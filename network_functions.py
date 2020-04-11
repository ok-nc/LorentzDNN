"""
Wrapper functions for the networks
"""
# Built-in
import os
import time
# import psutil

# Torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
#from torchsummary import summary
from torch.optim import lr_scheduler
from torchviz import make_dot

# Libs
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class Network(object):
    def __init__(self, model_fn, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        self.model_fn = model_fn                                # The network architecture object
        self.flags = flags                                      # The flags containing the hyperparameters
        if inference_mode:                                      # If inference mode, use saved model
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # Network training mode, create a new ckpt folder
            if flags.model_name is None:                    # Use custom name if possible, otherwise timestamp
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model = self.create_model()                        # The model itself
        self.loss = self.make_custom_loss()                            # The loss function
        self.optm = None                                        # The optimizer: Initialized at train()
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train()
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)     # Create a summary writer for tensorboard
        self.best_validation_loss = float('inf')    # Set the BVL to large number

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        #summary(model, input_size=(1, 8))
        print(model)
        return model

    def make_MSE_loss(self, logit=None, labels=None):
        """
        Create a tensor that represents the loss. This is consistent both at training time \
        and inference time for a backward model
        :param logit: The output of the network
        :return: the total loss
        """
        if logit is None:
            return None
        MSE_loss = nn.functional.mse_loss(logit, labels)          # The MSE Loss of the network
        return MSE_loss

    def make_custom_loss(self, logit=None, labels=None):

        if logit is None:
            return None

        # Loss function to handle the gradients of the Lorentz layer

        # custom_loss = torch.mean(torch.mean((logit - labels)**self.flags.err_exp, 1))
        # custom_loss = torch.mean(torch.norm((logit-labels),p=4))/logit.shape[0]
        custom_loss = nn.functional.mse_loss(logit, labels, reduction='mean')
        # custom_loss = nn.functional.smooth_l1_loss(logit, labels)
        # logit_diff = logit[1:] - logit[:-1]
        # labels_diff = labels[1:] - labels[:-1]
        # derivative_loss = nn.functional.mse_loss(logit_diff, labels_diff)
        # custom_loss += derivative_loss
        # logit_norm = nn.functional.instance_norm(logit)
        # labels_norm = nn.functional.instance_norm(labels)

        # dotproduct = torch.tensordot(logit, labels)
        # loss_penalty = torch.exp(-dotproduct/1000000)
        # # print('Loss penalty is '+str(dotproduct.cpu().data.numpy()/10000))
        # if custom_loss < 10:
        #     additional_loss_term = self.make_e2_KK_loss(logit)
        #     custom_loss += additional_loss_term
        return custom_loss

    def make_e2_KK_loss(self, logit=None):
        # Enforces Kramers-Kronig causality on e2 derivative

        if logit is None:
            return None

        num_spec_points = logit.shape[1]
        dw = (self.flags.freq_high - self.flags.freq_low) / num_spec_points
        w_numpy = np.arange(self.flags.freq_low, self.flags.freq_high, dw)
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            w = torch.tensor(w_numpy, requires_grad=False).cuda()
        else:
            w = torch.tensor(w_numpy, requires_grad=False)
        w = w.expand_as(logit)
        we_w = torch.mul(logit, w)
        # print(logit.shape, w.shape, we_w.shape)
        # we_w_diff = torch.zeros(we_w.shape, requires_grad=False)
        we_w_diff = (we_w[:, 1:] - we_w[:, :-1])
        dwe_dw = torch.div(we_w_diff, dw)
        diff = torch.abs(torch.min(dwe_dw[:, :]))
        diff_scaled = diff/logit.shape[0]
        # f = self.compare_spectra(Ypred=we_w[0, 1:].cpu().data.numpy(),
        #                          Ytruth=dwe_dw[0, :].cpu().data.numpy(), label_y1='w*e(w)', label_y2='d(w*e)/dw')
        # self.log.add_figure(tag='Sample ' + str(0) + ') derivative e2 Spectrum'.format(1),
        #                     figure=f)

        # zero = torch.zeros(max.shape, requires_grad=False, dtype=torch.float32)
        # print(loss.shape)
        # print(diff)
        return diff_scaled

    def make_optimizer(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed.
        :return:
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'AdamW':
            op = torch.optim.AdamW(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'Adamax':
            op = torch.optim.Adamax(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SparseAdam':
            op = torch.optim.SparseAdam(self.model.parameters(), lr=self.flags.lr)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale, momentum=0.9, nesterov=True)
        elif self.flags.optim == 'LBFGS':
            op = torch.optim.LBFGS(self.model.parameters(), lr=1, max_iter=20, history_size=100)
        else:
            raise Exception("Optimizer is not available at the moment.")
        return op

    def make_lr_scheduler(self):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        if self.flags.use_warm_restart:
            return lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optm,
                                                            T_0=self.flags.lr_warm_restart, T_mult=1)
        else:
            return lr_scheduler.ReduceLROnPlateau(optimizer=self.optm, mode='min',
                                            factor=self.flags.lr_decay_rate,
                                              patience=10, verbose=True, threshold=1e-4)



    def save(self):
        """
        Saving the model to the current check point folder with name best_model.pt
        :return: None
        """
        torch.save(self.model, os.path.join(self.ckpt_dir, 'best_model.pt'))

    def load(self):
        """
        Loading the model from the check point folder with name best_model.pt
        :return:
        """
        self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model.pt'))

    def record_weight(self, name='Weights', layer=-1, batch=999, epoch=999):
        """
        Record the weights for a specific layer to tensorboard (save to file if desired)
        :input: name: The name to save
        :input: layer: The layer to check
        """
        if batch == 0:
            weights_layer = self.model.linears[layer].weight.cpu().data.numpy()   # Get the weights

            # weights_w0 = weights_layer[:,::3]
            # weights_wp = weights_layer[:, 1::3]
            # weights_g = weights_layer[:, 2::3]

            # weights_all = [weights_w0, weights_wp, weights_g]
            weights_all = [weights_layer]

            # if epoch == 0:
            # np.savetxt('Training_Weights_Lorentz_Layer' + name,
            #     weights, fmt='%.3f', delimiter=',')
            # print(weights_layer.shape)
            for ind, weights in enumerate(weights_all):
                # Reshape the weights into a square dimension for plotting, zero padding if necessary
                wmin = np.amin(np.asarray(weights.shape))
                wmax = np.amax(np.asarray(weights.shape))
                sq = int(np.floor(np.sqrt(wmin * wmax)) + 1)
                diff = np.zeros((1, int(sq**2 - (wmin * wmax))), dtype='float64')
                weights = weights.reshape((1, -1))
                weights = np.concatenate((weights, diff), axis=1)
                # f = plt.figure(figsize=(10, 5))
                # c = plt.imshow(weights.reshape((sq, sq)), cmap=plt.get_cmap('viridis'))
                # plt.colorbar(c, fraction=0.03)
                f = self.plot_weights_3D(weights.reshape((sq, sq)), sq)
                self.log.add_figure(tag='1_Weights_' + name + '_Layer ' + str(layer) + ')_'+str(ind).format(1),
                                    figure=f, global_step=epoch)
                # if epoch == 0 or epoch == 999:
                #     save_file = 'Weights'+str(ind)+'_'+name+'.png'
                #     save_path = os.path.join(self.ckpt_dir,save_file)
                #     plt.savefig(save_path)
                #     plt.show()


    def record_grad(self, name='Gradients', layer=-1, batch=999, epoch=999):
        """
        Record the gradients for a specific layer to tensorboard (save to file if desired)
        :input: name: The name to save
        :input: layer: The layer to check
        """
        if batch == 0 and epoch > 0:
            gradients_layer = self.model.linears[layer].weight.grad.cpu().data.numpy()
            gradients_w0 = gradients_layer[:, ::3]
            gradients_wp = gradients_layer[:, 1::3]
            gradients_g = gradients_layer[:, 2::3]

            gradients_all = [gradients_w0, gradients_wp, gradients_g]
            # if epoch == 0:
            #     np.savetxt('Training_Gradients_Lorentz_Layer' + name,
            #                gradients, fmt='%.3f', delimiter=',')

            # Reshape the gradients into a square dimension for plotting, zero padding if necessary
            for ind, gradients in enumerate(gradients_all):
                grmin = np.amin(np.asarray(gradients.shape))
                grmax = np.amax(np.asarray(gradients.shape))
                sq = int(np.floor(np.sqrt(grmin * grmax)) + 1)
                diff = np.zeros((1, int(sq ** 2 - (grmin * grmax))), dtype='float64')
                gradients = gradients.reshape((1, -1))
                gradients = np.concatenate((gradients, diff), axis=1)
                # f = plt.figure(figsize=(10, 5))
                # c = plt.imshow(gradients.reshape((sq, sq)), cmap=plt.get_cmap('viridis'))
                # plt.colorbar(c, fraction=0.03)
                f = self.plot_weights_3D(gradients.reshape((sq, sq)), sq)

                self.log.add_figure(tag='2_Gradients_'+ name + '_Layer ' + str(layer) + ')_'+str(ind).format(1),
                                    figure=f, global_step=epoch)

    def train(self):
        """
        The major training function. This starts the training using parameters given in the flags
        :return: None
        """
        print("Starting training process")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        # self.record_weight(name='start_of_train', batch=0, epoch=0)

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler()

        # Start a tensorboard session for logging loss and training images
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.ckpt_dir])
        url = tb.launch()
        print("TensorBoard started at %s" % url)
        # pid = os.getpid()
        # print("PID = %d; use 'kill %d' to quit" % (pid, pid))

        for epoch in range(self.flags.train_step):
            # print("This is training Epoch {}".format(epoch))
            # Set to Training Mode
            train_loss = []
            train_loss_eval_mode_list = []
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                # Record weights and gradients to tb
                # if epoch % self.flags.record_step == 0:
                #     for ind in range(3):
                #     # for ind, fc_num in enumerate(self.flags.linear):
                #         self.record_weight(name='Training', layer=ind-1, batch=j, epoch=epoch)
                #         # self.record_grad(name='Training', layer=ind-1, batch=j, epoch=epoch)

                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU

                self.optm.zero_grad()                                   # Zero the gradient first
                logit = self.model(geometry)            # Get the output

                # print("logit type:", logit.dtype)
                # print("spectra type:", spectra.dtype)

                #loss = self.make_MSE_loss(logit, spectra)              # Get the loss tensor
                loss = self.make_custom_loss(logit, spectra)
                if j == 0 and epoch == 0:
                    im = make_dot(loss, params=dict(self.model.named_parameters())).render("Model Graph",
                                                                                           format="png",
                                                                                           directory=self.ckpt_dir)
                loss.backward()

                # Calculate the backward gradients
                # self.record_weight(name='after_backward', batch=j, epoch=epoch)

                # Clip gradients to help with training
                if self.flags.use_clip:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.flags.grad_clip)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.flags.grad_clip)

                self.optm.step()                                        # Move one step the optimizer
                # self.record_weight(name='after_optm_step', batch=j, epoch=epoch)

                train_loss.append(np.copy(loss.cpu().data.numpy()))     # Aggregate the loss

                # #############################################
                # # Extra test for err_test < err_train issue #
                # #############################################
                self.model.eval()
                logit = self.model(geometry)  # Get the output
                loss = self.make_custom_loss(logit, spectra)  # Get the loss tensor
                train_loss_eval_mode_list.append(np.copy(loss.cpu().data.numpy()))
                self.model.train()

            # Calculate the avg loss of training
            train_avg_loss = np.mean(train_loss)
            train_avg_eval_mode_loss = np.mean(train_loss_eval_mode_list)

            if epoch % self.flags.eval_step == 0:           # For eval steps, do the evaluations and tensor board
                # Record the training loss to tensorboard
                #train_avg_loss = train_loss.data.numpy() / (j+1)
                self.log.add_scalar('Loss/ Training Loss', train_avg_loss, epoch)
                self.log.add_scalar('Loss/ Batchnorm Training Loss', train_avg_eval_mode_loss, epoch)
                # if self.flags.use_lorentz:
                #     for j in range(self.flags.num_plot_compare):
                        # f = self.compare_spectra(Ypred=logit[j, :].cpu().data.numpy(),
                        #                          Ytruth=spectra[j, :].cpu().data.numpy(),
                        #                          T=self.model.T_each_lor[j, :],
                        #                          eps_inf = self.model.eps_inf[j])
                        #self.log.add_figure(tag='T{}'.format(j), figure=f, global_step=epoch)
                    # For debugging purpose, in model:forward function reocrd the tensor
                    # self.log.add_histogram("w0_histogram", self.model.w0s, epoch)
                    # self.log.add_histogram("wp_histogram", self.model.wps, epoch)
                    # self.log.add_histogram("g_histogram", self.model.gs, epoch)

                if epoch % self.flags.record_step == 0:
                    for j in range(self.flags.num_plot_compare):
                        f = self.compare_spectra(Ypred=logit[j, :].cpu().data.numpy(),
                                                 Ytruth=spectra[j, :].cpu().data.numpy())
                        self.log.add_figure(tag='Test ' + str(j) +') e2 Sample Spectrum'.format(1),
                                            figure=f, global_step=epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = []
                with torch.no_grad():
                    for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                        if cuda:
                            geometry = geometry.cuda()
                            spectra = spectra.cuda()
                        logit = self.model(geometry)
                        #loss = self.make_MSE_loss(logit, spectra)                   # compute the loss
                        loss = self.make_custom_loss(logit, spectra)
                        test_loss.append(np.copy(loss.cpu().data.numpy()))           # Aggregate the loss

                        if j == 0 and epoch % self.flags.record_step == 0:
                            # f2 = self.plotMSELossDistrib(test_loss)
                            f2 = self.plotMSELossDistrib(logit.cpu().data.numpy(), spectra.cpu().data.numpy())
                            self.log.add_figure(tag='0_Testing Loss Histogram'.format(1), figure=f2,
                                                global_step=epoch)

                # Record the testing loss to the tensorboard

                test_avg_loss = np.mean(test_loss)
                self.log.add_scalar('Loss/ Validation Loss', test_avg_loss, epoch)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_eval_mode_loss, test_avg_loss ))

                # Model improving, save the model
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.save()
                    print("Saving the model...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        return None

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)

            if epoch % self.flags.lr_warm_restart == 0:
                for param_group in self.optm.param_groups:
                    param_group['lr'] = self.flags.lr
                    print('Resetting learning rate to %.5f' % self.flags.lr)

        # print('Finished')
        self.log.close()
        # p = psutil.Process(pid)
        # p.terminate()

    def evaluate(self, save_dir='data/'):
        self.load()                             # load the model as constructed
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()                       # Evaluation mode

        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(self.saved_model))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(self.saved_model))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(self.saved_model))
        # Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(self.saved_model))  # For pure forward model, there is no Xpred

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt, open(Ypred_file, 'a') as fyp:
            # Loop through the eval data and evaluate
            with torch.no_grad:
                for ind, (geometry, spectra) in enumerate(self.test_loader):
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    logits = self.model(geometry)
                    np.savetxt(fxt, geometry.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fyt, spectra.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fyp, logits.cpu().data.numpy(), fmt='%.3f')
        return Ypred_file, Ytruth_file

    def compare_spectra(self, Ypred, Ytruth, T=None, title=None, figsize=[10, 5],
                        T_num=10, E1=None, E2=None, N=None, K=None, eps_inf=None, label_y1='Pred', label_y2='Truth'):
        """
        Function to plot the comparison for predicted spectra and truth spectra
        :param Ypred:  Predicted spectra, this should be a list of number of dimension 300, numpy
        :param Ytruth:  Truth spectra, this should be a list of number of dimension 300, numpy
        :param title: The title of the plot, usually it comes with the time
        :param figsize: The figure size of the plot
        :return: The identifier of the figure
        """
        # Make the frequency into real frequency in THz
        num_points = len(Ypred)
        frequency = self.flags.freq_low + (self.flags.freq_high - self.flags.freq_low) / len(Ytruth) * np.arange(num_points)
        f = plt.figure(figsize=figsize)
        plt.plot(frequency, Ypred, label=label_y1)
        plt.plot(frequency, Ytruth, label=label_y2)
        if T is not None:
            plt.plot(frequency, T, linewidth=1, linestyle='--')
        if E2 is not None:
            for i in range(np.shape(E2)[0]):
                plt.plot(frequency, E2[i, :], linewidth=1, linestyle=':', label="E2" + str(i))
        if E1 is not None:
            for i in range(np.shape(E1)[0]):
                plt.plot(frequency, E1[i, :], linewidth=1, linestyle='-', label="E1" + str(i))
        if N is not None:
            plt.plot(frequency, N, linewidth=1, linestyle=':', label="N")
        if K is not None:
            plt.plot(frequency, K, linewidth=1, linestyle='-', label="K")
        if eps_inf is not None:
            plt.plot(frequency, np.ones(np.shape(frequency)) * eps_inf, label="eps_inf")
        # plt.ylim([0, 1])
        plt.legend()
        #plt.xlim([self.flags.freq_low, self.flags.freq_high])
        plt.xlabel("Frequency (THz)")
        plt.ylabel("e2")
        if title is not None:
            plt.title(title)
        return f

    def plotMSELossDistrib(self, pred, truth):

        # mae, mse = compare_truth_pred(pred_file, truth_file)
        # mae = np.mean(np.abs(pred - truth), axis=1)
        mse = np.mean(np.square(pred - truth), axis=1)
        # mse = loss
        f = plt.figure(figsize=(12, 6))
        plt.hist(mse, bins=100)
        plt.xlabel('Validation Loss')
        plt.ylabel('Count')
        plt.suptitle('Model (Avg MSE={:.4e})'.format(np.mean(mse)))
        # plt.savefig(os.path.join(os.path.abspath(''), 'models',
        #                          'MSEdistrib_{}.png'.format(flags.model_name)))
        return f
        # plt.show()
        # print('Backprop (Avg MSE={:.4e})'.format(np.mean(mse)))


    def plot_weights_3D(self, data, dim, figsize=[10, 5]):

        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(121, projection='3d', proj_type='ortho')
        ax2 = fig.add_subplot(122)

        xx, yy = np.meshgrid(np.linspace(0,dim,dim), np.linspace(0,dim,dim))
        cmp = plt.get_cmap('viridis')

        ax1.plot_surface(xx, yy, data, cmap=cmp)
        ax1.view_init(10, -45)

        c2 = ax2.imshow(data, cmap=cmp)
        plt.colorbar(c2, fraction=0.03)

        return fig
