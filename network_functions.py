"""
The class wrapper for the networks
"""
# Built-in
import os
import time

# Torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
#from torchsummary import summary
from torch.optim import lr_scheduler
from network_architecture import Lorentz_layer

# Libs
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt



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
        self.pretrain_loss = self.make_MSE_loss()
        self.optm = None                                        # The optimizer: Initialized at train()
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train()
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)     # Create a summary writer for tensorboard
        self.best_validation_loss = float('inf')    # Set the BVL to large number
        self.best_pretrain_loss = float('inf')  # Set the BVL to large number

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
        # custom_loss = torch.abs(torch.mean((logit - labels)**2))
        custom_loss = nn.functional.mse_loss(logit, labels)
        # derivative_loss = logit
        # for i in range(logit.size()[1]-1):
        #     derivative_loss[:,i+1] = nn.functional.mse_loss(logit[:,i+1] - logit[:,i], labels[:,i+1] - labels[:,i])
        # custom_loss += derivative_loss
        return custom_loss

    def make_optimizer(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed.
        :return:
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        else:
            raise Exception("Optimizer is not available at the moment.")
        return op

    def make_lr_scheduler(self):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
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

    def record_weight(self, name, layer=-1, batch=999, epoch=999):
        """
        Record the weight to compare and see which layer it started to change
        :input: name: The name to save
        :input: layer: The layer to check
        """
        weights = self.model.linears[layer].weight.cpu().data.numpy()
        # if batch == 0 and epoch == 0:
        #     np.savetxt('Training_Weights_Lorentz_Layer' + name,
        #         weights, fmt='%.3f', delimiter=',')
        f = plt.figure(figsize=(10,10))
        c = plt.imshow(weights.reshape((65,100)), cmap=plt.get_cmap('viridis'))
        plt.colorbar(c, fraction=0.03)
        # f.axes.get_xaxis().set_visible(False)
        # f.axes.get_yaxis().set_visible(False)
        self.log.add_figure(tag='Layer ' + str(layer) + ') Weights'.format(1), figure=f, global_step=epoch)

    def record_grad(self, name, layer=-1, batch=999, epoch=999):
        """
        Record the weight to compare and see which layer it started to change
        :input: name: The name to save
        :input: layer: The layer to check
        """
        gradients = self.model.linears[layer].grad.data.cpu().data.numpy()
        if batch == 0 and epoch == 0:
            np.savetxt('Training_Gradients_Lorentz_Layer' + name,
                self.model.linears[layer].grad.data.cpu().data.numpy(),
                fmt='%.3f', delimiter=',')
        f = plt.figure(figsize=(10, 10))
        c = plt.imshow(gradients.reshape((65, 100)), cmap=plt.get_cmap('viridis'))
        plt.colorbar(c, fraction=0.03)
        self.log.add_figure(tag='Layer ' + str(layer) + ') Gradients'.format(1), figure=f, global_step=epoch)

    def train(self):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """
        print("Starting training process")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.record_weight(name='start_of_train', batch=0, epoch=0)

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler()
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.ckpt_dir])
        url = tb.launch()
        # self.optm.zero_grad()
        # self.model = torch.load(os.path.join(self.ckpt_dir, 'best_pretrained_model.pt'))

        for epoch in range(self.flags.train_step):
            # print("This is training Epoch {}".format(epoch))
            # Set to Training Mode
            train_loss = []
            train_loss_eval_mode_list = []
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                self.record_weight(name='before_cuda', batch=j, epoch=epoch)
                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU
                self.optm.zero_grad()
                # Zero the gradient first
                logit, last_Lor_layer = self.model(geometry)                        # Get the output

                # print("logit type:", logit.dtype)
                # print("spectra type:", spectra.dtype)
                #loss = self.make_MSE_loss(logit, spectra)              # Get the loss tensor

                loss = self.make_custom_loss(logit, spectra)
                loss.backward()                                # Calculate the backward gradients
                self.record_weight(name='after_backward', batch=j, epoch=epoch)

                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 10)
                self.optm.step()                                    # Move one step the optimizer
                self.record_weight(name='after_optm_step', batch=j, epoch=epoch)

                train_loss.append(np.copy(loss.cpu().data.numpy()))                                     # Aggregate the loss

                #############################################
                # Extra test for err_test < err_train issue #
                #############################################
                self.model.eval()
                logit, last_Lor_layer = self.model(geometry)  # Get the output
                loss = self.make_custom_loss(logit, spectra)  # Get the loss tensor
                train_loss_eval_mode_list.append(np.copy(loss.cpu().data.numpy()))
                self.model.train()

            # Calculate the avg loss of training
            train_avg_loss = np.mean(train_loss)
            train_avg_eval_mode_loss = np.mean(train_loss_eval_mode_list)

            if epoch % self.flags.eval_step == 0:                        # For eval steps, do the evaluations and tensor board
                # Record the training loss to the tensorboard
                #train_avg_loss = train_loss.data.numpy() / (j+1)
                self.log.add_scalar('Loss/train', train_avg_loss, epoch)
                self.log.add_scalar('Loss/train_eval_mode', train_avg_eval_mode_loss, epoch)
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

                for j in range(self.flags.num_plot_compare):
                    f = self.compare_spectra(Ypred=logit[j, :].cpu().data.numpy(),
                                             Ytruth=spectra[j, :].cpu().data.numpy())
                    self.log.add_figure(tag='Sample ' + str(j) +') e2 Spectrum'.format(1), figure=f, global_step=epoch)

                # f2 = self.plotMSELossDistrib(logit.cpu().data.numpy(), spectra.cpu().data.numpy())
                # self.log.add_figure(tag='Single Batch Training MSE Histogram'.format(1), figure=f2,
                #                     global_step=epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = []
                for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    logit, last_Lor_layer = self.model(geometry)
                    #loss = self.make_MSE_loss(logit, spectra)                   # compute the loss
                    loss = self.make_custom_loss(logit, spectra)
                    test_loss.append(np.copy(loss.cpu().data.numpy()))                                       # Aggregate the loss

                # Record the testing loss to the tensorboard
                test_avg_loss = np.mean(test_loss)
                self.log.add_scalar('Loss/test', test_avg_loss, epoch)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_loss, test_avg_loss ))

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

        self.log.close()

    def pretrain(self, pretrain_loader, pretest_loader):
        """
        The pretraining function. This would start the training using information given in the flags
        :return: None
        """
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler()
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.ckpt_dir])
        url = tb.launch()

        print("Starting pre-training process")
        for epoch in range(200):
            # print("This is training Epoch {}".format(epoch))
            # Set to Training Mode
            train_loss = []
            train_loss_eval_mode_list = []
            self.model.train()
            for j, (geometry, lor_params) in enumerate(pretrain_loader):
                self.record_weight(name='Pretraining', batch=0, epoch=epoch)
                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    lor_params = lor_params.cuda()                            # Put data onto GPU
                self.optm.zero_grad()                               # Zero the gradient first
                logit, last_Lor_layer = self.model(geometry)                        # Get the output
                # print("label size:", lor_params.size())
                # print("logit size:", last_Lor_layer.size())
                # print(logit)
                # print(lor_params)

                pretrain_loss = self.make_MSE_loss(last_Lor_layer, lor_params[:,:12])              # Get the loss tensor
                pretrain_loss.backward()                                # Calculate the backward gradients
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 10)
                self.optm.step()                                    # Move one step the optimizer
                train_loss.append(np.copy(pretrain_loss.cpu().data.numpy()))                                   # Aggregate the loss

                #############################################
                # Extra test for err_test < err_train issue #
                #############################################
                self.model.eval()
                logit, last_Lor_layer = self.model(geometry)  # Get the output
                pretrain_loss = self.make_MSE_loss(last_Lor_layer, lor_params[:,:12])  # Get the loss tensor
                train_loss_eval_mode_list.append(np.copy(pretrain_loss.cpu().data.numpy()))
                self.model.train()

            # Calculate the avg loss of training
            train_avg_loss = np.mean(train_loss)
            train_avg_eval_mode_loss = np.mean(train_loss_eval_mode_list)

            if epoch % 20 == 0:
            #if epoch % self.flags.eval_step == 0:                        # For eval steps, do the evaluations and tensor board
                # Record the training loss to the tensorboard
                #train_avg_loss = train_loss.data.numpy() / (j+1)
                self.log.add_scalar('Pretrain Loss', train_avg_loss, epoch)
                self.log.add_scalar('Pretrain Loss/ Evaluation Mode', train_avg_eval_mode_loss, epoch)

                for j in range(self.flags.num_plot_compare):
                    f = self.compare_Lor_params(pred=last_Lor_layer[j, :].cpu().data.numpy(),
                                             truth=lor_params[j, :12].cpu().data.numpy())
                    self.log.add_figure(tag='Sample ' + str(j) +') Lorentz Parameter Prediction'.format(1), figure=f, global_step=epoch)

                pretrain_sim_prediction = lor_params[:, 12:]
                pretrain_model_prediction = Lorentz_layer(last_Lor_layer)

                for j in range(self.flags.num_plot_compare):

                    f = self.compare_spectra(Ypred=pretrain_model_prediction[j, :].cpu().data.numpy(),
                                             Ytruth=pretrain_sim_prediction[j, :].cpu().data.numpy())
                    self.log.add_figure(tag='Model ' + str(j) +') e2 Prediction'.format(1), figure=f, global_step=epoch)

                # f2 = self.plotMSELossDistrib(last_Lor_layer.cpu().data.numpy(), lor_params.cpu().data.numpy())
                # self.log.add_figure(tag='Single Batch Pretraining MSE Histogram'.format(1), figure=f2,
                #                     global_step=epoch)

                print("This is Epoch %d, pretraining loss %.5f" % (epoch, train_avg_loss ))

                # Model improving, save the model
                if train_avg_loss < self.best_pretrain_loss:
                    self.best_pretrain_loss = train_avg_loss
                    self.save()
                    print("Saving the model...")

                    if self.best_pretrain_loss < self.flags.stop_threshold:
                        print("Pretraining finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_pretrain_loss))
                        return None

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)

            if epoch == 199:
                # weights = self.model.linears[-1].weight.cpu().data.numpy()
                # # print(weights.shape)
                # np.savetxt('Pretrain_Lorentz_Weights.csv', weights, fmt='%.3f', delimiter=',')
                torch.save(self.model, os.path.join(self.ckpt_dir, 'best_pretrained_model.pt'))

        self.log.close()

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
            for ind, (geometry, spectra) in enumerate(self.test_loader):
                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                logits = self.model(geometry)
                np.savetxt(fxt, geometry.cpu().data.numpy(), fmt='%.3f')
                np.savetxt(fyt, spectra.cpu().data.numpy(), fmt='%.3f')
                np.savetxt(fyp, logits.cpu().data.numpy(), fmt='%.3f')
        return Ypred_file, Ytruth_file

    def compare_Lor_params(self, pred, truth, title=None, figsize=[5, 5]):
        """
        Function to plot the comparison for predicted and truth Lorentz parameters
        :param pred:  Predicted spectra, this should be a list of number of dimension 300, numpy
        :param truth:  Truth spectra, this should be a list of number of dimension 300, numpy
        :param title: The title of the plot, usually it comes with the time
        :param figsize: The figure size of the plot
        :return: The identifier of the figure
        """
        x = np.ones(4)
        w0_pr = pred[0::3]
        wp_pr = pred[1::3]
        g_pr = pred[2::3]
        w0_tr = truth[0::3]
        wp_tr = truth[1::3]
        g_tr = truth[2::3]
        f = plt.figure(figsize=figsize)
        marker_size = 14
        plt.plot(x, w0_pr, markersize=marker_size, color='red', marker='o', fillstyle='none', linestyle='None', label='w_0 pr')
        plt.plot(x, w0_tr, markersize=marker_size, color='red', marker='o', fillstyle='full', linestyle='None', label='w_0 tr')
        plt.plot(2*x, wp_pr, markersize=marker_size, color='blue', marker='s', fillstyle='none', linestyle='None', label='w_0 pr')
        plt.plot(2*x, wp_tr, markersize=marker_size, color='blue', marker='s', fillstyle='full', linestyle='None', label='w_0 tr')
        plt.plot(3*x, g_pr, markersize=marker_size, color='green', marker='v', fillstyle='none', linestyle='None', label='w_0 pr')
        plt.plot(3*x, g_tr, markersize=marker_size, color='green', marker='v', fillstyle='full', linestyle='None', label='w_0 tr')
        plt.xlabel("Lorentz Parameters")
        plt.ylabel("Parameter value")
        return f

    def compare_spectra(self, Ypred, Ytruth, T=None, title=None, figsize=[15, 5],
                        T_num=10, E1=None, E2=None, N=None, K=None, eps_inf=None):
        """
        Function to plot the comparison for predicted spectra and truth spectra
        :param Ypred:  Predicted spectra, this should be a list of number of dimension 300, numpy
        :param Ytruth:  Truth spectra, this should be a list of number of dimension 300, numpy
        :param title: The title of the plot, usually it comes with the time
        :param figsize: The figure size of the plot
        :return: The identifier of the figure
        """
        # Make the frequency into real frequency in THz
        fre_low = 0.5
        fre_high = 5
        num_points = len(Ypred)
        frequency = fre_low + (fre_high - fre_low) / len(Ytruth) * np.arange(num_points)
        f = plt.figure(figsize=figsize)
        plt.plot(frequency, Ypred, label='Pred')
        plt.plot(frequency, Ytruth, label='Truth')
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
        #plt.xlim([fre_low, fre_high])
        plt.xlabel("Frequency (THz)")
        plt.ylabel("e2")
        if title is not None:
            plt.title(title)
        return f

    def plotMSELossDistrib(self, pred, truth):

        # mae, mse = compare_truth_pred(pred_file, truth_file)
        # mae = np.mean(np.abs(pred - truth), axis=1)
        mse = np.mean(np.square(pred - truth), axis=1)

        f = plt.figure(figsize=(12, 6))
        plt.hist(mse, bins=100)
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Count')
        plt.suptitle('Model (Avg MSE={:.4e})'.format(np.mean(mse)))
        # plt.savefig(os.path.join(os.path.abspath(''), 'models',
        #                          'MSEdistrib_{}.png'.format(flags.model_name)))
        return f
        # plt.show()
        # print('Backprop (Avg MSE={:.4e})'.format(np.mean(mse)))

