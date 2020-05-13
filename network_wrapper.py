"""
Wrapper functions for the networks
"""
# Built-in
import os
import time

# Torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
#from torchsummary import summary
from torch.optim import lr_scheduler
from torchviz import make_dot
from network_model import Lorentz_layer
from plotting_functions import plot_weights_3D, plotMSELossDistrib, \
    compare_spectra, compare_Lor_params

# Libs
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelmax



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
        self.best_pretrain_loss = float('inf')
        self.running_loss = []

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
        # additional_loss_term = self.lorentz_product_loss_term(logit, labels)
        # additional_loss_term = self.peak_finder_loss(logit, labels)
        # custom_loss += additional_loss_term

        return custom_loss

    def lorentz_product_loss_term(self, logit=None, labels=None):

        if logit is None:
            return None

        batch_size = labels.size()[0]
        loss_penalty = 100
        ascend = torch.tensor([0, 1, -1], requires_grad=False, dtype=torch.float32)
        descend = torch.tensor([-1, 1, 0], requires_grad=False, dtype=torch.float32)

        if torch.cuda.is_available():
            ascend = ascend.cuda()
            descend = descend.cuda()

        max = F.relu(F.conv1d(labels.view(batch_size, 1, -1),
                              ascend.view(1, 1, -1), bias=None, stride=1, padding=1))
        min = F.relu(F.conv1d(labels.view(batch_size, 1, -1),
                              descend.view(1, 1, -1), bias=None, stride=1, padding=1))
        zeros = torch.mul(max, min).squeeze()
        zeros = torch.logical_xor(zeros, torch.zeros_like(zeros))
        zeros = zeros.unsqueeze(1).expand_as(self.model.w_expand)
        # print(zeros.size())

        w = torch.mul(self.model.w_expand, zeros)
        w0 = torch.mul(self.model.w0, zeros)
        g = torch.mul(self.model.g, zeros)
        # print(w)
        # print(w0)

        # loss = torch.mean(torch.sum(torch.mul(torch.abs(torch.pow(w0, 2) - torch.pow(w, 2)), torch.mul(w, g)), 1))
        loss = torch.mean(torch.sum(torch.mul(torch.abs(w-w0), g), 1))
        # print(loss)
        loss = loss_penalty*loss
        return loss

    def peak_finder_loss(self, logit=None, labels=None):

        if logit is None:
            return None
        batch_size = labels.size()[0]
        loss_penalty = 10000

        ascend = torch.tensor([0, 1, -1], requires_grad=False, dtype=torch.float32)
        descend = torch.tensor([-1, 1, 0], requires_grad=False, dtype=torch.float32)

        if torch.cuda.is_available():
            ascend = ascend.cuda()
            descend = descend.cuda()

        max = F.relu(F.conv1d(labels.view(batch_size, 1, -1),
                               ascend.view(1, 1, -1), bias=None, stride=1, padding=1))
        min = F.relu(F.conv1d(labels.view(batch_size, 1, -1),
                                descend.view(1, 1, -1), bias=None, stride=1, padding=1))
        zeros = torch.mul(max, min).squeeze()
        zeros = torch.logical_xor(zeros, torch.zeros_like(zeros))
        loss = torch.mean(torch.mean(torch.mul(torch.abs(logit-labels), zeros), 1))
        # loss = 0
        # print(loss)
        loss = loss*100
        # print(loss)
        return loss

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
        # f = compare_spectra(Ypred=we_w[0, 1:].cpu().data.numpy(),
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

    def record_weight(self, name='Weights', layer=None, batch=999, epoch=999):
        """
        Record the weights for a specific layer to tensorboard (save to file if desired)
        :input: name: The name to save
        :input: layer: The layer to check
        """
        if batch == 0:
            weights = layer.weight.cpu().data.numpy()   # Get the weights

            # if epoch == 0:
            # np.savetxt('Training_Weights_Lorentz_Layer' + name,
            #     weights, fmt='%.3f', delimiter=',')
            # print(weights_layer.shape)

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
            f = plot_weights_3D(weights.reshape((sq, sq)), sq)
            self.log.add_figure(tag='1_Weights_' + name + '_Layer'.format(1),
                                figure=f, global_step=epoch)

    def record_grad(self, name='Gradients', layer=None, batch=999, epoch=999):
        """
        Record the gradients for a specific layer to tensorboard (save to file if desired)
        :input: name: The name to save
        :input: layer: The layer to check
        """
        if batch == 0 and epoch > 0:
            gradients = layer.weight.grad.cpu().data.numpy()

            # if epoch == 0:
            # np.savetxt('Training_Weights_Lorentz_Layer' + name,
            #     weights, fmt='%.3f', delimiter=',')
            # print(weights_layer.shape)

            # Reshape the weights into a square dimension for plotting, zero padding if necessary
            wmin = np.amin(np.asarray(gradients.shape))
            wmax = np.amax(np.asarray(gradients.shape))
            sq = int(np.floor(np.sqrt(wmin * wmax)) + 1)
            diff = np.zeros((1, int(sq ** 2 - (wmin * wmax))), dtype='float64')
            gradients = gradients.reshape((1, -1))
            gradients = np.concatenate((gradients, diff), axis=1)
            # f = plt.figure(figsize=(10, 5))
            # c = plt.imshow(weights.reshape((sq, sq)), cmap=plt.get_cmap('viridis'))
            # plt.colorbar(c, fraction=0.03)
            f = plot_weights_3D(gradients.reshape((sq, sq)), sq)
            self.log.add_figure(tag='1_Gradients_' + name + '_Layer'.format(1),
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

        self.model.lin_w0.weight.requires_grad_(False)

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

                # TODO: Create loop for this
                if epoch % self.flags.record_step == 0:
                    self.record_weight(name='w_p', layer=self.model.lin_wp, batch=j, epoch=epoch)
                    self.record_weight(name='w_0', layer=self.model.lin_w0, batch=j, epoch=epoch)
                    self.record_weight(name='g', layer=self.model.lin_g, batch=j, epoch=epoch)
                    self.record_grad(name='w_p', layer=self.model.lin_wp, batch=j, epoch=epoch)
                    # self.record_grad(name='w_0', layer=self.model.lin_w0, batch=j, epoch=epoch)
                    self.record_grad(name='g', layer=self.model.lin_g, batch=j, epoch=epoch)

                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU

                self.optm.zero_grad()                                   # Zero the gradient first
                logit,w0,wp,g = self.model(geometry)            # Get the output

                # print("logit type:", logit.dtype)
                # print("spectra type:", spectra.dtype)

                #loss = self.make_MSE_loss(logit, spectra)              # Get the loss tensor
                loss = self.make_custom_loss(logit, spectra[:, 12:])
                if j == 0 and epoch == 0:
                    im = make_dot(loss, params=dict(self.model.named_parameters())).render("Model Graph",
                                                                                           format="png",
                                                                                           directory=self.ckpt_dir)
                loss.backward()

                # Calculate the backward gradients
                # self.record_weight(name='after_backward', batch=j, epoch=epoch)

                # Clip gradients to help with training
                if self.flags.use_clip:
                    if self.flags.use_clip:
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.flags.grad_clip)
                        torch.nn.utils.clip_grad_value_(self.model.lin_w0.parameters(), self.flags.grad_clip)
                        torch.nn.utils.clip_grad_value_(self.model.lin_g.parameters(), self.flags.grad_clip)
                        torch.nn.utils.clip_grad_value_(self.model.lin_wp.parameters(), self.flags.grad_clip)
                        # torch.nn.utils.clip_grad_norm_(self.model.lin_w0.parameters(), self.flags.grad_clip, norm_type=2)
                        # torch.nn.utils.clip_grad_norm_(self.model.lin_g.parameters(), self.flags.grad_clip, norm_type=2)

                if epoch % self.flags.record_step == 0:
                    if j == 0:
                        for k in range(self.flags.num_plot_compare):
                            f = compare_spectra(Ypred=logit[k, :].cpu().data.numpy(),
                                                     Ytruth=spectra[k, 12:].cpu().data.numpy(), E2=self.model.e2[k,:,:], xmin=self.flags.freq_low,
                                                xmax=self.flags.freq_high, num_points=self.flags.num_spec_points)
                            self.log.add_figure(tag='Test ' + str(k) +') Sample e2 Spectrum'.format(1),
                                                figure=f, global_step=epoch)


                self.optm.step()                                        # Move one step the optimizer
                # self.record_weight(name='after_optm_step', batch=j, epoch=epoch)

                train_loss.append(np.copy(loss.cpu().data.numpy()))     # Aggregate the loss
                self.running_loss.append(np.copy(loss.cpu().data.numpy()))
                # #############################################
                # # Extra test for err_test < err_train issue #
                # #############################################
                self.model.eval()
                logit,w0,wp,g = self.model(geometry)  # Get the output
                loss = self.make_custom_loss(logit, spectra[:, 12:])  # Get the loss tensor
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
                self.log.add_scalar('Running Loss', train_avg_eval_mode_loss, epoch)

                # if self.flags.use_lorentz:
                    # for j in range(self.flags.num_plot_compare):
                    #     f = compare_spectra(Ypred=logit[j, :].cpu().data.numpy(),
                    #                              Ytruth=spectra[j, :].cpu().data.numpy(),
                    #                              T=self.model.T_each_lor[j, :],
                    #                              eps_inf = self.model.eps_inf[j])
                    #     self.log.add_figure(tag='T{}'.format(j), figure=f, global_step=epoch)
                    # For debugging purpose, in model:forward function reocrd the tensor
                    # self.log.add_histogram("w0_histogram", self.model.w0s, epoch)
                    # self.log.add_histogram("wp_histogram", self.model.wps, epoch)
                    # self.log.add_histogram("g_histogram", self.model.gs, epoch)

                # if epoch % self.flags.record_step == 0:
                #     for j in range(self.flags.num_plot_compare):
                #         f = compare_spectra(Ypred=logit[j, :].cpu().data.numpy(),
                #                                  Ytruth=spectra[j, 12:].cpu().data.numpy(), E2=self.model.e2[j,:,:], xmin=self.flags.freq_low,
                #                             xmax=self.flags.freq_high, num_points=self.flags.num_spec_points)
                #         self.log.add_figure(tag='Test ' + str(j) +') e2 Sample Spectrum'.format(1),
                #                             figure=f, global_step=epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = []
                with torch.no_grad():
                    for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                        if cuda:
                            geometry = geometry.cuda()
                            spectra = spectra.cuda()
                        logit,w0,wp,g = self.model(geometry)
                        #loss = self.make_MSE_loss(logit, spectra)                   # compute the loss
                        loss = self.make_custom_loss(logit, spectra[:, 12:])
                        test_loss.append(np.copy(loss.cpu().data.numpy()))           # Aggregate the loss

                        if j == 0 and epoch % self.flags.record_step == 0:
                            # f2 = plotMSELossDistrib(test_loss)
                            f2 = plotMSELossDistrib(logit.cpu().data.numpy(), spectra[:, 12:].cpu().data.numpy())
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
            # self.lr_scheduler.step(train_avg_loss)

            if self.flags.use_warm_restart:
                if epoch % self.flags.lr_warm_restart == 0:
                    for param_group in self.optm.param_groups:
                        param_group['lr'] = self.flags.lr
                        print('Resetting learning rate to %.5f' % self.flags.lr)

        # print('Finished')
        self.log.close()
        np.savetxt(time.strftime('%Y%m%d_%H%M%S', time.localtime())+'.csv', self.running_loss, delimiter=",")

    def pretrain(self):
        """
        The pretraining function. This starts the pretraining using parameters given in the flags
        :return: None
        """
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-3)
        self.lr_scheduler = self.make_lr_scheduler()

        # Start a tensorboard session for logging loss and training images
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.ckpt_dir])
        url = tb.launch()

        print("Starting pre-training process")
        pre_train_epoch = 250
        for epoch in range(pre_train_epoch):  # Only 200 epochs needed for pretraining
            # print("This is pretrainin Epoch {}".format(epoch))
            # Set to Training Mode
            train_loss = []
            train_loss_eval_mode_list = []
            sim_loss_list = []

            self.model.train()
            for j, (geometry, params_truth) in enumerate(self.train_loader):
                # if j == 0 and epoch == 0:
                    # print(geometry)
                # Record weights and gradients to tb
                if epoch % self.flags.record_step == 0:
                    self.record_weight(name='Pretrain w_p', layer=self.model.lin_wp, batch=j, epoch=epoch)
                    self.record_weight(name='Pretrain w_0', layer=self.model.lin_w0, batch=j, epoch=epoch)
                    self.record_weight(name='Pretrain g', layer=self.model.lin_g, batch=j, epoch=epoch)

                if cuda:
                    geometry = geometry.cuda()  # Put data onto GPU
                    params_truth = params_truth.cuda()  # Put data onto GPU
                self.optm.zero_grad()  # Zero the gradient first
                logit,w0,wp,g = self.model(geometry)  # Get the output
                # print("label size:", params_truth.size())
                # print("logit size:", params.size())

                pretrain_loss = self.make_MSE_loss(w0, params_truth[:, :4])  # Get the loss tensor
                pretrain_loss += self.make_MSE_loss(wp, params_truth[:, 4:8])  # Get the loss tensor
                pretrain_loss += self.make_MSE_loss(g, params_truth[:, 8:12]*10)  # Get the loss tensor

                pretrain_loss.backward()  # Calculate the backward gradients
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 10)
                train_loss.append(np.copy(pretrain_loss.cpu().data.numpy()))  # Aggregate the loss

                #############################################
                # Extra test for err_test < err_train issue #
                #############################################
                self.model.eval()
                logit,w0,wp,g = self.model(geometry)  # Get the output
                pretrain_loss_test = self.make_MSE_loss(w0, params_truth[:, :4])  # Get the loss tensor
                pretrain_loss_test += self.make_MSE_loss(wp, params_truth[:, 4:8])  # Get the loss tensor
                pretrain_loss_test += self.make_MSE_loss(g, params_truth[:, 8:12]*10)  # Get the loss tensor
                train_loss_eval_mode_list.append(np.copy(pretrain_loss_test.cpu().data.numpy()))
                pretrain_model_prediction = Lorentz_layer(w0, wp, g / 10)
                sim_loss = self.make_MSE_loss(pretrain_model_prediction,params_truth[:, 12:])
                sim_loss_list.append(sim_loss.cpu().data.numpy())
                self.model.train()

                #######################################
                # Monitor the same loss like training #
                #######################################

                # logit, params = self.model(geometry)  # Get the output
                # pre_train_spectra_loss = self.make_custom_loss(logit, params_truth[:, 12:])
                # # print(pre_train_spectra_loss)
                # # if torch.isnan(pre_train_spectra_loss):
                # #     print("!!! YOU ENCOUNTER NAN LOSS IN PRE_TRAINING SPECTRA LOSS PART")
                # # pre_train_spectra_loss_list.append(pre_train_spectra_loss.cpu().data.numpy())
                #
                # # update at the end, this is to make sure the last epoch does not update the weights
                if epoch < pre_train_epoch - 1:
                    self.optm.step()  # Move one step the optimizer

            # Calculate the avg loss of training
            train_avg_loss = np.mean(train_loss)
            train_avg_eval_mode_loss = np.mean(train_loss_eval_mode_list)
            sim_loss = np.mean(sim_loss_list)
            self.running_loss.append(sim_loss)

            if epoch % 10 == 0:  # Evaluate every 20 steps
                # Record the training loss to the tensorboard
                # train_avg_loss = train_loss.data.numpy() / (j+1)
                self.log.add_scalar('Pretrain Loss', train_avg_loss, epoch)
                self.log.add_scalar('Pretrain Loss/ Evaluation Mode', train_avg_eval_mode_loss, epoch)
                self.log.add_scalar('Simulation Loss', sim_loss, epoch)



                for j in range(self.flags.num_plot_compare):
                    f = compare_Lor_params(w0=w0[j, :].cpu().data.numpy(), wp=wp[j, :].cpu().data.numpy(),
                                           g=g[j, :].cpu().data.numpy(),
                                           truth=params_truth[j, :12].cpu().data.numpy())
                    self.log.add_figure(tag='Test ' + str(j) + ') e2 Lorentz Parameter Prediction'.
                                        format(1), figure=f, global_step=epoch)

                # Pretraining files contain both Lorentz parameters and simulated model spectra
                pretrain_sim_prediction = params_truth[:, 12:]
                pretrain_model_prediction = Lorentz_layer(w0,wp,g/10)

                for j in range(self.flags.num_plot_compare):
                    f = compare_spectra(Ypred=pretrain_model_prediction[j, :].cpu().data.numpy(),
                                             Ytruth=pretrain_sim_prediction[j, :].cpu().data.numpy())
                    self.log.add_figure(tag='Test ' + str(j) + ') e2 Model Prediction'.format(1),
                                        figure=f, global_step=epoch)

                # f2 = self.plotMSELossDistrib(params.cpu().data.numpy(), params_truth.cpu().data.numpy())
                # self.log.add_figure(tag='Single Batch Pretraining MSE Histogram'.format(1), figure=f2,
                #                     global_step=epoch)

                print("This is Epoch %d, pretrain loss %.5f, eval mode loss is %.5f, and sim loss is %.5f" % (
                epoch, train_avg_loss, train_avg_eval_mode_loss, sim_loss))

                # Model improving, save the model
                if train_avg_eval_mode_loss < self.best_pretrain_loss:
                    self.best_pretrain_loss = train_avg_loss
                    self.save()
                    print("Saving the model...")

                    if self.best_pretrain_loss < self.flags.stop_threshold:
                        print("Pretraining finished EARLIER at epoch %d, reaching loss of %.5f" % \
                              (epoch, self.best_pretrain_loss))
                        return None

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)

            # Save pretrained model at end
            # if epoch == 10:
            #     # weights = self.model.linears[-1].weight.cpu().data.numpy()
            #     # # print(weights.shape)
            #     # np.savetxt('Pretrain_Lorentz_Weights.csv', weights, fmt='%.3f', delimiter=',')
            #     torch.save(self.model, os.path.join(self.ckpt_dir, 'best_pretrained_model.pt'))
            #     # self.record_weight(name='Pretraining', batch=0, epoch=999)

        self.log.close()


    def evaluate(self, save_dir='eval/'):
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
            with torch.no_grad():
                for ind, (geometry, spectra) in enumerate(self.test_loader):
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    logits = self.model(geometry)
                    np.savetxt(fxt, geometry.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fyt, spectra.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fyp, logits.cpu().data.numpy(), fmt='%.3f')
        return Ypred_file, Ytruth_file






