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
import math
from torch import pow, add, mul, div, sqrt, abs, square
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from torchsummary import summary
from torch.optim import lr_scheduler
from torchviz import make_dot
# from network_model import Lorentz_layer
from plotting_functions import plot_weights_3D, plotMSELossDistrib, \
    compare_spectra, compare_spectra_with_params


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
        # summary(model, input_data=(8,))
        print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('There are %d trainable out of %d total parameters' %(pytorch_total_params, pytorch_total_params_train))
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
        MSE_loss = nn.functional.mse_loss(logit, labels,reduction='mean')          # The MSE Loss of the network
        return MSE_loss

    def make_custom_loss(self, logit=None, labels=None):

        if logit is None:
            return None

        # Loss function to handle the gradients of the Lorentz layer

        # custom_loss = torch.mean(torch.mean((logit - labels)**self.flags.err_exp, 1))
        # custom_loss = torch.mean(torch.norm((logit-labels),p=4))/logit.shape[0]
        # custom_loss = nn.functional.mse_loss(logit, labels, reduction='mean')
        # custom_loss = nn.functional.smooth_l1_loss(logit, labels)
        # additional_loss_term = self.lorentz_product_loss_term(logit, labels)
        # additional_loss_term = self.peak_finder_loss(logit, labels)
        # custom_loss += additional_loss_term

        # logit_diff = 715*(logit[:,1:]-logit[:,:-1])
        # labels_diff = 715*(labels[:, 1:] - labels[:, :-1])
        # deriv_loss = nn.functional.mse_loss(logit_diff, labels_diff, reduction='mean')
        mse_loss = nn.functional.mse_loss(logit, labels, reduction='mean')
        mse_loss *= 10000
        # print(mse_loss)
        # deriv_loss = nn.functional.l1_loss(logit_diff, labels_diff, reduction='mean')
        # custom_loss = 0.01*deriv_loss + mse_loss
        custom_loss = mse_loss

        return custom_loss



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
        # return lr_scheduler.StepLR(optimizer=self.optm, step_size=50, gamma=0.75, last_epoch=-1)
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

    def init_weights(self):

        for layer_name, child in self.model.named_children():
            for param in self.model.parameters():

                if (layer_name == 'lin_w0'):
                    torch.nn.init.uniform_(child.weight, a=0.0, b=0.02)
                if (layer_name == 'lin_wp'):
                    torch.nn.init.uniform_(child.weight, a=0.0, b=0.02)
                elif (layer_name == 'lin_g'):
                    torch.nn.init.uniform_(child.weight, a=0.0, b=0.02)
                # elif (layer_name == 'input_coupl1' or layer_name == 'input_coupl2'):
                #     torch.nn.init.zeros_(child.weight)
                    # torch.nn.init.normal_(child.weight, std=0.01)
                # elif (layer_name == 'lin_eps_inf'):
                    # torch.nn.init.uniform_(child.weight,a=1, b=3)
                else:
                    if ((type(child) == nn.Linear) | (type(child) == nn.Conv2d)):
                        torch.nn.init.xavier_uniform_(child.weight)

                        # torch.nn.init.uniform_(child.weight, a=0.0, b=0.05)
                        # if child.bias:
                        #     child.bias.data.fill_(0.00)

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
        # params = list(self.model.cyl1.parameters())
        # params.extend(list(self.model.cyl1_eps.parameters()))
        # params.extend(list(self.model.cyl2_eps.parameters()))
        # params.extend(list(self.model.cyl3_eps.parameters()))
        # params.extend(list(self.model.cyl4_eps.parameters()))
        # # params.extend(list(self.model.lin_g.parameters()))
        # # params.extend(list(self.model.lin_w0.parameters()))
        # # params.extend(list(self.model.lin_wp.parameters()))
        # params.extend(list(self.model.lin_eps_inf.parameters()))
        # self.optm = torch.optim.Adam(params,lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        self.lr_scheduler = self.make_lr_scheduler()
        #
        # params2 = list(self.model.input_coupl1.parameters())
        # params2.extend(list(self.model.bn1.parameters()))
        # params2.extend(list(self.model.input_coupl2.parameters()))
        # params2.extend(list(self.model.bn2.parameters()))
        # self.optm2 = torch.optim.Adam(params2, lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        # self.lr_scheduler2 = lr_scheduler.ReduceLROnPlateau(optimizer=self.optm2, mode='min',
        #                                 factor=self.flags.lr_decay_rate,
        #                                   patience=10, verbose=True, threshold=1e-4)


        self.init_weights()
        # self.model.divNN = torch.load('/home/omar/PycharmProjects/mlmOK_Pytorch/pretrained_div_network.pt')
        # for param in self.model.divNN.parameters():
        #     param.requires_grad = False

        # div_op = torch.optim.Adam(self.model.divNN.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)

        # Start a tensorboard session for logging loss and training images
        # tb = program.TensorBoard()
        # tb.configure(argv=[None, '--logdir', self.ckpt_dir])
        # url = tb.launch()
        # print("TensorBoard started at %s" % url)
        # pid = os.getpid()
        # print("PID = %d; use 'kill %d' to quit" % (pid, pid))

        # interaction_epoch = 0

        for epoch in range(self.flags.train_step):
            # print("This is training Epoch {}".format(epoch))
            # Set to Training Mode

        #     if epoch == interaction_epoch:
        #         for layer_name, child in self.model.named_children():
        #             for param in self.model.parameters():
        #                 if (layer_name == 'input_coupl1' or layer_name == 'input_coupl2'):
        #                     torch.nn.init.normal_(child.weight, std=0.01)

            train_loss = []
            train_loss_eval_mode_list = []
            self.model.train()

            # if epoch < 100:
            #     self.train_loader.batch_size = 64
            # elif epoch == 100:
            #     print('Pre-training with small batchsize done')
            #     self.train_loader.batch_size = self.flags.batch_size
            #     self.flags.lr = self.flags.lr/100

            for j, (geometry, spectra) in enumerate(self.train_loader):
                # Record weights and gradients to tb

                # TODO: Create loop for this
                # if epoch % self.flags.record_step == 0:
                # #     self.record_weight(name='lin0', layer=self.model.linears[0], batch=j, epoch=epoch)
                # #     self.record_weight(name='lin1', layer=self.model.linears[1], batch=j, epoch=epoch)
                # #     self.record_weight(name='w_p', layer=self.model.lin_wp, batch=j, epoch=epoch)
                # #     self.record_weight(name='w_0', layer=self.model.lin_w0, batch=j, epoch=epoch)
                # #     self.record_weight(name='g', layer=self.model.lin_g, batch=j, epoch=epoch)
                # #     self.record_weight(name='w_p', layer=self.model.lin_wp, batch=j, epoch=epoch)
                # #     self.record_weight(name='w_0', layer=self.model.lin_w0, batch=j, epoch=epoch)
                # #     self.record_grad(name='g', layer=self.model.lin_g, batch=j, epoch=epoch)
                # #     self.record_grad(name='lin0', layer=self.model.linears[0], batch=j, epoch=epoch)
                # #     self.record_grad(name='lin1', layer=self.model.linears[1], batch=j, epoch=epoch)
                #     self.record_weight(name='cyl1', layer=self.model.cyl1.linears[-1], batch=j, epoch=epoch)
                #     self.record_weight(name='int1', layer=self.model.input_coupl2, batch=j, epoch=epoch)
                #     self.record_weight(name='eps_inf', layer=self.model.lin_eps_inf, batch=j, epoch=epoch)

                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU

                self.optm.zero_grad()
                # self.optm2.zero_grad()   # Zero the gradient first
                # logit = self.model(geometry)
                # loss = self.make_MSE_loss(logit, spectra)
                if epoch % self.flags.record_step == 0 and j==0:
                    record = epoch

                else:
                    record = -1

                T,w0,wp,g,eps_inf = self.model(geometry)

                loss = self.make_custom_loss(T, spectra)  # compute the loss
                # print(loss)
                # loss = self.make_custom_loss(logit, spectra)
                if j == 0 and epoch == 0:
                    im = make_dot(loss, params=dict(self.model.named_parameters())).render("Model Graph",
                                                                                           format="png",
                                                                                           directory=self.ckpt_dir)
                # print(loss)
                loss.backward()

                if epoch % self.flags.record_step == 0:
                    if j == 0:
                        for k in range(self.flags.num_plot_compare):
                            f = compare_spectra(Ypred=T[k, :].cpu().data.numpy(),
                                                     Ytruth=spectra[k, :].cpu().data.numpy(),
                                                w_0=w0[k, :].cpu().data.numpy(),
                                                w_p=wp[k, :].cpu().data.numpy(), g=g[k, :].cpu().data.numpy(),
                                                E2=None, eps_inf= eps_inf[k].cpu().data.numpy(),
                                                test_var=None, xmin=self.flags.freq_low,
                                                xmax=self.flags.freq_high, num_points=self.flags.num_spec_points)

                            self.log.add_figure(tag='Test ' + str(k) +') Sample Transmission Spectrum'.format(1),
                                                figure=f, global_step=epoch)



                # if epoch < 2000:
                self.optm.step()
                # if (epoch > 2000):
                #     self.optm2.step()  # Move one step the optimizer
                # self.record_weight(name='after_optm_step', batch=j, epoch=epoch)

                train_loss.append(np.copy(loss.cpu().data.numpy()))     # Aggregate the loss
                self.running_loss.append(np.copy(loss.cpu().data.numpy()))
                # #############################################
                # # Extra test for err_test < err_train issue #
                # #############################################
                self.model.eval()
                T,w0,wp,g,eps_inf = self.model(geometry)


                loss = self.make_custom_loss(T, spectra)  # compute the loss
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
                # self.log.add_scalar('Running Loss', train_avg_eval_mode_loss, epoch)

                # if epoch % self.flags.record_step == 0:
                #     for j in range(self.flags.num_plot_compare):
                #         f,ax = compare_spectra(Ypred=logit[j, :].cpu().data.numpy(),
                #                                  Ytruth=spectra[j, :].cpu().data.numpy(), E2=self.model.e2[j,:,:], xmin=self.flags.freq_low,
                #                             xmax=self.flags.freq_high, num_points=self.flags.num_spec_points)
                #         self.log.add_figure(tag='Test ' + str(j) +') e2 Sample Spectrum'.format(1),
                #                             figure=f, global_step=epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = []
                test_loss2 = []
                with torch.no_grad():
                    for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                        if cuda:
                            geometry = geometry.cuda()
                            spectra = spectra.cuda()

                        T,w0,wp,g,eps_inf = self.model(geometry)

                        loss = self.make_custom_loss(T, spectra)  # compute the loss               # compute the loss
                        loss2 = self.make_MSE_loss(T, spectra)
                        # loss = self.make_custom_loss(logit, spectra)

                        test_loss.append(np.copy(loss.cpu().data.numpy()))           # Aggregate the loss
                        test_loss2.append(np.copy(loss2.cpu().data.numpy()))

                        # if j == 0 and epoch % self.flags.record_step == 0 and epoch > 20:
                        #     # f2 = plotMSELossDistrib(test_loss)
                        #     f2 = plotMSELossDistrib(T.cpu().data.numpy(), spectra[:, ].cpu().data.numpy())
                        #     self.log.add_figure(tag='0_Testing Loss Histogram'.format(1), figure=f2,
                        #                         global_step=epoch)



                # Record the testing loss to the tensorboard

                test_avg_loss = np.mean(test_loss)
                test_avg_loss2 = np.mean(test_loss2)
                self.log.add_scalar('Loss/ Validation Loss', test_avg_loss, epoch)
                self.log.add_scalar('Validation MSE Loss', test_avg_loss2, epoch)

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




            # # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
            # # self.lr_scheduler.step()


            if self.flags.use_warm_restart:
                if epoch % self.flags.lr_warm_restart == 0:
                    for param_group in self.optm.param_groups:
                        param_group['lr'] = self.flags.lr
                        print('Resetting learning rate to %.5f' % self.flags.lr)


            # # Separate learning rates for interaction layer
            #
            # # # Learning rate decay upon plateau
            # # if epoch < 2000:
            # self.lr_scheduler.step(train_avg_loss)
            # if epoch > interaction_epoch:
            #     self.lr_scheduler2.step(train_avg_loss)
            # # # self.lr_scheduler.step()
            #
            #
            # if self.flags.use_warm_restart:
            #     if epoch % self.flags.lr_warm_restart == 0:
            #         for param_group in self.optm.param_groups:
            #             param_group['lr'] = self.flags.lr/10
            #             print('Resetting learning rate to %.5f' % self.flags.lr)
            #         if epoch > interaction_epoch:
            #             for param_group in self.optm2.param_groups:
            #                 param_group['lr'] = self.flags.lr / 10

        # print('Finished')
        self.log.close()









