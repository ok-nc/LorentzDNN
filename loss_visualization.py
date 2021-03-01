
import os
import torch
import flagreader
from network_wrapper import Network
from network_model import Forward
import datareader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plotting_functions import plot_weights_3D, plotMSELossDistrib, \
    compare_spectra, compare_spectra_with_params

def visualize_loss(model_dir):
    # Retrieve the flag object
    if (model_dir.startswith("models")):
        model_dir = model_dir[7:]
        print("after removing prefix models/, now model_dir is:", model_dir)
    print("Retrieving flag object for parameters")
    flags = flagreader.load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode

    # Get the data
    # train_loader, test_loader = datareader.read_data(flags)
    train_loader, test_loader = datareader.read_data(x_range=flags.x_range,
                                                     y_range=flags.y_range,
                                                     geoboundary=flags.geoboundary,
                                                     batch_size=1,
                                                     normalize_input=flags.normalize_input,
                                                     data_dir=flags.data_dir,
                                                     test_ratio=0.999,shuffle=False)
    print("Making network now")

    # Make Network
    ntwk = Network(Forward, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)

    # Evaluation process
    ntwk.load()  # load the model as constructed
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        ntwk.model.cuda()
    ntwk.model.eval()

    # for param in ntwk.model.lin_w0.parameters():
    #     # print(param)
    #     param.data += torch.rand(1).cuda()*0.001

    # ntwk.model.lin_w0.weight.data[0] += torch.rand(1).cuda() * 0.01
    # ntwk.model.lin_g.weight.data[0] += torch.rand(1).cuda() * 0.01

    # weights = ntwk.model.linears[2].weight.cpu().data.numpy()  # Get the weights

    # # Reshape the weights into a square dimension for plotting, zero padding if necessary
    # wmin = np.amin(np.asarray(weights.shape))
    # wmax = np.amax(np.asarray(weights.shape))
    # sq = int(np.floor(np.sqrt(wmin * wmax)) + 1)
    # diff = np.zeros((1, int(sq ** 2 - (wmin * wmax))), dtype='float64')
    # weights = weights.reshape((1, -1))
    # weights = np.concatenate((weights, diff), axis=1)
    # # f = plt.figure(figsize=(10, 5))
    # # c = plt.imshow(weights.reshape((sq, sq)), cmap=plt.get_cmap('viridis'))
    # # plt.colorbar(c, fraction=0.03)
    # f = plot_weights_3D(weights.reshape((sq, sq)), sq)
    # fig = plt.figure(figsize=[10,10])
    # ax = fig.add_subplot(111)
    # cmp = plt.get_cmap('viridis')
    # c2 = ax.imshow(weights,aspect='auto',cmap=cmp,)
    # plt.colorbar(c2, fraction=0.03)
    # plt.grid(b=None)
    # plt.savefig('/home/omar/PycharmProjects/mlmOK_Pytorch/lin2.png')



    #
    # # 1D loop
    # loss_1D = np.empty(2*dim+1)
    # x = np.arange(-dim*dx, (dim+1)*dx, dx)
    # # ntwk.model.lin_w0.weight.data[wx2, wy2] += 5
    # for i,j in enumerate(x):
    #     print(str(i)+' of '+str(len(x)))
    #     # print(ntwk.model.lin_w0.weight.data[2, 10])
    #     eval_loss = []
    #
    #     ntwk.model.lin_g.weight.data[wx, wy] += j
    #     with torch.no_grad():
    #         for ind, (geometry, spectra) in enumerate(test_loader):
    #             if cuda:
    #                 geometry = geometry.cuda()
    #                 spectra = spectra.cuda()
    #             logit, w0, wp, g = ntwk.model(geometry)
    #             loss = ntwk.make_custom_loss(logit, spectra[:, 12:])
    #             eval_loss.append(np.copy(loss.cpu().data.numpy()))
    #     ntwk.model.lin_g.weight.data[wx, wy] -= j
    #     eval_avg_loss = np.mean(eval_loss)
    #     # print(str(j))
    #     # print(eval_avg_loss)
    #     loss_1D[i] = eval_avg_loss




    ntwk.model.eval()
    print("Doing Evaluation on the model now")
    test_loss = []
    with torch.no_grad():
        for j, (geometry, spectra) in enumerate(ntwk.test_loader):  # Loop through the eval set
            if cuda:
                geometry = geometry.cuda()
                spectra = spectra.cuda()

            # for n in [1,4,5,6,10,14,17,19,20,25]:
            #     if j == n:
            #         for in_geo in range(8):
            #             geom = geometry.clone()
            #             fig, ax = plt.subplots(1, figsize=(10, 5))
            #             frequency = ntwk.flags.freq_low + (ntwk.flags.freq_high - ntwk.flags.freq_low) / \
            #                         ntwk.flags.num_spec_points * np.arange(ntwk.flags.num_spec_points)
            #             ax.plot(frequency, spectra[0,:].cpu().data.numpy(), linewidth=5,label='Truth')
            #             colors = ['orange','red', 'green', 'blue', 'purple']
            #             for k in range(5):
            #                 delta = k*0.1
            #                 geom[0,in_geo] += delta
            #                 logit, w0, wp, g = ntwk.model(geom)  # Get the output
            #
            #                 ax.plot(frequency,logit[0,:].cpu().data.numpy(),color=colors[k],label=str(np.round(delta,3)))
            #             plt.xlabel("Frequency (THz)")
            #             plt.ylabel("Transmission")
            #             plt.legend(loc="lower left", frameon=False)
            #             plt.savefig('/home/omar/PycharmProjects/mlmOK_Pytorch/Perturbation_in'+str(in_geo+1)
            #                         +'_0.1_Spectrum'+str(j)+'.png')
            #         if n==25:
            #             break

            for n in [1,4,5,6,10,14,17,19,20,25]:
                if j == n:
                    for in_geo in range(8):
                        geom = geometry.clone()
                        fig, ax = plt.subplots(1, figsize=(10, 5))
                        frequency = ntwk.flags.freq_low + (ntwk.flags.freq_high - ntwk.flags.freq_low) / \
                                    ntwk.flags.num_spec_points * np.arange(ntwk.flags.num_spec_points)
                        x = np.ones(ntwk.flags.num_lorentz_osc)
                        marker_size = 14
                        logit, w0, wp, g = ntwk.model(geometry)  # Get the output
                        ax.plot(w0[0, :].cpu().data.numpy(),
                                markersize=marker_size, color='orange', marker='o', fillstyle='full',
                                linestyle='None', label='w_0')
                        ax.plot(g[0, :].cpu().data.numpy(),
                                markersize=marker_size, color='orange', marker='s', fillstyle='full',
                                linestyle='None', label='g')
                        ax.plot(wp[0, :].cpu().data.numpy(),
                                markersize=marker_size, color='orange', marker='v', fillstyle='full',
                                linestyle='None', label='w_p')
                        colors = ['red','green','blue','purple']
                        for k in range(1,5):
                            delta = k*0.1
                            geom[0,in_geo] += delta
                            logit, w0, wp, g = ntwk.model(geom)

                            ax.plot(w0[0,:].cpu().data.numpy(),
                                     markersize=marker_size, color=colors[k-1], marker='o', fillstyle='full',
                                     linestyle='None')
                            ax.plot(g[0,:].cpu().data.numpy(),
                                     markersize=marker_size, color=colors[k-1], marker='s', fillstyle='full',
                                     linestyle='None')
                            ax.plot(wp[0,:].cpu().data.numpy(),
                                     markersize=marker_size, color=colors[k-1], marker='v', fillstyle='full',
                                     linestyle='None')
                        plt.ylabel("Lorentzian parameter value")
                        plt.legend(loc="upper left", frameon=False)
                        plt.savefig('/home/omar/PycharmProjects/mlmOK_Pytorch/Perturb_in'+str(in_geo+1)+'_0.1_Params'+str(j)+'.png')
                    if n==25:
                        break

            # logit, w0, wp, g = ntwk.model(geometry)  # Get the output
            # loss = ntwk.make_MSE_loss(logit, spectra)  # compute the loss
            # mse_loss = loss.cpu().data.numpy()
            # test_loss.append(mse_loss)  # Aggregate the loss

            # if mse_loss < .0003:
            #     # Test spectra indices: 0,1,4,6,10
            #     if j > -1:
            #         # print(j)
            #         # print(mse_loss)
            #
            #         f = compare_spectra(Ypred=logit[0, :].cpu().data.numpy(),
            #                             Ytruth=spectra[0, :].cpu().data.numpy(), xmin=ntwk.flags.freq_low,
            #                             xmax=ntwk.flags.freq_high, num_points=ntwk.flags.num_spec_points)
            #         # plt.switch_backend('Qt4Agg')
            #         # plt.show(block=True)
            #         # plt.ion()
            #         plt.savefig('/home/omar/PycharmProjects/mlmOK_Pytorch/TestSpectra_'+str(j)+'.png')
            #         if j > 25:
            #             break




    # plt.switch_backend('Qt4Agg')
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # # plt.plot(x,loss_1D)
    # # plt.hist(ntwk.model.linears[1].weight.cpu().data.numpy())
    # plt.hist(test_loss, bins=100)
    # plt.xlabel('Mean Squared Error')
    # plt.ylabel('cnt')
    # plt.suptitle('(Avg MSE={:.4e})'.format(np.mean(test_loss)))
    # plt.show(block=True)
    # plt.ion()




    # data1 = ntwk.model.lin_g.weight.cpu().data.numpy()
    # data2 = ntwk.model.lin_w0.weight.cpu().data.numpy()
    # data3 = ntwk.model.lin_wp.weight.cpu().data.numpy()
    # data = np.vstack((data1,data2,data3))
    # cmp = plt.get_cmap('viridis')
    # c = ax1.imshow(data, cmap=cmp)
    # c = ax1.imshow(loss_surface, cmap=cmp)
    # plt.colorbar(c, fraction=0.03)
    # plt.grid(b=None)
    # plt.show()
    # plt.show(block=True)
    # plt.ion()





if __name__ == '__main__':
    # Read the flag, however only the flags.eval_model is used and others are not used
    flags = flagreader.read_flag()

    print(flags.eval_model)
    # Call the evaluate function from model
    visualize_loss(flags.eval_model)