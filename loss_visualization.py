
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
                                                     batch_size=flags.batch_size,
                                                     normalize_input=flags.normalize_input,
                                                     data_dir=flags.data_dir,
                                                     test_ratio=0.999)
    print("Making network now")

    # Make Network
    ntwk = Network(Forward, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)

    # Evaluation process
    ntwk.load_pretrain()  # load the model as constructed
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        ntwk.model.cuda()
    ntwk.model.eval()

    # for param in ntwk.model.lin_w0.parameters():
    #     # print(param)
    #     param.data += torch.rand(1).cuda()*0.001

    # ntwk.model.lin_w0.weight.data[0] += torch.rand(1).cuda() * 0.01
    # ntwk.model.lin_g.weight.data[0] += torch.rand(1).cuda() * 0.01



    # print(ntwk.model.lin_w0.weight.data[2,10])
    # print(ntwk.model.lin_w0.weight.data[2, 9])
    # print(ntwk.model.lin_w0.weight.data[3, 14])
    # print(ntwk.model.lin_w0.weight.data[0, 26])
    # print(ntwk.model.lin_w0.weight.data[1, 29])
    #
    # # print(ntwk.model.lin_w0.weight.data[1,3])
    # print(ntwk.model.lin_g.weight.data[1, 12])
    # print(ntwk.model.lin_g.weight.data[0, 13])
    # print(ntwk.model.lin_g.weight.data[2, 15])
    # print(ntwk.model.lin_g.weight.data[2, 19])
    # print(ntwk.model.lin_g.weight.data[3, 1])
    # print(ntwk.model.lin_g.weight.data[1, 3])

    # dim = 25
    # dx = 0.2
    # dy = 0.2
    #
    # wx = 0
    # wy = 11
    #
    # wx2 = 3
    # wy2 = 10
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

    # # 2D loop
    # loss_surface = np.empty((2*dim+1, 2*dim+1))
    # x = np.arange(-dim * dx, (dim + 1) * dx, dx)
    # y = np.arange(-dim * dy, (dim + 1) * dy, dy)
    # for i, j in enumerate(x):
    #     print(str(i) + ' of ' + str(len(x)))
    #     for k, l in enumerate(y):
    #
    #         eval_loss = []
    #         ntwk.model.lin_g.weight.data[wx, wy] += j
    #         ntwk.model.lin_g.weight.data[wx2, wy2] += l
    #         with torch.no_grad():
    #             for ind, (geometry, spectra) in enumerate(test_loader):
    #                 if cuda:
    #                     geometry = geometry.cuda()
    #                     spectra = spectra.cuda()
    #                 logit, w0, wp, g = ntwk.model(geometry)
    #                 loss = ntwk.make_custom_loss(logit, spectra[:, 12:])
    #                 eval_loss.append(np.copy(loss.cpu().data.numpy()))
    #         ntwk.model.lin_g.weight.data[wx, wy] -= j
    #         ntwk.model.lin_g.weight.data[wx2, wy2] -= l
    #         eval_avg_loss = np.mean(eval_loss)
    #         # print(str(j))
    #         # print(eval_avg_loss)
    #         loss_surface[i,k] = eval_avg_loss
    # #
    # np.savetxt('Loss_Surface_'+'g_('+str(wx)+','+str(wy)+')_dim'+str(dim)+'_dx'+str(dx)+
    #            '_g_('+str(wx2)+','+str(wy2)+')_dim'+str(dim)+'_dy'+str(dy)+'.csv',
    #            loss_surface, delimiter=",")


    # print(loss_surface)
    # np.savetxt('Loss_Surface.csv', loss_surface, delimiter=",")
    # array = np.array([x, loss_1D])
    # np.savetxt('Loss_Surface_'+'w0_('+str(wx)+','+str(wy)+')_dim'+str(dim)+'_dx'+str(dx)+'.csv',
    #            array, delimiter=",")
    #
    #
    plt.switch_backend('Qt4Agg')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # plt.plot(x,loss_1D)
    plt.hist(ntwk.model.linears[1].weight.cpu().data.numpy())
    plt.show(block=True)
    plt.ion()




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