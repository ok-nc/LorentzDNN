import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

weights_pre = np.loadtxt('Pretrain_Lorentz_Weights.csv', dtype='float32', delimiter=',')
weights0 = np.loadtxt('Training_Weights_Lorentz_Layer_0.csv', dtype='float32', delimiter=',')
weights1 = np.loadtxt('Training_Weights_Lorentz_Layer_1.csv', dtype='float32', delimiter=',')
weights2 = np.loadtxt('Training_Weights_Lorentz_Layer_Epoch_0_0.csv', dtype='float32', delimiter=',')
weights3 = np.loadtxt('Training_Weights_Lorentz_Layer_Epoch_0_1.csv', dtype='float32', delimiter=',')
weights4 = np.loadtxt('Training_Weights_Lorentz_Layer_Epoch_0_2.csv', dtype='float32', delimiter=',')
weights5 = np.loadtxt('Training_Weights_Lorentz_Layer_Epoch_0_3.csv', dtype='float32', delimiter=',')
weights6 = np.loadtxt('Training_Weights_Lorentz_Layer_Epoch_0_4.csv', dtype='float32', delimiter=',')
pre = weights_pre.reshape((65,100))
d0 = weights0.reshape((65,100))
d1 = weights1.reshape((65,100))
d2 = weights2.reshape((65,100))
d3 = weights3.reshape((65,100))
d4 = weights4.reshape((65,100))
d5 = weights5.reshape((65,100))
d6 = weights6.reshape((65,100))
imgs = [pre,d0,d1,d2,d3,d4,d5,d6]
# f, aximg = plt.subplots(1,3, figsize=(15,5))
f, aximg = plt.subplots(7,2, figsize=(10,30))
for i in range(7):
    a = aximg[i,0].imshow(imgs[i], cmap=plt.get_cmap('viridis'))
    aximg[i,0].set_title('Checkpoint_'+str(i), fontsize=24)
    aximg[i,0].axes.get_xaxis().set_visible(False)
    aximg[i,0].axes.get_yaxis().set_visible(False)
    plt.colorbar(a,ax=aximg[i,0],fraction=0.03)
    b = aximg[i,1].imshow(imgs[i]-imgs[0], cmap=plt.get_cmap('viridis'))
    aximg[i,1].set_title('Difference', fontsize=24)
    aximg[i,1].axes.get_xaxis().set_visible(False)
    aximg[i,1].axes.get_yaxis().set_visible(False)
    plt.colorbar(b,ax=aximg[i,1],fraction=0.03)

# a=aximg[0].imshow(pre, cmap=plt.get_cmap('viridis'))
# plt.colorbar(a,ax=aximg[0],fraction=0.03)
# b=aximg[1].imshow(d1, cmap=plt.get_cmap('viridis'))
# plt.colorbar(b,ax=aximg[1],fraction=0.03)
# c=aximg[2].imshow(d1-pre, cmap=plt.get_cmap('viridis'))
# plt.colorbar(c,ax=aximg[2],fraction=0.03)
# aximg[0].set_title('Pretraining Weights (13x500 reshaped)', fontsize=14)
# aximg[1].set_title('Mid-epoch 0 Weights (13x500 reshaped)', fontsize=14)
# aximg[2].set_title('Difference (13x500 reshaped)', fontsize=14)
# aximg[1].axes.get_xaxis().set_visible(False)
# aximg[1].axes.get_yaxis().set_visible(False)
# aximg[2].axes.get_xaxis().set_visible(False)
# aximg[2].axes.get_yaxis().set_visible(False)

plt.show()



# import os
# import imageio
#
# #jpg_dir = '../saves/png/'
# jpg_dir = '/home/omar/PycharmProjects/mlmOK_Pytorch/models/20200213_215704/out/Sample_1_Test_Prediction'
# images = []
# for file_name in os.listdir(jpg_dir):
#     if file_name.endswith('.jpg'):
#         file_path = os.path.join(jpg_dir, file_name)
#         images.append(imageio.imread(file_path))
# imageio.mimsave(os.path.join(jpg_dir, 'test.gif'), images, fps=5)