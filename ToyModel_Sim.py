# # Scratch file to learn Pytorch basics
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from datareader import MetaMaterialDataSet, read_data

# Visualize our data
import tkinter
import matplotlib
matplotlib.use('qt5agg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# # Defines Network structure, extends Pytorch nn module
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#
#         self.layer = torch.nn.Linear(1, 1)
#
#     def forward(self, x):
#         x = self.layer(x)
#         return x

# Returns single e2 simulation for given lorentzian parameters

def Lorentzian(w0, wp, g):

    freq_low = 0
    freq_high = 5
    num_freq = 300
    w = np.arange(freq_low, freq_high, (freq_high-freq_low)/num_freq)

    # e1 = np.divide(np.multiply(np.power(wp, 2), np.add(np.power(w0, 2), -np.power(w, 2))),
    #                   np.add(np.power(np.add(np.power(w0, 2), -np.power(w, 2)), 2),
    #                          np.multiply(np.power(w, 2), np.power(g, 2))))

    e2 = np.divide(np.multiply(np.power(wp, 2), np.multiply(w, np.power(g, 2))),
             np.add(np.power(np.add(np.power(w0, 2), -np.power(w, 2)), 2),
                    np.multiply(np.power(w, 2), np.power(g, 2))))

    return w, e2

def MM_Geom(n):

    # Parameter bounds for metamaterial radius and height
    r_min = 20
    r_max = 200
    h_min = 20
    h_max = 100

    # Defines hypergeometric space of parameters to choose from
    space = 10
    r_space = np.linspace(r_min, r_max, space+1)
    h_space = np.linspace(h_min, h_max, space+1)

    # Shuffles r,h arrays each iteration and then selects 0th element to generate random n x n parameter set
    r, h = np.zeros(n, dtype=float), np.zeros(n, dtype=float)
    for i in range(n):
        np.random.shuffle(r_space)
        np.random.shuffle(h_space)
        r[i] = r_space[0]
        h[i] = h_space[0]
    return r, h

def Make_MM_Model(n):

    r, h = MM_Geom(n)
    spectra = np.zeros(300)
    geom = np.concatenate((r, h), axis=0)
    for i in range(n):
        w0 = 100/h[i]
        wp = (1/100)*np.sqrt(np.pi)*r[i]
        g = (1/1000)*np.sqrt(np.pi)*r[i]
        w, e2 = Lorentzian(w0, wp, g)
        spectra += e2
    return geom, spectra


# Generates randomized dataset of simulated spectra for training and testing
def Prepare_Data(osc, sets, batch_size):

    features = []
    labels = []

    for i in range(sets):

        geom, spectra = Make_MM_Model(osc)
        features.append(geom)
        labels.append(spectra)

    features = np.array(features, dtype='float32')
    labels = np.array(labels, dtype='float32')

    ftrsize = features.size/sets
    lblsize = labels.size/sets
    print('Size of Features is %i, Size of Labels is %i' % (ftrsize, lblsize))
    print('There are %i datasets:' % sets)

    ftrTrain, ftrTest, lblTrain, lblTest = train_test_split(features, labels, test_size=0.2, random_state=1234)
    train_data = MetaMaterialDataSet(ftrTrain, lblTrain, bool_train= True)
    test_data = MetaMaterialDataSet(ftrTest, lblTest, bool_train= False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    print('Number of Training samples is {}'.format(len(ftrTrain)))
    print('Number of Test samples is {}'.format(len(ftrTest)))
    return train_loader, test_loader

# Read in training data from simulation files instead
def Read_Data_From_Files(dir):

    batch_size = 1024
    train_loader, test_loader = datareader.read_data(x_range=[i for i in range(0, 8 )],
                                                     range=[i for i in range(8 , 308 )],
                                                     geoboundary=[20, 200, 20, 100], batch_size=batch_size,
                                                     normalize_input=True, data_dir=dir,
                                                     test_ratio=0.2, pre_train=False)
    return train_loader, test_loader

# # # # Network creation and training code below
# net = Network()
# print(net)

# Create random test data
# w, e2 = Lorentzian(1, 2.5, 0.1)
# geom, spectra = Make_MM_Model(2)
test_loader, train_loader = Prepare_Data(2, 10, 5)



# w = np.arange(0.5,5,4.5/300)
# Visualize data
# plt.plot(w, spectra)
# plt.show()

# # Convert numpy array to torch tensors
# x = torch.from_numpy(x.reshape(-1,1)).float()
# y = torch.from_numpy(y.reshape(-1,1)).float()
# # print(x,y)
#
# # Define optimizer and loss function
# optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
# # optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-3)
# loss_func = torch.nn.MSELoss()
# im = make_dot(loss, params=dict(net.named_parameters())).render("Toy Model Graph",
#                                                                            format="png", directory=cwd)
#
# # Training loop
# inputs = x
# outputs = y
# for i in range(250):
#     prediction = net(inputs)
#     loss = loss_func(prediction, outputs)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if i % 10 == 0:
#         # plot and show learning process
#         plt.cla()
#         plt.scatter(x.data.numpy(), y.data.numpy())
#         plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
#         plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'red'})
#         plt.pause(0.1)
#
# plt.show()
