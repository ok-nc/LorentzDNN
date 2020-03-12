# # Scratch file to learn Pytorch basics
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from datareader import MetaMaterialDataSet

# Visualize our data
import matplotlib.pyplot as plt
import numpy as np

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        x = self.layer(x)
        return x

# Returns single e2 simulation for given lorentzian parameters
def Lorentzian_Sim(w0, wp, g):

    freq_low = 0
    freq_high = 5
    num_freq = 300
    w = np.arange(freq_low, freq_high, (freq_high-freq_low)/num_freq)

    e1 = np.divide(np.multiply(np.power(wp, 2), np.add(np.power(w0, 2), -np.power(w, 2))),
                      np.add(np.power(np.add(np.power(w0, 2), -np.power(w, 2)), 2),
                             np.multiply(np.power(w, 2), np.power(g, 2))))

    e2 = np.divide(np.multiply(np.power(wp, 2), np.multiply(w, np.power(g, 2))),
             np.add(np.power(np.add(np.power(w0, 2), -np.power(w, 2)), 2),
                    np.multiply(np.power(w, 2), np.power(g, 2))))

    return w, e2

# Generates randomized dataset of simulated spectra for training and testing
def Prepare_Data():

    batch_size = 1024

    features, labels = None

    ftrTrain, ftrTest, lblTrain, lblTest = train_test_split(features, labels, test_size=0.2, random_state=1234)
    train_data = MetaMaterialDataSet(ftrTrain, lblTrain, bool_train= True)
    test_data = MetaMaterialDataSet(ftrTest, lblTest, bool_train= False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader


# # # # Network creation and training code below

net = Network()
print(net)

cwd = 'C:/Users/labuser/mlmOK_Pytorch/Pytorch learning'
os.chdir(cwd)

# Create random test data
w,e2 = Lorentzian_Sim(1, 2.5, 0.1)

# Visualize data
plt.plot(w, e2)
plt.show()


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
