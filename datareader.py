import os
import numpy as np
import pandas as pd
import tkinter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from FastDataLoader import FastTensorDataLoader
from sklearn.model_selection import train_test_split
import torch

def importData(directory, x_range, y_range):
    # Import raw data into python, should be either for training set or evaluation set
    train_data_files = []
    for file in os.listdir(os.path.join(directory)):
        if file.endswith('.csv'):
            train_data_files.append(file)
    print(train_data_files)
    ftr = []
    lbl = []
    for file_name in train_data_files:
        # Import full arrays
        ftr_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',',header = None, usecols=x_range)
        lbl_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',',header = None, usecols=y_range)
        # Append each data point to ftr and lbl
        for params, curve in zip(ftr_array.values, lbl_array.values):
            ftr.append(params)
            lbl.append(curve)
    ftr = np.array(ftr, dtype='float32')
    lbl = np.array(lbl, dtype='float32')
    return ftr, lbl

# check that the data we're using is distributed uniformly and generate some plots
def check_data(input_directory, col_range=range(0, 8), col_names=('r1','r2','r3','r4','h1','h2','h3','h4')):
    for file in os.listdir(input_directory):
        if file.endswith('.csv'):
            print('\n histogram for file {}'.format(os.path.join(input_directory, file)))
            with open(os.path.join(input_directory, file)) as f:
                data = pd.read_csv(f, header=None, delimiter=',', usecols=col_range,
                                   names=col_names)
                for name in col_names:
                    print('{} unique values for {}: {}'.format(len(data[name].unique()),
                                                               name,
                                                               np.sort(data[name].unique()))
                          )
                hist = data.hist(bins=13, figsize=(10, 5))
                plt.tight_layout()
                plt.show()
                print('done plotting column data')


def read_data( x_range, y_range, geoboundary,  batch_size=128,
                 data_dir=os.path.abspath(''), rand_seed=1234, normalize_input = True, test_ratio = 0.2):
    """
      :param input_size: input size of the arrays
      :param output_size: output size of the arrays
      :param x_range: columns of input data in the txt file
      :param y_range: columns of output data in the txt file
      :param cross_val: number of cross validation folds
      :param val_fold: which fold to be used for validation
      :param batch_size: size of the batch read every time
      :param shuffle_size: size of the batch when shuffle the dataset
      :param data_dir: parent directory of where the data is stored, by default it's the current directory
      :param rand_seed: random seed
      :param test_ratio: if this is not 0, then split test data from training data at this ratio
                         if this is 0, use the dataIn/eval files to make the test set
      """

    # Import data files
    print('Importing data files...')

    ftrTrain, lblTrain = importData(os.path.join(data_dir, 'dataIn'), x_range, y_range)
    if (test_ratio > 0):
        print("Splitting data into training and test sets with a ratio of:", str(test_ratio))
        ftrTrain, ftrTest, lblTrain, lblTest = train_test_split(ftrTrain, lblTrain,
                                                                test_size=test_ratio, random_state=rand_seed)
        print('Total number of training samples is {}'.format(len(ftrTrain)))
        print('Total number of test samples is {}'.format(len(ftrTest)))
        print('Length of an output spectrum is {}'.format(len(lblTest[0])))
    else:
        print("Using separate file from dataIn/Eval as test set")
        ftrTest, lblTest = importData(os.path.join(data_dir, 'dataIn', 'eval'), x_range, y_range)

    # print('Total number of training samples is {}'.format(len(ftrTrain)))
    # print('Total number of test samples is {}'.format(len(ftrTest)))
    # print('Length of an output spectrum is {}'.format(len(lblTest[0])))
    # print('downsampling output curves')
    # resample the output curves so that there are not so many output points
    if len(lblTrain[0]) > 2000:                                 # For Omar data set
        lblTrain = lblTrain[::, len(lblTest[0])-1800::6]
        lblTest = lblTest[::, len(lblTest[0])-1800::6]

    # print('length of downsampled train spectra is {} for first, {} for final, '.format(len(lblTrain[0]),
    #                                                                                    len(lblTrain[-1])),
    #       'set final layer size to be compatible with this number')

    print('Generating torch datasets')
    assert np.shape(ftrTrain)[0] == np.shape(lblTrain)[0]
    assert np.shape(ftrTest)[0] == np.shape(lblTest)[0]

    # Normalize the data if instructed using boundary
    if normalize_input:
        ftrTrain[:,0:4] = (ftrTrain[:,0:4] - (geoboundary[0] + geoboundary[1]) / 2)/(geoboundary[1] - geoboundary[0]) * 2
        ftrTest[:,0:4] = (ftrTest[:,0:4] - (geoboundary[0] + geoboundary[1]) / 2)/(geoboundary[1] - geoboundary[0]) * 2
        ftrTrain[:,4:] = (ftrTrain[:,4:] - (geoboundary[2] + geoboundary[3]) / 2)/(geoboundary[3] - geoboundary[2]) * 2
        ftrTest[:,4:] = (ftrTest[:,4:] - (geoboundary[2] + geoboundary[3]) / 2)/(geoboundary[3] - geoboundary[2]) * 2

    train_data = MetaMaterialDataSet(ftrTrain, lblTrain, bool_train= True)
    test_data = MetaMaterialDataSet(ftrTest, lblTest, bool_train= False)
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    train_loader = FastTensorDataLoader(torch.from_numpy(ftrTrain),
                                        torch.from_numpy(lblTrain), batch_size=batch_size, shuffle=True)
    test_loader = FastTensorDataLoader(torch.from_numpy(ftrTest),
                                       torch.from_numpy(lblTest), batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


class MetaMaterialDataSet(Dataset):
    """ The Meta Material Dataset Class """
    def __init__(self, ftr, lbl, bool_train):
        """
        Instantiate the Dataset Object
        :param ftr: the features which is always the Geometry !!
        :param lbl: the labels, which is always the Spectra !!
        :param bool_train:
        """
        self.ftr = ftr
        self.lbl = lbl
        self.bool_train = bool_train
        self.len = len(ftr)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.ftr[ind, :], self.lbl[ind, :]