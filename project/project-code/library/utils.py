import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms 
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

class Dataset(Dataset):

    def __init__(self, subset, transform = None):
        self.subset = subset
        self.transformation = transform
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        return x, y

    def __len__(self):
        return len(self.subset)

def split_dataset(dataset, verbose = False, mode = 'train'):
    
    mode_name = 'Training/Valid' if mode == 'train' else 'Test'
    X, y, person = dataset
    X_arr = [] 
    y_arr = [] 

    for i in range(9):
        X_arr.append(X[np.where(person == i)[0]]) 
        y_arr.append(y[np.where(person == i)[0]]) 

    if verbose:
        j, k = 0, 0
        for X in X_arr:
            print (mode_name, 'subject shape: {}'.format(X.shape))
            j += 1
        for y in y_arr:
            print (mode_name, 'subject target shape: {}'.format(y.shape))
            k += 1

    return X_arr, y_arr


# def split_dataset(dataset, verbose = False, mode = 'train'):
    
#     mode_name = 'Training/Valid' if mode == 'train' else 'Test'
#     X, y, person = dataset

#     X_train_valid_subjects = np.empty(shape=[0, X.shape[1], X.shape[2]])
#     y_train_valid_subjects = np.empty(shape=[0])

#     for i in [2, 4]:
#         X_train_valid_subject = X[np.where(person == i)[0], :, :]
#         y_train_valid_subject = y[np.where(person == i)[0]]

#         X_train_valid_subjects = np.concatenate((X_train_valid_subjects, X_train_valid_subject), axis=0)
#         y_train_valid_subjects = np.concatenate((y_train_valid_subjects, y_train_valid_subject))

#     if verbose:
#         print (mode_name, 'total shape: {}'.format(X_train_valid_subjects.shape))
#         print (mode_name, 'total target shape: {}'.format(y_train_valid_subjects.shape))

#     return X_train_valid_subjects, y_train_valid_subjects


def np_to_tensor(dataset, verbose = False, mode = 'train'):
    
    mode_name = 'Training/Valid' if mode == 'train' else 'Test'
    X, y, person = dataset
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float().long()

    if verbose:
        print (mode_name, 'tensor shape: {}'.format(X_tensor.shape))
        print (mode_name, 'target tensor shape: {}'.format(y_tensor.shape))
        print ()

    return X_tensor, y_tensor

def np_to_tensor_subs(X_subs, y_subs, verbose = True, mode = 'train'):
    
    mode_name = 'Training/Valid' if mode == 'train' else 'Test'
    #X_tensor_subs = np.zeros(0)
    X_tensor_subs = []
    y_tensor_subs = [] 
    i, j = 0, 0

    for X in X_subs:
        X_tensor_subs.append(torch.from_numpy(X).float())
        #X_tensor_subs = np.append(X_tensor_subs, X)
        if verbose: 
            print (mode_name, 'subject tensor shape: {}'.format(X_tensor_subs[i].shape))
        i += 1
    #X_tensor_subs = torch.tensor(np.array(X_tensor_subs, dtype= np.float64))

    for y in y_subs:
        y_tensor_subs.append(torch.from_numpy(y).float().long())
        #y_tensor_subs = np.append(y_tensor_subs, y)
        if verbose: 
            print (mode_name, 'subject target tensor shape: {}'.format(y_tensor_subs[j].shape))
        j += 1
    #y_tensor_subs = torch.tensor(np.array(y_tensor_subs, dtype= np.float64)) 

    #X_tensor_subs = torch.FloatTensor(X_tensor_subs)
    #y_tensor_subs = torch.FloatTensor(y_tensor_subs)

    return X_tensor_subs, y_tensor_subs

def data_load(X_train_valid_tensor, y_train_valid_tensor, X_test_tensor, y_test_tensor, batch_size):

    # create dataloader for orignal data
    init_dataset = TensorDataset(X_train_valid_tensor, y_train_valid_tensor) 
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # split train and val
    lengths = [int(len(init_dataset)*0.8), int(len(init_dataset)*0.2)] 
    subset_train, subset_val = random_split(init_dataset, lengths) 

    train_data = Dataset(subset_train, transform=None)
    val_data = Dataset(subset_val, transform=None)


    # create dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0),
        'val': torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, num_workers=0),
        'test': test_dataset
    }
    

    return dataloaders

def data_subs_load(X_train_valid_tensor_subs, y_train_valid_tensor_subs, X_test_tensor_subs, y_test_tensor_subs, batch_size):

    dataloaders_subs = []
    test_data_subs = []

    for i in range(9):
        # create dataloader for orignal data
        init_dataset = TensorDataset(X_train_valid_tensor_subs[i], y_train_valid_tensor_subs[i]) 
        test_dataset = TensorDataset(X_test_tensor_subs[i], y_test_tensor_subs[i])

        # split train and val
        lengths = [int(len(init_dataset)*0.8), len(init_dataset) - int(len(init_dataset)*0.8)] 
        subset_train, subset_val = random_split(init_dataset, lengths) 

        train_data = Dataset(subset_train, transform=None)
        val_data = Dataset(subset_val, transform=None)


        # create dataloaders
        dataloaders_subs.append({
            'train': torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 0),
            'val': torch.utils.data.DataLoader(val_data, batch_size= batch_size, shuffle = False, num_workers = 0), 
            'test': test_data_subs.append(Dataset(test_dataset, transform=None))
        })
        
        

    return dataloaders_subs

def filter_data(data, fs, order, lowcut, highcut):

    filtered_data = np.zeros_like(data)
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq

    for n in np.arange(data.shape[0]):
        single_instance = data[n, :, :]

        for channel in np.arange(single_instance.shape[0]):
            X = single_instance[channel, :]
            b, a = signal.butter(order, [low, high], btype='band')
            y = signal.lfilter(b, a, X)
            filtered_data[n, channel, :] = y

    return filtered_data


def smooth_data(data, ws):
    kern = signal.hanning(ws)[None, None, :]
    kern /= kern.sum()
    return signal.convolve(data, kern, mode='same')