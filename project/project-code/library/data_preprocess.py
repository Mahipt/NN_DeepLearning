import numpy as np 
from library.utils import *


class load_data: 
    def __init__(self): 
        pass 

    def train_valid_data(self, verbose= False): 
        X_train_valid = np.load("data/X_train_valid.npy")
        # labels: [0, 1, 2, 3]
        y_train_valid = np.load("data/y_train_valid.npy") - 769   

        #X_train_valid = np.expand_dims(X_train_valid, axis= -1)        
        

        if (verbose == True): 
            print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
            print ('Training/Valid target shape: {}'.format(y_train_valid.shape))

        return X_train_valid, y_train_valid

    def test_data(self, verbose= False): 
        X_test = np.load("data/X_test.npy")
        # minus 769 so that the labels would be [0, 1, 2, 3]
        y_test = np.load("data/y_test.npy") - 769

        #X_test = np.expand_dims(X_test, axis= -1)

        if (verbose == True): 
            print ('Test data shape: {}'.format(X_test.shape))
            print ('Test target shape: {}'.format(y_test.shape)) 

        return X_test, y_test

    def person_data(self, verbose= False):
        person_train_valid = np.load("data/person_train_valid.npy")
        person_test = np.load("data/person_test.npy")
        

        if (verbose == True): 
            print ('Person train/valid shape: {}'.format(person_train_valid.shape))
            print ('Person test shape: {}'.format(person_test.shape))

        return person_train_valid, person_test


def proprocess(BATCH_SIZE= 34, verbose= False): 
    # Loading data: 
    ld = load_data()
    X_train_valid, y_train_valid = ld.train_valid_data(verbose= verbose)
    X_test, y_test = ld.test_data(verbose= verbose)
    person_train_valid, person_test = ld.person_data(verbose= verbose)
    if (verbose == True): 
        print ("====================")
        # count each subjects trials in train/valid
        count = np.zeros(9)
        for i in person_train_valid:
                count[int(i)] += 1.0
        print ("person_train_valid each sub count: ")
        print (count)
        # count each subjects trials in test
        count = np.zeros(9)
        for i in person_test:
            count[int(i)] += 1.0
        print ("person_test each sub count: ")
        print (count)
    
    #Start Preprocessing 
    # filter inputs
    X_train_valid = filter_data(X_train_valid, fs=250, order=6, lowcut=7, highcut=30)
    X_test = filter_data(X_test, fs=250, order=6, lowcut=7, highcut=30)

    # smooth inputs
    X_train_valid = smooth_data(X_train_valid, ws=5)
    X_test = smooth_data(X_test, ws=5)

    # merge dataset to feed into next step
    dataset_train = X_train_valid, y_train_valid, person_train_valid
    dataset_test = X_test, y_test, person_test

    # split data 
    # Toggle verbose [True/False] to choose if you want the content printed
    X_train_valid_subs, y_train_valid_subs = split_dataset(dataset_train, verbose = verbose, mode = 'train')
    X_test_subs, y_test_subs = split_dataset(dataset_test, verbose = verbose, mode = 'test')

    return X_train_valid_subs, y_train_valid_subs, X_test_subs, y_test_subs, dataset_train, dataset_test
