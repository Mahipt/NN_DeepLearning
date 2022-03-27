import numpy as np
import os 
import matplotlib.pyplot as plt
import time 

from livelossplot import PlotLosses

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def train_model(model, model_name, optimizer, num_epochs, dataloaders):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # for each epoch... 
    liveloss = PlotLosses()
    loss_fn = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')
    best_valid_acc = float(0)
    for epoch in range(num_epochs):
      print('Epoch {}/{}'.format(epoch, num_epochs - 1))
      print('-' * 10)
      logs = {}

      # let every epoch go through one training cycle and one validation cycle
      # TRAINING AND THEN VALIDATION LOOP...
      for phase in ['train', 'val']:
        train_loss = 0
        correct = 0
        total = 0
        batch_idx = 0

        start_time = time.time()
        # first loop is training, second loop through is validation
        # this conditional section picks out either a train mode or validation mode
        # depending on where we are in the overall training process
        # SELECT PROPER MODE- train or val
        if phase == 'train':
          for param_group in optimizer.param_groups:
            print("LR", param_group['lr']) # print out the learning rate
          model.train()  # Set model to training mode
        else:
          model.eval()   # Set model to evaluate mode
        
        for inputs, labels in dataloaders[phase]:
          inputs = inputs.to(device)
          labels = labels.to(device)
          batch_idx += 1
          
          optimizer.zero_grad()
          
          with torch.set_grad_enabled(phase == 'train'):
          #    the above line says to disable gradient tracking for validation
          #    which makes sense since the model is in evluation mode and we 
          #    don't want to track gradients for validation)
            outputs = model(inputs)
            # compute loss where the loss function will be defined later
            
            loss = loss_fn(outputs, labels)
            # backward + optimize only if in training phase
            if phase == 'train':
              loss.backward()
              optimizer.step()
            train_loss += loss
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # if phase == 'train':
        #   if  epoch%5 == 0:
        #   # prints for training and then validation (since the network will be in either train or eval mode at this point) 
        #     print(" Training Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

        # if phase == 'val' and epoch%5 == 0:
        #   print(" Validation Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))


        prefix = ''
        cur_loss = train_loss.item()/(batch_idx)
        acc = correct/total*100.
        if phase == 'val':
            prefix = 'val_'

            if cur_loss < best_valid_loss:
                best_valid_loss = cur_loss
                path = "../project/Best_Models/"
                path = os.path.join(path, model_name + ".pt") 
                torch.save(model, path)
            '''
            if acc > best_valid_acc: 
                best_valid_acc = acc  
                path = "../project/Best_Models/"
                path = os.path.join(path, model_name + ".pt") 
                torch.save(model, path)
            ''' 
        logs[prefix + 'loss'] = cur_loss
        logs[prefix + 'acc'] = correct/total*100.

      liveloss.update(logs)
      liveloss.send()

    # end of single epoch iteration... repeat of n epochs  
    return model

def train_model_time(model, model_name, optimizer, num_epochs, dataloaders):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # for each epoch... 
    loss_fn = nn.CrossEntropyLoss()
    #liveloss = PlotLosses()

    best_valid_loss = float('inf')
    best_valid_acc = float(0)
    best_train_acc = float(0)
    for epoch in range(num_epochs):
      #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
      #print('-' * 10)
      #logs = {}

      # let every epoch go through one training cycle and one validation cycle
      # TRAINING AND THEN VALIDATION LOOP...
      for phase in ['train', 'val']:
        train_loss = 0
        correct = 0
        total = 0
        batch_idx = 0

        start_time = time.time()
        # first loop is training, second loop through is validation
        # this conditional section picks out either a train mode or validation mode
        # depending on where we are in the overall training process
        # SELECT PROPER MODE- train or val
        if phase == 'train':
          for param_group in optimizer.param_groups:
            #print("LR", param_group['lr']) # print out the learning rate
            pass
          model.train()  # Set model to training mode
        else:
          model.eval()   # Set model to evaluate mode
        
        for inputs, labels in dataloaders[phase]:
          inputs = inputs.to(device)
          labels = labels.to(device)
          batch_idx += 1
          
          optimizer.zero_grad()
          
          with torch.set_grad_enabled(phase == 'train'):
          #    the above line says to disable gradient tracking for validation
          #    which makes sense since the model is in evluation mode and we 
          #    don't want to track gradients for validation)
            outputs = model(inputs)
            # compute loss where the loss function will be defined later
            
            loss = loss_fn(outputs, labels)
            # backward + optimize only if in training phase
            if phase == 'train':
              loss.backward()
              optimizer.step()
            train_loss += loss
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # if phase == 'train':
        #   if  epoch%5 == 0:
        #   # prints for training and then validation (since the network will be in either train or eval mode at this point) 
        #     print(" Training Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

        # if phase == 'val' and epoch%5 == 0:
        #   print(" Validation Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))


        prefix = ''
        cur_loss = train_loss.item()/(batch_idx)
        acc = correct/total*100.

        if phase == 'train': 
            if best_train_acc < acc: 
                best_train_acc = acc 

        if phase == 'val':
            prefix = 'val_'

            if cur_loss < best_valid_loss:
                best_valid_loss = cur_loss
                path = "../project/Best_Models/"
                path = os.path.join(path, model_name + ".pt") 
                torch.save(model, path)
            
            if acc > best_valid_acc: 
                best_valid_acc = acc  
        #logs[prefix + 'loss'] = cur_loss
        #logs[prefix + 'acc'] = correct/total*100.

      #liveloss.update(logs)
      #liveloss.send()

    # end of single epoch iteration... repeat of n epochs  
    return best_train_acc, best_valid_acc 


def test_model(model,test_data,criterion):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # for each epoch... 
    
    # Creating the test dataloader
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True, num_workers=0)
    
    # Making the predictions on the dataset
    
    total_test_preds = 0
    correct_test_preds = 0
    test_loss = 0
    
    model.eval()
    with torch.no_grad():
        
        for test_inputs, test_labels in test_dataloader:
            
            # Transfer test data and labels to device
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            
            # Perform forward pass
            test_outputs = model(test_inputs)
            
            # Compute loss
            test_loss = criterion(test_outputs,test_labels)
            
            # Compute test statistics
                    
            test_loss += test_loss.item()
            _, test_predicted = test_outputs.max(1)
            total_test_preds += test_labels.size(0)
            correct_test_preds += test_predicted.eq(test_labels).sum().item()
            
        test_acc = correct_test_preds/total_test_preds
        #print('Test loss', test_loss)
        #print('Test accuracy',test_acc*100)
        
    
    return test_acc