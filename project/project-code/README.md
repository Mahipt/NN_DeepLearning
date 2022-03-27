Project Name: Convolution Recurrent Neural Network for EGG 4-class Motor Classification. 

The goal of this project is to find the best classifier to classify EGG data into four categories. We implement CNN, CRNN, and RNN to classify the data, and the result shows that CRNN has the best performance. 


CNN, RNN, and CRNN Classifier for EGG data 
We use pytorch frame work to create each neural networks. 	
Keyrequirement: torch, torch.nn, torchvision, jupyter

Content 
project.ipnb: The main script to load data, data processing, and training model.
library folder: 
	data_preprocess.py: Including loading, filtering, and smoothing data.
	models.py: CNN, GRU, LSTM, CNN_GRU, CNN_LSTM, CNN_version1 models
	solver.py: Functions for training and testing model. 
	utils.py: Function for loading numpy data file, filtering numpy data, and smoothing numpy data. 
Best Models folder: Include all lowest lost model in each different neural network structures. 
data folder: EEG data. 


Usage: Run the code directly in the project.ipyb file. We run the code on both google Collab and VScode (using Pyenv and GPU) 



