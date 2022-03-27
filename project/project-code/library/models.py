from unicodedata import bidirectional
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


DROPOUT = 0.4



class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1, 3)

class Shallow_CNN(nn.Module): 
    def __init__(self): 
        super(Shallow_CNN, self).__init__()
        self.elu = nn.ELU()
        self.conv1 = nn.Conv2d(1, 40, (1, 25), stride= 1)
        self.fc1 = nn.Linear(880, 40)
        self.avgpool = nn.AvgPool1d(75, stride= 15)
        self.fc2 = nn.Linear(2440, 4)
        self.softmax = nn.Softmax(dim= 1)

    def forward(self, x): 
        x = x.view(-1, 1, 22, 1000)
        x = self.conv1(x)
        x = self.elu(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(-1, 976, 880)
        x = self.fc1(x)
        x = torch.square(x)
        x = x.permute(0, 2, 1)
        x = self.avgpool(x)
        x = torch.log(x)
        x = x.reshape(-1, 40 * 61)
        x = self.fc2(x)
        x = self.softmax(x)

        return x 

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(

           # Input: N x 1 x 22 x 1000

            ### Conv-Pool Block 1
            # Convolution (temporal)
            nn.Conv2d(1, 128, kernel_size=(1, 10), stride=1, padding=0),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(64, 32, kernel_size=(1, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(32, 25, kernel_size=(18, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            
            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 2
            # Convolution
            nn.Conv2d(1, 50, kernel_size=(25, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 3
            # Convolution
            nn.Conv2d(1, 100, kernel_size=(50, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 4
            # Convolution
            nn.Conv2d(1,200, kernel_size=(100, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features= 200, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )
        self.fc = nn.Sequential(
            nn.Linear(200, 54),
            nn.BatchNorm1d(num_features=54, eps=1e-05, momentum=0.2, affine=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(54, 44),
            nn.BatchNorm1d(num_features=44, eps=1e-05, momentum=0.2, affine=True),
            nn.ReLU(inplace = True),
            nn.Linear(44, 4)
        )

    def forward(self, x):
        
        # CNN
        x = x.view(-1, 1, 22, 1000)
        x = self.cnn(x)

        N, C, H, W = x.size()

        x = x.view(N, H, W).permute(0, 2, 1)
        
        # Fully Connected Layer
        out = self.fc(x[:, -1, :])

        return out


class CNN_GRU_v1(nn.Module):
    
    def __init__(self):
        super(CNN_GRU_v1, self).__init__()

        self.cnn = nn.Sequential(

           # Input: N x 1 x 22 x 1000

            ### Conv-Pool Block 1
            # Convolution (temporal)
            nn.Conv2d(1, 16, kernel_size=(1, 10), stride=1, padding=0),
            nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(16, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(64, 32, kernel_size=(1, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(32, 25, kernel_size=(18, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            
            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 2
            # Convolution
            nn.Conv2d(1, 50, kernel_size=(25, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 3
            # Convolution
            nn.Conv2d(1, 100, kernel_size=(50, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 4
            # Convolution
            nn.Conv2d(1,200, kernel_size=(100, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features= 200, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )

        self.gru = nn.GRU(200, 32, 2, batch_first = True, bidirectional= True) 
        
        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        
        # CNN
        x = x.view(-1, 1, 22, 1000)
        x = self.cnn(x)

        N, C, H, W = x.size()

        x = x.view(N, H, W).permute(0, 2, 1)
        out = x
        #x, _ = self.gru(x) 
        # Fully Connected Layer
        #out = self.fc(x[:, -1, :])

        return out


class CNN_GRU_v2(nn.Module):
    
    def __init__(self):
        super(CNN_GRU_v2, self).__init__()

        self.cnn = nn.Sequential(

           # Input: N x 1 x 22 x 1000

            ### Conv-Pool Block 1
            # Convolution (temporal)
            nn.Conv2d(1, 128, kernel_size=(1, 10), stride=1, padding=0),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(64, 32, kernel_size=(1, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(32, 25, kernel_size=(18, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            
            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 2
            # Convolution
            nn.Conv2d(1, 50, kernel_size=(25, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 3
            # Convolution
            nn.Conv2d(1, 100, kernel_size=(50, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 4
            # Convolution
            nn.Conv2d(1,200, kernel_size=(100, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features= 200, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )

        self.gru = nn.GRU(200, 32, 2, batch_first = True, bidirectional= True) 
        
        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        
        # CNN
        x = x.view(-1, 1, 22, 1000)
        x = self.cnn(x)

        N, C, H, W = x.size()

        x = x.view(N, H, W).permute(0, 2, 1)
        x, _ = self.gru(x) 
        # Fully Connected Layer
        out = self.fc(x[:, -1, :])

        return out

class CNN_LSTM(nn.Module):
    
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        self.cnn = nn.Sequential(

           # Input: N x 1 x 22 x 1000

            ### Conv-Pool Block 1
            # Convolution (temporal)
            nn.Conv2d(1, 128, kernel_size=(1, 10), stride=1, padding=0),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(64, 32, kernel_size=(1, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(32, 25, kernel_size=(18, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            
            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 2
            # Convolution
            nn.Conv2d(1, 50, kernel_size=(25, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 3
            # Convolution
            nn.Conv2d(1, 100, kernel_size=(50, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 4
            # Convolution
            nn.Conv2d(1,200, kernel_size=(100, 10), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features= 200, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )

        self.lstm = nn.LSTM(200, 32, 2, batch_first= True, dropout = DROPOUT, bidirectional= True)


        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        
        # CNN
        x = x.view(-1, 1, 22, 1000)
        x = self.cnn(x)

        N, C, H, W = x.size()

        x = x.view(N, H, W).permute(0, 2, 1)
        x, _ = self.lstm(x) 
        # Fully Connected Layer
        out = self.fc(x[:, -1, :])
        return out 



class GRU(nn.Module):
    
    def __init__(self):
        super(GRU, self).__init__()

        self.input_dim = 22
        self.hidden_dim = 64
        self.layer_dim = 3
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout=DROPOUT)

        self.fc = nn.Sequential(
            nn.Linear(64, 54),
            nn.BatchNorm1d(num_features=54, eps=1e-05, momentum=0.2, affine=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(54, 44),
            nn.BatchNorm1d(num_features=44, eps=1e-05, momentum=0.2, affine=True),
            nn.ReLU(inplace = True),
            nn.Linear(44, 4)
        )
    
    def forward(self, x, h=None):
        # GRU
        #print(x.size())
        N, H, W = x.size()
        x = x.view(N, H, W).permute(0, 2, 1)
        out, _ = self.gru(x)

        # Fully Connected Layer
        out = self.fc(out[:, -1, :])

        return out



class LSTM(nn.Module):

  def __init__(self): 
    super(LSTM, self).__init__()
    dropout = 0.4
    input_dim = 22
    hidden_dim = 64
    layer_dim = 3

    # LSTM layer
    self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, dropout=dropout, batch_first=True) # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, input_dim)
    # output layer
    self.fc = nn.Linear(hidden_dim, 4)
    
  def forward(self, x):
    x = x.permute(0,2,1) # (B, 1000, 22)
    x, _ = self.lstm(x)
    x = self.fc(x[:, -1, :]) # (B, 4)
    return x

    