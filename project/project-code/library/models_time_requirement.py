from re import I
import torch 
import torch.nn as nn

DROPOUT = 0.4


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1, 3)


class Shallow_CNN_time(nn.Module): 
    def __init__(self, time_length): 
        super(Shallow_CNN_time, self).__init__()
        self.TL = time_length

        self.conv0 = nn.Conv2d(self.TL, 1000, (1, 1))
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 40, (1, 25), stride= 1)
        self.fc1 = nn.Linear(880, 40)
        self.avgpool = nn.AvgPool1d(75, stride= 15)
        self.fc2 = nn.Linear(2440, 4)
        self.softmax = nn.Softmax(dim= 1)

    def forward(self, x): 
        # Here N_new can be seen as "-1"
        x = x.view(-1, 1, 22, self.TL) # Reshape (N, 22, self.TL) -> (N_new, 1, 22, self.TL)

        # convert all the input time series into the same size
        x = x.permute(0, 3, 2, 1)
        x = self.conv0(x)
        x = x.permute(0, 3, 2, 1)
        x = self.relu(x)

        x = self.conv1(x) # (N, 40, 22, self.TL - 24)
        X = self.relu(x)
        x = x.permute(0, 3, 1, 2) #(N, self.TL - 24, 20, 22)
        x = x.view(-1, 976, 880)
        x = self.fc1(x) # (-1, 976, 40)
        x = self.relu(x)
        x = torch.square(x) 
        x = x.permute(0, 2, 1) #(-1, 40, 976)
        x = self.avgpool(x) # (-1, 40, 61)
        x = torch.log(x) 
        x = x.reshape(-1, 40 * 61)
        x = self.fc2(x) # (2440) -> (4) categories
        x = self.softmax(x) # softmax output

        return x

class CNN_time(nn.Module):
    
    def __init__(self, time_period):
        super(CNN_time, self).__init__()


        self.tp = time_period
        self.convert = nn.Conv2d(self.tp, 1000, kernel_size= (1, 1))
        self.elu = nn.ELU()
        self.bn = nn.BatchNorm2d(1000)


        self.cnn = nn.Sequential(

           # Input: N x 1 x 22 x 100
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
        # convert input data into the same size 
        x = x.view(-1, 1, 22, self.tp)
        x = x.permute(0, 3, 2, 1)
        x = self.convert(x)
        x = self.elu(x)
        x = self.bn(x)
        x = x.permute(0, 3, 2, 1)

        # CNN
        x = self.cnn(x)

        N, C, H, W = x.size()

        x = x.view(N, H, W).permute(0, 2, 1)
        
        # Fully Connected Layer
        out = self.fc(x[:, -1, :])

        return out    