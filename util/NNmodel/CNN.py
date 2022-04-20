#view the sequence faeture of a picture of channel one.

import torch.nn as nn
import torch

feature_size = 49 # at time t, region (x,y) the 49 features
result_num = 1 # numerical value represting people num
proj_size = 1
dense_size = 64

class TaxiCNN(nn.Module):
    def __init__(
        self,
        dense_dim=dense_size,
    ):
        super(TaxiCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),#use out_channels filters to analysis the graph
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.mlp = nn.Sequential(
            nn.Linear(32*11+2+1, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, result_num)
        )
        
    def forward(self, features, locations, times):
        feature = features.unsqueeze(1)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = feature.reshape(feature.size(0), -1)
        times = times.unsqueeze(1)
        combine = torch.cat((feature, locations, times), axis=1)
        out = self.mlp(combine)
        return out

#to test the NN 
if __name__ == "__main__":
    a = torch.ones((4,49,8))
    b = torch.ones((4,1))
    c = torch.ones((4,2))
    model = CNN()

    s = model(a,b,c)
    
    print(s)