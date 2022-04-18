class TaxiCNN(nn.Module):
    def __init__(
        self,
        
    ):
        super(TaxiCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, out_channels=16, kernel_size=2, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, out_channels=32, kernel_size=2, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(352, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
