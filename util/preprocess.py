import torch
from torch.utils.data import Dataset

class TaxiDataset(Dataset):
    def __init__(self, x, y, loc, time):
        self.features = torch.from_numpy(x)
        self.labels = torch.from_numpy(y)
        self.locations = torch.from_numpy(loc)
        self.times = torch.from_numpy(time)
        
    def __getitem__(self,index):
        return self.features[index], self.labels[index], self.locations[index], self.times[index]
        
    def __len__(self):
        return len(self.labels)
