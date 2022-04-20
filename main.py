import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from util.NNmodel.LSTM import TaxiLSTM
import util.NNmodel.ConvLSTM
from util.preprocess import TaxiDataset

#training config
epochs = 80
batch_size = 72
learning_rate=0.001

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#train the model
def train_model(model, dataloader):
    model.train()
    all_loss = []
    #for _ in tqdm(range(total_step), ncols=80):
    for i, (features, labels, locations, times) in enumerate(train_dataloader):
        features = features.to(torch.float32).to(device)
        labels = labels.to(torch.float32).to(device)
        locations = locations.to(torch.float32).to(device)
        times = times.to(torch.float32).to(device)

        # Forward pass
        outputs = model(features, locations, times)
        loss = torch.sqrt(criterion(outputs, labels))
        all_loss.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return np.mean(all_loss)

# Test the model
def test_model(model, dataloader):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    all_loss = []
    with torch.no_grad():
        for features, labels, locations, times in dataloader:
            features = features.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            locations = locations.to(torch.float32).to(device)
            times = times.to(torch.float32).to(device)

            outputs = model(features, locations, times)
            loss = torch.sqrt(criterion(outputs, labels)).item()
            all_loss.append(loss)
        return np.mean(all_loss)

#read data
train_data = np.load('./train.npz')
train_features = train_data['x']#features
train_labels = train_data['y']#labels
train_locations = train_data['locations']#locations
train_times = train_data['times']#times
val_data = np.load('./val.npz')
val_features = val_data['x']#features
val_labels = val_data['y']#labels
val_locations = val_data['locations']#locations
val_times = val_data['times']#times

# dataset
train_dataset = TaxiDataset(train_features, train_labels, train_locations, train_times)
val_dataset = TaxiDataset(val_features, val_labels, val_locations, val_times)
# dataloader
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

# initialize model
model = TaxiLSTM().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    train_loss = train_model(model, train_dataloader)
    val_loss = test_model(model, val_dataloader)
    print(f'Epoch [{epoch+1}/{epochs}]: The traing set Root Main Square Error loss: {np.mean(train_loss)}. The RMSE of validation set is {val_loss}')
#with open('./time.txt',"a") as f:
#    for n in times:
#        f.write(str(n))
#    f.close()