# Hyper-parameter
feature_size = 49 # at time t, region (x,y) the 49 features
result_num = 1 # numerical value represting people num
hidden_size = 100
proj_size = 1
dense_size = 64
n_layers = 2

#try to decrise the demension of the feature sequence
class TaxiLSTM(nn.Module):
    def __init__(
        self,
        feature_size=feature_size,
        hidden_dim=hidden_size,
        proj_dim=proj_size,
        dense_dim=dense_size,
        n_layers=n_layers,
    ):
        super(TaxiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=feature_size, 
            hidden_size=hidden_dim, 
            proj_size=proj_dim, 
            batch_first=True, 
            num_layers=n_layers
        )
        self.gru = nn.GRU(feature_size, hidden_dim, batch_first=True, num_layers=n_layers)
        self.mlp = nn.Sequential(
            nn.Linear(8+2+1, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, result_num)
        )
        
    def forward(self, features, locations, times):
        #rnn to decrease the demension by reprsenting features with one feature
        #lstm: random initiate hidden and cell state 
        #h_0 = torch.rand((n_layers, batch_size, proj_size))
        #c_0 = torch.rand((n_layers, batch_size, proj_size))
        rnn,(_,_) = self.lstm(features)#self.lstm(features, (h_0,c_0))
        #gru
        #_,rnn = self.gru(features)
        squeeze = rnn.squeeze()
        times = times.unsqueeze(1)
        cat = torch.cat((squeeze, locations, times), axis=1)
        output = self.mlp(cat)
        return output

# class TaxiRNN(nn.Module):
#     def __init__(
#         self,
#         sequence_size,
#         embedding_dim=1,
#         hidden_dim=100,
#         dense_dim=32,
#         max_norm=2,
#         n_layers=1,
#     ):
#         super().__init__()
#         self.embedding = nn.Embedding(
#             sequence_size,
#             embedding_dim,
#             norm_type=2,
#             max_norm=max_norm,
#         )
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=n_layers)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim, dense_dim)
#             nn.ReLU()
#             nn.Linear(dense_dim, result_num)
#         )
        
#     def __forward__(self, features, locations, times):
#         embeds = self.embedding(features)
