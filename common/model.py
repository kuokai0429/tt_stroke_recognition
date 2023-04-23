# 2023.0420.0253 @Brian

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_SR(nn.Module):

    def __init__(self, input_feature_dim, hidden_feature_dim, num_layers, batch_size, num_classes, dropout):
        super(LSTM_SR, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.hidden_feature_dim = hidden_feature_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=input_feature_dim, hidden_size=hidden_feature_dim, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_feature_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.hidden = self.init_hidden() # (hidden state, cell state)

    def init_hidden(self):
        return (torch.zeros(self.hidden_layer_num, self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.hidden_layer_num, self.batch_size, self.hidden_dim).cuda())
        
    def forward(self, input):
        input = input.view(len(input), self.batch_size, -1)
        output, self.hidden = self.lstm(output, self.hidden)
        output = self.linear(output[-1].view(self.batch_size, -1))
        return output
    
class CNN_SR(nn.Module):

    def __init__(self):
        super().__init__()


    def forward():

        return None