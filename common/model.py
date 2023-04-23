# 2023.0420.0253 @Brian

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_SR(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, batch_size, num_classes):
        super(LSTM_SR, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.lstm1 = nn.LSTM(input_dim, self.hidden_dim, self.num_layers)
        self.lstm2 = nn.LSTM(input_dim, self.hidden_dim, self.num_layers)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
        
    def forward(self, video_frames):
        lstm_in = video_frames.view(len(video_frames), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(lstm_in, self.hidden)
        output = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return output
    
class CNN_SR(nn.Module):

    def __init__(self):
        super().__init__()
        

    def forward():

        return None