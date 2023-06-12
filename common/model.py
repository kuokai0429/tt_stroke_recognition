# 2023.0420.0253 @Brian

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_2D_SR(nn.Module):

    def __init__(self, num_classes):

        super(CNN_2D_SR, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1) # (16, 1, 17 * 2 * window)
        # self.norm1 = nn.LayerNorm([16, 1, ]) # [B, H, W]
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1) # (32, 1, )
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1) # (64, 1, )
        self.conv4 = nn.Conv1d(64, 64, kernel_size=4, stride=2) # (64, 1, )
        # self.norm2 = nn.LayerNorm([64, 1, ])
        self.conv5 = nn.Conv1d(64, 128, kernel_size=4, stride=2) # (128, 1, )
        self.conv6 = nn.Conv1d(128, 128, kernel_size=4, stride=2) # (128, 1, )
        self.fc1 = nn.Linear(7808, 64)
        self.fc2 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()           
        self.dropout = nn.Dropout(0.25)

        
    def forward(self, x):

        output = self.relu(self.conv1(x)) 
        # output = self.norm1(self.relu(self.conv1(x)))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv3(output))
        output = self.relu(self.conv4(output)) 
        # output = self.norm2(self.relu(self.conv4(output)))
        output = self.relu(self.conv5(output))
        output = self.dropout(self.relu(self.conv6(output)))
        output = torch.flatten(output, 1) # flatten all dimensions except batch
        output = self.fc1(output)
        output = self.fc2(output)

        return output
    

class CNN_3D_SR(nn.Module):

    def __init__(self, num_classes):

        super(CNN_3D_SR, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1) # (16, 1, 17 * 3 * window)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1) # (32, 1, )
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1) # (64, 1, )
        self.conv4 = nn.Conv1d(64, 64, kernel_size=4, stride=2) # (64, 1, )
        self.conv5 = nn.Conv1d(64, 128, kernel_size=4, stride=2) # (128, 1, )
        self.conv6 = nn.Conv1d(128, 128, kernel_size=4, stride=2) # (128, 1, )
        self.fc1 = nn.Linear(11904, 64)
        self.fc2 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()           
        self.dropout = nn.Dropout(0.25)

        
    def forward(self, x):

        output = self.relu(self.conv1(x)) 
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv3(output))
        output = self.relu(self.conv4(output)) 
        output = self.relu(self.conv5(output))
        output = self.dropout(self.relu(self.conv6(output)))
        output = torch.flatten(output, 1) # flatten all dimensions except batch
        output = self.fc1(output)
        output = self.fc2(output)

        return output
