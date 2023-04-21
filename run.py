# 2023.0417.1548 @Brian

from datetime import datetime
import time
import os
import glob
import re
import random
from time import time
from common.logging import Logger
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

from common.model import LSTM_SR

def train_history_graphic(history, history_key1, history_key2, y_label) :
    
	plt.plot( history.history[history_key1] )
	plt.plot( history.history[history_key2] )
	plt.title( 'train history' )
	plt.xlabel( 'epochs' )
	plt.ylabel( y_label )
	plt.legend( ['train', 'validate'], loc = 'upper left')
	plt.savefig('./%s_v2.png' %(y_label))
	plt.show()
	plt.close()


def getTrainData():

    return None


def train_model(model, train_features, train_labels, num_epochs, learning_rate, optimizer=None):
    
    since = time.time()
    if optimizer == None:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        model = model.cuda()

    loss_function = nn.CrossEntropyLoss()
    total_step = len(train_features)
    model.train()

    for epoch in range(num_epochs):

        data = list(zip(train_features, train_labels))
        random.shuffle(data)
        train_features, train_labels = zip(*data)

        correct = 0
        total = 0

        for i, (video, label) in enumerate(zip(train_features, train_labels)):

            # -1 here because loss function requires this to be between (0, num_classes]
            label = label.type(torch.LongTensor).view(-1) - 1

            if torch.cuda.is_available():
                video, label = video.cuda(), label.cuda()

            model.zero_grad()        
            model.hidden = model.init_hidden()

            predictions = model(video)
            loss = loss_function(predictions, label) 
            loss.backward()
            optimizer.step()

            # Track the accuracy
            _, predicted = torch.max(predictions.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            # if i != 0 and  i % (50) == 0:
            #   print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
            #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
            #                   (correct / total) * 100))

        print('Training Accuracy for epoch {}: {:.3f}%'.format(epoch + 1, (correct / total) * 100))

    elapsed = time.time() - since
    print('Train time elapsed in seconds: ', elapsed)

    return (correct / total) * 100


def test_model(model, test_features, test_labels):

    since = time.time()
    model.eval()

    with torch.no_grad():

        correct = 0
        total = 0

        for video, label in zip(test_features, test_labels):
            # -1 here because loss function during training required this to be between (0, num_classes]
            label = label.type(torch.LongTensor).view(-1) - 1

            if torch.cuda.is_available():
                # Move to GPU
                video, label = video.cuda(), label.cuda()

            outputs = model(video)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        print('Test Accuracy of the model on test images: {} %'.format((correct / total) * 100))
    
    elapsed = time.time() - since
    print('Test time elapsed in seconds: ', elapsed)

    return (correct / total) * 100


# def predVisualize(i, pred_mask):

#   global mp4, data_url

#   cap = cv2.VideoCapture("/content/gdrive/MyDrive/專題, 論文, 實驗室/高中夏令營 - AI4kids/學生影片/student_TrainData/1/20220811_161414/openpose/media/output.avi")
#   output = cv2.VideoWriter('PredictionResult_' + str(i) + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 
#                float(30), (int(cap.get(3)), int(cap.get(4))))
#   classes = ['Forehand Drive', 'Backhand Drive', 'Backhand Push', 'None']
#   color = [(255, 0, 0), (0, 100, 0), (0, 0, 255), (0, 0, 0)]

#   count = 0
#   while(cap.isOpened()):
      
#       count += 1
#       ret, frame = cap.read()
      
#       if ret == True:
#         cv2.rectangle(frame, (40, 10), (1000, 60), (255, 255, 255), -1, cv2.LINE_AA)
#         # cv2.putText(frame, "Frame: " + str(count) + " Predictions: " + str(classes[pred[i-1][0][0]]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
#         #       1.5, (0, 0, 0), 5, cv2.LINE_4)
#         cv2.putText(frame, "Predictions: " + str(classes[pred_mask[count-1]]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
#               1.5, color[pred_mask[count-1]], 5, cv2.LINE_4)
        
#         output.write(frame)
#       else:
#         break 

#   cap.release()
#   output.release()

#   !ffmpeg -i PredictionResult_{i}.avi PredictionResult_{i}.mp4

#   mp4 = open('PredictionResult_' + str(i) + '.mp4','rb').read()
#   data_url = "data:video/mp4;base64," + b64encode(mp4).decode()


if __name__ == "__main__":

    # Fetch Training Data.
    (X_All, y_All) = getTrainData()
    X_train, X_test, y_train, y_test = train_test_split(X_All, y_All, test_size=0.1, random_state=0)
    print(X_All.shape, y_All.shape)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Train Model.
    num_epochs = 5
    model = LSTM_SR(input_dim=4096, hidden_dim=512, num_layers=1, 
                        batch_size=1, num_classes=25)
    training_accuracy = train_model(model, X_train, y_train, num_epochs, 0.05)

    # Evaluate Model.
    test_accuracy = test_model(model, X_test, y_test)
    print('Training accuracy is %2.3f :' %(training_accuracy) )
    print('Test accuracy is %2.3f :' %(test_accuracy) )

    # Inference on Test Data.

