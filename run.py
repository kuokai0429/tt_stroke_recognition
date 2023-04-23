# 2023.0417.1548 @Brian

import os
import glob
import argparse
import pickle
import re
import random
import time
from datetime import datetime
from common.logging import Logger
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from common.model import LSTM_SR
from common.model import CNN_SR


parser = argparse.ArgumentParser(description='main')
parser.add_argument('--log', default="log/run", required=False, type=str, help="Log folder.")
parser.add_argument('--model', default="cnn", required=False, type=str, help="Model.")
args = parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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


def getTrainData(folder):

    temp = []
    for i in ['X_All', 'y_All']:
        with open(f'{folder}{i}.pkl', 'rb') as handle:
            temp.append(pickle.load(handle))

    return temp


def train_lstm(model, train_features, train_labels, num_epochs, learning_rate, optimizer=None):
    
    since = time.time()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        model = model.cuda()

    loss_function = nn.CrossEntropyLoss()
    total_step = len(train_features)

    model.train() # Set the model in training mode

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

            model.zero_grad()  # 清除 lstm 上個數據的偏微分暫存值，否則會一直累加      
            model.hidden = model.init_hidden()

            predictions = model(video)
            loss = loss_function(predictions, label) 
            loss.backward()
            optimizer.step()

            # Track the accuracy
            _, predicted = torch.max(predictions.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            if i != 0:
              print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

        training_accuracy = (correct / total) * 100
        print('Training Accuracy for epoch {}: {:.3f}%'.format(epoch + 1, training_accuracy))

    elapsed = time.time() - since
    print('Train time elapsed in seconds: ', elapsed)

    return training_accuracy


def test_lstm(model, test_features, test_labels):

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

        testing_accuracy = (correct / total) * 100
        print('Test Accuracy of the model on test images: {} %'.format(testing_accuracy))
    
    elapsed = time.time() - since
    print('Test time elapsed in seconds: ', elapsed)

    return testing_accuracy


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

    folder = "input\\"

    # Tensorboard logging settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    description = "Train!"
    TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
    writer = SummaryWriter(args.log+'_'+TIMESTAMP)
    writer.add_text('description', description)
    writer.add_text('command', 'python ' + ' '.join(os.sys.argv))

    logfile = os.path.join(args.log+'_'+TIMESTAMP, 'logging.log')
    os.sys.stdout = Logger(logfile)

    print(args.log+'_'+TIMESTAMP)
    print(description)
    print('python ' + ' '.join(os.sys.argv))
    print("CUDA Device Count: ", torch.cuda.device_count())
    print(args)

    # Define training hyperparameters
    INIT_LR = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 10

    # Define the train and val splits
    TRAIN_TEST_SPLIT = 0.1
    TRAIN_SPLIT = 0.75
    VAL_SPLIT = 1 - TRAIN_SPLIT

    # Set up random seed on everything
    SEED = 0
    seed_everything(SEED)

    # Fetch Training Data.
    print("[INFO] Fetching Data...")
    X_All, y_All = getTrainData(folder)[0], getTrainData(folder)[1]
    print(X_All.shape, y_All.shape)

    # Calculate the train/validation split
    print("[INFO] generating the train/validation/test split...")
    X_train, X_test, y_train, y_test = train_test_split(X_All, y_All, test_size=TRAIN_TEST_SPLIT, random_state=SEED)
    
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    print(len(train_dataset), len(test_dataset))
    print(int(len(X_train) * TRAIN_SPLIT), int(len(X_train) * VAL_SPLIT))

    train_dataset, val_dataset = random_split(train_dataset
                                    [int(len(X_train) * TRAIN_SPLIT), int(len(X_train) * VAL_SPLIT)],
                                    generator=torch.Generator().manual_seed(SEED))
    print(train_dataset.shape, val_dataset.shape, test_dataset.shape)

    '''

    if args.model.startswith("lstm"):

        # Train Model.
        model = LSTM_SR(input_dim=, hidden_dim=, num_layers=2, batch_size=32, num_classes=7)
        training_accuracy = train_lstm(model, X_train, y_train, EPOCHS, INIT_LR)

        # Evaluate Model.
        test_accuracy = test_lstm(model, X_test, y_test)
        print('Training accuracy is %2.3f :' %(training_accuracy) )
        print('Test accuracy is %2.3f :' %(test_accuracy) )

        # Inference on Test Data.

    elif args.model.startswith("cnn"):

        # initialize the train, validation, and test data loaders
        trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
        valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
        testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

        # calculate steps per epoch for training and validation set
        trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
        valSteps = len(valDataLoader.dataset) // BATCH_SIZE

        # initialize the LeNet model
        print("[INFO] initializing the LeNet model...")
        model = CNN_SR(
            numChannels=1,
            classes=len(trainData.dataset.classes)).to(device)
        
        # initialize our optimizer and loss function
        opt = optim.Adam(model.parameters(), lr=INIT_LR)
        lossFn = nn.NLLLoss()

        # initialize a dictionary to store training history
        H = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        # measure how long training is going to take
        print("[INFO] training the network...")
        startTime = time.time()

        # loop over our epochs
        for e in range(0, EPOCHS):
            # set the model in training mode
            model.train()
            # initialize the total training and validation loss
            totalTrainLoss = 0
            totalValLoss = 0
            # initialize the number of correct predictions in the training
            # and validation step
            trainCorrect = 0
            valCorrect = 0
            # loop over the training set
            for (x, y) in trainDataLoader:
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # perform a forward pass and calculate the training loss
                pred = model(x)
                loss = lossFn(pred, y)
                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far and
                # calculate the number of correct predictions
                totalTrainLoss += loss
                trainCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()

            # calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / trainSteps
            avgValLoss = totalValLoss / valSteps

            # calculate the training and validation accuracy
            trainCorrect = trainCorrect / len(trainDataLoader.dataset)
            valCorrect = valCorrect / len(valDataLoader.dataset)

            # update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["train_acc"].append(trainCorrect)
            H["val_loss"].append(avgValLoss.cpu().detach().numpy())
            H["val_acc"].append(valCorrect)
            
            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
            print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
            print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))

        # finish measuring how long training took
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime))
        
        # we can now evaluate the network on the test set
        print("[INFO] evaluating network...")

        # turn off autograd for testing evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            
            # initialize a list to store our predictions
            preds = []
            # loop over the test set
            for (x, y) in testDataLoader:
                # send the input to the device
                x = x.to(device)
                # make the predictions and add them to the list
                pred = model(x)
                preds.extend(pred.argmax(axis=1).cpu().numpy())

        # generate a classification report
        print(classification_report(testData.targets.cpu().numpy(),
            np.array(preds), target_names=testData.classes))
        
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H["train_loss"], label="train_loss")
        plt.plot(H["val_loss"], label="val_loss")
        plt.plot(H["train_acc"], label="train_acc")
        plt.plot(H["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(args["plot"])

        # serialize the model to disk
        torch.save(model, args["model"])

    '''