# 2023.0417.1548 @Brian

import os
import glob
import argparse
import pickle
import re
import random
import time
from datetime import datetime
from common.log import Logger
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import fontManager
import cv2
import seaborn as sns
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from common.model import CNN_2D_SR, CNN_3D_SR
from common.dataset import StrokeRecognitionDataset


def init_seed(seed):

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


def getTrainData(folder, datatype):

    temp = []
    for i in ['X_All', 'y_All']:
        with open(f'{folder}{datatype}_{i}.pkl', 'rb') as handle:
            temp.append(pickle.load(handle))

    return temp


def train_cnn(model, trainDataLoader, valDataLoader, trainSteps, valSteps):

    # initialize our optimizer and loss function
    opt = optim.Adam(model.parameters(), lr=INIT_LR)
    loss_function = nn.CrossEntropyLoss()

    # initialize a dictionary to store training history
    history = {
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
        for x, y in trainDataLoader:

            # send the input to the device
            x, y = (x.to(DEVICE), y.to(DEVICE))

            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = loss_function(pred, y)

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
            
        # switch off autograd for evaluation
        with torch.no_grad():

            # set the model in evaluation mode
            model.eval()

            # loop over the validation set
            for (x, y) in valDataLoader:

                # send the input to the device
                (x, y) = (x.to(DEVICE), y.to(DEVICE))

                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValLoss += loss_function(pred, y)

                # calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)

        # update our training history
        history["train_loss"].append(avgTrainLoss.detach().cpu().numpy())
        history["train_acc"].append(trainCorrect)
        history["val_loss"].append(avgValLoss.detach().cpu().numpy())
        history["val_acc"].append(valCorrect)
        
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))

    # finish measuring how long training took
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))
    
    return model, history


def test_cnn(model, history, testDataLoader, test_dataset):

    # turn off autograd for testing evaluation
    with torch.no_grad():

        # set the model in evaluation mode
        model.eval()
        
        # initialize a list to store our predictions
        preds = []

        # loop over the test set
        for x, y in testDataLoader:

            # send the input to the device
            x = x.to(DEVICE)

            # make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())

    # Generate a classification report
    print(classification_report(test_dataset.targets.cpu().numpy(),
        np.array(preds), target_names=test_dataset.classes))
    
    # Confusion Matrix
    cf_matrix = confusion_matrix(test_dataset.targets.cpu().numpy(), np.array(preds))
    svm = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
    figure = svm.get_figure()    
    figure.savefig(f'checkpoint/cm_{args.mode}_{TIMESTAMP[:-1]}.png', dpi=400)
    print(cf_matrix)
    
    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(f"checkpoint/loss_{TIMESTAMP[:-1]}")

    # Serialize the model to disk
    torch.save(model.state_dict(), f"checkpoint/epoch{EPOCHS}_{args.mode}_{TIMESTAMP[:-1]}.pth")


def classify_frames(keypoints, datatype):

    coord = 2 if datatype == "2D" else 3
    stride, pred_result = [1, 3, 5], [0] * (len(keypoints) + 1)
    temp = int((MODEL_INPUT_FRAMES - 1) / 2) * stride[2]

    for i in tqdm(range(1 + temp, len(keypoints) + 1 - temp)[:]):

        # print(f"\n--------------------- Frame {i} ---------------------")
        pred_temp = []

        for s in stride:
    
            window_range = range(i - int((MODEL_INPUT_FRAMES - 1) / 2) * s, i + 1 + int((MODEL_INPUT_FRAMES - 1) / 2) * s, s)
            window_frames = []
            # print(f"\nStride {s} (len{len(window_range)}): ")

            for j in window_range:

                # print(j, end=" ")
                window_frames.append(keypoints[j-1])

            X_features = torch.FloatTensor(window_frames).view(-1, 1, MODEL_INPUT_FRAMES * 17 * coord)
            X_features = X_features.to(DEVICE)
            pred_temp.append(model(X_features))

        pred_temp = np.array([t.detach().cpu().numpy() for t in pred_temp])
        pred_temp = np.mean(pred_temp, axis=0) # column-wise mean
        pred_result[i] = np.array(pred_temp.argmax(1)[0])
        # print(f"\n{pred_temp.argmax(1)}")

    pred_result = np.array(pred_result)
    count_pred_result = [np.count_nonzero(pred_result == 0), np.count_nonzero(pred_result == 1), 
        np.count_nonzero(pred_result == 2), np.count_nonzero(pred_result == 3), np.count_nonzero(pred_result == 4)]
    print(f"\nPredicted Segments: {pred_result.shape, np.unique(pred_result), count_pred_result}")

    return pred_result


def majority_filter_1d(pred_result, width):

    offset = width // 2
    pred_result = [0] * offset + list(pred_result)

    result = []
    for i in range(len(pred_result) - offset):
        a = pred_result[i:i+width]
        result.append(max(set(a), key=a.count))

    return result


def prepare_gt_segmts(keypoints, ground_truth_filepath):

    ground_truth = [0] * (len(keypoints) + 1)
    df = pd.read_csv(ground_truth_filepath, encoding='utf8')
    
    for index, row in df.iterrows():
        start, end, sc = row['start'], row['end'], row['label']

        for i in range(start, end + 1):
            ground_truth[i] = stroke_class[sc]

    ground_truth = np.array(ground_truth)
    count_ground_truth = [np.count_nonzero(ground_truth == 0), np.count_nonzero(ground_truth == 1), 
        np.count_nonzero(ground_truth == 2), np.count_nonzero(ground_truth == 3), np.count_nonzero(ground_truth == 4)]
    print(f"Ground Truth Segments: {ground_truth.shape, np.unique(ground_truth), count_ground_truth}")

    return ground_truth


def plot_segmts(ground_truth, pred_result, keypoints):
    '''
    Note: Short predicted segments won't show up in the barh plots.
    '''

    stroke_class =  {"其他": 0, "右正手發球": 1, "右反手發球": 2, "右正手回球": 3, "右反手回球": 4}
    facecolors_stroke_class = {0: 'tab:grey', 1: 'tab:blue', 2: 'tab:green', 3: 'tab:red', 4: 'tab:orange'}
    gt_barh, pred_barh, gt_facecolors, pred_facecolors = [], [], [], []

    for target in [(ground_truth, gt_barh, gt_facecolors, 'Ground Truth barh'), (pred_result, pred_barh, pred_facecolors, 'Predicted barh')]:

        # print(f"\n--------------------- {target[3]} ---------------------")

        pre_startframe, pre_class, length = 1, target[0][1], 1
        for i in range(2, len(keypoints) + 1)[:]:

            if target[0][i] == pre_class:
                length += 1
            else:
                # print((pre_startframe, length), facecolors_stroke_class[pre_class])
                target[1].append((pre_startframe, length))
                target[2].append(facecolors_stroke_class[pre_class])
                pre_startframe, length = i, 1

            pre_class = target[0][i]

        # print((pre_startframe, length), facecolors_stroke_class[pre_class])
        target[1].append((pre_startframe, length))
        target[2].append(facecolors_stroke_class[pre_class])

    print("\n[INFO] Drawing Ground Truth and Predicted Results Segments.")
    print(f"gt_barh length: {len(gt_barh)}")
    print(f"gt_facecolors length: {len(gt_facecolors)}")
    print(f"pred_barh length: {len(pred_barh)}")
    print(f"pred_facecolors length: {len(pred_facecolors)}", end="\n\n")

    fontManager.addfont('data/TaipeiSansTCBeta-Regular.ttf')
    plt.rc('font', family='Taipei Sans TC Beta')
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(17,6))
    ax.broken_barh(pred_barh, (11, 8), facecolors=pred_facecolors)
    ax.broken_barh(gt_barh, (21, 8), facecolors=gt_facecolors)
    ax.set_ylim(5, 35)
    ax.set_xlim(0, len(keypoints))
    ax.set_xlabel(f'frames since start ( total frames {len(keypoints)} )')
    ax.set_yticks([15, 25])
    ax.set_yticklabels(['Predicted Segments', 'Ground Truth Segments'])    
    ax.grid(True)   
    plt.legend(["其他", "右正手發球", "右反手發球", "右正手回球", "右反手回球"])
    h = [mpatches.Patch(color='grey', label="其他"), 
        mpatches.Patch(color='blue', label="右正手發球"),
        mpatches.Patch(color='green', label="右反手發球"),
        mpatches.Patch(color='red', label="右正手回球"),
        mpatches.Patch(color='orange', label="右反手回球")]
    plt.legend(handles=h, bbox_to_anchor =(1.10, 0.63))
    plt.savefig(f"output/ts_{args.mode[-2:]}({args.inference_target})_{TIMESTAMP[:-1]}")                                
    # plt.show()
    plt.clf()


def visualize_pred(TIMESTAMP, filepath, pred_mask, keypoints, datatype):

    cap = cv2.VideoCapture(filepath)
    output = cv2.VideoWriter(f'output/predictions_{args.mode[-2:]}({args.inference_target})_{TIMESTAMP}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 
                float(30), (int(cap.get(3)), int(cap.get(4))))
    
    classes = ['None', 'Forehand Serve', 'Backhand Serve', 'Forehand Push', 'Backhand Push']
    color = [(0, 0, 0), (255, 0, 0), (0, 100, 0), (0, 0, 255), (66, 144, 245)]
    count, progression_bar = 0, tqdm(total = len(pred_mask))

    while(cap.isOpened()):
        
        count += 1
        ret, frame = cap.read()
        
        if ret == True:

            cv2.rectangle(frame, (40, 10), (800, 60), (255, 255, 255), -1, cv2.LINE_AA)
            # cv2.putText(frame, "Frame: " + str(count) + " Predictions: " + str(classes[pred[i-1][0][0]]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
            #       1.5, (0, 0, 0), 5, cv2.LINE_4)
            cv2.putText(frame, f"Frame: {count}  Stroke Class: {str(classes[pred_mask[count-1]])}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color[pred_mask[count-1]], 5, cv2.LINE_4)
            
            if keypoints[count-1] is not None and datatype == "2D":
            
                for x, y in keypoints[count-1]:
                    frame = cv2.circle(frame, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)
            
            output.write(frame)
            progression_bar.update(1)
        else:
            break 

    cap.release()
    output.release()
    progression_bar.close()


def eval_1(ground_truth, pred_result):

    # Insure that each class has one prediction for confusion_matrix() and classification_report().
    ground_truth = np.append(ground_truth, [0, 1, 2, 3, 4])
    pred_result = np.append(pred_result, [0, 1, 2, 3, 4])

    # Confusion Matrix
    cm = confusion_matrix(ground_truth, pred_result)
    svm = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    figure = svm.get_figure()   
    figure.set_figwidth(12)
    figure.set_figheight(10)
    plt.style.use("ggplot")
    fontManager.addfont('data/TaipeiSansTCBeta-Regular.ttf')
    plt.rc('font', family='Taipei Sans TC Beta')
    plt.legend(["其他", "右正手發球", "右反手發球", "右正手回球", "右反手回球"])
    h = [mpatches.Patch(color='grey', label="0: 其他"), 
        mpatches.Patch(color='blue', label="1: 右正手發球"),
        mpatches.Patch(color='green', label="2: 右反手發球"),
        mpatches.Patch(color='red', label="3: 右正手回球"),
        mpatches.Patch(color='orange', label="4: 右反手回球")]
    plt.legend(handles=h, bbox_to_anchor =(1.4, 1))
    # plt.show() 
    figure.savefig(f'output/cm_{args.mode[-2:]}({args.inference_target})_{TIMESTAMP[:-1]}.png', dpi=400)
    
    # The IoU for class c, IoUc = cm(c, c) / sum(col(c)) + sum(row(c)) - cm(c, c)
    IoUc = [(cm[c][c] / (cm.sum(axis=0)[c] + cm.sum(axis=1)[c] - cm[c][c])) for c in range(NUMBER_OF_CLASSES)]

    # The mean IoU 
    mIoU = sum(IoUc) / NUMBER_OF_CLASSES

    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report: {classification_report(ground_truth, pred_result, target_names=StrokeRecognitionDataset().classes)}")
    print(f"IoU for class c (IoUc): {IoUc}")
    print(f"The mean IoU (mIoU): {mIoU}")

    # Remove the patches of [0, 1, 2, 3, 4].
    ground_truth = ground_truth[:-5]
    pred_result = pred_result[:-5]


def eval_2(ground_truth, pred_result):

    # Only for testing. 
    # ground_truth = np.array([0, 0, 0, 1, 1, 1, 0, 0, 2, 2, 3, 3, 3, 3, 1, 1, 0, 0, 2, 2, 2, 0, 0, 0, 0])
    # pred_result =  np.array([0, 0, 0, 1, 1, 0, 0, 0, 3, 3, 3, 4, 4, 3, 3, 1, 1, 0, 2, 1, 2, 0, 0, 0, 0])
    # ground_truth = ground_truth[:2000]
    # pred_result = pred_result[:2000]

    gt_class_mask = {0: [], 1: [], 2: [], 3: [], 4: []}
    pred_class_mask = {0: [], 1: [], 2: [], 3: [], 4: []}

    gt_class_segments = {0: [], 1: [], 2: [], 3: [], 4: []}
    pred_class_segments = {0: [], 1: [], 2: [], 3: [], 4: []}

    # Split multi-class segments into single-class segments.
    for c in range(NUMBER_OF_CLASSES):
        gt_class_mask[c] = np.array([1 if ground_truth[i] == c else 0 for i in range(len(ground_truth))])
        pred_class_mask[c] = np.array([1 if pred_result[i] == c else 0 for i in range(len(pred_result))])

    # Find the range of each single-class segments.
    for c in range(NUMBER_OF_CLASSES)[:]:

        # print(f"\nClass: {c}")
        # print(gt_class_mask[c][:], np.unique(gt_class_mask[c][:]))
        # print(pred_class_mask[c][:], np.unique(gt_class_mask[c][:]))

        for target in [(gt_class_mask[c], gt_class_segments[c], "Ground Truth Mask"), (pred_class_mask[c], pred_class_segments[c], "Predicted Mask")]:

            # print(target[2])

            pre_startframe, pre_state, length = 0, target[0][0], 1
            for i in range(1, len(target[0])):

                if target[0][i] == pre_state:
                    length += 1
                else:
                    # print(pre_state, (pre_startframe, pre_startframe + length - 1))
                    if pre_state == 1:
                        target[1].append((pre_startframe, pre_startframe + length - 1))
                    pre_startframe, length = i, 1

                pre_state = target[0][i]

            # print(pre_state, (pre_startframe, pre_startframe + length - 1))
            if pre_state == 1:
                target[1].append((pre_startframe, pre_startframe + length - 1))

    for c in range(NUMBER_OF_CLASSES)[:]:
        print(f"\nClass: {c}")
        print(f"gt_class_segments[{c}] length({len(gt_class_segments[c])}): {gt_class_segments[c]}")
        print(f"pred_class_segments[{c}] length({len(pred_class_segments[c])}): {pred_class_segments[c]}")

    
    # Count the stroke-wise TP, FN of each class

    # print("\nStroke-wise TP, FN of each class: ")
    tp_c, fn_c = [0] * NUMBER_OF_CLASSES, [0] * NUMBER_OF_CLASSES

    for c in range(NUMBER_OF_CLASSES)[:]:

        # print(f"\nClass: {c}")
        tp_c_temp, fn_c_temp = 0, 0

        for gt_s in gt_class_segments[c]: 
            
            count = 0
            for pre_s in pred_class_segments[c]:

                if not set(range(gt_s[0], gt_s[1]+1)).isdisjoint(set(range(pre_s[0], pre_s[1]+1))):
                    # print("tp: ", gt_s, pre_s)
                    tp_c_temp += 1
                    count += 1
                    break

            if count == 0:
                # print("fn: ", gt_s) 
                fn_c_temp += 1

        tp_c[c] = tp_c_temp
        fn_c[c] = fn_c_temp


    # Count the stroke-wise FP of each class

    # print("\nStroke-wise FP of each class: ")
    fp_c = [0] * NUMBER_OF_CLASSES

    for c in range(NUMBER_OF_CLASSES)[:]:

        # print(f"\nClass: {c}")
        fp_c_temp = 0

        for pre_s in pred_class_segments[c]:
            
            count = 0
            for gt_s in gt_class_segments[c]: 

                if not set(range(pre_s[0], pre_s[1]+1)).isdisjoint(set(range(gt_s[0], gt_s[1]+1))):
                    count += 1
                    break

            if count == 0:
                # print("fp: ", pre_s) 
                fp_c_temp += 1

        fp_c[c] = fp_c_temp

    print(f"\nTP_C: {tp_c}, FN_C: {fn_c}, FP_C: {fp_c}")

    precision_c = [(tp_c[c] / (tp_c[c] + fp_c[c])) if tp_c[c] != 0 else 0 for c in range(NUMBER_OF_CLASSES)]
    recall_c = [(tp_c[c] / (tp_c[c] + fn_c[c])) if tp_c[c] != 0 else 0 for c in range(NUMBER_OF_CLASSES)]
    f1score_c = [((2 * precision_c[c] * recall_c[c]) / (precision_c[c] + recall_c[c])) if precision_c[c] != 0 and recall_c[c] != 0 else 0 for c in range(NUMBER_OF_CLASSES)]

    print(f"Precision_C: {precision_c}")
    print(f"Recall_C: {recall_c}")
    print(f"F1-Score_C: {f1score_c}")


# Define training hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INIT_LR = 1e-3
BATCH_SIZE = 16
MODEL_INPUT_FRAMES = 15
EPOCHS = 50
TRAIN_TEST_SPLIT = 0.9
TRAIN_VAL_SPLIT = 0.8
SEED = 0
SOURCE_FOLDER = "data\\"
CHECKPOINT = "checkpoint/epoch50_20230503T15-05-00.pth"
NUMBER_OF_CLASSES = len(StrokeRecognitionDataset().classes)

# Argument Parser
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--mode', default="inference2d", required=True, type=str, help="Run Mode.")
parser.add_argument('--log', default="log/run", required=False, type=str, help="Log folder.")
args = parser.parse_known_args()[0]


if __name__ == "__main__":

    # Set up random seed on everything
    init_seed(SEED)

    if args.mode.startswith("inference"):

        TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())

        parser.add_argument('--inference_target', type=str, required=True, help='Inference Target.')
        parser.add_argument('--inference_with_gt', action="store_true", help='Inference With Ground Truth Label.')
        parser.add_argument('--checkpoint', type=str, required=True, help='Stroke Recognition Model Weight.')
        args = parser.parse_args()

        assert os.path.exists(args.checkpoint), "Stroke recognition classifier checkpoint file doesn't exist!"

        video2d_filepath = f"data/cropped_{args.inference_target}.mp4"
        video3d_filepath = f"common/pose3d/video/{args.inference_target}.mp4"
        keypoints2d_filepath = f'data/cropped_{args.inference_target}.npz'
        keypoints3d_filepath = f"common/pose3d/output/{args.inference_target}/keypoints_3d_mhformer.npz"

        stroke_class =  {"其他": 0, "右正手發球": 1, "右反手發球": 2, "右正手回球": 3, "右反手回球": 4}


        if args.mode == "inference2d":

            print("Inference 2D Model: ")

            ## Check if file exists

            assert os.path.exists(video2d_filepath), "2D video file doesn't exist!"
            assert os.path.exists(keypoints2d_filepath), "2D keypoints file doesn't exist!"

            ## Configurations

            video_filepath = video2d_filepath
            datatype = "2D"

            ## Load 2D Human pose estimation keypoints

            loaded_keypoints_2d = np.load(keypoints2d_filepath, encoding='latin1', allow_pickle=True)
            # print(loaded_keypoints_2d.files, loaded_keypoints_2d['positions_2d'])
            print(f'Number of frames: {len(dict(enumerate(loaded_keypoints_2d["positions_2d"].flatten()))[0]["myvideos.mp4"]["custom"][0])}')
            print(f'Number of keypoints: {len(dict(enumerate(loaded_keypoints_2d["positions_2d"].flatten()))[0]["myvideos.mp4"]["custom"][0][0])}')
            print(f'Number of coordinates: {len(dict(enumerate(loaded_keypoints_2d["positions_2d"].flatten()))[0]["myvideos.mp4"]["custom"][0][0][0])}')
            keypoints = dict(enumerate(loaded_keypoints_2d["positions_2d"].flatten()))[0]["myvideos.mp4"]["custom"][0]

            ## Load model weights

            print("[INFO] initializing the CNN_SR model...")
            model = CNN_2D_SR(num_classes=NUMBER_OF_CLASSES).to(DEVICE)
            model.load_state_dict(torch.load(args.checkpoint))
            model.eval()


        elif args.mode == "inference3d":

            print("Inference 3D Model: ")

            ## Check if file exists

            assert os.path.exists(video3d_filepath), "3D video file doesn't exist!"
            assert os.path.exists(keypoints3d_filepath), "3D keypoints file doesn't exist!"

            ## Configurations

            video_filepath = video3d_filepath
            datatype = "3D"

            ## Load 3D Human pose estimation keypoints

            loaded_keypoints_3d = np.load(keypoints3d_filepath, encoding='latin1', allow_pickle=True)
            # print(loaded_keypoints_3d.files, loaded_keypoints_3d['reconstruction'])
            print(f'Number of frames: {len(loaded_keypoints_3d["reconstruction"])}')
            print(f'Number of keypoints: {len(loaded_keypoints_3d["reconstruction"][0])}')
            print(f'Number of coordinates: {len(loaded_keypoints_3d["reconstruction"][0][0])}')
            keypoints = loaded_keypoints_3d["reconstruction"]

            ## Load model weights

            print("[INFO] initializing the CNN_SR model...")
            model = CNN_3D_SR(num_classes=NUMBER_OF_CLASSES).to(DEVICE)
            model.load_state_dict(torch.load(args.checkpoint))
            model.eval()


        ## Classifying stroke classes on each frame with different window stride (mid frame of window).

        print("[INFO] Classifying stroke classes on each frame....")
        pred_result = classify_frames(keypoints, datatype)
        pred_result = majority_filter_1d(pred_result, 30)

        ## Show the predicted segments in video

        print("[INFO] Saving predicted segments to video....")
        visualize_pred(TIMESTAMP[:-1], video_filepath, pred_result, keypoints, datatype)


        if args.inference_with_gt:

            # Read Ground Truth Filepath
            ground_truth_filepath = f"annotation/{args.inference_target}.csv"
            assert os.path.exists(ground_truth_filepath), "Stroke ground truth file doesn't exist!"

            # Prepare Ground Truth Segments
            ground_truth = prepare_gt_segmts(keypoints, ground_truth_filepath)

            # Plot the predicted segments compared with the ground-truth segments (https://matplotlib.org/devdocs/gallery/lines_bars_and_markers/broken_barh.html)
            plot_segmts(ground_truth, pred_result, keypoints)

            # Calculate the Confusion Matrix, IoU and DICE of Ground Truth and Predicted Segments. (https://github.com/qubvel/segmentation_models.pytorch/issues/278)
            print("\n[INFO] Calculating the Confusion Matrix, IoU and DICE of Ground Truth and Predicted Segments....")
            eval_1(ground_truth, pred_result)

            # Calculate the TP, FP, FN of Predicted Segments in a stroke-wise way.
            print("\n[INFO] Calculating the TP, FP, FN of Predicted Segments in a stroke-wise way....")
            eval_2(ground_truth, pred_result)


    elif args.mode.startswith("train"):

        ## Tensorboard logging settings

        description = "Train!"
        TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
        writer = SummaryWriter(f"{args.log}_{args.mode}_{TIMESTAMP}",)
        writer.add_text('description', description)
        writer.add_text('command', 'python ' + ' '.join(os.sys.argv))

        logfile = os.path.join(f"{args.log}_{args.mode}_{TIMESTAMP}", 'logging.log')
        os.sys.stdout = Logger(logfile)

        print(f"{args.log}_{args.mode}_{TIMESTAMP}",)
        print(description)
        print('python ' + ' '.join(os.sys.argv))
        print("CUDA Device Count: ", torch.cuda.device_count())
        print(args)

        ## Fetch Training Data.

        if args.mode == "train2d":

            print("Train 2D Model: ")
            print("[INFO] Fetching Data...")
            X_All, y_All = getTrainData(SOURCE_FOLDER, "2D")[0], getTrainData(SOURCE_FOLDER, "2D")[1]
            print(X_All.shape, y_All.shape)

        elif args.mode == "train3d":

            print("Train 3D Model: ")
            print("[INFO] Fetching Data...")
            X_All, y_All = getTrainData(SOURCE_FOLDER, "3D")[0], getTrainData(SOURCE_FOLDER, "3D")[1]
            print(X_All.shape, y_All.shape)
        

        ## Calculate the train/validation split

        print("[INFO] Generating the train/val/test split...")
        X_train, X_test, y_train, y_test = train_test_split(X_All, y_All, test_size=1-TRAIN_TEST_SPLIT, random_state=SEED)
        train_dataset = StrokeRecognitionDataset(X_train, y_train)
        test_dataset = StrokeRecognitionDataset(X_test, y_test)
        train_dataset, val_dataset = random_split(train_dataset, 
                                        [int(len(X_train) * TRAIN_VAL_SPLIT), len(X_train) - int(len(X_train) * TRAIN_VAL_SPLIT)],
                                        generator=torch.Generator().manual_seed(SEED))
        
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
        print(train_dataset.dataset.classes)

        ## Initialize the train, validation, and test data loaders

        trainDataLoader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
        valDataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        ## Calculate steps per epoch for training and validation set

        trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
        valSteps = len(valDataLoader.dataset) // BATCH_SIZE
        print(f"Train steps: {trainSteps}, Val steps: {valSteps}")
        
        ## Train Model.

        if args.mode == "train2d":

            print("[INFO] initializing the CNN_2D_SR model...")
            model = CNN_2D_SR(num_classes=len(train_dataset.dataset.classes)).to(DEVICE)
            model, history = train_cnn(model, trainDataLoader, valDataLoader, trainSteps, valSteps)

        elif args.mode == "train3d":

            print("[INFO] initializing the CNN_3D_SR model...")
            model = CNN_3D_SR(num_classes=len(train_dataset.dataset.classes)).to(DEVICE)
            model, history = train_cnn(model, trainDataLoader, valDataLoader, trainSteps, valSteps)

        ## Evaluate Model.

        print("[INFO] evaluating network...")
        test_cnn(model, history, testDataLoader, test_dataset)
