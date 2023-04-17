# 2023.0417.1548 @Brian

from datetime import datetime
import time
import os
import glob
import re

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

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

    # Get Training Data.
    (X_All, y_All) = getTrainData()
    X_train, X_test, y_train, y_test = train_test_split(X_All, y_All, test_size=0.1, random_state=0)

    print(X_All.shape, y_All.shape)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Train Model.

    # Inference on Test Data.

    