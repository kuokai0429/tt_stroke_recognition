# 2023.0417.1607 @Brian

from datetime import datetime
import time
import os
import glob
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
import cv2

# parser = argparse.ArgumentParser(description='main')
# parser.add_argument('--file_code', required=True, type=str, help="First character of the file name.")
# args = parser.parse_args()
# print(args.file_code)

def videoFrame(filename):

    cap = cv2.VideoCapture(f"{filename}.mp4")
    output = cv2.VideoWriter('result_' + str(5) + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 
                float(20), (int(cap.get(3)), int(cap.get(4))))
    color = [(255, 0, 0), (0, 0, 0), (0, 0, 255)]

    count = 0
    while(cap.isOpened()):
        
        count += 1
        ret, frame = cap.read()
        
        if ret == True:
            cv2.rectangle(frame, (40, 10), (1000, 60), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, "Frame: " + str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (0, 0, 0), 5, cv2.LINE_4)  
            output.write(frame)
        else:
            break 

    cap.release()
    output.release()

if __name__ == "__main__":

    for filepath in sorted(glob.glob(f".\input\*.MOV")):
  
        filename = filepath.rsplit('\\', 1)[1].rsplit('.', 1)[0]
        firstletter = filename[0]
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        fileID = re.findall(r'\d+', filename)[0]

        print(f"\nInput video: {filepath} -> {filename} -> {firstletter} -> {fileID} \n")

        # os.system(f"ffmpeg -i {firstletter}-{fileID}.MOV -vcodec h264 -acodec mp2 {firstletter}{fileID}.mp4")


    