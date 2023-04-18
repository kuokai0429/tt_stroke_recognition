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
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='main')
parser.add_argument('--mode', default="video", required=True, type=str, help="Mode.")
args = parser.parse_args()


def prepareData(source, side, window_size):

    stroke_class =  {"其他": 0, "正手發球": 1, "反手發球": 2, "正手推球": 3, "反手推球": 4, "正手切球": 5, "反手切球":6}

    train, train_label = [], []

    keypoints_2d = np.load(glob.glob(f"input/cropped_{source}*{side}.npz")[0], encoding='latin1', allow_pickle=True)
    print(keypoints_2d.files)
    print(keypoints_2d['positions_2d'])
    print(len(dict(enumerate(keypoints_2d['positions_2d'].flatten()))[0]["myvideos.mp4"]["custom"][0]))
    print(len(dict(enumerate(keypoints_2d['positions_2d'].flatten()))[0]["myvideos.mp4"]["custom"][0][0]))
    print(len(dict(enumerate(keypoints_2d['positions_2d'].flatten()))[0]["myvideos.mp4"]["custom"][0][0][0]))
    keypoints_2d = dict(enumerate(keypoints_2d['positions_2d'].flatten()))[0]["myvideos.mp4"]["custom"][0]
  
    for filepath in sorted(glob.glob(f"annotation/{source}*{side}.txt"))[:1]:

        print(filepath)

        with open(filepath, 'r', encoding= "utf-8") as f:

            for r1 in f.readlines():

                start, end, sc = int(r1.split()[0]), int(r1.split()[1]), str(r1.split()[2])
                print("frame range ", start, end)

                step = (end - start) // window_size
                extra = (end - start) % window_size
                print("step extra", step, extra)
                print("extra data", end - range(start, end, step)[5], end="\n")

                for z in range(end - range(start, end, step)[5]):
                    for count, i in enumerate(range(start+z, end, step)):

                        if count == window_size: break

                        train.append(keypoints_2d[i])
                        
                    train_label.append([stroke_class[sc]])

    train = np.asarray(train).reshape(-1, 17 * window_size, 2)
    train_label = np.asarray(train_label).reshape(-1, 1)

    print(len(train), len(train_label))
    print(train, train_label)

    return None, None


def videoCrop(filepath, filename, output_directory):

    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first_access, count = True, 0

    while True:

        count += 1
        ret, frame = cap.read()
        
        if not ret:
            print("End of video or Empty file.")
            break

        if first_access == True:
            
            first_access = False
            area = cv2.selectROI('videoCrop', frame, showCrosshair=False, fromCenter=False)
            out = cv2.VideoWriter(f'{output_directory}/cropped_{filename}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                fps, (area[2], area[3]))

            print("\nPress 'q' to exit!")
        
        keyName = cv2.waitKey(1)

        if keyName == ord('q'):
            print("Exit video!")
            break
        
        cv2.rectangle(frame, area, (255, 255, 255), 3)
        cv2.imshow('videoCrop', frame)

        frame_cropped = frame[int(area[1]):int(area[1]+area[3]), 
            int(area[0]):int(area[0]+area[2])] if any(area) > 0 else frame.copy()
        out.write(frame_cropped)

        time.sleep(0)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def getVideoInfo(filepath):

    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    return int(fps), int(duration)


def videoFrame(file, folder):

    print("Start adding frame!\n")
    
    cap = cv2.VideoCapture(f"{folder}{file}.mp4")
    output = cv2.VideoWriter(f'{folder}result_{file}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                             float(30), (int(cap.get(3)), int(cap.get(4))))

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

    print("Finished!\n")


if __name__ == "__main__":

    if args.mode.startswith("video"):

        folder = ".\\input\\"

        for filepath in sorted(glob.glob(f"{folder}*.MOV"))[:]:
    
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filepath.rsplit('\\', 1)[1].rsplit('.', 1)[0]
            firstletter = filename[0]
            fileID = re.findall(r'\d+', filename)[0]

            print(f"\nInput video: {filepath} -> {filename} -> {firstletter.lower()} -> {fileID} \n")
            
            # Convert video to mp4
            if args.mode == "video-convert":
                os.system(f"ffmpeg -i {folder}{firstletter}-{fileID}.MOV -vcodec h264 -acodec mp2 {folder}{firstletter.lower()}{fileID}.mp4")
            
            # Add frame
            elif args.mode == "video-addframe":
                videoFrame(f"{firstletter.lower()}{fileID}", folder)

            # Crop video
            elif args.mode == "video-crop":
                videoCrop(f"{folder}{firstletter.lower()}{fileID}.mp4", f"{firstletter.lower()}{fileID}", folder)

            elif args.mode == "video-test":
               print("Test")

    elif args.mode.startswith("annotation"):

        print("Mode: annotation")

        X_All, y_All = prepareData("m", "right", 10)
        X_train, X_test, y_train, y_test = train_test_split(X_All, y_All, test_size=0.1, random_state=0)

        print(X_All.shape, y_All.shape)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
