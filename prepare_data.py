# 2023.0417.1607 @Brian

import os
import glob
import re
import argparse
import itertools
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pandas as pd

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


# Define training data hyperparameters
MODEL_INPUT_FRAMES = 15

# Argument Parser
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--mode', default="video", required=True, type=str, help="Mode.")
args = parser.parse_args()


def getVideoInfo(filepath):

    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    return int(fps), int(duration)


def videoFrame(file, folder):

    print("Start adding frame!\n")
    
    cap = cv2.VideoCapture(f"{folder}{file}.mp4")
    output = cv2.VideoWriter(f'{folder}addframe_{file}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
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


def videoCrop(filename, folder):

    cap = cv2.VideoCapture(f"{folder}{filename}.mp4")
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
            out = cv2.VideoWriter(f'{folder}/cropped_{filename}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
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


def showLandmarks(keypoints_frame, total_frames, output_directory, source, side):

    keypoints_mask = [None] * (total_frames + 1)
    
    for kp, frame_id in keypoints_frame:
        keypoints_mask[frame_id] = kp

    for filepath in sorted(glob.glob(f"{output_directory}/cropped_{source}*{side}.mp4"))[:1]:
        
        print(f"File path: {filepath}")

        cap = cv2.VideoCapture(filepath)
        
        i = 1
        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if keypoints_mask[i] is not None:
                
                for x, y in keypoints_mask[i]:
                    frame = cv2.circle(frame, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)
                cv2.imshow('frame', frame)

            i += 1

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def prepareData_txt(filename, model_input_frames):

    print(f"File: {filename}")

    train, train_label, keypoints_frame = [], [], []
    stroke_class =  {"其他": 0, "正手發球": 1, "反手發球": 2, "正手推球": 3, "反手推球": 4, "正手切球": 5, "反手切球":6}

    loaded_keypoints_2d = np.load(f"input/cropped_{filename}.npz", encoding='latin1', allow_pickle=True)
    # print(loaded_keypoints_2d.files, loaded_keypoints_2d['positions_2d'])
    print(f'Number of frames: {len(dict(enumerate(loaded_keypoints_2d["positions_2d"].flatten()))[0]["myvideos.mp4"]["custom"][0])}')
    print(f'Number of keypoints: {len(dict(enumerate(loaded_keypoints_2d["positions_2d"].flatten()))[0]["myvideos.mp4"]["custom"][0][0])}')
    print(f'Number of coordinates: {len(dict(enumerate(loaded_keypoints_2d["positions_2d"].flatten()))[0]["myvideos.mp4"]["custom"][0][0][0])}')
    keypoints_2d = dict(enumerate(loaded_keypoints_2d["positions_2d"].flatten()))[0]["myvideos.mp4"]["custom"][0]

    with open(f"annotation/{filename}.txt", 'r', encoding= "utf-8") as f:

        for r1 in f.readlines():

            start, end, sc = int(r1.split()[0]), int(r1.split()[1]), str(r1.split()[2])
            step = (end - start) // model_input_frames
            extra = (end - start) % model_input_frames
            # print("frame range:", start, end)
            # print("(step, extra):", step, extra)
            # print("total extra data:", end - range(start, end, step)[model_input_frames-1] - 1, end="\n")

            for z in range(end - range(start, end, step)[model_input_frames-1]):
                for count, i in enumerate(range(start+z, end, step)):

                    if count == model_input_frames: break

                    keypoints_frame.append([keypoints_2d[i-1], i])
                    train.append(keypoints_2d[i-1])
                    
                train_label.append([stroke_class[sc]])

    train = np.asarray(train).reshape(-1, 17 * model_input_frames, 2)
    train_label = np.asarray(train_label).reshape(-1, 1)
    keypoints_frame = np.asarray(keypoints_frame, dtype=object)

    # print(train, train_label, keypoints_frame)
    print(f"Train Features Shape: {train.shape}")
    print(f"Train Label Shape: {train_label.shape}")
    print(f"Keypoints with Frame: {keypoints_frame.shape}")

    return train, train_label, keypoints_frame, len(keypoints_2d)


def prepareData_csv_ver1(model_input_frames):

    '''
    <All>:
    0 : 其他, 1 : 左正手發球, 2 : 左反手發球, 3 : 左正手回球, 4 : 左反手回球, 
    5 : 右正手發球, 6 : 右反手發球, 7 : 右正手回球, 8 : 右反手回球
    
    <Left>:  
    0: 其他, 1: 左正手發球, 2: 左反手發球, 3: 左正手回球, 4: 左反手回球
    
    <Right>: 
    0: 其他, 1: 右正手發球, 2: 右反手發球, 3: 右正手回球, 4: 右反手回球
    '''

    train, train_label, keypoints_frame_all, frame_length_all = [], [], {}, {}
    stroke_class =  {"其他": 0, "右正手發球": 1, "右反手發球": 2, "右正手回球": 3, "右反手回球": 4}

    for gender, side in [("f", "right"), ("m", "right")]:
  
        for filepath in sorted(glob.glob(f"annotation/{gender}*{side}.csv"))[:]:
            
            filename = filepath.rsplit('\\')[1].rsplit('.')[0]
            print(f"File path: {filepath} -> {filename}")
            
            keypoints_frame = []
            loaded_keypoints_2d = np.load(glob.glob(f"input/cropped_{filename}.npz")[0], encoding='latin1', allow_pickle=True)
            # print(loaded_keypoints_2d.files, loaded_keypoints_2d['positions_2d'])
            print(f'Number of frames: {len(dict(enumerate(loaded_keypoints_2d["positions_2d"].flatten()))[0]["myvideos.mp4"]["custom"][0])}')
            print(f'Number of keypoints: {len(dict(enumerate(loaded_keypoints_2d["positions_2d"].flatten()))[0]["myvideos.mp4"]["custom"][0][0])}')
            print(f'Number of coordinates: {len(dict(enumerate(loaded_keypoints_2d["positions_2d"].flatten()))[0]["myvideos.mp4"]["custom"][0][0][0])}')
            keypoints_2d = dict(enumerate(loaded_keypoints_2d["positions_2d"].flatten()))[0]["myvideos.mp4"]["custom"][0]
            df = pd.read_csv(filepath, encoding='utf8')

            for index, row in df.iterrows():
                
                start, end, sc = row['start'], row['end'], row['label']

                # Skip the stroke annotation data if (stroke length < model input frames)
                if (end - start) < model_input_frames: 
                    continue

                step = (end - start) // model_input_frames
                extra = (end - start) % model_input_frames
                # print("frame range:", start, end)
                # print("(step, extra):", step, extra)
                # print("total extra data:", end - range(start, end, step)[model_input_frames-1] - 1, end="\n")

                # Preparing train data for stroke classes from 1 ~ 4 
                for z in range(end - range(start, end, step)[model_input_frames-1]):
                        
                        for count, i in enumerate(range(start+z, end, step)):

                            if count == model_input_frames: break

                            keypoints_frame.append([keypoints_2d[i-1], i])
                            train.append(keypoints_2d[i-1])
                            
                        train_label.append([stroke_class[sc]])

                # Preparing train data for stroke class 0 from [window_tail ~ window_center] and [window_center ~ window_head]
                if (start - model_input_frames < 1) or (end + model_input_frames > len(keypoints_2d)):
                    continue
                else:
                    for x in range(int((model_input_frames - 1) / 2)):

                        # window_tail ~ window_center
                        for y in range(model_input_frames):
                            keypoints_frame.append([keypoints_2d[start - model_input_frames + x + y - 1], start - model_input_frames + x + y - 1])
                            train.append(keypoints_2d[start - model_input_frames + x + y - 1])
                        train_label.append([stroke_class["其他"]])

                        # window_center ~ window_head
                        for y in range(model_input_frames):
                            keypoints_frame.append([keypoints_2d[end + model_input_frames + x + y - 1], end + model_input_frames + x + y - 1])
                            train.append(keypoints_2d[end + model_input_frames + x + y - 1])
                        train_label.append([stroke_class["其他"]])

            print(f'Number of Train features/labels: {len(train)}/{len(train_label)}')
            keypoints_frame_all[filename] = np.asarray(keypoints_frame, dtype=object)
            frame_length_all[filename] = len(keypoints_2d)

    train = np.asarray(train).reshape(-1, 17 * model_input_frames, 2)
    train_label = np.asarray(train_label).reshape(-1, 1)

    # print(train, train_label, keypoints_frame_all)
    print(f"Train Features Shape: {train.shape}")
    print(f"Train Label Shape: {train_label.shape}")
    print(f"Keypoints with Frame: {[(k, v.shape) for k, v in keypoints_frame_all.items()]}")

    return train, train_label, keypoints_frame_all, frame_length_all


if __name__ == "__main__":

    folder = "input\\"

    if args.mode.startswith("video"):

        # 2d Pose estimation inference and preprocess.
        if args.mode == "video-pose2d":
            os.system("python common/pose2d/infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir input --image-ext mp4 --input input/myvideos.mp4")
            os.system("python common/pose2d/prepare_data_2d_custom.py -i input -o input -os myvideos")

        # 3d Pose estimation inference.
        elif args.mode == "video-pose3d":
            os.system("python common/pose3d/vis.py --video sample_video.mp4")

        # Video process related.
        else:

            filepath = "f-1.MOV"
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filepath.rsplit('\\', 1)[1].rsplit('.', 1)[0]
            
            # Convert video to mp4
            if args.mode == "video-convert":
                firstletter = filename[0]
                fileID = re.findall(r'\d+', filename)[0]
                os.system(f"ffmpeg -i {folder}{filepath}.MOV -vcodec h264 -acodec mp2 {folder}{firstletter.lower()}{fileID}.mp4")
            
            # Add frame
            elif args.mode == "video-addframe":
                videoFrame(f"{filename}", folder)

            # Crop video
            elif args.mode == "video-crop":
                videoCrop(f"{filename}", folder)


    elif args.mode.startswith("annotation"):

        # Prepare Training Data
        # X_All, y_All, kf, tf = prepareData_txt("m1_right", MODEL_INPUT_FRAMES)
        X_All, y_All, kfa, fla = prepareData_csv_ver1(MODEL_INPUT_FRAMES)
        print(f"Type of X_All, y_All: {type(X_All)}, {type(y_All)}")
        print(f"Shape of X_All, y_All: {X_All.shape}, {y_All.shape}")

        # Visualizing annotation keypoints.
        if args.mode.startswith("annotation-visualize"):
            src = args.mode.rsplit('-')[-1]
            showLandmarks(kfa[f'{src}_right'], fla[f'{src}_right'], folder, src, "right")

        # Convert Training Data to Pytorch Tensor
        X_All = torch.FloatTensor(X_All).view(-1, 1, MODEL_INPUT_FRAMES * 17 * 2)
        y_All = torch.LongTensor(y_All).view(-1)
        print(type(X_All), type(y_All))
        print(X_All.shape, y_All.shape)

        # Save Traing Data to pickle
        for k, v in [('X_All', X_All), ('y_All', y_All)]:
            with open(f'{folder}{k}.pkl', 'wb') as f:
                pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)