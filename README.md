# Table Tennis Stroke Recognition

We introduce a novel method for table tennis video data stroke detection and classification.

<br>

## Prepare Data 訓練資料集準備

<br>

**Prepare the annotation file for CNN tt-stroke-recognition model.**

- 3D Annotations:
```bash
python prepare_data.py --mode annotation-3d
```
- 2D Annotations with visualizations of labels:
```bash
python prepare_data.py --mode annotation-2d --vis_target nchu_f1_right
```
<br>

**Prepare the keypoints sequence by Pose estimation inference**

- 2D Pose estimation:
```bash
python prepare_data.py --mode video-pose2d --video sample_video.mp4
```

- 3D Pose estimation:
```bash
python prepare_data.py --mode video-pose3d --video sample_video.mp4 --out_video_sf 0 --out_video_dl 1000 --pose3d_rotation 0 0 0
```

<br>

**Other utilities for Labeling Data.**

- Add frames ID to input video:
```bash
python prepare_data.py --mode video-addframe --video nchu_m6_right.mp4
```

- Video cropping on input video:
```bash
python prepare_data.py --mode video-crop --video nchu_m6_right.mp4
```

<br>

## 模型訓練 Training 

To train a 3d keypoints model on table tennis video data:

```bash
python run.py --mode train3d
( python run.py --mode train2d )
```

<br>

## 模型預測 Run Inference

To test on a 

```bash
python run.py --mode inference3d --inference_target nchu_f1_right --inference_with_gt --checkpoint checkpoint/epoch50_train3d_20230620T15-16-49.pth
( python run.py --mode inference2d --inference_target nchu_f1_right --inference_with_gt --checkpoint checkpoint/epoch50_train2d_20230613T01-03-08.pth )
```
