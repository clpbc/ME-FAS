### Data prepoecessing - MCIO

- 对于MCIO数据集，先将视频提取成图像帧，之后使用[MTCNN](https://github.com/timesler/facenet-pytorch/tree/master)进行人脸图像对齐裁切

1. 对于每个视频，取视频中的两帧，frame[math.floor(1/4 * total_frame_num)]与frame[math.floor(3/4 * total_frame_num)],保存帧名为video_frame0.jpg/video_frame1.jpg.
2. 将所有帧输入MTCNN获取裁切对齐后人脸，图像尺寸为(224, 224, 3)，RGB三通道图像
3. 保存帧在data/MCIO/frame/,遵循在data/MCIO/txt/文件中的文件名，并且按照以下的文件夹格式进行文件组织。

```
data/MCIO/frame/
|-- casia
    |-- train
    |   |--real
    |   |  |--1_1_frame0.png, 1_1_frame1.png 
    |   |--fake
    |      |--1_3_frame0.png, 1_3_frame1.png 
    |-- test
        |--real
        |  |--1_1_frame0.png, 1_1_frame1.png 
        |--fake
           |--1_3_frame0.png, 1_3_frame1.png 
|-- msu
    |-- train
    |   |--real
    |   |  |--real_client002_android_SD_scene01_frame0.png, real_client002_android_SD_scene01_frame1.png
    |   |--fake
    |      |--attack_client002_android_SD_ipad_video_scene01_frame0.png, attack_client002_android_SD_ipad_video_scene01_frame1.png
    |-- test
        |--real
        |  |--real_client001_android_SD_scene01_frame0.png, real_client001_android_SD_scene01_frame1.png
        |--fake
           |--attack_client001_android_SD_ipad_video_scene01_frame0.png, attack_client001_android_SD_ipad_video_scene01_frame1.png
|-- replay
    |-- train
    |   |--real
    |   |  |--real_client001_session01_webcam_authenticate_adverse_1_frame0.png, real_client001_session01_webcam_authenticate_adverse_1_frame1.png
    |   |--fake
    |      |--fixed_attack_highdef_client001_session01_highdef_photo_adverse_frame0.png, fixed_attack_highdef_client001_session01_highdef_photo_adverse_frame1.png
    |-- test
        |--real
        |  |--real_client009_session01_webcam_authenticate_adverse_1_frame0.png, real_client009_session01_webcam_authenticate_adverse_1_frame1.png
        |--fake
           |--fixed_attack_highdef_client009_session01_highdef_photo_adverse_frame0.png, fixed_attack_highdef_client009_session01_highdef_photo_adverse_frame1.png
|-- oulu
    |-- train
    |   |--real
    |   |  |--1_1_01_1_frame0.png, 1_1_01_1_frame1.png
    |   |--fake
    |      |--1_1_01_2_frame0.png, 1_1_01_2_frame1.png
    |-- test
        |--real
        |  |--1_1_36_1_frame0.png, 1_1_36_1_frame1.png
        |--fake
           |--1_1_36_2_frame0.png, 1_1_36_2_frame1.png
|-- celeb
    |-- real
    |   |--167_live_096546.jpg
    |-- fake
        |--197_spoof_420156.jpg       
```







