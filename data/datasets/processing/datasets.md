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

### Data prepoecessing - WCS

- 对于WCS数据集，只使用原始给定帧图像同时裁切图像[多余黑边](https://github.com/AlexanderParkin/CASIA-SURF_CeFA/blob/205d3d976523ed0c15d1e709ed7f21d50d7cf19b/at_learner_core/at_learner_core/utils/transforms.py#L456)。

1. 使用surf数据集中的所有帧及其原始文件名。在cefa与wmca数据集中每个视频等距采样10帧，将采样帧另存为videoname_XX.jpg（其中XX表示采样帧的索引）详细文件名可以在data/WCS/txt/中找到。
2. 将所有帧输入MTCNN获取裁切对齐后人脸，图像尺寸为(224, 224, 3)，RGB三通道图像
3. 保存帧在data/WCS/frame/,遵循在data/WCS/txt/文件中的文件名，并且按照以下的文件夹格式进行文件组织。

```
data/WCS/frame/
|-- wmca
    |-- train
    |   |--real
    |   |  |--31.01.18_035_01_000_0_01_00.jpg, 31.01.18_035_01_000_0_01_05.jpg
    |   |--fake
    |      |--31.01.18_514_01_035_1_05_00.jpg, 31.01.18_514_01_035_1_05_05.jpg
    |-- test
        |--real
        |  |--31.01.18_036_01_000_0_00_00.jpg, 31.01.18_036_01_000_0_00_01.jpg
        |--fake
           |--31.01.18_098_01_035_3_13_00.jpg, 31.01.18_098_01_035_3_13_01.jpg
|-- cefa
    |-- train
    |   |--real
    |   |  |--3_499_1_1_1_00.jpg, 3_499_1_1_1_01.jpg
    |   |--fake
    |      |--3_499_3_2_2_00.jpg, 3_499_3_2_2_01.jpg
    |-- test
        |--real
        |  |--3_299_1_1_1_00.jpg, 3_299_1_1_1_01.jpg
        |--fake
           |--3_299_3_2_2_00.jpg, 3_299_3_2_2_01.jpg
|-- surf
    |-- train
    |   |--real
    |   |  |--Training_real_part_CLKJ_CS0110_real.rssdk_color_91.jpg
    |   |--fake
    |      |--Training_fake_part_CLKJ_CS0110_06_enm_b.rssdk_color_91.jpg
    |-- test
        |--real
        |  |--Val_0007_007243-color.jpg
        |--fake
           |--Val_0007_007193-color.jpg
```

## Acknowledgement

The data-preprocessing method mentioned above is followed directly from [few-shot-fas](https://github.com/hhsinping/few_shot_fas) repository. We thank the authors for their great work and for making the code public.
