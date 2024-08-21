# ME-FAS: Multimodal Text Enhancement for Cross-Domain Face Anti-Spoofing

## Updates ⏱️

- **2024-08-20**: Code released.（The training log file and training model will be uploaded later.）

## Highlights ⭐

<img src="https://clpbc-pic.oss-cn-nanjing.aliyuncs.com/img/202408210915632.png" alt="image-20240821091504565" style="zoom:50%;" />

1. We propose a novel early alignment network, ME-FAS, which utilizes prompts and masking as efficient intermediaries which significantly enhances the model's generalization capabilities across diverse domains.
2.  Extensive experiments and analysis, we demonstrate that our method significantly outperforms state-of-the-art competitors on widely used benchmark datasets, confirming its superiority in enhancing DG-FAS tasks.

## Instruction for code usage 📄

### **Setup**

- Get Code

```shell
git clone https://github.com/clpbc/ME-FAS.git
```

- Build Environment

```shell
cd ME-FAS
conda create -n mefas python=3.8
conda activate mefas
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### Dataset Pre-Processing

Please refer to [datasets.md](data\datasets\processing\Data preprocessing.md) for acquiring and pre-processing the datasets.

