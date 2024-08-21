# ME-FAS: Multimodal Text Enhancement for Cross-Domain Face Anti-Spoofing

## Updates ⏱️

- **2024-08-20**: Code released.（The training log file and training model will be uploaded later.）

## Highlights ⭐

<figure style="text-align: center;">
<img src="https://clpbc-pic.oss-cn-nanjing.aliyuncs.com/img/202408210915632.png" alt="image-20240821091504565" style="zoom:50%;" />
</figure>



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

Please refer to [datasets.md](https://github.com/clpbc/ME-FAS/blob/main/data/datasets/processing/datasets.md) for acquiring and pre-processing the datasets.

### Training

```shell
python main.py --config configs/me_fas.yaml --device cuda:0;
```

### Model Zoo

We will be uploading the training logger and training model soon.

## Results 📈

<figure style="text-align: center;">
<img src="https://clpbc-pic.oss-cn-nanjing.aliyuncs.com/img/202408211038408.png" alt="image-20240821103844258" style="zoom: 67%;" />
    <figcaption>Cross Domain performance in Protocol 1</figcaption>
</figure>

<figure style="text-align: center;">
<img src="https://clpbc-pic.oss-cn-nanjing.aliyuncs.com/img/202408211039557.png" alt="image-20240821103913424" style="zoom:67%;" />
    <figcaption>Cross Domain performance in Protocol 2</figcaption>
</figure>

<figure style="text-align: center;">
<img src="https://clpbc-pic.oss-cn-nanjing.aliyuncs.com/img/202408211040682.png" alt="image-20240821104030591" style="zoom:67%;" />
    <figcaption>Cross Domain performance in Protocol 3</figcaption>
</figure>

## Visualizations 🎨

<img src="https://clpbc-pic.oss-cn-nanjing.aliyuncs.com/img/202408211042310.png" alt="image-20240821104251168" style="zoom:67%;" />

### Acknowledgement 🙏

Our code is built on top of the [few_shot_fas](https://github.com/hhsinping/few_shot_fas) 、[FLIP](https://github.com/koushiksrivats/FLIP) and [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning)repository. We thank the authors for releasing their code.