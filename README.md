# 📷 EasyPhoto | Your Smart AI Photo Generator (AnyID).
🦜 EasyPhoto is a Webui UI plugin for generating AI portraits that can be used to train digital doppelgangers relevant to you.

Here, we expand the EasyPhoto pipeline, making it suitable for any identification (not limited to just the face).

We are still work on it for robust generation, have it for fun and welcome any suggestions!

The image is used for demo presentation. If it has specific copyright, please contact us and we will delete it as soon as possible.

🦜 🦜 Welcome!

English | [简体中文](./README_zh-CN.md)

# Table of Contents
- [Introduction](#introduction)
- [TODO List](#todo-list)
- [Quick Start](#quick-start)
    - [1. Cloud usage: AliyunDSW/AutoDL/Docker](#1-cloud-usage-aliyundswautodldocker) [Not suppport now]
    - [2. Local install: Check/Downloading/Installation](#2-local-install-environment-checkdownloadinginstallation)
- [How to use](#how-to-use)
    - [1. Model Training](#1-model-training)
    - [2. Inference](#2-inference)
- [Algorithm Detailed](#algorithm-detailed)
    - [1. Architectural Overview](#1-architectural-overview)
    - [2. Training Detailed](#2-training-detailed)
    - [3. Inference Detailed](#3-inference-detailed)
- [Reference](#reference)
- [Related Project](#Related-Project)
- [License](#license)
- [ContactUS](#contactus)

# Introduction
EasyPhoto can be used to generate your AI portraits within 5 to 20 images. We're working to expand the EasyPhoto pipeline, allowing you to create images for any desired identity (ID). By providing a set of training images featuring your target ID, you are allowed to first train a specific LoRA model. This empowers you to effortlessly generate personalized ID photos by seamlessly replacing the target in your template image. We now support for virtual try-on applications, and we're continuously exploring more exciting fetaures.

Please read our Contributor Covenant [covenant](./COVENANT.md) | [简体中文](./COVENANT_zh-CN.md).

If you meet some problems in the training, please refer to the [VQA](https://github.com/aigc-apps/sd-webui-EasyPhoto/wiki).

These are our generated results:
![results_1](https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/xinyi.zxy/anyid/result.png)

Our ui interface is as follows:

**train part:**

![train_ui](https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/xinyi.zxy/anyid/train_ui.png)

**inference part:**

![infer_ui](https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/xinyi.zxy/anyid/infer_ui.png)


# Quick Start

## a. From docker
If you are using docker, please make sure that the graphics card driver and CUDA environment have been installed correctly in your machine.

Then execute the following commands in this way:
```
# pull image
docker pull registry.cn-shanghai.aliyuncs.com/pai-ai-test/eas-service:ubuntu2204-cuda117-torch201-sdwebui-anyid01

# enter image
docker run -it -p 7860:7860 --network host --gpus all registry.cn-shanghai.aliyuncs.com/pai-ai-test/eas-service:ubuntu2204-cuda117-torch201-sdwebui-anyid01

# launch webui
python launch.py --xformers
```

If you've already pulled the images provided in the EasyPhoto master, you can navigate to the EasyPhoto directory and switch to the "anyid" branch. Then, refer to the [Install LightGlue](#####install-lightglue) to install lightglue.

The docker updates may be slightly slower than the github repository of sd-webui-EasyPhoto, so you can go to extensions/sd-webui-EasyPhoto and do a git pull first.
```
cd extensions/sd-webui-EasyPhoto/
# checkout anyid
git checkout -b anyid remotes/origin/anyid

# update code
git pull

cd /workspace
```

## b. Local install: Environment Check/Downloading/Installation

We have verified EasyPhoto execution on the following environment:
If you meet problem with WebUI auto killed by OOM, please refer to [ISSUE21](https://github.com/aigc-apps/sd-webui-EasyPhoto/issues/21), and setting some num_threads to 0 and report other fix to us, thanks.

The detailed of Linux:
- OS: Ubuntu 20.04, CentOS
- python: py3.10 & py3.11
- pytorch: torch2.0.1
- tensorflow-cpu: 2.13.0
- CUDA: 11.7
- CUDNN: 8+
- GPU: Nvidia-A10 24G & Nvidia-V100 16G & Nvidia-A100 40G

We need about 60GB available on disk (for saving weights and datasets process), please check!

### Relevant Repositories & Weights Downloading
#### i. Controlnet
We need to use Controlnet for inference. The related repo is [Mikubill/sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet). You need install this repo before using EasyPhoto.

#### ii. Other Dependencies.
We are mutually compatible with the existing stable-diffusion-webui environment, and the relevant repositories (except lightglue) are installed when starting stable-diffusion-webui.

The weights we need will be downloaded automatically when you start training first time.

- Install LightGlue

[LightGlue](https://github.com/cvg/LightGlue) is used for preprocess when training. You should mannually install LightGlue before start EasyPhoto of anyid version.
(Since the latest version of LightGlue misses the setup.py and allow only to download model from github, we recommend you to install LightGlue using our provided version.)
```
wget https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/models/LightGlue.zip
unzip LightGlue.zip
cd LightGlue
python -m pip install -e .
```


### c. Plug-in Installation
Now we support installing EasyPhoto from git. The url of our Repository is https://github.com/aigc-apps/sd-webui-EasyPhoto.

# How to use

The process closely resembles the EasyPhoto master pipeline, with the additional steps of selecting a main image and specifying the target region during inference.

### 1. Model Training
The EasyPhoto training interface is as follows:

![train_ui](https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/xinyi.zxy/anyid/train_ui.png)

- On the left is the training image. First upload a main image, and click Upload Photos to upload the image.
- On the right are the training parameters, which shares the same meaning as the master pipeline.

The main image is used to match the ROI in the training image that is used to segment the target id. Therefore, please choose a clean front image as the main image. Then, click the Upload Photos button to upload training images, the training images will be processed by LightGlue and SAM to obtain the main target. Then we click on "Start Training" below, and at this point, we need to fill in the User ID above, such as the user's name, to start training.


If you want to set parameters, the parsing of each parameter is as follows:

|Parameter Name | Meaning|
|--|--|
|Resolution | The size of the image fed into the network during training, with a default value of 512|
|Validation & save steps | The number of steps between validating the image and saving intermediate weights, with a default value of 100, representing verifying the image every 100 steps and saving the weights|
|Max train steps | Maximum number of training steps, default value is 800|
|Max steps per photos | The maximum number of training sessions per image, default to 200|
|Train batch size | The batch size of the training, with a default value of 1|
|Gradient accumulation steps | Whether to perform gradient accumulation. The default value is 4. Combined with the train batch size, each step is equivalent to feeding four images|
|Dataloader num workers | The number of jobs loaded with data, which does not take effect under Windows because an error will be reported if set, but is set normally on Linux|
|Learning rate | Train Lora's learning rate, default to 1e-4|
|Rank Lora | The feature length of the weight, default to 128|
|Network alpha | The regularization parameter for Lora training, usually half of the rank, defaults to 64|

### 2. Inference
- Step 1: Click the refresh button to query the model corresponding to the trained user ID.
- Step 2: Select the user ID.
- Step 3: Select the template and mask the target region that needs to be generated.
- Step 4: Click the Generate button to generate the results.


# Reference
- insightface：https://github.com/deepinsight/insightface
- cv_resnet50_face：https://www.modelscope.cn/models/damo/cv_resnet50_face-detection_retinaface/summary
- cv_u2net_salient：https://www.modelscope.cn/models/damo/cv_u2net_salient-detection/summary
- cv_unet_skin_retouching_torch：https://www.modelscope.cn/models/damo/cv_unet_skin_retouching_torch/summary
- cv_unet-image-face-fusion：https://www.modelscope.cn/models/damo/cv_unet-image-face-fusion_damo/summary
- kohya：https://github.com/bmaltais/kohya_ss
- controlnet-webui：https://github.com/Mikubill/sd-webui-controlnet

# Related Project
We've also listed some great open source projects as well as any extensions you might be interested in:
- [ModelScope](https://github.com/modelscope/modelscope).
- [FaceChain](https://github.com/modelscope/facechain).
- [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet).
- [sd-webui-roop](https://github.com/s0md3v/sd-webui-roop).
- [roop](https://github.com/s0md3v/roop).
- [sd-webui-deforum](https://github.com/deforum-art/sd-webui-deforum).
- [sd-webui-additional-networks](https://github.com/kohya-ss/sd-webui-additional-networks).
- [a1111-sd-webui-tagcomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete).
- [sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything).
- [sd-webui-tunnels](https://github.com/Bing-su/sd-webui-tunnels).
- [sd-webui-mov2mov](https://github.com/Scholar01/sd-webui-mov2mov).

# License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).

# ContactUS
1. Use [Dingding](https://www.dingtalk.com/) to search group 38250008552 or Scan to join
2. Since the WeChat group is full, you need to scan the image on the right to add this student as a friend first, and then join the WeChat group.

<figure>
<img src="images/erweima.jpg" width=300/>
<img src="images/wechat.jpg" width=300/>
</figure>
