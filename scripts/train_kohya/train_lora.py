# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""
import os
import sys
import gc

import argparse
import logging
import math
import random
import shutil
import time
from glob import glob
from pathlib import Path
from shutil import copyfile
from typing import Dict

import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline as modelscope_pipeline
from modelscope.utils.constant import Tasks
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import utils.lora_utils as network_module
from utils.model_utils import load_models_from_stable_diffusion_checkpoint
from utils.gpu_info import gpu_monitor_decorator

torch.backends.cudnn.benchmark = True

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0")

logger = get_logger(__name__, log_level="INFO")


def log_validation(
    network, noise_scheduler, vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch, global_step, **kwargs
):
    """
    This function, `log_validation`, serves as a validation step during training.
    It generates ID photo templates using controlnet if `template_dir` exists, otherwise, it creates random templates based on validation prompts.
    The resulting images are saved in the validation folder and logged in either TensorBoard or WandB.

    Args:
        model_dir (str): Directory path of the model.
        vae: Variational Autoencoder model.
        text_encoder: Text encoder model.
        tokenizer: Tokenizer for text data.
        unet: UNet model.
        args: Command line arguments.
        accelerator: Training accelerator.
        weight_dtype: Data type for model weights.
        epoch (int): Current training epoch.
        global_step (int): Current global training step.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
    # When template_dir doesn't exist, generate randomly based on validation prompts.
    text_encoder, vae, unet = load_models_from_stable_diffusion_checkpoint(False, args.pretrained_model_ckpt)

    pipeline = StableDiffusionInpaintPipeline(
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
        unet=unet.to(accelerator.device, weight_dtype),
        text_encoder=text_encoder.to(accelerator.device, weight_dtype),
        vae=vae.to(accelerator.device, weight_dtype),
        safety_checker=None,
        feature_extractor=None,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.safety_checker = None
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    network_module.merge_lora(pipeline, network.state_dict(), 1, "cuda", torch.float16)
    generator = torch.Generator(device=accelerator.device)

    if args.seed is not None:
        generator = generator.manual_seed(args.seed)

    # Predictions before the start
    images = []
    if args.template_dir is not None:
        # Iteratively generate ID photos
        jpgs = os.listdir(args.template_dir)
        for jpg, read_jpg, shape, read_mask in zip(jpgs, kwargs["input_images"], kwargs["input_images_shape"], kwargs["input_masks"]):
            image = pipeline(
                args.validation_prompt,
                image=read_jpg,
                mask_image=read_mask,
                strength=0.65,
                negative_prompt=args.neg_prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=20,
                generator=generator,
                height=kwargs["new_size"][1],
                width=kwargs["new_size"][0],
            ).images[0]

            images.append(image)

            save_name = jpg.split(".")[0]
            if not os.path.exists(os.path.join(args.output_dir, "validation")):
                os.makedirs(os.path.join(args.output_dir, "validation"))
            image.save(os.path.join(args.output_dir, "validation", f"global_step_{save_name}_{global_step}_0.jpg"))

    else:
        # Random Generate
        for _ in range(args.num_validation_images):
            images.append(
                pipeline(
                    args.validation_prompt,
                    negative_prompt=args.neg_prompt,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=50,
                    generator=generator,
                    height=args.resolution,
                    width=args.resolution,
                ).images[0]
            )
        for index, image in enumerate(images):
            if not os.path.exists(os.path.join(args.output_dir, "validation")):
                os.makedirs(os.path.join(args.output_dir, "validation"))
            image.save(os.path.join(args.output_dir, "validation", f"global_step_{global_step}_" + str(index) + ".jpg"))

    # Wandb or tensorboard if we have
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for index, image in enumerate(images):
                tracker.writer.add_images("validation_" + str(index), np.asarray(image), epoch, dataformats="HWC")
        if tracker.name == "wandb":
            tracker.log({"validation": [wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)]})

    del pipeline
    torch.cuda.empty_cache()
    vae.to(accelerator.device, dtype=weight_dtype)


def safe_get_box_mask_keypoints(image, retinaface_result, crop_ratio, face_seg, mask_type):
    """
    Inputs:
        image                   输入图片；
        retinaface_result       retinaface的检测结果；
        crop_ratio              人脸部分裁剪扩充比例；
        face_seg                人脸分割模型；
        mask_type               人脸分割的方式，一个是crop，一个是skin，人脸分割结果是人脸皮肤或者人脸框

    Outputs:
        retinaface_box          扩增后相对于原图的box
        retinaface_keypoints    相对于原图的keypoints
        retinaface_mask_pil     人脸分割结果
    """
    h, w, c = np.shape(image)
    if len(retinaface_result["boxes"]) != 0:
        # 获得retinaface的box并且做一手扩增
        retinaface_box = np.array(retinaface_result["boxes"][0])
        face_width = retinaface_box[2] - retinaface_box[0]
        face_height = retinaface_box[3] - retinaface_box[1]
        retinaface_box[0] = np.clip(np.array(retinaface_box[0], np.int32) - face_width * (crop_ratio - 1) / 2, 0, w - 1)
        retinaface_box[1] = np.clip(np.array(retinaface_box[1], np.int32) - face_height * (crop_ratio - 1) / 2, 0, h - 1)
        retinaface_box[2] = np.clip(np.array(retinaface_box[2], np.int32) + face_width * (crop_ratio - 1) / 2, 0, w - 1)
        retinaface_box[3] = np.clip(np.array(retinaface_box[3], np.int32) + face_height * (crop_ratio - 1) / 2, 0, h - 1)
        retinaface_box = np.array(retinaface_box, np.int32)

        # 检测关键点
        retinaface_keypoints = np.reshape(retinaface_result["keypoints"][0], [5, 2])
        retinaface_keypoints = np.array(retinaface_keypoints, np.float32)

        # mask部分
        retinaface_crop = image.crop(np.int32(retinaface_box))
        retinaface_mask = np.zeros_like(np.array(image, np.uint8))
        if mask_type == "skin":
            retinaface_sub_mask = face_seg(retinaface_crop)
            retinaface_mask[retinaface_box[1] : retinaface_box[3], retinaface_box[0] : retinaface_box[2]] = np.expand_dims(
                retinaface_sub_mask, -1
            )
        else:
            retinaface_mask[retinaface_box[1] : retinaface_box[3], retinaface_box[0] : retinaface_box[2]] = 255
        retinaface_mask_pil = Image.fromarray(np.uint8(retinaface_mask))
    else:
        retinaface_box = np.array([])
        retinaface_keypoints = np.array([])
        retinaface_mask = np.zeros_like(np.array(image, np.uint8))
        retinaface_mask_pil = Image.fromarray(np.uint8(retinaface_mask))

    return retinaface_box, retinaface_keypoints, retinaface_mask_pil


def call_face_crop(retinaface_detection, image, crop_ratio, prefix="tmp"):
    # retinaface检测部分
    # 检测人脸框
    retinaface_result = retinaface_detection(image)
    # 获取mask与关键点
    retinaface_box, retinaface_keypoints, retinaface_mask_pil = safe_get_box_mask_keypoints(
        image, retinaface_result, crop_ratio, None, "crop"
    )

    return retinaface_box, retinaface_keypoints, retinaface_mask_pil


def eval_jpg_with_faceid(pivot_dir, test_img_dir, top_merge=10):
    """
    Evaluate images using local face identification.

    Args:
        pivot_dir (str): Directory containing reference real human images.
        test_img_dir (str): Directory pointing to generated validation images for training.
            Image names follow the format xxxx_{step}_{indx}.jpg.
        top_merge (int, optional): Number of top weights to select for merging. Defaults to 10.

    Returns:
        list: List of evaluated results.

    Function:
        - Obtain faceid features locally.
        - Calculate the average feature of real human images.
        - Select top_merge weights for merging based on generated validation images.
    """
    try:
        # embedding
        face_recognition = modelscope_pipeline("face_recognition", model="bubbliiiing/cv_retinafce_recognition", model_revision="v1.0.3")
    except Exception as e:
        print(f"Load face recognition model failed. {e}")
        return [], [], []

    # get ID list
    face_image_list = (
        glob(os.path.join(pivot_dir, "*.jpg"))
        + glob(os.path.join(pivot_dir, "*.JPG"))
        + glob(os.path.join(pivot_dir, "*.png"))
        + glob(os.path.join(pivot_dir, "*.PNG"))
    )

    #  vstack all embedding
    embedding_list = []
    for img in face_image_list:
        try:
            image = Image.open(img)
            embedding = face_recognition(dict(user=image))[OutputKeys.IMG_EMBEDDING]
            if embedding is not None:
                embedding_list.append(embedding)
        except Exception as e:
            print("error at:", str(e))

    if len(embedding_list) == 0:
        print("Can't detect faces in processed images, return empty list")
        return [], [], []

    # exception cause by embedding = None
    try:
        embedding_array = np.vstack(embedding_list)
    except Exception as e:
        print(f"vstack embedding failed, caused by {str(e)}")
        return [], [], []

    #  mean, get pivot
    pivot_feature = np.mean(embedding_array, axis=0)
    pivot_feature = np.reshape(pivot_feature, [512, 1])

    # sort with cosine distance
    embedding_list = [[np.dot(emb, pivot_feature)[0][0], emb] for emb in embedding_list]
    embedding_list = sorted(embedding_list, key=lambda a: -a[0])

    top10_embedding = [emb[1] for emb in embedding_list]
    top10_embedding_array = np.vstack(top10_embedding)
    # [512, n]
    top10_embedding_array = np.swapaxes(top10_embedding_array, 0, 1)

    # sort all validation image
    result_list = []
    if not test_img_dir.endswith(".jpg"):
        img_list = (
            glob(os.path.join(test_img_dir, "*.jpg"))
            + glob(os.path.join(test_img_dir, "*.JPG"))
            + glob(os.path.join(test_img_dir, "*.png"))
            + glob(os.path.join(test_img_dir, "*.PNG"))
        )
        for img in img_list:
            try:
                # a average above all
                image = Image.open(img)
                embedding = face_recognition(dict(user=image))[OutputKeys.IMG_EMBEDDING]

                res = np.mean(np.dot(embedding, top10_embedding_array))
                result_list.append([res, img])
                result_list = sorted(result_list, key=lambda a: -a[0])
            except Exception as e:
                print("error at:", str(e))

    # pick most similar using faceid
    t_result_list = [i[1] for i in result_list][:top_merge]
    tlist = [i[1].split("_")[-2] for i in result_list][:top_merge]
    scores = [i[0] for i in result_list][:top_merge]
    return t_result_list, tlist, scores


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_model_ckpt",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument("--image_column", type=str, default="image", help="The column of the dataset containing an image.")
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="whether to validation in whole training.",
    )
    parser.add_argument("--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference.")
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help=(
            "Run fine-tuning validation every X steps. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="sketch, low quality, worst quality, low quality shadow, lowres, inaccurate eyes, huge eyes, longbody, bad anatomy, cropped, worst face, strange mouth, bad anatomy, inaccurate limb, bad composition, ugly, noface, disfigured, duplicate, ugly, text, logo",
        help="A prompt that is neg during training for inference.",
    )
    parser.add_argument("--guidance_scale", type=int, default=9, help="A guidance_scale during training for inference.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=("For debugging purposes or quicker training, truncate the number of training examples to this " "value if set."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=("The resolution for input images, all the images in the train/validation dataset will be resized to this" " resolution"),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--save_state", action="store_true", help="Whether or not to save state.")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alpha",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )

    parser.add_argument(
        "--template_dir",
        type=str,
        default=None,
        help=("The dir of template used, to make certificate photos."),
    )
    parser.add_argument(
        "--template_mask",
        default=False,
        action="store_true",
        help=("To mask certificate photos."),
    )
    parser.add_argument(
        "--template_mask_dir",
        type=str,
        default=None,
        help=("The dir of template masks used, to make certificate photos."),
    )

    parser.add_argument(
        "--merge_best_lora_based_face_id",
        default=False,
        action="store_true",
        help=("Merge the best loras based on face_id."),
    )
    parser.add_argument(
        "--merge_best_lora_name",
        type=str,
        default=None,
        help=("The output name for getting best loras."),
    )
    parser.add_argument(
        "--cache_log_file",
        type=str,
        default="train_kohya_log.txt",
        help=("The output log file path"),
    )
    parser.add_argument(
        "--faceid_post_url",
        type=str,
        default=None,
        help=("The post url to get faceid."),
    )
    parser.add_argument(
        "--train_scene_lora_bool",
        action="store_true",
        help=("Whether to train scene lora"),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
    r"""
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict


@gpu_monitor_decorator()
def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder, vae, unet = load_models_from_stable_diffusion_checkpoint(False, args.pretrained_model_ckpt)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Lora will work with this...
    network = network_module.create_network(
        1.0,
        args.rank,
        args.network_alpha,
        vae,
        text_encoder,
        unet,
        neuron_dropout=None,
    )
    network.apply_to(text_encoder, unet, args.train_text_encoder, True)
    trainable_params = network.prepare_optimizer_params(args.learning_rate / 2, args.learning_rate, args.learning_rate)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.set_use_memory_efficient_attention(True, False)
        else:
            logger.warn("xformers is not available. Make sure it is installed correctly")
            unet.set_use_memory_efficient_attention(False, True)
    else:
        unet.set_use_memory_efficient_attention(False, True)

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 or os.environ.get("ENABLE_TF32"):
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer_class = bnb.optim.AdamW8bit
    else:
        if os.environ.get("ENABLE_APEX_OPT"):
            try:
                import apex

                optimizer_class = apex.optimizers.FusedAdam
            except ImportError:
                logger.warn("To use apex FusedAdam, please install fusedAdam,https://github.com/NVIDIA/apex.")
                optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optimizer = optimizer_class(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}")
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}")

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(f"Caption column `{caption_column}` should contain either strings or lists of strings.")
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    persistent_workers = True
    if args.dataloader_num_workers == 0:
        persistent_workers = False
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        persistent_workers=persistent_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, network, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, network, optimizer, train_dataloader, lr_scheduler
        )

    def transform_models_if_DDP(models):
        from torch.nn.parallel import DistributedDataParallel as DDP

        # Transform text_encoder, unet and network from DistributedDataParallel
        return [model.module if type(model) == DDP else model for model in models if model is not None]

    # transform DDP after prepare (train_network here only)
    text_encoder = transform_models_if_DDP([text_encoder])[0]
    unet, network = transform_models_if_DDP([unet, network])

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.)
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint") and not d.endswith("safetensors")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            # network.load_weights(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    if args.template_dir is not None:
        input_images = []
        input_images_shape = []
        control_images = []
        input_masks = []
        if args.template_mask_dir is None:
            retinaface_detection = modelscope_pipeline(Tasks.face_detection, "damo/cv_resnet50_face-detection_retinaface")
        jpgs = os.listdir(args.template_dir)[:4]
        for jpg in jpgs:
            if not jpg.lower().endswith((".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")):
                continue
            read_jpg = os.path.join(args.template_dir, jpg)
            read_jpg = Image.open(read_jpg)
            shape = np.shape(read_jpg)

            short_side = min(read_jpg.width, read_jpg.height)
            resize = float(short_side / 512.0)
            new_size = (int(read_jpg.width // resize) // 64 * 64, int(read_jpg.height // resize) // 64 * 64)
            read_jpg = read_jpg.resize(new_size)

            if args.template_mask:
                if args.template_mask_dir is not None:
                    input_mask = Image.open(os.path.join(args.template_mask_dir, jpg))
                else:
                    _, _, input_mask = call_face_crop(retinaface_detection, read_jpg, crop_ratio=1.3)

            # append into list
            input_images.append(read_jpg)
            input_images_shape.append(shape)
            input_masks.append(input_mask if args.template_mask else None)
            control_images.append(None)
    else:
        new_size = None
        input_images = None
        input_images_shape = None
        input_masks = None
        control_images = None

    # function for saving/removing
    def save_model(ckpt_file, unwrapped_nw):
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        if args.train_scene_lora_bool:
            metadata = {
                "ep_lora_version": "scene",
                "ep_prompt": args.validation_prompt,
            }
        else:
            metadata = None
        unwrapped_nw.save_weights(ckpt_file, weight_dtype, metadata)

    user_id = os.path.basename(os.path.dirname(args.output_dir))
    # check log path
    if accelerator.is_main_process:
        output_log = open(args.cache_log_file, "w")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = network.get_trainable_params()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if global_step % 10 == 0:
                    log_line = f"{str(time.asctime(time.localtime(time.time())))} training lora of {user_id} at step {global_step} / {args.max_train_steps}\n"
                    if accelerator.is_main_process:
                        output_log.write(log_line)
                        output_log.flush()

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        safetensor_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                        accelerator_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_model(safetensor_save_path, accelerator.unwrap_model(network))
                        if args.save_state:
                            accelerator.save_state(accelerator_save_path)

                        logger.info(f"Saved state to {safetensor_save_path}, {accelerator_save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    if (
                        args.validation_steps is not None
                        and args.validation_prompt is not None
                        and global_step % args.validation_steps == 0
                        and args.validation
                    ):
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                            f" {args.validation_prompt}."
                        )
                        try:
                            log_validation(
                                network,
                                noise_scheduler,
                                vae,
                                text_encoder,
                                tokenizer,
                                unet,
                                args,
                                accelerator,
                                weight_dtype,
                                epoch,
                                global_step,
                                input_images=input_images,
                                input_images_shape=input_images_shape,
                                control_images=control_images,
                                input_masks=input_masks,
                                new_size=new_size,
                            )
                        except Exception as e:
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                            logger.info(f"Running validation error, skip it." f"Error info: {e}.")

        if accelerator.is_main_process:
            if (
                args.validation_steps is None
                and args.validation_prompt is not None
                and global_step % args.validation_epochs == 0
                and args.validation
            ):
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:" f" {args.validation_prompt}."
                )
                try:
                    log_validation(
                        network,
                        noise_scheduler,
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        args,
                        accelerator,
                        weight_dtype,
                        epoch,
                        global_step,
                        input_images=input_images,
                        input_images_shape=input_images_shape,
                        control_images=control_images,
                        input_masks=input_masks,
                        new_size=new_size,
                    )
                except Exception as e:
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    logger.info(f"Running validation error, skip it." f"Error info: {e}.")
    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        safetensor_save_path = os.path.join(args.output_dir, f"pytorch_lora_weights.safetensors")
        accelerator_save_path = os.path.join(args.output_dir, f"pytorch_lora_weights")
        save_model(safetensor_save_path, accelerator.unwrap_model(network))
        if args.save_state:
            accelerator.save_state(accelerator_save_path)

        if args.validation:
            try:
                log_validation(
                    network,
                    noise_scheduler,
                    vae,
                    text_encoder,
                    tokenizer,
                    unet,
                    args,
                    accelerator,
                    weight_dtype,
                    epoch,
                    global_step,
                    input_images=input_images,
                    input_images_shape=input_images_shape,
                    control_images=control_images,
                    input_masks=input_masks,
                    new_size=new_size,
                )
            except Exception as e:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info(f"Running validation error, skip it." f"Error info: {e}.")

        if args.merge_best_lora_based_face_id and args.validation:
            pivot_dir = os.path.join(args.train_data_dir, "train")
            merge_best_lora_name = args.train_data_dir.split("/")[-1] if args.merge_best_lora_name is None else args.merge_best_lora_name
            t_result_list, tlist, scores = eval_jpg_with_faceid(pivot_dir, os.path.join(args.output_dir, "validation"))

            for index, line in enumerate(zip(tlist, scores)):
                print(f"Top-{str(index)}: {str(line)}")
                logger.info(f"Top-{str(index)}: {str(line)}")

            best_outputs_dir = os.path.join(args.output_dir, "best_outputs")
            os.makedirs(best_outputs_dir, exist_ok=True)

            # If all training images cannot detect faces, Lora fusion will not be performed
            # Otherwise, the face ID score will be calculated based on the training images and the validated images for Lora fusion.
            if len(t_result_list) == 0:
                print("Dectect no face in training data, move last weights and validation image to best_outputs")
                test_img_dir = os.path.join(args.output_dir, "validation")
                img_list = (
                    glob(os.path.join(test_img_dir, "*.jpg"))
                    + glob(os.path.join(test_img_dir, "*.JPG"))
                    + glob(os.path.join(test_img_dir, "*.png"))
                    + glob(os.path.join(test_img_dir, "*.PNG"))
                )

                t_result_list = []
                for img in img_list:
                    res = int(img.split("_")[-2])
                    t_result_list.append([res, img])
                    t_result_list = sorted(t_result_list, key=lambda a: -a[0])

                copyfile(t_result_list[0][1], os.path.join(best_outputs_dir, os.path.basename(t_result_list[0][1])))
                copyfile(
                    os.path.join(args.output_dir, "pytorch_lora_weights.safetensors"),
                    os.path.join(best_outputs_dir, merge_best_lora_name + ".safetensors"),
                )
            else:
                lora_save_path = network_module.merge_from_name_and_index(merge_best_lora_name, tlist, output_dir=args.output_dir)
                logger.info(f"Save Best Merged Loras To:{lora_save_path}.")

                for result in t_result_list[:1]:
                    copyfile(result, os.path.join(best_outputs_dir, os.path.basename(result)))
                copyfile(lora_save_path, os.path.join(best_outputs_dir, os.path.basename(lora_save_path)))
        else:
            best_outputs_dir = os.path.join(args.output_dir, "best_outputs")
            os.makedirs(best_outputs_dir, exist_ok=True)
            merge_best_lora_name = args.train_data_dir.split("/")[-1] if args.merge_best_lora_name is None else args.merge_best_lora_name
            copyfile(
                os.path.join(args.output_dir, "pytorch_lora_weights.safetensors"),
                os.path.join(best_outputs_dir, merge_best_lora_name + ".safetensors"),
            )

        # we will remove cache_log_file after train
        open(args.cache_log_file, "w")

    accelerator.end_training()


if __name__ == "__main__":
    main()
