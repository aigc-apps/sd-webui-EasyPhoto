import numpy as np

import requests
import json
import base64
import time
import modules.scripts as scripts
from modules.api import api
from PIL import Image
from modules import shared
from modules.shared import opts, state
from threading import Thread
from io import BytesIO
from modules import processing, sd_samplers, shared
from modules.generation_parameters_copypaste import \
    create_override_settings_dict
from modules.images import save_image
from modules.processing import (Processed, StableDiffusionProcessingImg2Img,
                                StableDiffusionProcessingTxt2Img)
from modules.ui import plaintext_to_html
from scripts.paiya_config import user_id_outpath_samples, processed_image_outpath_samples, validation_prompt
from scripts.paiya_utils import get_controlnet_args, init_default_script_args
from scripts.api import reload_conv_forward
from modules.sd_models import get_closet_checkpoint_match, load_model

import importlib
external_code = importlib.import_module('extensions-builtin.sd-webui-controlnet.scripts.external_code', 'external_code')

def paiya_train(
    id_task: str,
    user_id,
    instance_images,
    *args
):
    # 原始数据备份
    original_backup_path    = os.path.join(user_id_outpath_samples, user_id, "original_backup")
    # 人脸的参考备份
    ref_image_path          = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")

    # 训练数据保存
    user_path               = os.path.join(user_id_outpath_samples, user_id, "processed_images")
    images_save_path        = os.path.join(user_id_outpath_samples, user_id, "processed_images", "train")
    json_save_path          = os.path.join(user_id_outpath_samples, user_id, "processed_images", "metadata.jsonl")

    # 训练权重保存
    weights_save_path       = os.path.join("/photog_oss/photog/user_weights", user_id)
    webui_save_path         = os.path.join("/photog_oss/photog/webui/models/Lora", user_id)

    os.makedirs(original_backup_path, exist_ok=True)
    os.makedirs(user_path, exist_ok=True)
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(webui_save_path, exist_ok=True)

    max_train_steps         = min(len(instance_images) * 200, 800)

    os.system(
        f'''
        python preprocess.py --validation_prompt "{validation_prompt}" --ref_image_path {ref_image_path} \
                                    --inputs_dir {original_backup_path} --images_save_path {images_save_path} \
                                    --json_save_path {json_save_path} --crop_ratio 3 --model_cache_dir ./
        '''
    )

    os.system(
        f'''
        accelerate launch --mixed_precision="fp16" --main_process_port=3456 train_kohya/train_lora.py \
        --pretrained_model_name_or_path="ChilloutMix-ni-fp16" \
        --train_data_dir="{user_path}" --caption_column="text" \
        --resolution=512 --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --dataloader_num_workers=16 \
        --max_train_steps={600} --checkpointing_steps=100 \
        --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
        --train_text_encoder \
        --seed=42 \
        --rank=128 --network_alpha=64 \
        --validation_prompt="{validation_prompt}" \
        --validation_steps=100 \
        --output_dir="{weights_save_path}" \
        --logging_dir="{weights_save_path}" \
        --enable_xformers_memory_efficient_attention \
        --mixed_precision='fp16' \
        --template_dir="./images/template" \
        --template_mask \
        --template_mask_dir="./images/template_mask" \
        --merge_best_lora_based_face_id \
        --merge_best_lora_name="{user_id}"
        '''
    )
    
    best_weight_path = os.path.join(weights_save_path, f"best_outputs/{user_id}.safetensors")
    os.system(f"cp -rf {best_weight_path} {webui_save_path}")

    return "训练已经完成。"