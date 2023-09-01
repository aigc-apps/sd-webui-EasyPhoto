
import os
import platform
import subprocess
import threading
from shutil import copyfile

import time
import requests
from PIL import Image
from glob import glob
from tqdm import tqdm

from modules import shared
from modules.paths import models_path
from scripts.paiya_config import (id_path, user_id_outpath_samples,
                                  validation_prompt)
from scripts.preprocess import preprocess_images


def urldownload_progressbar(url, filepath):
    start = time.time() 
    response = requests.get(url, stream=True)
    size = 0 
    chunk_size = 1024
    content_size = int(response.headers['content-length']) 
    try:
        if response.status_code == 200: 
            print('Start download,[File size]:{size:.2f} MB'.format(size = content_size / chunk_size /1024))  
            with open(filepath,'wb') as file:  
                for data in response.iter_content(chunk_size = chunk_size):
                    file.write(data)
                    size +=len(data)
                    print('\r'+'[下载进度]:%s%.2f%%' % ('>'*int(size*50/ content_size), float(size / content_size * 100)) ,end=' ')
        end = time.time()
        print('Download completed!,times: %.2f秒' % (end - start))
    except:
        print('Error!')

def check_files_exists_and_download():
    urls        = [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/ChilloutMix-ni-fp16.safetensors", 
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11p_sd15_openpose.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11p_sd15_canny.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11f1e_sd15_tile.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_sd15_random_color.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/w600k_r50.onnx",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/2d106det.onnx",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/det_10g.onnx",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/1.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/2.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/3.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/4.jpg",
    ]
    filenames = [
        os.path.join(models_path, f"Stable-diffusion/Chilloutmix-Ni-pruned-fp16-fix.safetensors"),
        os.path.join(models_path, f"ControlNet/control_v11p_sd15_openpose.pth"),
        os.path.join(models_path, f"ControlNet/control_v11p_sd15_canny.pth"),
        os.path.join(models_path, f"ControlNet/control_v11f1e_sd15_tile.pth"),
        os.path.join(models_path, f"ControlNet/control_sd15_random_color.pth"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "buffalo_l", "w600k_r50.onnx"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "buffalo_l", "2d106det.onnx"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "buffalo_l", "det_10g.onnx"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "1.jpg"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "2.jpg"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "3.jpg"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "4.jpg"),
    ]
    print("Start Downloading weights")
    for url, filename in zip(urls, filenames):
        if os.path.exists(filename):
            continue
        print(f"Start Downloading: {url}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        urldownload_progressbar(url, filename)

# Attention! Output of js is str or list, not float or int
def paiya_train_forward(
    id_task: str,
    user_id: str,
    resolution: int, val_and_checkpointing_steps: int, max_train_steps: int, steps_per_photos: int,
    train_batch_size: int, gradient_accumulation_steps: int, dataloader_num_workers: int, learning_rate: float, 
    rank: int, network_alpha: int,
    instance_images: list,
    *args
):  
    if user_id == "":
        return "User id不能为空。"
    
    if os.path.exists(id_path):
        with open(id_path, "r") as f:
            ids = f.readlines()
        ids = [_id.strip() for _id in ids]
    else:
        ids = []
    if user_id in ids:
        return "User id 不能重复。"

    check_files_exists_and_download()
    # 模板的地址
    training_templates_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates")
    # 原始数据备份
    original_backup_path    = os.path.join(user_id_outpath_samples, user_id, "original_backup")
    # 人脸的参考备份
    ref_image_path          = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")

    # 训练数据保存
    user_path               = os.path.join(user_id_outpath_samples, user_id, "processed_images")
    images_save_path        = os.path.join(user_id_outpath_samples, user_id, "processed_images", "train")
    json_save_path          = os.path.join(user_id_outpath_samples, user_id, "processed_images", "metadata.jsonl")

    # 训练权重保存
    weights_save_path       = os.path.join(user_id_outpath_samples, user_id, "user_weights")
    webui_save_path         = os.path.join(models_path, f"Lora/{user_id}.safetensors")
    webui_load_path         = os.path.join(models_path, f"Stable-diffusion/Chilloutmix-Ni-pruned-fp16-fix.safetensors")

    os.makedirs(original_backup_path, exist_ok=True)
    os.makedirs(user_path, exist_ok=True)
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(webui_save_path)), exist_ok=True)

    max_train_steps         = int(min(len(instance_images) * int(steps_per_photos), int(max_train_steps)))

    for index, user_image in enumerate(instance_images):
        image = Image.open(user_image['name']).convert("RGB")
        image.save(os.path.join(original_backup_path, str(index) + ".jpg"))
        
    sub_threading = threading.Thread(target=preprocess_images, args=(images_save_path, json_save_path, validation_prompt, original_backup_path, ref_image_path,))
    sub_threading.start()
    sub_threading.join()

    train_images = glob(os.path.join(images_save_path, "*.jpg"))
    if len(images_save_path) == 0:
        return "未能获得预处理后的图片，请检查训练过程。"
    if not os.path.exists(json_save_path):
        return "未能获得预处理后的metadata.jsonl，请检查训练过程。"

    train_kohya_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_kohya/train_lora.py")
    print(train_kohya_path)
    if platform.system() == 'Windows':
        command = [
            'accelerate', 'launch', '--mixed_precision="fp16', "--main_process_port=3456", f'{train_kohya_path}',
            '--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"', 
            f'--pretrained_model_ckpt="{webui_load_path}"', 
            f'--train_data_dir="{user_path}" ', 
            '--caption_column="text"', 
            f'--resolution={resolution} ', 
            '--random_flip ', 
            f'--train_batch_size={train_batch_size} ', 
            f'--gradient_accumulation_steps={gradient_accumulation_steps} ', 
            f'--dataloader_num_workers={dataloader_num_workers}', 
            f'--max_train_steps={max_train_steps} ', 
            f'--checkpointing_steps={val_and_checkpointing_steps}', 
            f'--learning_rate={learning_rate} ', 
            '--lr_scheduler="constant" ', 
            '--lr_warmup_steps=0', 
            '--train_text_encoder', 
            '--seed=42', 
            f'--rank={rank} ', 
            f'--network_alpha={network_alpha}', 
            f'--validation_prompt="{validation_prompt}"', 
            f'--validation_steps={val_and_checkpointing_steps}', 
            f'--output_dir="{weights_save_path}"', 
            f'--logging_dir="{weights_save_path}"', 
            '--enable_xformers_memory_efficient_attention', 
            '--mixed_precision="fp16"', 
            f'--template_dir="{training_templates_path}"', 
            '--template_mask', 
            '--merge_best_lora_based_face_id', 
            f'--merge_best_lora_name="{user_id}"', 
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")
    else:
        os.system(
            f'''
            accelerate launch --mixed_precision="fp16" --main_process_port=3456 {train_kohya_path} \
            --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
            --pretrained_model_ckpt="{webui_load_path}" \
            --train_data_dir="{user_path}" --caption_column="text" \
            --resolution={resolution} --random_flip --train_batch_size={train_batch_size} --gradient_accumulation_steps={gradient_accumulation_steps} --dataloader_num_workers={dataloader_num_workers} \
            --max_train_steps={max_train_steps} --checkpointing_steps={val_and_checkpointing_steps} \
            --learning_rate={learning_rate} --lr_scheduler="constant" --lr_warmup_steps=0 \
            --train_text_encoder \
            --seed=42 \
            --rank={rank} --network_alpha={network_alpha} \
            --validation_prompt="{validation_prompt}" \
            --validation_steps={val_and_checkpointing_steps} \
            --output_dir="{weights_save_path}" \
            --logging_dir="{weights_save_path}" \
            --enable_xformers_memory_efficient_attention \
            --mixed_precision='fp16' \
            --template_dir="{training_templates_path}" \
            --template_mask \
            --merge_best_lora_based_face_id \
            --merge_best_lora_name="{user_id}"
            '''
        )
    
    best_weight_path = os.path.join(weights_save_path, f"best_outputs/{user_id}.safetensors")
    copyfile(best_weight_path, webui_save_path)
    if not os.path.exists(webui_save_path):
        return "未能获得训练后的Lora，请检查训练过程。"

    with open(id_path, "a") as f:
        f.write(f"{user_id}\n")
    return "训练已经完成。"