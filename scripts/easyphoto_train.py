
import os
import platform
import subprocess
import sys
import threading
import time
from glob import glob
from shutil import copyfile

from PIL import Image, ImageOps

from scripts.easyphoto_config import (models_path, cache_log_file_path,
                                      user_id_outpath_samples,
                                      validation_prompt)
from scripts.easyphoto_utils import (check_files_exists_and_download,
                                     check_id_valid)


python_executable_path = sys.executable
check_hash             = True

# Attention! Output of js is str or list, not float or int
def easyphoto_train_forward(
    sd_model_checkpoint: str,
    id_task: str,
    user_id: str,
    resolution: int, val_and_checkpointing_steps: int, max_train_steps: int, steps_per_photos: int,
    train_batch_size: int, gradient_accumulation_steps: int, dataloader_num_workers: int, learning_rate: float, 
    rank: int, network_alpha: int,
    validation: bool,
    main_image: str,
    instance_images: list,
    enable_rl: bool,
    max_rl_time: float,
    timestep_fraction: float,
    *args
):  
    global check_hash

    if user_id == "" or user_id is None:
        return "User id cannot be set to empty."
    if user_id == "none" :
        return "User id cannot be set to none."
    
    ids = []
    if os.path.exists(user_id_outpath_samples):
        _ids = os.listdir(user_id_outpath_samples)
        for _id in _ids:
            if check_id_valid(_id, user_id_outpath_samples, models_path):
                ids.append(_id)
    ids = sorted(ids)

    if user_id in ids:
        return "User id non-repeatability."

    check_files_exists_and_download(check_hash)
    check_hash = False

    # Raw data backup
    original_backup_path    = os.path.join(user_id_outpath_samples, user_id, "original_backup")
    # ref image copy
    ref_image_path          = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")
    # Sam vit path
    model_path              = os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "sam_vit_l_0b3195.pth")

    # Training data retention
    user_path               = os.path.join(user_id_outpath_samples, user_id, "processed_images")
    images_save_path        = os.path.join(user_id_outpath_samples, user_id, "processed_images", "train")
    json_save_path          = os.path.join(user_id_outpath_samples, user_id, "processed_images", "metadata.jsonl")

    # Training weight saving
    weights_save_path       = os.path.join(user_id_outpath_samples, user_id, "user_weights")
    webui_save_path         = os.path.join(models_path, f"Lora/{user_id}.safetensors")
    webui_load_path         = os.path.join(models_path, f"Stable-diffusion", sd_model_checkpoint)
    sd15_save_path          = os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "stable-diffusion-v1-5")
    
    os.makedirs(original_backup_path, exist_ok=True)
    os.makedirs(user_path, exist_ok=True)
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(webui_save_path)), exist_ok=True)

    max_train_steps         = int(min(len(instance_images) * int(steps_per_photos), int(max_train_steps)))
    for index, user_image in enumerate(instance_images):
        image = Image.open(user_image['name'])
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.save(os.path.join(original_backup_path, str(index) + ".jpg"))

    # preprocess
    preprocess_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess.py")
    command = [
        f'{python_executable_path}', f'{preprocess_path}',
        f'--images_save_path={images_save_path}',
        f'--json_save_path={json_save_path}', 
        f'--validation_prompt={validation_prompt}',
        f'--inputs_dir={original_backup_path}',
        f'--main_image_path={main_image}',
        f'--ref_image_path={ref_image_path}',
        f'--sam_model_path={model_path}'
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing the command: {e}")
    
    # check preprocess results
    train_images = glob(os.path.join(images_save_path, "*.jpg"))
    if len(train_images) == 0:
        return "Failed to obtain preprocessed images, please check the preprocessing process"
    if not os.path.exists(json_save_path):
        return "Failed to obtain preprocessed metadata.jsonl, please check the preprocessing process."

    train_kohya_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_kohya/train_lora.py")
    print("train_file_path : ", train_kohya_path)
    
    # extensions/sd-webui-EasyPhoto/train_kohya_log.txt, use to cache log and flush to UI
    print("cache_log_file_path:", cache_log_file_path)
    if not os.path.exists(os.path.dirname(cache_log_file_path)):
        os.makedirs(os.path.dirname(cache_log_file_path), exist_ok=True)

    if platform.system() == 'Windows':
        pwd = os.getcwd()
        dataloader_num_workers = 0 # for solve multi process bug
        command = [
            f'{python_executable_path}', '-m', 'accelerate.commands.launch', '--mixed_precision=fp16', "--main_process_port=3456", f'{train_kohya_path}',
            f'--pretrained_model_name_or_path={os.path.relpath(sd15_save_path, pwd)}',
            f'--pretrained_model_ckpt={os.path.relpath(webui_load_path, pwd)}', 
            f'--train_data_dir={os.path.relpath(user_path, pwd)}',
            '--caption_column=text', 
            f'--resolution={resolution}',
            '--random_flip',
            f'--train_batch_size={train_batch_size}',
            f'--gradient_accumulation_steps={gradient_accumulation_steps}',
            f'--dataloader_num_workers={dataloader_num_workers}', 
            f'--max_train_steps={max_train_steps}',
            f'--checkpointing_steps={val_and_checkpointing_steps}', 
            f'--learning_rate={learning_rate}',
            '--lr_scheduler=constant',
            '--lr_warmup_steps=0', 
            '--train_text_encoder', 
            '--seed=42', 
            f'--rank={rank}',
            f'--network_alpha={network_alpha}', 
            f'--output_dir={os.path.relpath(weights_save_path, pwd)}', 
            f'--logging_dir={os.path.relpath(weights_save_path, pwd)}', 
            '--enable_xformers_memory_efficient_attention', 
            '--mixed_precision=fp16', 
            f'--cache_log_file={cache_log_file_path}'
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")
        
    else:
        command = [
            f'{python_executable_path}', '-m', 'accelerate.commands.launch', '--mixed_precision=fp16', "--main_process_port=3456", f'{train_kohya_path}',
            f'--pretrained_model_name_or_path={sd15_save_path}',
            f'--pretrained_model_ckpt={webui_load_path}', 
            f'--train_data_dir={user_path}',
            '--caption_column=text', 
            f'--resolution={resolution}',
            '--random_flip',
            f'--train_batch_size={train_batch_size}',
            f'--gradient_accumulation_steps={gradient_accumulation_steps}',
            f'--dataloader_num_workers={dataloader_num_workers}', 
            f'--max_train_steps={max_train_steps}',
            f'--checkpointing_steps={val_and_checkpointing_steps}', 
            f'--learning_rate={learning_rate}',
            '--lr_scheduler=constant',
            '--lr_warmup_steps=0', 
            '--train_text_encoder', 
            '--seed=42', 
            f'--rank={rank}',
            f'--network_alpha={network_alpha}', 
            f'--output_dir={weights_save_path}', 
            f'--logging_dir={weights_save_path}', 
            '--enable_xformers_memory_efficient_attention', 
            '--mixed_precision=fp16', 
            f'--cache_log_file={cache_log_file_path}'
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")

    best_weight_path = os.path.join(weights_save_path, f"pytorch_lora_weights.safetensors")
    if not os.path.exists(best_weight_path):
        return "Failed to obtain Lora after training, please check the training process."

    copyfile(best_weight_path, webui_save_path)

    return "The training has been completed."
