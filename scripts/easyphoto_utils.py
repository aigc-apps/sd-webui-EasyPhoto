import datetime
import hashlib
import logging
import os
import time
from glob import glob

import requests
from modules.paths import models_path
from tqdm import tqdm

from scripts.easyphoto_config import data_path

# Set the level of the logger
log_level = os.environ.get('LOG_LEVEL', 'INFO')
logging.getLogger().setLevel(log_level)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s')  

def check_id_valid(user_id, user_id_outpath_samples, models_path):
    face_id_image_path = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg") 
    if not os.path.exists(face_id_image_path):
        return False
    
    safetensors_lora_path   = os.path.join(models_path, "Lora", f"{user_id}.safetensors") 
    ckpt_lora_path          = os.path.join(models_path, "Lora", f"{user_id}.ckpt") 
    if not (os.path.exists(safetensors_lora_path) or os.path.exists(ckpt_lora_path)):
        return False
    return True

def urldownload_progressbar(url, file_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()

def check_files_exists_and_download(check_hash):
    controlnet_extensions_path          = os.path.join(data_path, "extensions", "sd-webui-controlnet")
    controlnet_extensions_builtin_path  = os.path.join(data_path, "extensions-builtin", "sd-webui-controlnet")
    models_annotator_path               = os.path.join(data_path, "models")
    if os.path.exists(controlnet_extensions_path):
        controlnet_annotator_cache_path = os.path.join(controlnet_extensions_path, "annotator/downloads/openpose")
    elif os.path.exists(controlnet_extensions_builtin_path):
        controlnet_annotator_cache_path = os.path.join(controlnet_extensions_builtin_path, "annotator/downloads/openpose")
    else:
        controlnet_annotator_cache_path = os.path.join(models_annotator_path, "annotator/downloads/openpose")

    # The models are from civitai/6424 & civitai/118913, we saved them to oss for your convenience in downloading the models.
    urls        = [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/ChilloutMix-ni-fp16.safetensors", 
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/SDXL_1.0_ArienMixXL_v2.0.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11p_sd15_openpose.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11p_sd15_canny.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11f1e_sd15_tile.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_sd15_random_color.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/FilmVelvia3.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/body_pose_model.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/facenet.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/hand_pose_model.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/vae-ft-mse-840000-ema-pruned.ckpt",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/face_skin.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/1.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/2.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/3.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/4.jpg",
    ]
    filenames = [
        os.path.join(models_path, f"Stable-diffusion/Chilloutmix-Ni-pruned-fp16-fix.safetensors"),
        os.path.join(models_path, f"Stable-diffusion/SDXL_1.0_ArienMixXL_v2.0.safetensors"),
        os.path.join(models_path, f"ControlNet/control_v11p_sd15_openpose.pth"),
        os.path.join(models_path, f"ControlNet/control_v11p_sd15_canny.pth"),
        os.path.join(models_path, f"ControlNet/control_v11f1e_sd15_tile.pth"),
        os.path.join(models_path, f"ControlNet/control_sd15_random_color.pth"),
        os.path.join(models_path, f"Lora/FilmVelvia3.safetensors"),
        os.path.join(controlnet_annotator_cache_path, f"body_pose_model.pth"),
        os.path.join(controlnet_annotator_cache_path, f"facenet.pth"),
        os.path.join(controlnet_annotator_cache_path, f"hand_pose_model.pth"),
        os.path.join(models_path, f"VAE/vae-ft-mse-840000-ema-pruned.ckpt"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "face_skin.pth"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "1.jpg"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "2.jpg"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "3.jpg"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "4.jpg"),
    ]
    print("Start Downloading weights")
    for url, filename in zip(urls, filenames):
        if not check_hash:
            if os.path.exists(filename):
                continue
        else:
            if os.path.exists(filename) and compare_hasd_link_file(url, filename):
                continue
        print(f"Start Downloading: {url}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        urldownload_progressbar(url, filename)
       
# Calculate the hash value of the download link and downloaded_file by sha256
def compare_hasd_link_file(url, file_path):
    r           = requests.head(url)
    total_size  = int(r.headers['Content-Length'])
    
    res = requests.get(url, stream=True)
    remote_head_hash = hashlib.sha256(res.raw.read(1000)).hexdigest()  
    res.close()
    
    end_pos = total_size - 1000
    headers = {'Range': f'bytes={end_pos}-{total_size-1}'}
    res = requests.get(url, headers=headers, stream=True)
    remote_end_hash = hashlib.sha256(res.content).hexdigest()
    res.close()
    
    with open(file_path,'rb') as f:
        local_head_data = f.read(1000)
        local_head_hash = hashlib.sha256(local_head_data).hexdigest()
    
        f.seek(end_pos)
        local_end_data = f.read(1000) 
        local_end_hash = hashlib.sha256(local_end_data).hexdigest()
     
    if remote_head_hash == local_head_hash and remote_end_hash == local_end_hash:
        print(f"{file_path} : Hash match")
        return True
      
    else:
        print(f" {file_path} : Hash mismatch")
        return False
      
