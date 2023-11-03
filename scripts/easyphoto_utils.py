import datetime
import hashlib
import logging
import gc
import os
import time
from glob import glob

import requests
import torch
from modules.paths import models_path
from tqdm import tqdm

import scripts.easyphoto_infer
from scripts.easyphoto_config import data_path
from modelscope.utils.logger import get_logger as ms_get_logger

# Ms logger set
ms_logger = ms_get_logger()
ms_logger.setLevel(logging.ERROR)

# ep logger set
ep_logger_name = __name__.split('.')[0]
ep_logger = logging.getLogger(ep_logger_name)
ep_logger.propagate = False

for handler in ep_logger.root.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
handlers = [stream_handler]

for handler in handlers:
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
    handler.setLevel("INFO")
    ep_logger.addHandler(handler)

ep_logger.setLevel("INFO")

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
        controlnet_cache_path = controlnet_extensions_path
    elif os.path.exists(controlnet_extensions_builtin_path):
        controlnet_annotator_cache_path = os.path.join(controlnet_extensions_builtin_path, "annotator/downloads/openpose")
        controlnet_cache_path = controlnet_extensions_builtin_path
    else:
        controlnet_annotator_cache_path = os.path.join(models_annotator_path, "annotator/downloads/openpose")
        controlnet_cache_path = controlnet_extensions_path

    # The models are from civitai/6424 & civitai/118913, we saved them to oss for your convenience in downloading the models.
    urls        = [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/ChilloutMix-ni-fp16.safetensors", 
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/SDXL_1.0_ArienMixXL_v2.0.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11p_sd15_openpose.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11p_sd15_canny.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11f1e_sd15_tile.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_sd15_random_color.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/diffusers_xl_canny_mid.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/FilmVelvia3.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/body_pose_model.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/facenet.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/hand_pose_model.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/vae-ft-mse-840000-ema-pruned.ckpt",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/madebyollin_sdxl_vae_fp16_fix/diffusion_pytorch_model.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/madebyollin-sdxl-vae-fp16-fix.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/face_skin.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/face_landmarks.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/makeup_transfer.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/1.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/2.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/3.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/4.jpg",
    ]
    filenames = [
        os.path.join(models_path, f"Stable-diffusion/Chilloutmix-Ni-pruned-fp16-fix.safetensors"),
        os.path.join(models_path, f"Stable-diffusion/SDXL_1.0_ArienMixXL_v2.0.safetensors"),
        [os.path.join(models_path, f"ControlNet/control_v11p_sd15_openpose.pth"), os.path.join(controlnet_cache_path, f"models/control_v11p_sd15_openpose.pth")],
        [os.path.join(models_path, f"ControlNet/control_v11p_sd15_canny.pth"), os.path.join(controlnet_cache_path, f"models/control_v11p_sd15_canny.pth")],
        [os.path.join(models_path, f"ControlNet/control_v11f1e_sd15_tile.pth"), os.path.join(controlnet_cache_path, f"models/control_v11f1e_sd15_tile.pth")],
        [os.path.join(models_path, f"ControlNet/control_sd15_random_color.pth"), os.path.join(controlnet_cache_path, f"models/control_sd15_random_color.pth")],
        [os.path.join(models_path, f"ControlNet/diffusers_xl_canny_mid.safetensors"), os.path.join(controlnet_cache_path, f"models/diffusers_xl_canny_mid.safetensors")],
        os.path.join(models_path, f"Lora/FilmVelvia3.safetensors"),
        os.path.join(controlnet_annotator_cache_path, f"body_pose_model.pth"),
        os.path.join(controlnet_annotator_cache_path, f"facenet.pth"),
        os.path.join(controlnet_annotator_cache_path, f"hand_pose_model.pth"),
        os.path.join(models_path, f"VAE/vae-ft-mse-840000-ema-pruned.ckpt"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models/stable-diffusion-xl/madebyollin_sdxl_vae_fp16_fix"), "diffusion_pytorch_model.safetensors"),
        os.path.join(models_path, f"VAE/madebyollin-sdxl-vae-fp16-fix.safetensors"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "face_skin.pth"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "face_landmarks.pth"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "makeup_transfer.pth"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "1.jpg"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "2.jpg"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "3.jpg"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "4.jpg"),
    ]
    # This print will introduce some misundertand
    # print("Start Downloading weights")
    for url, filename in zip(urls, filenames):
        if type(filename) is str:
            filename = [filename]
        
        exist_flag = False
        for _filename in filename:
            if not check_hash:
                if os.path.exists(_filename):
                    exist_flag = True
                    break
            else:
                if os.path.exists(_filename) and compare_hasd_link_file(url, _filename):
                    exist_flag = True
                    break
        if exist_flag:
            continue

        ep_logger.info(f"Start Downloading: {url}")
        os.makedirs(os.path.dirname(filename[0]), exist_ok=True)
        urldownload_progressbar(url, filename[0])
       
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
        ep_logger.info(f"{file_path} : Hash match")
        return True
      
    else:
        ep_logger.info(f" {file_path} : Hash mismatch")
        return False
      

def unload_models():
    """Unload models to free VRAM.
    """
    scripts.easyphoto_infer.retinaface_detection = None
    scripts.easyphoto_infer.image_face_fusion = None
    scripts.easyphoto_infer.skin_retouching = None
    scripts.easyphoto_infer.portrait_enhancement = None
    scripts.easyphoto_infer.face_skin = None
    scripts.easyphoto_infer.face_recognition = None
    scripts.easyphoto_infer.psgan_inference = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()