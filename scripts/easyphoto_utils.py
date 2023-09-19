import logging
import os
import time
from glob import glob

import requests
from modules.paths import models_path
from scripts.easyphoto_config import data_path
import hashlib

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
    controlnet_extensions_path          = os.path.join(data_path, "extensions", "sd-webui-controlnet")
    controlnet_extensions_builtin_path  = os.path.join(data_path, "extensions-builtin", "sd-webui-controlnet")
    models_annotator_path               = os.path.join(data_path, "models")
    if os.path.exists(controlnet_extensions_path):
        controlnet_annotator_cache_path = os.path.join(controlnet_extensions_path, "annotator/downloads/openpose")
    elif os.path.exists(controlnet_extensions_builtin_path):
        controlnet_annotator_cache_path = os.path.join(controlnet_extensions_builtin_path, "annotator/downloads/openpose")
    else:
        controlnet_annotator_cache_path = os.path.join(models_annotator_path, "annotator/downloads/openpose")

    urls        = [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/ChilloutMix-ni-fp16.safetensors", 
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
        os.path.join(models_path, f"Lora/FilmVelvia3.safetensors"),
        os.path.join(controlnet_annotator_cache_path, f"body_pose_model.pth"),
        os.path.join(controlnet_annotator_cache_path, f"facenet.pth"),
        os.path.join(controlnet_annotator_cache_path, f"hand_pose_model.pth"),
        os.path.join(models_path, f"VAE/vae-ft-mse-840000-ema-pruned.ckpt"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "face_skin.pth"),
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
        if os.path.exists(filename) and check_weight_file(url,filename):
            continue
        print(f"Start Downloading: {url}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        urldownload_progressbar(url, filename)


# check if the weight file has been downloaded completely
def check_weight_file(url,filename):
    if compare_hasd_link_file(url,filename) and compare_size_link_file(url, filename):
        print("link and file all same ")
        return True
    else:
        print("link and file  is different  ")
        return False
    

#Calculate the hash value of the download link and downloaded_file by md5
def compare_hasd_link_file(url, file_path):
    hash_url = hashlib.md5(requests.get(url).content).hexdigest()
    
    hash_object = hashlib.md5()  
    with open(file_path, 'rb') as file:  
        buffer = file.read(4096) 
        while len(buffer) > 0: 
            hash_object.update(buffer)  
            buffer = file.read(4096)  
    hash_file=  hash_object.hexdigest()  
    
    if hash_url == hash_file:
        print("link and file hash same ")
        return True
    else: 
        print("link and file hash different ")
        return False

# Calculate the size of the download link and downloaded_file
def compare_size_link_file(url, file_path):
    size_url = requests.get(url).headers.get('Content-Length')
    size_file = os.path.getsize(file_path)
    if int(size_url) == int(size_file):
        print("link and file size same ")
        return True
    else:
        print("link and file size different ")
        return False