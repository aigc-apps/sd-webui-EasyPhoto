import datetime
import gc
import hashlib
import logging
import os
import traceback
from contextlib import ContextDecorator
from glob import glob

import cv2
import numpy as np
import requests
import scripts.easyphoto_infer
import torch
import torchvision
from modelscope.utils.logger import get_logger as ms_get_logger
from modules.paths import models_path
from PIL import Image
from scripts.easyphoto_config import data_path
from tqdm import tqdm

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

# download path
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

download_urls = {
    # The models are from civitai/6424 & civitai/118913, we saved them to oss for your convenience in downloading the models.
    "base": [
        # base model
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/ChilloutMix-ni-fp16.safetensors", 
        # controlnets
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11p_sd15_openpose.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11p_sd15_canny.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_v11f1e_sd15_tile.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_sd15_random_color.pth",
        # loras
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/FilmVelvia3.safetensors",
        # controlnet annotator
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/body_pose_model.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/facenet.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/hand_pose_model.pth",
        # vaes
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/vae-ft-mse-840000-ema-pruned.ckpt",
        # other models
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/face_skin.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/face_landmarks.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/makeup_transfer.pth",
        # templates
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/1.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/2.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/3.jpg",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/4.jpg",
    ],
    "sdxl": [
        # sdxl
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/diffusers_xl_canny_mid.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/madebyollin_sdxl_vae_fp16_fix/diffusion_pytorch_model.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/madebyollin-sdxl-vae-fp16-fix.safetensors",
    ],
    "add_text2image": [
        # LZ 16k for text2image
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/LZ-16K%2BOptics.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/001.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/002.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/003.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/004.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/005.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/006.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/007.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/008.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/009.png",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/pose_templates/010.png",
    ],
    "add_video": [
        # new backbone for video
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/majicmixRealistic_v7.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/mm_sd_v15_v2.ckpt",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/flownet.pkl",
    ],

    # Scene Lora Collection
    "Christmas_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/Christmas_1.safetensors",
    ],
    "FairMaidenStyle_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/FairMaidenStyle_1.safetensors",
    ],
    "Gentleman_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/Gentleman_1.safetensors",
    ],
    "GuoFeng_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/GuoFeng_1.safetensors",
    ],
    "GuoFeng_2": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/GuoFeng_2.safetensors",
    ],
    "GuoFeng_3": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/GuoFeng_3.safetensors",
    ],
    "GuoFeng_4": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/GuoFeng_4.safetensors",
    ],
    "Minimalism_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/Minimalism_1.safetensors",
    ],
    "NaturalWind_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/NaturalWind_1.safetensors",
    ],
    "Princess_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/Princess_1.safetensors",
    ],
    "Princess_2": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/Princess_2.safetensors",
    ],
    "Princess_3": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/Princess_3.safetensors",
    ],
    "SchoolUniform_1": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/SchoolUniform_1.safetensors",
    ],
    "SchoolUniform_2": [
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/scene_lora/SchoolUniform_2.safetensors",
    ],
}
save_filenames = {
    # The models are from civitai/6424 & civitai/118913, we saved them to oss for your convenience in downloading the models.
    "base": [
        # base model
        os.path.join(models_path, f"Stable-diffusion/Chilloutmix-Ni-pruned-fp16-fix.safetensors"),
        # controlnets
        [os.path.join(models_path, f"ControlNet/control_v11p_sd15_openpose.pth"), os.path.join(controlnet_cache_path, f"models/control_v11p_sd15_openpose.pth")],
        [os.path.join(models_path, f"ControlNet/control_v11p_sd15_canny.pth"), os.path.join(controlnet_cache_path, f"models/control_v11p_sd15_canny.pth")],
        [os.path.join(models_path, f"ControlNet/control_v11f1e_sd15_tile.pth"), os.path.join(controlnet_cache_path, f"models/control_v11f1e_sd15_tile.pth")],
        [os.path.join(models_path, f"ControlNet/control_sd15_random_color.pth"), os.path.join(controlnet_cache_path, f"models/control_sd15_random_color.pth")],
        # loras
        os.path.join(models_path, f"Lora/FilmVelvia3.safetensors"),
        # controlnet annotator
        os.path.join(controlnet_annotator_cache_path, f"body_pose_model.pth"),
        os.path.join(controlnet_annotator_cache_path, f"facenet.pth"),
        os.path.join(controlnet_annotator_cache_path, f"hand_pose_model.pth"),
        # vaes
        os.path.join(models_path, f"VAE/vae-ft-mse-840000-ema-pruned.ckpt"),
        # other models
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "face_skin.pth"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "face_landmarks.pth"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "makeup_transfer.pth"),
        # templates
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "1.jpg"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "2.jpg"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "3.jpg"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "training_templates", "4.jpg"),
    ],
    "sdxl": [
        [os.path.join(models_path, f"ControlNet/diffusers_xl_canny_mid.safetensors"), os.path.join(controlnet_cache_path, f"models/diffusers_xl_canny_mid.safetensors")],
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models/stable-diffusion-xl/madebyollin_sdxl_vae_fp16_fix"), "diffusion_pytorch_model.safetensors"),
        os.path.join(models_path, f"VAE/madebyollin-sdxl-vae-fp16-fix.safetensors"),
    ],
    "add_text2image": [
        # sdxl for text2image
        os.path.join(models_path, f"Stable-diffusion/LZ-16K+Optics.safetensors"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "pose_templates", "001.png"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "pose_templates", "002.png"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "pose_templates", "003.png"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "pose_templates", "004.png"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "pose_templates", "005.png"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "pose_templates", "006.png"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "pose_templates", "007.png"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "pose_templates", "008.png"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "pose_templates", "009.png"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "pose_templates", "010.png")
    ],
    "add_video": [
        # new backbone for video
        os.path.join(models_path, f"Stable-diffusion/majicmixRealistic_v7.safetensors"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "mm_sd_v15_v2.ckpt"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "flownet.pkl"),
    ], 

    # Scene Lora Collection
    "Christmas_1": [
       os.path.join(models_path, f"Lora/Christmas_1.safetensors"),
    ],
    "FairMaidenStyle_1": [
       os.path.join(models_path, f"Lora/FairMaidenStyle_1.safetensors"),
    ],
    "Gentleman_1": [
       os.path.join(models_path, f"Lora/Gentleman_1.safetensors"),
    ],
    "GuoFeng_1": [
       os.path.join(models_path, f"Lora/GuoFeng_1.safetensors"),
    ],
    "GuoFeng_2": [
       os.path.join(models_path, f"Lora/GuoFeng_2.safetensors"),
    ],
    "GuoFeng_3": [
       os.path.join(models_path, f"Lora/GuoFeng_3.safetensors"),
    ],
    "GuoFeng_4": [
       os.path.join(models_path, f"Lora/GuoFeng_4.safetensors"),
    ],
    "Minimalism_1": [
       os.path.join(models_path, f"Lora/Minimalism_1.safetensors"),
    ],
    "NaturalWind_1": [
       os.path.join(models_path, f"Lora/NaturalWind_1.safetensors"),
    ],
    "Princess_1": [
       os.path.join(models_path, f"Lora/Princess_1.safetensors"),
    ],
    "Princess_2": [
       os.path.join(models_path, f"Lora/Princess_2.safetensors"),
    ],
    "Princess_3": [
       os.path.join(models_path, f"Lora/Princess_3.safetensors"),
    ],
    "SchoolUniform_1": [
       os.path.join(models_path, f"Lora/SchoolUniform_1.safetensors"),
    ],
    "SchoolUniform_2": [
       os.path.join(models_path, f"Lora/SchoolUniform_2.safetensors"),
    ]
}

def check_scene_valid(lora_path, models_path):
    from scripts.sdwebui import get_lora_type
    safetensors_lora_path = os.path.join(models_path, "Lora", lora_path) 
    if not safetensors_lora_path.endswith("safetensors"):
        return False
    lora_type = get_lora_type(safetensors_lora_path)
    if lora_type == 4:
        return True
    return False

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

def check_files_exists_and_download(check_hash, download_mode="base"):
    urls, filenames = download_urls[download_mode], save_filenames[download_mode]

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
      
def get_mov_all_images(file: str, required_fps: int) -> tuple:
    """
    Extracts a specific number of frames uniformly from a video file and converts them to a list of RGB images.

    required_fps < video fps , uniform sampling frames
    required_fps >= video fps, get all frames

    Parameters:
    - file (str): The path to the video file.
    - required_fps (int): The required frame per second to extract.

    Returns:
    - image_list (tuple): A tuple containing a list of RGB images and the actual number of frames extracted.
             Returns None if the file cannot be opened or if 'file' is None.
    - required_fps (int): The actual fps after extracting a specific number of frames uniformly from a video file.
    """
    if file is None:
        return None
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        return None
    
    # Frames cannot be greater than the actual fps for sampling
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if required_fps > fps:
        print('Waring: The set number of frames is greater than the number of video frames')
        required_fps = fps

    # Get all frames
    movies = []
    while (True):
        flag, frame = cap.read()
        if not flag:
            break
        else:
            movies.append(frame)
    # Obtain the required frame
    # Extracts a specific number of frames uniformly from a video
    num_pics        = int(required_fps / fps * len(movies))
    target_indexs   = list(np.rint(np.linspace(0, len(movies)-1, num=num_pics)))
    image_list = []
    for index in target_indexs:
        frame = movies[int(index)]
        image_list.append(frame)

    cap.release()

    image_list = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in image_list]
    return image_list, required_fps

def convert_to_video(path, frames, fps, prefix = None, mode="gif"):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    index = len([path for path in os.listdir(path)]) + 1
    if prefix is None:
        prefix = str(index).zfill(8)
    video_path = os.path.join(path, prefix + f'.{mode}')

    if mode == "gif":
        import imageio.v3 as imageio
        try:
            import av
        except ImportError:
            from launch import run_pip
            run_pip(
                "install imageio[pyav]",
                "sd-webui-animatediff GIF palette optimization requirement: imageio[pyav]",
            )
        video_array = [np.array(v) for v in frames]
        imageio.imwrite(
            video_path, video_array, plugin='pyav', fps=fps, 
            codec='gif', out_pixel_format='pal8',
            filter_graph=(
                {
                    "split": ("split", ""),
                    "palgen": ("palettegen", ""),
                    "paluse": ("paletteuse", ""),
                    "scale": ("scale", f"{frames[0].width}:{frames[0].height}")
                },
                [
                    ("video_in", "scale", 0, 0),
                    ("scale", "split", 0, 0),
                    ("split", "palgen", 1, 0),
                    ("split", "paluse", 0, 0),
                    ("palgen", "paluse", 0, 1),
                    ("paluse", "video_out", 0, 0),
                ]
            )
        )
        
        return None, video_path, prefix
    else:
        frames = [np.array(frame) for frame in frames]
        frames = torch.from_numpy(np.array(frames))
        if not os.path.exists(os.path.dirname(video_path)):
            os.makedirs(os.path.dirname(video_path))
        torchvision.io.write_video(video_path, frames, fps=fps, video_codec="libx264")
    
        return video_path, None, prefix

def modelscope_models_to_cpu():
    """Load models to cpu to free VRAM.
    """
    ms_models = [
        scripts.easyphoto_infer.retinaface_detection,
        scripts.easyphoto_infer.image_face_fusion,
        scripts.easyphoto_infer.skin_retouching,
        scripts.easyphoto_infer.portrait_enhancement,
        scripts.easyphoto_infer.face_skin,
        scripts.easyphoto_infer.face_recognition,
        scripts.easyphoto_infer.psgan_inference,
    ]
    for ms_model in ms_models:
        if hasattr(ms_model, "__dict__"):
            for key in ms_model.__dict__.keys():
                try:
                    if hasattr(getattr(ms_model, key), "cpu"):
                        getattr(ms_model, key).cpu()
                except Exception as e:
                    traceback.print_exc()
                    ep_logger.info(f"{str(ms_model)}.{key} has no cpu(), detailed error infor is {e}")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def modelscope_models_to_gpu():
    """Load models to cuda.
    """
    ms_models = [
        scripts.easyphoto_infer.retinaface_detection,
        scripts.easyphoto_infer.image_face_fusion,
        scripts.easyphoto_infer.skin_retouching,
        scripts.easyphoto_infer.portrait_enhancement,
        scripts.easyphoto_infer.face_skin,
        scripts.easyphoto_infer.face_recognition,
        scripts.easyphoto_infer.psgan_inference,
    ]
    for ms_model in ms_models:
        if hasattr(ms_model, "__dict__"):
            for key in ms_model.__dict__.keys():
                try:
                    if hasattr(getattr(ms_model, key), "cuda"):
                        getattr(ms_model, key).cuda()
                except Exception as e:
                    traceback.print_exc()
                    ep_logger.info(f"{str(ms_model)}.{key} has no cuda(), detailed error infor is {e}")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

class switch_ms_model_cpu(ContextDecorator):
    """Context-manager that supports switch modelscope models to cpu and cuda
    """
    def __enter__(self):
        modelscope_models_to_cpu()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        modelscope_models_to_gpu()

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
