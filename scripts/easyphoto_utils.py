import hashlib
import logging
import os

import requests
from modules.paths import models_path
from tqdm import tqdm

from scripts.easyphoto_config import data_path

# Set the level of the logger
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.getLogger().setLevel(log_level)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(message)s")


# ep logger set
ep_logger_name = "EasyPhoto"
ep_logger = logging.getLogger(ep_logger_name)
ep_logger.propagate = False

for handler in ep_logger.root.handlers:
    if type(handler) is logging.StreamHandler:
        handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
handlers = [stream_handler]

for handler in handlers:
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    handler.setLevel("INFO")
    ep_logger.addHandler(handler)

ep_logger.setLevel("INFO")


def check_id_valid(user_id, user_id_outpath_samples, models_path):
    face_id_image_path = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")
    if not os.path.exists(face_id_image_path):
        return False

    safetensors_lora_path = os.path.join(models_path, "Lora", f"{user_id}.safetensors")
    ckpt_lora_path = os.path.join(models_path, "Lora", f"{user_id}.ckpt")
    if not (os.path.exists(safetensors_lora_path) or os.path.exists(ckpt_lora_path)):
        return False
    return True


def urldownload_progressbar(url, file_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()


controlnet_extensions_path = os.path.join(data_path, "extensions", "sd-webui-controlnet")
controlnet_extensions_builtin_path = os.path.join(data_path, "extensions-builtin", "sd-webui-controlnet")
models_annotator_path = os.path.join(data_path, "models")

if os.path.exists(controlnet_extensions_path):
    controlnet_annotator_cache_path = os.path.join(controlnet_extensions_path, "annotator/downloads/midas")
    controlnet_annotator_cache_path_ipa = os.path.join(controlnet_extensions_path, "annotator/downloads/clip_vision")
elif os.path.exists(controlnet_extensions_builtin_path):
    controlnet_annotator_cache_path = os.path.join(controlnet_extensions_builtin_path, "annotator/downloads/midas")
    controlnet_annotator_cache_path_ipa = os.path.join(controlnet_extensions_path, "annotator/downloads/clip_vision")
else:
    controlnet_annotator_cache_path = os.path.join(models_annotator_path, "annotator/downloads/midas")
    controlnet_annotator_cache_path_ipa = os.path.join(controlnet_extensions_path, "annotator/downloads/clip_vision")

# The models are from civitai/6424 & civitai/118913, we saved them to oss for your convenience in downloading the models.
download_urls = {
    "base": [
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/models/Chilloutmix-Ni-pruned-fp16-fix.safetensors",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/models/control_v11p_sd15_canny.pth",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/models/dpt_hybrid-midas-501f0c75.pt",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/models/control_v11f1p_sd15_depth.pth",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/control_sd15_random_color.pth",
    ],
    "seg": [
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/models/sam_vit_l_0b3195.pth",
    ],
    "anydoor": [
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/models/anydoor/epoch%3D1-step%3D8687.ckpt",
        "https://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com/aigc-data/easyphoto/models/anydoor/dinov2_vitg14_pretrain.pth"
    ]
}

save_filenames = {
    "base": [
        os.path.join(models_path, f"Stable-diffusion/Chilloutmix-Ni-pruned-fp16-fix.safetensors"),
        os.path.join(models_path, f"ControlNet/control_v11p_sd15_canny.pth"),
        os.path.join(controlnet_annotator_cache_path, f"dpt_hybrid-midas-501f0c75.pt"),
        os.path.join(models_path, f"ControlNet/control_v11f1p_sd15_depth.pth"),
        os.path.join(models_path, f"ControlNet/control_sd15_random_color.pth"),    
    ],
    "seg": [
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "sam_vit_l_0b3195.pth"),
    ],
    "anydoor": [
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "epoch=1-step=8687.ckpt"),
        os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "dinov2_vitg14_pretrain.pth"),
    ]   
}

def check_files_exists_and_download(check_hash, download_mode="base"):
    urls, filenames = download_urls[download_mode], save_filenames[download_mode]
    # This print will introduce some misundertand
    # print("Start Downloading weights")
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
    r = requests.head(url)
    total_size = int(r.headers["Content-Length"])

    res = requests.get(url, stream=True)
    remote_head_hash = hashlib.sha256(res.raw.read(1000)).hexdigest()
    res.close()

    end_pos = total_size - 1000
    headers = {"Range": f"bytes={end_pos}-{total_size-1}"}
    res = requests.get(url, headers=headers, stream=True)
    remote_end_hash = hashlib.sha256(res.content).hexdigest()
    res.close()

    with open(file_path, "rb") as f:
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


def check_image_mask_input(image, mask, type):
    if image is None:
        info = f"Please upload a {type} image."
        return False, info

    if mask is None and image["mask"].max() == 0:
        info = f"Please give a hint of {type}, or upload {type} mask."
        return False, info

    if mask is not None and mask.shape != image["image"].shape:
        info = f"Please upload a mask with the same size as {type}. Or remove the uploaded mask and generate automatically by given hints"
        return False, info
    
    return True, ''
    