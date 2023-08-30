
import os
import threading
from shutil import copyfile

from modules import shared
from PIL import Image
from scripts.paiya_config import user_id_outpath_samples, validation_prompt
from scripts.preprocess import preprocess_images
from modules.paths import models_path


def paiya_train_forward(
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
    weights_save_path       = os.path.join(user_id_outpath_samples, user_id, "user_weights")
    webui_save_path         = os.path.join(models_path, f"Lora/{user_id}.safetensors")
    webui_load_path         = os.path.join(models_path, f"Stable-diffusion/Chilloutmix-Ni-pruned-fp16-fix.safetensors")

    os.makedirs(original_backup_path, exist_ok=True)
    os.makedirs(user_path, exist_ok=True)
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(webui_save_path)), exist_ok=True)

    max_train_steps         = min(len(instance_images) * 200, 800)

    for index, user_image in enumerate(instance_images):
        image = Image.open(user_image['name']).convert("RGB")
        image.save(os.path.join(original_backup_path, str(index) + ".jpg"))
        
    sub_threading = threading.Thread(target=preprocess_images, args=(images_save_path, json_save_path, validation_prompt, original_backup_path, ref_image_path,))
    sub_threading.start()
    sub_threading.join()

    train_kohya_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_kohya/train_lora.py")
    os.system(
        f'''
        accelerate launch --mixed_precision="fp16" --main_process_port=3456 {train_kohya_path} \
        --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
        --pretrained_model_ckpt="{webui_load_path}" \
        --train_data_dir="{user_path}" --caption_column="text" \
        --resolution=512 --random_flip --train_batch_size=1 --gradient_accumulation_steps=4 --dataloader_num_workers=16 \
        --max_train_steps={max_train_steps} --checkpointing_steps=100 \
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
        --template_dir="extensions-builtin/paiya-sd-webui/models/templates" \
        --template_mask \
        --merge_best_lora_based_face_id \
        --merge_best_lora_name="{user_id}"
        '''
    )
    
    best_weight_path = os.path.join(weights_save_path, f"best_outputs/{user_id}.safetensors")
    copyfile(best_weight_path, webui_save_path)
    return "训练已经完成。"