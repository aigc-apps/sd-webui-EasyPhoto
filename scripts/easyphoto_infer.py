import copy
import logging
import os
import cv2
import numpy as np
import torch
import sys
import subprocess
import platform
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modules import script_callbacks, shared
from modules.images import save_image
from modules.shared import opts, state
from PIL import Image
from scipy.optimize import minimize
from scripts.dragdiffusion_utils import run_drag
from glob import glob
from shutil import copyfile
from typing import Any, List, Union
from scripts.easyphoto_config import (
    DEFAULT_NEGATIVE,
    DEFAULT_POSITIVE,
    easyphoto_img2img_samples,
    easyphoto_txt2img_samples,
    models_path,
    user_id_outpath_samples,
    validation_prompt,
    easyphoto_outpath_samples,
    zero123_model_path,
    cache_log_file_path,
    anydoor_weight_path,
)

from scripts.easyphoto_process_utils import (
    align_and_overlay_images,
    mask_to_polygon,
    mask_to_box,
    seg_by_box,
    apply_mask_to_image,
    merge_with_inner_canny,
    crop_image,
    resize_image_with_pad,
    copy_white_mask_to_template,
    get_background_color,
    find_best_angle_ratio,
    resize_and_stretch,
    compute_rotation_angle,
    expand_box_by_pad,
    expand_roi,
    find_connected_components,
    prepare_train_data_with_single_input,
    resize_to_512
)

from scripts.easyphoto_utils import check_files_exists_and_download, check_id_valid, ep_logger, check_image_mask_input
from scripts.sdwebui import ControlNetUnit, i2i_inpaint_call, t2i_call
from scripts.train_kohya.utils.gpu_info import gpu_monitor_decorator

from segment_anything import SamPredictor, sam_model_registry
from shapely.geometry import Polygon
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

try:
    from scripts.thirdparty.zero123.infer import zero123_infer, load_model_from_config
    from lovely_numpy import lo
    from omegaconf import OmegaConf
    from scripts.thirdparty.zero123.ldm_zero123.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import AutoFeatureExtractor
    from torchvision import transforms
    from scripts.thirdparty.zero123.ldm_zero123.models.diffusion.ddim import DDIMSampler
    from einops import rearrange
except:
    print("Please install Zero123 following the instruction of https://github.com/cvlab-columbia/zero123")

try:
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'thirdparty/anydoor'))
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'thirdparty/anydoor/dinov2'))
    from scripts.thirdparty.anydoor.infer import  inference_single_image, load_anydoor_model
    any_door_env = True
except:
    print("Anydoor Load Failed.")
    any_door_env = False


python_executable_path = sys.executable

def resize_image(input_image, resolution, nearest=False, crop264=True):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    if crop264:
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
    else:
        H = int(H)
        W = int(W)
    if not nearest:
        img = cv2.resize(
            input_image,
            (W, H),
            interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
        )
    else:
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_NEAREST)
    return img


# Add control_mode=1 means Prompt is more important, to better control lips and eyes,
# this comments will be delete after 10 PR and for those who are not familiar with SDWebUIControlNetAPI
def get_controlnet_unit(
    unit: str, input_image: Union[Any, List[Any]], weight: float, is_batch: bool = False, control_mode: int = 1
):  # Any should be replaced with a more specific image type  # Default to False, assuming single image input by default
    if unit == "canny":
        control_unit = dict(
            input_image=None,
            module="canny",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            threshold_a=100,
            threshold_b=200,
            model="control_v11p_sd15_canny",
        )

    elif unit == "sdxl_canny_mid":
        control_unit = dict(
            input_image={"image": np.asarray(input_image), "mask": None},
            module="canny",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            processor_res=1024,
            resize_mode="Just Resize",
            threshold_a=100,
            threshold_b=200,
            model="diffusers_xl_canny_mid",
        )

    elif unit == "openpose":
        control_unit = dict(
            image=None,
            module="openpose_full",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            model="control_v11p_sd15_openpose",
        )

    elif unit == "dwpose":
        control_unit = dict(
            image=None,
            module="dw_openpose_full",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            model="control_v11p_sd15_openpose",
        )

    elif unit == "sdxl_openpose_lora":
        control_unit = dict(
            input_image={"image": np.asarray(input_image), "mask": None},
            module="openpose_full",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            processor_res=1024,
            resize_mode="Just Resize",
            model="thibaud_xl_openpose_256lora",
        )

    elif unit == "color":
        control_unit = dict(
            input_image=None,
            module="none",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            model="control_sd15_random_color",
        )

        blur_ratio = 1
        if is_batch:
            new_input_image = []
            for _input_image in input_image:
                h, w, c = np.shape(_input_image)
                color_image = np.array(_input_image, np.uint8)

                color_image = resize_image(color_image, 1024)
                now_h, now_w = color_image.shape[:2]

                color_image = cv2.resize(color_image, (int(now_w // blur_ratio), int(now_h // blur_ratio)), interpolation=cv2.INTER_CUBIC)
                color_image = cv2.resize(color_image, (now_w, now_h), interpolation=cv2.INTER_NEAREST)
                color_image = cv2.resize(color_image, (w, h), interpolation=cv2.INTER_CUBIC)
                color_image = Image.fromarray(np.uint8(color_image))
                new_input_image.append(color_image)

            control_unit["batch_images"] = [np.array(_input_image, np.uint8) for _input_image in new_input_image]
        else:
            h, w, c = np.shape(input_image)
            color_image = np.array(input_image, np.uint8)

            color_image = resize_image(color_image, 1024)
            now_h, now_w = color_image.shape[:2]

            color_image = cv2.resize(color_image, (int(now_w // blur_ratio), int(now_h // blur_ratio)), interpolation=cv2.INTER_CUBIC)
            color_image = cv2.resize(color_image, (now_w, now_h), interpolation=cv2.INTER_NEAREST)
            color_image = cv2.resize(color_image, (w, h), interpolation=cv2.INTER_CUBIC)
            color_image = Image.fromarray(np.uint8(color_image))

            control_unit["input_image"] = {"image": np.asarray(color_image), "mask": None}

    elif unit == "tile":
        control_unit = dict(
            input_image=None,
            module="tile_resample",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            threshold_a=1,
            threshold_b=200,
            model="control_v11f1e_sd15_tile",
        )

    elif unit == "ipa_full_face":
        control_unit = dict(
            input_image=None,
            module="ip-adapter_clip_sd15",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            model="ip-adapter-full-face_sd15",
        )
    elif unit == "ipa_sdxl_plus_face":
        control_unit = dict(
            input_image={"image": np.asarray(input_image), "mask": None},
            module="ip-adapter_clip_sdxl_plus_vith",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            model="ip-adapter-plus-face_sdxl_vit-h",
        )
    elif unit == "depth":
        control_unit = dict(
            input_image=input_image,
            module="depth_midas",
            weight=weight,
            guidance_end=1,
            control_mode=control_mode,
            resize_mode="Just Resize",
            model="control_v11f1p_sd15_depth",
        )

    elif unit == "ipa":
        control_unit = dict(
            input_image=input_image,
            module="ip-adapter_clip_sd15",
            weight=weight,
            guidance_end=1,
            control_mode=control_mode,
            resize_mode="Just Resize",
            model="ip-adapter_sd15",
        )

    elif unit == "canny_no_pre":
        control_unit = dict(
            input_image=input_image,
            module=None,
            weight=weight,
            guidance_end=1,
            control_mode=control_mode,
            resize_mode="Crop and Resize",
            threshold_a=100,
            threshold_b=200,
            model="control_v11p_sd15_canny",
        )

    if unit != "color" and not unit.startswith("sdxl"):
        if is_batch:
            control_unit["batch_images"] = [np.array(_input_image, np.uint8) for _input_image in input_image]
        else:
            control_unit["input_image"] = {
                "image": np.asarray(input_image),
                "mask": None,
            }

    return control_unit


def inpaint(
    input_image: Image.Image,
    select_mask_input: Image.Image,
    controlnet_pairs: list,
    input_prompt="1girl",
    diffusion_steps=50,
    denoising_strength=0.45,
    hr_scale: float = 1.0,
    default_positive_prompt=DEFAULT_POSITIVE,
    default_negative_prompt=DEFAULT_NEGATIVE,
    seed: int = 123456,
    sd_model_checkpoint="Chilloutmix-Ni-pruned-fp16-fix.safetensors",
    sampler="DPM++ 2M SDE Karras",
):
    assert input_image is not None, f"input_image must not be none"
    controlnet_units_list = []
    w = int(input_image.width)
    h = int(input_image.height)

    # if controlnet_pairs:
    for pair in controlnet_pairs:
        controlnet_units_list.append(get_controlnet_unit(pair[0], pair[1], pair[2], pair[3]))

    positive = f"{input_prompt}, {default_positive_prompt}"
    negative = f"{default_negative_prompt}"

    image = i2i_inpaint_call(
        images=[input_image],
        mask_image=select_mask_input,
        inpainting_fill=1,
        steps=diffusion_steps,
        denoising_strength=denoising_strength,
        cfg_scale=7,
        inpainting_mask_invert=0,
        width=int(w * hr_scale),
        height=int(h * hr_scale),
        inpaint_full_res=False,
        seed=seed,
        prompt=positive,
        negative_prompt=negative,
        controlnet_units=controlnet_units_list,
        sd_model_checkpoint=sd_model_checkpoint,
        outpath_samples=easyphoto_img2img_samples,
        sampler=sampler,
    )

    return image


check_hash = {}
sam_predictor = None
model_anyid = None
models_zero123 = None
ddim_sampler = None
universal_matting = None

# this decorate is default to be closed, not every needs this, more for developers
# @gpu_monitor_decorator()
def easyphoto_infer_forward(
    sd_model_checkpoint,
    infer_way,
    template_image,
    template_mask,
    reference_image,
    reference_mask,
    additional_prompt,
    seed,
    first_diffusion_steps,
    first_denoising_strength,
    lora_weight,
    iou_threshold,
    angle,
    azimuth,
    ratio,
    batch_size,
    refine_input_mask,
    optimize_angle_and_ratio,
    refine_bound,
    pure_image,
    global_inpaint,
    match_and_paste,
    remove_target,
    user_id,
    enhance_with_lora
):
    # global
    global check_hash, models_zero123, model_anyid, ddim_sampler, universal_matting

    dx = 0
    dy = 0

    # check & download weights
    check_files_exists_and_download(check_hash.get("base", True), "base")
    check_files_exists_and_download(check_hash.get("seg", True), "seg")
    check_hash["base"] = False
    check_hash["seg"] = False

    # get random seed
    if int(seed) == -1:
        seed = np.random.randint(0, 65536)

    # Template Input
    input_valid_flag, return_info = check_image_mask_input(template_image, template_mask, 'template')
    if not input_valid_flag:
        ep_logger.error(return_info)
        return return_info, [], template_mask, reference_mask
    
    # Reference Input
    input_valid_flag, return_info = check_image_mask_input(reference_image, reference_mask, 'reference')
    if not input_valid_flag:
        ep_logger.error(return_info)
        return return_info, [], template_mask, reference_mask
    
    # check userid
    return_msg=''

    if (infer_way == 'Infer with Anydoor' and enhance_with_lora) or infer_way == 'Infer with LoRA':
        if user_id == "" or user_id is None or user_id =='none':
            info = "The user id cannot be empty or none."
            ep_logger.error(info)
            return info, [], template_mask, reference_mask
        
        ep_logger.info(f"[Infer with LoRA] User id: {user_id}")
        webui_save_path = os.path.join(models_path, f"Lora/{user_id}.safetensors")

        # train user lora if model not exist
        if os.path.exists(webui_save_path):
            return_msg += f"Use exists LoRA of {user_id}.\n"
            ep_logger.info(f"LoRA of user id: {user_id} exists. Start Infer.")
        else:
            ep_logger.info("No exists LoRA model found. Start Training")
            # ref image copy
            ref_image_path = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")

            # Training data retention
            user_path = os.path.join(user_id_outpath_samples, user_id, "processed_images")
            images_save_path = os.path.join(user_id_outpath_samples, user_id, "processed_images", "train")
            json_save_path = os.path.join(user_id_outpath_samples, user_id, "processed_images", "metadata.jsonl")

            weights_save_path = os.path.join(user_id_outpath_samples, user_id, "user_weights")
            webui_load_path = os.path.join(models_path, f"Stable-diffusion", sd_model_checkpoint)
            sd15_save_path = os.path.join(
                os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"),
                "stable-diffusion-v1-5",
            )

            os.makedirs(user_path, exist_ok=True)
            os.makedirs(images_save_path, exist_ok=True)
            os.makedirs(os.path.dirname(os.path.abspath(webui_save_path)), exist_ok=True)

            if reference_mask is None:
                _, reference_mask = easyphoto_mask_forward(reference_image, "Reference")

            prepare_train_data_with_single_input(reference_image, reference_mask, ref_image_path, images_save_path, json_save_path, validation_prompt)

            # start train
            # check preprocess results
            train_images = glob(os.path.join(images_save_path, "*.jpg"))
            if len(train_images) == 0:
                return (
                    "Failed to obtain preprocessed images, please check the preprocessing process",
                    [],
                    template_mask,
                    reference_mask,
                )
            if not os.path.exists(json_save_path):
                return (
                    "Failed to obtain preprocessed metadata.jsonl, please check the preprocessing process.",
                    [],
                    template_mask,
                    reference_mask,
                )

            train_kohya_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_kohya/train_lora.py")
            ep_logger.info(f"train_file_path : {train_kohya_path}")

            # extensions/sd-webui-EasyPhoto/train_kohya_log.txt, use to cache log and flush to UI
            ep_logger.info(f"cache_log_file_path: {cache_log_file_path}")
            if not os.path.exists(os.path.dirname(cache_log_file_path)):
                os.makedirs(os.path.dirname(cache_log_file_path), exist_ok=True)

            # default
            resolution = 512
            train_batch_size = 1
            gradient_accumulation_steps = 4
            val_and_checkpointing_steps = 200
            learning_rate = 0.0001
            rank = 128
            network_alpha = 64
            dataloader_num_workers = 16
            max_train_steps = 200

            ep_logger.info(f"Delete sam model before training to save CUDA memory.")
            sam_predictor = None
            torch.cuda.empty_cache()

            if platform.system() == "Windows":
                pwd = os.getcwd()
                dataloader_num_workers = 0  # for solve multi process bug

                command = [
                    f"{python_executable_path}",
                    "-m",
                    "accelerate.commands.launch",
                    "--mixed_precision=fp16",
                    "--main_process_port=3456",
                    f"{train_kohya_path}",
                    f"--pretrained_model_name_or_path={os.path.relpath(sd15_save_path, pwd)}",
                    f"--pretrained_model_ckpt={os.path.relpath(webui_load_path, pwd)}",
                    f"--train_data_dir={os.path.relpath(user_path, pwd)}",
                    "--caption_column=text",
                    f"--resolution={resolution}",
                    f"--train_batch_size={train_batch_size}",
                    f"--gradient_accumulation_steps={gradient_accumulation_steps}",
                    f"--dataloader_num_workers={dataloader_num_workers}",
                    f"--max_train_steps={max_train_steps}",
                    f"--checkpointing_steps={val_and_checkpointing_steps}",
                    f"--learning_rate={learning_rate}",
                    "--lr_scheduler=constant",
                    "--lr_warmup_steps=0",
                    "--train_text_encoder",
                    "--seed=42",
                    f"--rank={rank}",
                    f"--network_alpha={network_alpha}",
                    f"--output_dir={weights_save_path}",
                    f"--logging_dir={weights_save_path}",
                    "--enable_xformers_memory_efficient_attention",
                    "--mixed_precision=fp16",
                    f"--cache_log_file={cache_log_file_path}",
                ]
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    ep_logger.error(f"Error executing the command: {e}")

            else:
                command = [
                    f"{python_executable_path}",
                    "-m",
                    "accelerate.commands.launch",
                    "--mixed_precision=fp16",
                    "--main_process_port=3456",
                    f"{train_kohya_path}",
                    f"--pretrained_model_name_or_path={sd15_save_path}",
                    f"--pretrained_model_ckpt={webui_load_path}",
                    f"--train_data_dir={user_path}",
                    "--caption_column=text",
                    f"--resolution={resolution}",
                    f"--train_batch_size={train_batch_size}",
                    f"--gradient_accumulation_steps={gradient_accumulation_steps}",
                    f"--dataloader_num_workers={dataloader_num_workers}",
                    f"--max_train_steps={max_train_steps}",
                    f"--checkpointing_steps={val_and_checkpointing_steps}",
                    f"--learning_rate={learning_rate}",
                    "--lr_scheduler=constant",
                    "--lr_warmup_steps=0",
                    "--train_text_encoder",
                    "--seed=42",
                    f"--rank={rank}",
                    f"--network_alpha={network_alpha}",
                    f"--output_dir={weights_save_path}",
                    f"--logging_dir={weights_save_path}",
                    "--enable_xformers_memory_efficient_attention",
                    "--mixed_precision=fp16",
                    f"--cache_log_file={cache_log_file_path}",
                ]
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    ep_logger.error(f"Error executing the command: {e}")

            best_weight_path = os.path.join(weights_save_path, f"pytorch_lora_weights.safetensors")
            if not os.path.exists(best_weight_path):
                return ("Failed to obtain Lora after training, please check the training process.", [], template_mask, reference_mask)

            # save to models/LoRA
            copyfile(best_weight_path, webui_save_path)


    # anydoor infer
    return_res = []

    if infer_way == 'Infer with Anydoor':
        if not any_door_env:
            return 'Failed to load Anydoor. You can only choose Infer with LoRA', [], template_mask, reference_mask


        check_files_exists_and_download(check_hash.get("anydoor", True), "anydoor")
        check_hash["anydoor"] = False

        # use anydoor to generate result
        # reference image
        img_ref = np.uint8(Image.fromarray(np.uint8(reference_image["image"])))
        mask_ref_input = np.uint8(Image.fromarray(np.uint8(reference_image["mask"])))
        if reference_mask is None:
            # refine
            _, mask_ref = easyphoto_mask_forward(reference_image, True, "Reference")
        else:
            mask_ref = reference_mask[:, :, 0]

        # template image
        img_template = np.uint8(Image.fromarray(np.uint8(template_image["image"])))
        mask_template_input = np.uint8(Image.fromarray(np.uint8(template_image["mask"])))
        if template_mask is None:
            # refine
            _, mask_template = easyphoto_mask_forward(template_image, True, "Template")
            # update for return
            template_mask = mask_template
        else:
            mask_template = template_mask[:, :, 0]

        sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'thirdparty/anydoor'))
        sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),'thirdparty/anydoor/dinov2'))
        from scripts.thirdparty.anydoor.infer import  inference_single_image, load_anydoor_model

        if model_anyid is None:
            model_anyid, ddim_sampler = load_anydoor_model(anydoor_weight_path)

        mask_ref_1 = mask_ref/255    
        synthesis = inference_single_image(model_anyid, ddim_sampler, img_ref.copy(), mask_ref_1.copy(), img_template.copy(), mask_template.copy())

        return_res.append(Image.fromarray(np.uint8(synthesis)))

        # masked
        update_mask = False
        if update_mask:
            if universal_matting is None:
                universal_matting = pipeline(Tasks.universal_matting, model='damo/cv_unet_universal-matting')
            result = universal_matting(synthesis)[OutputKeys.OUTPUT_IMG]
            mask_template = result[:,:,3]
        
        masked_synthesis = apply_mask_to_image(synthesis, img_template, mask_template)
    
        return_res.append(Image.fromarray(np.uint8(masked_synthesis)))
        match_and_paste = False

        if not enhance_with_lora:
            return "Success\n", return_res, template_mask, reference_mask

    else:
        # paste
        img_ref = np.uint8(Image.open(os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")))
        mask_ref = np.uint8(Image.open(os.path.join(user_id_outpath_samples, user_id, "ref_image_mask.jpg")))

        if len(mask_ref.shape) == 2:
            mask_ref = np.repeat(mask_ref[:, :, np.newaxis], 3, axis=2)

        # template image
        img_template = np.uint8(Image.fromarray(np.uint8(template_image["image"])))
        mask_template_input = np.uint8(Image.fromarray(np.uint8(template_image["mask"])))

        if template_mask is None:
            # refine
            _, mask_template = easyphoto_mask_forward(template_image, True, "Template")
            # update for return
            template_mask = mask_template
        else:
            mask_template = template_mask[:, :, 0]


    # infer with pretrained lora
    input_prompt = f"{validation_prompt}, <lora:{user_id}:{lora_weight}>" + additional_prompt
    ep_logger.info(f"input_prompt: {input_prompt}")

    # for final paste
    _, box_main = mask_to_box(np.uint8(mask_ref))
    _, box_template = mask_to_box(mask_template)
    template_copy = copy.deepcopy(img_template)

    # crop to get local img
    W, H = np.array(img_template).shape[1], np.array(img_template).shape[0]
    expand_ratio = 1.2
    img_ref = crop_image(np.array(img_ref), box_main, expand_ratio=expand_ratio)
    mask_ref = crop_image(np.array(mask_ref), box_main, expand_ratio=expand_ratio)
    img_template = crop_image(np.array(img_template), box_template, expand_ratio=expand_ratio)
    mask_template = crop_image(np.array(mask_template), box_template, expand_ratio=expand_ratio)

    if infer_way == 'Infer with Anydoor':
        masked_synthesis = crop_image(np.array(masked_synthesis), box_template, expand_ratio=expand_ratio)

    box_template = expand_roi(box_template, ratio=expand_ratio, max_box=[0, 0, W, H])

    if match_and_paste:
        # Step2: prepare background image for paste
        # main background with most frequent color
        color = get_background_color(img_ref, mask_ref[:, :, 0])
        color_img = np.full((img_template.shape[0], img_template.shape[1], 3), color, dtype=np.uint8)
        background_img = apply_mask_to_image(color_img, img_template, mask_template)

        if azimuth != 0:
            # zero123
            try:
                # no need to always load zero123 model
                x, y, z = 0, azimuth, 0
                result = zero123_infer(
                    models_zero123, x, y, z, Image.fromarray(np.uint8(img_ref))
                )  # TODO [PIL.Image] May choose the best merge one by lightglue

                # result[0].save('res_zero123.jpg')
                success = 1
            except Exception as e:
                print(e)
                success = 0

            print("success:", success)
            if not success:
                try:
                    from scripts.thirdparty.zero123.infer import zero123_infer, load_model_from_config
                    from lovely_numpy import lo
                    from omegaconf import OmegaConf
                    from scripts.thirdparty.zero123.ldm_zero123.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
                    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
                    from transformers import AutoFeatureExtractor
                    from torchvision import transforms
                    from scripts.thirdparty.zero123.ldm_zero123.models.diffusion.ddim import DDIMSampler
                    from contextlib import nullcontext
                    from einops import rearrange
                    import math

                    # load model
                    print(f"Loading zero123 model from {zero123_model_path}")
                    ckpt = os.path.join(zero123_model_path, "105000.ckpt")
                    config = os.path.join(zero123_model_path, "configs/sd-objaverse-finetune-c_concat-256.yaml")

                    config = OmegaConf.load(config)
                    models_zero123 = dict()
                    print("Instantiating LatentDiffusion...")
                    device = "cuda"
                    models_zero123["turncam"] = load_model_from_config(config, ckpt, device=device)
                    models_zero123["nsfw"] = StableDiffusionSafetyChecker.from_pretrained(zero123_model_path).to(device)
                    print("Instantiating AutoFeatureExtractor...")
                    models_zero123["clip_fe"] = AutoFeatureExtractor.from_pretrained(zero123_model_path)
                    print("Instantiating Carvekit HiInterface...")
                    models_zero123["carvekit"] = create_carvekit_interface()

                    x, y, z = 0, azimuth, 0
                    result, has_nsfw_concept = zero123_infer(
                        models_zero123, x, y, z, Image.fromarray(np.uint8(img_ref))
                    )  # TODO [PIL.Image] May choose the best merge one by lightglue
                    if not has_nsfw_concept:
                        # crop and get mask
                        img_ref_3d = np.array(result[0])
                        mask_ref = salient_detect(img_ref_3d)[OutputKeys.MASKS]
                        mask_ref, box_gen = mask_to_box(mask_ref)

                        mask_ref = cv2.erode(np.array(mask_ref), np.ones((10, 10), np.uint8), iterations=1)

                        # crop image again
                        img_ref = crop_image(img_ref_3d, box_gen, expand_ratio=expand_ratio)
                        mask_ref = crop_image(mask_ref, box_gen, expand_ratio=expand_ratio)

                        cv2.imwrite("img_ref_3d_crop.jpg", img_ref)
                        cv2.imwrite("mask_ref_3d_crop.jpg", mask_ref)
                    else:
                        print("To 3d failed. NSFW Image occured!")

                    # result[0].save('res_zero123.jpg')
                except:
                    raise ImportError("Please install Zero123 following the instruction of https://github.com/cvlab-columbia/zero123")
    
        if optimize_angle_and_ratio:
            ep_logger.info("Start optimize angle and ratio!")
            # find optimzal angle and ratio
            # resize mask_ref to same size as mask_template (init ratio as 1)
            resized_mask_ref = resize_and_stretch(mask_ref, target_size=(mask_template.shape[1], mask_template.shape[0]))
            resized_mask_ref = resized_mask_ref[:, :, 0]

            # get polygon
            polygon1 = mask_to_polygon(resized_mask_ref)
            polygon2 = mask_to_polygon(mask_template)

            # target angle: 2 to 0
            rotation_angle2 = compute_rotation_angle(polygon2)
            # target angle: 1 to 0
            rotation_angle1 = compute_rotation_angle(polygon1)

            # wrong result of big angle (suppose the angle is small)
            if rotation_angle2 > 20:
                rotation_angle2 = 0
            if rotation_angle1 > 20:
                rotation_angle1 = 0
            # polygon angle is reverse to img angle
            angle_target = rotation_angle2 - rotation_angle1

            ep_logger.info(f"target rotation: 1 to 0: {rotation_angle1}, 2 to 0: {rotation_angle2}, final_rotate: {angle_target}")

            # center
            x, y = mask_template.shape[1] // 2, mask_template.shape[0] // 2

            initial_parameters = np.array([angle, ratio])
            max_iters = 100

            angle, ratio, = find_best_angle_ratio(
                polygon1,
                polygon2,
                initial_parameters,
                x,
                y,
                angle_target,
                max_iters,
                iou_threshold,
            )

        ep_logger.info(f"Set angle:{angle}, ratio: {ratio}, azimuth: {azimuth}")

        # paste
        result_img, rotate_img_ref, mask_ref, mask_template, iou = align_and_overlay_images(
            np.array(img_ref), np.array(background_img), np.array(mask_ref), np.array(mask_template), angle=-angle, ratio=ratio, dx=dx, dy=dy
        )

        return_msg += "Paste with angle {:.2f}, ratio: {:.2f}, dx: {:.2f}, dy: {:.2f}, Match IoU: {:.2f}, optimize: {}. \n See paste result above, if you are not satisfatory with the optimized result, close the optimize_angle_and_ratio and manually set a angle and ratio.\n".format(
            angle, ratio, dx, dy, iou, optimize_angle_and_ratio
        )

        # Step4: prepare for control image
        h_expand, w_expand = result_img.shape[:2]
        h2, w2 = np.array(img_template).shape[:2]
        crop_img_template_box_first = [
            (w_expand - w2) // 2,
            (h_expand - h2) // 2,
            (w_expand + w2) // 2,
            (h_expand + h2) // 2,
        ]

        first_paste = crop_image(result_img, crop_img_template_box_first)
        first_paste = Image.fromarray(np.uint8(first_paste))

        result_img = crop_image(result_img, crop_img_template_box_first)
        mask_ref = crop_image(mask_ref, crop_img_template_box_first)
        mask_template = crop_image(mask_template, crop_img_template_box_first)
        resize_img_ref = crop_image(rotate_img_ref, crop_img_template_box_first)

        result_img = Image.fromarray(np.uint8(result_img))

        # get inner canny and resize img to 512
        resize_image, res_canny, inner_bound_mask = merge_with_inner_canny(np.array(result_img).astype(np.uint8), mask_ref, mask_template)
        inner_bound_mask = Image.fromarray(np.uint8(inner_bound_mask))

        resize_mask_template, remove_pad = resize_image_with_pad(mask_template, resolution=512)
        resize_mask_template = remove_pad(resize_mask_template)

        resize_img_template, remove_pad = resize_image_with_pad(img_template, resolution=512)
        resize_img_template = remove_pad(resize_img_template)

        resize_mask_ref, remove_pad = resize_image_with_pad(mask_ref, resolution=512)
        resize_mask_ref = remove_pad(resize_mask_ref)

        resize_img_ref, remove_pad = resize_image_with_pad(resize_img_ref, resolution=512)
        resize_img_ref = remove_pad(resize_img_ref)

        mask_template = Image.fromarray(np.uint8(resize_mask_template))
        resize_image = Image.fromarray(resize_image)
        resize_image_input = copy.deepcopy(resize_image)
        mask_template_input = copy.deepcopy(mask_template)

    else:
        # resize
        resize_img_ref = resize_to_512(img_ref)
        resize_img_template = resize_to_512(img_template)
        resize_mask_ref = resize_to_512(mask_ref)
        resize_mask_template = resize_to_512(mask_template)

        mask_template = Image.fromarray(np.uint8(resize_mask_template))
        mask_template_input = copy.deepcopy(mask_template)

        if infer_way == 'Infer with Anydoor':
            resized_masked_synthesis = resize_to_512(masked_synthesis)
            print(resized_masked_synthesis.shape)
            resize_image_input = Image.fromarray(np.uint8(resized_masked_synthesis))
        else:
            remove_target = True # remove target from template img
            resize_image_input = Image.fromarray(np.uint8(resize_img_template))


    # Step5: generation
    for i in range(batch_size):
        ep_logger.info("Start First diffusion.")

        if match_and_paste:
            controlnet_pairs = [
                ["canny_no_pre", res_canny, 1.0, 0],
                ["depth", resize_img_template, 1.0, 0],
                ["color", resize_image_input, 0.5, 0],
            ]
        else:
            controlnet_pairs = [
                ["depth", resize_img_template, 1.0, 0],
            ]
            # controlnet_pairs = []

        result_img = inpaint(
            resize_image_input,
            mask_template_input,
            controlnet_pairs,
            diffusion_steps=first_diffusion_steps,
            denoising_strength=first_denoising_strength,
            input_prompt=input_prompt,
            default_positive_prompt="",
            default_negative_prompt="",
            hr_scale=1.0,
            seed=str(seed),
            sd_model_checkpoint=sd_model_checkpoint,
        )

        if infer_way != 'Infer with Anydoor':
            # start inner bound refine
            controlnet_pairs = [
                ["canny", resize_img_template, 1.0, 0],
                ["depth", resize_img_template, 1.0, 0],
            ]
            refine_diffusion_steps = 30
            refine_denoising_strength = 0.7

            result_img = inpaint(
                result_img,
                inner_bound_mask,
                controlnet_pairs,
                diffusion_steps=refine_diffusion_steps,
                denoising_strength=refine_denoising_strength,
                input_prompt=input_prompt,
                default_positive_prompt="",
                default_negative_prompt="",
                hr_scale=1.0,
                seed=str(seed),
                sd_model_checkpoint=sd_model_checkpoint,
            )

        # resize diffusion results
        target_width = box_template[2] - box_template[0]
        target_height = box_template[3] - box_template[1]
        result_img = result_img.resize((target_width, target_height))
        resize_mask_template = mask_template.resize((target_width, target_height))

        # copy back
        template_copy = np.array(template_copy, np.uint8)

        if len(np.array(np.uint8(resize_mask_template)).shape) == 2:
            init_generation = copy_white_mask_to_template(
                np.array(result_img),
                np.array(np.uint8(resize_mask_template)),
                template_copy,
                box_template,
            )
        else:
            init_generation = copy_white_mask_to_template(
                np.array(result_img),
                np.array(np.uint8(resize_mask_template))[:, :, 0],
                template_copy,
                box_template,
            )

        return_res.append(Image.fromarray(np.uint8(init_generation)))

        if refine_bound:
            ep_logger.info("Start Refine Boundary.")
            # refine bound
            padding = 30

            box_pad = expand_box_by_pad(
                box_template,
                max_size=(template_copy.shape[1], template_copy.shape[0]),
                padding_size=padding,
            )
            padding_size = abs(np.array(box_pad) - np.array(box_template))

            input_img = init_generation[box_pad[1] : box_pad[3], box_pad[0] : box_pad[2]]
            input_control_img = template_copy[box_pad[1] : box_pad[3], box_pad[0] : box_pad[2]]

            if len(np.array(np.uint8(resize_mask_template)).shape) == 2:
                mask_array = np.array(np.uint8(resize_mask_template))
            else:
                mask_array = np.array(np.uint8(resize_mask_template))[:, :, 0]

            input_mask = np.pad(
                mask_array,
                (
                    (padding_size[1], padding_size[3]),
                    (padding_size[0], padding_size[2]),
                ),
                mode="constant",
                constant_values=0,
            )

            input_mask_copy = copy.deepcopy(input_mask)

            input_mask = np.uint8(
                cv2.dilate(np.array(input_mask), np.ones((10, 10), np.uint8), iterations=1)
                - cv2.erode(np.array(input_mask), np.ones((10, 10), np.uint8), iterations=1)
            )

            # generate
            controlnet_pairs = [["canny", input_control_img, 1.0, 0]]

            input_mask = Image.fromarray(np.uint8(input_mask))
            input_img = Image.fromarray(input_img)

            refine_diffusion_steps = 20
            refine_denoising_strength = 0.5

            result_img = inpaint(
                input_img,
                input_mask,
                controlnet_pairs,
                diffusion_steps=refine_diffusion_steps,
                denoising_strength=refine_denoising_strength,
                input_prompt=input_prompt,
                hr_scale=1.0,
                seed=str(seed),
                sd_model_checkpoint=sd_model_checkpoint,
            )

            # resize diffusion results
            target_width = box_pad[2] - box_pad[0]
            target_height = box_pad[3] - box_pad[1]
            result_img = result_img.resize((target_width, target_height))
            input_mask = input_mask.resize((target_width, target_height))

            final_generation = copy_white_mask_to_template(
                np.array(result_img),
                np.array(np.uint8(input_mask_copy)),
                template_copy,
                box_pad,
            )

            return_res.append(Image.fromarray(np.uint8(final_generation)))
        else:
            final_generation = init_generation

        save_image(
            Image.fromarray(np.uint8(final_generation)),
            easyphoto_outpath_samples,
            "EasyPhoto",
            None,
            None,
            opts.grid_format,
            info=None,
            short_filename=not opts.grid_extended_filename,
            grid=True,
            p=None,
        )

        if match_and_paste:
            # show paste result for debug
            return_res.append(first_paste)

        torch.cuda.empty_cache()

        ep_logger.info("Finished")

    return "Success\n" + return_msg, return_res, template_mask, reference_mask


def easyphoto_mask_forward(input_image, refine_mask, img_type):
    global check_hash, sam_predictor

    check_files_exists_and_download(check_hash.get("seg", True), "seg")
    check_hash["seg"] = False

    if input_image is None:
        info = f"Please upload a {img_type} image."
        ep_logger.error(info)
        return info, None

    img = np.uint8(Image.fromarray(np.uint8(input_image["image"])))
    mask = np.uint8(Image.fromarray(np.uint8(input_image["mask"])))

    if mask.max() == 0:
        # no input mask
        info = f"({img_type}) No input hint given. Upload a mask image or give some hints."
        ep_logger.info(info)
        return info, None

    if not refine_mask:
        return f"[{img_type}] Use input mask.", mask

    num_connected, centroids = find_connected_components(mask[:, :, 0])
    ep_logger.info(f"({img_type}) Find input mask connected num: {num_connected}.")

    # model init
    if sam_predictor is None:
        sam_checkpoint = os.path.join(
            os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"),
            "sam_vit_l_0b3195.pth",
        )

        sam = sam_model_registry["vit_l"]()
        sam.load_state_dict(torch.load(sam_checkpoint))
        sam_predictor = SamPredictor(sam.cuda())

    if num_connected < 2:
        ep_logger.info(f"{(img_type)} Refine input mask of by mask.")
        # support the input is a mask, we use box and sam to refine mask
        _, box_template = mask_to_box(mask[:, :, 0])

        mask = np.uint8(seg_by_box(np.array(img), box_template, sam_predictor))
    else:
        ep_logger.info(f"{(img_type)} Refine input mask of by points.")
        # support points is given, points are used to refine mask
        centroids = np.array(centroids)
        input_label = np.array([1] * centroids.shape[0])
        sam_predictor.set_image(np.array(img))
        masks, _, _ = sam_predictor.predict(
            point_coords=centroids,
            point_labels=input_label,
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8)  # mask = [sam_outputs_num, h, w]
        mask = mask * 255

    return f"[{img_type}] Show Mask Success", mask
