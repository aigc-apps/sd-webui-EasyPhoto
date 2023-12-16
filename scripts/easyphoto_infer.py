import copy
import glob
import math
import os
import traceback
from typing import Any, List, Union

import cv2
import numpy as np
import torch
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modules import shared
from modules.images import save_image
from modules.shared import opts
from PIL import Image, ImageChops, ImageOps

from scripts.easyphoto_config import (
    DEFAULT_NEGATIVE,
    DEFAULT_NEGATIVE_AD,
    DEFAULT_NEGATIVE_T2I,
    DEFAULT_POSITIVE,
    DEFAULT_POSITIVE_AD,
    DEFAULT_POSITIVE_T2I,
    easyphoto_img2img_samples,
    easyphoto_models_path,
    easyphoto_outpath_samples,
    easyphoto_txt2img_samples,
    easyphoto_video_outpath_samples,
    models_path,
    user_id_outpath_samples,
    validation_prompt,
)
from scripts.easyphoto_utils import (
    Face_Skin,
    FIRE_forward,
    PSGAN_Inference,
    alignment_photo,
    call_face_crop,
    call_face_crop_templates,
    check_files_exists_and_download,
    check_id_valid,
    color_transfer,
    convert_to_video,
    crop_and_paste,
    ep_logger,
    get_controlnet_version,
    get_mov_all_images,
    modelscope_models_to_cpu,
    modelscope_models_to_gpu,
    switch_ms_model_cpu,
    unload_models,
    seed_everything,
)
from scripts.sdwebui import (
    get_checkpoint_type,
    get_lora_type,
    get_scene_prompt,
    i2i_inpaint_call,
    refresh_model_vae,
    reload_sd_model_vae,
    switch_sd_model_vae,
    t2i_call,
)


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
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
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

    elif unit == "sdxl_openpose":
        control_unit = dict(
            input_image={"image": np.asarray(input_image), "mask": None},
            module="openpose_full",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            processor_res=1024,
            resize_mode="Just Resize",
            model="thibaud_xl_openpose",
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


@switch_ms_model_cpu()
def txt2img(
    controlnet_pairs: list,
    input_prompt="1girl",
    diffusion_steps=50,
    cfg_scale=7,
    width: int = 1024,
    height: int = 1024,
    default_positive_prompt=DEFAULT_POSITIVE,
    default_negative_prompt=DEFAULT_NEGATIVE,
    seed: int = 123456,
    sd_model_checkpoint="Chilloutmix-Ni-pruned-fp16-fix.safetensors",
    sampler="DPM++ 2M SDE Karras",
    outpath_samples=easyphoto_txt2img_samples,
    do_not_save_samples=False,
    animatediff_flag=False,
    animatediff_video_length=0,
    animatediff_fps=0,
    loractl_flag=False,
):
    controlnet_units_list = []

    for pair in controlnet_pairs:
        if len(pair) == 4:
            controlnet_units_list.append(
                get_controlnet_unit(pair[0], pair[1], pair[2], False if type(pair[1]) is not list else True, pair[3])
            )
        else:
            controlnet_units_list.append(
                get_controlnet_unit(
                    pair[0],
                    pair[1],
                    pair[2],
                    False if type(1) is not list else True,
                )
            )

    positive = f"{input_prompt}, {default_positive_prompt}"
    negative = f"{default_negative_prompt}"

    image = t2i_call(
        steps=diffusion_steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        seed=seed,
        subseed=seed,
        prompt=positive,
        negative_prompt=negative,
        controlnet_units=controlnet_units_list,
        outpath_samples=outpath_samples,
        do_not_save_samples=do_not_save_samples,
        sampler=sampler,
        animatediff_flag=animatediff_flag,
        animatediff_video_length=animatediff_video_length,
        animatediff_fps=animatediff_fps,
        loractl_flag=False,
    )

    return image


@switch_ms_model_cpu()
def inpaint(
    input_image: Image.Image,
    select_mask_input: Image.Image,
    controlnet_pairs: list,
    input_prompt="1girl",
    diffusion_steps=50,
    denoising_strength=0.45,
    cfg_scale=7,
    hr_scale: float = 1.0,
    default_positive_prompt=DEFAULT_POSITIVE,
    default_negative_prompt=DEFAULT_NEGATIVE,
    seed: int = 123456,
    sd_model_checkpoint="Chilloutmix-Ni-pruned-fp16-fix.safetensors",
    sampler="DPM++ 2M SDE Karras",
    outpath_samples=easyphoto_img2img_samples,
    do_not_save_samples=False,
    animatediff_flag=False,
    animatediff_video_length=0,
    animatediff_fps=0,
    animatediff_reserve_scale=1,
    animatediff_last_image=None,
    loractl_flag=False,
):
    assert input_image is not None, f"input_image must not be none"
    controlnet_units_list = []
    w = int(input_image.width) if type(input_image) is not list else int(input_image[0].width)
    h = int(input_image.height) if type(input_image) is not list else int(input_image[0].height)

    for pair in controlnet_pairs:
        if len(pair) == 4:
            # if control_mode is additional given (default 1 prompt is better)
            controlnet_units_list.append(
                get_controlnet_unit(
                    pair[0],
                    pair[1],
                    pair[2],
                    False if type(input_image) is not list else True,
                    pair[3],
                )
            )
        else:
            controlnet_units_list.append(
                get_controlnet_unit(
                    pair[0],
                    pair[1],
                    pair[2],
                    False if type(input_image) is not list else True,
                )
            )

    positive = f"{input_prompt}, {default_positive_prompt}"
    negative = f"{default_negative_prompt}"

    image = i2i_inpaint_call(
        images=[input_image] if type(input_image) is not list else input_image,
        mask_image=select_mask_input,
        inpainting_fill=1,
        steps=diffusion_steps,
        denoising_strength=denoising_strength,
        cfg_scale=cfg_scale,
        inpainting_mask_invert=0,
        width=int(w * hr_scale),
        height=int(h * hr_scale),
        inpaint_full_res=False,
        seed=seed,
        subseed=seed,
        prompt=positive,
        negative_prompt=negative,
        controlnet_units=controlnet_units_list,
        outpath_samples=outpath_samples,
        do_not_save_samples=do_not_save_samples,
        sampler=sampler,
        animatediff_flag=animatediff_flag,
        animatediff_video_length=animatediff_video_length,
        animatediff_fps=animatediff_fps,
        animatediff_reserve_scale=animatediff_reserve_scale,
        animatediff_last_image=animatediff_last_image,
        loractl_flag=loractl_flag,
    )

    return image


retinaface_detection = None
image_face_fusion = None
skin_retouching = None
portrait_enhancement = None
old_super_resolution_method = None
face_skin = None
face_recognition = None
psgan_inference = None
check_hash = {}
sdxl_txt2img_flag = False


# this decorate is default to be closed, not every needs this, more for developers
# @gpu_monitor_decorator()
@switch_sd_model_vae()
def easyphoto_infer_forward(
    sd_model_checkpoint,
    selected_template_images,
    init_image,
    uploaded_template_images,
    text_to_image_input_prompt,
    text_to_image_width,
    text_to_image_height,
    scene_id,
    prompt_generate_sd_model_checkpoint,
    additional_prompt,
    before_face_fusion_ratio,
    after_face_fusion_ratio,
    first_diffusion_steps,
    first_denoising_strength,
    second_diffusion_steps,
    second_denoising_strength,
    seed,
    crop_face_preprocess,
    apply_face_fusion_before,
    apply_face_fusion_after,
    color_shift_middle,
    color_shift_last,
    super_resolution,
    super_resolution_method,
    skin_retouching_bool,
    display_score,
    background_restore,
    background_restore_denoising_strength,
    makeup_transfer,
    makeup_transfer_ratio,
    face_shape_match,
    tabs,
    ipa_control,
    ipa_weight,
    ipa_image_path,
    ref_mode_choose,
    ipa_only_weight,
    ipa_only_image_path,
    lcm_accelerate,
    *user_ids,
):
    # global
    global retinaface_detection, image_face_fusion, skin_retouching, portrait_enhancement, old_super_resolution_method, face_skin, face_recognition, psgan_inference, check_hash, sdxl_txt2img_flag

    # check & download weights of basemodel/controlnet+annotator/VAE/face_skin/buffalo/validation_template
    check_files_exists_and_download(check_hash.get("base", True), download_mode="base")
    check_files_exists_and_download(check_hash.get("portrait", True), download_mode="portrait")
    if check_hash.get("base", True) or check_hash.get("portrait", True):
        refresh_model_vae()
    check_hash["base"] = False
    check_hash["portrait"] = False

    # check the checkpoint_type of sd_model_checkpoint
    checkpoint_type = get_checkpoint_type(sd_model_checkpoint)
    if checkpoint_type == 2:
        return "EasyPhoto does not support the SD2 checkpoint.", [], []
    sdxl_pipeline_flag = True if checkpoint_type == 3 else False

    # infer with IPA only
    if ref_mode_choose == "Infer with IPA only(without Pretraining Lora)":
        ipa_control = True
        ipa_weight = ipa_only_weight
        ipa_image_path = ipa_only_image_path
        user_ids = ["ipa_control_only", "none", "none", "none", "none"]

    # check & download weights of others models
    if sdxl_pipeline_flag or tabs == 3:
        check_files_exists_and_download(check_hash.get("sdxl", True), download_mode="sdxl")
        if check_hash.get("sdxl", True):
            refresh_model_vae()
        check_hash["sdxl"] = False
    if tabs == 3:
        check_files_exists_and_download(check_hash.get("add_text2image", True), download_mode="add_text2image")
        if check_hash.get("add_text2image", True):
            refresh_model_vae()
        check_hash["add_text2image"] = False
    if ipa_control:
        if not sdxl_pipeline_flag:
            check_files_exists_and_download(check_hash.get("add_ipa_base", True), download_mode="add_ipa_base")
            if check_hash.get("add_ipa_base", True):
                refresh_model_vae()
            check_hash["add_ipa_base"] = False
        else:
            check_files_exists_and_download(check_hash.get("add_ipa_sdxl", True), download_mode="add_ipa_sdxl")
            if check_hash.get("add_ipa_sdxl", True):
                refresh_model_vae()
            check_hash["add_ipa_sdxl"] = False
    if lcm_accelerate:
        check_files_exists_and_download(check_hash.get("lcm", True), download_mode="lcm")
        check_hash["lcm"] = False

    # Check if the user_id is valid and if the type of the stable diffusion model and the user LoRA match
    for user_id in user_ids:
        if user_id != "none" and user_id != "ipa_control_only":
            # Check if the user_id is valid
            if not check_id_valid(user_id, user_id_outpath_samples, models_path):
                return "User id is not exist", [], []
            # Check if the type of the stable diffusion model and the user LoRA match
            sdxl_lora_type = get_lora_type(os.path.join(models_path, f"Lora/{user_id}.safetensors"))
            sdxl_lora_flag = True if sdxl_lora_type == 3 else False
            if sdxl_lora_flag != sdxl_pipeline_flag:
                checkpoint_type_name = "SDXL" if sdxl_pipeline_flag else "SD1"
                lora_type_name = "SDXL" if sdxl_lora_flag else "SD1"
                error_info = "The type of the stable diffusion model {} ({}) and the user id {} ({}) does not " "match ".format(
                    sd_model_checkpoint, checkpoint_type_name, user_id, lora_type_name
                )
                return error_info, [], []

    loractl_flag = False
    if "sliders" in additional_prompt:
        if ("sdxl_sliders" in additional_prompt and not sdxl_pipeline_flag) or ("sd1_sliders" in additional_prompt and sdxl_pipeline_flag):
            error_info = "The type of the stable diffusion model {} and attribute edit sliders ({}) does not match.".format(
                sd_model_checkpoint, additional_prompt
            )
            ep_logger.error(error_info)
            return error_info, [], []
        # download all sliders here.
        check_files_exists_and_download(check_hash.get("sliders", True), download_mode="sliders")
        check_hash["sliders"] = False
        loractl_flag = True

    # update donot delete but use "none" as placeholder and will pass this face inpaint later
    passed_userid_list = []
    last_user_id_none_num = 0
    for idx, user_id in enumerate(user_ids):
        if user_id == "none":
            last_user_id_none_num += 1
            passed_userid_list.append(idx)
        else:
            last_user_id_none_num = 0

    if len(user_ids) == last_user_id_none_num:
        return "Please choose a user id.", [], []

    # check the version of controlnets
    controlnet_version = get_controlnet_version()
    major, minor, patch = map(int, controlnet_version.split("."))
    if major == 0 and minor == 0 and patch == 0:
        return "Please install sd-webui-controlnet from https://github.com/Mikubill/sd-webui-controlnet.", [], []
    if ipa_control:
        if major < 1 or minor < 1 or patch < 417:
            return "To use IP-Adapter Control, please upgrade sd-webui-controlnet to the latest version.", [], []

    # check the number of controlnets
    max_control_net_unit_count = 3 if not ipa_control else 4
    control_net_unit_count = shared.opts.data.get("control_net_unit_count", 3)
    ep_logger.info("ControlNet unit number: {}".format(control_net_unit_count))
    if control_net_unit_count < max_control_net_unit_count:
        error_info = (
            "Please go to Settings/ControlNet and at least set {} for "
            "Multi-ControlNet: ControlNet unit number (requires restart).".format(max_control_net_unit_count)
        )
        return error_info, [], []

    if ipa_control:
        ipa_image_paths = ["none"] * 5  # consistent with user_ids
        ipa_flag = False
        valid_user_id_num, valid_ipa_image_path_num = 0, 0
        for index, user_id in enumerate(user_ids):
            if not ipa_flag and user_id != "none" and ipa_image_path is not None:
                ipa_image_paths[index] = ipa_image_path
                ipa_flag = True
                valid_ipa_image_path_num += 1
            if user_id != "none":
                valid_user_id_num += 1

        if valid_user_id_num > 1:
            ep_logger.error("EasyPhoto does not support IP-Adapter Control with multiple user ids currently.")
            return "EasyPhoto does not support IP-Adapter Control with multiple user ids currently.", [], []
        if ipa_control and valid_user_id_num != valid_ipa_image_path_num:
            ep_logger.warning(
                "Found {} user id(s), but only {} image prompt(s) for IP-Adapter Control. Use the reference image "
                "corresponding to the user instead.".format(valid_user_id_num, valid_ipa_image_path_num)
            )
        if not display_score:
            display_score = True
            ep_logger.warning("Display score is forced to be true when IP-Adapter Control is enabled.")

    if lcm_accelerate:
        lcm_lora_name_and_weight = "lcm_lora_sdxl:0.40" if sdxl_pipeline_flag else "lcm_lora_sd15:0.80"

    try:
        # choose tabs select
        if tabs == 0:
            template_images = eval(selected_template_images)
        elif tabs == 1:
            template_images = [init_image]
        elif tabs == 2:
            template_images = [file_d["name"] for file_d in uploaded_template_images]
        elif tabs == 3:
            # load sd and vae
            prompt_generate_sd_model_checkpoint_type = get_checkpoint_type(prompt_generate_sd_model_checkpoint)
            if prompt_generate_sd_model_checkpoint_type == 3:
                prompt_generate_vae = "madebyollin-sdxl-vae-fp16-fix.safetensors"
            else:
                prompt_generate_vae = "vae-ft-mse-840000-ema-pruned.ckpt"

            if prompt_generate_sd_model_checkpoint_type == 3 and scene_id != "none":
                return "EasyPhoto does not support infer scene lora with the SDXL checkpoint.", [], []
            ep_logger.info("Template images will be generated when you use text2photo")
    except Exception:
        torch.cuda.empty_cache()
        traceback.print_exc()
        return "Please choose or upload a template.", [], []

    # create modelscope model
    if retinaface_detection is None:
        retinaface_detection = pipeline(Tasks.face_detection, "damo/cv_resnet50_face-detection_retinaface", model_revision="v2.0.2")
    if image_face_fusion is None:
        image_face_fusion = pipeline(Tasks.image_face_fusion, model="damo/cv_unet-image-face-fusion_damo", model_revision="v1.3")
    if face_skin is None:
        face_skin = Face_Skin(os.path.join(easyphoto_models_path, "face_skin.pth"))
    if skin_retouching is None:
        try:
            skin_retouching = pipeline("skin-retouching-torch", model="damo/cv_unet_skin_retouching_torch", model_revision="v1.0.2")
        except Exception as e:
            torch.cuda.empty_cache()
            traceback.print_exc()
            ep_logger.error(f"Skin Retouching model load error. Error Info: {e}")
    if portrait_enhancement is None or old_super_resolution_method != super_resolution_method:
        try:
            if super_resolution_method == "gpen":
                portrait_enhancement = pipeline(
                    Tasks.image_portrait_enhancement, model="damo/cv_gpen_image-portrait-enhancement", model_revision="v1.0.0"
                )
            elif super_resolution_method == "realesrgan":
                portrait_enhancement = pipeline(
                    "image-super-resolution-x2", model="bubbliiiing/cv_rrdb_image-super-resolution_x2", model_revision="v1.0.2"
                )
            old_super_resolution_method = super_resolution_method
        except Exception as e:
            torch.cuda.empty_cache()
            traceback.print_exc()
            ep_logger.error(f"Portrait Enhancement model load error. Error Info: {e}")

    # To save the GPU memory, create the face recognition model for computing FaceID if the user intend to show it.
    if display_score and face_recognition is None:
        face_recognition = pipeline("face_recognition", model="bubbliiiing/cv_retinafce_recognition", model_revision="v1.0.3")

    # psgan for transfer makeup
    if makeup_transfer and psgan_inference is None:
        try:
            makeup_transfer_model_path = os.path.join(easyphoto_models_path, "makeup_transfer.pth")
            face_landmarks_model_path = os.path.join(easyphoto_models_path, "face_landmarks.pth")
            psgan_inference = PSGAN_Inference(
                "cuda", makeup_transfer_model_path, retinaface_detection, face_skin, face_landmarks_model_path
            )
        except Exception as e:
            torch.cuda.empty_cache()
            traceback.print_exc()
            ep_logger.error(f"MakeUp Transfer model load error. Error Info: {e}")

    # This is to increase the fault tolerance of the code.
    # If the code exits abnormally, it may cause the model to not function properly on the CPU
    modelscope_models_to_gpu()

    # get random seed
    if int(seed) == -1:
        seed = np.random.randint(0, 65536)

    seed_everything(int(seed))

    # params init
    input_prompts = []
    face_id_images = []
    roop_images = []
    face_id_retinaface_boxes = []
    face_id_retinaface_keypoints = []
    face_id_retinaface_masks = []
    input_prompt_without_lora = additional_prompt
    best_lora_weights = str(0.9)
    multi_user_facecrop_ratio = 1.5
    multi_user_safecrop_ratio = 1.0
    # Second diffusion hr scale
    default_hr_scale = 1.0
    need_mouth_fix = True
    input_mask_face_part_only = True

    if ipa_control:
        ipa_images = []
        ipa_retinaface_boxes = []
        ipa_retinaface_keypoints = []
        ipa_retinaface_masks = []
        ipa_face_part_only = False

    if lcm_accelerate:
        input_prompt_without_lora += f"<lora:{lcm_lora_name_and_weight}>, "

    ep_logger.info("Start templates and user_ids preprocess.")

    if tabs == 3:
        reload_sd_model_vae(prompt_generate_sd_model_checkpoint, prompt_generate_vae)

        if scene_id != "none":
            # scene lora path
            scene_lora_model_path = os.path.join(models_path, "Lora", f"{scene_id}.safetensors")
            if not os.path.exists(scene_lora_model_path):
                return "Please check scene lora is exist or not.", [], []
            is_scene_lora, scene_lora_prompt = get_scene_prompt(scene_lora_model_path)
            if not is_scene_lora:
                return "Please use the lora trained by ep.", [], []

            # get lora scene prompt
            if user_ids[0] != "ipa_control_only":
                last_scene_lora_prompt_high_weight = (
                    text_to_image_input_prompt
                    + f", <lora:{scene_id}:0.80>, look at viewer, "
                    + f"{validation_prompt}, <lora:{user_ids[0]}:0.25>, "
                )
                last_scene_lora_prompt_low_weight = (
                    text_to_image_input_prompt
                    + f", <lora:{scene_id}:0.40>, look at viewer, "
                    + f"{validation_prompt}, <lora:{user_ids[0]}:0.25>, "
                )
            else:
                last_scene_lora_prompt_high_weight = text_to_image_input_prompt + f", <lora:{scene_id}:0.80>, look at viewer, "
                last_scene_lora_prompt_low_weight = text_to_image_input_prompt + f", <lora:{scene_id}:0.40>, look at viewer, "

            if lcm_accelerate:
                last_scene_lora_prompt_high_weight += f"<lora:{lcm_lora_name_and_weight}>, "
                last_scene_lora_prompt_low_weight += f"<lora:{lcm_lora_name_and_weight}>, "

            # text to image with scene lora
            ep_logger.info(f"Text to Image with prompt: {last_scene_lora_prompt_high_weight} and lora: {scene_lora_model_path}")
            pose_templates = glob.glob(os.path.join(easyphoto_models_path, "pose_templates/*.jpg")) + glob.glob(
                os.path.join(easyphoto_models_path, "pose_templates/*.png")
            )

            pose_template = Image.open(np.random.choice(pose_templates))
            template_images = txt2img(
                [["openpose", pose_template, 0.50, 1]],
                input_prompt=last_scene_lora_prompt_high_weight,
                diffusion_steps=30 if not lcm_accelerate else 8,
                cfg_scale=7 if not lcm_accelerate else 2,
                width=text_to_image_width,
                height=text_to_image_height,
                default_positive_prompt=DEFAULT_POSITIVE_T2I,
                default_negative_prompt=DEFAULT_NEGATIVE_T2I,
                seed=seed,
                sampler="Euler a",
            )
            ep_logger.info(f"Hire Fix with prompt: {last_scene_lora_prompt_low_weight} and lora: {scene_lora_model_path}")
            template_images = inpaint(
                template_images,
                None,
                [],
                input_prompt=last_scene_lora_prompt_low_weight,
                diffusion_steps=30 if not lcm_accelerate else 8,
                cfg_scale=7 if not lcm_accelerate else 2,
                denoising_strength=0.20,
                hr_scale=1.5,
                default_positive_prompt=DEFAULT_POSITIVE_T2I,
                default_negative_prompt=DEFAULT_NEGATIVE_T2I,
                seed=seed,
                sampler="Euler a",
            )
            template_images = [np.uint8(template_images)]
        else:
            # text to image for template
            if lcm_accelerate:
                text_to_image_input_prompt += f"<lora:{lcm_lora_name_and_weight}>, "
            ep_logger.info(f"Text to Image with prompt: {text_to_image_input_prompt}")
            template_images = txt2img(
                [],
                input_prompt=text_to_image_input_prompt,
                diffusion_steps=30 if not lcm_accelerate else 8,
                cfg_scale=7 if not lcm_accelerate else 2,
                width=text_to_image_width,
                height=text_to_image_height,
                default_positive_prompt=DEFAULT_POSITIVE_T2I,
                default_negative_prompt=DEFAULT_NEGATIVE_T2I,
                seed=seed,
                sampler="DPM++ 2M SDE Karras" if not lcm_accelerate else "Euler a",
            )
            template_images = [np.uint8(template_images)]

    if not sdxl_pipeline_flag:
        reload_sd_model_vae(sd_model_checkpoint, "vae-ft-mse-840000-ema-pruned.ckpt")
    else:
        reload_sd_model_vae(sd_model_checkpoint, "madebyollin-sdxl-vae-fp16-fix.safetensors")

    # SD web UI will raise the `Error: A tensor with all NaNs was produced in Unet.`
    # when users do img2img with SDXL currently (v1.6.0). Users should launch SD web UI with `--no-half`
    # or do txt2img with SDXL once before img2img.
    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/6923#issuecomment-1713104376.
    if sdxl_pipeline_flag and not sdxl_txt2img_flag:
        txt2img([], diffusion_steps=3, do_not_save_samples=True)
        sdxl_txt2img_flag = True
    for index, user_id in enumerate(user_ids):
        if user_id == "none":
            # use some placeholder
            input_prompts.append("none")
            face_id_images.append("none")
            roop_images.append("none")
            face_id_retinaface_boxes.append([])
            face_id_retinaface_keypoints.append([])
            face_id_retinaface_masks.append([])
            if ipa_control:
                ipa_images.append("none")
                ipa_retinaface_boxes.append([])
                ipa_retinaface_keypoints.append([])
                ipa_retinaface_masks.append([])
        elif user_id == "ipa_control_only":
            # get prompt
            input_prompt = f"1person, face, portrait, " + "<lora:FilmVelvia3:0.65>, " + additional_prompt
            if lcm_accelerate:
                input_prompt += f"<lora:{lcm_lora_name_and_weight}>, "
            ipa_image = Image.open(ipa_image_paths[index])
            ipa_image = ImageOps.exif_transpose(ipa_image).convert("RGB")

            roop_image = ipa_image

            _ipa_retinaface_boxes, _ipa_retinaface_keypoints, _ipa_retinaface_masks = call_face_crop(
                retinaface_detection, ipa_image, 1.05, "crop"
            )
            if len(_ipa_retinaface_boxes) == 0:
                ep_logger.error("No face is detected in the uploaded image prompt.")
                return "Please upload a image prompt with face.", [], []
            if len(_ipa_retinaface_boxes) > 1:
                ep_logger.warning(
                    "{} faces are detected in the uploaded image prompt. "
                    "Only the left one will be used.".format(len(_ipa_retinaface_boxes))
                )

            input_prompts.append(input_prompt)
            face_id_images.append("none")
            roop_images.append(roop_image)
            face_id_retinaface_boxes.append([])
            face_id_retinaface_keypoints.append([])
            face_id_retinaface_masks.append([])
            if ipa_control:
                ipa_images.append(ipa_image)
                ipa_retinaface_boxes.append(_ipa_retinaface_boxes[0])
                ipa_retinaface_keypoints.append(_ipa_retinaface_keypoints[0])
                ipa_retinaface_masks.append(_ipa_retinaface_masks[0])
        else:
            # get prompt
            input_prompt = f"{validation_prompt}, <lora:{user_id}:{best_lora_weights}>, " + "<lora:FilmVelvia3:0.65>, " + additional_prompt
            # Add the ddpo LoRA into the input prompt if available.
            lora_model_path = os.path.join(models_path, "Lora")
            if os.path.exists(os.path.join(lora_model_path, "ddpo_{}.safetensors".format(user_id))):
                input_prompt += "<lora:ddpo_{}>, ".format(user_id)

            # TODO: face_id_image_path may have to be picked with pitch yaw angle in video mode.
            if sdxl_pipeline_flag:
                input_prompt = f"{validation_prompt}, <lora:{user_id}>, " + additional_prompt

            if lcm_accelerate:
                input_prompt += f"<lora:{lcm_lora_name_and_weight}>, "
            # get best image after training
            best_outputs_paths = glob.glob(os.path.join(user_id_outpath_samples, user_id, "user_weights", "best_outputs", "*.jpg"))
            # get roop image
            if len(best_outputs_paths) > 0:
                face_id_image_path = best_outputs_paths[0]
            else:
                face_id_image_path = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")
            roop_image_path = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")

            face_id_image = Image.open(face_id_image_path).convert("RGB")
            roop_image = Image.open(roop_image_path).convert("RGB")

            if ipa_control:
                if ipa_image_paths[index] != "none":
                    ipa_image = Image.open(ipa_image_paths[index])
                    ipa_image = ImageOps.exif_transpose(ipa_image).convert("RGB")
                else:
                    ipa_image = copy.deepcopy(roop_image)

                _ipa_retinaface_boxes, _ipa_retinaface_keypoints, _ipa_retinaface_masks = call_face_crop(
                    retinaface_detection, ipa_image, 1, "crop"
                )
                if len(_ipa_retinaface_boxes) == 0:
                    ep_logger.error("No face is detected in the uploaded image prompt.")
                    return "Please upload a image prompt with face.", [], []
                if len(_ipa_retinaface_boxes) > 1:
                    ep_logger.warning(
                        "{} faces are detected in the uploaded image prompt. "
                        "Only the left one will be used.".format(len(_ipa_retinaface_boxes))
                    )

            # Crop user images to obtain portrait boxes, facial keypoints, and masks
            _face_id_retinaface_boxes, _face_id_retinaface_keypoints, _face_id_retinaface_masks = call_face_crop(
                retinaface_detection, face_id_image, multi_user_facecrop_ratio, "face_id"
            )
            _face_id_retinaface_box = _face_id_retinaface_boxes[0]
            _face_id_retinaface_keypoint = _face_id_retinaface_keypoints[0]
            _face_id_retinaface_mask = _face_id_retinaface_masks[0]

            input_prompts.append(input_prompt)
            face_id_images.append(face_id_image)
            roop_images.append(roop_image)
            face_id_retinaface_boxes.append(_face_id_retinaface_box)
            face_id_retinaface_keypoints.append(_face_id_retinaface_keypoint)
            face_id_retinaface_masks.append(_face_id_retinaface_mask)
            if ipa_control:
                ipa_images.append(ipa_image)
                ipa_retinaface_boxes.append(_ipa_retinaface_boxes[0])
                ipa_retinaface_keypoints.append(_ipa_retinaface_keypoints[0])
                ipa_retinaface_masks.append(_ipa_retinaface_masks[0])

    outputs, face_id_outputs = [], []
    loop_message = ""
    for template_idx, template_image in enumerate(template_images):
        template_idx_info = f"""
            Start Generate template                 : {str(template_idx + 1)};
            user_ids                                : {str(user_ids)};
            sd_model_checkpoint                     : {str(sd_model_checkpoint)};
            input_prompts                           : {str(input_prompts)};
            before_face_fusion_ratio                : {str(before_face_fusion_ratio)};
            after_face_fusion_ratio                 : {str(after_face_fusion_ratio)};
            first_diffusion_steps                   : {str(first_diffusion_steps)};
            first_denoising_strength                : {str(first_denoising_strength)};
            second_diffusion_steps                  : {str(second_diffusion_steps)};
            second_denoising_strength               : {str(second_denoising_strength)};
            seed                                    : {seed}
            crop_face_preprocess                    : {str(crop_face_preprocess)}
            apply_face_fusion_before                : {str(apply_face_fusion_before)}
            apply_face_fusion_after                 : {str(apply_face_fusion_after)}
            color_shift_middle                      : {str(color_shift_middle)}
            color_shift_last                        : {str(color_shift_last)}
            super_resolution                        : {str(super_resolution)}
            super_resolution_method                 : {str(super_resolution_method)}
            display_score                           : {str(display_score)}
            background_restore                      : {str(background_restore)}
            background_restore_denoising_strength   : {str(background_restore_denoising_strength)}
            makeup_transfer                         : {str(makeup_transfer)}
            makeup_transfer_ratio                   : {str(makeup_transfer_ratio)}
            skin_retouching_bool                    : {str(skin_retouching_bool)}
            face_shape_match                        : {str(face_shape_match)}
            ipa_control                             : {str(ipa_control)}
            ipa_weight                              : {str(ipa_weight)}
            ipa_image_path                          : {str(ipa_image_path)}
            ref_mode_choose                         : {str(ref_mode_choose)}
            ipa_only_weight                         : {str(ipa_only_weight)}
            ipa_only_image_path                     : {str(ipa_only_image_path)}
        """
        ep_logger.info(template_idx_info)
        try:
            # open the template image
            if tabs == 0 or tabs == 2:
                template_image = Image.open(template_image).convert("RGB")
            else:
                template_image = Image.fromarray(template_image).convert("RGB")

            template_face_safe_boxes, _, _ = call_face_crop(retinaface_detection, template_image, multi_user_safecrop_ratio, "crop")
            if len(template_face_safe_boxes) == 0:
                return "Please upload a template with face.", [], []
            template_detected_facenum = len(template_face_safe_boxes)

            # use some print/log to record mismatch of detectionface and user_ids
            if template_detected_facenum > len(user_ids) - last_user_id_none_num:
                ep_logger.warning(
                    f"User set {len(user_ids) - last_user_id_none_num} face but detected {template_detected_facenum} face in template image,\
                the last {template_detected_facenum - len(user_ids) - last_user_id_none_num} face will remains"
                )

            if len(user_ids) - last_user_id_none_num > template_detected_facenum:
                ep_logger.warning(
                    f"User set {len(user_ids) - last_user_id_none_num} face but detected {template_detected_facenum} face in template image,\
                the last {len(user_ids) - last_user_id_none_num - template_detected_facenum} set user_ids is useless"
                )

            if background_restore:
                output_image = np.array(copy.deepcopy(template_image))
                output_mask = np.ones_like(output_image) * 255

                for index in range(len(template_face_safe_boxes)):
                    retinaface_box = template_face_safe_boxes[index]
                    output_mask[retinaface_box[1] : retinaface_box[3], retinaface_box[0] : retinaface_box[2]] = 0
                output_mask = Image.fromarray(np.uint8(cv2.dilate(np.array(output_mask), np.ones((32, 32), np.uint8), iterations=1)))
            else:
                if min(template_detected_facenum, len(user_ids) - last_user_id_none_num) > 1:
                    output_image = np.array(copy.deepcopy(template_image))
                    output_mask = np.ones_like(output_image)

                    # get mask in final diffusion for multi people
                    for index in range(len(template_face_safe_boxes)):
                        # pass this userid, not mask the face
                        if index in passed_userid_list:
                            continue
                        else:
                            retinaface_box = template_face_safe_boxes[index]
                            output_mask[retinaface_box[1] : retinaface_box[3], retinaface_box[0] : retinaface_box[2]] = 255
                    output_mask = Image.fromarray(
                        np.uint8(
                            cv2.dilate(np.array(output_mask), np.ones((64, 64), np.uint8), iterations=1)
                            - cv2.erode(np.array(output_mask), np.ones((32, 32), np.uint8), iterations=1)
                        )
                    )

            total_processed_person = 0
            for index in range(min(len(template_face_safe_boxes), len(user_ids) - last_user_id_none_num)):
                # pass this userid, not do anything
                if index in passed_userid_list:
                    continue
                total_processed_person += 1

                loop_template_image = copy.deepcopy(template_image)

                # mask other people face use 255 in this term, to transfer multi user to single user situation
                if min(len(template_face_safe_boxes), len(user_ids) - last_user_id_none_num) > 1:
                    loop_template_image = np.array(loop_template_image)
                    for sub_index in range(len(template_face_safe_boxes)):
                        if index != sub_index:
                            retinaface_box = template_face_safe_boxes[sub_index]
                            loop_template_image[retinaface_box[1] : retinaface_box[3], retinaface_box[0] : retinaface_box[2]] = 255
                    loop_template_image = Image.fromarray(np.uint8(loop_template_image))

                # Crop the template image to retain only the portion of the portrait
                if crop_face_preprocess:
                    loop_template_crop_safe_boxes, _, _ = call_face_crop(retinaface_detection, loop_template_image, 3, "crop")
                    loop_template_crop_safe_box = loop_template_crop_safe_boxes[0]
                    input_image = copy.deepcopy(loop_template_image).crop(loop_template_crop_safe_box)
                else:
                    input_image = copy.deepcopy(loop_template_image)

                if sdxl_pipeline_flag:
                    # Fix total pixels in the generated image in SDXL.
                    target_area = 1024 * 1024
                    ratio = math.sqrt(target_area / (input_image.width * input_image.height))
                    new_size = (int(input_image.width * ratio), int(input_image.height * ratio))
                    ep_logger.info("Start resize image from {} to {}.".format(input_image.size, new_size))
                else:
                    input_short_size = 512.0
                    ep_logger.info("Start Image resize to {}.".format(input_short_size))
                    short_side = min(input_image.width, input_image.height)
                    resize = float(short_side / input_short_size)
                    new_size = (int(input_image.width // resize), int(input_image.height // resize))
                input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)

                if crop_face_preprocess:
                    # In order to ensure that the size of the inpainting output image produced by webui is consistent
                    # with the input image, the height and width of the input image need to be a multiple of 32.
                    new_width = int(np.shape(input_image)[1] // 32 * 32)
                    new_height = int(np.shape(input_image)[0] // 32 * 32)
                    input_image = input_image.resize([new_width, new_height], Image.Resampling.LANCZOS)

                # Detect the box where the face of the template image is located and obtain its corresponding small mask
                ep_logger.info("Start face detect.")
                input_image_retinaface_boxes, input_image_retinaface_keypoints, input_masks = call_face_crop(
                    retinaface_detection, input_image, 1.1, "template"
                )
                input_image_retinaface_box = input_image_retinaface_boxes[0]
                input_image_retinaface_keypoint = input_image_retinaface_keypoints[0]
                input_mask = input_masks[0]

                # backup input template and mask
                copy.deepcopy(input_mask)
                original_input_template = copy.deepcopy(input_image)

                if user_ids[index] == "ipa_control_only":
                    replaced_input_image = None
                else:
                    # Paste user images onto template images
                    replaced_input_image = crop_and_paste(
                        face_id_images[index],
                        face_id_retinaface_masks[index],
                        input_image,
                        face_id_retinaface_keypoints[index],
                        input_image_retinaface_keypoint,
                        face_id_retinaface_boxes[index],
                    )
                    replaced_input_image = Image.fromarray(np.uint8(replaced_input_image))

                # The cropped face area (square) in the reference image will be used in IP-Adapter.
                if ipa_control:
                    ipa_retinaface_box = ipa_retinaface_boxes[index]
                    ipa_retinaface_keypoint = ipa_retinaface_keypoints[index]
                    ipa_retinaface_mask = ipa_retinaface_masks[index]
                    ipa_face_width = ipa_retinaface_box[2] - ipa_retinaface_box[0]

                    if not ipa_face_part_only:
                        ipa_mask, brow_mask = face_skin(
                            ipa_images[index], retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 11, 12, 13], [2, 3]]
                        )
                        ipa_kernel_size = np.ones((int(ipa_face_width // 10), int(ipa_face_width // 10)), np.uint8)
                        # Fill small holes with a close operation (w/o cv2.dilate)
                        ipa_mask = Image.fromarray(np.uint8(cv2.morphologyEx(np.array(ipa_mask), cv2.MORPH_CLOSE, ipa_kernel_size)))
                    else:
                        # Expand the reference image in the x-axis direction to include the ears.
                        h, w, c = np.shape(ipa_retinaface_mask)
                        ipa_mask = np.zeros_like(np.array(ipa_retinaface_mask, np.uint8))
                        ipa_retinaface_box[0] = np.clip(np.array(ipa_retinaface_box[0], np.int32) - ipa_face_width * 0.15, 0, w - 1)
                        ipa_retinaface_box[2] = np.clip(np.array(ipa_retinaface_box[2], np.int32) + ipa_face_width * 0.15, 0, w - 1)
                        ipa_mask[ipa_retinaface_box[1] : ipa_retinaface_box[3], ipa_retinaface_box[0] : ipa_retinaface_box[2]] = 255
                        ipa_mask = Image.fromarray(np.uint8(ipa_mask))
                        brow_mask = None

                    # Since the image encoder of IP-Adapter will crop/resize the image prompt to (224, 224),
                    # we pad the face w.r.t the long side for an aspect ratio of 1.
                    ipa_mask = np.array(ipa_mask, np.uint8) / 255
                    ipa_image_face = np.ones_like(np.array(ipa_images[index])) * 255
                    ipa_image_face = Image.fromarray(np.uint8(np.array(ipa_images[index]) * ipa_mask + ipa_image_face * (1 - ipa_mask)))
                    ipa_image_face = ipa_image_face.crop(ipa_retinaface_box)

                    # Align the ipa face
                    ipa_retinaface_keypoint[:, 0] -= ipa_retinaface_box[0]
                    ipa_retinaface_keypoint[:, 1] -= ipa_retinaface_box[1]
                    ipa_image_face = Image.fromarray(
                        np.uint8(alignment_photo(np.array(ipa_image_face), np.array(ipa_retinaface_keypoint, np.int))[0])
                    )

                    # If brow_mask is not None, remove the skin above brows
                    # Only edit the facial area here, hair is useless
                    if brow_mask is not None:
                        brow_mask = brow_mask.crop(ipa_retinaface_box)
                        brow_mask = Image.fromarray(
                            np.uint8(
                                alignment_photo(np.array(brow_mask), np.array(ipa_retinaface_keypoint, np.int), borderValue=(0, 0, 0))[0]
                            )
                        )
                        y_coords, _, _ = np.where(np.array(brow_mask) > 0)
                        min_y = max(int(np.min(y_coords)), 1)
                        ipa_image_face = np.array(ipa_image_face, np.uint8)
                        ipa_image_face[:min_y, :, :] = 255
                        ipa_image_face = Image.fromarray(ipa_image_face)

                    padded_size = (max(ipa_image_face.size), max(ipa_image_face.size))
                    ipa_image_face = ImageOps.pad(ipa_image_face, padded_size, color=(255, 255, 255))

                # Fusion of user reference images and input images as canny input
                if roop_images[index] is not None and apply_face_fusion_before:
                    fusion_image = image_face_fusion(dict(template=input_image, user=roop_images[index]))[OutputKeys.OUTPUT_IMG]
                    fusion_image = Image.fromarray(cv2.cvtColor(fusion_image, cv2.COLOR_BGR2RGB))

                    # The edge shadows generated by fusion are filtered out by taking intersections of masks of faces before and after fusion.
                    # detect face area
                    fusion_image_mask = np.int32(
                        np.float32(face_skin(fusion_image, retinaface_detection, needs_index=[[1, 2, 3, 10, 11, 12, 13]])[0]) > 128
                    )
                    input_image_mask = np.int32(
                        np.float32(face_skin(input_image, retinaface_detection, needs_index=[[1, 2, 3, 10, 11, 12, 13]])[0]) > 128
                    )
                    combine_mask = cv2.blur(np.uint8(input_image_mask * fusion_image_mask * 255), (8, 8)) / 255

                    # paste back to photo
                    fusion_image = np.array(fusion_image) * combine_mask + np.array(input_image) * (1 - combine_mask)
                    fusion_image = Image.fromarray(np.uint8(fusion_image))

                    input_image = Image.fromarray(
                        np.uint8(
                            (
                                np.array(input_image, np.float32) * (1 - before_face_fusion_ratio)
                                + np.array(fusion_image, np.float32) * before_face_fusion_ratio
                            )
                        )
                    )

                if input_mask_face_part_only:
                    face_width = input_image_retinaface_box[2] - input_image_retinaface_box[0]
                    input_mask = face_skin(input_image, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 11, 12, 13]])[0]

                    kernel_size = np.ones((int(face_width // 10), int(face_width // 10)), np.uint8)
                    # Fill small holes with a close operation
                    input_mask = Image.fromarray(np.uint8(cv2.morphologyEx(np.array(input_mask), cv2.MORPH_CLOSE, kernel_size)))
                    # Use dilate to reconstruct the surrounding area of the face
                    input_mask = Image.fromarray(np.uint8(cv2.dilate(np.array(input_mask), kernel_size, iterations=1)))
                else:
                    # Expand the template image in the x-axis direction to include the ears.
                    h, w, c = np.shape(input_mask)
                    input_mask = np.zeros_like(np.array(input_mask, np.uint8))
                    input_image_retinaface_box = np.int32(input_image_retinaface_box)

                    face_width = input_image_retinaface_box[2] - input_image_retinaface_box[0]
                    input_image_retinaface_box[0] = np.clip(np.array(input_image_retinaface_box[0], np.int32) - face_width * 0.10, 0, w - 1)
                    input_image_retinaface_box[2] = np.clip(np.array(input_image_retinaface_box[2], np.int32) + face_width * 0.10, 0, w - 1)

                    # get new input_mask
                    input_mask[
                        input_image_retinaface_box[1] : input_image_retinaface_box[3],
                        input_image_retinaface_box[0] : input_image_retinaface_box[2],
                    ] = 255
                    input_mask = Image.fromarray(np.uint8(input_mask))

                # here we get the retinaface_box, we should use this Input box and face pixel to refine the output face pixel colors
                template_image_original_face_area = np.array(original_input_template)[
                    input_image_retinaface_box[1] : input_image_retinaface_box[3],
                    input_image_retinaface_box[0] : input_image_retinaface_box[2],
                    :,
                ]

                if user_ids[index] == "ipa_control_only":
                    replaced_input_image = input_image

                # First diffusion, facial reconstruction
                ep_logger.info("Start First diffusion.")
                ImageChops.multiply(input_image, input_mask)
                if not face_shape_match:
                    if not sdxl_pipeline_flag:
                        controlnet_pairs = [
                            ["canny", input_image, 0.50],
                            ["openpose", replaced_input_image, 0.50],
                            ["color", input_image, 0.85],
                        ]
                        if ipa_control:
                            controlnet_pairs.append(["ipa_full_face", ipa_image_face, ipa_weight])
                    else:
                        controlnet_pairs = [["sdxl_canny_mid", input_image, 0.50]]
                        if ipa_control:
                            controlnet_pairs.append(["ipa_sdxl_plus_face", ipa_image_face, ipa_weight])
                    first_diffusion_output_image = inpaint(
                        input_image,
                        input_mask,
                        controlnet_pairs,
                        diffusion_steps=first_diffusion_steps,
                        cfg_scale=7 if not lcm_accelerate else 2,
                        denoising_strength=first_denoising_strength,
                        input_prompt=input_prompts[index],
                        hr_scale=1.0,
                        seed=seed,
                        sampler="DPM++ 2M SDE Karras" if not lcm_accelerate else "Euler a",
                        loractl_flag=loractl_flag,
                    )
                    # We only save the lora weight image in the first diffusion.
                    if loractl_flag:
                        first_diffusion_output_image, lora_weight_image = first_diffusion_output_image
                else:
                    if not sdxl_pipeline_flag:
                        controlnet_pairs = [["canny", input_image, 0.50], ["openpose", replaced_input_image, 0.50]]
                        if ipa_control:
                            controlnet_pairs.append(["ipa_full_face", ipa_image_face, ipa_weight])
                    else:
                        controlnet_pairs = [["sdxl_canny_mid", input_image, 0.50]]
                        if ipa_control:
                            controlnet_pairs.append(["ipa_sdxl_plus_face", ipa_image_face, ipa_weight])
                    first_diffusion_output_image = inpaint(
                        input_image,
                        None,
                        controlnet_pairs,
                        diffusion_steps=first_diffusion_steps,
                        cfg_scale=7 if not lcm_accelerate else 2,
                        denoising_strength=first_denoising_strength,
                        input_prompt=input_prompts[index],
                        hr_scale=1.0,
                        seed=seed,
                        sampler="DPM++ 2M SDE Karras" if not lcm_accelerate else "Euler a",
                    )
                    # We only save the lora weight image in the first diffusion.
                    if loractl_flag:
                        first_diffusion_output_image, lora_weight_image = first_diffusion_output_image

                    # detect face area
                    face_skin_mask = face_skin(
                        first_diffusion_output_image, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]]
                    )[0]
                    kernel_size = np.ones((int(face_width // 10), int(face_width // 10)), np.uint8)

                    # Fill small holes with a close operation
                    face_skin_mask = Image.fromarray(np.uint8(cv2.morphologyEx(np.array(face_skin_mask), cv2.MORPH_CLOSE, kernel_size)))

                    # Use dilate to reconstruct the surrounding area of the face
                    face_skin_mask = Image.fromarray(np.uint8(cv2.dilate(np.array(face_skin_mask), kernel_size, iterations=1)))
                    face_skin_mask = cv2.blur(np.float32(face_skin_mask), (32, 32)) / 255

                    # paste back to photo, Using I2I generation controlled solely by OpenPose, even with a very small denoise amplitude,
                    # still carries the risk of introducing NSFW and global incoherence.!!! important!!!
                    input_image_uint8 = np.array(first_diffusion_output_image) * face_skin_mask + np.array(input_image) * (
                        1 - face_skin_mask
                    )
                    first_diffusion_output_image = Image.fromarray(np.uint8(input_image_uint8))

                if color_shift_middle:
                    # apply color shift
                    ep_logger.info("Start color shift middle.")
                    first_diffusion_output_image_uint8 = np.uint8(np.array(first_diffusion_output_image))
                    # crop image first
                    first_diffusion_output_image_crop = Image.fromarray(
                        first_diffusion_output_image_uint8[
                            input_image_retinaface_box[1] : input_image_retinaface_box[3],
                            input_image_retinaface_box[0] : input_image_retinaface_box[2],
                            :,
                        ]
                    )

                    # apply color shift
                    first_diffusion_output_image_crop_color_shift = np.array(copy.deepcopy(first_diffusion_output_image_crop))
                    first_diffusion_output_image_crop_color_shift = color_transfer(
                        first_diffusion_output_image_crop_color_shift, template_image_original_face_area
                    )

                    # detect face area
                    face_skin_mask = np.float32(
                        face_skin(first_diffusion_output_image_crop, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 11, 12, 13]])[0]
                    )
                    face_skin_mask = cv2.blur(face_skin_mask, (32, 32)) / 255

                    # paste back to photo
                    first_diffusion_output_image_uint8[
                        input_image_retinaface_box[1] : input_image_retinaface_box[3],
                        input_image_retinaface_box[0] : input_image_retinaface_box[2],
                        :,
                    ] = first_diffusion_output_image_crop_color_shift * face_skin_mask + np.array(first_diffusion_output_image_crop) * (
                        1 - face_skin_mask
                    )
                    first_diffusion_output_image = Image.fromarray(np.uint8(first_diffusion_output_image_uint8))

                # Second diffusion
                if roop_images[index] is not None and apply_face_fusion_after:
                    # Fusion of facial photos with user photos
                    ep_logger.info("Start second face fusion.")
                    fusion_image = image_face_fusion(dict(template=first_diffusion_output_image, user=roop_images[index]))[
                        OutputKeys.OUTPUT_IMG
                    ]  # swap_face(target_img=output_image, source_img=roop_image, model="inswapper_128.onnx", upscale_options=UpscaleOptions())
                    fusion_image = Image.fromarray(cv2.cvtColor(fusion_image, cv2.COLOR_BGR2RGB))

                    # The edge shadows generated by fusion are filtered out by taking intersections of masks of faces before and after fusion.
                    # detect face area
                    # fusion_image_mask and input_image_mask are 0, 1 masks of shape [h, w, 3]
                    fusion_image_mask = np.int32(
                        np.float32(face_skin(fusion_image, retinaface_detection, needs_index=[[1, 2, 3, 11, 12, 13]])[0]) > 128
                    )
                    input_image_mask = np.int32(
                        np.float32(face_skin(first_diffusion_output_image, retinaface_detection, needs_index=[[1, 2, 3, 11, 12, 13]])[0])
                        > 128
                    )
                    combine_mask = cv2.blur(np.uint8(input_image_mask * fusion_image_mask * 255), (8, 8)) / 255

                    # paste back to photo
                    fusion_image = np.array(fusion_image) * combine_mask + np.array(first_diffusion_output_image) * (1 - combine_mask)
                    fusion_image = Image.fromarray(np.uint8(fusion_image))

                    input_image = Image.fromarray(
                        np.uint8(
                            (
                                np.array(first_diffusion_output_image, np.float32) * (1 - after_face_fusion_ratio)
                                + np.array(fusion_image, np.float32) * after_face_fusion_ratio
                            )
                        )
                    )
                else:
                    fusion_image = first_diffusion_output_image
                    input_image = first_diffusion_output_image

                # Add mouth_mask to avoid some fault lips, close if you dont need
                if need_mouth_fix:
                    ep_logger.info("Start mouth detect.")
                    mouth_mask, face_mask = face_skin(input_image, retinaface_detection, [[4, 5, 12, 13], [1, 2, 3, 4, 5, 10, 11, 12, 13]])
                    # Obtain the mask of the area around the face
                    face_mask = Image.fromarray(
                        np.uint8(
                            cv2.dilate(np.array(face_mask), np.ones((32, 32), np.uint8), iterations=1)
                            - cv2.erode(np.array(face_mask), np.ones((16, 16), np.uint8), iterations=1)
                        )
                    )

                    i_h, i_w, i_c = np.shape(face_mask)
                    m_h, m_w, m_c = np.shape(mouth_mask)
                    if i_h != m_h or i_w != m_w:
                        face_mask = face_mask.resize([m_w, m_h])
                    input_mask = Image.fromarray(np.uint8(np.clip(np.float32(face_mask) + np.float32(mouth_mask), 0, 255)))

                ep_logger.info("Start Second diffusion.")
                if not sdxl_pipeline_flag:
                    controlnet_pairs = [["canny", fusion_image, 1.00], ["tile", fusion_image, 1.00]]
                    if ipa_control:
                        controlnet_pairs = [["canny", fusion_image, 1.00], ["ipa_full_face", ipa_image_face, ipa_weight]]
                else:
                    controlnet_pairs = [["sdxl_canny_mid", fusion_image, 1.00]]
                    if ipa_control:
                        controlnet_pairs = [["sdxl_canny_mid", fusion_image, 1.00], ["ipa_sdxl_plus_face", ipa_image_face, ipa_weight]]

                second_diffusion_output_image = inpaint(
                    input_image,
                    input_mask,
                    controlnet_pairs,
                    input_prompts[index],
                    diffusion_steps=second_diffusion_steps,
                    cfg_scale=7 if not lcm_accelerate else 2,
                    denoising_strength=second_denoising_strength,
                    hr_scale=default_hr_scale,
                    seed=seed,
                    sampler="DPM++ 2M SDE Karras" if not lcm_accelerate else "Euler a",
                )

                # use original template face area to shift generated face color at last
                if color_shift_last:
                    ep_logger.info("Start color shift last.")
                    # scale box
                    rescale_retinaface_box = [int(i * default_hr_scale) for i in input_image_retinaface_box]
                    second_diffusion_output_image_uint8 = np.uint8(np.array(second_diffusion_output_image))
                    second_diffusion_output_image_crop = Image.fromarray(
                        second_diffusion_output_image_uint8[
                            rescale_retinaface_box[1] : rescale_retinaface_box[3], rescale_retinaface_box[0] : rescale_retinaface_box[2], :
                        ]
                    )

                    # apply color shift
                    second_diffusion_output_image_crop_color_shift = np.array(copy.deepcopy(second_diffusion_output_image_crop))
                    second_diffusion_output_image_crop_color_shift = color_transfer(
                        second_diffusion_output_image_crop_color_shift, template_image_original_face_area
                    )

                    # detect face area
                    face_skin_mask = np.float32(
                        face_skin(second_diffusion_output_image_crop, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10]])[0]
                    )
                    face_skin_mask = cv2.blur(face_skin_mask, (32, 32)) / 255

                    # paste back to photo
                    second_diffusion_output_image_uint8[
                        rescale_retinaface_box[1] : rescale_retinaface_box[3], rescale_retinaface_box[0] : rescale_retinaface_box[2], :
                    ] = second_diffusion_output_image_crop_color_shift * face_skin_mask + np.array(second_diffusion_output_image_crop) * (
                        1 - face_skin_mask
                    )
                    second_diffusion_output_image = Image.fromarray(second_diffusion_output_image_uint8)

                # use original template face area to transfer makeup
                if makeup_transfer:
                    rescale_retinaface_box = [int(i * default_hr_scale) for i in input_image_retinaface_box]
                    second_diffusion_output_image_uint8 = np.uint8(np.array(second_diffusion_output_image))
                    second_diffusion_output_image_crop = Image.fromarray(
                        second_diffusion_output_image_uint8[
                            rescale_retinaface_box[1] : rescale_retinaface_box[3], rescale_retinaface_box[0] : rescale_retinaface_box[2], :
                        ]
                    )
                    template_image_original_face_area = Image.fromarray(np.uint8(template_image_original_face_area))

                    # makeup transfer
                    second_diffusion_output_image_crop_makeup_transfer = second_diffusion_output_image_crop.resize([256, 256])
                    template_image_original_face_area = Image.fromarray(np.uint8(template_image_original_face_area)).resize([256, 256])
                    second_diffusion_output_image_crop_makeup_transfer = psgan_inference.transfer(
                        second_diffusion_output_image_crop_makeup_transfer, template_image_original_face_area
                    )
                    second_diffusion_output_image_crop_makeup_transfer = second_diffusion_output_image_crop_makeup_transfer.resize(
                        [np.shape(second_diffusion_output_image_crop)[1], np.shape(second_diffusion_output_image_crop)[0]]
                    )

                    # detect face area
                    face_skin_mask = np.float32(
                        face_skin(second_diffusion_output_image_crop, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 11, 12, 13]])[
                            0
                        ]
                    )
                    face_skin_mask = cv2.blur(face_skin_mask, (32, 32)) / 255 * makeup_transfer_ratio

                    # paste back to photo
                    second_diffusion_output_image_uint8[
                        rescale_retinaface_box[1] : rescale_retinaface_box[3], rescale_retinaface_box[0] : rescale_retinaface_box[2], :
                    ] = np.array(second_diffusion_output_image_crop_makeup_transfer) * face_skin_mask + np.array(
                        second_diffusion_output_image_crop
                    ) * (
                        1 - face_skin_mask
                    )
                    second_diffusion_output_image = Image.fromarray(np.uint8(np.clip(second_diffusion_output_image_uint8, 0, 255)))

                # If it is a large template for cutting, paste the reconstructed image back
                if crop_face_preprocess:
                    ep_logger.info("Start paste crop image to origin template.")
                    origin_loop_template_image = np.array(copy.deepcopy(loop_template_image))

                    x1, y1, x2, y2 = loop_template_crop_safe_box
                    second_diffusion_output_image = second_diffusion_output_image.resize([x2 - x1, y2 - y1], Image.Resampling.LANCZOS)
                    origin_loop_template_image[y1:y2, x1:x2] = np.array(second_diffusion_output_image)

                    loop_output_image = Image.fromarray(np.uint8(origin_loop_template_image))
                else:
                    loop_output_image = second_diffusion_output_image.resize([loop_template_image.width, loop_template_image.height])

                # Given the current user id, compute the Face ID of the generation w.r.t the roop image. Considering
                # the multi-person template, we don't compute the FaceID of the final output image for simplicity.
                if display_score:
                    loop_output_image = np.array(loop_output_image)
                    if crop_face_preprocess:
                        x1, y1, x2, y2 = loop_template_crop_safe_box
                        loop_output_image_face = loop_output_image[y1:y2, x1:x2]
                    else:
                        loop_output_image_face = loop_output_image

                    embedding = face_recognition(dict(user=Image.fromarray(np.uint8(loop_output_image_face))))[OutputKeys.IMG_EMBEDDING]
                    roop_image_embedding = face_recognition(dict(user=Image.fromarray(np.uint8(roop_images[index]))))[
                        OutputKeys.IMG_EMBEDDING
                    ]
                    loop_output_image_faceid = np.dot(embedding, np.transpose(roop_image_embedding))[0][0]
                    face_id_outputs.append(
                        (roop_images[index], "{:.2f}, {}, the reference image".format(loop_output_image_faceid, user_ids[index]))
                    )
                    if ipa_control and user_ids[index] != "ipa_control_only":
                        ipa_image_embedding = face_recognition(dict(user=Image.fromarray(np.uint8(ipa_images[index]))))[
                            OutputKeys.IMG_EMBEDDING
                        ]
                        ipa_image_faceid = np.dot(embedding, np.transpose(ipa_image_embedding))[0][0]
                        face_id_outputs.append(
                            (ipa_images[index], "{:.2f}, {}, the image prompt".format(ipa_image_faceid, user_ids[index]))
                        )
                    loop_output_image = Image.fromarray(loop_output_image)

                if min(len(template_face_safe_boxes), len(user_ids) - last_user_id_none_num) > 1:
                    ep_logger.info("Start paste crop image to origin template in multi people.")
                    template_face_safe_box = template_face_safe_boxes[index]
                    output_image_mask = np.zeros_like(np.array(output_image))
                    output_image_mask[
                        template_face_safe_box[1] : template_face_safe_box[3], template_face_safe_box[0] : template_face_safe_box[2]
                    ] = 255
                    output_image_mask = cv2.blur(output_image_mask, (32, 32)) / 255

                    output_image = np.array(loop_output_image, np.float32) * output_image_mask + np.array(output_image) * (
                        1 - output_image_mask
                    )
                    output_image = np.uint8(output_image)
                else:
                    output_image = loop_output_image

            try:
                if min(len(template_face_safe_boxes), len(user_ids) - last_user_id_none_num) > 1 or background_restore:
                    ep_logger.info("Start Third diffusion for background.")
                    output_image = Image.fromarray(np.uint8(output_image))
                    # When reconstructing the entire background, use smaller denoise values with larger diffusion_steps to prevent discordant scenes and image collapse.
                    denoising_strength = background_restore_denoising_strength if background_restore else 0.3

                    if not background_restore:
                        h, w, c = np.shape(output_image)

                        # Set the padding size for edge of faces
                        background_padding_size = 50

                        # Calculate the left, top, right, bottom of all faces now
                        left, top, right, bottom = [
                            np.min(np.array(template_face_safe_boxes)[:, 0]) - background_padding_size,
                            np.min(np.array(template_face_safe_boxes)[:, 1]) - background_padding_size,
                            np.max(np.array(template_face_safe_boxes)[:, 2]) + background_padding_size,
                            np.max(np.array(template_face_safe_boxes)[:, 3]) + background_padding_size,
                        ]
                        # Calculate the width, height, center_x, and center_y of all faces, and get the long side for rec
                        width, height, center_x, center_y = [right - left, bottom - top, (left + right) / 2, (top + bottom) / 2]
                        long_side = max(width, height)

                        # Calculate the new left, top, right, bottom of all faces for clipping
                        # Pad the box to square for saving GPU memomry
                        left, top = int(np.clip(center_x - long_side // 2, 0, w - 1)), int(np.clip(center_y - long_side // 2, 0, h - 1))
                        right, bottom = int(np.clip(left + long_side, 0, w - 1)), int(np.clip(top + long_side, 0, h - 1))

                        # Crop image and mask for Diffusion
                        sub_output_image = output_image.crop([left, top, right, bottom])
                        sub_output_mask = output_mask.crop([left, top, right, bottom])

                        # record origin width and height
                        sub_output_image_width = sub_output_image.width
                        sub_output_image_height = sub_output_image.height

                        # get target_short_side base on the ratio of width and height
                        if sub_output_image.width / sub_output_image.height > 1.5 or sub_output_image.height / sub_output_image.width > 1.5:
                            target_short_side = 512
                        else:
                            target_short_side = 768

                        # If the short side is greater than target_short_side, we will resize the image with the short side of target_short_side
                        short_side = min(sub_output_image.width, sub_output_image.height)
                        if min(sub_output_image.width, sub_output_image.height) > target_short_side:
                            resize = float(short_side / target_short_side)
                        else:
                            resize = 1
                        new_size = (int(sub_output_image.width // resize // 32 * 32), int(sub_output_image.height // resize // 32 * 32))
                        sub_output_image = sub_output_image.resize(new_size, Image.Resampling.LANCZOS)

                        # Diffusion
                        if not sdxl_pipeline_flag:
                            controlnet_pairs = [["canny", sub_output_image, 1.00, 1], ["color", sub_output_image, 1.00, 1]]
                        else:
                            controlnet_pairs = [["sdxl_canny_mid", sub_output_image, 1.00, 1]]
                        sub_output_image = inpaint(
                            sub_output_image,
                            sub_output_mask,
                            controlnet_pairs,
                            input_prompt_without_lora,
                            diffusion_steps=30 if not lcm_accelerate else 8,
                            cfg_scale=7 if not lcm_accelerate else 2,
                            denoising_strength=denoising_strength,
                            hr_scale=1,
                            seed=seed,
                            sampler="DPM++ 2M SDE Karras" if not lcm_accelerate else "Euler a",
                        )

                        # Paste the image back to the background
                        sub_output_image = sub_output_image.resize([sub_output_image_width, sub_output_image_height])
                        output_image = np.array(output_image)
                        output_image[top:bottom, left:right] = np.array(sub_output_image)
                        output_image = Image.fromarray(output_image)
                    else:
                        short_side = min(output_image.width, output_image.height)
                        # get target_short_side base on the ratio of width and height
                        if output_image.width / output_image.height > 1.5 or output_image.height / output_image.width > 1.5:
                            target_short_side = 512
                        else:
                            target_short_side = 768
                        resize = float(short_side / target_short_side)
                        new_size = (int(output_image.width // resize), int(output_image.height // resize))
                        output_image = output_image.resize(new_size, Image.Resampling.LANCZOS)

                        if not sdxl_pipeline_flag:
                            controlnet_pairs = [["canny", output_image, 1.00, 1], ["color", output_image, 1.00, 1]]
                        else:
                            controlnet_pairs = [["sdxl_canny_mid", output_image, 1.00, 1]]
                        output_image = inpaint(
                            output_image,
                            output_mask,
                            controlnet_pairs,
                            input_prompt_without_lora,
                            diffusion_steps=30 if not lcm_accelerate else 8,
                            cfg_scale=7 if not lcm_accelerate else 2,
                            denoising_strength=denoising_strength,
                            hr_scale=1,
                            seed=seed,
                            sampler="DPM++ 2M SDE Karras" if not lcm_accelerate else "Euler a",
                        )

            except Exception as e:
                torch.cuda.empty_cache()
                traceback.print_exc()
                ep_logger.error(f"Background Restore Failed, Please check the ratio of height and width in template. Error Info: {e}")
                return f"Background Restore Failed, Please check the ratio of height and width in template. Error Info: {e}", outputs, []

            if total_processed_person != 0:
                if skin_retouching_bool:
                    try:
                        ep_logger.info("Start Skin Retouching.")
                        # Skin Retouching is performed here.
                        output_image = Image.fromarray(
                            cv2.cvtColor(skin_retouching(output_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB)
                        )
                    except Exception as e:
                        torch.cuda.empty_cache()
                        traceback.print_exc()
                        ep_logger.error(f"Skin Retouching error: {e}")

                if super_resolution:
                    try:
                        ep_logger.info("Start Portrait enhancement.")
                        h, w, c = np.shape(np.array(output_image))
                        # Super-resolution is performed here.
                        output_image = Image.fromarray(
                            cv2.cvtColor(portrait_enhancement(output_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB)
                        )
                    except Exception as e:
                        torch.cuda.empty_cache()
                        traceback.print_exc()
                        ep_logger.error(f"Portrait enhancement error: {e}")
            else:
                output_image = template_image

            outputs.append(output_image)
            if loractl_flag:
                outputs.append(lora_weight_image)
            save_image(
                output_image,
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

            if loop_message != "":
                loop_message += "\n"
            loop_message += f"Template {str(template_idx + 1)} Success."
        except Exception as e:
            torch.cuda.empty_cache()
            traceback.print_exc()
            ep_logger.error(f"Template {str(template_idx + 1)} error: Error info is {e}, skip it.")

            if loop_message != "":
                loop_message += "\n"
            loop_message += f"Template {str(template_idx + 1)} error: Error info is {e}."

    if not shared.opts.data.get("easyphoto_cache_model", True):
        unload_models()

    torch.cuda.empty_cache()
    return loop_message, outputs, face_id_outputs


@switch_sd_model_vae()
def easyphoto_video_infer_forward(
    sd_model_checkpoint,
    sd_model_checkpoint_for_animatediff_text2video,
    sd_model_checkpoint_for_animatediff_image2video,
    t2v_input_prompt,
    t2v_input_width,
    t2v_input_height,
    scene_id,
    upload_control_video,
    upload_control_video_type,
    openpose_video,
    init_image,
    init_image_prompt,
    last_image,
    init_video,
    additional_prompt,
    max_frames,
    max_fps,
    save_as,
    before_face_fusion_ratio,
    after_face_fusion_ratio,
    first_diffusion_steps,
    first_denoising_strength,
    seed,
    crop_face_preprocess,
    apply_face_fusion_before,
    apply_face_fusion_after,
    color_shift_middle,
    super_resolution,
    super_resolution_method,
    skin_retouching_bool,
    display_score,
    makeup_transfer,
    makeup_transfer_ratio,
    face_shape_match,
    video_interpolation,
    video_interpolation_ext,
    tabs,
    ipa_control,
    ipa_weight,
    ipa_image_path,
    lcm_accelerate,
    *user_ids,
):
    # global
    global retinaface_detection, image_face_fusion, skin_retouching, portrait_enhancement, old_super_resolution_method, face_skin, face_recognition, psgan_inference, check_hash

    # check & download weights of basemodel/controlnet+annotator/VAE/face_skin/buffalo/validation_template
    check_files_exists_and_download(check_hash.get("base", True), download_mode="base")
    check_files_exists_and_download(check_hash.get("portrait", True), download_mode="portrait")
    check_files_exists_and_download(check_hash.get("add_video", True), download_mode="add_video")
    if check_hash.get("base", True) or check_hash.get("portrait", True) or check_hash.get("add_video", True):
        refresh_model_vae()
    check_hash["base"] = False
    check_hash["portrait"] = False
    check_hash["add_video"] = False

    checkpoint_type = get_checkpoint_type(sd_model_checkpoint)
    checkpoint_type_text2video = get_checkpoint_type(sd_model_checkpoint_for_animatediff_text2video)
    checkpoint_type_image2video = get_checkpoint_type(sd_model_checkpoint_for_animatediff_image2video)
    if (
        checkpoint_type == 2
        or checkpoint_type == 3
        or checkpoint_type_text2video == 2
        or checkpoint_type_text2video == 3
        or checkpoint_type_image2video == 2
        or checkpoint_type_image2video == 3
    ):
        return "EasyPhoto video infer does not support the SD2 checkpoint and sdxl.", None, None, []

    if ipa_control:
        check_files_exists_and_download(check_hash.get("add_ipa_sdxl", True), download_mode="add_ipa_sdxl")
        if check_hash.get("add_ipa_sdxl", True):
            refresh_model_vae()
        check_hash["add_ipa_sdxl"] = False
    if lcm_accelerate:
        check_files_exists_and_download(check_hash.get("lcm", True), download_mode="lcm")
        check_hash["lcm"] = False

    for user_id in user_ids:
        if user_id != "none":
            if not check_id_valid(user_id, user_id_outpath_samples, models_path):
                return "User id is not exist", None, None, []

    # update donot delete but use "none" as placeholder and will pass this face inpaint later
    passed_userid_list = []
    last_user_id_none_num = 0
    for idx, user_id in enumerate(user_ids):
        if user_id == "none":
            last_user_id_none_num += 1
            passed_userid_list.append(idx)
        else:
            last_user_id_none_num = 0

    if len(user_ids) == last_user_id_none_num:
        return "Please choose a user id.", None, None, []

    # check the version of controlnets, reuse code at L538
    controlnet_version = get_controlnet_version()
    major, minor, patch = map(int, controlnet_version.split("."))
    if major == 0 and minor == 0 and patch == 0:
        return "Please install sd-webui-controlnet from https://github.com/Mikubill/sd-webui-controlnet.", None, None, []
    if ipa_control:
        if major < 1 or minor < 1 or patch < 417:
            return "To use IP-Adapter Control, please upgrade sd-webui-controlnet to the latest version.", None, None, []

    # check the number of controlnets
    max_control_net_unit_count = 3 if not ipa_control else 4
    control_net_unit_count = shared.opts.data.get("control_net_unit_count", 3)
    ep_logger.info("ControlNet unit number: {}".format(control_net_unit_count))
    if control_net_unit_count < max_control_net_unit_count:
        error_info = (
            "Please go to Settings/ControlNet and at least set {} for "
            "Multi-ControlNet: ControlNet unit number (requires restart).".format(max_control_net_unit_count)
        )
        return error_info, None, None, []

    if ipa_control:
        ipa_image_paths = ["none"] * 5  # consistent with user_ids
        ipa_flag = False
        valid_user_id_num, valid_ipa_image_path_num = 0, 0
        for index, user_id in enumerate(user_ids):
            if not ipa_flag and user_id != "none" and ipa_image_path is not None:
                ipa_image_paths[index] = ipa_image_path
                ipa_flag = True
                valid_ipa_image_path_num += 1
            if user_id != "none":
                valid_user_id_num += 1

        if valid_user_id_num > 1:
            ep_logger.error("EasyPhoto does not support IP-Adapter Control with multiple user ids currently.")
            return "EasyPhoto does not support IP-Adapter Control with multiple user ids currently.", None, None, []
        if ipa_control and valid_user_id_num != valid_ipa_image_path_num:
            ep_logger.warning(
                "Found {} user id(s), but only {} image prompt(s) for IP-Adapter Control. Use the reference image "
                "corresponding to the user instead.".format(valid_user_id_num, valid_ipa_image_path_num)
            )
    if lcm_accelerate:
        lcm_lora_name_and_weight = "lcm_lora_sd15:0.60"

    try:
        # choose tabs select
        #
        # max_frames represents the maximum frames in t2v and i2v;
        # it represents the max_frames before interception for conversion in v2v.
        #
        # max_fps represents the desired frame rate;
        # in v2v, if the frame rate of video is less than max_fps, the frame rate of video will be used as the desired frame rate.
        if tabs == 0:  # t2v
            max_frames = int(max_frames)

            if upload_control_video:
                max_fps = int(max_fps)
                template_images, actual_fps = get_mov_all_images(openpose_video, max_fps)
                template_images = [template_images[:max_frames]] if max_frames != -1 else [template_images]
            else:
                actual_fps = int(max_fps)
                template_images = None
        elif tabs == 1:  # i2v
            max_frames = int(max_frames)
            actual_fps = int(max_fps)
            template_images = init_image
        elif tabs == 2:  # v2v
            max_frames = int(max_frames)
            max_fps = int(max_fps)

            template_images, actual_fps = get_mov_all_images(init_video, max_fps)
            template_images = [template_images[:max_frames]] if max_frames != -1 else [template_images]
    except Exception:
        torch.cuda.empty_cache()
        traceback.print_exc()
        return "Please input the correct params or upload a template.", None, None, []

    # create modelscope model
    if retinaface_detection is None:
        retinaface_detection = pipeline(Tasks.face_detection, "damo/cv_resnet50_face-detection_retinaface", model_revision="v2.0.2")
    if image_face_fusion is None:
        image_face_fusion = pipeline(Tasks.image_face_fusion, model="damo/cv_unet-image-face-fusion_damo", model_revision="v1.3")
    if face_skin is None:
        face_skin = Face_Skin(os.path.join(easyphoto_models_path, "face_skin.pth"))
    if skin_retouching is None:
        try:
            skin_retouching = pipeline("skin-retouching-torch", model="damo/cv_unet_skin_retouching_torch", model_revision="v1.0.2")
        except Exception as e:
            torch.cuda.empty_cache()
            traceback.print_exc()
            ep_logger.error(f"Skin Retouching model load error. Error Info: {e}")
    if portrait_enhancement is None or old_super_resolution_method != super_resolution_method:
        try:
            if super_resolution_method == "gpen":
                portrait_enhancement = pipeline(
                    Tasks.image_portrait_enhancement, model="damo/cv_gpen_image-portrait-enhancement", model_revision="v1.0.0"
                )
            elif super_resolution_method == "realesrgan":
                portrait_enhancement = pipeline(
                    "image-super-resolution-x2", model="bubbliiiing/cv_rrdb_image-super-resolution_x2", model_revision="v1.0.2"
                )
            old_super_resolution_method = super_resolution_method
        except Exception as e:
            torch.cuda.empty_cache()
            traceback.print_exc()
            ep_logger.error(f"Portrait Enhancement model load error. Error Info: {e}")

    # psgan for transfer makeup
    if makeup_transfer and psgan_inference is None:
        try:
            makeup_transfer_model_path = os.path.join(easyphoto_models_path, "makeup_transfer.pth")
            face_landmarks_model_path = os.path.join(easyphoto_models_path, "face_landmarks.pth")
            psgan_inference = PSGAN_Inference(
                "cuda", makeup_transfer_model_path, retinaface_detection, face_skin, face_landmarks_model_path
            )
        except Exception as e:
            torch.cuda.empty_cache()
            traceback.print_exc()
            ep_logger.error(f"MakeUp Transfer model load error. Error Info: {e}")

    # To save the GPU memory, create the face recognition model for computing FaceID if the user intend to show it.
    if display_score and face_recognition is None:
        face_recognition = pipeline("face_recognition", model="bubbliiiing/cv_retinafce_recognition", model_revision="v1.0.3")

    # This is to increase the fault tolerance of the code.
    # If the code exits abnormally, it may cause the model to not function properly on the CPU
    modelscope_models_to_gpu()

    # get random seed
    if int(seed) == -1:
        seed = np.random.randint(0, 65536)

    seed_everything(int(seed))

    # params init
    input_prompts = []
    face_id_images = []
    roop_images = []
    face_id_retinaface_boxes = []
    face_id_retinaface_keypoints = []
    face_id_retinaface_masks = []
    best_lora_weights = str(0.9)
    multi_user_facecrop_ratio = 1.5
    input_mask_face_part_only = True
    # safe params
    crop_at_last = True
    crop_at_last_ratio = 3

    if ipa_control:
        ipa_images = []
        ipa_retinaface_boxes = []
        ipa_retinaface_keypoints = []
        ipa_retinaface_masks = []
        ipa_face_part_only = False

    ep_logger.info("Start templates and user_ids preprocess.")

    if tabs == 0:
        reload_sd_model_vae(sd_model_checkpoint_for_animatediff_text2video, "vae-ft-mse-840000-ema-pruned.ckpt")
        if scene_id != "none":
            # scene lora path
            scene_lora_model_path = os.path.join(models_path, "Lora", f"{scene_id}.safetensors")
            if not os.path.exists(scene_lora_model_path):
                return "Please check scene lora is exist or not.", None, None, []
            is_scene_lora, scene_lora_prompt = get_scene_prompt(scene_lora_model_path)
            if not is_scene_lora:
                return "Please use the lora trained by ep.", None, None, []

            t2v_input_prompt = t2v_input_prompt + f"<lora:{scene_id}:0.80>, "
            if lcm_accelerate:
                t2v_input_prompt += f"<lora:{lcm_lora_name_and_weight}>, "

            # text to image with scene lora
            ep_logger.info(f"Text to Image with prompt: {t2v_input_prompt} and lora: {scene_lora_model_path}")
            if upload_control_video:
                image = Image.fromarray(np.uint8(template_images[0][0]))
                # Resize the template image with short edges on 512
                short_side = min(image.width, image.height)
                resize = float(short_side / 512.0)
                new_size = (int(image.width // resize), int(image.height // resize))

                if upload_control_video_type == 'depth':
                    ep_logger.info(f"Using depth control for video control input")
                    controlnet_pairs = [["depth", template_images[0], 1, 1]]

                if upload_control_video_type == 'openpose':
                    ep_logger.info(f"Using openpose control for video control input")
                    controlnet_pairs = [["openpose", template_images[0], 1, 1]]

                template_images = txt2img(
                    controlnet_pairs,
                    input_prompt=t2v_input_prompt,
                    diffusion_steps=30 if not lcm_accelerate else 8,
                    cfg_scale=7 if not lcm_accelerate else 2,
                    width=new_size[0],
                    height=new_size[1],
                    default_positive_prompt=DEFAULT_POSITIVE_AD,
                    default_negative_prompt=DEFAULT_NEGATIVE_AD,
                    seed=seed,
                    sampler="DPM++ 2M SDE Karras" if not lcm_accelerate else "Euler a",
                    animatediff_flag=True,
                    animatediff_video_length=len(template_images[0]),
                    animatediff_fps=int(actual_fps),
                )
            else:
                template_images = txt2img(
                    [],
                    input_prompt=t2v_input_prompt,
                    diffusion_steps=30 if not lcm_accelerate else 8,
                    cfg_scale=7 if not lcm_accelerate else 2,
                    width=t2v_input_width,
                    height=t2v_input_height,
                    default_positive_prompt=DEFAULT_POSITIVE_AD,
                    default_negative_prompt=DEFAULT_NEGATIVE_AD,
                    seed=seed,
                    sampler="Euler a",
                    animatediff_flag=True,
                    animatediff_video_length=int(max_frames),
                    animatediff_fps=int(actual_fps),
                )
            template_images = [template_images]
        else:
            if lcm_accelerate:
                t2v_input_prompt += f"<lora:{lcm_lora_name_and_weight}>, "
            if upload_control_video:
                image = Image.fromarray(np.uint8(template_images[0][0]))
                # Resize the template image with short edges on 512
                short_side = min(image.width, image.height)
                resize = float(short_side / 512.0)
                new_size = (int(image.width // resize), int(image.height // resize))

                controlnet_pairs = [["openpose", template_images[0], 1, 1]]
                template_images = txt2img(
                    controlnet_pairs,
                    input_prompt=t2v_input_prompt,
                    diffusion_steps=30 if not lcm_accelerate else 8,
                    cfg_scale=7 if not lcm_accelerate else 2,
                    width=new_size[0],
                    height=new_size[1],
                    default_positive_prompt=DEFAULT_POSITIVE_AD,
                    default_negative_prompt=DEFAULT_NEGATIVE_AD,
                    seed=seed,
                    sampler="DPM++ 2M SDE Karras" if not lcm_accelerate else "Euler a",
                    animatediff_flag=True,
                    animatediff_video_length=len(template_images[0]),
                    animatediff_fps=int(actual_fps),
                )
            else:
                template_images = txt2img(
                    [],
                    input_prompt=t2v_input_prompt,
                    diffusion_steps=30 if not lcm_accelerate else 8,
                    cfg_scale=7 if not lcm_accelerate else 2,
                    width=t2v_input_width,
                    height=t2v_input_height,
                    default_positive_prompt=DEFAULT_POSITIVE_AD,
                    default_negative_prompt=DEFAULT_NEGATIVE_AD,
                    seed=seed,
                    sampler="DPM++ 2M SDE Karras" if not lcm_accelerate else "Euler a",
                    animatediff_flag=True,
                    animatediff_video_length=int(max_frames),
                    animatediff_fps=int(actual_fps),
                )
            template_images = [template_images]
    elif tabs == 1:
        reload_sd_model_vae(sd_model_checkpoint_for_animatediff_image2video, "vae-ft-mse-840000-ema-pruned.ckpt")
        image = Image.fromarray(np.uint8(template_images)).convert("RGB")
        if last_image is not None:
            last_image = Image.fromarray(np.uint8(last_image)).convert("RGB")
            animatediff_reserve_scale = 1.00
            denoising_strength = 0.55
        else:
            animatediff_reserve_scale = 0.75
            denoising_strength = 0.65
        if lcm_accelerate:
            init_image_prompt += f"<lora:{lcm_lora_name_and_weight}>, "
        # Resize the template image with short edges on 512
        short_side = min(image.width, image.height)
        resize = float(short_side / 512.0)
        new_size = (int(image.width // resize), int(image.height // resize))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        template_images = inpaint(
            image,
            None,
            [],
            input_prompt=init_image_prompt,
            diffusion_steps=30 if not lcm_accelerate else 8,
            cfg_scale=7 if not lcm_accelerate else 2,
            denoising_strength=denoising_strength,
            hr_scale=1,
            default_positive_prompt=DEFAULT_POSITIVE_AD,
            default_negative_prompt=DEFAULT_NEGATIVE_AD,
            seed=seed,
            sampler="DPM++ 2M SDE Karras" if not lcm_accelerate else "Euler a",
            animatediff_flag=True,
            animatediff_video_length=int(max_frames),
            animatediff_fps=int(actual_fps),
            animatediff_reserve_scale=animatediff_reserve_scale,
            animatediff_last_image=last_image,
        )
        template_images = [template_images]

    reload_sd_model_vae(sd_model_checkpoint, "vae-ft-mse-840000-ema-pruned.ckpt")

    # TODO ： multiuser in VideoModel is unable to use at 23/11/03, we keep code for future test
    for user_id in user_ids:
        if user_id == "none":
            # use some placeholder
            input_prompts.append("none")
            face_id_images.append("none")
            roop_images.append("none")
            face_id_retinaface_boxes.append([])
            face_id_retinaface_keypoints.append([])
            face_id_retinaface_masks.append([])
            if ipa_control:
                ipa_images.append("none")
                ipa_retinaface_boxes.append([])
                ipa_retinaface_keypoints.append([])
                ipa_retinaface_masks.append([])
        else:
            # get prompt
            input_prompt = f"{validation_prompt}, <lora:{user_id}:{best_lora_weights}>" + "<lora:FilmVelvia3:0.65>" + additional_prompt
            # Add the ddpo LoRA into the input prompt if available.
            lora_model_path = os.path.join(models_path, "Lora")
            if os.path.exists(os.path.join(lora_model_path, "ddpo_{}.safetensors".format(user_id))):
                input_prompt += "<lora:ddpo_{}>".format(user_id)

            if lcm_accelerate:
                input_prompt += f"<lora:{lcm_lora_name_and_weight}>, "
            # get best image after training
            best_outputs_paths = glob.glob(os.path.join(user_id_outpath_samples, user_id, "user_weights", "best_outputs", "*.jpg"))
            # get roop image
            if len(best_outputs_paths) > 0:
                face_id_image_path = best_outputs_paths[0]
            else:
                face_id_image_path = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")
            roop_image_path = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")

            face_id_image = Image.open(face_id_image_path).convert("RGB")
            roop_image = Image.open(roop_image_path).convert("RGB")

            if ipa_control:
                if ipa_image_paths[index] != "none":
                    ipa_image = Image.open(ipa_image_paths[index])
                    ipa_image = ImageOps.exif_transpose(ipa_image).convert("RGB")
                else:
                    ipa_image = copy.deepcopy(roop_image)

                _ipa_retinaface_boxes, _ipa_retinaface_keypoints, _ipa_retinaface_masks = call_face_crop(
                    retinaface_detection, ipa_image, 1, "crop"
                )
                if len(_ipa_retinaface_boxes) == 0:
                    ep_logger.error("No face is detected in the uploaded image prompt.")
                    return "Please upload a image prompt with face.", None, None, []
                if len(_ipa_retinaface_boxes) > 1:
                    ep_logger.warning(
                        "{} faces are detected in the uploaded image prompt. "
                        "Only the left one will be used.".format(len(_ipa_retinaface_boxes))
                    )

            # Crop user images to obtain portrait boxes, facial keypoints, and masks
            _face_id_retinaface_boxes, _face_id_retinaface_keypoints, _face_id_retinaface_masks = call_face_crop(
                retinaface_detection, face_id_image, multi_user_facecrop_ratio, "face_id"
            )
            _face_id_retinaface_box = _face_id_retinaface_boxes[0]
            _face_id_retinaface_keypoint = _face_id_retinaface_keypoints[0]
            _face_id_retinaface_mask = _face_id_retinaface_masks[0]

            input_prompts.append(input_prompt)
            face_id_images.append(face_id_image)
            roop_images.append(roop_image)
            face_id_retinaface_boxes.append(_face_id_retinaface_box)
            face_id_retinaface_keypoints.append(_face_id_retinaface_keypoint)
            face_id_retinaface_masks.append(_face_id_retinaface_mask)
            if ipa_control:
                ipa_images.append(ipa_image)
                ipa_retinaface_boxes.append(_ipa_retinaface_boxes[0])
                ipa_retinaface_keypoints.append(_ipa_retinaface_keypoints[0])
                ipa_retinaface_masks.append(_ipa_retinaface_masks[0])

    outputs = []
    loop_message = ""
    for template_idx, template_image in enumerate(template_images):
        template_idx_info = f"""
            Start Generate template                 : {str(template_idx + 1)};
            user_ids                                : {str(user_ids)};
            input_prompts                           : {str(input_prompts)};
            before_face_fusion_ratio                : {str(before_face_fusion_ratio)};
            after_face_fusion_ratio                 : {str(after_face_fusion_ratio)};
            first_diffusion_steps                   : {str(first_diffusion_steps)};
            first_denoising_strength                : {str(first_denoising_strength)};
            seed                                    : {seed}
            apply_face_fusion_before                : {str(apply_face_fusion_before)}
            apply_face_fusion_after                 : {str(apply_face_fusion_after)}
            color_shift_middle                      : {str(color_shift_middle)}
            super_resolution                        : {str(super_resolution)}
            ipa_control                             : {str(ipa_control)}
            ipa_weight                              : {str(ipa_weight)}
            ipa_image_path                          : {str(ipa_image_path)}
        """
        ep_logger.info(template_idx_info)

        try:
            # open the template image
            template_image = [Image.fromarray(np.uint8(_)).convert("RGB") for _ in template_image]
            loop_template_image = copy.deepcopy(template_image)

            # crop images from templates and get the box of each photos
            input_image, loop_template_crop_safe_box = call_face_crop_templates(
                loop_template_image, retinaface_detection, crop_face_preprocess
            )

            # Resize the template image with short edges on 512
            new_input_image = []
            for idx, _input_image in enumerate(input_image):
                ep_logger.info(f"Start {idx} Image resize to 512.")
                short_side = min(_input_image.width, _input_image.height)
                resize = float(short_side / 512.0)
                new_size = (int(_input_image.width // resize), int(_input_image.height // resize))
                _input_image = _input_image.resize(new_size, Image.Resampling.LANCZOS)

                if crop_face_preprocess:
                    new_width = int(np.shape(_input_image)[1] // 32 * 32)
                    new_height = int(np.shape(_input_image)[0] // 32 * 32)
                    _input_image = _input_image.resize([new_width, new_height], Image.Resampling.LANCZOS)

                new_input_image.append(_input_image)
            input_image = new_input_image

            # Detect the box where the face of the template image is located and obtain its corresponding small mask
            input_image_retinaface_boxes = []
            input_image_retinaface_keypoints = []
            input_masks = []

            for idx, _input_image in enumerate(input_image):
                ep_logger.info(f"Start {idx} face detect.")
                _input_image_retinaface_boxes, _input_image_retinaface_keypoints, _input_masks = call_face_crop(
                    retinaface_detection, _input_image, 1.05, "template"
                )
                if len(_input_image_retinaface_boxes) == 0:
                    input_image_retinaface_boxes.append(None)
                    input_image_retinaface_keypoints.append(None)
                    input_masks.append(None)
                else:
                    _input_image_retinaface_box = _input_image_retinaface_boxes[0]
                    _input_image_retinaface_keypoint = _input_image_retinaface_keypoints[0]
                    _input_mask = _input_masks[0]

                    input_image_retinaface_boxes.append(_input_image_retinaface_box)
                    input_image_retinaface_keypoints.append(_input_image_retinaface_keypoint)
                    input_masks.append(_input_mask)

            replaced_input_image = []
            ipa_image_face = []
            new_input_image = []
            new_input_mask = []
            template_image_original_face_area = []
            for _input_image, _input_image_retinaface_box, _input_image_retinaface_keypoint, _input_mask in zip(
                input_image, input_image_retinaface_boxes, input_image_retinaface_keypoints, input_masks
            ):
                # backup input template and mask
                original_input_template = copy.deepcopy(_input_image)
                if _input_image_retinaface_box is None:
                    replaced_input_image.append(_input_image)
                    new_input_image.append(_input_image)
                    new_input_mask.append(None)
                    template_image_original_face_area.append(None)
                    continue

                # Paste user images onto template images
                _replaced_input_image = crop_and_paste(
                    face_id_images[0],
                    face_id_retinaface_masks[0],
                    _input_image,
                    face_id_retinaface_keypoints[0],
                    _input_image_retinaface_keypoint,
                    face_id_retinaface_boxes[0],
                )
                _replaced_input_image = Image.fromarray(np.uint8(_replaced_input_image))
                replaced_input_image.append(_replaced_input_image)

                # The cropped face area (square) in the reference image will be used in IP-Adapter.
                if ipa_control:
                    try:
                        _ipa_retinaface_box = ipa_retinaface_boxes[0]
                        _ipa_retinaface_keypoint = ipa_retinaface_keypoints[0]
                        _ipa_retinaface_mask = ipa_retinaface_masks[0]
                        _ipa_face_width = _ipa_retinaface_box[2] - _ipa_retinaface_box[0]

                        if not ipa_face_part_only:
                            _ipa_mask, _brow_mask = face_skin(
                                ipa_images[0], retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 11, 12, 13], [2, 3]]
                            )
                            _ipa_kernel_size = np.ones((int(_ipa_face_width // 10), int(_ipa_face_width // 10)), np.uint8)
                            # Fill small holes with a close operation (w/o cv2.dilate)
                            _ipa_mask = Image.fromarray(np.uint8(cv2.morphologyEx(np.array(_ipa_mask), cv2.MORPH_CLOSE, _ipa_kernel_size)))
                        else:
                            # Expand the reference image in the x-axis direction to include the ears.
                            h, w, c = np.shape(_ipa_retinaface_mask)
                            _ipa_mask = np.zeros_like(np.array(_ipa_retinaface_mask, np.uint8))
                            _ipa_retinaface_box[0] = np.clip(np.array(_ipa_retinaface_box[0], np.int32) - _ipa_face_width * 0.15, 0, w - 1)
                            _ipa_retinaface_box[2] = np.clip(np.array(_ipa_retinaface_box[2], np.int32) + _ipa_face_width * 0.15, 0, w - 1)
                            _ipa_mask[
                                _ipa_retinaface_box[1] : _ipa_retinaface_box[3], _ipa_retinaface_box[0] : _ipa_retinaface_box[2]
                            ] = 255
                            _ipa_mask = Image.fromarray(np.uint8(_ipa_mask))
                            _brow_mask = None

                        # Since the image encoder of IP-Adapter will crop/resize the image prompt to (224, 224),
                        # we pad the face w.r.t the long side for an aspect ratio of 1.
                        _ipa_mask = np.array(_ipa_mask, np.uint8) / 255
                        _ipa_image_face = np.ones_like(np.array(ipa_images[0])) * 255
                        _ipa_image_face = Image.fromarray(np.uint8(np.array(ipa_images[0]) * _ipa_mask + _ipa_image_face * (1 - _ipa_mask)))
                        _ipa_image_face = _ipa_image_face.crop(_ipa_retinaface_box)

                        # Align the ipa face
                        _ipa_retinaface_keypoint[:, 0] -= _ipa_retinaface_box[0]
                        _ipa_retinaface_keypoint[:, 1] -= _ipa_retinaface_box[1]
                        _ipa_image_face = Image.fromarray(
                            np.uint8(alignment_photo(np.array(_ipa_image_face), np.array(_ipa_retinaface_keypoint, np.int))[0])
                        )

                        # If brow_mask is not None, remove the skin above brows
                        # Only edit the facial area here, hair is useless
                        if _brow_mask is not None:
                            _brow_mask = _brow_mask.crop(_ipa_retinaface_box)
                            _brow_mask = Image.fromarray(
                                np.uint8(
                                    alignment_photo(
                                        np.array(_brow_mask), np.array(_ipa_retinaface_keypoint, np.int), borderValue=(0, 0, 0)
                                    )[0]
                                )
                            )
                            y_coords, _, _ = np.where(np.array(_brow_mask) > 0)
                            min_y = max(int(np.min(y_coords)), 1)
                            _ipa_image_face = np.array(_ipa_image_face, np.uint8)
                            _ipa_image_face[:min_y, :, :] = 255
                            _ipa_image_face = Image.fromarray(_ipa_image_face)

                        _padded_size = (max(_ipa_image_face.size), max(_ipa_image_face.size))
                        _ipa_image_face = ImageOps.pad(_ipa_image_face, _padded_size, color=(255, 255, 255))
                        ipa_image_face.append(_ipa_image_face)
                    except Exception as e:
                        ipa_image_face.append(None)

                        torch.cuda.empty_cache()
                        traceback.print_exc()
                        ep_logger.error(f"Crop ipa image error. Continue. Error Info: {e}")
                        continue

                # Fusion of user reference images and input images as canny input
                if roop_images[0] is not None and apply_face_fusion_before:
                    try:
                        _fusion_image = image_face_fusion(dict(template=_input_image, user=roop_images[0]))[OutputKeys.OUTPUT_IMG]
                        _fusion_image = Image.fromarray(cv2.cvtColor(_fusion_image, cv2.COLOR_BGR2RGB))

                        # The edge shadows generated by fusion are filtered out by taking intersections of masks of faces before and after fusion.
                        # detect face area
                        _fusion_image_mask = np.int32(
                            np.float32(face_skin(_fusion_image, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 11, 12, 13]])[0])
                            > 128
                        )
                        _input_image_mask = np.int32(
                            np.float32(face_skin(_input_image, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 11, 12, 13]])[0])
                            > 128
                        )
                        # paste back to photo
                        _fusion_image = _fusion_image * _fusion_image_mask * _input_image_mask + np.array(_input_image) * (
                            1 - _fusion_image_mask * _input_image_mask
                        )
                        _fusion_image = cv2.medianBlur(np.uint8(_fusion_image), 3)
                        _fusion_image = Image.fromarray(_fusion_image)

                        _input_image = Image.fromarray(
                            np.uint8(
                                (
                                    np.array(_input_image, np.float32) * (1 - before_face_fusion_ratio)
                                    + np.array(_fusion_image, np.float32) * before_face_fusion_ratio
                                )
                            )
                        )
                    except Exception as e:
                        new_input_image.append(_input_image)
                        new_input_mask.append(None)
                        template_image_original_face_area.append(None)

                        torch.cuda.empty_cache()
                        traceback.print_exc()
                        ep_logger.error(f"Apply face fusion before error. Continue. Error Info: {e}")
                        continue

                if input_mask_face_part_only:
                    try:
                        face_width = _input_image_retinaface_box[2] - _input_image_retinaface_box[0]
                        _input_mask = face_skin(_input_image, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 11, 12, 13]])[0]

                        kernel_size = np.ones((int(face_width // 10), int(face_width // 10)), np.uint8)
                        # Fill small holes with a close operation
                        _input_mask = Image.fromarray(np.uint8(cv2.morphologyEx(np.array(_input_mask), cv2.MORPH_CLOSE, kernel_size)))
                        # Use dilate to reconstruct the surrounding area of the face
                        _input_mask = Image.fromarray(np.uint8(cv2.dilate(np.array(_input_mask), kernel_size, iterations=1)))
                    except Exception as e:
                        new_input_image.append(_input_image)
                        new_input_mask.append(None)
                        template_image_original_face_area.append(None)

                        torch.cuda.empty_cache()
                        traceback.print_exc()
                        ep_logger.error(f"Mask Calculation error. Continue. Error Info: {e}")
                        continue
                else:
                    # Expand the template image in the x-axis direction to include the ears.
                    h, w, c = np.shape(_input_mask)
                    _input_mask = np.zeros_like(np.array(_input_mask, np.uint8))
                    _input_image_retinaface_box = np.int32(_input_image_retinaface_box)

                    face_width = _input_image_retinaface_box[2] - _input_image_retinaface_box[0]
                    _input_image_retinaface_box[0] = np.clip(
                        np.array(_input_image_retinaface_box[0], np.int32) - face_width * 0.10, 0, w - 1
                    )
                    _input_image_retinaface_box[2] = np.clip(
                        np.array(_input_image_retinaface_box[2], np.int32) + face_width * 0.10, 0, w - 1
                    )

                    # get new input_mask
                    _input_mask[
                        _input_image_retinaface_box[1] : _input_image_retinaface_box[3],
                        _input_image_retinaface_box[0] : _input_image_retinaface_box[2],
                    ] = 255
                    _input_mask = Image.fromarray(np.uint8(_input_mask))

                # here we get the retinaface_box, we should use this Input box and face pixel to refine the output face pixel colors
                _template_image_original_face_area = np.array(original_input_template)[
                    _input_image_retinaface_box[1] : _input_image_retinaface_box[3],
                    _input_image_retinaface_box[0] : _input_image_retinaface_box[2],
                    :,
                ]

                new_input_image.append(_input_image)
                new_input_mask.append(_input_mask)
                template_image_original_face_area.append(_template_image_original_face_area)

            input_image = new_input_image
            input_mask = new_input_mask

            # First diffusion, facial reconstruction
            ep_logger.info("Start First diffusion.")
            if not face_shape_match:
                controlnet_pairs = [
                    ["canny", input_image, 0.50, 1],
                    ["openpose", replaced_input_image, 0.50, 1],
                    ["color", input_image, 0.85, 1],
                ]
                if ipa_control and not (None in ipa_image_face):
                    controlnet_pairs.append(["ipa_full_face", ipa_image_face, ipa_weight])
            else:
                controlnet_pairs = [["canny", input_image, 0.50, 1], ["openpose", replaced_input_image, 0.50, 1]]
                if ipa_control and not (None in ipa_image_face):
                    controlnet_pairs.append(["ipa_full_face", ipa_image_face, ipa_weight])

            sum_input_mask = []
            for _input_mask in input_mask:
                if _input_mask is not None:
                    sum_input_mask.append(np.array(_input_mask))
            if len(sum_input_mask) == 0:
                sum_input_mask = None
            else:
                sum_input_mask = Image.fromarray(np.uint8(np.max(np.array(sum_input_mask), axis=0)))

            first_diffusion_output_image = inpaint(
                input_image,
                sum_input_mask,
                controlnet_pairs,
                diffusion_steps=first_diffusion_steps,
                cfg_scale=7 if not lcm_accelerate else 2,
                denoising_strength=first_denoising_strength,
                input_prompt=input_prompts[0],
                hr_scale=1.0,
                seed=seed,
                sd_model_checkpoint=sd_model_checkpoint,
                default_positive_prompt=DEFAULT_POSITIVE_AD,
                default_negative_prompt=DEFAULT_NEGATIVE_AD,
                sampler="DPM++ 2M SDE Karras" if not lcm_accelerate else "Euler a",
                animatediff_flag=True,
                animatediff_fps=int(actual_fps),
            )

            _outputs = []
            frame_idx = 0
            for idx, [
                _first_diffusion_output_image,
                _loop_template_image,
                _loop_template_crop_safe_box,
                _input_image_retinaface_box,
                _template_image_original_face_area,
            ] in enumerate(
                zip(
                    first_diffusion_output_image,
                    loop_template_image,
                    loop_template_crop_safe_box,
                    input_image_retinaface_boxes,
                    template_image_original_face_area,
                )
            ):
                if _input_image_retinaface_box is not None:
                    # TODO : this color shift is too hardcode and naive for video
                    if color_shift_middle:
                        try:
                            # apply color shift
                            ep_logger.info(f"Start {idx} color shift middle.")
                            _first_diffusion_output_image_uint8 = np.uint8(np.array(_first_diffusion_output_image))
                            # crop image first
                            _first_diffusion_output_image_crop = Image.fromarray(
                                _first_diffusion_output_image_uint8[
                                    _input_image_retinaface_box[1] : _input_image_retinaface_box[3],
                                    _input_image_retinaface_box[0] : _input_image_retinaface_box[2],
                                    :,
                                ]
                            )

                            # apply color shift
                            _first_diffusion_output_image_crop_color_shift = np.array(copy.deepcopy(_first_diffusion_output_image_crop))
                            _first_diffusion_output_image_crop_color_shift = color_transfer(
                                _first_diffusion_output_image_crop_color_shift, _template_image_original_face_area
                            )

                            # detect face area
                            face_skin_mask = np.float32(
                                face_skin(
                                    _first_diffusion_output_image_crop, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 11, 12, 13]]
                                )[0]
                            )
                            face_skin_mask = cv2.blur(face_skin_mask, (32, 32)) / 255

                            # paste back to photo
                            _first_diffusion_output_image_uint8[
                                _input_image_retinaface_box[1] : _input_image_retinaface_box[3],
                                _input_image_retinaface_box[0] : _input_image_retinaface_box[2],
                                :,
                            ] = _first_diffusion_output_image_crop_color_shift * face_skin_mask + np.array(
                                _first_diffusion_output_image_crop
                            ) * (
                                1 - face_skin_mask
                            )
                            _first_diffusion_output_image = Image.fromarray(np.uint8(_first_diffusion_output_image_uint8))
                        except Exception as e:
                            torch.cuda.empty_cache()
                            traceback.print_exc()
                            ep_logger.error(f"Color Shift Middle {idx} error. Continue. Error Info: {e}")

                    if roop_images[0] is not None and apply_face_fusion_after:
                        try:
                            # Fusion of facial photos with user photos
                            ep_logger.info(f"Start {idx} second face fusion.")
                            _fusion_image = image_face_fusion(dict(template=_first_diffusion_output_image, user=roop_images[0]))[
                                OutputKeys.OUTPUT_IMG
                            ]  # swap_face(target_img=output_image, source_img=roop_image, model="inswapper_128.onnx", upscale_options=UpscaleOptions())
                            _fusion_image = Image.fromarray(cv2.cvtColor(_fusion_image, cv2.COLOR_BGR2RGB))

                            # The edge shadows generated by fusion are filtered out by taking intersections of masks of faces before and after fusion.
                            # detect face area
                            _fusion_image_mask = np.int32(
                                np.float32(face_skin(_fusion_image, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 11, 12, 13]])[0])
                                > 128
                            )
                            _input_image_mask = np.int32(
                                np.float32(
                                    face_skin(
                                        _first_diffusion_output_image, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 11, 12, 13]]
                                    )[0]
                                )
                                > 128
                            )
                            # paste back to photo
                            _fusion_image = _fusion_image * _fusion_image_mask * _input_image_mask + np.array(
                                _first_diffusion_output_image
                            ) * (1 - _fusion_image_mask * _input_image_mask)
                            _fusion_image = cv2.medianBlur(np.uint8(_fusion_image), 3)
                            _fusion_image = Image.fromarray(_fusion_image)

                            _input_image = Image.fromarray(
                                np.uint8(
                                    (
                                        np.array(_first_diffusion_output_image, np.float32) * (1 - after_face_fusion_ratio)
                                        + np.array(_fusion_image, np.float32) * after_face_fusion_ratio
                                    )
                                )
                            )
                        except Exception as e:
                            _fusion_image = _first_diffusion_output_image
                            _input_image = _first_diffusion_output_image
                            torch.cuda.empty_cache()
                            traceback.print_exc()
                            ep_logger.error(f"Apply Face Fusion After {idx} error. Continue. Error Info: {e}")
                    else:
                        _fusion_image = _first_diffusion_output_image
                        _input_image = _first_diffusion_output_image

                    # use original template face area to transfer makeup
                    if makeup_transfer:
                        try:
                            _input_image_uint8 = np.uint8(np.array(_input_image))
                            _input_image_crop = Image.fromarray(
                                _input_image_uint8[
                                    _input_image_retinaface_box[1] : _input_image_retinaface_box[3],
                                    _input_image_retinaface_box[0] : _input_image_retinaface_box[2],
                                    :,
                                ]
                            )
                            _template_image_original_face_area = Image.fromarray(np.uint8(_template_image_original_face_area))

                            # makeup transfer
                            _input_image_crop_makeup_transfer = _input_image_crop.resize([256, 256])
                            _template_image_original_face_area = Image.fromarray(np.uint8(_template_image_original_face_area)).resize(
                                [256, 256]
                            )
                            _input_image_crop_makeup_transfer = psgan_inference.transfer(
                                _input_image_crop_makeup_transfer, _template_image_original_face_area
                            )
                            _input_image_crop_makeup_transfer = _input_image_crop_makeup_transfer.resize(
                                [np.shape(_input_image_crop)[1], np.shape(_input_image_crop)[0]]
                            )

                            # detect face area
                            face_skin_mask = np.float32(
                                face_skin(_input_image_crop, retinaface_detection, needs_index=[[1, 2, 3, 4, 5, 10, 11, 12, 13]])[0]
                            )
                            face_skin_mask = cv2.blur(face_skin_mask, (32, 32)) / 255 * makeup_transfer_ratio

                            # paste back to photo
                            _input_image_uint8[
                                _input_image_retinaface_box[1] : _input_image_retinaface_box[3],
                                _input_image_retinaface_box[0] : _input_image_retinaface_box[2],
                                :,
                            ] = np.array(_input_image_crop_makeup_transfer) * face_skin_mask + np.array(_input_image_crop) * (
                                1 - face_skin_mask
                            )
                            _input_image = Image.fromarray(np.uint8(np.clip(_input_image_uint8, 0, 255)))
                        except Exception as e:
                            torch.cuda.empty_cache()
                            traceback.print_exc()
                            ep_logger.error(f"Makeup Transfer {idx} error. Continue. Error Info: {e}")
                else:
                    _input_image = _loop_template_image

                # If it is a large template for cutting, paste the reconstructed image back
                if crop_face_preprocess:
                    if _loop_template_crop_safe_box is not None:
                        ep_logger.info(f"Start {idx} paste crop image to origin template.")

                        x1, y1, x2, y2 = _loop_template_crop_safe_box
                        _loop_template_image = np.array(_loop_template_image)
                        _loop_template_image[y1:y2, x1:x2] = np.array(_input_image.resize([x2 - x1, y2 - y1], Image.Resampling.LANCZOS))

                        # backup for old code, will be delete in 2 weeks.
                        # _loop_template_image = _loop_template_image[_loop_template_padding_size: -_loop_template_padding_size, _loop_template_padding_size: -_loop_template_padding_size]

                    _input_image = Image.fromarray(np.uint8(_loop_template_image))

                if skin_retouching_bool:
                    try:
                        ep_logger.info(f"Start {idx} Skin Retouching.")
                        # Skin Retouching is performed here.
                        _input_image = Image.fromarray(
                            cv2.cvtColor(skin_retouching(_input_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB)
                        )
                    except Exception as e:
                        torch.cuda.empty_cache()
                        traceback.print_exc()
                        ep_logger.error(f"Skin Retouching error: {e}")

                if super_resolution:
                    try:
                        ep_logger.info(f"Start {idx} Portrait enhancement.")
                        h, w, c = np.shape(np.array(_input_image))
                        # Super-resolution is performed here.
                        _input_image = Image.fromarray(
                            cv2.cvtColor(portrait_enhancement(_input_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB)
                        )
                    except Exception as e:
                        torch.cuda.empty_cache()
                        traceback.print_exc()
                        ep_logger.error(f"Portrait enhancement error: {e}")

                if display_score:
                    try:
                        # count face id
                        embedding = face_recognition(dict(user=Image.fromarray(np.uint8(_input_image))))[OutputKeys.IMG_EMBEDDING]
                        roop_image_embedding = face_recognition(dict(user=Image.fromarray(np.uint8(roop_images[0]))))[
                            OutputKeys.IMG_EMBEDDING
                        ]
                        loop_output_image_faceid = np.dot(embedding, np.transpose(roop_image_embedding))[0][0]

                        # define font and label
                        _input_image = cv2.putText(
                            np.array(_input_image, np.uint8),
                            "frame_idx: {}, similarity score: {:.2f}".format(frame_idx, loop_output_image_faceid),
                            (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                        )
                        _input_image = Image.fromarray(np.uint8(_input_image))
                    except Exception as e:
                        torch.cuda.empty_cache()
                        traceback.print_exc()
                        ep_logger.error(f"Count similarity error: {e}")

                frame_idx += 1
                _outputs.append(_input_image)

            if video_interpolation:
                modelscope_models_to_cpu()
                try:
                    _outputs = [np.array(_output, np.uint8) for _output in _outputs]
                    _outputs, actual_fps = FIRE_forward(
                        _outputs, actual_fps, os.path.join(easyphoto_models_path, "flownet.pkl"), video_interpolation_ext, 1, fp16=False
                    )
                    _outputs = [Image.fromarray(np.uint8(_output)) for _output in _outputs]
                except Exception as e:
                    torch.cuda.empty_cache()
                    traceback.print_exc()
                    ep_logger.error(f"Video Interpolation error. Continue. Error Info: {e}")
                modelscope_models_to_gpu()

            output_video, output_gif, prefix = convert_to_video(
                os.path.join(easyphoto_video_outpath_samples, "origin"), _outputs, actual_fps, mode=save_as
            )

            if crop_at_last:
                # get max box of face
                last_retinaface_box = []
                for _output in _outputs:
                    _last_retinaface_boxes, _, _ = call_face_crop(retinaface_detection, _output, crop_at_last_ratio, "last_image")
                    if len(_last_retinaface_boxes) == 0:
                        continue
                    _last_retinaface_box = _last_retinaface_boxes[0]
                    last_retinaface_box.append(_last_retinaface_box)
                last_retinaface_box = np.array(last_retinaface_box)
                last_retinaface_box = [
                    np.min(last_retinaface_box[:, 0]),
                    np.min(last_retinaface_box[:, 1]),
                    np.max(last_retinaface_box[:, 2]),
                    np.max(last_retinaface_box[:, 3]),
                ]

                # make width and height can be divisible by 2
                width, height = (last_retinaface_box[2] - last_retinaface_box[0]) // 2 * 2, (
                    last_retinaface_box[3] - last_retinaface_box[1]
                ) // 2 * 2
                last_retinaface_box = [
                    last_retinaface_box[0],
                    last_retinaface_box[1],
                    last_retinaface_box[0] + width,
                    last_retinaface_box[1] + height,
                ]

                # crop
                _new_outputs = []
                for _output in _outputs:
                    _new_outputs.append(_output.crop(last_retinaface_box))
                output_video, output_gif, _ = convert_to_video(
                    os.path.join(easyphoto_video_outpath_samples, "crop"), _new_outputs, actual_fps, prefix=prefix + "_crop", mode=save_as
                )
                _outputs = _new_outputs

            outputs += _outputs
            if loop_message != "":
                loop_message += "\n"
            loop_message += f"Template {str(template_idx + 1)} Success."
        except Exception as e:
            output_video, output_gif = None, None

            torch.cuda.empty_cache()
            traceback.print_exc()
            ep_logger.error(f"Template {str(template_idx + 1)} error: Error info is {e}, skip it.")

            if loop_message != "":
                loop_message += "\n"
            loop_message += f"Template {str(template_idx + 1)} error: Error info is {e}."

    if not shared.opts.data.get("easyphoto_cache_model", True):
        unload_models()

    torch.cuda.empty_cache()
    return loop_message, output_video, output_gif, outputs
