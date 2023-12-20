import copy
import os
import platform
import subprocess
import sys
from glob import glob
from shutil import copyfile

import cv2
import numpy as np
import torch
from modules.images import save_image
from modules.paths import models_path
from modules.shared import opts
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

from scripts.easyphoto_config import (
    cache_log_file_path,
    cloth_id_outpath_samples,
    easyphoto_outpath_samples,
    validation_tryon_prompt,
    tryon_gallery_dir,
    DEFAULT_CLOTH_LORA,
)
from scripts.easyphoto_infer import inpaint
from scripts.easyphoto_utils import (
    align_and_overlay_images,
    apply_mask_to_image,
    check_files_exists_and_download,
    compute_rotation_angle,
    copy_white_mask_to_template,
    crop_image,
    ep_logger,
    expand_box_by_pad,
    expand_roi,
    find_best_angle_ratio,
    get_background_color,
    mask_to_box,
    mask_to_polygon,
    merge_with_inner_canny,
    prepare_tryon_train_data,
    resize_and_stretch,
    resize_image_with_pad,
    seg_by_box,
    find_connected_components,
)
from scripts.sdwebui import get_checkpoint_type, reload_sd_model_vae, switch_sd_model_vae

python_executable_path = sys.executable
# base portrait sdxl add_text2image add_ipa_base add_ipa_sdxl add_video add_tryon
check_hash = {}
sam_predictor = None


@switch_sd_model_vae()
def easyphoto_tryon_infer_forward(
    sd_model_checkpoint,
    template_image,
    template_mask,
    selected_cloth_images,
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
    dx,
    dy,
    batch_size,
    optimize_angle_and_ratio,
    refine_bound,
    ref_image_selected_tab,
    cloth_uuid,
    max_train_steps,
):
    global check_hash, sam_predictor
    # change system ckpt if not match
    reload_sd_model_vae(sd_model_checkpoint, "vae-ft-mse-840000-ema-pruned.ckpt")

    check_files_exists_and_download(check_hash.get("base", True), "base")
    check_files_exists_and_download(check_hash.get("add_tryon", True), "add_tryon")
    check_hash["base"] = False
    check_hash["add_tryon"] = False

    checkpoint_type = get_checkpoint_type(sd_model_checkpoint)
    if checkpoint_type == 2 or checkpoint_type == 3:
        info = "Tryon does not support the SD2 / SDXL checkpoint."
        ep_logger.error(info)
        return info, [], template_mask, reference_mask

    cloth_gallery_dir = os.path.join(tryon_gallery_dir, "cloth")
    gallery_lists = glob(os.path.join(cloth_gallery_dir, "*.jpg")) + glob(os.path.join(cloth_gallery_dir, "*.png"))
    user_ids = [os.path.basename(i).split(".")[0] for i in gallery_lists]
    ep_logger.info(f"user_ids: {user_ids}")

    # Template Input
    if template_image is None:
        info = "Please upload a template image."
        ep_logger.error(info)
        return info, [], template_mask, reference_mask

    if template_mask is None and template_image["mask"].max() == 0:
        info = "Please give a hint of template, or upload template mask by clicking show mask."
        ep_logger.error(info)
        return info, [], template_mask, reference_mask

    if template_mask is not None and template_mask.shape != template_image["image"].shape:
        info = "Please upload a mask with the same size as template. Or remove the uploaded mask and generate automatically by given hints"
        ep_logger.error(info)
        return info, [], template_mask, reference_mask

    # Reference: choose tabs select
    if ref_image_selected_tab == 0:
        # choose from template
        try:
            input_ref_img_path = eval(selected_cloth_images)[0]
            cloth_uuid = os.path.basename(input_ref_img_path).split(".")[0]

            # clean previous reference mask result
            reference_mask = None
        except Exception:
            ep_logger.info(f"selected_cloth_images: {selected_cloth_images}")
            info = "Please choose a cloth image from gallery."
            return info, [], template_mask, reference_mask
    else:
        # use uploaded reference
        if cloth_uuid == "" or cloth_uuid is None:
            info = "The user id cannot be empty."
            ep_logger.error(info)
            return info, [], template_mask, reference_mask

        cloth_uuid = cloth_uuid + "_" + str(max_train_steps)

        if cloth_uuid in user_ids:
            info = "The user id cannot be repeat. Please check in the cloth gallery. Or use a new id to mark the new cloth"
            ep_logger.error(info)
            return info, [], template_mask, reference_mask

        if reference_mask is None and reference_image["mask"].max() == 0:
            info = "Please give a hint of reference, or upload reference mask by clicking show mask."
            ep_logger.error(info)
            return info, [], template_mask, reference_mask

        if reference_mask is not None and reference_mask.shape != reference_image["image"].shape:
            info = "Please upload a mask with the same size as reference. Or remove the uploaded mask and generate automatically by given hints"
            ep_logger.error(info)
            return info, [], template_mask, reference_mask

    ep_logger.info(f"cloth user id: {cloth_uuid}")

    return_msg = ""

    webui_save_path = os.path.join(models_path, f"Lora/{cloth_uuid}.safetensors")
    if os.path.exists(webui_save_path):
        return_msg += f"Use exists LoRA of {cloth_uuid}.\n"
        ep_logger.info(f"LoRA of user id: {cloth_uuid} exists. Start Infer.")
    else:

        def has_duplicate_uid(input_user_id, existing_user_ids):
            input_uid = input_user_id.split("_")[1]
            existing_uids = [uid.split("_")[1] for uid in existing_user_ids]
            return input_uid in existing_uids

        if has_duplicate_uid(cloth_uuid, user_ids):
            uuid = cloth_uuid.split("_")[1]
            return_msg += f"Update LoRA of {uuid} with {max_train_steps} steps.\n"
        else:
            return_msg += f"Train a new LoRA of {cloth_uuid} of {max_train_steps} steps.\n"

        ep_logger.info("No exists LoRA model found. Start Training")

        # ref image copy
        ref_image_path = os.path.join(cloth_id_outpath_samples, cloth_uuid, "ref_image.jpg")

        # Training data retention
        user_path = os.path.join(cloth_id_outpath_samples, cloth_uuid, "processed_images")
        images_save_path = os.path.join(cloth_id_outpath_samples, cloth_uuid, "processed_images", "train")
        json_save_path = os.path.join(cloth_id_outpath_samples, cloth_uuid, "processed_images", "metadata.jsonl")

        # Training weight saving
        weights_save_path = os.path.join(cloth_id_outpath_samples, cloth_uuid, "user_weights")
        webui_load_path = os.path.join(models_path, f"Stable-diffusion", sd_model_checkpoint)
        sd15_save_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"),
            "stable-diffusion-v1-5",
        )

        os.makedirs(user_path, exist_ok=True)
        os.makedirs(images_save_path, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(webui_save_path)), exist_ok=True)

        if reference_mask is None:
            _, reference_mask = easyphoto_tryon_mask_forward(reference_image, "Reference")

        prepare_tryon_train_data(reference_image, reference_mask, ref_image_path, images_save_path, json_save_path, validation_tryon_prompt)

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

        ep_logger.info(f"Delete sam model before training to save CUDA memory.")
        del sam_predictor
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

        # save to gallery
        os.makedirs(cloth_gallery_dir, exist_ok=True)
        Image.open(ref_image_path).save(os.path.join(cloth_gallery_dir, f"{cloth_uuid}.jpg"))
        # save to models/LoRA
        copyfile(best_weight_path, webui_save_path)

    # infer
    # get random seed
    if int(seed) == -1:
        seed = np.random.randint(0, 65536)

    # prompt init
    input_prompt = f"{validation_tryon_prompt}, <lora:{cloth_uuid}:{lora_weight}>" + additional_prompt
    ep_logger.info(f"input_prompt: {input_prompt}")

    # reference image
    if cloth_uuid in DEFAULT_CLOTH_LORA:
        img_ref = np.uint8(Image.open(os.path.join(cloth_gallery_dir, cloth_uuid + ".jpg")))
        mask_ref = np.uint8(Image.open(os.path.join(cloth_gallery_dir, cloth_uuid + "_mask.jpg")))
    else:
        img_ref = np.uint8(Image.open(os.path.join(cloth_id_outpath_samples, cloth_uuid, "ref_image.jpg")))
        mask_ref = np.uint8(Image.open(os.path.join(cloth_id_outpath_samples, cloth_uuid, "ref_image_mask.jpg")))

    if len(mask_ref.shape) == 2:
        mask_ref = np.repeat(mask_ref[:, :, np.newaxis], 3, axis=2)

    # template image
    img_template = np.uint8(Image.fromarray(np.uint8(template_image["image"])))
    mask_template_input = np.uint8(Image.fromarray(np.uint8(template_image["mask"])))
    if template_mask is None:
        # refine
        _, mask_template = easyphoto_tryon_mask_forward(template_image, "Template")
        # update for return
        template_mask = mask_template
    else:
        mask_template = template_mask[:, :, 0]

    # for final paste
    _, box_main = mask_to_box(np.uint8(mask_ref[:, :, 0]))
    _, box_template = mask_to_box(mask_template)
    template_copy = copy.deepcopy(img_template)
    copy.deepcopy(mask_template)

    # crop to get local img
    W, H = np.array(img_template).shape[1], np.array(img_template).shape[0]
    expand_ratio = 1.2
    img_ref = crop_image(np.array(img_ref), box_main, expand_ratio=expand_ratio)
    mask_ref = crop_image(np.array(mask_ref), box_main, expand_ratio=expand_ratio)
    img_template = crop_image(np.array(img_template), box_template, expand_ratio=expand_ratio)
    mask_template = crop_image(np.array(mask_template), box_template, expand_ratio=expand_ratio)

    box_template = expand_roi(box_template, ratio=expand_ratio, max_box=[0, 0, W, H])

    # Step2: prepare background image for paste
    # main background with most frequent color
    color = get_background_color(img_ref, mask_ref[:, :, 0])
    color_img = np.full((img_template.shape[0], img_template.shape[1], 3), color, dtype=np.uint8)
    background_img = apply_mask_to_image(color_img, img_template, mask_template)

    # Step3: optimize match and paste
    if azimuth != 0:
        return_msg += "Please refer to the anyid branch to use zero123 for a 3d rotation. (Set to 0 here.)\n"
        azimuth = 0

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

    # Step5: generation
    return_res = []
    for i in range(batch_size):
        ep_logger.info("Start First diffusion.")
        # inpaint the main region
        controlnet_pairs = [
            ["canny_no_pre", res_canny, 1.0, 0],
            ["depth", resize_img_template, 1.0, 0],
            ["color", resize_image_input, 0.5, 0],
        ]

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
        result_img = result_img[0]

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
        result_img = result_img[0]

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
            result_img = result_img[0]

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

        return_res.append(first_paste)
        torch.cuda.empty_cache()

        ep_logger.info("Finished")

    return "Success\n" + return_msg, return_res, template_mask, reference_mask


def easyphoto_tryon_mask_forward(input_image, img_type):
    global check_hash, sam_predictor

    check_files_exists_and_download(check_hash.get("add_tryon", True), "add_tryon")
    check_hash["add_tryon"] = False

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

    return "Show Mask Success", mask
