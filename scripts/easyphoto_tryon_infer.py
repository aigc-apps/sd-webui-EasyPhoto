import copy
import os
import platform
import shutil
import subprocess
import sys
import traceback
from glob import glob
from shutil import copyfile

import numpy as np
from PIL import Image, ImageOps

import cv2
import torch
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modules.images import save_image
from modules.paths import models_path
from modules.shared import opts, state
from scripts.easyphoto_config import (CLOTH_LORA_PREFIX, cache_log_file_path,
                                      cloth_id_outpath_samples,
                                      easyphoto_outpath_samples,
                                      validation_tryon_prompt)
from scripts.easyphoto_infer import inpaint
from scripts.easyphoto_tryon_process_utils import (align_and_overlay_images,
                                                   apply_mask_to_image,
                                                   compute_rotation_angle,
                                                   copy_white_mask_to_template,
                                                   crop_image,
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
                                                   seg_by_box)
from scripts.easyphoto_utils import check_tryon_files_exists_and_download
from scripts.sdwebui import get_checkpoint_type, switch_sd_model_vae, reload_sd_model_vae
from segment_anything import SamPredictor, sam_model_registry

python_executable_path = sys.executable
check_hash = True

@switch_sd_model_vae()
def easyphoto_tryon_infer_forward(
    sd_model_checkpoint, template_image, selected_cloth_template_images, input_ref_img_path, additional_prompt, seed, first_diffusion_steps,
    first_denoising_strength, lora_weight, iou_threshold, angle, azimuth, ratio, batch_size, refine_input_mask, optimize_angle_and_ratio, refine_bound,
    pure_image, ref_image_selected_tab, cloth_uuid, max_train_steps
):
    global check_hash
    # change system ckpt if not match
    reload_sd_model_vae(sd_model_checkpoint,"vae-ft-mse-840000-ema-pruned.ckpt")

    # check input
    check_tryon_files_exists_and_download(check_hash)
    check_hash = False

    checkpoint_type = get_checkpoint_type(sd_model_checkpoint)
    if checkpoint_type == 2 or checkpoint_type == 3:
        info = "Tryon does not support the SD2 / SDXL checkpoint."
        print(info)
        return info, [], []

    cloth_gallery_dir = os.path.join(cloth_id_outpath_samples, 'gallery')
    gallery_lists = glob(os.path.join(cloth_gallery_dir, '*.jpg')) + \
        glob(os.path.join(cloth_gallery_dir, '*.png'))
    user_ids = [i.split('/')[-1].split('.')[0] for i in gallery_lists]
    print('user_ids:', user_ids)

    try:
        # choose tabs select
        if ref_image_selected_tab == 0:
            # choose from gallery
            input_ref_img_path = eval(selected_cloth_template_images)[0]
            cloth_uuid = input_ref_img_path.split('/')[-1].split('.')[0]
        elif ref_image_selected_tab == 1:
            if cloth_uuid == '' or cloth_uuid is None:
                info = "The user id cannot be empty."
                print(info)
                return info, [], []
    
            cloth_uuid = CLOTH_LORA_PREFIX+cloth_uuid+'_'+str(max_train_steps)

            if cloth_uuid in user_ids:
                info = "The user id cannot be repeat. Please check in the cloth gallery. Or use a new id to mark the new cloth"
                print(info)
                return info, [], []

    except Exception as e:
        torch.cuda.empty_cache()
        info = "Please choose or upload a reference image."
        print(info)
        return info, [], []

    if template_image is None:
        info = "Please upload a template image."
        print(info)
        return info, [], []

    print(f'cloth user id: {cloth_uuid}')

    return_msg = ''

    webui_save_path = os.path.join(
        models_path, f"Lora/{cloth_uuid}.safetensors")

    if os.path.exists(webui_save_path):
        return_msg += f'Use exists LoRA of {cloth_uuid}.\n'
        print(f'LoRA of user id: {cloth_uuid} exists. Start Infer.')
    else:
        def has_duplicate_uid(input_user_id, existing_user_ids):
            input_uid = input_user_id.split('_')[1]
            existing_uids = [uid.split('_')[1] for uid in existing_user_ids]
            return input_uid in existing_uids

        if has_duplicate_uid(cloth_uuid, user_ids):
            uuid = cloth_uuid.split('_')[1]
            return_msg += f'Update LoRA of {uuid} with {max_train_steps} steps.\n'
        else:
            return_msg += f'Train a new LoRA of {cloth_uuid} of {max_train_steps} steps.\n'

        print('Start Training')
        # ref image copy
        ref_image_path = os.path.join(
            cloth_id_outpath_samples, cloth_uuid, "ref_image.jpg")

        # Training data retention
        user_path = os.path.join(
            cloth_id_outpath_samples, cloth_uuid, "processed_images")
        images_save_path = os.path.join(
            cloth_id_outpath_samples, cloth_uuid, "processed_images", "train")
        json_save_path = os.path.join(
            cloth_id_outpath_samples, cloth_uuid, "processed_images", "metadata.jsonl")

        # Training weight saving
        weights_save_path = os.path.join(
            cloth_id_outpath_samples, cloth_uuid, "user_weights")
        webui_load_path = os.path.join(
            models_path, f"Stable-diffusion", sd_model_checkpoint)
        sd15_save_path = os.path.join(os.path.abspath(os.path.dirname(
            __file__)).replace("scripts", "models"), "stable-diffusion-v1-5")

        os.makedirs(user_path, exist_ok=True)
        os.makedirs(images_save_path, exist_ok=True)
        os.makedirs(os.path.dirname(
            os.path.abspath(webui_save_path)), exist_ok=True)

        shutil.copy(input_ref_img_path, ref_image_path)
        prepare_tryon_train_data(
            ref_image_path, images_save_path, json_save_path, validation_tryon_prompt)

        # start train
        # check preprocess results
        train_images = glob(os.path.join(images_save_path, "*.jpg"))
        if len(train_images) == 0:
            return "Failed to obtain preprocessed images, please check the preprocessing process"
        if not os.path.exists(json_save_path):
            return "Failed to obtain preprocessed metadata.jsonl, please check the preprocessing process."

        train_kohya_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "train_kohya/train_lora.py")
        print("train_file_path : ", train_kohya_path)

        # extensions/sd-webui-EasyPhoto/train_kohya_log.txt, use to cache log and flush to UI
        print("cache_log_file_path:", cache_log_file_path)
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

        if platform.system() == 'Windows':
            pwd = os.getcwd()
            dataloader_num_workers = 0  # for solve multi process bug

            command = [
                f'{python_executable_path}', '-m', 'accelerate.commands.launch', '--mixed_precision=fp16', "--main_process_port=3456", f'{train_kohya_path}',
                f'--pretrained_model_name_or_path={os.path.relpath(sd15_save_path, pwd)}',
                f'--pretrained_model_ckpt={os.path.relpath(webui_load_path, pwd)}',
                f'--train_data_dir={os.path.relpath(user_path, pwd)}',
                '--caption_column=text',
                f'--resolution={resolution}',
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

        best_weight_path = os.path.join(
            weights_save_path, f"pytorch_lora_weights.safetensors")
        if not os.path.exists(best_weight_path):
            return "Failed to obtain Lora after training, please check the training process.", []

        # save to gallery
        Image.open(input_ref_img_path).save(
            os.path.join(cloth_gallery_dir, f'{cloth_uuid}.jpg'))
        # save to models/LoRA
        copyfile(best_weight_path, webui_save_path)

    # infer
    # get random seed
    if int(seed) == -1:
        seed = np.random.randint(0, 65536)

    # path init
    model_path = os.path.join(
        models_path, f"Stable-diffusion", sd_model_checkpoint)
    sd_base15_checkpoint = os.path.join(
        os.path.abspath(os.path.dirname(__file__)
                        ).replace("scripts", "models"),
        "stable-diffusion-v1-5",
    )
    sam_checkpoint = os.path.join(
        os.path.abspath(os.path.dirname(__file__)
                        ).replace("scripts", "models"),
        "sam_vit_l_0b3195.pth",
    )

    # prompt init
    input_prompt = f"{validation_tryon_prompt}, <lora:{cloth_uuid}:{lora_weight}>"
    print("input_prompt:", input_prompt)

    # model init
    sam = sam_model_registry["vit_l"]()
    sam.load_state_dict(torch.load(sam_checkpoint))
    predictor = SamPredictor(sam.cuda())

    # Step1: open image and prepare for mask
    # reference image
    img_ref = np.uint8(
        Image.open(os.path.join(cloth_id_outpath_samples,
                   cloth_uuid, "ref_image.jpg"))
    )
    mask_ref = np.uint8(
        Image.open(
            os.path.join(cloth_id_outpath_samples,
                         cloth_uuid, "ref_image_mask.jpg")
        )
    )
    if len(mask_ref.shape) == 2:
        mask_ref = np.repeat(mask_ref[:, :, np.newaxis], 3, axis=2)

    # template image
    img_template = np.uint8(Image.fromarray(np.uint8(template_image["image"])))
    mask_template_input = np.uint8(
        Image.fromarray(np.uint8(template_image["mask"])))

    if mask_template_input.max() == 0:
        print('Please mark the target region on the inference template!')
        return 'Please mark the target region on the inference template!', [], []

    _, box_main = mask_to_box(np.uint8(mask_ref[:, :, 0]))

    if refine_input_mask:
        _, box_template = mask_to_box(mask_template_input[:, :, 0])
        mask_template = np.uint8(seg_by_box(
            np.array(img_template), box_template, predictor))
    else:
        mask_template = mask_template_input[:, :, 0]

    # for final paste
    _, box_template = mask_to_box(mask_template)
    template_copy = copy.deepcopy(img_template)
    mask_copy = copy.deepcopy(mask_template)

    # crop to get local img
    W, H = np.array(img_template).shape[1], np.array(img_template).shape[0]
    expand_ratio = 1.2
    img_ref = crop_image(np.array(img_ref), box_main,
                         expand_ratio=expand_ratio)
    mask_ref = crop_image(np.array(mask_ref), box_main,
                          expand_ratio=expand_ratio)
    img_template = crop_image(np.array(img_template),
                              box_template, expand_ratio=expand_ratio)
    mask_template = crop_image(
        np.array(mask_template), box_template, expand_ratio=expand_ratio)

    box_template = expand_roi(
        box_template, ratio=expand_ratio, max_box=[0, 0, W, H])

    # Step2: prepare background image for paste
    if pure_image:
        # main background with most frequent color
        color = get_background_color(img_ref, mask_ref[:, :, 0])
        color_img = np.full(
            (img_template.shape[0], img_template.shape[1], 3), color, dtype=np.uint8)
        background_img = apply_mask_to_image(
            color_img, img_template, mask_template)
    else:
        # generate background_image with ipa (referenced img_ref)
        controlnet_pairs = [
            ["ipa", img_ref, 2.0],
            ["depth", img_template, 1.0],
        ]

        background_diffusion_steps = 20
        background_denoising_strength = 0.8
        background_img = inpaint(
            Image.fromarray(np.uint8(img_template)),
            Image.fromarray(np.uint8(mask_template)),
            controlnet_pairs,
            diffusion_steps=background_diffusion_steps,
            denoising_strength=background_denoising_strength,
            input_prompt=input_prompt,
            hr_scale=1.0,
            seed=str(seed),
            sd_model_checkpoint=sd_model_checkpoint,
        )

        background_img = background_img.resize(
            (img_template.shape[1], img_template.shape[0]), Image.Resampling.LANCZOS)
        background_img = np.array(background_img)

    # Step3: optimize match and paste
    if azimuth != 0:
        return_msg += 'Please refer to the anyid branch to use zero123 for a 3d rotation. (Set to 0 here.)\n'
        azimuth = 0

    if optimize_angle_and_ratio:
        print('Start optimize angle and ratio!')
        # find optimzal angle and ratio
        # resize mask_ref to same size as mask_template (init ratio as 1)
        resized_mask_ref = resize_and_stretch(
            mask_ref, target_size=(
                mask_template.shape[1], mask_template.shape[0])
        )
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

        print(
            f"target rotation: 1 to 0: {rotation_angle1}, 2 to 0: {rotation_angle2}, final_rotate: {angle_target}")

        # center
        x, y = mask_template.shape[1] // 2, mask_template.shape[0] // 2

        initial_parameters = np.array([angle, ratio])
        max_iters = 100

        angle, ratio = find_best_angle_ratio(
            polygon1,
            polygon2,
            initial_parameters,
            x,
            y,
            angle_target,
            max_iters,
            iou_threshold,
        )

    print(f'Set angle:{angle}, ratio: {ratio}, azimuth: {azimuth}')

    # paste
    result_img, rotate_img_ref, mask_ref, mask_template, iou = align_and_overlay_images(
        np.array(img_ref),
        np.array(background_img),
        np.array(mask_ref),
        np.array(mask_template),
        angle=-angle,
        ratio=ratio,
    )

    return_msg += f'Paste with angle {angle}, ratio: {ratio}, Match IoU: {iou}, optimize: {optimize_angle_and_ratio}. \n See paste result above, if you are not satisfatory with the optimized result, close the optimize_angle_and_ratio and manually set a angle and ratio.\n'

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
    resize_image, res_canny = merge_with_inner_canny(
        np.array(result_img).astype(np.uint8), mask_ref, mask_template
    )

    resize_mask_template, remove_pad = resize_image_with_pad(
        mask_template, resolution=512)
    resize_mask_template = remove_pad(resize_mask_template)

    resize_img_template, remove_pad = resize_image_with_pad(
        img_template, resolution=512)
    resize_img_template = remove_pad(resize_img_template)

    resize_mask_ref, remove_pad = resize_image_with_pad(
        mask_ref, resolution=512)
    resize_mask_ref = remove_pad(resize_mask_ref)

    resize_img_ref, remove_pad = resize_image_with_pad(
        resize_img_ref, resolution=512)
    resize_img_ref = remove_pad(resize_img_ref)

    mask_template = Image.fromarray(
        np.uint8(resize_mask_template))
    resize_image = Image.fromarray(resize_image)
    resize_image_input = copy.deepcopy(resize_image)
    mask_template_input = copy.deepcopy(mask_template)

    # Step5: generation
    return_res = []
    for i in range(batch_size):
        print("Start First diffusion.")

        # control image
        cv2.imwrite('debug_res_canny.jpg',res_canny)
        cv2.imwrite('debug_resize_img_template.jpg',resize_img_template[:,:,::-1])
        resize_image_input.save('debug_resize_image_input.jpg')
        mask_template_input.save('debug_mask_template_input.jpg')

        controlnet_pairs = [
            ["canny_no_pre", res_canny, 1.0, 0],
            ["depth", resize_img_template, 1.0, 0],
        ]

        result_img = inpaint(
            resize_image_input,
            mask_template_input,
            controlnet_pairs,
            diffusion_steps=first_diffusion_steps,
            denoising_strength=first_denoising_strength,
            input_prompt=input_prompt,
            hr_scale=1.0,
            seed=str(seed),
            sd_model_checkpoint=sd_model_checkpoint,
        )

        # resize diffusion results
        target_width = box_template[2] - box_template[0]
        target_height = box_template[3] - box_template[1]
        result_img = result_img.resize((target_width, target_height))
        resize_mask_template = mask_template.resize(
            (target_width, target_height))

        result_img.save('debug_first_inpaint_output.jpg')

        # copy back
        template_copy = np.array(template_copy, np.uint8)

        if len(np.array(np.uint8(resize_mask_template)).shape) == 2:
            init_generation = copy_white_mask_to_template(
                np.array(result_img), np.array(
                    np.uint8(resize_mask_template)), template_copy, box_template
            )
        else:
            init_generation = copy_white_mask_to_template(
                np.array(result_img), np.array(np.uint8(resize_mask_template))[
                    :, :, 0], template_copy, box_template
            )

        return_res.append(Image.fromarray(np.uint8(init_generation)))

        if refine_bound:
            print('Start Refine Boundary.')
            # refine bound
            padding = 30

            box_pad = expand_box_by_pad(
                box_template,
                max_size=(template_copy.shape[1], template_copy.shape[0]),
                padding_size=padding,
            )
            padding_size = abs(np.array(box_pad) - np.array(box_template))

            input_img = init_generation[box_pad[1]: box_pad[3], box_pad[0]: box_pad[2]]
            input_control_img = template_copy[
                box_pad[1]: box_pad[3], box_pad[0]: box_pad[2]
            ]

            if len(np.array(np.uint8(resize_mask_template)).shape) == 2:
                mask_array = np.array(np.uint8(resize_mask_template))
            else:
                mask_array = np.array(np.uint8(resize_mask_template))[:, :, 0]

            input_mask = np.pad(
                mask_array,
                ((padding_size[1], padding_size[3]),
                 (padding_size[0], padding_size[2])),
                mode="constant",
                constant_values=0,
            )

            input_mask_copy = copy.deepcopy(input_mask)

            input_mask = np.uint8(
                cv2.dilate(np.array(input_mask), np.ones(
                    (10, 10), np.uint8), iterations=1)
                - cv2.erode(np.array(input_mask),
                            np.ones((10, 10), np.uint8), iterations=1)
            )

            # generate
            controlnet_pairs = [["canny", input_control_img, 1.0, False, 0]]

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
                np.array(result_img), np.array(
                    np.uint8(input_mask_copy)), template_copy, box_pad
            )

            return_res.append(Image.fromarray(np.uint8(final_generation)))
        else:
            final_generation = init_generation

        save_image(Image.fromarray(np.uint8(final_generation)), easyphoto_outpath_samples, "EasyPhoto", None,
                   None, opts.grid_format, info=None, short_filename=not opts.grid_extended_filename, grid=True, p=None)

        return_res.append(first_paste)
        torch.cuda.empty_cache()

        print('Finished')

    return 'Success\n'+return_msg, return_res
