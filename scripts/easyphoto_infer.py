import copy
import glob
import logging
import os
import random
import sys

import gradio as gr
import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
from scripts.face_process_utils import call_face_crop, crop_and_paste, color_transfer
from scripts.easyphoto_config import user_id_outpath_samples, easyphoto_outpath_samples, validation_prompt, DEFAULT_POSITIVE, DEFAULT_NEGATIVE, easyphoto_img2img_samples
from scripts.sdwebui import ControlNetUnit, i2i_inpaint_call
from scripts.swapper import UpscaleOptions, swap_face

from modules.images import save_image
from modules.shared import opts, state

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 设置日志记录器的级别
log_level = os.environ.get('LOG_LEVEL', 'INFO')
logging.getLogger().setLevel(log_level)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s')  

def resize_image(input_image, resolution, nearest = False, crop264 = True):
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

def inpaint_with_mask_face(
    input_image: Image.Image,
    select_mask_input: Image.Image,
    replaced_input_image: Image.Image,
    diffusion_steps = 50,
    denoising_strength = 0.45,
    input_prompt = '1girl',
    hr_scale: float = 1.0,
    default_positive_prompt = DEFAULT_POSITIVE,
    default_negative_prompt = DEFAULT_NEGATIVE,
    seed: int = 123456,
    sd_model_checkpoint = "ChilloutMix-ni-fp16.safetensors",
):
    assert input_image is not None, f'input_image must not be none'
    controlnet_units_list = []
    w = int(input_image.width)
    h = int(input_image.height)

    control_weight_canny = 1.0
    canny_weight = 0.50
    if 1:
        controlnet_units_list.append(
            ControlNetUnit(
                input_image=input_image, module='canny',
                weight=0.50,
                guidance_end=1,
                resize_mode='Just Resize',
                threshold_a=100,
                threshold_b=200,
                model='control_v11p_sd15_canny [d14c016b]'
            )
        )
    if 1:
        controlnet_units_list.append(
            ControlNetUnit(
                input_image=replaced_input_image, module='openpose_full',
                weight=control_weight_canny - canny_weight,
                guidance_end=1,
                resize_mode='Just Resize',
                model='control_v11p_sd15_openpose [cab727d4]'
            )
        )

    if 1:
        blur_ratio      = 24
        h, w, c         = np.shape(input_image)
        color_image     = np.array(input_image, np.uint8)

        color_image     = resize_image(color_image, 1024)
        now_h, now_w    = color_image.shape[:2]

        color_image = cv2.resize(color_image, (int(now_w//blur_ratio), int(now_h//blur_ratio)), interpolation=cv2.INTER_CUBIC)  
        color_image = cv2.resize(color_image, (now_w, now_h), interpolation=cv2.INTER_NEAREST)
        color_image = cv2.resize(color_image, (w, h), interpolation=cv2.INTER_CUBIC)
        color_image = Image.fromarray(np.uint8(color_image))

        control_unit_canny = ControlNetUnit(input_image=color_image, module='none',
                                            weight=0.85,
                                            guidance_end=1,
                                            resize_mode='Just Resize',
                                            model='control_sd15_random_color')
        controlnet_units_list.append(control_unit_canny)

    positive = f'{input_prompt}, {default_positive_prompt}'
    negative = f'{default_negative_prompt}'

    image = i2i_inpaint_call(
        images=[input_image],
        mask_image=select_mask_input,
        inpainting_fill=1, 
        steps=diffusion_steps,
        denoising_strength=denoising_strength,
        cfg_scale=7,
        inpainting_mask_invert=0,
        width=int(w*hr_scale),
        height=int(h*hr_scale),
        inpaint_full_res=False,
        seed=seed,
        prompt=positive,
        negative_prompt=negative,
        controlnet_units=controlnet_units_list,
        sd_model_checkpoint=sd_model_checkpoint,
        outpath_samples=easyphoto_img2img_samples,
    )

    return image

def inpaint_only(        
    input_image: Image.Image,
    input_mask: Image.Image,
    input_prompt = '1girl',
    fusion_image = None,
    diffusion_steps = 50,
    denoising_strength = 0.2, 
    hr_scale: float = 1.0,
    default_positive_prompt = DEFAULT_POSITIVE,
    default_negative_prompt = DEFAULT_NEGATIVE,
    seed: int = 123456,
    sd_model_checkpoint = "ChilloutMix-ni-fp16.safetensors",
):
    assert input_image is not None, f'input_image must not be none'
    controlnet_units_list = []
    w = int(input_image.width)
    h = int(input_image.height)

    if 1:
        controlnet_units_list.append(
            ControlNetUnit(
                    input_image=input_image if fusion_image is None else fusion_image, module='canny',
                    weight=1,
                    guidance_end=1,
                    resize_mode='Just Resize',
                    threshold_a=100,
                    threshold_b=200,
                    model='control_v11p_sd15_canny [d14c016b]'
            )
        )
    if 1:
        controlnet_units_list.append(
            ControlNetUnit(
                input_image=input_image if fusion_image is None else fusion_image, module='tile_resample',
                weight=1,
                guidance_end=1,
                resize_mode='Just Resize',
                threshold_a=1,
                threshold_b=200,
                model='control_v11f1e_sd15_tile [a371b31b]'
            )
        )

    positive = f'{input_prompt}, {default_positive_prompt}'
    negative = f'{default_negative_prompt}'

    image = i2i_inpaint_call(
        images=[input_image], 
        mask_image=input_mask, 
        inpainting_fill=1, 
        steps=diffusion_steps,
        denoising_strength=denoising_strength, 
        inpainting_mask_invert=0, 
        width=int(hr_scale * w), 
        height=int(hr_scale * h), 
        inpaint_full_res=False, 
        seed=seed, 
        prompt=positive, 
        negative_prompt=negative, 
        controlnet_units=controlnet_units_list, 
        sd_model_checkpoint=sd_model_checkpoint, 
        outpath_samples=easyphoto_img2img_samples,
    )
    return image

def easyphoto_infer_forward(selected_template_images, init_image, additional_prompt, after_face_fusion_ratio, first_diffusion_steps, first_denoising_strength, second_diffusion_steps, second_denoising_strength, seed, crop_face_preprocess, apply_face_fusion_before, apply_face_fusion_after, color_shift_middle, color_shift_last, tabs, *user_ids): 
    # create modelscope model
    retinaface_detection    = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
    image_face_fusion       = pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo')
    skin_retouching         = pipeline('skin-retouching-torch', model='damo/cv_unet_skin_retouching_torch', model_revision='v1.0.1')

    if int(seed) == -1:
        seed = np.random.randint(0, 65536)

    if tabs == 0:
        template_images = eval(selected_template_images)
    else:
        template_images = [init_image]
    
    _user_ids = []
    for user_id in user_ids:
        if user_id != "none":
            _user_ids.append(user_id)
    user_ids = _user_ids
    if len(user_ids) == 0:
        return "Please choose a user id.", []

    if len(user_ids) == 1:
        user_id                 = user_ids[0]
        # get prompt
        input_prompt            = f"{validation_prompt}, <lora:{user_id}:0.9>" + "<lora:FilmVelvia3:0.65>" + additional_prompt
        
        # get best image after training
        best_outputs_paths = glob.glob(os.path.join(user_id_outpath_samples, user_id, "user_weights", "best_outputs", "*.jpg"))
        if len(best_outputs_paths) > 0:
            face_id_image_path  = best_outputs_paths[0]
        else:
            face_id_image_path  = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg") 
        roop_image_path         = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")

        face_id_image           = Image.open(face_id_image_path).convert("RGB")
        roop_image              = Image.open(roop_image_path).convert("RGB")

        # Crop user images to obtain portrait boxes, facial keypoints, and masks
        roop_face_retinaface_box, roop_face_retinaface_keypoints, roop_face_retinaface_mask = call_face_crop(retinaface_detection, face_id_image, 1.5, "roop")
    else:
        input_prompt_without_lora       = f"{validation_prompt}" + additional_prompt

        input_prompts                   = []
        face_id_images                  = []
        roop_images                     = []
        roop_face_retinaface_boxs       = []
        roop_face_retinaface_keypoints  = []
        roop_face_retinaface_masks      = []
        
        for user_id in user_ids:
            # get prompt
            input_prompt            = f"{validation_prompt}, <lora:{user_id}:0.9>" + "<lora:FilmVelvia3:0.65>" + additional_prompt
            
            # get best image after training
            best_outputs_paths = glob.glob(os.path.join(user_id_outpath_samples, user_id, "user_weights", "best_outputs", "*.jpg"))
            if len(best_outputs_paths) > 0:
                face_id_image_path  = best_outputs_paths[0]
            else:
                face_id_image_path  = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg") 
            roop_image_path         = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")

            face_id_image           = Image.open(face_id_image_path).convert("RGB")
            roop_image              = Image.open(roop_image_path).convert("RGB")

            # Crop user images to obtain portrait boxes, facial keypoints, and masks
            roop_face_retinaface_box, roop_face_retinaface_keypoint, roop_face_retinaface_mask = call_face_crop(retinaface_detection, face_id_image, 1.5, "roop")

            input_prompts.append(input_prompt)
            face_id_images.append(face_id_image)
            roop_images.append(roop_image)
            roop_face_retinaface_boxs.append(roop_face_retinaface_box)
            roop_face_retinaface_keypoints.append(roop_face_retinaface_keypoint)
            roop_face_retinaface_masks.append(roop_face_retinaface_mask)

    outputs                 = []
    for template_image in template_images:
        # open the template image
        if tabs == 0:
            template_image = Image.open(template_image).convert("RGB")
        else:
            template_image = Image.fromarray(template_image)

        if len(user_ids) == 1:
            # Crop the template image to retain only the portion of the portrait
            if crop_face_preprocess:
                crop_safe_box, _, _ = call_face_crop(retinaface_detection, template_image, 3, "crop")
                input_image = copy.deepcopy(template_image).crop(crop_safe_box)
            else:
                input_image = copy.deepcopy(template_image)

            # Resize the template image with short edges on 512
            short_side  = min(input_image.width, input_image.height)
            resize      = float(short_side / 512.0)
            new_size    = (int(input_image.width//resize), int(input_image.height//resize))
            input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
            if crop_face_preprocess:
                new_width   = int(np.shape(input_image)[1] // 32 * 32)
                new_height  = int(np.shape(input_image)[0] // 32 * 32)
                input_image = input_image.resize([new_width, new_height], Image.Resampling.LANCZOS)
            
            # Detect the box where the face of the template image is located and obtain its corresponding small mask
            retinaface_box, retinaface_keypoints, input_mask = call_face_crop(retinaface_detection, input_image, 1.1, "template")
            origin_input_mask = copy.deepcopy(input_mask)
            
            # backup input template
            original_input_template = copy.deepcopy(input_image)

            # Paste user images onto template images
            replaced_input_image = crop_and_paste(face_id_image, roop_face_retinaface_mask, input_image, roop_face_retinaface_keypoints, retinaface_keypoints, roop_face_retinaface_box)
            replaced_input_image = Image.fromarray(np.uint8(replaced_input_image))
            
            # Fusion of user reference images and input images as canny input
            if roop_image is not None and apply_face_fusion_before:
                input_image = image_face_fusion(dict(template=input_image, user=roop_image))[OutputKeys.OUTPUT_IMG]# swap_face(target_img=input_image, source_img=roop_image, model="inswapper_128.onnx", upscale_options=UpscaleOptions())
                input_image = Image.fromarray(np.uint8(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)))

            # Expand the template image in the x-axis direction to include the ears.
            h, w, c             = np.shape(input_mask)
            retinaface_box      = np.int32(retinaface_box)
            face_width          = retinaface_box[2] - retinaface_box[0]
            retinaface_box[0]   = np.clip(np.array(retinaface_box[0], np.int32) - face_width * 0.15, 0, w - 1)
            retinaface_box[2]   = np.clip(np.array(retinaface_box[2], np.int32) + face_width * 0.15, 0, w - 1)
            input_mask          = np.zeros_like(np.array(input_mask, np.uint8))
            input_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = 255
            input_mask          = Image.fromarray(np.uint8(input_mask))
            
            # here we get the retinaface_box, we should use this Input box and face pixel to refine the output face pixel colors
            template_image_original_face_area = np.array(original_input_template)[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2],:] 
            
            # First diffusion, facial reconstruction
            output_image = inpaint_with_mask_face(input_image, input_mask, replaced_input_image, diffusion_steps=first_diffusion_steps, denoising_strength=first_denoising_strength, input_prompt=input_prompt, hr_scale=1.0, seed=str(seed))

            # Obtain the mask of the area around the face
            input_mask  = Image.fromarray(np.uint8(cv2.dilate(np.array(origin_input_mask), np.ones((96, 96), np.uint8), iterations=1) - cv2.erode(np.array(origin_input_mask), np.ones((48, 48), np.uint8), iterations=1)))

            # Before second diffusion, we trans the image's face color according to template_image_original_face_area
            # use original template face area to shift generated face color in the middle
            if color_shift_middle:
                output_image_face_area = np.array(copy.deepcopy(output_image))[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2],:] 
                # color_transfer(target_to_trans, shift_reference)
                output_image_face_area = color_transfer(output_image_face_area,template_image_original_face_area)
                output_image = np.array(output_image)
                output_image[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2],:] = output_image_face_area
                output_image = Image.fromarray(output_image)

            # Second diffusion
            default_hr_scale = 1.5
            if roop_image is not None and apply_face_fusion_after:
                # Fusion of facial photos with user photos
                fusion_image = image_face_fusion(dict(template=output_image, user=roop_image))[OutputKeys.OUTPUT_IMG] # swap_face(target_img=output_image, source_img=roop_image, model="inswapper_128.onnx", upscale_options=UpscaleOptions())
                fusion_image = Image.fromarray(cv2.cvtColor(fusion_image, cv2.COLOR_BGR2RGB))
                output_image = Image.fromarray(np.uint8((np.array(output_image, np.float32) * (1 - after_face_fusion_ratio) + np.array(fusion_image, np.float32) * after_face_fusion_ratio)))
                generate_image = inpaint_only(output_image, input_mask, input_prompt, diffusion_steps=second_diffusion_steps, denoising_strength=second_denoising_strength, fusion_image=fusion_image, hr_scale=default_hr_scale)
            else:
                generate_image = inpaint_only(output_image, input_mask, input_prompt, diffusion_steps=second_diffusion_steps, denoising_strength=second_denoising_strength, hr_scale=default_hr_scale)

            # use original template face area to shift generated face color at last
            if color_shift_last:
                rescale_retinaface_box = [int(i * default_hr_scale) for i in retinaface_box]
                output_image_face_area = np.array(copy.deepcopy(generate_image))[rescale_retinaface_box[1]:rescale_retinaface_box[3], rescale_retinaface_box[0]:rescale_retinaface_box[2],:] 
                output_image_face_area = color_transfer(output_image_face_area,template_image_original_face_area)
                generate_image = np.array(generate_image)
                generate_image[rescale_retinaface_box[1]:rescale_retinaface_box[3], rescale_retinaface_box[0]:rescale_retinaface_box[2],:] = output_image_face_area
                generate_image = Image.fromarray(generate_image)


            # If it is a large template for cutting, paste the reconstructed image back
            if crop_face_preprocess:
                origin_image    = np.array(copy.deepcopy(template_image))
                x1,y1,x2,y2     = crop_safe_box
                generate_image  = generate_image.resize([x2-x1, y2-y1], Image.Resampling.LANCZOS)
                origin_image[y1:y2,x1:x2] = np.array(generate_image) 
                origin_image    = Image.fromarray(np.uint8(origin_image))
            else:
                origin_image    = generate_image

        else:
            crop_safe_box, _, _ = call_face_crop(retinaface_detection, template_image, 1.10, "crop")

            output_image = copy.deepcopy(template_image)
            output_image = np.array(output_image)
            output_mask = np.ones_like(output_image)
            for index in range(len(crop_safe_box)):
                retinaface_box = crop_safe_box[index]
                output_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = 255
            output_mask  = Image.fromarray(np.uint8(cv2.dilate(np.array(output_mask), np.ones((64, 64), np.uint8), iterations=1) - cv2.erode(np.array(output_mask), np.ones((32, 32), np.uint8), iterations=1)))

            for index in range(len(crop_safe_box)):
                _template_image = copy.deepcopy(template_image)
                _template_image = np.array(_template_image)

                for sub_index in range(len(crop_safe_box)):
                    if index != sub_index:
                        retinaface_box = crop_safe_box[sub_index]
                        _template_image[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = 255
                _template_image = Image.fromarray(np.uint8(_template_image))

                # Crop the template image to retain only the portion of the portrait
                if crop_face_preprocess:
                    _crop_safe_box, _, _ = call_face_crop(retinaface_detection, _template_image, 2, "crop")
                    input_image = copy.deepcopy(_template_image).crop(_crop_safe_box)
                else:
                    input_image = copy.deepcopy(_template_image)

                # backup input template
                original_input_template = copy.deepcopy(input_image)

                # Resize the template image with short edges on 512
                short_side  = min(input_image.width, input_image.height)
                resize      = float(short_side / 512.0)
                new_size    = (int(input_image.width//resize), int(input_image.height//resize))
                input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
                if crop_face_preprocess:
                    new_width   = int(np.shape(input_image)[1] // 32 * 32)
                    new_height  = int(np.shape(input_image)[0] // 32 * 32)
                    input_image = input_image.resize([new_width, new_height], Image.Resampling.LANCZOS)
                
                # Detect the box where the face of the template image is located and obtain its corresponding small mask
                retinaface_box, retinaface_keypoints, input_mask = call_face_crop(retinaface_detection, input_image, 1.1, "template")
                origin_input_mask = copy.deepcopy(input_mask)

                # Paste user images onto template images
                replaced_input_image = crop_and_paste(face_id_images[index], roop_face_retinaface_masks[index], input_image, roop_face_retinaface_keypoints[index], retinaface_keypoints, roop_face_retinaface_boxs[index])
                replaced_input_image = Image.fromarray(np.uint8(replaced_input_image))
                
                # Fusion of user reference images and input images as canny input
                if roop_images[index] is not None and apply_face_fusion_before:
                    input_image = image_face_fusion(dict(template=input_image, user=roop_images[index]))[OutputKeys.OUTPUT_IMG]# swap_face(target_img=input_image, source_img=roop_image, model="inswapper_128.onnx", upscale_options=UpscaleOptions())
                    input_image = Image.fromarray(np.uint8(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)))

                # Expand the template image in the x-axis direction to include the ears.
                h, w, c             = np.shape(input_mask)
                retinaface_box      = np.int32(retinaface_box)
                face_width          = retinaface_box[2] - retinaface_box[0]
                retinaface_box[0]   = np.clip(np.array(retinaface_box[0], np.int32) - face_width * 0.15, 0, w - 1)
                retinaface_box[2]   = np.clip(np.array(retinaface_box[2], np.int32) + face_width * 0.15, 0, w - 1)
                input_mask          = np.zeros_like(np.array(input_mask, np.uint8))
                input_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = 255
                input_mask          = Image.fromarray(np.uint8(input_mask))
                
                # here we get the retinaface_box, we should use this Input box and face pixel to refine the output face pixel colors
                template_image_original_face_area = np.array(original_input_template)[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2],:] 
                
                # First diffusion, facial reconstruction
                _output_image = inpaint_with_mask_face(input_image, input_mask, replaced_input_image, diffusion_steps=first_diffusion_steps, denoising_strength=first_denoising_strength, input_prompt=input_prompts[index], hr_scale=1.0, seed=str(seed))

                if color_shift_middle:
                    _output_image_face_area = np.array(copy.deepcopy(_output_image))[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2],:] 
                    # color_transfer(target_to_trans, shift_reference)
                    _output_image_face_area = color_transfer(_output_image_face_area,template_image_original_face_area)
                    _output_image = np.array(_output_image)
                    _output_image[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2],:] = _output_image_face_area
                    _output_image = Image.fromarray(_output_image)
                # Obtain the mask of the area around the face
                input_mask  = Image.fromarray(np.uint8(cv2.dilate(np.array(origin_input_mask), np.ones((96, 96), np.uint8), iterations=1) - cv2.erode(np.array(origin_input_mask), np.ones((48, 48), np.uint8), iterations=1)))

                # Second diffusion
                default_hr_scale = 1.5
                # Second diffusion
                if roop_images[index] is not None and apply_face_fusion_after:
                    # Fusion of facial photos with user photos
                    fusion_image = image_face_fusion(dict(template=_output_image, user=roop_images[index]))[OutputKeys.OUTPUT_IMG] # swap_face(target_img=output_image, source_img=roop_image, model="inswapper_128.onnx", upscale_options=UpscaleOptions())
                    fusion_image = Image.fromarray(cv2.cvtColor(fusion_image, cv2.COLOR_BGR2RGB))
                    _output_image = Image.fromarray(np.uint8((np.array(_output_image, np.float32) * (1 - after_face_fusion_ratio) + np.array(fusion_image, np.float32) * after_face_fusion_ratio)))

                    generate_image = inpaint_only(_output_image, input_mask, input_prompts[index], diffusion_steps=second_diffusion_steps, denoising_strength=second_denoising_strength, fusion_image=fusion_image, hr_scale=default_hr_scale)
                else:
                    generate_image = inpaint_only(_output_image, input_mask, input_prompts[index], diffusion_steps=second_diffusion_steps, denoising_strength=second_denoising_strength, hr_scale=default_hr_scale)

                # use original template face area to shift generated face color at last
                if color_shift_last:
                    rescale_retinaface_box = [int(i * default_hr_scale) for i in retinaface_box]
                    output_image_face_area = np.array(copy.deepcopy(generate_image))[rescale_retinaface_box[1]:rescale_retinaface_box[3], rescale_retinaface_box[0]:rescale_retinaface_box[2],:] 
                    output_image_face_area = color_transfer(output_image_face_area,template_image_original_face_area)
                    generate_image = np.array(generate_image)
                    generate_image[rescale_retinaface_box[1]:rescale_retinaface_box[3], rescale_retinaface_box[0]:rescale_retinaface_box[2],:] = output_image_face_area
                    generate_image = Image.fromarray(generate_image)
                    
                # If it is a large template for cutting, paste the reconstructed image back
                if crop_face_preprocess:
                    origin_image    = np.array(copy.deepcopy(_template_image))

                    x1,y1,x2,y2     = _crop_safe_box
                    generate_image  = generate_image.resize([x2-x1, y2-y1], Image.Resampling.LANCZOS)
                    origin_image[y1:y2,x1:x2] = np.array(generate_image) 

                    origin_image    = Image.fromarray(np.uint8(origin_image))
                else:
                    origin_image    = generate_image
                
                retinaface_box = crop_safe_box[index]
                output_image[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = np.array(origin_image, np.float32)[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]]
            
            output_image    = Image.fromarray(np.uint8(output_image))
            short_side      = min(output_image.width, output_image.height)
            resize          = float(short_side / 768.0)
            new_size        = (int(output_image.width//resize), int(output_image.height//resize))
            output_image    = output_image.resize(new_size, Image.Resampling.LANCZOS)
            origin_image    = inpaint_only(output_image, output_mask, input_prompt_without_lora, diffusion_steps=20, denoising_strength=0.3, hr_scale=1)

            
        try:
            origin_image    = Image.fromarray(cv2.cvtColor(skin_retouching(origin_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
        except:
            logging.info("Skin Retouching error, but pass.")

        outputs.append(origin_image)

        save_image(origin_image, easyphoto_outpath_samples, "EasyPhoto", None, None, opts.grid_format, info=None, short_filename=not opts.grid_extended_filename, grid=True, p=None)

    return "SUCCESS", outputs