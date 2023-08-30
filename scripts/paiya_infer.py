import copy
import glob
import logging
import os
import random
import sys

import cv2
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
from scripts.face_process_utils import call_face_crop, crop_and_paste
from scripts.paiya_config import user_id_outpath_samples, validation_prompt, DEFAULT_POSITIVE, DEFAULT_NEGATIVE
from scripts.sdwebui import ControlNetUnit, i2i_inpaint_call
from scripts.swapper import UpscaleOptions, swap_face

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 设置日志记录器的级别
log_level = os.environ.get('LOG_LEVEL', 'INFO')
logging.getLogger().setLevel(log_level)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s')  

def inpaint_with_mask_face(
    input_image: Image.Image,
    select_mask_input: Image.Image,
    replaced_input_image: Image.Image,
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

    positive = f'{input_prompt}, {default_positive_prompt}'
    negative = f'{default_negative_prompt}'

    image = i2i_inpaint_call(
        images=[input_image],
        mask_image=select_mask_input,
        inpainting_fill=1, 
        denoising_strength=0.45,
        cfg_scale=7,
        inpainting_mask_invert=0,
        width=int(w*hr_scale),
        height=int(h*hr_scale),
        inpaint_full_res=False,
        seed=seed,
        steps=50,
        prompt=positive,
        negative_prompt=negative,
        controlnet_units=controlnet_units_list,
        sd_model_checkpoint=sd_model_checkpoint,
    )

    return image

def inpaint_only(        
    input_image: Image.Image,
    input_mask: Image.Image,
    input_prompt = '1girl',
    fusion_image = None,
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
        denoising_strength=0.10, 
        inpainting_mask_invert=0, 
        width=int(hr_scale * w), 
        height=int(hr_scale * h), 
        inpaint_full_res=False, 
        seed=seed, 
        prompt=positive, 
        negative_prompt=negative, 
        controlnet_units=controlnet_units_list, 
        sd_model_checkpoint=sd_model_checkpoint, 
    )
    return image

def paiya_infer_forward(user_id, selected_template_images, append_pos_prompt, final_fusion_ratio, use_fusion_before, use_fusion_after, args): 
    # create modelscope model
    retinaface_detection    = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
    image_face_fusion       = pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo')

    # Whether to perform reconstruction after cropping the image, suit for large image with small people
    crop_face_preprocess    = True
    # random seed TODO: show in ui
    seed                    = random.randint(0, 10000)
    # get prompt
    input_prompt            = f"{validation_prompt}, <lora:{user_id}:1>" + append_pos_prompt
    
    # get best image after training
    best_outputs_paths = glob.glob(os.path.join(user_id_outpath_samples, user_id, "user_weights", "best_outputs", "*.jpg"))
    if len(best_outputs_paths) > 0:
        face_id_image_path  = best_outputs_paths[0]
    else:
        face_id_image_path  = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg") 
    roop_image_path         = os.path.join(user_id_outpath_samples, user_id, "ref_image.jpg")

    face_id_image           = Image.open(face_id_image_path).convert("RGB")
    roop_image              = Image.open(roop_image_path).convert("RGB")

    # get template
    template_images         = eval(selected_template_images)

    # Crop user images to obtain portrait boxes, facial keypoints, and masks
    roop_face_retinaface_box, roop_face_retinaface_keypoints, roop_face_retinaface_mask = call_face_crop(retinaface_detection, face_id_image, 1.5, "roop")

    outputs                 = []
    for template_image in template_images:
        # open the template image
        template_image = Image.open(template_image).convert("RGB")

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

        # Paste user images onto template images
        replaced_input_image = crop_and_paste(face_id_image, roop_face_retinaface_mask, input_image, roop_face_retinaface_keypoints, retinaface_keypoints, roop_face_retinaface_box)
        replaced_input_image = Image.fromarray(np.uint8(replaced_input_image))
        
        # Fusion of user reference images and input images as canny input
        if roop_image is not None and use_fusion_before:
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
        
        # First diffusion, facial reconstruction
        output_image = inpaint_with_mask_face(input_image, input_mask, replaced_input_image, input_prompt=input_prompt, hr_scale=1.0, seed=str(seed))

        # Obtain the mask of the area around the face
        input_mask  = Image.fromarray(np.uint8(cv2.dilate(np.array(input_mask), np.ones((96, 96), np.uint8), iterations=1) - cv2.erode(np.array(input_mask), np.ones((16, 16), np.uint8), iterations=1)))

        # Second diffusion
        if roop_image is not None and use_fusion_after:
            # Fusion of facial photos with user photos
            fusion_image = image_face_fusion(dict(template=output_image, user=roop_image))[OutputKeys.OUTPUT_IMG] # swap_face(target_img=output_image, source_img=roop_image, model="inswapper_128.onnx", upscale_options=UpscaleOptions())
            fusion_image = Image.fromarray(cv2.cvtColor(fusion_image, cv2.COLOR_BGR2RGB))
            output_image = Image.fromarray(np.uint8((np.array(output_image, np.float32) * final_fusion_ratio + np.array(fusion_image, np.float32) * (1 - final_fusion_ratio))))

            generate_image = inpaint_only(output_image, input_mask, input_prompt, fusion_image=fusion_image, hr_scale=1.5)
        else:
            generate_image = inpaint_only(output_image, input_mask, input_prompt, hr_scale=1.5)

        # If it is a large template for cutting, paste the reconstructed image back
        if crop_face_preprocess:
            origin_image    = np.array(copy.deepcopy(template_image))
            x1,y1,x2,y2     = crop_safe_box
            generate_image  = generate_image.resize([x2-x1, y2-y1], Image.Resampling.LANCZOS)
            origin_image[y1:y2,x1:x2] = np.array(generate_image)
            origin_image    = Image.fromarray(np.uint8(origin_image))
        else:
            origin_image    = generate_image
        
        outputs.append(origin_image)

    return "SUCCESS", outputs