import copy
import glob
import logging
import os
import sys
import cv2
import numpy as np
import torch
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modules import script_callbacks, shared
from modules.images import save_image
from modules.shared import opts, state
from PIL import Image
from scipy.optimize import minimize
from scripts.dragdiffusion_utils import run_drag
from scripts.easyphoto_config import (DEFAULT_NEGATIVE, 
                                      DEFAULT_POSITIVE, 
                                      easyphoto_img2img_samples,
                                      easyphoto_txt2img_samples, models_path,
                                      user_id_outpath_samples,
                                      validation_prompt)
from scripts.easyphoto_process_utils import (align_and_overlay_images,
                                             calculate_average_distance, calculate_polygon_iou,
                                             expand_polygon_vertex, adjust_B_to_match_A,
                                             mask_to_polygon)
from scripts.easyphoto_utils import (check_files_exists_and_download,
                                     check_id_valid)
from scripts.sdwebui import ControlNetUnit, i2i_inpaint_call, t2i_call
from scripts.train_kohya.utils.gpu_info import gpu_monitor_decorator
from segment_anything import SamPredictor, sam_model_registry
from shapely.geometry import Polygon
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys


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

# Add control_mode=1 means Prompt is more important, to better control lips and eyes,
# this comments will be delete after 10 PR and for those who are not familiar with SDWebUIControlNetAPI
def get_controlnet_unit(unit, input_image, weight):
    if unit == "canny":
        control_unit = ControlNetUnit(
            input_image=input_image, module='canny',
            weight=weight,
            guidance_end=1,
            control_mode=1, 
            resize_mode='Just Resize',
            threshold_a=100,
            threshold_b=200,
            model='control_v11p_sd15_canny'
        )
    elif unit == "openpose":
        control_unit = ControlNetUnit(
            input_image=input_image, module='openpose_full',
            weight=weight,
            guidance_end=1,
            control_mode=1, 
            resize_mode='Just Resize',
            model='control_v11p_sd15_openpose'
        )
    elif unit == "color":
        blur_ratio      = 24
        h, w, c         = np.shape(input_image)
        color_image     = np.array(input_image, np.uint8)

        color_image     = resize_image(color_image, 1024)
        now_h, now_w    = color_image.shape[:2]

        color_image = cv2.resize(color_image, (int(now_w//blur_ratio), int(now_h//blur_ratio)), interpolation=cv2.INTER_CUBIC)  
        color_image = cv2.resize(color_image, (now_w, now_h), interpolation=cv2.INTER_NEAREST)
        color_image = cv2.resize(color_image, (w, h), interpolation=cv2.INTER_CUBIC)
        color_image = Image.fromarray(np.uint8(color_image))

        control_unit = ControlNetUnit(input_image=color_image, module='none',
                                            weight=weight,
                                            guidance_end=1,
                                            control_mode=1,
                                            resize_mode='Just Resize',
                                            model='control_sd15_random_color')
    elif unit == "tile":
        control_unit = ControlNetUnit(
            input_image=input_image, module='tile_resample',
            weight=weight,
            guidance_end=1,
            control_mode=1, 
            resize_mode='Just Resize',
            threshold_a=1,
            threshold_b=200,
            model='control_v11f1e_sd15_tile'
        )
    return control_unit

def txt2img(
    controlnet_pairs: list,
    input_prompt = '1girl',
    diffusion_steps = 50,
    width: int = 1024,
    height: int = 1024,
    default_positive_prompt = DEFAULT_POSITIVE,
    default_negative_prompt = DEFAULT_NEGATIVE,
    seed: int = 123456,
    sd_model_checkpoint = "Chilloutmix-Ni-pruned-fp16-fix.safetensors",
    sampler = "DPM++ 2M SDE Karras"
):
    controlnet_units_list = []

    for pair in controlnet_pairs:
        controlnet_units_list.append(
            get_controlnet_unit(pair[0], pair[1], pair[2])
        )

    positive = f'{input_prompt}, {default_positive_prompt}'
    negative = f'{default_negative_prompt}'

    image = t2i_call(
        steps=diffusion_steps,
        cfg_scale=7,
        width=width,
        height=height,
        seed=seed,
        prompt=positive,
        negative_prompt=negative,
        controlnet_units=controlnet_units_list,
        sd_model_checkpoint=sd_model_checkpoint,
        outpath_samples=easyphoto_txt2img_samples,
        sampler=sampler,
    )

    return image

def inpaint(
    input_image: Image.Image,
    select_mask_input: Image.Image,
    controlnet_pairs: list,
    input_prompt = '1girl',
    diffusion_steps = 50,
    denoising_strength = 0.45,
    hr_scale: float = 1.0,
    default_positive_prompt = DEFAULT_POSITIVE,
    default_negative_prompt = DEFAULT_NEGATIVE,
    seed: int = 123456,
    sd_model_checkpoint = "Chilloutmix-Ni-pruned-fp16-fix.safetensors",
    sampler = "DPM++ 2M SDE Karras"
):
    assert input_image is not None, f'input_image must not be none'
    controlnet_units_list = []
    w = int(input_image.width)
    h = int(input_image.height)

    for pair in controlnet_pairs:
        controlnet_units_list.append(
            get_controlnet_unit(pair[0], pair[1], pair[2])
        )

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
        sampler=sampler,
    )

    return image

def seg_by_box(raw_image_rgb, boxes_filt, segmentor):
    h, w = raw_image_rgb.shape[:2]
    # [1,2,3,4]

    segmentor.set_image(raw_image_rgb)
    transformed_boxes   = segmentor.transform.apply_boxes_torch(torch.from_numpy(np.expand_dims(boxes_filt, 0)), (h, w)).cuda()
    masks, scores, _    = segmentor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=True)

    # muilt mask
    mask = masks[0]
    mask = torch.sum(mask, dim=0)
    mask = mask.cpu().numpy().astype(np.uint8)
    mask_image = mask * 255

    return mask_image

def crop_image(img, box):
    """
    使用OpenCV裁剪图像

    参数:
    img (numpy.ndarray): 输入的图像 (使用cv2读取的图像)
    box (tuple): 裁剪框的坐标 (left, upper, right, lower)

    返回:
    numpy.ndarray: 裁剪后的图像
    """
    left, upper, right, lower = box
    cropped_img = img[upper:lower, left:right]
    
    return cropped_img

def invert_mask_and_get_largest_connected_component(mask):
    # Take the inverse of the mask
    # inverted_mask = cv2.bitwise_not(mask)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # Find the largest connected component
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # Ignore the background label

    # Create a new mask with only the largest connected component
    largest_connected_component_mask = np.zeros_like(mask)
    largest_connected_component_mask[labels == largest_label] = 255

    # Find the bounding box for the largest connected component
    x, y, width, height = cv2.boundingRect(largest_connected_component_mask)

    # Set all regions outside of the bounding box to black
    largest_connected_component_mask[:y, :] = 0
    largest_connected_component_mask[y+height:, :] = 0
    largest_connected_component_mask[:, :x] = 0
    largest_connected_component_mask[:, x+width:] = 0

    return largest_connected_component_mask, (x, y, x + width, y + height)

retinaface_detection = None
image_face_fusion = None
skin_retouching = None
portrait_enhancement = None
face_skin = None
face_recognition = None
check_hash = True

# this decorate is default to be closed, not every needs this, more for developers
# @gpu_monitor_decorator() 
def easyphoto_infer_forward(
    sd_model_checkpoint, init_image, additional_prompt, seed, first_diffusion_steps, first_denoising_strength, \
    model_selected_tab, *user_ids,
): 
    # global
    global retinaface_detection, image_face_fusion, skin_retouching, portrait_enhancement, face_skin, face_recognition, check_hash

    # check & download weights of basemodel/controlnet+annotator/VAE/face_skin/buffalo/validation_template
    check_files_exists_and_download(check_hash)
    check_hash = False

    for user_id in user_ids:
        if user_id != "none":
            if not check_id_valid(user_id, user_id_outpath_samples, models_path):
                return "User id is not exist", [], []  
    
    # update donot delete but use "none" as placeholder and will pass this face inpaint later
    passed_userid_list = []
    for idx, user_id in enumerate(user_ids):
        if user_id == "none":
            passed_userid_list.append(idx)
    
    if len(user_ids) == len(passed_userid_list):
        return "Please choose a user id.", [], []
    
    # get random seed 
    if int(seed) == -1:
        seed = np.random.randint(0, 65536)

    # path init
    model_path              = os.path.join(models_path, f"Stable-diffusion", sd_model_checkpoint)
    lora_path               = os.path.join(models_path, f"Lora/{user_ids[0]}.safetensors")
    sd_base15_checkpoint    = os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "stable-diffusion-v1-5")
    sam_checkpoint          = os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "sam_vit_l_0b3195.pth")
    portrait_enhancement    = pipeline(Tasks.image_portrait_enhancement, model='damo/cv_gpen_image-portrait-enhancement', model_revision='v1.0.0')

    # params init
    box_template            = [64, 298, 406, 588]
    inversion_strength      = 0.75
    lam                     = 0.1
    latent_lr               = 0.01
    n_pix_step              = 40
    start_step              = 0
    start_layer             = 10
    padding_size            = 50

    # prompt init
    prompt                  = validation_prompt
    input_prompt            = f"{validation_prompt}, <lora:{user_ids[0]}:0.95>"

    # model init
    sam                 = sam_model_registry['vit_l']()
    sam.load_state_dict(torch.load(sam_checkpoint))
    predictor           = SamPredictor(sam.cuda())
    salient_detect      = pipeline(Tasks.semantic_segmentation, model='damo/cv_u2net_salient-detection')

    # open image
    img1            = np.uint8(Image.open(os.path.join(user_id_outpath_samples, user_ids[0], "ref_image.jpg")))
    img2            = np.uint8(Image.fromarray(np.uint8(init_image)))
    template_copy   = copy.deepcopy(img2)
    mask1           = np.uint8(Image.open(os.path.join(user_id_outpath_samples, user_ids[0], "ref_image_mask.jpg")))
    mask2           = np.uint8(seg_by_box(np.array(img2), box_template, predictor))

    # box cal
    _, box_main     = invert_mask_and_get_largest_connected_component(np.uint8(mask1[:, :, 0]))

    img1            = crop_image(np.array(img1), box_main)
    mask1           = crop_image(np.array(mask1), box_main)
    img2            = crop_image(np.array(img2), box_template)
    mask2           = crop_image(np.array(mask2), box_template)

    # paste first
    result_img, rotate_img1, mask1, mask2 = align_and_overlay_images(np.array(img1), np.array(img2), np.array(mask1), np.array(mask2))
    result_img = Image.fromarray(np.uint8(result_img))

    epsilon_multiplier = 0.005
    polygon1 = mask_to_polygon(np.uint8(mask1), epsilon_multiplier)
    polygon2 = mask_to_polygon(np.uint8(mask2), epsilon_multiplier)
    num_k = int(len(polygon1) * 2)
    expand_A = expand_polygon_vertex(polygon1, num_k)
    expand_B = expand_polygon_vertex(polygon2, num_k)
    expand_B = adjust_B_to_match_A(expand_A, expand_B)
    expand_B = expand_B.reshape(-1)

    # Define the constraint functions
    def constraint_1(params,A):
        K = len(params) // 2
        new_A = params.reshape((K, 2))
        iou = calculate_polygon_iou(A, new_A)
        return iou - 0.9

    def constraint_2(params, A):
        K = len(params) // 2
        new_A = params.reshape((K, 2))
        avg_distance = calculate_average_distance(A, new_A)
        return avg_distance - 10
    
    def objective_function(params, A, B, lambda_1, lambda_2):
        K = len(params) // 2
        new_A = params.reshape((K, 2))
        iou = calculate_polygon_iou(B, new_A)
        avg_distance = calculate_average_distance(A, new_A)
        # total_loss = -iou
        total_loss = lambda_1 * (1 - iou) + lambda_2 * avg_distance
        return total_loss
    
    # Create constraint objects
    constraint_1_obj = {'type': 'ineq', 'fun': constraint_1, 'args': (polygon2,)}
    constraint_2_obj = {'type': 'ineq', 'fun': constraint_2, 'args': (expand_A,)}

    # Combine constraints into a list
    constraints = [constraint_1_obj, constraint_2_obj]

    # optimize
    res = minimize(objective_function, expand_B, args=(expand_A, polygon2, 5, 0.5), constraints=constraints, method='COBYLA')
    new_polygon = res.x.reshape(-1,2)
    
    # source_image cal
    source_image = rotate_img1 * np.float32(np.array(np.expand_dims(mask1, -1), np.uint8) > 128) + np.ones_like(rotate_img1) * 255 * (1 - np.float32(np.array(np.expand_dims(mask1, -1), np.uint8) > 128))
    source_image = np.pad(source_image, [(padding_size, padding_size), (padding_size, padding_size), (0, 0)], constant_values=255)
    
    # mask cal
    mask = np.uint8(cv2.dilate(np.array(mask1), np.ones((30, 30), np.uint8), iterations=1) - cv2.erode(np.array(mask1), np.ones((30, 30), np.uint8), iterations=1))
    mask = np.pad(mask,[(padding_size, padding_size), (padding_size, padding_size)])

    # points cal
    final_points = []
    source_points = expand_A
    target_points = new_polygon
    for i in range(len(source_points)):
        final_points.append([int(k) for k in source_points[i]])
        final_points.append([int(k) for k in target_points[i]])
    final_points = np.array(final_points) + padding_size

    out_image = run_drag(
        source_image,
        mask,
        prompt,
        final_points,
        inversion_strength,
        lam,
        latent_lr,
        n_pix_step,
        model_path,
        sd_base15_checkpoint,
        lora_path,
        start_step,
        start_layer
    )

    out_image   = out_image[padding_size:-padding_size, padding_size:-padding_size]
    mask1       = salient_detect(out_image)[OutputKeys.MASKS]
    mask_gen, box_gen = invert_mask_and_get_largest_connected_component(mask1)

    # crop image again
    gen_img     = crop_image(out_image, box_gen)
    mask_gen    = crop_image(mask_gen, box_gen)

    # paste again after drag
    result_img, rotate_img1, mask1, mask2 = align_and_overlay_images(np.array(gen_img), np.array(img2), np.array(mask_gen), np.array(mask2), box2=box_gen)
    result_img = Image.fromarray(np.uint8(result_img))

    # resize 
    short_side  = min(result_img.width, result_img.height)
    resize      = float(short_side / 512.0)
    new_size    = (int(result_img.width // resize // 32 * 32), int(result_img.height // resize // 32 * 32))
    result_img  = result_img.resize(new_size, Image.Resampling.LANCZOS)

    # first diffusion
    logging.info("Start First diffusion.")
    controlnet_pairs = [["canny", result_img, 0.50]]
    mask2 = Image.fromarray(np.uint8(np.clip((np.float32(mask2) * 255), 0, 255)))

    result_img = inpaint(result_img, mask2, controlnet_pairs, diffusion_steps=first_diffusion_steps, denoising_strength=first_denoising_strength, input_prompt=input_prompt, hr_scale=1.0, seed=str(seed), sd_model_checkpoint=sd_model_checkpoint)
    
    # resize diffusion results
    target_width = box_template[2] - box_template[0]
    target_height = box_template[3] - box_template[1]
    result_img = result_img.resize((target_width, target_height))

    # copy back
    template_copy = np.array(template_copy, np.uint8)
    template_copy[box_template[1]:box_template[3], box_template[0]:box_template[2]] = np.array(result_img, np.uint8)
    template_copy = Image.fromarray(np.uint8(template_copy))

    # add portrait enhancement
    template_copy = Image.fromarray(cv2.cvtColor(portrait_enhancement(template_copy)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
    return "Success", [template_copy]