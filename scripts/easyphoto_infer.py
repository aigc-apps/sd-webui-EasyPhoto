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
from scripts.easyphoto_config import (
    DEFAULT_NEGATIVE,
    DEFAULT_POSITIVE,
    easyphoto_img2img_samples,
    easyphoto_txt2img_samples,
    models_path,
    user_id_outpath_samples,
    validation_prompt,
    easyphoto_outpath_samples,
    zero123_model_path
)

from scripts.easyphoto_process_utils import (
    align_and_overlay_images,
    calculate_average_distance,
    calculate_polygon_iou,
    expand_polygon_vertex,
    adjust_B_to_match_A,
    mask_to_polygon,
    draw_vertex_polygon,
    mask_to_box,
    draw_box_on_image,
    seg_by_box,
    apply_mask_to_image,
    merge_with_inner_canny,
    crop_image,
    resize_image_with_pad,
    copy_white_mask_to_template,
    wrap_image_by_vertex,
    get_background_color,
    find_best_angle_ratio,
    resize_and_stretch,
    compute_rotation_angle,
    expand_box_by_pad,
    expand_roi,
    resize_to_512
)
from scripts.easyphoto_utils import check_files_exists_and_download, check_id_valid
from scripts.sdwebui import ControlNetUnit, i2i_inpaint_call, t2i_call
from scripts.train_kohya.utils.gpu_info import gpu_monitor_decorator

from segment_anything import SamPredictor, sam_model_registry
from shapely.geometry import Polygon
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import time

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
except:
    print('Please install Zero123 following the instruction of https://github.com/cvlab-columbia/zero123')

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
def get_controlnet_unit(unit, input_image, weight, use_preprocess=True):
    if unit == "canny":
        if use_preprocess:
            control_unit = ControlNetUnit(
                image=input_image,
                module="canny",
                weight=weight,
                guidance_end=1,
                control_mode=1,
                resize_mode="Just Resize",
                threshold_a=100,
                threshold_b=200,
                model="control_v11p_sd15_canny",
            )
            print("Processor is used for canny!")
            print(input_image.shape)
        else:
            # direct use the inout canny image with inner line
            control_unit = ControlNetUnit(
                image=input_image,
                module=None,
                weight=weight,
                guidance_end=1,
                control_mode=1,
                resize_mode="Crop and Resize",
                threshold_a=100,
                threshold_b=200,
                model="control_v11p_sd15_canny",
            )
            print("No processor is used for canny!")
            print(input_image.shape)
    elif unit == "openpose":
        control_unit = ControlNetUnit(
            image=input_image,
            module="openpose_full",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            model="control_v11p_sd15_openpose",
        )
    elif unit == "color":
        blur_ratio = 24
        h, w, c = np.shape(input_image)
        color_image = np.array(input_image, np.uint8)

        color_image = resize_image(color_image, 1024)
        now_h, now_w = color_image.shape[:2]

        color_image = cv2.resize(
            color_image,
            (int(now_w // blur_ratio), int(now_h // blur_ratio)),
            interpolation=cv2.INTER_CUBIC,
        )
        color_image = cv2.resize(
            color_image, (now_w, now_h), interpolation=cv2.INTER_NEAREST
        )
        color_image = cv2.resize(
            color_image, (w, h), interpolation=cv2.INTER_CUBIC)
        color_image = Image.fromarray(np.uint8(color_image))

        control_unit = ControlNetUnit(
            image=color_image,
            module="none",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            model="control_sd15_random_color",
        )
    elif unit == "tile":
        control_unit = ControlNetUnit(
            image=input_image,
            module="tile_resample",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            threshold_a=1,
            threshold_b=200,
            model="control_v11f1e_sd15_tile",
        )

    elif unit == "depth":
        control_unit = ControlNetUnit(
            image=input_image,
            module="depth_midas",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            model="control_v11f1p_sd15_depth",
        )

    elif unit == "ipa":
        control_unit = ControlNetUnit(
            image=input_image,
            module="ip-adapter_clip_sd15",
            weight=weight,
            guidance_end=1,
            control_mode=1,
            resize_mode="Just Resize",
            model="ip-adapter_sd15",
        )

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
        controlnet_units_list.append(
            get_controlnet_unit(pair[0], pair[1], pair[2], pair[3])
        )

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
    lora_weight, iou_threshold, angle, azimuth, ratio, batch_size, refine_input_mask, optimize_angle_and_ratio, refine_bound, \
    pure_image, global_inpaint, match_and_paste, remove_target, model_selected_tab, *user_ids,
):
    # global
    global check_hash, models_zero123

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
        print('Please choose a user id.')
        return "Please choose a user id.", [], []

    # get random seed
    if int(seed) == -1:
        seed = np.random.randint(0, 65536)

    # path init
    model_path = os.path.join(
        models_path, f"Stable-diffusion", sd_model_checkpoint)
    lora_path = os.path.join(models_path, f"Lora/{user_ids[0]}.safetensors")
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

    # params init
    inversion_strength = 0.75
    lam = 0.1
    latent_lr = 0.01
    n_pix_step = 40
    start_step = 0
    start_layer = 10
    padding_size = 50

    # prompt init
    prompt = validation_prompt
    input_prompt = f"{validation_prompt}, <lora:{user_ids[0]}:{lora_weight}>"
    print("input_prompt:", input_prompt)

    # model init
    sam = sam_model_registry["vit_l"]()
    sam.load_state_dict(torch.load(sam_checkpoint))
    predictor = SamPredictor(sam.cuda())
    salient_detect = pipeline(
        Tasks.semantic_segmentation, model="damo/cv_u2net_salient-detection"
    )

    # open image
    img1 = np.uint8(
        Image.open(os.path.join(user_id_outpath_samples,
                   user_ids[0], "ref_image.jpg"))
    )  # main
    img2 = np.uint8(Image.fromarray(np.uint8(init_image["image"])))  # template
    mask2_input = np.uint8(Image.fromarray(np.uint8(init_image["mask"])))

    if mask2_input.max()==0:
        print('Please mark the target region on the inference template!')
        return 'Please mark the target region on the inference template!', [], []

    # get mask1
    mask1 = np.uint8(
        Image.open(
            os.path.join(user_id_outpath_samples,
                         user_ids[0], "ref_image_mask.jpg")
        )
    )

    # box cal & refine mask
    if len(mask1.shape)==2:
        mask1 = np.repeat(mask1[:, :, np.newaxis], 3, axis=2)

    _, box_main = mask_to_box(np.uint8(mask1[:,:,0]))
    # draw_box_on_image(img1, box_main, "box1.jpg")

    # get mask2
    if refine_input_mask:
        _, box_template = mask_to_box(mask2_input[:, :, 0])
        # draw_box_on_image(img2, box_template,'box.jpg')
        mask2 = np.uint8(seg_by_box(np.array(img2), box_template, predictor))
    else:
        mask2 = mask2_input[:, :, 0]

    if remove_target:
        # remove mask2 on img2
        expand_kernal = 20
        mask_expand = cv2.dilate(np.array(mask2), np.ones(
                    (expand_kernal, expand_kernal), np.uint8), iterations=5)
        input = {
                'img':Image.fromarray(np.uint8(img2)),
                'mask':Image.fromarray(np.uint8(mask_expand)),
        }

        print('before refine:',img2.shape)
        W, H = img2.shape[0], img2.shape[1]
        inpainting = pipeline(Tasks.image_inpainting, model='damo/cv_fft_inpainting_lama', refine=True)
        result = inpainting(input)
        print(result.keys())
        img2_bg = result[OutputKeys.OUTPUT_IMG]
        img2_bg = cv2.resize(img2_bg, (H, W))
        print('after refine:',img2_bg.shape)
        cv2.imwrite('template_refine.jpg', img2_bg)


    # for final paste
    _, box_template = mask_to_box(mask2)
    if remove_target:
        template_copy = copy.deepcopy(img2_bg[:,:,::-1])
    else:
        template_copy = copy.deepcopy(img2)
    mask_copy = copy.deepcopy(mask2)

    cv2.imwrite("mask2_input.jpg", mask2)
    # draw_box_on_image(img2, box_template, "box2.jpg")

    # crop to get local img
    W, H = np.array(img2).shape[1], np.array(img2).shape[0]

    expand_ratio = 1.2
    img1 = crop_image(np.array(img1), box_main, expand_ratio=expand_ratio)
    mask1 = crop_image(np.array(mask1), box_main, expand_ratio=expand_ratio)
    img2 = crop_image(np.array(img2), box_template, expand_ratio=expand_ratio)
    mask2 = crop_image(np.array(mask2), box_template, expand_ratio=expand_ratio)

    # cv2.imwrite("croped_img1.jpg", img1[:,:,::-1])
    # cv2.imwrite("croped_img2.jpg", img2[:,:,::-1])
    # cv2.imwrite("croped_mask1.jpg", mask1)
    # cv2.imwrite("croped_mask2.jpg", mask2)

    box_template = expand_roi(box_template, ratio=expand_ratio, max_box=[0, 0, W, H])

    mask2_copy = copy.deepcopy(mask2)  # use for second paste

    if match_and_paste:
        # clothes or something that needs high detail texture
        # generate image background 
        if pure_image:
            # main background with most frequent color
            color = get_background_color(img1, mask1[:, :, 0])
            color_img = np.full((img2.shape[0], img2.shape[1], 3), color, dtype=np.uint8)
            background_img = apply_mask_to_image(color_img, img2, mask2)
            # mask_blur = cv2.GaussianBlur(
            #     np.array(np.uint8(mask2)), (5, 5), 0
            # )
        
            # color_img = np.full((img2.shape[0], img2.shape[1], 3), color, dtype=np.uint8)

            # mask_blur = np.stack((mask_blur,) * 3, axis=-1)
            # background_img = np.array(color_img, np.uint8)*(mask_blur/255.) + np.array(img2, np.uint8)*((255-mask_blur)/255.)

        else:
            # generate background_image with ipa (referenced img1)
            controlnet_pairs = [
                    ["ipa", img1, 2.0, True],
                    ["depth", img2, 1.0, True],
                ]

            background_diffusion_steps = 20
            background_denoising_strength = 0.8
            background_img = inpaint(
                    Image.fromarray(np.uint8(img2)),
                    Image.fromarray(np.uint8(mask2)),
                    controlnet_pairs,
                    diffusion_steps=background_diffusion_steps,
                    denoising_strength=background_denoising_strength,
                    input_prompt=input_prompt,
                    hr_scale=1.0,
                    seed=str(seed),
                    sd_model_checkpoint=sd_model_checkpoint,
                )

            background_img = background_img.resize((img2.shape[1],img2.shape[0]), Image.Resampling.LANCZOS)
            background_img = np.array(background_img)


        cv2.imwrite('background_img.jpg', background_img[:,:,::-1])
        # background_img.save('background_img.jpg')

        if azimuth !=0:
            # zero123   
            try:
                # no need to always load zero123 model
                x,y,z = 0,azimuth,0
                result = zero123_infer(models_zero123, x, y, z, Image.fromarray(np.uint8(img1))) # TODO [PIL.Image] May choose the best merge one by lightglue 
                
                # result[0].save('res_zero123.jpg')
                success=1
            except Exception as e:
                print(e)
                success=0
           
            print('success:',success)
            if not success:
                try:
                    # from scripts.thirdparty.zero123.infer import zero123_infer, load_model_from_config
                    # from lovely_numpy import lo
                    # from omegaconf import OmegaConf
                    # from scripts.thirdparty.zero123.ldm_zero123.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
                    # from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
                    # from transformers import AutoFeatureExtractor
                    # from torchvision import transforms
                    # from scripts.thirdparty.zero123.ldm_zero123.models.diffusion.ddim import DDIMSampler
                    # from contextlib import nullcontext
                    # from einops import rearrange
                    # import math

                    # load model
                    print(f'Loading zero123 model from {zero123_model_path}')
                    ckpt = os.path.join(zero123_model_path, '105000.ckpt')
                    config = os.path.join(zero123_model_path, 'configs/sd-objaverse-finetune-c_concat-256.yaml')

                    config = OmegaConf.load(config)
                    models_zero123 = dict()
                    print('Instantiating LatentDiffusion...')
                    device = 'cuda'
                    models_zero123['turncam'] = load_model_from_config(config, ckpt, device=device)
                    models_zero123['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(zero123_model_path).to(device)
                    print('Instantiating AutoFeatureExtractor...')
                    models_zero123['clip_fe'] = AutoFeatureExtractor.from_pretrained(zero123_model_path)
                    print('Instantiating Carvekit HiInterface...')
                    models_zero123['carvekit'] = create_carvekit_interface()

                    x,y,z = 0,azimuth,0
                    result = zero123_infer(models_zero123, x, y, z, Image.fromarray(np.uint8(img1))) # TODO [PIL.Image] May choose the best merge one by lightglue 
                    # result[0].save('res_zero123.jpg')
                except:
                    raise ImportError('Please install Zero123 following the instruction of https://github.com/cvlab-columbia/zero123')    

            # crop and get mask
            img1_3d = np.array(result[0])
            mask1 = salient_detect(img1_3d)[OutputKeys.MASKS]
            mask1, box_gen = mask_to_box(mask1)

            mask1 = cv2.erode(np.array(mask1),
                                np.ones((10, 10), np.uint8), iterations=1)

            # crop image again
            img1 = crop_image(img1_3d, box_gen,expand_ratio=expand_ratio)
            mask1 = crop_image(mask1, box_gen,expand_ratio=expand_ratio)

            cv2.imwrite('img1_3d_crop.jpg', img1)
            cv2.imwrite('mask1_3d_crop.jpg', mask1)

        # first paste: paste img1 to img2 to get the control image
        if optimize_angle_and_ratio:
            print('Start Optimize!')
            # find optimzal angle and ratio
            # resize mask1 to same size as mask2 (init ratio as 1)
            resized_mask1 = resize_and_stretch(
                mask1, target_size=(mask2.shape[1], mask2.shape[0])
            )
            resized_mask1 = resized_mask1[:, :, 0]

            # cv2.imwrite("before_optimize_mask1.jpg", resized_mask1)
            # cv2.imwrite("before_optimize_mask2.jpg", mask2)

            # print(resized_mask1.shape)
            # print(mask2.shape)

            # get polygon
            polygon1 = mask_to_polygon(resized_mask1)
            polygon2 = mask_to_polygon(mask2)

            # target angle: 2 to 0
            rotation_angle2 = compute_rotation_angle(polygon2)
            # target angle: 1 to 0
            rotation_angle1 = compute_rotation_angle(polygon1)

            # wrong result of big angle (suppose the angle is small)
            if rotation_angle2 > 20:
                rotation_angle2 = 0
            if rotation_angle1 > 20:
                rotation_angle1 = 0
            angle_target = rotation_angle1 - rotation_angle2

            print(
                f"target rotation: 1 to 0: {rotation_angle1}, 2 to 0: {rotation_angle2}")

            # center
            x, y = mask2.shape[1] // 2, mask2.shape[0] // 2

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

   
        # paste first
        print("before first paste:", img1.shape)
        print("before first paste:", img2.shape)
        print("before first paste:", mask1.shape)
        print("before first paste:", mask2.shape)

        print(f'Set angle:{angle}, ratio: {ratio}, azimuth: {azimuth}')

        cv2.imwrite('first_img1.jpg',img1)
        cv2.imwrite('first_img2.jpg',img2)
        cv2.imwrite('first_mask1.jpg',mask1)
        cv2.imwrite('first_mask2.jpg',mask2)
        cv2.imwrite('first_background.jpg',background_img)

        print('before align!')
        print(img2.shape)
        print(background_img.shape)

        result_img, rotate_img1, mask1, mask2 = align_and_overlay_images(
            np.array(img1),
            np.array(background_img),
            np.array(mask1),
            np.array(mask2),
            angle=angle,
            ratio=ratio,
        )

        # shape not the same as img2 (img1 is rotate) but in the center
        print("after first paste:", rotate_img1.shape)
        print("after first paste:", result_img.shape)
        print("after first paste:", mask1.shape)
        print("after first paste:", mask2.shape)

        print("after first paste (img2):", np.array(img2).shape)

        # get the img2 box
        h_expand, w_expand = result_img.shape[:2]
        h2, w2 = np.array(img2).shape[:2]
        crop_img2_box_first = [
            (w_expand - w2) // 2,
            (h_expand - h2) // 2,
            (w_expand + w2) // 2,
            (h_expand + h2) // 2,
        ]

        # cv2.imwrite("first_paste_rotate_img1.jpg", rotate_img1)
        # cv2.imwrite("first_paste_mask1.jpg", mask1)
        # cv2.imwrite("first_paste_mask2.jpg", mask2)
        # cv2.imwrite("first_paste_result_img.jpg", result_img)

        first_paste = crop_image(result_img, crop_img2_box_first)
        first_paste = Image.fromarray(np.uint8(first_paste))

        first_paste.save('first_paste.jpg')

        second_paste = Image.fromarray(np.uint8(np.zeros((512, 512, 3))))

        # TODO drag diffusion to change shape is unused for not good results.
        # if change_shape:
        if 0:
            epsilon_multiplier = 0.005
            # mask to polygon
            polygon1 = mask_to_polygon(np.uint8(mask1), epsilon_multiplier)
            polygon2 = mask_to_polygon(np.uint8(mask2), epsilon_multiplier)

            # expand vertex and match polygon
            num_k = int(len(polygon1) * 2)

            expand_A = expand_polygon_vertex(polygon1, num_k)
            expand_B = expand_polygon_vertex(polygon2, num_k)
            expand_B = adjust_B_to_match_A(expand_A, expand_B)

            draw_vertex_polygon(
                rotate_img1,
                expand_A,
                "expand_A",
                line_color=(0, 0, 255),
                font_color=(0, 0, 255),
            )
            draw_vertex_polygon(
                rotate_img1,
                expand_B,
                "expand_B",
                line_color=(255, 0, 0),
                font_color=(255, 0, 0),
            )

            if optimize_vertex:
                # optimize the target polygon to obtain the larget iou and the least moving distance
                expand_B = expand_B.reshape(-1)

                # Define the constraint functions
                def constraint_1(params, A):
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
                constraint_1_obj = {
                    "type": "ineq",
                    "fun": constraint_1,
                    "args": (polygon2,),
                }
                constraint_2_obj = {
                    "type": "ineq",
                    "fun": constraint_2,
                    "args": (expand_A,),
                }

                # Combine constraints into a list
                constraints = [constraint_1_obj, constraint_2_obj]

                print(objective_function(expand_B, expand_A, polygon2, 0.5, 0.5))

                # optimize
                res = minimize(
                    objective_function,
                    expand_B,
                    args=(expand_A, polygon2, 5, 0.5),
                    constraints=constraints,
                    method="COBYLA",
                )
                new_polygon = res.x.reshape(-1, 2)

                draw_vertex_polygon(
                    rotate_img1,
                    new_polygon,
                    "new_polygon",
                    line_color=(0, 255, 0),
                    font_color=(0, 255, 0),
                )
            else:
                # default is to match vertex on A to vertex B
                new_polygon = expand_B

            if use_dragdiffusion:
                # drag the input image to target polygon by drag diffusion
                # source_image cal
                source_image = rotate_img1 * np.float32(
                    np.array(np.expand_dims(mask1, -1), np.uint8) > 128
                ) + np.ones_like(rotate_img1) * 255 * (
                    1 -
                    np.float32(np.array(np.expand_dims(mask1, -1), np.uint8) > 128)
                )
                source_image = np.pad(
                    source_image,
                    [(padding_size, padding_size),
                    (padding_size, padding_size), (0, 0)],
                    constant_values=255,
                )

                print("source_image:", source_image.shape)
                cv2.imwrite("source_img.jpg", source_image)

                # mask cal
                mask = np.uint8(
                    cv2.dilate(np.array(mask1), np.ones(
                        (30, 30), np.uint8), iterations=1)
                    - cv2.erode(np.array(mask1),
                                np.ones((30, 30), np.uint8), iterations=1)
                )
                mask = np.pad(
                    mask, [(padding_size, padding_size),
                        (padding_size, padding_size)]
                )

                print("drag_diffusion_mask:", mask.shape)
                # cv2.imwrite("drag_diffusion_mask.jpg", mask)

                # points cal
                final_points = []
                source_points = expand_A
                target_points = new_polygon
                for i in range(len(source_points)):
                    final_points.append([int(k) for k in source_points[i]])
                    final_points.append([int(k) for k in target_points[i]])
                final_points = np.array(final_points) + padding_size

                # print(len(final_points))
                draw_vertex_polygon(
                    source_image, np.array(source_points) +
                    padding_size, "pad_source"
                )
                draw_vertex_polygon(
                    source_image, np.array(target_points) +
                    padding_size, "pad_target"
                )

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
                    start_layer,
                )
                print("drag out:", out_image.shape)

                # cv2.imwrite("drag_diffusion_out.jpg", out_image[:, :, ::-1])
                out_image = out_image[
                    padding_size:-padding_size, padding_size:-padding_size
                ]

                print("after crop:", out_image.shape)
                mask1 = salient_detect(out_image)[OutputKeys.MASKS]
                mask_gen, box_gen = mask_to_box(mask1)

                # cv2.imwrite("drag_out_mask.jpg", mask_gen)
                # draw_box_on_image(out_image, box_gen, "drag_out_box.jpg")

                # crop image again
                gen_img = crop_image(out_image, box_gen)
                mask_gen = crop_image(mask_gen, box_gen)

                # paste again after drag
                print("Start second paste")
                print(gen_img.shape)
                print(mask_gen.shape)
                print(img2.shape)
                print(mask2_copy.shape)

                # cv2.imwrite("second_paste_img2.jpg", img2)

                result_img, rotate_img1, mask1, mask2 = align_and_overlay_images(
                    np.array(gen_img),
                    np.array(img2),
                    np.array(mask_gen),
                    np.array(mask2_copy),
                )
                # cv2.imwrite("second_paste_res.jpg", result_img)

                h_expand, w_expand = result_img.shape[:2]
                h2, w2 = np.array(img2).shape[:2]
                crop_img2_box_second = [
                    (w_expand - w2) // 2,
                    (h_expand - h2) // 2,
                    (w_expand + w2) // 2,
                    (h_expand + h2) // 2,
                ]

                second_paste = crop_image(result_img, crop_img2_box_second)
                second_paste = Image.fromarray(np.uint8(second_paste))

                # if 0:
                #     # resize
                #     short_side  = min(result_img.width, result_img.height)
                #     resize      = float(short_side / 512.0)
                #     new_size    = (int(result_img.width // resize // 32 * 32), int(result_img.height // resize // 32 * 32))
                #     result_img  = result_img.resize(new_size, Image.Resampling.LANCZOS)

                #     print('after_resize:',result_img.size)
                #     result_img.save('after_drag_paste.jpg')

            # else:
            #     # direct wrap the image by matching vertex
            #     result_img = wrap_image_by_vertex(rotate_img1, expand_A,new_polygon)

        # crop the result to the same size as cropped_img2
        # if change_shape:
        if 0:
            result_img = crop_image(result_img, crop_img2_box_second)
            mask1 = crop_image(mask1, crop_img2_box_second)
            mask2 = crop_image(mask2, crop_img2_box_second)
            resize_img1 = crop_image(rotate_img1, crop_img2_box_second)
        else:
            result_img = crop_image(result_img, crop_img2_box_first)
            mask1 = crop_image(mask1, crop_img2_box_first)
            mask2 = crop_image(mask2, crop_img2_box_first)
            resize_img1 = crop_image(rotate_img1, crop_img2_box_first)

        # final paste result
        result_img = Image.fromarray(np.uint8(result_img))

        # get inner canny and resize img to 512
        resize_image, res_canny = merge_with_inner_canny(
            np.array(result_img).astype(np.uint8), mask1, mask2
        )

        resize_mask2, remove_pad = resize_image_with_pad(mask2, resolution=512)
        resize_mask2 = remove_pad(resize_mask2)

        resize_img2, remove_pad = resize_image_with_pad(img2, resolution=512)
        resize_img2 = remove_pad(resize_img2)

        resize_mask1, remove_pad = resize_image_with_pad(mask1, resolution=512)
        resize_mask1 = remove_pad(resize_mask1)

        resize_img1, remove_pad = resize_image_with_pad(
            resize_img1, resolution=512)
        resize_img1 = remove_pad(resize_img1)

        print(
            "after canny:",
            resize_image.shape,
            res_canny.shape,
            resize_mask2.shape,
            resize_img2.shape,
            resize_mask1.shape,
            resize_img1.shape,
        )

        cv2.imwrite("after_canny_res_canny.jpg", res_canny)
        # cv2.imwrite("after_canny_res_image.jpg", resize_image[:, :, ::-1])
        # cv2.imwrite("after_canny_res_mask.jpg", resize_mask2)
        # cv2.imwrite("after_canny_res_control_depth.jpg", resize_img2[:, :, ::-1])
        # cv2.imwrite("after_canny_res_mask1.jpg", resize_mask1)
        # cv2.imwrite("after_canny_res_image1.jpg", resize_img1[:, :, ::-1])
         
        mask2 = Image.fromarray(
            np.uint8(resize_mask2))

        resize_image = Image.fromarray(resize_image)
        resize_image_input = copy.deepcopy(resize_image)
        mask2_input = copy.deepcopy(mask2)

    else:
        # # resize
        resize_img1 = resize_to_512(img1)
        resize_img2 = resize_to_512(img2)
        resize_mask1 = resize_to_512(mask1)
        resize_mask2 = resize_to_512(mask2)

        resize_image_input = Image.fromarray(np.uint8(resize_img2))
        mask2 = Image.fromarray(
            np.uint8(resize_mask2))

        mask2_input = copy.deepcopy(mask2)

   
    # generation
    return_res = []
    for i in range(batch_size):
        logging.info("Start First diffusion.")

        if match_and_paste:
            controlnet_pairs = [
                ["canny", res_canny, 1.0, False],
                ["depth", resize_img2, 1.0, True],
            ]
        else:
            controlnet_pairs = [
                
                ["depth", resize_img2, 1.0, True],
                # ["ipa", resize_img1, 1.0, True],
            ]
        
        result_img = inpaint(
            resize_image_input,
            mask2_input,
            controlnet_pairs,
            diffusion_steps=first_diffusion_steps,
            denoising_strength=first_denoising_strength,
            input_prompt=input_prompt,
            hr_scale=1.0,
            seed=str(seed),
            sd_model_checkpoint=sd_model_checkpoint,
        )

        print("inpaint:", result_img.size)
        result_img.save("inpaint_res1.jpg")

        # second diffusion
        # second_diffusion_steps = 20
        # # second_input_image =
        # result_img = inpaint(
        #     result_img,
        #     mask2_input,
        #     controlnet_pairs,
        #     diffusion_steps=second_diffusion_steps,
        #     denoising_strength=first_denoising_strength,
        #     input_prompt=input_prompt,
        #     hr_scale=1.0,
        #     seed=str(seed),
        #     sd_model_checkpoint=sd_model_checkpoint,
        # )
        # result_img.save("inpaint_res2.jpg")

        # resize diffusion results
        target_width = box_template[2] - box_template[0]
        target_height = box_template[3] - box_template[1]
        result_img = result_img.resize((target_width, target_height))
        resize_mask2 = mask2.resize((target_width, target_height))

        # result_img.save("inpaint_res_resize.jpg")
        # resize_mask2.save("inpaint_res_mask.jpg")

        print("resize:", result_img.size)
        print("box_template:", box_template)

        # copy back
        template_copy = np.array(template_copy, np.uint8)

        # mask_blur = cv2.GaussianBlur(
        #     np.array(np.uint8(resize_mask2))[:, :, 0], (5, 5), 0
        # )

        if len(np.array(np.uint8(resize_mask2)).shape)==2:
            init_generation = copy_white_mask_to_template(
                np.array(result_img), np.array(np.uint8(resize_mask2)), template_copy, box_template
            )
        else:
            init_generation = copy_white_mask_to_template(
                np.array(result_img), np.array(np.uint8(resize_mask2))[:, :, 0], template_copy, box_template
            )

        # cv2.imwrite("mask_blur.jpg", mask_blur)
        # cv2.imwrite("template_copy_init.jpg", init_generation)

        return_res.append(Image.fromarray(np.uint8(init_generation)))

        if refine_bound:
            # refine bound
            padding = 30

            box_pad = expand_box_by_pad(
                box_template,
                max_size=(template_copy.shape[1], template_copy.shape[0]),
                padding_size=padding,
            )
            print("box_pad:", box_pad)
            padding_size = abs(np.array(box_pad) - np.array(box_template))
            print("padding_size:", padding_size)

            input_img = init_generation[box_pad[1]: box_pad[3], box_pad[0]: box_pad[2]]
            input_control_img = template_copy[
                box_pad[1]: box_pad[3], box_pad[0]: box_pad[2]
            ]
            # up down left right

            if len(np.array(np.uint8(resize_mask2)).shape)==2:
                mask_array = np.array(np.uint8(resize_mask2))
            else:
                mask_array = np.array(np.uint8(resize_mask2))[:, :, 0]
            
            input_mask = np.pad(
                mask_array,
                ((padding_size[1], padding_size[3]),
                (padding_size[0], padding_size[2])),
                mode="constant",
                constant_values=0,
            )
            # cv2.imwrite("input_mask.jpg", input_mask)
            input_mask_copy = copy.deepcopy(input_mask)
            print("input_mask_copy:", input_mask_copy.shape)

            input_mask = np.uint8(
                cv2.dilate(np.array(input_mask), np.ones(
                    (10, 10), np.uint8), iterations=1)
                - cv2.erode(np.array(input_mask),
                            np.ones((10, 10), np.uint8), iterations=1)
            )


            # cv2.imwrite("input_mask_outline.jpg", input_mask)
            # cv2.imwrite("input_img.jpg", input_img)
            # cv2.imwrite("input_control_img.jpg", input_control_img)

            # print(input_mask.shape)
            # print(input_img.shape)
            # print("input_control:", input_control_img.shape)

            # generate
            controlnet_pairs = [["canny", input_control_img, 1.0, True]]
            # input_mask = Image.fromarray(
            #     np.uint8(np.clip((np.float32(input_mask) * 255), 0, 255))
            # )
            
            print(input_mask.max())
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

            print("res shape:", result_img.size)

            # resize diffusion results
            target_width = box_pad[2] - box_pad[0]
            target_height = box_pad[3] - box_pad[1]
            result_img = result_img.resize((target_width, target_height))
            input_mask = input_mask.resize((target_width, target_height))

            # result_img.save('result_img.jpg')
            # cv2.imwrite('template_copy.jpg', template_copy[:,:,::-1])
            # cv2.imwrite('input_mask_copy.jpg', input_mask_copy)

            print(box_pad)

            # copy back
            # mask_blur = cv2.GaussianBlur(
            #     np.array(np.uint8(input_mask_copy)), (5, 5), 0)
            # cv2.imwrite("mask_blur2.jpg", mask_blur)

            final_generation = copy_white_mask_to_template(
                np.array(result_img), np.array(np.uint8(input_mask_copy)), template_copy, box_pad
            )

            return_res.append(Image.fromarray(np.uint8(final_generation)))
        else:
            final_generation = init_generation

        # global inpaint
        if global_inpaint:
            global_diffusion_steps = 20
            global_denoising_strength = 0.3
            controlnet_pairs = [
                ["canny", final_generation, 1.0, True],
                ["depth", template_copy, 1.0, True],
            ]

            result_img = inpaint(
                Image.fromarray(np.uint8(final_generation)),
                Image.fromarray(np.uint8(mask_copy)),
                controlnet_pairs,
                diffusion_steps=global_diffusion_steps,
                denoising_strength=global_denoising_strength,
                input_prompt=input_prompt,
                hr_scale=1.0,
                seed=str(seed),
                sd_model_checkpoint=sd_model_checkpoint,
            )
            final_generation = np.array(result_img.resize((template_copy.shape[1], template_copy.shape[0])))


        save_image(Image.fromarray(np.uint8(final_generation)), easyphoto_outpath_samples, "EasyPhoto", None, None, opts.grid_format, info=None, short_filename=not opts.grid_extended_filename, grid=True, p=None)
        return_res.append(Image.fromarray(np.uint8(final_generation)))

    if match_and_paste:
        # show paste result for debug
        return_res.append(first_paste)
    
    if azimuth !=0:
        return_res.append(Image.fromarray(np.uint8(img1_3d)))
    
    torch.cuda.empty_cache()

    return "Success", return_res
