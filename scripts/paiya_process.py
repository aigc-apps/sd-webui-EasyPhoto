import numpy as np

import requests
import json
import base64
import time
import modules.scripts as scripts
from modules.api import api
from PIL import Image
from modules import shared
from modules.shared import opts, state
from threading import Thread
from io import BytesIO
from modules import processing, sd_samplers, shared
from modules.generation_parameters_copypaste import \
    create_override_settings_dict
from modules.images import save_image
from modules.processing import (Processed, StableDiffusionProcessingImg2Img,
                                StableDiffusionProcessingTxt2Img)
from modules.ui import plaintext_to_html
from scripts.skybox_config import skybox_outpath_samples
from scripts.skybox_util import get_controlnet_args, init_default_script_args
from scripts.api import reload_conv_forward
from modules.sd_models import get_closet_checkpoint_match, load_model
import base64
import copy
import io
import json
import logging
import os
import random
import sys
import time
import allspark
import cv2
import numpy as np
import glob
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
from sdwebui import ControlNetUnit, i2i_inpaint_call
from skimage import transform
import importlib
from scripts.paiya_utils import crop_and_paste, call_face_crop

external_code = importlib.import_module('extensions-builtin.sd-webui-controlnet.scripts.external_code', 'external_code')
# 使用端口与Prompt
APP_PORT            = 7861
DEFAULT_POSITIVE    = '(best quality), (realistic, photo-realistic:1.2), beautiful, cool, finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw phot, put on makeup'
DEFAULT_NEGATIVE    = '(worst quality:2), (low quality:2), (normal quality:2), hair, teeth, lowres, watermark, badhand, ((naked:2, nude:2, nsfw)), (normal quality:2), lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale)), mural,'

# 获得当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 设置日志记录器的级别
log_level = os.environ.get('LOG_LEVEL', 'INFO')
logging.getLogger().setLevel(log_level)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s')  

def download_image(image_link: str, remove = True) -> Image.Image:
    """
    Download an image from the given image_link and return it as a PIL Image object.
    Args:
        save_dir (str): The result dir to be saved
        image_link (str): The URL of the image to download.
    Returns:
        Image.Image: The downloaded image as a PIL Image object.
    """
    img_name = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())) + ".jpg"
    os.system(f"wget -O {img_name} {image_link}")
    image = Image.open(img_name)
    if remove:
        os.remove(img_name)
    return image

def encode_cv2image_to_base64jpeg(image: np.ndarray) -> str:
    buffer          = Image.fromarray(image)
    byte_io         = io.BytesIO()
    buffer.save(byte_io, format='JPEG')
    byte_data       = byte_io.getvalue()
    base64_image    = base64.b64encode(byte_data).decode()
    return base64_image

def encode_image_to_base64jpeg(image: Image.Image) -> str:
    byte_io         = io.BytesIO()
    image.save(byte_io, format='JPEG')
    byte_data       = byte_io.getvalue()
    base64_image    = base64.b64encode(byte_data).decode()
    return base64_image

def decode_image_from_base64jpeg(base64_image):
    if base64_image == "":
        return None
    image_bytes = base64.b64decode(base64_image)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def inpaint_with_mask_face(
        input_image: Image.Image,
        select_mask_input: Image.Image,
        replaced_input_image: Image.Image,
        roop_image:Image.Image = None,
        input_prompt = '1girl',
        hr_scale: float = 1.0,
        default_positive_prompt = DEFAULT_POSITIVE,
        default_negative_prompt = DEFAULT_NEGATIVE,
        seed: int = 123456,
        sd_model_checkpoint = "ChilloutMix-ni-fp16.safetensors",
        output_path = None,
        url = "http://127.0.0.1:7860",
        authorization = None
):
    seed = random.randint(0, 10000000)
    assert input_image is not None, f'input_image must not be none'
    controlnet_units_list = []
    w = int(input_image.width)
    h = int(input_image.height)

    control_weight_canny = 1.0
    canny_weight = 0.50
    if 1:
        # canny_pil = Image.fromarray(cv2.cvtColor(image_rembg, cv2.COLOR_RGB2BGR))
        control_unit_canny = ControlNetUnit(input_image=input_image, module='canny',
                                            weight=0.50,
                                            guidance_end=1,
                                            resize_mode='Just Resize',
                                            threshold_a=100,
                                            threshold_b=200,
                                            model='control_v11p_sd15_canny [d14c016b]')
        controlnet_units_list.append(control_unit_canny)
    if 1:
        control_unit_canny = ControlNetUnit(input_image=replaced_input_image, module='openpose_full',
                                            weight=control_weight_canny - canny_weight,
                                            guidance_end=1,
                                            resize_mode='Just Resize',
                                            model='control_v11p_sd15_openpose [cab727d4]')
        controlnet_units_list.append(control_unit_canny)

    print(len(controlnet_units_list))
    positive = f'{input_prompt}, {default_positive_prompt}'
    negative = f'{default_negative_prompt}'

    image_names, image_filepaths, params = i2i_inpaint_call(images=[input_image], \
                                                            roop_image=roop_image,
                                                            mask_image=select_mask_input,
                                                            inpainting_fill=1, \
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
                                                            sd_vae="vae-ft-mse-840000-ema-pruned.safetensors",
                                                            sd_model_checkpoint=sd_model_checkpoint,
                                                            url=url,
                                                            authorization=authorization,
                                                            )

    reform_path = image_filepaths[0]
    if output_path is not None:
        os.system(f'mv {reform_path} {output_path}')
        return output_path
    return image_filepaths[0]

def inpaint_only(        
        input_image: Image.Image,
        input_mask: Image.Image,
        input_prompt = '1girl',
        roop_image = None,
        fusion_image = None,
        hr_scale: float = 1.0,
        default_positive_prompt = DEFAULT_POSITIVE,
        default_negative_prompt = DEFAULT_NEGATIVE,
        seed: int = 123456,
        sd_model_checkpoint = "ChilloutMix-ni-fp16.safetensors",
        output_path = None,
        url = "http://127.0.0.1:7860",
        authorization = None,
    ):
    seed = random.randint(0, 10000000)
    assert input_image is not None, f'input_image must not be none'
    controlnet_units_list = []
    w = int(input_image.width)
    h = int(input_image.height)

    if 1:
        control_unit_canny = ControlNetUnit(input_image=input_image if fusion_image is None else fusion_image, module='canny',
                                            weight=1,
                                            guidance_end=1,
                                            resize_mode='Just Resize',
                                            threshold_a=100,
                                            threshold_b=200,
                                            model='control_v11p_sd15_canny [d14c016b]')
        controlnet_units_list.append(control_unit_canny)
    if 1:
        control_unit_canny = ControlNetUnit(input_image=input_image if fusion_image is None else fusion_image, module='tile_resample',
                                            weight=1,
                                            guidance_end=1,
                                            resize_mode='Just Resize',
                                            threshold_a=1,
                                            threshold_b=200,
                                            model='control_v11f1e_sd15_tile [a371b31b]')
        controlnet_units_list.append(control_unit_canny)
    if 0:
        control_unit_canny = ControlNetUnit(input_image=input_image, module='inpaint_only',
                                            weight=1,
                                            guidance_end=1,
                                            resize_mode='Just Resize',
                                            model='control_v11p_sd15_inpaint [ebff9138]')
        controlnet_units_list.append(control_unit_canny)


    print(len(controlnet_units_list))
    positive = f'{input_prompt}, {default_positive_prompt}'
    negative = f'{default_negative_prompt}'

    # select_mask_input = Image.fromarray(select_mask_input)
    image_names, image_filepaths, params = i2i_inpaint_call(images=[input_image], \
                                                            roop_image=roop_image,
                                                            mask_image=input_mask,
                                                            inpainting_fill=1, \
                                                            denoising_strength=0.10,
                                                            inpainting_mask_invert=0,
                                                            width=int(hr_scale * w),
                                                            height=int(hr_scale * h),
                                                            inpaint_full_res=False,
                                                            seed=seed,
                                                            prompt=positive,
                                                            negative_prompt=negative,
                                                            controlnet_units=controlnet_units_list,
                                                            sd_vae="vae-ft-mse-840000-ema-pruned.safetensors",
                                                            sd_model_checkpoint=sd_model_checkpoint,
                                                            url=url,
                                                            authorization=authorization
                                                            )

    reform_path = image_filepaths[0]
    if output_path is not None:
        os.system(f'mv {reform_path} {output_path}')
        return output_path
    return image_filepaths[0]

def paiya_forward(user_id, selected_template_images, append_pos_prompt, first_control_weight, second_control_weight,
                            final_fusion_ratio, use_fusion_before, use_fusion_after, args): 
                            
    retinaface_detection    = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
    skin_retouching         = pipeline(Tasks.skin_retouching, 'damo/cv_unet_skin-retouching')
    image_face_fusion       = pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo')

    # 输出图像路径
    output_dir              = 'output'
    # 是否裁剪图像后进行重建
    crop_face_preprocess    = True
    # 随机seed
    seed                    = random.randint(0, 10000)

    input_prompt            = f"paiya_face, paiya, 1person, " + append_pos_prompt + DEFAULT_POSITIVE
    
    weights_save_path       = os.path.join("/photog_oss/photog/user_weights", user_id)
    best_images_path        = glob.glob(os.path.join(weights_save_path, "best_outputs/*.jpg"))
    face_id_image           = Image.open(best_images_path[-1]).convert("RGB")
    template_image          = Image.open(selected_template_images).convert("RGB")

    roop_image_path         = os.path.join("/photog_oss/photog/user_images", user_id, "ref_image.jpg")
    roop_image              = Image.open(roop_image_path).convert("RGB")

    # 对用户的图像进行裁剪，获取人像box、人脸关键点与mask
    roop_face_retinaface_box, roop_face_retinaface_keypoints, roop_face_retinaface_mask = call_face_crop(face_id_image, 1.5, "roop")

    # 对模板图像进行裁剪，只保留人像的部分
    if crop_face_preprocess:
        crop_safe_box, _, _ = call_face_crop(template_image, 3, "crop")
        input_image = copy.deepcopy(template_image).crop(crop_safe_box)
    else:
        input_image = copy.deepcopy(template_image)

    # 对模板图像进行resize短边到512上
    short_side  = min(input_image.width, input_image.height)
    resize      = float(short_side / 512.0)
    new_size    = (int(input_image.width//resize), int(input_image.height//resize))
    input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)

    if crop_face_preprocess:
        new_width   = int(np.shape(input_image)[1] // 32 * 32)
        new_height  = int(np.shape(input_image)[0] // 32 * 32)
        input_image = input_image.resize([new_width, new_height], Image.Resampling.LANCZOS)
    
    # 检测模板图像人脸所在的box，获取其对应的小mask
    retinaface_box, retinaface_keypoints, input_mask = call_face_crop(input_image, 1.1, "template")

    # 将用户图像贴到模板图像上
    replaced_input_image = crop_and_paste(face_id_image, roop_face_retinaface_mask, input_image, roop_face_retinaface_keypoints, retinaface_keypoints, roop_face_retinaface_box)
    replaced_input_image = Image.fromarray(np.uint8(replaced_input_image))
    replaced_input_image.save(os.path.join(output_dir, "paste_face.jpg"))
    
    if roop_image is not None and use_fusion_before:
        # 融合用户参考图像和输入图片，canny输入
        input_image = image_face_fusion(dict(template=input_image, user=roop_image))[OutputKeys.OUTPUT_IMG]
        input_image = Image.fromarray(np.uint8(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)))

    # 将模板图像在x轴方向进行扩张，这样可以包括到耳朵。
    h, w, c             = np.shape(input_mask)
    retinaface_box      = np.int32(retinaface_box)
    face_width          = retinaface_box[2] - retinaface_box[0]
    retinaface_box[0]   = np.clip(np.array(retinaface_box[0], np.int32) - face_width * 0.15, 0, w - 1)
    retinaface_box[2]   = np.clip(np.array(retinaface_box[2], np.int32) + face_width * 0.15, 0, w - 1)
    input_mask          = np.zeros_like(np.array(input_mask, np.uint8))
    input_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = 255
    input_mask          = Image.fromarray(np.uint8(input_mask))
    
    # 第一次diffusion，进行人脸重建
    output_path = os.path.join(output_dir, "tmp.jpg")
    output_path = inpaint_with_mask_face(input_image, input_mask, replaced_input_image, None, input_prompt=input_prompt, hr_scale=1.0, seed=str(seed), output_path=output_path, authorization=self.authorization, url=self.url)

    
    # 获取人脸周围区域的mask
    input_mask  = Image.fromarray(np.uint8(cv2.dilate(np.array(input_mask), np.ones((96, 96), np.uint8), iterations=1) - cv2.erode(np.array(input_mask), np.ones((16, 16), np.uint8), iterations=1)))
    # 获取第一次diffusion的结果
    output_image = Image.open(output_path)

    if roop_image is not None and use_fusion_after:
        # 脸型照片与用户照片融合
        fusion_image = image_face_fusion(dict(template=output_image, user=roop_image))[OutputKeys.OUTPUT_IMG]
        fusion_image = Image.fromarray(cv2.cvtColor(fusion_image, cv2.COLOR_BGR2RGB))
        output_image = Image.fromarray(np.uint8((np.array(output_image, np.float32) * final_fusion_ratio + np.array(fusion_image, np.float32) * (1 - final_fusion_ratio))))

        # 进行第二次diffusion
        toutput_path = inpaint_only(output_image, input_mask, input_prompt, fusion_image=fusion_image, hr_scale=1.5, output_path=output_path[:-4]+'_hq.jpg', authorization=self.authorization, url=self.url)
    else:
        toutput_path = inpaint_only(output_image, input_mask, input_prompt, hr_scale=1.5, output_path=output_path[:-4]+'_hq.jpg', authorization=self.authorization, url=self.url)

    # 最后进行一次美颜
    generate_image = Image.open(toutput_path)

    # 如果是大模板进行裁剪的话，就把重建好的图贴回去
    if crop_face_preprocess:
        origin_image    = np.array(copy.deepcopy(template_image))
        x1,y1,x2,y2     = crop_safe_box
        generate_image  = generate_image.resize([x2-x1, y2-y1], Image.Resampling.LANCZOS)
        origin_image[y1:y2,x1:x2] = np.array(generate_image)
        origin_image    = Image.fromarray(np.uint8(origin_image))
        origin_image.save(toutput_path)
    else:
        origin_image    = generate_image

    return origin_image

def paiya_process(
    id_task: str,
    init_image, 
    prompt,
    negative_prompt,
    steps,
    sampler_index,
    restore_faces,
    tiling,

    noise_multiplier,
    cfg_scale,
    image_cfg_scale,
    denoising_strength,
    seed,
    subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras,

    frame_interpolation_ratio,

    width,
    height,
    override_settings_text, 
    *args
):
    # 需要使用到的controlnet
    args = ['tile', 1, 512, 'hed', 1, 512, 'sd_color', 1, 1024]
    model_info = shared.sd_model.sd_checkpoint_info.model_name
    if shared.cmd_opts.just_ui:
        # 只使用前端
        read_progress = ReadProgress()
        read_progress.id_tasks[id_task] = True

        simple_req = dict(
            id_task = id_task,
            init_image = api.encode_pil_to_base64(Image.fromarray(np.uint8(init_image))) if init_image is not None else None, 
            prompt = prompt,
            negative_prompt = negative_prompt,
            steps = steps,
            sampler_index = sampler_index,
            restore_faces = restore_faces,
            tiling = tiling,

            noise_multiplier = noise_multiplier,
            cfg_scale = cfg_scale,
            image_cfg_scale = image_cfg_scale,
            denoising_strength = denoising_strength,
            seed = seed,
            subseed = subseed, subseed_strength = subseed_strength, seed_resize_from_h = seed_resize_from_h, seed_resize_from_w = seed_resize_from_w, seed_enable_extras = seed_enable_extras,

            frame_interpolation_ratio = frame_interpolation_ratio,

            width = width,
            height = height,
            override_settings_text = override_settings_text, 
            model_info = model_info,
            args = args
        )
        url_skybox  = '/'.join([shared.cmd_opts.server_path, 'skybox/forward'])
        data        = requests.post(url_skybox, json=simple_req)

        read_progress.id_tasks.pop(id_task, None)
        read_progress.flag = False
        
        gen_image   = Image.fromarray(np.array(api.decode_base64_to_image(json.loads(data.text)['image_b64']), np.uint8))
        js          = json.loads(data.text)['js']
        info        = json.loads(data.text)['info']
        comments    = json.loads(data.text)['comments']
        return [gen_image], js, info, comments
    else:
        opts.control_net_allow_script_control   = True
        override_settings = create_override_settings_dict(override_settings_text)
        # pipeline参数初始化
        p_txt2img = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            outpath_samples=shared.opts.data.get("skybox_outpath_samples", skybox_outpath_samples),
            outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
            prompt=prompt,
            negative_prompt=negative_prompt,
            styles=[],
            seed=seed,
            subseed=subseed,
            subseed_strength=subseed_strength,
            seed_resize_from_h=seed_resize_from_h,
            seed_resize_from_w=seed_resize_from_w,
            seed_enable_extras=seed_enable_extras,
            sampler_name=sd_samplers.samplers_for_img2img[sampler_index].name,
            batch_size=1,
            n_iter=1,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            restore_faces=restore_faces,
            tiling=tiling,
            override_settings=override_settings,
        )

        mask_blur                   = 4
        inpainting_fill             = 1
        inpaint_full_res            = False
        inpaint_full_res_padding    = 32
        inpainting_mask_invert      = 0
        p_img2img = StableDiffusionProcessingImg2Img(
            sd_model=shared.sd_model,
            outpath_samples=shared.opts.data.get("skybox_outpath_samples", skybox_outpath_samples),
            outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
            prompt=prompt,
            negative_prompt=negative_prompt,
            styles=[],
            seed=seed,
            subseed=subseed,
            subseed_strength=subseed_strength,
            seed_resize_from_h=seed_resize_from_h,
            seed_resize_from_w=seed_resize_from_w,
            seed_enable_extras=seed_enable_extras,
            sampler_name=sd_samplers.samplers_for_img2img[sampler_index].name,
            batch_size=1,
            n_iter=1,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            restore_faces=restore_faces,
            tiling=tiling,
            init_images=[None],
            mask=None,
            mask_blur=mask_blur,
            inpainting_fill=inpainting_fill,
            resize_mode=0,
            denoising_strength=denoising_strength,
            image_cfg_scale=image_cfg_scale,
            inpaint_full_res=inpaint_full_res,
            inpaint_full_res_padding=inpaint_full_res_padding,
            inpainting_mask_invert=inpainting_mask_invert,
            override_settings=override_settings,
            initial_noise_multiplier=noise_multiplier
        )

        p_txt2img.scripts       = scripts.scripts_txt2img
        p_txt2img.script_args   = init_default_script_args(scripts.scripts_txt2img)
        p_img2img.scripts       = scripts.scripts_img2img
        p_img2img.script_args   = init_default_script_args(scripts.scripts_img2img)
        p_img2img.extra_generation_params["Mask blur"] = mask_blur

        # 开始处理图片
        gen_image = process_skybox(
            p_txt2img, p_img2img, init_image, frame_interpolation_ratio, height, width, model_info, args
        )
        
        processed = Processed(p_img2img, [], p_img2img.seed, "")
        # 关闭process
        p_txt2img.close()
        p_img2img.close()
        shared.total_tqdm.clear()

        generation_info_js = processed.js()
        if opts.samples_log_stdout:
            print(generation_info_js)
            
        return [gen_image], generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments)
