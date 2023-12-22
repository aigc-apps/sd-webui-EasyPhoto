# code copy from https://github.com/continue-revolution/sd-webui-animatediff
# rewrite AnimateDiffControl for batch control
# rewrite AnimateDiffMM for load model
# rewrite AnimateDiffOutput for get image
# rewrite AnimateDiffScript for easyphoto inject

import copy
import os
import shutil
from pathlib import Path
from types import MethodType
from typing import List, Optional

import cv2
import gradio as gr
import imageio.v3 as imageio
import numpy as np
import piexif
import PIL.features
import torch
from modules import (devices, hashes, images, img2img, masking, processing,
                     prompt_parser, scripts, sd_models, sd_samplers,
                     sd_samplers_common, shared)
from modules.devices import device, dtype_vae, torch_gc
from modules.paths import data_path
from modules.processing import (Processed, StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img,
                                StableDiffusionProcessingTxt2Img)
from modules.script_callbacks import (AfterCFGCallbackParams,
                                      CFGDenoisedParams, CFGDenoiserParams,
                                      cfg_after_cfg_callback,
                                      cfg_denoised_callback,
                                      cfg_denoiser_callback)
from modules.shared import opts, state
from PIL import Image, ImageFilter, ImageOps, PngImagePlugin
from scripts.easyphoto_config import easyphoto_models_path
from tqdm import tqdm

try:
    from modules.sd_samplers_common import (approximation_indexes,
                                            images_tensor_to_samples)
    from modules.sd_samplers_cfg_denoiser import (CFGDenoiser, catenate_conds,
                                                pad_cond, subscript_cond)

    from .animatediff.animatediff_i2ibatch import animatediff_i2ibatch
    from .animatediff.animatediff_infotext import update_infotext
    from .animatediff.animatediff_infv2v import AnimateDiffInfV2V
    from .animatediff.animatediff_logger import logger_animatediff as logger
    from .animatediff.animatediff_lora import AnimateDiffLora
    from .animatediff.animatediff_mm import AnimateDiffMM
    from .animatediff.animatediff_output import AnimateDiffOutput
    from .animatediff.animatediff_prompt import AnimateDiffPromptSchedule
    from .animatediff.animatediff_ui import (AnimateDiffProcess,
                                             AnimateDiffUiGroup)

    video_visible = True
except Exception as e:
    print(f"Animatediff is not Support when stable-diffusion webui is under v1.6.0. Animatediff import error detailed is follow: {e}")
    animatediff_i2ibatch = None
    update_infotext = None
    AnimateDiffInfV2V = None
    AnimateDiffLora = None
    AnimateDiffMM = None
    AnimateDiffOutput = None
    AnimateDiffPromptSchedule = None
    AnimateDiffProcess = None
    AnimateDiffUiGroup = None
    AnimateDiffControl = None
    AnimateDiffI2VLatent = None
    motion_module = None
    video_visible = False

if video_visible:

    class AnimateDiffControl:
        original_processing_process_images_hijack = None
        original_controlnet_main_entry = None
        original_postprocess_batch = None

        def __init__(self, p: StableDiffusionProcessing, prompt_scheduler: AnimateDiffPromptSchedule):
            try:
                from scripts.external_code import find_cn_script
                self.cn_script = find_cn_script(p.scripts)
            except:
                self.cn_script = None
            self.prompt_scheduler = prompt_scheduler

        def hack_batchhijack(self, params: AnimateDiffProcess):
            self.cn_script
            prompt_scheduler = self.prompt_scheduler

            from scripts import external_code
            from scripts.batch_hijack import BatchHijack, instance

            def hacked_processing_process_images_hijack(self, p: StableDiffusionProcessing, *args, **kwargs):
                if self.is_batch:  # AnimateDiff does not support this.
                    # we are in img2img batch tab, do a single batch iteration
                    return self.process_images_cn_batch(p, *args, **kwargs)

                units = external_code.get_all_units_in_processing(p)
                units = [unit for unit in units if getattr(unit, "enabled", False)]

                if len(units) > 0:
                    unit_batch_list = [len(unit.batch_images) for unit in units]

                    if len(unit_batch_list) > 0:
                        video_length = min(unit_batch_list)
                        # ensure that params.video_length <= video_length and params.batch_size <= video_length
                        if params.video_length > video_length:
                            params.video_length = video_length
                        if params.batch_size > video_length:
                            params.batch_size = video_length
                        if params.video_default:
                            params.video_length = video_length
                            p.batch_size = video_length
                        for unit in units:
                            unit.batch_images = unit.batch_images[: params.video_length]

                prompt_scheduler.parse_prompt(p)
                update_infotext(p, params)
                return getattr(processing, "__controlnet_original_process_images_inner")(p, *args, **kwargs)

            if AnimateDiffControl.original_processing_process_images_hijack is not None:
                logger.info('BatchHijack already hacked.')
                return
        
            AnimateDiffControl.original_processing_process_images_hijack = BatchHijack.processing_process_images_hijack
            BatchHijack.processing_process_images_hijack = hacked_processing_process_images_hijack
            processing.process_images_inner = instance.processing_process_images_hijack

        def restore_batchhijack(self):
            if AnimateDiffControl.original_processing_process_images_hijack is not None:
                from scripts.batch_hijack import BatchHijack, instance
                BatchHijack.processing_process_images_hijack = AnimateDiffControl.original_processing_process_images_hijack
                AnimateDiffControl.original_processing_process_images_hijack = None
                processing.process_images_inner = instance.processing_process_images_hijack

        def hack_cn(self):
            cn_script = self.cn_script


            def hacked_main_entry(self, p: StableDiffusionProcessing):
                from scripts import external_code, global_state, hook
                from scripts.adapter import (Adapter, Adapter_light,
                                             StyleAdapter)
                from scripts.batch_hijack import InputMode
                from scripts.controlmodel_ipadapter import (
                    PlugableIPAdapter, clear_all_ip_adapter)
                from scripts.controlnet_lllite import (PlugableControlLLLite,
                                                       clear_all_lllite)
                from scripts.controlnet_lora import bind_control_lora
                from scripts.hook import (ControlModelType, ControlParams,
                                          UnetHook)
                from scripts.logging import logger
                from scripts.processor import model_free_preprocessors

                # TODO: i2i-batch mode, what should I change?
                def image_has_mask(input_image: np.ndarray) -> bool:
                    return (
                        input_image.ndim == 3 and 
                        input_image.shape[2] == 4 and 
                        np.max(input_image[:, :, 3]) > 127
                    )


                def prepare_mask(
                    mask: Image.Image, p: processing.StableDiffusionProcessing
                ) -> Image.Image:
                    mask = mask.convert("L")
                    if getattr(p, "inpainting_mask_invert", False):
                        mask = ImageOps.invert(mask)
                    
                    if hasattr(p, 'mask_blur_x'):
                        if getattr(p, "mask_blur_x", 0) > 0:
                            np_mask = np.array(mask)
                            kernel_size = 2 * int(2.5 * p.mask_blur_x + 0.5) + 1
                            np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), p.mask_blur_x)
                            mask = Image.fromarray(np_mask)
                        if getattr(p, "mask_blur_y", 0) > 0:
                            np_mask = np.array(mask)
                            kernel_size = 2 * int(2.5 * p.mask_blur_y + 0.5) + 1
                            np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), p.mask_blur_y)
                            mask = Image.fromarray(np_mask)
                    else:
                        if getattr(p, "mask_blur", 0) > 0:
                            mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))
                    
                    return mask


                def set_numpy_seed(p: processing.StableDiffusionProcessing) -> Optional[int]:
                    try:
                        tmp_seed = int(p.all_seeds[0] if p.seed == -1 else max(int(p.seed), 0))
                        tmp_subseed = int(p.all_seeds[0] if p.subseed == -1 else max(int(p.subseed), 0))
                        seed = (tmp_seed + tmp_subseed) & 0xFFFFFFFF
                        np.random.seed(seed)
                        return seed
                    except Exception as e:
                        logger.warning(e)
                        logger.warning('Warning: Failed to use consistent random seed.')
                        return None

                sd_ldm = p.sd_model
                unet = sd_ldm.model.diffusion_model
                self.noise_modifier = None

                setattr(p, 'controlnet_control_loras', [])

                if self.latest_network is not None:
                    # always restore (~0.05s)
                    self.latest_network.restore()

                # always clear (~0.05s)
                clear_all_lllite()
                clear_all_ip_adapter()

                self.enabled_units = cn_script.get_enabled_units(p)

                if len(self.enabled_units) == 0:
                    self.latest_network = None
                    return

                detected_maps = []
                forward_params = []
                post_processors = []

                # cache stuff
                if self.latest_model_hash != p.sd_model.sd_model_hash:
                    cn_script.clear_control_model_cache()

                for idx, unit in enumerate(self.enabled_units):
                    unit.module = global_state.get_module_basename(unit.module)

                # unload unused preproc
                module_list = [unit.module for unit in self.enabled_units]
                for key in self.unloadable:
                    if key not in module_list:
                        self.unloadable.get(key, lambda:None)()

                self.latest_model_hash = p.sd_model.sd_model_hash
                for idx, unit in enumerate(self.enabled_units):
                    cn_script.bound_check_params(unit)

                    resize_mode = external_code.resize_mode_from_value(unit.resize_mode)
                    control_mode = external_code.control_mode_from_value(unit.control_mode)

                    if unit.module in model_free_preprocessors:
                        model_net = None
                    else:
                        model_net = cn_script.load_control_model(p, unet, unit.model)
                        model_net.reset()
                        if model_net is not None and getattr(devices, "fp8", False) and not isinstance(model_net, PlugableIPAdapter):
                            for _module in model_net.modules():
                                if isinstance(_module, (torch.nn.Conv2d, torch.nn.Linear)):
                                    _module.to(torch.float8_e4m3fn)

                        if getattr(model_net, 'is_control_lora', False):
                            control_lora = model_net.control_model
                            bind_control_lora(unet, control_lora)
                            p.controlnet_control_loras.append(control_lora)

                    input_images = []
                    for img in unit.batch_images:
                        unit.image = img
                        input_image, _ = cn_script.choose_input_image(p, unit, idx)
                        input_images.append(input_image)

                    for idx, input_image in enumerate(input_images):
                        a1111_mask_image : Optional[Image.Image] = getattr(p, "image_mask", None)
                        if a1111_mask_image and isinstance(a1111_mask_image, list):
                            a1111_mask_image = a1111_mask_image[idx]
                        if 'inpaint' in unit.module and not image_has_mask(input_image) and a1111_mask_image is not None:
                            a1111_mask = np.array(prepare_mask(a1111_mask_image, p))
                            if a1111_mask.ndim == 2:
                                if a1111_mask.shape[0] == input_image.shape[0]:
                                    if a1111_mask.shape[1] == input_image.shape[1]:
                                        input_image = np.concatenate([input_image[:, :, 0:3], a1111_mask[:, :, None]], axis=2)
                                        a1111_i2i_resize_mode = getattr(p, "resize_mode", None)
                                        if a1111_i2i_resize_mode is not None:
                                            resize_mode = external_code.resize_mode_from_value(a1111_i2i_resize_mode)

                        if 'reference' not in unit.module and issubclass(type(p), StableDiffusionProcessingImg2Img) \
                                and p.inpaint_full_res and a1111_mask_image is not None:
                            logger.debug("A1111 inpaint mask START")
                            input_image = [input_image[:, :, i] for i in range(input_image.shape[2])]
                            input_image = [Image.fromarray(x) for x in input_image]

                            mask = prepare_mask(a1111_mask_image, p)

                            crop_region = masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding)
                            crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)

                            input_image = [
                                images.resize_image(resize_mode.int_value(), i, mask.width, mask.height) 
                                for i in input_image
                            ]

                            input_image = [x.crop(crop_region) for x in input_image]
                            input_image = [
                                images.resize_image(external_code.ResizeMode.OUTER_FIT.int_value(), x, p.width, p.height) 
                                for x in input_image
                            ]

                            input_image = [np.asarray(x)[:, :, 0] for x in input_image]
                            input_image = np.stack(input_image, axis=2)
                            logger.debug("A1111 inpaint mask END")

                        # safe numpy
                        logger.debug("Safe numpy convertion START")
                        input_image = np.ascontiguousarray(input_image.copy()).copy()
                        logger.debug("Safe numpy convertion END")

                        input_images[idx] = input_image

                    if 'inpaint_only' == unit.module and issubclass(type(p), StableDiffusionProcessingImg2Img) and p.image_mask is not None:
                        logger.warning('A1111 inpaint and ControlNet inpaint duplicated. ControlNet support enabled.')
                        unit.module = 'inpaint'

                    logger.info(f"Loading preprocessor: {unit.module}")
                    preprocessor = self.preprocessor[unit.module]

                    high_res_fix = isinstance(p, StableDiffusionProcessingTxt2Img) and getattr(p, 'enable_hr', False)

                    h = (p.height // 8) * 8
                    w = (p.width // 8) * 8

                    if high_res_fix:
                        if p.hr_resize_x == 0 and p.hr_resize_y == 0:
                            hr_y = int(p.height * p.hr_scale)
                            hr_x = int(p.width * p.hr_scale)
                        else:
                            hr_y, hr_x = p.hr_resize_y, p.hr_resize_x
                        hr_y = (hr_y // 8) * 8
                        hr_x = (hr_x // 8) * 8
                    else:
                        hr_y = h
                        hr_x = w

                    if unit.module == 'inpaint_only+lama' and resize_mode == external_code.ResizeMode.OUTER_FIT:
                        # inpaint_only+lama is special and required outpaint fix
                        for idx, input_image in enumerate(input_images):
                            _, input_image = cn_script.detectmap_proc(input_image, unit.module, resize_mode, hr_y, hr_x)
                            input_images[idx] = input_image

                    control_model_type = ControlModelType.ControlNet
                    global_average_pooling = False

                    if 'reference' in unit.module:
                        control_model_type = ControlModelType.AttentionInjection
                    elif 'revision' in unit.module:
                        control_model_type = ControlModelType.ReVision
                    elif hasattr(model_net, 'control_model') and (isinstance(model_net.control_model, Adapter) or isinstance(model_net.control_model, Adapter_light)):
                        control_model_type = ControlModelType.T2I_Adapter
                    elif hasattr(model_net, 'control_model') and isinstance(model_net.control_model, StyleAdapter):
                        control_model_type = ControlModelType.T2I_StyleAdapter
                    elif isinstance(model_net, PlugableIPAdapter):
                        control_model_type = ControlModelType.IPAdapter
                    elif isinstance(model_net, PlugableControlLLLite):
                        control_model_type = ControlModelType.Controlllite

                    if control_model_type is ControlModelType.ControlNet:
                        global_average_pooling = model_net.control_model.global_average_pooling

                    preprocessor_resolution = unit.processor_res
                    if unit.pixel_perfect:
                        preprocessor_resolution = external_code.pixel_perfect_resolution(
                            input_images[0],
                            target_H=h,
                            target_W=w,
                            resize_mode=resize_mode
                        )

                    logger.info(f'preprocessor resolution = {preprocessor_resolution}')
                    # Preprocessor result may depend on numpy random operations, use the
                    # random seed in `StableDiffusionProcessing` to make the 
                    # preprocessor result reproducable.
                    # Currently following preprocessors use numpy random:
                    # - shuffle
                    seed = set_numpy_seed(p)
                    logger.debug(f"Use numpy seed {seed}.")

                    controls = []
                    hr_controls = []
                    controls_ipadapter = {'hidden_states': [], 'image_embeds': []}
                    hr_controls_ipadapter = {'hidden_states': [], 'image_embeds': []}
                    for idx, input_image in tqdm(enumerate(input_images), total=len(input_images)):
                        detected_map, is_image = preprocessor(
                            input_image, 
                            res=preprocessor_resolution, 
                            thr_a=unit.threshold_a,
                            thr_b=unit.threshold_b,
                        )

                        if high_res_fix:
                            if is_image:
                                hr_control, hr_detected_map = cn_script.detectmap_proc(detected_map, unit.module, resize_mode, hr_y, hr_x)
                                detected_maps.append((hr_detected_map, unit.module))
                            else:
                                hr_control = detected_map
                        else:
                            hr_control = None

                        if is_image:
                            control, detected_map = cn_script.detectmap_proc(detected_map, unit.module, resize_mode, h, w)
                            detected_maps.append((detected_map, unit.module))
                        else:
                            control = detected_map
                            detected_maps.append((input_image, unit.module))

                        if control_model_type == ControlModelType.T2I_StyleAdapter:
                            control = control['last_hidden_state']

                        if control_model_type == ControlModelType.ReVision:
                            control = control['image_embeds']

                        if control_model_type == ControlModelType.IPAdapter:
                            if model_net.is_plus:
                                controls_ipadapter['hidden_states'].append(control['hidden_states'][-2].cpu())
                            else:
                                controls_ipadapter['image_embeds'].append(control['image_embeds'].cpu())
                            if hr_control is not None:
                                if model_net.is_plus:
                                    hr_controls_ipadapter['hidden_states'].append(hr_control['hidden_states'][-2].cpu())
                                else:
                                    hr_controls_ipadapter['image_embeds'].append(hr_control['image_embeds'].cpu())
                            else:
                                hr_controls_ipadapter = None
                                hr_controls = None
                        else:
                            controls.append(control.cpu())
                            if hr_control is not None:
                                hr_controls.append(hr_control.cpu())
                            else:
                                hr_controls = None
                    
                    if control_model_type == ControlModelType.IPAdapter:
                        ipadapter_key = 'hidden_states' if model_net.is_plus else 'image_embeds'
                        controls = {ipadapter_key: torch.cat(controls_ipadapter[ipadapter_key], dim=0)}
                        if controls[ipadapter_key].shape[0] > 1:
                            controls[ipadapter_key] = torch.cat([controls[ipadapter_key], controls[ipadapter_key]], dim=0)
                        if model_net.is_plus:
                            controls[ipadapter_key] = [controls[ipadapter_key], None]
                        if hr_controls_ipadapter is not None:
                            hr_controls = {ipadapter_key: torch.cat(hr_controls_ipadapter[ipadapter_key], dim=0)}
                            if hr_controls[ipadapter_key].shape[0] > 1:
                                hr_controls[ipadapter_key] = torch.cat([hr_controls[ipadapter_key], hr_controls[ipadapter_key]], dim=0)
                            if model_net.is_plus:
                                hr_controls[ipadapter_key] = [hr_controls[ipadapter_key], None]
                    else:
                        controls = torch.cat(controls, dim=0)
                        if controls.shape[0] > 1:
                            controls = torch.cat([controls, controls], dim=0)
                        if hr_controls is not None:
                            hr_controls = torch.cat(hr_controls, dim=0)
                            if hr_controls.shape[0] > 1:
                                hr_controls = torch.cat([hr_controls, hr_controls], dim=0)

                    preprocessor_dict = dict(
                        name=unit.module,
                        preprocessor_resolution=preprocessor_resolution,
                        threshold_a=unit.threshold_a,
                        threshold_b=unit.threshold_b
                    )

                    forward_param = ControlParams(
                        control_model=model_net,
                        preprocessor=preprocessor_dict,
                        hint_cond=controls,
                        weight=unit.weight,
                        guidance_stopped=False,
                        start_guidance_percent=unit.guidance_start,
                        stop_guidance_percent=unit.guidance_end,
                        advanced_weighting=None,
                        control_model_type=control_model_type,
                        global_average_pooling=global_average_pooling,
                        hr_hint_cond=hr_controls,
                        soft_injection=control_mode != external_code.ControlMode.BALANCED,
                        cfg_injection=control_mode == external_code.ControlMode.CONTROL,
                    )
                    forward_params.append(forward_param)

                    unit_is_batch = InputMode.BATCH
                    if 'inpaint_only' in unit.module:
                        final_inpaint_raws = []
                        final_inpaint_masks = []
                        for i in range(len(controls)):
                            final_inpaint_feed = hr_controls[i] if hr_controls is not None else controls[i]
                            final_inpaint_feed = final_inpaint_feed.detach().cpu().numpy()
                            final_inpaint_feed = np.ascontiguousarray(final_inpaint_feed).copy()
                            final_inpaint_mask = final_inpaint_feed[0, 3, :, :].astype(np.float32)
                            final_inpaint_raw = final_inpaint_feed[0, :3].astype(np.float32)
                            sigma = shared.opts.data.get("control_net_inpaint_blur_sigma", 7)
                            final_inpaint_mask = cv2.dilate(final_inpaint_mask, np.ones((sigma, sigma), dtype=np.uint8))
                            final_inpaint_mask = cv2.blur(final_inpaint_mask, (sigma, sigma))[None]
                            _, Hmask, Wmask = final_inpaint_mask.shape
                            final_inpaint_raw = torch.from_numpy(np.ascontiguousarray(final_inpaint_raw).copy())
                            final_inpaint_mask = torch.from_numpy(np.ascontiguousarray(final_inpaint_mask).copy())
                            final_inpaint_raws.append(final_inpaint_raw)
                            final_inpaint_masks.append(final_inpaint_mask)

                        def inpaint_only_post_processing(x, i):
                            _, H, W = x.shape
                            if Hmask != H or Wmask != W:
                                logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                                return x
                            idx = i if unit_is_batch else 0
                            r = final_inpaint_raw[idx].to(x.dtype).to(x.device)
                            m = final_inpaint_mask[idx].to(x.dtype).to(x.device)
                            y = m * x.clip(0, 1) + (1 - m) * r
                            y = y.clip(0, 1)
                            return y

                        post_processors.append(inpaint_only_post_processing)

                    if 'recolor' in unit.module:
                        final_feeds = []
                        for i in range(len(controls)):
                            final_feed = hr_control if hr_control is not None else control
                            final_feed = final_feed.detach().cpu().numpy()
                            final_feed = np.ascontiguousarray(final_feed).copy()
                            final_feed = final_feed[0, 0, :, :].astype(np.float32)
                            final_feed = (final_feed * 255).clip(0, 255).astype(np.uint8)
                            Hfeed, Wfeed = final_feed.shape
                            final_feeds.append(final_feed)

                        if 'luminance' in unit.module:

                            def recolor_luminance_post_processing(x, i):
                                C, H, W = x.shape
                                if Hfeed != H or Wfeed != W or C != 3:
                                    logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                                    return x
                                h = x.detach().cpu().numpy().transpose((1, 2, 0))
                                h = (h * 255).clip(0, 255).astype(np.uint8)
                                h = cv2.cvtColor(h, cv2.COLOR_RGB2LAB)
                                h[:, :, 0] = final_feed[i if unit_is_batch else 0]
                                h = cv2.cvtColor(h, cv2.COLOR_LAB2RGB)
                                h = (h.astype(np.float32) / 255.0).transpose((2, 0, 1))
                                y = torch.from_numpy(h).clip(0, 1).to(x)
                                return y

                            post_processors.append(recolor_luminance_post_processing)

                        if 'intensity' in unit.module:

                            def recolor_intensity_post_processing(x, i):
                                C, H, W = x.shape
                                if Hfeed != H or Wfeed != W or C != 3:
                                    logger.error('Error: ControlNet find post-processing resolution mismatch. This could be related to other extensions hacked processing.')
                                    return x
                                h = x.detach().cpu().numpy().transpose((1, 2, 0))
                                h = (h * 255).clip(0, 255).astype(np.uint8)
                                h = cv2.cvtColor(h, cv2.COLOR_RGB2HSV)
                                h[:, :, 2] = final_feed[i if unit_is_batch else 0]
                                h = cv2.cvtColor(h, cv2.COLOR_HSV2RGB)
                                h = (h.astype(np.float32) / 255.0).transpose((2, 0, 1))
                                y = torch.from_numpy(h).clip(0, 1).to(x)
                                return y

                            post_processors.append(recolor_intensity_post_processing)

                    if '+lama' in unit.module:
                        forward_param.used_hint_cond_latent = hook.UnetHook.call_vae_using_process(p, control)
                        self.noise_modifier = forward_param.used_hint_cond_latent

                    del model_net

                is_low_vram = any(unit.low_vram for unit in self.enabled_units)

                self.latest_network = UnetHook(lowvram=is_low_vram)
                self.latest_network.hook(model=unet, sd_ldm=sd_ldm, control_params=forward_params, process=p)

                for param in forward_params:
                    if param.control_model_type == ControlModelType.IPAdapter:
                        param.control_model.hook(
                            model=unet,
                            clip_vision_output=param.hint_cond,
                            weight=param.weight,
                            dtype=torch.float32,
                            start=param.start_guidance_percent,
                            end=param.stop_guidance_percent
                        ) 
                    if param.control_model_type == ControlModelType.Controlllite:
                        param.control_model.hook(
                            model=unet,
                            cond=param.hint_cond,
                            weight=param.weight,
                            start=param.start_guidance_percent,
                            end=param.stop_guidance_percent
                        )

                self.detected_map = detected_maps
                self.post_processors = post_processors

                if os.path.exists(f'{data_path}/tmp/animatediff-frames/'):
                    shutil.rmtree(f'{data_path}/tmp/animatediff-frames/')

            def hacked_postprocess_batch(self, p, *args, **kwargs):
                images = kwargs.get('images', [])
                for post_processor in self.post_processors:
                    for i in range(len(images)):
                        images[i] = post_processor(images[i], i)
                return

            if AnimateDiffControl.original_controlnet_main_entry is not None:
                logger.info('ControlNet Main Entry already hacked.')
                return

            AnimateDiffControl.original_controlnet_main_entry = self.cn_script.controlnet_main_entry
            AnimateDiffControl.original_postprocess_batch = self.cn_script.postprocess_batch
            self.cn_script.controlnet_main_entry = MethodType(hacked_main_entry, self.cn_script)
            self.cn_script.postprocess_batch = MethodType(hacked_postprocess_batch, self.cn_script)
            
        def restore_cn(self):
            if AnimateDiffControl.original_controlnet_main_entry is not None:
                self.cn_script.controlnet_main_entry = AnimateDiffControl.original_controlnet_main_entry
                AnimateDiffControl.original_controlnet_main_entry = None
            if AnimateDiffControl.original_postprocess_batch is not None:
                self.cn_script.postprocess_batch = AnimateDiffControl.original_postprocess_batch
                AnimateDiffControl.original_postprocess_batch = None

        def hack(self, params: AnimateDiffProcess):
            if self.cn_script is not None:
                logger.info(f"Hacking ControlNet.")
                self.hack_batchhijack(params)
                self.hack_cn()

        def restore(self):
            if self.cn_script is not None:
                logger.info(f"Restoring ControlNet.")
                self.restore_batchhijack()
                self.restore_cn()

    class AnimateDiffInfV2V(AnimateDiffInfV2V):
        cfg_original_forward = None

        def __init__(self, p, prompt_scheduler: AnimateDiffPromptSchedule):
            try:
                from scripts.external_code import find_cn_script
                self.cn_script = find_cn_script(p.scripts)
            except:
                self.cn_script = None
            self.prompt_scheduler = prompt_scheduler


        # Returns fraction that has denominator that is a power of 2
        @staticmethod
        def ordered_halving(val):
            # get binary value, padded with 0s for 64 bits
            bin_str = f"{val:064b}"
            # flip binary value, padding included
            bin_flip = bin_str[::-1]
            # convert binary to int
            as_int = int(bin_flip, 2)
            # divide by 1 << 64, equivalent to 2**64, or 18446744073709551616,
            # or b10000000000000000000000000000000000000000000000000000000000000000 (1 with 64 zero's)
            final = as_int / (1 << 64)
            return final


        # Generator that returns lists of latent indeces to diffuse on
        @staticmethod
        def uniform(
            step: int = ...,
            video_length: int = 0,
            batch_size: int = 16,
            stride: int = 1,
            overlap: int = 4,
            loop_setting: str = 'R-P',
        ):
            if video_length <= batch_size:
                yield list(range(batch_size))
                return

            closed_loop = (loop_setting == 'A')
            stride = min(stride, int(np.ceil(np.log2(video_length / batch_size))) + 1)

            for context_step in 1 << np.arange(stride):
                pad = int(round(video_length * AnimateDiffInfV2V.ordered_halving(step)))
                both_close_loop = False
                for j in range(
                    int(AnimateDiffInfV2V.ordered_halving(step) * context_step) + pad,
                    video_length + pad + (0 if closed_loop else -overlap),
                    (batch_size * context_step - overlap),
                ):
                    if loop_setting == 'N' and context_step == 1:
                        current_context = [e % video_length for e in range(j, j + batch_size * context_step, context_step)]
                        first_context = [e % video_length for e in range(0, batch_size * context_step, context_step)]
                        last_context = [e % video_length for e in range(video_length - batch_size * context_step, video_length, context_step)]
                        def get_unsorted_index(lst):
                            for i in range(1, len(lst)):
                                if lst[i] < lst[i-1]:
                                    return i
                            return None
                        unsorted_index = get_unsorted_index(current_context)
                        if unsorted_index is None:
                            yield current_context
                        elif both_close_loop: # last and this context are close loop
                            both_close_loop = False
                            yield first_context
                        elif unsorted_index < batch_size - overlap: # only this context is close loop
                            yield last_context
                            yield first_context
                        else: # this and next context are close loop
                            both_close_loop = True
                            yield last_context
                    else:
                        yield [e % video_length for e in range(j, j + batch_size * context_step, context_step)]

        def hack(self, params: AnimateDiffProcess):
            if AnimateDiffInfV2V.cfg_original_forward is not None:
                logger.info("CFGDenoiser already hacked")
                return

            logger.info(f"Hacking CFGDenoiser forward function.")
            AnimateDiffInfV2V.cfg_original_forward = CFGDenoiser.forward
            cn_script = self.cn_script
            prompt_scheduler = self.prompt_scheduler

            def mm_cn_select(context: List[int]):
                # take control images for current context.
                if cn_script and cn_script.latest_network:
                    from scripts.hook import ControlModelType
                    for control in cn_script.latest_network.control_params:
                        if control.control_model_type not in [ControlModelType.IPAdapter, ControlModelType.Controlllite]:
                            if control.hint_cond.shape[0] > len(context):
                                control.hint_cond_backup = control.hint_cond
                                control.hint_cond = control.hint_cond[context]
                            control.hint_cond = control.hint_cond.to(device=devices.get_device_for("controlnet"))
                            if control.hr_hint_cond is not None:
                                if control.hr_hint_cond.shape[0] > len(context):
                                    control.hr_hint_cond_backup = control.hr_hint_cond
                                    control.hr_hint_cond = control.hr_hint_cond[context]
                                control.hr_hint_cond = control.hr_hint_cond.to(device=devices.get_device_for("controlnet"))
                        # IPAdapter and Controlllite are always on CPU.
                        elif control.control_model_type == ControlModelType.IPAdapter and control.control_model.image_emb.shape[0] > len(context):
                            control.control_model.image_emb_backup = control.control_model.image_emb
                            control.control_model.image_emb = control.control_model.image_emb[context]
                            control.control_model.uncond_image_emb_backup = control.control_model.uncond_image_emb
                            control.control_model.uncond_image_emb = control.control_model.uncond_image_emb
                        elif control.control_model_type == ControlModelType.Controlllite:
                            for module in control.control_model.modules.values():
                                if module.cond_image.shape[0] > len(context):
                                    module.cond_image_backup = module.cond_image
                                    module.set_cond_image(module.cond_image[context])
            
            def mm_cn_restore(context: List[int]):
                # restore control images for next context
                if cn_script and cn_script.latest_network:
                    from scripts.hook import ControlModelType
                    for control in cn_script.latest_network.control_params:
                        if control.control_model_type not in [ControlModelType.IPAdapter, ControlModelType.Controlllite]:
                            if getattr(control, "hint_cond_backup", None) is not None:
                                control.hint_cond_backup[context] = control.hint_cond.to(device="cpu")
                                control.hint_cond = control.hint_cond_backup
                            if control.hr_hint_cond is not None and getattr(control, "hr_hint_cond_backup", None) is not None:
                                control.hr_hint_cond_backup[context] = control.hr_hint_cond.to(device="cpu")
                                control.hr_hint_cond = control.hr_hint_cond_backup
                        elif control.control_model_type == ControlModelType.IPAdapter and getattr(control.control_model, "image_emb_backup", None) is not None:
                            control.control_model.image_emb = control.control_model.image_emb_backup
                            control.control_model.uncond_image_emb = control.control_model.uncond_image_emb_backup
                        elif control.control_model_type == ControlModelType.Controlllite:
                            for module in control.control_model.modules.values():
                                if getattr(module, "cond_image_backup", None) is not None:
                                    module.set_cond_image(module.cond_image_backup)

            def mm_sd_forward(self, x_in, sigma_in, cond_in, image_cond_in, make_condition_dict):
                x_out = torch.zeros_like(x_in)
                for context in AnimateDiffInfV2V.uniform(self.step, params.video_length, params.batch_size, params.stride, params.overlap, params.closed_loop):
                    if shared.opts.batch_cond_uncond:
                        _context = context + [c + params.video_length for c in context]
                    else:
                        _context = context
                    mm_cn_select(_context)
                    out = self.inner_model(
                        x_in[_context], sigma_in[_context],
                        cond=make_condition_dict(
                            cond_in[_context] if not isinstance(cond_in, dict) else {k: v[_context] for k, v in cond_in.items()},
                            image_cond_in[_context]))
                    x_out = x_out.to(dtype=out.dtype)
                    x_out[_context] = out
                    mm_cn_restore(_context)
                return x_out

            def mm_cfg_forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
                if state.interrupted or state.skipped:
                    raise sd_samplers_common.InterruptedException

                if sd_samplers_common.apply_refiner(self):
                    cond = self.sampler.sampler_extra_args['cond']
                    uncond = self.sampler.sampler_extra_args['uncond']

                # at self.image_cfg_scale == 1.0 produced results for edit model are the same as with normal sampling,
                # so is_edit_model is set to False to support AND composition.
                is_edit_model = shared.sd_model.cond_stage_key == "edit" and self.image_cfg_scale is not None and self.image_cfg_scale != 1.0

                conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
                uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)

                assert not is_edit_model or all(len(conds) == 1 for conds in conds_list), "AND is not supported for InstructPix2Pix checkpoint (unless using Image CFG scale = 1.0)"

                if self.mask_before_denoising and self.mask is not None:
                    x = self.init_latent * self.mask + self.nmask * x

                batch_size = len(conds_list)
                repeats = [len(conds_list[i]) for i in range(batch_size)]

                if shared.sd_model.model.conditioning_key == "crossattn-adm":
                    image_uncond = torch.zeros_like(image_cond) # this should not be supported.
                    make_condition_dict = lambda c_crossattn, c_adm: {"c_crossattn": [c_crossattn], "c_adm": c_adm}
                else:
                    image_uncond = image_cond
                    if isinstance(uncond, dict):
                        make_condition_dict = lambda c_crossattn, c_concat: {**c_crossattn, "c_concat": [c_concat]}
                    else:
                        make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": [c_crossattn], "c_concat": [c_concat]}

                if not is_edit_model:
                    x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x])
                    sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma])
                    image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond])
                else:
                    x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x] + [x])
                    sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma] + [sigma])
                    image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond] + [torch.zeros_like(self.init_latent)])

                denoiser_params = CFGDenoiserParams(x_in, image_cond_in, sigma_in, state.sampling_step, state.sampling_steps, tensor, uncond)
                cfg_denoiser_callback(denoiser_params)
                x_in = denoiser_params.x
                image_cond_in = denoiser_params.image_cond
                sigma_in = denoiser_params.sigma
                tensor = denoiser_params.text_cond
                uncond = denoiser_params.text_uncond
                skip_uncond = False

                # alternating uncond allows for higher thresholds without the quality loss normally expected from raising it
                if self.step % 2 and s_min_uncond > 0 and sigma[0] < s_min_uncond and not is_edit_model:
                    skip_uncond = True
                    x_in = x_in[:-batch_size]
                    sigma_in = sigma_in[:-batch_size]

                self.padded_cond_uncond = False
                if shared.opts.pad_cond_uncond and tensor.shape[1] != uncond.shape[1]:
                    empty = shared.sd_model.cond_stage_model_empty_prompt
                    num_repeats = (tensor.shape[1] - uncond.shape[1]) // empty.shape[1]

                    if num_repeats < 0:
                        tensor = pad_cond(tensor, -num_repeats, empty)
                        self.padded_cond_uncond = True
                    elif num_repeats > 0:
                        uncond = pad_cond(uncond, num_repeats, empty)
                        self.padded_cond_uncond = True

                if tensor.shape[1] == uncond.shape[1] or skip_uncond:
                    prompt_closed_loop = (params.video_length > params.batch_size) and (params.closed_loop in ['R+P', 'A']) # hook
                    tensor = prompt_scheduler.multi_cond(tensor, prompt_closed_loop) # hook
                    if is_edit_model:
                        cond_in = catenate_conds([tensor, uncond, uncond])
                    elif skip_uncond:
                        cond_in = tensor
                    else:
                        cond_in = catenate_conds([tensor, uncond])

                    if shared.opts.batch_cond_uncond:
                        x_out = mm_sd_forward(self, x_in, sigma_in, cond_in, image_cond_in, make_condition_dict) # hook
                    else:
                        x_out = torch.zeros_like(x_in)
                        for batch_offset in range(0, x_out.shape[0], batch_size):
                            a = batch_offset
                            b = a + batch_size
                            x_out[a:b] = mm_sd_forward(self, x_in[a:b], sigma_in[a:b], subscript_cond(cond_in, a, b), subscript_cond(image_cond_in, a, b), make_condition_dict) # hook
                else:
                    x_out = torch.zeros_like(x_in)
                    batch_size = batch_size*2 if shared.opts.batch_cond_uncond else batch_size
                    for batch_offset in range(0, tensor.shape[0], batch_size):
                        a = batch_offset
                        b = min(a + batch_size, tensor.shape[0])

                        if not is_edit_model:
                            c_crossattn = subscript_cond(tensor, a, b)
                        else:
                            c_crossattn = torch.cat([tensor[a:b]], uncond)

                        x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond=make_condition_dict(c_crossattn, image_cond_in[a:b]))

                    if not skip_uncond:
                        x_out[-uncond.shape[0]:] = self.inner_model(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:], cond=make_condition_dict(uncond, image_cond_in[-uncond.shape[0]:]))

                denoised_image_indexes = [x[0][0] for x in conds_list]
                if skip_uncond:
                    fake_uncond = torch.cat([x_out[i:i+1] for i in denoised_image_indexes])
                    x_out = torch.cat([x_out, fake_uncond])  # we skipped uncond denoising, so we put cond-denoised image to where the uncond-denoised image should be

                denoised_params = CFGDenoisedParams(x_out, state.sampling_step, state.sampling_steps, self.inner_model)
                cfg_denoised_callback(denoised_params)

                devices.test_for_nans(x_out, "unet")

                if is_edit_model:
                    denoised = self.combine_denoised_for_edit_model(x_out, cond_scale)
                elif skip_uncond:
                    denoised = self.combine_denoised(x_out, conds_list, uncond, 1.0)
                else:
                    denoised = self.combine_denoised(x_out, conds_list, uncond, cond_scale)

                if not self.mask_before_denoising and self.mask is not None:
                    denoised = self.init_latent * self.mask + self.nmask * denoised

                self.sampler.last_latent = self.get_pred_x0(torch.cat([x_in[i:i + 1] for i in denoised_image_indexes]), torch.cat([x_out[i:i + 1] for i in denoised_image_indexes]), sigma)

                if opts.live_preview_content == "Prompt":
                    preview = self.sampler.last_latent
                elif opts.live_preview_content == "Negative prompt":
                    preview = self.get_pred_x0(x_in[-uncond.shape[0]:], x_out[-uncond.shape[0]:], sigma)
                else:
                    preview = self.get_pred_x0(torch.cat([x_in[i:i+1] for i in denoised_image_indexes]), torch.cat([denoised[i:i+1] for i in denoised_image_indexes]), sigma)

                sd_samplers_common.store_latent(preview)

                after_cfg_callback_params = AfterCFGCallbackParams(denoised, state.sampling_step, state.sampling_steps)
                cfg_after_cfg_callback(after_cfg_callback_params)
                denoised = after_cfg_callback_params.x

                self.step += 1
                return denoised

            CFGDenoiser.forward = mm_cfg_forward


        def restore(self):
            if AnimateDiffInfV2V.cfg_original_forward is None:
                logger.info("CFGDenoiser already restored.")
                return

            logger.info(f"Restoring CFGDenoiser forward function.")
            CFGDenoiser.forward = AnimateDiffInfV2V.cfg_original_forward
            AnimateDiffInfV2V.cfg_original_forward = None


    class AnimateDiffOutput(AnimateDiffOutput):
        def output(self, p: StableDiffusionProcessing, res: Processed, params: AnimateDiffProcess):
            video_paths = []
            logger.info("Merging images into GIF.")
            Path(f"{p.outpath_samples}/AnimateDiff").mkdir(exist_ok=True, parents=True)
            step = params.video_length if params.video_length > params.batch_size else params.batch_size

            origin_video = []
            index = 0
            for i in range(res.index_of_first_image, len(res.images), step):
                # frame interpolation replaces video_list with interpolated frames
                # so make a copy instead of a slice (reference), to avoid modifying res
                video_list = [image.copy() for image in res.images[i : i + params.video_length]]

                seq = images.get_next_sequence_number(f"{p.outpath_samples}/AnimateDiff", "")
                filename = f"{seq:05}"
                video_path_prefix = f"{p.outpath_samples}/AnimateDiff/{filename}"

                video_list = self._add_reverse(params, video_list)
                video_list = self._interp(p, params, video_list, filename)
                video_paths += self._save(params, video_list, video_path_prefix, res, i)
                if index == 0:
                    origin_video = copy.deepcopy(video_list)
                index += 1

            res.images = origin_video

        def _save(
            self,
            params: AnimateDiffProcess,
            video_list: list,
            video_path_prefix: str,
            res: Processed,
            index: int,
        ):
            video_paths = []
            video_array = [np.array(v) for v in video_list]
            infotext = None
            use_infotext = shared.opts.enable_pnginfo and infotext is not None
            if "PNG" in params.format and shared.opts.data.get("animatediff_save_to_custom", False):
                Path(video_path_prefix).mkdir(exist_ok=True, parents=True)
                for i, frame in enumerate(video_list):
                    png_filename = f"{video_path_prefix}/{i:05}.png"
                    png_info = PngImagePlugin.PngInfo()
                    png_info.add_text("parameters", "")
                    imageio.imwrite(png_filename, frame, pnginfo=png_info)

            if "GIF" in params.format:
                video_path_gif = video_path_prefix + ".gif"
                video_paths.append(video_path_gif)
                if shared.opts.data.get("animatediff_optimize_gif_palette", False):
                    try:
                        pass
                    except ImportError:
                        from launch import run_pip

                        run_pip(
                            "install imageio[pyav]",
                            "sd-webui-animatediff GIF palette optimization requirement: imageio[pyav]",
                        )
                    imageio.imwrite(
                        video_path_gif,
                        video_array,
                        plugin="pyav",
                        fps=params.fps,
                        codec="gif",
                        out_pixel_format="pal8",
                        filter_graph=(
                            {"split": ("split", ""), "palgen": ("palettegen", ""), "paluse": ("paletteuse", ""), "scale": ("scale", f"{video_list[0].width}:{video_list[0].height}")},
                            [
                                ("video_in", "scale", 0, 0),
                                ("scale", "split", 0, 0),
                                ("split", "palgen", 1, 0),
                                ("split", "paluse", 0, 0),
                                ("palgen", "paluse", 0, 1),
                                ("paluse", "video_out", 0, 0),
                            ],
                        ),
                    )
                    # imageio[pyav].imwrite doesn't support comment parameter
                    if use_infotext:
                        try:
                            import exiftool
                        except ImportError:
                            from launch import run_pip

                            run_pip(
                                "install PyExifTool",
                                "sd-webui-animatediff GIF palette optimization requirement: PyExifTool",
                            )
                            import exiftool
                        finally:
                            try:
                                exif_tool = exiftool.ExifTool()
                                with exif_tool:
                                    escaped_infotext = infotext.replace("\n", r"\n")
                                    exif_tool.execute("-overwrite_original", f"-Comment={escaped_infotext}", video_path_gif)
                            except FileNotFoundError:
                                logger.warn("exiftool not found, required for infotext with optimized GIF palette, try: apt install libimage-exiftool-perl or https://exiftool.org/")
                else:
                    imageio.imwrite(video_path_gif, video_array, plugin="pillow", duration=(1000 / params.fps), loop=params.loop_number, comment=(infotext if use_infotext else ""))
                if shared.opts.data.get("animatediff_optimize_gif_gifsicle", False):
                    self._optimize_gif(video_path_gif)
            if "MP4" in params.format:
                video_path_mp4 = video_path_prefix + ".mp4"
                video_paths.append(video_path_mp4)
                try:
                    imageio.imwrite(video_path_mp4, video_array, fps=params.fps, codec="h264")
                except Exception:
                    from launch import run_pip

                    run_pip(
                        "install imageio[ffmpeg]",
                        "sd-webui-animatediff save mp4 requirement: imageio[ffmpeg]",
                    )
                    imageio.imwrite(video_path_mp4, video_array, fps=params.fps, codec="h264")
            if "TXT" in params.format and res.images[index].info is not None:
                video_path_txt = video_path_prefix + ".txt"
                self._save_txt(video_path_txt, infotext)
            if "WEBP" in params.format:
                if PIL.features.check("webp_anim"):
                    video_path_webp = video_path_prefix + ".webp"
                    video_paths.append(video_path_webp)
                    exif_bytes = b""
                    if use_infotext:
                        exif_bytes = piexif.dump({"Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(infotext, encoding="unicode")}})
                    lossless = shared.opts.data.get("animatediff_webp_lossless", False)
                    quality = shared.opts.data.get("animatediff_webp_quality", 80)
                    logger.info(f"Saving {video_path_webp} with lossless={lossless} and quality={quality}")
                    imageio.imwrite(video_path_webp, video_array, plugin="pillow", duration=int(1 / params.fps * 1000), loop=params.loop_number, lossless=lossless, quality=quality, exif=exif_bytes)
                    # see additional Pillow WebP options at https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#webp
                else:
                    logger.warn("WebP animation in Pillow requires system WebP library v0.5.0 or later")
            return video_paths

    class AnimateDiffMM(AnimateDiffMM):
        def _load(self, model_name):
            from .animatediff.motion_module import (MotionModuleType,
                                                    MotionWrapper)

            model_path = os.path.join(self.script_dir, model_name)
            if not os.path.isfile(model_path):
                raise RuntimeError("Please download models manually.")
            if self.mm is None or self.mm.mm_name != model_name:
                logger.info(f"Loading motion module {model_name} from {model_path}")
                model_hash = hashes.sha256(model_path, f"AnimateDiff/{model_name}")
                mm_state_dict = sd_models.read_state_dict(model_path)
                model_type = MotionModuleType.get_mm_type(mm_state_dict)
                logger.info(f"Guessed {model_name} architecture: {model_type}")
                self.mm = MotionWrapper(model_name, model_hash, model_type)
                missed_keys = self.mm.load_state_dict(mm_state_dict)
                logger.warn(f"Missing keys {missed_keys}")
            self.mm.to(device).eval()
            if not shared.cmd_opts.no_half:
                self.mm.half()

    motion_module = AnimateDiffMM()
    motion_module.set_script_dir(easyphoto_models_path)

    class AnimateDiffUiGroup:
        txt2img_submit_button = None
        img2img_submit_button = None

        def __init__(self):
            self.params = AnimateDiffProcess()

        def render(self):
            return gr.State(value=AnimateDiffProcess)

    class AnimateDiffProcess:
        def __init__(
            self,
            model="mm_sd_v15_v2.ckpt",
            enable=False,
            video_length=0,
            fps=8,
            loop_number=0,
            closed_loop="R-P",
            batch_size=16,
            stride=1,
            overlap=-1,
            format=["GIF", "PNG"],
            interp="Off",
            interp_x=10,
            video_source=None,
            video_path="",
            latent_power=1,
            latent_scale=32,
            last_frame=None,
            latent_power_last=1,
            latent_scale_last=32,
            i2i_reserve_scale=1,
        ):
            self.model = model
            self.enable = enable
            self.video_length = video_length
            self.fps = fps
            self.loop_number = loop_number
            self.closed_loop = closed_loop
            self.batch_size = batch_size
            self.stride = stride
            self.overlap = overlap
            self.format = format
            self.interp = interp
            self.interp_x = interp_x
            self.video_source = video_source
            self.video_path = video_path
            self.latent_power = latent_power
            self.latent_scale = latent_scale
            self.last_frame = last_frame
            self.latent_power_last = latent_power_last
            self.latent_scale_last = latent_scale_last
            self.i2i_reserve_scale = i2i_reserve_scale

        def get_list(self, is_img2img: bool):
            list_var = list(vars(self).values())
            if is_img2img:
                animatediff_i2ibatch.hack()
            else:
                list_var = list_var[:-5]
            return list_var

        def get_dict(self, is_img2img: bool):
            infotext = {
                "enable": self.enable,
                "model": self.model,
                "video_length": self.video_length,
                "fps": self.fps,
                "loop_number": self.loop_number,
                "closed_loop": self.closed_loop,
                "batch_size": self.batch_size,
                "stride": self.stride,
                "overlap": self.overlap,
                "interp": self.interp,
                "interp_x": self.interp_x,
            }
            if motion_module.mm is not None and motion_module.mm.mm_hash is not None:
                infotext["mm_hash"] = motion_module.mm.mm_hash[:8]
            if is_img2img:
                infotext.update(
                    {
                        "latent_power": self.latent_power,
                        "latent_scale": self.latent_scale,
                        "latent_power_last": self.latent_power_last,
                        "latent_scale_last": self.latent_scale_last,
                    }
                )
            infotext_str = ", ".join(f"{k}: {v}" for k, v in infotext.items())
            return infotext_str

        def _check(self):
            assert self.video_length >= 0 and self.fps > 0, "Video length and FPS should be positive."
            assert not set(["GIF", "MP4", "PNG", "WEBP"]).isdisjoint(self.format), "At least one saving format should be selected."

        def set_p(self, p: StableDiffusionProcessing):
            self._check()
            if self.video_length < self.batch_size:
                p.batch_size = self.batch_size
            else:
                p.batch_size = self.video_length
            if self.video_length == 0:
                self.video_length = p.batch_size
                self.video_default = True
            else:
                self.video_default = False
            if self.overlap == -1:
                self.overlap = self.batch_size // 4
            if "PNG" not in self.format or shared.opts.data.get("animatediff_save_to_custom", False):
                p.do_not_save_samples = True

    class AnimateDiffI2VLatent:
        def randomize(self, p: StableDiffusionProcessingImg2Img, params: AnimateDiffProcess):
            # Get init_alpha
            step = (1 - params.i2i_reserve_scale) / (params.video_length - 1)
            reserve_scale = [1 - i * step for i in range(params.video_length)]

            logger.info(f"Randomizing reserve_scale according to {reserve_scale}.")
            reserve_scale = torch.tensor(reserve_scale, dtype=torch.float32, device=device)[:, None, None, None]
            reserve_scale[reserve_scale < 0] = 0

            if params.last_frame is not None:
                init_alpha = [1 - pow(i, params.latent_power) / params.latent_scale for i in range(params.video_length)]
                logger.info(f"Randomizing init_latent according to {init_alpha}.")
                init_alpha = torch.tensor(init_alpha, dtype=torch.float32, device=device)[:, None, None, None]
                init_alpha[init_alpha < 0] = 0

                last_frame = params.last_frame
                if type(last_frame) == str:
                    from modules.api.api import decode_base64_to_image

                    last_frame = decode_base64_to_image(last_frame)
                # Get last_alpha
                last_alpha = [1 - pow(i, params.latent_power_last) / params.latent_scale_last for i in range(params.video_length)]
                last_alpha.reverse()
                logger.info(f"Randomizing last_latent according to {last_alpha}.")
                last_alpha = torch.tensor(last_alpha, dtype=torch.float32, device=device)[:, None, None, None]
                last_alpha[last_alpha < 0] = 0

                # Normalize alpha
                sum_alpha = init_alpha + last_alpha
                mask_alpha = sum_alpha > 1
                scaling_factor = 1 / sum_alpha[mask_alpha]
                init_alpha[mask_alpha] *= scaling_factor
                last_alpha[mask_alpha] *= scaling_factor
                init_alpha[0] = 1
                init_alpha[-1] = 0
                last_alpha[0] = 0
                last_alpha[-1] = 1

                # Calculate last_latent
                if p.resize_mode != 3:
                    last_frame = images.resize_image(p.resize_mode, last_frame, p.width, p.height)
                    last_frame = np.array(last_frame).astype(np.float32) / 255.0
                    last_frame = np.moveaxis(last_frame, 2, 0)[None, ...]
                last_frame = torch.from_numpy(last_frame).to(device).to(dtype_vae)
                last_latent = images_tensor_to_samples(
                    last_frame,
                    approximation_indexes.get(shared.opts.sd_vae_encode_method),
                    p.sd_model,
                )
                torch_gc()
                if p.resize_mode == 3:
                    opt_f = 8
                    last_latent = torch.nn.functional.interpolate(
                        last_latent,
                        size=(p.height // opt_f, p.width // opt_f),
                        mode="bilinear",
                    )
                reserve_scale[0] = 1
                reserve_scale[-1] = 1
                # Modify init_latent
                p.init_latent = (p.init_latent * init_alpha + last_latent * last_alpha + p.rng.next() * (1 - init_alpha - last_alpha)) * reserve_scale + p.rng.next() * (1 - reserve_scale)
            else:
                p.init_latent = p.init_latent * reserve_scale + p.rng.next() * (1 - reserve_scale)
