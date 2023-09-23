import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import modules
import modules.scripts as scripts
import numpy as np
from modules import processing, scripts, sd_models, sd_samplers, shared, sd_vae
from modules.api.models import *
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.sd_models import get_closet_checkpoint_match, load_model
from modules.sd_vae import find_vae_near_checkpoint
from modules.shared import opts, state

output_pic_dir = os.path.join(os.path.dirname(__file__), "online_files/output")

InputImage = Union[np.ndarray, str]
InputImage = Union[Dict[str, InputImage], Tuple[InputImage, InputImage], InputImage]

class ControlMode(Enum):
    """
    The improved guess mode.
    """

    BALANCED = "Balanced"
    PROMPT = "My prompt is more important"
    CONTROL = "ControlNet is more important"

class ResizeMode(Enum):
    """
    Resize modes for ControlNet input images.
    """

    RESIZE = "Just Resize"
    INNER_FIT = "Crop and Resize"
    OUTER_FIT = "Resize and Fill"

    def int_value(self):
        if self == ResizeMode.RESIZE:
            return 0
        elif self == ResizeMode.INNER_FIT:
            return 1
        elif self == ResizeMode.OUTER_FIT:
            return 2
        assert False, "NOTREACHED"

class ControlNetUnit:
    """
    Represents an entire ControlNet processing unit.
    """

    def __init__(
        self,
        enabled: bool=True,
        module: Optional[str]=None,
        model: Optional[str]=None,
        weight: float=1.0,
        image: Optional[InputImage]=None,
        resize_mode: Union[ResizeMode, int, str] = ResizeMode.INNER_FIT,
        low_vram: bool=False,
        processor_res: int=-1,
        threshold_a: float=-1,
        threshold_b: float=-1,
        guidance_start: float=0.0,
        guidance_end: float=1.0,
        pixel_perfect: bool=False,
        control_mode: Union[ControlMode, int, str] = ControlMode.BALANCED,
        **_kwargs,
    ):
        self.enabled = enabled
        self.module = module
        self.model = model
        self.weight = weight
        self.image = image
        self.resize_mode = resize_mode
        self.low_vram = low_vram
        self.processor_res = processor_res
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.guidance_start = guidance_start
        self.guidance_end = guidance_end
        self.pixel_perfect = pixel_perfect
        self.control_mode = control_mode

    def __eq__(self, other):
        if not isinstance(other, ControlNetUnit):
            return False

        return vars(self) == vars(other)

def find_cn_script(script_runner: scripts.ScriptRunner) -> Optional[scripts.Script]:
    """
    Find the ControlNet script in `script_runner`. Returns `None` if `script_runner` does not contain a ControlNet script.
    """

    if script_runner is None:
        return None

    for script in script_runner.alwayson_scripts:
        if is_cn_script(script):
            return script

def update_cn_script_in_place(
    script_runner: scripts.ScriptRunner,
    script_args: List[Any],
    cn_units: List[ControlNetUnit],
    **_kwargs, # for backwards compatibility
):
    """
    Update the arguments of the ControlNet script in `script_args` in place, reading from `cn_units`.
    `cn_units` and its elements are not modified. You can call this function repeatedly, as many times as you want.

    Does not update `script_args` if any of the folling is true:
    - ControlNet is not present in `script_runner`
    - `script_args` is not filled with script arguments for scripts that are processed before ControlNet
    """

    cn_script = find_cn_script(script_runner)
    if cn_script is None or len(script_args) < cn_script.args_from:
        return

    # fill in remaining parameters to satisfy max models, just in case script needs it.
    max_models = shared.opts.data.get("control_net_max_models_num", 1)
    cn_units = cn_units + [ControlNetUnit(enabled=False)] * max(max_models - len(cn_units), 0)

    cn_script_args_diff = 0
    for script in script_runner.alwayson_scripts:
        if script is cn_script:
            cn_script_args_diff = len(cn_units) - (cn_script.args_to - cn_script.args_from)
            script_args[script.args_from:script.args_to] = cn_units
            script.args_to = script.args_from + len(cn_units)
        else:
            script.args_from += cn_script_args_diff
            script.args_to += cn_script_args_diff

def update_cn_script_in_processing(
    p: processing.StableDiffusionProcessing,
    cn_units: List[ControlNetUnit],
    **_kwargs, # for backwards compatibility
):
    """
    Update the arguments of the ControlNet script in `p.script_args` in place, reading from `cn_units`.
    `cn_units` and its elements are not modified. You can call this function repeatedly, as many times as you want.

    Does not update `p.script_args` if any of the folling is true:
    - ControlNet is not present in `p.scripts`
    - `p.script_args` is not filled with script arguments for scripts that are processed before ControlNet
    """

    cn_units_type = type(cn_units) if type(cn_units) in (list, tuple) else list
    script_args = list(p.script_args)
    update_cn_script_in_place(p.scripts, script_args, cn_units)
    p.script_args = cn_units_type(script_args)

def is_cn_script(script: scripts.Script) -> bool:
    """
    Determine whether `script` is a ControlNet script.
    """

    return script.title().lower() == 'controlnet'

def init_default_script_args(script_runner):
    #find max idx from the scripts in runner and generate a none array to init script_args
    last_arg_index = 1
    for script in script_runner.scripts:
        if last_arg_index < script.args_to:
            last_arg_index = script.args_to
    # None everywhere except position 0 to initialize script args
    script_args = [None]*last_arg_index
    script_args[0] = 0

    # get default values
    with gr.Blocks(): # will throw errors calling ui function without this
        for script in script_runner.scripts:
            if script.ui(script.is_img2img):
                ui_default_values = []
                for elem in script.ui(script.is_img2img):
                    ui_default_values.append(elem.value)
                script_args[script.args_from:script.args_to] = ui_default_values
    return script_args

def reload_model(k, v):
    opts.set(k, v)
    if k == 'sd_model_checkpoint':
        sd_models.reload_model_weights()
    if k == 'sd_vae':
        sd_vae.reload_vae_weights()

def t2i_call(
        resize_mode=0,
        prompt="",
        styles=[],
        seed=-1,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,

        batch_size=1,
        n_iter=1,
        steps=None,
        cfg_scale=7.0,
        width=640,
        height=768,
        restore_faces=False,
        tiling=False,
        do_not_save_samples=False,
        do_not_save_grid=False,
        negative_prompt="",
        eta=1.0,
        s_churn=0,
        s_tmax=0,
        s_tmin=0,
        s_noise=1,
        override_settings={},
        override_settings_restore_afterwards=True,
        sampler_index=None,  # deprecated: use sampler_name
        include_init_images=False,

        controlnet_units: List[ControlNetUnit] = [],
        use_deprecated_controlnet=False,
        outpath_samples = "",
        sd_vae = None, 
        sd_model_checkpoint = "Chilloutmix-Ni-pruned-fp16-fix.safetensors",
):
    if sampler_index is None:
        sampler_index = 0
    if steps is None:
        steps = 20

    try:
        origin_sd_model_checkpoint  = opts.sd_model_checkpoint
        origin_sd_vae               = opts.sd_vae
    except:
        origin_sd_model_checkpoint  = ""
        origin_sd_vae               = ""

    sd_model_checkpoint = get_closet_checkpoint_match(sd_model_checkpoint).model_name
    if sd_vae is not None:
        sd_vae = os.path.basename(vae_near_checkpoint = find_vae_near_checkpoint(sd_vae))
    else:
        sd_vae = None

    p_txt2img = StableDiffusionProcessingTxt2Img(
        sd_model=origin_sd_model_checkpoint,
        outpath_samples=outpath_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=[],
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        sampler_name=sd_samplers.samplers[sampler_index].name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        override_settings=override_settings
    )

    p_txt2img.scripts = scripts.scripts_img2img
    p_txt2img.script_args = init_default_script_args(p_txt2img.scripts)

    for alwayson_scripts in modules.scripts.scripts_img2img.alwayson_scripts:
        if alwayson_scripts.name is None:
            continue
        if alwayson_scripts.name=='controlnet':
            p_txt2img.script_args[alwayson_scripts.args_from:alwayson_scripts.args_from + len(controlnet_units)] = controlnet_units
    
    if sd_model_checkpoint != origin_sd_model_checkpoint:
        reload_model('sd_model_checkpoint', sd_model_checkpoint)
    
    if origin_sd_vae != sd_vae:
        reload_model('sd_vae', sd_vae)

    processed = processing.process_images(p_txt2img)

    if sd_model_checkpoint != origin_sd_model_checkpoint:
        reload_model('sd_model_checkpoint', origin_sd_model_checkpoint)
    if origin_sd_vae != sd_vae:
        reload_model('sd_vae', origin_sd_vae)

    if len(processed.images) > 1:
        # get the generate image!
        h_0, w_0, c_0 = np.shape(processed.images[0])
        h_1, w_1, c_1 = np.shape(processed.images[1])
        if w_1 != w_0:
            gen_image = processed.images[1]
        else:
            gen_image = processed.images[0]
    else:
        gen_image = processed.images[0]
    return gen_image

def i2i_inpaint_call(
        images=[],  
        resize_mode=0,
        denoising_strength=0.75,
        image_cfg_scale=1.5,
        mask_image=None,  # PIL Image mask
        mask_blur=8,
        inpainting_fill=0,
        inpaint_full_res=True,
        inpaint_full_res_padding=0,
        inpainting_mask_invert=0,
        initial_noise_multiplier=1,
        prompt="",
        styles=[],
        seed=-1,
        subseed=-1,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,

        batch_size=1,
        n_iter=1,
        steps=None,
        cfg_scale=7.0,
        width=640,
        height=768,
        restore_faces=False,
        tiling=False,
        do_not_save_samples=False,
        do_not_save_grid=False,
        negative_prompt="",
        eta=1.0,
        s_churn=0,
        s_tmax=0,
        s_tmin=0,
        s_noise=1,
        override_settings={},
        override_settings_restore_afterwards=True,
        sampler_index=None,  # deprecated: use sampler_name
        include_init_images=False,

        controlnet_units: List[ControlNetUnit] = [],
        use_deprecated_controlnet=False,
        outpath_samples = "",
        sd_vae = "vae-ft-mse-840000-ema-pruned.ckpt", 
        sd_model_checkpoint = "Chilloutmix-Ni-pruned-fp16-fix.safetensors",
):
    if sampler_index is None:
        sampler_index = 0
    if steps is None:
        steps = 20

    try:
        origin_sd_model_checkpoint  = opts.sd_model_checkpoint
        origin_sd_vae               = opts.sd_vae
    except:
        origin_sd_model_checkpoint  = ""
        origin_sd_vae               = ""

    sd_model_checkpoint = get_closet_checkpoint_match(sd_model_checkpoint).model_name
    vae_near_checkpoint = find_vae_near_checkpoint(sd_vae)
    if vae_near_checkpoint is not None:
        sd_vae = os.path.basename(vae_near_checkpoint)
    else:
        sd_vae = None

    p_img2img = StableDiffusionProcessingImg2Img(
        sd_model=origin_sd_model_checkpoint,
        outpath_samples=outpath_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=[],
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        sampler_name=sd_samplers.samplers_for_img2img[sampler_index].name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        init_images=images,
        mask=mask_image,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        override_settings=override_settings,
        initial_noise_multiplier=initial_noise_multiplier
    )

    p_img2img.scripts = scripts.scripts_img2img
    p_img2img.extra_generation_params["Mask blur"] = mask_blur
    p_img2img.script_args = init_default_script_args(p_img2img.scripts)

    for alwayson_scripts in modules.scripts.scripts_img2img.alwayson_scripts:
        if alwayson_scripts.name is None:
            continue
        if alwayson_scripts.name=='controlnet':
            p_img2img.script_args[alwayson_scripts.args_from:alwayson_scripts.args_from + len(controlnet_units)] = controlnet_units
    
    if sd_model_checkpoint != origin_sd_model_checkpoint:
        reload_model('sd_model_checkpoint', sd_model_checkpoint)
    
    if sd_vae is not None:
        if origin_sd_vae != sd_vae:
            reload_model('sd_vae', sd_vae)

    processed = processing.process_images(p_img2img)

    if sd_model_checkpoint != origin_sd_model_checkpoint:
        reload_model('sd_model_checkpoint', origin_sd_model_checkpoint)
    if sd_vae is not None:
        if origin_sd_vae != sd_vae:
            reload_model('sd_vae', origin_sd_vae)

    if len(processed.images) > 1:
        # get the generate image!
        h_0, w_0, c_0 = np.shape(processed.images[0])
        h_1, w_1, c_1 = np.shape(processed.images[1])
        if w_1 != w_0:
            gen_image = processed.images[1]
        else:
            gen_image = processed.images[0]
    else:
        gen_image = processed.images[0]
    return gen_image