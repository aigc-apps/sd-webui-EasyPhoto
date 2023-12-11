import gradio as gr
from modules import script_callbacks, scripts, shared
from modules.processing import (Processed, StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img)
from modules.scripts import PostprocessBatchListArgs, PostprocessImageArgs

from scripts.animatediff_cn import AnimateDiffControl
from scripts.animatediff_infv2v import AnimateDiffInfV2V
from scripts.animatediff_latent import AnimateDiffI2VLatent
from scripts.animatediff_logger import logger_animatediff as logger
from scripts.animatediff_lora import AnimateDiffLora
from scripts.animatediff_mm import mm_animatediff as motion_module
from scripts.animatediff_prompt import AnimateDiffPromptSchedule
from scripts.animatediff_output import AnimateDiffOutput
from scripts.animatediff_ui import AnimateDiffProcess, AnimateDiffUiGroup
from scripts.animatediff_infotext import update_infotext

script_dir = scripts.basedir()
motion_module.set_script_dir(script_dir)


class AnimateDiffScript(scripts.Script):

    def __init__(self):
        self.lora_hacker = None
        self.cfg_hacker = None
        self.cn_hacker = None
        self.prompt_scheduler = None
        self.hacked = False


    def title(self):
        return "AnimateDiff"


    def show(self, is_img2img):
        return scripts.AlwaysVisible


    def ui(self, is_img2img):
        return (AnimateDiffUiGroup().render(is_img2img, motion_module.get_model_dir()),)


    def before_process(self, p: StableDiffusionProcessing, params: AnimateDiffProcess):
        if p.is_api and isinstance(params, dict):
            self.ad_params = AnimateDiffProcess(**params)
            params = self.ad_params
        if params.enable:
            logger.info("AnimateDiff process start.")
            params.set_p(p)
            motion_module.inject(p.sd_model, params.model)
            self.prompt_scheduler = AnimateDiffPromptSchedule()
            self.lora_hacker = AnimateDiffLora(motion_module.mm.is_v2)
            self.lora_hacker.hack()
            self.cfg_hacker = AnimateDiffInfV2V(p, self.prompt_scheduler)
            self.cfg_hacker.hack(params)
            self.cn_hacker = AnimateDiffControl(p, self.prompt_scheduler)
            self.cn_hacker.hack(params)
            update_infotext(p, params)
            self.hacked = True
        elif self.hacked:
            self.cn_hacker.restore()
            self.cfg_hacker.restore()
            self.lora_hacker.restore()
            motion_module.restore(p.sd_model)
            self.hacked = False


    def before_process_batch(self, p: StableDiffusionProcessing, params: AnimateDiffProcess, **kwargs):
        if p.is_api and isinstance(params, dict): params = self.ad_params
        if params.enable and isinstance(p, StableDiffusionProcessingImg2Img) and not hasattr(p, '_animatediff_i2i_batch'):
            AnimateDiffI2VLatent().randomize(p, params)


    def postprocess_batch_list(self, p: StableDiffusionProcessing, pp: PostprocessBatchListArgs, params: AnimateDiffProcess, **kwargs):
        if p.is_api and isinstance(params, dict): params = self.ad_params
        if params.enable:
            self.prompt_scheduler.save_infotext_img(p)


    def postprocess_image(self, p: StableDiffusionProcessing, pp: PostprocessImageArgs, params: AnimateDiffProcess, *args):
        if p.is_api and isinstance(params, dict): params = self.ad_params
        if params.enable and isinstance(p, StableDiffusionProcessingImg2Img) and hasattr(p, '_animatediff_paste_to_full'):
            p.paste_to = p._animatediff_paste_to_full[p.batch_index]


    def postprocess(self, p: StableDiffusionProcessing, res: Processed, params: AnimateDiffProcess):
        if p.is_api and isinstance(params, dict): params = self.ad_params
        if params.enable:
            self.prompt_scheduler.save_infotext_txt(res)
            self.cn_hacker.restore()
            self.cfg_hacker.restore()
            self.lora_hacker.restore()
            motion_module.restore(p.sd_model)
            self.hacked = False
            AnimateDiffOutput().output(p, res, params)
            logger.info("AnimateDiff process end.")


def on_ui_settings():
    section = ("animatediff", "AnimateDiff")
    s3_selection =("animatediff", "AnimateDiff AWS") 
    shared.opts.add_option(
        "animatediff_model_path",
        shared.OptionInfo(
            None,
            "Path to save AnimateDiff motion modules",
            gr.Textbox,
            section=section,
        ),
    )
    shared.opts.add_option(
        "animatediff_optimize_gif_palette",
        shared.OptionInfo(
            False,
            "Calculate the optimal GIF palette, improves quality significantly, removes banding",
            gr.Checkbox,
            section=section
        )
    )
    shared.opts.add_option(
        "animatediff_optimize_gif_gifsicle",
        shared.OptionInfo(
            False,
            "Optimize GIFs with gifsicle, reduces file size",
            gr.Checkbox,
            section=section
        )
    )
    shared.opts.add_option(
        key="animatediff_mp4_crf",
        info=shared.OptionInfo(
            default=23,
            label="MP4 Quality (CRF)",
            component=gr.Slider,
            component_args={
                "minimum": 0,
                "maximum": 51,
                "step": 1},
            section=section
        )
        .link("docs", "https://trac.ffmpeg.org/wiki/Encode/H.264#crf")
        .info("17 for best quality, up to 28 for smaller size")
    )
    shared.opts.add_option(
        key="animatediff_mp4_preset",
        info=shared.OptionInfo(
            default="",
            label="MP4 Encoding Preset",
            component=gr.Dropdown,
            component_args={"choices": ["", 'veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast', 'superfast', 'ultrafast']},
            section=section,
        )
        .link("docs", "https://trac.ffmpeg.org/wiki/Encode/H.264#Preset")
        .info("encoding speed, use the slowest you can tolerate")
    )
    shared.opts.add_option(
        key="animatediff_mp4_tune",
        info=shared.OptionInfo(
            default="",
            label="MP4 Tune encoding for content type",
            component=gr.Dropdown,
            component_args={"choices": ["", "film", "animation", "grain"]},
            section=section
        )
        .link("docs", "https://trac.ffmpeg.org/wiki/Encode/H.264#Tune")
        .info("optimize for specific content types")
    )
    shared.opts.add_option(
        "animatediff_webp_quality",
        shared.OptionInfo(
            80,
            "WebP Quality (if lossless=True, increases compression and CPU usage)",
            gr.Slider,
            {
                "minimum": 1,
                "maximum": 100,
                "step": 1},
            section=section
        )
    )
    shared.opts.add_option(
        "animatediff_webp_lossless",
        shared.OptionInfo(
            False,
            "Save WebP in lossless format (highest quality, largest file size)",
            gr.Checkbox,
            section=section
        )
    )
    shared.opts.add_option(
        "animatediff_save_to_custom",
        shared.OptionInfo(
            False,
            "Save frames to stable-diffusion-webui/outputs/{ txt|img }2img-images/AnimateDiff/{gif filename}/{date} "
            "instead of stable-diffusion-webui/outputs/{ txt|img }2img-images/{date}/.",
            gr.Checkbox,
            section=section
        )
    )
    shared.opts.add_option(
        "animatediff_xformers",
        shared.OptionInfo(
            "Optimize attention layers with xformers",
            "When you have --xformers in your command line args, you want AnimateDiff to ",
            gr.Radio,
            {"choices": ["Optimize attention layers with xformers",
                         "Optimize attention layers with sdp (torch >= 2.0.0 required)",
                         "Do not optimize attention layers"]},
            section=section
        )
    )
    shared.opts.add_option(
        "animatediff_s3_enable",
        shared.OptionInfo(
            False,
            "Enable to Store file in object storage that supports the s3 protocol",
            gr.Checkbox,
            section=s3_selection
        )
    )
    shared.opts.add_option(
        "animatediff_s3_host",
        shared.OptionInfo(
            None,
            "S3 protocol host",
            gr.Textbox,
            section=s3_selection,
        ),
    )
    shared.opts.add_option(
        "animatediff_s3_port",
        shared.OptionInfo(
            None,
            "S3 protocol port",
            gr.Textbox,
            section=s3_selection,
        ),
    )
    shared.opts.add_option(
        "animatediff_s3_access_key",
        shared.OptionInfo(
            None,
            "S3 protocol access_key",
            gr.Textbox,
            section=s3_selection,
        ),
    )
    shared.opts.add_option(
        "animatediff_s3_secret_key",
        shared.OptionInfo(
            None,
            "S3 protocol secret_key",
            gr.Textbox,
            section=s3_selection,
        ),
    )
    shared.opts.add_option(
        "animatediff_s3_storge_bucket",
        shared.OptionInfo(
            None,
            "Bucket for file storage",
            gr.Textbox,
            section=s3_selection,
        ),
    )    
    
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(AnimateDiffUiGroup.on_after_component)
script_callbacks.on_before_ui(AnimateDiffUiGroup.on_before_ui)
