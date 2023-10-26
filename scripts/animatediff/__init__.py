from modules import processing, scripts, sd_models, sd_samplers, sd_vae, shared
from modules.api.models import *
from modules.processing import (Processed, StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img,
                                StableDiffusionProcessingTxt2Img)
from modules.sd_models import get_closet_checkpoint_match, load_model
from modules.sd_vae import find_vae_near_checkpoint
from modules.shared import opts, state
from scripts.animatediff.animatediff_cn import AnimateDiffControl
from scripts.animatediff.animatediff_infotext import update_infotext
from scripts.animatediff.animatediff_infv2v import AnimateDiffInfV2V
from scripts.animatediff.animatediff_latent import AnimateDiffI2VLatent
from scripts.animatediff.animatediff_lora import AnimateDiffLora
from scripts.animatediff.animatediff_mm import mm_animatediff as motion_module
from scripts.animatediff.animatediff_output import AnimateDiffOutput
from scripts.animatediff.animatediff_prompt import AnimateDiffPromptSchedule
from scripts.animatediff.animatediff_ui import AnimateDiffProcess

class AnimateDiffScript():
    def __init__(self):
        self.lora_hacker = None
        self.cfg_hacker = None
        self.cn_hacker = None
        self.prompt_scheduler = None

    def before_process(self, p: processing.StableDiffusionProcessing, params: AnimateDiffProcess):
        if isinstance(params, dict): params = AnimateDiffProcess(**params)
        params.set_p(p)
        motion_module.inject(p.sd_model, params.model)
        self.prompt_scheduler = AnimateDiffPromptSchedule()
        self.lora_hacker = AnimateDiffLora(motion_module.mm.using_v2)
        self.lora_hacker.hack()
        self.cfg_hacker = AnimateDiffInfV2V(p, self.prompt_scheduler)
        self.cfg_hacker.hack(params)
        self.cn_hacker = AnimateDiffControl(p, self.prompt_scheduler)
        self.cn_hacker.hack(params)
        update_infotext(p, params)

    def before_process_batch(self, p: processing.StableDiffusionProcessing, params: AnimateDiffProcess, **kwargs):
        if isinstance(params, dict): params = AnimateDiffProcess(**params)
        if isinstance(p, StableDiffusionProcessingImg2Img):
            AnimateDiffI2VLatent().randomize(p, params)

    def postprocess(self, p: processing.StableDiffusionProcessing, res: Processed, params: AnimateDiffProcess):
        if isinstance(params, dict): params = AnimateDiffProcess(**params)
        self.prompt_scheduler.set_infotext(res)
        self.cn_hacker.restore()
        self.cfg_hacker.restore()
        self.lora_hacker.restore()
        motion_module.restore(p.sd_model)
        AnimateDiffOutput().output(p, res, params)