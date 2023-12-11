import gc
import os

import torch
from einops import rearrange
from modules import hashes, shared, sd_models, devices
from modules.devices import cpu, device, torch_gc

from .motion_module import MotionWrapper, MotionModuleType
from .animatediff_logger import logger_animatediff as logger


class AnimateDiffMM:
    mm_injected = False

    def __init__(self):
        self.mm: MotionWrapper = None
        self.script_dir = None
        self.prev_alpha_cumprod = None
        self.gn32_original_forward = None


    def set_script_dir(self, script_dir):
        self.script_dir = script_dir


    def get_model_dir(self):
        model_dir = shared.opts.data.get("animatediff_model_path", os.path.join(self.script_dir, "model"))
        if not model_dir:
            model_dir = os.path.join(self.script_dir, "model")
        return model_dir


    def _load(self, model_name):
        model_path = os.path.join(self.get_model_dir(), model_name)
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
            if getattr(devices, "fp8", False):
                for module in self.mm.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                        module.to(torch.float8_e4m3fn)


    def inject(self, sd_model, model_name="mm_sd_v15.ckpt"):
        if AnimateDiffMM.mm_injected:
            logger.info("Motion module already injected. Trying to restore.")
            self.restore(sd_model)

        unet = sd_model.model.diffusion_model
        self._load(model_name)
        inject_sdxl = sd_model.is_sdxl or self.mm.is_xl
        sd_ver = "SDXL" if sd_model.is_sdxl else "SD1.5"
        assert sd_model.is_sdxl == self.mm.is_xl, f"Motion module incompatible with SD. You are using {sd_ver} with {self.mm.mm_type}."

        if self.mm.is_v2:
            logger.info(f"Injecting motion module {model_name} into {sd_ver} UNet middle block.")
            unet.middle_block.insert(-1, self.mm.mid_block.motion_modules[0])
        elif not self.mm.is_adxl:
            logger.info(f"Hacking {sd_ver} GroupNorm32 forward function.")
            if self.mm.is_hotshot:
                from sgm.modules.diffusionmodules.util import GroupNorm32
            else:
                from ldm.modules.diffusionmodules.util import GroupNorm32
            self.gn32_original_forward = GroupNorm32.forward
            gn32_original_forward = self.gn32_original_forward

            def groupnorm32_mm_forward(self, x):
                x = rearrange(x, "(b f) c h w -> b c f h w", b=2)
                x = gn32_original_forward(self, x)
                x = rearrange(x, "b c f h w -> (b f) c h w", b=2)
                return x

            GroupNorm32.forward = groupnorm32_mm_forward

        logger.info(f"Injecting motion module {model_name} into {sd_ver} UNet input blocks.")
        for mm_idx, unet_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11]):
            if inject_sdxl and mm_idx >= 6:
                break
            mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
            mm_inject = getattr(self.mm.down_blocks[mm_idx0], "temporal_attentions" if self.mm.is_hotshot else "motion_modules")[mm_idx1]
            unet.input_blocks[unet_idx].append(mm_inject)

        logger.info(f"Injecting motion module {model_name} into {sd_ver} UNet output blocks.")
        for unet_idx in range(12):
            if inject_sdxl and unet_idx >= 9:
                break
            mm_idx0, mm_idx1 = unet_idx // 3, unet_idx % 3
            mm_inject = getattr(self.mm.up_blocks[mm_idx0], "temporal_attentions" if self.mm.is_hotshot else "motion_modules")[mm_idx1]
            if unet_idx % 3 == 2 and unet_idx != (8 if self.mm.is_xl else 11):
                unet.output_blocks[unet_idx].insert(-1, mm_inject)
            else:
                unet.output_blocks[unet_idx].append(mm_inject)

        self._set_ddim_alpha(sd_model)
        self._set_layer_mapping(sd_model)
        AnimateDiffMM.mm_injected = True
        logger.info(f"Injection finished.")


    def restore(self, sd_model):
        if not AnimateDiffMM.mm_injected:
            logger.info("Motion module already removed.")
            return

        inject_sdxl = sd_model.is_sdxl or self.mm.is_xl
        sd_ver = "SDXL" if sd_model.is_sdxl else "SD1.5"
        self._restore_ddim_alpha(sd_model)
        unet = sd_model.model.diffusion_model

        logger.info(f"Removing motion module from {sd_ver} UNet input blocks.")
        for unet_idx in [1, 2, 4, 5, 7, 8, 10, 11]:
            if inject_sdxl and unet_idx >= 9:
                break
            unet.input_blocks[unet_idx].pop(-1)

        logger.info(f"Removing motion module from {sd_ver} UNet output blocks.")
        for unet_idx in range(12):
            if inject_sdxl and unet_idx >= 9:
                break
            if unet_idx % 3 == 2 and unet_idx != (8 if self.mm.is_xl else 11):
                unet.output_blocks[unet_idx].pop(-2)
            else:
                unet.output_blocks[unet_idx].pop(-1)

        if self.mm.is_v2:
            logger.info(f"Removing motion module from {sd_ver} UNet middle block.")
            unet.middle_block.pop(-2)
        elif not self.mm.is_adxl:
            logger.info(f"Restoring {sd_ver} GroupNorm32 forward function.")
            if self.mm.is_hotshot:
                from sgm.modules.diffusionmodules.util import GroupNorm32
            else:
                from ldm.modules.diffusionmodules.util import GroupNorm32
            GroupNorm32.forward = self.gn32_original_forward
            self.gn32_original_forward = None

        AnimateDiffMM.mm_injected = False
        logger.info(f"Removal finished.")
        if shared.cmd_opts.lowvram:
            self.unload()


    def _set_ddim_alpha(self, sd_model):
        logger.info(f"Setting DDIM alpha.")
        beta_start = 0.00085
        beta_end = 0.020 if self.mm.is_adxl else 0.012
        if self.mm.is_adxl:
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, 1000, dtype=torch.float32, device=device) ** 2
        else:
            betas = torch.linspace(
                beta_start,
                beta_end,
                1000 if sd_model.is_sdxl else sd_model.num_timesteps,
                dtype=torch.float32,
                device=device,
            )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.prev_alpha_cumprod = sd_model.alphas_cumprod
        sd_model.alphas_cumprod = alphas_cumprod
    

    def _set_layer_mapping(self, sd_model):
        if hasattr(sd_model, 'network_layer_mapping'):
            for name, module in self.mm.named_modules():
                sd_model.network_layer_mapping[name] = module
                module.network_layer_name = name


    def _restore_ddim_alpha(self, sd_model):
        logger.info(f"Restoring DDIM alpha.")
        sd_model.alphas_cumprod = self.prev_alpha_cumprod
        self.prev_alpha_cumprod = None


    def unload(self):
        logger.info("Moving motion module to CPU")
        if self.mm is not None:
            self.mm.to(cpu)
        torch_gc()
        gc.collect()


    def remove(self):
        logger.info("Removing motion module from any memory")
        del self.mm
        self.mm = None
        torch_gc()
        gc.collect()


mm_animatediff = AnimateDiffMM()
