import gc
import json
import os

import torch
from einops import rearrange
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (TimestepBlock,
                                                      TimestepEmbedSequential)
from ldm.modules.diffusionmodules.util import GroupNorm32
from modules import hashes, shared, sd_models
from modules.devices import cpu, device, torch_gc

# from scripts.animatediff_mm import mm_animatediff as motion_module
# from scripts.animatediff_logger import logger_animatediff as logger

try:
    from scripts.animatediff_logger import logger_animatediff as logger
    from motion_module import MotionWrapper, VanillaTemporalModule
except ImportError:
    from scripts.animatediff.animatediff_logger import logger_animatediff as logger
    from scripts.animatediff.motion_module  import MotionWrapper, VanillaTemporalModule


class AnimateDiffMM:
    def __init__(self):
        self.mm: MotionWrapper = None
        self.script_dir = None
        self.prev_beta = None
        self.prev_alpha_cumprod = None
        self.prev_alpha_cumprod_prev = None
        self.gn32_original_forward = None
        self.tes_original_forward = None


    def set_script_dir(self, script_dir):
        self.script_dir = script_dir

    def _load(self, model_name):
        model_path = os.path.join(self.script_dir, model_name)
        model_hash, using_v2, guess = self._hash(model_path, model_name)
        if not os.path.isfile(model_path):
            raise RuntimeError("Please download models manually.")
        if self.mm is None or self.mm.mm_hash != model_hash:
            logger.info(f"Loading motion module {model_name} from {model_path}")
            mm_state_dict = sd_models.read_state_dict(model_path)
            if guess:
                using_v2 = "mid_block.motion_modules.0.temporal_transformer.proj_out.bias" in mm_state_dict.keys()
                logger.warn(f"Guessed mm architecture : {'v2' if using_v2 else 'v1'}")
            self.mm = MotionWrapper(model_hash, using_v2)
            missed_keys = self.mm.load_state_dict(mm_state_dict)
            logger.warn(f"Missing keys {missed_keys}")
        self.mm.to(device).eval()
        if not shared.cmd_opts.no_half:
            self.mm.half()

    def _hash(self, model_path: str, model_name="mm_sd_v15.ckpt"):
        model_hash = hashes.sha256(model_path, f"AnimateDiff/{model_name}")
        with open(os.path.join(self.script_dir, "mm_zoo.json"), "r") as f:
            model_zoo = json.load(f)
        if model_hash in model_zoo:
            model_official_name = model_zoo[model_hash]["name"]
            logger.info(f"You are using tested mm {model_official_name}.")
            return model_hash, model_zoo[model_hash]["arch"] == 2, False
        else:
            logger.warn(
                f"You are using unknown mm {model_name}. "
                "Either your download is incomplete or your model has not been tested. "
                "Please use at your own risk. "
                "AnimateDiff will guess mm architecture via state_dict."
            )
            return model_hash, False, True


    def inject(self, sd_model, model_name="mm_sd_v15.ckpt"):
        unet = sd_model.model.diffusion_model
        self._load(model_name)
        self.gn32_original_forward = GroupNorm32.forward
        self.tes_original_forward = TimestepEmbedSequential.forward
        gn32_original_forward = self.gn32_original_forward

        def mm_tes_forward(self, x, emb, context=None):
            for layer in self:
                if isinstance(layer, TimestepBlock):
                    x = layer(x, emb)
                elif isinstance(layer, (SpatialTransformer, VanillaTemporalModule)):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        TimestepEmbedSequential.forward = mm_tes_forward
        if self.mm.using_v2:
            logger.info(f"Injecting motion module {model_name} into SD1.5 UNet middle block.")
            unet.middle_block.insert(-1, self.mm.mid_block.motion_modules[0])
        else:
            logger.info(f"Hacking GroupNorm32 forward function.")

            def groupnorm32_mm_forward(self, x):
                x = rearrange(x, "(b f) c h w -> b c f h w", b=2)
                x = gn32_original_forward(self, x)
                x = rearrange(x, "b c f h w -> (b f) c h w", b=2)
                return x

            GroupNorm32.forward = groupnorm32_mm_forward

        logger.info(f"Injecting motion module {model_name} into SD1.5 UNet input blocks.")
        for mm_idx, unet_idx in enumerate([1, 2, 4, 5, 7, 8, 10, 11]):
            mm_idx0, mm_idx1 = mm_idx // 2, mm_idx % 2
            unet.input_blocks[unet_idx].append(
                self.mm.down_blocks[mm_idx0].motion_modules[mm_idx1]
            )

        logger.info(f"Injecting motion module {model_name} into SD1.5 UNet output blocks.")
        for unet_idx in range(12):
            mm_idx0, mm_idx1 = unet_idx // 3, unet_idx % 3
            if unet_idx % 3 == 2 and unet_idx != 11:
                unet.output_blocks[unet_idx].insert(
                    -1, self.mm.up_blocks[mm_idx0].motion_modules[mm_idx1]
                )
            else:
                unet.output_blocks[unet_idx].append(
                    self.mm.up_blocks[mm_idx0].motion_modules[mm_idx1]
                )

        self._set_ddim_alpha(sd_model)
        self._set_layer_mapping(sd_model)
        logger.info(f"Injection finished.")


    def restore(self, sd_model):
        self._restore_ddim_alpha(sd_model)
        unet = sd_model.model.diffusion_model
        logger.info(f"Removing motion module from SD1.5 UNet input blocks.")
        for unet_idx in [1, 2, 4, 5, 7, 8, 10, 11]:
            unet.input_blocks[unet_idx].pop(-1)
        logger.info(f"Removing motion module from SD1.5 UNet output blocks.")
        for unet_idx in range(12):
            if unet_idx % 3 == 2 and unet_idx != 11:
                unet.output_blocks[unet_idx].pop(-2)
            else:
                unet.output_blocks[unet_idx].pop(-1)
        if self.mm.using_v2:
            logger.info(f"Removing motion module from SD1.5 UNet middle block.")
            unet.middle_block.pop(-2)
        else:
            logger.info(f"Restoring GroupNorm32 forward function.")
            GroupNorm32.forward = self.gn32_original_forward
        TimestepEmbedSequential.forward = self.tes_original_forward
        logger.info(f"Removal finished.")
        if shared.cmd_opts.lowvram:
            self.unload()


    def _set_ddim_alpha(self, sd_model):
        logger.info(f"Setting DDIM alpha.")
        beta_start = 0.00085
        beta_end = 0.012
        betas = torch.linspace(
            beta_start,
            beta_end,
            sd_model.num_timesteps,
            dtype=torch.float32,
            device=device,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            (
                torch.tensor([1.0], dtype=torch.float32, device=device),
                alphas_cumprod[:-1],
            )
        )
        self.prev_beta = sd_model.betas
        self.prev_alpha_cumprod = sd_model.alphas_cumprod
        self.prev_alpha_cumprod_prev = sd_model.alphas_cumprod_prev
        sd_model.betas = betas
        sd_model.alphas_cumprod = alphas_cumprod
        sd_model.alphas_cumprod_prev = alphas_cumprod_prev
    

    def _set_layer_mapping(self, sd_model):
        if hasattr(sd_model, 'network_layer_mapping'):
            for name, module in self.mm.named_modules():
                sd_model.network_layer_mapping[name] = module
                module.network_layer_name = name

    def _restore_ddim_alpha(self, sd_model):
        logger.info(f"Restoring DDIM alpha.")
        sd_model.betas = self.prev_beta
        sd_model.alphas_cumprod = self.prev_alpha_cumprod
        sd_model.alphas_cumprod_prev = self.prev_alpha_cumprod_prev
        self.prev_beta = None
        self.prev_alpha_cumprod = None
        self.prev_alpha_cumprod_prev = None


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
