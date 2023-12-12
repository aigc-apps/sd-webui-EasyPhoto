from typing import List

import numpy as np
import torch

from modules import prompt_parser, devices, sd_samplers_common, shared
from modules.shared import opts, state
from modules.script_callbacks import CFGDenoiserParams, cfg_denoiser_callback
from modules.script_callbacks import CFGDenoisedParams, cfg_denoised_callback
from modules.script_callbacks import AfterCFGCallbackParams, cfg_after_cfg_callback
from modules.sd_samplers_cfg_denoiser import CFGDenoiser, catenate_conds, subscript_cond, pad_cond

from .animatediff_logger import logger_animatediff as logger
from .animatediff_ui import AnimateDiffProcess
from .animatediff_prompt import AnimateDiffPromptSchedule


class AnimateDiffInfV2V:
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
                        control.control_model.uncond_image_emb = control.control_model.uncond_image_emb[context]
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
