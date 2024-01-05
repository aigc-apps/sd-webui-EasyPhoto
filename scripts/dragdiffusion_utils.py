# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************

import logging
import sys

sys.path.append("/root/zhoumo/AICamera/EasyPhoto")
import copy
import logging
import os
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import DDIMScheduler, StableDiffusionPipeline
from einops import rearrange
from PIL import Image
from pytorch_lightning import seed_everything
from safetensors.torch import load_file
from scripts.train_kohya.utils.model_utils import load_models_from_stable_diffusion_checkpoint
from tqdm import tqdm
from transformers import CLIPTokenizer

# weight_dtype = torch.float16
weight_dtype = torch.float32

# TODO: Refactor with merge_lora.
def merge_lora(pipeline, lora_path, multiplier, from_safetensor=False, device="cpu", dtype=torch.float32):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    if from_safetensor:
        state_dict = load_file(lora_path, device=device)
    else:
        checkpoint = torch.load(os.path.join(lora_path, "pytorch_lora_weights.bin"), map_location=torch.device(device))
        new_dict = dict()
        for idx, key in enumerate(checkpoint):
            new_key = re.sub(r"\.processor\.", "_", key)
            new_key = re.sub(r"mid_block\.", "mid_block_", new_key)
            new_key = re.sub("_lora.up.", ".lora_up.", new_key)
            new_key = re.sub("_lora.down.", ".lora_down.", new_key)
            new_key = re.sub(r"\.(\d+)\.", "_\\1_", new_key)
            new_key = re.sub("to_out", "to_out_0", new_key)
            new_key = "lora_unet_" + new_key
            new_dict[new_key] = checkpoint[key]
            state_dict = new_dict
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        layer, elem = key.split(".", 1)
        updates[layer][elem] = value

    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(layer_infos) == 0:
                    print("Error loading layer")
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        weight_up = elems["lora_up.weight"].to(dtype)
        weight_down = elems["lora_down.weight"].to(dtype)
        if "alpha" in elems.keys():
            alpha = elems["alpha"].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        curr_layer.weight.data = curr_layer.weight.data.to(device)
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += (
                multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            )
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline


# -------------------------------------------------------- #
#   Drag utils
# -------------------------------------------------------- #
def point_tracking(F0, F1, handle_points, handle_points_init, r_p):
    with torch.no_grad():
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

            r1, r2 = int(pi[0]) - r_p, int(pi[0]) + r_p + 1
            c1, c2 = int(pi[1]) - r_p, int(pi[1]) + r_p + 1
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            all_dist = all_dist.squeeze(dim=0)
            # WARNING: no boundary protection right now
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            handle_points[i][0] = pi[0] - r_p + row
            handle_points[i][1] = pi[1] - r_p + col
        return handle_points


def check_handle_reach_target(handle_points, target_points):
    all_dist = list(map(lambda p, q: (p - q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 2.0).all()


# obtain the bilinear interpolated feature patch centered around (x, y) with radius r
def interpolate_feature_patch(feat, y, x, r):
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    Ia = feat[:, :, y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
    Ib = feat[:, :, y1 - r : y1 + r + 1, x0 - r : x0 + r + 1]
    Ic = feat[:, :, y0 - r : y0 + r + 1, x1 - r : x1 + r + 1]
    Id = feat[:, :, y1 - r : y1 + r + 1, x1 - r : x1 + r + 1]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def drag_diffusion_update(
    model, init_code, t, handle_points, target_points, mask, prompt, unet_feature_idx, lam, lr, sup_res_h, sup_res_w, n_pix_step, r_m, r_p
):
    assert len(handle_points) == len(target_points), "number of handle point must equals target points"

    text_emb = model.get_text_embeddings(prompt).detach()
    # the init output feature of unet
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(
            init_code, t, encoder_hidden_states=text_emb, layer_idx=unet_feature_idx, interp_res_h=sup_res_h, interp_res_w=sup_res_w
        )
        x_prev_0, _ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=lr)

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2], init_code.shape[3]), mode="nearest")

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    for step_idx in range(n_pix_step):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            unet_output, F1 = model.forward_unet_features(
                init_code, t, encoder_hidden_states=text_emb, layer_idx=unet_feature_idx, interp_res_h=sup_res_h, interp_res_w=sup_res_w
            )
            x_prev_updated, _ = model.step(unet_output, t, init_code)

            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                handle_points = point_tracking(F0, F1, handle_points, handle_points_init, r_p)
                print("new handle points", handle_points)

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                break

            loss = 0.0
            for i in range(len(handle_points)):
                pi, ti = handle_points[i], target_points[i]
                # skip if the distance between target and source is less than 1
                if (ti - pi).norm() < 2.0:
                    continue

                di = (ti - pi) / (ti - pi).norm()

                # motion supervision
                f0_patch = F1[:, :, int(pi[0]) - r_m : int(pi[0]) + r_m + 1, int(pi[1]) - r_m : int(pi[1]) + r_m + 1].detach()
                f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], r_m)
                loss += ((2 * r_m + 1) ** 2) * F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            loss += lam * ((x_prev_updated - x_prev_0) * (1.0 - interp_mask)).abs().sum()
            # loss += lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
            print("loss total=%f" % (loss.item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return init_code


# -------------------------------------------------------- #
#   Attn utils
# -------------------------------------------------------- #
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, is_cross, place_in_unet, num_heads, **kwargs):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = rearrange(out, "b h n d -> b n (h d)")
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class MutualSelfAttentionControl(AttentionBase):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, guidance_scale=7.5):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
        """
        super().__init__()
        self.total_steps = total_steps
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, 16))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        # store the guidance scale to decide whether there are unconditional branch
        self.guidance_scale = guidance_scale
        print("step_idx: ", self.step_idx)
        print("layer_idx: ", self.layer_idx)

    def forward(self, q, k, v, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, is_cross, place_in_unet, num_heads, **kwargs)

        if self.guidance_scale > 1.0:
            qu, qc = q[0:2], q[2:4]
            ku, kc = k[0:2], k[2:4]
            vu, vc = v[0:2], v[2:4]

            # merge queries of source and target branch into one so we can use torch API
            qu = torch.cat([qu[0:1], qu[1:2]], dim=2)
            qc = torch.cat([qc[0:1], qc[1:2]], dim=2)

            out_u = F.scaled_dot_product_attention(qu, ku[0:1], vu[0:1], attn_mask=None, dropout_p=0.0, is_causal=False)
            out_u = torch.cat(out_u.chunk(2, dim=2), dim=0)  # split the queries into source and target batch
            out_u = rearrange(out_u, "b h n d -> b n (h d)")

            out_c = F.scaled_dot_product_attention(qc, kc[0:1], vc[0:1], attn_mask=None, dropout_p=0.0, is_causal=False)
            out_c = torch.cat(out_c.chunk(2, dim=2), dim=0)  # split the queries into source and target batch
            out_c = rearrange(out_c, "b h n d -> b n (h d)")

            out = torch.cat([out_u, out_c], dim=0)
        else:
            q = torch.cat([q[0:1], q[1:2]], dim=2)
            out = F.scaled_dot_product_attention(q, k[0:1], v[0:1], attn_mask=None, dropout_p=0.0, is_causal=False)
            out = torch.cat(out.chunk(2, dim=2), dim=0)  # split the queries into source and target batch
            out = rearrange(out, "b h n d -> b n (h d)")
        return out


# forward function for default attention processor
# modified from __call__ function of AttnProcessor in diffusers
def override_attn_proc_forward(attn, editor, place_in_unet):
    def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
        """
        The attention is similar to the original implementation of LDM CrossAttention class
        except adding some modifications on the attention
        """
        if encoder_hidden_states is not None:
            context = encoder_hidden_states
        if attention_mask is not None:
            pass

        to_out = attn.to_out
        if isinstance(to_out, nn.modules.container.ModuleList):
            to_out = attn.to_out[0]
        else:
            to_out = attn.to_out

        h = attn.heads
        q = attn.to_q(x)
        is_cross = context is not None
        context = context if is_cross else x
        k = attn.to_k(context)
        v = attn.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # the only difference
        out = editor(q, k, v, is_cross, place_in_unet, attn.heads, scale=attn.scale)

        return to_out(out)

    return forward


# forward function for lora attention processor
# modified from __call__ function of LoRAAttnProcessor2_0 in diffusers v0.17.1
def override_lora_attn_proc_forward(attn, editor, place_in_unet):
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, lora_scale=1.0):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        is_cross = encoder_hidden_states is not None

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if hasattr(attn.processor, "to_q_lora"):
            query = attn.to_q(hidden_states) + lora_scale * attn.processor.to_q_lora(hidden_states)
        else:
            query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if hasattr(attn.processor, "to_q_lora"):
            key = attn.to_k(encoder_hidden_states) + lora_scale * attn.processor.to_k_lora(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states) + lora_scale * attn.processor.to_v_lora(encoder_hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        query, key, value = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=attn.heads), (query, key, value))

        # the only difference
        hidden_states = editor(query, key, value, is_cross, place_in_unet, attn.heads, scale=attn.scale)

        # linear proj
        if hasattr(attn.processor, "to_out_lora"):
            hidden_states = attn.to_out[0](hidden_states) + lora_scale * attn.processor.to_out_lora(hidden_states)
        else:
            hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    return forward


def register_attention_editor_diffusers(model, editor: AttentionBase, attn_processor="attn_proc"):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == "Attention":  # spatial Transformer layer
                if attn_processor == "attn_proc":
                    net.forward = override_attn_proc_forward(net, editor, place_in_unet)
                elif attn_processor == "lora_attn_proc":
                    net.forward = override_lora_attn_proc_forward(net, editor, place_in_unet)
                else:
                    raise NotImplementedError("not implemented")
                return count + 1
            elif hasattr(net, "children"):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count


# -------------------------------------------------------- #
#   Unet Hack
# -------------------------------------------------------- #
# override unet forward
# The only difference from diffusers:
# return intermediate UNet features of all UpSample blocks
def override_forward(self):
    def forward(
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        last_up_block_idx: int = None,
    ):
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(down_block_res_samples, down_block_additional_residuals):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        # only difference from diffusers:
        # save the intermediate features of unet upsample blocks
        # the 0-th element is the mid-block output
        all_intermediate_features = [sample]
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size)
            all_intermediate_features.append(sample)
            # return early to save computation time if needed
            if last_up_block_idx is not None and i == last_up_block_idx:
                return all_intermediate_features

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # only difference from diffusers, return intermediate results
        if return_intermediates:
            return sample, all_intermediate_features
        else:
            return sample

    return forward


class DragPipeline(StableDiffusionPipeline):
    # must call this function when initialize
    def modify_unet_forward(self):
        self.unet.forward = override_forward(self.unet)

    def inv_step(self, model_output: torch.FloatTensor, timestep: int, x: torch.FloatTensor, eta=0.0, verbose=False):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
    ):
        """
        predict the sample of the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type="np"):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)["sample"]

        return image  # range [-1, 1]

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        # text embeddings
        text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.cuda())[0]
        return text_embeddings

    # get all intermediate features and then do bilinear interpolation
    # return features in the layer_idx list
    def forward_unet_features(self, z, t, encoder_hidden_states, layer_idx=[0], interp_res_h=256, interp_res_w=256):
        unet_output, all_intermediate_features = self.unet(z, t, encoder_hidden_states=encoder_hidden_states, return_intermediates=True)

        all_return_features = []
        for idx in layer_idx:
            feat = all_intermediate_features[idx]
            feat = F.interpolate(feat, (interp_res_h, interp_res_w), mode="bilinear")
            all_return_features.append(feat)
        return_features = torch.cat(all_return_features, dim=1)
        return unet_output, return_features

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        prompt_embeds=None,  # whether text embedding is directly provided.
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        num_actual_inference_steps=None,
        guidance_scale=7.5,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        return_intermediates=False,
        **kwds,
    ):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if prompt_embeds is None:
            if isinstance(prompt, list):
                batch_size = len(prompt)
            elif isinstance(prompt, str):
                if batch_size > 1:
                    prompt = [prompt] * batch_size

            # text embeddings
            text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
            text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        else:
            batch_size = prompt_embeds.shape[0]
            text_embeddings = prompt_embeds
        print("input text embeddings :", text_embeddings.shape)

        # define initial latents if not predefined
        if latents is None:
            latents_shape = (batch_size, self.unet.in_channels, height // 8, width // 8)
            latents = torch.randn(latents_shape, device=DEVICE, dtype=self.vae.dtype)

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.0:
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            unconditional_input = self.tokenizer([uc_text] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        # print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue

            if guidance_scale > 1.0:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings])
            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t -> x_t-1
            # YUJUN: right now, the only difference between step here and step in scheduler
            # is that scheduler version would clamp pred_x0 between [-1,1]
            # don't know if that's gonna have huge impact
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents_list.append(latents)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            return image, latents_list
        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        num_actual_inference_steps=None,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds,
    ):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.image2latent(image)

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.0:
            text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if num_actual_inference_steps is not None and i >= num_actual_inference_steps:
                continue

            if guidance_scale > 1.0:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
            if guidance_scale > 1.0:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.inv_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents


# -------------------------------------------------------- #
#   Run Drag Diffusion
# -------------------------------------------------------- #
def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def preprocess_mask(mask, device):
    mask = torch.from_numpy(mask).float() / 255.0
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w")
    mask = mask.to(device)
    return mask


def run_drag(
    source_image,
    mask,
    prompt,
    points,
    inversion_strength,
    lam,
    latent_lr,
    n_pix_step,
    model_path,
    sd_base15_checkpoint,
    lora_path,
    start_step,
    start_layer,
    seed=42,
    n_inference_step=50,
    guidance_scale=1.0,
    unet_feature_idx=[3],
):
    logging.info("start call drag_gen")

    # initialize parameters
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    n_actual_inference_step = round(inversion_strength * n_inference_step)

    full_h, full_w = source_image.shape[:2]
    sup_res_h = int(0.5 * full_h)
    sup_res_w = int(0.5 * full_w)
    r_m = 1
    r_p = 3

    # initialize seed
    seed_everything(seed)

    # initialize scheduler
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1
    )

    # load lora
    if model_path.endswith("safetensors"):
        logging.info("build from safetensors")
        tokenizer = CLIPTokenizer.from_pretrained(sd_base15_checkpoint, subfolder="tokenizer")
        text_encoder, vae, unet = load_models_from_stable_diffusion_checkpoint(False, model_path)
        model = DragPipeline(
            unet=unet.to(weight_dtype),
            text_encoder=text_encoder.to(weight_dtype),
            vae=vae.to(weight_dtype),
            scheduler=scheduler,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=None,
        ).to("cuda")
        merge_lora(model, lora_path, 0.3, from_safetensor=True, device="cuda", dtype=weight_dtype)
        model.unet.set_default_attn_processor()

        try:
            import xformers

            model.enable_xformers_memory_efficient_attention()
        except:
            logging.warning("No module named xformers. Infer without using xformers. You can run pip install xformers to install it.")

    else:
        logging.info("applying lora: " + lora_path)
        model = DragPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
        model.unet.load_attn_procs(lora_path)

    logging.info(f"load model from {model_path} success!")

    # call this function to override unet forward function,
    # so that intermediate features are returned after forward
    model.modify_unet_forward()

    # preprocess
    source_image = preprocess_image(source_image, device)
    mask = preprocess_mask(mask, device)
    mask = F.interpolate(mask, (sup_res_h, sup_res_w), mode="nearest")

    # here, the point is in x,y coordinate
    handle_points = []
    target_points = []
    for idx, point in enumerate(points):
        cur_point = torch.tensor([point[1] / full_h * sup_res_h, point[0] / full_w * sup_res_w])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    print("handle points:", handle_points)
    print("target points:", target_points)

    model.scheduler.set_timesteps(n_inference_step)
    t = model.scheduler.timesteps[n_inference_step - n_actual_inference_step]

    # invert the source image
    # the latent code resolution is too small, only 64*64
    init_code = model.invert(
        source_image,
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=n_inference_step,
        num_actual_inference_steps=n_actual_inference_step,
    )
    orig_init_code = deepcopy(init_code)

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # update according to the given supervision
    updated_init_code = drag_diffusion_update(
        model,
        init_code,
        t,
        handle_points,
        target_points,
        mask,
        prompt,
        unet_feature_idx,
        lam,
        latent_lr,
        sup_res_h,
        sup_res_w,
        n_pix_step,
        r_m,
        r_p,
    )

    # hijack the attention module
    # inject the reference branch to guide the generation
    editor = MutualSelfAttentionControl(
        start_step=start_step, start_layer=start_layer, total_steps=n_inference_step, guidance_scale=guidance_scale
    )
    if not model_path.endswith("safetensors"):
        register_attention_editor_diffusers(model, editor, attn_processor="lora_attn_proc")
    else:
        register_attention_editor_diffusers(model, editor, attn_processor="attn_proc")

    # inference the synthesized image
    gen_image = model(
        prompt=prompt,
        batch_size=2,
        latents=torch.cat([orig_init_code, updated_init_code], dim=0),
        guidance_scale=guidance_scale,
        num_inference_steps=n_inference_step,
        num_actual_inference_steps=n_actual_inference_step,
    )[1].unsqueeze(dim=0)

    # resize gen_image into the size of source_image
    # we do this because shape of gen_image will be rounded to multipliers of 8
    gen_image = F.interpolate(gen_image, (full_h, full_w), mode="bilinear")

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    return out_image
