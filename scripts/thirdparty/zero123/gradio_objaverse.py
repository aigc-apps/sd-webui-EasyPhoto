from contextlib import nullcontext
from functools import partial

import math
import fire
import gradio as gr
import numpy as np
import torch
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms
from ldm.util import load_and_preprocess, instantiate_from_config

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=device)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, \
                 ddim_eta, x, y, z):
    precision_scope = autocast if precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples,1,1)
            T = torch.tensor([math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            c_concat = model.encode_first_stage((input_im.to(c.device))).mode().detach()
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()\
                               .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def main(
    model,
    device,
    input_im,
    x=0.,
    y=0.,
    z=0.,
    scale=3.0,
    n_samples=4,
    ddim_steps=50,
    preprocess=True,
    ddim_eta=1.0,
    precision="fp32",
    h=256,
    w=256,
    ):
    # input_im[input_im == [0., 0., 0.]] = [1., 1., 1., 1.]
    print(input_im.size)
    if preprocess:
        input_im = load_and_preprocess(input_im)
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.
        input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.] # very important, thresholding background
        input_im = input_im[:, :, :3]
    print(input_im.shape)
    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    input_im = transforms.functional.resize(input_im, [h, w])

    sampler = DDIMSampler(model)

    x_samples_ddim = sample_model(input_im, model, sampler, precision, h, w,\
                                  ddim_steps, n_samples, scale, ddim_eta, x, y, z)
    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    return output_ims


description = \
"""Generate variations on an input image using a fine-tuned version of Stable Diffision.
Trained by [Justin Pinkney](https://www.justinpinkney.com) ([@Buntworthy](https://twitter.com/Buntworthy)) at [Lambda](https://lambdalabs.com/)
__Get the [code](https://github.com/justinpinkney/stable-diffusion) and [model](https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned).__
![](https://raw.githubusercontent.com/justinpinkney/stable-diffusion/main/assets/im-vars-thin.jpg)
"""

article = \
"""
## How does this work?
The normal Stable Diffusion model is trained to be conditioned on text input. This version has had the original text encoder (from CLIP) removed, and replaced with
the CLIP _image_ encoder instead. So instead of generating images based a text input, images are generated to match CLIP's embedding of the image.
This creates images which have the same rough style and content, but different details, in particular the composition is generally quite different.
This is a totally different approach to the img2img script of the original Stable Diffusion and gives very different results.
The model was fine tuned on the [LAION aethetics v2 6+ dataset](https://laion.ai/blog/laion-aesthetics/) to accept the new conditioning.
Training was done on 4xA6000 GPUs on [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud).
More details on the method and training will come in a future blog post.
"""


def run_demo(
    device_idx=0,
    ckpt="last.ckpt",
    config="configs/sd-objaverse-finetune-c_concat-256.yaml",
    ):

    device = f"cuda:{device_idx}"
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device=device)

    inputs = [
        gr.Image(type="pil", image_mode="RGBA"), # shape=[512, 512]
        gr.Number(label="polar (between axis z+)"),
        gr.Number(label="azimuth (between axis x+)"),
        gr.Number(label="z (distance from center)"),
        gr.Slider(0, 100, value=3, step=1, label="cfg scale"),
        gr.Slider(1, 8, value=4, step=1, label="Number images"),
        gr.Slider(5, 200, value=100, step=5, label="steps"),
        gr.Checkbox(True, label="image preprocess (background removal and recenter)"),
    ]
    output = gr.Gallery(label="Generated variations")
    output.style(grid=2)

    fn_with_model = partial(main, model, device)
    fn_with_model.__name__ = "fn_with_model"

    examples = [
        # ["assets/zero-shot/bear.png", 0, 0, 0, 3, 4, 100],
        # ["assets/zero-shot/car.png", 0, 0, 0, 3, 4, 100],
        # ["assets/zero-shot/elephant.png", 0, 0, 0, 3, 4, 100],
        # ["assets/zero-shot/pikachu.png", 0, 0, 0, 3, 4, 100],
        # ["assets/zero-shot/spyro.png", 0, 0, 0, 3, 4, 100],
        # ["assets/zero-shot/taxi.png", 0, 0, 0, 3, 4, 100],
    ]

    demo = gr.Interface(
        fn=fn_with_model,
        title="Stable Diffusion Novel View Synthesis (Image)",
        # description=description,
        # article=article,
        inputs=inputs,
        outputs=output,
        examples=examples,
        allow_flagging="never",
        )
    demo.launch(enable_queue=True, share=True)

if __name__ == "__main__":
    fire.Fire(run_demo)