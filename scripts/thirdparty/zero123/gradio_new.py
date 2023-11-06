'''
conda activate zero123
cd stable-diffusion
python gradio_new.py 0
'''

import diffusers  # 0.12.1
import math
import fire
import gradio as gr
import lovely_numpy
import lovely_tensors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import rich
import sys
import time
import torch
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from functools import partial
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import AutoFeatureExtractor #, CLIPImageProcessor
from torch import autocast
from torchvision import transforms


_SHOW_DESC = True
_SHOW_INTERMEDIATE = False
# _SHOW_INTERMEDIATE = True
_GPU_INDEX = 0
# _GPU_INDEX = 2

# _TITLE = 'Zero-Shot Control of Camera Viewpoints within a Single Image'
_TITLE = 'Zero-1-to-3: Zero-shot One Image to 3D Object'

# This demo allows you to generate novel viewpoints of an object depicted in an input image using a fine-tuned version of Stable Diffusion.
_DESCRIPTION = '''
This demo allows you to control camera rotation and thereby generate novel viewpoints of an object within a single image.
It is based on Stable Diffusion. Check out our [project webpage](https://zero123.cs.columbia.edu/) and [paper](https://arxiv.org/) if you want to learn more about the method!
Note that this model is not intended for images of humans or faces, and is unlikely to work well for them.
'''

_ARTICLE = 'See uses.md'


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
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


class CameraVisualizer:
    def __init__(self, gradio_plot):
        self._gradio_plot = gradio_plot
        self._fig = None
        self._polar = 0.0
        self._azimuth = 0.0
        self._radius = 0.0
        self._raw_image = None
        self._8bit_image = None
        self._image_colorscale = None

    def polar_change(self, value):
        self._polar = value
        # return self.update_figure()

    def azimuth_change(self, value):
        self._azimuth = value
        # return self.update_figure()

    def radius_change(self, value):
        self._radius = value
        # return self.update_figure()

    def encode_image(self, raw_image):
        '''
        :param raw_image (H, W, 3) array of uint8 in [0, 255].
        '''
        # https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot

        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        self._raw_image = raw_image
        self._8bit_image = Image.fromarray(raw_image).convert('P', palette='WEB', dither=None)
        # self._8bit_image = Image.fromarray(raw_image.clip(0, 254)).convert(
        #     'P', palette='WEB', dither=None)
        self._image_colorscale = [
            [i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]

        # return self.update_figure()

    def update_figure(self):
        fig = go.Figure()

        if self._raw_image is not None:
            (H, W, C) = self._raw_image.shape

            x = np.zeros((H, W))
            (y, z) = np.meshgrid(np.linspace(-1.0, 1.0, W), np.linspace(1.0, -1.0, H) * H / W)
            print('x:', lo(x))
            print('y:', lo(y))
            print('z:', lo(z))

            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=self._8bit_image,
                cmin=0,
                cmax=255,
                colorscale=self._image_colorscale,
                showscale=False,
                lighting_diffuse=1.0,
                lighting_ambient=1.0,
                lighting_fresnel=1.0,
                lighting_roughness=1.0,
                lighting_specular=0.3))

            scene_bounds = 3.5
            base_radius = 2.5
            zoom_scale = 1.5  # Note that input radius offset is in [-0.5, 0.5].
            fov_deg = 50.0
            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

            input_cone = calc_cam_cone_pts_3d(
                0.0, 0.0, base_radius, fov_deg)  # (5, 3).
            output_cone = calc_cam_cone_pts_3d(
                self._polar, self._azimuth, base_radius + self._radius * zoom_scale, fov_deg)  # (5, 3).
            # print('input_cone:', lo(input_cone).v)
            # print('output_cone:', lo(output_cone).v)

            for (cone, clr, legend) in [(input_cone, 'green', 'Input view'),
                                        (output_cone, 'blue', 'Target view')]:

                for (i, edge) in enumerate(edges):
                    (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                    (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                    (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                    fig.add_trace(go.Scatter3d(
                        x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                        line=dict(color=clr, width=3),
                        name=legend, showlegend=(i == 0)))
                    # text=(legend if i == 0 else None),
                    # textposition='bottom center'))
                    # hoverinfo='text',
                    # hovertext='hovertext'))

                # Add label.
                if cone[0, 2] <= base_radius / 2.0:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] - 0.05], showlegend=False,
                        mode='text', text=legend, textposition='bottom center'))
                else:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] + 0.05], showlegend=False,
                        mode='text', text=legend, textposition='top center'))

            # look at center of scene
            fig.update_layout(
                # width=640,
                # height=480,
                # height=400,
                height=360,
                autosize=True,
                hovermode=False,
                margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                showlegend=True,
                legend=dict(
                    yanchor='bottom',
                    y=0.01,
                    xanchor='right',
                    x=0.99,
                ),
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=1.0),
                    camera=dict(
                        eye=dict(x=base_radius - 1.6, y=0.0, z=0.6),
                        center=dict(x=0.0, y=0.0, z=0.0),
                        up=dict(x=0.0, y=0.0, z=1.0)),
                    xaxis_title='',
                    yaxis_title='',
                    zaxis_title='',
                    xaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    yaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    zaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks='')))

        self._fig = fig
        return fig


def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    print('new input_im:', lo(input_im))

    return input_im


def main_run(models, device, cam_vis, return_what,
             x=0.0, y=0.0, z=0.0,
             raw_im=None, preprocess=True,
             scale=3.0, n_samples=4, ddim_steps=50, ddim_eta=1.0,
             precision='fp32', h=256, w=256):
    '''
    :param raw_im (PIL Image).
    '''
    print(raw_im)
    print('return what:',return_what)
    print('x,y,z:', x,y,z)
    raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    safety_checker_input = models['clip_fe'](raw_im, return_tensors='pt').to(device)
    (image, has_nsfw_concept) = models['nsfw'](
        images=np.ones((1, 3)), clip_input=safety_checker_input.pixel_values)
    print('has_nsfw_concept:', has_nsfw_concept)
    if np.any(has_nsfw_concept):
        print('NSFW content detected.')
        to_return = [None] * 10
        description = ('###  <span style="color:red"> Unfortunately, '
                       'potential NSFW content was detected, '
                       'which is not supported by our model. '
                       'Please try again with a different image. </span>')
        if 'angles' in return_what:
            to_return[0] = 0.0
            to_return[1] = 0.0
            to_return[2] = 0.0
            to_return[3] = description
        else:
            to_return[0] = description
        return to_return

    else:
        print('Safety check passed.')

    input_im = preprocess_image(models, raw_im, preprocess)

    # if np.random.rand() < 0.3:
    #     description = ('Unfortunately, a human, a face, or potential NSFW content was detected, '
    #                    'which is not supported by our model.')
    #     if vis_only:
    #         return (None, None, description)
    #     else:
    #         return (None, None, None, description)

    show_in_im1 = (input_im * 255.0).astype(np.uint8)
    show_in_im2 = Image.fromarray(show_in_im1)

    if 'rand' in return_what:
        x = int(np.round(np.arcsin(np.random.uniform(-1.0, 1.0)) * 160.0 / np.pi))  # [-80, 80].
        y = int(np.round(np.random.uniform(-150.0, 150.0)))
        z = 0.0

    cam_vis.polar_change(x)
    cam_vis.azimuth_change(y)
    cam_vis.radius_change(z)
    cam_vis.encode_image(show_in_im1)
    new_fig = cam_vis.update_figure()

    if 'vis' in return_what:
        description = ('The viewpoints are visualized on the top right. '
                       'Click Run Generation to update the results on the bottom right.')

        if 'angles' in return_what:
            return (x, y, z, description, new_fig, show_in_im2)
        else:
            return (description, new_fig, show_in_im2)

    elif 'gen' in return_what:
        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
        input_im = input_im * 2 - 1
        input_im = transforms.functional.resize(input_im, [h, w])

        sampler = DDIMSampler(models['turncam'])
        # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
        used_x = x  # NOTE: Set this way for consistency.
        x_samples_ddim = sample_model(input_im, models['turncam'], sampler, precision, h, w,
                                      ddim_steps, n_samples, scale, ddim_eta, used_x, y, z)

        output_ims = []
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

        description = None

        if 'angles' in return_what:
            return (x, y, z, description, new_fig, show_in_im2, output_ims)
        else:
            return (description, new_fig, show_in_im2, output_ims)


def calc_cam_cone_pts_3d(polar_deg, azimuth_deg, radius_m, fov_deg):
    '''
    :param polar_deg (float).
    :param azimuth_deg (float).
    :param radius_m (float).
    :param fov_deg (float).
    :return (5, 3) array of float with (x, y, z).
    '''
    polar_rad = np.deg2rad(polar_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    fov_rad = np.deg2rad(fov_deg)
    polar_rad = -polar_rad  # NOTE: Inverse of how used_x relates to x.

    # Camera pose center:
    cam_x = radius_m * np.cos(azimuth_rad) * np.cos(polar_rad)
    cam_y = radius_m * np.sin(azimuth_rad) * np.cos(polar_rad)
    cam_z = radius_m * np.sin(polar_rad)

    # Obtain four corners of camera frustum, assuming it is looking at origin.
    # First, obtain camera extrinsics (rotation matrix only):
    camera_R = np.array([[np.cos(azimuth_rad) * np.cos(polar_rad),
                          -np.sin(azimuth_rad),
                          -np.cos(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(azimuth_rad) * np.cos(polar_rad),
                          np.cos(azimuth_rad),
                          -np.sin(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(polar_rad),
                          0.0,
                          np.cos(polar_rad)]])
    # print('camera_R:', lo(camera_R).v)

    # Multiply by corners in camera space to obtain go to space:
    corn1 = [-1.0, np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn2 = [-1.0, -np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn3 = [-1.0, -np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn4 = [-1.0, np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn1 = np.dot(camera_R, corn1)
    corn2 = np.dot(camera_R, corn2)
    corn3 = np.dot(camera_R, corn3)
    corn4 = np.dot(camera_R, corn4)

    # Now attach as offset to actual 3D camera position:
    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2)
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2)
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2)
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1]
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2)
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4]

    return np.array([xs, ys, zs]).T


def run_demo(
        device_idx=_GPU_INDEX,
        ckpt='105000.ckpt',
        config='configs/sd-objaverse-finetune-c_concat-256.yaml'):

    print('sys.argv:', sys.argv)
    if len(sys.argv) > 1:
        print('old device_idx:', device_idx)
        device_idx = int(sys.argv[1])
        print('new device_idx:', device_idx)

    device = f'cuda:{device_idx}'
    config = OmegaConf.load(config)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = load_model_from_config(config, ckpt, device=device)
    print('Instantiating Carvekit HiInterface...')
    models['carvekit'] = create_carvekit_interface()
    print('Instantiating StableDiffusionSafetyChecker...')
    models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
        'CompVis/stable-diffusion-safety-checker').to(device)
    print('Instantiating AutoFeatureExtractor...')
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        'CompVis/stable-diffusion-safety-checker')

    # Reduce NSFW false positives.
    # NOTE: At the time of writing, and for diffusers 0.12.1, the default parameters are:
    # models['nsfw'].concept_embeds_weights:
    # [0.1800, 0.1900, 0.2060, 0.2100, 0.1950, 0.1900, 0.1940, 0.1900, 0.1900, 0.2200, 0.1900,
    #  0.1900, 0.1950, 0.1984, 0.2100, 0.2140, 0.2000].
    # models['nsfw'].special_care_embeds_weights:
    # [0.1950, 0.2000, 0.2200].
    # We multiply all by some factor > 1 to make them less likely to be triggered.
    models['nsfw'].concept_embeds_weights *= 1.07
    models['nsfw'].special_care_embeds_weights *= 1.07

    with open('instructions.md', 'r') as f:
        article = f.read()

    # Compose demo layout & data flow.
    demo = gr.Blocks(title=_TITLE)

    with demo:
        gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=0.9, variant='panel'):

                image_block = gr.Image(type='pil', image_mode='RGBA',
                                       label='Input image of single object')
                preprocess_chk = gr.Checkbox(
                    True, label='Preprocess image automatically (remove background and recenter object)')
                # info='If enabled, the uploaded image will be preprocessed to remove the background and recenter the object by cropping and/or padding as necessary. '
                # 'If disabled, the image will be used as-is, *BUT* a fully transparent or white background is required.'),

                gr.Markdown('*Try camera position presets:*')
                with gr.Row():
                    left_btn = gr.Button('View from the Left', variant='primary')
                    above_btn = gr.Button('View from Above', variant='primary')
                    right_btn = gr.Button('View from the Right', variant='primary')
                with gr.Row():
                    random_btn = gr.Button('Random Rotation', variant='primary')
                    below_btn = gr.Button('View from Below', variant='primary')
                    behind_btn = gr.Button('View from Behind', variant='primary')

                gr.Markdown('*Control camera position manually:*')
                polar_slider = gr.Slider(
                    -90, 90, value=0, step=5, label='Polar angle (vertical rotation in degrees)')
                # info='Positive values move the camera down, while negative values move the camera up.')
                azimuth_slider = gr.Slider(
                    -180, 180, value=0, step=5, label='Azimuth angle (horizontal rotation in degrees)')
                # info='Positive values move the camera right, while negative values move the camera left.')
                radius_slider = gr.Slider(
                    -0.5, 0.5, value=0.0, step=0.1, label='Zoom (relative distance from center)')
                # info='Positive values move the camera further away, while negative values move the camera closer.')

                samples_slider = gr.Slider(1, 8, value=4, step=1,
                                           label='Number of samples to generate')

                with gr.Accordion('Advanced options', open=False):
                    scale_slider = gr.Slider(0, 30, value=3, step=1,
                                             label='Diffusion guidance scale')
                    steps_slider = gr.Slider(5, 200, value=75, step=5,
                                             label='Number of diffusion inference steps')

                with gr.Row():
                    vis_btn = gr.Button('Visualize Angles', variant='secondary')
                    run_btn = gr.Button('Run Generation', variant='primary')

                desc_output = gr.Markdown('The results will appear on the right.', visible=_SHOW_DESC)

            with gr.Column(scale=1.1, variant='panel'):

                vis_output = gr.Plot(
                    label='Relationship between input (green) and output (blue) camera poses')

                gen_output = gr.Gallery(label='Generated images from specified new viewpoint')
                gen_output.style(grid=2)

                preproc_output = gr.Image(type='pil', image_mode='RGB',
                                          label='Preprocessed input image', visible=_SHOW_INTERMEDIATE)

        gr.Markdown(article)

        # NOTE: I am forced to update vis_output for these preset buttons,
        # because otherwise the gradio plot always resets the plotly 3D viewpoint for some reason,
        # which might confuse the user into thinking that the plot has been updated too.

        # OLD 1:
        # left_btn.click(fn=lambda: [0.0, -90.0], #, 0.0],
        #                inputs=[], outputs=[polar_slider, azimuth_slider]), #], radius_slider])
        # above_btn.click(fn=lambda: [90.0, 0.0], #, 0.0],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider]), #, radius_slider])
        # right_btn.click(fn=lambda: [0.0, 90.0], #, 0.0],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider]), #, radius_slider])
        # random_btn.click(fn=lambda: [int(np.round(np.random.uniform(-60.0, 60.0))),
        #                              int(np.round(np.random.uniform(-150.0, 150.0)))], #, 0.0],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider]), #, radius_slider])
        # below_btn.click(fn=lambda: [-90.0, 0.0], #, 0.0],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider]), #, radius_slider])
        # behind_btn.click(fn=lambda: [0.0, 180.0], #, 0.0],
        #                  inputs=[], outputs=[polar_slider, azimuth_slider]), #, radius_slider])

        # OLD 2:
        # preset_text = ('You have selected a preset target camera view. '
        #                'Now click Run Generation to update the results!')

        # left_btn.click(fn=lambda: [0.0, -90.0, None, preset_text],
        #                inputs=[], outputs=[polar_slider, azimuth_slider, vis_output, desc_output])
        # above_btn.click(fn=lambda: [90.0, 0.0, None, preset_text],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider, vis_output, desc_output])
        # right_btn.click(fn=lambda: [0.0, 90.0, None, preset_text],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider, vis_output, desc_output])
        # random_btn.click(fn=lambda: [int(np.round(np.random.uniform(-60.0, 60.0))),
        #                              int(np.round(np.random.uniform(-150.0, 150.0))),
        #                              None, preset_text],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider, vis_output, desc_output])
        # below_btn.click(fn=lambda: [-90.0, 0.0, None, preset_text],
        #                 inputs=[], outputs=[polar_slider, azimuth_slider, vis_output, desc_output])
        # behind_btn.click(fn=lambda: [0.0, 180.0, None, preset_text],
        #                  inputs=[], outputs=[polar_slider, azimuth_slider, vis_output, desc_output])

        # OLD 3 (does not work at all):
        # def a():
        #     polar_slider.value = 77.7
        #     polar_slider.postprocess(77.7)
        #     print('testa')
        # left_btn.click(fn=a)

        cam_vis = CameraVisualizer(vis_output)

        vis_btn.click(fn=partial(main_run, models, device, cam_vis, 'vis'),
                      inputs=[polar_slider, azimuth_slider, radius_slider,
                              image_block, preprocess_chk],
                      outputs=[desc_output, vis_output, preproc_output])

        run_btn.click(fn=partial(main_run, models, device, cam_vis, 'gen'),
                      inputs=[polar_slider, azimuth_slider, radius_slider,
                              image_block, preprocess_chk,
                              scale_slider, samples_slider, steps_slider],
                      outputs=[desc_output, vis_output, preproc_output, gen_output])

        # NEW:
        preset_inputs = [image_block, preprocess_chk,
                         scale_slider, samples_slider, steps_slider]
        preset_outputs = [polar_slider, azimuth_slider, radius_slider,
                          desc_output, vis_output, preproc_output, gen_output]
        left_btn.click(fn=partial(main_run, models, device, cam_vis, 'angles_gen',
                                  0.0, -90.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        above_btn.click(fn=partial(main_run, models, device, cam_vis, 'angles_gen',
                                   -90.0, 0.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        right_btn.click(fn=partial(main_run, models, device, cam_vis, 'angles_gen',
                                   0.0, 90.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        random_btn.click(fn=partial(main_run, models, device, cam_vis, 'rand_angles_gen',
                                    -1.0, -1.0, -1.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        below_btn.click(fn=partial(main_run, models, device, cam_vis, 'angles_gen',
                                   90.0, 0.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)
        behind_btn.click(fn=partial(main_run, models, device, cam_vis, 'angles_gen',
                                    0.0, 180.0, 0.0),
                       inputs=preset_inputs, outputs=preset_outputs)

    demo.launch(enable_queue=True, share=True)


if __name__ == '__main__':

    fire.Fire(run_demo)
