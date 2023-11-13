import os

import cv2
import gradio as gr

from modules import shared
from modules.processing import StableDiffusionProcessing

from .animatediff_mm import mm_animatediff as motion_module
from .animatediff_i2ibatch import animatediff_i2ibatch


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)


    def get_block_name(self):
        return "button"


class AnimateDiffProcess:

    def __init__(
        self,
        model="mm_sd_v15_v2.ckpt",
        enable=False,
        video_length=0,
        fps=8,
        loop_number=0,
        closed_loop='R-P',
        batch_size=16,
        stride=1,
        overlap=-1,
        format=["GIF", "PNG"],
        interp='Off',
        interp_x=10,
        video_source=None,
        video_path='',
        latent_power=1,
        latent_scale=32,
        last_frame=None,
        latent_power_last=1,
        latent_scale_last=32,
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
            infotext['mm_hash'] = motion_module.mm.mm_hash[:8]
        if is_img2img:
            infotext.update({
                "latent_power": self.latent_power,
                "latent_scale": self.latent_scale,
                "latent_power_last": self.latent_power_last,
                "latent_scale_last": self.latent_scale_last,
            })
        infotext_str = ', '.join(f"{k}: {v}" for k, v in infotext.items())
        return infotext_str


    def _check(self):
        assert (
            self.video_length >= 0 and self.fps > 0
        ), "Video length and FPS should be positive."
        assert not set(["GIF", "MP4", "PNG", "WEBP"]).isdisjoint(
            self.format
        ), "At least one saving format should be selected."


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


class AnimateDiffUiGroup:
    txt2img_submit_button = None
    img2img_submit_button = None

    def __init__(self):
        self.params = AnimateDiffProcess()


    def render(self, is_img2img: bool, model_dir: str):
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        elemid_prefix = "img2img-ad-" if is_img2img else "txt2img-ad-"
        model_list = [f for f in os.listdir(model_dir) if f != ".gitkeep"]
        with gr.Accordion("AnimateDiff", open=False):
            gr.Markdown(value="Please click [this link](https://github.com/continue-revolution/sd-webui-animatediff#webui-parameters) to read the documentation of each parameter.")
            with gr.Row():

                def refresh_models(*inputs):
                    new_model_list = [
                        f for f in os.listdir(model_dir) if f != ".gitkeep"
                    ]
                    dd = inputs[0]
                    if dd in new_model_list:
                        selected = dd
                    elif len(new_model_list) > 0:
                        selected = new_model_list[0]
                    else:
                        selected = None
                    return gr.Dropdown.update(choices=new_model_list, value=selected)

                with gr.Row():
                    self.params.model = gr.Dropdown(
                        choices=model_list,
                        value=(self.params.model if self.params.model in model_list else None),
                        label="Motion module",
                        type="value",
                        elem_id=f"{elemid_prefix}motion-module",
                    )
                    refresh_model = ToolButton(value="\U0001f504")
                    refresh_model.click(refresh_models, self.params.model, self.params.model)

                self.params.format = gr.CheckboxGroup(
                    choices=["GIF", "MP4", "WEBP", "PNG", "TXT"],
                    label="Save format",
                    type="value",
                    elem_id=f"{elemid_prefix}save-format",
                    value=self.params.format,
                )
            with gr.Row():
                self.params.enable = gr.Checkbox(
                    value=self.params.enable, label="Enable AnimateDiff", 
                    elem_id=f"{elemid_prefix}enable"
                )
                self.params.video_length = gr.Number(
                    minimum=0,
                    value=self.params.video_length,
                    label="Number of frames",
                    precision=0,
                    elem_id=f"{elemid_prefix}video-length",
                )
                self.params.fps = gr.Number(
                    value=self.params.fps, label="FPS", precision=0, 
                    elem_id=f"{elemid_prefix}fps"
                )
                self.params.loop_number = gr.Number(
                    minimum=0,
                    value=self.params.loop_number,
                    label="Display loop number",
                    precision=0,
                    elem_id=f"{elemid_prefix}loop-number",
                )
            with gr.Row():
                self.params.closed_loop = gr.Radio(
                    choices=["N", "R-P", "R+P", "A"],
                    value=self.params.closed_loop,
                    label="Closed loop",
                    elem_id=f"{elemid_prefix}closed-loop",
                )
                self.params.batch_size = gr.Slider(
                    minimum=1,
                    maximum=32,
                    value=self.params.batch_size,
                    label="Context batch size",
                    step=1,
                    precision=0,
                    elem_id=f"{elemid_prefix}batch-size",
                )
                self.params.stride = gr.Number(
                    minimum=1,
                    value=self.params.stride,
                    label="Stride",
                    precision=0,
                    elem_id=f"{elemid_prefix}stride",
                )
                self.params.overlap = gr.Number(
                    minimum=-1,
                    value=self.params.overlap,
                    label="Overlap",
                    precision=0,
                    elem_id=f"{elemid_prefix}overlap",
                )
            with gr.Row():
                self.params.interp = gr.Radio(
                    choices=["Off", "FILM"],
                    label="Frame Interpolation",
                    elem_id=f"{elemid_prefix}interp-choice",
                    value=self.params.interp
                )
                self.params.interp_x = gr.Number(
                    value=self.params.interp_x, label="Interp X", precision=0, 
                    elem_id=f"{elemid_prefix}interp-x"
                )
            self.params.video_source = gr.Video(
                value=self.params.video_source,
                label="Video source",
            )
            def update_fps(video_source):
                if video_source is not None and video_source != '':
                    cap = cv2.VideoCapture(video_source)
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    cap.release()
                    return fps
                else:
                    return int(self.params.fps.value)
            self.params.video_source.change(update_fps, inputs=self.params.video_source, outputs=self.params.fps)
            def update_frames(video_source):
                if video_source is not None and video_source != '':
                    cap = cv2.VideoCapture(video_source)
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    return frames
                else:
                    return int(self.params.video_length.value)
            self.params.video_source.change(update_frames, inputs=self.params.video_source, outputs=self.params.video_length)
            self.params.video_path = gr.Textbox(
                value=self.params.video_path,
                label="Video path",
                elem_id=f"{elemid_prefix}video-path"
            )
            if is_img2img:
                with gr.Row():
                    self.params.latent_power = gr.Slider(
                        minimum=0.1,
                        maximum=10,
                        value=self.params.latent_power,
                        step=0.1,
                        label="Latent power",
                        elem_id=f"{elemid_prefix}latent-power",
                    )
                    self.params.latent_scale = gr.Slider(
                        minimum=1,
                        maximum=128,
                        value=self.params.latent_scale,
                        label="Latent scale",
                        elem_id=f"{elemid_prefix}latent-scale"
                    )
                    self.params.latent_power_last = gr.Slider(
                        minimum=0.1,
                        maximum=10,
                        value=self.params.latent_power_last,
                        step=0.1,
                        label="Optional latent power for last frame",
                        elem_id=f"{elemid_prefix}latent-power-last",
                    )
                    self.params.latent_scale_last = gr.Slider(
                        minimum=1,
                        maximum=128,
                        value=self.params.latent_scale_last,
                        label="Optional latent scale for last frame",
                        elem_id=f"{elemid_prefix}latent-scale-last"
                    )
                self.params.last_frame = gr.Image(
                    label="Optional last frame. Leave it blank if you do not need one.",
                    type="pil",
                )
            with gr.Row():
                unload = gr.Button(value="Move motion module to CPU (default if lowvram)")
                remove = gr.Button(value="Remove motion module from any memory")
                unload.click(fn=motion_module.unload)
                remove.click(fn=motion_module.remove)
        return self.register_unit(is_img2img)


    def register_unit(self, is_img2img: bool):
        unit = gr.State(value=AnimateDiffProcess)
        (
            AnimateDiffUiGroup.img2img_submit_button
            if is_img2img
            else AnimateDiffUiGroup.txt2img_submit_button
        ).click(
            fn=AnimateDiffProcess,
            inputs=self.params.get_list(is_img2img),
            outputs=unit,
            queue=False,
        )
        return unit


    @staticmethod
    def on_after_component(component, **_kwargs):
        elem_id = getattr(component, "elem_id", None)

        if elem_id == "txt2img_generate":
            AnimateDiffUiGroup.txt2img_submit_button = component
            return

        if elem_id == "img2img_generate":
            AnimateDiffUiGroup.img2img_submit_button = component
            return
