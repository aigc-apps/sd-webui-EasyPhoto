import os
import platform
import shutil
import subprocess as sp
import sys

import gradio as gr

from modules import call_queue, script_callbacks, shared, ui_postprocessing
from modules.call_queue import wrap_gradio_gpu_call
from modules.sd_samplers import samplers_for_img2img
from modules.shared import opts
from modules.ui import (create_override_settings_dropdown,
                        create_sampler_and_steps_selection, create_seed_inputs,
                        ordered_ui_categories)
from modules.ui_common import plaintext_to_html
from modules.ui_components import FormGroup, FormRow
from scripts.paiya_process import paiya_process, paiya_forward
from scripts.paiya_config import paiya_outpath_samples
from paiya_train import paiya_train

gradio_compat = True
try:
    from distutils.version import LooseVersion

    from importlib_metadata import version
    if LooseVersion(version("gradio")) < LooseVersion("3.10"):
        gradio_compat = False
except ImportError:
    pass

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as skybox_tabs:
        uuid = gr.Text(label="paiya", visible=True)
        with gr.TabItem('数字分身训练(Train)'):
            dummy_component = gr.Label(visible=False)
            with gr.Blocks():
                with gr.Row():
                    with gr.Column():
                        with gr.Box():
                            gr.Markdown('训练图片(Training photos)')
                            instance_images = gr.Gallery()
                            upload_button = gr.UploadButton(
                                "选择图片进行上传", file_types=["image"], file_count="multiple"
                            )

                            clear_button = gr.Button("清空图片(Clear photos)")
                            clear_button.click(fn=lambda: [], inputs=None, outputs=instance_images)

                            upload_button.upload(upload_file, inputs=[upload_button, instance_images], outputs=instance_images, queue=False)
                            
                            gr.Markdown('''
                                - 步骤1、上传需要训练的图片，5～10张日常照片。
                                - 步骤2、点击下方的 开始训练 按钮 ，启动训练流程，约20分钟。
                                - 步骤3、切换至 艺术照生成(Inference)，根据生模板成照片。
                                ''')

                run_button = gr.Button('开始训练'
                                    'Start training')

                with gr.Box():
                    gr.Markdown('''
                    请等待训练完成
                    
                    Please wait for the training to complete.
                    ''')
                    output_message = gr.Markdown()

                run_button.click(fn=paiya_train,
                                inputs=[
                                    dummy_component,
                                    uuid,
                                    instance_images,
                                ],
                                outputs=[output_message])

        with gr.TabItem('艺术照生成(Inference)'):
            dummy_component = gr.Label(visible=False)

            preset_template=glob(os.path.join('resources/inpaint_template/*.jpg'))
            with gr.Blocks() as demo:
                # Initialize the GUI
                with gr.Row():
                    with gr.Column():
                        template_gallery_list = [(i, i) for i in preset_template]
                        gallery = gr.Gallery(template_gallery_list).style(grid=4, height=300)
                        # new inplementation with gr.select callback function
                        def select_function(evt: gr.SelectData):
                            return [preset_template[evt.index]]
                        # selected_template_images = []
                        selected_template_images = gr.Text(show_label=False, placeholder="Selected")
                        gallery.select(select_function, None, selected_template_images)
                        
                        with gr.Accordion("Advanced Options", open=False):
                            append_pos_prompt = gr.Textbox(
                                label="Prompt",
                                lines=3,
                                value='masterpiece, smile, beauty',
                                interactive=True
                            )
                            first_control_weight = gr.Slider(
                                minimum=0.35, maximum=0.6, value=0.45,
                                step=0.02, label='初始权重(Initial Control Weight)'
                            )

                            second_control_weight = gr.Slider(
                                minimum=0.04, maximum=0.2, value=0.1,
                                step=0.02, label='二次权重(Secondary Control Weight)'
                            )
                            final_fusion_ratio = gr.Slider(
                                minimum=0.2, maximum=0.8, value=0.5,
                                step=0.1, label='融合系数(Final Fusion Ratio)'
                            )
                            use_fusion_before = gr.Radio(
                                label="前融合(Apply Fusion Before)", type="value", choices=[True, False],
                                value=True
                            )
                            use_fusion_after = gr.Radio(
                                label="后融合(Apply Fusion After)", type="value", choices=[True, False],
                                value=True
                            )
                        
                display_button = gr.Button('Start Generation')
                with gr.Box():
                    infer_progress = gr.Textbox(
                        label="生成(Generation Progress)",
                        value="No task currently",
                        interactive=False
                    )
                with gr.Box():
                    gr.Markdown('Generated Results')
                    output_images = gr.Gallery(
                        label='输出(Output)',
                        show_label=False
                    ).style(columns=3, rows=2, height=600, object_fit="contain")
                    
                display_button.click(
                    fn=paiya_forward,
                    inputs=[uuid, selected_template_images, append_pos_prompt, first_control_weight, second_control_weight,
                            final_fusion_ratio, use_fusion_before, use_fusion_after],
                    outputs=[infer_progress, output_images]
                )
        
    return [(skybox_tabs, "SkyBox", f"{id_part}_tabs")]


# 注册设置页的配置项
def on_ui_settings():
    section = ('skybox', "SkyBox")
    shared.opts.add_option("paiya_outpath_samples", shared.OptionInfo(
        paiya_outpath_samples, "SkyBox output path for image", section=section))  # 图片保存路径


script_callbacks.on_ui_settings(on_ui_settings)  # 注册进设置页
script_callbacks.on_ui_tabs(on_ui_tabs)
