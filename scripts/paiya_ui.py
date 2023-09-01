import os

import gradio as gr
import glob
import requests

from scripts.paiya_infer import paiya_infer_forward
from scripts.paiya_config import paiya_outpath_samples, id_path
from scripts.paiya_train import paiya_train_forward
from modules import script_callbacks, shared
from modules.paths import models_path

gradio_compat = True

try:
    from distutils.version import LooseVersion

    from importlib_metadata import version
    if LooseVersion(version("gradio")) < LooseVersion("3.10"):
        gradio_compat = False
except ImportError:
    pass


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", 
                         elem_classes=kwargs.pop('elem_classes', []) + ["cnet-toolbutton"], 
                         **kwargs)

    def get_block_name(self):
        return "button"

def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as paiya_tabs:
        with gr.TabItem('数字分身训练(Train)'):
            dummy_component = gr.Label(visible=False)
            with gr.Blocks():
                with gr.Row():
                    uuid = gr.Text(label="User_ID", value="", visible=False)

                    with gr.Column():
                        gr.Markdown('训练图片(Training photos)')

                        instance_images = gr.Gallery().style(columns=[4], rows=[2], object_fit="contain", height="auto")

                        with gr.Row():
                            upload_button = gr.UploadButton(
                                "选择图片进行上传", file_types=["image"], file_count="multiple"
                            )
                            clear_button = gr.Button("清空图片")
                        clear_button.click(fn=lambda: [], inputs=None, outputs=instance_images)

                        upload_button.upload(upload_file, inputs=[upload_button, instance_images], outputs=instance_images, queue=False)
                        
                    with gr.Column():
                        gr.Markdown('参数设置(Params Setting)')
                        with gr.Accordion("Advanced Options", open=True):
                            with gr.Row():
                                resolution = gr.Textbox(
                                    label="resolution",
                                    value=512,
                                    interactive=True
                                )
                                val_and_checkpointing_steps = gr.Textbox(
                                    label="validation & save steps",
                                    value=100,
                                    interactive=True
                                )
                                max_train_steps = gr.Textbox(
                                    label="max train steps",
                                    value=800,
                                    interactive=True
                                )
                                steps_per_photos = gr.Textbox(
                                    label="max steps per photos",
                                    value=200,
                                    interactive=True
                                )

                            with gr.Row():
                                train_batch_size = gr.Textbox(
                                    label="train batch size",
                                    value=1,
                                    interactive=True
                                )
                                gradient_accumulation_steps = gr.Textbox(
                                    label="gradient accumulationsteps",
                                    value=4,
                                    interactive=True
                                )
                                dataloader_num_workers =  gr.Textbox(
                                    label="dataloader num workers",
                                    value=16,
                                    interactive=True
                                )
                                learning_rate = gr.Textbox(
                                    label="learning rate",
                                    value=1e-4,
                                    interactive=True
                                )
                            with gr.Row():
                                rank = gr.Textbox(
                                    label="rank",
                                    value=128,
                                    interactive=True
                                )
                                network_alpha = gr.Textbox(
                                    label="network alpha",
                                    value=64,
                                    interactive=True
                                )
                        gr.Markdown('''
                            - 步骤1、上传需要训练的图片，5～10张日常照片。
                            - 步骤2、点击下方的 开始训练 按钮 ，启动训练流程，约20分钟。
                            - 步骤3、切换至 艺术照生成(Inference)，根据生模板成照片。
                            ''')

                with gr.Column():
                    run_button = gr.Button('开始训练'
                                        'Start training')

                with gr.Box():
                    gr.Markdown('''
                    请等待训练完成
                    
                    Please wait for the training to complete.
                    ''')
                    output_message = gr.Markdown()
            
                run_button.click(fn=paiya_train_forward,
                                _js="ask_for_style_name",
                                inputs=[
                                    dummy_component,
                                    uuid,
                                    resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos, train_batch_size, gradient_accumulation_steps, dataloader_num_workers, learning_rate, rank, network_alpha, instance_images,
                                ],
                                outputs=[output_message])

        with gr.TabItem('艺术照生成(Inference)'):
            dummy_component = gr.Label(visible=False)
            preset_template = glob.glob(os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), 'templates/*.jpg'))

            with gr.Blocks() as demo:
                with gr.Row():
                    with gr.Column():
                        model_selected_tab = gr.State(0)
                        # Initialize the GUI
                        with gr.TabItem("template images") as template_images_tab:
                            template_gallery_list = [(i, i) for i in preset_template]
                            gallery = gr.Gallery(template_gallery_list).style(columns=[4], rows=[2], object_fit="contain", height="auto")
                            # new inplementation with gr.select callback function
                            def select_function(evt: gr.SelectData):
                                return [preset_template[evt.index]]
                            # selected_template_images = []
                            selected_template_images = gr.Text(show_label=False, visible=False, placeholder="Selected")
                            gallery.select(select_function, None, selected_template_images)
                            
                        with gr.TabItem("upload image") as upload_image_tab:
                            init_image = gr.Image(label="Image for skybox", elem_id="{id_part}_image", show_label=False, source="upload")
                            
                        model_selected_tabs = [template_images_tab, upload_image_tab]
                        for i, tab in enumerate(model_selected_tabs):
                            tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[model_selected_tab])

                        with gr.Row():
                            def select_function():
                                if os.path.exists(id_path):
                                    with open(id_path, "r") as f:
                                        ids = f.readlines()
                                    ids = [_id.strip() for _id in ids]
                                else:
                                    ids = []
                                return gr.update(choices=ids)

                            if os.path.exists(id_path):
                                with open(id_path, "r") as f:
                                    ids = f.readlines()
                                ids = [_id.strip() for _id in ids]
                            else:
                                ids = []
                            uuid    = gr.Dropdown(value=ids[0] if len(ids) > 0 else "none", choices=ids, label="Used id (The User id you provide while training)", visible=True)

                            refresh = ToolButton(value="\U0001f504")
                            refresh.click(
                                fn=select_function,
                                inputs=[],
                                outputs=[uuid]
                            )

                        with gr.Accordion("Advanced Options", open=False):
                            append_pos_prompt = gr.Textbox(
                                label="Prompt",
                                lines=3,
                                value='masterpiece, beauty',
                                interactive=True
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

                    with gr.Column():
                        gr.Markdown('Generated Results')

                        output_images = gr.Gallery(
                            label='输出(Output)',
                            show_label=False
                        ).style(columns=[4], rows=[2], object_fit="contain", height="auto")
                        infer_progress = gr.Textbox(
                            label="生成(Generation Progress)",
                            value="No task currently",
                            interactive=False
                        )
                    
                display_button.click(
                    fn=paiya_infer_forward,
                    inputs=[uuid, selected_template_images, init_image, append_pos_prompt, 
                            final_fusion_ratio, use_fusion_before, use_fusion_after, model_selected_tab],
                    outputs=[infer_progress, output_images]
                )
            
    return [(paiya_tabs, "Paiya", f"paiya_tabs")]

# 注册设置页的配置项
def on_ui_settings():
    section = ('paiya', "Paiya")
    shared.opts.add_option("paiya_outpath_samples", shared.OptionInfo(
        paiya_outpath_samples, "Paiya output path for image", section=section))  # 图片保存路径

script_callbacks.on_ui_settings(on_ui_settings)  # 注册进设置页
script_callbacks.on_ui_tabs(on_ui_tabs)
