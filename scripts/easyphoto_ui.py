import glob
import os
import time

import gradio as gr
import requests
from modules import script_callbacks, shared

from scripts.easyphoto_config import (cache_log_file_path, models_path,
                                      user_id_outpath_samples)
from scripts.easyphoto_infer import easyphoto_infer_forward
from scripts.easyphoto_train import easyphoto_train_forward
from scripts.easyphoto_utils import check_id_valid

gradio_compat = True

try:
    from distutils.version import LooseVersion

    from importlib_metadata import version
    if LooseVersion(version("gradio")) < LooseVersion("3.10"):
        gradio_compat = False
except ImportError:
    pass

def get_external_ckpts():
    external_checkpoints = []
    external_ckpt_dir = shared.cmd_opts.ckpt_dir if shared.cmd_opts.ckpt_dir else []
    if len(external_ckpt_dir) > 0:
        for _checkpoint in os.listdir(external_ckpt_dir):
            if _checkpoint.endswith(("pth", "safetensors", "ckpt")):
                external_checkpoints.append(_checkpoint)
    return external_checkpoints
external_checkpoints = get_external_ckpts()

def upload_file(files, current_files):
    file_paths = [file_d['name'] for file_d in current_files] + [file.name for file in files]
    return file_paths

def refresh_display():
    if not os.path.exists(os.path.dirname(cache_log_file_path)):
        os.makedirs(os.path.dirname(cache_log_file_path), exist_ok=True)
    lines_limit = 3
    try:
        with open(cache_log_file_path, "r", newline="") as f:
            lines = []
            for s in f.readlines():
                line = s.replace("\x00", "")
                if line.strip() == "" or line.strip() == "\r":
                    continue
                lines.append(line)

            total_lines = len(lines)
            if total_lines <= lines_limit:
                chatbot = [(None, ''.join(lines))]
            else:
                chatbot = [(None, ''.join(lines[total_lines-lines_limit:]))]
            return chatbot
    except Exception:
        with open(cache_log_file_path, "w") as f:
            pass
        return None

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", 
                         elem_classes=kwargs.pop('elem_classes', []) + ["cnet-toolbutton"], 
                         **kwargs)

    def get_block_name(self):
        return "button"
    
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as easyphoto_tabs:
        with gr.TabItem('Train'):
            dummy_component = gr.Label(visible=False)
            with gr.Blocks():
                with gr.Row():
                    uuid = gr.Text(label="User_ID", value="", visible=False)

                    with gr.Column():
                        gr.Markdown('Training photos')

                        instance_images = gr.Gallery().style(columns=[4], rows=[2], object_fit="contain", height="auto")

                        with gr.Row():
                            upload_button = gr.UploadButton(
                                "Upload Photos", file_types=["image"], file_count="multiple"
                            )
                            clear_button = gr.Button("Clear Photos")
                        clear_button.click(fn=lambda: [], inputs=None, outputs=instance_images)

                        upload_button.upload(upload_file, inputs=[upload_button, instance_images], outputs=instance_images, queue=False)
                    
                        gr.Markdown(
                            '''
                            Training steps:
                            1. Please upload 5-20 half-body photos or head-and-shoulder photos, and please don't make the proportion of your face too small.
                            2. Click on the Start Training button below to start the training process, approximately 25 minutes.
                            3. Switch to Inference and generate photos based on the template. 
                            4. If you encounter lag when uploading, please modify the size of the uploaded pictures and try to limit it to 1.5MB.
                            '''
                        )
                    with gr.Column():
                        gr.Markdown('Params Setting')
                        with gr.Accordion("Advanced Options", open=True):
                            with gr.Row():
                                def checkpoint_refresh_function():
                                    checkpoints = []
                                    for _checkpoint in os.listdir(os.path.join(models_path, "Stable-diffusion")):
                                        if _checkpoint.endswith(("pth", "safetensors", "ckpt")):
                                            checkpoints.append(_checkpoint)
                                    return gr.update(choices=list(set(["Chilloutmix-Ni-pruned-fp16-fix.safetensors"] + checkpoints + external_checkpoints)))
                                
                                checkpoints = []
                                for _checkpoint in os.listdir(os.path.join(models_path, "Stable-diffusion")):
                                    if _checkpoint.endswith(("pth", "safetensors", "ckpt")):
                                        checkpoints.append(_checkpoint)
                                sd_model_checkpoint = gr.Dropdown(value="Chilloutmix-Ni-pruned-fp16-fix.safetensors", choices=list(set(["Chilloutmix-Ni-pruned-fp16-fix.safetensors"] + checkpoints + external_checkpoints)), label="The base checkpoint you use.", visible=True)

                                checkpoint_refresh = ToolButton(value="\U0001f504")
                                checkpoint_refresh.click(
                                    fn=checkpoint_refresh_function,
                                    inputs=[],
                                    outputs=[sd_model_checkpoint]
                                )

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
                            with gr.Row():
                                validation = gr.Checkbox(
                                    label="Validation",  
                                    value=True
                                )

                        gr.Markdown(
                            '''
                            Parameter parsing:
                            - **max steps per photo** represents the maximum number of training steps per photo.
                            - **max train steps** represents the maximum training step.
                            - **Validation** Whether to validate at training time.
                            - Final training step = Min(photo_num * max_steps_per_photos, max_train_steps)
                            '''
                        )

                with gr.Row():
                    with gr.Column(width=3):
                        run_button = gr.Button('Start Training')
                    with gr.Column(width=1):
                        refresh_button = gr.Button('Refresh Log')

                gr.Markdown(
                    '''
                    We need to train first to predict, please wait for the training to complete, thank you for your patience.  
                    '''
                )
                output_message  = gr.Markdown()

                with gr.Box():
                    logs_out        = gr.Chatbot(label='Training Logs', height=200)
                    block           = gr.Blocks()
                    with block:
                        block.load(refresh_display, None, logs_out, every=3)

                    refresh_button.click(
                        fn = refresh_display,
                        inputs = [],
                        outputs = [logs_out]
                    )

                run_button.click(fn=easyphoto_train_forward,
                                _js="ask_for_style_name",
                                inputs=[
                                    sd_model_checkpoint, dummy_component,
                                    uuid,
                                    resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos, train_batch_size, gradient_accumulation_steps, dataloader_num_workers, learning_rate, rank, network_alpha, validation, instance_images,
                                ],
                                outputs=[output_message])
                                
        with gr.TabItem('Inference'):
            dummy_component = gr.Label(visible=False)
            training_templates = glob.glob(os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), 'training_templates/*.jpg'))
            infer_templates = glob.glob(os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), 'infer_templates/*.jpg'))
            preset_template = list(training_templates) + list(infer_templates)

            with gr.Blocks() as demo:
                with gr.Row():
                    with gr.Column():
                        model_selected_tab = gr.State(0)

                        with gr.TabItem("template images") as template_images_tab:
                            template_gallery_list = [(i, i) for i in preset_template]
                            gallery = gr.Gallery(template_gallery_list).style(columns=[4], rows=[2], object_fit="contain", height="auto")
                            
                            def select_function(evt: gr.SelectData):
                                return [preset_template[evt.index]]

                            selected_template_images = gr.Text(show_label=False, visible=False, placeholder="Selected")
                            gallery.select(select_function, None, selected_template_images)
                            
                        with gr.TabItem("upload image") as upload_image_tab:
                            init_image = gr.Image(label="Image for skybox", elem_id="{id_part}_image", show_label=False, source="upload")
                            

                        with gr.TabItem("upload dir") as upload_dir_tab:
                            uploaded_template_images = gr.Gallery().style(columns=[4], rows=[2], object_fit="contain", height="auto")

                            with gr.Row():
                                upload_dir_button = gr.UploadButton(
                                    "Upload Photos", file_types=["image"], file_count="multiple"
                                )
                                clear_dir_button = gr.Button("Clear Photos")
                            clear_dir_button.click(fn=lambda: [], inputs=None, outputs=uploaded_template_images)

                            upload_dir_button.upload(upload_file, inputs=[upload_dir_button, uploaded_template_images], outputs=uploaded_template_images, queue=False)

                        with gr.TabItem("generate from none") as generate_tab:
                            with gr.Row():
                                portrait_ratio  = gr.Dropdown(value="Half-body photo", elem_id='dropdown', choices=["upper-body", "headshot"], label="The Portrait ratio.", visible=True)
                                gender          = gr.Dropdown(value="girl", elem_id='dropdown', choices=["girl", "woman", "boy", "man"], label="The gender of the person.", visible=True)
                                cloth_color     = gr.Dropdown(value="white", elem_id='dropdown', choices=["white", "orange", "pink", "black", "red", "blue"], label="The Color of the cloth.", visible=True)
                                cloth           = gr.Dropdown(value="shirt", elem_id='dropdown', choices=["shirt", "overcoat", "dress", "coat", "vest"], label="The cloth on the person.", visible=True)
                            with gr.Row():
                                doing           = gr.Dropdown(value="standing", elem_id='dropdown', choices=["standing", "sit"], label="What does the person do?", visible=True)
                                where           = gr.Dropdown(value="in the garden", elem_id='dropdown', choices=["in the garden", "in the house", "on the lawn", "besides the sea", "besides the lake", "on the bridge"], label="Where is the person?", visible=True)
                                time_of_photo   = gr.Dropdown(value="daytime", elem_id='dropdown', choices=["daytime", "night"], label="Where is the time?", visible=True)

                            sd_xl_input_prompt = gr.Text(
                                value="upper-body, look at viewer, one twenty years old girl, wear white shit, standing, in the garden, daytime, f32", show_label=False, visible=True
                            )

                            def update_sd_xl_input_prompt(portrait_ratio, gender, cloth_color, cloth, doing, where, time_of_photo):
                                input_prompt = f"{portrait_ratio}, look at viewer, one twenty years old {gender}, wear {cloth_color} {cloth}, {doing}, {where}, {time_of_photo}, f32"
                                return input_prompt

                            prompt_inputs = [portrait_ratio, gender, cloth_color, cloth, doing, where, time_of_photo]
                            portrait_ratio.change(update_sd_xl_input_prompt, inputs=prompt_inputs, outputs=sd_xl_input_prompt)

                        model_selected_tabs = [template_images_tab, upload_image_tab, upload_dir_tab, generate_tab]
                        for i, tab in enumerate(model_selected_tabs):
                            tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[model_selected_tab])
                        
                        with gr.Row():
                            def checkpoint_refresh_function():
                                checkpoints = []
                                for _checkpoint in os.listdir(os.path.join(models_path, "Stable-diffusion")):
                                    if _checkpoint.endswith(("pth", "safetensors", "ckpt")):
                                        checkpoints.append(_checkpoint)
                                return gr.update(choices=list(set(["Chilloutmix-Ni-pruned-fp16-fix.safetensors"] + checkpoints + external_checkpoints)))
                            
                            checkpoints = []
                            for _checkpoint in os.listdir(os.path.join(models_path, "Stable-diffusion")):
                                if _checkpoint.endswith(("pth", "safetensors", "ckpt")):
                                    checkpoints.append(_checkpoint)
                            sd_model_checkpoint = gr.Dropdown(value="Chilloutmix-Ni-pruned-fp16-fix.safetensors", choices=list(set(["Chilloutmix-Ni-pruned-fp16-fix.safetensors"] + checkpoints + external_checkpoints)), label="The base checkpoint you use.", visible=True)

                            checkpoint_refresh = ToolButton(value="\U0001f504")
                            checkpoint_refresh.click(
                                fn=checkpoint_refresh_function,
                                inputs=[],
                                outputs=[sd_model_checkpoint]
                            )

                        with gr.Row():
                            def select_function():
                                ids = []
                                if os.path.exists(user_id_outpath_samples):
                                    _ids = os.listdir(user_id_outpath_samples)
                                    for _id in _ids:
                                        if check_id_valid(_id, user_id_outpath_samples, models_path):
                                            ids.append(_id)
                                ids = sorted(ids)
                                return gr.update(choices=["none"] + ids)

                            ids = []
                            if os.path.exists(user_id_outpath_samples):
                                _ids = os.listdir(user_id_outpath_samples)
                                for _id in _ids:
                                    if check_id_valid(_id, user_id_outpath_samples, models_path):
                                        ids.append(_id)
                                ids = sorted(ids)

                            num_of_faceid = gr.Dropdown(value=str(1), elem_id='dropdown', choices=[1, 2, 3, 4, 5], label=f"Num of Faceid")

                            uuids           = []
                            visibles        = [True, False, False, False, False]
                            for i in range(int(5)):
                                uuid = gr.Dropdown(value="none", elem_id='dropdown', choices=["none"] + ids, min_width=140, label=f"User_{i} id", visible=visibles[i])
                                uuids.append(uuid)

                            def update_uuids(_num_of_faceid):
                                _uuids = []
                                for i in range(int(_num_of_faceid)):
                                    _uuids.append(gr.update(value="none", visible=True))
                                for i in range(int(5 - int(_num_of_faceid))):
                                    _uuids.append(gr.update(value="none", visible=False))
                                return _uuids
                            
                            num_of_faceid.change(update_uuids, inputs=[num_of_faceid], outputs=uuids)
                            
                            refresh = ToolButton(value="\U0001f504")
                            for i in range(int(5)):
                                refresh.click(
                                    fn=select_function,
                                    inputs=[],
                                    outputs=[uuids[i]]
                                )

                        with gr.Accordion("Advanced Options", open=False):
                            additional_prompt = gr.Textbox(
                                label="Additional Prompt",
                                lines=3,
                                value='masterpiece, beauty',
                                interactive=True
                            )
                            seed = gr.Textbox(
                                label="Seed", 
                                value=12345,
                            )
                            with gr.Row():
                                before_face_fusion_ratio = gr.Slider(
                                    minimum=0.2, maximum=0.8, value=0.50,
                                    step=0.05, label='Face Fusion Ratio Before'
                                )
                                after_face_fusion_ratio = gr.Slider(
                                    minimum=0.2, maximum=0.8, value=0.50,
                                    step=0.05, label='Face Fusion Ratio After'
                                )

                            with gr.Row():
                                first_diffusion_steps = gr.Slider(
                                    minimum=15, maximum=50, value=50,
                                    step=1, label='First Diffusion steps'
                                )
                                first_denoising_strength = gr.Slider(
                                    minimum=0.30, maximum=0.60, value=0.45,
                                    step=0.05, label='First Diffusion denoising strength'
                                )
                            with gr.Row():
                                second_diffusion_steps = gr.Slider(
                                    minimum=15, maximum=50, value=20,
                                    step=1, label='Second Diffusion steps'
                                )
                                second_denoising_strength = gr.Slider(
                                    minimum=0.20, maximum=0.40, value=0.30,
                                    step=0.05, label='Second Diffusion denoising strength'
                                )
                            with gr.Row():
                                crop_face_preprocess = gr.Checkbox(
                                    label="Crop Face Preprocess",  
                                    value=True
                                )
                                apply_face_fusion_before = gr.Checkbox(
                                    label="Apply Face Fusion Before", 
                                    value=True
                                )
                                apply_face_fusion_after = gr.Checkbox(
                                    label="Apply Face Fusion After",  
                                    value=True
                                )
                            with gr.Row():
                                color_shift_middle = gr.Checkbox(
                                    label="Apply color shift first",  
                                    value=True
                                )
                                color_shift_last = gr.Checkbox(
                                    label="Apply color shift last",  
                                    value=True
                                )
                                super_resolution = gr.Checkbox(
                                    label="Super Resolution at last",  
                                    value=True
                                )
                            with gr.Row():
                                display_score = gr.Checkbox(
                                    label="Display Face Similarity Scores",  
                                    value=False
                                )
                                background_restore = gr.Checkbox(
                                    label="Background Restore",  
                                    value=False
                                )
                            with gr.Row():
                                background_restore_denoising_strength = gr.Slider(
                                    minimum=0.10, maximum=0.60, value=0.35,
                                    step=0.05, label='Background restore denoising strength',
                                    visible=False
                                )
                                background_restore.change(lambda x: background_restore_denoising_strength.update(visible=x), inputs=[background_restore], outputs=[background_restore_denoising_strength])

                            with gr.Box():
                                gr.Markdown(
                                    '''
                                    Parameter parsing:
                                    1. **Face Fusion Ratio Before** represents the proportion of the first facial fusion, which is higher and more similar to the training object.  
                                    2. **Face Fusion Ratio After** represents the proportion of the second facial fusion, which is higher and more similar to the training object.  
                                    3. **Crop Face Preprocess** represents whether to crop the image before generation, which can adapt to images with smaller faces.  
                                    4. **Apply Face Fusion Before** represents whether to perform the first facial fusion.  
                                    5. **Apply Face Fusion After** represents whether to perform the second facial fusion. 
                                    6. **Display Face Similarity Scores** represents whether to compute the face similarity score of the generated image with the ID photo.
                                    7. **Background Restore** represents whether to give a different background.
                                    '''
                                )
                            
                        display_button = gr.Button('Start Generation')

                    with gr.Column():
                        gr.Markdown('Generated Results')

                        output_images = gr.Gallery(
                            label='Output',
                            show_label=False
                        ).style(columns=[4], rows=[2], object_fit="contain", height="auto")

                        face_id_text    = gr.Markdown("Face Similarity Scores", visible=False)
                        face_id_outputs = gr.Gallery(
                            label ="Face Similarity Scores",
                            show_label=False,
                            visible=False,
                        ).style(columns=[4], rows=[1], object_fit="contain", height="auto")
                        # Display Face Similarity Scores if the user intend to do it.
                        display_score.change(lambda x: face_id_text.update(visible=x), inputs=[display_score], outputs=[face_id_text])
                        display_score.change(lambda x: face_id_outputs.update(visible=x), inputs=[display_score], outputs=[face_id_outputs])

                        infer_progress = gr.Textbox(
                            label="Generation Progress",
                            value="No task currently",
                            interactive=False
                        )
                    
                display_button.click(
                    fn=easyphoto_infer_forward,
                    inputs=[sd_model_checkpoint, selected_template_images, init_image, uploaded_template_images, additional_prompt, 
                            before_face_fusion_ratio, after_face_fusion_ratio, first_diffusion_steps, first_denoising_strength, second_diffusion_steps, second_denoising_strength, \
                            seed, crop_face_preprocess, apply_face_fusion_before, apply_face_fusion_after, color_shift_middle, color_shift_last, super_resolution, display_score, \
                            background_restore, background_restore_denoising_strength, sd_xl_input_prompt, model_selected_tab, *uuids],
                    outputs=[infer_progress, output_images, face_id_outputs]

                )
            
    return [(easyphoto_tabs, "EasyPhoto", f"EasyPhoto_tabs")]

# 注册设置页的配置项
def on_ui_settings():
    section = ('EasyPhoto', "EasyPhoto")
    shared.opts.add_option("easyphoto_cache_model", shared.OptionInfo(
        True, "Cache preprocess model in Inference", gr.Checkbox, {}, section=section))

script_callbacks.on_ui_settings(on_ui_settings)  # 注册进设置页
script_callbacks.on_ui_tabs(on_ui_tabs)
