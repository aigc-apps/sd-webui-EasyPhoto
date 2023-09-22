import gradio as gr
from fastapi import FastAPI
from modules.api import api
from modules.api.models import *
from scripts.easyphoto_infer import *
from scripts.easyphoto_train import *
import hashlib
from PIL import Image
import os

def easyphoto_train_forward_api(_: gr.Blocks, app: FastAPI):
    @app.post("/easyphoto/easyphoto_train_forward")
    def _easyphoto_train_forward_api(
        imgs: dict,
    ):
        sd_model_checkpoint     = imgs.get("sd_model_checkpoint", "Chilloutmix-Ni-pruned-fp16-fix.safetensors")
        id_task                 = imgs.get("id_task", "")
        user_id                 = imgs.get("user_id", "tmp")
        resolution              = imgs.get("resolution", 512)
        val_and_checkpointing_steps = imgs.get("val_and_checkpointing_steps", 100)
        max_train_steps         = imgs.get("max_train_steps", 800)
        steps_per_photos        = imgs.get("steps_per_photos", 200)
        train_batch_size        = imgs.get("train_batch_size", 1)

        gradient_accumulation_steps = imgs.get("gradient_accumulation_steps", 4)
        dataloader_num_workers  = imgs.get("dataloader_num_workers", 16)
        learning_rate           = imgs.get("learning_rate", 1e-4)
        rank                    = imgs.get("rank", 128)
        network_alpha           = imgs.get("network_alpha", 64)
        instance_images         = imgs.get("instance_images", [])
        validation              = imgs.get("validation", True)
        args                    = imgs.get("args", []) 

        instance_images         = [api.decode_base64_to_image(init_image) for init_image in instance_images]
        _instance_images        = []
        for instance_image in instance_images:
            hash_value = hashlib.md5(instance_image.tobytes()).hexdigest()
            save_path = os.path.join('/tmp', hash_value + '.jpg')
            instance_image.save(save_path)
            _instance_images.append(
                {"name" : save_path}
            )
        instance_images = _instance_images

        message = easyphoto_train_forward(
            sd_model_checkpoint,
            id_task,
            user_id,
            resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos,
            train_batch_size, gradient_accumulation_steps, dataloader_num_workers, learning_rate, 
            rank, network_alpha,
            validation,
            instance_images,
            *args
        )
        return {"message": message}
    
def easyphoto_infer_forward_api(_: gr.Blocks, app: FastAPI):
    @app.post("/easyphoto/easyphoto_infer_forward")
    def _easyphoto_infer_forward_api(
        imgs: dict,
    ):
        user_ids                    = imgs.get("user_ids", [])
        sd_model_checkpoint         = imgs.get("sd_model_checkpoint", "Chilloutmix-Ni-pruned-fp16-fix.safetensors")
        selected_template_images    = imgs.get("selected_template_images", [])
        init_image                  = imgs.get("init_image", None)
        uploaded_template_images    = imgs.get("uploaded_template_images", [])
        additional_prompt           = imgs.get("additional_prompt", "")

        first_diffusion_steps       = imgs.get("first_diffusion_steps", 50)
        first_denoising_strength    = imgs.get("first_denoising_strength", 0.45)

        second_diffusion_steps      = imgs.get("second_diffusion_steps", 20)
        second_denoising_strength   = imgs.get("second_denoising_strength", 0.35)
        seed                        = imgs.get("seed", -1)
        crop_face_preprocess        = imgs.get("crop_face_preprocess", True)

        before_face_fusion_ratio    = imgs.get("before_face_fusion_ratio", 0.50)
        after_face_fusion_ratio     = imgs.get("after_face_fusion_ratio", 0.50)
        apply_face_fusion_before    = imgs.get("apply_face_fusion_before", True)
        apply_face_fusion_after     = imgs.get("apply_face_fusion_after", True)
        color_shift_middle          = imgs.get("color_shift_middle", True)
        color_shift_last            = imgs.get("color_shift_last", True)
        super_resolution            = imgs.get("super_resolution", True)
        display_score               = imgs.get("display_score", False)
        background_restore          = imgs.get("background_restore", False)
        background_restore_denoising_strength = imgs.get("background_restore", 0.35)
        tabs                        = imgs.get("tabs", 1)

        selected_template_images    = [api.decode_base64_to_image(_) for _ in selected_template_images]
        init_image                  = None if init_image is None else api.decode_base64_to_image(init_image)
        uploaded_template_images    = [api.decode_base64_to_image(_) for _ in uploaded_template_images]

        _selected_template_images = []
        for selected_template_image in selected_template_images:
            hash_value = hashlib.md5(selected_template_image.tobytes()).hexdigest()
            save_path = os.path.join('/tmp', hash_value + '.jpg')
            selected_template_image.save(save_path)
            _selected_template_images.append(save_path)
        selected_template_images = str(_selected_template_images)

        if init_image is not None:
            init_image = np.uint8(init_image)
        
        _uploaded_template_images = []
        for uploaded_template_image in uploaded_template_images:
            hash_value = hashlib.md5(uploaded_template_image.tobytes()).hexdigest()
            _uploaded_template_images.append(
                {"name" : save_path}
            )
        uploaded_template_images = _uploaded_template_images
        
        comment, outputs, face_id_outputs = easyphoto_infer_forward(
            sd_model_checkpoint, selected_template_images, init_image, uploaded_template_images, additional_prompt, \
            before_face_fusion_ratio, after_face_fusion_ratio, first_diffusion_steps, first_denoising_strength, second_diffusion_steps, second_denoising_strength, \
            seed, crop_face_preprocess, apply_face_fusion_before, apply_face_fusion_after, color_shift_middle, color_shift_last, super_resolution, display_score, background_restore, \
            background_restore_denoising_strength, tabs, *user_ids
        )
        outputs = [api.encode_pil_to_base64(output) for output in outputs]
        
        return {"message": comment, "outputs": outputs, "face_id_outputs": face_id_outputs}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(easyphoto_train_forward_api)
    script_callbacks.on_app_started(easyphoto_infer_forward_api)
except:
    pass