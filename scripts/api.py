import base64
import hashlib
import os

import gradio as gr
import numpy as np
import torch
from fastapi import FastAPI
from modules.api import api

from scripts.easyphoto_infer import easyphoto_infer_forward
from scripts.easyphoto_train import easyphoto_train_forward
from scripts.easyphoto_utils import ep_logger


def easyphoto_train_forward_api(_: gr.Blocks, app: FastAPI):
    @app.post("/easyphoto/easyphoto_train_forward")
    def _easyphoto_train_forward_api(
        datas: dict,
    ):
        sd_model_checkpoint = datas.get("sd_model_checkpoint", "Chilloutmix-Ni-pruned-fp16-fix.safetensors")
        id_task = datas.get("id_task", "")
        user_id = datas.get("user_id", "tmp")
        train_mode_choose = datas.get("train_mode_choose", "Train Human Lora")
        resolution = datas.get("resolution", 512)
        val_and_checkpointing_steps = datas.get("val_and_checkpointing_steps", 100)
        max_train_steps = datas.get("max_train_steps", 800)
        steps_per_photos = datas.get("steps_per_photos", 200)
        train_batch_size = datas.get("train_batch_size", 1)

        gradient_accumulation_steps = datas.get("gradient_accumulation_steps", 4)
        dataloader_num_workers = datas.get("dataloader_num_workers", 16)
        learning_rate = datas.get("learning_rate", 1e-4)
        rank = datas.get("rank", 128)
        network_alpha = datas.get("network_alpha", 64)
        instance_images = datas.get("instance_images", [])
        validation = datas.get("validation", True)

        enable_rl = datas.get("enable_rl", False)
        max_rl_time = datas.get("max_rl_time", 1)
        timestep_fraction = datas.get("timestep_fraction", 1)
        skin_retouching_bool = datas.get("skin_retouching_bool", False)
        training_prefix_prompt = datas.get("training_prefix_prompt", "")
        crop_ratio = datas.get("crop_ratio", 3)
        args = datas.get("args", [])

        instance_images = [api.decode_base64_to_image(init_image) for init_image in instance_images]
        _instance_images = []
        for instance_image in instance_images:
            hash_value = hashlib.md5(instance_image.tobytes()).hexdigest()
            save_path = os.path.join("/tmp", hash_value + ".jpg")
            instance_image = instance_image.convert("RGB")
            instance_image.save(save_path)
            _instance_images.append({"name": save_path})
        instance_images = _instance_images

        try:
            message = easyphoto_train_forward(
                sd_model_checkpoint,
                id_task,
                user_id,
                train_mode_choose,
                resolution,
                val_and_checkpointing_steps,
                max_train_steps,
                steps_per_photos,
                train_batch_size,
                gradient_accumulation_steps,
                dataloader_num_workers,
                learning_rate,
                rank,
                network_alpha,
                validation,
                instance_images,
                enable_rl,
                max_rl_time,
                timestep_fraction,
                skin_retouching_bool,
                training_prefix_prompt,
                crop_ratio,
                *args,
            )
        except Exception as e:
            torch.cuda.empty_cache()
            message = f"Train error, error info:{str(e)}"
            ep_logger.error(message)

        return {"message": message}


def easyphoto_infer_forward_api(_: gr.Blocks, app: FastAPI):
    @app.post("/easyphoto/easyphoto_infer_forward")
    def _easyphoto_infer_forward_api(
        datas: dict,
    ):
        user_ids = datas.get("user_ids", [])
        sd_model_checkpoint = datas.get("sd_model_checkpoint", "Chilloutmix-Ni-pruned-fp16-fix.safetensors")
        selected_template_images = datas.get("selected_template_images", [])
        init_image = datas.get("init_image", None)
        uploaded_template_images = datas.get("uploaded_template_images", [])

        text_to_image_input_prompt = datas.get(
            "text_to_image_input_prompt",
            "upper-body, look at viewer, one twenty years old girl, wear white dress, standing, in the garden with flowers, in the winter, daytime, snow, f32",
        )
        text_to_image_width = datas.get("text_to_image_width", 624)
        text_to_image_height = datas.get("text_to_image_height", 832)

        t2i_control_way = datas.get("t2i_control_way", "Control with inner template")
        t2i_pose_template = datas.get("t2i_pose_template", None)

        scene_id = datas.get("scene_id", "none")
        prompt_generate_sd_model_checkpoint = datas.get("sd_model_checkpoint", "LZ-16K+Optics.safetensors")

        additional_prompt = datas.get("additional_prompt", "")
        lora_weights = datas.get("lora_weights", 0.9)

        first_diffusion_steps = datas.get("first_diffusion_steps", 50)
        first_denoising_strength = datas.get("first_denoising_strength", 0.45)

        second_diffusion_steps = datas.get("second_diffusion_steps", 20)
        second_denoising_strength = datas.get("second_denoising_strength", 0.35)
        seed = datas.get("seed", -1)
        crop_face_preprocess = datas.get("crop_face_preprocess", True)

        before_face_fusion_ratio = datas.get("before_face_fusion_ratio", 0.50)
        after_face_fusion_ratio = datas.get("after_face_fusion_ratio", 0.50)
        apply_face_fusion_before = datas.get("apply_face_fusion_before", True)
        apply_face_fusion_after = datas.get("apply_face_fusion_after", True)
        color_shift_middle = datas.get("color_shift_middle", True)
        color_shift_last = datas.get("color_shift_last", True)
        super_resolution = datas.get("super_resolution", True)
        super_resolution_method = datas.get("super_resolution_method", "gpen")
        skin_retouching_bool = datas.get("skin_retouching_bool", False)
        display_score = datas.get("display_score", False)
        background_restore = datas.get("background_restore", False)
        background_restore_denoising_strength = datas.get("background_restore_denoising_strength", 0.35)
        makeup_transfer = datas.get("makeup_transfer", False)
        makeup_transfer_ratio = datas.get("makeup_transfer_ratio", 0.50)
        face_shape_match = datas.get("face_shape_match", False)
        tabs = datas.get("tabs", 1)

        ipa_control = datas.get("ipa_control", False)
        ipa_weight = datas.get("ipa_weight", 0.50)
        ipa_image = datas.get("ipa_image", None)

        ref_mode_choose = datas.get("ref_mode_choose", "Infer with Pretrained Lora")
        ipa_only_weight = datas.get("ipa_only_weight", 0.60)
        ipa_only_image = datas.get("ipa_only_image", None)

        lcm_accelerate = datas.get("lcm_accelerate", None)

        if type(user_ids) == str:
            user_ids = [user_ids]

        selected_template_images = [api.decode_base64_to_image(_) for _ in selected_template_images]
        init_image = None if init_image is None else api.decode_base64_to_image(init_image)
        uploaded_template_images = [api.decode_base64_to_image(_) for _ in uploaded_template_images]
        t2i_pose_template = None if t2i_pose_template is None else api.decode_base64_to_image(t2i_pose_template)
        ipa_image = None if ipa_image is None else api.decode_base64_to_image(ipa_image)
        ipa_only_image = None if ipa_only_image is None else api.decode_base64_to_image(ipa_only_image)

        _selected_template_images = []
        for selected_template_image in selected_template_images:
            hash_value = hashlib.md5(selected_template_image.tobytes()).hexdigest()
            save_path = os.path.join("/tmp", hash_value + ".jpg")
            selected_template_image.save(save_path)
            _selected_template_images.append(save_path)
        selected_template_images = str(_selected_template_images)

        if init_image is not None:
            init_image = np.uint8(init_image)

        _uploaded_template_images = []
        for uploaded_template_image in uploaded_template_images:
            hash_value = hashlib.md5(uploaded_template_image.tobytes()).hexdigest()
            save_path = os.path.join("/tmp", hash_value + ".jpg")
            uploaded_template_image.save(save_path)
            _uploaded_template_images.append({"name": save_path})
        uploaded_template_images = _uploaded_template_images

        if t2i_pose_template is not None:
            t2i_pose_template = np.uint8(t2i_pose_template)

        if ipa_image is not None:
            hash_value = hashlib.md5(ipa_image.tobytes()).hexdigest()
            save_path = os.path.join("/tmp", hash_value + ".jpg")
            ipa_image.save(save_path)
            ipa_image_path = save_path
        else:
            ipa_image_path = None

        if ipa_only_image is not None:
            hash_value = hashlib.md5(ipa_only_image.tobytes()).hexdigest()
            save_path = os.path.join("/tmp", hash_value + ".jpg")
            ipa_only_image.save(save_path)
            ipa_only_image_path = save_path
        else:
            ipa_only_image_path = None

        tabs = int(tabs)
        try:
            comment, outputs, face_id_outputs = easyphoto_infer_forward(
                sd_model_checkpoint,
                selected_template_images,
                init_image,
                uploaded_template_images,
                text_to_image_input_prompt,
                text_to_image_width,
                text_to_image_height,
                t2i_control_way,
                t2i_pose_template,
                scene_id,
                prompt_generate_sd_model_checkpoint,
                additional_prompt,
                lora_weights,
                before_face_fusion_ratio,
                after_face_fusion_ratio,
                first_diffusion_steps,
                first_denoising_strength,
                second_diffusion_steps,
                second_denoising_strength,
                seed,
                crop_face_preprocess,
                apply_face_fusion_before,
                apply_face_fusion_after,
                color_shift_middle,
                color_shift_last,
                super_resolution,
                super_resolution_method,
                skin_retouching_bool,
                display_score,
                background_restore,
                background_restore_denoising_strength,
                makeup_transfer,
                makeup_transfer_ratio,
                face_shape_match,
                tabs,
                ipa_control,
                ipa_weight,
                ipa_image_path,
                ref_mode_choose,
                ipa_only_weight,
                ipa_only_image_path,
                lcm_accelerate,
                *user_ids,
            )
            outputs = [api.encode_pil_to_base64(output) for output in outputs]
            face_id_outputs_base64 = []
            if len(face_id_outputs) != 0:
                for item in face_id_outputs:
                    pil_base64 = api.encode_pil_to_base64(item[0])
                    score_base64 = base64.b64encode(item[1].encode("utf-8")).decode("utf-8")
                    face_id_outputs_base64.append((pil_base64, score_base64))
        except Exception as e:
            torch.cuda.empty_cache()
            comment = f"Infer error, error info:{str(e)}"
            outputs = []
            face_id_outputs_base64 = []
            ep_logger.error(comment)

        return {"message": comment, "outputs": outputs, "face_id_outputs": face_id_outputs_base64}


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(easyphoto_train_forward_api)
    script_callbacks.on_app_started(easyphoto_infer_forward_api)
except Exception as e:
    print(e)
