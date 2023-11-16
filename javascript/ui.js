function ask_for_style_name(sd_model_checkpoint, dummy_component, _, resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos, train_batch_size, gradient_accumulation_steps, dataloader_num_workers, learning_rate, rank, network_alpha, validation, instance_images, enable_rl, max_rl_time, timestep_fraction, skin_retouching) {
    var name_ = prompt('User id:');
    return [sd_model_checkpoint, dummy_component, name_, resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos, train_batch_size, gradient_accumulation_steps, dataloader_num_workers, learning_rate, rank, network_alpha, validation, instance_images, enable_rl, max_rl_time, timestep_fraction, skin_retouching, ];
}

function ask_for_tryon_style_name(sd_model_checkpoint, template_image, selected_cloth_template_images, main_image, additional_prompt, seed, first_diffusion_steps, first_denoising_strength, lora_weight, iou_threshold, angle, azimuth, ratio, batch_size, refine_input_mask, optimize_angle_and_ratio, refine_bound, pure_image, model_selected_tab, ref_image_selected_tab, _) {
    var name_ = prompt('User id:');
    return [sd_model_checkpoint, template_image, selected_cloth_template_images, main_image, additional_prompt, seed, first_diffusion_steps, first_denoising_strength, lora_weight, iou_threshold, angle, azimuth, ratio, batch_size, refine_input_mask, optimize_angle_and_ratio, refine_bound, pure_image, model_selected_tab, ref_image_selected_tab, name_, ];
}