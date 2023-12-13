function ask_for_style_name(sd_model_checkpoint, dummy_component, _, train_mode_choose, resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos, train_batch_size, gradient_accumulation_steps, dataloader_num_workers, learning_rate, rank, network_alpha, validation, instance_images, enable_rl, max_rl_time, timestep_fraction, skin_retouching, training_prefix_prompt, crop_ratio) {
    var name_ = prompt('User id:');
    return [sd_model_checkpoint, dummy_component, name_, train_mode_choose, resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos, train_batch_size, gradient_accumulation_steps, dataloader_num_workers, learning_rate, rank, network_alpha, validation, instance_images, enable_rl, max_rl_time, timestep_fraction, skin_retouching, training_prefix_prompt, crop_ratio];
}

function switch_to_ep_photoinfer_upload() {
    gradioApp().getElementById('mode_easyphoto').querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_easyphoto_photo_inference').querySelectorAll('button')[0].click();

    return Array.from(arguments);
}

function switch_to_ep_tryon() {
    gradioApp().getElementById('mode_easyphoto').querySelectorAll('button')[3].click();

    return Array.from(arguments);
}
