function ask_for_style_name(sd_model_checkpoint, dummy_component, _, resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos, train_batch_size, gradient_accumulation_steps, dataloader_num_workers, learning_rate, rank, network_alpha, validation, instance_images,) {
    var name_ = prompt('User id:');
    return [sd_model_checkpoint, dummy_component, name_, resolution, val_and_checkpointing_steps, max_train_steps, steps_per_photos, train_batch_size, gradient_accumulation_steps, dataloader_num_workers, learning_rate, rank, network_alpha, validation, instance_images, ];
}