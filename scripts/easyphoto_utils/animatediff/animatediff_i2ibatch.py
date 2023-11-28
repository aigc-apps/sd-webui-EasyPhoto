from pathlib import Path
from types import MethodType

import os
import cv2
import numpy as np
import torch
import hashlib
from PIL import Image, ImageOps, UnidentifiedImageError
from modules import processing, shared, scripts, img2img, devices, masking, sd_samplers, images
from modules.processing import (StableDiffusionProcessingImg2Img,
                                process_images,
                                create_binary_mask,
                                create_random_tensors,
                                images_tensor_to_samples,
                                setup_color_correction,
                                opt_f)
from modules.shared import opts
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes

from .animatediff_logger import logger_animatediff as logger


class AnimateDiffI2IBatch:

    def hack(self):
        logger.info("Hacking i2i-batch.")
        original_img2img_process_batch = img2img.process_batch

        def hacked_i2i_init(self, all_prompts, all_seeds, all_subseeds): # only hack this when i2i-batch with batch mask
            self.image_cfg_scale: float = self.image_cfg_scale if shared.sd_model.cond_stage_key == "edit" else None

            self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
            crop_regions = []
            paste_to = []
            masks_for_overlay = []

            image_masks = self.image_mask

            for idx, image_mask in enumerate(image_masks):
                # image_mask is passed in as RGBA by Gradio to support alpha masks,
                # but we still want to support binary masks.
                image_mask = create_binary_mask(image_mask)

                if self.inpainting_mask_invert:
                    image_mask = ImageOps.invert(image_mask)

                if self.mask_blur_x > 0:
                    np_mask = np.array(image_mask)
                    kernel_size = 2 * int(2.5 * self.mask_blur_x + 0.5) + 1
                    np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur_x)
                    image_mask = Image.fromarray(np_mask)

                if self.mask_blur_y > 0:
                    np_mask = np.array(image_mask)
                    kernel_size = 2 * int(2.5 * self.mask_blur_y + 0.5) + 1
                    np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur_y)
                    image_mask = Image.fromarray(np_mask)

                if self.inpaint_full_res:
                    masks_for_overlay.append(image_mask)
                    mask = image_mask.convert('L')
                    crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                    crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                    crop_regions.append(crop_region)
                    x1, y1, x2, y2 = crop_region

                    mask = mask.crop(crop_region)
                    image_mask = images.resize_image(2, mask, self.width, self.height)
                    paste_to.append((x1, y1, x2-x1, y2-y1))
                else:
                    image_mask = images.resize_image(self.resize_mode, image_mask, self.width, self.height)
                    np_mask = np.array(image_mask)
                    np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                    masks_for_overlay.append(Image.fromarray(np_mask))

                image_masks[idx] = image_mask

            self.mask_for_overlay = masks_for_overlay[0] # only for saving purpose
            if paste_to:
                self.paste_to = paste_to[0]
                self._animatediff_paste_to_full = paste_to

            self.overlay_images = []
            add_color_corrections = opts.img2img_color_correction and self.color_corrections is None
            if add_color_corrections:
                self.color_corrections = []
            imgs = []
            for idx, img in enumerate(self.init_images):
                latent_mask = (self.latent_mask[idx] if isinstance(self.latent_mask, list) else self.latent_mask) if self.latent_mask is not None else image_masks[idx]
                # Save init image
                if opts.save_init_img:
                    self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
                    images.save_image(img, path=opts.outdir_init_images, basename=None, forced_filename=self.init_img_hash, save_to_dirs=False)

                image = images.flatten(img, opts.img2img_background_color)

                if not crop_regions and self.resize_mode != 3:
                    image = images.resize_image(self.resize_mode, image, self.width, self.height)

                if image_masks:
                    image_masked = Image.new('RGBa', (image.width, image.height))
                    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(masks_for_overlay[idx].convert('L')))

                    self.overlay_images.append(image_masked.convert('RGBA'))

                # crop_region is not None if we are doing inpaint full res
                if crop_regions:
                    image = image.crop(crop_regions[idx])
                    image = images.resize_image(2, image, self.width, self.height)

                if image_masks:
                    if self.inpainting_fill != 1:
                        image = masking.fill(image, latent_mask)

                if add_color_corrections:
                    self.color_corrections.append(setup_color_correction(image))

                image = np.array(image).astype(np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)

                imgs.append(image)

            if len(imgs) == 1:
                batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
                if self.overlay_images is not None:
                    self.overlay_images = self.overlay_images * self.batch_size

                if self.color_corrections is not None and len(self.color_corrections) == 1:
                    self.color_corrections = self.color_corrections * self.batch_size

            elif len(imgs) <= self.batch_size:
                self.batch_size = len(imgs)
                batch_images = np.array(imgs)
            else:
                raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

            image = torch.from_numpy(batch_images)
            image = image.to(shared.device, dtype=devices.dtype_vae)

            if opts.sd_vae_encode_method != 'Full':
                self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method

            self.init_latent = images_tensor_to_samples(image, approximation_indexes.get(opts.sd_vae_encode_method), self.sd_model)
            devices.torch_gc()

            if self.resize_mode == 3:
                self.init_latent = torch.nn.functional.interpolate(self.init_latent, size=(self.height // opt_f, self.width // opt_f), mode="bilinear")

            if image_masks is not None:
                def process_letmask(init_mask):
                    # init_mask = latent_mask
                    latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
                    latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
                    latmask = latmask[0]
                    latmask = np.around(latmask)
                    return np.tile(latmask[None], (4, 1, 1))

                if self.latent_mask is not None and not isinstance(self.latent_mask, list):
                    latmask = process_letmask(self.latent_mask)
                elif self.latent_mask is not None:
                    if isinstance(self.latent_mask, list):
                        latmask = [process_letmask(x) for x in self.latent_mask]
                    else:
                        latmask = [process_letmask(x) for x in image_masks]
                    latmask = np.stack(latmask, axis=0)

                self.mask = torch.asarray(1.0 - latmask).to(shared.device).type(self.sd_model.dtype)
                self.nmask = torch.asarray(latmask).to(shared.device).type(self.sd_model.dtype)

                # this needs to be fixed to be done in sample() using actual seeds for batches
                if self.inpainting_fill == 2:
                    self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:], all_seeds[0:self.init_latent.shape[0]]) * self.nmask
                elif self.inpainting_fill == 3:
                    self.init_latent = self.init_latent * self.mask

            self.image_conditioning = self.img2img_image_conditioning(image * 2 - 1, self.init_latent, image_masks) # let's ignore this image_masks which is related to inpaint model with different arch

        def hacked_img2img_process_batch_hijack(
                p: StableDiffusionProcessingImg2Img, input_dir: str, output_dir: str, inpaint_mask_dir: str,
                args, to_scale=False, scale_by=1.0, use_png_info=False, png_info_props=None, png_info_dir=None):
            if p.scripts:
                for script in p.scripts.alwayson_scripts:
                    if script.title().lower() == "animatediff":
                        ad_arg = p.script_args[script.args_from]
                        ad_enabled = ad_arg.get('enable', False) if isinstance(ad_arg, dict) else getattr(ad_arg, 'enable', False)
                        if ad_enabled:
                            p._animatediff_i2i_batch = 1 # i2i-batch mode, ordinary

            if not hasattr(p, '_animatediff_i2i_batch'):
                return original_img2img_process_batch(p, input_dir, output_dir, inpaint_mask_dir, args, to_scale, scale_by, use_png_info, png_info_props, png_info_dir)
            output_dir = output_dir.strip()
            processing.fix_seed(p)

            images = list(shared.walk_files(input_dir, allowed_extensions=(".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")))

            is_inpaint_batch = False
            if inpaint_mask_dir:
                inpaint_masks = shared.listfiles(inpaint_mask_dir)
                is_inpaint_batch = bool(inpaint_masks)

                if is_inpaint_batch:
                    assert len(inpaint_masks) == 1 or len(inpaint_masks) == len(images), 'The number of masks must be 1 or equal to the number of images.'
                    logger.info(f"\n[i2i batch] Inpaint batch is enabled. {len(inpaint_masks)} masks found.")
                    if len(inpaint_masks) > 1: # batch mask
                        p.init = MethodType(hacked_i2i_init, p)

            logger.info(f"[i2i batch] Will process {len(images)} images, creating {p.n_iter} new videos.")

            # extract "default" params to use in case getting png info fails
            prompt = p.prompt
            negative_prompt = p.negative_prompt
            seed = p.seed
            cfg_scale = p.cfg_scale
            sampler_name = p.sampler_name
            steps = p.steps
            frame_images = []
            frame_masks = []

            for i, image in enumerate(images):

                try:
                    img = Image.open(image)
                except UnidentifiedImageError as e:
                    print(e)
                    continue
                # Use the EXIF orientation of photos taken by smartphones.
                img = ImageOps.exif_transpose(img)

                if to_scale:
                    p.width = int(img.width * scale_by)
                    p.height = int(img.height * scale_by)

                frame_images.append(img)

                image_path = Path(image)
                if is_inpaint_batch:
                    if len(inpaint_masks) == 1:
                        mask_image_path = inpaint_masks[0]
                        p.image_mask = Image.open(mask_image_path)
                    else:
                        # try to find corresponding mask for an image using index matching
                        mask_image_path = inpaint_masks[i]
                        frame_masks.append(Image.open(mask_image_path))

                    mask_image = Image.open(mask_image_path)
                    p.image_mask = mask_image

            if use_png_info:
                try:
                    info_img = frame_images[0]
                    if png_info_dir:
                        info_img_path = os.path.join(png_info_dir, os.path.basename(image))
                        info_img = Image.open(info_img_path)
                    from modules import images as imgutil
                    from modules.generation_parameters_copypaste import parse_generation_parameters
                    geninfo, _ = imgutil.read_info_from_image(info_img)
                    parsed_parameters = parse_generation_parameters(geninfo)
                    parsed_parameters = {k: v for k, v in parsed_parameters.items() if k in (png_info_props or {})}
                except Exception:
                    parsed_parameters = {}

                p.prompt = prompt + (" " + parsed_parameters["Prompt"] if "Prompt" in parsed_parameters else "")
                p.negative_prompt = negative_prompt + (" " + parsed_parameters["Negative prompt"] if "Negative prompt" in parsed_parameters else "")
                p.seed = int(parsed_parameters.get("Seed", seed))
                p.cfg_scale = float(parsed_parameters.get("CFG scale", cfg_scale))
                p.sampler_name = parsed_parameters.get("Sampler", sampler_name)
                p.steps = int(parsed_parameters.get("Steps", steps))
            
            p.init_images = frame_images
            if len(frame_masks) > 0:
                p.image_mask = frame_masks

            proc = scripts.scripts_img2img.run(p, *args) # we should not support this, but just leave it here
            if proc is None:
                if output_dir:
                    p.outpath_samples = output_dir
                    p.override_settings['save_to_dirs'] = False
                    if p.n_iter > 1 or p.batch_size > 1:
                        p.override_settings['samples_filename_pattern'] = f'{image_path.stem}-[generation_number]'
                    else:
                        p.override_settings['samples_filename_pattern'] = f'{image_path.stem}'
                process_images(p)
            else:
                logger.warn("Warning: you are using an unsupported external script. AnimateDiff may not work properly.")

        img2img.process_batch = hacked_img2img_process_batch_hijack


    def cap_init_image(self, p: StableDiffusionProcessingImg2Img, params):
        if params.enable and isinstance(p, StableDiffusionProcessingImg2Img) and hasattr(p, '_animatediff_i2i_batch'):
            if len(p.init_images) > params.video_length:
                p.init_images = p.init_images[:params.video_length]
                if p.image_mask and isinstance(p.image_mask, list) and len(p.image_mask) > params.video_length:
                    p.image_mask = p.image_mask[:params.video_length]
            if len(p.init_images) < params.video_length:
                params.video_length = len(p.init_images)
            if len(p.init_images) < params.batch_size:
                params.batch_size = len(p.init_images)



animatediff_i2ibatch = AnimateDiffI2IBatch()
