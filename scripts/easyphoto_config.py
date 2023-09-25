import os, glob
from modules.paths import data_path, models_path

# save_dirs
data_dir                        = data_path
models_path                     = models_path
easyphoto_img2img_samples       = os.path.join(data_dir, 'outputs/img2img-images')
easyphoto_outpath_samples       = os.path.join(data_dir, 'outputs/easyphoto-outputs')
user_id_outpath_samples         = os.path.join(data_dir, 'outputs/easyphoto-user-id-infos')
cache_log_file_path             = os.path.join(data_dir, "outputs/easyphoto-tmp/train_kohya_log.txt")

# prompts 
validation_prompt   = "easyphoto_face, easyphoto, 1person"
DEFAULT_POSITIVE    = '(cloth:1.7), (best quality), (realistic, photo-realistic:1.2), detailed skin, beautiful, cool, finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw photo, put on makeup'
DEFAULT_NEGATIVE    = '(bags under the eyes:1.5), (Bags under eyes:1.5), (glasses:1.5), (naked:2.0), nude, (nsfw:2.0), breasts, penis, cum, (worst quality:2), (low quality:2), (normal quality:2), over red lips, hair, teeth, lowres, watermark, badhand, (normal quality:2), lowres, bad anatomy, bad hands, normal quality, mural,'