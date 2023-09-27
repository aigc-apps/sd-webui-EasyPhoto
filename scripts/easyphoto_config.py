import os, glob
from modules.paths import data_path, models_path

# save_dirs
data_dir                        = data_path
models_path                     = models_path
easyphoto_img2img_samples       = os.path.join(data_dir, 'outputs/img2img-images')
easyphoto_txt2img_samples       = os.path.join(data_dir, 'outputs/txt2img-images')
easyphoto_outpath_samples       = os.path.join(data_dir, 'outputs/easyphoto-outputs')
user_id_outpath_samples         = os.path.join(data_dir, 'outputs/easyphoto-user-id-infos')
cache_log_file_path             = os.path.join(data_dir, "outputs/easyphoto-tmp/train_kohya_log.txt")

# prompts 
validation_prompt   = "easyphoto_face, easyphoto, 1person"
DEFAULT_POSITIVE    = '(cloth:1.7), (best quality), (realistic, photo-realistic:1.2), detailed skin, beautiful, cool, finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw photo, put on makeup'
DEFAULT_NEGATIVE    = '(bags under the eyes:1.5), (Bags under eyes:1.5), (glasses:1.5), (naked:2.0), nude, (nsfw:2.0), breasts, penis, cum, (worst quality:2), (low quality:2), (normal quality:2), over red lips, hair, teeth, lowres, watermark, badhand, (normal quality:2), lowres, bad anatomy, bad hands, normal quality, mural,'
DEFAULT_POSITIVE_XL = 'film photography, a clear face, minor acne, (high resolution detail of human skin texture:1.4, rough skin:1.2), (portrait, :1.8), (indirect lighting)'
DEFAULT_NEGATIVE_XL = '(bokeh:2), cgi, illustration, cartoon, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy,ugly, deformed, blurry,Noisy, log, text'

# ModelName
SDXL_MODEL_NAME     = 'SDXL_1.0_ArienMixXL_v2.0.safetensors'