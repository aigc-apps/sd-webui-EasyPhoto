import os, glob
from modules.paths import data_path, models_path

# save_dirs
data_dir                        = data_path
models_path                     = models_path
easyphoto_img2img_samples       = os.path.join(data_dir, 'outputs/img2img-images')
easyphoto_txt2img_samples       = os.path.join(data_dir, 'outputs/txt2img-images')
easyphoto_outpath_samples       = os.path.join(data_dir, 'outputs/easyphoto-outputs')
user_id_outpath_samples         = os.path.join(data_dir, 'outputs/easyphoto-user-id-infos')
scene_id_outpath_samples        = os.path.join(data_dir, 'outputs/easyphoto-scene-id-infos')
cache_log_file_path             = os.path.join(data_dir, "outputs/easyphoto-tmp/train_kohya_log.txt")

# prompts 
validation_prompt       = "easyphoto_face, easyphoto, 1person"
validation_prompt_scene = "special_scene, scene, 1person"
DEFAULT_POSITIVE    = '(cloth:1.5), (best quality), (realistic, photo-realistic:1.3), (detailed skin:1.3), (rough skin:1.3), (beautiful eyes:1.3), (sparkling eyes:1.3), (beautiful mouth:1.3), finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw photo, put on makeup'
DEFAULT_NEGATIVE    = '(bags under the eyes:1.5), (bags under eyes:1.5), (earrings:1.3), (glasses:1.2), (naked:1.5), (nsfw:1.5), nude, breasts, penis, cum, (over red lips: 1.3), (bad lips: 1.3), (bad ears:1.3), (bad hair: 1.3), (bad teeth: 1.3), (worst quality:2), (low quality:2), (normal quality:2), lowres, watermark, badhand, lowres, bad anatomy, bad hands, normal quality, mural,'
DEFAULT_POSITIVE_XL = 'film photography, a clear face, minor acne, (high resolution detail of human skin texture:1.4, rough skin:1.2), (portrait, :1.8), (indirect lighting)'
DEFAULT_NEGATIVE_XL = '(bokeh:2), cgi, illustration, cartoon, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy,ugly, deformed, blurry,Noisy, log, text'

# ModelName
SDXL_MODEL_NAME     = 'SDXL_1.0_ArienMixXL_v2.0.safetensors'