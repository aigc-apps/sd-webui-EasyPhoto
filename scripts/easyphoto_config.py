import os, glob
from modules.paths import data_path

# save_dirs
data_dir                        = data_path
easyphoto_img2img_samples       = os.path.join(data_dir, 'outputs/img2img-images')
easyphoto_outpath_samples       = os.path.join(data_dir, 'outputs/easyphoto-outputs')
user_id_outpath_samples         = os.path.join(data_dir, 'outputs/easyphoto-user-id-infos')

# prompts 
validation_prompt   = "easyphoto_face, easyphoto, 1person, <lora:FilmVelvia3:0.65>"
DEFAULT_POSITIVE    = '(best quality), (realistic, photo-realistic:1.2), detailed skin, beautiful, cool, finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw phot, put on makeup'
DEFAULT_NEGATIVE    = '(glasses:1.5), (worst quality:2), (low quality:2), (normal quality:2), hair, teeth, lowres, watermark, badhand, ((naked:2, nude:2, nsfw)), (normal quality:2), lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale)), mural,'

id_path             = os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "ids.txt")