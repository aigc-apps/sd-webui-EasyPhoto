import os, glob
from modules.paths import data_path

# save_dirs
data_dir                = data_path
paiya_outpath_samples   = os.path.join(data_dir, 'outputs/paiya-outputs')
user_id_outpath_samples = os.path.join(data_dir, 'outputs/paiya-user-id-infos')
processed_image_outpath_samples = os.path.join(data_dir, 'outputs/paiya-processed-images')

# prompts 
validation_prompt       = "paiya_face, paiya, 1person"
DEFAULT_POSITIVE    = '(best quality), (realistic, photo-realistic:1.2), beautiful, cool, finely detail, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw phot, put on makeup'
DEFAULT_NEGATIVE    = '(worst quality:2), (low quality:2), (normal quality:2), hair, teeth, lowres, watermark, badhand, ((naked:2, nude:2, nsfw)), (normal quality:2), lowres, bad anatomy, bad hands, normal quality, ((monochrome)), ((grayscale)), mural,'