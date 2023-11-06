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
zero123_model_path              = os.path.join(os.path.abspath(os.path.dirname(__file__)),'thirdparty/zero123/models')

# prompts 
validation_prompt   = "easyphoto, 1thing"
DEFAULT_POSITIVE    = '(cloth:1.5), (best quality),'
DEFAULT_NEGATIVE    = 'bad detail'
DEFAULT_POSITIVE_XL = 'film photography, a clear face, minor acne, (high resolution detail of human skin texture:1.4, rough skin:1.2), (portrait, :1.8), (indirect lighting)'
DEFAULT_NEGATIVE_XL = '(bokeh:2), cgi, illustration, cartoon, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy,ugly, deformed, blurry,Noisy, log, text'

# ModelName
SDXL_MODEL_NAME     = 'SDXL_1.0_ArienMixXL_v2.0.safetensors'