import os, glob
from modules import shared

# save_dirs
data_dir                = shared.cmd_opts.data_dir
paiya_outpath_samples   = os.path.join(data_dir, 'outputs/paiya-outputs')
user_id_outpath_samples = os.path.join(data_dir, 'outputs/paiya-user-id-infos')
processed_image_outpath_samples = os.path.join(data_dir, 'outputs/paiya-processed-images')

# prompts 
validation_prompt       = "paiya_face, paiya, 1person"