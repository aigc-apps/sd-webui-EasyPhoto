import argparse
import base64
import json
import sys
import os
import time
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import requests
from glob import glob 
from tqdm import tqdm


def decode_image_from_base64jpeg(base64_image):
    image_bytes = base64.b64decode(base64_image)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def post(encoded_image, user_id=None, url='http://0.0.0.0:7860'):
    if user_id is None:
        user_id = 'test'
    datas = json.dumps({
        "user_ids"              : [user_id], 
        "sd_model_checkpoint"   : "Chilloutmix-Ni-pruned-fp16-fix.safetensors",
        "init_image"            : encoded_image, 

        "first_diffusion_steps"     : 50,
        "first_denoising_strength"  : 0.45,
        "second_diffusion_steps"    : 20,
        "second_denoising_strength" : 0.35,
        "seed"                      : 12345, 
        "crop_face_preprocess"      : True,

        "before_face_fusion_ratio"  : 0.5,
        "after_face_fusion_ratio"   : 0.5,
        "apply_face_fusion_before"  : True,
        "apply_face_fusion_after"   : True,

        "color_shift_middle"        : True,
        "color_shift_last"          : True,
        "super_resolution"          : True,
        "super_resolution_method"   : "gpen",

        "skin_retouching_bool"      : True,

        "background_restore"        : False,
        "background_restore_denoising_strength" : 0.35,

        "makeup_transfer"           : False,
        "makeup_transfer_ratio"     : 0.50,
        "face_shape_match"          : False,
        "tabs"                      : 1,

        "ipa_control"               : True,
        "ipa_weight"                : 0.50,
        "ipa_image"                 : None,

        "ref_mode_choose"           : "Infer with Pretrained Lora",
        "ipa_only_weight"           : 0.60,
        "ipa_only_image"            : None,
    })
    r = requests.post(f'{url}/easyphoto/easyphoto_infer_forward', data=datas, timeout=1500)
    data = r.content.decode('utf-8')
    return data

if __name__ == '__main__':
    '''
    There are two ways to test:
        The first: make sure the directory is full of readable images
        The second: public link of readable picture
    '''
    parser = argparse.ArgumentParser(description='Description of your script')

    parser.add_argument('--template_dir', type=str, default='', help='Path to the template directory')
    parser.add_argument('--output_path', type=str, default='./',help='Path to the output directory')
    parser.add_argument('--user_ids', type=str, default='test',help='Test user ids, split with space')

    args = parser.parse_args()

    template_dir = args.template_dir
    output_path = args.output_path
    user_ids = args.user_ids.split(' ')

    if output_path !='./':
        os.makedirs(output_path, exist_ok=True)

    # initiate time
    now_date    = datetime.now()
    time_start  = time.time()  
    
    # -------------------test infer------------------- #
    # When there is no parameter input.
    if template_dir == '':
        encoded_image = 'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/template/template1.jpeg'
        encoded_image = requests.get(encoded_image) 
        encoded_image = base64.b64encode(BytesIO(encoded_image.content).read()).decode('utf-8')

        outputs = post(encoded_image)
        outputs = json.loads(outputs)
        image = decode_image_from_base64jpeg(outputs["outputs"][0])
        toutput_path = os.path.join(os.path.join(output_path), "tmp.jpg")
        cv2.imwrite(toutput_path, image)

    # When selecting a local file as a parameter input.
    else:
        image_formats = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        img_list = []
        for image_format in image_formats:
            img_list.extend(glob(os.path.join(template_dir, image_format)))
      
        if len(img_list) == 0:
            print(f' Input template dir {template_dir} contains no images')
        else:
            print(f' Total {len(img_list)} templates to test for {len(user_ids)} ID')


        # please set your test user ids in args
        for user_id in tqdm(user_ids):
            for img_path in tqdm(img_list):
                print(f' Call generate for ID ({user_id}) and Template ({img_path})')
     
                with open(img_path, 'rb') as f:
                    encoded_image = base64.b64encode(f.read()).decode('utf-8')
                    outputs = post(encoded_image, user_id)
                    outputs = json.loads(outputs)

                    if len(outputs["outputs"]):
                        image = decode_image_from_base64jpeg(outputs["outputs"][0])
                        toutput_path = os.path.join(os.path.join(output_path), f'{user_id}_' + os.path.basename(img_path))
                        print(output_path)
                        cv2.imwrite(toutput_path, image)
                    else:
                        print('Error!', outputs['message'])

    # End of record time
    # The calculated time difference is the execution time of the program, expressed in seconds / s
    time_end = time.time()  
    time_sum = time_end - time_start 
    print('# --------------------------------------------------------- #')
    print(f'#   Total expenditure: {time_sum}s')
    print('# --------------------------------------------------------- #')
