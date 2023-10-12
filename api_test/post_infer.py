import base64
import json
import sys
import time
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
import requests


def decode_image_from_base64jpeg(base64_image):
    image_bytes = base64.b64decode(base64_image)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def post(encoded_image, url='http://0.0.0.0:7860'):
    datas = json.dumps({
        "user_ids"              : ["test"], 
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
        "skin_retouching_bool"      : True,

        "background_restore"        : False,
        "tabs"                      : 1
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
    
    # initiate time
    now_date    = datetime.now()
    time_start  = time.time()  
    
    # -------------------test infer------------------- #
    # When there is no parameter input.
    if len(sys.argv) == 1:
        encoded_image = 'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/template/template1.jpeg'
        encoded_image = requests.get(encoded_image) 
        encoded_image = base64.b64encode(BytesIO(encoded_image.content).read()).decode('utf-8')

        outputs = post(encoded_image)
        outputs = json.loads(outputs)
        image = decode_image_from_base64jpeg(outputs["outputs"][0])
        cv2.imwrite(f"{now_date.hour}-{now_date.minute}-{now_date.second}" + ".jpg", image)

    # When selecting a local file as a parameter input.
    elif len(sys.argv) == 2:
        with open(sys.argv[1], 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf-8')
            outputs = post(encoded_image)
            outputs = json.loads(outputs)
            image = decode_image_from_base64jpeg(outputs["outputs"][0])
            cv2.imwrite(f"{now_date.hour}-{now_date.minute}-{now_date.second}" + ".jpg", image)

    # End of record time
    # The calculated time difference is the execution time of the program, expressed in seconds / s
    time_end = time.time()  
    time_sum = (time_end - time_start) % 60 
    print('# --------------------------------------------------------- #')
    print(f'#   Total expenditure: {time_sum}s')
    print('# --------------------------------------------------------- #')