import base64
import json
import cv2
import numpy as np
import requests
import time

def decode_image_from_base64jpeg(base64_image):
    image_bytes = base64.b64decode(base64_image)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def post(encoded_image, url='http://0.0.0.0:7860'):
    datas = json.dumps({
        "user_ids"              : ["liushishi3"], 
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
        "background_restore"        : False,
        "tabs"                      : 1
    })
    r = requests.post(f'{url}/easyphoto/easyphoto_infer_forward', data=datas, timeout=1500)
    data = r.content.decode('utf-8')
    return data

if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    # function()   执行的程序
    # 测试训练
    encoded_image = 'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/template/template1.jpeg'
    outputs = post(encoded_image)
    outputs = json.loads(outputs)
    image = decode_image_from_base64jpeg(outputs["outputs"][0])
    cv2.imwrite(str('liushishi') + ".jpg", image)
    time_end = time.time()  # 记录结束时间
    time_sum = (time_end - time_start) / 60  # 计算的时间差为程序的执行时间，单位为秒/s
    print('########################################################')
    print(f"######      总共花销：{time_sum} 分钟     #######")
    print('########################################################')