import base64
import json
import os
import requests
import time
import sys
from glob import glob


def post_train(encoded_images, url='http://0.0.0.0:7860'):
    datas = json.dumps({
        "user_id"               : "test", # A custom ID that identifies the trained face model
        "sd_model_checkpoint"   : "Chilloutmix-Ni-pruned-fp16-fix.safetensors",
        "resolution"            : 512,
        "val_and_checkpointing_steps" : 100,
        "max_train_steps"       : 100, # Training batch
        "steps_per_photos"      : 200,
        "train_batch_size"      : 1,
        "gradient_accumulation_steps" : 4,
        "dataloader_num_workers" : 16,
        "learning_rate"         : 1e-4,
        "rank"                  : 64,
        "network_alpha"         : 64,
        "instance_images"       : encoded_images, 
    })
    r = requests.post(f'{url}/easyphoto/easyphoto_train_forward', data=datas, timeout=1500)
    data = r.content.decode('utf-8')
    return data

if __name__ == '__main__':

    '''
    There are two ways to test:
        The first: make sure the directory is full of readable images
        The second: public link of readable picture
    '''

    time_start = time.time()  # initiate time
    
    ##        Test training procedure      ##
    ## The first
    if len(sys.argv) == 2:
        # 如需有不同格式的图片，可自行改下下方代码读取图片
        img_list = glob(os.path.join(sys.argv[1], "*")) 
        encoded_images = []
        for idx, img_path in enumerate(img_list):
            with open(img_path, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
                encoded_images.append(encoded_image)
        outputs = post_train(encoded_images)
        outputs = json.loads(outputs)
        print(outputs['message'])
    
    ## The second:
    elif len(sys.argv) == 1:
        img_src = [
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi1.webp',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi2.jpeg',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi3.jpeg',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi4.webp',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi5.webp',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi6.jpeg',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi7.jpeg',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi8.webp',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi9.webp',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi10.webp',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi11.webp',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi12.jpeg',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi13.jpeg',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi14.jpeg',
            'https://pai-vision-data-inner.oss-accelerate.aliyuncs.com/data/easyphoto/train_data/shishi/shishi/shi15.webp'
        ]
        outputs = post_train(img_src)
        outputs = json.loads(outputs)
        print(outputs['message'])
        
    else:
        print("other modes except url and local read are not supported")
        
    time_end = time.time()  # End of record time
    time_sum = (time_end - time_start) // 60  # The calculated time difference is the execution time of the program, expressed in minute / m
    
    
    print('########################################################')
    print(f"##########      Total expenditure：{time_sum} minutes    ##########")
    print('########################################################')