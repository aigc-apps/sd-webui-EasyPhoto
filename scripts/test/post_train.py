import json
import numpy as np
import requests
import time


def post_train(encoded_images, url='http://0.0.0.0:7860'):
    datas = json.dumps({
        "user_id"               : "liushishi3", 
        "sd_model_checkpoint"   : "Chilloutmix-Ni-pruned-fp16-fix.safetensors",
        "resolution"            : 512,
        "val_and_checkpointing_steps" : 100,
        "max_train_steps"       : 100,
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
    time_start = time.time()  # 记录开始时间
    # function()   执行的程序
    # 测试训练
    outputs = post_train(img_src)
    outputs = json.loads(outputs)
    print(outputs['message'])
    time_end = time.time()  # 记录结束时间
    time_sum = (time_end - time_start) // 60  # 计算的时间差为程序的执行时间，单位为秒/s
    print('########################################################')
    print(f"##########      总共花销：{time_sum} 分钟     ##########")
    print('########################################################')
