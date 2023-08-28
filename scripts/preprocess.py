import argparse
import json
import os
import sys
import math

import numpy as np
from tqdm import tqdm
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
from paiya_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=(
            "The validation_prompt of the user."
        ),
    )
    parser.add_argument(
        "--ref_image_path",
        type=str,
        default=None,
        help=(
            "The ref_image_path."
        ),
    )
    parser.add_argument(
        "--images_save_path",
        type=str,
        default=None,
        help=(
            "The images_save_path."
        ),
    )
    parser.add_argument(
        "--json_save_path",
        type=str,
        default=None,
        help=(
            "The json_save_path."
        ),
    )
    parser.add_argument(
        "--inputs_dir",
        type=str,
        default=None,
        help=(
            "The inputs dir of the data for preprocessing."
        ),
    )
    parser.add_argument(
        "--crop_ratio",
        type=int,
        default=3,
        help=(
            "The expand ratio for input data to crop."
        ),
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default='./',
        help=(
            "The expand ratio for input data to crop."
        ),
    )
    args = parser.parse_args()
    return args

def compare_jpg_with_face_id(embedding_list):
    embedding_array = np.vstack(embedding_list)
    # 然后对真人图片取mean，获取真人图片的平均特征
    pivot_feature   = np.mean(embedding_array, axis=0)
    pivot_feature   = np.reshape(pivot_feature, [512, 1])

    # 计算一个文件夹中，和中位值最接近的图片排序
    scores = [np.dot(emb, pivot_feature)[0][0] for emb in embedding_list]
    return scores

if __name__ == "__main__":
    args        = parse_args()

    # 创建输出文件夹
    images_save_path    = args.images_save_path
    json_save_path      = args.json_save_path
    validation_prompt   = args.validation_prompt

    # 人脸评分
    face_quality_func       = pipeline(Tasks.face_quality_assessment, 'damo/cv_manual_face-quality-assessment_fqa')
    # embedding
    face_recognition        = pipeline(Tasks.face_recognition, model='damo/cv_ir101_facerecognition_cfglint')
    # 人脸检测
    retinaface_detection    = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
    # 显著性检测
    salient_detect          = pipeline(Tasks.semantic_segmentation, model='damo/cv_u2net_salient-detection')
    
    # 获得jpg列表
    jpgs            = os.listdir(args.inputs_dir)
    # ---------------------------人脸得分计算-------------------------- #
    face_id_scores  = []
    quality_scores  = []
    face_angles     = []
    copy_jpgs       = []
    selected_paths  = []
    for index, jpg in enumerate(tqdm(jpgs)):
        try:
            if not jpg.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                continue
            _image_path = os.path.join(args.inputs_dir, jpg)
            image       = Image.open(_image_path)
            h, w, c     = np.shape(image)

            retinaface_box, retinaface_keypoint, _ = call_face_crop(retinaface_detection, image, 3, prefix="tmp")
            retinaface_keypoint = np.reshape(retinaface_keypoint, [5, 2])
            # 计算人脸偏移角度
            x = retinaface_keypoint[0,0] - retinaface_keypoint[1,0]
            y = retinaface_keypoint[0,1] - retinaface_keypoint[1,1]
            angle = 0 if x==0 else abs(math.atan(y/x)*180/math.pi)
            angle = (90 - angle)/ 90 

            # 人脸宽度判断
            face_width  = (retinaface_box[2] - retinaface_box[0]) / (3 - 1)
            face_height = (retinaface_box[3] - retinaface_box[1]) / (3 - 1)
            if face_width / w < 1/8 or face_height / h < 1/8:
                continue

            sub_image = image.crop(retinaface_box)

            embedding   = np.array(face_recognition(sub_image)[OutputKeys.IMG_EMBEDDING])
            score       = face_quality_func(sub_image)[OutputKeys.SCORES]
            score       = 0 if score is None else score[0]

            face_id_scores.append(embedding)
            quality_scores.append(score)
            face_angles.append(angle)

            copy_jpgs.append(jpg)
            selected_paths.append(_image_path)
        except:
            pass

    # 根据得分进行参考人脸的筛选，考虑质量分，相似分与角度分
    face_id_scores      = compare_jpg_with_face_id(face_id_scores)
    ref_total_scores    = np.array(face_id_scores) * np.array(quality_scores) * np.array(face_angles)
    ref_indexes         = np.argsort(ref_total_scores)[::-1]
    for index in ref_indexes:
        print("selected paths:", selected_paths[index], "total scores: ", ref_total_scores[index], "face id score", face_id_scores[index], "face angles", face_angles[index])
    os.system(f"cp -rf {selected_paths[ref_indexes[0]]} {args.ref_image_path}")
    
    # 根据得分进行训练人脸的筛选，考虑相似分
    total_scores    = np.array(face_id_scores) # np.array(face_id_scores) * np.array(scores)
    indexes         = np.argsort(total_scores)[::-1][:15]
    
    selected_jpgs   = []
    selected_scores = []
    for index in indexes:
        selected_jpgs.append(copy_jpgs[index])
        selected_scores.append(quality_scores[index])
        print("jpg:", copy_jpgs[index], "face_id_scores", face_id_scores[index])
                             
    images              = []
    for index, jpg in tqdm(enumerate(selected_jpgs[::-1])):
        if not jpg.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            continue
        try:
            _image_path             = os.path.join(args.inputs_dir, jpg)
            image                   = Image.open(_image_path)
            retinaface_box, _, _    = call_face_crop(retinaface_detection, image, args.crop_ratio, prefix="tmp")
            sub_image               = image.crop(retinaface_box)

            # 显著性检测，合并人脸mask
            result      = salient_detect(sub_image)[OutputKeys.MASKS]
            mask        = np.float32(np.expand_dims(result > 128, -1))
            # 获得mask后的图像
            mask_sub_image = np.array(sub_image) * np.array(mask) + np.ones_like(sub_image) * 255 * (1 - np.array(mask))
            mask_sub_image = Image.fromarray(np.uint8(mask_sub_image))
            if np.sum(np.array(mask)) != 0:
                images.append(mask_sub_image)
        except:
            pass

    # 写入结果
    for index, base64_pilimage in enumerate(images):
        image = base64_pilimage.convert("RGB")
        image.save(os.path.join(images_save_path, str(index) + ".jpg"))
        print("save processed image to "+ os.path.join(images_save_path, str(index) + ".jpg"))
        with open(os.path.join(images_save_path, str(index) + ".txt"), "w") as f:
            f.write(validation_prompt)

    with open(json_save_path, 'w', encoding="utf-8") as f:
        for root, dirs, files in os.walk(images_save_path, topdown=False):
            for file in files:
                path = os.path.join(root, file)
                if not file.endswith('txt'):
                    txt_path = ".".join(path.split(".")[:-1]) + ".txt"
                    if os.path.exists(txt_path):
                        prompt          = open(txt_path, 'r').readline().strip()
                        jpg_path_split  = path.split("/")
                        file_name = os.path.join(*jpg_path_split[-2:])
                        a = {
                            "file_name": file_name, 
                            "text": prompt
                        }
                        f.write(json.dumps(eval(str(a))))
                        f.write("\n")
