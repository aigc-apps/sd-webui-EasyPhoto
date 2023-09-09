import argparse
import json
import math
import os
import sys
import cv2
import torch
import insightface
import numpy as np
import platform
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
from scripts.face_process_utils import call_face_crop
from scripts.easyphoto_utils import logging
from tqdm import tqdm
from shutil import copyfile


def compare_jpg_with_face_id(embedding_list):
    embedding_array = np.vstack(embedding_list)
    # Take mean from the user image to obtain the average features of the real person image
    pivot_feature   = np.mean(embedding_array, axis=0)
    pivot_feature   = np.reshape(pivot_feature, [512, 1])

    # Sort the images in a folder that are closest to the median value
    scores = [np.dot(emb, pivot_feature)[0][0] for emb in embedding_list]
    return scores


def preprocess_images(images_save_path, json_save_path, validation_prompt, inputs_dir, ref_image_path):
    # face embedding
    providers           = ["CPUExecutionProvider"]
    face_recognition    = insightface.model_zoo.get_model(os.path.join(os.path.abspath(os.path.dirname(__file__)).replace("scripts", "models"), "buffalo_l", "w600k_r50.onnx"), providers=providers)
    face_recognition.prepare(ctx_id=0)
    
    face_analyser       = insightface.app.FaceAnalysis(name="buffalo_l", root=os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""), providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    # face detection
    retinaface_detection    = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
    # semantic segmentation
    salient_detect          = pipeline(Tasks.semantic_segmentation, 'damo/cv_u2net_salient-detection')
    # skin retouching
    try:
        skin_retouching = pipeline('skin-retouching-torch', model='damo/cv_unet_skin_retouching_torch', model_revision='v1.0.2')
    except:
        logging.info("Skin Retouching model load error, but pass.")
    # portrait enhancement
    try:
        portrait_enhancement = pipeline(Tasks.image_portrait_enhancement, model='damo/cv_gpen_image-portrait-enhancement')
    except:
        logging.info("Portrait Enhancement model load error, but pass.")
    
    # jpg list
    jpgs            = os.listdir(inputs_dir)
    # ---------------------------FaceID score calculate-------------------------- #
    face_id_scores  = []
    face_angles     = []
    copy_jpgs       = []
    selected_paths  = []
    for index, jpg in enumerate(tqdm(jpgs)):
        if not jpg.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            continue
        _image_path = os.path.join(inputs_dir, jpg)
        image       = Image.open(_image_path)

        h, w, c     = np.shape(image)

        retinaface_boxes, retinaface_keypoints, _ = call_face_crop(retinaface_detection, image, 3, prefix="tmp")
        retinaface_box      = retinaface_boxes[0]
        retinaface_keypoint = retinaface_keypoints[0]

        # get key point
        retinaface_keypoint = np.reshape(retinaface_keypoint, [5, 2])
        # get angle
        x = retinaface_keypoint[0,0] - retinaface_keypoint[1,0]
        y = retinaface_keypoint[0,1] - retinaface_keypoint[1,1]
        angle = 0 if x==0 else abs(math.atan(y/x)*180/math.pi)
        angle = (90 - angle)/ 90 

        # face size judge
        face_width  = (retinaface_box[2] - retinaface_box[0]) / (3 - 1)
        face_height = (retinaface_box[3] - retinaface_box[1]) / (3 - 1)
        if min(face_width, face_height) < 128:
            print("Face size in {} is small than 128. Ignore it.".format(jpg))
            continue

        # face crop
        sub_image = image.crop(retinaface_box)

        # get embedding
        embedding = face_recognition.get(np.array(image), face_analyser.get(np.array(image))[0])
        embedding = np.array([embedding / np.linalg.norm(embedding, 2)])

        face_id_scores.append(embedding)
        face_angles.append(angle)

        copy_jpgs.append(jpg)
        selected_paths.append(_image_path)
    
    if len(face_id_scores) == 0:
        return "No faces detected. Please increase the size of the face in the upload photos."

    # Filter reference faces based on scores, considering quality scores, similarity scores, and angle scores
    face_id_scores      = compare_jpg_with_face_id(face_id_scores)
    ref_total_scores    = np.array(face_angles) * np.array(face_id_scores)
    ref_indexes         = np.argsort(ref_total_scores)[::-1]
    for index in ref_indexes:
        print("selected paths:", selected_paths[index], "total scores: ", ref_total_scores[index], "face angles", face_angles[index])
    copyfile(selected_paths[ref_indexes[0]], ref_image_path)
             
    # Select faces based on scores, considering similarity scores
    total_scores    = np.array(face_id_scores)
    indexes         = np.argsort(total_scores)[::-1][:15]
    
    selected_jpgs   = []
    selected_scores = []
    for index in indexes:
        selected_jpgs.append(copy_jpgs[index])
        selected_scores.append(ref_total_scores[index])
        print("jpg:", copy_jpgs[index], "face_id_scores", ref_total_scores[index])
                             
    images              = []
    enhancement_num      = 0
    max_enhancement_num  = len(selected_jpgs) // 2
    for index, jpg in tqdm(enumerate(selected_jpgs[::-1])):
        if not jpg.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            continue
        _image_path             = os.path.join(inputs_dir, jpg)
        image                   = Image.open(_image_path)
        retinaface_boxes, _, _  = call_face_crop(retinaface_detection, image, 3, prefix="tmp")
        retinaface_box          = retinaface_boxes[0]
        # crop image
        sub_image               = image.crop(retinaface_box)
        try:
            sub_image           = Image.fromarray(cv2.cvtColor(skin_retouching(sub_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
        except:
            logging.info("Skin Retouching model detect error, but pass.")

        try:
            # Determine which images to enhance based on quality score and image size
            if (np.shape(sub_image)[0] < 512 or np.shape(sub_image)[1] < 512) and enhancement_num < max_enhancement_num:
                sub_image = Image.fromarray(cv2.cvtColor(portrait_enhancement(sub_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
                enhancement_num += 1
        except:
            logging.info("Portrait Enhancement model detect error, but pass.")

        # Correct the mask area of the face
        sub_boxes, _, sub_masks = call_face_crop(retinaface_detection, sub_image, 1, prefix="tmp")
        sub_box     = sub_boxes[0]
        sub_mask    = sub_masks[0]

        h, w, c     = np.shape(sub_mask)
        face_width  = sub_box[2] - sub_box[0]
        face_height = sub_box[3] - sub_box[1]
        sub_box[0]  = np.clip(np.array(sub_box[0], np.int32) - face_width * 0.3, 1, w - 1)
        sub_box[2]  = np.clip(np.array(sub_box[2], np.int32) + face_width * 0.3, 1, w - 1)
        sub_box[1]  = np.clip(np.array(sub_box[1], np.int32) + face_height * 0.15, 1, h - 1)
        sub_box[3]  = np.clip(np.array(sub_box[3], np.int32) + face_height * 0.15, 1, h - 1)
        sub_mask    = np.zeros_like(np.array(sub_mask, np.uint8))
        sub_mask[sub_box[1]:sub_box[3], sub_box[0]:sub_box[2]] = 1

        # Significance detection, merging facial masks
        result      = salient_detect(sub_image)[OutputKeys.MASKS]
        mask        = np.float32(np.expand_dims(result > 128, -1)) * sub_mask

        # Obtain the image after the mask
        mask_sub_image = np.array(sub_image) * np.array(mask) + np.ones_like(sub_image) * 255 * (1 - np.array(mask))
        mask_sub_image = Image.fromarray(np.uint8(mask_sub_image))
        if np.sum(np.array(mask)) != 0:
            images.append(mask_sub_image)

    # write resuilts
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
                        if platform.system() == 'Windows':
                            path = path.replace('\\', '/')
                        jpg_path_split  = path.split("/")
                        file_name = os.path.join(*jpg_path_split[-2:])
                        a = {
                            "file_name": file_name, 
                            "text": prompt
                        }
                        f.write(json.dumps(eval(str(a))))
                        f.write("\n")

    del retinaface_detection
    del salient_detect
    torch.cuda.empty_cache()