import argparse
import json
import logging
import math
import os
import platform
import sys
import re
from shutil import copyfile

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "easyphoto_utils"))

import cv2
import numpy as np
import torch
from face_process_utils import call_face_crop
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=("The validation_prompt of the user."),
    )
    parser.add_argument(
        "--ref_image_path",
        type=str,
        default=None,
        help=("The ref_image_path."),
    )
    parser.add_argument(
        "--images_save_path",
        type=str,
        default=None,
        help=("The images_save_path."),
    )
    parser.add_argument(
        "--json_save_path",
        type=str,
        default=None,
        help=("The json_save_path."),
    )
    parser.add_argument(
        "--inputs_dir",
        type=str,
        default=None,
        help=("The inputs dir of the data for preprocessing."),
    )
    parser.add_argument(
        "--crop_ratio",
        type=float,
        default=3,
        help=("The crop ratio of the data for scene lora preprocessing."),
    )
    parser.add_argument(
        "--skin_retouching_bool",
        action="store_true",
        help=("Whether to use beauty"),
    )
    parser.add_argument(
        "--train_scene_lora_bool",
        action="store_true",
        help=("Whether to train scene lora"),
    )
    args = parser.parse_args()
    return args


def compare_jpg_with_face_id(embedding_list):
    embedding_array = np.vstack(embedding_list)
    # Take mean from the user image to obtain the average features of the real person image
    pivot_feature = np.mean(embedding_array, axis=0)
    pivot_feature = np.reshape(pivot_feature, [512, 1])

    # Sort the images in a folder that are closest to the median value
    scores = [np.dot(emb, pivot_feature)[0][0] for emb in embedding_list]
    return scores


if __name__ == "__main__":
    args = parse_args()
    images_save_path = args.images_save_path
    json_save_path = args.json_save_path
    validation_prompt = args.validation_prompt
    # Convert a list of strings (type: str) into an list of strings (type: List).
    if validation_prompt[0] == "[" and validation_prompt[-1] == "]":
        validation_prompt = re.findall(r"'(.*?)'", validation_prompt)
    inputs_dir = args.inputs_dir
    ref_image_path = args.ref_image_path
    skin_retouching_bool = args.skin_retouching_bool
    train_scene_lora_bool = args.train_scene_lora_bool

    logging.info(
        f"""
        preprocess params:
        images_save_path     = {images_save_path}
        json_save_path       = {json_save_path}
        validation_prompt    = {validation_prompt}
        inputs_dir           = {inputs_dir}
        ref_image_path       = {ref_image_path}
        skin_retouching_bool = {skin_retouching_bool}
        train_scene_lora_bool = {train_scene_lora_bool}
        """
    )
    # embedding
    face_recognition = pipeline("face_recognition", model="bubbliiiing/cv_retinafce_recognition", model_revision="v1.0.3")
    # face detection
    retinaface_detection = pipeline(Tasks.face_detection, "damo/cv_resnet50_face-detection_retinaface", model_revision="v2.0.2")
    # semantic segmentation
    salient_detect = pipeline(Tasks.semantic_segmentation, "damo/cv_u2net_salient-detection", model_revision="v1.0.0")
    # skin retouching
    try:
        skin_retouching = pipeline("skin-retouching-torch", model="damo/cv_unet_skin_retouching_torch", model_revision="v1.0.2")
    except Exception as e:
        skin_retouching = None
        logging.info(f"Skin Retouching model load error, but pass. Error info {e}")
    # portrait enhancement
    try:
        portrait_enhancement = pipeline(
            Tasks.image_portrait_enhancement, model="damo/cv_gpen_image-portrait-enhancement", model_revision="v1.0.0"
        )
    except Exception as e:
        portrait_enhancement = None
        logging.info(f"Portrait Enhancement model load error, but pass. Error info {e}")

    if not train_scene_lora_bool:
        # jpg list
        jpgs = os.listdir(inputs_dir)
        # ---------------------------FaceID score calculate-------------------------- #
        face_id_scores = []
        face_angles = []
        copy_jpgs = []
        selected_paths = []
        sub_images = []
        for index, jpg in enumerate(tqdm(jpgs)):
            try:
                if not jpg.lower().endswith((".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")):
                    continue
                _image_path = os.path.join(inputs_dir, jpg)
                image = Image.open(_image_path)

                h, w, c = np.shape(image)

                retinaface_boxes, retinaface_keypoints, _ = call_face_crop(retinaface_detection, image, 3, prefix="tmp")
                retinaface_box = retinaface_boxes[0]
                retinaface_keypoint = retinaface_keypoints[0]

                # get key point
                retinaface_keypoint = np.reshape(retinaface_keypoint, [5, 2])
                # get angle
                x = retinaface_keypoint[0, 0] - retinaface_keypoint[1, 0]
                y = retinaface_keypoint[0, 1] - retinaface_keypoint[1, 1]
                angle = 0 if x == 0 else abs(math.atan(y / x) * 180 / math.pi)
                angle = (90 - angle) / 90

                # face crop
                sub_image = image.crop(retinaface_box)
                if skin_retouching_bool:
                    try:
                        sub_image = Image.fromarray(cv2.cvtColor(skin_retouching(sub_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
                    except Exception as e:
                        torch.cuda.empty_cache()
                        logging.error(f"Photo skin_retouching error, error info: {e}")

                # get embedding
                embedding = face_recognition(dict(user=image))[OutputKeys.IMG_EMBEDDING]

                face_id_scores.append(embedding)
                face_angles.append(angle)

                copy_jpgs.append(jpg)
                selected_paths.append(_image_path)
                sub_images.append(sub_image)
            except Exception as e:
                torch.cuda.empty_cache()
                logging.error(f"Photo detect and count score error, error info: {e}")

        # Filter reference faces based on scores, considering quality scores, similarity scores, and angle scores
        face_id_scores = compare_jpg_with_face_id(face_id_scores)
        ref_total_scores = np.array(face_angles) * np.array(face_id_scores)
        ref_indexes = np.argsort(ref_total_scores)[::-1]
        for index in ref_indexes:
            print("selected paths:", selected_paths[index], "total scores: ", ref_total_scores[index], "face angles", face_angles[index])
        copyfile(selected_paths[ref_indexes[0]], ref_image_path)

        # Select faces based on scores, considering similarity scores
        total_scores = np.array(face_id_scores)
        indexes = np.argsort(total_scores)[::-1][:15]

        selected_jpgs = []
        selected_scores = []
        selected_sub_images = []
        for index in indexes:
            selected_jpgs.append(copy_jpgs[index])
            selected_scores.append(ref_total_scores[index])
            selected_sub_images.append(sub_images[index])
            print("jpg:", copy_jpgs[index], "face_id_scores", ref_total_scores[index])

        images = []
        enhancement_num = 0
        max_enhancement_num = len(selected_jpgs) // 2
        for index, jpg in tqdm(enumerate(selected_jpgs[::-1])):
            try:
                sub_image = selected_sub_images[index]
                try:
                    if (np.shape(sub_image)[0] < 512 or np.shape(sub_image)[1] < 512) and enhancement_num < max_enhancement_num:
                        sub_image = Image.fromarray(cv2.cvtColor(portrait_enhancement(sub_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
                        enhancement_num += 1
                except Exception as e:
                    torch.cuda.empty_cache()
                    logging.error(f"Photo enhance error, error info: {e}")

                # Correct the mask area of the face
                sub_boxes, _, sub_masks = call_face_crop(retinaface_detection, sub_image, 1, prefix="tmp")
                sub_box = sub_boxes[0]
                sub_mask = sub_masks[0]

                h, w, c = np.shape(sub_mask)
                face_width = sub_box[2] - sub_box[0]
                face_height = sub_box[3] - sub_box[1]
                sub_box[0] = np.clip(np.array(sub_box[0], np.int32) - face_width * 0.3, 1, w - 1)
                sub_box[2] = np.clip(np.array(sub_box[2], np.int32) + face_width * 0.3, 1, w - 1)
                sub_box[1] = np.clip(np.array(sub_box[1], np.int32) + face_height * 0.15, 1, h - 1)
                sub_box[3] = np.clip(np.array(sub_box[3], np.int32) + face_height * 0.15, 1, h - 1)
                sub_mask = np.zeros_like(np.array(sub_mask, np.uint8))
                sub_mask[sub_box[1] : sub_box[3], sub_box[0] : sub_box[2]] = 1

                # Significance detection, merging facial masks
                result = salient_detect(sub_image)[OutputKeys.MASKS]
                mask = np.float32(np.expand_dims(result > 128, -1)) * sub_mask

                # Obtain the image after the mask
                mask_sub_image = np.array(sub_image) * np.array(mask) + np.ones_like(sub_image) * 255 * (1 - np.array(mask))
                mask_sub_image = Image.fromarray(np.uint8(mask_sub_image))
                if np.sum(np.array(mask)) != 0:
                    images.append(mask_sub_image)
            except Exception as e:
                torch.cuda.empty_cache()
                logging.error(f"Photo face crop and salient_detect error, error info: {e}")
    else:
        # jpg list
        jpgs = os.listdir(inputs_dir)
        # sort photo files to correspond with captions.
        jpgs = sorted(jpgs, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        images = []
        for index, jpg in enumerate(tqdm(jpgs)):
            try:
                if not jpg.lower().endswith((".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")):
                    continue
                _image_path = os.path.join(inputs_dir, jpg)
                image = Image.open(_image_path)

                # Use original training photos if crop_ratio < 1.
                if float(args.crop_ratio) >= 1:
                    h, w, c = np.shape(image)

                    retinaface_boxes, retinaface_keypoints, _ = call_face_crop(retinaface_detection, image, 1, prefix="tmp")
                    retinaface_box = retinaface_boxes[0]
                    retinaface_keypoint = retinaface_keypoints[0]

                    face_width = retinaface_box[2] - retinaface_box[0]
                    face_height = retinaface_box[3] - retinaface_box[1]

                    crop_ratio = float(args.crop_ratio)
                    retinaface_box[0] = np.clip(np.array(retinaface_box[0], np.int32) - face_width * (crop_ratio - 1) / 2, 0, w - 1)
                    retinaface_box[1] = np.clip(np.array(retinaface_box[1], np.int32) - face_height * (crop_ratio - 1) / 4, 0, h - 1)
                    retinaface_box[2] = np.clip(np.array(retinaface_box[2], np.int32) + face_width * (crop_ratio - 1) / 2, 0, w - 1)
                    retinaface_box[3] = np.clip(np.array(retinaface_box[3], np.int32) + face_height * (crop_ratio - 1) / 4 * 3, 0, h - 1)

                    # Calculate the left, top, right, bottom of all faces now
                    left, top, right, bottom = retinaface_box
                    # Calculate the width, height, center_x, and center_y of all faces, and get the long side for rec
                    width, height, center_x, center_y = [right - left, bottom - top, (left + right) / 2, (top + bottom) / 2]
                    long_side = min(width, height)

                    # Calculate the new left, top, right, bottom of all faces for clipping
                    # Pad the box to square for saving GPU memomry
                    left, top = int(np.clip(center_x - long_side // 2, 0, w - 1)), int(np.clip(center_y - long_side // 2, 0, h - 1))
                    right, bottom = int(np.clip(left + long_side, 0, w - 1)), int(np.clip(top + long_side, 0, h - 1))
                    retinaface_box = [left, top, right, bottom]

                    # face crop
                    sub_image = image.crop(retinaface_box)
                else:
                    sub_image = image
                if skin_retouching_bool:
                    try:
                        sub_image = Image.fromarray(cv2.cvtColor(skin_retouching(sub_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
                    except Exception as e:
                        torch.cuda.empty_cache()
                        logging.error(f"Photo skin_retouching error, error info: {e}")

                try:
                    if np.shape(sub_image)[0] < 768 or np.shape(sub_image)[1] < 768:
                        sub_image = Image.fromarray(cv2.cvtColor(portrait_enhancement(sub_image)[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
                except Exception as e:
                    torch.cuda.empty_cache()
                    logging.error(f"Photo enhance error, error info: {e}")

                images.append(sub_image)
            except Exception as e:
                torch.cuda.empty_cache()
                logging.error(f"Photo detect and count score error, error info: {e}")

        if len(images) > 0:
            # Save the reference image displayed in the Scene Lora gallery.
            images[0].save(ref_image_path)

    # write results
    for index, base64_pilimage in enumerate(images):
        image = base64_pilimage.convert("RGB")
        image.save(os.path.join(images_save_path, str(index) + ".jpg"))
        print("save processed image to " + os.path.join(images_save_path, str(index) + ".jpg"))
        with open(os.path.join(images_save_path, str(index) + ".txt"), "w") as f:
            if isinstance(validation_prompt, list):
                f.write(validation_prompt[index])
            else:
                f.write(validation_prompt)

    with open(json_save_path, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(images_save_path, topdown=False):
            for file in files:
                path = os.path.join(root, file)
                if not file.endswith("txt"):
                    txt_path = ".".join(path.split(".")[:-1]) + ".txt"
                    if os.path.exists(txt_path):
                        prompt = open(txt_path, "r").readline().strip()
                        if platform.system() == "Windows":
                            path = path.replace("\\", "/")
                        jpg_path_split = path.split("/")
                        file_name = os.path.join(*jpg_path_split[-2:])
                        a = {"file_name": file_name, "text": prompt}
                        f.write(json.dumps(eval(str(a))))
                        f.write("\n")

    del retinaface_detection
    del salient_detect
    del skin_retouching
    del portrait_enhancement
    del face_recognition
    torch.cuda.empty_cache()
