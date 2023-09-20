from pathlib import Path
from typing import Callable, List, Union, Tuple

import numpy as np
import torch
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys


def _convert_images(images: Union[List[Image.Image], np.array, torch.Tensor]) -> List[Image.Image]:
    if isinstance(images, List) and isinstance(images[0], Image.Image):
        return images
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    images = [Image.fromarray(image) for image in images]

    return images


def faceid_v0(target_image_dir: str) -> Callable:
    # This pipeline requires the extra package mmcv-full.
    face_recognition = pipeline(Tasks.face_recognition, model="damo/cv_ir101_facerecognition_cfglint")
    target_image_files = Path(target_image_dir).glob("*.jpg")
    target_images = [Image.open(f).convert("RGB") for f in target_image_files]
    target_embs = [face_recognition(t)[OutputKeys.IMG_EMBEDDING][0] for t in target_images]  # (M, 512)
    target_mean_emb = np.mean(np.vstack(target_embs), axis=0)  # (512,)

    # Redundant parameters are for backward compatibility.
    def __call__(src_images: Union[List[Image.Image], np.array, torch.Tensor], prompts: List[str]) -> Tuple[np.array, dict]:
        src_images = _convert_images(src_images)
        src_embs = []  # (N, 512)
        for s in src_images:
            try:
                emb = face_recognition(s)[OutputKeys.IMG_EMBEDDING][0]
            except TypeError:
                emb = np.array([0] * 512)
                print("No face is detected or the size of the detected face size is not enough. Set FaceID to zero.")
            finally:
                src_embs.append(emb)
        faceid_list = np.dot(src_embs, target_mean_emb)

        return faceid_list

    return __call__


def faceid_v2(target_image_dir: str) -> Callable:
    # The retinaface detection is built into the face recognition pipeline.
    face_recognition = pipeline("face_recognition", model='bubbliiiing/cv_retinafce_recognition', model_revision='v1.0.3')
    target_image_files = Path(target_image_dir).glob("*.jpg")
    target_images = [Image.open(f).convert("RGB") for f in target_image_files]

    target_embs = [face_recognition(dict(user=f))[OutputKeys.IMG_EMBEDDING] for f in target_images]  # (M, 512)
    target_mean_emb = np.mean(np.vstack(target_embs), axis=0)  # (512,)

    # Redundant parameters are for backward compatibility.
    def __call__(src_images: Union[List[Image.Image], np.array, torch.Tensor], prompts: List[str]) -> Tuple[np.array, dict]:
        src_images = _convert_images(src_images)
        src_embs = []  # (N, 512)
        for s in src_images:
            try:
                emb = face_recognition(dict(user=s))[OutputKeys.IMG_EMBEDDING][0]
            except:  # cv2.error; TypeError.
                emb = np.array([0] * 512)
                print("No face is detected or the size of the detected face size is not enough. Set the embedding to zero.")
            finally:
                src_embs.append(emb)
        faceid_list = np.dot(src_embs, target_mean_emb)

        return faceid_list

    return __call__