"""This file defines customize reward funtions. The reward function which takes in a batch of images
and corresponding prompts returns a batch of rewards each time it is called.
"""

from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image


def _convert_images(images: Union[List[Image.Image], np.array, torch.Tensor]) -> List[Image.Image]:
    if isinstance(images, List) and isinstance(images[0], Image.Image):
        return images
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    images = [Image.fromarray(image) for image in images]

    return images


def faceid_SCRFD(target_image_dir: str) -> Callable:
    """The closure returns the Face ID reward function given a user ID. It uses SCRFD to detect the face and then
    use CurricularFace to extract the face feature. SCRFD requires the extra package mmcv-full.

    Args:
        target_image_dir (str): The directory of processed face image files (.jpg) given a user ID.
    """
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
            except Exception as e:  # TypeError
                print("Catch Exception in the reward function faceid_retina: {}".format(e))
                emb = np.array([0] * 512)
                print("No face is detected or the size of the detected face size is not enough. Set the embedding to zero.")
            finally:
                src_embs.append(emb)
        faceid_list = np.dot(src_embs, target_mean_emb)

        return faceid_list

    return __call__


def faceid_retina(target_image_dir: str) -> Callable:
    """The closure returns the Face ID reward function given a user ID. It uses RetinaFace to detect the face and then
    use CurricularFace to extract the face feature. As the detection capability of RetinaFace is weaker than SCRFD,
    many generated side faces cannot be detected.

    Args:
        target_image_dir (str): The directory of processed face image files (.jpg) given a user ID.
    """
    # The retinaface detection is built into the face recognition pipeline.
    face_recognition = pipeline("face_recognition", model="bubbliiiing/cv_retinafce_recognition", model_revision="v1.0.3")
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
            except Exception as e:  # cv2.error; TypeError.
                print("Catch Exception in the reward function faceid_retina: {}".format(e))
                emb = np.array([0] * 512)
                print("No face is detected or the size of the detected face size is not enough. Set the embedding to zero.")
            finally:
                src_embs.append(emb)
        faceid_list = np.dot(src_embs, target_mean_emb)

        return faceid_list

    return __call__
