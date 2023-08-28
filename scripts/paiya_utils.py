import importlib
import os.path
import stat
from collections import OrderedDict

import cv2
import requests
import glob
import gradio as gr
import modules.scripts as scripts
from modules import sd_models, shared
from modules.paths import models_path

external_code = importlib.import_module('extensions-builtin.sd-webui-controlnet.scripts.external_code', 'external_code')

def init_default_script_args(script_runner):
    #find max idx from the scripts in runner and generate a none array to init script_args
    last_arg_index = 1
    for script in script_runner.scripts:
        if last_arg_index < script.args_to:
            last_arg_index = script.args_to
    # None everywhere except position 0 to initialize script args
    script_args = [None]*last_arg_index
    script_args[0] = 0

    # get default values
    with gr.Blocks(): # will throw errors calling ui function without this
        for script in script_runner.scripts:
            if script.ui(script.is_img2img):
                ui_default_values = []
                for elem in script.ui(script.is_img2img):
                    ui_default_values.append(elem.value)
                script_args[script.args_from:script.args_to] = ui_default_values
    return script_args

def urldownload(url, filename):
    """
    下载文件到指定目录
    :param url: 文件下载的url
    :param filename: 要存放的目录及文件名，例如：./test.xls
    :return:
    """
    down_res = requests.get(url)
    with open(filename,'wb') as file:
        file.write(down_res.content)

def self_save_image(save_path, sub_name, image):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, sub_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def traverse_all_files(curr_path, model_list):
    f_list = [(os.path.join(curr_path, entry.name), entry.stat())
              for entry in os.scandir(curr_path)]
    for f_info in f_list:
        fname, fstat = f_info
        if os.path.splitext(fname)[1] in CN_MODEL_EXTS:
            model_list.append(f_info)
        elif stat.S_ISDIR(fstat.st_mode):
            model_list = traverse_all_files(fname, model_list)
    return model_list

def get_all_models(sort_by, filter_by, path):
    res = OrderedDict()
    fileinfos = traverse_all_files(path, [])
    filter_by = filter_by.strip(" ")
    if len(filter_by) != 0:
        fileinfos = [x for x in fileinfos if filter_by.lower()
                     in os.path.basename(x[0]).lower()]
    if sort_by == "name":
        fileinfos = sorted(fileinfos, key=lambda x: os.path.basename(x[0]))
    elif sort_by == "date":
        fileinfos = sorted(fileinfos, key=lambda x: -x[1].st_mtime)
    elif sort_by == "path name":
        fileinfos = sorted(fileinfos)

    for finfo in fileinfos:
        filename = finfo[0]
        name = os.path.splitext(os.path.basename(filename))[0]
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[name] = name + f" [{sd_models.model_hash(filename)}]"

    return res

CN_MODEL_EXTS = [".pt", ".pth", ".ckpt", ".safetensors"]
cn_models_dir = os.path.join(models_path, "ControlNet")
cn_models_dir_old = os.path.join(scripts.basedir(), "models")
cn_models = OrderedDict()      # "My_Lora(abcd1234)" -> C:/path/to/model.safetensors
cn_models_names = {}  # "my_lora" -> "My_Lora(abcd1234)"

ext_dirs = (shared.opts.data.get("control_net_models_path", None), getattr(shared.cmd_opts, 'controlnet_dir', None))
extra_lora_paths = (extra_lora_path for extra_lora_path in ext_dirs
            if extra_lora_path is not None and os.path.exists(extra_lora_path))
paths = [cn_models_dir, cn_models_dir_old, *extra_lora_paths]
if os.path.isdir(shared.cmd_opts.data_dir):
    paths = paths + glob.glob(os.path.join(shared.cmd_opts.data_dir, '*/models/ControlNet')) 
    paths = paths + glob.glob(os.path.join(shared.cmd_opts.data_dir, '*/*/models/ControlNet'))
    paths = list(set(paths))
    
for path in paths:
    sort_by = shared.opts.data.get(
        "control_net_models_sort_models_by", "name")
    filter_by = shared.opts.data.get("control_net_models_name_filter", "")
    found = get_all_models(sort_by, filter_by, path)
    cn_models.update({**found, **cn_models})

def get_controlnet_args(module, weight, blur_ratio):
    global cn_models
    enabled = False
    if module != "none":
        enabled = True

    model_key = {
        "none"          : "none",
        "hed"           : "control_sd15_hed",
        "skybox_seg"    : "control_v11p_sd15_seg",
        "tile"          : "control_v11f1e_sd15_tile",
        "sd_color"      : "control_sd15_sd_random_color",
    }[module]
    if model_key not in cn_models.keys():
        for path in paths:
            sort_by = shared.opts.data.get(
                "control_net_models_sort_models_by", "name")
            filter_by = shared.opts.data.get("control_net_models_name_filter", "")
            found = get_all_models(sort_by, filter_by, path)
            cn_models.update({**found, **cn_models})
    model = cn_models.get(model_key, "none")
    print(model_key, cn_models)

    processor_res = {
        "none"          : 512,
        "hed"           : 512,
        "skybox_seg"    : 512,
        "tile"          : 1024,
        "sd_color"      : 1024,
    }[module]
    control_mode = {
        "none"          : "My prompt is more important",
        "hed"           : "My prompt is more important",
        "skybox_seg"    : "Balanced",
        "tile"          : "My prompt is more important",
        "sd_color"      : "My prompt is more important",
    }[module]
    module = {
        "none"          : "none",
        "hed"           : "hed",
        "skybox_seg"    : "none",
        "tile"          : "none",
        "sd_color"      : "sd_color",
    }[module]

    __controlnet_args   = dict(
        enabled         = enabled,
        module          = module,
        model           = model,
        weight          = weight,
        image           = None,
        invert_image    = False,
        resize_mode     = "Just Resize",
        rgbbgr_mode     = False,
        low_vram        = False,
        processor_res   = processor_res,
        threshold_a     = blur_ratio,
        threshold_b     = 64,
        guidance_start  = 0,
        guidance_end    = 1,
        control_mode    = control_mode
    )
    return __controlnet_args

JPGS_EXTS   = [".pt", ".pth", ".ckpt", ".safetensors"]
jpgs_dir    = os.path.join(models_path, "ControlNet")
jpgs        = OrderedDict()
jpgs_names  = {}

def update_jpgs():
    jpgs.clear()
    ext_dirs = (shared.opts.data.get("control_net_models_path", None), getattr(shared.cmd_opts, 'controlnet_dir', None))
    extra_lora_paths = (extra_lora_path for extra_lora_path in ext_dirs
                if extra_lora_path is not None and os.path.exists(extra_lora_path))
    paths = [cn_models_dir, cn_models_dir_old, *extra_lora_paths]

    for path in paths:
        sort_by = shared.opts.data.get(
            "control_net_models_sort_models_by", "name")
        filter_by = shared.opts.data.get("control_net_models_name_filter", "")
        found = get_all_models(sort_by, filter_by, path)
        cn_models.update({**found, **cn_models})

    # insert "None" at the beginning of `cn_models` in-place
    cn_models_copy = OrderedDict(cn_models)
    cn_models.clear()
    cn_models.update({**{"None": None}, **cn_models_copy})

    cn_models_names.clear()
    for name_and_hash, filename in cn_models.items():
        if filename is None:
            continue
        name = os.path.splitext(os.path.basename(filename))[0].lower()
        cn_models_names[name] = name_and_hash

import cv2
import numpy as np
import os
from PIL import Image
from skimage import transform


def safe_get_box_mask_keypoints(image, retinaface_result, crop_ratio, face_seg, mask_type):
    '''
    Inputs:
        image                   输入图片；
        retinaface_result       retinaface的检测结果；
        crop_ratio              人脸部分裁剪扩充比例；
        face_seg                人脸分割模型；
        mask_type               人脸分割的方式，一个是crop，一个是skin，人脸分割结果是人脸皮肤或者人脸框
    
    Outputs:
        retinaface_box          扩增后相对于原图的box
        retinaface_keypoints    相对于原图的keypoints
        retinaface_mask_pil     人脸分割结果
    '''
    h, w, c = np.shape(image)
    if len(retinaface_result['boxes']) != 0:
        # 获得retinaface的box并且做一手扩增
        retinaface_box      = np.array(retinaface_result['boxes'][0])
        face_width          = retinaface_box[2] - retinaface_box[0]
        face_height         = retinaface_box[3] - retinaface_box[1]
        retinaface_box[0]   = np.clip(np.array(retinaface_box[0], np.int32) - face_width * (crop_ratio - 1) / 2, 0, w - 1)
        retinaface_box[1]   = np.clip(np.array(retinaface_box[1], np.int32) - face_height * (crop_ratio - 1) / 2, 0, h - 1)
        retinaface_box[2]   = np.clip(np.array(retinaface_box[2], np.int32) + face_width * (crop_ratio - 1) / 2, 0, w - 1)
        retinaface_box[3]   = np.clip(np.array(retinaface_box[3], np.int32) + face_height * (crop_ratio - 1) / 2, 0, h - 1)
        retinaface_box      = np.array(retinaface_box, np.int32)

        # 检测关键点
        retinaface_keypoints = np.reshape(retinaface_result['keypoints'][0], [5, 2])
        retinaface_keypoints = np.array(retinaface_keypoints, np.float32)

        # mask部分
        retinaface_crop     = image.crop(np.int32(retinaface_box))
        retinaface_mask     = np.zeros_like(np.array(image, np.uint8))
        if mask_type == "skin":
            retinaface_sub_mask = face_seg(retinaface_crop)
            retinaface_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = np.expand_dims(retinaface_sub_mask, -1)
        else:
            retinaface_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = 255
        retinaface_mask_pil = Image.fromarray(np.uint8(retinaface_mask))
    else:
        retinaface_box          = np.array([])
        retinaface_keypoints    = np.array([])
        retinaface_mask         = np.zeros_like(np.array(image, np.uint8))
        retinaface_mask_pil     = Image.fromarray(np.uint8(retinaface_mask))
        
    return retinaface_box, retinaface_keypoints, retinaface_mask_pil

def crop_and_paste(Source_image, Source_image_mask, Target_image, Source_Five_Point, Target_Five_Point, Source_box):
    '''
    Inputs:
        Source_image            原图像；
        Source_image_mask       原图像人脸的mask比例；
        Target_image            目标模板图像；
        Source_Five_Point       原图像五个人脸关键点；
        Target_Five_Point       目标图像五个人脸关键点；
        Source_box              原图像人脸的坐标；
    
    Outputs:
        output                  贴脸后的人像
    '''
    Source_Five_Point = np.reshape(Source_Five_Point, [5, 2]) - np.array(Source_box[:2])
    Target_Five_Point = np.reshape(Target_Five_Point, [5, 2])

    Crop_Source_image                       = Source_image.crop(np.int32(Source_box))
    Crop_Source_image_mask                  = Source_image_mask.crop(np.int32(Source_box))
    Source_Five_Point, Target_Five_Point    = np.array(Source_Five_Point), np.array(Target_Five_Point)

    tform = transform.SimilarityTransform()
    # 程序直接估算出转换矩阵M
    tform.estimate(Source_Five_Point, Target_Five_Point)
    M = tform.params[0:2, :]

    warped      = cv2.warpAffine(np.array(Crop_Source_image), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)
    warped_mask = cv2.warpAffine(np.array(Crop_Source_image_mask), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)

    mask        = np.float32(warped_mask == 0)
    output      = mask * np.float32(Target_image) + (1 - mask) * np.float32(warped)
    return output

def call_face_crop(retinaface_detection, image, crop_ratio, prefix="tmp"):
    # retinaface检测部分
    # 检测人脸框
    retinaface_result                                           = retinaface_detection(image) 
    # 获取mask与关键点
    retinaface_box, retinaface_keypoints, retinaface_mask_pil   = safe_get_box_mask_keypoints(image, retinaface_result, crop_ratio, None, "crop")

    return retinaface_box, retinaface_keypoints, retinaface_mask_pil