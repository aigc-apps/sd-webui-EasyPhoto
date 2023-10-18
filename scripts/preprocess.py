import argparse
import json
import logging
import math
import os
import platform
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import time
from collections import defaultdict
from glob import glob
from operator import itemgetter
from shutil import copyfile

import cv2
import numpy as np
# Third-party libraries
import torch
# Custom libraries
from lightglue import DISK, LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
import numpy as np
import torch
import cv2
from sklearn.cluster import KMeans
from segment_anything import SamPredictor, sam_model_registry
from easyphoto_process_utils import rotate_resize_image

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
        "--main_image_path",
        type=str,
        default=None,
        help=(
            "The main_image_path."
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
        "--sam_model_path",
        type=str,
        default=None,
        help=(
            "The path of Sam model."
        ),
    )
    args = parser.parse_args()
    return args

def get_match_points(img_path1, img_path2, extractor, matcher):
    key = (img_path1, img_path2)
    if key not in MATCH_CACHE.keys():
        logging.info("=== Extracting features and matching ===")

        # Feature extraction and matching
        base_image = load_image(img_path1).cuda()
        base_feats = extractor.extract(base_image)
        image = load_image(img_path2).cuda()
        feats = extractor.extract(image)

        # Running the matcher
        matches = matcher({'image0': base_feats, 'image1': feats})

        # Prepare for visualization
        feats0, feats1, matches = [rbd(x) for x in [base_feats, feats, matches]]
        kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        MATCH_CACHE[key] = {'m_kpts0':m_kpts0.cpu().numpy(), 'm_kpts1':m_kpts1.cpu().numpy(), 'matches':matches.cpu().numpy()}
        torch.cuda.empty_cache()
    return MATCH_CACHE[key]
    
def find_farthest_points(points, k):
    # 初始化结果列表和起始点
    result = []
    result.append(points[np.random.randint(len(points))])
    
    for _ in range(k - 1):
        max_avg_distance = -1
        farthest_point = None
        
        for point in points:
            avg_distance = np.mean(np.linalg.norm(result - point, axis=1))
            
            if avg_distance > max_avg_distance:
                max_avg_distance = avg_distance
                farthest_point = point
                
        result.append(farthest_point)
        
    return np.array(result)

def find_farthest_points_kmeans(points, k):
    # 使用 KMeans 对数据点进行聚类
    kmeans = KMeans(n_clusters=k).fit(points)
    
    # 获得每个聚类的中心
    cluster_centers = kmeans.cluster_centers_
    
    # 初始化一个数组来保存最终的 k 个点
    farthest_points = np.zeros((k, 2))
    
    for i, center in enumerate(cluster_centers):
        # 对于每个聚类中心，找到距离它最远的点
        cluster_points = points[kmeans.labels_ == i]
        distances = np.linalg.norm(cluster_points - center, axis=1)
        farthest_point = cluster_points[np.argmax(distances)]
        
        # 将找到的点保存到结果数组中
        farthest_points[i] = farthest_point
    
    return farthest_points

def refine_matched_pairs_per_image(matched_pairs_per_image, sam_points=5):
    def find_k_centers(matched_pairs_per_image, k=5):
        all_points = []
        
        for pairs in matched_pairs_per_image.values():
            for ref_point, _ in pairs:
                all_points.append(ref_point)
        
        all_points = np.array(all_points)
        kmeans = KMeans(n_clusters=k).fit(all_points)
        centers = kmeans.cluster_centers_
        
        return centers
    
    def filter_pairs_by_centers(matched_pairs_per_image, centers):
        filtered_pairs_per_image = {}
        
        for key, pairs in matched_pairs_per_image.items():
            closest_pairs = []
            
            for center in centers:
                min_distance = float('inf')
                closest_pair = None
                
                for ref_point, match_point in pairs:
                    distance = np.linalg.norm(np.array(ref_point) - center)
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_pair = (ref_point, match_point)
                        
                closest_pairs.append(closest_pair)
            
            filtered_pairs_per_image[key] = closest_pairs
        
        return filtered_pairs_per_image
    centers = find_k_centers(matched_pairs_per_image, sam_points)
    filtered_pairs_per_image = filter_pairs_by_centers(matched_pairs_per_image, centers)
    return filtered_pairs_per_image

def plot_sam_result(image_list, matched_pairs_per_image, idx, pair_idx=1):
    # mask_idx in 0,1,2
    # sam_point 5 is good enought, not much better
    image1 = cv2.imread(image_list[idx])
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    # matched pair hasn't record 0-0 pair
    if idx == 0:
        pair_idx = 0
        pair_1 = matched_pairs_per_image[list(matched_pairs_per_image.keys())[0]]
    else:
        pair_1 = matched_pairs_per_image[idx]
        
    kpts1 = np.array([pair[pair_idx] for pair in pair_1])

    input_point = kpts1
    input_label = np.array([1]*kpts1.shape[0])
    predictor.set_image(image1)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    return masks, kpts1, image1

def safe_expand(x, y, w, h, img_width, img_height, expand_ratio=1.2):
    """
    安全地扩大一个矩形框。

    参数：
    x, y, w, h - 矩形框的坐标和尺寸
    img_width, img_height - 图像的宽度和高度
    expand_ratio - 扩大比例，默认为 1.2

    返回：
    新的 x, y, w, h
    """
    # 计算扩大后的宽度和高度
    new_w = int(w * expand_ratio)
    new_h = int(h * expand_ratio)

    # 计算新的左上角坐标，以保持矩形框居中
    new_x = x + w // 2 - new_w // 2
    new_y = y + h // 2 - new_h // 2

    # 确保新的矩形框在图像内部
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    new_w = min(img_width - new_x, new_w)
    new_h = min(img_height - new_y, new_h)

    return new_x, new_y, new_w, new_h

def extract_subimage_with_mask_and_kpts(image, mask, kpts, kernel_size=5, safe_expand_ratio=1.2, background=255):
    mask = mask[0].astype(np.uint8)  # mask = [sam_outputs_num, h, w]
    mask = mask * 255

    kernel_size = min(kernel_size, min(mask.shape[0], mask.shape[1])//20)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel)
    dilated_mask = cv2.dilate(eroded_mask, kernel)
    
    # 寻找联通区域
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找到面积最大的联通区域（主体）
    max_contour = max(contours, key=cv2.contourArea)
    
    # 获取主体的矩形框
    x, y, w, h = cv2.boundingRect(max_contour)
    # 获取 mask 中为 1 的点的坐标
    x, y, w, h = safe_expand(x, y, w, h, image.shape[1], image.shape[0], safe_expand_ratio)
    
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # print(mask.shape, image.shape, mask_3d.shape)
    image[mask_3d == 0] = background
    cropped_image = image[y:y+h, x:x+w]
    cropped_mask = mask_3d[y:y+h, x:x+w]
    return cropped_image, cropped_mask

if __name__ == "__main__":
    args                = parse_args()
    images_save_path    = args.images_save_path
    json_save_path      = args.json_save_path
    validation_prompt   = args.validation_prompt
    inputs_dir          = args.inputs_dir
    main_image_path     = args.main_image_path
    ref_image_path      = args.ref_image_path
    sam_model_path      = args.sam_model_path

    max_match_num       = 2048
    final_point         = 50
    sigma_ratio         = -0.4
    K                   = 5
    point_url           = 'http://pai-vision-exp.oss-cn-zhangjiakou.aliyuncs.com/wzh-zhoulou/lightglue/superpoint_v1.pth'
    glue_url            = 'http://pai-vision-exp.oss-cn-zhangjiakou.aliyuncs.com/wzh-zhoulou/lightglue/superpoint_lightglue.pth'

    default_extractor   = SuperPoint(max_num_keypoints=max_match_num, url=point_url).eval().cuda()
    default_matcher     = LightGlue(features='superpoint', url=glue_url).eval().cuda()
    sam                 = sam_model_registry['vit_l'](checkpoint=sam_model_path)
    predictor           = SamPredictor(sam)
    
    # image list and main image
    image_list          = glob(f'{inputs_dir}/*.jpg')  + glob(f'{inputs_dir}/*.png') + glob(f'{inputs_dir}/*.jpeg') 
    main_path           = main_image_path
    if main_path is not None:
        image_list      = [main_path] + image_list
    N                   = len(image_list)
    
    # cache init
    MATCH_CACHE         = defaultdict(lambda: {'m_kpts0':[], 'm_kpts1':[]})
    keypoint_counter    = defaultdict(lambda: {'count': 0, 'score': 0})
    keypoint_matches    = defaultdict(list)
    image_pair_scores   = defaultdict(int)
    keypoints_info      = {}
    num_matches_list    = []

    # Loop to collect data
    start_time = time.time()
    for i in range(1, N):
        iter_start_time             = time.time()
        match_points                = get_match_points(image_list[0], image_list[i], default_extractor, default_matcher)
        m_kpts0, m_kpts1, matches   = match_points['m_kpts0'], match_points['m_kpts1'], match_points['matches']

        num_matches                 = m_kpts0.shape[0]
        keypoints_info[(0, i)]      = matches
        image_pair_scores[(0, i)]   = num_matches
        num_matches_list.append(num_matches)
        
        # Update the number of occurrences and trust scores of keypoints and record the matches in the top K images
        for idx, (kpt0, kpt1) in enumerate(zip(m_kpts0, m_kpts1)):
            key_tuple = tuple(map(int, kpt0))
            keypoint_counter[key_tuple]['count'] += 1
            keypoint_counter[key_tuple]['score'] += 1 / (idx + 1)
            if i < K:
                keypoint_matches[key_tuple].append(tuple(map(int, kpt1)))
        logging.info(f"Time taken for image {i}: {time.time() - iter_start_time:.2f} seconds")

    # set threshold dynamic
    logging.info(num_matches_list)
    mean_matches    = np.mean(num_matches_list)
    std_matches     = np.std(num_matches_list)
    threshold       = mean_matches + sigma_ratio * std_matches 
    logging.info(mean_matches, std_matches, threshold)

    # choose image based on threshold
    selected_image_indices = set()
    for (i, j), num_matches in image_pair_scores.items():
        if num_matches > threshold:
            selected_image_indices.add(i)
            selected_image_indices.add(j)

    logging.info(f"Dynamic threshold is set at: {threshold}")
    logging.info(f"Selected images for the common subject are: {sorted(list(selected_image_indices))}")

    # get final list
    image_list = [image_list[i] for i in sorted(selected_image_indices)]
    N = len(image_list)  # Number of images
    keypoint_counter = defaultdict(lambda: {'count': 0, 'score': 0.0, 'matches': {}})
    matched_pairs_per_image = defaultdict(list)

    final_point = int(threshold) - 1
    logging.info(f"Now set final_point to {final_point}")

    # Main Loop
    start_time = time.time()

    for i in range(1, N):
        iter_start_time = time.time()
        match_points = get_match_points(image_list[0], image_list[i], default_extractor, default_matcher)
        m_kpts0, m_kpts1, matches = itemgetter('m_kpts0', 'm_kpts1', 'matches')(match_points)

        # Update keypoint counter and collect matches
        for idx, (kpt0, kpt1) in enumerate(zip(m_kpts0, m_kpts1)):
            key_tuple = tuple(map(int, kpt0))
            kpt_counter = keypoint_counter[key_tuple]
            
            kpt_counter['count'] += 1
            kpt_counter['score'] += 1 / (idx + 1)
            kpt_counter['matches'].setdefault(i, []).append(tuple(map(int, kpt1)))

        logging.info(f"Time taken for image {i}: {time.time() - iter_start_time:.2f} seconds")

    # Sort and Select Top 50 Keypoints
    sorted_keypoints = sorted(
        keypoint_counter.keys(), 
        key=lambda k: (keypoint_counter[k]['count'], keypoint_counter[k]['score']), 
        reverse=True)[:final_point]

    # Collect Matched Pairs for Each Image
    for kpt in sorted_keypoints:
        for image_idx, matched_kpts in keypoint_counter[kpt]['matches'].items():
            matched_pairs_per_image[image_idx].extend((kpt, mkpt) for mkpt in matched_kpts)

    plot_num = min(5, len(image_list))
    filtered_pairs_per_image = refine_matched_pairs_per_image(matched_pairs_per_image, 5)
    mask_list = []
    ref_list = []

    os.makedirs(images_save_path, exist_ok=True)
    for i in range(0, plot_num):
        logging.info(f"show sam result on matched points of image{i}")
        mask, kpts, image = plot_sam_result(image_list, filtered_pairs_per_image, i)
        
        if i == 0:
            tkpts =  np.array([pair[0] for pair in matched_pairs_per_image[1]])
        else:
            tkpts =  np.array([pair[1] for pair in matched_pairs_per_image[i]])

        sub_image, sub_mask = extract_subimage_with_mask_and_kpts(image, mask, tkpts)

        # rotate
        # angle_low = -60
        # angle_high = 60
        # angle_num = 5
        # angles = np.linspace(angle_low, angle_high, angle_num)
        
        # if 0 not in angles:
        #     angles = np.insert(angles, 0, 0)
    
        # for angle_id, angle in enumerate(angles):
        #     rotate_sub = rotate_resize_image(sub_image, angle, 1.0)

        #     cv2.imwrite(f'{images_save_path}/{i}_{angle_id}.jpg',rotate_sub[:, :, ::-1])
        #     with open(os.path.join(images_save_path, str(i) + '_'+str(angle_id)+".txt"), "w") as f:
        #         f.write(validation_prompt)

        mask_list.append(sub_mask)
        ref_list.append(sub_image)

        cv2.imwrite(f'{images_save_path}/{i}.jpg',sub_image[:, :, ::-1])
        with open(os.path.join(images_save_path, str(i) + ".txt"), "w") as f:
            f.write(validation_prompt)

        if i == 0:
            cv2.imwrite(ref_image_path, sub_image[:, :, ::-1])
            cv2.imwrite(os.path.join(os.path.dirname(ref_image_path), os.path.basename(ref_image_path).split(".")[0] + "_mask.jpg"), sub_mask[:, :, ::-1])



    with open(json_save_path, 'w', encoding="utf-8") as f:
        for root, dirs, files in os.walk(images_save_path, topdown=False):
            for file in files:
                path = os.path.join(root, file)
                if not file.endswith('txt'):
                    txt_path = ".".join(path.split(".")[:-1]) + ".txt"
                    if os.path.exists(txt_path):
                        prompt = open(txt_path, 'r').readline().strip()
                        if platform.system() == 'Windows':
                            path = path.replace('\\', '/')
                        jpg_path_split = path.split("/")
                        file_name = os.path.join(*jpg_path_split[-2:])
                        a = {
                            "file_name": file_name, 
                            "text": prompt
                        }
                        f.write(json.dumps(eval(str(a))))
                        f.write("\n")
