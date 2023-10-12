import cv2
import numpy as np
from PIL import Image
import math
import time
import random
import scipy.spatial
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon

import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt

def mask_to_polygon(mask, epsilon_multiplier):
    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 近似多边形
    epsilon = epsilon_multiplier * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 创建多边形mask
    polygon_mask = np.zeros_like(mask)
    cv2.drawContours(polygon_mask, [approx_polygon], -1, 255, thickness=cv2.FILLED)

    # 创建图像用于绘制多边形和标签
    polygon_image = np.zeros_like(mask)
    cv2.drawContours(polygon_image, [approx_polygon], -1, 255, thickness=cv2.FILLED)

    # 获取多边形的顶点坐标
    approx_polygon = approx_polygon.reshape(-1, 2)

    # 绘制顶点标记和标签
    for i, (x, y) in enumerate(approx_polygon):
        cv2.circle(polygon_image, (x, y), 3, (255, 255, 255), -1)  # 绘制红色的圆点
        cv2.putText(polygon_image, str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 计算IoU（如果需要）

    # 显示图像
    plt.imshow(polygon_image)
    plt.title('Polygon with Labels')
    plt.show()

    return approx_polygon
    

def calculate_iou(mask1, mask2):
    # 计算交集
    intersection = np.logical_and(mask1, mask2)

    # 计算并集
    union = np.logical_or(mask1, mask2)

    # 计算IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou

def mask_to_polygon_with_labels(mask, epsilon_multiplier, iou_threshold):
    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 近似多边形
    epsilon = epsilon_multiplier * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 创建多边形mask
    polygon_mask = np.zeros_like(mask)
    cv2.drawContours(polygon_mask, [approx_polygon], -1, 255, thickness=cv2.FILLED)

    # 创建图像用于绘制多边形和标签
    polygon_image = np.zeros_like(mask)
    cv2.drawContours(polygon_image, [approx_polygon], -1, 255, thickness=cv2.FILLED)

    # 获取多边形的顶点坐标
    approx_polygon = approx_polygon.reshape(-1, 2)

    # 绘制顶点标记和标签
    for i, (x, y) in enumerate(approx_polygon):
        cv2.circle(polygon_image, (x, y), 3, (255, 255, 255), -1)  # 绘制红色的圆点
        cv2.putText(polygon_image, str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 计算IoU（如果需要）

    # 显示图像
    plt.imshow(polygon_image)
    plt.title('Polygon with Labels')
    plt.show()

    # 如果IoU小于阈值，则返回None
    # if iou < iou_threshold:
    #     return None

    return approx_polygon
    

def calculate_iou(mask1, mask2):
    # 计算交集
    intersection = np.logical_and(mask1, mask2)

    # 计算并集
    union = np.logical_or(mask1, mask2)

    # 计算IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou
    
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} time: {execution_time} s")
        return result
    return wrapper



def draw_points_on_image(img, point):
    img_with_points = img.copy()
    x, y = point
    cv2.circle(img_with_points, (int(x), int(y)), 5, (0, 0, 255), -1)

    return img_with_points


def calculate_polygon_center(polygon):
    center_x = polygon[:, 0].mean()
    center_y = polygon[:, 1].mean()
    return np.array((int(center_x), int(center_y)))


def calculate_box_center(box):
    return np.array((int((box[0]+box[2])/2),int((box[1]+box[3])/2)))


def resize_and_stretch(img, target_size, is_mask = False, white_back = False):
    """
        crop img，mask by box, resize to target size
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
          
    # crop main
    # img = img.crop((box[0], box[1], box[2], box[3]))
    img.thumbnail(target_size)

    if is_mask:
        resized_img = Image.new("L", target_size, color='white' if white_back else 'black')
    else:
        resized_img = Image.new("RGB", target_size, color='white' if white_back else 'black')

    # add to center
    offset_x = (target_size[0] - img.size[0]) // 2
    offset_y = (target_size[1] - img.size[1]) // 2

    # paste
    resized_img.paste(img, (offset_x, offset_y))
    resized_img = np.array(resized_img)
    return resized_img


def paste_image_centered_at(img1, img2, mask1, mask2, x, y, is_canny = True):
    """
        paste img1 to img2 with the center of x,y
        filter the final result by mask1, mask2 and calculate the iou
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # left up 
    paste_x = int(x - w1 // 2)
    paste_y = int(y - h1 // 2)

    # ensure img1 is smaller than img2
    if paste_x < 0:
        paste_x = 0
    if paste_y < 0:
        paste_y = 0
    if paste_x + w1 > w2:
        img1 = img1[:, :w2 - paste_x]
        mask1 = mask1[:, :w2 - paste_x]
    if paste_y + h1 > h2:
        img1 = img1[:h2 - paste_y, :]
        mask1 = mask1[:h2 - paste_y, :]

    # direct paste img1 to img2
    expand_img1 = np.ones((h2,w2,3)) * 255.
    expand_mask1 = np.zeros((h2,w2))
    
    # final result
    result_img = np.zeros((h2,w2,3))
    
    # merge mask1 mask2, white use img1, black use img2
    merge_mask = np.zeros((h2,w2))

    merge_mask[paste_y:paste_y + h1, paste_x:paste_x + w1] = mask1
    merge_mask = np.where(np.logical_and(merge_mask > 128, mask2 > 128), 255, 0)

    # paste img1, mask1
    expand_img1[paste_y:paste_y + h1, paste_x:paste_x + w1:] = img1
    expand_mask1[paste_y:paste_y + h1, paste_x:paste_x + w1:] = mask1

    iou = calculate_iou(expand_mask1,mask2)
    
    # filter the final result
    result_img[merge_mask == 255] = expand_img1[merge_mask == 255]
    result_img[merge_mask == 0] = img2[merge_mask == 0]

    return result_img,iou,expand_mask1
    

def rotate_resize_image(array, angle, scale_ratio):
    """
        rotate img with angle and scale by scale_ratio
    """
    height, width = array.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, int(angle), 1)

    # resize the output image to save corner
    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += (bound_w - width) / 2
    rotation_mat[1, 2] += (bound_h - height) / 2

    rotated_mat = cv2.warpAffine(array, rotation_mat, (bound_w, bound_h))

     # scale
    scaled_mat = cv2.resize(rotated_mat, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)
    return scaled_mat

    
@timing_decorator
def crop_and_paste(img1, mask1, img2, mask2, angle, x, y, ratio, use_local = False):
    # rotate and resize img1 mask1
    img1 = rotate_resize_image(img1, angle, ratio)
    mask1 = rotate_resize_image(mask1, angle, ratio)
    rotate_img1 = copy.deepcopy(img1)

    # use local
    if use_local:
        img2 = paste_image_center(img2, img1)
        mask2 = paste_image_center(mask2, mask1)

    # paste to mask2
    result_img,iou,mask1 = paste_image_centered_at(img1, img2, mask1, mask2, x, y)

    return result_img, rotate_img1, iou, mask1, mask2


def match_polygon_points(polygon1, polygon2):
    # 将输入多边形转换为NumPy数组
    polygon1 = np.array(polygon1)
    polygon2 = np.array(polygon2)

    # 初始化匹配字典，用于存储匹配结果
    match_dict = {}

    # 遍历第一个多边形的每个顶点
    for idx, point1 in enumerate(polygon1):
        # 如果polygon2中的点都已被匹配，跳出循环
        if len(match_dict) == len(polygon2):
            break

        # 计算第一个多边形上的顶点与尚未匹配的polygon2上所有顶点的距离
        unmatched_polygon2_indices = [i for i in range(len(polygon2)) if i not in match_dict.values()]
        distances = scipy.spatial.distance.cdist([point1], polygon2[unmatched_polygon2_indices])

        # 找到距离最小的索引，即第二个多边形上最近的点的索引
        min_distance_idx = unmatched_polygon2_indices[np.argmin(distances)]

        # 将第一个多边形上的顶点与最近的第二个多边形上的顶点进行匹配
        match_dict[idx] = min_distance_idx

    return match_dict


def calculate_polygon_iou(polygon1_coords, polygon2_coords):
    polygon1_coords = np.array(polygon1_coords).astype(np.int32)
    polygon2_coords = np.array(polygon2_coords).astype(np.int32)
    
    # 创建两个Shapely Polygon对象
    polygon1 = Polygon(polygon1_coords)
    polygon2 = Polygon(polygon2_coords)

    # 计算交集区域的面积
    intersection = polygon1.intersection(polygon2).area

    # 计算并集区域的面积
    union = polygon1.union(polygon2).area

    # 计算IoU
    iou = intersection / union
    
    return iou


def label_polygon_vertices(image, polygon):
    # 复制输入图像以避免更改原始图像
    output_image = image.copy()

    # 绘制多边形
    cv2.polylines(output_image, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)

    # 标注顶点
    for i, (x, y) in enumerate(polygon):
        cv2.circle(output_image, (x, y), 5, (0, 0, 255), -1)  # 绘制红色的圆点
        cv2.putText(output_image, str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return output_image
    

def show_polygons(polygon1,polygon2):
    # 创建一个空白图
    fig, ax = plt.subplots()
    
    # 绘制polygon1
    ax.plot(polygon1[:, 0], polygon1[:, 1], color='blue')
    # 标注polygon1的顶点序号
    for i, point in enumerate(polygon1):
        ax.text(point[0], point[1], str(i+1), color='blue')
    
    # 绘制polygon2
    ax.plot(polygon2[:, 0], polygon2[:, 1], color='red')
    # 标注polygon2的顶点序号
    for i, point in enumerate(polygon2):
        ax.text(point[0], point[1], str(i+1), color='red')
    
    # 设置坐标轴范围
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    
    # 显示图形
    plt.show()

def paste_image_center(img1, img2):
    """
    将图像1粘贴在图像2的中心，不调整图像1的大小

    参数:
    img1 (numpy.ndarray): 要粘贴的图像
    img2 (numpy.ndarray): 目标图像

    返回:
    numpy.ndarray: 结果图像
    """
    # 获取图像2的宽度和高度
    width2, height2 = img2.shape[1], img2.shape[0]

    # 计算将图像1放置在图像2中心的坐标
    left = (width2 - img1.shape[1]) // 2
    upper = (height2 - img1.shape[0]) // 2
    right = left + img1.shape[1]
    lower = upper + img1.shape[0]

    # 确保坐标是非负的
    left = max(left, 0)
    upper = max(upper, 0)

    # 从图像2中复制相应区域
    # centered_image = img2.copy()
    # centered_image = np.full(img2.shape, 255, dtype=np.uint8)
    centered_image = np.zeros_like(img2)

    centered_image[upper:lower, left:right] = img1

    return centered_image
    

def align_and_overlay_images(img1, img2, mask1, mask2, angle=0.0, ratio=1.0, box2=None, find_param={}):
    # crop img1 box region to same size as img2 box region
    if box2:
        # use box2 to mark center
        resized_img1 = resize_and_stretch(img1, target_size=(box2[2]-box2[0],box2[3]-box2[1]), white_back=True)
        resized_mask1 = resize_and_stretch(mask1, target_size=(box2[2]-box2[0],box2[3]-box2[1]))
        resized_mask1 = resized_mask1[:,:,0]
    else:
        # default center is the center of img2
        resized_img1 = resize_and_stretch(img1, target_size=(img2.shape[1],img2.shape[0]), white_back=True)
        resized_mask1 = resize_and_stretch(mask1,target_size=(mask2.shape[1],mask2.shape[0]))
        resized_mask1 = resized_mask1[:,:,0]

    # find optimal angle, ratio
    if box2:
        x, y = calculate_box_center(box2)
    else:
        x, y = img2.shape[1]/2, img2.shape[0]/2
    
    ious = []
    max_iou = 0
    if find_param:
        angle_low   = find_param.get('angle_low', -60)
        angle_high  = find_param.get('angle_high', 60)
        angle_num   = find_param.get('angle_num', 5)
        ratio_low   = find_param.get('ratio_low', 0.8)
        ratio_high  = find_param.get('ratio_high', 1.5)
        ratio_num   = find_param.get('ratio_num', 5)
        
        angles      = [random.uniform(angle_low, angle_high) for _ in range(angle_num)]
        ratios      = [random.uniform(ratio_low, ratio_high) for _ in range(ratio_num)]
        
        for angle in angles:
            for ratio in ratios:
                res_img, rotate_img1 ,iou, mask1, mask2 = crop_and_paste(resized_img1, resized_mask1, img2, mask2, angle, x, y, ratio)
                ious.append(iou)
                if iou > max_iou:
                    max_iou = iou
                    final_res = res_img
    else:
        final_res, rotate_img1, iou, mask1, mask2 = crop_and_paste(resized_img1, resized_mask1, img2, mask2, angle, x, y, ratio)

    return final_res, rotate_img1, mask1, mask2

def crop_image(img, box):
    """
    使用OpenCV裁剪图像

    参数:
    img (numpy.ndarray): 输入的图像 (使用cv2读取的图像)
    box (tuple): 裁剪框的坐标 (left, upper, right, lower)

    返回:
    numpy.ndarray: 裁剪后的图像
    """
    left, upper, right, lower = box
    cropped_img = img[upper:lower, left:right]
    
    return cropped_img

def calculate_average_distance(original_points, new_points):
    """计算K个点的平均移动距离。
    original_points 和 new_points 是两个形状为(K, 2)的NumPy数组。
    """
    if original_points.shape != new_points.shape:
        print("Shapes of original_points and new_points must match.")
        return None

    distances = np.linalg.norm(new_points - original_points, axis=1)
    average_distance = np.mean(distances)

    return average_distance

def expand_polygon_vertex(A, n):
    # 计算多边形A的边数
    num_edges = len(A)

    # 计算每条边上应该采样的点数
    edge_lengths = [np.linalg.norm(A[(i + 1) % num_edges] - A[i]) for i in range(num_edges)]
    total_length = sum(edge_lengths)
    
    # 计算每条边上的点数，采用四舍五入方式分配
    points_per_edge = [int(round(n * length / total_length)) for length in edge_lengths]
    
    # 计算剩余的点数
    remaining_points = n - sum(points_per_edge)
    
    # 将剩余的点数分配给最长的边
    max_edge_index = points_per_edge.index(max(points_per_edge))
    points_per_edge[max_edge_index] += remaining_points

    print(points_per_edge)
    print(n,sum(points_per_edge))
    # 创建一个新的顶点列表，用于存储扩展后的多边形顶点
    expanded_A = []

    for i in range(num_edges):
        # 获取当前边的起点和终点
        start_point = A[i]
        end_point = A[(i + 1) % num_edges]

        # 计算当前边上的步长
        num_points_on_edge = points_per_edge[i]
        if num_points_on_edge!=0:
            step = 1.0 / num_points_on_edge
    
            # 生成均匀分布的点，包括起点但不包括终点
            for j in range(num_points_on_edge):
                t = j * step
                interpolated_point = (1 - t) * start_point + t * end_point
                expanded_A.append(interpolated_point)

    # 转换为NumPy数组
    expanded_A = np.array(expanded_A)
    return expanded_A

def adjust_B_to_match_A(A, B):
    # 找到多边形B中与多边形A中0号点最近的点的索引
    distances = np.linalg.norm(A[0] - B, axis=1)
    nearest_point_index = np.argmin(distances)

    # 重新排列多边形B的顶点，以匹配A的0号点
    adjusted_B = np.roll(B, -nearest_point_index, axis=0)

    return adjusted_B