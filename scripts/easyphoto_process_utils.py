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
import torch
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
    # plt.imshow(polygon_image)
    # plt.title('Polygon with Labels')
    # plt.show()

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
        crop img, mask by box, resize to target size
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
          
    # crop main
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

# def paste_image_centered_at(img1, img2, mask1, mask2, x, y):
#     """
#     Paste img1 onto img2 at the specified (x, y) coordinates with mask filtering.
#     """
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]

#     # Calculate the coordinates to place img1 at the specified (x, y) location
#     paste_x = int(x - w1 // 2)
#     paste_y = int(y - h1 // 2)

#     # Ensure img1 is within the bounds of img2
#     if paste_x < 0:
#         paste_x2 = min(w1, w2)
#         paste_x1 = 0
#     else:
#         paste_x2 = min(w1, w2 - paste_x)
#         paste_x1 = paste_x
#     if paste_y < 0:
#         paste_y2 = min(h1, h2)
#         paste_y1 = 0
#     else:
#         paste_y2 = min(h1, h2 - paste_y)
#         paste_y1 = paste_y

#     # Create an empty result image
#     result_img = img2.copy()

#     # Crop img1 and mask1 to fit within the bounds of img2
#     img1_cropped = img1[paste_y1:paste_y2, paste_x1:paste_x2]
#     mask1_cropped = mask1[paste_y1:paste_y2, paste_x1:paste_x2]

#     # Apply mask filtering to result_img
#     for c in range(img2.shape[2]):
#         result_img[paste_y:paste_y + img1_cropped.shape[0], paste_x:paste_x + img1_cropped.shape[1], c] = \
#             img1_cropped * (mask1_cropped / 255.0) + img2[paste_y:paste_y + img1_cropped.shape[0], paste_x:paste_x + img1_cropped.shape[1], c] * (1 - mask1_cropped / 255.0)

#     return result_img

def merge_images(img1, img2, mask1, mask2, x, y, is_canny = True):
    """
        paste img1 to img2 with the center of x,y
        filter the final result by mask1, mask2 and calculate the iou
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    h_max, w_max = max(h1,h2), max(w1,w2)
    # final result large
    expand_img1 = np.zeros((h_max,w_max,3))
    expand_img2 = np.zeros((h_max,w_max,3))
    result_img = np.zeros((h_max,w_max,3))
    expand_mask1 = np.zeros((h_max,w_max))
    expand_mask2 = np.zeros((h_max,w_max))

    img2_box = [(w_max-w2)//2, (h_max-h2)//2, (w_max + w2)//2, (h_max+h2)//2]
    print('img2_box:', img2_box[2]-img2_box[0],img2_box[3]-img2_box[1])

    # merge mask1 mask2, white use img1, black use img2
    merge_mask = np.zeros((h_max,w_max))

    expand_img1 = paste_image_center(img1,expand_img1)
    expand_img2 = paste_image_center(img2,expand_img2)
    expand_mask1 = paste_image_center(mask1,expand_mask1)
    expand_mask2 = paste_image_center(mask2,expand_mask2)

    # cv2.imwrite('expand_img1.jpg',expand_img1)
    # cv2.imwrite('expand_img2.jpg',expand_img2)
    # cv2.imwrite('expand_mask1.jpg',expand_mask1)
    # cv2.imwrite('expand_mask2.jpg',expand_mask2)

    iou = calculate_iou(expand_mask1,expand_mask2)
    merge_mask = np.where(np.logical_and(expand_mask1 > 128, expand_mask2 > 128), 255, 0)
    cv2.imwrite('merge_mask.jpg',merge_mask)

    result_img[merge_mask == 255] = expand_img1[merge_mask == 255]
    result_img[merge_mask == 0] = expand_img2[merge_mask == 0]

    expand_region_mask = copy.deepcopy(expand_mask2)
    expand_region_mask[merge_mask==255]=0

    cv2.imwrite('result_img11.jpg',result_img)
    cv2.imwrite('expand_img11.jpg',expand_img1)
    # cv2.imwrite('mask2.jpg',mask2)
    # cv2.imwrite('merge_mask.jpg',merge_mask)
    cv2.imwrite('expand_region_mask.jpg',expand_region_mask)

    # print(result_img.shape)
    # print(expand_img1.shape)
    # print(mask2.shape)
    # print(merge_mask.shape)

    # expand
    expand_ratio=1.5 
    result_img = blend_images(result_img, expand_img1, expand_ratio, expand_region_mask)

    # crop to img2
    result_img = crop_image(result_img,img2_box)
    expand_img1 = crop_image(expand_img1,img2_box)
    expand_img2 = crop_image(expand_img2,img2_box)
    expand_mask1 = crop_image(expand_mask1,img2_box)
    expand_mask2 = crop_image(expand_mask2,img2_box)

    cv2.imwrite('expand_img1.jpg',expand_img1)
    cv2.imwrite('expand_img2.jpg',expand_img2)
    cv2.imwrite('expand_mask1.jpg',expand_mask1)
    cv2.imwrite('expand_mask2.jpg',expand_mask2)

    return result_img, expand_img1, expand_img2, expand_mask1, expand_mask2, iou
    # return result_img,expand_mask1


def rotate_resize_image(array, angle=0.0, scale_ratio=1.0, use_white_bg=False):
    """
    Rotate img with angle and scale by scale_ratio and set excess border to white.
    """
    height, width = array.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, scale_ratio)

    # Calculate the size of the new image to accommodate the rotated image
    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    # Adjust the rotation matrix to translate to the center of the new image
    rotation_mat[0, 2] += (bound_w - width) / 2
    rotation_mat[1, 2] += (bound_h - height) / 2

    rotated_mat = cv2.warpAffine(array, rotation_mat, (bound_w, bound_h))

    if use_white_bg:
        # Create a white background with the same size
        white_bg = np.zeros_like(rotated_mat)
        white_bg.fill(255)  # Set the background to white

        # Combine the rotated image and the white background using a mask
        mask = (rotated_mat == 0)
        rotated_mat = np.where(mask, white_bg, rotated_mat)

    # Scale
    scaled_mat = cv2.resize(rotated_mat, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)

    return scaled_mat

# def rotate_resize_image(array, angle=0.0, scale_ratio=1.0):
#     """
#         rotate img with angle and scale by scale_ratio
#     """
#     height, width = array.shape[:2]
#     image_center = (width / 2, height / 2)

#     rotation_mat = cv2.getRotationMatrix2D(image_center, int(angle), 1)

#     # resize the output image to save corner
#     radians = math.radians(angle)
#     sin = math.sin(radians)
#     cos = math.cos(radians)
#     bound_w = int((height * abs(sin)) + (width * abs(cos)))
#     bound_h = int((height * abs(cos)) + (width * abs(sin)))

#     rotation_mat[0, 2] += (bound_w - width) / 2
#     rotation_mat[1, 2] += (bound_h - height) / 2

#     rotated_mat = cv2.warpAffine(array, rotation_mat, (bound_w, bound_h))

#      # scale
#     scaled_mat = cv2.resize(rotated_mat, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)
#     return scaled_mat

    
@timing_decorator
def crop_and_paste(img1, mask1, img2, mask2, angle, x, y, ratio, use_local = False):
    # rotate and resize img1 mask1
    img1 = rotate_resize_image(img1, angle, ratio)
    mask1 = rotate_resize_image(mask1, angle, ratio)
    # rotate_img1 = copy.deepcopy(img1)

    # paste to mask2 (img1 img2 is not the same size now)
    print('bf merge:',img2.shape)
    result_img, img1, img2, mask1, mask2, iou = merge_images(img1, img2, mask1, mask2, x, y)
    print('af merge:',img2.shape)

    return result_img, img1, img2, mask1, mask2, iou


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
    Paste img1 at the center of img2.

    Args:
        img1 (numpy.ndarray): The image to paste.
        img2 (numpy.ndarray): The target image.

    Returns:
        numpy.ndarray: The resulting image.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Calculate the coordinates to place img1 at the center of img2
    paste_x = (w2 - w1) // 2
    paste_y = (h2 - h1) // 2

    # Ensure img1 is within the bounds of img2
    paste_x = max(0, paste_x)
    paste_y = max(0, paste_y)
    paste_x = min(paste_x, w2 - w1)
    paste_y = min(paste_y, h2 - h1)

    # Create an empty result image
    result_img = img2.copy()

    # Paste img1 onto result_img using the calculated coordinates
    result_img[paste_y:paste_y + h1, paste_x:paste_x + w1] = img1

    return result_img

    

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

    cv2.imwrite('resized_img1.jpg',resized_img1)
    cv2.imwrite('resized_mask1.jpg',resized_mask1)
    cv2.imwrite('resized_img2.jpg',img2)
    print('align and overlay (resized img1):',resized_img1.shape)
    print('align and overlay (resized mask1):',resized_mask1.shape)
    print('align and overlay (resized img2):',img2.shape)

    # find optimal angle, ratio
    if box2:
        x, y = calculate_box_center(box2)
    else:
        x, y = img2.shape[1]/2, img2.shape[0]/2
    
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
                # res_img, rotate_img1 ,iou, mask1, mask2 = crop_and_paste(resized_img1, resized_mask1, img2, mask2, angle, x, y, ratio)
                res_img, res_img1, res_img2, res_mask1, res_mask2, iou = crop_and_paste(resized_img1, resized_mask1, img2, mask2, angle, x, y, ratio)

                # print(f'iou: {iou}, angle: {angle}, ratio: {ratio}')
                # print(res_img.shape)
                # print(res_img1.shape)
                # print(res_img2.shape)
                # print(res_mask1.shape)
                # print(res_mask2.shape)

                if iou > max_iou:
                    max_iou = iou
                    final_res = res_img
                    final_img1 = res_img1
                    final_mask1 = res_mask1
                    final_mask2 = res_mask2
                    print(f'update iou! iou: {iou}, angle: {angle}, ratio: {ratio}')
    else:
        final_res, final_img1, final_img2, final_mask1, final_mask2, iou = crop_and_paste(resized_img1, resized_mask1, img2, mask2, angle, x, y, ratio)
        # final_res, rotate_img1, iou, mask1, mask2 = crop_and_paste(resized_img1, resized_mask1, img2, mask2, angle, x, y, ratio)

    return final_res, final_img1, final_mask1, final_mask2

def expand_roi(box, ratio, max_box, eps=0):
    centerx = (box[0] + box[2]) / 2
    centery = (box[1] + box[3]) / 2
    w = box[2] - box[0]
    h = box[3] - box[1]

    nw = w * ratio + eps
    nh = h * ratio + eps

    b0 = int(max(centerx - nw / 2, max_box[0]))
    b1 = int(max(centery - nh / 2, max_box[1]))
    b2 = int(min(centerx + nw / 2, max_box[2]))
    b3 = int(min(centery + nh / 2, max_box[3]))

    return [b0, b1, b2, b3]


def crop_image(img, box, expand_ratio=1.0):
    """
    使用OpenCV裁剪图像

    参数:
    img (numpy.ndarray): 输入的图像 (使用cv2读取的图像)
    box (tuple): 裁剪框的坐标 (left, upper, right, lower)

    返回:
    numpy.ndarray: 裁剪后的图像
    """
    W, H = img.shape[1], img.shape[0]
    # print(W,H)
    x1,y1,x2,y2 = expand_roi(box,ratio=expand_ratio,max_box=[0,0,W,H])

    cropped_img = img[y1:y2, x1:x2]
    
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

def draw_vertex_polygon(image, polygon, name):
    # image = np.zeros((1008, 512, 3))
    polygon = np.array(polygon, dtype=np.int32)
    cv2.polylines(image, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)
    
    # 标注顶点序号
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (x, y) in enumerate(polygon):
        cv2.putText(image, str(i), (x, y), font, 0.5, (255, 255, 255), 2)
    
    cv2.imwrite(f'{name}.jpg',image)


def mask_to_box(mask):
    """
        get the only largest componet and return the bouding box
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # Find the largest connected component
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # Ignore the background label

    # Create a new mask with only the largest connected component
    largest_connected_component_mask = np.zeros_like(mask)
    largest_connected_component_mask[labels == largest_label] = 255

    # Find the bounding box for the largest connected component
    x, y, width, height = cv2.boundingRect(largest_connected_component_mask)

    # Set all regions outside of the bounding box to black
    largest_connected_component_mask[:y, :] = 0
    largest_connected_component_mask[y+height:, :] = 0
    largest_connected_component_mask[:, :x] = 0
    largest_connected_component_mask[:, x+width:] = 0

    return largest_connected_component_mask, (x, y, x + width, y + height)


def draw_box_on_image(image, box, det_path):
   
    plot_image = image.copy()
    height, width, _ = image.shape

    x1,y1,x2,y2 = box
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    # plot
    cv2.rectangle(plot_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    cv2.imwrite(det_path, plot_image)


def seg_by_box(raw_image_rgb, boxes_filt, segmentor,point_coords=None):
    h, w = raw_image_rgb.shape[:2]
    # [1,2,3,4]

    segmentor.set_image(raw_image_rgb)
    transformed_boxes   = segmentor.transform.apply_boxes_torch(torch.from_numpy(np.expand_dims(boxes_filt, 0)), (h, w)).cuda()

    masks, scores, _    = segmentor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=True)

    # muilt mask
    mask = masks[0]
    mask = torch.sum(mask, dim=0)
    mask = mask.cpu().numpy().astype(np.uint8)
    mask_image = mask * 255

    return mask_image


def apply_mask_to_image(img, mask):
    """
    根据掩码将图像中的像素保留，其他地方设为白色

    参数:
    img (numpy.ndarray): 输入的图像
    mask (numpy.ndarray): 掩码图像，值为255的像素将被保留，其他地方将变为白色

    返回:
    numpy.ndarray: 处理后的图像
    """
    # 创建一个白色背景
    white_background = np.full_like(img, 255, dtype=np.uint8)

    # 使用掩码来保留图像中值为255的像素
    result = cv2.bitwise_and(img, img, mask=mask)

    # 将未被掩码保留的区域设置为白色
    result[np.where(mask == 0)] = [255, 255, 255]

    return result


def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad

def canny(img, res=512, thr_a=100, thr_b=200, **kwargs):
    l, h = thr_a, thr_b
    img, remove_pad = resize_image_with_pad(img, res)
    # global model_canny
    # if model_canny is None:
    model_canny = apply_canny
    result = model_canny(img, l, h)
    return remove_pad(result), remove_pad(img)

def HWC3(x):
    x=x.astype(np.uint8)
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def apply_canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

def fill_mask(mask):
    # fill in the logo
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filled_mask = np.zeros_like(mask)
    
    for contour in contours:
        cv2.drawContours(filled_mask, [contour], 0, (255), -1)
    
    return filled_mask

def remove_outline(img, outline_mask):
    inverted_outline = np.logical_not(outline_mask)
    
    filtered_img = np.multiply(img, inverted_outline)
    return filtered_img



def merge_with_inner_canny(image, mask1, mask2):
    print(image.shape)
    print(mask1.shape)
    print(mask2.shape)
    canny_image, resize_image = canny(image)
    cv2.imwrite('canny_image.jpg', canny_image)
    _, resize_mask1 = canny(mask1)
    _, resize_mask2 = canny(mask2)

    mask1_outline  = np.uint8(cv2.dilate(np.array(resize_mask1), np.ones((10, 10), np.uint8), iterations=1) - cv2.erode(np.array(resize_mask1), np.ones((5, 5), np.uint8), iterations=1))
    cv2.imwrite('mask1_outline.jpg', mask1_outline)
    
    mask1_outline = cv2.cvtColor(np.uint8(mask1_outline), cv2.COLOR_BGR2GRAY)
    canny_image_inner = remove_outline(canny_image, mask1_outline)
    cv2.imwrite('canny_image_inner.jpg', canny_image_inner)
    
    # print('after canny:',canny_image.shape, resize_image.shape)
    return resize_image, canny_image_inner
     

def blend_images(img1, img2, expand_ratio, mask):
    # 获取图像的高度和宽度
    height, width = img1.shape[:2]

    # 将img2中心放大expand_ratio
    new_height = int(height * expand_ratio)
    new_width = int(width * expand_ratio)
    resized_img2 = cv2.resize(img2, (new_width, new_height))

    # 计算对齐后的坐标
    align_x = (new_width-width) // 2
    align_y = (new_height-height) // 2

    # 裁剪resized_img2到与result_img相同的大小
    cropped_resized_img2 = resized_img2[align_y:align_y+height, align_x:align_x+width]

    print(cropped_resized_img2.shape)
    # 创建一个蒙版，将img1的mask为白色的区域设置为1，其余为0
    mask = mask / 255

    # 创建一个与img1相同大小的掩码，将img1的mask为白色的区域保留，其余部分用cropped_resized_img2替换
    result_img = img1.copy()
    for c in range(img1.shape[2]):
        result_img[:, :, c] = result_img[:, :, c] * (1 - mask) + cropped_resized_img2[:, :, c] * mask

    return result_img


def copy_white_mask_to_template(img, mask, template, box):
    h,w,_ = template.shape
    expand_mask = np.zeros((h,w))

    print(mask.shape)
    expand_mask[box[1]:box[3], box[0]:box[2]] = mask

    result = np.zeros_like(template)
    result[box[1]:box[3], box[0]:box[2]] = np.array(img, np.uint8)
    result[expand_mask==0] = template[expand_mask==0]
    result[expand_mask==0] = template[expand_mask==0]
    return result

