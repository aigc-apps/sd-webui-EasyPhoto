import cv2
from PIL import Image, ImageEnhance, ImageFilter
import math
import time
import random
import scipy.spatial
import os
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
import torch
import cv2
import copy
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from typing import List, Tuple, Union, Optional


def mask_to_polygon(
    mask: np.ndarray, epsilon_multiplier: float = 0.005
) -> List[List[int]]:
    """
    Convert a binary mask to a polygon by finding the largest contour and approximating it.

    Args:
        mask (np.ndarray): The input binary mask.
        epsilon_multiplier (float, optional): The multiplier to control the approximation (default is 0.005).

    Returns:
        List[List[int]]: List of polygon vertices represented as coordinate pairs.
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the polygon
    epsilon = epsilon_multiplier * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Get the vertices of the polygon
    approx_polygon = approx_polygon.reshape(-1, 2)

    return approx_polygon


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate the Intersection over Union (IoU) of two binary masks.

    Args:
        mask1 (np.ndarray): The first binary mask as a NumPy array.
        mask2 (np.ndarray): The second binary mask as a NumPy array.

    Returns:
        float: The IoU (Intersection over Union) between the two masks.
    """
    # Calculate the intersection
    intersection = np.logical_and(mask1, mask2)

    # Calculate the union
    union = np.logical_or(mask1, mask2)

    # Calculate IoU
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
    return np.array((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))


def resize_and_stretch(
    img: Union[np.ndarray, Image.Image],
    target_size: Tuple[int, int],
    is_mask: bool = False,
    white_back: bool = False,
) -> np.ndarray:
    """
    Crop the image or mask, resize it to the target size, and stretch it.

    Args:
        img (Union[np.ndarray, Image.Image]): The input image or mask.
        target_size (Tuple[int, int]): The desired target size (width, height).
        is_mask (bool, optional): Whether the input is a mask (default is False).
        white_back (bool, optional): Whether to use white background (default is False).

    Returns:
        np.ndarray: The resized and stretched image as a NumPy array.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Calculate the aspect ratio
    width, height = img.size
    aspect_ratio = width / height

    # Calculate the new size while preserving the aspect ratio
    new_width = int(min(target_size[0], target_size[1] * aspect_ratio))
    new_height = int(min(target_size[1], target_size[0] / aspect_ratio))

    img = img.resize((new_width, new_height))

    if is_mask:
        resized_img = Image.new(
            "L", target_size, color="white" if white_back else "black"
        )
    else:
        resized_img = Image.new(
            "RGB", target_size, color="white" if white_back else "black"
        )

    # Add to the center
    offset_x = (target_size[0] - img.size[0]) // 2
    offset_y = (target_size[1] - img.size[1]) // 2

    # Paste the image
    resized_img.paste(img, (offset_x, offset_y))
    resized_img = np.array(resized_img)
    return resized_img


def merge_images(
    img1: np.ndarray,
    img2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    x: float,
    y: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Paste img1 onto img2 with a specified center point (x, y), filter the result by mask1 and mask2, and calculate IoU.

    Args:
        img1 (np.ndarray): The first input image to be pasted.
        img2 (np.ndarray): The second input image.
        mask1 (np.ndarray): The mask corresponding to img1.
        mask2 (np.ndarray): The mask corresponding to img2.
        x (float): The x-coordinate of the center point for pasting.
        y (float): The y-coordinate of the center point for pasting.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
            - The resulting merged image.
            - The processed img1.
            - The processed img2.
            - The processed mask1.
            - The processed mask2.
            - The Intersection over Union (IoU) value.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    h_max, w_max = max(h1, h2), max(w1, w2)

    # Create large arrays for the final result and intermediate steps
    expand_img1 = np.zeros((h_max, w_max, 3))
    expand_img2 = np.zeros((h_max, w_max, 3))
    result_img = np.zeros((h_max, w_max, 3))
    expand_mask1 = np.zeros((h_max, w_max))
    expand_mask2 = np.zeros((h_max, w_max))

    # Merge masks to determine where to use img1 and where to use img2
    merge_mask = np.zeros((h_max, w_max))

    # Paste images and masks to the center of the larger arrays
    expand_img1 = paste_image_center(img1, expand_img1)
    expand_img2 = paste_image_center(img2, expand_img2)
    expand_mask1 = paste_image_center(mask1, expand_mask1)
    expand_mask2 = paste_image_center(mask2, expand_mask2)

    cv2.imwrite('expand_mask1.jpg',expand_mask1)
    cv2.imwrite('expand_mask2.jpg',expand_mask2)
    # Calculate IoU between the expanded masks
    iou = calculate_mask_iou(expand_mask1, expand_mask2)

    # Create a merge mask to determine which pixels to use from each image
    merge_mask = np.where(
        np.logical_and(expand_mask1 > 128, expand_mask2 > 128), 255, 0
    )

    # Combine images based on the merge mask
    result_img[merge_mask == 255] = expand_img1[merge_mask == 255]
    result_img[merge_mask == 0] = expand_img2[merge_mask == 0]

    # Determine areas that were expanded with img2 and fill them with the specified color
    # expand_region_mask = copy.deepcopy(expand_mask2)
    # expand_region_mask[merge_mask == 255] = 0

    # white_area = np.where(expand_region_mask > 128)
    # result_img[white_area] = color

    return result_img, expand_img1, expand_img2, expand_mask1, expand_mask2, iou


def rotate_resize_image(
    array: np.ndarray,
    angle: float = 0.0,
    scale_ratio: float = 1.0,
    use_white_bg: Optional[bool] = False
) -> np.ndarray:
    """
    Rotate and resize an image while optionally setting excess border to white.

    Args:
        array (np.ndarray): The input image.
        angle (float, optional): The rotation angle in degrees (default is 0.0).
        scale_ratio (float, optional): The scaling ratio (default is 1.0).
        use_white_bg (bool, optional): Whether to set the excess border to white (default is False).

    Returns:
        np.ndarray: The rotated and resized image.
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
        mask = rotated_mat == 0
        rotated_mat = np.where(mask, white_bg, rotated_mat)

    # Scale
    scaled_mat = cv2.resize(
        rotated_mat,
        None,
        fx=scale_ratio,
        fy=scale_ratio,
        interpolation=cv2.INTER_LINEAR,
    )

    return scaled_mat


@timing_decorator
def crop_and_paste(
    img1: np.ndarray,
    mask1: np.ndarray,
    img2: np.ndarray,
    mask2: np.ndarray,
    angle: float,
    x: float,
    y: float,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Crop and paste one image onto another image while considering rotation, resizing, and color fill.

    Args:
        img1 (np.ndarray): The first input image.
        mask1 (np.ndarray): The mask corresponding to img1.
        img2 (np.ndarray): The second input image.
        mask2 (np.ndarray): The mask corresponding to img2.
        angle (float): The rotation angle for img1.
        x (float): The x-coordinate of the center point for transformation.
        y (float): The y-coordinate of the center point for transformation.
        ratio (float): The scaling ratio for img1.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
            - The resulting merged image.
            - The processed img1.
            - The processed img2.
            - The processed mask1.
            - The processed mask2.
            - The Intersection over Union (IoU) value.
    """
    # Rotate and resize img1 and mask1
    img1 = rotate_resize_image(img1, angle, ratio)
    mask1 = rotate_resize_image(mask1, angle, ratio)

    # Paste img1 onto img2, considering masks
    result_img, img1, img2, mask1, mask2, iou = merge_images(
        img1, img2, mask1, mask2, x, y)

    return result_img, img1, img2, mask1, mask2, iou


def calculate_polygon_iou(polygon1_coords: np.ndarray, polygon2_coords: np.ndarray) -> float:
    """
    Calculate the Intersection over Union (IoU) of two polygons.

    Args:
        polygon1_coords (np.ndarray): Coordinates of the first polygon as a NumPy array.
        polygon2_coords (np.ndarray): Coordinates of the second polygon as a NumPy array.

    Returns:
        float: The IoU (Intersection over Union) between the two polygons.
    """
    # Convert input coordinates to integer arrays
    polygon1_coords = np.array(polygon1_coords).astype(np.int32)
    polygon2_coords = np.array(polygon2_coords).astype(np.int32)

    # Create two Shapely Polygon objects
    polygon1 = Polygon(polygon1_coords)
    polygon2 = Polygon(polygon2_coords)

    # Calculate the area of the intersection region
    intersection = polygon1.intersection(polygon2).area

    # Calculate the area of the union region
    union = polygon1.union(polygon2).area

    # Calculate IoU
    iou = intersection / union

    return iou


def paste_image_center(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Paste img1 at the center of img2.

    Args:
        img1 (np.ndarray): The image to paste.
        img2 (np.ndarray): The target image.

    Returns:
        np.ndarray: The resulting image.
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
    result_img[paste_y: paste_y + h1, paste_x: paste_x + w1] = img1

    return result_img


def align_and_overlay_images(
    img1: np.ndarray,
    img2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    angle: float = 0.0,
    ratio: float = 1.0,
    box2: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Paste img1 to img2 with angle, ratio and mask, while considering optional transformations.

    Args:
        img1 (np.ndarray): The first input image.
        img2 (np.ndarray): The second input image.
        mask1 (np.ndarray): The mask corresponding to img1.
        mask2 (np.ndarray): The mask corresponding to img2.
        angle (float, optional): The rotation angle for img1 (default is 0.0).
        ratio (float, optional): The scaling ratio for img1 (default is 1.0).
        box2 (Optional[Tuple[int, int, int, int]]): The bounding box for ROI in img2 (left, upper, right, lower) (default is None).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
            - The resulting overlay image.
            - The processed img1.
            - The processed mask1.
            - The mask2.
            - The mask2 used in the final overlay.
            - The Intersection over Union (IoU) value.
    """
    # Crop img1 to the same size as img2 based on the provided bounding box or default center.
    if box2:
        x, y = calculate_box_center(box2)
        resized_img1 = resize_and_stretch(
            img1, target_size=(box2[2] - box2[0], box2[3] - box2[1]), white_back=True
        )
        resized_mask1 = resize_and_stretch(
            mask1, target_size=(box2[2] - box2[0], box2[3] - box2[1])
        )
        resized_mask1 = resized_mask1[:, :, 0]
    else:
        x, y = img2.shape[1] / 2, img2.shape[0] / 2
        resized_img1 = resize_and_stretch(
            img1, target_size=(img2.shape[1], img2.shape[0]), white_back=True
        )
        resized_mask1 = resize_and_stretch(
            mask1, target_size=(mask2.shape[1], mask2.shape[0])
        )
        resized_mask1 = resized_mask1[:, :, 0]

    # cv2.imwrite("resized_img1.jpg", resized_img1)
    # cv2.imwrite("resized_mask1.jpg", resized_mask1)
    # cv2.imwrite("resized_img2.jpg", img2)
    print("align and overlay (resized img1):", resized_img1.shape)
    print("align and overlay (resized mask1):", resized_mask1.shape)
    print("align and overlay (resized img2):", img2.shape)
    print("align and overlay (resized mask2):", mask2.shape)

    # rotate & expand img1, and paste to the center of img2
    final_res, final_img1, final_img2, final_mask1, final_mask2, iou = crop_and_paste(
        resized_img1, resized_mask1, img2, mask2, angle, x, y, ratio
    )

    print(f'Merge img1, img2! Mask IoU: {iou}')

    return final_res, final_img1, final_mask1, final_mask2


def expand_roi(
    box: List[int], ratio: float, max_box: List[int], eps: int = 0
) -> List[int]:
    """
    Expand a region of interest (ROI) box by a given ratio while ensuring it doesn't exceed the specified maximum box.

    Args:
        box (List[int]): The original ROI box coordinates [left, upper, right, lower].
        ratio (float): The expansion ratio applied to the ROI box.
        max_box (List[int]): The maximum bounding box within which the ROI must be contained [left, upper, right, lower].
        eps (int, optional): An epsilon value to add to the width and height during expansion (default is 0).

    Returns:
        List[int]: The expanded ROI box coordinates [left, upper, right, lower].
    """
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


def crop_image(
    img: np.ndarray, box: Tuple[int, int, int, int], expand_ratio: float = 1.0
) -> np.ndarray:
    """
    Crop an image by box and expand by expand_ratio.

    Args:
        img (np.ndarray): Input image (image read using cv2).
        box (Tuple[int, int, int, int]): Coordinates of the cropping box (left, upper, right, lower).
        expand_ratio (float, optional): A ratio to expand the cropping box (default is 1.0).

    Returns:
        np.ndarray: Cropped image.
    """
    W, H = img.shape[1], img.shape[0]
    x1, y1, x2, y2 = expand_roi(box, ratio=expand_ratio, max_box=[0, 0, W, H])

    cropped_img = img[y1:y2, x1:x2]

    return cropped_img


def calculate_average_distance(original_points: np.ndarray, new_points: np.ndarray) -> float:
    """
    Calculate the average movement distance of K points.

    Args:
        original_points (np.ndarray): An array of original points with shape (K, 2).
        new_points (np.ndarray): An array of new points with shape (K, 2).

    Returns:
        float: The average distance between corresponding points in the two arrays.
    """
    if original_points.shape != new_points.shape:
        print("Shapes of original_points and new_points must match.")
        return None

    distances = np.linalg.norm(new_points - original_points, axis=1)
    average_distance = np.mean(distances)

    return average_distance


def expand_polygon_vertex(A: np.ndarray, n: int) -> np.ndarray:
    """
    Expand a polygon A by adding vertices to create a polygon with n vertices.

    Args:
        A (np.ndarray): The input polygon as a NumPy array, where each row represents a vertex.
        n (int): The desired number of vertices in the expanded polygon.

    Returns:
        np.ndarray: The expanded polygon as a NumPy array with n vertices.
    """
    # Calculate the number of edges in polygon A
    num_edges = len(A)

    # Calculate the number of points to sample on each edge
    edge_lengths = [
        np.linalg.norm(A[(i + 1) % num_edges] - A[i]) for i in range(num_edges)
    ]
    total_length = sum(edge_lengths)

    # Calculate the number of points on each edge, rounding to the nearest integer
    points_per_edge = [int(round(n * length / total_length))
                       for length in edge_lengths]

    # Calculate the remaining points
    remaining_points = n - sum(points_per_edge)

    # Distribute the remaining points to the longest edge
    max_edge_index = points_per_edge.index(max(points_per_edge))
    points_per_edge[max_edge_index] += remaining_points

    # Create a new list of vertices to store the expanded polygon
    expanded_A = []

    for i in range(num_edges):
        # Get the start and end points of the current edge
        start_point = A[i]
        end_point = A[(i + 1) % num_edges]

        # Calculate the step size along the current edge
        num_points_on_edge = points_per_edge[i]
        if num_points_on_edge != 0:
            step = 1.0 / num_points_on_edge

            # Generate evenly spaced points along the edge, including the start but not the end
            for j in range(num_points_on_edge):
                t = j * step
                interpolated_point = (1 - t) * start_point + t * end_point
                expanded_A.append(interpolated_point)

    # Convert the expanded vertices to a NumPy array
    expanded_A = np.array(expanded_A)
    return expanded_A


def adjust_B_to_match_A(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Adjust the vertices of polygon B to match the orientation of polygon A by finding the closest point in B to the first vertex of A.

    Args:
        A (np.ndarray): The reference polygon A as a NumPy array.
        B (np.ndarray): The polygon B to be adjusted as a NumPy array.

    Returns:
        np.ndarray: The adjusted polygon B with vertices rearranged to match the orientation of polygon A.
    """
    # Find the index of the point in polygon B that is closest to the first vertex of polygon A
    distances = np.linalg.norm(A[0] - B, axis=1)
    nearest_point_index = np.argmin(distances)

    # Rearrange the vertices of polygon B to match the orientation of polygon A with the first vertex aligned
    adjusted_B = np.roll(B, -nearest_point_index, axis=0)

    return adjusted_B


def draw_vertex_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    name: str,
    line_color: tuple = (0, 0, 255),
    font_color: tuple = (0, 0, 255)
):
    """
    Draw a polygon on an image and label its vertices with numbers.

    Args:
        image (np.ndarray): The target image on which to draw the polygon.
        polygon (np.ndarray): The polygon vertices as a NumPy array.
        name (str): The name for saving the output image.
        line_color (tuple, optional): The color for drawing the polygon lines. Default is (0, 0, 255) (red).
        font_color (tuple, optional): The color for labeling vertex numbers. Default is (0, 0, 255) (red).

    Returns:
        None
    """
    plot_image = copy.deepcopy(image)
    polygon = np.array(polygon, dtype=np.int32)
    cv2.polylines(plot_image, [polygon], isClosed=True,
                  color=line_color, thickness=2)

    # Label vertex numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (x, y) in enumerate(polygon):
        cv2.putText(plot_image, str(i), (x, y), font, 0.5, font_color, 2)

    cv2.imwrite(f"{name}.jpg", plot_image)


def mask_to_box(mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Get the largest connected component and return its bounding box.

    Args:
        mask (np.ndarray): The input binary mask.

    Returns:
        Tuple[np.ndarray, Tuple[int, int, int, int]]:
            - Largest connected component binary mask.
            - Bounding box coordinates (x, y, x + width, y + height).
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # Find the largest connected component
    largest_label = (
        np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    )  # Ignore the background label

    # Create a new mask with only the largest connected component
    largest_connected_component_mask = np.zeros_like(mask)
    largest_connected_component_mask[labels == largest_label] = 255

    # Find the bounding box for the largest connected component
    x, y, width, height = cv2.boundingRect(largest_connected_component_mask)

    # Set all regions outside of the bounding box to black
    largest_connected_component_mask[:y, :] = 0
    largest_connected_component_mask[y + height:, :] = 0
    largest_connected_component_mask[:, :x] = 0
    largest_connected_component_mask[:, x + width:] = 0

    return largest_connected_component_mask, (x, y, x + width, y + height)


def draw_box_on_image(
    image: np.ndarray,
    box: tuple,
    det_path: str
):
    """
    Draw a bounding box on an image and save the result to a file.

    Args:
        image (np.ndarray): The input image.
        box (tuple): The bounding box coordinates in the format (x1, y1, x2, y2).
        det_path (str): The path to save the resulting image with the bounding box.

    Returns:
        None
    """
    plot_image = image.copy()
    height, width, _ = image.shape

    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Draw the bounding box on the image
    cv2.rectangle(plot_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the image with the bounding box to the specified path
    cv2.imwrite(det_path, plot_image)


def seg_by_box(
    raw_image_rgb: np.ndarray,
    boxes_filt: np.ndarray,
    segmentor,
    point_coords: np.ndarray = None
) -> np.ndarray:
    """
    Segment regions within specified boxes in an RGB image.

    Args:
        raw_image_rgb (np.ndarray): The input RGB image.
        boxes_filt (np.ndarray): An array of bounding boxes in the format [x1, y1, x2, y2].
        segmentor: The segmentor object used for segmentation.
        point_coords (np.ndarray): Point coordinates (default: None).

    Returns:
        np.ndarray: The segmented mask image with regions within the specified boxes.
    """
    h, w = raw_image_rgb.shape[:2]

    # Set the image for the segmentor
    segmentor.set_image(raw_image_rgb)

    # Transform the bounding boxes to match the image dimensions
    transformed_boxes = segmentor.transform.apply_boxes_torch(
        torch.from_numpy(np.expand_dims(boxes_filt, 0)), (h, w)
    ).cuda()

    # Predict masks, scores, etc.
    masks, scores, _ = segmentor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )

    # Sum the multi-mask outputs to obtain a single mask
    mask = masks[0]
    mask = torch.sum(mask, dim=0)
    mask = mask.cpu().numpy().astype(np.uint8)
    mask_image = mask * 255

    return mask_image


def apply_mask_to_image(img_foreground: np.ndarray, img_background: np.ndarray, mask: np.ndarray, mask_blur: int = 5, expand_kernal=5) -> np.ndarray:
    """
    Apply a mask to an image to keep pixels where the mask is 255 and set other areas to white.

    Args:
        img_foreground (np.ndarray): The foreground image.
        img_background (np.ndarray): The background image.
        mask (np.ndarray): mask wthite use foreground, the other use background
        mask_blur (int): kernal size of mask_blur

    Returns:
        np.ndarray: The processed image with the specified pixels retained and other areas set to white.
    """
    # # Create a white background
    # white_background = np.full_like(img, 255, dtype=np.uint8)

    # # Use the mask to retain pixels in the image where the mask is 255
    # result = cv2.bitwise_and(img, img, mask=mask)

    # # Set areas not retained by the mask to white
    # result[np.where(mask == 0)] = [255, 255, 255]
    mask = cv2.dilate(np.array(mask), np.ones(
            (expand_kernal, expand_kernal), np.uint8), iterations=1)
    mask_blur = cv2.GaussianBlur(
        np.array(np.uint8(mask)), (mask_blur, mask_blur), 0
    )
    mask_blur = np.stack((mask_blur,) * 3, axis=-1)
    result = np.array(img_foreground, np.uint8)*(mask_blur/255.) + np.array(img_background, np.uint8)*((255-mask_blur)/255.)

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
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode="edge")

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
    x = x.astype(np.uint8)
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


def fill_mask(mask: np.ndarray) -> np.ndarray:
    """
    Fill in the connected components of the input binary mask. (remove logo)

    Args:
        mask (np.ndarray): The binary mask with connected components to be filled.

    Returns:
        np.ndarray: The filled binary mask with the connected components completely filled.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled_mask = np.zeros_like(mask)

    for contour in contours:
        cv2.drawContours(filled_mask, [contour], 0, (255), -1)

    return filled_mask


def remove_outline(img: np.ndarray, outline_mask: np.ndarray) -> np.ndarray:
    """
    Remove the outlined region from an image based on an outline mask.

    Args:
        img (np.ndarray): The input image as a NumPy array.
        outline_mask (np.ndarray): The binary mask that defines the outline region to be removed.

    Returns:
        np.ndarray: The resulting image with the outlined region removed.
    """
    inverted_outline = np.logical_not(outline_mask)

    filtered_img = np.multiply(img, inverted_outline)

    return filtered_img


def merge_with_inner_canny(image: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Merge an image with inner Canny edges by applying Canny edge detection to the image and masks, and then removing
    the outline of mask1 to obtain inner edges.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        mask1 (np.ndarray): The first binary mask.
        mask2 (np.ndarray): The second binary mask.

    Returns:
        tuple: A tuple containing two images - the first is the resized image, and the second is the Canny edges image with inner edges.
    """
    canny_image, resize_image = canny(image)

    _, resize_mask1 = canny(mask1)
    _, resize_mask2 = canny(mask2)

    # Create a mask outline using morphological operations
    # mask1_outline = np.uint8(
    #     cv2.dilate(np.array(resize_mask1), np.ones(
    #         (10, 10), np.uint8), iterations=1)
    #     - cv2.erode(np.array(resize_mask1),
    #                 np.ones((5, 5), np.uint8), iterations=1)
    # )

    # erode_kernal = int(0.1* min(resize_mask1.shape[0], resize_mask1.shape[1]))
    mask1_outline = np.uint8(
            cv2.dilate(np.array(resize_mask1), np.ones(
                (10, 10), np.uint8), iterations=1)
            - cv2.erode(np.array(resize_mask1),
                        np.ones((10, 10), np.uint8), iterations=1)
        )

    mask1_outline = cv2.cvtColor(np.uint8(mask1_outline), cv2.COLOR_BGR2GRAY)

    cv2.imwrite('mask1_outline.jpg', mask1_outline)

    # Remove the mask1 outline from the Canny image to obtain inner edges
    canny_image_inner = remove_outline(canny_image, mask1_outline)

    return resize_image, canny_image_inner


def copy_white_mask_to_template(img: np.ndarray, mask: np.ndarray, template: np.ndarray, box: list) -> np.ndarray:
    """
    Copy a masked region from an image to a template.

    Args:
        img (np.ndarray): The source image from which to copy the masked region.
        mask (np.ndarray): The binary mask defining the region to be copied.
        template (np.ndarray): The template image where the region will be pasted.
        box (list): The bounding box coordinates in the format [left, upper, right, lower].

    Returns:
        np.ndarray: The resulting image with the masked region copied to the template.
    """
    h, w, _ = template.shape
    # expand_mask = np.zeros((h, w))

    # Expand the mask to match the specified bounding box
    # expand_mask[box[1]:box[3], box[0]:box[2]] = mask

    # result = np.zeros_like(template)
    result = template
    template_crop = template[box[1]:box[3], box[0]:box[2]]
    # result[box[1]:box[3], box[0]:box[2]] = np.array(img, np.uint8)*(mask/255.) + np.array(template_crop, np.uint8)*((255-mask)/255.)
    result[box[1]:box[3], box[0]:box[2]] = apply_mask_to_image(img, template_crop, mask)

    # Copy the region from the template to the result where the mask is 0
    # result[expand_mask == 0] = template[expand_mask == 0]

    return result


def wrap_image_by_vertex(img, polygon1, polygon2):
    # TODO traditional way to translate img of polygon1 to polygon2
    return img


def compute_color_similarity(color1, color2):
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    distance = ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
    return distance


def get_no_white(colors, threshold=10):
    for color in colors:
        dis = compute_color_similarity(color,[255,255,255])
        if dis>threshold:
            return color
    return colors[0]

def get_background_color(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Get the background color of the image within the masked region.

    Args:
        img (np.ndarray): The input image as a NumPy array.
        mask (np.ndarray): The binary mask that defines the region of interest.

    Returns:
        np.ndarray: The most frequent color within the masked region.
    """
    # Get the coordinates of the mask area
    mask_area = np.where(mask > 0)

    # Calculate the pixel value distribution within the masked region
    img_mask_area = img[mask_area]
    unique_values, counts = np.unique(
        img_mask_area, axis=0, return_counts=True)

    # print(np.array(counts).max(), np.array(counts).sum())
    # print(np.array(counts).max()/ np.array(counts).sum())
    # Find the most frequent pixel value
    # most_frequent_index = np.argmax(counts)
    # most_frequent_value = unique_values[most_frequent_index]

    sorted_indices = np.argsort(-counts)
    sorted_colors = unique_values[sorted_indices]
    # sorted_counts = counts[sorted_indices]

    TopK = min(len(sorted_colors), 10)
    sorted_colors = sorted_colors[:TopK]
    most_frequent_value = get_no_white(sorted_colors)

    return most_frequent_value


def compute_rotation_angle(polygon: List[List[int]]) -> float:
    """
    Compute the rotation angle of the minimum bounding box that fits around a given polygon.

    Args:
        polygon (List[List[int]]): List of polygon vertices represented as coordinate pairs.

    Returns:
        float: The rotation angle in degrees.
    """
    # Create a copy of the polygon in the format expected by OpenCV
    points = np.array(polygon, dtype=np.int32)

    # Fit a rotated rectangle around the polygon
    rect = cv2.minAreaRect(points)

    # Get the rotation angle of the minimum bounding box
    rotation_angle = rect[2]

    return rotation_angle


def find_best_angle_ratio(
    polygon1: List[List[int]],
    polygon2: List[List[int]],
    initial_parameters: Tuple[float, float],
    x: float,
    y: float,
    angle_target: float,
    max_iters: int = 100,
    iou_threshold: float = 0.7,
) -> Tuple[float, float]:
    """
    Find the best angle and scaling ratio to maximize IoU (Intersection over Union) while satisfying a constraint.

    Args:
        polygon1 (List[List[int]]): List of vertices of the first polygon.
        polygon2 (List[List[int]]): List of vertices of the second polygon.
        initial_parameters (Tuple[float, float]): Initial parameters for optimization (angle, ratio).
        x (float): X-coordinate of the center point for transformation.
        y (float): Y-coordinate of the center point for transformation.
        angle_target (float): Target angle for optimization.
        max_iters (int, optional): Maximum number of iterations for optimization (default is 100).
        iou_threshold (float, optional): IoU threshold for the constraint (default is 0.7).

    Returns:
        Tuple[float, float]: The optimal angle and scaling ratio.
    """

    # Define the optimization target function
    def target_function(parameters: Tuple[float, float]) -> float:
        iou, in_iou = align_and_compute_iou(
            polygon1, polygon2, x, y, parameters
        )
        return - iou - 0.01* in_iou + 0.1 * angle_loss(angle_target, parameters)

    # Define the constraint function
    def constraint_function(parameters: Tuple[float, float]) -> float:
        iou,_ = align_and_compute_iou(polygon1, polygon2, x, y, parameters)
        return iou - iou_threshold

    # Define the angle loss function
    def angle_loss(angle_target: float, parameters: Tuple[float, float]) -> float:
        loss = (angle_target - parameters[0]) ** 2
        return loss

    # Define the transformation and IoU calculation function
    def align_and_compute_iou(
        polygon1: List[List[int]],
        polygon2: List[List[int]],
        x: float,
        y: float,
        parameters: Tuple[float, float],
    ) -> float:
        angle, ratio = parameters

        # Create a rotation matrix
        rotation_matrix = np.array(
            [
                [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                [np.sin(np.radians(angle)), np.cos(np.radians(angle))],
            ]
        )

        # Translate the polygon to the origin
        translated_polygon = polygon1 - [x, y]

        # Rotate the translated polygon
        rotated_polygon = np.dot(translated_polygon, rotation_matrix.T)

        # Scale the rotated polygon
        scaled_polygon = rotated_polygon * ratio

        # Translate the scaled polygon back to its original position
        polygon1_transformed = scaled_polygon + [x, y]

        # Create Shapely polygon objects
        poly1 = Polygon(polygon1_transformed)
        poly2 = Polygon(polygon2)

        # Calculate IoU
        try:
            iou = poly1.intersection(poly2).area / poly1.union(poly2).area
            in_iou = poly1.intersection(poly2).area / poly2.area
        except:
            if not poly1.is_valid:
                poly1 = poly1.buffer(0)
            if not poly2.is_valid:
                poly2 = poly2.buffer(0)

            iou = poly1.intersection(poly2).area / poly1.union(poly2).area
            in_iou = poly1.intersection(poly2).area / poly2.area

        print(f'iou: {iou}, in_iou: {in_iou}, angle:{angle}, ratio:{ratio}')

        return iou, in_iou

    # Define the constraint
    constraint = {"type": "ineq", "fun": constraint_function}

    # Define optimization options, including learning rate
    options = {"disp": True, "maxiter": max_iters, "ftol": 1e-6, "eps": 1e-5}

    # Define parameter bounds
    bounds = [(angle_target - 10, angle_target + 10), (0.1, 3.0)]

    # Perform the optimization to maximize IoU while satisfying the constraint
    result = minimize(
        target_function,
        initial_parameters,
        method="SLSQP",
        constraints=constraint,
        options=options,
        bounds=bounds,
    )

    # Get the optimal parameters and maximum IoU value
    optimal_parameters = result.x
    max_iou = -result.fun

    return optimal_parameters[0], optimal_parameters[1]


def expand_box_by_pad(box, max_size, padding_size):
    """
    Expand a bounding box by adding padding.

    Args:
        box (list): The bounding box to be expanded in the format [left, upper, right, lower].
        max_size (tuple): The maximum size of the bounding box in the format (max_width, max_height).
        padding_size (int): The size of the padding to add to each side of the bounding box.

    Returns:
        list: The expanded bounding box in the format [left, upper, right, lower].
    """
    expanded_box = [
        max(0, box[0] - padding_size),
        max(0, box[1] - padding_size),
        min(max_size[0], box[2] + padding_size),
        min(max_size[1], box[3] + padding_size),
    ]
    return expanded_box


def resize_to_512(image): 
    short_side  = min(image.shape[0], image.shape[1])
    resize      = float(short_side / 512.0)
    new_size    = (int(image.shape[1] // resize // 32 * 32), int(image.shape[0] // resize // 32 * 32))
    result_img  = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)

    return result_img