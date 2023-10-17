import cv2
import numpy as np
from PIL import Image
import math
# from easyphoto_process_utils import mask_to_box

# def replace_white_regions_with_most_frequent(img1, img2, mask):
#     """
#     Replace white regions in img1 with the most frequent color in img2.

#     Args:
#         img1 (numpy.ndarray): The first image.
#         img2 (numpy.ndarray): The second image from which the color will be taken.
#         mask (numpy.ndarray): A mask specifying the regions to be replaced.

#     Returns:
#         numpy.ndarray: The resulting image.
#     """
#     def most_frequent_color(image, mask=None):
#         if mask is not None:
#             masked_image = cv2.bitwise_and(image, image, mask=mask)
#         else:
#             masked_image = image

#         pixel_values, counts = np.unique(masked_image.reshape(-1, masked_image.shape[2]), axis=0, return_counts=True)
#         most_frequent_pixel = pixel_values[np.argmax(counts)]

#         return most_frequent_pixel

#     most_frequent_pixel = most_frequent_color(img2, mask)

#     white_pixels = (mask == 255)
#     img1[white_pixels] = most_frequent_pixel

#     return img1

# img1 = cv2.imread('/mnt/xinyi.zxy/easyphoto/sdwebui/workspace/result_img11.jpg')
# img2=cv2.imread('/mnt/xinyi.zxy/easyphoto/sdwebui/workspace/expand_img11.jpg')
# mask = cv2.imread('/mnt/xinyi.zxy/easyphoto/sdwebui/workspace/expand_region_mask.jpg')[:,:,0]

# res = replace_white_regions_with_most_frequent(img1,img2,mask)
# cv2.imwrite('res.jpg',res)

if 0:
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



    box_template = [67, 308, 404, 586]
    template_copy = cv2.imread('/mnt/xinyi.zxy/easyphoto/template_img.jpg')
    result_img = Image.open('/mnt/xinyi.zxy/easyphoto/sdwebui/workspace/inpaint_res_resize.jpg')
    resize_mask2 = Image.open('/mnt/xinyi.zxy/easyphoto/sdwebui/workspace/inpaint_res_mask.jpg')

    template_copy = copy_white_mask_to_template(np.array(result_img),np.array(np.uint8(resize_mask2))[:,:,0], template_copy, box_template)


    # template_copy[box_template[1]:box_template[3], box_template[0]:box_template[2]][ = np.array(result_img, np.uint8)
    cv2.imwrite('gen.jpg',template_copy)

if 0:
    # rotate result with white background
    def rotate_resize_image(array, angle=0.0, scale_ratio=1.0):
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

        # Create a white background with the same size
        white_bg = np.zeros_like(rotated_mat)
        white_bg.fill(255)  # Set the background to white

        # Combine the rotated image and the white background using a mask
        mask = (rotated_mat == 0)
        result = np.where(mask, white_bg, rotated_mat)

        # Scale
        scaled_mat = cv2.resize(result, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)

        return scaled_mat

    img = cv2.imread('/mnt/xinyi.zxy/easyphoto/sdwebui/workspace/outputs/easyphoto-user-id-infos/tshirt/original_backup/1.jpg')
    rotate_img = rotate_resize_image(img,60)
    cv2.imwrite('rotate.jpg',rotate_img)

if 0:
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

    # remove useless boundary after sam
    mask = cv2.imread('/mnt/xinyi.zxy/easyphoto/sdwebui/workspace/outputs/easyphoto-user-id-infos/t1/ref_image_mask.jpg')[:,:,0]
    mask_clean,box = mask_to_box(mask)
    cv2.imwrite('mask_clean.jpg',mask_clean)

if 1:
    img = cv2.imread('/mnt/xinyi.zxy/easyphoto/sdwebui/workspace/after_canny_res_image_ori.jpg')
    cv2.imwrite('/mnt/xinyi.zxy/easyphoto/sdwebui/workspace/after_canny_res_image_ori.jpg',img[:,:,::-1])