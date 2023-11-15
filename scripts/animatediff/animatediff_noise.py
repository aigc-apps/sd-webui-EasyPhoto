import torch
import torch.nn.functional as F
import numpy as np

def gaussian_kernel(size, sigma):
    x = np.linspace(-size, size, 2 * size + 1)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return torch.tensor(kernel, dtype=torch.float32)

def split_pad_and_smooth_tensor_gaussian(tensor, b, window_size=1, sigma=1):
    k, c, h, w = tensor.shape

    # Pad the tensor if k is not divisible by b
    padding_needed = k % b
    if padding_needed != 0:
        padding_amount = b - padding_needed
        padding_values = tensor[:padding_amount]
        tensor = torch.cat((tensor, padding_values), dim=0)

    # Split the tensor
    split_tensors = torch.split(tensor, split_size_or_sections=b, dim=0)

    # Create a Gaussian kernel for smoothing
    kernel_size = 2 * window_size + 1
    gauss_kernel = gaussian_kernel(window_size, sigma)
    gauss_kernel = gauss_kernel.view(1, 1, kernel_size).repeat(1, c, 1)

    # Apply smoothing at the boundaries of each split
    for i in range(len(split_tensors)):
        if i > 0:
            start_smoothing = F.conv1d(split_tensors[i-1][-window_size:].unsqueeze(0), gauss_kernel, padding=0)
            split_tensors[i][0:window_size] = start_smoothing[0]

        if i < len(split_tensors) - 1:
            end_smoothing = F.conv1d(split_tensors[i+1][0:window_size].unsqueeze(0), gauss_kernel, padding=0)
            split_tensors[i][-window_size:] = end_smoothing[0]

    # Concatenate the split tensors back together
    smoothed_tensor = torch.cat(split_tensors, dim=0)

    # Trim off the padding if it was added
    if padding_needed != 0:
        smoothed_tensor = smoothed_tensor[:-(padding_amount)]

    return smoothed_tensor

def split_pad_and_smooth_tensor(tensor, b, window_size=1):
    """
    This function takes a tensor with shape [k, c, h, w] and a split size b.
    It splits the tensor along the k dimension, pads it if k is not divisible by b,
    and applies smoothing at the boundaries of each split.

    :param tensor: A PyTorch tensor with shape [k, c, h, w]
    :param b: The size of each split along the k dimension
    :param window_size: The size of the smoothing window
    :return: A tensor with the same shape [k, c, h, w], with smoothing applied
    """
    k, c, h, w = tensor.shape

    # Pad the tensor if k is not divisible by b
    padding_needed = k % b
    if padding_needed != 0:
        padding_amount = b - padding_needed
        padding_values = tensor[:padding_amount]
        tensor = torch.cat((tensor, padding_values), dim=0)

    # Split the tensor
    split_tensors = torch.split(tensor, split_size_or_sections=b, dim=0)

    # Apply smoothing at the boundaries of each split
    for i in range(len(split_tensors)):
        if i > 0:
            start_smoothing = (split_tensors[i][0:window_size] + split_tensors[i-1][-window_size:]) / 2
            split_tensors[i][0:window_size] = start_smoothing

        if i < len(split_tensors) - 1:
            end_smoothing = (split_tensors[i][-window_size:] + split_tensors[i+1][0:window_size]) / 2
            split_tensors[i][-window_size:] = end_smoothing

    # Concatenate the split tensors back together
    smoothed_tensor = torch.cat(split_tensors, dim=0)

    # Trim off the padding if it was added
    if padding_needed != 0:
        smoothed_tensor = smoothed_tensor[:-(padding_amount)]

    return smoothed_tensor
