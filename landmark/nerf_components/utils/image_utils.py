#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import cv2
import numpy as np
import torch


@torch.no_grad()
def psnr(img1, img2):
    loss = torch.mean((img1 - img2) ** 2)
    value = -10.0 * np.log(loss.item()) / np.log(10.0)
    return value


@torch.no_grad()
def mse2psnr_npy(x):
    return -10.0 * np.log(x) / np.log(10.0)


@torch.no_grad()
def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    Visualize depth map.

    Args:
        depth (numpy.ndarray): The depth map.
        minmax (list): The min and max depth.

    Returns:
        numpy.ndarray: The visualized depth map.
        list: The min and max depth.
    """
    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = np.clip(x, a_min=0, a_max=1)
    x = x / 1.1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]
