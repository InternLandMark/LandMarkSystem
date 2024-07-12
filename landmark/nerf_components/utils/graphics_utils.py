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

import math
from typing import NamedTuple

import numpy as np
import torch


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def pose_rotate_translate(pose, rot_x, rot_y, rot_z, trans):
    R_z = np.array(
        [
            [math.cos(np.deg2rad(rot_z)), -math.sin(np.deg2rad(rot_z)), 0, 0],
            [math.sin(np.deg2rad(rot_z)), math.cos(np.deg2rad(rot_z)), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    R_y = np.array(
        [
            [math.cos(np.deg2rad(rot_y)), 0, math.sin(np.deg2rad(rot_y)), 0],
            [0, 1, 0, 0],
            [-math.sin(np.deg2rad(rot_y)), 0, math.cos(np.deg2rad(rot_y)), 0],
            [0, 0, 0, 1],
        ]
    )

    R_x = np.array(
        [
            [1, 0, 0, 0],
            [0, math.cos(np.deg2rad(rot_x)), -math.sin(np.deg2rad(rot_x)), 0],
            [0, math.sin(np.deg2rad(rot_x)), math.cos(np.deg2rad(rot_x)), 0],
            [0, 0, 0, 1],
        ]
    )

    rotation_matrix = R_z @ R_y @ R_x

    c2w = np.linalg.inv(pose)
    new_c2w = rotation_matrix @ c2w
    new_c2w[:3, 3] = new_c2w[:3, 3] + trans
    new_pose = np.linalg.inv(new_c2w)

    return np.float32(new_pose)