import os

import laspy
import numpy as np
from colorama import Fore, Style
from plyfile import PlyData, PlyElement
from typing_extensions import Literal

from landmark.nerf_components.data.datasets.dataloader.utils.colmap_loader import (
    read_points3D_binary,
    read_points3D_text,
)
from landmark.nerf_components.utils.graphics_utils import BasicPointCloud
from landmark.nerf_components.utils.sh_utils import SH2RGB


def init_ply(folder, aabb, init_type: Literal["ply", "colmap"] = "ply", init_points_num: int = 100000):
    if init_type == "ply":
        ply_path = os.path.join(folder, "points3d_init.ply")
        num_pts = init_points_num
        print(f"Generating random point cloud ({num_pts})...")

        xyz = np.random.random((num_pts, 3))
        xyz[:, 0] = xyz[:, 0] * (aabb[1][0] - aabb[0][0]) + aabb[0][0]
        xyz[:, 1] = xyz[:, 1] * (aabb[1][1] - aabb[0][1]) + aabb[0][1]
        xyz[:, 2] = xyz[:, 2] * 0

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    elif init_type == "colmap":
        ply_path = os.path.join(folder, "sparse/0/points3D.ply")
        bin_path = os.path.join(folder, "sparse/0/points3D.bin")
        txt_path = os.path.join(folder, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except FileNotFoundError as e:
                print(f"Error reading binary file: {e}")
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except FileNotFoundError as e:
            print(f"Error loading PLY file: {e}")
            pcd = None
    else:
        raise ValueError(f"Unknown init type {init_type}")

    return pcd


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T

    try:
        colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    except (KeyError, ValueError, TypeError, IndexError) as e:
        print(f"Error in obtaining colors: {e}")
        print("randomized color")
        colors = np.random.rand(*positions.shape)

    try:
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    except (KeyError, ValueError, TypeError, IndexError) as e:
        print(f"Error in obtaining normals: {e}")
        print("randomized normal")
        normals = np.random.rand(*positions.shape) * 0

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def fetchLas(path):
    las = laspy.read(path)
    positions = np.vstack((las.x, las.y, las.z)).transpose()
    try:
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
    except (KeyError, ValueError, TypeError, IndexError) as e:
        print(f"Error in obtaining colors: {e}")
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.random.rand(positions.shape[0], positions.shape[1])

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def get_pcd(loaded_iter, config, aabb, logfolder, model):
    if loaded_iter:
        pcd_path = os.path.join(config.logfolder, "point_cloud", "iteration_" + str(loaded_iter))
        print(Fore.CYAN + f"Loading from {pcd_path}" + Style.RESET_ALL)
        state_dict = model.get_state_dict_from_ckpt(config.ckpt, config.device)
        model.load_from_state_dict(state_dict)
    else:
        init_type = "colmap" if config.dataset_name == "Colmap" else "ply"
        pcd = init_ply(config.datadir, aabb, init_type, config.init_points_num)
        if config.point_path is not None:
            assert os.path.exists(config.point_path)
            print(f"find point cloud at {config.point_path}, start preprocessing...")
            if os.path.splitext(config.point_path)[1] == ".ply":
                data = fetchPly(config.point_path)
            elif os.path.splitext(config.point_path)[1] == ".las":
                data = fetchLas(config.point_path)
            xyz = data.points
            print(f"original point nums: {xyz.shape[0]}")
            if config.max_init_points > 0:
                if xyz.shape[0] > config.max_init_points:
                    idx = np.random.choice(xyz.shape[0], config.max_init_points, replace=False)
                    xyz = xyz[idx]
            print(f"after sampling for init anchors: {xyz.shape[0]}")
            print("load points before reformat:", xyz.shape[0])
            shs = np.random.random((xyz.shape[0], 3)) / 255.0
            storePly(os.path.join(logfolder, "orig_points3d.ply"), xyz, SH2RGB(shs) * 255)
            print(f"save processed point cloud to {os.path.join(logfolder, 'orig_points3d.ply')}")
            pcd = fetchPly(os.path.join(logfolder, "orig_points3d.ply"))
    return pcd
