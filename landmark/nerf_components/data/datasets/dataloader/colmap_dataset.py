import os

import numpy as np
import torch
from tqdm import tqdm

from landmark.nerf_components.data.cameras import Camera
from landmark.nerf_components.utils.graphics_utils import (
    getProjectionMatrix,
    getWorld2View2,
    pose_rotate_translate,
)

from .base_dataset import GaussianSplattingDataset
from .utils.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from .utils.utils import focal2fov, getNeRFppNorm


class ColmapGaussianDataset(GaussianSplattingDataset):
    """
    A dataset class for handling Colmap 3DGS datasets with Gaussian splatting.

    This class extends `GaussianSplattingDataset` to support operations specific to Colmap 3DGS datasets,
    such as reading metadata, processing images, and handling camera parameters for 3D rendering.

    Attributes:
        llffhold (int): Determines the holdout set size for training and testing.
        img_wh (list): Stores the width and height of images.
        cameras_extent (float): The extent of the camera's field of view.

    Methods:
        read_meta(): Reads and processes the Colmap dataset metadata.
    """

    def __init__(self, split="train", downsample=1, args=None, loaded_cams=None):
        """
        Initializes the ColmapGaussianDataset with configuration parameters.

        Parameters:
            split (str): The dataset split, e.g., 'train', 'test'.
            downsample (float): The factor by which images are downsampled.
            args (Namespace): Configuration arguments for the dataset.
            loaded_cams (list): Preloaded camera parameters.

        Raises:
            ValueError: If `loaded_cams` is None and camera parameters cannot be read from metadata.
        """
        super().__init__(split, downsample, args)

        self.llffhold = 40  # 7 for training, 1 for testing
        self.img_wh = [None, None]
        if loaded_cams:
            self.cams = loaded_cams
        else:
            self.read_meta()
        self.cameras_extent = getNeRFppNorm(self.cams)["radius"]

    def read_meta(self):
        try:
            cameras_extrinsic_file = os.path.join(self.datadir, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.datadir, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except FileNotFoundError as e:
            print(f"Error reading binary file: {e}")
            cameras_extrinsic_file = os.path.join(self.datadir, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.datadir, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        if self.split == "train":
            train_extrinsics = {}
            for idx, key in enumerate(cam_extrinsics):
                if idx % self.llffhold != 0:
                    train_extrinsics[key] = cam_extrinsics[key]
            cam_extrinsics = train_extrinsics

        elif self.split == "test":
            test_extrinsics = {}
            for idx, key in enumerate(cam_extrinsics):
                if idx % self.llffhold == 0:
                    test_extrinsics[key] = cam_extrinsics[key]
            cam_extrinsics = test_extrinsics

        reading_dir = "images" if self.args.images is None else self.args.images
        cams = self.readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(self.datadir, reading_dir),
        )
        cams = sorted(cams, key=lambda x: x.image_name)

        self.cams = cams

    def readColmapCameras(self, cam_extrinsics, cam_intrinsics, images_folder):
        cams = []
        for key in tqdm(cam_extrinsics, desc=f"Reading {self.split} dataset metadata"):

            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width

            uid = intr.id
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            if intr.model == "SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, (
                    "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras)"
                    " supported!"
                )

            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            if not os.path.exists(image_path):
                image_path = image_path.replace(".png", ".JPG")  # fix for loading zhita_5k dataset

            trans = np.array([0.0, 0.0, 0.0])
            scale = 1.0

            zfar = 100.0
            znear = 0.01

            world_view_transform = torch.tensor(
                pose_rotate_translate(
                    getWorld2View2(R, T, trans, scale),
                    rot_x=self.args.pose_rotation[0],
                    rot_y=self.args.pose_rotation[1],
                    rot_z=self.args.pose_rotation[2],
                    trans=self.args.pose_translation,
                )
            ).transpose(0, 1)
            projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            cams.append(
                Camera(
                    uid=uid,
                    image_name=image_name,
                    image_path=image_path,
                    width=width,
                    height=height,
                    c2w=None,
                    FovY=FovY,
                    FovX=FovX,
                    R=torch.tensor(R),
                    T=torch.tensor(T),
                    zfar=zfar,
                    znear=znear,
                    trans=torch.tensor(trans),
                    scale=scale,
                    world_view_transform=world_view_transform,
                    projection_matrix=projection_matrix,
                    full_proj_transform=full_proj_transform,
                    camera_center=camera_center,
                    image=self.load_and_process_image(image_path) if self.args.preload else None,
                )
            )
        return cams
