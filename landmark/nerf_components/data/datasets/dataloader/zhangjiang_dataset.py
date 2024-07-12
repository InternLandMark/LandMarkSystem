import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from landmark.nerf_components.data.cameras import Camera
from landmark.nerf_components.utils.graphics_utils import (
    getProjectionMatrix,
    getWorld2View2,
)

from .city_dataset import CityGaussianDataset
from .utils.utils import focal2fov, fov2focal

# SCALE = 1
# SCALE = 0.5
# SCALE = 100
SCALE = 1000


class ZhangjiangGaussianDataset(CityGaussianDataset):
    """Class for Zhangjiang Dataset"""

    def read_meta_file(self):
        if self.split in ["train", "test"]:
            self.meta_json = Path(self.datadir) / f"transforms_{self.split}_3dgs_ds10.json"
        elif self.split in ["path"]:
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown split {self.split}")

        assert os.path.exists(self.meta_json), f"Meta file {self.meta_json} does not exist!"

    def read_meta(self):
        with open(self.meta_json, "r", encoding="utf-8") as f:
            contents = json.load(f)
            try:
                fovx = contents["camera_angle_x"]
            except FileNotFoundError as e:
                print(f"Error loading PLY file: {e}")
                fovx = None

            FovX, FovY = None, None
            img_width, img_height = None, None

            frames = contents["frames"]
            # check if filename already contain postfix
            if frames[0]["file_path"].split(".")[-1] in ["jpg", "jpeg", "JPG", "png"]:
                extension = ""

            for idx, frame in enumerate(tqdm(frames, desc=f"Reading {self.split} dataset metadata")):
                image_path = os.path.join(self.args.image_folder, os.path.basename(frame["file_path"]) + extension)
                # image_path = os.path.join(self.datadir, frame["file_path"] + extension)
                assert os.path.exists(image_path), f"Image {image_path} does not exist!"
                image_name = Path(image_path).stem

                c2w = np.array(frame["transform_matrix"])

                if not (c2w[:2, 3].max() < SCALE and c2w[:2, 3].min() > -SCALE):
                    continue

                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                if not (img_width and img_height):
                    img = Image.open(image_path)
                    img_width, img_height = img.size[0], img.size[1]

                if fovx is not None:
                    fovy = focal2fov(fov2focal(fovx, img_width), img_height)
                    FovY = fovy
                    FovX = fovx
                else:
                    # given focal in pixel unit
                    FovY = focal2fov(frame["fl_y"], img_height)
                    FovX = focal2fov(frame["fl_x"], img_width)

                trans = np.array([0.0, 0.0, 0.0])
                scale = 1.0

                zfar = 100.0
                znear = 0.01

                world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
                projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0, 1)
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]

                self.cams.append(
                    Camera(
                        uid=idx,
                        image_name=image_name,
                        image_path=image_path,
                        width=img_width,
                        height=img_height,
                        c2w=torch.tensor(c2w),
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
                    )
                )
