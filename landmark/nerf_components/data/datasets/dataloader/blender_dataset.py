import json
import os
from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import tqdm

from .base_dataset import VolumeRenderDataset
from .utils.ray_utils import (
    get_ray_directions_blender,
    get_rays_with_directions,
    read_Image,
)


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


class BlenderDataset(VolumeRenderDataset):
    """
    A dataset class for handling Blender datasets in volume rendering.

    This class extends `VolumeRenderDataset` to support operations specific to Blender datasets,
    such as reading metadata, processing images, and handling camera parameters for 3D rendering.

    Attributes:
        Inherits all attributes from `VolumeRenderDataset`.

    Methods:
        read_meta(): Reads and processes the Blender dataset metadata.
    """

    def __init__(self, split="train", downsample=10, is_stack=False, enable_lpips=False, args=None, preprocess=False):
        """
        Initializes the BlenderDataset with configuration parameters.

        Parameters:
            split (str): The dataset split, e.g., 'train', 'test'.
            downsample (float): The factor by which images are downsampled.
            is_stack (bool): Whether to stack rays tensors.
            enable_lpips (bool): Whether LPIPS loss is enabled.
            args (Namespace): Configuration arguments for the dataset.
            preprocess (bool): Whether to preprocess the dataset.
        """
        super().__init__(split, downsample, is_stack, enable_lpips, args)
        if preprocess:
            pass
        else:
            if self.split == "path":
                self.read_meta_path()
            else:
                self.read_meta()

    def read_meta(self):

        """
        Read blender dataset from rootdir.
        """
        meta = load_from_json(Path(os.path.join(self.datadir, f"transforms_{self.split}.json")))
        # meta = load_from_json(Path(os.path.join(self.datadir, f"transforms_train.json")))
        image_filenames = []
        poses = []

        for frame in meta["frames"]:
            fname = Path(self.datadir) / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.v2.imread(image_filenames[0])
        H, W = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * W / np.tan(0.5 * camera_angle_x)

        # in x,y,z order
        # camera_to_world[..., 3] *= self.scale_factor

        N = len(poses)
        idxs = list(range(0, N))
        self.img_wh = [W, H]
        directions = get_ray_directions_blender(int(H), int(W), (focal_length, focal_length))

        if self.lpips:
            ps = self.patch_size
            coords = torch.stack(
                torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                -1,
            )
            patches = (
                coords.unfold(0, size=ps, step=ps).unfold(1, size=ps, step=ps).unfold(2, 2, 2)
            )  # slice image into patches
            patches = torch.reshape(patches, [-1, ps, ps, 2])

        for i in tqdm(idxs):
            pose = torch.FloatTensor(poses[i])
            f_path = os.path.join(image_filenames[i])
            self.poses.append(pose)
            self.image_paths.append(f_path)

            img = read_Image(f_path, self.transform)
            rays_o, rays_d = get_rays_with_directions(directions, pose)

            if self.lpips and self.split == "train":
                for coord in patches:
                    coord = torch.reshape(coord, [-1, 2])
                    inds = (coord[:, 0] * W + coord[:, 1]).long()
                    self.all_rgbs += [img[inds]]
                    self.all_rays += [torch.cat([rays_o, rays_d], 1)[inds]]
                    self.all_idxs += [(torch.zeros(rays_d.shape[0]) + i)[inds]]
            else:
                self.all_rgbs += [img]
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]
                self.all_idxs += [(torch.zeros(rays_d.shape[0]) + i)]

        self.poses = torch.stack(self.poses, 0)
        print(
            f"{self.split} poses bds",
            self.poses[:, :3, -1].min(0)[0],
            self.poses[:, :3, -1].max(0)[0],
        )

        self.stack_rays()
