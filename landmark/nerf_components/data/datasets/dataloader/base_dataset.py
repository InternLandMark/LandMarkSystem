import os
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from .utils.ray_utils import (
    get_ray_directions_blender,
    get_rays,
    get_rays_with_directions,
    load_json_render_path,
    pose_spherical,
)
from .utils.utils import PILtoTorch

WARNED = False


class BaseDataset(Dataset):
    """
    A base class for datasets in Neural Radiance Fields (NeRF) applications.

    This class provides a template for dataset classes used in NeRF, including methods
    for reading metadata, accessing dataset length, and retrieving items by index.

    Attributes:
        args (Namespace): Configuration arguments for the dataset.
        datadir (str): Directory where the dataset is located.
        meta_json (str, optional): Path to a JSON file containing metadata. Defaults to None.
        split (str): The dataset split, e.g., 'train', 'test', or 'path'.
        white_bg (bool): Whether to use a white background in images.
        image_scale (int): The scale factor for downsampling images.

    Methods:
        read_meta(): Abstract method to read dataset metadata.
        __len__(): Abstract method to return the size of the dataset.
        __getitem__(idx): Abstract method to get an item from the dataset by index.
    """

    def __init__(self, split="train", downsample=1.0, args=None):
        """
        Initializes the BaseDataset with configuration parameters.

        Parameters:
            split (str): The dataset split, e.g., 'train', 'test', or 'path'.
            downsample (float): The factor by which images are downsampled.
            args (Namespace): Configuration arguments for the dataset.
        """
        self.args = args
        self.datadir = args.datadir
        self.meta_json = None
        self.split = split  # train, test, path
        self.white_bg = args.white_bkgd
        self.image_scale = int(downsample)

    def read_meta(self):
        """
        Abstract method to read dataset metadata.

        This method should be implemented by subclasses to read and process
        dataset-specific metadata.
        """
        raise NotImplementedError

    def __len__(self):
        """
        Abstract method to return the size of the dataset.

        This method should be implemented by subclasses to return the number
        of items in the dataset.
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Abstract method to get an item from the dataset by index.

        This method should be implemented by subclasses to retrieve a specific
        item from the dataset based on its index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            The item at the specified index.
        """
        raise NotImplementedError


class VolumeRenderDataset(BaseDataset):
    """Class of NeRF Dataset"""

    def __init__(self, split="train", downsample=1.0, is_stack=False, enable_lpips=False, args=None):
        super().__init__(split, downsample, args)
        self.is_stack = is_stack
        self.transform = T.ToTensor()
        self.camera = args.camera

        self.partition = args.partition
        self.debug = args.debug
        if self.debug:
            print("dataset for debug mode")
        self.patch_size = args.patch_size
        self.lpips = enable_lpips

        # render args
        self.render_nframes = args.render_nframes
        self.render_ncircle = args.render_ncircle
        self.render_downward = args.render_downward
        self.render_px = args.render_px
        self.render_fov = args.render_fov
        self.render_focal = 0.5 * self.render_px / np.tan(0.5 * np.deg2rad(self.render_fov))
        self.render_hwf = [
            int(self.render_px),
            int(1920 / 1080 * self.render_px),
            self.render_focal,
        ]

        self.render_spherical = args.render_spherical
        self.render_spherical_zdiff = args.render_spherical_zdiff
        self.render_spherical_radius = args.render_spherical_radius

        self.render_skip = args.render_skip
        self.render_pathid = args.render_pathid

        self.cxyz = None

        # define scene near/far & bbox
        self.near_far = args.render_near_far if self.split == "path" else args.train_near_far
        if args.lb and args.ub:
            self.scene_bbox = torch.tensor([args.lb, args.ub])
            if split == "path":
                self.render_scene_bbox = torch.tensor([args.render_lb, args.render_ub])
        else:
            pass

        # read meta
        self.poses = []
        self.all_rgbs = []
        self.all_rays = []
        self.all_idxs = []
        self.image_paths = []

        self.render_poses = []
        self.render_rays = []

    def read_meta_path(self):
        """
        Read meta dataset splited by path.
        """
        H, W, focal = self.render_hwf
        self.img_wh = [W, H]
        print("path fov", self.render_fov, "hwf", H, W, self.render_focal)
        directions = get_ray_directions_blender(H, W, (focal, focal))

        if self.render_spherical:
            nframes, radius, zdiff = (
                self.render_nframes // self.render_skip,
                self.render_spherical_radius,
                self.render_spherical_zdiff,
            )
            downward = self.render_downward
            angles = np.linspace(0, 360 * self.render_ncircle, nframes + 1)[:-1]
            radiuss = np.linspace(radius, radius, nframes + 1)  # [:-1]
            poses = torch.stack(
                [pose_spherical(angle, downward, radius) for angle, radius in zip(angles, radiuss)],
                0,
            )
            # recenter pose
            if self.cxyz is not None:
                poses[:, 0, 3] += self.cxyz[0]
                poses[:, 1, 3] += self.cxyz[1]
                if self.partition == "sjt":
                    poses[:, 2, 3] += self.cxyz[2] + 2
                else:
                    poses[:, 2, 3] += self.cxyz[2]

            if zdiff > 0:
                zstep = zdiff / nframes * 2
                for i in range(nframes // 2):
                    poses[i, 2, 3] -= zstep * i

                for i in range(nframes // 2, nframes):
                    poses[i, 2, 3] = poses[i, 2, 3] - zdiff + zstep * (i - nframes // 2)

            print("num of poses", len(poses))
        else:
            poses = load_json_render_path(
                pathfolder=os.path.join(self.datadir, "trajectories"),
                posefile=f"path{self.render_pathid}.json",
                render_skip=self.render_skip,
            )

            nframes = len(poses)

        idxs = list(range(len(poses)))
        for i in tqdm(idxs, desc=f"Loading data {self.split} ({len(idxs)})"):  # img_list:#
            pose = torch.FloatTensor(poses[i])
            if self.camera == "normal":
                rays_o, rays_d = get_rays_with_directions(directions, pose)
            else:
                K = torch.Tensor([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
                rays_o, rays_d = get_rays(H, W, K, pose, camera_mode=self.camera)

            self.render_poses.append(pose)
            self.render_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        self.render_poses = torch.stack(self.render_poses)
        self.render_rays = torch.stack(self.render_rays, 0)  # (len(self.meta['frames]),h*w, 3)
        self.all_rays = self.render_rays

    def read_meta(self):
        """
        Abstract method to read dataset metadata.

        This method should be implemented by subclasses to read and process
        dataset-specific metadata.
        """
        raise NotImplementedError

    def stack_rays(self):
        """
        Stacks all rays tensors.

        Depending on the configuration, this method stacks the rays tensors either
        horizontally or vertically, and optionally reshapes the RGB tensors if LPIPS
        loss is enabled.
        """
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            self.all_idxs = torch.cat(self.all_idxs, 0)
        elif self.lpips and self.split == "train":
            self.all_rays = torch.stack(self.all_rays, 0)
            self.all_rgbs = torch.stack(self.all_rgbs, 0)
            self.all_idxs = torch.stack(self.all_idxs, 0)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3)
            self.all_idxs = torch.stack(self.all_idxs, 0)

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        sample = {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}
        return sample


class GaussianSplattingDataset(BaseDataset):
    """
    A dataset class for Gaussian Splatting in 3D scene reconstruction.

    This class extends `BaseDataset` to support operations specific to Gaussian Splatting,
    such as loading and processing images, handling camera parameters, and resizing images
    according to specified resolution scales.

    Attributes:
        cams (list): List of Camera objects containing camera parameters and image paths.
        resolution (int): The target resolution for the images.
        resolution_scales (list): List of resolution scales to process the images at different resolutions.

    Methods:
        load_and_process_image(image_path, resolution_scale): Loads and processes an image from a given path.
        __getitem__(idx): Retrieves a camera and its processed image by index.
        __len__(): Returns the total number of items in the dataset.
        collate_fn(batch): Static method to collate data into a batch.
        resize_img_resolution(img, resolution, resolution_scale): Resizes an image to a specified resolution.
    """

    def __init__(self, split="train", downsample: float = 1.0, args=None):
        """
        Initializes the GaussianSplattingDataset with configuration parameters.

        Parameters:
            split (str): The dataset split, e.g., 'train', 'test'.
            downsample (float): The factor by which images are downsampled.
            args (Namespace): Configuration arguments for the dataset, including resolution and resolution_scales.
        """
        super().__init__(split, downsample, args)
        self.cams = []  # list of Camera objects
        self.resolution = args.resolution
        self.resolution_scales = args.resolution_scales

    def load_and_process_image(self, image_path, resolution_scale=1):
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1, 1, 1]) if self.white_bg else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        # process image
        im_data = self.resize_img_resolution(image, self.resolution, resolution_scale)
        im_data = PILtoTorch(im_data)

        loaded_mask = None
        if im_data.shape[1] == 4:
            loaded_mask = im_data[3:4, ...]

        im_data = im_data.clamp(0.0, 1.0)
        if loaded_mask is not None:
            im_data *= loaded_mask
        else:
            im_data *= torch.ones((1, im_data.shape[1], im_data.shape[2]))

        return im_data

    def __getitem__(self, idx):
        cam_id = idx // len(self.resolution_scales)
        reso_scale_idx = idx % len(self.resolution_scales)
        resolution_scale = self.resolution_scales[reso_scale_idx]

        cam = self.cams[cam_id]
        if self.args.preload:
            im_data = cam.image
        else:
            im_data = self.load_and_process_image(cam.image_path, resolution_scale)
            cam.resolution_scale = resolution_scale

        assert self.split in ["train", "test"], "Not support more than one resolution scale for path split"

        return cam, im_data

    def __len__(self):
        return len(self.cams) * len(self.resolution_scales)

    @staticmethod
    def collate_fn(batch):
        cams = []
        im_datas = []
        for cam, im_data in batch:
            cams.append(cam)
            im_datas.append(im_data)

        return cams[0], torch.stack(im_datas, dim=0)  # TODO support batching

    @staticmethod
    def resize_img_resolution(img: Image, resolution: Union[float, int], resolution_scale=1.0) -> Image:
        orig_w, orig_h = img.size
        if resolution == -1:
            return img
        elif resolution in [1, 2, 4, 8]:
            resolution = round(orig_w / (resolution_scale * resolution)), round(
                orig_h / (resolution_scale * resolution)
            )
        else:  # should be a type that converts to float
            global_down = orig_w / resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))

        return img.resize(resolution)
