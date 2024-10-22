import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import distributed as dist
from tqdm import tqdm

from landmark.nerf_components.data.cameras import Camera
from landmark.nerf_components.utils.graphics_utils import (
    getProjectionMatrix,
    pose_rotate_translate,
)

from .base_dataset import GaussianSplattingDataset, VolumeRenderDataset
from .utils.ray_utils import (
    get_ray_directions_blender,
    get_rays_with_directions,
    load_json_drone_data,
    read_Image,
)
from .utils.utils import focal2fov, fov2focal, getNeRFppNorm, getWorld2View2


class CityDataset(VolumeRenderDataset):
    """
    A dataset class for handling city datasets in volume rendering.

    This class extends `VolumeRenderDataset` to support operations specific to city datasets,
    such as reading metadata, processing images, and handling camera parameters for 3D rendering.

    Attributes:
        Inherits all attributes from `VolumeRenderDataset`.

    Methods:
        read_meta(): Reads and processes the city dataset metadata.
    """

    def __init__(self, split="train", downsample=10, is_stack=False, enable_lpips=False, args=None, preprocess=False):
        """
        Initializes the CityDataset with configuration parameters.

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
        Read drone dataset from rootdir.
        """
        meta = load_json_drone_data(
            self.datadir,
            self.split,
            self.image_scale,
            subfolder=self.args.subfolder,
            debug=self.debug,
        )
        poses, focals, fnames, hw, imgfolder = meta.values()
        N = len(poses)
        idxs = list(range(0, N))

        H, W = hw

        self.img_wh = [W, H]

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
            focal = focals
            f_path = os.path.join(imgfolder, fnames[i])
            self.poses.append(pose)
            self.image_paths.append(f_path)

            img = read_Image(f_path, self.transform)
            if len(focal) == 1:
                directions = get_ray_directions_blender(int(H), int(W), (focal, focal))
            else:
                directions = get_ray_directions_blender(int(H), int(W), (focal[i], focal[i]))
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

    def preprocess(self, args, split_num=20, batch_size=8192, client=None):
        """
        Load and preprocess dataset.

        Args:
            args(ArgsConfig): training or render config.
            split_num(int): chunk split number.
            batch_size(int): length in a batch.
            client(Union[None, Client]): None or petrel_client.
        """
        meta = load_json_drone_data(
            self.datadir,
            self.split,
            self.image_scale,
            subfolder=self.args.subfolder,
            debug=self.debug,
        )
        poses, focals, fnames, hw, imgfolder = meta.values()
        N = len(poses)
        idxs = list(range(0, N))

        H, W = hw
        self.img_wh = [W, H]

        chunk_size = len(idxs) // split_num  # split_num = 20
        chunk_list = []
        for i in range(0, len(idxs), chunk_size):
            chunk_list.append(idxs[i : i + chunk_size])

        assert not self.lpips

        rest_rays = []
        rest_rgbs = []
        rest_idxs = []
        num = 0
        for partial_list in chunk_list:
            part_rgbs = []
            part_rays = []
            part_idxs = []
            for i in partial_list:
                pose = torch.FloatTensor(poses[i])
                focal = focals
                f_path = os.path.join(imgfolder, fnames[i])
                self.poses.append(pose)
                self.image_paths.append(f_path)

                img = read_Image(f_path, self.transform)
                if len(focal) == 1:
                    directions = get_ray_directions_blender(int(H), int(W), (focal, focal))
                else:
                    directions = get_ray_directions_blender(int(H), int(W), (focal[i], focal[i]))
                rays_o, rays_d = get_rays_with_directions(directions, pose)

                part_rgbs += [img]
                part_rays += [torch.cat([rays_o, rays_d], 1)]
                part_idxs += [torch.zeros(rays_d.shape[0]) + i]

            part_rays = torch.cat(part_rays, 0)
            part_rgbs = torch.cat(part_rgbs, 0)
            part_idxs = torch.cat(part_idxs, 0)

            part_num = part_rays.shape[0]
            g = torch.Generator()
            g.manual_seed(args.random_seed * num)
            ids = torch.randperm(part_num, generator=g)
            curr = 0
            while curr + batch_size < part_num:
                batch_id = ids[curr : curr + batch_size]
                rays = part_rays[batch_id]
                rgbs = part_rgbs[batch_id]
                idxs = part_idxs[batch_id]
                if args.processed_data_type == "ceph":
                    batch_data = torch.cat([rays, rgbs, idxs], dim=1).numpy().tobytes()
                    num_str = str(num).zfill(10)
                    data_url = (
                        args.preprocessed_dir
                        + args.dataset_name
                        + "/"
                        + args.datadir
                        + "/ds"
                        + str(args.downsample_train)
                        + "_"
                        + args.partition
                        + "/"
                        + num_str
                        + ".npy"
                    )
                    client.put(data_url, batch_data)
                else:
                    num_str = str(num).zfill(10)
                    outfolder = (
                        args.preprocessed_dir
                        + args.dataset_name
                        + "/"
                        + args.datadir
                        + "/ds"
                        + str(args.downsample_train)
                        + "_"
                        + args.partition
                        + "/"
                    )
                    os.makedirs(outfolder, exist_ok=True)
                    outfile = outfolder + num_str + ".npz"
                    np.savez(outfile, rays=rays.numpy(), rgbs=rgbs.numpy(), idxs=idxs.numpy())
                curr += batch_size
                num += 1
            if curr < part_num:
                batch_id = ids[curr:part_num]
                rays = part_rays[batch_id]
                rgbs = part_rgbs[batch_id]
                idxs = part_idxs[batch_id]
                rest_rays.append(rays)
                rest_rgbs.append(rgbs)
                rest_idxs.append(idxs)

        rest_rays = torch.cat(rest_rays, 0)
        rest_rgbs = torch.cat(rest_rgbs, 0)
        rest_idxs = torch.cat(rest_idxs, 0)
        rest_num = rest_rays.shape[0]
        g = torch.Generator()
        g.manual_seed(args.random_seed)
        ids = torch.randperm(rest_num, generator=g)
        curr = 0
        while curr + batch_size < rest_num:
            batch_id = ids[curr : curr + batch_size]
            rays = rest_rays[batch_id]
            rgbs = rest_rgbs[batch_id]
            idxs = rest_idxs[batch_id]
            if args.processed_data_type == "ceph":
                batch_data = torch.cat([rays, rgbs, idxs], dim=1).numpy().tobytes()
                num_str = str(num).zfill(10)
                data_url = (
                    args.preprocessed_dir
                    + args.dataset_name
                    + "/"
                    + args.datadir
                    + "/ds"
                    + str(args.downsample_train)
                    + "_"
                    + args.partition
                    + "/"
                    + num_str
                    + ".npy"
                )
                client.put(data_url, batch_data)
            else:
                num_str = str(num).zfill(10)
                outfolder = (
                    args.preprocessed_dir
                    + args.dataset_name
                    + "/"
                    + args.datadir
                    + "/ds"
                    + str(args.downsample_train)
                    + "_"
                    + args.partition
                    + "/"
                )
                os.makedirs(outfolder, exist_ok=True)
                outfile = outfolder + num_str + ".npz"
                np.savez(outfile, rays=rays.numpy(), rgbs=rgbs.numpy(), idxs=idxs.numpy())
            curr += batch_size
            num += 1

        if args.processed_data_type == "ceph":
            bytedata = np.array([num], dtype=np.longlong).tobytes()
            data_url = (
                args.preprocessed_dir
                + args.dataset_name
                + "/"
                + args.datadir
                + "/ds"
                + str(args.downsample_train)
                + "_"
                + args.partition
                + "/num.npy"
            )
            client.put(data_url, bytedata)
        else:
            num = np.array([num], dtype=np.longlong)
            outfolder = (
                args.preprocessed_dir
                + args.dataset_name
                + "/"
                + args.datadir
                + "/ds"
                + str(args.downsample_train)
                + "_"
                + args.partition
                + "/"
            )
            os.makedirs(outfolder, exist_ok=True)
            outfile = outfolder + "num.npz"
            np.savez(outfile, num=num)

    def distpreprocess(self, args, batch_size=8192, client=None):
        """
        Load dataset and do a distributed preprocessing.

        Args:
            args(ArgsConfig): training or render config.
            batch_size(int): length in a batch.
            client(Union[None, Client]): None or petrel_client.
        """
        meta = load_json_drone_data(
            self.datadir,
            self.split,
            self.image_scale,
            subfolder=self.args.subfolder,
            debug=self.debug,
        )
        poses, focals, fnames, hw, imgfolder = meta.values()
        N = len(poses)
        idxs = list(range(0, N))

        H, W = hw
        self.img_wh = [W, H]

        chunk_size = len(idxs) // args.world_size  # split_num = 20
        chunk_list = []
        for i in range(0, len(idxs), chunk_size):
            chunk_list.append(idxs[i : i + chunk_size])
        rest_list = idxs[args.world_size * chunk_size : len(idxs)]
        num = args.rank

        partial_list = chunk_list[args.rank]
        part_rgbs = []
        part_rays = []
        part_idxs = []
        for i in partial_list:
            pose = torch.FloatTensor(poses[i])
            focal = focals
            f_path = os.path.join(imgfolder, fnames[i])
            self.poses.append(pose)
            self.image_paths.append(f_path)

            img = read_Image(f_path, self.transform)
            if len(focal) == 1:
                directions = get_ray_directions_blender(int(H), int(W), (focal, focal))
            else:
                directions = get_ray_directions_blender(int(H), int(W), (focal[i], focal[i]))
            rays_o, rays_d = get_rays_with_directions(directions, pose)

            part_rgbs += [img]
            part_rays += [torch.cat([rays_o, rays_d], 1)]
            part_idxs += [torch.zeros(rays_d.shape[0]) + i]

        part_rays = torch.cat(part_rays, 0)
        part_rgbs = torch.cat(part_rgbs, 0)
        part_idxs = torch.cat(part_idxs, 0)

        part_num = part_rays.shape[0]
        g = torch.Generator()
        g.manual_seed(args.random_seed * num)
        ids = torch.randperm(part_num, generator=g)
        curr = 0
        while curr + batch_size < part_num:
            batch_id = ids[curr : curr + batch_size]
            rays = part_rays[batch_id]
            rgbs = part_rgbs[batch_id]
            idxs = part_idxs[batch_id]
            if args.processed_data_type == "ceph":
                batch_data = torch.cat([rays, rgbs, idxs], dim=1).numpy().tobytes()
                num_str = str(num).zfill(10)
                data_url = (
                    args.preprocessed_dir
                    + args.dataset_name
                    + "/"
                    + args.datadir
                    + "/ds"
                    + str(args.downsample_train)
                    + "_"
                    + args.partition
                    + "/"
                    + num_str
                    + ".npy"
                )
                client.put(data_url, batch_data)
            else:
                num_str = str(num).zfill(10)
                outfolder = (
                    args.preprocessed_dir
                    + args.dataset_name
                    + "/"
                    + args.datadir
                    + "/ds"
                    + str(args.downsample_train)
                    + "_"
                    + args.partition
                    + "/"
                )
                os.makedirs(outfolder, exist_ok=True)
                outfile = outfolder + num_str + ".npz"
                np.savez(outfile, rays=rays.numpy(), rgbs=rgbs.numpy(), idxs=idxs.numpy())
                if args.rank == 0:
                    print("finish writing: ", outfile)
            curr += batch_size
            num += args.world_size
        if curr < part_num:
            batch_id = ids[curr:part_num]
            rays = part_rays[batch_id]
            rgbs = part_rgbs[batch_id]
            idxs = part_idxs[batch_id]
            rest_rays = rays
            rest_rgbs = rgbs
            rest_idxs = idxs

        all_rest_rays = [None for i in range(args.world_size)]
        all_rest_rgbs = [None for i in range(args.world_size)]
        all_rest_idxs = [None for i in range(args.world_size)]
        dist.all_gather_object(all_rest_rays, rest_rays)
        dist.all_gather_object(all_rest_rgbs, rest_rgbs)
        dist.all_gather_object(all_rest_idxs, rest_idxs)
        if args.rank == 0:
            for i in rest_list:
                pose = torch.FloatTensor(poses[i])
                f_path = os.path.join(imgfolder, fnames[i])
                self.poses.append(pose)
                self.image_paths.append(f_path)

                img = read_Image(f_path, self.transform)
                rays_o, rays_d = get_rays_with_directions(directions, pose)

                all_rest_rgbs += [img]
                all_rest_rays += [torch.cat([rays_o, rays_d], 1)]
                all_rest_idxs += [torch.zeros(rays_d.shape[0]) + i]

            rest_rays = torch.cat(all_rest_rays, 0)
            rest_rgbs = torch.cat(all_rest_rgbs, 0)
            rest_idxs = torch.cat(all_rest_idxs, 0)

            rest_num = rest_rays.shape[0]
            g = torch.Generator()
            g.manual_seed(args.random_seed)
            ids = torch.randperm(rest_num, generator=g)
            curr = 0
            while curr + batch_size < rest_num:
                batch_id = ids[curr : curr + batch_size]
                rays = rest_rays[batch_id]
                rgbs = rest_rgbs[batch_id]
                idxs = rest_idxs[batch_id]
                if args.processed_data_type == "ceph":
                    batch_data = torch.cat([rays, rgbs, idxs], dim=1).numpy().tobytes()
                    num_str = str(num).zfill(10)
                    data_url = (
                        args.preprocessed_dir
                        + args.dataset_name
                        + "/"
                        + args.datadir
                        + "/ds"
                        + str(args.downsample_train)
                        + "_"
                        + args.partition
                        + "/"
                        + num_str
                        + ".npy"
                    )
                    client.put(data_url, batch_data)
                else:
                    num_str = str(num).zfill(10)
                    outfolder = (
                        args.preprocessed_dir
                        + args.dataset_name
                        + "/"
                        + args.datadir
                        + "/ds"
                        + str(args.downsample_train)
                        + "_"
                        + args.partition
                        + "/"
                    )
                    os.makedirs(outfolder, exist_ok=True)
                    outfile = outfolder + num_str + ".npz"
                    np.savez(outfile, rays=rays.numpy(), rgbs=rgbs.numpy(), idxs=idxs.numpy())
                    if args.rank == 0:
                        print("finish writing: ", outfile)
                curr += batch_size
                num += 1

            if args.processed_data_type == "ceph":
                bytedata = np.array([num], dtype=np.longlong).tobytes()
                data_url = (
                    args.preprocessed_dir
                    + args.dataset_name
                    + "/"
                    + args.datadir
                    + "/ds"
                    + str(args.downsample_train)
                    + "_"
                    + args.partition
                    + "/num.npy"
                )
                client.put(data_url, bytedata)
            else:
                num = np.array([num], dtype=np.longlong)
                outfolder = (
                    args.preprocessed_dir
                    + args.dataset_name
                    + "/"
                    + args.datadir
                    + "/ds"
                    + str(args.downsample_train)
                    + "_"
                    + args.partition
                    + "/"
                )
                os.makedirs(outfolder, exist_ok=True)
                outfile = outfolder + "num.npz"
                np.savez(outfile, num=num)
                if args.rank == 0:
                    print("finish writing: ", outfile)


class CityGaussianDataset(GaussianSplattingDataset):
    """Class for City 3DGS Dataset"""

    def __init__(self, split="train", downsample=1, args=None, block_mask=None, loaded_cams=None):
        super().__init__(split, downsample, args)

        self.img_wh = [None, None]
        if loaded_cams:
            self.cams = loaded_cams
        else:
            self.read_meta_file()
            self.read_meta()

        self.cameras_extent = getNeRFppNorm(self.cams)["radius"]

        if block_mask is not None:
            self.cams = [cam for valid, cam in zip(block_mask, self.cams) if valid]

    def read_meta_file(self):
        if self.split in ["train", "test"]:
            self.meta_json = Path(self.datadir) / f"transforms_{self.split}.json"
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
                image_path = os.path.join(self.datadir, frame["file_path"] + extension)
                assert os.path.exists(image_path), f"Image {image_path} does not exist!"
                image_name = Path(image_path).stem

                c2w = np.array(frame["transform_matrix"])

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
                        image=self.load_and_process_image(image_path) if self.args.preload else None,
                    )
                )
