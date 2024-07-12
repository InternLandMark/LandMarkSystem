import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from landmark.nerf_components.data.datasets.dataloader import dataset_dict_gs
from landmark.nerf_components.data.datasets.dataloader.utils.dataset_utils import (
    DatasetInfo,
    get_preprocessed_loader,
    prep_dataset,
    prep_sampler,
    prep_testdataset,
    prep_testdataset_gs,
    prep_traindataset,
    prep_traindataset_gs,
)
from landmark.utils import EnvSetting


class DatasetManager:
    """Dataset Manager Class for NeRF"""

    def __init__(self, config, enable_lpips=False):
        self.config = config

        self.enable_lpips = enable_lpips
        # create dataset
        if config.dataset_type == "nerf":
            if config.use_preprocessed_data:
                if config.dataset_type == "nerf":
                    self.test_dataset = prep_testdataset(config)
                    if config.is_train:
                        self.train_dataloader = get_preprocessed_loader(config)
            else:
                if EnvSetting.RANK == 0:
                    if config.is_train:
                        self.train_dataset = prep_traindataset(self.enable_lpips, config)
                        self.allrays, self.allrgbs, self.allidxs, self.trainingsampler = prep_sampler(
                            self.enable_lpips, config, self.train_dataset
                        )
                        print("prepare sampler done", flush=True)
                    self.test_dataset = prep_testdataset(config)
                else:
                    if config.dataset_type == "nerf":
                        self.test_dataset = prep_testdataset(config)
            self.dataset_info = DatasetInfo(
                self.test_dataset.scene_bbox, self.test_dataset.near_far, self.test_dataset.white_bg
            )
            self.white_bg = self.dataset_info.white_bg

        elif config.dataset_type == "gaussian":
            self.test_dataset = prep_testdataset_gs(config)
            if config.is_train:
                self.train_dataset = prep_traindataset_gs(config)
                if config.train_all_data:
                    self.train_dataset = ConcatDataset([self.train_dataset, self.test_dataset])
                if config.DDP:
                    self.distributed_sampler = DistributedSampler(self.train_dataset)
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=config.batch_size,
                    collate_fn=self.train_dataset.collate_fn,
                    shuffle=(not config.DDP),
                    num_workers=0,
                    pin_memory=False,
                    sampler=self.distributed_sampler if config.DDP else None,
                )
                self.train_loader_iter = iter(self.train_loader)
                self.train_data_size = len(self.train_dataset)

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=config.batch_size,
                collate_fn=self.test_dataset.collate_fn,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            self.test_loader_iter = iter(self.test_loader)
            self.white_bg = self.test_dataset.white_bg
            self.test_data_size = len(self.test_dataset)

        else:
            raise ValueError(f"Unknown dataset type {config.dataset_type}")

    def get_camera(self, split="train"):  # TODO for gaussian currently
        """
        Get the next pose of camera.

        Parameters:
        - split (str): The split to get the data from. Default is "train".

        Returns:
        - tuple or any: The pose of the camera for Gaussian rendering from the specified split.
        """
        if self.config.max_iter_each_block != -1 and split == "train":
            self.iter_cnt += 1
            # TODO raise warning insead of assert
            assert (
                self.iter_cnt <= self.config.max_iter_each_block
            ), "Exceed the max iteration times in the current block, please swith to the next block."

        ret = None
        try:
            if split == "train":
                ret = next(self.train_loader_iter)
            elif split == "test":
                ret = next(self.test_loader_iter)
            else:
                raise ValueError(f"Unknown split {split}")
        except StopIteration as exc:
            if split == "train":
                self.train_loader_iter = iter(self.train_loader)
                ret = next(self.train_loader_iter)
            elif split == "test":
                self.test_loader_iter = iter(self.test_loader)
                ret = next(self.test_loader_iter)
            else:
                raise ValueError(f"Unknown split {split}") from exc

        if self.config.no_batching:
            assert self.config.batch_size == 1, "batch size must be 1 when batching is disabled"
            return ret[0], ret[1].squeeze(0)
        else:
            return ret

    def get_rays(self, iteration):
        # assert nerf not gs
        config = self.config
        if config.use_preprocessed_data:
            if iteration % len(self.train_dataloader) == 0 or iteration == config.start_iters:
                batch_iter = iter(self.train_dataloader)
            rays, rgbs, idxs = next(batch_iter)
            rays_train, rgb_train, idxs_train = rays.to(config.device), rgbs.to(config.device), idxs.to(config.device)
            rays_train = rays_train.view(-1, 6)
            rgb_train = rgb_train.view(-1, 3)
            idxs_train = idxs_train.view(rays_train.shape[0])
        else:
            if config.rank == 0:
                if config.add_upsample and iteration == config.add_upsample:
                    self.train_dataset, self.test_dataset = prep_dataset(self.enable_lpips, config)
                    self.allrays, self.allrgbs, self.allidxs, self.trainingsampler = prep_sampler(
                        self.enable_lpips, config, self.train_dataset
                    )
                    print("upsample training dataset by x2")

                if config.add_lpips > 0 and iteration == config.add_lpips:
                    self.enable_lpips = True
                    self.train_dataset, self.test_dataset = prep_dataset(self.enable_lpips, config)
                    self.allrays, self.allrgbs, self.allidxs, self.trainingsampler = prep_sampler(
                        self.enable_lpips, config, self.train_dataset
                    )
                    print("reformat dataset with patch samples")

                ray_idx = self.trainingsampler.nextids()
                rays_train, rgb_train, idxs_train = (
                    self.allrays[ray_idx].to(config.device),
                    self.allrgbs[ray_idx].to(config.device),
                    self.allidxs[ray_idx].to(config.device),
                )
            else:
                if self.enable_lpips or (config.add_lpips > 0 and iteration == config.add_lpips):
                    self.enable_lpips = True
                    ps = config.patch_size
                    rays_train = torch.zeros([ps * ps, 6], dtype=torch.float32, device=config.device)
                    rgb_train = torch.zeros([ps * ps, 3], dtype=torch.float32, device=config.device)
                    idxs_train = torch.zeros([ps * ps], dtype=torch.float32, device=config.device)
                else:
                    rays_train = torch.zeros([config.batch_size, 6], dtype=torch.float32, device=config.device)
                    rgb_train = torch.zeros([config.batch_size, 3], dtype=torch.float32, device=config.device)
                    idxs_train = torch.zeros([config.batch_size], dtype=torch.float32, device=config.device)

            if config.world_size > 1:
                dist.broadcast(rays_train, src=0)
                dist.broadcast(rgb_train, src=0)
                dist.broadcast(idxs_train, src=0)

            if config.DDP:
                if config.model_parallel_and_DDP:
                    num_replicas = config.num_mp_groups
                    dp_rank = config.dp_rank
                else:
                    num_replicas = config.world_size
                    dp_rank = config.rank
                rays_train = torch.chunk(rays_train, num_replicas, dim=0)[dp_rank]
                rgb_train = torch.chunk(rgb_train, num_replicas, dim=0)[dp_rank]
                idxs_train = torch.chunk(idxs_train, num_replicas, dim=0)[dp_rank]

        if self.enable_lpips:
            rays_train = rays_train.view(-1, 6)
            rgb_train = rgb_train.view(-1, 3)
            idxs_train = idxs_train.view(rays_train.shape[0])

        return rays_train, rgb_train, idxs_train

    def divide_dataset(self):
        config = self.config
        assert config.dynamic_load, "divide_dataset is only supported for dynamic_load."

        plane_division = config.plane_division
        aabb = torch.tensor([config.lb, config.ub])
        x_min, y_min, _ = aabb[0].numpy()
        x_max, y_max, _ = aabb[1].numpy()
        x_step = (x_max - x_min) / plane_division[0]
        y_step = (y_max - y_min) / plane_division[1]
        x_split = [x_min + x * x_step for x in range(plane_division[0] + 1)]
        y_split = [y_min + y * y_step for y in range(plane_division[1] + 1)]

        block_train_dataset = [[[None] for _ in range(plane_division[1])] for _ in range(plane_division[0])]
        block_test_dataset = [[[None] for _ in range(plane_division[1])] for _ in range(plane_division[0])]

        all_train_camera_xy = torch.stack([cam.camera_center[:2] for cam in self.train_dataset.cams])
        all_test_camera_xy = torch.stack([cam.camera_center[:2] for cam in self.test_dataset.cams])

        dataset = dataset_dict_gs[config.dataset_name]  # only support for gaussian dataset_type currently

        for y_idx in range(plane_division[1]):
            for x_idx in range(plane_division[0]):
                coord_x1, coord_x2 = x_split[x_idx], x_split[x_idx + 1]
                coord_y1, coord_y2 = y_split[y_idx], y_split[y_idx + 1]
                train_block_mask = (
                    (all_train_camera_xy[:, 0] >= coord_x1)
                    & (all_train_camera_xy[:, 0] < coord_x2)
                    & (all_train_camera_xy[:, 1] >= coord_y1)
                    & (all_train_camera_xy[:, 1] < coord_y2)
                )
                test_block_mask = (
                    (all_test_camera_xy[:, 0] >= coord_x1)
                    & (all_test_camera_xy[:, 0] < coord_x2)
                    & (all_test_camera_xy[:, 1] >= coord_y1)
                    & (all_test_camera_xy[:, 1] < coord_y2)
                )

                block_train_dataset[x_idx][y_idx] = dataset(
                    split="train",
                    downsample=config.downsample_train,
                    args=config,
                    block_mask=train_block_mask,
                    loaded_cams=self.train_dataset.cams,
                )
                block_test_dataset[x_idx][y_idx] = dataset(
                    split="test",
                    downsample=config.downsample_train,
                    args=config,
                    block_mask=test_block_mask,
                    loaded_cams=self.test_dataset.cams,
                )

        self.block_train_dataset = block_train_dataset
        self.block_test_dataset = block_test_dataset

        self.block_train_loader_iter = [[None for _ in range(plane_division[1])] for _ in range(plane_division[0])]
        self.block_test_loader_iter = [[None for _ in range(plane_division[1])] for _ in range(plane_division[0])]
        for y_idx in range(plane_division[1]):
            for x_idx in range(plane_division[0]):
                if config.DDP:
                    distributed_sampler = DistributedSampler(self.block_train_dataset[x_idx][y_idx])
                if len(self.block_train_dataset[x_idx][y_idx]):
                    self.block_train_loader_iter[x_idx][y_idx] = iter(
                        DataLoader(
                            self.block_train_dataset[x_idx][y_idx],
                            batch_size=config.batch_size,
                            collate_fn=self.train_dataset.collate_fn,
                            shuffle=(not config.DDP),
                            sampler=distributed_sampler if config.DDP else None,
                        )
                    )
                else:
                    self.block_train_loader_iter[x_idx][y_idx] = None

                if len(self.block_test_dataset[x_idx][y_idx]):
                    self.block_test_loader_iter[x_idx][y_idx] = iter(
                        DataLoader(
                            self.block_test_dataset[x_idx][y_idx],
                            batch_size=config.batch_size,
                            collate_fn=self.test_dataset.collate_fn,
                            shuffle=False,
                        )
                    )
                else:
                    self.block_test_loader_iter[x_idx][y_idx] = None

        # print len info
        cnt = 0
        print("blocked train dataset info:")
        for y_idx in range(plane_division[1]):
            for x_idx in range(plane_division[0]):
                cnt += len(block_train_dataset[x_idx][y_idx])
                print(f"{len(block_train_dataset[x_idx][y_idx])}\t", end=" ")
            print("")
        assert cnt == len(all_train_camera_xy), (
            f"Error, some of the camera may be out of the aabb range, only {cnt} cams are selected from"
            f" {len(all_train_camera_xy)} cams in total"
        )

        cnt = 0
        print("blocked test dataset info:")
        for y_idx in range(plane_division[1]):
            for x_idx in range(plane_division[0]):
                cnt += len(block_test_dataset[x_idx][y_idx])
                print(f"{len(block_test_dataset[x_idx][y_idx])}\t", end=" ")
            print("")
        assert cnt == len(all_test_camera_xy), (
            f"Error, some of the test camera may be out of the aabb range, only {cnt} cams are selected from"
            f" {len(all_test_camera_xy)} cams in total"
        )

    def switch_cell(self, block_coord):
        config = self.config
        assert config.dynamic_load, "set_sub_dataset is only supported for dynamic_load."

        self.train_loader_iter = self.block_train_loader_iter[block_coord[0]][block_coord[1]]
        self.test_loader_iter = self.block_test_loader_iter[block_coord[0]][block_coord[1]]

        # set current train data size
        self.train_data_size = len(self.block_train_dataset[block_coord[0]][block_coord[1]])
        self.test_data_size = len(self.block_test_dataset[block_coord[0]][block_coord[1]])

        # reset iter count
        self.iter_cnt = 0

    def use_all_data(self):
        config = self.config
        assert config.dynamic_load, "use_all_data is only supported for dynamic_load."
        self.train_loader_iter = iter(self.train_loader)
        self.test_loader_iter = iter(self.test_loader)
        self.train_data_size = len(self.train_dataset)
        self.test_data_size = len(self.test_dataset)
        self.iter_cnt = 0
