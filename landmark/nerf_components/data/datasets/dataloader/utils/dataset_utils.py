import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler as DSampler

from landmark.nerf_components.data.datasets.dataloader import (
    dataset_dict,
    dataset_dict_gs,
)
from landmark.nerf_components.data.datasets.dataloader.processeddataset import (
    PreprocessedDataset,
)


class DatasetInfo:
    """Class used to describe dataset information."""

    def __init__(self, aabb, near_far, white_bg):
        self.aabb = aabb
        self.near_far = near_far
        self.white_bg = white_bg


class SimpleSampler:
    """Sampler for randomly sampling rays."""

    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


class DistributedSampler:
    """Sampler for randomly sampling rays in DDP"""

    def __init__(self, total, batch, rank, world_size, seed: int = 0):
        self.total = total
        self.batch = batch
        self.curr = total
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.ids = None

    def nextids(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.curr += self.batch * self.world_size
        if self.curr + self.batch * self.world_size > self.total:
            self.ids = torch.randperm(self.total, generator=g)
            self.curr = 0
            self.epoch += 1
        return self.ids[self.curr + self.rank * self.batch : self.curr + self.rank * self.batch + self.batch]


def prep_dataset(enable_lpips, args):
    """Prepare dataset used to train."""

    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(
        split="train",
        downsample=args.downsample_train,
        is_stack=enable_lpips,
        enable_lpips=enable_lpips,
        args=args,
    )
    test_dataset = dataset(
        split="test",
        downsample=args.downsample_train,
        is_stack=True,
        enable_lpips=enable_lpips,
        args=args,
    )
    return train_dataset, test_dataset


def prep_traindataset(enable_lpips, args):
    """Prepare dataset used to train."""
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(
        split="train",
        downsample=args.downsample_train,
        is_stack=enable_lpips,
        enable_lpips=enable_lpips,
        args=args,
    )
    return train_dataset


def prep_testdataset(args):
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        split="test",
        downsample=args.downsample_train,
        is_stack=True,
        args=args,
    )
    return test_dataset


def prep_dataset_gs(args):
    """Prepare dataset used to train."""
    dataset = dataset_dict_gs[args.dataset_name]
    train_dataset = dataset(
        split="train",
        downsample=args.downsample_train,
        args=args,
    )
    test_dataset = dataset(
        split="test",
        downsample=args.downsample_train,
        args=args,
    )
    return train_dataset, test_dataset


def prep_traindataset_gs(args):
    """Prepare dataset used to train."""
    dataset = dataset_dict_gs[args.dataset_name]
    train_dataset = dataset(
        split="train",
        downsample=args.downsample_train,
        args=args,
    )
    return train_dataset


def prep_testdataset_gs(args):
    dataset = dataset_dict_gs[args.dataset_name]
    test_dataset = dataset(
        split="test",
        downsample=args.downsample_train,
        args=args,
    )
    return test_dataset


def get_preprocessed_loader(args):
    conf_path = "~/petreloss.conf"
    data_type = args.processed_data_type
    filefolder = (
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
    batch_size = args.batch_size // 8192
    dataset = PreprocessedDataset(filefolder, data_type, conf_path)
    if args.DDP:
        if args.model_parallel_and_DDP:
            batch_size = batch_size // args.num_mp_groups
            sampler = DSampler(
                dataset, num_replicas=args.num_mp_groups, rank=args.dp_rank, shuffle=True, seed=0, drop_last=False
            )
        else:
            batch_size = batch_size // args.world_size
            sampler = DSampler(
                dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, seed=0, drop_last=False
            )
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return dataloader


def prep_sampler(enable_lpips, args, train_dataset):
    """Prepare rays sampler for training"""
    allrays, allrgbs, allidxs = train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.all_idxs

    if args.preload:
        print("preload to cuda")
        allrays = allrays.cuda()
        allrgbs = allrgbs.cuda()
        allidxs = allidxs.cuda()

    # if args.DDP:
    #     if args.model_parallel_and_DDP:
    #         num_replicas = args.num_mp_groups
    #     else:
    #         num_replicas = args.world_size
    #     sample_batch = args.batch_size * num_replicas
    # else:
    #     sample_batch = args.batch_size

    if enable_lpips:
        trainingsampler = SimpleSampler(allrays.shape[0], 1)
    else:
        trainingsampler = SimpleSampler(allrays.shape[0], args.batch_size)
    return allrays, allrgbs, allidxs, trainingsampler
