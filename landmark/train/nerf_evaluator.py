import datetime
import os

import numpy as np
import torch
from torch import nn

from landmark.communicator.comm_context import CommContext
from landmark.communicator.group_initializer import CommMode, ParallelMode
from landmark.nerf_components.data.data_manager import DatasetManager
from landmark.nerf_components.scene import SceneManager
from landmark.train.utils.distributed_utils import is_main_rank
from landmark.utils import EnvSetting
from landmark.utils.config import Config


class NeRFEvaluator:
    """Base class for NeRF evaluator"""

    def __init__(self, config):
        self.config = config

        self.model: nn.Module = None
        self.data_mgr: DatasetManager = None
        self.scene_mgr: SceneManager = None

        # set log
        if config.add_timestamp:
            self.logfolder = f'{config.basedir}/{config.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
        else:
            self.logfolder = f"{config.basedir}/{config.expname}"

        self.config.logfolder = self.logfolder
        self.save_folder = f"{self.logfolder}/imgs_test_all/"

        if is_main_rank():
            os.makedirs(self.logfolder, exist_ok=True)
            os.makedirs(self.save_folder, exist_ok=True)

    def create_model(self):
        raise NotImplementedError

    def check_args(self):
        assert self.config.ckpt is not None, "Error, ckpt is None."

    def init_render_env(
        self,
    ):
        config = self.config

        torch.set_default_dtype(torch.float32)
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        # setup distributed
        comm_conf = Config({"world_size": EnvSetting.WORLD_SIZE, "rank": EnvSetting.RANK})
        if config.channel_parallel:
            comm_conf.update({"tp_size": config.channel_parallel_size})
        CommContext().init_distributed_env(world_size=comm_conf.world_size, rank=comm_conf.rank)
        CommContext().init_groups(comm_conf)
        config.local_rank = CommContext().get_local_rank(comm_mode=CommMode.GLOBAL) % torch.cuda.device_count()
        config.device = torch.device("cuda", config.local_rank)
        config.rank = CommContext().get_global_rank()
        config.world_size = EnvSetting.WORLD_SIZE
        if config.channel_parallel:
            config.mp_group = CommContext().get_group(comm_mode=ParallelMode.TENSOR_PARALLEL)
            config.mp_rank = CommContext().get_local_rank(comm_mode=ParallelMode.TENSOR_PARALLEL)
        if config.DDP:
            config.dp_group = CommContext().get_group(comm_mode=ParallelMode.DATA_PARALLEL)
            config.dp_rank = CommContext().get_local_rank(comm_mode=ParallelMode.DATA_PARALLEL)

        config.model_parallel = bool(config.channel_parallel or config.branch_parallel)

        print("rank", config.rank)
        print("world_size", config.world_size)
        print("local rank", config.local_rank)
        print("device", config.device)
        print(
            f'{"Rendering in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."}'
            % (config.rank, config.world_size)
        )
        if config.channel_parallel:
            config.DDP = (config.world_size // config.channel_parallel_size) > 1
        else:
            config.DDP = config.world_size > 1
        assert config.rank >= 0

        if config.model_parallel:
            plane_division = config.plane_division
            config.model_parallel_degree = plane_division[0] * plane_division[1]
        elif config.channel_parallel:
            config.model_parallel_degree = config.channel_parallel_size

        self.config = config

    @torch.no_grad()
    def evaluation(
        self,
        test_dataset,
        config,
        savePath=None,
        N_vis=5,
        prtx="",
        N_samples=-1,
        compute_extra_metrics=False,
        device="cuda",
    ):
        raise NotImplementedError

    @torch.no_grad()
    def eval(self):
        config = self.config
        self.model.eval()

        PSNRs_test = self.evaluation(
            test_dataset=self.data_mgr.test_dataset,
            config=config,
            savePath=self.save_folder,
            N_vis=config.N_vis,
            N_samples=-1,
            compute_extra_metrics=config.compute_extra_metrics,
            device=config.device,
        )
        all_psnr = np.mean(PSNRs_test)
        if config.rank == 0:
            print(f"======> {config.expname} test all psnr: {all_psnr} <========================")
        return all_psnr
