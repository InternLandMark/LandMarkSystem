import datetime
import os
import random
from abc import abstractmethod

import imageio
import numpy as np
import torch
import wandb
from torch import nn
from tqdm import tqdm

from landmark.communicator.comm_context import CommContext, CommMode
from landmark.nerf_components.components_convertor import init_train_groups
from landmark.nerf_components.data.data_manager import DatasetManager
from landmark.nerf_components.scene import SceneManager
from landmark.nerf_components.utils.image_utils import visualize_depth_numpy
from landmark.nerf_components.utils.loss_utils import rgb_lpips, rgb_ssim
from landmark.nerf_components.utils.system_utils import set_print_with_timestamp
from landmark.train.utils.distributed_utils import is_main_rank
from landmark.utils import EnvSetting
from landmark.utils.config import Config


class NeRFTrainer:
    """
    Base class for Neural Radiance Field (NeRF) trainer.

    This class provides a structured way to train NeRF models, handling initialization, training,
    evaluation, and logging. It is designed to be extended by specific implementations that define
    the model architecture, optimizer, and training/evaluation procedures.

    Attributes:
        config (Config): Configuration object containing all settings for the training process.
        device (torch.device): The device (CPU/GPU) on which the model will be trained.
        model (nn.Module): The neural network model to be trained.
        module (nn.Module): An additional module, if any, used during training.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        data_mgr (DatasetManager): Manages the dataset for training and evaluation.
        scene_mgr (SceneManager): Manages scene-specific configurations and data.
        metrics (dict): A dictionary to store various metrics computed during training.
        logfolder (str): Path to the folder where logs and outputs are saved.
        optim_dir (str): Directory within `logfolder` for saving optimizer states.
        save_folder (str): Directory for saving test images.
        writer (SummaryWriter): TensorBoard writer for logging metrics.
    """

    def __init__(self, config):
        """
        Initializes the NeRFTrainer instance with the given configuration.

        Args:
            config (Config): The configuration object containing settings for the training process.
        """

        self.config = config
        self.device = config.device

        self.model: nn.Module = None
        self.module: nn.Module = None
        self.optimizer: torch.optim.Optimizer = None

        self.data_mgr: DatasetManager = None
        self.scene_mgr: SceneManager = None

        # set metric
        self.metrics = {}

        # set log
        if config.add_timestamp:
            self.logfolder = f'{config.basedir}/{config.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
        else:
            self.logfolder = f"{config.basedir}/{config.expname}"
        self.config.logfolder = self.logfolder

        if config.optim_dir is not None:
            self.optim_dir = config.optim_dir
        else:
            self.optim_dir = config.logfolder + "/optim/"

        self.save_folder = f"{self.logfolder}/imgs_test_all/"

        if is_main_rank():
            os.makedirs(self.logfolder, exist_ok=True)
            os.makedirs(f"{self.logfolder}/imgs_vis", exist_ok=True)
            os.makedirs(self.optim_dir, exist_ok=True)
            os.makedirs(self.save_folder, exist_ok=True)
            if config.tensorboard:
                from torch.utils.tensorboard import SummaryWriter

                tb_logfolder = (
                    f'{self.logfolder}/runs/{config.expname}_{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
                )
                self.writer = SummaryWriter(log_dir=tb_logfolder)
            if config.wandb:
                wandb.init(project=f"SH32TEST-{config.datadir}-{config.partition}", name=config.expname)
                wandb.config.update(config)

            # save args
            f = os.path.join(self.logfolder, f'args-{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}.txt')
            config.save_config(f)
            if config.config is not None:
                f = os.path.join(self.logfolder, "config.txt")
                with open(f, "w", encoding="utf-8") as file:
                    with open(config.config, "r", encoding="utf-8") as sfile:
                        file.write(sfile.read())

    def init_train_env(self):
        """
        Initializes the training environment, including setting up distributed training and
        logging directories.
        """
        config = self.config

        torch.set_default_dtype(torch.float32)
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

        torch.backends.cudnn.deterministic = True

        set_print_with_timestamp(config.print_timestamp)

        # setup distributed
        comm_conf = Config({"world_size": EnvSetting.WORLD_SIZE, "rank": EnvSetting.RANK})
        CommContext().init_distributed_env(world_size=comm_conf.world_size, rank=comm_conf.rank)
        config.local_rank = CommContext().get_local_rank(comm_mode=CommMode.GLOBAL) % torch.cuda.device_count()
        config.device = torch.device("cuda", config.local_rank)
        config.rank = CommContext().get_global_rank()
        config.world_size = EnvSetting.WORLD_SIZE
        config.model_parallel = bool(config.channel_parallel or config.branch_parallel)

        config = init_train_groups(config)

        print("rank", config.rank)
        print("world_size", config.world_size)
        print("local rank", config.local_rank)
        print("device", config.device)
        print(
            f"Training in distributed mode with multiple processes, 1 GPU per process. Process {config.rank}, total"
            f" {config.world_size}."
        )

        if not config.model_parallel and config.world_size > 1:
            assert config.DDP, "The world size is bigger than the required, but DDP is not enabled."
        else:
            print("Training with a single process on 1 GPUs.")
        assert config.rank >= 0

        if config.branch_parallel:
            plane_division = config.plane_division
            config.model_parallel_degree = plane_division[0] * plane_division[1]
        elif config.channel_parallel:
            config.model_parallel_degree = config.channel_parallel_size

        self.config = config

    @abstractmethod
    def create_model(self):
        """
        Creates the model to be trained. This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def create_optimizer(self):
        """
        Creates the optimizer for training the model. This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def check_args(self):
        """
        Checks the consistency and validity of the arguments provided in the config.
        """
        config = self.config
        assert (
            sum([config.channel_parallel, config.branch_parallel]) <= 1
        ), "Only one of the channel/plane/block parallel modes can be True currently"
        # check world size
        mp_size = self.config.model_parallel_degree if config.model_parallel else 1
        if config.DDP:
            assert (
                config.world_size > mp_size
            ), f"world size({config.world_size}) should be bigger than {mp_size} when using DDP"
            assert config.world_size % mp_size == 0, f"world size should be divisible by {mp_size} when using DDP"
        else:
            assert (
                config.world_size == mp_size
            ), f"world size({config.world_size}) should be equal to model parallel size ({mp_size})"

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Trains the model. This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluation(self, *args, **kwargs):
        """
        Evaluates the model. This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @torch.no_grad()
    def eval(self):
        """
        Evaluates the model on the test dataset, computes metrics, and saves the results.
        """
        self.model.eval()

        config = self.config
        test_dataset = self.data_mgr.test_dataset

        near_far = test_dataset.near_far
        img_eval_interval = 1 if config.N_vis < 0 else max(test_dataset.all_rays.shape[0] // config.N_vis, 1)
        img_indice = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))

        PSNRs = []  # to be refactored (frank)
        ssims, l_alex, l_vgg = [], [], []

        print("test_dataset render images", len(test_dataset.all_rays[0::img_eval_interval]))

        for idx, samples in enumerate(tqdm(test_dataset.all_rays[0::img_eval_interval])):
            W, H = test_dataset.img_wh
            # load groundtruth
            assert len(test_dataset.all_rgbs) > 0
            path = test_dataset.image_paths[img_indice[idx]]
            postfix = path.split("/")[-1]
            rgb_gt = test_dataset.all_rgbs[img_indice[idx]].view(H, W, 3)

            rays = samples.view(-1, samples.shape[-1])

            if config.encode_app:
                dummy_idxs = torch.zeros_like(rays[:, 0], dtype=torch.long).to(self.device)  # TODO need check
            else:
                dummy_idxs = None

            all_ret, _ = self.renderer_fn(
                rays, chunk=config.batch_size, near_far=near_far, N_samples=self.nsamples, idxs=dummy_idxs
            )

            rgb_map, depth_map = all_ret["rgb_map"], all_ret["depth_map"]
            rgb_map = rgb_map.clamp(0.0, 1.0)

            rgb_map, depth_map = (
                rgb_map.reshape(H, W, 3).cpu(),
                depth_map.reshape(H, W).cpu(),
            )

            # compute metrics
            loss = torch.mean((rgb_map - rgb_gt) ** 2)
            psnr = -10.0 * np.log(loss.item()) / np.log(10.0)
            PSNRs.append(psnr)

            ssim = rgb_ssim(rgb_map, rgb_gt, 1)
            l_a = rgb_lpips(rgb_gt.numpy(), rgb_map.numpy(), "alex", self.device)
            l_v = rgb_lpips(rgb_gt.numpy(), rgb_map.numpy(), "vgg", self.device)
            ssims.append(ssim)
            l_alex.append(l_a)
            l_vgg.append(l_v)

            depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)  # TODO remove near_far here (frank)

            torch.cuda.empty_cache()

            rgb_gt = (rgb_gt.numpy() * 255).astype("uint8")
            rgb_map = (rgb_map.numpy() * 255).astype("uint8")

            rgb_map = np.concatenate((rgb_gt, rgb_map, depth_map), axis=1)

            img_save_fp = f"{self.save_folder}/{self.iteration}_{postfix}"
            print(f"save to: {img_save_fp}, psnr: {psnr}")
            imageio.imwrite(img_save_fp, rgb_map)
