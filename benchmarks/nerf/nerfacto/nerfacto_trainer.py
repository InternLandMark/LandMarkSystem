# pylint: disable=W0621
import math
import os
import sys

import imageio
import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from benchmarks.nerf.nerfacto.nerfacto import Nerfacto
from landmark.nerf_components.components_convertor import ComponentConvertorFactory
from landmark.nerf_components.configs import (
    BaseConfig,
    get_default_parser,
    parse_console_args,
)
from landmark.nerf_components.data import DatasetManager
from landmark.nerf_components.scene import SceneManager
from landmark.nerf_components.utils.image_utils import (
    mse2psnr_npy,
    visualize_depth_numpy,
)
from landmark.nerf_components.utils.loss_utils import distortion_loss, interlevel_loss
from landmark.train.nerf_trainer import NeRFTrainer
from landmark.train.utils.utils import st


class NerfactoTrainer(NeRFTrainer):
    "landmark.nerf_components.configs.config_parser"

    def __init__(
        self,
        config: BaseConfig,
    ):
        super().__init__(config)
        self.config = config
        self.init_train_env()

        self.scene_mgr = SceneManager(config)
        self.data_mgr = DatasetManager(config)

        self.model = self.create_model()
        self.module = self.model.module if isinstance(self.model, DDP) else self.model

        self.optimizer = self.create_optimizer()

        self.mse_loss = nn.MSELoss()
        self.grad_scaler = GradScaler(enabled=True)

        self.check_args()

    def check_args(self):
        super().check_args()
        config = self.config
        if config.rank == 0:
            assert config.appearance_embedding_size == len(
                self.data_mgr.train_dataset.poses
            ), "embedding size must match with the train dataset size"

    def create_model(self):
        """
        Create model based on args when training.

        Args:
            None

        Returns:
            nerfacto: Nerfacto model.
        """
        config = self.config
        aabb = self.data_mgr.dataset_info.aabb.to(config.device)
        near_far = self.data_mgr.dataset_info.near_far

        if config.ckpt == "auto":
            if config.model_parallel:
                config.ckpt = f"{config.logfolder}/state_dict-sub{config.part}.th"
            else:
                config.ckpt = f"{config.logfolder}/state_dict.th"

        if config.model_parallel and config.DDP:
            ddp_group = config.dp_group
            mp_group = config.mp_group
        else:
            ddp_group = None
            mp_group = None

        state_dict = None
        if config.ckpt is not None:
            kwargs = Nerfacto.get_kwargs_from_ckpt(config.kwargs, config.device)
            kwargs.update({"device": config.device, "config": config})
            kwargs.update({"group": mp_group})
            nerfacto = eval(config.model_name)(**kwargs)  # pylint: disable=W0123
            state_dict = nerfacto.get_state_dict_from_ckpt(config.ckpt, config.device)
            nerfacto.load_from_state_dict(state_dict, False)
            print("load ckpt from", config.ckpt)
        else:
            nerfacto = Nerfacto(  # pylint: disable=W0123
                aabb,
                device=config.device,
                near_far=near_far,
                scene_manager=self.scene_mgr,
                config=config,
                group=mp_group,
            )
        nerfacto = nerfacto.cuda()

        if config.model_parallel:
            convertor = ComponentConvertorFactory.get_convertor("parallelize")
            nerfacto = convertor.convert(nerfacto, config)

        if hasattr(config, "runtime"):
            runtime_convertor = ComponentConvertorFactory.get_convertor("runtime")
            if state_dict is not None:
                nerfacto = runtime_convertor.convert(nerfacto, config, state_dict)
                nerfacto.load_from_state_dict(state_dict, False)
            else:
                nerfacto = runtime_convertor.convert(nerfacto, config)

        if config.DDP and (not config.model_parallel or config.channel_parallel):
            nerfacto = DDP(
                nerfacto, device_ids=[config.local_rank], process_group=ddp_group, find_unused_parameters=True
            )
        print(nerfacto)
        return nerfacto

    def create_optimizer(self):
        """create optimizer.

        Returns:
            optimizer (torch.optim.Optimizer)
        """
        config = self.config
        param_groups = self.module.get_optparam_groups(config.lr)
        optimizer = torch.optim.RAdam(param_groups, eps=config.eps, lr=config.lr)
        if config.optim_dir is not None:
            print("optim_dir is not none, try to load old optimizer")

            if config.model_parallel:
                optim_file = f"{config.optim_dir}/{config.expname}_opt-sub{config.part}.th"
            else:
                optim_file = f"{config.optim_dir}/{config.expname}_opt.th"
            checkpoint = torch.load(optim_file, "cpu")
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"load optimizer from {optim_file}")
        else:
            print("optim_dir is none, try to create new optimizer")
        return optimizer

    def save_optimizer(self, path):
        """
            Save optimizer to a specific path.

        Args:
            path (string): path to save the optimizer.
        """
        state = {"optimizer": self.optimizer.state_dict()}
        torch.save(state, path)

    def train(self):
        torch.cuda.synchronize()
        config = self.config
        self.model.train()

        lr_init = config.lr
        lr_final = config.lr_final

        torch.cuda.empty_cache()
        distortions, psnrs, psnrs_test = [], [], [0]

        pbar = tqdm(
            range(config.start_iters, config.n_iters),
            file=sys.stdout,
        )

        for self.iteration in pbar:
            rays_train, rgb_train, idxs_train = self.data_mgr.get_rays(self.iteration)

            if config.add_lpips > 0 and self.iteration == config.add_lpips:
                self.data_mgr.enable_lpips = True

            self.module.set_anneal(self.iteration)

            with torch.autocast(device_type=self.module.device.type, enabled=True):
                rays_chunk = torch.cat((rays_train, idxs_train.unsqueeze(-1)), dim=-1)
                all_ret = self.model(rays_chunk)

                mse_loss = torch.mean((all_ret["rgb_map"] - rgb_train) ** 2)
                psnr = mse2psnr_npy(mse_loss.item())
                distortion = (
                    distortion_loss(
                        all_ret["weights_list"], all_ret["spacing_starts_list"], all_ret["spacing_ends_list"]
                    )
                    * self.config.distortion_loss_mult
                )
                interlevel = (
                    interlevel_loss(
                        all_ret["weights_list"], all_ret["spacing_starts_list"], all_ret["spacing_ends_list"]
                    )
                    * self.config.interlevel_loss_mult
                )

                loss = mse_loss + distortion + interlevel

            self.grad_scaler.scale(loss).backward()
            if any(any(p.grad is not None for p in g["params"]) for g in self.optimizer.param_groups):
                self.grad_scaler.step(self.optimizer)

            scale = self.grad_scaler.get_scale()
            self.grad_scaler.update()

            if scale <= self.grad_scaler.get_scale():
                for optimizer in self.optimizer.param_groups:
                    if optimizer["name"] == "fields":
                        if self.config.lr_decay_max_steps > 0:
                            lr_decay_max_steps = self.config.lr_decay_max_steps
                        else:
                            lr_decay_max_steps = self.config.n_iters
                        delta_t = np.clip((self.iteration) / (lr_decay_max_steps), 0, 1)
                        lr = np.exp(np.log(lr_init) * (1 - delta_t) + np.log(lr_final) * delta_t)
                        optimizer["lr"] = lr

            self.optimizer.zero_grad()

            psnrs.append(psnr)
            distortions.append(distortion.item())

            if self.iteration % config.progress_refresh_rate == 0:
                datadir = config.datadir.split("/")[-1]
                string = (
                    st.BLUE
                    + f"[{datadir}-{config.partition}][{config.expname}]"
                    + st.RESET
                    + st.YELLOW
                    + f" Iter {self.iteration:06d}:"
                    + st.RESET
                    + st.RED
                    + f" psnr={float(np.mean(psnrs)):.2f}"
                    + st.RESET
                    + st.GREEN
                    + f" test={float(np.mean(psnrs_test)):.2f}"
                    + st.RESET
                    + f" mse={loss:.3f}"
                )

                if config.rank == 0:
                    pbar.set_description(string)
                psnrs = []

            if config.tensorboard and config.rank == 0:
                repre_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("Train/Loss", loss, self.iteration)
                self.writer.add_scalar("Train/PSNR", psnr, self.iteration)
                self.writer.add_scalar("Test/PSNR", psnrs_test[-1], self.iteration)
                self.writer.add_scalar("Train/tf_new_lrate", repre_lr, self.iteration)
                self.writer.add_scalar("Train/distortion", distortion, self.iteration)

            if self.iteration % config.vis_every == config.vis_every - 1 and config.N_vis != 0:
                prtx = f"{self.iteration:06d}_"
                self.model.eval()

                psnrs_test = self.evaluation(
                    self.data_mgr.test_dataset,
                    savePath=f"{self.logfolder}/imgs_vis/",
                    N_vis=config.N_vis,
                    prtx=prtx,
                )
                if config.model_parallel:
                    self.module.save_model(self.logfolder, config.rank)
                    self.save_optimizer(f"{self.logfolder}/{config.expname}_opt-sub{config.rank}.th")
                    dist.barrier()
                else:
                    if config.rank == 0:
                        self.module.save_model(self.logfolder)
                        self.save_optimizer(f"{self.optim_dir}/{config.expname}_opt.th")

                self.model.train()

            if isinstance(self.module.proposal_sampler, DDP):
                self.module.proposal_sampler.module.step_cb(self.iteration)
            else:
                self.module.proposal_sampler.step_cb(self.iteration)

        if config.model_parallel:
            if config.rank < config.model_parallel_degree:
                self.module.save_model(self.logfolder, config.rank)
                self.save_optimizer(f"{self.logfolder}/{config.expname}_opt-sub{config.rank}.th")
                dist.barrier()
                if config.rank == 0:
                    ckpt_fp_list = [
                        f"{self.logfolder}/state_dict-sub{i}.th" for i in range(config.model_parallel_degree)
                    ]
                    kwargs_fp_list = [f"{self.logfolder}/kwargs-sub{i}.th" for i in range(config.model_parallel_degree)]
                    merged_ckpt, merged_kwargs = self.module.merge_ckpts(ckpt_fp_list, kwargs_fp_list)
                    torch.save(merged_ckpt, f"{self.logfolder}/state_dict-merged.th")
                    torch.save(merged_kwargs, f"{self.logfolder}/kwargs-merged.th")
        else:
            if config.rank == 0:
                self.module.save_model(self.logfolder)
                opt_save_path = f"{self.optim_dir}/{config.expname}_opt.th"
                print(f"save optimizer to {opt_save_path}")
                self.save_optimizer(opt_save_path)

        folder = f"{self.logfolder}/imgs_test_all"
        os.makedirs(folder, exist_ok=True)
        psnrs_test = self.evaluation(
            self.data_mgr.test_dataset,
            folder,
            N_vis=-1,
            device=config.device,
        )
        all_psnr = np.mean(psnrs_test)
        print(f"======> {config.expname} test all psnr: {all_psnr} <========================")
        if config.tensorboard and config.rank == 0:
            self.writer.close()
        return all_psnr

    @torch.no_grad()
    def evaluation(
        self,
        test_dataset,
        savePath=None,
        N_vis=5,
        prtx="",
        device="cuda",
    ):
        """
            Generate rays from test dataset. Render images based on them and compute psnr.

        Args:
            test_dataset (Dataset): Dataset used for testing.
            savePath (str): Path where image, video, etc. information is stored.
            N_vis (int): Control the visualization images.
            prtx (str): Prefix of finename.
            device (str): The device on which a tensor is or will be allocated.

        Returns:
            list: A list of PSNR for each image.
        """
        config = self.config
        self.model.eval()

        PSNRs, rgb_maps, depth_maps = [], [], []

        if savePath is not None and self.config.rank == 0:
            os.makedirs(savePath, exist_ok=True)

        tqdm._instances.clear()

        near_far = test_dataset.near_far
        img_eval_interval = max(test_dataset.all_rays.shape[0] // max(N_vis, 1), 1)
        idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))

        render_images = len(test_dataset.all_rays[0::img_eval_interval])
        print("test_dataset render images", render_images)

        for idx, samples in enumerate(tqdm(test_dataset.all_rays[0::img_eval_interval])):
            W, H = test_dataset.img_wh
            rays = samples.view(-1, samples.shape[-1])

            if config.DDP:
                rank_size = (
                    math.ceil(rays.shape[0] / config.num_mp_groups)
                    if config.model_parallel
                    else math.ceil(rays.shape[0] / config.world_size)
                )
                rays_list = rays.split(rank_size)
                rays = rays_list[config.dp_rank]

            dummy_idxs = torch.zeros_like(rays[:, 0], dtype=torch.long).to(device)  # TODO need check
            rays = rays.to(device)

            all_ret = self.module.render_all_rays(
                rays,
                chunk_size=config.render_batch_size,
                idxs=dummy_idxs,
            )

            rgb_map, depth_map = all_ret["rgb_map"], all_ret["depth_map"]
            rgb_map = rgb_map.clamp(0.0, 1.0)

            if config.DDP:
                world_size = config.num_mp_groups if config.model_parallel else config.world_size
                group = config.dp_group if config.model_parallel else None

                rgb_map_all = [
                    torch.zeros((rays_list[i].shape[0], 3), dtype=torch.float32, device=device)
                    for i in range(world_size)
                ]
                depth_map_all = [
                    torch.zeros((rays_list[i].shape[0]), dtype=torch.float32, device=device) for i in range(world_size)
                ]
                dist.all_gather(rgb_map_all, rgb_map, group=group)
                dist.all_gather(depth_map_all, depth_map, group=group)
                rgb_map = torch.cat(rgb_map_all, 0)
                depth_map = torch.cat(depth_map_all, 0)
                all_ret["rgb_map"] = rgb_map.view(-1, 3)
                all_ret["depth_map"] = depth_map.view(-1, 1)

            rgb_map, depth_map = (
                rgb_map.reshape(H, W, 3).cpu(),
                depth_map.reshape(H, W).cpu(),
            )

            if len(test_dataset.all_rgbs):
                path = test_dataset.image_paths[idxs[idx]]
                postfix = path.split("/")[-1]

                gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
                loss = torch.mean((rgb_map - gt_rgb) ** 2)
                psnr = mse2psnr_npy(loss.item())
                PSNRs.append(psnr.item())

                rgb_gt = (gt_rgb.numpy() * 255).astype("uint8")
            else:
                postfix = f"{idx:03d}.png"

            if config.rank == 0:
                print(
                    f"{savePath}{prtx}{postfix}",
                    depth_map.min(),
                    depth_map.max(),
                    near_far,
                    flush=True,
                )

            torch.cuda.empty_cache()
            rgb_map = (rgb_map.numpy() * 255).astype("uint8")
            depth_map, _ = visualize_depth_numpy(depth_map.numpy())

            rgb_maps.append(rgb_map)
            depth_maps.append(depth_map)

            if savePath is not None:
                rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
                if len(test_dataset.all_rgbs):
                    rgb_map = np.concatenate((rgb_gt, rgb_map), axis=1)

                if config.rank == 0:
                    imageio.imwrite(f"{savePath}/{prtx}{postfix}", rgb_map)

        if PSNRs and config.rank == 0 and savePath is not None:
            psnr = np.mean(np.asarray(PSNRs))
            np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray([psnr]))

        return PSNRs


if __name__ == "__main__":
    parser = get_default_parser()
    args, other_args = parser.parse_known_args()
    console_args = parse_console_args(other_args)
    config = BaseConfig.from_file(args.config, console_args)

    trainer = NerfactoTrainer(config)
    trainer.train()
