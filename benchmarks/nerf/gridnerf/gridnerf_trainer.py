# pylint: disable=W0621
import math
import os
import sys

import imageio
import lpips
import numpy as np
import torch
from torch import distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from tqdm.auto import tqdm

from benchmarks.nerf.gridnerf.gridnerf import GridNeRF, cal_n_samples, n_to_reso
from landmark.nerf_components.components_convertor import ComponentConvertorFactory
from landmark.nerf_components.configs import (
    BaseConfig,
    get_default_parser,
    parse_console_args,
)
from landmark.nerf_components.data import DatasetManager, Rays
from landmark.nerf_components.scene import SceneManager
from landmark.nerf_components.utils.image_utils import (
    mse2psnr_npy,
    visualize_depth_numpy,
)
from landmark.nerf_components.utils.loss_utils import rgb_lpips, rgb_ssim
from landmark.train.nerf_trainer import NeRFTrainer
from landmark.train.utils.utils import st


class GridNeRFTrainer(NeRFTrainer):
    """Trainer class for GridNeRF"""

    def __init__(
        self,
        config: BaseConfig,
    ):
        super().__init__(config)
        self.init_train_env()

        self.scene_mgr = SceneManager(config)

        self.data_mgr = DatasetManager(config)
        self.model = self.create_model()
        self.module = self.model.module if isinstance(self.model, DDP) else self.model

        self.optimizer = self.create_optimizer()

        self.check_args()

    def check_args(self):
        super().check_args()
        config = self.config
        if config.rank == 0:
            assert config.appearance_embedding_size == len(
                self.data_mgr.train_dataset.poses
            ), "embedding size must match with the train dataset size"

        if config.DDP:
            assert config.add_lpips == -1, "Do not support lpips in DDP mode"

    def create_model(self):
        config = self.config
        aabb = self.data_mgr.dataset_info.aabb.to(config.device)
        near_far = self.data_mgr.dataset_info.near_far

        if config.model_parallel and config.DDP:
            config.part = config.mp_rank
        else:
            config.part = config.rank

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

        if config.ckpt is not None:
            if config.add_nerf > 0 and config.start_iters is not None and config.start_iters > config.add_nerf:
                config.run_nerf = True
            kwargs = GridNeRF.get_kwargs_from_ckpt(config.kwargs, config.device)
            kwargs.update({"device": config.device, "config": config})
            kwargs.update({"group": mp_group})
            gridnerf = eval(config.model_name)(**kwargs)  # pylint: disable=W0123
            state_dict = gridnerf.get_state_dict_from_ckpt(config.ckpt, config.device)
            gridnerf.load_from_state_dict(state_dict)
            reso_cur = gridnerf.gridSize.tolist()
            print("load ckpt from", config.ckpt)
            print("load kwargs from", config.kwargs)
        else:
            reso_cur = n_to_reso(config.N_voxel_init, aabb)
            gridnerf = GridNeRF(  # pylint: disable=W0123
                aabb,
                reso_cur,
                device=config.device,
                near_far=near_far,
                scene_manager=self.scene_mgr,
                config=config,
            )
        print(gridnerf)

        if config.model_parallel:
            parallelize_convert = ComponentConvertorFactory.get_convertor("parallelize")
            gridnerf = parallelize_convert.convert(gridnerf, config)
            if config.branch_parallel:
                gridnerf.register_grid_ddp_hook()  # TODO move to the parallelize
                if config.add_nerf > 0 or config.run_nerf:
                    gridnerf.register_nerf_ddp_hook()

        if config.DDP and (not config.model_parallel or config.channel_parallel):
            gridnerf = DDP(
                gridnerf, device_ids=[config.local_rank], process_group=ddp_group, find_unused_parameters=True
            )
        print(gridnerf)
        return gridnerf

    def create_optimizer(self):
        """
        create optimizer.

        Returns:
            optimizer (torch.optim.Optimizer)
        """
        config = self.config
        grad_vars = self.module.get_optparam_groups(config.lr_init, config.lr_basis)
        optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        if config.optim_dir is not None:
            print("optim_dir is not none, try to load old optimizer")

            if config.model_parallel and config.DDP:  # TODO(yzy) simplify
                config.part = config.mp_rank
            else:
                config.part = config.rank

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
        config = self.config
        self.model.train()

        grad_scaler = GradScaler(enabled=True)

        enable_lpips = False
        enable_distort = False
        ckpt = config.ckpt

        nsamples = min(config.nSamples, cal_n_samples(self.module.gridSize.tolist(), config.step_ratio))
        if config.rank == 0:
            print(self.module.gridSize.tolist(), nsamples)

        if config.lr_decay_iters > 0:
            lr_factor = config.lr_decay_target_ratio ** (1 / config.lr_decay_iters)
        else:
            config.lr_decay_iters = config.n_iters
            lr_factor = config.lr_decay_target_ratio ** (1 / config.n_iters)

        # create optimizer
        update_alphamask_list = config.update_AlphaMask_list

        # upsaple list
        upsamp_list = config.upsamp_list
        n_voxel_list = (
            torch.round(
                torch.exp(
                    torch.linspace(
                        np.log(config.N_voxel_init),
                        np.log(config.N_voxel_final),
                        len(upsamp_list) + 1,
                    )
                )
            ).long()
        ).tolist()[1:]
        print(f"upsamp_list: {upsamp_list}")
        print(f"n_voxel_list: {n_voxel_list}")

        torch.cuda.empty_cache()
        psnrs, psnrs_nerf, psnrs_test = [], [], [0]

        ortho_reg_weight = config.Ortho_weight
        l1_reg_weight = config.L1_weight_inital
        tv_weight_density, tv_weight_app = config.TV_weight_density, config.TV_weight_app

        if enable_lpips or config.add_lpips > 0:
            lp_loss = lpips.LPIPS(net="vgg").to(config.device)

        pbar = tqdm(range(config.start_iters, config.n_iters), file=sys.stdout)

        for self.iteration in pbar:
            rays_train, rgb_train, idxs_train = self.data_mgr.get_rays(self.iteration)
            if config.add_distort > 0 and self.iteration == config.add_distort:
                enable_distort = True

            if config.add_lpips > 0 and self.iteration == config.add_lpips:
                enable_lpips = True
                self.data_mgr.enable_lpips = True

            with torch.autocast(device_type=self.module.device.type, enabled=True):
                rays_chunk = torch.cat((rays_train, idxs_train.unsqueeze(-1)), dim=-1).to(config.device)

                rays_chunk = Rays(
                    origin=None,
                    dirs=None,
                    camera_idx=None,
                    near=self.config.train_near_far[0],
                    far=self.config.train_near_far[1],
                    _data=rays_chunk,
                    rank=self.config.mp_rank,
                    group=self.config.mp_group,
                )

                all_ret = self.model(rays_chunk, N_samples=nsamples)

                rgb_map = all_ret["rgb_map"]
                loss = torch.mean((rgb_map - rgb_train) ** 2)
                total_loss = loss

                # additional rgb supervision
                if "rgb_map1" in all_ret:
                    rgb_map1 = all_ret["rgb_map1"]
                    loss1 = torch.mean((rgb_map1 - rgb_train) ** 2)
                    total_loss += loss1
                if self.module.run_nerf:
                    nerf_loss = torch.mean((all_ret["rgb_map_nerf"] - rgb_train) ** 2)
                    total_loss = total_loss + nerf_loss

                # regularization loss
                if ortho_reg_weight > 0:
                    total_loss += (
                        ortho_reg_weight * self.module.density_encoding.vector_diffs()
                        + self.module.rgb_encoding.vector_diffs()
                    )
                if l1_reg_weight > 0:
                    total_loss += l1_reg_weight * self.module.density_encoding.L1_loss()
                if tv_weight_density > 0:
                    tv_weight_density *= lr_factor
                    total_loss += tv_weight_density * self.module.density_encoding.TV_loss()
                if tv_weight_app > 0:
                    tv_weight_app *= lr_factor
                    total_loss += tv_weight_app * self.module.rgb_encoding.TV_loss()

                # lpips loss
                if enable_lpips:
                    lpips_w = 0.01
                    ps = config.patch_size
                    batch_sample_target_s = torch.reshape(rgb_train, [-1, ps, ps, 3]).permute(0, 3, 1, 2).clamp(0, 1)
                    ##
                    batch_sample_fake = torch.reshape(rgb_map, [-1, ps, ps, 3]).permute(0, 3, 1, 2).clamp(0, 1)
                    lpips_loss = torch.mean(lp_loss.forward(batch_sample_fake * 2 - 1, batch_sample_target_s * 2 - 1))
                    total_loss = total_loss + lpips_loss * lpips_w

                    if self.module.run_nerf:
                        batch_sample_fake_nerf = (
                            torch.reshape(all_ret["rgb_map_nerf"], [-1, ps, ps, 3]).permute(0, 3, 1, 2).clamp(0, 1)
                        )
                        lpips_loss_nerf = torch.mean(
                            lp_loss.forward(batch_sample_fake_nerf * 2 - 1, batch_sample_target_s * 2 - 1)
                        )
                        total_loss = total_loss + lpips_loss_nerf * lpips_w

                if enable_distort:
                    total_loss += torch.mean(all_ret["distort_loss"])
                    if "distort_loss1" in all_ret:
                        total_loss += torch.mean(all_ret["distort_loss1"])
                    if "distort_loss_nerf" in all_ret:
                        total_loss += torch.mean(all_ret["distort_loss_nerf"])

            grad_scaler.scale(total_loss).backward()
            grad_scaler.step(self.optimizer)
            grad_scaler.update()

            self.optimizer.zero_grad()

            loss = torch.mean((rgb_map - rgb_train) ** 2).detach().item()
            psnr = mse2psnr_npy(loss)
            psnrs.append(psnr)

            if enable_lpips:
                lpips_loss = lpips_loss.detach().item()

            if enable_distort:
                distort_loss = torch.mean(all_ret["distort_loss"]).detach().item()

            if self.module.run_nerf:
                nerf_loss = nerf_loss.detach().item()
                nerf_psnr = mse2psnr_npy(nerf_loss)
                psnrs_nerf.append(nerf_psnr)

            if "rgb_map1" in all_ret:
                loss1 = loss1.detach().item()
                psnr1 = mse2psnr_npy(loss1)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * lr_factor

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
                if enable_lpips:
                    string += f" lpips = {lpips_loss:.2f}"
                if enable_distort:
                    string += f" distort = {distort_loss:.10f}"
                if self.module.run_nerf:
                    string += f" nerf_psnr = {float(np.mean(psnrs_nerf)):.2f}"
                if "rgb_map1" in all_ret:
                    string += f" psnr1 = {psnr1:.2f}"

                if config.rank == 0:
                    pbar.set_description(string)
                psnrs = []
                psnrs_nerf = []

            if config.tensorboard and config.rank == 0:
                repre_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("Train/Loss", loss, self.iteration)
                self.writer.add_scalar("Train/PSNR", psnr, self.iteration)
                self.writer.add_scalar("Test/PSNR", psnrs_test[-1], self.iteration)
                self.writer.add_scalar("Train/tf_new_lrate", repre_lr, self.iteration)
                if enable_lpips:
                    self.writer.add_scalar("Train/LPIPSLoss", lpips_loss, self.iteration)
                if self.module.run_nerf:
                    self.writer.add_scalar("Train/PSNR_nerf", nerf_psnr, self.iteration)

            if self.iteration % config.vis_every == config.vis_every - 1 and config.N_vis != 0:
                prtx = f"{self.iteration:06d}_"
                self.model.eval()

                psnrs_test = self.evaluation(
                    self.data_mgr.test_dataset,
                    config,
                    f"{self.logfolder}/imgs_vis/",
                    N_vis=config.N_vis,
                    prtx=prtx,
                    N_samples=nsamples,
                    compute_extra_metrics=False,
                )

                if config.model_parallel:
                    self.module.save_model(self.logfolder, config.rank)
                    print("render saved to", f"{self.logfolder}/imgs_vis/")

                    opt_save_path = f"{self.optim_dir}/{config.expname}_opt-sub{config.rank}.th"
                    print(f"optimizer save to {self.optim_dir}")
                    self.save_optimizer(opt_save_path)

                    for i in [0]:
                        save_image(
                            self.module.density_encoding.plane_coef[i].squeeze(2).permute(1, 0, 2, 3),
                            f"{self.logfolder}/den_plane_{i}_sub{config.rank}.png",
                        )
                        save_image(
                            self.module.rgb_encoding.plane_coef[i].squeeze(2).permute(1, 0, 2, 3),
                            f"{self.logfolder}/app_plane_{i}_sub{config.rank}.png",
                        )
                else:
                    if config.rank == 0:
                        self.module.save_model(self.logfolder)
                        print("render saved to", f"{self.logfolder}/imgs_vis/")

                        opt_save_path = f"{self.optim_dir}/{config.expname}_opt.th"
                        print(f"optimizer save to {self.optim_dir}")
                        self.save_optimizer(opt_save_path)

                        for i in [0]:
                            save_image(
                                self.module.density_encoding.plane_coef[i].squeeze(2).permute(1, 0, 2, 3),
                                f"{self.logfolder}/den_plane_{i}.png",
                            )
                            save_image(
                                self.module.rgb_encoding.plane_coef[i].squeeze(2).permute(1, 0, 2, 3),
                                f"{self.logfolder}/app_plane_{i}.png",
                            )

                self.model.train()

            if not ckpt and not self.module.run_nerf:
                increase_alpha_thresh = 0
                if self.iteration in update_alphamask_list:
                    reso_cur = self.module.gridSize.tolist()
                    if reso_cur[0] * reso_cur[1] * reso_cur[2] < config.alpha_grid_reso**3:
                        reso_mask = reso_cur

                    self.module.updateAlphaMask(tuple(reso_mask), increase_alpha_thresh)
                    if config.progressive_alpha:
                        increase_alpha_thresh += 1

            if self.iteration in upsamp_list:
                if not ckpt:
                    n_voxels = n_voxel_list.pop(0)
                else:
                    for it in upsamp_list:
                        if self.iteration >= it:
                            n_voxels = n_voxel_list.pop(0)
                    ckpt = None
                reso_cur = n_to_reso(n_voxels, self.module.aabb)
                nsamples = min(config.nSamples, cal_n_samples(reso_cur, config.step_ratio))
                self.module.upsample_volume_grid(reso_cur)

                if config.lr_upsample_reset:
                    if config.rank == 0:
                        print(st.CYAN + "reset lr to initial" + st.RESET)
                    lr_scale = 1
                else:
                    lr_scale = config.lr_decay_target_ratio ** (self.iteration / config.n_iters)
                if config.DDP and (not config.model_parallel or config.channel_parallel):
                    if config.channel_parallel:
                        ddp_group = config.dp_group
                    else:
                        ddp_group = None
                    self.model = DDP(
                        self.module,
                        device_ids=[config.local_rank],
                        process_group=ddp_group,
                        find_unused_parameters=True,
                    )
                    self.module = self.model.module if isinstance(self.model, DDP) else self.model
                grad_vars = self.module.get_optparam_groups(config.lr_init * lr_scale, config.lr_basis * lr_scale)
                self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

            if config.add_nerf > 0 and self.iteration == config.add_nerf:
                self.module.run_nerf = True
                if config.lr_upsample_reset:
                    if config.rank == 0:
                        print(st.CYAN + "reset lr to initial" + st.RESET)
                    lr_scale = 1
                else:
                    lr_scale = config.lr_decay_target_ratio ** (self.iteration / config.n_iters)
                if config.DDP and (not config.model_parallel or config.channel_parallel):
                    if config.channel_parallel:
                        ddp_group = config.dp_group
                    else:
                        ddp_group = None
                    self.model = DDP(
                        self.module,
                        device_ids=[config.local_rank],
                        process_group=ddp_group,
                        find_unused_parameters=True,
                    )
                    self.module = self.model.module if isinstance(self.model, DDP) else self.model
                grad_vars = self.module.get_optparam_groups(config.lr_init * lr_scale, config.lr_basis * lr_scale)
                if config.rank == 0:
                    print("reload grad_vars")
                    print(grad_vars[-1])
                self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        if config.model_parallel:
            if config.rank < config.model_parallel_degree:
                print(f"saving model to {self.logfolder}")
                self.module.save_model(self.logfolder, config.rank)
                print(f"saving optimizer to {self.optim_dir}/{config.expname}_opt-sub{config.rank}.th")
                self.save_optimizer(f"{self.optim_dir}/{config.expname}_opt-sub{config.rank}.th")
                dist.barrier()

                if config.rank == 0:
                    print(f"merging ckpts to {self.logfolder}/state_dict-merged.th")
                    ckpt_fp_list = [
                        f"{self.logfolder}/state_dict-sub{i}.th" for i in range(config.model_parallel_degree)
                    ]
                    kwargs_fp_list = [f"{self.logfolder}/kwargs-sub{i}.th" for i in range(config.model_parallel_degree)]
                    merged_ckpt, merged_kwargs = self.module.merge_ckpts(ckpt_fp_list, kwargs_fp_list)
                    torch.save(merged_ckpt, f"{self.logfolder}/state_dict-merged.th")
                    torch.save(merged_kwargs, f"{self.logfolder}/kwargs-merged.th")
        else:
            if config.rank == 0:
                print(f"saving model to {self.logfolder}/state_dict.th")
                self.module.save_model(self.logfolder)
                print(f"saving optimizer to {self.optim_dir}/{config.expname}_opt.th")
                self.save_optimizer(f"{self.optim_dir}/{config.expname}_opt.th")

        folder = f"{self.logfolder}/imgs_test_all"
        os.makedirs(folder, exist_ok=True)
        self.model.eval()
        psnrs_test = self.evaluation(
            self.data_mgr.test_dataset,
            config,
            folder,
            N_vis=config.N_vis,
            N_samples=-1,
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
        config,
        savePath=None,
        N_vis=5,
        prtx="",
        N_samples=-1,
        compute_extra_metrics=True,
        device="cuda",
    ):
        """
        Generate rays from the test dataset and render images based on them.
        Compute the peak signal-to-noise ratio (PSNR) for each image.

        Args:
            test_dataset (Dataset): The dataset used for testing.
            config (BaseConfig): The configuration object.
            savePath (str, optional): The path where image, video, etc. information is stored. Defaults to None.
            N_vis (int, optional): The number of visualization images. Defaults to 5.
            prtx (str, optional): The prefix of the filename. Defaults to "".
            N_samples (int, optional): The number of sample points along a ray in total. Defaults to -1.
            compute_extra_metrics (bool, optional): Whether to compute SSIM and LPIPS metrics. Defaults to True.
            device (str, optional): The device on which a tensor is or will be allocated. Defaults to "cuda".

        Returns:
            list: A list of PSNR values for each image.
        """
        self.model.eval()

        PSNRs, rgb_maps, depth_maps = [], [], []
        ssims, l_alex, l_vgg = [], [], []

        if savePath is not None and config.rank == 0:
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
                N_samples=N_samples,
                idxs=dummy_idxs,
            )

            rgb_map, depth_map = all_ret["rgb_map"], all_ret["depth_map"]
            rgb_map = rgb_map.clamp(0.0, 1.0)

            if config.DDP:
                if config.model_parallel:
                    world_size = config.num_mp_groups
                    group = config.dp_group
                else:
                    world_size = config.world_size
                    group = None
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

            rgb_map, depth_map = (
                rgb_map.reshape(H, W, 3).cpu(),
                depth_map.reshape(H, W).cpu(),
            )

            if len(test_dataset.all_rgbs):
                path = test_dataset.image_paths[idxs[idx]]
                postfix = path.split("/")[-1]

                gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
                loss = torch.mean((rgb_map - gt_rgb) ** 2)
                PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

                if compute_extra_metrics:
                    ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                    l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "alex", config.device)
                    l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", config.device)
                    ssims.append(ssim)
                    l_alex.append(l_a)
                    l_vgg.append(l_v)

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

            depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

            if self.module.run_nerf:
                rgb_map_nerf, depth_map_nerf = (
                    all_ret["rgb_map_nerf"],
                    all_ret["depth_map_nerf"],
                )

                if config.DDP:
                    if config.model_parallel:
                        world_size = config.num_mp_groups
                        group = config.dp_group
                    else:
                        world_size = config.world_size
                        group = None
                    rgb_map_nerf_all = [
                        torch.zeros((rays_list[i].shape[0], 3), dtype=torch.float32, device=device)
                        for i in range(world_size)
                    ]
                    depth_map_nerf_all = [
                        torch.zeros((rays_list[i].shape[0]), dtype=torch.float32, device=device)
                        for i in range(world_size)
                    ]
                    dist.all_gather(rgb_map_nerf_all, rgb_map_nerf, group=group)
                    dist.all_gather(depth_map_nerf_all, depth_map_nerf, group=group)
                    rgb_map_nerf = torch.cat(rgb_map_nerf_all, 0)
                    depth_map_nerf = torch.cat(depth_map_nerf_all, 0)

                rgb_map_nerf, depth_map_nerf = (
                    rgb_map_nerf.reshape(H, W, 3).cpu(),
                    depth_map_nerf.reshape(H, W).cpu(),
                )
                depth_map_nerf, _ = visualize_depth_numpy(depth_map_nerf.numpy(), near_far)
                if len(test_dataset.all_rgbs):
                    loss_nerf = torch.mean((rgb_map_nerf - gt_rgb) ** 2)
                    if config.rank == 0:
                        print("psnr", -10.0 * np.log(loss_nerf.item()) / np.log(10.0))
            torch.cuda.empty_cache()
            rgb_map = (rgb_map.numpy() * 255).astype("uint8")

            rgb_maps.append(rgb_map)
            depth_maps.append(depth_map)

            if savePath is not None:
                rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
                if len(test_dataset.all_rgbs):
                    rgb_map = np.concatenate((rgb_gt, rgb_map), axis=1)

                if self.module.run_nerf:
                    rgb_map_nerf = (rgb_map_nerf.numpy() * 255).astype("uint8")
                    rgb_map_nerf = np.concatenate((rgb_map_nerf, depth_map_nerf), axis=1)
                    rgb_map = np.concatenate((rgb_map, rgb_map_nerf), axis=1)

                if config.rank == 0:
                    imageio.imwrite(f"{savePath}/{prtx}{postfix}", rgb_map)

        if PSNRs and config.rank == 0 and savePath is not None:
            psnr = np.mean(np.asarray(PSNRs))
            if compute_extra_metrics:
                ssim = np.mean(np.asarray(ssims))
                l_a = np.mean(np.asarray(l_alex))
                l_v = np.mean(np.asarray(l_vgg))
                np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray([psnr, ssim, l_a, l_v]))
            else:
                np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray(PSNRs))

        return PSNRs


if __name__ == "__main__":
    parser = get_default_parser()
    args, other_args = parser.parse_known_args()
    console_args = parse_console_args(other_args)
    config = BaseConfig.from_file(args.config, console_args)

    trainer = GridNeRFTrainer(config)
    trainer.train()
