# pylint: disable=W0621
import math
import os

import imageio
import numpy as np
import torch
from tqdm import tqdm

from benchmarks.nerf.octree_gs.gs_model import OctreeGS
from landmark.nerf_components.configs import (
    BaseConfig,
    get_default_parser,
    parse_console_args,
)
from landmark.nerf_components.data.data_manager import DatasetManager
from landmark.nerf_components.scene import SceneManager
from landmark.nerf_components.utils.general_utils import get_expon_lr_func
from landmark.nerf_components.utils.image_utils import psnr
from landmark.nerf_components.utils.loss_utils import l1_loss, ssim
from landmark.nerf_components.utils.ply_utils import get_pcd
from landmark.nerf_components.utils.system_utils import searchForMaxIteration
from landmark.train.nerf_trainer import NeRFTrainer


class OctreeGSTrainer(NeRFTrainer):
    """Class for OctreeGS Training"""

    def __init__(
        self,
        config: BaseConfig,
    ):
        super().__init__(config)
        self.config = config
        self.aabb = np.array([config.lb, config.ub])
        self.init_train_env()

        self.check_args()

        self.scene_mgr = SceneManager(config)

        self.data_mgr = DatasetManager(config)

        self.model = self.create_model()

        self.optimizer = self.create_optimizer()

    def create_model(self):
        config = self.config
        gaussians = OctreeGS(config)

        loaded_iter = None
        load_iteration = None
        if load_iteration:
            if load_iteration == -1:
                loaded_iter = searchForMaxIteration(os.path.join(config.logfolder, "point_cloud"))
            else:
                loaded_iter = load_iteration
            print(f"Loading trained model at iteration {loaded_iter}")

        pcd = get_pcd(loaded_iter, config, self.aabb, self.logfolder, gaussians)

        if not load_iteration:
            print("setting appearance embedding...")
            gaussians.set_appearance(self.data_mgr.train_data_size)

            points = torch.tensor(pcd.points[:: config.ratio]).float().cuda()
            unique_points = torch.unique(points, dim=0)
            print("setting LOD levels...")
            gaussians.set_level(unique_points, self.data_mgr.train_dataset.cams)
            print("setting progressive intervals...")
            gaussians.set_coarse_interval()
            # Initilize octree on cuda
            print("setting initial octree...")
            pcd._replace(points=unique_points.cpu().numpy())
            gaussians.create_from_pcd(pcd, self.data_mgr.train_dataset.cameras_extent)
            self.spatial_lr_scale = self.data_mgr.train_dataset.cameras_extent

        if config.start_checkpoint:
            state_dict = gaussians.get_state_dict_from_ckpt(config.start_checkpoint, config.device)
            gaussians.load_from_state_dict(state_dict)

        return gaussians

    def create_optimizer(self):
        config = self.config
        param_groups = self.model.get_optparam_groups(config)

        optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(
            lr_init=config.position_lr_init * self.spatial_lr_scale,
            lr_final=config.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=config.position_lr_delay_mult,
            max_steps=config.position_lr_max_steps,
        )
        self.offset_scheduler_args = get_expon_lr_func(
            lr_init=config.offset_lr_init * self.spatial_lr_scale,
            lr_final=config.offset_lr_final * self.spatial_lr_scale,
            lr_delay_mult=config.offset_lr_delay_mult,
            max_steps=config.offset_lr_max_steps,
        )

        self.mlp_opacity_scheduler_args = get_expon_lr_func(
            lr_init=config.mlp_opacity_lr_init,
            lr_final=config.mlp_opacity_lr_final,
            lr_delay_mult=config.mlp_opacity_lr_delay_mult,
            max_steps=config.mlp_opacity_lr_max_steps,
        )

        self.mlp_cov_scheduler_args = get_expon_lr_func(
            lr_init=config.mlp_cov_lr_init,
            lr_final=config.mlp_cov_lr_final,
            lr_delay_mult=config.mlp_cov_lr_delay_mult,
            max_steps=config.mlp_cov_lr_max_steps,
        )

        self.mlp_color_scheduler_args = get_expon_lr_func(
            lr_init=config.mlp_color_lr_init,
            lr_final=config.mlp_color_lr_final,
            lr_delay_mult=config.mlp_color_lr_delay_mult,
            max_steps=config.mlp_color_lr_max_steps,
        )
        if config.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(
                lr_init=config.mlp_featurebank_lr_init,
                lr_final=config.mlp_featurebank_lr_final,
                lr_delay_mult=config.mlp_featurebank_lr_delay_mult,
                max_steps=config.mlp_featurebank_lr_max_steps,
            )
        if self.config.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(
                lr_init=config.appearance_lr_init,
                lr_final=config.appearance_lr_final,
                lr_delay_mult=config.appearance_lr_delay_mult,
                max_steps=config.appearance_lr_max_steps,
            )

        if config.start_checkpoint:
            opt_dir = os.path.dirname(config.start_checkpoint)
            self.opt_dict = torch.load(os.path.join(opt_dir, "opt.th"))
            optimizer.load_state_dict(self.opt_dict)
        return optimizer

    def update_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(self.iteration)
                param_group["lr"] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(self.iteration)
                param_group["lr"] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(self.iteration)
                param_group["lr"] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(self.iteration)
                param_group["lr"] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(self.iteration)
                param_group["lr"] = lr
            if self.config.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(self.iteration)
                param_group["lr"] = lr
            if self.config.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(self.iteration)
                param_group["lr"] = lr

    def train(self):
        config = self.config

        self.model.train()
        torch.cuda.synchronize()

        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(config.start_iters + 1, config.n_iters + 1), desc="Training progress")
        for self.iteration in progress_bar:
            # get next training data
            viewpoint_cam, gt_image = self.data_mgr.get_camera("train")
            viewpoint_cam, gt_image = viewpoint_cam.cuda(), gt_image.cuda()

            gt_image = gt_image.float() / 255.0 if config.img_uint8 else gt_image

            # update learning rate
            self.update_learning_rate()

            # render
            self.model.set_anchor_mask(viewpoint_cam.camera_center, self.iteration, viewpoint_cam.resolution_scale)
            retain_grad = 0 <= self.iteration < config.update_until
            image, scaling = self.model(viewpoint_cam, retain_grad=retain_grad)

            # Loss
            Ll1 = l1_loss(image, gt_image)
            Lssim = 1.0 - ssim(image, gt_image)
            if scaling.shape[0] > 0:
                scaling_reg = scaling.prod(dim=1).mean()
            else:
                scaling_reg = torch.tensor(0.0, device="cuda")
            loss = (1.0 - config.lambda_dssim) * Ll1 + config.lambda_dssim * Lssim + 0.01 * scaling_reg
            loss.backward()

            ps = psnr(image, gt_image).mean()

            with torch.no_grad():
                # progress bar
                if math.isnan(ema_loss_for_log):
                    ema_loss_for_log = loss.item()
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

                # Densification
                self.model.densification(self.optimizer, self.iteration)

                # optimzer step
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # saving iter
                if self.iteration in config.save_iterations:
                    print(f"\n[ITER {self.iteration}] Saving Checkpoint")
                    logfolder = os.path.join(self.logfolder, "point_cloud", "iteration_" + str(self.iteration))
                    self.model.save_model(logfolder)
                    torch.save(
                        self.optimizer.state_dict(),
                        os.path.join(logfolder, "opt.th"),
                    )

                # testing iter
                if self.iteration in config.test_iterations:
                    print(f"\n[ITER {self.iteration}] Evaluating")
                    ps = self.evaluation(os.path.join(config.logfolder, "imgs_vis"))
                    print(f"Test PSNR: {ps:.2f}")
                    self.model.train()

                # progress bar
                if self.iteration % 10 == 0:
                    primitives_num = self.model.get_primitives_num
                    postfix = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "psnr": f"{ps.item():.2f}",
                        "Num": f"{primitives_num}",
                    }
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)

                    if self.writer and config.rank == 0:
                        self.writer.add_scalar("train_loss/l1_loss", Ll1.item(), self.iteration)
                        self.writer.add_scalar("train_loss/total_loss", loss.item(), self.iteration)
                        self.writer.add_scalar("train_psnr", ps.item(), self.iteration)
                        self.writer.add_scalar("primitives_num", primitives_num, self.iteration)
                        for param_group in self.optimizer.param_groups:
                            if param_group["name"] == "anchor":
                                self.writer.add_scalar("anchor_lr", param_group["lr"], self.iteration)
                        self.writer.add_scalar(
                            "gpu/cur_memory_allocated", torch.cuda.memory_allocated() / 1024**3, self.iteration
                        )

        # save model
        logfolder = os.path.join(self.logfolder, "point_cloud", "iteration_" + str(self.iteration))
        self.model.save_model(logfolder)
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(logfolder, "opt.th"),
        )

        # evaluation
        ps = self.evaluation(os.path.join(config.logfolder, "imgs_test_all"))
        return ps

    @torch.no_grad()
    def evaluation(self, save_path):
        self.model.eval()
        self.model.plot_levels()
        PSNRs = []
        os.makedirs(save_path, exist_ok=True)

        tqdm._instances.clear()

        for idx in tqdm(range(0, self.data_mgr.test_data_size, 20), desc="Evaluation progress"):
            viewpoint_cam, gt_image = self.data_mgr.get_camera("test")
            viewpoint_cam, gt_image = viewpoint_cam.cuda(), gt_image.cuda()

            gt_image = gt_image.float() / 255.0 if self.config.img_uint8 else gt_image

            image = self.model(viewpoint_cam)

            image = image.clamp(0.0, 1.0)

            ps = psnr(image, gt_image).mean()
            PSNRs.append(ps.item())

            gt_image = (gt_image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            img = np.concatenate((gt_image, image), axis=1)

            imageio.imwrite(f"{save_path}/{idx:03d}.png", img)

        if PSNRs and self.config.rank == 0:
            ps = np.mean(np.asarray(PSNRs))

        torch.cuda.empty_cache()
        return ps


if __name__ == "__main__":
    parser = get_default_parser()
    args, other_args = parser.parse_known_args()
    console_args = parse_console_args(other_args)
    config = BaseConfig.from_file(args.config, console_args)

    trainer = OctreeGSTrainer(config)
    trainer.train()
