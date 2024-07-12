# pylint: disable=W0621
import math
import os

import imageio
import numpy as np
import torch
from tqdm import tqdm

from benchmarks.nerf.origin_gs.gs_model import GaussianModel
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


class Gaussian3DTrainer(NeRFTrainer):
    """Class for 3D Gaussian Splatting Training"""

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
        gaussians = GaussianModel(config=config, cam_extent=self.data_mgr.train_dataset.cameras_extent)

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
            gaussians.create_from_pcd(pcd, gaussians.cam_extent)
            self.spatial_lr_scale = self.data_mgr.train_dataset.cameras_extent

        if config.start_checkpoint:
            state_dict = gaussians.get_state_dict_from_ckpt(config.start_checkpoint, config.device)
            gaussians.load_from_state_dict(state_dict)

        return gaussians

    def create_optimizer(self):
        config = self.config
        param_groups = self.model.get_optparam_groups(config)
        optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=config.position_lr_init * self.spatial_lr_scale,
            lr_final=config.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=config.position_lr_delay_mult,
            max_steps=config.position_lr_max_steps,
        )
        if config.start_checkpoint:
            opt_dir = os.path.dirname(config.start_checkpoint)
            self.opt_dict = torch.load(os.path.join(opt_dir, "opt.th"))
            optimizer.load_state_dict(self.opt_dict)
        return optimizer

    def update_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(self.iteration)
                param_group["lr"] = lr
                break

    def train(self):
        config = self.config

        self.model.train()

        ema_loss_for_log = 0.0
        pbar = tqdm(range(config.start_iters + 1, config.n_iters + 1), desc="Training progress")
        for self.iteration in pbar:
            # get next training data
            viewpoint_cam, gt_image = self.data_mgr.get_camera("train")
            viewpoint_cam, gt_image = viewpoint_cam.cuda(), gt_image.cuda()

            gt_image = gt_image.float() / 255.0 if config.img_uint8 else gt_image

            # update learning rate
            self.update_learning_rate()

            if self.iteration % self.data_mgr.train_data_size == 0:
                self.model.oneupSHdegree()

            # render
            image = self.model(viewpoint_cam)

            # Loss
            Ll1 = l1_loss(image, gt_image)
            Lssim = 1.0 - ssim(image, gt_image)
            loss = (1.0 - config.lambda_dssim) * Ll1 + config.lambda_dssim * Lssim
            loss.backward()

            ps = psnr(image, gt_image).mean()

            with torch.no_grad():
                # progress bar
                if math.isnan(ema_loss_for_log):
                    ema_loss_for_log = loss.item()
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

                # Densification
                self.model.densification(self.optimizer, self.iteration)

                # optimizer step
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
                    pbar.set_postfix(postfix)
                    pbar.update(10)

                    if self.writer and config.rank == 0:
                        self.writer.add_scalar("train_loss/l1_loss", Ll1.item(), self.iteration)
                        self.writer.add_scalar("train_loss/total_loss", loss.item(), self.iteration)
                        self.writer.add_scalar("train_psnr", ps.item(), self.iteration)
                        self.writer.add_scalar("primitives_num", primitives_num, self.iteration)
                        for param_group in self.optimizer.param_groups:
                            if param_group["name"] == "xyz":
                                self.writer.add_scalar("xyz_lr", param_group["lr"], self.iteration)
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

    trainer = Gaussian3DTrainer(config)
    trainer.train()
