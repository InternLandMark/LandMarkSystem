# pylint: disable=W0621
import math
import os

import imageio
import numpy as np
import torch
from torch import distributed as dist
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
from landmark.nerf_components.utils.image_utils import visualize_depth_numpy
from landmark.nerf_components.utils.loss_utils import rgb_lpips, rgb_ssim
from landmark.train.nerf_evaluator import NeRFEvaluator


class NerfactoEvaluator(NeRFEvaluator):
    """Evaluator Class for Nerfacto"""

    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self.init_render_env()

        self.scene_mgr = SceneManager(config)
        self.data_mgr = DatasetManager(config)
        self.model = self.create_model()

        self.check_args()

    def check_args(self):
        super().check_args()
        config = self.config

        if config.branch_parallel:
            assert config.ckpt is not None and ".th" in config.ckpt, f"Error, config.ckpt is incorrect: {config.ckpt}."
            # check whether loading the stacked-merged ckpt, since the concat-merged ckpt won't be supported.
            assert "stack" in config.ckpt if "merged" in config.ckpt else True, (
                f"Error, you may load the concated-merged ckpt: {config.ckpt}, which is deprecated and won't be"
                " supported. Please load the ckpt with suffix 'merged-stack.th'"
            )

    def create_model(self):
        config = self.config

        if config.ckpt_type == "sub":
            config.part = config.rank

        if config.ckpt == "auto":
            if config.model_parallel:
                if config.ckpt_type == "sub":
                    config.ckpt = f"{config.logfolder}/{config.expname}-sub{config.part}.th"
                    config.part = config.rank
                elif config.ckpt_type == "full":
                    if config.branch_parallel:
                        config.ckpt = f"{config.logfolder}/{config.expname}-merged-stack.th"
                    else:
                        config.ckpt = f"{config.logfolder}/{config.expname}-merged.th"
                elif config.branch_parallel and config.ckpt_type == "part":
                    raise NotImplementedError
                else:
                    raise Exception(
                        "Don't known how to load checkpoints, please check the config.ckpt and config.ckpt_type configs"
                        " setting."
                    )
            else:
                config.ckpt = f"{config.logfolder}/{config.expname}.th"

        def kwargs_tensors_to_device(kwargs, device):
            # move the tensors in kwargs to target device
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    kwargs_tensors_to_device(value, device)
                elif isinstance(value, torch.Tensor):
                    kwargs[key] = value.to(device)

        print("load ckpt from", config.ckpt)
        kwargs = Nerfacto.get_kwargs_from_ckpt(config.kwargs, config.device)
        kwargs["config"] = config
        kwargs["scene_manager"] = self.scene_mgr

        nerfacto = Nerfacto(**kwargs)  # pylint: disable=W0123
        nerfacto = nerfacto.cuda()

        state_dict = nerfacto.get_state_dict_from_ckpt(config.ckpt, config.device)

        nerfacto.load_from_state_dict(state_dict, False)

        if hasattr(config, "runtime"):
            runtime_convertor = ComponentConvertorFactory.get_convertor("runtime")
            nerfacto = runtime_convertor.convert(nerfacto, config, state_dict)
        nerfacto.load_from_state_dict(state_dict, False)

        print(nerfacto)

        nerfacto.eval()
        return nerfacto

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
        """
            Generate rays from test dataset. Render images based on them and compute psnr.

        Args:
            test_dataset (Dataset): Dataset used for testing.
            config (tools.configs.ArgsConfig): Evaluation configs.
            savePath (str): Path where image, video, etc. information is stored.
            N_vis (int): Control the visualization images.
            prtx (str): Prefix of finename.
            compute_extra_metrics (bool): Compute extra metrics.
            device (str): The device on which a tensor is or will be allocated.

        Returns:
            list: A list of PSNR for each image.
        """
        self.model.eval()

        PSNRs, rgb_maps, depth_maps = [], [], []
        ssims, l_alex, l_vgg = [], [], []

        if savePath is not None:
            os.makedirs(savePath, exist_ok=True)

        tqdm._instances.clear()

        near_far = test_dataset.near_far
        img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
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

            dummy_idxs = torch.zeros_like(rays[:, 0], dtype=torch.long, device=device)
            rays = rays.to(device)
            all_ret = self.model.render_all_rays(
                rays,
                chunk_size=config.render_batch_size,
                N_samples=N_samples,
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

            rgb_map, depth_map = (
                rgb_map.reshape(H, W, 3).cpu(),
                depth_map.reshape(H, W).cpu(),
            )

            if len(test_dataset.all_rgbs):
                path = test_dataset.image_paths[idxs[idx]]
                postfix = path.split("/")[-1]

                gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)

                loss = torch.mean((rgb_map - gt_rgb) ** 2)
                psnr = -10.0 * np.log(loss.item()) / np.log(10.0)

                PSNRs.append(psnr)

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

            torch.cuda.empty_cache()
            rgb_map = (rgb_map.numpy() * 255).astype("uint8")
            depth_map, _ = visualize_depth_numpy(depth_map)

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

    evaluator = NerfactoEvaluator(config)
    evaluator.eval()
