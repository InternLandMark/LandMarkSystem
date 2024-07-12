# pylint: disable=W0621, W0613
import os
import time

import imageio
import numpy as np
import torch
from colorama import Fore, Style
from tqdm import tqdm

from benchmarks.nerf.octree_gs.gs_model import OctreeGS
from landmark.nerf_components.configs import (
    BaseConfig,
    get_default_parser,
    parse_console_args,
)
from landmark.nerf_components.data.data_manager import DatasetManager
from landmark.nerf_components.scene import SceneManager
from landmark.nerf_components.utils.image_utils import psnr
from landmark.nerf_components.utils.loss_utils import rgb_lpips, rgb_ssim
from landmark.train.nerf_evaluator import NeRFEvaluator


class Gaussian3DEvaluator(NeRFEvaluator):
    """Evaluator Class for 3D Gaussian Splatting"""

    def __init__(
        self,
        config: BaseConfig,
    ):
        super().__init__(config)
        self.init_render_env()

        self.scene_mgr = SceneManager(config)
        self.data_mgr = DatasetManager(config)
        self.model = self.create_model()

        self.check_args()

    def create_model(self):
        config = self.config
        gaussians = OctreeGS(config)

        pcd_path = config.ckpt
        print(Fore.CYAN + f"Loading from {pcd_path}" + Style.RESET_ALL)
        state_dict = gaussians.get_state_dict_from_ckpt(config.ckpt, config.device)
        gaussians.load_from_state_dict(state_dict)
        print(f"{gaussians=}")

        if config.start_checkpoint:
            (model_params, _, self.first_iter) = torch.load(config.start_checkpoint)
            gaussians.load(model_params)

        return gaussians

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
    ):  # pylint:disable=W0237
        """
        Args:
            config (tools.configs.ArgsConfig): Evaluation configs.
            savePath (str): Path where image, video, etc. information is stored.
            N_vis (int): Control the visualization images.
            prtx (str): Prefix of finename.
            compute_extra_metrics (bool): Whether to compute extra metrics.

        Returns:
            list: A list of PSNR for each image.
        """

        self.model.eval()
        self.model.plot_levels()

        PSNRs = []
        ssims, l_alex, l_vgg = [], [], []
        if savePath is not None:
            os.makedirs(savePath, exist_ok=True)

        tqdm._instances.clear()

        num_test = len(test_dataset)
        img_eval_interval = 1 if N_vis < 0 else max(num_test // N_vis, 1)
        idxs = list(range(0, num_test, img_eval_interval))

        render_images = len(idxs)
        print("test_dataset render images", render_images)

        render_times = []

        for idx in tqdm(idxs):
            viewpoint_cam, gt_image = self.data_mgr.get_camera("test")
            viewpoint_cam = viewpoint_cam.cuda()
            gt_image = gt_image.cuda()

            postfix = f"{idx:03d}.png"

            torch.cuda.synchronize()
            render_start = time.time()
            self.model.set_anchor_mask(viewpoint_cam.camera_center, idx, viewpoint_cam.resolution_scale)
            image = self.model(viewpoint_cam)
            torch.cuda.synchronize()
            render_end = time.time()
            render_times.append((render_end - render_start))

            if config.rank == 0:
                image = image.clamp(0.0, 1.0)

                ps = psnr(image, gt_image).mean()
                PSNRs.append(ps.item())
                if compute_extra_metrics:
                    ssim = rgb_ssim(image, gt_image, 1)
                    l_a = rgb_lpips(gt_image.numpy(), image.numpy(), "alex", device)
                    l_v = rgb_lpips(gt_image.numpy(), image.numpy(), "vgg", device)
                    ssims.append(ssim)
                    l_alex.append(l_a)
                    l_vgg.append(l_v)

                if savePath is not None:
                    gt_image = (gt_image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                    image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                    img = np.concatenate((gt_image, image), axis=1)

                    if config.rank == 0:
                        imageio.imwrite(f"{savePath}/{prtx}{postfix}", img)
                        print(f"saved to {savePath}/{prtx}{postfix}, got psnr: {ps.item()}")

        if config.rank == 0:
            print(f"  AVG_Render_Time : {torch.tensor(render_times[5:]).mean()*1000} ms")
            print(f"  AVG_Render_FPS : {1/torch.tensor(render_times[5:]).mean()} FPS")

        if PSNRs and config.rank == 0 and savePath is not None:
            ps = np.mean(np.asarray(PSNRs))
            if compute_extra_metrics:
                ssim = np.mean(np.asarray(ssims))
                l_a = np.mean(np.asarray(l_alex))
                l_v = np.mean(np.asarray(l_vgg))
                np.savetxt(f"{savePath}/{prtx}mean.txt", np.asarray([ps, ssim, l_a, l_v]))
            else:
                np.savetxt(f"{savePath}/{prtx}mean_{ps:03f}.txt", np.asarray(PSNRs))
        return PSNRs


if __name__ == "__main__":
    parser = get_default_parser()
    args, other_args = parser.parse_known_args()
    console_args = parse_console_args(other_args)
    config = BaseConfig.from_file(args.config, console_args)

    evaluator = Gaussian3DEvaluator(config)
    evaluator.eval()
