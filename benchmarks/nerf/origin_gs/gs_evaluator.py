# pylint: disable=W0621, W0613
import os
import time

import imageio
import numpy as np
import torch
from colorama import Fore, Style
from tqdm import tqdm

from benchmarks.nerf.origin_gs.gs_model import GaussianModel
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
        gaussians = GaussianModel(config)

        print(Fore.CYAN + f"Loading from {config.ckpt}" + Style.RESET_ALL)
        state_dict = gaussians.get_state_dict_from_ckpt(config.ckpt, config.device)
        gaussians.load_from_state_dict(state_dict)

        self.first_iter = 0

        if config.start_checkpoint:
            state_dict = gaussians.get_state_dict_from_ckpt(config.start_checkpoint, config.device)
            gaussians.load_from_state_dict(state_dict)

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
            test_dataset: Test dataset.
            config (tools.configs.ArgsConfig): Evaluation configs.
            savePath (str): Path where image, video, etc. information is stored.
            N_vis (int): Control the visualization images.
            prtx (str): Prefix of finename.
            compute_extra_metrics (bool): Whether to compute extra metrics.
            device (str): Device on which a tensor is or will be allocated.

        Returns:
            list: A list of PSNR for each image.
        """
        self.model.eval()

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
            gt_image = gt_image.cuda()

            torch.cuda.synchronize()
            if config.rank == 0:
                render_start = time.time()

            postfix = f"{idx:03d}.png"
            image = self.model(viewpoint_cam)

            torch.cuda.synchronize()
            if config.rank == 0:
                render_end = time.time()
                render_times.append((render_end - render_start))

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

        if config.rank == 0:
            print(f"  AVG_Render_Time : {torch.tensor(render_times[5:]).mean()*1000} ms")

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
