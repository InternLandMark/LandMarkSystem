import json
import os
import time

import imageio
import numpy as np
import torch
from PIL import Image

from benchmarks.nerf.gridnerf.gridnerf import GridNeRF
from benchmarks.nerf.instant_ngp.instant_ngp import InstantNGP
from benchmarks.nerf.nerfacto.nerfacto import Nerfacto
from benchmarks.nerf.octree_gs.gs_model import OctreeGS
from benchmarks.nerf.origin_gs.gs_model import GaussianModel
from benchmarks.nerf.scaffold_gs.gs_model import ScaffoldGS
from landmark import init_inference
from landmark.nerf_components.configs.config_parser import BaseConfig
from landmark.nerf_components.data import DatasetManager
from landmark.nerf_components.data.cameras import Camera
from landmark.nerf_components.utils.graphics_utils import (
    focal2fov,
    getProjectionMatrix,
    getWorld2View2,
)
from landmark.nerf_components.utils.image_utils import psnr as cal_gs_psnr
from landmark.utils.env import EnvSetting
from tests.utils import (
    InferenceGridNerfModule,
    InferenceInstantNGPModule,
    InferenceNerfactoModule,
    InferenceOctreeGSModule,
    SimpleModel,
)


class TestTorchInference:
    """
    Test for pure inference
    """

    def setup_class(self):
        self.cur_dir_path = os.path.dirname(os.path.abspath(__file__))

    def teardown_class(self):
        torch.cuda.empty_cache()

    def sync_barrier(self):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        torch.cuda.synchronize()

    def prepare(self, config_path, render_idx_st=60, render_idx_ed=80, dataset_mgr_cls=DatasetManager):
        self.model_config = BaseConfig.from_file(config_path)
        self.render_idxes = list(range(render_idx_st, render_idx_ed))
        self.render_png = len(self.render_idxes)  # <= self.dataset.poses
        self.compute_overlap = False
        self.save_png = False  # When save_png is true, compute_overlap is false regardless of its value.
        self.compute_psnr = True
        self.saving_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "pictures",
        )
        if EnvSetting.RANK == 0:
            self.dataset_mgr = dataset_mgr_cls(self.model_config)
            self.dataset = self.dataset_mgr.test_dataset
            self.W, self.H = self.dataset.img_wh

    def prepare_big(self, config_path, render_idx_st=0, render_idx_ed=1800):
        self.model_config = BaseConfig.from_file(config_path)
        self.render_idxes = list(range(render_idx_st, render_idx_ed))
        self.render_png = len(self.render_idxes)  # <= self.dataset.poses
        self.compute_overlap = True
        self.save_png = False
        self.compute_psnr = False
        self.saving_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "offload_kernel_overlap",
        )
        self.test_json_dir = (
            "/cpfs01/shared/pjlab-lingjun-landmarks/3dgs_render_engine/cityeyes/test_traj/zigzag_traj.json"
        )

    def _inter_inference_nerf(self, idx, gt_rgb, inference, *forward_args):
        if EnvSetting.RANK == 0:
            if self.compute_overlap:
                if idx % 100 == 0:
                    self.sync_barrier()
            else:
                self.sync_barrier()
            time_start = time.time()
            nerf_out = inference(*forward_args)
            if self.compute_overlap:
                if idx % 100 == 0:
                    self.sync_barrier()
            else:
                self.sync_barrier()
            latency = time.time() - time_start
        else:
            if self.compute_overlap:
                if idx % 100 == 0:
                    self.sync_barrier()
            else:
                self.sync_barrier()
            nerf_out = inference()
            if self.compute_overlap:
                if idx % 100 == 0:
                    self.sync_barrier()
            else:
                self.sync_barrier()
        assert nerf_out is not None
        if EnvSetting.RANK == 0:
            psnr = 0.0
            if self.save_png:
                save_out = nerf_out.byte().detach()
                save_out = save_out.to("cpu")
                save_out = save_out.numpy()
                if not os.path.exists(self.saving_path):
                    os.mkdir(self.saving_path)
                im_png = Image.fromarray(save_out)
                im_png.save(os.path.join(self.saving_path, f"{idx}.png"))
            if self.compute_psnr:
                nerf_out = nerf_out[..., :-1]
                rgb_map = nerf_out / 255.0
                rgb_map = rgb_map.cpu()
                psnr = cal_gs_psnr(rgb_map, gt_rgb).mean().item()
            return latency, psnr

    def _inference_nerf(self, inference_config, nerf_model, infer_model):
        torch.cuda.reset_max_memory_allocated()
        inference = init_inference(nerf_model, infer_model, self.model_config, inference_config)
        chunk_size = self.model_config.render_batch_size
        edit_mode = {"idx", 0}
        app_code = 0
        Latency = []
        PSNRs = []
        Max_mem = []
        print(f"render nums:{self.render_png}")

        for idx in self.render_idxes:
            if EnvSetting.RANK == 0:
                gt_rgb = self.dataset.all_rgbs[idx].view(self.H, self.W, 3)
                forward_args = (self.dataset.poses[idx], chunk_size, self.H, self.W, app_code, edit_mode)
                latency, psnr = self._inter_inference_nerf(idx, gt_rgb, inference, *forward_args)
                Latency.append(latency)
                PSNRs.append(psnr)
            else:
                self._inter_inference_nerf(idx, None, inference)
            Max_mem.append(torch.cuda.max_memory_allocated())

        if EnvSetting.RANK == 0:
            print(
                "Avg latency:",
                sum(Latency[5:]) / len(Latency[5:]),
                " throughput:",
                len(Latency[5:]) / sum(Latency[5:]),
                " max mem:",
                max(Max_mem) / 1024**3,
            )

        return PSNRs

    def _inter_inference_gs(self, idx, gt_rgb, inference, *forward_args):
        if EnvSetting.RANK == 0:
            if self.compute_overlap:
                if idx % 100 == 0:
                    self.sync_barrier()
            else:
                self.sync_barrier()
            time_start = time.time()
            gs_out = inference(*forward_args)
            if self.compute_overlap:
                if idx % 100 == 0:
                    self.sync_barrier()
            else:
                self.sync_barrier()
            latency = time.time() - time_start
        else:
            if self.compute_overlap:
                if idx % 100 == 0:
                    self.sync_barrier()
            else:
                self.sync_barrier()
            gs_out = inference()
            if self.compute_overlap:
                if idx % 100 == 0:
                    self.sync_barrier()
            else:
                self.sync_barrier()
        assert gs_out is not None
        if EnvSetting.RANK == 0:
            image = gs_out.clamp(0.0, 1.0)
            if self.save_png:
                saved_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                if not os.path.exists(self.saving_path):
                    os.mkdir(self.saving_path)
                imageio.imwrite(os.path.join(self.saving_path, f"{idx}.png"), saved_image)
            psnr = 0.0
            if self.compute_psnr:
                psnr = cal_gs_psnr(image, gt_rgb.cuda()).mean().item()
            return latency, psnr

    def _inter_inference_gs_big(self, idx, gt_rgb, inference, *forward_args):
        if EnvSetting.RANK == 0:
            if self.compute_overlap:
                if idx % 100 == 0:
                    self.sync_barrier()
            else:
                self.sync_barrier()
            time_start = time.time()
            gs_out = inference(*forward_args)
            if self.compute_overlap:
                if idx % 100 == 0:
                    self.sync_barrier()
            else:
                self.sync_barrier()
            latency = time.time() - time_start
        else:
            if self.compute_overlap:
                if idx % 100 == 0:
                    self.sync_barrier()
            else:
                self.sync_barrier()
            gs_out = inference()
            if self.compute_overlap:
                if idx % 100 == 0:
                    self.sync_barrier()
            else:
                self.sync_barrier()
        assert gs_out is not None
        if EnvSetting.RANK == 0:
            image = gs_out.clamp(0.0, 1.0)
            if self.save_png:
                if idx % 60 == 0:
                    saved_image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                    if not os.path.exists(self.saving_path):
                        os.mkdir(self.saving_path)
                    imageio.imwrite(os.path.join(self.saving_path, f"{idx}.png"), saved_image)
            psnr = 0.0
            if self.compute_psnr:
                psnr = cal_gs_psnr(image, gt_rgb.cuda()).mean().item()
            return latency, psnr

    def load_poses(self, path: str = "/cpfs01/shared/pjlab-lingjun-landmarks/wuguohao/pose5x5_300.pth"):
        poses = torch.load(path)
        if isinstance(poses, list):
            print("len(poses) = ", len(poses))
        elif isinstance(poses, torch.Tensor):
            print("poses.shape = ", poses.shape)
        return poses

    def _inference_gs(self, inference_config, gs_model, infer_model=None):
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        inference = init_inference(gs_model, infer_model, self.model_config, inference_config)

        # edit_mode = {"idx", 0}
        # app_code = 0
        Latency = []
        PSNRs = []
        Max_mem = []
        Max_reserve_mem = []
        print(f"render nums:{self.render_png}")

        for idx in self.render_idxes:
            if EnvSetting.RANK == 0:
                viewpoint_cam, gt_rgb = self.dataset_mgr.get_camera("test")
                viewpoint_cam.image = None
                viewpoint_cam.c2w = None
                viewpoint_cam.R = None
                viewpoint_cam.T = None
                viewpoint_cam.trans = None
                viewpoint_cam.image_name = None
                viewpoint_cam.image_path = None
                forward_args = (viewpoint_cam, 1.0, None, None)
                latency, psnr = self._inter_inference_gs(idx, gt_rgb, inference, *forward_args)
                Latency.append(latency)
                PSNRs.append(psnr)
            else:
                self._inter_inference_gs(idx, None, inference)
            Max_mem.append(torch.cuda.max_memory_allocated())
            Max_reserve_mem.append(torch.cuda.max_memory_reserved())

        if EnvSetting.RANK == 0:
            print(
                "Avg latency:",
                sum(Latency[5:]) / len(Latency[5:]),
                " throughput:",
                len(Latency[5:]) / sum(Latency[5:]),
                " max allocated mem:",
                max(Max_mem) / 1024**3,
                " max reserved mem:",
                max(Max_reserve_mem) / 1024**3,
            )
        return PSNRs

    def _inference_gs_big(self, inference_config, gs_model, infer_model=None):
        def cam_wrap(c2w_pose):
            image_width = 1920
            image_height = 1080
            fl_x = 1483.6378
            fl_y = 1483.6378
            znear = 0.01
            zfar = 10.0
            fovx = focal2fov(fl_x, image_width)
            fovy = focal2fov(fl_y, image_height)

            c2w_pose = np.array(c2w_pose)
            c2w_pose = np.concatenate((c2w_pose, np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])))[:4, :4]

            c2w_pose[:3, 1:3] *= -1

            w2c_pose = np.linalg.inv(c2w_pose)
            R = np.transpose(w2c_pose[:3, :3])
            T = w2c_pose[:3, 3]
            world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
            projection_matrix = getProjectionMatrix(znear=0.01, zfar=1.0, fovX=fovx, fovY=fovy).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            c2w_minicam = Camera(
                None,
                None,
                None,
                image_width,
                image_height,
                None,
                fovx,
                fovy,
                None,
                None,
                zfar,
                znear,
                None,
                None,
                world_view_transform,
                projection_matrix,
                full_proj_transform,
                torch.tensor(c2w_pose[:3, 3]).float(),
                None,
            )
            return c2w_minicam

        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        inference = init_inference(gs_model, infer_model, self.model_config, inference_config)

        Latency = []
        PSNRs = []
        Max_mem = []
        Max_reserve_mem = []
        print(f"render nums:{self.render_png}")

        with open(self.test_json_dir, encoding="utf-8") as json_file:
            contents = json.load(json_file)
            frames = contents["frames"]

            for idx, frame in enumerate(frames):
                if idx in self.render_idxes:
                    if EnvSetting.RANK == 0:
                        pose = np.array(frame["transform_matrix"])
                        pose[0, 3] *= 5
                        pose[1, 3] *= 5
                        pose = np.concatenate((pose[:3, :4], np.array([[0.0], [0.0], [0.0]])), axis=1)[:3, :5]
                        viewpoint_cam = cam_wrap(pose)
                        forward_args = (viewpoint_cam, 1.0, None, None)
                        latency, psnr = self._inter_inference_gs_big(idx, None, inference, *forward_args)
                        Latency.append(latency)
                        PSNRs.append(psnr)
                    else:
                        self._inter_inference_gs_big(idx, None, inference)
                    Max_mem.append(torch.cuda.max_memory_allocated())
                    Max_reserve_mem.append(torch.cuda.max_memory_reserved())

        if EnvSetting.RANK == 0:
            print(
                "Avg latency:",
                sum(Latency[5:]) / len(Latency[5:]),
                " throughput:",
                len(Latency[5:]) / sum(Latency[5:]),
                " max allocated mem:",
                max(Max_mem) / 1024**3,
                " max reserved mem:",
                max(Max_reserve_mem) / 1024**3,
            )
        return PSNRs

    def inference_simplemodel(self, inference_config):
        inference = init_inference(SimpleModel, None, None, inference_config)

        nerf_out = inference(torch.ones(10, 10).cuda())
        print(nerf_out)

    def inference_gridnerf(self, inference_config):
        return self._inference_nerf(inference_config, GridNeRF, InferenceGridNerfModule)

    def inference_instantNGP(self, inference_config):
        return self._inference_nerf(inference_config, InstantNGP, InferenceInstantNGPModule)

    def inference_nerfacto(self, inference_config):
        return self._inference_nerf(inference_config, Nerfacto, InferenceNerfactoModule)

    def inference_origin_gs(self, inference_config):
        return self._inference_gs(inference_config, GaussianModel)

    def inference_origin_gs_big(self, inference_config):
        return self._inference_gs_big(inference_config, GaussianModel)

    def inference_scaffold_gs(self, inference_config):
        return self._inference_gs(inference_config, ScaffoldGS)

    def inference_scaffold_gs_big(self, inference_config):
        return self._inference_gs_big(inference_config, ScaffoldGS)

    def inference_octree_gs(self, inference_config):
        return self._inference_gs(inference_config, OctreeGS, InferenceOctreeGSModule)

    def inference_octree_gs_big(self, inference_config):
        return self._inference_gs_big(inference_config, OctreeGS, InferenceOctreeGSModule)
