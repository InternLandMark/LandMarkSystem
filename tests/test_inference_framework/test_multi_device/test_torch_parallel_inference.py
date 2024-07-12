# pylint: disable=R1735
import os

import numpy as np
import pytest
import torch

from landmark.communicator import destroy_communication_context
from landmark.nerf_components.data import DatasetManager
from landmark.utils.env import EnvSetting
from tests.test_inference_framework.test_inference import TestTorchInference


def _destroy():
    """
    threre is no self instance in teardown_class func
    """
    torch.cuda.empty_cache()
    destroy_communication_context()


class TestTorchInferenceDP(TestTorchInference):
    """
    Test data parallel inference
    """

    def teardown_class(self):
        _destroy()

    def _torch_inference_dp(self, render_config_path, infer_func, psnr, dataset_mgr_cls=DatasetManager):
        self.prepare(render_config_path, render_idx_st=60, render_idx_ed=80, dataset_mgr_cls=dataset_mgr_cls)

        inference_config = dict(
            parallel_config=dict(
                tp_size=1,
            ),
            runtime="Torch",
            kernel_fusion=False,
        )

        psnrs = infer_func(inference_config)
        if EnvSetting.RANK == 0:
            avg_psnr = np.mean(psnrs)
            print(f"{avg_psnr=}")
            assert len(psnrs) == 20
            assert round(avg_psnr, 2) == psnr
        _destroy()

    def _torch_inference_dp_big(self, render_config_path, infer_func):
        self.prepare_big(render_config_path, render_idx_st=0, render_idx_ed=1800)

        inference_config = dict(
            parallel_config=dict(
                tp_size=1,
            ),
            runtime="Torch",
            kernel_fusion=False,
        )

        psnrs = infer_func(inference_config)
        if EnvSetting.RANK == 0:
            avg_psnr = np.mean(psnrs)
            print(f"{avg_psnr=}")
            assert False
        _destroy()

    @pytest.mark.group_2
    @pytest.mark.group_gn
    def test_gridnerf_torch_inference_dp(self):
        gridnerf_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/gridnerf/confs/matrixcity_2block_multi_render.py",
        )
        self._torch_inference_dp(gridnerf_render_config_path, self.inference_gridnerf, 25.29)

    @pytest.mark.group_2
    @pytest.mark.group_in
    def test_instantNGP_torch_inference_dp(self):
        instantNGP_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/instant_ngp/confs/matrixcity_2block_plconfig_render.py",
        )
        self.prepare(instantNGP_render_config_path, render_idx_st=60, render_idx_ed=80, dataset_mgr_cls=DatasetManager)

        inference_config = dict(
            parallel_config=dict(
                tp_size=1,
            ),
            runtime="Kernel",
            kernel_fusion=True,
        )

        psnrs = self.inference_instantNGP(inference_config)
        if EnvSetting.RANK == 0:
            assert len(psnrs) == 20
            avg_psnr = np.mean(psnrs)
            print(f"{avg_psnr=}")
            assert round(avg_psnr, 1) == 24.0
        _destroy()

        # self._torch_inference_dp(instantNGP_render_config_path, self.inference_instantNGP, 24.0)

    @pytest.mark.group_2
    @pytest.mark.group_nf
    def test_nerfacto_torch_inference_dp(self):
        nerfacto_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/nerfacto/confs/matrixcity_2block_huge_render.py",
        )
        self.prepare(nerfacto_render_config_path, render_idx_st=60, render_idx_ed=80, dataset_mgr_cls=DatasetManager)

        inference_config = dict(
            parallel_config=dict(
                tp_size=1,
            ),
            runtime="Kernel",
            kernel_fusion=True,
        )

        psnrs = self.inference_nerfacto(inference_config)
        if EnvSetting.RANK == 0:
            assert len(psnrs) == 20
            avg_psnr = np.mean(psnrs)
            print(f"{avg_psnr=}")
            assert round(avg_psnr, 1) == 24.3
        _destroy()

        # self._torch_inference_dp(nerfacto_render_config_path, self.inference_nerfacto, 24.16)


class TestTorchInferenceTP(TestTorchInference):
    """
    Test tensor parallel inference
    """

    def teardown_class(self):
        _destroy()

    @pytest.mark.group_2
    @pytest.mark.group_gn
    def test_gridnerf_torch_inference_tp(self):
        gridnerf_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/gridnerf/confs/matrixcity_2block_multi_Branch2ChannelParallel_render.py",
        )
        self.prepare(gridnerf_render_config_path, render_idx_st=60, render_idx_ed=80)

        inference_config = dict(
            parallel_config=dict(
                tp_size=2,
            ),
            runtime="Torch",
            kernel_fusion=False,
        )

        psnrs = self.inference_gridnerf(inference_config)
        if EnvSetting.RANK == 0:
            avg_psnr = np.mean(psnrs)
            print(f"{avg_psnr=}")
            assert len(psnrs) == 20
            assert round(avg_psnr, 5) == 25.29503
        _destroy()

    @pytest.mark.group_4
    @pytest.mark.group_gn
    def test_gridnerf_torch_inference_dptp(self):
        self.test_gridnerf_torch_inference_tp()


class TestTorchInferenceOffloadDP(TestTorchInference):
    """
    Test data parallel offload inference
    """

    def teardown_class(self):
        _destroy()

    @pytest.mark.group_2
    @pytest.mark.group_gn
    def test_gridnerf_offload_inference_dp(self):
        gridnerf_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/gridnerf/confs/matrixcity_2block_multi_BranchParallel_render.py",
        )
        self.prepare(gridnerf_render_config_path, render_idx_st=0, render_idx_ed=20)

        local_plane_split = [1, 1]
        inference_config = dict(
            parallel_config=dict(
                tp_size=1,
            ),
            runtime="Torch",
            kernel_fusion=False,
            offload_config=dict(
                local_plane_split=local_plane_split,
                use_nccl=True,
            ),
        )

        psnrs = self.inference_gridnerf(inference_config)
        if EnvSetting.RANK == 0:
            avg_psnr = np.mean(psnrs)
            print(f"{avg_psnr=}")
            assert len(psnrs) == 20
            assert round(avg_psnr, 5) == 25.35770
        _destroy()

    @pytest.mark.group_2
    @pytest.mark.group_gn
    def test_gridnerf_offload_inference_dp_channel_last(self):
        gridnerf_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/gridnerf/confs/matrixcity_2block_multi_BranchParallel_channel_last_render.py",
        )
        self.prepare(gridnerf_render_config_path, render_idx_st=0, render_idx_ed=20)

        local_plane_split = [1, 1]
        inference_config = dict(
            parallel_config=dict(
                tp_size=1,
            ),
            runtime="Torch",
            kernel_fusion=False,
            offload_config=dict(
                local_plane_split=local_plane_split,
                use_nccl=True,
                channel_last=True,
            ),
        )

        psnrs = self.inference_gridnerf(inference_config)
        if EnvSetting.RANK == 0:
            avg_psnr = np.mean(psnrs)
            print(f"{avg_psnr=}")
            assert len(psnrs) == 20
            assert round(avg_psnr, 5) == 25.35770
        _destroy()

    @pytest.mark.group_2
    @pytest.mark.group_in
    def test_instantNGP_torch_inference_dp(self):
        instantNGP_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/instant_ngp/confs/matrixcity_2block_BranchParallel_render.py",
        )
        self.prepare(instantNGP_render_config_path, render_idx_st=0, render_idx_ed=20, dataset_mgr_cls=DatasetManager)

        inference_config = dict(
            parallel_config=dict(
                tp_size=1,
            ),
            offload_config=dict(
                local_plane_split=[1, 2],
                use_nccl=False,
                channel_last=False,
            ),
            runtime="Kernel",
            kernel_fusion=True,
        )

        psnrs = self.inference_instantNGP(inference_config)
        if EnvSetting.RANK == 0:
            avg_psnr = np.mean(psnrs)
            print(f"{avg_psnr=}")
            assert len(psnrs) == 20
            assert round(avg_psnr, 1) == 25.3
        _destroy()

    @pytest.mark.group_2
    @pytest.mark.group_nf
    def test_nerfacto_torch_inference_dp(self):
        nerfacto_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/nerfacto/confs/matrixcity_2block_BranchParallel_render.py",
        )
        self.prepare(nerfacto_render_config_path, render_idx_st=0, render_idx_ed=20, dataset_mgr_cls=DatasetManager)

        inference_config = dict(
            parallel_config=dict(
                tp_size=1,
            ),
            offload_config=dict(
                local_plane_split=[1, 1],
                use_nccl=False,
                channel_last=False,
            ),
            runtime="Kernel",
            kernel_fusion=True,
        )

        psnrs = self.inference_nerfacto(inference_config)
        if EnvSetting.RANK == 0:
            avg_psnr = np.mean(psnrs)
            print(f"{avg_psnr=}")
            assert len(psnrs) == 20
            assert round(avg_psnr, 1) == 26.2
        _destroy()
