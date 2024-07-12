# pylint: disable=R1735
import os

import numpy as np
import pytest

from landmark.nerf_components.data import DatasetManager
from tests.test_inference_framework.test_inference import TestTorchInference


class TestTorchInferenceNoParallel(TestTorchInference):
    """
    Test inference on one cuda device
    """

    def _inference_no_parallel(self, render_config_path, infer_func, psnr, dataset_mgr_cls=DatasetManager):
        self.prepare(render_config_path, render_idx_st=60, render_idx_ed=80, dataset_mgr_cls=dataset_mgr_cls)

        inference_config = dict(
            runtime="Torch",
            kernel_fusion=False,
        )

        psnrs = infer_func(inference_config)
        avg_psnr = np.mean(psnrs)
        print(f"{avg_psnr=}")
        assert len(psnrs) == 20
        assert round(avg_psnr, 2) == psnr

    def _inference_no_parallel_big(self, render_config_path, infer_func):
        self.prepare_big(render_config_path, render_idx_st=0, render_idx_ed=1800)

        inference_config = dict(
            runtime="Torch",
            kernel_fusion=False,
        )

        psnrs = infer_func(inference_config)
        avg_psnr = np.mean(psnrs)
        print(f"{avg_psnr=}")

    @pytest.mark.group_gs
    def test_inference_simple_no_parallel(self):
        inference_config = dict(
            runtime="Torch",
            kernel_fusion=False,
            ckpt_path=None,
        )
        self.inference_simplemodel(inference_config)

    @pytest.mark.group_gn
    def test_inference_gridnerf_no_parallel(self):
        gridnerf_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/gridnerf/confs/matrixcity_2block_multi_render.py",
        )
        self._inference_no_parallel(gridnerf_render_config_path, self.inference_gridnerf, 25.29)

    @pytest.mark.group_in
    def test_inference_instantNGP_no_parallel(self):
        instantNGP_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/instant_ngp/confs/matrixcity_2block_plconfig_render.py",
        )
        self.prepare(instantNGP_render_config_path, render_idx_st=60, render_idx_ed=80, dataset_mgr_cls=DatasetManager)

        inference_config = dict(
            runtime="Kernel",
            kernel_fusion=True,
        )

        psnrs = self.inference_instantNGP(inference_config)
        avg_psnr = np.mean(psnrs)
        print(f"{avg_psnr=}")
        assert len(psnrs) == 20
        assert round(avg_psnr, 1) == 24.0
        # self._inference_no_parallel(instantNGP_render_config_path, self.inference_instantNGP, 24.0)

    @pytest.mark.group_in
    def test_inference_instantNGP_bp_no_parallel(self):
        instantNGP_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/instant_ngp/confs/matrixcity_2block_BranchParallel_render.py",
        )
        self.prepare(instantNGP_render_config_path, render_idx_st=60, render_idx_ed=80, dataset_mgr_cls=DatasetManager)

        inference_config = dict(
            runtime="Kernel",
            kernel_fusion=True,
        )

        psnrs = self.inference_instantNGP(inference_config)
        avg_psnr = np.mean(psnrs)
        print(f"{avg_psnr=}")
        assert len(psnrs) == 20
        assert round(avg_psnr, 1) == 24.2

        # self._inference_no_parallel(instantNGP_render_config_path, self.inference_instantNGP, 24.2)

    @pytest.mark.group_nf
    def test_inference_nerfacto_no_parallel(self):
        nerfacto_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/nerfacto/confs/matrixcity_2block_huge_render.py",
        )
        self.prepare(nerfacto_render_config_path, render_idx_st=60, render_idx_ed=80, dataset_mgr_cls=DatasetManager)

        inference_config = dict(
            runtime="Kernel",
            kernel_fusion=True,
        )

        psnrs = self.inference_nerfacto(inference_config)
        avg_psnr = np.mean(psnrs)
        print(f"{avg_psnr=}")
        assert len(psnrs) == 20
        assert round(avg_psnr, 1) == 24.3

        # self._inference_no_parallel(nerfacto_render_config_path, self.inference_nerfacto, 24.16)

    @pytest.mark.group_nf
    def test_inference_nerfacto_bp_no_parallel(self):
        nerfacto_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/nerfacto/confs/matrixcity_2block_BranchParallel_render.py",
        )
        self.prepare(nerfacto_render_config_path, render_idx_st=60, render_idx_ed=80, dataset_mgr_cls=DatasetManager)

        inference_config = dict(
            runtime="Kernel",
            kernel_fusion=True,
        )

        psnrs = self.inference_nerfacto(inference_config)
        avg_psnr = np.mean(psnrs)
        print(f"{avg_psnr=}")
        assert len(psnrs) == 20
        assert round(avg_psnr, 1) == 24.7

        # self._inference_no_parallel(nerfacto_render_config_path, self.inference_nerfacto, 24.67)

    @pytest.mark.group_gs
    def test_inference_origin_gs_no_parallel(self):
        origings_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/origin_gs/confs/matrixcity_2block_render.py",
        )
        self._inference_no_parallel(origings_render_config_path, self.inference_origin_gs, 25.53, DatasetManager)

    def test_inference_origin_gs_no_parallel_big(self):
        origings_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/origin_gs/confs/matrixcity_2block_render_big.py",
        )
        self._inference_no_parallel_big(origings_render_config_path, self.inference_origin_gs_big)

    @pytest.mark.group_gs
    def test_inference_scaffold_gs_no_parallel(self):
        scaffold_gs_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/scaffold_gs/confs/matrixcity_2block_render.py",
        )
        self._inference_no_parallel(scaffold_gs_render_config_path, self.inference_scaffold_gs, 28.20, DatasetManager)

    def test_inference_scaffold_gs_no_parallel_big(self):
        scaffold_gs_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/scaffold_gs/confs/matrixcity_2block_render_big.py",
        )
        self._inference_no_parallel_big(scaffold_gs_render_config_path, self.inference_scaffold_gs_big)

    @pytest.mark.group_gs
    def test_inference_octree_gs_no_parallel(self):
        octree_gs_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/octree_gs/confs/matrixcity_2block_render.py",
        )
        self._inference_no_parallel(octree_gs_render_config_path, self.inference_octree_gs, 27.85, DatasetManager)

    def test_inference_octree_gs_no_parallel_big(self):
        octree_gs_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/octree_gs/confs/matrixcity_2block_render_big.py",
        )
        self._inference_no_parallel_big(octree_gs_render_config_path, self.inference_octree_gs_big)


class TestTorchInferenceOffload(TestTorchInference):
    """
    Test data parallel offload inference
    """

    @pytest.mark.group_gn
    def test_offload_inference_gridnerf(self):
        config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/gridnerf/confs/matrixcity_2block_multi_BranchParallel_render.py",
        )
        self.prepare(config_path, render_idx_st=0, render_idx_ed=20)

        local_plane_split = [1, 1]
        inference_config = dict(
            runtime="Torch",
            kernel_fusion=False,
            offload_config=dict(
                local_plane_split=local_plane_split,
                use_nccl=False,
            ),
        )

        psnrs = self.inference_gridnerf(inference_config)
        avg_psnr = np.mean(psnrs)
        print(f"{avg_psnr=}")
        assert len(psnrs) == 20
        assert round(avg_psnr, 5) == 25.35770

    @pytest.mark.group_gn
    def test_offload_inference_gridnerf_channel_last(self):
        config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/gridnerf/confs/matrixcity_2block_multi_BranchParallel_channel_last_render.py",
        )
        self.prepare(config_path, render_idx_st=0, render_idx_ed=20)

        local_plane_split = [1, 1]
        inference_config = dict(
            runtime="Torch",
            kernel_fusion=False,
            offload_config=dict(
                plane_split=None,
                local_plane_split=local_plane_split,
                use_nccl=False,
                channel_last=True,
            ),
        )

        psnrs = self.inference_gridnerf(inference_config)
        avg_psnr = np.mean(psnrs)
        print(f"{avg_psnr=}")
        assert len(psnrs) == 20
        assert round(avg_psnr, 5) == 25.35770

    @pytest.mark.group_in
    def test_offload_inference_instantNGP_no_parallel(self):
        instantNGP_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/instant_ngp/confs/matrixcity_2block_BranchParallel_render.py",
        )
        self.prepare(instantNGP_render_config_path, render_idx_st=0, render_idx_ed=20, dataset_mgr_cls=DatasetManager)

        inference_config = dict(
            runtime="Kernel",
            kernel_fusion=True,
            offload_config=dict(
                local_plane_split=[1, 2],
                use_nccl=False,
                channel_last=False,
            ),
        )

        psnrs = self.inference_instantNGP(inference_config)
        avg_psnr = np.mean(psnrs)
        print(f"{avg_psnr=}")
        assert len(psnrs) == 20
        assert round(avg_psnr, 1) == 25.3

    @pytest.mark.group_nf
    def test_offload_inference_nerfacto_no_parallel(self):
        instantNGP_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/nerfacto/confs/matrixcity_2block_BranchParallel_render.py",
        )
        self.prepare(instantNGP_render_config_path, render_idx_st=0, render_idx_ed=20, dataset_mgr_cls=DatasetManager)

        inference_config = dict(
            runtime="Kernel",
            kernel_fusion=True,
            offload_config=dict(
                local_plane_split=[1, 1],
                use_nccl=False,
                channel_last=False,
            ),
        )

        psnrs = self.inference_nerfacto(inference_config)
        avg_psnr = np.mean(psnrs)
        print(f"{avg_psnr=}")
        assert len(psnrs) == 20
        assert round(avg_psnr, 1) == 26.2
