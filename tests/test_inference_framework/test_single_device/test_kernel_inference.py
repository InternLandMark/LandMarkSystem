# pylint: disable=R1735
import os

import numpy as np
import pytest

from landmark.nerf_components.data import DatasetManager
from tests.test_inference_framework.test_inference import TestTorchInference


class TestKernelInferenceNoParallel(TestTorchInference):
    """
    Test inference on one cuda device
    """

    def _inference_no_parallel(self, render_config_path, infer_func, psnr, dataset_mgr_cls=DatasetManager):
        self.prepare(render_config_path, render_idx_st=60, render_idx_ed=80, dataset_mgr_cls=dataset_mgr_cls)

        inference_config = dict(
            runtime="Kernel",
            kernel_fusion=True,
        )

        psnrs = infer_func(inference_config)
        avg_psnr = np.mean(psnrs)
        print(f"{avg_psnr=}")
        assert len(psnrs) == 20
        assert round(avg_psnr, 2) == psnr

    @pytest.mark.group_gs
    def test_inference_scaffold_gs_no_parallel(self):
        scaffold_gs_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/scaffold_gs/confs/matrixcity_2block_render.py",
        )
        self._inference_no_parallel(scaffold_gs_render_config_path, self.inference_scaffold_gs, 28.20, DatasetManager)

    @pytest.mark.group_gs
    def test_inference_octree_gs_no_parallel(self):
        octree_gs_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "benchmarks/nerf/octree_gs/confs/matrixcity_2block_render.py",
        )
        self._inference_no_parallel(octree_gs_render_config_path, self.inference_octree_gs, 27.85, DatasetManager)
