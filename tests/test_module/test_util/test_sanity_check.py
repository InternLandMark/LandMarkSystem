import os

import pytest

from benchmarks.nerf.origin_gs.gs_model import GaussianModel
from landmark import init_inference
from landmark.nerf_components.configs.config_parser import BaseConfig


class TestSanityCheck:
    """
    Test register cache
    """

    def setup_class(self):
        self.cur_dir_path = os.path.dirname(os.path.abspath(__file__))

    @pytest.mark.skip
    def test_infer_config_sanity_check(self):
        config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "..",
            "benchmarks/nerf/origin_gs/confs/matrixcity_2block_render.py",
        )
        model_config = BaseConfig.from_file(config_path)

        local_plane_split = [1, 1]
        inference_config = dict(
            runtime="Torch",
            kernel_fusion=False,
            parallel_config=dict(
                tp_size=4,
            ),
            offload_config=dict(
                local_plane_split=local_plane_split,
                use_nccl=False,
            ),
        )

        engine = init_inference(GaussianModel, None, model_config, inference_config)

        print(engine)
