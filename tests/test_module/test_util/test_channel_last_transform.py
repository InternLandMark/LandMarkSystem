import os

import pytest

from benchmarks.nerf.gridnerf.gridnerf import GridNeRF
from landmark.nerf_components.configs.config_parser import BaseConfig
from landmark.nerf_components.scene import SceneManager
from landmark.render.util import transform_gridnerf_model_to_channel_last
from tests.test_inference_framework.test_inference import TestTorchInference


class TestRenderUtils(TestTorchInference):
    """
    Test register cache
    """

    def setup_class(self):
        self.cur_dir_path = os.path.dirname(os.path.abspath(__file__))

    @pytest.mark.skip
    def test_channel_last_transfer(self):
        gridnerf_render_config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "..",
            "benchmarks/nerf/gridnerf/confs/matrixcity_2block_multi_BranchParallel_10x5_render.py",
        )

        config = BaseConfig.from_file(gridnerf_render_config_path)

        print("load ckpt from", config.ckpt)

        device = "cpu"  # or "cuda"
        config.device = device
        kwargs = GridNeRF.get_kwargs_from_ckpt(config.kwargs, device)
        kwargs["config"] = config
        kwargs["scene_manager"] = SceneManager(config)

        gridnerf_model = GridNeRF(**kwargs)  # pylint: disable=W0123

        state_dict = gridnerf_model.get_state_dict_from_ckpt(config.ckpt, device)
        gridnerf_model.load_from_state_dict(state_dict, strict=False)
        gridnerf_model.eval()
        transform_gridnerf_model_to_channel_last(gridnerf_model, config.ckpt)
