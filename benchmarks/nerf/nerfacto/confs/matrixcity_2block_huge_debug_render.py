# pylint: disable=W0401, W0614, E0602
# flake8: noqa: F405
from benchmarks.nerf.nerfacto.confs.matrixcity_2block_huge_debug import *
from landmark.nerf_components.configs import RenderConfig

render_config = RenderConfig(
    ckpt_type="full",
    ckpt="log/matrix_city_blockA_huge_nerfacto_5/state_dict.th",
    kwargs="log/matrix_city_blockA_huge_nerfacto_5/kwargs.th",
).from_train_config(train_config)

# model_config.sampling_opt = True

config = [render_config, dataset_config, model_config]
