# pylint: disable=W0401, W0614, E0602
# flake8: noqa: F405
from benchmarks.nerf.instant_ngp.confs.matrixcity_2block_lowquality_debug import *
from landmark.nerf_components.configs import RenderConfig

render_config = RenderConfig(
    ckpt="log/matrix_city_blockA_lowquality_debug_new18/state_dict.th",
    kwargs="log/matrix_city_blockA_lowquality_debug_new18/kwargs.th",
    ckpt_type="full",
).from_train_config(train_config)

# model_config.sampling_opt = True

config = [render_config, dataset_config, model_config]
