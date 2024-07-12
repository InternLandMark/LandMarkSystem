# pylint: disable=W0401, W0614, E0602
# flake8: noqa: F405
from benchmarks.nerf.gridnerf.confs.matrixcity_2block_multi import *
from landmark.nerf_components.configs import RenderConfig

render_config = RenderConfig(
    ckpt="/mnt/hwfile/landmark/checkpoint/landmark_sys/gridnerf/matrix_city_blockA_multi/state_dict.th",
    kwargs="/mnt/hwfile/landmark/checkpoint/landmark_sys/gridnerf/matrix_city_blockA_multi/kwargs.th",
    ckpt_type="full",
).from_train_config(train_config)

model_config.sampling_opt = True

config = [render_config, dataset_config, model_config]
