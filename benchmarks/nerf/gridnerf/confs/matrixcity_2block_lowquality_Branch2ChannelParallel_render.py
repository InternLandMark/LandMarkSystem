# pylint: disable=W0401, W0614, E0602
# flake8: noqa: F405
from benchmarks.nerf.gridnerf.confs.matrixcity_2block_lowquality_BranchParallel import (
    dataset_config,
    model_config,
    train_config,
)
from landmark.nerf_components.configs import RenderConfig

render_config = RenderConfig(
    ckpt="log/matrix_city_blockA_lowquality_BranchParallel/state_dict-merged.th",
    kwargs="log/matrix_city_blockA_lowquality_BranchParallel/kwargs-merged.th",
    ckpt_type="full",
    channel_parallel=True,
    channel_parallel_size=2,
).from_train_config(train_config)

model_config.sampling_opt = True

config = [render_config, dataset_config, model_config]