# pylint: disable=W0401, W0614, E0602
# flake8: noqa: F405
from benchmarks.nerf.gridnerf.confs.matrixcity_2block_multi_BranchParallel_10x5 import *
from landmark.nerf_components.configs import RenderConfig

render_config = RenderConfig(
    ckpt="log/merged_gridnerf_5x5/state_dict-merged.th",
    kwargs="log/merged_gridnerf_5x5/kwargs-merged.th",
    ckpt_type="full",
    branch_parallel=False,
).from_train_config(train_config)

model_config.sampling_opt = True

config = [render_config, dataset_config, model_config]
