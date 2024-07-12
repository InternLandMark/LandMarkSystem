# pylint: disable=W0401, W0614, E0602
# flake8: noqa: F405
from benchmarks.nerf.nerfacto.confs.matrixcity_2block_BranchParallel import *
from landmark.nerf_components.configs import RenderConfig

render_config = RenderConfig(
    ckpt="log/matrix_city_blockA_nerfacto_BranchParallel_debug/state_dict-merged.th",
    ckpt_type="full",
    kwargs="log/matrix_city_blockA_nerfacto_BranchParallel_debug/kwargs-merged.th",
    branch_parallel=False,
).from_train_config(train_config)

config = [render_config, dataset_config, model_config]
