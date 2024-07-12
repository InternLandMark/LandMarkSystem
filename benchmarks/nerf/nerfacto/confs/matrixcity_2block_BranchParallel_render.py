# pylint: disable=W0401, W0614, E0602
# flake8: noqa: F405
from benchmarks.nerf.nerfacto.confs.matrixcity_2block_BranchParallel import *
from landmark.nerf_components.configs import RenderConfig

render_config = RenderConfig(
    ckpt=(
        "/mnt/hwfile/landmark/checkpoint/landmark_sys/nerfacto/"
        "matrix_city_blockA_nerfacto_BranchParallel/state_dict-merged.th"
    ),
    kwargs=(
        "/mnt/hwfile/landmark/checkpoint/landmark_sys/nerfacto/"
        "matrix_city_blockA_nerfacto_BranchParallel/kwargs-merged.th"
    ),
    ckpt_type="full",
    branch_parallel=False,
).from_train_config(train_config)

config = [render_config, dataset_config, model_config]
