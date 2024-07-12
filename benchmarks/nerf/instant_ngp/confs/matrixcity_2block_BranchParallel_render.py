# pylint: disable=W0401, W0614, E0602
# flake8: noqa: F405
from benchmarks.nerf.instant_ngp.confs.matrixcity_2block_BranchParallel import (
    dataset_config,
    model_config,
    train_config,
)
from landmark.nerf_components.configs import RenderConfig

render_config = RenderConfig(
    ckpt=(
        "/mnt/hwfile/landmark/checkpoint/landmark_sys/instant_ngp/"
        "matrix_city_blockA_instant_ngp_branch/state_dict-merged.th"
    ),
    kwargs=(
        "/mnt/hwfile/landmark/checkpoint/landmark_sys/instant_ngp/"
        "matrix_city_blockA_instant_ngp_branch/kwargs-merged.th"
    ),
    branch_parallel=False,
    ckpt_type="full",
).from_train_config(train_config)


train_config.branch_parallel = False

config = [render_config, dataset_config, model_config]