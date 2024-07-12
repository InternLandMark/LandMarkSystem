# pylint: disable=W0401, W0614, E0602
# flake8: noqa: F405
from benchmarks.nerf.scaffold_gs.confs.matrixcity_2block_train import *
from landmark.nerf_components.configs import RenderConfig

render_config = RenderConfig(
    ckpt=(
        "/mnt/hwfile/landmark/checkpoint/landmark_sys/scaffold_gs/"
        "ScaffoldGS_matrix_city_blockA/refactor/point_cloud.ply"
    ),
    ckpt_type="full",
    batch_size=1,
    N_vis=20,
).from_train_config(train_config)

config = [render_config, dataset_config, model_config]
