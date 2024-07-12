# pylint: disable=W0401, W0614, E0602
# flake8: noqa: F405
from benchmarks.nerf.octree_gs.confs.matrixcity_2block_train import *
from landmark.nerf_components.configs import RenderConfig

render_config = RenderConfig(
    ckpt="/mnt/hwfile/landmark/checkpoint/landmark_sys/octree_gs/OctreeGS_matrix_city_blockA/point_cloud.ply",
    ckpt_type="full",
    batch_size=1,
    N_vis=20,
).from_train_config(train_config)

model_config.enable_lod = True

config = [render_config, dataset_config, model_config]
