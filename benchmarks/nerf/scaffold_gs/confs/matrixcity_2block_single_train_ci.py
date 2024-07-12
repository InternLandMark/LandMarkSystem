# pylint: disable=W0401, W0614, E0602
# flake8: noqa: F405
from benchmarks.nerf.scaffold_gs.confs.matrixcity_2block_train import *
from landmark.nerf_components.configs import TrainConfig

train_config = TrainConfig(
    model_name="ScaffoldGS",
    expname="ScaffoldGS_matrix_city_blockA",
    basedir="./log_for_ci_test",
    batch_size=1,
    n_iters=30000,
    test_iterations=[30000],
    tensorboard=True,
)

config = [train_config, dataset_config, model_config]
