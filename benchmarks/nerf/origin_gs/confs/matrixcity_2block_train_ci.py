# pylint: disable=W0401, W0614, E0602
# flake8: noqa: F405
from benchmarks.nerf.origin_gs.confs.matrixcity_2block_train import *

train_config.basedir = "./log_for_ci_test"
train_config.N_vis = 0

config = [train_config, dataset_config, model_config]
