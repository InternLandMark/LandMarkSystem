# pylint: disable=W0401, W0614, E0602
# flake8: noqa: F405
from benchmarks.nerf.instant_ngp.confs.matrixcity_2block_ddp_train import *
from landmark.nerf_components.configs import RenderConfig

render_config = RenderConfig(
    ckpt="log/iNGP_matrixcity_blockA_ddp_train/state_dict.th",
    kwargs="log/iNGP_matrixcity_blockA_ddp_train/kwargs.th",
    ckpt_type="full",
).from_train_config(train_config)

# model_config.sampling_opt = True

config = [render_config, dataset_config, model_config]
