from benchmarks.nerf.instant_ngp.instant_ngp import InstantNGPConfig
from landmark.nerf_components.configs import DatasetConfig, TrainConfig

train_config = TrainConfig(
    model_name="InstantNGP",
    expname="iNGP_matrixcity_blockA_ddp_train",
    basedir="./log",
    batch_size=16384,
    vis_every=1000,
    n_iters=100000,
    N_vis=5,
    tensorboard=True,
    DDP=True,
)

dataset_config = DatasetConfig(
    dataset_name="matrixcity",
    datadir="/mnt/hwfile/landmark/train_data/base/MatrixCity/small_city/aerial/pose/block_A",
    downsample_train=1,
    lb=[-1.0378e01, -7.4430e00, -1.1921e-07],
    ub=[1.2078e00, 2.4078e00, 9.0000e-01],
)

model_config = InstantNGPConfig(
    # lb=[-1.0378e01, -7.4430e00, -1.1921e-07],
    # ub=[1.2078e00, 2.4078e00, 9.0000e-01],
    # train_near_far=[0.6, 3.5],
    train_near_far=[0.05, 1000],
    log2_hashmap_size=24,
    appearance_embedding_size=1063,
)

config = [train_config, dataset_config, model_config]
