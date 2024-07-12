from benchmarks.nerf.origin_gs.gs_model import Gaussian3DConfig
from landmark.nerf_components.configs import DatasetConfig, TrainConfig

train_config = TrainConfig(
    model_name="Gaussian3D",
    expname="Gaussian3D_matrix_city_blockA",
    basedir="./log",
    batch_size=1,
    n_iters=30000,
    test_iterations=[7000, 30000],
    save_iterations=[7000, 30000],
    tensorboard=True,
)

dataset_config = DatasetConfig(
    datadir="/mnt/hwfile/landmark/train_data/base/MatrixCity/small_city/aerial/pose/block_A",
    downsample_train=10,
    dataset_type="gaussian",
    dataset_name="city",
    preload=True,
    lb=[-1.0378e01, -7.4430e00, -1.1921e-07],
    ub=[1.2078e00, 2.4078e00, 9.0000e-01],
)

model_config = Gaussian3DConfig(
    position_lr_max_steps=30000,
    densification_interval=100,
    opacity_reset_interval=3000,
    densify_from_iter=500,
    densify_until_iter=15000,
    densify_grad_threshold=0.0002,
    white_bkgd=False,
    resolution=1,
    resolution_scales=[1.0],
)

config = [train_config, dataset_config, model_config]
