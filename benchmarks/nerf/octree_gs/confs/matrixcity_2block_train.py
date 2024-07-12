from benchmarks.nerf.octree_gs.gs_model import OctreeGSConfig
from landmark.nerf_components.configs import DatasetConfig, TrainConfig

train_config = TrainConfig(
    model_name="OctreeGS",
    expname="OctreeGS_matrix_city_blockA",
    basedir="./log",
    batch_size=1,
    n_iters=40000,
    test_iterations=[10000, 20000, 30000, 40000],
    save_iterations=[10000, 20000, 30000, 40000],
    tensorboard=True,
)

dataset_config = DatasetConfig(
    datadir="/mnt/hwfile/landmark/train_data/base/MatrixCity/small_city/aerial/pose/block_A/",
    downsample_train=1,
    dataset_type="gaussian",
    dataset_name="city",
    preload=True,
    lb=[-1.0378e01, -7.4430e00, -1.1921e-07],
    ub=[1.2078e00, 2.4078e00, 9.0000e-01],
)

model_config = OctreeGSConfig(
    appearance_dim=0,
    fork=2,
    base_layer=-1,
    visible_threshold=0.9,
    dist2level="round",
    update_ratio=0.2,
    progressive=True,
    dist_ratio=0.999,  # 0.99
    levels=-1,
    init_level=-1,
    extra_ratio=0.5,
    extra_up=0.01,
    white_bkgd=False,
    resolution=1,
    resolution_scales=[1.0],
    point_path=(
        "/mnt/hwfile/landmark/checkpoint/landmark_sys/vanilla_gs/Gaussian3D_matrix_city_blockA/refactor/point_cloud.ply"
    ),
)

config = [train_config, dataset_config, model_config]
