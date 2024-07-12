from benchmarks.nerf.gridnerf.gridnerf import GridNeRFConfig
from landmark.nerf_components.configs import DatasetConfig, TrainConfig

train_config = TrainConfig(
    model_name="GridNeRF",
    expname="matrix_city_blockA_lowquality_ChannelParallel",
    basedir="./log",
    batch_size=8192,
    debug=False,
    vis_every=5000,
    n_iters=8000,
    N_vis=5,
    channel_parallel=True,
    channel_parallel_size=2,
)

dataset_config = DatasetConfig(
    dataset_name="matrixcity",
    datadir="/mnt/hwfile/landmark/train_data/base/MatrixCity/small_city/aerial/pose/block_A",
    downsample_train=1,
    lb=[-1.0378e01, -7.4430e00, -1.1921e-07],
    ub=[1.2078e00, 2.4078e00, 9.0000e-01],
)

model_config = GridNeRFConfig(
    ndims=1,
    resMode=[1],
    encode_app=False,
    nonlinear_density=True,
    add_nerf=5000,
    train_near_far=[0.6, 3.5],
    N_voxel_init=128**3,
    N_voxel_final=300**3,
    upsamp_list=[2000, 3000, 4000, 5500, 7000],
    update_AlphaMask_list=[2000, 4000, 8000, 10000],
    lr_decay_iters=8000,
    n_lamb_sigma=[16, 16, 16],
    n_lamb_sh=[48, 48, 48],
    fea2denseAct="relu",
    view_pe=2,
    fea_pe=2,
    L1_weight_inital=8e-5,
    L1_weight_rest=4e-5,
    rm_weight_mask_thre=1e-4,
    TV_weight_density=0.1,
    TV_weight_app=0.01,
    compute_extra_metrics=1,
    run_nerf=False,
    white_bkgd=True,
    sampling_opt=False,
    appearance_embedding_size=1063,
)

config = [train_config, dataset_config, model_config]
