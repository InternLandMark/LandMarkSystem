from benchmarks.nerf.nerfacto.nerfacto import NerfactoConfig
from landmark.nerf_components.configs import DatasetConfig, TrainConfig

train_config = TrainConfig(
    model_name="Nerfacto",
    expname="matrix_city_blockA_nerfacto",
    basedir="./log",
    batch_size=16384,
    # debug=True,
    debug=False,
    vis_every=1000,
    n_iters=100000,
    N_vis=5,
    tensorboard=True,
)

dataset_config = DatasetConfig(
    dataset_name="matrixcity",
    datadir="/mnt/hwfile/landmark/train_data/base/MatrixCity/small_city/aerial/pose/block_A",
    downsample_train=1,
    lb=[-1.0378e01, -7.4430e00, -1.1921e-07],
    ub=[1.2078e00, 2.4078e00, 9.0000e-01],
)

model_config = NerfactoConfig(
    # lb=[-1.0378e01, -7.4430e00, -1.1921e-07],
    # ub=[1.2078e00, 2.4078e00, 9.0000e-01],
    # train_near_far=[0.6, 3.5],
    train_near_far=[0.05, 1000],
    num_nerf_samples_per_ray=64,
    num_proposal_samples_per_ray=(512, 512),
    proposal_net_args_list=[
        {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 512, "use_linear": False},
        {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 7, "max_res": 2048, "use_linear": False},
    ],
    hidden_dim=256,
    hidden_dim_color=256,
    appearance_embed_dim=32,
    appearance_embedding_size=1063,
    max_res=8192,
    proposal_weights_anneal_max_num_iters=5000,
    log2_hashmap_size=21,
)

config = [train_config, dataset_config, model_config]
