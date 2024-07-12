# pylint: disable=E1111, W0127
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_efficient_distloss import eff_distloss
from typing_extensions import Literal

from landmark.nerf_components.configs import ConfigClass
from landmark.nerf_components.data import Rays, Samples
from landmark.nerf_components.model import BaseNeRF
from landmark.nerf_components.model_components import (
    AlphaGridMask,
    AppearanceEmbedding,
    MLPDecoder,
    NeRF,
    TensorVMEncoding,
    VolumeEncoding,
    VolumeRenderer,
    raw2outputs,
)
from landmark.nerf_components.model_components.fields.mlp_decoder import (
    positional_encoding,
)
from landmark.nerf_components.ray_samplers import PDFSampler, UniformSampler
from landmark.train.utils.utils import st


class GridNeRFConfig(ConfigClass):
    """Configuration for Grid NeRF."""

    resMode: List[int] = [1]  # resolution_mode
    encode_app: bool = False  # whether to enable encode appearance
    run_nerf: bool = False
    nerf_D: int = 6
    nerf_D_a: int = 2
    nerf_W: int = 128
    nerf_freq: int = 16
    n_importance: int = 128
    nerf_n_importance: int = 16
    nonlinear_density: bool = False

    n_lamb_sigma: List[int] = [16, 16, 16]  # density_n_components
    n_lamb_sh: List[int] = [48, 48, 48]  # app_n_components

    data_dim_color: int = 27  # data_dim_color
    pos_pe: int = 6  # number of positional encoding for position    # pos_pe_dim
    view_pe: int = 6  # number of positional encoding for view    # view_pe_dim
    fea_pe: int = 6  # number of positional encoding for feature # feature_pe_dim
    featureC: int = 128  # hidden feature channel in MLP
    fea2denseAct: Literal["softplus", "relu"] = "relu"  # density_activation
    density_shift: float = -10  # shift density in softplus
    ndims: int = 1

    appearance_embedding_size: int
    """Size of the appearance embedding, equal to the training dataset size."""

    add_upsample: int = -1
    add_lpips: int = -1
    add_nerf: int = -1
    add_distort: int = -1
    patch_size: int = 128
    residnerf: bool = False
    train_near_far: list
    lr_init: float = 0.02
    lr_basis: float = 1e-3
    lr_decay_iters: int = -1
    lr_decay_target_ratio: float = 0.1
    lr_upsample_reset: bool = True

    L1_weight_inital: float = 0.0
    Ortho_weight: float = 0.0
    TV_weight_density: float = 0.0
    TV_weight_app: float = 0.0

    alpha_mask_thre: float = 0.0001
    nSamples: int = 1e6

    step_ratio: float = 0.5  # how many grids to walk in one step during sampling # sampling_step_ratio

    N_voxel_init: int = 100**3
    N_voxel_final: int = 300**3

    upsamp_list: list
    update_AlphaMask_list: list

    progressive_alpha: int = 0

    alpha_grid_reso: int = 256**3

    rayMarch_weight_thres: float = 0.001

    # render
    render_px: int = 720
    render_fov: float = 65.0
    render_nframes: int = 100
    render_skip: int = 1
    render_fps: int = 30
    render_spherical: bool = False
    render_spherical_zdiff: float = 1.0
    render_spherical_radius: float = 4.0
    render_downward: float = -45.0
    render_ncircle: float = 1
    render_path: bool = False
    render_pathid: int = 0
    render_near_far: Optional[list] = None
    render_lb: Optional[list] = None
    render_ub: Optional[list] = None
    render_batch_size: int = 8192
    distance_scale: float = 25

    sampling_opt: bool = False
    alpha_mask_filter_thre: float = 0
    white_bkgd: bool = False

    dynamic_fetching: bool = False
    neighbour_size: int = 9

    camera: str = "normal"

    L1_weight_rest: float = 0.0
    rm_weight_mask_thre: float = 0.0

    runtime: Literal["Torch", "Kernel"] = "Torch"

    def check_args(self):
        if isinstance(self.train_near_far, list):
            assert len(self.train_near_far) == 2
        if isinstance(self.train_near_far, torch.Tensor):
            assert self.train_near_far.shape == (2,)


class RGBDecoder(nn.Module):
    """Class for RGB decoder."""

    def __init__(self, n_component, resMode, app_dim, view_pe=6, fea_pe=6, featureC=128, encode_app=False) -> None:
        super().__init__()
        self.basis_mat = nn.Linear(sum(n_component) * len(resMode), app_dim, bias=False)
        self.encode_app = encode_app
        self.viewpe = view_pe
        self.feape = fea_pe
        if self.encode_app:
            in_mlpC = 2 * max(view_pe, 0) * 3 + 2 * fea_pe * app_dim + 3 * (view_pe > -1) + app_dim + 48
        else:
            in_mlpC = 2 * max(view_pe, 0) * 3 + 2 * fea_pe * app_dim + 3 * (view_pe > -1) + app_dim
        self.decoder = MLPDecoder(
            inChanel=in_mlpC,
            layer_width=featureC,
            num_layers=3,
            out_dim=3,
            out_activation=torch.nn.Sigmoid(),
        )

    def forward(self, samples_dir, app_features, app_latent):
        app_features = self.basis_mat(app_features)

        if self.encode_app and app_latent is not None:
            indata = [app_features, app_latent]
        else:
            indata = [app_features]
        if self.viewpe > -1:
            indata += [samples_dir]
        if self.feape > 0:
            indata += [positional_encoding(app_features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(samples_dir, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)

        rgb = self.decoder(mlp_in)
        return rgb


class Feature2Density(nn.Module):
    """
    Feature2Density
    """

    def __init__(self, sum_n_component, resModeLen, nonlinear_density, density_shift, fea2denseAct):
        super().__init__()
        self.feature2density = nn.Sequential()
        if nonlinear_density:
            self.feature2density.append(nn.Linear(sum_n_component * resModeLen, 1, bias=False))
            self.feature2density.append(nn.ReLU())
        if fea2denseAct == "softplus":
            self.feature2density.append(lambda x: nn.Softplus(x + density_shift))
        elif fea2denseAct == "relu":
            self.feature2density.append(nn.ReLU())
        else:
            raise NotImplementedError(f"Unknown density activation function: {fea2denseAct}.")

    def forward(self, sigma_feature):
        density = self.feature2density(sigma_feature)
        return density


class GridNeRF(BaseNeRF):
    """
    Base class for GridNeRF
    """

    def __init__(  # pylint: disable=W0102
        self,
        aabb,
        gridSize,
        device,
        alphaMask=None,
        near_far=[2.0, 6.0],
        scene_manager=None,
        grid_shape=None,  # state_dict's gridshape saved in kwargs
        config=None,
    ):
        super().__init__(
            config,
            scene_manager,
            aabb=aabb,
            gridSize=gridSize,
            device=device,
            alphaMask=alphaMask,
            grid_shape=grid_shape,
            near_far=near_far,
        )
        print(f"aabb.device = {aabb.device}, device = {device}", flush=True)
        self.run_nerf = config.run_nerf
        self.density_n_comp = config.n_lamb_sigma[: config.ndims]
        self.app_n_comp = config.n_lamb_sh[: config.ndims]
        self.app_dim = config.data_dim_color
        self.plane_division = config.plane_division

        self.matMode = [[0, 1], [0, 2], [1, 2]][: config.ndims]
        self.vecMode = [2, 1, 0][: config.ndims]
        self.comp_w = [1, 1, 1][: config.ndims]

        self.update_stepSize(gridSize)

        self.init_field_components()
        self.init_sampler()
        self.init_renderer()

    def init_sampler(self):
        config = self.config
        self.sampler = UniformSampler(
            num_samples=self.nSamples,
            step_size=self.stepSize,
            aabb=self.aabb,
            near_far=self.near_far,
        )
        self.pdf_sampler = PDFSampler(
            num_samples=config.nerf_n_importance,
        )

    def init_field_components(self):
        config = self.config
        grid_size = self.gridSize.clone()
        grid_size[:2] = self.gridSize[:2] // self.scene_manager.block_partition
        self.density_encoding = TensorVMEncoding(
            ndims=config.ndims,
            resolution_mode=config.resMode,
            n_component=config.n_lamb_sigma,
            grid_size=grid_size,
            param_shape=self.grid_shape["density_encoding"] if self.grid_shape else None,
            device=self.device,
        )
        self.rgb_encoding = TensorVMEncoding(
            ndims=config.ndims,
            resolution_mode=config.resMode,
            n_component=config.n_lamb_sh,
            grid_size=grid_size,
            param_shape=self.grid_shape["rgb_encoding"] if self.grid_shape else None,
            device=self.device,
        )

        self.feature2density = Feature2Density(
            sum(self.density_encoding.n_component),
            len(config.resMode),
            config.nonlinear_density,
            config.density_shift,
            config.fea2denseAct,
        )
        self.feature2density = self.feature2density.to(self.device)

        # rgb decoder
        self.rgb_decoder = RGBDecoder(
            self.rgb_encoding.n_component,
            config.resMode,
            self.app_dim,
            config.view_pe,
            config.fea_pe,
            config.featureC,
            config.encode_app,
        )
        self.rgb_decoder = self.rgb_decoder.to(self.device)

        if config.encode_app:
            self.embedding_app = AppearanceEmbedding(
                config.appearance_embedding_size, self.app_n_comp[0], self.device
            )  # (N, ch), N: total num of imgs in dataset

        if config.add_nerf > 0 or self.run_nerf:
            self.init_nerf()

    def init_renderer(self):
        config = self.config
        self.renderer = VolumeRenderer(white_bg=config.white_bkgd)

    def init_nerf(self):
        """create nerf branch"""
        config = self.config

        self.nerf = NeRF(
            config,
            sum(self.density_n_comp) * len(config.resMode),
            sum(self.app_n_comp) * len(config.resMode),
        ).to(self.device)
        self.residnerf = config.residnerf
        self.nerf_n_importance = config.nerf_n_importance
        print("init run_nerf", self.nerf)

    def channel_last(self):
        self.density_encoding.channel_last()
        self.rgb_encoding.channel_last()

    def update_stepSize(self, gridSize):
        """
        update step size according to new grid size

        Args:
            gridSize (list): grid size
        """
        config = self.config
        print("", flush=True)
        print(st.GREEN + "grid size" + st.RESET, gridSize, flush=True)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.tensor(gridSize, device=self.aabbSize.device)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * config.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print(
            st.BLUE + f"sample step size: {self.stepSize:.3f} unit" + st.RESET,
            flush=True,
        )
        print(st.BLUE + f"default number: {self.nSamples}" + st.RESET, flush=True)

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        """
        Computes dense alpha values and corresponding locations on a grid.

        Args:
            gridSize (tuple): The size of the grid to use for computing the dense alpha values.

        Returns:
            torch.Tensor: The dense alpha values.
            torch.Tensor: The corresponding locations.
        """
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, gridSize[0]),
                torch.linspace(0, 1, gridSize[1]),
                torch.linspace(0, 1, gridSize[2]),
            ),
            -1,
        ).to(self.device)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples
        alpha = torch.zeros_like(dense_xyz[..., 0])

        for i in range(gridSize[0]):
            alpha_pred = self.compute_alpha(dense_xyz[i].view(-1, 3), self.stepSize)
            alpha[i] = alpha_pred.view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200), increase_alpha_thresh=0):
        """
        Updates the alpha mask based on the current density field.

        Args:
            gridSize (tuple): The size of the grid to use for computing the alpha mask.
            increase_alpha_thresh (int): The number of orders of magnitude to increase the alpha mask threshold by.

        Returns:
            torch.Tensor: The new AABB for the alpha mask.
        """
        config = self.config
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha >= config.alpha_mask_thre * (10**increase_alpha_thresh)] = 1
        alpha[alpha < config.alpha_mask_thre * (10**increase_alpha_thresh)] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f" % (total / total_voxels * 100))
        return new_aabb

    def compute_alpha(self, xyz_locs, length=1):
        """
        Computes the alpha values for the given locations.

        Args:
            xyz_locs (torch.Tensor): The locations to compute alpha values for.
            length (float): The length of the segment to compute alpha values for.

        Returns:
            torch.Tensor: The alpha values.
        """
        config = self.config
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if alpha_mask.any():
            valid_samples = Samples(xyz=xyz_locs[alpha_mask], dirs=None, rank=config.mp_rank, group=config.mp_group)
            valid_samples.xyz = self.scene_manager.normalize_coord(valid_samples.xyz)
            self.scene_manager.assign_blockIdx(valid_samples, norm_coord=True)
            if config.branch_parallel:
                block_mask = valid_samples.mask
                if block_mask.sum():  # in case all samples are out of mask screen
                    sigma_feature = self.density_encoding(valid_samples.xyzb[block_mask])
                    validsigma = self.feature2density(sigma_feature).squeeze(-1).float()
                    valid_samples.sigma[block_mask] = validsigma
                valid_samples.sync_sigma()
            else:
                sigma_feature = self.density_encoding(valid_samples.xyzb)
                validsigma = self.feature2density(sigma_feature).squeeze(-1).float()
                valid_samples.sigma = validsigma

            sigma[alpha_mask] = valid_samples.sigma

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])
        return alpha

    def distortion_loss(self, weight, z_vals, dists):
        # compute distort loss
        w = weight
        m = torch.cat(
            (
                (z_vals[:, 1:] + z_vals[:, :-1]) * 0.5,
                ((z_vals[:, 1:] + z_vals[:, :-1]) * 0.5)[:, -1:],
            ),
            dim=-1,
        )
        dist_loss = 0.01 * eff_distloss(w, m, dists).unsqueeze(0)
        return dist_loss

    def forward(
        self,
        rays_chunk,
        app_code=None,
        N_samples=-1,
    ):
        config = self.config
        if config.is_train or (not config.is_train and not config.sampling_opt):
            # dense sample
            samples, ray_valid = self.sampler(rays_chunk, num_samples=N_samples, random_sampling=config.is_train)

            if self.alphaMask is not None:
                filter_thresh = torch.tensor([config.alpha_mask_filter_thre], device=rays_chunk.device)
                ray_valid = self.alphaMask(samples.xyz, filter_thresh, ray_valid)

            self.sigma_batch = ray_valid.sum()

            if ray_valid.any():
                samples.xyz = self.scene_manager.normalize_coord(samples.xyz)
                if config.branch_parallel:
                    valid_samples = samples[ray_valid]
                    self.scene_manager.assign_blockIdx(valid_samples, norm_coord=True)
                    block_mask = valid_samples.mask
                    self.sigma_batch = block_mask.sum()
                    if block_mask.sum():
                        sigma_feature = self.density_encoding(valid_samples.xyzb[block_mask])
                        validsigma = self.feature2density(sigma_feature).squeeze(-1).float()
                        valid_samples.sigma[block_mask] = validsigma
                    valid_samples.sync_sigma()
                    samples.sigma[ray_valid] = valid_samples.sigma
                else:
                    sigma_feature = self.density_encoding(samples.xyzb)
                    validsigma = self.feature2density(sigma_feature).squeeze(-1).float()
                    samples.sigma = validsigma

            weight = self.renderer.get_weights(samples.sigma, samples.dists * config.distance_scale)

            app_mask = weight > config.rayMarch_weight_thres
            self.app_batch = app_mask.sum()
            if app_mask.any():
                valid_app_samples = samples[app_mask]
                self.scene_manager.assign_blockIdx(valid_app_samples, norm_coord=True)
                app_latent = (
                    self.embedding_app(valid_app_samples.camera_idx, valid_app_samples.xyz, app_code)
                    if config.encode_app
                    else None
                )
                samples_dir = valid_app_samples.dirs
                if config.branch_parallel:
                    block_mask = valid_app_samples.mask
                    self.app_batch = block_mask.sum()

                    if block_mask.sum():
                        app_features = self.rgb_encoding(valid_app_samples.xyzb[block_mask])
                        if app_latent is not None:
                            app_latent = app_latent[block_mask]
                        samples_dir = samples_dir[block_mask]
                        valid_rgbs = self.rgb_decoder(samples_dir, app_features, app_latent).float()
                        valid_app_samples.rgb[block_mask] = valid_rgbs
                    valid_app_samples.sync_rgb()
                else:
                    app_features = self.rgb_encoding(valid_app_samples.xyzb)
                    valid_rgbs = self.rgb_decoder(samples_dir, app_features, app_latent).float()
                    valid_app_samples.rgb = valid_rgbs

                samples.rgb[app_mask] = valid_app_samples.rgb
            else:
                if config.branch_parallel:
                    if config.is_train:
                        tmp_input = valid_samples[:1]
                        app_features = self.rgb_encoding(tmp_input.xyzb)
                        app_latent = None
                        if config.encode_app:
                            app_latent = self.embedding_app(tmp_input.camera_idx, tmp_input.xyz, app_code)
                        valid_rgbs = self.rgb_decoder(tmp_input.dirs, app_features, app_latent).float()
                        valid_samples.rgb[:1] = valid_rgbs
                    valid_samples.sync_rgb()
                    samples.rgb[ray_valid] = valid_samples.rgb

            random_white_bg_ratio = 0.5 if self.training else 0.0
            rgb_map, depth_map = self.renderer(weight, samples.rgb, samples.z_vals, random_white_bg_ratio)

            if not config.is_train:
                outputs = (rgb_map, depth_map)
            else:
                # compute distort loss
                dist_loss = self.distortion_loss(weight * app_mask, samples.z_vals, samples.dists)
                outputs = {
                    "rgb_map": rgb_map,
                    "depth_map": depth_map,
                    "distort_loss": dist_loss,
                }

        if (not config.is_train and config.sampling_opt) or (config.is_train and self.run_nerf):
            # importance sample for grid branch (optional)
            samples, ray_valid = self.sampler(
                rays_chunk,
                num_samples=config.n_importance,
                random_sampling=config.is_train,
                sample_within_hull=True,
            )
            if self.alphaMask is not None:
                filter_thresh = torch.tensor([config.alpha_mask_filter_thre], device=rays_chunk.device)
                ray_valid = self.alphaMask(samples.xyz, filter_thresh, ray_valid)

            self.sigma_batch = ray_valid.sum()

            if ray_valid.any():
                samples.xyz = self.scene_manager.normalize_coord(samples.xyz)
                if config.branch_parallel:
                    valid_samples = samples[ray_valid]
                    self.scene_manager.assign_blockIdx(valid_samples, norm_coord=True)
                    block_mask = valid_samples.mask
                    self.sigma_batch = block_mask.sum()
                    if block_mask.sum():
                        sigma_feature = self.density_encoding(valid_samples.xyzb[block_mask])
                        validsigma = self.feature2density(sigma_feature).squeeze(-1).float()
                        valid_samples.sigma[block_mask] = validsigma
                    valid_samples.sync_sigma()
                    samples.sigma[ray_valid] = valid_samples.sigma
                else:
                    self.scene_manager.assign_blockIdx(samples, norm_coord=True)
                    valid_samples = samples[ray_valid]
                    valid_sigma_feature = self.density_encoding(valid_samples.xyzb)
                    valid_sigma = self.feature2density(valid_sigma_feature).squeeze(-1).float()
                    samples.sigma[ray_valid] = valid_sigma

            weight = self.renderer.get_weights(samples.sigma, samples.dists * config.distance_scale)
            app_mask = weight > config.rayMarch_weight_thres
            self.app_batch = app_mask.sum()

            if app_mask.any():
                if config.branch_parallel:
                    valid_app_samples = samples[app_mask]
                    self.scene_manager.assign_blockIdx(valid_app_samples, norm_coord=True)
                    app_latent = (
                        self.embedding_app(valid_app_samples.camera_idx, valid_app_samples.xyz, app_code)
                        if config.encode_app
                        else None
                    )
                    samples_dir = valid_app_samples.dirs

                    block_mask = valid_app_samples.mask
                    self.app_batch = block_mask.sum()
                    if block_mask.sum():
                        app_features = self.rgb_encoding(valid_app_samples.xyzb[block_mask])
                        if app_latent is not None:
                            app_latent = app_latent[block_mask]
                        samples_dir = samples_dir[block_mask]
                        valid_rgbs = self.rgb_decoder(samples_dir, app_features, app_latent).float()
                        valid_app_samples.rgb[block_mask] = valid_rgbs
                    valid_app_samples.sync_rgb()
                    samples.rgb[app_mask] = valid_app_samples.rgb
                else:
                    valid_app_samples = samples[app_mask]
                    app_features = self.rgb_encoding(valid_app_samples.xyzb)
                    app_latent = (
                        self.embedding_app(valid_app_samples.camera_idx, valid_app_samples.xyzb, app_code)
                        if config.encode_app
                        else None
                    )
                    rgb = self.rgb_decoder(valid_app_samples.dirs, app_features, app_latent).float()
                    samples.rgb[app_mask] = rgb
            else:
                if config.branch_parallel:
                    if config.is_train:
                        tmp_input = valid_samples[:1]
                        app_features = self.rgb_encoding(tmp_input.xyzb)
                        app_latent = None
                        if config.encode_app:
                            app_latent = self.embedding_app(tmp_input.camera_idx, tmp_input.xyz, app_code)
                        valid_rgbs = self.rgb_decoder(tmp_input.dirs, app_features, app_latent).float()
                        valid_samples.rgb[:1] = valid_rgbs
                    valid_samples.sync_rgb()
                    samples.rgb[ray_valid] = valid_samples.rgb

            random_white_bg_ratio = 0.5 if self.training else 0.0
            rgb_map, depth_map = self.renderer(weight, samples.rgb, samples.z_vals, random_white_bg_ratio)
            if not config.is_train:
                outputs = (rgb_map, depth_map)
            else:
                dist_loss = self.distortion_loss(weight * app_mask, samples.z_vals, samples.dists)
                outputs.update(
                    {
                        "rgb_map1": rgb_map,
                        "depth_map1": depth_map,
                        "distort_loss1": dist_loss,
                    }
                )

                samples = self.pdf_sampler(rays_chunk, samples.z_vals, weight[..., 1:-1], self.nerf_n_importance)
                orig_samples_xyz = samples.xyz.clone().detach()
                samples.xyz = self.scene_manager.normalize_coord(samples.xyz)
                self.scene_manager.assign_blockIdx(samples, norm_coord=True)
                self.sigma_batch = self.app_batch = self.nerf_batch = len(samples)

                app_latent = (
                    self.embedding_app(samples.camera_idx, samples.xyz, app_code) if config.encode_app else None
                )  # TODO a litte different from the original implementation on the processing of fake idxs
                nray, npts = samples.xyz.shape[:2]
                if config.branch_parallel:
                    block_mask = samples.mask
                    self.sigma_batch = self.app_batch = self.nerf_batch = block_mask.sum()

                    if block_mask.sum():
                        den_feat = self.density_encoding(samples.xyzb[block_mask])
                        app_feat = self.rgb_encoding(samples.xyzb[block_mask])
                        if app_latent is not None:
                            app_latent = app_latent[block_mask]
                        nerf_outputs = self.nerf(  # pylint: disable=E1102
                            orig_samples_xyz[block_mask],
                            samples.dirs[block_mask],
                            den_feat,
                            app_feat,
                            app_latent,
                        ).float()  # TODO use parallel version of nerf branch
                        nerf_rgb = nerf_outputs[..., :3]
                        nerf_sigma = nerf_outputs[..., -1]
                        samples.rgb[block_mask] = nerf_rgb
                        samples.sigma[block_mask] = nerf_sigma
                    samples.sync_rgb()
                    samples.sync_sigma()
                    nerf_out = torch.cat((samples.rgb, samples.sigma.unsqueeze(-1)), dim=-1).view(nray, npts, -1)
                    extras = raw2outputs(nerf_out, samples.dists)

                else:
                    den_feat = self.density_encoding(samples.xyzb.view(-1, 4))
                    app_feat = self.rgb_encoding(samples.xyzb.view(-1, 4))
                    nerf_outputs = self.nerf(  # pylint: disable=E1102
                        orig_samples_xyz.view(-1, 3), samples.dirs.view(-1, 3), den_feat, app_feat, app_latent
                    )
                    nerf_out = nerf_outputs.view(nray, npts, -1)
                    extras = raw2outputs(nerf_out, samples.dists)

                depth_map_nerf = torch.sum(extras["weights"] * samples.z_vals, -1)

                if self.residnerf:
                    outputs.update(
                        {
                            "rgb_map_nerf": (extras["rgb_map"] + rgb_map).clamp(min=0.0, max=1.0),
                            "depth_map_nerf": depth_map_nerf,
                        }
                    )
                else:
                    outputs.update(
                        {
                            "rgb_map_nerf": extras["rgb_map"],
                            "depth_map_nerf": depth_map_nerf,
                        }
                    )

        return outputs

    def render_all_rays(self, rays, chunk_size: Optional[int] = None, N_samples=-1, idxs=None, app_code=0):
        assert app_code is not None
        all_ret = {"rgb_map": [], "depth_map": []}
        N_rays_all = rays.shape[0]
        for chunk_idx in range(N_rays_all // chunk_size + int(N_rays_all % chunk_size > 0)):
            rays_chunk = rays[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
            idxs_chunk = idxs[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]

            rays_chunk = torch.cat((rays_chunk, idxs_chunk.unsqueeze(-1)), dim=-1)
            if self.config.is_train:  # TODO to be optimized
                rays_chunk = Rays(
                    origin=None,
                    dirs=None,
                    camera_idx=None,
                    near=self.config.train_near_far[0],
                    far=self.config.train_near_far[1],
                    _data=rays_chunk,
                    rank=self.config.mp_rank,
                    group=self.config.mp_group,
                )

            ret = self.forward(  # pylint: disable=E1123
                rays_chunk,
                N_samples=N_samples,  # TODO not support other models except gridnerf (frank)
            )
            if not self.config.is_train:
                rgb_map, depth_map = ret
                all_ret["rgb_map"].append(rgb_map)
                all_ret["depth_map"].append(depth_map)
            else:
                for k in ret:
                    if k not in all_ret:
                        all_ret[k] = []
                    all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(v, 0) for k, v in all_ret.items()}
        return all_ret

    # optimization
    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        """
        Returns the parameter groups for optimization.

        Args:
            lr_init_spatial (float): The initial learning rate for spatial parameters.
            lr_init_network (float): The initial learning rate for network parameters.

        Returns:
            list: A list of parameter groups for optimization.
        """
        config = self.config
        grad_vars = [
            {"params": self.density_encoding.parameters(), "lr": lr_init_spatial},
            {"params": self.rgb_encoding.parameters(), "lr": lr_init_spatial},
            {"params": self.feature2density.parameters(), "lr": lr_init_network},
            {"params": self.rgb_decoder.parameters(), "lr": lr_init_network},
        ]
        if config.encode_app:
            grad_vars += [{"params": self.embedding_app.parameters(), "lr": lr_init_network}]
        if self.run_nerf:
            grad_vars += [{"params": self.nerf.parameters(), "lr": 5e-4}]
        return grad_vars

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        """
        Upsamples the volume grid to a target resolution.

        Args:
            res_target (list): The target resolution.
        """
        grid_size = torch.tensor(res_target, device=self.device)
        grid_size[:2] = grid_size[:2] // torch.tensor(self.scene_manager.block_partition, device=self.device)
        self.density_encoding.upsample_volume_grid(grid_size)
        self.rgb_encoding.upsample_volume_grid(grid_size)
        self.update_stepSize(res_target)
        self.sampler.step_size = self.stepSize
        self.sampler.num_samples = self.nSamples
        print(f"upsamping to {res_target}")

    def merge_ckpts(self, ckpt_fp_list, kwargs_fp_list, map_location="cpu"):
        config = self.config
        if config.rank != 0:
            return

        state_dicts = [torch.load(ckpt_fp, map_location=map_location) for ckpt_fp in ckpt_fp_list]
        kwargs = torch.load(kwargs_fp_list[0], map_location=map_location)

        from collections import OrderedDict

        merged_state_dict = OrderedDict()
        grid_shape = OrderedDict()  # for saving grid shape
        for name, module in self.named_children():
            if isinstance(module, VolumeEncoding):
                encoding_state_dicts = [OrderedDict() for _ in range(len(state_dicts))]
                for key in self.state_dict().keys():
                    if name in key.split(".")[0]:
                        for i in range(len(state_dicts)):
                            encoding_state_dicts[i][key] = state_dicts[i][key]
                merged_encoding_state_dict = module.merge_state_dicts(encoding_state_dicts)
                merged_state_dict.update(merged_encoding_state_dict)
                grid_shape[name] = {}
                for key in merged_encoding_state_dict.keys():
                    grid_shape[name].update({key.replace(name + ".", ""): merged_encoding_state_dict[key].shape})
            else:
                for key in self.state_dict().keys():
                    if name in key.split(".")[0]:
                        # check consistency
                        for i in range(len(state_dicts) - 1):
                            assert torch.all(state_dicts[i][key] == state_dicts[i + 1][key])
                        merged_state_dict.update({key: state_dicts[0][key]})

        kwargs.update({"grid_shape": grid_shape})
        return merged_state_dict, kwargs

    def edit_model(self, edit_mode: dict):
        pass

    def register_grid_ddp_hook(self):
        """
        A DDP communication hook function. When training with branch / plane parallel,
        it should be registered on modules that belong to grid branch.
        """

        def sigma_allreduce_hook(
            process_group: dist.ProcessGroup, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:

            batch = self.sigma_batch
            batch_sum = batch.clone()

            group_to_use = process_group if process_group is not None else dist.group.WORLD

            dist.all_reduce(batch_sum, group=group_to_use)
            if batch_sum == 0:
                weight = 0
            else:
                weight = batch / batch_sum

            # Apply the division first to avoid overflow, especially for FP16.
            tensor = bucket.buffer()
            tensor.mul_(weight)

            return (
                dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future().then(lambda fut: fut.value()[0])
            )

        def app_allreduce_hook(
            process_group: dist.ProcessGroup, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:

            batch = self.app_batch
            batch_sum = batch.clone().detach()

            group_to_use = process_group if process_group is not None else dist.group.WORLD

            dist.all_reduce(batch_sum, group=group_to_use)
            if batch_sum == 0:
                weight = 0
            else:
                weight = batch / batch_sum

            # Apply the division first to avoid overflow, especially for FP16.
            tensor = bucket.buffer()
            tensor.mul_(weight)

            return (
                dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future().then(lambda fut: fut.value()[0])
            )

        if isinstance(self.density_encoding.line_coef, DDP) and isinstance(self.rgb_encoding.line_coef, DDP):
            # if the line is wrapped by DDP, it should be registered ddp comm hook
            self.density_encoding.line_coef.register_comm_hook(self.config.mp_group, sigma_allreduce_hook)
            self.rgb_encoding.line_coef.register_comm_hook(self.config.mp_group, app_allreduce_hook)
        self.feature2density.register_comm_hook(self.config.mp_group, sigma_allreduce_hook)
        self.rgb_decoder.register_comm_hook(self.config.mp_group, app_allreduce_hook)
        if self.config.encode_app:
            self.embedding_app.register_comm_hook(self.config.mp_group, app_allreduce_hook)

    def register_nerf_ddp_hook(self):
        """
        A DDP communication hook function. When training with branch / plane parallel,
        it should be registered on modules that belong to nerf branch.
        """

        def nerf_allreduce_hook(
            process_group: dist.ProcessGroup, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            batch = self.nerf_batch
            batch_sum = batch.clone().detach()

            group_to_use = process_group if process_group is not None else dist.group.WORLD

            dist.all_reduce(batch_sum, group=group_to_use)
            if batch_sum == 0:
                weight = 0
            else:
                weight = batch / batch_sum

            # Apply the division first to avoid overflow, especially for FP16.
            tensor = bucket.buffer()
            tensor.mul_(weight)

            return (
                dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future().then(lambda fut: fut.value()[0])
            )

        self.nerf.register_comm_hook(self.config.mp_group, nerf_allreduce_hook)


def n_to_reso(n_voxels, bbox):
    """Compute new grid size.
    Args:
        n_voxels (int): The number of voxels
        bbox (torch.Tensor): The representation of Axis Aligned Bounding Box(aabb)
    Returns:
        list: The current grid size
    """
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso) / step_ratio)
