# pylint: disable=E1111
from collections import OrderedDict
from typing import List, Optional, Union

import nerfacc
import torch
import torch.distributed as dist
from torch import nn
from typing_extensions import Literal

from landmark.nerf_components.configs import ConfigClass
from landmark.nerf_components.data import Samples
from landmark.nerf_components.model import BaseNeRF
from landmark.nerf_components.model_components import (
    HashEncoding,
    MLPDecoder,
    SHEncoding,
    VolumeRenderer,
)
from landmark.nerf_components.ray_samplers import VolumetricSampler
from landmark.nerf_components.utils.activation_utils import trunc_exp


class InstantNGPConfig(ConfigClass):
    """Configuration for InstantNGP."""

    # hash encoding
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Minimum resolution of the hashmap for the base mlp."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""

    # mlp decoder
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    appearance_embedding_dim: int = 32
    """Dimension of the appearance embedding."""
    appearance_embedding_size: int
    """Size of the appearance embedding, equal to the training dataset size."""
    geo_feat_dim: int = 15
    num_layers: int = 2
    num_layers_color: int = 3

    # sampler
    grid_resolution: int = 128
    """Resolution of the grid used for the field, for OccGrid."""
    grid_levels: int = 4
    """Levels of the grid used for the field, for OccGrid."""
    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering, for OccGrid."""

    # renderer
    background_color: Literal["black", "white"] = "white"
    """The color that is given to untrained areas."""

    # basics
    train_near_far: List[float]
    aabb_norm_range: Union[List[float], torch.Tensor] = [-1, 1]
    white_bkgd: bool = False
    camera: str = "normal"
    patch_size: int = 128
    add_upsample: int = -1
    add_lpips: int = -1
    runtime: Literal["Torch", "Kernel"] = "Kernel"

    # optim
    lr_init: float = 1e-2
    lr_final: float = 0.0001
    lr_decay_max_steps: int = 200000
    eps: float = 1e-15

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
    render_near_far: Optional[List[float]] = None
    render_lb: Optional[List[float]] = None
    render_ub: Optional[List[float]] = None
    render_batch_size: int = 8192
    distance_scale: float = 25

    render_stratified_sampling: bool = True


class InstantNGP(BaseNeRF):
    """
    Base class for InstantNGP

    Args:
        aabb (torch.Tensor): Axis-aligned bounding box
        device (torch.device): Device that the model runs on.
        near_far (list): Near and far plane along the sample ray
        is_train (bool): Distinguish between training and rendering in order to init modules correctly.
    """

    def __init__(
        self,
        aabb,
        device,
        near_far,
        scene_manager=None,
        config=None,
        group=None,
    ):
        super().__init__(config=config, scene_manager=scene_manager, aabb=aabb, device=device, near_far=near_far)

        self.group = group
        print(f"aabb.device = {aabb.device}, device = {device}", flush=True)

        # new features
        self.plane_division = config.plane_division
        self.num_hash_table = config.plane_division[0] * config.plane_division[1]
        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.config.render_step_size = ((self.aabb[1] - self.aabb[0]) ** 2).sum().sqrt().item() / 1000

        self.init_field_components()
        self.init_sampler()
        self.init_renderer()

    def init_sampler(self):
        config = self.config
        aabb = torch.cat((self.aabb[0], self.aabb[1]))
        # Occupancy Grid.
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        ).to(self.device)

        # Sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.density_fn,
            near_far=self.near_far,
            step_size=config.render_step_size,
            alpha_thre=config.alpha_thre,
            cone_angle=config.cone_angle,
        )

    def init_renderer(self):
        config = self.config
        self.renderer = VolumeRenderer(white_bg=config.white_bkgd)

    def init_field_components(self):
        config = self.config

        self.geo_feat_dim = config.geo_feat_dim

        self.base_res = config.base_res
        self.step = 0

        self.direction_encoding = SHEncoding(
            levels=4,
        )
        self.direction_encoding = self.direction_encoding.to(self.device)

        self.mlp_base_grid = HashEncoding(
            num_levels=config.num_levels,
            min_res=config.base_res,
            max_res=config.max_res,
            log2_hashmap_size=config.log2_hashmap_size,
            features_per_level=config.features_per_level,
            device=self.device,
            num_hash_table=self.num_hash_table,
        )
        self.mlp_base_grid = self.mlp_base_grid.to(self.device)

        self.mlp_base_mlp = MLPDecoder(
            inChanel=self.mlp_base_grid.get_out_dim(),
            num_layers=config.num_layers,
            layer_width=config.hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            out_activation=None,
            mlp_bias=False,
        ).to(self.device)

        self.mlp_base_mlp = self.mlp_base_mlp.to(self.device)

        self.mlp_head = MLPDecoder(
            inChanel=self.direction_encoding.get_out_dim() + self.geo_feat_dim + config.appearance_embedding_dim,
            num_layers=config.num_layers_color,
            layer_width=config.hidden_dim_color,
            out_dim=3,
            out_activation=nn.Sigmoid(),
            mlp_bias=False,
        )
        self.mlp_head = self.mlp_head.to(self.device)

        self.embedding_appearance = torch.nn.Embedding(
            config.appearance_embedding_size, config.appearance_embedding_dim
        )
        self.embedding_appearance = self.embedding_appearance.to(self.device)

    def density_fn(self, positions):
        """Returns only the density. Used primarily with the density grid."""
        origins = positions
        directions = torch.ones_like(positions)
        starts = torch.zeros_like(positions[..., :1])
        ends = torch.zeros_like(positions[..., :1])

        density, _ = self.get_density(origins, directions, (starts + ends) / 2)

        return density

    def get_density(self, origins, directions, z_vals):
        """Computes and returns the densities."""
        pos = origins + directions * z_vals

        positions = get_normalized_positions(pos, self.aabb)  # TODO use scene_manger.normalize_coord
        # positions = self.scene_manager.normalize_coord(pos)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        # selector = ((positions > -1.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        rank = self.config.mp_rank if self.config.branch_parallel else None
        samples = Samples(xyz=positions, dirs=directions, z_vals=z_vals, rank=rank, group=self.group)
        self.scene_manager.assign_blockIdx(samples)
        if self.config.branch_parallel:

            block_mask = samples.mask
            # TODO dtype is same to mlp_base_grid's output
            global_base_mlp_out = torch.zeros(
                size=(positions.shape[0], self.mlp_base_mlp.module.out_dim - 1), device=self.device
            )
            if block_mask.sum():
                block_positions_flat = samples.xyzb[block_mask].view(-1, 4)
                hash_features = self.mlp_base_grid(block_positions_flat)
                h = self.mlp_base_mlp(hash_features).view(samples.xyz[block_mask].shape[0], -1)
                density_before_activation, block_base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
                density = trunc_exp(density_before_activation.to(positions))
                samples.sigma[block_mask] = density.view(-1)
                global_base_mlp_out = global_base_mlp_out.to(block_base_mlp_out)
                global_base_mlp_out[block_mask] = block_base_mlp_out

                samples.sync_sigma()
                # TODO base_mlp_out sync need to encapsulation
                sample_nums = [mask.sum() for mask in samples.masks]
                base_mlp_out_tensor_list = [
                    torch.zeros((num, self.mlp_base_mlp.module.out_dim - 1), device=self.device).to(block_base_mlp_out)
                    for num in sample_nums
                ]
                with torch.no_grad():
                    dist.all_gather(base_mlp_out_tensor_list, global_base_mlp_out[block_mask], group=samples.group)

                base_mlp_out_tensor_list[self.config.rank] = global_base_mlp_out[block_mask]
                for mask, validbase_mlp_out in zip(samples.masks, base_mlp_out_tensor_list):
                    global_base_mlp_out.masked_scatter_(
                        mask.unsqueeze(-1).expand(*mask.shape, self.mlp_base_mlp.module.out_dim - 1), validbase_mlp_out
                    )

            else:
                block_positions_flat = samples.xyzb[:1].view(-1, 4)
                hash_features = self.mlp_base_grid(block_positions_flat)
                h = self.mlp_base_mlp(hash_features).view(samples.xyz[:1].shape[0], -1)
                density_before_activation, block_base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
                density = trunc_exp(density_before_activation.to(positions))
                global_base_mlp_out = global_base_mlp_out.to(block_base_mlp_out)

                samples.sync_sigma()
                # TODO base_mlp_out sync need to encapsulation
                sample_nums = [mask.sum() for mask in samples.masks]
                base_mlp_out_tensor_list = [
                    torch.zeros((num, self.mlp_base_mlp.module.out_dim - 1), device=self.device).to(block_base_mlp_out)
                    for num in sample_nums
                ]
                with torch.no_grad():
                    dist.all_gather(base_mlp_out_tensor_list, global_base_mlp_out[block_mask], group=samples.group)

                base_mlp_out_tensor_list[self.config.rank] = global_base_mlp_out[block_mask]
                for mask, validbase_mlp_out in zip(samples.masks, base_mlp_out_tensor_list):
                    global_base_mlp_out.masked_scatter_(
                        mask.unsqueeze(-1).expand(*mask.shape, self.mlp_base_mlp.module.out_dim - 1), validbase_mlp_out
                    )

                global_base_mlp_out[:1] = block_base_mlp_out

            base_mlp_out = global_base_mlp_out
            density = samples.sigma.view(-1, 1)
        else:
            positions_flat = samples.xyzb.view(-1, 4)
            hash_features = self.mlp_base_grid(positions_flat)
            h = self.mlp_base_mlp(hash_features).view(positions.shape[0], -1)
            density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
            self._density_before_activation = density_before_activation

            # Rectifying the density with an exponential is much more stable than a ReLU or
            # softplus, because it enables high post-activation (float32) density outputs
            # from smaller internal (float16) parameters.
            density = trunc_exp(density_before_activation.to(positions))

        density = density * selector[..., None]
        return density, base_mlp_out

    def update_occupancy_grid(self, step: int):
        self.occupancy_grid.update_every_n_steps(
            step=step,
            occ_eval_fn=lambda x: self.density_fn(x) * self.config.render_step_size,
        )
        if self.config.branch_parallel or self.config.DDP:
            if step % 16 == 0 and self.occupancy_grid.training:
                dist.broadcast(self.occupancy_grid.occs, src=0)
                self.occupancy_grid.binaries = (
                    self.occupancy_grid.occs > torch.clamp(self.occupancy_grid.occs.mean(), max=0.01)
                ).view(self.occupancy_grid.binaries.shape)

    def forward(self, rays_chunk):
        with torch.no_grad():
            ray_samples, valid_mask = self.sampler(
                ray_bundle=rays_chunk,
                training=self.training,
                render_stratified_sampling=self.config.render_stratified_sampling,
            )
        valid_samples = ray_samples[valid_mask]
        origins, directions, camera_indices = (
            valid_samples.xyz,
            valid_samples.dirs,
            valid_samples.camera_idx,
        )

        density, density_embedding = self.get_density(origins, directions, valid_samples.z_vals[..., None])

        camera_indices = camera_indices.squeeze()
        directions = get_normalized_directions(directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)
        outputs_shape = directions.shape[:-1]

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices.to(torch.int))
        else:
            embedded_appearance = torch.zeros(
                (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
            )

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ],
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)

        # accumulation
        ray_samples.rgb[valid_mask] = rgb
        ray_samples.sigma[valid_mask] = density[..., 0]

        density = ray_samples.sigma
        rgb = ray_samples.rgb

        dists = torch.full(ray_samples.shape, self.config.render_step_size, device=ray_samples.device)

        weight = self.renderer.get_weights(ray_samples.sigma, dists)
        rgb_map, depth_map = self.renderer(weight, ray_samples.rgb, ray_samples.z_vals, 1.0)

        outputs = {
            "rgb_map": rgb_map,
            "depth_map": depth_map.unsqueeze(-1),
        }
        return outputs

    def get_optparam_groups(self):
        config = self.config
        grad_vars = [
            {"params": self.mlp_base_grid.parameters(), "lr": config.lr_init},
            {"params": self.mlp_base_mlp.parameters(), "lr": config.lr_init},
            {"params": self.mlp_head.parameters(), "lr": config.lr_init},
            {"params": self.embedding_appearance.parameters(), "lr": config.lr_init},
        ]
        return grad_vars

    def merge_ckpts(self, ckpt_fp_list, kwargs_fp_list, map_location="cpu"):
        config = self.config
        if config.rank != 0:
            return

        state_dicts = [torch.load(ckpt_fp, map_location=map_location) for ckpt_fp in ckpt_fp_list]
        kwargs = torch.load(kwargs_fp_list[0], map_location=map_location)

        merged_state_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, HashEncoding):
                encoding_state_dicts = [OrderedDict() for _ in range(len(state_dicts))]
                for key in self.state_dict().keys():
                    if name in key.split(".")[0]:
                        for i in range(len(state_dicts)):
                            encoding_state_dicts[i][key.replace("params", f"{i}.params")] = state_dicts[i][key]
                for d in encoding_state_dicts:
                    merged_state_dict.update(d)

            elif isinstance(module, nerfacc.OccGridEstimator):
                for key in self.state_dict().keys():
                    if name in key.split(".")[0]:
                        merged_state_dict.update({key: state_dicts[0][key]})

            elif isinstance(module, VolumetricSampler):
                for key in self.state_dict().keys():
                    if name in key.split(".")[0]:
                        merged_state_dict.update({key: state_dicts[0][key]})

            else:
                for key in self.state_dict().keys():
                    if name in key.split(".")[0]:
                        # check consistency
                        for i in range(len(state_dicts) - 1):
                            assert torch.all(state_dicts[i][key] == state_dicts[i + 1][key])
                        merged_state_dict.update({key: state_dicts[0][key]})

        return merged_state_dict, kwargs


def get_normalized_positions(positions, aabb):
    """Return normalized positions in range [0, 1] based on the aabb axis-aligned bounding box.

    Args:
        positions: the xyz positions
        aabb: the axis-aligned bounding box
    """
    aabb_lengths = aabb[1] - aabb[0]
    normalized_positions = (positions - aabb[0]) / aabb_lengths
    return normalized_positions


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0
