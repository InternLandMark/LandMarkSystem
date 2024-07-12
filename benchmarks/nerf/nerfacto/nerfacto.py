# pylint: disable=E1111
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
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
from landmark.nerf_components.ray_samplers import ProposalNetworkSampler
from landmark.nerf_components.utils.activation_utils import trunc_exp


class NerfactoConfig(ConfigClass):
    """Configuration for InstantNGP."""

    aabb_norm_range: Union[List[float], torch.Tensor] = [-1, 1]

    # hash encoding
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
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
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""
    appearance_embedding_size: int
    """Size of the appearance embedding, equal to the training dataset size."""

    geo_feat_dim: int = 15
    num_layers: int = 2
    num_layers_color: int = 3

    # sampler
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    proposal_net_args_list: List[Dict] = [
        {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
        {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
    ]
    """Arguments for the proposal density fields."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""

    # loss
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""

    # basics
    train_near_far: List[float]
    white_bkgd: bool = False
    camera: str = "normal"
    patch_size: int = 128
    add_upsample: int = -1
    add_lpips: int = -1
    runtime: Literal["Torch", "Kernel"] = "Kernel"

    # optim
    lr: float = 1e-2
    lr_final: float = 0.0001
    lr_decay_max_steps: int = 50000
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


class Nerfacto(BaseNeRF):
    """
    Base class for Nerfacto

    Args:
        aabb (torch.Tensor): Axis-aligned bounding box
        device (torch.device): Device that the model runs on.
        near_far (list): Near and far plane along the sample ray
        scene_manager (SceneManager): SceneManager object
        config: Configuration for the model
        group: Process group for distributed training
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
        super().__init__(config, scene_manager, aabb=aabb, device=device, near_far=near_far)

        self.group = group

        print(f"aabb.device = {aabb.device}, device = {device}", flush=True)

        # new features
        self.plane_division = config.plane_division
        self.num_hash_table = config.plane_division[0] * config.plane_division[1]

        self.init_field_components()
        self.init_sampler()
        self.init_renderer()

    def init_sampler(self):
        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        self.proposal_sampler = ProposalNetworkSampler(
            near_far=self.near_far,
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            config=self.config,
            aabb=self.aabb,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=None,
        )

    def init_renderer(self):
        config = self.config
        self.renderer = VolumeRenderer(white_bg=config.white_bkgd)

    def init_field_components(self):
        config = self.config

        self.geo_feat_dim = config.geo_feat_dim

        self.appearance_embedding_dim = config.appearance_embed_dim
        self.base_res = config.base_res
        self.step = 0

        self.direction_encoding = SHEncoding(
            levels=4,
        )

        self.mlp_base_grid = HashEncoding(
            num_levels=config.num_levels,
            min_res=config.base_res,
            max_res=config.max_res,
            log2_hashmap_size=config.log2_hashmap_size,
            features_per_level=config.features_per_level,
            device=self.device,
            num_hash_table=self.num_hash_table,
        )
        # NOTE: MLPDecoder for nerfacto and instant-ngp only supports tcnn implementation
        self.mlp_base_mlp = MLPDecoder(
            inChanel=self.mlp_base_grid.get_out_dim(),
            num_layers=config.num_layers,
            layer_width=config.hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            out_activation=None,
            mlp_bias=False,
        )
        # self.mlp_base = torch.nn.Sequential(self.mlp_base_grid, self.mlp_base_mlp)

        self.mlp_head = MLPDecoder(
            inChanel=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=config.num_layers_color,
            layer_width=config.hidden_dim_color,
            out_dim=3,
            out_activation=nn.Sigmoid(),
            mlp_bias=False,
        )

        self.embedding_appearance = torch.nn.Embedding(config.appearance_embedding_size, self.appearance_embedding_dim)

    def density_fn(self, positions):
        """Returns only the density. Used primarily with the density grid."""
        origins = positions
        directions = torch.ones_like(positions)
        starts = torch.zeros_like(positions[..., :1])
        ends = torch.zeros_like(positions[..., :1])

        density, _ = self.get_density(origins, directions, (starts + ends) / 2)
        return density

    def get_density(self, origins, directions, z_vals) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        pos = origins + directions * z_vals[..., None]
        positions = get_normalized_positions(pos, self.aabb)  # TODO use scene_manger.normalize_coord

        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True

        rank = self.config.mp_rank if self.config.branch_parallel else None
        samples = Samples(xyz=positions, dirs=directions, z_vals=z_vals, rank=rank, group=self.group)  # z_vals * 2
        self.scene_manager.assign_blockIdx(samples)

        if self.config.branch_parallel:
            block_mask = samples.mask
            # TODO dtype is same to mlp_base_grid's output
            global_base_mlp_out = torch.zeros(
                size=(*positions.shape[:-1], self.mlp_base_mlp.module.out_dim - 1), device=self.device
            )
            if block_mask.sum():
                block_positions_contract = contract(samples.xyz[block_mask])
                block_positions_contract = (block_positions_contract + 2.0) / 4.0
                block_positions_flat = block_positions_contract
                block_idx_flat = samples.block_idx[block_mask].view(-1, 1)
                hash_features = self.mlp_base_grid(torch.concat((block_positions_flat, block_idx_flat), dim=1))
                h = self.mlp_base_mlp(hash_features).view(samples.xyz[block_mask].shape[0], -1)
                density_before_activation, block_base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
                density = trunc_exp(density_before_activation.to(positions))
                samples.sigma[block_mask] = density.view(-1)
                global_base_mlp_out = global_base_mlp_out.to(block_base_mlp_out)
                global_base_mlp_out[block_mask] = block_base_mlp_out

                samples.sync_sigma()

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
                        mask.unsqueeze(-1).expand(*mask.shape, self.mlp_base_mlp.module.out_dim - 1),
                        validbase_mlp_out,
                    )
            else:
                block_positions_contract = contract(samples.xyz[:1].view(-1, 3))
                block_positions_contract = (block_positions_contract + 2.0) / 4.0
                block_positions_flat = block_positions_contract
                block_idx_flat = samples.block_idx[:1].view(-1, 1)
                hash_features = self.mlp_base_grid(torch.concat((block_positions_flat, block_idx_flat), dim=1))
                h = self.mlp_base_mlp(hash_features).view(*samples.xyz[:1].shape[:-1], -1)
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
                # samples.sigma[:1] = density.view(-1)

                global_base_mlp_out[:1] = block_base_mlp_out

            base_mlp_out = global_base_mlp_out
            density = samples.sigma.unsqueeze(-1)
        else:
            positions_contract = contract(positions)
            positions_contract = (positions_contract + 2.0) / 4.0
            positions_flat = positions_contract.view(-1, 3)
            block_idx_flat = samples.block_idx.view(-1, 1)
            hash_features = self.mlp_base_grid(torch.concat((positions_flat, block_idx_flat), dim=1))
            h = self.mlp_base_mlp(hash_features).view(positions.shape[0], positions.shape[1], -1)
            density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
            self._density_before_activation = density_before_activation

            # Rectifying the density with an exponential is much more stable than a ReLU or
            # softplus, because it enables high post-activation (float32) density outputs
            # from smaller internal (float16) parameters.
            density = trunc_exp(density_before_activation.to(positions))

        density = density * selector[..., None]
        return density, base_mlp_out

    def set_anneal(self, step):
        # https://arxiv.org/pdf/2111.12077.pdf eq. 18
        self.step = step
        train_frac = np.clip(step / self.config.proposal_weights_anneal_max_num_iters, 0, 1)
        self.step = step

        def bias(x, b):
            return b * x / ((b - 1) * x + 1)

        anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
        if isinstance(self.proposal_sampler, DDP):
            self.proposal_sampler.module.set_anneal(anneal)
        else:
            self.proposal_sampler.set_anneal(anneal)

    def forward(
        self,
        rays_chunk,
    ):
        (
            samples,
            spacing_starts,
            spacing_ends,
            weights_list,
            spacing_starts_list,
            spacing_ends_list,
        ) = self.proposal_sampler(
            ray_bundle=rays_chunk,
            training=self.training,
            render_stratified_sampling=self.config.render_stratified_sampling,
        )

        density, density_embedding = self.get_density(samples.xyz, samples.dirs, samples.z_vals)

        camera_indices = samples.camera_idx.squeeze()
        directions = get_normalized_directions(samples.dirs)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)
        outputs_shape = directions.shape[:-1]

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices.to(torch.int))
        else:
            weight_mean = (
                self.embedding_appearance.weight.mean(dim=0)
                if not isinstance(self.embedding_appearance, DDP)
                else self.embedding_appearance.module.weight.mean(dim=0)
            )
            embedded_appearance = (
                torch.ones((*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device)
                * weight_mean
            )

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)

        weights = self.renderer.get_weights(density[..., 0], samples.dists)

        weights_list.append(weights)
        spacing_starts_list.append(spacing_starts)
        spacing_ends_list.append(spacing_ends)

        rgb_map, depth_map = self.renderer(weights, rgb, samples.z_vals, 1.0)

        outputs = {
            "rgb_map": rgb_map,
            "depth_map": depth_map[..., None],
        }
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["spacing_starts_list"] = spacing_starts_list
            outputs["spacing_ends_list"] = spacing_ends_list

        return outputs

    def get_optparam_groups(self, lr=0.01):
        """
        Returns the parameter groups for optimization.

        Args:
            lr: learning rate

        Returns:
            list: A list of parameter groups for optimization.
        """
        param_groups = [
            {"params": self.mlp_base_grid.parameters(), "lr": lr, "name": "fields"},
            {"params": self.mlp_base_mlp.parameters(), "lr": lr, "name": "fields"},
            {"params": self.mlp_head.parameters(), "lr": lr, "name": "fields"},
            {"params": self.embedding_appearance.parameters(), "lr": lr, "name": "fields"},
            {"params": self.proposal_sampler.parameters(), "lr": lr, "name": "proposal_networks"},
        ]
        return param_groups

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
            else:
                for key in self.state_dict().keys():
                    if name in key.split(".")[0]:
                        # check consistency
                        for i in range(len(state_dicts) - 1):
                            assert torch.equal(state_dicts[i][key], state_dicts[i + 1][key])
                        merged_state_dict.update({key: state_dicts[0][key]})

        return merged_state_dict, kwargs


def contract(x):
    mag = torch.linalg.norm(x, ord=float("inf"), dim=-1)[..., None]  # pylint: disable=E1102
    return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))


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
