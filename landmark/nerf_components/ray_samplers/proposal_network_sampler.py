from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

# from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from landmark.nerf_components.data import Rays, Samples
from landmark.nerf_components.model.base_module import BaseModule
from landmark.nerf_components.model_components import HashEncoding, MLPDecoder
from landmark.nerf_components.utils.activation_utils import trunc_exp

from .base_sampler import BaseSampler


def contract(x):
    mag = torch.linalg.norm(x, ord=float("inf"), dim=-1)[..., None]  # pylint: disable=E1102
    return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))


class ProposalNetworks(BaseModule):
    """A lightweight density field module.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        spatial_distortion: spatial distortion module
        use_linear: whether to skip the MLP and use a single linear layer instead
    """

    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        use_linear: bool = False,
        num_levels: int = 8,
        max_res: int = 1024,
        base_res: int = 16,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.use_linear = use_linear

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.encoding = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
        )

        if not self.use_linear:
            network = MLPDecoder(
                inChanel=self.encoding.get_out_dim(),
                num_layers=num_layers,
                layer_width=hidden_dim,
                out_dim=1,
                out_activation=None,
                mlp_bias=False,
            )
            self.mlp_base = torch.nn.Sequential(self.encoding, network)
        else:
            self.linear = torch.nn.Linear(self.encoding.get_out_dim(), 1)

    def get_density(self, origins, directions, starts, ends):
        """Computes and returns the densities."""
        pos = origins + directions * (starts + ends) / 2
        pos = contract(pos)
        positions = (pos + 2.0) / 4.0
        # positions = get_normalized_positions(pos, self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        positions_flat = positions.view(-1, 3)
        if not self.use_linear:
            density_before_activation = self.mlp_base(positions_flat).view(positions.shape[0], -1).to(positions)
        else:
            x = self.encoding(positions_flat).to(positions)
            density_before_activation = self.linear(x).view(positions.shape[0], -1)
        # density_before_activation = (self.hashmlp(positions_flat).view(positions.shape[0], -1).to(positions))

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        density = density[..., None] * selector[..., None]
        return density

    def density_fn(self, positions):
        """Returns only the density. Used primarily with the density grid."""
        origins = positions
        directions = torch.ones_like(positions)
        starts = torch.zeros_like(positions[..., :1])
        ends = torch.zeros_like(positions[..., :1])

        density = self.get_density(origins, directions, starts, ends)
        return density


class ProposalNetworkSampler(BaseSampler):
    """Sampler that uses a proposal network to generate samples.

    Args:
        num_proposal_samples_per_ray: Number of samples to generate per ray for each proposal step.
        num_nerf_samples_per_ray: Number of samples to generate per ray for the NERF model.
        num_proposal_network_iterations: Number of proposal network iterations to run.
        single_jitter: Use a same random jitter for all samples along a ray.
        update_sched: A function that takes the iteration number of steps between updates.
        initial_sampler: Sampler to use for the first iteration. Uses UniformLinDispPiecewise if not set.
        pdf_sampler: PDFSampler to use after the first iteration. Uses PDFSampler if not set.
    """

    def __init__(
        self,
        near_far,
        num_proposal_samples_per_ray: Tuple[int, ...] = (64,),
        num_nerf_samples_per_ray: int = 32,
        num_proposal_network_iterations: int = 2,
        config=None,
        aabb=None,
        single_jitter: bool = False,
        update_sched: Callable = lambda x: 1,
        initial_sampler=None,
        pdf_sampler=None,
    ) -> None:
        super().__init__()
        self.near_far = near_far
        self.num_proposal_samples_per_ray = num_proposal_samples_per_ray
        self.num_nerf_samples_per_ray = num_nerf_samples_per_ray
        self.num_proposal_network_iterations = num_proposal_network_iterations
        self.update_sched = update_sched
        if self.num_proposal_network_iterations < 1:
            raise ValueError("num_proposal_network_iterations must be >= 1")

        # samplers
        if initial_sampler is None:
            self.initial_sampler = SpacedSampler(
                num_samples=None,
                spacing_fn=lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x)),
                spacing_fn_inv=lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x)),
                train_stratified=True,
                single_jitter=single_jitter,
            )
        else:
            self.initial_sampler = initial_sampler
        if pdf_sampler is None:
            self.pdf_sampler = PropPDFSampler(include_original=False, single_jitter=single_jitter)
        else:
            self.pdf_sampler = pdf_sampler

        self._anneal = 1.0
        self._steps_since_update = 0
        self._step = 0

        self.density_fns = []
        num_prop_nets = num_proposal_network_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        for i in range(num_prop_nets):
            prop_net_args = config.proposal_net_args_list[min(i, len(config.proposal_net_args_list) - 1)]
            network = ProposalNetworks(
                aabb,
                **prop_net_args,
            )
            self.proposal_networks.append(network)
        self.density_fns.extend([network.density_fn for network in self.proposal_networks])

    def set_anneal(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._anneal = anneal

    def step_cb(self, step):
        """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
        self._step = step
        self._steps_since_update += 1

    def forward(
        self,
        ray_bundle,
        render_stratified_sampling: bool,
        training: bool = False,
    ):
        weights_list = []
        spacing_starts_list = []
        spacing_ends_list = []

        n = self.num_proposal_network_iterations
        weights = None
        ray_samples = None
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10

        stratified = training or render_stratified_sampling

        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples, spacing_starts, spacing_ends, spacing_to_euclidean_fn = self.initial_sampler(
                    ray_bundle=ray_bundle,
                    stratified=stratified,
                    num_samples=num_samples,
                    near_plane=self.near_far[0],
                    far_plane=self.near_far[1],
                )
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples, spacing_starts, spacing_ends, spacing_to_euclidean_fn = self.pdf_sampler(
                    ray_bundle=ray_bundle,
                    ray_samples=ray_samples,
                    weights=annealed_weights,
                    stratified=stratified,
                    num_samples=num_samples,
                    spacing_starts=spacing_starts,
                    spacing_ends=spacing_ends,
                    spacing_to_euclidean_fn=spacing_to_euclidean_fn,
                )
            if is_prop:
                z_vals = ray_samples.z_vals[..., None]
                if updated:
                    # always update on the first step or the inf check in grad scaling crashes
                    pos = ray_samples.xyz + ray_samples.dirs * z_vals
                    density = self.density_fns[i_level](pos)
                else:
                    with torch.no_grad():
                        pos = ray_samples.xyz + ray_samples.dirs * z_vals
                        density = self.density_fns[i_level](pos)

                delta_density = ray_samples.dists[..., None] * density
                alphas = 1 - torch.exp(-delta_density)

                transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
                transmittance = torch.cat(
                    [torch.zeros((*transmittance.shape[:1], 1, 1), device=density.device), transmittance], dim=-2
                )
                transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

                weights = alphas * transmittance  # [..., "num_samples"]
                weights = torch.nan_to_num(weights)

                weights_list.append(weights)  # (num_rays, num_samples)
                spacing_starts_list.append(spacing_starts)
                spacing_ends_list.append(spacing_ends)

        if updated:
            self._steps_since_update = 0

        return ray_samples, spacing_starts, spacing_ends, weights_list, spacing_starts_list, spacing_ends_list


class SpacedSampler(BaseSampler):
    """Sample points according to a function.

    Args:
        num_samples: Number of samples per ray
        spacing_fn: Function that dictates sample spacing (ie `lambda x : x` is uniform).
        spacing_fn_inv: The inverse of spacing_fn.
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        spacing_fn: Callable,
        spacing_fn_inv: Callable,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.train_stratified = train_stratified
        self.single_jitter = single_jitter
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    def forward(
        self,
        stratified,
        ray_bundle=None,
        num_samples: Optional[int] = None,
        near_plane: float = 0.0,
        far_plane: Optional[float] = None,
    ):
        """Generates position samples according to spacing function.

        Args:
            ray_bundle: Rays to generate samples for
            num_samples: Number of samples per ray

        Returns:
            Positions for samples along a ray
        """
        rays_data = ray_bundle.data if isinstance(ray_bundle, Rays) else ray_bundle
        rays_o = rays_data[..., :3]
        rays_d = rays_data[..., 3:6]
        rays_idx = rays_data[..., 6]

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_rays = rays_o.shape[0]

        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(rays_o.device)[None, ...]  # [1, num_samples+1]

        # TODO More complicated than it needs to be.
        if self.train_stratified and stratified:
            if self.single_jitter:
                t_rand = torch.rand((num_rays, 1), dtype=bins.dtype, device=bins.device)
            else:
                t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand
        elif not stratified:
            bins = bins.repeat(num_rays, 1)

        nears = torch.full((num_rays, 1), near_plane).to(bins.device)
        fars = torch.full((num_rays, 1), far_plane).to(bins.device)
        s_near, s_far = (self.spacing_fn(x) for x in (nears, fars))

        def spacing_to_euclidean_fn(x):
            return self.spacing_fn_inv(x * s_far + (1 - x) * s_near)

        euclidean_bins = spacing_to_euclidean_fn(bins)  # [num_rays, num_samples+1]

        bin_starts = euclidean_bins[..., :-1, None]
        bin_ends = euclidean_bins[..., 1:, None]
        spacing_starts = bins[..., :-1, None]
        spacing_ends = bins[..., 1:, None]

        if rays_idx is not None:
            camera_indices = rays_idx.unsqueeze(1).repeat(1, num_samples)
        else:
            camera_indices = None

        origins = rays_o.unsqueeze(1).repeat(1, num_samples, 1)  # [..., 512, 3]
        dirs = rays_d.unsqueeze(1).repeat(1, num_samples, 1)  # [..., 512, 3]
        starts = bin_starts  # [..., num_samples, 1]
        ends = bin_ends  # [..., num_samples, 1]

        samples = Samples(
            xyz=origins,
            dirs=dirs,
            z_vals=(starts + ends) / 2,
            camera_idx=camera_indices,
        )
        return samples, spacing_starts, spacing_ends, spacing_to_euclidean_fn


class PropPDFSampler(BaseSampler):
    """Sample based on probability distribution

    Args:
        num_samples: Number of samples per ray
        train_stratified: Randomize location within each bin during training.
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
        include_original: Add original samples to ray.
        histogram_padding: Amount to weights prior to computing PDF.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified: bool = True,
        single_jitter: bool = False,
        include_original: bool = True,
        histogram_padding: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.train_stratified = train_stratified
        self.include_original = include_original
        self.histogram_padding = histogram_padding
        self.single_jitter = single_jitter

    def forward(
        self,
        stratified,
        ray_bundle=None,
        ray_samples=None,
        weights=None,
        num_samples: Optional[int] = None,
        spacing_starts=None,
        spacing_ends=None,
        spacing_to_euclidean_fn=None,
    ):
        """Generates position samples given a distribution.

        Args:
            ray_bundle: Rays to generate samples for
            ray_samples: Existing ray samples
            weights: Weights for each bin
            num_samples: Number of samples per ray

        Returns:
            Positions for samples along a ray
        """

        if ray_samples is None or ray_bundle is None:
            raise ValueError("ray_samples and ray_bundle must be provided")
        assert weights is not None, "weights must be provided"

        rays_data = ray_bundle.data if isinstance(ray_bundle, Rays) else ray_bundle
        rays_o = rays_data[..., :3]
        rays_d = rays_data[..., 3:6]
        rays_idx = rays_data[..., 6]

        num_samples = num_samples or self.num_samples
        assert num_samples is not None
        num_bins = num_samples + 1

        weights = weights[..., 0] + self.histogram_padding

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(1e-5 - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        if self.train_stratified and stratified:
            # Stratified samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
            if self.single_jitter:
                rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
            else:
                rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
            u = u + rand
        else:
            # Uniform samples between 0 and 1
            u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
            u = u + 1.0 / (2 * num_bins)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
        u = u.contiguous()

        assert (
            spacing_starts is not None and spacing_ends is not None
        ), "ray_sample spacing_starts and spacing_ends must be provided"
        assert spacing_to_euclidean_fn is not None, "spacing_to_euclidean_fn must be provided"

        existing_bins = torch.cat(
            [
                spacing_starts[..., 0],
                spacing_ends[..., -1:, 0],
            ],
            dim=-1,
        )

        inds = torch.searchsorted(cdf, u, side="right")
        below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
        above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
        cdf_g0 = torch.gather(cdf, -1, below)
        bins_g0 = torch.gather(existing_bins, -1, below)
        cdf_g1 = torch.gather(cdf, -1, above)
        bins_g1 = torch.gather(existing_bins, -1, above)

        t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)

        if self.include_original:
            bins, _ = torch.sort(torch.cat([existing_bins, bins], -1), -1)

        # Stop gradients
        bins = bins.detach()

        euclidean_bins = spacing_to_euclidean_fn(bins)

        bin_starts = euclidean_bins[..., :-1, None]
        bin_ends = euclidean_bins[..., 1:, None]
        spacing_starts = bins[..., :-1, None]
        spacing_ends = bins[..., 1:, None]

        if rays_idx is not None:
            camera_indices = rays_idx.unsqueeze(1).repeat(1, num_samples)
        else:
            camera_indices = None

        origins = rays_o.unsqueeze(1).repeat(1, num_samples, 1)  # [..., 512, 3]
        dirs = rays_d.unsqueeze(1).repeat(1, num_samples, 1)  # [..., 512, 3]
        samples = Samples(
            xyz=origins,
            dirs=dirs,
            camera_idx=camera_indices,
            z_vals=(bin_starts + bin_ends) / 2,
        )
        return samples, spacing_starts, spacing_ends, spacing_to_euclidean_fn
