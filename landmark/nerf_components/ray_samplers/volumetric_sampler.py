from typing import Callable, Optional, Union

import torch

from landmark.nerf_components.data import Rays, Samples

from .base_sampler import BaseSampler


class VolumetricSampler(BaseSampler):
    """Sampler inspired by the one proposed in the Instant-NGP paper.
    Generates samples along a ray by sampling the occupancy field.
    Optionally removes occluded samples if the density_fn is provided.

    Args:
    occupancy_grid: Occupancy grid to sample from.
    density_fn: Function that evaluates density at a given point.
    scene_aabb: Axis-aligned bounding box of the scene, should be set to None if the scene is unbounded.
    """

    def __init__(self, occupancy_grid, density_fn, near_far, step_size, alpha_thre=0.01, cone_angle=0.0):
        super().__init__()
        assert occupancy_grid is not None
        self.density_fn = density_fn
        self.occupancy_grid = occupancy_grid
        self.near_far = near_far
        self.step_size = step_size  # render_step_size
        self.alpha_thre = alpha_thre
        self.cone_angle = cone_angle

    def get_sigma_fn(self, origins, directions, training, times=None) -> Optional[Callable]:
        """Returns a function that returns the density of a point.

        Args:
            origins: Origins of rays
            directions: Directions of rays
            times: Times at which rays are sampled
        Returns:
            Function that returns the density of a point or None if a density function is not provided.
        """

        if self.density_fn is None or not training:
            return None

        density_fn = self.density_fn

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = directions[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            if times is None:
                return density_fn(positions).squeeze(-1)
            return density_fn(positions, times[ray_indices]).squeeze(-1)

        return sigma_fn

    def forward(
        self,
        ray_bundle: Union[Rays, torch.Tensor],
        training: bool,
        render_stratified_sampling: bool,
    ):
        """Generate ray samples in a bounding box.

        Args:
            ray_bundle: Rays to generate samples for
            render_step_size: Minimum step size to use for rendering
            near_plane: Near plane for raymarching
            far_plane: Far plane for raymarching
            alpha_thre: Opacity threshold skipping samples.
            cone_angle: Cone angle for raymarching, set to 0 for uniform marching.

        Returns:
            a tuple of (ray_samples, packed_info, ray_indices)
            The ray_samples are packed, only storing the valid samples.
            The ray_indices contains the indices of the rays that each sample belongs to.
        """
        rays_data = ray_bundle.data if isinstance(ray_bundle, Rays) else ray_bundle
        rays_o = rays_data[..., :3]
        rays_d = rays_data[..., 3:6]
        rays_idx = rays_data[..., 6]
        t_min = None
        t_max = None

        if rays_idx is not None:
            camera_indices = rays_idx.contiguous()
        else:
            camera_indices = None

        stratified = training or render_stratified_sampling

        ray_indices, starts, ends = self.occupancy_grid.sampling(
            rays_o=rays_o,
            rays_d=rays_d,
            t_min=t_min,
            t_max=t_max,
            sigma_fn=self.get_sigma_fn(rays_o, rays_d, training=training),
            render_step_size=self.step_size,
            near_plane=self.near_far[0],
            far_plane=self.near_far[1],
            stratified=stratified,
            cone_angle=self.cone_angle,
            alpha_thre=self.alpha_thre,
        )

        ray_sample_nums = torch.bincount(ray_indices, minlength=rays_o.shape[0])
        max_ray_sample_num = ray_sample_nums.max()
        indices = torch.arange(max_ray_sample_num, device=rays_o.device).expand(rays_o.shape[0], -1)
        valid_mask = indices < ray_sample_nums[:, None]

        samples = Samples(
            xyz=rays_o[..., None, :].expand(rays_o.shape[0], max_ray_sample_num, -1),
            dirs=rays_d[..., None, :].expand(rays_d.shape[0], max_ray_sample_num, -1),
            camera_idx=camera_indices[..., None].expand(camera_indices.shape[0], max_ray_sample_num),
        )
        samples._has_z_vals = True
        samples.z_vals.masked_scatter_(valid_mask, (starts + ends) / 2)

        return samples, valid_mask
