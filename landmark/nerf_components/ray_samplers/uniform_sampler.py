from typing import Union

import torch

from landmark.nerf_components.data import Rays, Samples

from .base_sampler import BaseSampler


class UniformSampler(BaseSampler):
    """ Sample N_samples number points on a ray. If is_train is True, use segmented \
        random sampling. Otherwise use uniform sampling.

    Args:
        num_samples (int): number of samples on each ray.

    """

    def __init__(self, num_samples: int, step_size: int, aabb: torch.Tensor, near_far: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.step_size = step_size
        self.aabb = aabb
        self.near_far = near_far
        self.save_init_kwargs(locals())  # save for converting to fusion kernel

    def forward(
        self,
        rays: Union[Rays, torch.Tensor],
        num_samples: int = -1,
        random_sampling: bool = False,
        sample_within_hull: bool = False,
    ) -> (Samples, torch.Tensor):
        """Sample points along the rays.
        Args:
            rays (Rays): rays to sample.
            num_samples (int): number of samples on each ray. If None, use self.num_samples.
        Returns:
            (Samples): sampled points.
            (torch.Tensor): mask of the points outside the bounding box.
        """
        rays_data = rays.data if isinstance(rays, Rays) else rays
        ray_origin = rays_data[..., :3]
        ray_dirs = rays_data[..., 3:6]
        camera_indice = rays_data[..., 6]
        num_samples = num_samples if num_samples > 0 else self.num_samples
        # near, far = rays.nears.clone(), rays.fars.clone()
        near, far = self.near_far
        if not sample_within_hull:
            vec = torch.where(ray_dirs == 0, torch.full_like(ray_dirs, 1e-6), ray_dirs)
            self.aabb = self.aabb.to(rays_data.device)
            rate_a = (self.aabb[1] - ray_origin) / vec
            rate_b = (self.aabb[0] - ray_origin) / vec
            t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

            rng = torch.arange(num_samples)[None].float()

            if random_sampling:
                rng = rng.repeat(ray_dirs.shape[-2], 1)
                rng += torch.rand_like(rng[:, [0]])
            step = self.step_size * rng.to(ray_dirs.device)
            z_vals = t_min[..., None] + step
        else:
            o_z = ray_origin[:, -1:] - self.aabb[0, 2].item()
            d_z = ray_dirs[:, -1:]
            far = -(o_z / d_z)
            # far[ray_dirs[:, 2] >= 0] = self.near_far[-1]
            far.scatter_(0, (ray_dirs[:, 2] >= 0).nonzero(), torch.full(far.shape, float(self.near_far[-1])).cuda())
            t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=ray_origin.device)
            z_vals = near * (1.0 - t_vals) + far * (t_vals)
            if random_sampling:
                mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                t_rand = torch.rand(z_vals.shape, device=ray_origin.device)
                z_vals = lower + (upper - lower) * t_rand

        rays_pts = ray_origin[..., None, :] + ray_dirs[..., None, :] * z_vals[..., None]
        if camera_indice is not None:
            rays_pts_idxs = camera_indice.unsqueeze(-1).repeat(1, rays_pts.shape[1]).type(torch.long)
        else:
            rays_pts_idxs = None

        samples_dirs = ray_dirs.clone().view(-1, 1, 3).expand(rays_pts.shape)

        rank = rays.rank if isinstance(rays, Rays) else None
        group = rays.group if isinstance(rays, Rays) else None
        samples = Samples(
            xyz=rays_pts, dirs=samples_dirs, z_vals=z_vals, camera_idx=rays_pts_idxs, rank=rank, group=group
        )
        samples = samples.to(ray_origin.device)

        aabb = self.aabb.clone()
        mask1 = aabb[0] > rays_pts
        mask2 = rays_pts > aabb[1]
        mask_outbbox = torch.any(torch.add(mask1, mask2), dim=-1)
        # mask_outbbox = ((aabb[0] > rays_pts) | (rays_pts > aabb[1])).any(dim=-1)
        return samples, ~mask_outbbox
