from typing import Optional, Union

import torch
from torch import searchsorted

from landmark.nerf_components.data import Rays, Samples

from .base_sampler import BaseSampler


class PDFSampler(BaseSampler):
    """
    A sampler that generates samples based on a Probability Density Function (PDF).

    This class extends the BaseSampler to implement sampling methods that are specifically designed to generate
    samples according to a given Probability Density Function. It is useful in scenarios where the distribution
    of the samples needs to follow a specific statistical distribution.

    Args:
        num_samples (int): The number of samples to generate.
        *args: Variable length argument list for the base class.
        **kwargs: Arbitrary keyword arguments for the base class.
    """

    def __init__(self, num_samples: int, *args, **kwargs):
        """
        Initializes the PDFSampler with the specified number of samples.

        Args:
            num_samples (int): The number of samples to generate.
            *args: Variable length argument list for the base class.
            **kwargs: Arbitrary keyword arguments for the base class.
        """
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples

    def forward(
        self,
        rays: Union[Rays, torch.Tensor],
        z_vals,
        weights,
        num_samples: Optional[int] = None,
        det=False,
    ) -> (Samples, torch.Tensor):
        """
        Sample points along the rays based on a probability density function.

        Args:
            rays (Union[Rays, torch.Tensor]): Rays to sample from.
            z_vals (torch.Tensor): Values along the rays.
            weights (torch.Tensor): Weights for the PDF.
            num_samples (Optional[int], optional): Number of samples on each ray. If None, use self.num_samples.
            det (bool, optional): Deterministic sampling. Defaults to False.

        Returns:
            (Samples, torch.Tensor): A tuple containing the sampled points
            and a mask of the points outside the bounding box.
        """

        rays_data = rays.data if isinstance(rays, Rays) else rays
        rays_origin = rays_data[..., :3]
        rays_dirs = rays_data[..., 3:6]
        rays_camera_indice = rays_data[..., 6]
        z_vals_mid = 0.5 * (z_vals[..., 1:] - z_vals[..., :-1])
        device = weights.device

        num_samples = self.num_samples if num_samples is None else num_samples
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0.0, 1.0, steps=num_samples, device=device)
            u = u.expand(list(cdf.shape[:-1]) + [num_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=device)

        # Invert CDF
        u = u.contiguous()
        inds = searchsorted(cdf.detach(), u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(z_vals_mid.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        z_samples = samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        xyz_sampled = (
            rays_origin[..., None, :] + rays_dirs[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples, 3]
        if rays_camera_indice is not None:
            xyz_sampled_idxs = rays_camera_indice.unsqueeze(1).repeat(1, xyz_sampled.shape[1]).type(torch.long)
        else:
            xyz_sampled_idxs = None
        samples_dirs = rays_dirs.clone().view(-1, 1, 3).expand(xyz_sampled.shape)

        rank = rays.rank if isinstance(rays, Rays) else None
        group = rays.group if isinstance(rays, Rays) else None

        samples = Samples(
            xyz=xyz_sampled, dirs=samples_dirs, z_vals=z_vals, camera_idx=xyz_sampled_idxs, rank=rank, group=group
        )

        return samples
