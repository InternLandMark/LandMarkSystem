import torch

from .base_renderer import BaseRenderer


class VolumeRenderer(BaseRenderer):
    """
    Class for volume rendering.

    Args:
        white_bg (bool, optional): Flag indicating whether to use a white background. Defaults to False.
    """

    def __init__(self, white_bg: bool = False) -> None:
        super().__init__()
        self.white_bg = white_bg

        self.save_init_kwargs(locals())  # save for converting to parallel encoding

    def get_weights(self, sigma: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute the weights for each sample.

        Args:
            sigma (torch.Tensor): Sigma values. Shape: [N_rays, N_samples]
            distances (torch.Tensor): Distances. Shape: [N_rays, N_samples]

        Returns:
            torch.Tensor: Weights. Shape: [N_rays, N_samples]
        """
        alpha = 1.0 - torch.exp(-sigma * distances)

        tensor = torch.cumprod(
            torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1.0 - alpha + 1e-10], -1),
            -1,
        )

        weights = alpha * tensor[:, :-1]  # [N_rays, N_samples]
        return weights

    def forward(
        self, weights: torch.Tensor, rgbs: torch.Tensor, z_vals: torch.Tensor, random_white_bg_ratio: float = 0.0
    ):
        """
        Compute the weighted sum of the RGB values.

        Args:
            weights (torch.Tensor): Weights. Shape: [N_rays, N_samples]
            rgbs (torch.Tensor): RGB values. Shape: [N_rays, N_samples, 3]
            z_vals (torch.Tensor): Depth values. Shape: [N_rays, N_samples]
            random_white_bg_ratio (float, optional): Random white background ratio. Defaults to 0.0.

        Returns:
            torch.Tensor: RGB map. Shape: [N_rays, 3]
            torch.Tensor: Depth map. Shape: [N_rays]
        """
        acc_map = torch.sum(weights, -1)
        rgb_map = torch.sum(weights[..., None] * rgbs, -2)
        if self.white_bg or (random_white_bg_ratio > 0.0 and torch.rand((1,)) < random_white_bg_ratio):
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weights * z_vals, -1)

        return rgb_map, depth_map
