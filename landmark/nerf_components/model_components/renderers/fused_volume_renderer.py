import torch

from .volume_renderer import VolumeRenderer


class FusedVolumeRenderer(VolumeRenderer):
    """Class for volume rendering."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self, weights: torch.Tensor, rgbs: torch.Tensor, z_vals: torch.Tensor, random_white_bg_ratio: float = 0.0
    ) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass of the fused volume renderer.

        Args:
            weights (torch.Tensor): Tensor containing the weights of each volume sample.
            rgbs (torch.Tensor): Tensor containing the RGB values of each volume sample.
            z_vals (torch.Tensor): Tensor containing the depth values of each volume sample.
            random_white_bg_ratio (float, optional): Ratio of random white background color. Defaults to 0.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the rendered RGB image and the rendered depth map.
        """
        raise NotImplementedError("TODO")
