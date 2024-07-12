import torch
import torch.nn
import torch.nn.functional as F


class AlphaGridMask(torch.nn.Module):
    """
    A class for the alpha grid mask.

    Args:
        device (str): The device to use.
        aabb (torch.Tensor): The axis-aligned bounding box.
        alpha_volume (torch.Tensor): The alpha volume.
    """

    def __init__(self, device, aabb, alpha_volume):
        """
        Initializes an AlphaMask object.

        Args:
            device (torch.device): The device to be used for computations.
            aabb (torch.Tensor): The axis-aligned bounding box (AABB) of the volume.
            alpha_volume (torch.Tensor): The alpha volume.

        """
        super().__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]).to(
            self.device
        )

    def update_device(self, device):
        """
        Updates the device used by the AlphaGridMask instance.

        Args:
            device (str): The new device to use.
        """
        self.device = device
        self.aabb = self.aabb.to(device)
        self.invgridSize = self.invgridSize.to(device)
        self.gridSize = self.gridSize.to(device)
        self.alpha_volume = self.alpha_volume.to(device)

    def sample_alpha(self, xyz_sampled):
        """
        Samples the alpha values.

        Args:
            xyz_sampled (torch.Tensor): The sampled coordinates.

        Returns:
            torch.Tensor: The alpha values.
        """
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)
        return alpha_vals

    def forward(self, xyz_sampled, filter_thresh: torch.Tensor, mask: torch.Tensor = None):
        """
        Computes the forward pass of the AlphaGridMask.

        Args:
            xyz_sampled (torch.Tensor): The sampled coordinates.
            filter_thresh (torch.Tensor): The filter threshold.
            mask (torch.Tensor, optional): The mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The computed mask.
        """
        if mask is not None:
            xyz_sampled = xyz_sampled[mask]

        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)
        alpha_mask = alpha_vals > filter_thresh

        if mask is not None:
            mask_invalid = ~mask
            mask_invalid[mask] |= ~alpha_mask
            mask_valid = ~mask_invalid
        else:
            mask_valid = alpha_mask

        return mask_valid

    def generate_alpha_mask(self, xyz_sampled, filter_thresh=0, mask=None):
        """
        Generates the alpha mask.

        Args:
            xyz_sampled (torch.Tensor): The sampled coordinates.
            filter_thresh (int, optional): The filter threshold. Defaults to 0.
            mask (torch.Tensor, optional): The mask tensor. Defaults to None.

        Returns:
            torch.Tensor or tuple: The generated alpha mask or a tuple of alpha mask and valid mask.
        """
        if mask is not None:
            xyz_sampled = xyz_sampled[mask]
            xyz_sampled = self.normalize_coord(xyz_sampled)
            alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)
            alpha_mask = alpha_vals > filter_thresh
            mask_invalid = ~mask
            mask_invalid[mask] |= ~alpha_mask
            mask_valid = ~mask_invalid
            return alpha_mask, mask_valid
        else:
            xyz_sampled = self.normalize_coord(xyz_sampled)
            alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)
            alpha_mask = alpha_vals > filter_thresh
            return alpha_mask

    def normalize_coord(self, xyz_sampled):
        """
        Normalizes the sampled coordinates.

        Args:
            xyz_sampled (torch.Tensor): The sampled coordinates.

        Returns:
            torch.Tensor: The normalized coordinates.
        """
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1
