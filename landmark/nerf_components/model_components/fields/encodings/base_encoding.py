# pylint: disable=W0613
from abc import abstractmethod

import torch

from landmark.nerf_components.model.base_module import BaseModule
from landmark.nerf_components.utils.graphics_utils import BasicPointCloud


class VolumeEncoding(BaseModule):
    """Class for volume encoding"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._kwargs = kwargs

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Encode the scene into a latent feature representation.

        Args:
            xyz: (N, 3) tensor, the coordinates of the points

        Returns:
            torch.Tensor: (N, C) tensor, the latent feature representation of the scene
        """
        pass

    def feature2density(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Convert the latent feature representation to density.

        Args:
            feature: (N, C) tensor, the latent feature representation of the scene

        Returns:
            torch.Tensor: (N, 1) tensor, the density of the scene
        """
        pass


class BaseGaussianEncoding(BaseModule):
    """Class for gaussian encoding"""

    @abstractmethod
    def create_from_pcd(self, pcd: BasicPointCloud):
        """Create gaussian encoding from a point cloud.

        Args:
            pcd (BasicPointCloud): The input point cloud.

        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def save_ply(self, path):
        """Save the gaussian encoding as a PLY file.

        Args:
            path (str): The path to save the PLY file.

        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        """
        raise NotImplementedError
