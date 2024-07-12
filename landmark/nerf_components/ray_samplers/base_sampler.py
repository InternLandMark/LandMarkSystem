from abc import abstractmethod

from landmark.nerf_components.data import Samples
from landmark.nerf_components.model.base_module import BaseModule


class BaseSampler(BaseModule):
    """
    Sample points along the Rays.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Samples:
        raise NotImplementedError
