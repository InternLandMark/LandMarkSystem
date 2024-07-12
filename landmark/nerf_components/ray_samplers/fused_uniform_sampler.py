from typing import Union

import torch

from landmark.nerf_components.data import Rays, Samples

from .uniform_sampler import UniformSampler


class FusedUniformSampler(UniformSampler):
    """ Sample N_samples number points on a ray. If is_train is True, use segmented \
        random sampling. Otherwise use uniform sampling.

    Args:
        num_samples (int): number of samples on each ray.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        rays: Union[Rays, torch.Tensor],
        num_samples: int = -1,
        random_sampling: bool = False,
        sample_within_hull: bool = False,
    ) -> (Samples, torch.Tensor):
        raise NotImplementedError("TODO")
