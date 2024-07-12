import torch

from .tensor_vm_encoding import ChannelParallelTensorVMEncoding, TensorVMEncoding


class FusedTensorVMEncoding(TensorVMEncoding):
    """Class for TensorVMEncoding"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:  # pylint:disable=W0237
        """
        Computes the feature for given samples coordinates.
        It is the same as the compute_densityfeature or compute_appfeature in the original tensorf code

        Args:
            xyz: (N, 3, 1) tensor, the coordinates of the points

        Returns:
            (N, n_component) tensor, the feature of the points

        """
        raise NotImplementedError("TODO")


class FusedChannelParallelTensorVMEncoding(ChannelParallelTensorVMEncoding):
    """Channel Parallel version of TensorVMEncoding"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:  # pylint:disable=W0237
        """
        Computes the feature for given samples coordinates.
        It is the same as the compute_densityfeature or compute_appfeature in the original tensorf code

        Args:
            xyz: should be xyzb (N, 4) tensor, which b is 'block', the coordinates of the points

        Returns:
            (N, n_component) tensor, the feature of the points

        """
        raise NotImplementedError("TODO")
