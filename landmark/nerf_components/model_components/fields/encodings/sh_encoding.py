from torch import nn

from landmark.nerf_components.utils.tcnn_utils import tcnn


class SHEncoding(nn.Module):
    """Spherical harmonic encoding

    Args:
        levels (int): Number of spherical harmonic levels to encode.

    Attributes:
        in_dim (int): Input dimension of the encoding.
        levels (int): Number of spherical harmonic levels to encode.
        tcnn_encoding (tcnn.Encoding): TCNN encoding module.

    Methods:
        get_out_dim(): Returns the output dimension of the encoding.
        forward(xyz): Performs the forward pass of the encoding.

    """

    def __init__(self, levels: int = 4) -> None:
        super().__init__()
        self.in_dim = 3
        if levels <= 0 or levels > 4:
            raise ValueError(f"Spherical harmonic encoding only supports 1 to 4 levels, requested {levels}")

        self.levels = levels

        encoding_config = {
            "otype": "SphericalHarmonics",
            "degree": levels,
        }
        self.tcnn_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=encoding_config,
        )

    def get_out_dim(self) -> int:
        """Returns the output dimension of the encoding.

        Returns:
            int: The output dimension of the encoding.

        """
        return self.levels**2

    def forward(self, xyz):
        """Performs the forward pass of the encoding.

        Args:
            xyz (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The encoded tensor.

        """
        return self.tcnn_encoding(xyz)
