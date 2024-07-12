from typing import Optional, Union

import torch

from landmark.nerf_components.model.base_module import BaseModule


def positional_encoding(positions, freqs):
    """
    Apply positional encoding to the given positions.

    Args:
        positions (torch.Tensor): The positions to be encoded. Shape: (..., N)
        freqs (int): The number of frequency bands to use for encoding.

    Returns:
        torch.Tensor: The encoded positions. Shape: (..., D*F), where D is the dimension of positions and F
        is the number of frequency bands.
    """
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def activation_to_tcnn_string(activation: Union[torch.nn.Module, None]) -> str:
    """Converts a torch.nn activation function to a string that can be used to
    initialize a TCNN activation function.

    Args:
        activation: torch.nn activation function
    Returns:
        str: TCNN activation function string
    """

    if isinstance(activation, torch.nn.ReLU):
        return "ReLU"
    if isinstance(activation, torch.nn.LeakyReLU):
        return "Leaky ReLU"
    if isinstance(activation, torch.nn.Sigmoid):
        return "Sigmoid"
    if isinstance(activation, torch.nn.Softplus):
        return "Softplus"
    if isinstance(activation, torch.nn.Tanh):
        return "Tanh"
    if isinstance(activation, type(None)):
        return "None"
    tcnn_documentation_url = "https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#activation-functions"
    raise ValueError(
        f"TCNN activation {activation} not supported for now.\nSee {tcnn_documentation_url} for TCNN documentation."
    )


class MLPDecoder(BaseModule):
    """
    A MLP for rendering feature

    Args:
        inChanel (int): The number of input channels.
        layer_width (int): The number of output channels.
        num_layers (int): The number of layers in the MLP.
        layer_width (int): The width of the MLP.
        out_dim (int): The output dimension of the MLP.
        out_activation (torch.nn.Module): The output activation function.
        mlp_bias (bool): The learnable bias of the module.
    """

    def __init__(
        self,
        inChanel,
        layer_width=128,
        num_layers: Optional[int] = None,
        out_dim: Optional[int] = None,
        out_activation: Optional[torch.nn.Module] = None,
        mlp_bias: bool = True,
    ):
        super().__init__()
        self.mlp_bias = mlp_bias
        layers = []
        if num_layers == 1:
            layers.append(torch.nn.Linear(inChanel, out_dim, bias=self.mlp_bias))
        else:
            for i in range(num_layers - 1):
                if i == 0:
                    layers.append(torch.nn.Linear(inChanel, layer_width, bias=self.mlp_bias))
                else:
                    layers.append(torch.nn.Linear(layer_width, layer_width, bias=self.mlp_bias))
                layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(layer_width, out_dim, bias=self.mlp_bias))
        if out_activation is not None:
            layers.append(out_activation)
        self.mlp = torch.nn.Sequential(*layers)
        self.save_init_kwargs(locals())  # save for converting to fusion kernel

    def forward(self, features):
        """
        Forward pass of the MLP Render Feature.

        Args:
            viewdirs (torch.Tensor): The view direction tensor.
            features (torch.Tensor): The input feature tensor.

        Returns:
            torch.Tensor: The output RGB tensor.
        """
        x = self.mlp(features)
        return x
