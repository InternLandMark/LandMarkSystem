from typing import Optional, Union

import torch

from landmark.nerf_components.model_components.fields.mlp_decoder import MLPDecoder
from landmark.nerf_components.utils.tcnn_utils import tcnn


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


class FusedMLPDecoder(MLPDecoder):
    """
    A MLP decoder for rendering features.

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
        super(MLPDecoder, self).__init__()
        self.mlp_bias = mlp_bias
        if self.mlp_bias:
            # TODO: kernel implementation
            pass
        else:
            assert inChanel > 0
            self.out_dim = out_dim if out_dim is not None else layer_width
            output_activation_str = activation_to_tcnn_string(out_activation)
            if layer_width in [16, 32, 64, 128]:
                network_config = {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": output_activation_str,
                    "n_neurons": layer_width,
                    "n_hidden_layers": num_layers - 1,
                }
            else:
                network_config = {
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": output_activation_str,
                    "n_neurons": layer_width,
                    "n_hidden_layers": num_layers - 1,
                }

            self.tcnn_encoding = tcnn.Network(
                n_input_dims=inChanel,
                n_output_dims=self.out_dim,
                network_config=network_config,
            )

    def forward(self, features):
        """
        Forward pass of the MLP Render Feature.

        Args:
            features (torch.Tensor): The input feature tensor.

        Returns:
            torch.Tensor: The output RGB tensor.
        """
        if self.mlp_bias:
            raise NotImplementedError("TODO")
        else:
            return self.tcnn_encoding(features)
