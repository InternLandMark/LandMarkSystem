from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from typing_extensions import Literal

from landmark.nerf_components.utils.tcnn_utils import tcnn
from landmark.render.util.types import MergeType

from .base_encoding import VolumeEncoding


class HashEncoding(VolumeEncoding):
    """Class for HashEncoding.

    Args:
        num_levels (int): Number of feature grids.
        min_res (int): Resolution of smallest feature grid.
        max_res (int): Resolution of largest feature grid.
        log2_hashmap_size (int): Size of hash map is 2^log2_hashmap_size.
        features_per_level (int): Number of features per level.
        interpolation (Optional[Literal["Nearest", "Linear", "Smoothstep"]]): Interpolation override for tcnn hashgrid.
            Not supported for torch unless linear.
        device (str): Device to use for computation. Defaults to "cpu".
        num_hash_table (int): Number of hash tables. Defaults to 1.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
        device="cpu",
        num_hash_table=1,
    ):
        super().__init__()
        self.in_dim = 3
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size
        self.interpolation = interpolation
        self.min_res = min_res
        self.max_res = max_res
        self.device = device
        self.num_levels = num_levels

        levels = torch.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1
        self.scalings = torch.floor(min_res * growth_factor**levels)

        self.hash_offset = levels * self.hash_table_size

        self.tcnn_encoding = None
        self.tcnn_encoding_list = None

        self.hash_table = torch.empty(0)
        self.num_hash_table = num_hash_table

        self.init_hash_table(min_res, growth_factor, num_levels)

        self.save_init_kwargs(locals())  # save for converting to parallel encoding

        self.merge_config = {
            "merge_type": MergeType.List,
            "tcnn_encoding.*.params": 0,
        }

    def init_hash_table(self, min_res, growth_factor, num_levels):
        """Initialize the hash table.

        Args:
            min_res (int): Resolution of smallest feature grid.
            growth_factor (float): Growth factor for scaling the feature grids.
            num_levels (int): Number of feature grids.
        """
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": self.features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": min_res,
            "per_level_scale": growth_factor,
        }
        if self.interpolation is not None:
            encoding_config["interpolation"] = self.interpolation

        if self.num_hash_table > 1:
            self.tcnn_encoding_list = []
            for _ in range(self.num_hash_table):
                self.tcnn_encoding_list.append(
                    tcnn.Encoding(
                        n_input_dims=3,
                        encoding_config=encoding_config,
                    )
                )
            self.tcnn_encoding = torch.nn.ModuleList(self.tcnn_encoding_list)
        else:
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )

        if self.tcnn_encoding is None and self.tcnn_encoding_list is None:
            assert (
                self.interpolation is None or self.interpolation == "Linear"
            ), f"interpolation '{self.interpolation}' is not supported for torch encoding backend"

    def get_out_dim(self) -> int:
        """Get the output dimension of the encoding.

        Returns:
            int: The output dimension.
        """
        return self.num_levels * self.features_per_level

    def forward(self, xyz: torch.Tensor):
        """Forward pass of the encoding.

        Args:
            xyz (torch.Tensor): Input tensor of shape (batch_size, num_points, 3) or (batch_size, num_points, 4).

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, num_points, out_dim).
        """
        if xyz.shape[1] == 3:
            return self.tcnn_encoding(xyz)
        elif xyz.shape[1] == 4:
            if isinstance(self.tcnn_encoding, nn.ModuleList):
                masks = []
                for block_idx in range(len(self.tcnn_encoding)):
                    masks.append(xyz[:, -1] == block_idx)  # block index of a sample is given in last dimension

                out_tensor = torch.zeros(
                    [xyz.shape[0], self.num_levels * self.features_per_level], dtype=torch.float16, device=xyz.device
                )
                for i in range(len(self.tcnn_encoding)):
                    out_tensor[masks[i]] = self.tcnn_encoding[i](xyz[:, :-1][masks[i]])
                return out_tensor
            else:
                return self.tcnn_encoding(xyz[:, :-1])
        else:
            raise NotImplementedError


class BranchParallelHashEncoding(HashEncoding):
    """
    Branch Parallel version for HashEncoding.

    Inherits from HashEncoding class and implements a branch parallel version of the encoding.

    Args:
        parallel_degree (int): The degree of parallelism.
        parallel_part (int): The parallel part.
        group: The group.
        init_state_dict: The initial state dictionary.
        **kwargs: Additional keyword arguments.

    Attributes:
        parallel_degree (int): The degree of parallelism.
        parallel_part (int): The parallel part.
        init_state_dict: The initial state dictionary.
        group: The group.
    """

    def __init__(self, parallel_degree, parallel_part, group, init_state_dict, **kwargs):
        """
        Initializes a new instance of the BranchParallelHashEncoding class.

        Args:
            parallel_degree (int): The degree of parallelism.
            parallel_part (int): The parallel part.
            group: The group.
            init_state_dict: The initial state dictionary.
            **kwargs: Additional keyword arguments.
        """
        self.parallel_degree = parallel_degree
        self.parallel_part = parallel_part
        self.init_state_dict = init_state_dict
        self.group = group
        super().__init__(**kwargs)

    def init_hash_table(self, min_res, growth_factor, num_levels):
        """
        Initializes the hash table.

        Args:
            min_res: The minimum resolution.
            growth_factor: The growth factor.
            num_levels: The number of levels.
        """
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": self.features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": min_res,
            "per_level_scale": growth_factor,
        }
        if self.interpolation is not None:
            encoding_config["interpolation"] = self.interpolation

        self.tcnn_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=encoding_config,
        )
        local_params_shape = self.tcnn_encoding.params.shape
        global_params_shape = local_params_shape + (self.parallel_degree,)
        global_hash_table = torch.rand(size=global_params_shape, device=self.device) * 2 - 1
        dist.broadcast(global_hash_table, src=0, group=self.group)
        local_hash_table_params = global_hash_table[..., self.parallel_part].clone()
        local_hash_table_params = torch.nn.Parameter(local_hash_table_params, requires_grad=True)
        self.tcnn_encoding.register_parameter(name="params", param=local_hash_table_params)

        if self.tcnn_encoding is None:
            assert (
                self.interpolation is None or self.interpolation == "Linear"
            ), f"interpolation '{self.interpolation}' is not supported for torch encoding backend"
