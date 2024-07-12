import copy
from typing import List, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed import ReduceOp

from landmark.nerf_components.utils.comm_utils import (  # TODO to be changed (frank);
    AllGather,
    AllReduce,
)
from landmark.nerf_components.utils.loss_utils import TVLoss
from landmark.render.util.types import MergeType

from .base_encoding import VolumeEncoding


class TensorVMEncoding(VolumeEncoding):
    """Class for TensorVMEncoding"""

    def __init__(
        self,
        ndims: int,
        resolution_mode: List[int],
        n_component: int,  # channel nums
        grid_size: torch.Tensor,
        device: torch.device,
        param_shape: List[int] = None,
    ):
        """
        Initializes a TensorVMEncoding object.

        Args:
            ndims (int): The number of dimensions.
            resolution_mode (List[int]): The resolution mode.
            n_component (int): The number of components.
            grid_size (torch.Tensor): The grid size.
            device (torch.device): The device to use for computation.
            param_shape (List[int], optional): The shape of the parameters. Defaults to None.
        """
        super().__init__()

        self.ndims = ndims
        self.matrix_mode = [[0, 1], [0, 2], [1, 2]][: self.ndims]  # matMode
        self.vector_mode = [2, 1, 0][: self.ndims]  # vecMode
        self.n_component = n_component if isinstance(n_component, list) else [n_component] * 3
        self.n_component = self.n_component[: self.ndims]
        # self.grid_size = grid_size
        self.reso_mode = resolution_mode

        self.device = device

        self.plane_coef, self.line_coef = self.init_one_svd(self.n_component, grid_size, param_shape, scale=0.1)

        self.tv_reg = TVLoss()

        self.save_init_kwargs(locals())  # save for converting to parallel encoding

        self.is_channel_last = False
        # TODO: use a more elegent method to auto-gen this config.
        self.merge_config = {
            "merge_type": MergeType.Evenly,
            "plane_coef.0": 2,
            "plane_coef.1": 2,
            "plane_coef.2": 2,
            "line_coef.0": 3,
            "line_coef.1": 3,
            "line_coef.2": 3,
        }

    def init_one_svd(self, n_component, grid_size, param_shape=None, scale=0.1):
        """
        Initialize one SVD volume for a density or appearance.

        Args:
            n_component (int): The number of components.
            grid_size (list): The size of the grid.
            scale (float): The scaling factor.
            device (torch.device): The device to use for computation.

        Returns:
            tuple: A tuple containing the plane and line coefficients.
        """
        plane_coef, line_coef = [], []
        for i in range(len(self.vector_mode)):
            vec_id = self.vector_mode[i]
            mat_id_0, mat_id_1 = self.matrix_mode[i]
            for idx, j in enumerate(self.reso_mode):
                if param_shape is not None:
                    plane_coef_shape = param_shape[f"plane_coef.{idx}"]
                    line_coef_shape = param_shape[f"line_coef.{idx}"]
                else:
                    plane_coef_shape = (
                        1,
                        n_component[i],
                        1,
                        grid_size[mat_id_1] * j,
                        grid_size[mat_id_0] * j,
                    )
                    line_coef_shape = (1, n_component[i], grid_size[vec_id] * j, 1)
                plane_coef.append(
                    torch.nn.Parameter(scale * torch.randn(plane_coef_shape))  # TODO why not use mat_id_0 firstly.
                )
                line_coef.append(torch.nn.Parameter(scale * torch.randn(line_coef_shape)))

        # return CoefParam(plane_coef).to(self.device), CoefParam(line_coef).to(self.device)
        return torch.nn.ParameterList(plane_coef).to(self.device), torch.nn.ParameterList(line_coef).to(self.device)

    def channel_last(self):
        """
        Converts the encoding to channel last format.

        This method is used to convert the encoding to channel last format. It sets the `is_channel_last` flag to
        True and updates the merge configuration accordingly. It also permutes the `plane_coef` tensor to match the
        channel last format.

        Note: This method does not support backward operation when the model is in training mode.
        """
        assert not self.training, "channel_last format not support backward for now."
        self.is_channel_last = True
        self.merge_config["plane_coef.0"] = 1
        self.merge_config["plane_coef.1"] = 1
        self.merge_config["plane_coef.2"] = 1
        for i in range(len(self.reso_mode)):
            self.plane_coef[i] = self.plane_coef[i].permute(0, 2, 3, 4, 1).contiguous()

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:  # pylint:disable=W0237
        """
        Computes the feature for given samples coordinates.
        It is the same as the compute_densityfeature or compute_appfeature in the original tensorf code

        Args:
            xyz: (N, 3, 1) tensor, the coordinates of the points

        Returns:
            (N, n_component) tensor, the feature of the points

        """

        N = self.ndims
        coordinate_plane = (
            torch.stack([xyz[..., [*self.matrix_mode[i], self.matrix_mode[i][1] + 2]] for i in range(N)])
            .detach()
            .view(N, 1, -1, 1, 3)
        )  # 5D: (W, H, D)
        coordinate_line = (
            torch.stack([xyz[..., [self.vector_mode[i] + 1, self.vector_mode[i]]] for i in range(N)])
            .detach()
            .view(N, -1, 1, 2)
        )  # 4D: (W, H)
        plane_coef_point, line_coef_point = [], []

        for idx_plane in range(len(self.plane_coef)):
            idx_dim = idx_plane // len(self.reso_mode)
            if self.is_channel_last:
                plane_coef = torch.zeros(
                    (
                        1,
                        self.plane_coef[idx_plane].shape[-1],
                        coordinate_plane[[idx_dim]].shape[1],
                        coordinate_plane[[idx_dim]].shape[2],
                        1,
                    ),
                    device="cuda",
                )
                import grid_sample_ndhwc2ncdhw_3d

                grid_sample_ndhwc2ncdhw_3d.cuda(
                    self.plane_coef[idx_plane],
                    coordinate_plane[[idx_dim]],
                    plane_coef,
                    True,
                    True,
                )
                plane_coef = plane_coef.view(-1, *xyz.shape[:-1])
            else:
                plane_coef = F.grid_sample(
                    self.plane_coef[idx_plane],
                    coordinate_plane[[idx_dim]],
                    align_corners=True,
                ).reshape(-1, *xyz.shape[:-1])
            line_coef = F.grid_sample(
                self.line_coef[idx_plane],
                coordinate_line[[idx_dim]],
                align_corners=True,
            ).reshape(-1, *xyz.shape[:-1])

            plane_coef_point.append(plane_coef)
            line_coef_point.append(line_coef)

        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        rets = plane_coef_point * line_coef_point
        rets = rets.permute(*range(1, rets.dim()), 0)

        return rets

    @torch.no_grad()
    def __upsample_VM(self, plane_coef, line_coef, grid_size):
        """
        Upsamples the plane and line coefficients to a target resolution.

        Args:
            plane_coef (list): The plane coefficients.
            line_coef (list): The line coefficients.
            grid_size (list): The target grid size.

        Returns:
            tuple: A tuple containing the upsampled plane and line coefficients.
        """
        for i in range(len(self.vector_mode)):
            vec_id = self.vector_mode[i]
            mat_id_0, mat_id_1 = self.matrix_mode[i]
            for j in self.reso_mode:
                plane_idx = i * len(self.reso_mode) + self.reso_mode.index(j)
                plane_coef[plane_idx] = torch.nn.Parameter(
                    F.interpolate(
                        plane_coef[plane_idx].data,
                        size=(1, grid_size[mat_id_1] * j, grid_size[mat_id_0] * j),
                        mode="trilinear",
                        align_corners=True,
                    )
                )
                line_coef[plane_idx] = torch.nn.Parameter(
                    F.interpolate(
                        line_coef[plane_idx].data,
                        size=(grid_size[vec_id] * j, 1),
                        mode="bilinear",
                        align_corners=True,
                    )
                )

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, grid_size: Union[torch.Tensor, List]) -> torch.Tensor:
        """
        Upsamples the volume grid to a target grid size.

        Args:
            grid_size (list, Tensor): The target grid size.
        """
        self.plane_coef, self.line_coef = self.__upsample_VM(self.plane_coef, self.line_coef, grid_size)
        # self.grid_size = grid_size

        print(f"upsamping to {grid_size} for each branch")

    def L1_loss(self):
        """
        Computes the L1 loss for plane and line

        Returns:
            torch.Tensor: The loss value
        """
        total = 0
        for idx in range(len(self.plane_coef)):
            total = total + torch.mean(torch.abs(self.plane_coef[idx])) + torch.mean(torch.abs(self.line_coef[idx]))
        return total

    def TV_loss(self):
        """
        Computes the total variation loss for the plane coefficients.

        Args:
            reg (function): The regularization function.

        Returns:
            torch.Tensor: The total variation loss for the plane coefficients.
        """
        total = 0
        for idx in range(len(self.plane_coef)):
            total = total + self.tv_reg(self.plane_coef[idx].squeeze(2)) * 1e-2  # squeeze the block_idx dim
        return total

    def vector_diffs(self):
        """
        Computes the mean absolute difference between non-diagonal elements of the dot product of the input vectors.

        Args:
            vector_comps (list): A list of vectors.

        Returns:
            torch.Tensor: The mean absolute difference between non-diagonal elements of the dot product of \
                the input vectors.
        """
        vector_comps = self.line_coef
        total = 0
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            dotp = torch.matmul(
                vector_comps[idx].view(n_comp, n_size),
                vector_comps[idx].view(n_comp, n_size).transpose(-1, -2),
            )
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def merge_state_dicts(self, state_dicts, device="cpu"):
        """
        Merges a list of state dictionaries into a single state dictionary.

        Args:
            state_dicts (list): A list of state dictionaries to be merged.
            device (str, optional): The device to be used for the merged tensor. Defaults to "cpu".

        Returns:
            dict: The merged state dictionary.

        """
        merged_state_dict = copy.deepcopy(state_dicts[0])
        keys = sorted(merged_state_dict.keys())
        branch_nums = len(state_dicts)
        for key in keys:
            if "plane_coef" in key:
                _, n_channel, _, h, w = merged_state_dict[key].shape
                global_shape = (1, n_channel, branch_nums, h, w)
                merged_tensor = torch.zeros(global_shape, device=device)
                for branch_idx in range(branch_nums):
                    merged_tensor[:, :, [branch_idx]] = state_dicts[branch_idx][key]
            elif "line_coef" in key:
                global_shape = (*merged_state_dict[key].shape[:-1], branch_nums)
                merged_tensor = torch.zeros(global_shape, device=device)
                for branch_idx in range(branch_nums):
                    merged_tensor[..., [branch_idx]] = state_dicts[branch_idx][key]
            merged_state_dict[key] = merged_tensor

        return merged_state_dict


# class PlaneParallelTensorVMEncoding(TensorVMEncoding):
#     ...


class ChannelParallelTensorVMEncoding(TensorVMEncoding):
    """Channel Parallel version of TensorVMEncoding"""

    def __init__(self, parallel_degree, parallel_part, group, init_state_dict, **kwargs):
        """
        Initializes the TensorVMEncoding object.

        Args:
            parallel_degree (int): The degree of parallelism.
            parallel_part (int): The parallel part.
            group (str): The group.
            init_state_dict (dict): The initial state dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.parallel_degree = parallel_degree
        self.parallel_part = parallel_part
        self.init_state_dict = init_state_dict
        self.group = group
        self.save_init_kwargs(locals())  # save for converting to fusion kernel
        super().__init__(**kwargs)

    def init_one_svd(self, n_component, grid_size, param_shape=None, scale=0.1):
        """
        Initialize the SVD parameters for the tensor VM encoding.

        Args:
            n_component (int): Number of components.
            grid_size (int): Size of the grid.
            param_shape (tuple, optional): Shape of the parameters. Defaults to None.
            scale (float, optional): Scaling factor for initialization. Defaults to 0.1.

        Returns:
            tuple: A tuple containing two torch.nn.ParameterList objects, one for plane coefficients and one
            for line coefficients.
        """
        plane_coef, line_coef = [], []
        for _ in range(len(self.vector_mode)):
            for idx, _ in enumerate(self.reso_mode):
                plane_coef.append(
                    torch.nn.Parameter(
                        torch.chunk(
                            self.init_state_dict[f"plane_coef.{idx}"],
                            self.parallel_degree,
                            dim=1,
                        )[self.parallel_part]
                    )
                )
                line_coef.append(
                    torch.nn.Parameter(
                        torch.chunk(
                            self.init_state_dict[f"line_coef.{idx}"],
                            self.parallel_degree,
                            dim=1,
                        )[self.parallel_part]
                    )
                )
        return torch.nn.ParameterList(plane_coef).to(self.device), torch.nn.ParameterList(line_coef).to(self.device)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Computes the feature for given samples coordinates.
        It is the same as the compute_densityfeature or compute_appfeature in the original tensorf code

        Args:
            xyz: should be xyzb (N, 4) tensor, which b is 'block', the coordinates of the points

        Returns:
            (N, n_component) tensor, the feature of the points

        """
        N = self.ndims
        coordinate_plane = (
            torch.stack([xyz[..., [*self.matrix_mode[i], self.matrix_mode[i][1] + 2]] for i in range(N)])
            .detach()
            .view(N, 1, -1, 1, 3)
        )  # 5D: (W, H, D)
        coordinate_line = (
            torch.stack([xyz[..., [self.vector_mode[i] + 1, self.vector_mode[i]]] for i in range(N)])
            .detach()
            .view(N, -1, 1, 2)
        )  # 4D: (W, H)
        plane_coef_point, line_coef_point = [], []

        for idx_plane in range(len(self.plane_coef)):
            idx_dim = idx_plane // len(self.reso_mode)
            plane_coef = F.grid_sample(
                self.plane_coef[idx_plane],
                coordinate_plane[[idx_dim]],
                align_corners=True,
            ).reshape(-1, *xyz.shape[:-1])
            line_coef = F.grid_sample(
                self.line_coef[idx_plane],
                coordinate_line[[idx_dim]],
                align_corners=True,
            ).reshape(-1, *xyz.shape[:-1])

            plane_coef_point.append(self.all_gather_func(plane_coef))
            line_coef_point.append(self.all_gather_func(line_coef))

        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)
        # rets = (plane_coef_point * line_coef_point).T  # shape: [Batch, n_component]
        rets = plane_coef_point * line_coef_point
        rets = rets.permute(*range(1, rets.dim()), 0)

        return rets

    def all_gather_func(self, tensor):
        """wrapped all_gather function

        Args:
            tensor (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: garhered tensor
        """
        return AllGather.apply(tensor, self.parallel_part, self.parallel_degree, self.group)

    # wrapped all_reduce function
    def all_reduce_func(self, tensor):
        """
        Applies all-reduce operation on the given tensor.

        Args:
            tensor (torch.Tensor): The input tensor to be reduced.

        Returns:
            torch.Tensor: The reduced tensor after all-reduce operation.
        """
        tensor.div_(self.args.model_parallel_degree)
        return AllReduce.apply(ReduceOp.SUM, self.group, tensor)

    def vector_diffs(self):
        """
        Computes the mean absolute difference between non-diagonal elements of the dot product of the input vectors.

        Args:
            vector_comps (list): A list of vectors.

        Returns:
            torch.Tensor: The mean absolute difference between non-diagonal elements of the dot product of \
                the input vectors.
        """
        vector_comps = self.line_coef
        total = 0
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            vcmat = self.all_gather_func(vector_comps[idx].view(n_comp, n_size))
            dotp = torch.matmul(vcmat, vcmat.transpose(-1, -2))
            n_comp = n_comp * self.parallel_degree
            non_diagonal = dotp.view(-1)[1:].view(n_comp - 1, n_comp + 1)[..., :-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def L1_loss(self):
        """
        Computes the L1 loss for plane and line

        Returns:
            torch.Tensor: The loss value
        """
        total = 0
        for idx in range(len(self.plane_coef)):
            dp = self.all_gather_func(torch.mean(torch.abs(self.plane_coef[idx])).view([1, 1]))
            dl = self.all_gather_func(torch.mean(torch.abs(self.line_coef[idx])).view([1, 1]))
            total = total + torch.mean(dp) + torch.mean(dl)
        return total

    def TV_loss(self):
        """
        Computes the total variation loss for the plane coefficients.

        Returns:
            torch.Tensor: The total variation loss for the plane coefficients.
        """
        total = 0
        for idx in range(len(self.plane_coef)):
            total = total + self.tv_reg(self.plane_coef[idx].squeeze(2)) * 1e-2
        return torch.mean(self.all_gather_func(total.view([1, 1])))

    def merge_state_dicts(self, state_dicts, device="cpu"):
        """
        Merges a list of state dictionaries into a single state dictionary.

        Args:
            state_dicts (list): A list of state dictionaries to be merged.
            device (str, optional): The device to be used for the merged tensor. Defaults to "cpu".

        Returns:
            dict: The merged state dictionary.

        """
        merged_state_dict = copy.deepcopy(state_dicts[0])
        keys = sorted(merged_state_dict.keys())
        channel_size = len(state_dicts)
        for key in keys:
            if "plane_coef" in key:
                _, n_channel, b, h, w = merged_state_dict[key].shape
                global_shape = (1, n_channel * channel_size, b, h, w)
                merged_tensor = torch.zeros(global_shape, device=device)
                for channel_idx in range(channel_size):
                    merged_tensor[:, channel_idx * n_channel : (channel_idx + 1) * n_channel] = state_dicts[
                        channel_idx
                    ][key]
            elif "line_coef" in key:
                _, n_channel, h, b = merged_state_dict[key].shape
                global_shape = (1, n_channel * channel_size, h, b)
                merged_tensor = torch.zeros(global_shape, device=device)
                for channel_idx in range(channel_size):
                    merged_tensor[:, channel_idx * n_channel : (channel_idx + 1) * n_channel] = state_dicts[
                        channel_idx
                    ][key]
            merged_state_dict[key] = merged_tensor

        return merged_state_dict


class BranchParallelTensorVMEncoding(TensorVMEncoding):
    """Branch Parallel version of TensorVMEncoding"""

    def __init__(self, parallel_degree, parallel_part, group, init_state_dict, **kwargs):
        """
        Initializes the TensorVMEncoding object.

        Args:
            parallel_degree (int): The degree of parallelism.
            parallel_part (int): The parallel part.
            group (str): The group.
            init_state_dict (dict): The initial state dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.parallel_degree = parallel_degree
        self.parallel_part = parallel_part
        self.init_state_dict = init_state_dict
        self.group = group
        super().__init__(**kwargs)

    def init_one_svd(self, n_component, grid_size, param_shape=None, scale=0.1):
        """
        Initialize the parameters for the SVD encoding.

        Args:
            n_component (list): A list of integers representing the number of components for each vector mode.
            grid_size (list): A list of integers representing the grid size for each matrix mode.
            param_shape (tuple, optional): The shape of the parameter tensor. Defaults to None.
            scale (float, optional): The scaling factor for the initialized parameters. Defaults to 0.1.

        Returns:
            tuple: A tuple containing two torch.nn.ParameterList objects - one for the plane coefficients and one
            for the line coefficients.
        """
        plane_coef, line_coef = [], []
        for i in range(len(self.vector_mode)):
            vec_id = self.vector_mode[i]
            mat_id_0, mat_id_1 = self.matrix_mode[i]
            for j in self.reso_mode:
                global_plane = scale * torch.randn(
                    (
                        1,
                        n_component[i],
                        1,
                        grid_size[mat_id_1] * j,
                        grid_size[mat_id_0] * j,
                        self.parallel_degree,
                    ),
                    device=self.device,
                )
                dist.broadcast(global_plane, src=0, group=self.group)
                local_plane = global_plane[..., self.parallel_part].clone()
                del global_plane
                plane_coef.append(torch.nn.Parameter(local_plane))
                line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], grid_size[vec_id] * j, 1))))
        return torch.nn.ParameterList(plane_coef).to(self.device), torch.nn.ParameterList(line_coef).to(self.device)
        # return CoefParam(plane_coef).to(self.device), CoefParam(line_coef).to(self.device)

    def all_gather_func(self, tensor):
        """wrapped all_gather function

        Args:
            tensor (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: garhered tensor
        """
        return AllGather.apply(tensor, self.parallel_part, self.parallel_degree, self.group)

    def TV_loss(self):
        """
        Calculates the total variation (TV) loss for the plane coefficients.

        Returns:
            torch.Tensor: The mean TV loss across all plane coefficients.
        """
        total = 0
        for idx in range(len(self.plane_coef)):
            total = total + self.tv_reg(self.plane_coef[idx].squeeze(2)) * 1e-2  # squeeze the block_idx dim
        return torch.mean(self.all_gather_func(total.view([1, 1])))


class CoefParam(nn.Module):
    """
    A wrap for plane and line, so that it can be wrapped by DDP.

    Args:
        coefs (list): A list of coefficients.
    """

    def __init__(self, coefs) -> None:
        super().__init__()
        self.params = torch.nn.ParameterList(coefs)

    def forward(self, coordinates, idx_plane, idx_dim, align_corners=True):
        """
        Do F.grid_sample on the params.

        Args:
            coordinate (torch.Tensor): The coordinate to sample from.
            idx_plane (int): The index of the plane to sample from.
            idx_dim (int): The index of the dimension to sample from.
            align_corners (bool): Whether to align the corners of the grid.

        Returns:
            torch.Tensor: The sampled values.
        """
        return F.grid_sample(
            self.params[idx_plane],
            coordinates[[idx_dim]],
            align_corners=align_corners,
        )

    def __setitem__(self, idx, value):
        self.params[idx] = value

    def __getitem__(self, idx):
        return self.params[idx]

    def __len__(self):
        return len(self.params)
