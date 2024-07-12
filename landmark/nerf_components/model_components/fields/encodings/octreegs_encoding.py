import math
import os
import time
from functools import reduce
from typing import Any, Dict, Mapping, Union

import numpy as np
import torch
from einops import repeat
from plyfile import PlyData, PlyElement
from torch import nn
from torch_scatter import scatter_max

from landmark.nerf_components.utils.general_utils import (
    build_scaling_rotation,
    inverse_sigmoid,
    strip_symmetric,
)
from landmark.nerf_components.utils.graphics_utils import BasicPointCloud
from landmark.render.util.types import MergeType

from .base_encoding import BaseGaussianEncoding


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    """
    Build a covariance matrix from scaling and rotation.

    Parameters:
    scaling (torch.Tensor): The scaling tensor.
    scaling_modifier (float): The scaling modifier.
    rotation (torch.Tensor): The rotation tensor.

    Returns:
    torch.Tensor: The covariance matrix.
    """
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


class OctreeGSEncoding(BaseGaussianEncoding):
    """Class for OctreeGS encoding"""

    def __init__(
        self,
        feat_dim: int = 32,
        n_offsets: int = 5,
        fork: int = 2,
        visible_threshold: float = -1,
        dist2level: str = "round",
        base_layer: int = 10,
        progressive: bool = True,
        extend: float = 1.1,
        percent_dense: float = 0,
        enable_lod: bool = True,
    ):
        super().__init__()

        # Initialize class attributes
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.fork = fork
        self.progressive = progressive
        self.percent_dense = percent_dense
        self.enable_lod = enable_lod

        # Octree attributes
        self.sub_pos_offsets = (
            torch.tensor([[i % fork, (i // fork) % fork, i // (fork * fork)] for i in range(fork**3)]).float().cuda()
        )
        self.extend = extend
        self.visible_threshold = visible_threshold
        self.dist2level = dist2level
        self.base_layer = base_layer

        # Initialize other attributes
        self._anchor = torch.empty(0)
        self._level = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self.opacity_accum = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.merge_config = {
            "merge_type": MergeType.Unevenly,
            "_anchor_feat": 0,
            "_offset": 0,
            "_anchor": 0,
            "_opacity": 0,
            "_scaling": 0,
            "_rotation": 0,
            "_level": 0,
            "_extra_level": 0,
        }

    def create_from_pcd(self, pcd: BasicPointCloud):
        """
        Create OctreeGS encoding from a point cloud.

        Args:
            pcd (BasicPointCloud): The input point cloud.

        Returns:
            None
        """
        from simple_knn._C import distCUDA2
        # Convert point cloud to tensor
        points = torch.tensor(pcd.points, device="cuda").float()

        # Calculate bounding box
        box_min = torch.min(points) * self.extend
        box_max = torch.max(points) * self.extend
        box_d = box_max - box_min

        # Calculate voxel size and initial position
        if self.base_layer < 0:
            init_dist = distCUDA2(points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0] * 0.5))
            self.voxel_size = median_dist.item()
            self.base_layer = int(torch.log2(box_d / self.voxel_size).item()) - self.levels + 1
            del init_dist
            torch.cuda.empty_cache()
        self.voxel_size = box_d / (float(self.fork) ** self.base_layer)
        self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda()

        # Perform octree sampling
        self.octree_sample(points, self.init_pos)

        # Weed out invisible voxels
        if self.visible_threshold < 0:
            self.visible_threshold = 0.0
            self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
        self.positions, self._level, _, _ = self.weed_out(self.positions, self._level)

        # Print information about the octree
        print(f"Branches of Tree: {self.fork}")
        print(f"Base Layer of Tree: {self.base_layer}")
        print(f"Visible Threshold: {self.visible_threshold}")
        print(f"LOD Levels: {self.levels}")
        print(f"Initial Levels: {self.init_level}")
        print(f"Initial Voxel Number: {self.positions.shape[0]}")
        print(f"Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}")
        print(f"Max Voxel Size: {self.voxel_size}")

        # Initialize offsets, anchor features, scales, rotations, and opacities
        offsets = torch.zeros((self.positions.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((self.positions.shape[0], self.feat_dim)).float().cuda()
        dist2 = torch.clamp_min(distCUDA2(self.positions).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)
        rots = torch.zeros((self.positions.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((self.positions.shape[0], 1), dtype=torch.float, device="cuda"))

        # Initialize parameters as nn.Parameters
        self._anchor = nn.Parameter(self.positions.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self._level = nn.Parameter(self._level.float().unsqueeze(dim=1)).requires_grad_(False)
        self._extra_level = nn.Parameter(
            torch.zeros(self._anchor.shape[0], dtype=torch.float, device="cuda")
        ).requires_grad_(False)
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")

        # Initialize other attributes
        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

    def octree_sample(self, data, init_pos):
        """
        Perform octree sampling on the input data.

        Args:
            data: The input data.
            init_pos: The initial position.

        Returns:
            None
        """
        torch.cuda.synchronize()
        t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self._level.data = torch.empty(0).int().cuda()
        for cur_level in range(self.levels):
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            new_positions = torch.unique(torch.round((data - init_pos) / cur_size), dim=0) * cur_size + init_pos
            new_level = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self._level.data = torch.concat((self._level.data, new_level), dim=0)
        torch.cuda.synchronize()
        t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def weed_out(self, anchor_positions, anchor_levels):
        """
        Weed out invisible voxels based on visibility threshold.

        Args:
            anchor_positions: The anchor positions.
            anchor_levels: The anchor levels.

        Returns:
            Tuple containing:
                - anchor_positions (torch.Tensor): The filtered anchor positions.
                - anchor_levels (torch.Tensor): The filtered anchor levels.
                - mean_visible (float): The mean visibility.
                - weed_mask (torch.Tensor): The mask indicating which voxels were filtered out.
        """
        visible_count = torch.zeros(anchor_positions.shape[0], dtype=torch.int, device="cuda")
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((anchor_positions - cam_center) ** 2, dim=1)) * scale
            pred_level = torch.log2(self.standard_dist / dist) / math.log2(self.fork)
            int_level = self.map_to_int_level(pred_level, self.levels - 1)
            visible_count += (anchor_levels <= int_level).int()
        visible_count = visible_count / len(self.cam_infos)
        weed_mask = visible_count > self.visible_threshold
        mean_visible = torch.mean(visible_count)
        return anchor_positions[weed_mask], anchor_levels[weed_mask], mean_visible, weed_mask

    def set_anchor_mask(self, cam_center, iteration, resolution_scale, mlp_is_training):
        """
        Set the anchor mask based on the camera center, iteration, resolution scale, and MLP training status.

        Args:
            cam_center: The camera center.
            iteration: The current iteration.
            resolution_scale: The resolution scale.
            mlp_is_training: Whether the MLP is training or not.

        Returns:
            None
        """
        if self.enable_lod:
            anchor_pos = self._anchor + (self.voxel_size / 2) / (float(self.fork) ** self._level)
            dist = torch.sqrt(torch.sum((anchor_pos - cam_center) ** 2, dim=1)) * resolution_scale
            pred_level = torch.log2(self.standard_dist / dist) / math.log2(self.fork) + self._extra_level

            if self.progressive and mlp_is_training:
                coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
            else:
                coarse_index = torch.max(self._level) - torch.min(self._level) + 1

            int_level = self.map_to_int_level(pred_level, coarse_index - 1)
            self._anchor_mask = self._level.squeeze(dim=1) <= int_level
        else:
            self._anchor_mask = torch.ones(self._level.shape[0], dtype=torch.bool, device=self._level.device)

    def map_to_int_level(self, pred_level, cur_level):
        """
        Map predicted levels to integer levels based on the dist2level parameter.

        Args:
            pred_level: The predicted levels.
            cur_level: The current level.

        Returns:
            int_level (torch.Tensor): The mapped integer levels.
        """
        if self.dist2level == "floor":
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level == "round":
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level == "ceil":
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level == "progressive":
            pred_level = torch.clamp(pred_level + 1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = self._level.squeeze(dim=1) == int_level
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")

        return int_level

    def get_state_dict_from_ckpt(
        self, file_path: Union[str, os.PathLike], map_location: Union[str, Dict[str, str]], prefix: str = ""
    ) -> Mapping[str, Any]:
        """
        Get the state dictionary from a checkpoint file.

        Args:
            file_path (str or os.PathLike): The path to the checkpoint file.
            map_location (str or dict): The map location for loading the checkpoint.
            prefix (str): The prefix to prepend to the keys in the state dictionary.

        Returns:
            Mapping[str, Any]: The state dictionary.
        """
        plydata = PlyData.read(file_path)

        anchor = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        ).astype(np.float32)

        levels = np.asarray(plydata.elements[0]["level"])[..., np.newaxis].astype(np.int_)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[..., np.newaxis].astype(np.float32)
        self.voxel_size = torch.tensor(plydata.elements[0]["info"][0]).float()
        self.standard_dist = torch.tensor(plydata.elements[0]["info"][1]).float()

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key=lambda x: int(x.split("_")[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key=lambda x: int(x.split("_")[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        _anchor_feat = torch.tensor(anchor_feats, dtype=torch.float, device=map_location)
        _level = torch.tensor(levels, dtype=torch.int, device=map_location)
        _extra_level = torch.tensor(extra_levels, dtype=torch.float, device=map_location).squeeze(dim=1)
        _offset = torch.tensor(offsets, dtype=torch.float, device=map_location).transpose(1, 2)
        _anchor = torch.tensor(anchor, dtype=torch.float, device=map_location)
        _scaling = torch.tensor(scales, dtype=torch.float, device=map_location)
        _opacity = torch.tensor(opacities, dtype=torch.float, device=map_location)
        _rotation = torch.tensor(rots, dtype=torch.float, device=map_location)

        return {
            prefix + "_anchor_feat": _anchor_feat,
            prefix + "_level": _level,
            prefix + "_extra_level": _extra_level,
            prefix + "_offset": _offset,
            prefix + "_anchor": _anchor,
            prefix + "_scaling": _scaling,
            prefix + "_opacity": _opacity,
            prefix + "_rotation": _rotation,
        }

    def load_from_state_dict(self, state_dict, strict: bool = False):
        """
        Loads the model's state from a state dictionary.

        Args:
            state_dict (dict): The state dictionary containing the model's parameters.
            strict (bool, optional): Whether to strictly enforce that all keys in the state dictionary are expected.
                Defaults to False.
        """
        params_key = [
            "_anchor_feat",
            "_level",
            "_extra_level",
            "_offset",
            "_anchor",
            "_scaling",
            "_opacity",
            "_rotation",
        ]
        duplicate_keys = set(state_dict.keys()) & set(params_key)
        if strict is True and len(duplicate_keys) != len(params_key):
            raise ValueError(f"only found keys {duplicate_keys} in state dict")
        for param in duplicate_keys:
            setattr(self, param, nn.Parameter(torch.zeros(state_dict[param].shape, device=state_dict[param].device)))
        super().load_state_dict(state_dict, strict)
        if "_level" in duplicate_keys:
            self.levels = torch.max(self._level).int() - torch.min(self._level).int() + 1
        if "_anchor" in duplicate_keys:
            self._anchor_mask = torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device=self.get_anchor.device)
            self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device=self.get_anchor.device)
            self.offset_gradient_accum = torch.zeros(
                (self.get_anchor.shape[0] * self.n_offsets, 1), device=self.get_anchor.device
            )
            self.offset_denom = torch.zeros(
                (self.get_anchor.shape[0] * self.n_offsets, 1), device=self.get_anchor.device
            )
            self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device=self.get_anchor.device)

    def save_ply(self, path):
        """
        Save the encoded octree as a PLY file.

        Args:
            path (str): The path to save the PLY file.

        Returns:
            None
        """
        anchor = self._anchor.detach().cpu().numpy()
        levels = self._level.detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()
        infos = np.zeros_like(levels, dtype=np.float32)
        infos[0, 0] = self.voxel_size
        infos[1, 0] = self.standard_dist

        anchor_feats = self._anchor_feat.detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rots = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (anchor, levels, extra_levels, infos, offsets, anchor_feats, opacities, scales, rots), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        """
        Constructs a list of attributes for the octreegs encoding.

        Returns:
            attr_list (list): A list of attributes including 'x', 'y', 'z', 'level', 'extra_level', 'info',
                              'f_offset_i' (for each i in range(self._offset.shape[1] * self._offset.shape[2])),
                              'f_anchor_feat_i' (for each i in range(self._anchor_feat.shape[1])),
                              'opacity', 'scale_i' (for each i in range(self._scaling.shape[1])),
                              'rot_i' (for each i in range(self._rotation.shape[1])).
        """
        attr_list = ["x", "y", "z", "level", "extra_level", "info"]
        for i in range(self._offset.shape[1] * self._offset.shape[2]):
            attr_list.append(f"f_offset_{i}")
        for i in range(self._anchor_feat.shape[1]):
            attr_list.append(f"f_anchor_feat_{i}")
        attr_list.append("opacity")
        for i in range(self._scaling.shape[1]):
            attr_list.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            attr_list.append(f"rot_{i}")
        return attr_list

    def cat_tensors_to_optimizer(self, tensors_dict, optimizer):
        """
        Concatenates extension tensors to the optimizer's parameters and returns the optimizable tensors.

        Args:
            tensors_dict (dict): A dictionary containing the extension tensors.
            optimizer (torch.optim.Optimizer): The optimizer object.

        Returns:
            dict: A dictionary containing the optimizable tensors.

        Raises:
            AssertionError: If the number of parameters in a group is not equal to 1.

        """
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if (
                "mlp" in group["name"]
                or "conv" in group["name"]
                or "feat_base" in group["name"]
                or "embedding" in group["name"]
            ):
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                del optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # statis grad information to guide liftting.
    def training_statis(
        self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask
    ):
        """
        Update the training statistics for the octreegs encoding.

        Args:
            viewspace_point_tensor (torch.Tensor): The viewspace point tensor.
            opacity (torch.Tensor): The opacity tensor.
            update_filter (torch.Tensor): The update filter tensor.
            offset_selection_mask (torch.Tensor): The offset selection mask tensor.
            anchor_visible_mask (torch.Tensor): The anchor visible mask tensor.

        Returns:
            None
        """
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0

        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)

        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask, optimizer):
        """
        Prunes the optimizer by applying a mask to the optimizer's parameters.

        Args:
            mask (torch.Tensor): A boolean mask indicating which parameters to keep.
            optimizer (torch.optim.Optimizer): The optimizer to be pruned.

        Returns:
            dict: A dictionary containing the pruned optimizable tensors, where the keys are the names of the
            parameter groups and the values are the pruned parameters.
        """
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if (
                "mlp" in group["name"]
                or "conv" in group["name"]
                or "feat_base" in group["name"]
                or "embedding" in group["name"]
            ):
                continue

            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizer.state[group["params"][0]] = stored_state
                if group["name"] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group["name"] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_anchor(self, mask, optimizer):
        """
        Prunes the anchor based on the given mask and optimizer.

        Args:
            mask (torch.Tensor): A boolean mask indicating which points to keep.
            optimizer: The optimizer used for optimization.

        Returns:
            None

        """
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask, optimizer)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._level.data = self._level[valid_points_mask]
        self._extra_level.data = self._extra_level[valid_points_mask]

    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, use_chunk=True):
        """
        Returns a boolean mask indicating which grid coordinates in `grid_coords` are duplicates of any of the
        grid coordinates in `selected_grid_coords_unique`.

        Args:
            grid_coords (torch.Tensor): Tensor of shape (N, D) representing the grid coordinates.
            selected_grid_coords_unique (torch.Tensor): Tensor of shape (M, D) representing the unique grid
            coordinates.
            use_chunk (bool, optional): Whether to process the grid coordinates in chunks. Defaults to True.

        Returns:
            torch.Tensor: Boolean mask of shape (N,) indicating which grid coordinates are duplicates.
        """
        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for i in range(max_iters):
                cur_remove_duplicates = (
                    (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i * chunk_size : (i + 1) * chunk_size, :])
                    .all(-1)
                    .any(-1)
                    .view(-1)
                )
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
        else:
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
        return remove_duplicates

    def anchor_growing(self, iteration, grads, threshold, update_ratio, extra_ratio, extra_up, offset_mask, optimizer):
        """
        Performs anchor growing for the octreegs_encoding.

        Args:
            iteration (int): The current iteration.
            grads (torch.Tensor): The gradients.
            threshold (float): The threshold value.
            update_ratio (float): The update ratio.
            extra_ratio (float): The extra ratio.
            extra_up (float): The extra up value.
            offset_mask (torch.Tensor): The offset mask.
            optimizer: The optimizer.

        Returns:
            None
        """
        init_length = self.get_anchor.shape[0]
        grads[~offset_mask] = 0.0
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (
            torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6
        )
        for cur_level in range(self.levels):
            update_value = self.fork**update_ratio
            level_mask = (self.get_level == cur_level).squeeze(dim=1)
            if torch.sum(level_mask) == 0:
                continue
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            # update threshold
            cur_threshold = threshold * (update_value**cur_level)
            ds_size = cur_size / self.fork
            ds_threshold = cur_threshold * update_value
            extra_threshold = cur_threshold * extra_ratio
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)
            candidate_ds_mask = grads >= ds_threshold
            candidate_extra_mask = anchor_grads >= extra_threshold

            length_inc = self.get_anchor.shape[0] - init_length
            if length_inc > 0:
                candidate_mask = torch.cat(
                    [candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device="cuda")], dim=0
                )
                candidate_ds_mask = torch.cat(
                    [candidate_ds_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device="cuda")],
                    dim=0,
                )
                candidate_extra_mask = torch.cat(
                    [candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device="cuda")], dim=0
                )

            repeated_mask = repeat(level_mask, "n -> (n k)", k=self.n_offsets)
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)
            candidate_ds_mask = torch.logical_and(candidate_ds_mask, repeated_mask)
            if ~self.progressive or iteration > self.coarse_intervals[-1]:
                self._extra_level.data += extra_up * candidate_extra_mask.float()

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            grid_coords = torch.round((self.get_anchor[level_mask] - self.init_pos) / cur_size).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round((selected_xyz - self.init_pos) / cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(
                selected_grid_coords, return_inverse=True, dim=0
            )
            if selected_grid_coords_unique.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates  # pylint: disable=E1130
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + self.init_pos
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device="cuda") * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device="cuda")
                remove_duplicates = torch.ones([0], dtype=torch.bool, device="cuda")
                new_level = torch.zeros([0], dtype=torch.int, device="cuda")

            if (~self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:
                grid_coords_ds = torch.round((self.get_anchor[level_mask] - self.init_pos) / ds_size).int()
                selected_xyz_ds = all_xyz.view([-1, 3])[candidate_ds_mask]
                selected_grid_coords_ds = torch.round((selected_xyz_ds - self.init_pos) / ds_size).int()
                selected_grid_coords_unique_ds, _ = torch.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
                if selected_grid_coords_unique_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds  # pylint: disable=E1130
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds] * ds_size + self.init_pos
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device="cuda") * (
                        cur_level + 1
                    )
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(
                        candidate_anchor_ds, new_level_ds
                    )
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                else:
                    candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device="cuda")
                    remove_duplicates_ds = torch.ones([0], dtype=torch.bool, device="cuda")
                    new_level_ds = torch.zeros([0], dtype=torch.int, device="cuda")
            else:
                candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device="cuda")
                remove_duplicates_ds = torch.ones([0], dtype=torch.bool, device="cuda")
                new_level_ds = torch.zeros([0], dtype=torch.int, device="cuda")

            if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:

                new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)
                new_level = torch.cat([new_level, new_level_ds]).unsqueeze(dim=1).float().cuda()

                new_feat = (
                    self._anchor_feat.unsqueeze(dim=1)
                    .repeat([1, self.n_offsets, 1])
                    .view([-1, self.feat_dim])[candidate_mask]
                )
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][
                    remove_duplicates
                ]
                new_feat_ds = torch.zeros(
                    [candidate_anchor_ds.shape[0], self.feat_dim], dtype=torch.float, device="cuda"
                )
                new_feat = torch.cat([new_feat, new_feat_ds], dim=0)

                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size  # *0.05
                new_scaling_ds = torch.ones_like(candidate_anchor_ds).repeat([1, 2]).float().cuda() * ds_size  # *0.05
                new_scaling = torch.cat([new_scaling, new_scaling_ds], dim=0)
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], dtype=torch.float, device="cuda")
                new_rotation_ds = torch.zeros([candidate_anchor_ds.shape[0], 4], dtype=torch.float, device="cuda")
                new_rotation = torch.cat([new_rotation, new_rotation_ds], dim=0)
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(
                    0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda")
                )
                new_opacities_ds = inverse_sigmoid(
                    0.1 * torch.ones((candidate_anchor_ds.shape[0], 1), dtype=torch.float, device="cuda")
                )
                new_opacities = torch.cat([new_opacities, new_opacities_ds], dim=0)

                new_offsets = (
                    torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                )
                new_offsets_ds = (
                    torch.zeros_like(candidate_anchor_ds).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                )
                new_offsets = torch.cat([new_offsets, new_offsets_ds], dim=0)

                new_extra_level = torch.zeros(candidate_anchor.shape[0], dtype=torch.float, device="cuda")
                new_extra_level_ds = torch.zeros(candidate_anchor_ds.shape[0], dtype=torch.float, device="cuda")
                new_extra_level = torch.cat([new_extra_level, new_extra_level_ds])

                d = {
                    "anchor": new_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat(
                    [self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device="cuda").float()], dim=0
                )
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat(
                    [self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device="cuda").float()], dim=0
                )
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d, optimizer)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                self._level.data = torch.cat([self._level, new_level], dim=0)
                self._extra_level.data = torch.cat([self._extra_level, new_extra_level], dim=0)

    def adjust_anchor(
        self,
        iteration,
        check_interval=100,
        success_threshold=0.8,
        grad_threshold=0.0002,
        update_ratio=0.5,
        extra_ratio=4.0,
        extra_up=0.25,
        min_opacity=0.005,
        optimizer=None,
    ):
        """
        Adjusts the anchor points based on the given parameters.

        Args:
            iteration (int): The current iteration number.
            check_interval (int, optional): The interval at which to check the success threshold. Defaults to 100.
            success_threshold (float, optional): The threshold for determining the success of the adjustment.
            Defaults to 0.8.
            grad_threshold (float, optional): The threshold for the gradient. Defaults to 0.0002.
            update_ratio (float, optional): The ratio for updating the anchor points. Defaults to 0.5.
            extra_ratio (float, optional): The extra ratio for updating the anchor points. Defaults to 4.0.
            extra_up (float, optional): The extra up value for updating the anchor points. Defaults to 0.25.
            min_opacity (float, optional): The minimum opacity value. Defaults to 0.005.
            optimizer (object, optional): The optimizer object to use for optimization. Defaults to None.
        """
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom  # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)

        self.anchor_growing(
            iteration, grads_norm, grad_threshold, update_ratio, extra_ratio, extra_up, offset_mask, optimizer
        )

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros(
            [self.get_anchor.shape[0] * self.n_offsets - self.offset_denom.shape[0], 1],
            dtype=torch.int32,
            device=self.offset_denom.device,
        )
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros(
            [self.get_anchor.shape[0] * self.n_offsets - self.offset_gradient_accum.shape[0], 1],
            dtype=torch.int32,
            device=self.offset_gradient_accum.device,
        )
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity * self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval * success_threshold).squeeze(dim=1)  # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum() > 0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device="cuda").float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0] > 0:
            self.prune_anchor(prune_mask, optimizer)

    @property
    def get_scaling(self):
        """
        Returns the scaling value for the octreegs encoding.

        Returns:
            float: The scaling value.
        """
        return 1.0 * self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        """
        Returns the rotation of the encoding.

        Returns:
            The rotation of the encoding.
        """
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        """
        Returns the anchor of the octree grid structure.

        Returns:
            The anchor of the octree grid structure.
        """
        return self._anchor

    @property
    def get_offset(self):
        """
        Returns the offset value.

        Returns:
            The offset value.
        """
        return self._offset

    @property
    def get_level(self):
        """
        Get the level of the octree encoding.

        Returns:
            int: The level of the octree encoding.
        """
        return self._level

    @property
    def get_extra_level(self):
        """
        Returns the extra level of the octree encoding.

        Returns:
            int: The extra level of the octree encoding.
        """
        return self._extra_level

    @property
    def get_opacity(self):
        """
        Get the opacity value of the encoding.

        Returns:
            float: The opacity value.
        """
        return self.opacity_activation(self._opacity)

    @property
    def get_anchor_feat(self):
        """
        Returns the anchor feature.

        Returns:
            The anchor feature.
        """
        return self._anchor_feat

    def get_covariance(self, scaling_modifier=1):
        """
        Returns the covariance of the encoding.

        Parameters:
        scaling_modifier (float): A scaling factor to modify the covariance.

        Returns:
        numpy.ndarray: The covariance of the encoding.
        """
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def set_coarse_interval(self, coarse_iter, coarse_factor):
        """
        Sets the coarse intervals for the encoding.

        Args:
            coarse_iter (int): The number of coarse iterations.
            coarse_factor (float): The coarse factor.

        Returns:
            None
        """
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1 / coarse_factor
            a1 = coarse_iter * (1 - q) / (1 - q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q**i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_level(self, points, cameras, scales, dist_ratio=0.95, init_level=-1, levels=-1):
        """
        Sets the level of the octree encoding based on the given points, cameras, and scales.

        Args:
            points (torch.Tensor): Tensor of shape (N, 3) representing the 3D points.
            cameras (list): List of camera objects.
            scales (list): List of scales.
            dist_ratio (float, optional): Distance ratio used to determine the maximum and
            minimum distances. Defaults to 0.95.
            init_level (int, optional): Initial level of the octree encoding. Defaults to -1.
            levels (int, optional): Number of levels in the octree encoding. Defaults to -1.

        Returns:
            None
        """
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras:
                if cam.resolution_scale == scale:
                    cam_center = cam.camera_center.float().cuda()
                    cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                    self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                    dist = torch.sqrt(torch.sum((points - cam_center) ** 2, dim=1))
                    dist_max = torch.quantile(dist, dist_ratio)
                    dist_min = torch.quantile(dist, 1 - dist_ratio)
                    new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                    new_dist = new_dist * scale
                    all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - dist_ratio)
        self.standard_dist = dist_max
        if levels == -1:
            self.levels = torch.round(torch.log2(dist_max / dist_min) / math.log2(self.fork)).int().item() + 1
        else:
            self.levels = levels
        if init_level == -1:
            self.init_level = int(self.levels / 2)
        else:
            self.init_level = init_level

    def plot_levels(self):
        """
        Plot the number of points in each level of the octree encoding.

        This method iterates over each level of the octree encoding and prints the number of points
        in each level along with the ratio of points in that level to the total number of points.

        Returns:
            None
        """
        for level in range(self.levels):
            level_mask = (self._level.data == level).squeeze(dim=1)
            print(
                f"Level {level}: {torch.sum(level_mask).item()}, Ratio:"
                f" {torch.sum(level_mask).item()/self._level.shape[0]}"
            )
