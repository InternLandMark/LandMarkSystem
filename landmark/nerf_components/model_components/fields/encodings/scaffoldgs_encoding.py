import os
from functools import reduce
from typing import Any, Dict, Mapping, Union

import numpy as np
import torch
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
    scaling (torch.Tensor): The scaling factor.
    scaling_modifier (float): The scaling modifier.
    rotation (torch.Tensor): The rotation matrix.

    Returns:
    torch.Tensor: The covariance matrix.

    """
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


class ScaffoldGSEncoding(BaseGaussianEncoding):
    """Class for 3DGS encoding"""

    def __init__(
        self,
        percent_dense,
        feat_dim: int = 32,
        n_offsets: int = 5,
        voxel_size: float = 0.01,
        update_depth: int = 3,
        update_init_factor: int = 100,
        update_hierachy_factor: int = 4,
        use_feat_bank=False,
    ):
        """
        Initializes a ScaffoldGSEncoding object.

        Args:
            percent_dense (float): The percentage of dense points.
            feat_dim (int, optional): The dimension of the feature vector. Defaults to 32.
            n_offsets (int, optional): The number of offsets. Defaults to 5.
            voxel_size (float, optional): The size of the voxel. Defaults to 0.01.
            update_depth (int, optional): The depth of the update. Defaults to 3.
            update_init_factor (int, optional): The initial factor for update. Defaults to 100.
            update_hierachy_factor (int, optional): The hierarchy factor for update. Defaults to 4.
            use_feat_bank (bool, optional): Whether to use feature bank. Defaults to False.
        """
        super().__init__()

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.percent_dense = percent_dense

        # Initialize empty tensors
        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self.opacity_accum = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.anchor_demon = torch.empty(0)

        # Define activation functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # Define merge configuration
        self.merge_config = {
            "merge_type": MergeType.Unevenly,
            "_anchor_feat": 0,
            "_offset": 0,
            "_anchor": 0,
            "_opacity": 0,
            "_scaling": 0,
            "_rotation": 0,
        }

    def create_from_pcd(self, pcd: BasicPointCloud):
        """
        Creates a scaffoldgs encoding from a given point cloud.

        Args:
            pcd (BasicPointCloud): The input point cloud.

        Returns:
            None
        """
        from simple_knn._C import distCUDA2
        points = pcd.points

        if self.voxel_size <= 0:
            init_points = torch.tensor(points, device="cuda").float()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0] * 0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f"Initial voxel_size: {self.voxel_size}")

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points), device="cuda").float()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3), device="cuda").float()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim), device="cuda").float()

        print("Number of points at initialization : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.cuda()).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

    def get_state_dict_from_ckpt(
        self,
        file_path: Union[str, os.PathLike],
        map_location: Union[str, Dict[str, str]],
        prefix: str = "",
    ) -> Mapping[str, Any]:
        """
        Reads a checkpoint file and returns a state dictionary containing the encoded data.

        Args:
            file_path (Union[str, os.PathLike]): The path to the checkpoint file.
            map_location (Union[str, Dict[str, str]]): The device location to store the tensors.
            prefix (str, optional): A prefix to add to the keys of the state dictionary. Defaults to "".

        Returns:
            Mapping[str, Any]: A dictionary containing the encoded data with the specified prefix.

        Raises:
            FileNotFoundError: If the specified file_path does not exist.
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
        _offset = torch.tensor(offsets, dtype=torch.float, device=map_location).transpose(1, 2)
        _anchor = torch.tensor(anchor, dtype=torch.float, device=map_location)
        _opacity = torch.tensor(opacities, dtype=torch.float, device=map_location)
        _scaling = torch.tensor(scales, dtype=torch.float, device=map_location)
        _rotation = torch.tensor(rots, dtype=torch.float, device=map_location)
        return {
            prefix + "_anchor_feat": _anchor_feat,
            prefix + "_offset": _offset,
            prefix + "_anchor": _anchor,
            prefix + "_opacity": _opacity,
            prefix + "_scaling": _scaling,
            prefix + "_rotation": _rotation,
        }

    def load_from_state_dict(self, state_dict, strict: bool = False):
        """
        Loads the model's parameters from a state dictionary.

        Args:
            state_dict (dict): A dictionary containing the model's state.
            strict (bool, optional): Whether to strictly enforce that all keys in the state dictionary match the
            model's parameters.
                                        Defaults to False.

        Raises:
            ValueError: If strict is True and not all keys in the state dictionary match the model's parameters.

        """
        params_key = [
            "_anchor_feat",
            "_offset",
            "_anchor",
            "_opacity",
            "_scaling",
            "_rotation",
        ]
        duplicate_keys = set(state_dict.keys()) & set(params_key)
        if strict is True and len(duplicate_keys) != len(params_key):
            raise ValueError(f"only found keys {duplicate_keys} in state dict")
        for param in duplicate_keys:
            setattr(
                self,
                param,
                nn.Parameter(torch.zeros(state_dict[param].shape, device=state_dict[param].device)),
            )
        super().load_state_dict(state_dict, strict)
        if "_anchor" in duplicate_keys:
            self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device=self.get_anchor.device)
            self.offset_gradient_accum = torch.zeros(
                (self.get_anchor.shape[0] * self.n_offsets, 1),
                device=self.get_anchor.device,
            )
            self.offset_denom = torch.zeros(
                (self.get_anchor.shape[0] * self.n_offsets, 1),
                device=self.get_anchor.device,
            )
            self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device=self.get_anchor.device)

    def save_ply(self, path):
        """
        Save the encoding as a PLY file.

        Args:
            path (str): The path to save the PLY file.

        Returns:
            None
        """
        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        """
        Constructs a list of attributes for the encoding.

        Returns:
            attr_list (list): A list of attributes including x, y, z, nx, ny, nz, f_offset_i, f_anchor_feat_i,
            opacity, scale_i, and rot_i.
        """
        attr_list = ["x", "y", "z", "nx", "ny", "nz"]
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
            AssertionError: If the length of the optimizer's parameter group is not equal to 1.
        """
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if "mlp" in group["name"] or "conv" in group["name"] or "feat_base" in group["name"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
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

    def training_statis(
        self,
        viewspace_point_tensor,
        opacity,
        update_filter,
        offset_selection_mask,
        anchor_visible_mask,
    ):
        """
        Update the statistics used for training the model.

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
            mask (torch.Tensor): The mask to be applied to the optimizer's parameters.
            optimizer (torch.optim.Optimizer): The optimizer to be pruned.

        Returns:
            dict: A dictionary containing the pruned optimizable tensors, where the keys are the names of the
            parameter groups and the values are the pruned parameters.
        """
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if "mlp" in group["name"] or "conv" in group["name"] or "feat_base" in group["name"]:
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
            mask (bool tensor): A boolean mask indicating which points to prune.
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

    def anchor_growing(self, grads, threshold, offset_mask, optimizer):
        """
        Performs anchor growing based on the given gradients, threshold, offset mask, and optimizer.

        Args:
            grads (torch.Tensor): The gradients used for anchor growing.
            threshold (float): The threshold value for selecting candidate anchors.
            offset_mask (torch.Tensor): The offset mask used for filtering candidate anchors.
            optimizer: The optimizer used for updating the anchor parameters.

        Returns:
            None
        """
        init_length = self.get_anchor.shape[0] * self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold * ((self.update_hierachy_factor // 2) ** i)
            # mask from grad threshold
            candidate_mask = grads >= cur_threshold
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            # random pick
            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5 ** (i + 1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0] * self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat(
                    [
                        candidate_mask,
                        torch.zeros(length_inc, dtype=torch.bool, device="cuda"),
                    ],
                    dim=0,
                )

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size * size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(
                selected_grid_coords, return_inverse=True, dim=0
            )

            # split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (
                        (
                            selected_grid_coords_unique.unsqueeze(1)
                            == grid_coords[i * chunk_size : (i + 1) * chunk_size, :]
                        )
                        .all(-1)
                        .any(-1)
                        .view(-1)
                    )
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates  # pylint: disable=E1130
            candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size  # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(
                    0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda")
                )

                new_feat = (
                    self._anchor_feat.unsqueeze(dim=1)
                    .repeat([1, self.n_offsets, 1])
                    .view([-1, self.feat_dim])[candidate_mask]
                )

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0,)[
                    0
                ][remove_duplicates]

                new_offsets = (
                    torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                )

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat(
                    [
                        self.anchor_demon,
                        torch.zeros([new_opacities.shape[0], 1], device="cuda").float(),
                    ],
                    dim=0,
                )
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat(
                    [
                        self.opacity_accum,
                        torch.zeros([new_opacities.shape[0], 1], device="cuda").float(),
                    ],
                    dim=0,
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

    def adjust_anchor(
        self,
        check_interval=100,
        success_threshold=0.8,
        grad_threshold=0.0002,
        min_opacity=0.005,
        optimizer=None,
    ):
        """
        Adjusts the anchors based on certain criteria.

        Args:
            check_interval (int): The interval at which to check the success threshold.
            success_threshold (float): The threshold for determining the success of an anchor.
            grad_threshold (float): The threshold for the gradient.
            min_opacity (float): The minimum opacity value.
            optimizer: The optimizer to use for updating the anchor parameters.

        Returns:
            None
        """
        # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom  # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask, optimizer)

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
            [
                self.get_anchor.shape[0] * self.n_offsets - self.offset_gradient_accum.shape[0],
                1,
            ],
            dtype=torch.int32,
            device=self.offset_gradient_accum.device,
        )
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # prune anchors
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

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    @property
    def get_scaling(self):
        """
        Returns the scaling factor for the encoding.

        Returns:
            float: The scaling factor.
        """
        return 1.0 * self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        """
        Returns the rotation value of the encoding.

        Returns:
            float: The rotation value.
        """
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        """
        Returns the anchor value.

        Returns:
            The anchor value.
        """
        return self._anchor

    @property
    def get_anchor_feat(self):
        """
        Returns the anchor feature.

        Returns:
            The anchor feature.
        """
        return self._anchor_feat

    @property
    def get_offset(self):
        """
        Returns the offset value.

        Returns:
            The offset value.
        """
        return self._offset

    def set_anchor(self, new_anchor):
        """
        Sets the anchor for the scaffoldgs encoding.

        Args:
            new_anchor: The new anchor tensor.

        Raises:
            AssertionError: If the shape of the new anchor tensor does not match the shape of the
            existing anchor tensor.
        """
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        """
        Returns the opacity value of the encoding.

        Returns:
            float: The opacity value.
        """
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        """
        Returns the covariance matrix of the encoding.

        Parameters:
        scaling_modifier (float): A scaling factor to modify the scaling of the covariance matrix. Default is 1.

        Returns:
        numpy.ndarray: The covariance matrix of the encoding.
        """
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        """
        Voxelize the given sample data.

        Args:
            data (numpy.ndarray): The input sample data.
            voxel_size (float): The size of each voxel.

        Returns:
            numpy.ndarray: The voxelized sample data.
        """
        np.random.shuffle(data)
        data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size

        return data
