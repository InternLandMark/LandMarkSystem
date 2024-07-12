import os
from typing import Any, Dict, Mapping, Union

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from landmark.nerf_components.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    inverse_sigmoid,
    strip_symmetric,
)
from landmark.nerf_components.utils.graphics_utils import BasicPointCloud
from landmark.nerf_components.utils.sh_utils import RGB2SH
from landmark.render.util.types import MergeType

from .base_encoding import BaseGaussianEncoding


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    """
    Builds a covariance matrix from scaling and rotation matrices.

    Args:
        scaling (torch.Tensor): The scaling matrix.
        scaling_modifier (float): The scaling modifier.
        rotation (torch.Tensor): The rotation matrix.

    Returns:
        torch.Tensor: The covariance matrix.

    """
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


class GaussianEncoding(BaseGaussianEncoding):
    """Class for 3DGS encoding"""

    def __init__(self, sh_degree, percent_dense, act_cache):
        """
        Initialize the GaussianEncoding object.

        Args:
            sh_degree (int): The maximum spherical harmonics degree.
            percent_dense (float): The percentage of dense points.
            act_cache (bool): Flag indicating whether to cache activations.

        Attributes:
            active_sh_degree (int): The currently active spherical harmonics degree.
            max_sh_degree (int): The maximum spherical harmonics degree.
            percent_dense (float): The percentage of dense points.
            act_cache (bool): Flag indicating whether to cache activations.
            _xyz (torch.Tensor): Empty tensor for storing XYZ coordinates.
            _features_dc (torch.Tensor): Empty tensor for storing DC features.
            _features_rest (torch.Tensor): Empty tensor for storing rest features.
            _scaling (torch.Tensor): Empty tensor for storing scaling values.
            _rotation (torch.Tensor): Empty tensor for storing rotation values.
            _opacity (torch.Tensor): Empty tensor for storing opacity values.
            scaling_activation (function): Activation function for scaling values.
            scaling_inverse_activation (function): Inverse activation function for scaling values.
            covariance_activation (function): Activation function for computing covariance from scaling and rotation.
            opacity_activation (function): Activation function for opacity values.
            inverse_opacity_activation (function): Inverse activation function for opacity values.
            rotation_activation (function): Activation function for normalizing rotation values.
            merge_config (dict): Configuration for merging different components.
            _cache_scaling (None or torch.Tensor): Cached scaling values.
            _cache_features (None or torch.Tensor): Cached features.
            _cache_rotation (None or torch.Tensor): Cached rotation values.
            _cache_opacity (None or torch.Tensor): Cached opacity values.
        """
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.percent_dense = percent_dense
        self.act_cache = act_cache

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.merge_config = {
            "merge_type": MergeType.Unevenly,
            "_xyz": 0,
            "_opacity": 0,
            "_features_dc": 0,
            "_features_rest": 0,
            "_scaling": 0,
            "_rotation": 0,
        }

        self._cache_scaling = None
        self._cache_features = None
        self._cache_rotation = None
        self._cache_opacity = None

    def create_from_pcd(self, pcd: BasicPointCloud):
        """
        Creates a Gaussian encoding from a given point cloud.

        Args:
            pcd (BasicPointCloud): The input point cloud.

        Returns:
            None
        """
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def get_state_dict_from_ckpt(
        self,
        file_path: Union[str, os.PathLike],
        map_location: Union[str, Dict[str, str]],
        prefix: str = "",
    ) -> Mapping[str, Any]:
        """
        Reads a checkpoint file and returns the state dictionary.

        Args:
            file_path (Union[str, os.PathLike]): The path to the checkpoint file.
            map_location (Union[str, Dict[str, str]]): The device location to load the tensors onto.
            prefix (str, optional): The prefix to add to the state dictionary keys. Defaults to "".

        Returns:
            Mapping[str, Any]: The state dictionary containing the loaded tensors.

        Raises:
            NotImplementedError: If the checkpoint file contains unsupported keys.
        """
        plydata = PlyData.read(file_path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        if "opacity" in plydata.elements[0]:
            opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

            scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
            scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

            rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
            rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
            rots = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

            _xyz = torch.tensor(xyz, dtype=torch.float, device=map_location)
            _features_dc = (
                torch.tensor(features_dc, dtype=torch.float, device=map_location).transpose(1, 2).contiguous()
            )  # .transpose(1, 2)
            _features_rest = (
                torch.tensor(features_extra, dtype=torch.float, device=map_location).transpose(1, 2).contiguous()
            )  # .transpose(1, 2)
            _opacity = torch.tensor(opacities, dtype=torch.float, device=map_location)
            _scaling = torch.tensor(scales, dtype=torch.float, device=map_location)
            _rotation = torch.tensor(rots, dtype=torch.float, device=map_location)
            state_dict = {
                prefix + "_xyz": _xyz,
                prefix + "_features_dc": _features_dc,
                prefix + "_features_rest": _features_rest,
                prefix + "_opacity": _opacity,
                prefix + "_scaling": _scaling,
                prefix + "_rotation": _rotation,
            }
        elif "cache_opacity" in plydata.elements[0]:
            cache_opacities = np.asarray(plydata.elements[0]["cache_opacity"])[..., np.newaxis]

            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("cache_f_")]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2))

            scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("cache_scale_")]
            scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
            cache_scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                cache_scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

            rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("cache_rot_")]
            rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
            cache_rots = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                cache_rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

            _xyz = torch.tensor(xyz, dtype=torch.float, device=map_location)
            _cache_features = (
                torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
            )
            _cache_opacity = torch.tensor(cache_opacities, dtype=torch.float, device="cuda")
            _cache_scaling = torch.tensor(cache_scales, dtype=torch.float, device="cuda")
            _cache_rotation = torch.tensor(cache_rots, dtype=torch.float, device="cuda")
            state_dict = {
                prefix + "_xyz": _xyz,
                prefix + "_cache_features": _cache_features,
                prefix + "_cache_opacity": _cache_opacity,
                prefix + "_cache_scaling": _cache_scaling,
                prefix + "_cache_rotation": _cache_rotation,
            }
        else:
            raise NotImplementedError(f"unsupport keys {plydata.elements[0].keys()}")
        return state_dict

    def load_from_state_dict(self, state_dict, strict: bool = False):
        """
        Loads the model's parameters from a state dictionary.

        Args:
            state_dict (dict): A dictionary containing the model's parameters.
            strict (bool, optional): Whether to strictly enforce that all keys in the state dictionary
            match the expected parameter keys. Defaults to False.

        Raises:
            ValueError: If strict is True and some keys in the state dictionary do not match the expected
            parameter keys.

        """
        params_key = [
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
        ]
        cache_params_key = [
            "_xyz",
            "_cache_features",
            "_cache_opacity",
            "_cache_scaling",
            "_cache_rotation",
        ]
        duplicate_keys = set(state_dict.keys()) & set(params_key)
        cache_duplicate_keys = set(state_dict.keys()) & set(cache_params_key)
        if len(duplicate_keys) > 0:
            if strict is True and len(duplicate_keys) != len(params_key):
                raise ValueError(f"only found keys {duplicate_keys} in state dict")
            for param in duplicate_keys:
                setattr(
                    self,
                    param,
                    nn.Parameter(torch.zeros(state_dict[param].shape, device=state_dict[param].device)),
                )
            super().load_state_dict(state_dict, strict)
            self.active_sh_degree = self.max_sh_degree
            if "_xyz" in duplicate_keys:
                self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.get_xyz.device)
                self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.get_xyz.device)
            self._cache_scaling = None
            self._cache_features = None
            self._cache_rotation = None
            self._cache_opacity = None
        elif len(cache_duplicate_keys) > 0:
            if strict is True and len(cache_duplicate_keys) != len(params_key):
                raise ValueError(f"only found keys {cache_duplicate_keys} in state dict")
            for param in cache_duplicate_keys:
                setattr(
                    self,
                    param,
                    nn.Parameter(torch.zeros(state_dict[param].shape, device=state_dict[param].device)),
                )
            super().load_state_dict(state_dict, strict)
            self.active_sh_degree = self.max_sh_degree
            if "_xyz" in cache_duplicate_keys:
                self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.get_xyz.device)
                self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.get_xyz.device)
        else:
            if strict:
                raise ValueError(f"not found any param keys in state dict, state dict keys: {state_dict.keys()}")

    def save_ply(self, path):
        """
        Save the encoded data as a PLY file.

        Args:
            path (str): The path to save the PLY file.

        Raises:
            None

        Returns:
            None
        """
        if self.act_cache is True and self._cache_features is not None:
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_all = self._cache_features.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            act_opacities = self._cache_opacity.detach().cpu().numpy()
            act_scale = self._cache_scaling.detach().cpu().numpy()
            act_rotation = self._cache_rotation.detach().cpu().numpy()

            dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_cache_attributes()]

            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_all, act_opacities, act_scale, act_rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, "vertex")
            PlyData([el]).write(path)
        else:
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

            dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, "vertex")
            PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        """
        Constructs a list of attributes for the encoding.

        Returns:
            attr_list (list): A list of attributes including x, y, z, nx, ny, nz, f_dc_i,
            f_rest_i, opacity, scale_i, and rot_i.
        """
        attr_list = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attr_list.append(f"f_dc_{i}")
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attr_list.append(f"f_rest_{i}")
        attr_list.append("opacity")
        for i in range(self._scaling.shape[1]):
            attr_list.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            attr_list.append(f"rot_{i}")
        return attr_list

    def construct_list_of_cache_attributes(self):
        """
        Constructs a list of cache attributes based on the available cache features, opacity, scaling, and rotation.

        Returns:
            attr_list (list): A list of cache attributes including x, y, z, nx, ny, nz, cache_f_i, cache_opacity,
            cache_scale_i, and cache_rot_i.
        """
        assert self._cache_features is not None
        assert self._cache_opacity is not None
        assert self._cache_scaling is not None
        assert self._cache_rotation is not None
        attr_list = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._cache_features.shape[1] * self._cache_features.shape[2]):
            attr_list.append(f"cache_f_{i}")
        attr_list.append("cache_opacity")
        for i in range(self._cache_scaling.shape[1]):
            attr_list.append(f"cache_scale_{i}")
        for i in range(self._cache_rotation.shape[1]):
            attr_list.append(f"cache_rot_{i}")
        return attr_list

    def cat_tensors_to_optimizer(self, tensors_dict, optimizer):
        """
        Concatenates extension tensors to the optimizer's parameter groups.

        Args:
            tensors_dict (dict): A dictionary containing extension tensors for each parameter group.
            optimizer (torch.optim.Optimizer): The optimizer object.

        Returns:
            dict: A dictionary containing the optimizable tensors after concatenation.

        Raises:
            AssertionError: If the number of parameters in a group is not equal to 1.
        """
        optimizable_tensors = {}
        for group in optimizer.param_groups:
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

    def replace_tensor_to_optimizer(self, tensor, name, optimizer):
        """
        Replaces a tensor in the optimizer with a new tensor.

        Args:
            tensor (torch.Tensor): The new tensor to replace the existing tensor.
            name (str): The name of the optimizer group to modify.
            optimizer (torch.optim.Optimizer): The optimizer object.

        Returns:
            dict: A dictionary containing the optimizable tensors, where the keys are the group names and the
            values are the modified tensors.

        """
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"] == name:
                stored_state = optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        optimizer,
    ):
        """
        Perform densification postfix operation.

        Args:
            new_xyz (torch.Tensor): New xyz coordinates.
            new_features_dc (torch.Tensor): New DC features.
            new_features_rest (torch.Tensor): New rest features.
            new_opacities (torch.Tensor): New opacities.
            new_scaling (torch.Tensor): New scaling values.
            new_rotation (torch.Tensor): New rotation values.
            optimizer: Optimizer object.

        Returns:
            None
        """
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d, optimizer)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _prune_optimizer(self, mask, optimizer):
        """
        Prunes the optimizer by removing the parameters and state corresponding to the given mask.

        Args:
            mask (torch.Tensor): A boolean mask indicating which parameters to keep.
            optimizer (torch.optim.Optimizer): The optimizer to be pruned.

        Returns:
            dict: A dictionary mapping the names of the optimizable tensors to their pruned versions.

        """
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, optimizer):
        """
        Prunes the points based on the given mask and updates the relevant attributes.

        Args:
            mask (bool tensor): A boolean mask indicating which points to prune.
            optimizer: The optimizer used for pruning.

        Returns:
            None
        """
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, optimizer)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densify_and_split(self, grads, grad_threshold, scene_extent, optimizer, N=2):
        """
        Densifies and splits the points based on gradient condition.

        Args:
            grads (torch.Tensor): The gradients of the points.
            grad_threshold (float): The threshold for the gradient condition.
            scene_extent (float): The extent of the scene.
            optimizer: The optimizer used for optimization.
            N (int, optional): The number of times to repeat the points. Defaults to 2.
        """
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            optimizer,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter, optimizer)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, optimizer):
        """
        Densifies and clones the encoding based on the gradient condition.

        Args:
            grads (torch.Tensor): The gradients.
            grad_threshold (float): The threshold for the gradient condition.
            scene_extent (float): The extent of the scene.
            optimizer: The optimizer.

        Returns:
            None
        """
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            optimizer,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, optimizer):
        """
        Densifies and prunes the encoded points based on specified criteria.

        Args:
            max_grad (float): The maximum gradient value.
            min_opacity (float): The minimum opacity value.
            extent (float): The extent value.
            max_screen_size (float): The maximum screen size value.
            optimizer: The optimizer used for pruning.

        Returns:
            None
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, optimizer)
        self.densify_and_split(grads, max_grad, extent, optimizer)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask, optimizer)

        torch.cuda.empty_cache()

    def reset_opacity(self, optimizer):
        """
        Resets the opacity values of the encoding.

        Args:
            optimizer: The optimizer used for updating the opacity values.

        Returns:
            None
        """
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity", optimizer)
        self._opacity = optimizable_tensors["opacity"]

    def add_densification_stats(self, viewspace_point, update_filter):
        """
        Adds densification statistics for a given viewspace point and update filter.

        Args:
            viewspace_point (torch.Tensor): The viewspace point.
            update_filter (torch.Tensor): The update filter.

        Returns:
            None
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    @property
    def get_scaling(self):
        """
        Returns the scaling value for the encoding.

        If the activation cache is available and the model is not in training mode,
        the cached scaling value is returned. Otherwise, the scaling value is computed
        using the scaling_activation function.

        Returns:
            The scaling value for the encoding.
        """
        if self.act_cache and not self.training:
            if self._cache_scaling is not None:
                return self._cache_scaling
            else:
                self._cache_scaling = self.scaling_activation(self._scaling)
                del self._scaling
                return self._cache_scaling
        else:
            return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        """
        Returns the rotation value.

        If the `act_cache` is not empty and the model is not in training mode,
        it checks if the cached rotation value exists. If it does, it returns
        the cached value. Otherwise, it calculates the rotation activation value
        using the `_rotation` attribute, caches it, and returns the cached value.

        If the `act_cache` is empty or the model is in training mode, it directly
        calculates the rotation activation value using the `_rotation` attribute
        and returns it.

        Returns:
            The rotation activation value.
        """
        if self.act_cache and not self.training:
            if self._cache_rotation is not None:
                return self._cache_rotation
            else:
                self._cache_rotation = self.rotation_activation(self._rotation)
                del self._rotation
                return self._cache_rotation
        else:
            return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        """
        Returns the XYZ coordinates of the encoding.

        Returns:
            Tuple[float, float, float]: The XYZ coordinates of the encoding.
        """
        return self._xyz

    @property
    def get_features(self):
        """
        Returns the encoded features.

        If the `act_cache` is enabled and the model is not in training mode, it checks if the cached features
        are available.
        If the cached features are available, it returns them. Otherwise, it concatenates the `_features_dc`
        and `_features_rest`
        tensors along the second dimension, caches the result, and returns it.

        If the `act_cache` is disabled or the model is in training mode, it simply concatenates the `_features_dc`
        and `_features_rest`
        tensors along the second dimension and returns the result.

        Returns:
            torch.Tensor: The encoded features tensor.
        """
        if self.act_cache and not self.training:
            if self._cache_features is not None:
                return self._cache_features
            else:
                self._cache_features = torch.cat((self._features_dc, self._features_rest), dim=1)
                del self._features_dc
                del self._features_rest
                return self._cache_features
        else:
            return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        """
        Returns the opacity value for the encoding.

        If the activation cache is available and the model is not in training mode,
        the cached opacity value is returned. Otherwise, the opacity value is computed
        using the opacity_activation function.

        Returns:
            float: The opacity value for the encoding.
        """
        if self.act_cache and not self.training:
            if self._cache_opacity is not None:
                return self._cache_opacity
            else:
                self._cache_opacity = self.opacity_activation(self._opacity)
                del self._opacity
                return self._cache_opacity
        else:
            return self.opacity_activation(self._opacity)

    def set_scaling(self, value):
        """
        Set the scaling value for the encoding.

        Parameters:
        value (float): The scaling value to be set.

        Returns:
        None
        """
        if self.act_cache and not self.training:
            self._cache_scaling = value
        else:
            self._scaling = value

    def set_rotation(self, value):
        """
        Sets the rotation value for the encoding.

        If the `act_cache` is enabled and the model is not in training mode,
        the rotation value is cached instead of directly setting it.

        Args:
            value: The rotation value to be set.
        """
        if self.act_cache and not self.training:
            self._cache_rotation = value
        else:
            self._rotation = value

    def set_xyz(self, value):
        """
        Sets the value of the xyz attribute.

        Args:
            value: The new value for the xyz attribute.
        """
        if isinstance(self._xyz, nn.Parameter):
            self._xyz.data = value
        else:
            self._xyz = value

    def set_features(self, value):
        """
        Sets the features of the encoding.

        Args:
            value: The value to set as the features.

        Raises:
            NotImplementedError: If the `act_cache` is enabled and the model is not in training mode.
        """
        if self.act_cache and not self.training:
            self._cache_features = value
        else:
            raise NotImplementedError("not support yet")

    def set_opacity(self, value):
        """
        Set the opacity value for the encoding.

        Parameters:
        value (float): The opacity value to be set.

        Returns:
        None
        """
        if self.act_cache and not self.training:
            self._cache_opacity = value
        else:
            self._opacity = value

    def get_covariance(self, scaling_modifier=1):
        """
        Returns the covariance matrix of the Gaussian encoding.

        Parameters:
            scaling_modifier (float): A scaling factor applied to the covariance matrix. Default is 1.

        Returns:
            numpy.ndarray: The covariance matrix of the Gaussian encoding.
        """
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        """
        Increases the active spherical harmonics (SH) degree by one if it is less than the maximum SH degree.

        Returns:
            None
        """
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
