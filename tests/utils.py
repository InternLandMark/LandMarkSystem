from typing import Any, Mapping

import torch
from kornia import create_meshgrid
from torch.nn import Module

from benchmarks.nerf.gridnerf.gridnerf import GridNeRF
from benchmarks.nerf.instant_ngp.instant_ngp import InstantNGP
from benchmarks.nerf.nerfacto.nerfacto import Nerfacto
from benchmarks.nerf.octree_gs.gs_model import OctreeGS
from landmark import InferenceModule
from landmark.nerf_components.model import BaseNeRF


class SimpleModel(BaseNeRF):
    """simple demo model"""

    def __init__(self, N=10):
        super().__init__()
        self.layer1 = torch.nn.Linear(N, N)
        self.layer2 = torch.nn.ReLU()

    @staticmethod
    def get_kwargs() -> Mapping[str, Any]:
        return {"N": 10}

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class InferenceNerfModule(InferenceModule):
    """
    Nerf inference module
    """

    def __init__(self, model: Module):
        self.render_1080p = True
        self.H, self.W = None, None
        super().__init__(model=model)

    def preprocess(self, pose, chunk_size, H, W, app_code, edit_mode):
        assert edit_mode is not None
        # self.model.edit_model(edit_mode)

        if self.render_1080p:
            self.H, self.W = 1080, 1920
        else:
            self.H, self.W = H, W

        focal = 2317.6449482429634 if self.render_1080p else pose[-1, -1]
        rays = self.generate_rays(pose, H, W, focal)
        args = (rays, chunk_size, app_code)
        kwargs = {}
        return args, kwargs

    def forward(self, rays, chunk_size, app_code):
        N_samples = -1
        idxs = torch.zeros_like(rays[:, 0], dtype=torch.long, device=rays.device)  # TODO need check
        all_ret = self.model.render_all_rays(rays, chunk_size, N_samples, idxs, app_code)
        return all_ret["rgb_map"]

    def postprocess(self, result):
        result = result.clamp(0.0, 1.0)
        result = result.reshape(self.H, self.W, 3) * 255
        result = torch.cat([result, torch.ones((self.H, self.W, 1), device=result.device) * 255], dim=-1)
        return result

    def get_ray_directions_blender(self, H, W, focal, center=None, device=torch.device("cuda")):
        """
        Generate blender ray directions.

        Args:
            H(int): image height.
            W(int): image width.
            focal(tuple): focal value.
            center(Union[List, None]): center point.

        Returns:
            Tensor: ray directions.
        """
        grid = create_meshgrid(H, W, normalized_coordinates=False, device=device)[0]  # +0.5
        i, j = grid.unbind(-1)
        cent = center if center is not None else [W / 2, H / 2]
        directions = torch.stack(
            [
                ((i - cent[0]) / focal[0]).cuda(),
                (-(j - cent[1]) / focal[1]).cuda(),
                -torch.ones_like(i, device=device),
            ],
            -1,
        )  # (H, W, 3)
        # directions / torch.norm(directions, dim=-1, keepdim=True)
        return directions

    def get_rays_with_directions(self, directions, c2w):
        """
        Apply directions on pose.

        Args:
            directions(Tensor): ray directions.
            c2w(Tensor): pose.

        Returns:
            Tensor: rays tensor.
            Tensor: ray direction tensor.
        """
        rays_d = directions @ c2w[:3, :3].T
        rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
        return rays_o, rays_d

    def generate_rays(self, single_pose, H, W, focal=None):
        """
        Generate rays according to pose.

        Args:
            single_pose(list): single pose with focal.
            H(int): height.
            W(int): width.
            focal(float): flocal value.

        Returns:
            Tensor: rays.
        """
        if focal is None:
            focal = single_pose[-1, -1]
        directions = self.get_ray_directions_blender(int(H), int(W), (focal, focal))
        pose = torch.FloatTensor(single_pose[:3, :4]).cuda()
        rays_o, rays_d = self.get_rays_with_directions(directions, pose)
        return torch.cat([rays_o, rays_d], 1)


class InferenceGridNerfModule(InferenceNerfModule):
    """
    Gridnerf inference module
    """

    def __init__(self, *args, **kwargs):
        model = GridNeRF(*args, **kwargs)
        super().__init__(model=model)


class InferenceInstantNGPModule(InferenceNerfModule):
    """
    InstantNGP inference module
    """

    def __init__(self, *args, **kwargs):
        model = InstantNGP(*args, **kwargs)
        super().__init__(model=model)


class InferenceNerfactoModule(InferenceNerfModule):
    """
    Nerfacto inference module
    """

    def __init__(self, *args, **kwargs):
        model = Nerfacto(*args, **kwargs)
        super().__init__(model=model)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        assert strict or not strict
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict, map_location="cpu")
        return self.model.load(state_dict)


class InferenceOctreeGSModule(InferenceModule):
    """
    OctreeGS inference module
    """

    def __init__(self, *args, **kwargs):
        model = OctreeGS(*args, **kwargs)
        super().__init__(model=model)

    def preprocess(self, viewpoint_cam, scaling_modifier, retain_grad, render_exclude_filter):
        viewpoint_cam.image = None
        viewpoint_cam.c2w = None
        viewpoint_cam.R = None
        viewpoint_cam.T = None
        viewpoint_cam.trans = None
        viewpoint_cam.image_name = None
        viewpoint_cam.image_path = None

        ape_code = -1
        args = (viewpoint_cam, scaling_modifier, retain_grad, render_exclude_filter, ape_code)

        kwargs = {}
        return args, kwargs

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        return out

    def postprocess(self, out):
        return out
