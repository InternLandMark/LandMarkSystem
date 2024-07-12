import math
from typing import List, Optional

import torch
from opt_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from landmark.nerf_components.configs import BaseConfig, ConfigClass
from landmark.nerf_components.model.base_model import BaseGaussian
from landmark.nerf_components.model_components import GaussianEncoding
from landmark.nerf_components.scene.scene_manager import SceneManager
from landmark.nerf_components.utils.graphics_utils import BasicPointCloud
from landmark.nerf_components.utils.sh_utils import eval_sh


class Gaussian3DConfig(ConfigClass):
    """3DSG Config"""

    sh_degree: int = 3
    images: str = "images"
    model_path: str = None

    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000

    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

    percent_dense: float = 0.01
    lambda_dssim: float = 0.2

    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002

    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False

    no_batching: bool = False

    white_bkgd: bool = False
    resolution: int = -1
    resolution_scales: List[float] = [1.0]

    point_path: Optional[str] = None
    max_init_points: int = -1

    neighbour_size: int = 1

    init_points_num: int = 100000

    act_cache: bool = False


class GaussianModel(BaseGaussian):
    """3DGS Model"""

    def __init__(self, config: BaseConfig, scene_manager: SceneManager = None, cam_extent=None):
        super().__init__(config, scene_manager)
        self.cam_extent = cam_extent

        # set background color
        bg_color = [1, 1, 1] if config.white_bkgd else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.spatial_lr_scale = 0
        self.init_field_components()
        self.init_renderer()

    def init_field_components(self):
        config = self.config
        self.gaussian_encoding = GaussianEncoding(
            sh_degree=config.sh_degree, percent_dense=config.percent_dense, act_cache=config.act_cache
        )

    def init_renderer(self):

        raster_settings = GaussianRasterizationSettings(
            image_height=0,
            image_width=0,
            tanfovx=0,
            tanfovy=0,
            bg=self.background,
            scale_modifier=0,
            viewmatrix=torch.Tensor([]),
            projmatrix=torch.Tensor([]),
            sh_degree=self.gaussian_encoding.active_sh_degree,
            campos=torch.Tensor([]),
            prefiltered=False,
            debug=self.config.debug,
        )

        self.renderer = GaussianRasterizer(raster_settings=raster_settings)

    def construct_list_of_attributes(self):
        return self.gaussian_encoding.construct_list_of_attributes()

    def construct_list_of_cache_attributes(self):
        return self.gaussian_encoding.construct_list_of_cache_attributes()

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        self.gaussian_encoding.create_from_pcd(pcd)

    def oneupSHdegree(self):
        self.gaussian_encoding.oneupSHdegree()

    def forward(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        override_color=None,
        render_exclude_filter=None,
    ):
        viewpoint_camera = viewpoint_camera.cuda()
        config = self.config

        voxel_visible_mask = self.prefilter_voxel(
            viewpoint_camera,
        )
        if render_exclude_filter is not None:
            voxel_visible_mask = voxel_visible_mask & render_exclude_filter

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussian_encoding.get_xyz,
                dtype=self.gaussian_encoding.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        if self.training:
            screenspace_points.retain_grad()

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FovX * 0.5)
        tanfovy = math.tan(viewpoint_camera.FovY * 0.5)

        image_height = viewpoint_camera.height
        image_width = viewpoint_camera.width

        raster_settings = GaussianRasterizationSettings(
            image_height=int(image_height),
            image_width=int(image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.background,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussian_encoding.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.config.debug,
        )
        self.renderer.raster_settings = raster_settings

        means3D = self.gaussian_encoding.get_xyz
        means2D = screenspace_points
        opacity = self.gaussian_encoding.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if config.compute_cov3D_python:
            cov3D_precomp = self.gaussian_encoding.get_covariance(scaling_modifier)
        else:
            scales = self.gaussian_encoding.get_scaling
            rotations = self.gaussian_encoding.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if config.convert_SHs_python:
                shs_view = self.gaussian_encoding.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussian_encoding.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussian_encoding.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussian_encoding.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.gaussian_encoding.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussian_encoding.get_features
        else:
            colors_precomp = override_color

        if not self.training:
            mask = voxel_visible_mask.nonzero().squeeze(-1)
            means3D = means3D[mask]
            means2D = means2D[mask]
            shs = shs[mask]
            colors_precomp = colors_precomp[mask] if colors_precomp is not None else None
            opacity = opacity[mask]
            scales = scales[mask]
            rotations = rotations[mask]
            cov3D_precomp = cov3D_precomp[mask] if cov3D_precomp is not None else None

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, _ = self.renderer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.

        self.viewspace_points = screenspace_points
        self.visibility_filter = voxel_visible_mask
        self.radii = radii
        return rendered_image

    def densification(self, optimizer, iteration):
        config = self.config
        if iteration < config.densify_until_iter:
            self.gaussian_encoding.max_radii2D[self.visibility_filter] = torch.max(
                self.gaussian_encoding.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter]
            )
            self.gaussian_encoding.add_densification_stats(self.viewspace_points, self.visibility_filter)

            if iteration > config.densify_from_iter and iteration % config.densification_interval == 0:
                size_threshold = 20 if iteration > config.opacity_reset_interval else None

                self.gaussian_encoding.densify_and_prune(
                    config.densify_grad_threshold,
                    0.005,
                    self.cam_extent,
                    size_threshold,
                    optimizer,
                )

            if iteration % config.opacity_reset_interval == 0 or (
                config.white_bkgd and iteration == config.densify_from_iter
            ):
                self.gaussian_encoding.reset_opacity(optimizer)

    def get_optparam_groups(self, training_args):
        """
        Returns the parameter groups for optimization.

        Returns:
            list: A list of parameter groups for optimization.
        """
        param_groups = [
            {
                "params": [self.gaussian_encoding._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {"params": [self.gaussian_encoding._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {
                "params": [self.gaussian_encoding._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {"params": [self.gaussian_encoding._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self.gaussian_encoding._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self.gaussian_encoding._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        return param_groups

    def prefilter_voxel(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FovX * 0.5)
        tanfovy = math.tan(viewpoint_camera.FovY * 0.5)

        image_height = viewpoint_camera.height
        image_width = viewpoint_camera.width

        raster_settings = GaussianRasterizationSettings(
            image_height=int(image_height),
            image_width=int(image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.background,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.config.debug,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussian_encoding.get_xyz

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.config.compute_cov3D_python:
            cov3D_precomp = self.gaussian_encoding.get_covariance(scaling_modifier)
        else:
            scales = self.gaussian_encoding.get_scaling
            rotations = self.gaussian_encoding.get_rotation

        radii_pure = rasterizer.visible_filter(
            means3D=means3D, scales=scales[:, :3], rotations=rotations, cov3D_precomp=cov3D_precomp
        )

        return radii_pure > 0

    @property
    def get_primitives_num(self):
        assert not self.config.dynamic_load, (
            "Do not use get_primitives_num when using dynamic load training, please refer to get_global_primitives_num"
            " or get_local_primitives_num instead."
        )
        return len(self.gaussian_encoding._xyz)
