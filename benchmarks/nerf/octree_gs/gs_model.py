import math
import os
from typing import List, Optional

import torch

# Octree-GS uses the same rasterizer as Scaffold-GS
from opt_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from landmark.nerf_components.configs import BaseConfig, ConfigClass
from landmark.nerf_components.model import BaseGaussian
from landmark.nerf_components.model_components import (
    AnchorDecoder,
    AppearanceEmbedding,
    OctreeGSEncoding,
)
from landmark.nerf_components.scene.scene_manager import SceneManager


class OctreeGSConfig(ConfigClass):
    """Octree-GS Config"""

    feat_dim: int = 32
    view_dim: int = 3
    n_offsets: int = 10
    fork: int = 2

    use_feat_bank: bool = False
    source_path: str = ""
    model_path: str = ""
    images: str = "images"
    resolution: int = 1
    white_background: bool = False
    random_background: bool = False
    resolution_scales: List[float] = [1.0]

    data_device: str = "cuda"
    eval: bool = False
    ds: int = 1
    ratio: float = 1  # sampling the input point cloud
    undistorted: bool = False

    appearance_dim: int = 32
    add_opacity_dist: bool = False
    add_cov_dist: bool = False
    add_color_dist: bool = False
    add_level: bool = False

    extend: float = 1.1
    dist2level: str = "round"
    base_layer: int = -1  # -1(adaptive) or 10 (default) or 0 ~
    visible_threshold: float = 0.0  # -1(adaptive) or 0.0 ~ 1.0
    update_ratio: float = 0.2

    progressive: bool = False
    dist_ratio: float = 0.999  # 0.99/0.999
    levels: int = -1  # -1(adaptive) or 0 ~
    init_level: int = -1  # -1(adaptive) or 0 ~ levels-1
    extra_ratio: float = 0.5
    extra_up: float = 0.01

    # optim
    feature_lr: float = 0.0075
    opacity_lr: float = 0.02
    scaling_lr: float = 0.007
    rotation_lr: float = 0.002

    position_lr_init: float = 0.0
    position_lr_final: float = 0.0
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 40000

    offset_lr_init: float = 0.01
    offset_lr_final: float = 0.0001
    offset_lr_delay_mult: float = 0.01
    offset_lr_max_steps: int = 40000

    mlp_opacity_lr_init: float = 0.002
    mlp_opacity_lr_final: float = 0.00002
    mlp_opacity_lr_delay_mult: float = 0.01
    mlp_opacity_lr_max_steps: int = 40000

    mlp_cov_lr_init: float = 0.004
    mlp_cov_lr_final: float = 0.004
    mlp_cov_lr_delay_mult: float = 0.01
    mlp_cov_lr_max_steps: int = 40000

    mlp_color_lr_init: float = 0.008
    mlp_color_lr_final: float = 0.00005
    mlp_color_lr_delay_mult: float = 0.01
    mlp_color_lr_max_steps: int = 40000

    mlp_featurebank_lr_init: float = 0.01
    mlp_featurebank_lr_final: float = 0.00001
    mlp_featurebank_lr_delay_mult: float = 0.01
    mlp_featurebank_lr_max_steps: int = 40000

    appearance_lr_init: float = 0.05
    appearance_lr_final: float = 0.0005
    appearance_lr_delay_mult: float = 0.01
    appearance_lr_max_steps: int = 40000

    percent_dense: float = 0.01
    lambda_dssim: float = 0.2

    # for anchor densification
    start_stat: int = 500
    update_from: int = 1500
    coarse_iter: int = 5000
    coarse_factor: float = 1.5
    update_interval: int = 100
    update_until: int = 20000
    update_anchor: bool = True

    min_opacity: float = 0.005
    success_threshold: float = 0.8
    densify_grad_threshold: float = 0.0002

    # for extra use
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    enable_lod: bool = True

    no_batching: bool = False

    white_bkgd: bool = False
    resolution: float = -1
    resolution_scales: List[float] = [1.0]

    point_path: Optional[str] = None
    max_init_points: int = -1

    neighbour_size: int = 1

    init_points_num: int = 100000


class OctreeGS(BaseGaussian):
    """Octree-GS Model"""

    def __init__(self, config: BaseConfig, scene_manager: SceneManager = None):
        super().__init__(config, scene_manager)
        self.aabb = torch.tensor([config.lb, config.ub])
        # set background color
        bg_color = [1, 1, 1] if config.white_bkgd else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.init_field_components()
        self.init_renderer()

    def init_field_components(self):
        config = self.config
        self.gaussian_encoding = OctreeGSEncoding(
            feat_dim=config.feat_dim,
            n_offsets=config.n_offsets,
            fork=config.fork,
            visible_threshold=config.visible_threshold,
            dist2level=config.dist2level,
            base_layer=config.base_layer,
            progressive=config.progressive,
            extend=config.extend,
            percent_dense=config.percent_dense,
            enable_lod=config.enable_lod,
        )

        self.anchor_decoder = AnchorDecoder(
            use_feat_bank=config.use_feat_bank,
            add_opacity_dist=config.add_opacity_dist,
            add_color_dist=config.add_color_dist,
            add_cov_dist=config.add_cov_dist,
            view_dim=config.view_dim,
            feat_dim=config.feat_dim,
            n_offsets=config.n_offsets,
            appearance_dim=config.appearance_dim,
            add_level=config.add_level,
            dist2level=config.dist2level,
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
            sh_degree=1,
            campos=torch.Tensor([]),
            prefiltered=False,
            debug=self.config.debug,
        )

        self.renderer = GaussianRasterizer(raster_settings=raster_settings)

    def set_appearance(self, num_cameras):
        if self.config.appearance_dim > 0:
            self.embedding_appearance = AppearanceEmbedding(num_cameras, self.config.appearance_dim, "cuda")

    def get_appearance_code(self, appearance_code_idx):
        return self.embedding_appearance(appearance_code_idx)

    def set_level(self, pcd, train_cameras):
        self.gaussian_encoding.set_level(
            pcd,
            train_cameras,
            self.config.resolution_scales,
            self.config.dist_ratio,
            self.config.init_level,
            self.config.levels,
        )

    def set_coarse_interval(self):
        self.gaussian_encoding.set_coarse_interval(self.config.coarse_iter, self.config.coarse_factor)

    def set_anchor_mask(self, cam_center, iteration, resolution_scale):
        mlp_is_training = self.anchor_decoder.training
        self.gaussian_encoding.set_anchor_mask(cam_center, iteration, resolution_scale, mlp_is_training)

    def plot_levels(self):
        self.gaussian_encoding.plot_levels()

    def create_from_pcd(self, pcd, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        self.gaussian_encoding.create_from_pcd(pcd)
        print(f"Appearance Embedding Dimension: {self.config.appearance_dim}")

    def forward(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        retain_grad=False,
        render_exclude_filter=None,
        ape_code=-1,
    ):
        viewpoint_camera = viewpoint_camera.cuda()

        if self.training is False:
            self.set_anchor_mask(viewpoint_camera.camera_center, -1, viewpoint_camera.resolution_scale)

        voxel_visible_mask = self.prefilter_voxel(
            viewpoint_camera,
        )
        if render_exclude_filter is not None:
            voxel_visible_mask = voxel_visible_mask & render_exclude_filter

        if self.config.dist2level == "progressve":
            prog_ratio = self.gaussian_encoding._prog_ratio
            transition_mask = self.gaussian_encoding.transition_mask
        else:
            prog_ratio = None
            transition_mask = None

        if self.config.appearance_dim > 0:
            if self.training or ape_code < 0:
                appearance_code = self.get_appearance_code(viewpoint_camera.uid)
            else:
                appearance_code = self.get_appearance_code(ape_code)
        else:
            appearance_code = None

        xyz, color, opacity, scaling, rot, neural_opacity, mask = self.anchor_decoder(
            camera_center=viewpoint_camera.camera_center,
            anchor=self.gaussian_encoding.get_anchor,
            anchor_feat=self.gaussian_encoding.get_anchor_feat,
            offset=self.gaussian_encoding.get_offset,
            scaling=self.gaussian_encoding.get_scaling,
            level=self.gaussian_encoding.levels,
            visible_mask=voxel_visible_mask,
            prog_ratio=prog_ratio,
            transition_mask=transition_mask,
            appearance_code=appearance_code,
        )

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(xyz, dtype=self.gaussian_encoding.get_anchor.dtype, requires_grad=True, device="cuda") + 0
        )
        if retain_grad and self.training:
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
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.config.debug,
        )
        self.renderer.raster_settings = raster_settings

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, _ = self.renderer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=color,
            opacities=opacity,
            scales=scaling,
            rotations=rot,
            cov3D_precomp=None,
        )
        self.viewspace_points = screenspace_points
        self.visibility_filter = radii > 0
        self.voxel_visible_mask = voxel_visible_mask

        if self.training:
            self.offset_selection_mask = mask
            self.opacity = neural_opacity
            return rendered_image, scaling
        else:
            return rendered_image

    def densification(self, optimizer, iteration):
        config = self.config
        if config.start_stat < iteration < config.update_until:
            self.gaussian_encoding.training_statis(
                self.viewspace_points,
                self.opacity,
                self.visibility_filter,
                self.offset_selection_mask,
                self.voxel_visible_mask,
            )

            # densification
            if config.update_anchor and iteration > config.update_from and iteration % config.update_interval == 0:
                self.gaussian_encoding.adjust_anchor(
                    iteration=iteration,
                    check_interval=config.update_interval,
                    success_threshold=config.success_threshold,
                    grad_threshold=config.densify_grad_threshold,
                    update_ratio=config.update_ratio,
                    extra_ratio=config.extra_ratio,
                    extra_up=config.extra_up,
                    min_opacity=config.min_opacity,
                    optimizer=optimizer,
                )
        elif iteration == config.update_until:
            del self.gaussian_encoding.opacity_accum
            del self.gaussian_encoding.offset_gradient_accum
            del self.gaussian_encoding.offset_denom

            # delete attributes in the gaussian_cells
            if config.dynamic_load:
                for y_idx in range(config.plane_division[1]):
                    for x_idx in range(config.plane_division[0]):
                        del self.gaussians_cells[x_idx][y_idx]["opacity_accum"]
                        del self.gaussians_cells[x_idx][y_idx]["offset_gradient_accum"]
                        del self.gaussians_cells[x_idx][y_idx]["offset_denom"]

            torch.cuda.empty_cache()

    def get_optparam_groups(self, training_args):
        """
        Returns the parameter groups for optimization.

        Returns:
            list: A list of parameter groups for optimization.
        """
        param_groups = [
            {
                "params": [self.gaussian_encoding._anchor],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "anchor",
            },
            {
                "params": [self.gaussian_encoding._offset],
                "lr": training_args.offset_lr_init * self.spatial_lr_scale,
                "name": "offset",
            },
            {
                "params": [self.gaussian_encoding._anchor_feat],
                "lr": training_args.feature_lr,
                "name": "anchor_feat",
            },
            {"params": [self.gaussian_encoding._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self.gaussian_encoding._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self.gaussian_encoding._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
            {
                "params": self.anchor_decoder.mlp_opacity.parameters(),
                "lr": training_args.mlp_opacity_lr_init,
                "name": "mlp_opacity",
            },
            {
                "params": self.anchor_decoder.mlp_cov.parameters(),
                "lr": training_args.mlp_cov_lr_init,
                "name": "mlp_cov",
            },
            {
                "params": self.anchor_decoder.mlp_color.parameters(),
                "lr": training_args.mlp_color_lr_init,
                "name": "mlp_color",
            },
        ]

        if self.config.use_feat_bank:
            param_groups.append(
                {
                    "params": self.anchor_decoder.mlp_feature_bank.parameters(),
                    "lr": training_args.mlp_featurebank_lr_init,
                    "name": "mlp_featurebank",
                }
            )
        if self.config.appearance_dim > 0:
            param_groups.append(
                {
                    "params": self.embedding_appearance.parameters(),
                    "lr": training_args.appearance_lr_init,
                    "name": "embedding_appearance",
                }
            )

        return param_groups

    def save_mlp_checkpoints(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        mlp_checkpoints_dict = {
            "anchor_decoder.opacity_mlp": self.anchor_decoder.mlp_opacity.state_dict(),
            "anchor_decoder.cov_mlp": self.anchor_decoder.mlp_cov.state_dict(),
            "anchor_decoder.color_mlp": self.anchor_decoder.mlp_color.state_dict(),
        }

        if self.config.use_feat_bank:
            mlp_checkpoints_dict["anchor_decoder.mlp_feature_bank"] = self.anchor_decoder.mlp_feature_bank.state_dict()

        if self.config.appearance_dim > 0:
            mlp_checkpoints_dict["anchor_decoder.embedding_appearance"] = self.embedding_appearance.state_dict()

        torch.save(mlp_checkpoints_dict, path)

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
        means3D = self.gaussian_encoding.get_anchor[self.gaussian_encoding._anchor_mask]

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.config.compute_cov3D_python:
            cov3D_precomp = self.gaussian_encoding.get_covariance(scaling_modifier)
        else:
            scales = self.gaussian_encoding.get_scaling[self.gaussian_encoding._anchor_mask]
            rotations = self.gaussian_encoding.get_rotation[self.gaussian_encoding._anchor_mask]

        radii_pure = rasterizer.visible_filter(
            means3D=means3D, scales=scales[:, :3], rotations=rotations, cov3D_precomp=cov3D_precomp
        )

        visible_mask = self.gaussian_encoding._anchor_mask.clone()
        visible_mask[self.gaussian_encoding._anchor_mask] = radii_pure > 0
        return visible_mask

    @property
    def get_primitives_num(self):
        assert not self.config.dynamic_load, (
            "Do not use get_primitives_num when using dynamic load training, please refer to get_global_primitives_num"
            " or get_local_primitives_num instead."
        )
        return len(self.gaussian_encoding._anchor)
