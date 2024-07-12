from typing import Optional

import torch
from einops import repeat
from torch import nn

from landmark.nerf_components.model.base_module import BaseModule


class AnchorDecoder(BaseModule):
    """
    Decode the anchors to neural gaussians

    Args:
        use_feat_bank (bool): Whether to use feature bank.
        add_opacity_dist (bool): Whether to add opacity distribution.
        add_color_dist (bool): Whether to add color distribution.
        add_cov_dist (bool): Whether to add covariance distribution.
        view_dim (int): Dimension of the view.
        feat_dim (int): Dimension of the features.
        n_offsets (int): Number of offsets.
        appearance_dim (int, optional): Dimension of the appearance code. Defaults to 0.
        add_level (bool, optional): Whether to add level. Defaults to False.
        dist2level (str, optional): Distance to level. Defaults to None.

    Attributes:
        use_feat_bank (bool): Whether to use feature bank.
        add_opacity_dist (bool): Whether to add opacity distribution.
        add_color_dist (bool): Whether to add color distribution.
        add_cov_dist (bool): Whether to add covariance distribution.
        add_level (bool): Whether to add level.
        dist2level (str): Distance to level.
        view_dim (int): Dimension of the view.
        feat_dim (int): Dimension of the features.
        n_offsets (int): Number of offsets.
        appearance_dim (int): Dimension of the appearance code.
        mlp_feature_bank (nn.Sequential): MLP for feature bank.
        mlp_opacity (nn.Sequential): MLP for opacity.
        mlp_cov (nn.Sequential): MLP for covariance.
        mlp_color (nn.Sequential): MLP for color.

    Methods:
        forward(camera_center, anchor, anchor_feat, offset, scaling, visible_mask, level, prog_ratio,
        transition_mask, appearance_code):
            Forward pass of the Anchor Decoder.

    """

    def __init__(
        self,
        use_feat_bank: bool,
        add_opacity_dist: bool,
        add_color_dist: bool,
        add_cov_dist: bool,
        view_dim: int,
        feat_dim: int,
        n_offsets: int,
        appearance_dim: int = 0,
        add_level: bool = False,
        dist2level: str = None,
    ):
        super().__init__()
        self.use_feat_bank = use_feat_bank
        self.add_opacity_dist = add_opacity_dist
        self.add_color_dist = add_color_dist
        self.add_cov_dist = add_cov_dist
        self.add_level = add_level
        self.dist2level = dist2level
        self.view_dim = view_dim
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.appearance_dim = appearance_dim

        opacity_dist_dim = 1 if self.add_opacity_dist else 0
        cov_dist_dim = 1 if self.add_cov_dist else 0
        color_dist_dim = 1 if self.add_color_dist else 0
        level_dim = 1 if self.add_level else 0

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(self.view_dim + level_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1),
            ).cuda()
        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim + self.view_dim + opacity_dist_dim + level_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh(),
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim + self.view_dim + cov_dist_dim + level_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7 * self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim + self.view_dim + color_dist_dim + level_dim + self.appearance_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3 * self.n_offsets),
            nn.Sigmoid(),
        ).cuda()

        self.save_init_kwargs(locals())  # save for converting to fusion kernel

    def forward(
        self,
        camera_center: torch.Tensor,
        anchor: torch.Tensor,
        anchor_feat: torch.Tensor,
        offset: torch.Tensor,
        scaling: torch.Tensor,
        visible_mask: Optional[torch.Tensor],
        level: Optional[torch.Tensor] = None,
        prog_ratio: Optional[torch.Tensor] = None,
        transition_mask: Optional[torch.Tensor] = None,
        appearance_code: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the Anchor Decoder.

        Args:
            camera_center (torch.Tensor): Camera center.
            anchor (torch.Tensor): Anchor.
            anchor_feat (torch.Tensor): Anchor features.
            offset (torch.Tensor): Offset.
            scaling (torch.Tensor): Scaling.
            visible_mask (torch.Tensor, optional): Visible mask. Defaults to None.
            level (torch.Tensor, optional): Level. Defaults to None.
            prog_ratio (torch.Tensor, optional): Progress ratio. Defaults to None.
            transition_mask (torch.Tensor, optional): Transition mask. Defaults to None.
            appearance_code (torch.Tensor, optional): Appearance code. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor],
            Optional[torch.Tensor]]:
                - xyz (torch.Tensor): XYZ coordinates.
                - color (torch.Tensor): Color.
                - opacity (torch.Tensor): Opacity.
                - scaling (torch.Tensor): Scaling.
                - rot (torch.Tensor): Rotation.
                - neural_opacity (torch.Tensor, optional): Neural opacity. Defaults to None.
                - mask (torch.Tensor, optional): Mask. Defaults to None.
        """
        # view frustum filtering for acceleration
        if visible_mask is None:
            visible_mask = torch.ones(anchor.shape[0], dtype=torch.bool, device=anchor.device)

        anchor = anchor[visible_mask]
        feat = anchor_feat[visible_mask]
        grid_offsets = offset[visible_mask]
        grid_scaling = scaling[visible_mask]

        # feature from octree-gs
        if self.add_level and level is not None:
            level = level[visible_mask]

        ob_view = anchor - camera_center
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / ob_dist

        if self.use_feat_bank:
            if self.add_level and level is not None:
                cat_view = torch.cat([ob_view, level], dim=1)
            else:
                cat_view = ob_view

            bank_weight = self.mlp_feature_bank(cat_view).unsqueeze(dim=1)  # [n, 1, 3]

            feat = feat.unsqueeze(dim=-1)
            feat = (
                feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1]
                + feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2]
                + feat[:, ::1, :1] * bank_weight[:, :, 2:]
            )
            feat = feat.squeeze(dim=-1)  # [n, c]

        if self.add_level and level is not None:
            cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1)  # [N, c+3]
            cat_local_view_wodist = cat_local_view = torch.cat([feat, ob_view, level], dim=1)  # [N, c+3+1]
        else:
            cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]
            cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]

        if self.add_opacity_dist:
            neural_opacity = self.mlp_opacity(cat_local_view)  # [N, k]
        else:
            neural_opacity = self.mlp_opacity(cat_local_view_wodist)

        if self.dist2level == "progressive" and prog_ratio is not None and transition_mask is not None:
            prog = prog_ratio[visible_mask]
            transition_mask = transition_mask[visible_mask]
            prog[~transition_mask] = 1.0
            neural_opacity = neural_opacity * prog

        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = neural_opacity > 0.0
        mask = mask.view(-1)

        opacity = neural_opacity[mask]

        if self.appearance_dim > 0 and appearance_code is not None:
            appearance_code = (
                torch.ones_like(cat_local_view[:, 0], dtype=torch.long, device=ob_dist.device) * appearance_code[0]
            )
            if self.add_color_dist:
                color = self.mlp_color(torch.cat([cat_local_view, appearance_code], dim=1))
            else:
                color = self.mlp_color(torch.cat([cat_local_view_wodist, appearance_code], dim=1))
        else:
            if self.add_color_dist:
                color = self.mlp_color(cat_local_view)
            else:
                color = self.mlp_color(cat_local_view_wodist)
        color = color.reshape([anchor.shape[0] * self.n_offsets, 3])  # [mask]

        if self.add_cov_dist:
            scale_rot = self.mlp_cov(cat_local_view)
        else:
            scale_rot = self.mlp_cov(cat_local_view_wodist)
        scale_rot = scale_rot.reshape([anchor.shape[0] * self.n_offsets, 7])  # [mask]

        offsets = grid_offsets.view([-1, 3])  # [mask]

        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, "n (c) -> (n k) (c)", k=self.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

        scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
        rot = torch.nn.functional.normalize(scale_rot[:, 3:7])

        offsets = offsets * scaling_repeat[:, :3]
        xyz = repeat_anchor + offsets

        if self.training:
            return xyz, color, opacity, scaling, rot, neural_opacity, mask
        else:
            return xyz, color, opacity, scaling, rot, None, None
