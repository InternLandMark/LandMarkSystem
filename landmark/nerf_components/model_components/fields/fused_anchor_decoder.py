# pylint: disable=E1136
from typing import Optional

try:
    import fusedKernels
    import FusedMatmul
except:
    pass
import torch

from landmark.nerf_components.model_components.fields.anchor_decoder import (
    AnchorDecoder,
)


class FusedAnchorDecoder(AnchorDecoder):
    """
    Decode the anchors to neural gaussians

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        w0_opa (torch.Tensor): Weight tensor of the first layer of the opacity MLP.
        b0_opa (torch.Tensor): Bias tensor of the first layer of the opacity MLP.
        w1_opa (torch.Tensor): Weight tensor of the second layer of the opacity MLP.
        b1_opa (torch.Tensor): Bias tensor of the second layer of the opacity MLP.
        a0_opa (str): Activation function of the first layer of the opacity MLP.
        a1_opa (str): Activation function of the second layer of the opacity MLP.
        w0_cov (torch.Tensor): Weight tensor of the first layer of the covariance MLP.
        b0_cov (torch.Tensor): Bias tensor of the first layer of the covariance MLP.
        w1_cov (torch.Tensor): Weight tensor of the second layer of the covariance MLP.
        b1_cov (torch.Tensor): Bias tensor of the second layer of the covariance MLP.
        a0_cov (str): Activation function of the first layer of the covariance MLP.
        a1_cov (str): Activation function of the second layer of the covariance MLP.
        w0_color (torch.Tensor): Weight tensor of the first layer of the color MLP.
        b0_color (torch.Tensor): Bias tensor of the first layer of the color MLP.
        w1_color (torch.Tensor): Weight tensor of the second layer of the color MLP.
        b1_color (torch.Tensor): Bias tensor of the second layer of the color MLP.
        a0_color (str): Activation function of the first layer of the color MLP.
        a1_color (str): Activation function of the second layer of the color MLP.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w0_opa = self.mlp_opacity.state_dict()["0.weight"]
        self.b0_opa = self.mlp_opacity.state_dict()["0.bias"]
        self.w1_opa = self.mlp_opacity.state_dict()["2.weight"]
        self.b1_opa = self.mlp_opacity.state_dict()["2.bias"]
        self.a0_opa = "relu"
        self.a1_opa = "tanh"

        self.w0_cov = self.mlp_cov.state_dict()["0.weight"]
        self.b0_cov = self.mlp_cov.state_dict()["0.bias"]
        self.w1_cov = self.mlp_cov.state_dict()["2.weight"]
        self.b1_cov = self.mlp_cov.state_dict()["2.bias"]
        self.a0_cov = "relu"
        self.a1_cov = "none"

        self.w0_color = self.mlp_color.state_dict()["0.weight"]
        self.b0_color = self.mlp_color.state_dict()["0.bias"]
        self.w1_color = self.mlp_color.state_dict()["2.weight"]
        self.b1_color = self.mlp_color.state_dict()["2.bias"]
        self.a0_color = "relu"
        self.a1_color = "sigmoid"

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
            camera_center (torch.Tensor): The camera center tensor.
            anchor (torch.Tensor): The anchor tensor.
            anchor_feat (torch.Tensor): The anchor feature tensor.
            offset (torch.Tensor): The offset tensor.
            scaling (torch.Tensor): The scaling tensor.
            visible_mask (torch.Tensor, optional): The visible mask tensor. Defaults to None.
            level (torch.Tensor, optional): The level tensor. Defaults to None.
            prog_ratio (torch.Tensor, optional): The progressive ratio tensor. Defaults to None.
            transition_mask (torch.Tensor, optional): The transition mask tensor. Defaults to None.
            appearance_code (torch.Tensor, optional): The appearance code tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor],
            Optional[torch.Tensor]]:
            - xyz (torch.Tensor): The tensor containing the centers for gaussians.
            - color (torch.Tensor): The tensor containing the color values.
            - opacity (torch.Tensor): The tensor containing the opacity values.
            - scaling (torch.Tensor): The tensor containing the scaling values.
            - rot (torch.Tensor): The tensor containing the rotation values.
            - neural_opacity (torch.Tensor, optional): The tensor containing the neural opacity values.
            Defaults to None.
            - mask (torch.Tensor, optional): The tensor containing the mask values. Defaults to None.

        """

        # view frustum filtering for acceleration
        if visible_mask is None:
            visible_mask = torch.ones(anchor.shape[0], dtype=torch.bool, device=anchor.device)

        visible_idx = torch.nonzero(visible_mask)
        feat = fusedKernels.simpleMask(anchor_feat, visible_idx)  # [904794, 32]
        grid_offsets = fusedKernels.simpleMask(offset, visible_idx)  # [904794, 10, 3]
        grid_scaling = fusedKernels.simpleMask(scaling, visible_idx)  # [904794, 6]
        # feature from octree-gs
        if self.add_level and level is not None:
            level = fusedKernels.simpleMask(level, visible_idx)

        # get view properties for anchor
        anchor, ob_view, ob_dist = fusedKernels.ob_property(
            anchor, visible_idx, camera_center[0].item(), camera_center[1].item(), camera_center[2].item(), 1
        )

        # view-adaptive feature
        if self.use_feat_bank:
            if self.add_level and level is not None:
                cat_view = torch.cat([ob_view, level], dim=1)
            else:
                cat_view = ob_view

            bank_weight = self.mlp_feature_bank(cat_view).unsqueeze(dim=1)  # [n, 1, 3]

            # multi-resolution feat
            feat = feat.unsqueeze(dim=-1)
            fusedKernels.SelfContainedFeat(feat, bank_weight)
            feat = feat.squeeze(dim=-1)  # [n, c]

        if self.add_level and level is not None:
            cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1)  # [N, c+3]
            cat_local_view_wodist = cat_local_view = torch.cat([feat, ob_view, level], dim=1)  # [N, c+3+1]
        else:
            cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]
            cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]

        # get offset's opacity
        if self.add_opacity_dist:
            # neural_opacity = self.mlp_opacity(cat_local_view)  # [N, k]
            neural_opacity = FusedMatmul.simple2layer(
                cat_local_view, self.w0_opa, self.b0_opa, self.a0_opa, self.w1_opa, self.b1_opa, self.a1_opa
            )
        else:
            # neural_opacity = self.mlp_opacity(cat_local_view_wodist)
            neural_opacity = FusedMatmul.simple2layer(
                cat_local_view_wodist, self.w0_opa, self.b0_opa, self.a0_opa, self.w1_opa, self.b1_opa, self.a1_opa
            )

        # feature from octree-gs
        if self.dist2level == "progressive" and prog_ratio is not None and transition_mask is not None:
            prog = prog_ratio[visible_mask]
            transition_mask = transition_mask[visible_mask]
            prog[~transition_mask] = 1.0
            neural_opacity = neural_opacity * prog

        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = neural_opacity > 0.0
        mask = mask.view(-1)

        # select opacity
        mask_idx = torch.nonzero(mask)
        opacity = fusedKernels.simpleMask(neural_opacity, mask_idx)

        # get offset's color
        if self.appearance_dim > 0 and appearance_code is not None:
            appearance_code = (
                torch.ones_like(cat_local_view[:, 0], dtype=torch.long, device=ob_dist.device) * appearance_code[0]
            )
            if self.add_color_dist:
                color_input = torch.cat([cat_local_view, appearance_code], dim=1)
                # color = self.mlp_color(torch.cat([cat_local_view, appearance_code], dim=1))
            else:
                color_input = torch.cat([cat_local_view_wodist, appearance_code], dim=1)
                # color = self.mlp_color(torch.cat([cat_local_view_wodist, appearance_code], dim=1))
        else:
            if self.add_color_dist:
                color_input = cat_local_view
                # color = self.mlp_color(cat_local_view)
            else:
                color_input = cat_local_view_wodist
                # color = self.mlp_color(cat_local_view_wodist)
        color = FusedMatmul.simple2layer(
            color_input, self.w0_color, self.b0_color, self.a0_color, self.w1_color, self.b1_color, self.a1_color
        )
        color = color.reshape([anchor.shape[0] * self.n_offsets, 3])  # [mask]

        # get offset's cov
        if self.add_cov_dist:
            # scale_rot = self.mlp_cov(cat_local_view)
            scale_rot = FusedMatmul.simple2layer(
                cat_local_view, self.w0_cov, self.b0_cov, self.a0_cov, self.w1_cov, self.b1_cov, self.a1_cov
            )
        else:
            # scale_rot = self.mlp_cov(cat_local_view_wodist)
            scale_rot = FusedMatmul.simple2layer(
                cat_local_view_wodist, self.w0_cov, self.b0_cov, self.a0_cov, self.w1_cov, self.b1_cov, self.a1_cov
            )
        scale_rot = scale_rot.reshape([anchor.shape[0] * self.n_offsets, 7])  # [mask]

        # offsets
        offsets = grid_offsets.view([-1, 3])  # [mask]

        # repeat and mask
        scaling_repeat = torch.empty(
            (mask_idx.shape[0], grid_scaling.shape[1]), dtype=grid_scaling.dtype, device=grid_scaling.device
        )
        fusedKernels.RepeatMask(grid_scaling, self.n_offsets, mask_idx, scaling_repeat)
        # mask
        scale_rot_tmp = torch.empty(
            (mask_idx.shape[0], scale_rot.shape[1]), dtype=scale_rot.dtype, device=scale_rot.device
        )
        fusedKernels.simpleIdx(scale_rot, mask_idx, scale_rot_tmp)
        scale_rot = scale_rot_tmp

        # mask and post-process color
        override_color = torch.empty((mask_idx.shape[0], color.shape[1]), dtype=color.dtype, device=color.device)
        fusedKernels.MaskPostProcessColor(color, mask_idx, override_color, 0, 1.0)
        color = override_color

        # post-process cov
        scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])  # * torch.sigmoid(repeat_dist)
        rot = torch.nn.functional.normalize(scale_rot[:, 3:7])

        # post-process offsets to get centers for gaussians
        xyz = torch.empty((mask_idx.shape[0], anchor.shape[1]), device=anchor.device, dtype=anchor.dtype)
        fusedKernels.RepeatMaskPostProcessOffsets(anchor, offsets, scaling_repeat, mask_idx, xyz, self.n_offsets)

        if self.training:
            return xyz, color, opacity, scaling, rot, neural_opacity, mask
        else:
            return xyz, color, opacity, scaling, rot, None, None
