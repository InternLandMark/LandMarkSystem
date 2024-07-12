# pylint: disable=E1136
import torch

from landmark.nerf_components.data.samples import Samples


class SceneManager:
    """Class for managing scene information"""

    def __init__(self, config=None) -> None:
        self.meta = {}
        self.aabb = torch.tensor(
            [config.lb, config.ub], device=config.device
        )  # global_aabb, [[x_min, y_min, z_min], [x_max, y_max, z_max]],
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.aabb_norm_range = (
            torch.tensor(config.aabb_norm_range, device=config.device)
            if hasattr(config, "aabb_norm_range")
            else torch.tensor([-1, 1], device=config.device)
        )
        self.aabb_norm_size = self.aabb_norm_range[1] - self.aabb_norm_range[0]
        self.invaabbSize = self.aabb_norm_size / self.aabbSize
        self.block_partition = (
            torch.tensor(config.plane_division, device=config.device) if hasattr(config, "plane_division") else None
        )
        self.block_width = (self.aabb[1][:2] - self.aabb[0][:2]) / self.block_partition
        self.norm_block_width = self.aabb_norm_size / self.block_partition

        # for offload
        self.local_block_partition = (
            config.local_block_partition if hasattr(config, "local_block_partition") else self.block_partition
        )
        relative_block_idx = config.relative_block_idx if hasattr(config, "relative_block_idx") else 0
        self.assign_relative_block_idx(relative_block_idx)
        self.config = config

    def select_block_idx(self, input_args):
        rays = input_args
        curr_block_idx_xy = torch.floor((rays[0, :2] - self.aabb[0, :2]) // self.block_width)

        curr_block_idx_xy = torch.min(
            torch.max(curr_block_idx_xy, torch.zeros(2, device=curr_block_idx_xy.device)),
            self.block_partition - 1,
        )
        curr_block_idx = curr_block_idx_xy[0] * self.block_partition[1] + curr_block_idx_xy[1]
        return curr_block_idx.to(torch.int)

    def assign_local_block_partition(self, local_block_partition):
        self.local_block_partition = local_block_partition

    def assign_block_partition(self, block_partition):
        if self.block_partition is not None:
            self.block_partition[0] = block_partition[0]
            self.block_partition[1] = block_partition[1]
        else:
            self.block_partition = torch.tensor(block_partition, device=self.config.device)

    def assign_relative_block_idx(self, relative_block_idx=0):
        self.relative_block_idx = relative_block_idx
        self.relative_block_idx_x = self.relative_block_idx // self.block_partition[1].item()
        self.relative_block_idx_y = self.relative_block_idx % self.block_partition[1].item()

    def normalize_coord(self, xyz_sampled):
        if xyz_sampled.shape[-1] == 3:
            return (xyz_sampled - self.aabb[0]) * self.invaabbSize + self.aabb_norm_range[0]
        elif xyz_sampled.shape[-1] == 4:
            xyzb_sampled = xyz_sampled.clone().detach()
            xyzb_sampled[..., :3] = (xyz_sampled[..., :3] - self.aabb[0]) * self.invaabbSize + self.aabb_norm_range[0]
            return xyzb_sampled
        else:
            assert False  # TODO raise Error (frank)

    @torch.no_grad()
    def assign_blockIdx(self, samples: Samples, norm_coord=False) -> Samples:
        """ Assign branch index to each sample by querying the partition. This is an inplace \
            operation for Sample object.
        """
        # generate masks
        if self.norm_block_width[0] == self.aabb_norm_size and self.norm_block_width[1] == self.aabb_norm_size:
            samples.block_idx = torch.zeros_like(samples.xyz[..., 0])
        else:
            xyz = samples.xyz.clone()  # TODO (frank) remove clone
            xyz[..., :2] -= self.aabb_norm_range[0]

            xyz[..., :2] /= self.norm_block_width
            block_idx_x = torch.floor(xyz[..., 0])
            block_idx_y = torch.floor(xyz[..., 1])
            samples.block_idx = (
                (block_idx_x - self.relative_block_idx_x) * self.local_block_partition[1]
                + block_idx_y
                - self.relative_block_idx_y
            )
            if self.local_block_partition[0] == 1 and self.local_block_partition[1] == 1:
                # if local_block_partition is (1,1), torch grid_sample D dimension won't work,
                # so we manually filter the sample point belongs to other blocks except block 0.
                padding = torch.zeros_like(samples.xyz[..., 0])
                samples.xyz[..., 0] = torch.where(samples.block_idx == 0, samples.xyz[..., 0], padding)
                samples.xyz[..., 1] = torch.where(samples.block_idx == 0, samples.xyz[..., 1], padding)

            block_num = self.local_block_partition[0] * self.local_block_partition[1]

            # compute masks for samples
            masks = []
            for block_idx in range(block_num):
                masks.append(samples.block_idx == block_idx)
            samples.masks = torch.stack(masks, dim=0)

            # TODO decoupling assign block_idx with normalize block_idx (frank)
            if norm_coord:
                # normalized xyb coords (when using stack-merged ckpt)
                coord_min = torch.stack(
                    [
                        -1
                        + (samples.block_idx // self.local_block_partition[1] + self.relative_block_idx_x)
                        * self.norm_block_width[0],
                        -1
                        + (samples.block_idx % self.local_block_partition[1] + self.relative_block_idx_y)
                        * self.norm_block_width[1],
                    ],
                    dim=-1,
                )
                samples.xyz[..., :2] = (samples.xyz[..., :2] - coord_min) * self.block_partition - 1

                samples.block_idx = -1 + 2 * samples.block_idx / (block_num - 1 + 1e-6)

    def filter_valid_samples(self, samples: Samples) -> (torch.Tensor, Samples):
        """
        filter the samples that are outside the scene
        """
        out_aabb_mask = (samples.xyz < self.aabb[0]) | (samples.xyz > self.aabb[1])
        aabb_mask = ~out_aabb_mask
        valid_samples = samples[out_aabb_mask]
        return aabb_mask, valid_samples
