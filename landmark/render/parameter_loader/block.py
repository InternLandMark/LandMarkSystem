import torch

from .utils import create_splited_tensor


class Block:
    """
    Block class description
    """

    def __init__(self, index=-1, coord: torch.Tensor = None):
        # location metadata
        self.block_idx = index
        self.coord = coord  # 定义 block 对应的 xyz 坐标
        self.aabb = []
        self.state_dict = {}

    @staticmethod
    def create_blocks(merge_config, merged_state_dict, plane_split, use_h2d):
        block_num = plane_split[0] * plane_split[1]
        blocks = [Block() for _ in range(block_num)]
        if use_h2d:
            split_state_dicts = create_splited_tensor(merge_config, merged_state_dict, plane_split)
            for block_idx in range(block_num):
                blocks[block_idx].state_dict.update(split_state_dicts[block_idx])
        return blocks
