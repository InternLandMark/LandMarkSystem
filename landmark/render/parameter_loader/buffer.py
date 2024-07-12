from landmark.communicator import _broadcast_user_inputs_message

from .utils import (
    create_buffer_attr_shape,
    create_merged_tensor,
    search_maximum_memory_used_per_param,
)

BUFFER_SIZE = 2


class Buffer:
    """
    Buffer class
    """

    def __init__(self, local_plane_split):
        # location metadata
        self.local_plane_split = local_plane_split
        self.local_block_idxes = [-1 for _ in range(local_plane_split[0] * local_plane_split[1])]
        self.state_dict = {}

    @staticmethod
    def create_buffers(merge_config, blocks, plane_split, local_plane_split, use_nccl, use_h2d):
        assert len(blocks) != 0
        print(f"len(blocks) = {len(blocks)}", flush=True)

        if use_h2d:
            max_neighbours = search_maximum_memory_used_per_param(blocks, plane_split, local_plane_split)

            max_state_dicts = [{} for _ in range(local_plane_split[0] * local_plane_split[1])]
            for attr_name, neighbours in max_neighbours.items():
                for idx, neighbour in enumerate(neighbours):
                    max_state_dicts[idx][attr_name] = blocks[neighbour].state_dict[attr_name]
            buffer_attr_shape = create_buffer_attr_shape(merge_config, max_state_dicts)
        else:
            buffer_attr_shape = None

        if use_nccl:
            tmp = [buffer_attr_shape]
            _broadcast_user_inputs_message(tmp)
            buffer_attr_shape = tmp[0]

        buffers = []
        for _ in range(BUFFER_SIZE):
            buffer = Buffer.create_buffer(buffer_attr_shape, local_plane_split)
            buffers.append(buffer)
        return buffers

    @staticmethod
    def create_buffer(buffer_attr_shape, local_plane_split):
        buffer = Buffer(local_plane_split)
        buffer.state_dict = create_merged_tensor(buffer_attr_shape, device="cuda")
        return buffer
