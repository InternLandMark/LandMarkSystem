import torch
from torch import nn

from landmark.communicator.collective import all_reduce
from landmark.communicator.comm_context import CommContext
from landmark.communicator.group_initializer import CommMode

from .block import Block
from .buffer import Buffer
from .utils import broadcast_updated_tensor, update_merged_tensor


class BufferManager:
    """
    Desc:
        control the buffer swap/update/init during runtime.
    """

    def __init__(
        self,
        model: nn.Module,
        merged_state_dict,
        plane_split,
        local_plane_split,
        use_nccl=True,
        use_h2d=True,
        sync_switch=True,
    ):
        self.h2d_stream = torch.cuda.Stream()
        self.d2d_stream = torch.cuda.Stream()
        self.broadcast_stream = torch.cuda.Stream()

        self.blocks = []
        self.buffers = []
        self.curr_buffer_idx = 0
        self.merge_config = model.merge_config if model.merge_config != {} else model.get_merge_config()

        self.use_nccl = use_nccl
        self.use_h2d = use_h2d
        self.create_buffers(merged_state_dict, plane_split, local_plane_split)
        self.updated = False
        self.sync_switch = sync_switch

    def create_buffers(self, merged_state_dict, plane_split, local_plane_split):
        # TODO(wuguohao): those who don't need to do h2d copy shouldn't load full ckpt.
        # TODO(wuguohao): need a block to coord mapping.
        self.blocks = Block.create_blocks(self.merge_config, merged_state_dict, plane_split, self.use_h2d)
        self.buffers = Buffer.create_buffers(
            self.merge_config, self.blocks, plane_split, local_plane_split, self.use_nccl, self.use_h2d
        )

    def check_already_have(self, new_block_idxes):
        buffer = self.buffers[self.curr_buffer_idx]
        for block_idx in new_block_idxes:
            if block_idx not in buffer.local_block_idxes:
                return False
        # TODO(wuguohao): can use buffer_to_load's index to check also.
        return True

    def check_out_of_bound(self, new_block_idxes):
        for block_idx in new_block_idxes:
            if block_idx < 0 or block_idx >= len(self.blocks):
                return True
        return False

    def update_buffer(self, new_block_idxes, init=False):
        if self.updated or self.check_already_have(new_block_idxes) or self.check_out_of_bound(new_block_idxes):
            return
        print(f"new_block_idxes = {new_block_idxes}", flush=True)
        buffer_idx = self.curr_buffer_idx if init else self.curr_buffer_idx ^ 1
        buffer_to_load = self.buffers[buffer_idx]
        # buffer_in_use = self.buffers[self.curr_buffer_idx]
        non_blocking = not init
        new_blocks = [self.blocks[idx] for idx in new_block_idxes]
        new_state_dicts = [new_block.state_dict for new_block in new_blocks]
        event_dict = {}
        if self.use_h2d:
            event_dict = update_merged_tensor(
                self.merge_config,
                new_state_dicts,
                buffer_to_load.state_dict,
                new_block_idxes,
                cuda_stream=self.h2d_stream,
                non_blocking=non_blocking,
            )
        if self.use_nccl:
            broadcast_updated_tensor(
                self.merge_config,
                len(new_state_dicts),
                buffer_to_load.state_dict,
                event_dict=event_dict,
                cuda_stream=self.broadcast_stream,
                async_op=non_blocking,
            )
        buffer_to_load.local_block_idxes = sorted(new_block_idxes)
        self.updated = True

    def update_done(self):
        tmp = torch.cuda.FloatTensor([1.0])
        if self.updated:
            if self.h2d_stream.query() and self.broadcast_stream.query():
                tmp[0] = 0.0
        if CommContext().is_initialized() and self.sync_switch:
            all_reduce(tmp, comm_mode=CommMode.GLOBAL, op=torch.distributed.ReduceOp.AVG)
        if tmp[0].item() == 0:
            self.updated = False
            return not self.updated
        return False

    def wait_update_done(self):
        if self.updated:
            self.updated = False
            self.h2d_stream.synchronize()
            self.broadcast_stream.synchronize()

    def get_buffer(self, buffer_idx=None):
        if buffer_idx is None:
            buffer_idx = self.curr_buffer_idx
        return self.buffers[buffer_idx]

    def change_buffer(self):
        self.curr_buffer_idx ^= 1
