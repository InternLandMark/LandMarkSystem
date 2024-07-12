from typing import Any, Mapping

from torch import nn

from landmark.communicator.comm_context import CommContext
from landmark.communicator.group_initializer import _InterCommMode
from landmark.render.util import InferenceModule
from landmark.utils import Config

from .buffer_manager import BufferManager
from .utils import calculate_block_neighbours, load_buffer, rm_origin_module_param


class OffloadModule(nn.Module):
    "A model wrapper for parameter offloading."

    def __init__(self, model: InferenceModule, merged_state_dict, config: Config):
        super().__init__()
        self.model = model
        self.scene_mgr = model.scene_manager
        self.scene_mgr.assign_local_block_partition(config.local_plane_split)
        if hasattr(config, "plane_split") and isinstance(config.plane_split, list):
            self.scene_mgr.assign_block_partition(config.plane_split)
        print(
            f"self.scene_mgr.block_partition = {self.scene_mgr.block_partition}, "
            f"self.scene_mgr.local_block_partition = {self.scene_mgr.local_block_partition}",
            flush=True,
        )
        use_nccl = True if not hasattr(config, "use_nccl") else config.use_nccl
        use_h2d = True
        if hasattr(config, "use_h2d"):
            use_h2d = config.use_h2d
        elif use_nccl and CommContext().get_local_rank(_InterCommMode.DYNAMIC_LOAD) != 0:
            use_h2d = False
        if hasattr(config, "channel_last") and config.channel_last:
            self.model.channel_last()
        sync_switch = True
        if hasattr(config, "sync_switch"):
            sync_switch = config.sync_switch

        merged_state_dict = self._find_state_dict(merged_state_dict)
        merge_config = model.merge_config if model.merge_config != {} else model.get_merge_config()
        local_block_num = self.scene_mgr.local_block_partition[0] * self.scene_mgr.local_block_partition[1]
        if isinstance(model, InferenceModule):
            rm_origin_module_param(model.model, merge_config, local_block_num)
            self.buffer_manager = BufferManager(
                model,
                merged_state_dict,
                self.scene_mgr.block_partition,
                config.local_plane_split,
                use_nccl,
                use_h2d,
                sync_switch,
            )
            # load empty buffer to assign memory space to model's params to forbid error during load_state_dict
            load_buffer(self.model.model, self.buffer_manager.get_buffer(), local_block_num)
        else:
            rm_origin_module_param(model, merge_config, local_block_num)
            self.buffer_manager = BufferManager(
                model,
                merged_state_dict,
                self.scene_mgr.block_partition,
                config.local_plane_split,
                use_nccl,
                use_h2d,
                sync_switch,
            )
            # load empty buffer to assign memory space to model's params to forbid error during load_state_dict
            load_buffer(self.model, self.buffer_manager.get_buffer(), local_block_num)

        self.init = True

    def _find_state_dict(self, state_dict):
        if "state_dict" in state_dict:
            return state_dict["state_dict"]
        return state_dict

    def __getattr__(self, item):
        """
        This method is to adjust the gs model getter func.
        """
        if item == "model":
            return super().__getattr__(item)
        return getattr(self.model, item)

    def update_state_dict(self, state_dict):
        """
        Using partially merged state_dict to update the state_dict.
        """
        state_dict = self._find_state_dict(state_dict)
        state_dict.update(self.buffer_manager.get_buffer().state_dict)
        return state_dict

    def load_from_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        self.update_state_dict(state_dict)
        self.model.load_from_state_dict(state_dict, strict)

    def _pre_forward(self, *args):
        camera = args[0].cuda()
        new_block_idx = self.scene_mgr.select_block_idx(camera)  # TODO: need to standardize the input type
        new_block_idxes = calculate_block_neighbours(
            new_block_idx, self.scene_mgr.block_partition, self.scene_mgr.local_block_partition
        )
        local_block_num = self.scene_mgr.local_block_partition[0] * self.scene_mgr.local_block_partition[1]
        self.buffer_manager.update_buffer(new_block_idxes, self.init)
        if self.init:
            self.init = False
            self.buffer_manager.wait_update_done()
            buffer = self.buffer_manager.get_buffer()

            if isinstance(self.model, InferenceModule):
                load_buffer(self.model.model, buffer, local_block_num)
            else:
                load_buffer(self.model, buffer, local_block_num)
            self.scene_mgr.assign_relative_block_idx(buffer.local_block_idxes[0])
        elif self.buffer_manager.update_done():
            self.buffer_manager.change_buffer()
            buffer = self.buffer_manager.get_buffer()
            if isinstance(self.model, InferenceModule):
                load_buffer(self.model.model, buffer, local_block_num)
            else:
                load_buffer(self.model, buffer, local_block_num)
            self.scene_mgr.assign_relative_block_idx(buffer.local_block_idxes[0])

    def _post_forward(self):
        pass

    def forward(self, *args, **kwargs):
        self._pre_forward(*args)
        outputs = self.model(*args, **kwargs)
        self._post_forward()
        return outputs

    def preprocess(self, *args, **kwargs):
        return self.model.preprocess(*args, **kwargs)

    def postprocess(self, *args, **kwargs):
        return self.model.postprocess(*args, **kwargs)
