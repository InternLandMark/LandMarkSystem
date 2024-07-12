import torch

from landmark.communicator import CommContext, broadcast
from landmark.communicator.group_initializer import _InterCommMode
from landmark.nerf_components.model.base_module import BaseModule
from landmark.render.util.types import MergeType
from landmark.utils import MERGE_TYPE_STR

INT_MAX = 0x3F3F3F3F3F
MAX_DEPTH = 3


def search_maximum_memory_used_per_param(blocks, plane_split, local_plane_split):
    # find the minimal memory space of blocks' param
    max_attr_numel_sum = {}
    max_neighbours = {}
    for attr_name in blocks[0].state_dict.keys():
        for block_idx, _ in enumerate(blocks):
            neighbours = calculate_block_neighbours(block_idx, plane_split, local_plane_split)
            if -1 in neighbours:
                continue
            attr_numel = 0
            for neighbour in neighbours:
                attr_numel += blocks[neighbour].state_dict[attr_name].numel()
            if attr_name not in max_attr_numel_sum:
                max_attr_numel_sum[attr_name] = 0

            if attr_numel > max_attr_numel_sum[attr_name]:
                max_attr_numel_sum[attr_name] = attr_numel
                max_neighbours[attr_name] = neighbours

    return max_neighbours


# assume that block are distributed according to the index.
def calculate_block_neighbours(block_idx, plane_split, local_plane_split):
    idx_x = block_idx // plane_split[1]
    idx_y = block_idx % plane_split[1]
    neighbours = []

    lbound_x = (local_plane_split[0] - 1) // 2
    rbound_x = local_plane_split[0] - lbound_x

    lbound_y = (local_plane_split[1] - 1) // 2
    rbound_y = local_plane_split[1] - lbound_y

    LX = (local_plane_split[0] - 1) // 2
    RX = plane_split[0] - (local_plane_split[0] - LX)
    LY = (local_plane_split[1] - 1) // 2
    RY = plane_split[1] - (local_plane_split[1] - LY)

    if idx_x < LX:
        idx_x = LX
    elif idx_x > RX:
        idx_x = RX
    if idx_y < LY:
        idx_y = LY
    elif idx_y > RY:
        idx_y = RY

    for offset_x in range(-lbound_x, rbound_x):
        for offset_y in range(-lbound_y, rbound_y):
            n_idx_x = idx_x + offset_x
            n_idx_y = idx_y + offset_y
            # if n_idx = -1, it means out of bound. can use dummy tensor to fill this block.
            if n_idx_x < 0 or n_idx_x >= plane_split[0] or n_idx_y < 0 or n_idx_y >= plane_split[1]:
                n_idx = -1
            else:
                n_idx = n_idx_x * plane_split[1] + n_idx_y
            neighbours.append(n_idx)
    return neighbours


def assign_block_to_pose(pose: torch.Tensor, aabb, plane_split):
    block_width = aabb[0] // plane_split[0]
    block_height = aabb[1] // plane_split[1]

    block_idx_x = pose[0] // block_width
    block_idx_y = pose[1] // block_height

    return block_idx_x * plane_split[1] + block_idx_y


def match(key, _dict):
    """
    Desc:
        only used in MergeType.List.
        This function assumes that params stored in list form the blocks.
        And assume that param key with `*` will correspon to at least one block,
        whose key is `0`.

    Example:
        mlp_base_grid.tcnn_encoding.*.params -> mlp_base_grid.tcnn_encoding.0.params
    """
    pos = key.rfind("*")
    if pos != -1 and key.replace("*", "0", 1) in _dict:
        return True

    return False


def rm_origin_module_param(model, merge_config, local_block_num):
    def clear_data(key, module, merge_type, prefix):
        if merge_type is MergeType.Unevenly:
            param = module.get_parameter(key)
            param.data = torch.tensor([], device=param.device)
        elif merge_type is MergeType.Evenly:
            param = module.get_parameter(key)
            param.data = torch.tensor([], device=param.device)
        elif merge_type is MergeType.List:
            # TODO: need to find a better way to handle MergeType.List
            pos = key.find("*")
            assert pos != -1
            key_prefix = key[:pos]
            key_suffix = key[pos + 1 :]

            model_param_names = []
            for name, _ in module.named_parameters(prefix=prefix[:-1]):
                next_token_pos = name[pos:].find(".")
                if name.startswith(key_prefix) and name[pos + next_token_pos :].startswith(key_suffix):
                    model_param_names.append(name)
                if len(model_param_names) == local_block_num:
                    break

            for model_param_name in model_param_names:
                param = module.get_parameter(model_param_name)
                param.data = torch.tensor([], device=param.device)

            # truncate the list
            module_name = key_prefix[:-1] if key_prefix.endswith(".") else key_prefix
            module_list = module.get_submodule(module_name)
            if len(module_list) > len(model_param_names):
                print("Offload: triger the module list truncating")
                module_list = module_list[: len(model_param_names)]

                # update the parent module
                nested_module = module
                sub_modules = module_name.split(".")
                for sub_module in sub_modules[:-1]:
                    nested_module = getattr(nested_module, sub_module)
                setattr(nested_module, sub_modules[-1], module_list)
        else:
            raise Exception(f"Not expected merge_config type {merge_type} found.")

    def rm_param(module, local_merge_config, prefix=""):
        model_param_names = [name for name, _ in module.named_parameters(prefix=prefix[:-1])]
        for key in local_merge_config.keys():
            if key in model_param_names or match(key, model_param_names):
                _, merge_type = get_merge_type(key, local_merge_config)
                clear_data(key, module, merge_type, prefix)

        for name, child in module.named_children():
            if child is not None and isinstance(child, BaseModule):
                child_prefix = prefix + name + "."
                child_merge_config = {
                    k[len(child_prefix) :]: v for k, v in merge_config.items() if k.startswith(child_prefix)
                }
                rm_param(child, child_merge_config, child_prefix)

    assert isinstance(model, BaseModule)
    rm_param(model, merge_config)


def load_buffer(model, buffer, local_block_num):
    def load(module, prefix=""):
        merge_type = (
            model.merge_config[prefix + MERGE_TYPE_STR] if prefix + MERGE_TYPE_STR in model.merge_config else None
        )
        if merge_type is not None:
            for key in buffer.state_dict.keys():
                if not key.startswith(prefix):
                    continue

                if merge_type is not MergeType.List:
                    for name, param in module.named_parameters(prefix=prefix[:-1]):
                        if key == name:
                            param.data = buffer.state_dict[key]
                    continue

                # Chunk buffer tensor into list of tensor to satisfy the `MergeType.List` requirement.
                assert merge_type is MergeType.List
                block_tensors = torch.chunk(buffer.state_dict[key], local_block_num)
                assert id(block_tensors[0].storage()) == id(buffer.state_dict[key].storage())
                new_keys = [key.replace("*", str(idx), 1) for idx in range(local_block_num)]
                for name, param in module.named_parameters(prefix=prefix[:-1]):
                    if name in new_keys:
                        index = new_keys.index(name)
                        param.data = block_tensors[index]

        for name, child in module.named_children():
            if isinstance(child, BaseModule):
                load(child, prefix + name + ".")

    assert isinstance(model, BaseModule), f"Expected type BaseModule, but type {type(model)} was given."
    load(model)


def get_merge_type(key, merge_config):
    maxsplit = 1
    merge_type_key = key.rsplit(sep=".", maxsplit=maxsplit)[0] + "." + MERGE_TYPE_STR
    while merge_type_key not in merge_config:
        maxsplit += 1
        merge_type_key = key.rsplit(sep=".", maxsplit=maxsplit)[0] + "." + MERGE_TYPE_STR
    assert merge_type_key in merge_config
    return merge_type_key, merge_config[merge_type_key]


def split_tensor_into_block(key, merge_config, merged_state_dict, merge_type, block_num):
    if merge_type is MergeType.Unevenly:
        split_dim = 0
        split_nums = merge_config[key]
        merged_tensor = merged_state_dict[key]
        split_tensors = list(torch.split(merged_tensor, split_nums, dim=split_dim))
    elif merge_type is MergeType.Evenly:
        split_dim = merge_config[key]
        split_num = block_num
        merged_tensor = merged_state_dict[key]
        split_tensors = list(torch.chunk(merged_tensor, split_num, dim=split_dim))
        assert len(split_tensors) == split_num
    elif merge_type is MergeType.List:
        split_tensors = []
        block_idx = 0
        while True:
            block_key = key.replace("*", str(block_idx), 1)
            if block_key not in merged_state_dict:
                break
            split_tensors.append(merged_state_dict[block_key])
            block_idx += 1
    else:
        raise Exception(f"Not expected merge_config type {merge_type} found.")

    return split_tensors


def create_splited_tensor(merge_config, merged_state_dict, plane_split, pin_memory=True):
    # TODO: can use pre-defined split_state_dicts to reduce the memory creation time costs.
    print(f"merge_config = {merge_config}", flush=True)
    print(f"merged_state_dict.keys() = {merged_state_dict.keys()}", flush=True)
    print(f"plane_split = {plane_split}", flush=True)

    block_num = plane_split[0] * plane_split[1]
    split_state_dicts = [{} for _ in range(block_num)]
    # for key in merged_state_dict.keys():
    #     if key in merge_config:
    for key in merge_config.keys():
        if key in merged_state_dict or match(key, merged_state_dict):
            _, merge_type = get_merge_type(key, merge_config)

            # split tensor into block num
            split_tensors = split_tensor_into_block(key, merge_config, merged_state_dict, merge_type, block_num)

            for idx, split_tensor in enumerate(split_tensors):
                split_state_dicts[idx][key] = split_tensor.contiguous()
                if pin_memory:
                    # pin splited tensor to pin memory
                    split_state_dicts[idx][key] = split_state_dicts[idx][key].pin_memory()

    return split_state_dicts


def create_buffer_attr_shape(merge_config, state_dicts, merged_state_dict=None):
    def modify_shape(origin_shape, dim, num):
        origin_shape_list = list(origin_shape)
        origin_shape_list[dim] = num
        return tuple(origin_shape_list)

    if merged_state_dict is None:
        merged_state_dict = {}

    buffer_attr_shape = {}
    block_num = len(state_dicts)
    for key in state_dicts[0].keys():
        # skip already created tensor
        if key in merged_state_dict.keys():
            continue

        # filter components' config
        assert key in merge_config
        _, merge_type = get_merge_type(key, merge_config)
        if merge_type is MergeType.Evenly:
            merge_dim = merge_config[key]
            merge_num = block_num
            buffer_attr_shape[key] = modify_shape(state_dicts[0][key].shape, merge_dim, merge_num)
        elif merge_type is MergeType.Unevenly:
            element_num_per_block = merge_config[key]
            merge_dim = 0
            merge_num = sum(element_num_per_block)
            buffer_attr_shape[key] = modify_shape(state_dicts[0][key].shape, merge_dim, merge_num)
        elif merge_type is MergeType.List:
            for i in range(block_num):
                assert (
                    state_dicts[0][key].shape == state_dicts[i][key].shape
                ), f"{state_dicts[0][key].shape} != {state_dicts[i][key].shape}"
            merge_dim = 0
            merge_num = state_dicts[0][key].shape[0] * block_num
            buffer_attr_shape[key] = modify_shape(state_dicts[0][key].shape, merge_dim, merge_num)
        else:
            raise Exception(f"type(merge_config[{key}]) = {type(merge_config[key])}, not support now.")

    return buffer_attr_shape


def create_merged_tensor(buffer_attr_shape, merged_state_dict=None, device="cpu"):
    if merged_state_dict is None:
        merged_state_dict = {}

    for key in buffer_attr_shape.keys():
        # skip already created tensor
        if key in merged_state_dict.keys():
            continue

        global_shape = buffer_attr_shape[key]
        merged_tensor = torch.zeros(global_shape, device=device)
        merged_state_dict[key] = merged_tensor

    return merged_state_dict


def update_merged_tensor(
    merge_config,
    state_dicts,
    merged_state_dict,
    new_block_idxes,
    cuda_stream=None,
    non_blocking=False,
):
    block_num = len(state_dicts)
    event_dict = {}
    new_block_idxes = [int(new_block_idx) for new_block_idx in new_block_idxes]
    for key in merged_state_dict.keys():
        # filter components' config
        if key in merge_config:
            _, merge_type = get_merge_type(key, merge_config)
            if merge_type is MergeType.Evenly:
                merge_dim = merge_config[key]
                merged_tensor = merged_state_dict[key]
                padding_dim = [slice(None, None, None) for _ in range(merge_dim + 1)]
                src_padding_dim = [slice(None, None, None) for _ in range(merge_dim + 1)]
                src_padding_dim[-1] = 0
                with torch.cuda.stream(cuda_stream):
                    for block_idx in range(block_num):
                        padding_dim[-1] = block_idx
                        merged_tensor[padding_dim].copy_(
                            state_dicts[block_idx][key][src_padding_dim],
                            non_blocking=non_blocking,
                        )
                        event = torch.cuda.Event()
                        event.record()
                        event_dict[key + str(block_idx)] = event
            elif merge_type is MergeType.Unevenly:
                # merge_dim = 0
                # reset the numel to real numel
                orign_shape = merged_state_dict[key].shape[1:]
                merged_state_dict[key] = (
                    merged_state_dict[key].set_(merged_state_dict[key].storage()).reshape(-1, *orign_shape)
                )
                merged_tensor = merged_state_dict[key]
                merge_nums = merge_config[key]  # element number per block
                merged_num = 0  # offset of target tensor
                with torch.cuda.stream(cuda_stream):
                    slice_list = []
                    for idx, block_idx in enumerate(new_block_idxes):
                        slice_list.append(merged_tensor[merged_num : merged_num + merge_nums[block_idx]])
                        merged_num += merge_nums[block_idx]
                    # set the numel to used numel
                    merged_state_dict[key] = merged_state_dict[key][:merged_num]
                    for idx, block_idx in enumerate(new_block_idxes):
                        slice_list[idx].copy_(
                            state_dicts[idx][key],
                            non_blocking=non_blocking,
                        )
                        event = torch.cuda.Event()
                        event.record()
                        event_dict[key + str(block_idx)] = event
            elif merge_type is MergeType.List:
                merge_dim = merge_config[key]
                merged_tensor = merged_state_dict[key]
                start_offset = 0
                with torch.cuda.stream(cuda_stream):
                    for block_idx in range(block_num):
                        block_shape0 = state_dicts[block_idx][key].shape[0]
                        end_offset = start_offset + block_shape0
                        merged_tensor[start_offset:end_offset].copy_(
                            state_dicts[block_idx][key],
                            non_blocking=non_blocking,
                        )
                        start_offset = end_offset
                        event = torch.cuda.Event()
                        event.record()
                        event_dict[key + str(block_idx)] = event

    return event_dict


def broadcast_updated_tensor(
    merge_config,
    state_dicts_num,
    merged_state_dict,
    event_dict=None,
    cuda_stream=None,
    async_op=True,
):
    if event_dict is None:
        event_dict = {}
    block_num = state_dicts_num
    for key in merged_state_dict.keys():
        # filter components' config
        if key in merge_config:
            merge_dim = merge_config[key]
            merged_tensor = merged_state_dict[key]
            if merge_dim < MAX_DEPTH:
                with torch.cuda.stream(cuda_stream):
                    for block_idx in range(block_num):
                        dict_key = key + str(block_idx)
                        if dict_key in event_dict:
                            event_dict[dict_key].wait()
                        recurve_broadcast(merged_tensor, block_idx, merge_dim, 0, async_op=async_op)
            else:
                with torch.cuda.stream(cuda_stream):
                    for block_idx in range(block_num):
                        dict_key = key + str(block_idx)
                        if dict_key in event_dict:
                            event_dict[dict_key].wait()
                    src = CommContext().get_ranks_in_group(comm_mode=_InterCommMode.DYNAMIC_LOAD)[0]
                    broadcast(merged_tensor, src=src, comm_mode=_InterCommMode.DYNAMIC_LOAD, async_op=async_op)


def recurve_broadcast(tensor, block_idx, target_dim, curr_dim, async_op=True):
    if target_dim == curr_dim:
        src = CommContext().get_ranks_in_group(comm_mode=_InterCommMode.DYNAMIC_LOAD)[0]
        broadcast(tensor[block_idx], src=src, comm_mode=_InterCommMode.DYNAMIC_LOAD, async_op=async_op)
        return

    for j in range(tensor.shape[0]):
        recurve_broadcast(tensor[j], block_idx, target_dim, curr_dim + 1, async_op=async_op)
