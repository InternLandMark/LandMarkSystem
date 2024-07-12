from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .convert_maps import kernel_fusion_components_convert_map


def _filter_dict_by_strings(dictionary, strings):
    """
    Remove keys from a dictionary if they do not contain any of the specified strings.

    Args:
        dictionary (dict): The dictionary to filter.
        strings (list of str): A list of strings to check against the dictionary keys.

    Returns:
        None: Modifies the dictionary in place by removing keys that do not match the criteria.
    """
    keys_to_remove = []
    for key in dictionary:
        if not any(string in key for string in strings):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del dictionary[key]


def _convert_components_by_kernel_runtime(
    model: nn.Module, verbose: bool = True, trans_comp=None, parent_attr_name: str = ""
):
    """
    Recursively replace torch components with their kernel component equivalents within a model.

    Args:
        model (nn.Module): The model to convert.
        verbose (bool, optional): If True, print information about the conversion process. Defaults to True.
        trans_comp (list, optional): A list to keep track of the components that have been converted. Defaults to None.
        parent_attr_name (str, optional): The parent attribute name for nested models. Defaults to "".

    Returns:
        None: Modifies the model in place by replacing components.
    """
    assert isinstance(trans_comp, list)

    for attr_name, module in model.named_children():
        if isinstance(module, nn.Module):
            full_attr_name = f"{parent_attr_name}.{attr_name}" if parent_attr_name else attr_name
            module_name = type(module).__name__

            try:
                try:
                    init_kwargs = module.get_init_kwargs()
                    if "device" in init_kwargs:
                        init_kwargs["device"] = "cuda"
                except AttributeError:
                    pass
                if module_name in kernel_fusion_components_convert_map:
                    kernel_module = kernel_fusion_components_convert_map[module_name](**init_kwargs)
                    setattr(model, attr_name, kernel_module)
                    print(f"{attr_name}: {module_name} -> {type(kernel_module).__name__}")
                    trans_comp.append(full_attr_name)
            except KeyError:
                if verbose:
                    print(f"{attr_name}: {module_name} -> SKIP")

            _convert_components_by_kernel_runtime(module, verbose, trans_comp, full_attr_name)


def _rewarp_ddp_module(model: nn.Module, trans_comp=None, parent_attr_name: str = ""):
    """
    Rewrap DDP modules around newly converted kernel components within a model.

    Args:
        model (nn.Module): The model to rewrap.
        trans_comp (list, optional): A list of converted components to consider for rewrapping. Defaults to None.
        parent_attr_name (str, optional): The parent attribute name for nested models. Defaults to "".

    Returns:
        None: Modifies the model in place by rewrapping DDP modules.
    """
    assert isinstance(trans_comp, list)

    for attr_name, module in model.named_children():
        full_attr_name = f"{parent_attr_name}.{attr_name}" if parent_attr_name else attr_name
        _rewarp_ddp_module(module, trans_comp, full_attr_name)
        if isinstance(module, DDP):
            for key in trans_comp:
                if key.startswith(full_attr_name):
                    module = DDP(module.module, device_ids=module.device_ids, process_group=module.process_group)
                    setattr(model, attr_name, module)
                    print(f"found DDP wraped kernel module {key}, re-warp {attr_name} -> {type(module).__name__}")
