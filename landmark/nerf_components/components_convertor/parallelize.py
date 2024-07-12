from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from landmark.train.utils.distributed_utils import get_local_rank

from .convert_maps import encoding_convert_map


def _convert_module_by_parallel_mode(
    model: nn.Module,
    parallel_mode: str,
    parallel_degree: int,
    parallel_part: int,
    parallel_group,
    verbose: bool = True,
):
    """
    Replace sequence components with parallel components based on the specified parallel mode.

    This function iterates through all modules in a given PyTorch model and replaces them with their parallel
     equivalents
    if available in the `encoding_convert_map`. It supports different parallel modes, such as branch parallel, by
    wrapping modules with the appropriate parallel wrapper (e.g., DDP for distributed data parallelism).

    Args:
        model (nn.Module): The model whose modules are to be converted.
        parallel_mode (str): The mode of parallelism to apply (e.g., "BranchParallel").
        parallel_degree (int): The degree of parallelism.
        parallel_part (int): The part of the model to parallelize.
        parallel_group: The group of processes for distributed training.
        verbose (bool, optional): If True, prints detailed information about the conversion process. Defaults to True.

    Returns:
        nn.Module: The modified model with converted modules.

    Note:
        Currently, this function primarily supports branch parallelism. Future optimizations may extend support
        to other parallel modes.
    """
    converted_result = []
    for attr_name, module in model.named_children():
        module_name = type(module).__name__

        has_parameters = sum(1 for _ in module.parameters()) > 0
        if not has_parameters:
            converted_result.append(f"{attr_name}: {module_name} -> SKIP")
            continue

        try:
            try:
                init_kwargs = module.get_init_kwargs()
                if "device" in init_kwargs:
                    init_kwargs["device"] = "cuda"
            except AttributeError:
                pass
            parallel_module = encoding_convert_map[parallel_mode][module_name](
                parallel_part=parallel_part,
                parallel_degree=parallel_degree,
                group=parallel_group,
                init_state_dict=module.state_dict(),
                **init_kwargs,
            )  # TODO (frank) set by parallel_degree and config
            setattr(model, attr_name, parallel_module)
            converted_result.append(f"{attr_name}: {module_name} -> {type(parallel_module).__name__}")
        except KeyError:
            if parallel_mode == "BranchParallel":
                module = DDP(module, device_ids=[get_local_rank()], process_group=parallel_group)
                setattr(model, attr_name, module)
                converted_result.append(f"{attr_name}: {module_name} -> Replica")
            else:
                converted_result.append(f"{attr_name}: {module_name} -> SKIP")

    if verbose:
        print("Converted result:")
        for line in converted_result:
            print("    ", line)

    return model
