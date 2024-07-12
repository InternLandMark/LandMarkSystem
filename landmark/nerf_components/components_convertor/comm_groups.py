import torch
import torch.distributed as dist

from landmark.nerf_components.configs import BaseConfig

# _DP_GROUP = None
# _BRANCH_PARALLEL_GROUP = None
# _BRANCH_PARALLEL_RANKS = []
# _CHANNEL_PARALLEL_GROUP = None
# _CHANNEL_PARALLEL_RANKS = []

# Global variables for distributed training configuration
_DP_GROUP = None  # Distributed Processing Group
_MODEL_PARALLEL_GROUP = None  # Model Parallel Group
_MODEL_PARALLEL_RANKS = []  # List of ranks in the model parallel group
_MODEL_PARALLEL_RANK = None  # Current rank in the model parallel group


def init_train_groups(config: BaseConfig):
    """
    Initialize parallel communication groups based on the training configuration.

    This function sets up the distributed processing (DP) and model parallel (MP) groups
    according to the specified configuration. It calculates the degrees of parallelism
    for branch and channel parallelism, and organizes the ranks into groups accordingly.

    Args:
        config (BaseConfig): The configuration object containing settings for parallelism.

    Returns:
        BaseConfig: The updated configuration object with added attributes for DP and MP groups.
    """
    global _DP_GROUP
    global _MODEL_PARALLEL_GROUP
    global _MODEL_PARALLEL_RANKS
    global _MODEL_PARALLEL_RANK

    assert dist.is_initialized()
    world_size = dist.get_world_size()
    cur_rank = dist.get_rank()

    if config.branch_parallel:
        assert config.plane_division is not None
        branch_parallel_degree = config.plane_division[0] * config.plane_division[1]
    else:
        branch_parallel_degree = 1
    if config.channel_parallel_size is not None:
        channel_parallel_degree = config.channel_parallel_size
    else:
        channel_parallel_degree = 1
    model_parallel_degree = branch_parallel_degree * channel_parallel_degree
    assert world_size % model_parallel_degree == 0
    num_mp_groups = int(world_size // model_parallel_degree)
    for i in range(num_mp_groups):
        mp_ranks = range(model_parallel_degree * i, model_parallel_degree * (i + 1))
        mp_group = torch.distributed.new_group(mp_ranks)
        if cur_rank in mp_ranks:
            _MODEL_PARALLEL_GROUP = mp_group
            _MODEL_PARALLEL_RANKS = mp_ranks

    num_dp_groups = model_parallel_degree
    dp_group_size = int(world_size // model_parallel_degree)
    for g in range(num_dp_groups):
        dp_ranks = [g + j * num_dp_groups for j in range(dp_group_size)]
        dp_group = torch.distributed.new_group(dp_ranks)
        if cur_rank in dp_ranks:
            _DP_GROUP = dp_group

    _MODEL_PARALLEL_RANK = cur_rank % model_parallel_degree

    config.dp_group = get_dp_group()
    config.dp_rank = cur_rank // model_parallel_degree
    config.num_mp_groups = world_size // model_parallel_degree
    config.mp_group = get_mp_group()
    config.mp_rank = get_mp_rank()

    return config


def get_dp_group():
    """Returns the current Distributed Processing group."""
    return _DP_GROUP


def get_mp_group():
    """Returns the current Model Parallel group."""
    return _MODEL_PARALLEL_GROUP


def get_mp_rank0():
    """Returns the rank 0 of the Model Parallel group."""
    return _MODEL_PARALLEL_RANKS[0]


def get_mp_rank():
    """Returns the current rank in the Model Parallel group."""
    return _MODEL_PARALLEL_RANK
