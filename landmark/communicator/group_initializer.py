from abc import ABC, abstractmethod

from torch import distributed as dist


class CommMode:
    GLOBAL = "GLOBAL"


class ParallelMode(CommMode):
    DATA_PARALLEL = "DATA_PARALLEL"
    TENSOR_PARALLEL = "TENSOR_PARALLEL"


class _InterCommMode(CommMode):
    MESSAGE_TRANSFER = "MESSAGE_TRANSFER"
    DYNAMIC_LOAD = "DYNAMIC_LOAD"


class ProcessGroupInitializer(ABC):
    """Base class to initialize process group"""

    def __init__(self, rank, world_size) -> None:
        self.rank = rank
        self.world_size = world_size

        self.local_rank = None
        self.ranks_in_group = None
        self.process_group = None
        self.group_world_size = None
        self.mode = None

    @abstractmethod
    def init_dist_group(self):
        pass

    def _new_and_update_group_info(self, ranks, use_cpu=False):
        backend = "gloo" if use_cpu else "nccl"
        group = dist.new_group(ranks, backend=backend)

        if self.rank in ranks:
            self.local_rank = ranks.index(self.rank)
            self.group_world_size = len(ranks)
            self.process_group = group
            self.ranks_in_group = ranks
        else:
            self.group_world_size = len(ranks)


class DPGroupInitializer(ProcessGroupInitializer):
    """data parallel group initializer"""

    def __init__(self, rank, world_size, data_parallel_size) -> None:
        super().__init__(rank, world_size)
        self.data_parallel_size = data_parallel_size
        self.process_num_between_dp_rank = world_size // data_parallel_size
        self.mode = ParallelMode.DATA_PARALLEL

    def init_dist_group(self):
        ranks = [i * self.process_num_between_dp_rank for i in range(self.data_parallel_size)]
        self._new_and_update_group_info(ranks)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )


class TPGroupInitializer(ProcessGroupInitializer):
    """tensor parallel group initializer"""

    def __init__(self, rank, world_size, tensor_parallel_size) -> None:
        super().__init__(rank, world_size)
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_group_num = world_size // tensor_parallel_size
        self.mode = ParallelMode.TENSOR_PARALLEL

    def init_dist_group(self):
        for i in range(self.tensor_parallel_group_num):
            ranks = [i * self.tensor_parallel_size + j for j in range(self.tensor_parallel_size)]
            self._new_and_update_group_info(ranks)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )


class MessageTransferInitializer(ProcessGroupInitializer):
    """message transfer initializer"""

    def __init__(self, rank, world_size) -> None:
        super().__init__(rank, world_size)
        self.mode = _InterCommMode.MESSAGE_TRANSFER

    def init_dist_group(self):
        ranks = list(range(self.world_size))
        self._new_and_update_group_info(ranks, use_cpu=True)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )


class DynamicLoadGroupInitializer(ProcessGroupInitializer):
    """dynamic loading group initializer"""

    def __init__(self, rank, world_size, tensor_parallel_size, node_cuda_num) -> None:
        super().__init__(rank, world_size)
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_group_num = world_size // tensor_parallel_size
        self.node_cuda_num = node_cuda_num
        self.mode = _InterCommMode.DYNAMIC_LOAD

    def init_dist_group(self):
        for i in range(self.tensor_parallel_size):
            # TODO(wuguohao): use intra-node group instead of inter-node group
            ranks = [i + j * self.tensor_parallel_size for j in range(self.tensor_parallel_group_num)]
            # assert ranks is ordered
            ranks_within_node = []
            node_idx = 0
            for rank in ranks:
                if (rank // self.node_cuda_num) != node_idx:
                    if len(ranks_within_node) > 0:
                        self._new_and_update_group_info(ranks_within_node)
                    ranks_within_node = []
                    node_idx = rank // self.node_cuda_num
                ranks_within_node.append(rank)
            if len(ranks_within_node) > 0:
                self._new_and_update_group_info(ranks_within_node)

        return (
            self.local_rank,
            self.group_world_size,
            self.process_group,
            self.ranks_in_group,
            self.mode,
        )
