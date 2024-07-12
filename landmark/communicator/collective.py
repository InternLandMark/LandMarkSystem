# pylint: disable=W0613
from typing import Optional

from torch import distributed as dist
from torch.distributed import ReduceOp

from .comm_context import CommContext
from .group_initializer import CommMode, _InterCommMode

_all_gather_func = dist._all_gather_base if "all_gather_into_tensor" not in dir(dist) else dist.all_gather_into_tensor


def scatter(
    tensor,
    comm_mode: CommMode,
    scatter_list: Optional[list] = None,
    src: int = 0,
    async_op: bool = False,
):
    """
    custom scatter operation.

    Args:
        tensor(Tensor): Output tensor.
        comm_mode (CommMode): Communication mode registered in CommContext.
        scatter_list(list[Tensor]): List of tensors to scatter (default is
            None, must be specified on the source rank).
        src(int): Src rank.
        async_op(bool): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.scatter(
        tensor=tensor,
        scatter_list=scatter_list,
        src=src,
        group=group,
        async_op=async_op,
    )


def broadcast_object_list(object_list: list, comm_mode: CommMode, src: int = 0):
    """
    Broadcasts python objects based on torch.distributed.broadcast_object_list

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
        comm_mode (CommMode): Communication mode registered in CommContext.

    Returns:
        ``None``
    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.broadcast_object_list(object_list, src=src, group=group)


def all_gather(output_tensor, input_tensor, comm_mode: CommMode, async_op: bool = False):
    """
    Single tensor all gather. Gathers a single tensor from all ranks, and puts them in a single output tensor.

    Args:
        output_tensor (Tensor): Output tensor. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return _all_gather_func(output_tensor=output_tensor, input_tensor=input_tensor, group=group, async_op=async_op)


def all_reduce(tensor, comm_mode: CommMode, op=ReduceOp.SUM, async_op: bool = False):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Complex tensors are supported.

    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        comm_mode (CommMode): Communication mode registered in CommContext.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.all_reduce(tensor=tensor, op=op, group=group, async_op=async_op)


def broadcast(tensor, comm_mode: CommMode, src: int = 0, async_op: bool = False):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Args:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        comm_mode (CommMode): Communication mode registered in CommContext.
        src (int): Source rank.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)


def gather(tensor, comm_mode: CommMode, gather_list: Optional[list] = None, dst: int = 0, async_op: bool = False):
    """
    Gathers a list of tensors in a single process.

    Args:
        tensor (Tensor): Input tensor.
        comm_mode (CommMode): Communication mode registered in CommContext.
        gather_list (list[Tensor], optional): List of appropriately-sized
            tensors to use for gathered data (default is None, must be specified
            on the destination rank)
        dst (int, optional): Destination rank (default is 0)
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    group = CommContext().get_group(comm_mode=comm_mode)
    return dist.gather(tensor, gather_list=gather_list, dst=dst, group=group, async_op=async_op)


def _broadcast_user_inputs_message(object_list: list):
    broadcast_object_list(object_list=object_list, comm_mode=_InterCommMode.MESSAGE_TRANSFER, src=0)
