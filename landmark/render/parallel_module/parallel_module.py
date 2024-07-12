from abc import abstractmethod
from typing import Any, Mapping

import torch
from torch.nn import Module

from landmark.communicator import CommContext, ParallelMode, all_gather
from landmark.nerf_components.components_convertor.parallelize import (
    _convert_module_by_parallel_mode,
)
from landmark.render.util import InferenceModule, StageInput
from landmark.utils import Config

from .optimization_manager import OptimizationManager


class ParallelModule(Module):
    """
    Parallelize normal torch module
    """

    def __init__(self, model: Module, config: Config) -> None:
        super().__init__()

        self._inference_config = config
        if hasattr(config, "parallel_config"):
            self._parallel_config = config.parallel_config
        # self._dp_config, self._gate = self._parse_dp_config(config)
        self._dp_gap_size = 0
        self._gs_model = False
        self.model = self._parallize(model)

    def reset_dp_gap_size(self):
        self._dp_gap_size = 0

    def _parse_dp_config(self, config):
        if not hasattr(config, "dp_config"):
            return None, None
        dp_config = config.dp_config
        dp_index_or_key = dp_config.dp_index_or_key

        if isinstance(dp_index_or_key, int):
            assert dp_index_or_key >= 0
            return dp_config, True
        elif isinstance(dp_index_or_key, str):
            return dp_config, False
        else:
            raise NotImplementedError()

    def load_from_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        self.model.load_from_state_dict(state_dict, strict)

    @abstractmethod
    def _parallize(self, model: Module):
        pass

    def auto_tp(self, model):
        """
        Replace normal nerf model components with parallel components.
        """
        parallel_degree = self._parallel_config.tp_size
        tp_local_rank = CommContext().get_local_rank(comm_mode=ParallelMode.TENSOR_PARALLEL)
        tp_group = CommContext().get_group(comm_mode=ParallelMode.TENSOR_PARALLEL)

        if isinstance(model, InferenceModule):
            model.inter_model = _convert_module_by_parallel_mode(
                model.inter_model, "ChannelParallel", parallel_degree, tp_local_rank, tp_group, True
            )
        else:
            raise NotImplementedError()
        return model

    def preprocess(self, *args, **kwargs):
        return self.model.preprocess(*args, **kwargs)

    def postprocess(self, *args, **kwargs):
        return self.model.postprocess(*args, **kwargs)

    def dp_scatter(self, stage_input: StageInput):
        # TODO(liujun1): get tensor place from user dp_config
        first_ele = stage_input.Args[0]
        global_world_size = CommContext().get_world_size(comm_mode=ParallelMode.GLOBAL)
        dp_world_size = CommContext().get_world_size(comm_mode=ParallelMode.DATA_PARALLEL)
        process_num_between_dp_rank = global_world_size // dp_world_size
        local_part = CommContext().get_global_rank() // process_num_between_dp_rank
        # Nerf DP
        if isinstance(first_ele, torch.Tensor):
            tensor = first_ele
            part_size = tensor.shape[0] // dp_world_size
            gap = tensor.shape[0] % dp_world_size

            if gap > 0:
                self._dp_gap_size = dp_world_size - gap
                dp_gap_shape = (self._dp_gap_size,) + tensor.shape[1:]
                gap_tensor = torch.zeros(dp_gap_shape, device="cuda")
                tensor = torch.cat((tensor, gap_tensor), dim=0)
            tensor = tensor[local_part * part_size : (local_part + 1) * part_size]

            new_args = list(stage_input.Args)
            new_args[0] = tensor
            return tuple(new_args), stage_input.Kwargs
        else:
            raise NotImplementedError()

    def dp_gather(self, stage_input: StageInput):
        single_result_tensor = stage_input.Args[0]
        gathered_res_tensor = torch.Tensor([])
        dp_world_size = CommContext().get_world_size(comm_mode=ParallelMode.DATA_PARALLEL)
        global_world_size = CommContext().get_world_size(comm_mode=ParallelMode.GLOBAL)
        process_num_between_dp_rank = global_world_size // dp_world_size
        if dp_world_size == 1:
            return stage_input.Args, stage_input.Kwargs

        if not self._gs_model:
            out_shape = (single_result_tensor.shape[0] * dp_world_size,) + single_result_tensor.shape[1:]
            gathered_res_tensor = torch.empty(
                out_shape,
                dtype=single_result_tensor.dtype,
                device=single_result_tensor.device,
            )

            if CommContext().get_global_rank() % process_num_between_dp_rank == 0:
                all_gather(
                    gathered_res_tensor,
                    single_result_tensor,
                    comm_mode=ParallelMode.DATA_PARALLEL,
                )

            if self._dp_gap_size > 0:
                gathered_res_tensor = gathered_res_tensor[: -self._dp_gap_size, :]
                self.reset_dp_gap_size()

        new_args = list(stage_input.Args)
        new_args[0] = gathered_res_tensor
        return tuple(new_args), stage_input.Kwargs

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class TorchParallelModule(ParallelModule):
    def _parallize(self, model: Module):
        if hasattr(self, "_parallel_config") and self._parallel_config.tp_size > 1:
            model = self.auto_tp(model)
        return model


class KernelParallelModule(ParallelModule):
    def __init__(self, model: Module, config: Config, merged_state_dict) -> None:
        self.merged_state_dict = merged_state_dict
        super().__init__(model, config)

    def _parallize(self, model: Module):
        if hasattr(self, "_parallel_config") and self._parallel_config.tp_size > 1:
            model = self.auto_tp(model)
        model = OptimizationManager(self._inference_config, self.merged_state_dict).build(model)
        return model
