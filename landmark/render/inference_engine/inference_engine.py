from types import MethodType

import torch
from torch.nn import Module

from landmark.communicator import (
    _broadcast_user_inputs_message,
    init_communication_context,
)
from landmark.render.parallel_module import KernelParallelModule, TorchParallelModule
from landmark.render.parameter_loader import OffloadModule
from landmark.render.scheduler.instructions import (
    BroadcastInputsInstruction,
    CallInstruction,
    DefModInstruction,
    DPGatherInstruction,
    DPScatterInstruction,
    PostprocessInstruction,
    PreprocessInstruction,
)
from landmark.render.scheduler.sches import gen_scheduler
from landmark.render.util import (
    InferenceModule,
    RuntimeType,
    StageInput,
    _bind_empty_preprocess_and_postprocess,
)
from landmark.utils import _OFFLOAD_CONFIG_STR, _PARALLEL_CONFIG_STR, Config


class InferenceEngine(Module):
    """
    Model inference
    """

    def __init__(self, model: Module, config: Config) -> None:
        super().__init__()

        self.model = model
        if not isinstance(self.model, InferenceModule):
            _bind_empty_preprocess_and_postprocess(self.model)
            self.base_model = self.model
        else:
            self.base_model = self.model.inter_model

        self._config = config
        self._parallel = hasattr(self._config, _PARALLEL_CONFIG_STR)
        self._model_initialized = False
        self._stage_input = StageInput()
        self._inference_output = None
        if self._parallel:
            init_communication_context(self._config.parallel_config)
        self.sche_list = gen_scheduler(config).steps()
        self._exec_init()

    def _exec_init(self):
        if self._model_initialized:
            return

        self.model.eval()

        if self._config.ckpt_path is not None:
            merged_state_dict = self.model.get_state_dict_from_ckpt(self._config.ckpt_path, "cpu")

        if hasattr(self._config, _OFFLOAD_CONFIG_STR):
            assert self._config.ckpt_path is not None
            self.model = OffloadModule(
                self.model, merged_state_dict=merged_state_dict, config=self._config.offload_config
            )
            self.model.load_from_state_dict(merged_state_dict)
        elif self._config.ckpt_path is not None:
            self.model.load_from_state_dict(merged_state_dict)

        runtime_type = RuntimeType(self._config.runtime)
        if runtime_type is RuntimeType.Kernel:
            self.model = KernelParallelModule(self.model, self._config, merged_state_dict)
            self.model.load_from_state_dict(merged_state_dict)
        elif runtime_type is RuntimeType.Torch:
            self.model = TorchParallelModule(self.model, self._config)
        else:
            raise NotImplementedError()

        # move user model on any device to current rank device
        self.model.to("cuda")

        self.model.eval()
        self._model_initialized = True

    @torch.no_grad()
    def _exec_forward(self):
        outputs = self.model.forward(*self._stage_input.Args, **self._stage_input.Kwargs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self._stage_input = StageInput(outputs, {})

    def _exec_broadcast_inputs(self):
        input_list = [self._stage_input.Args, self._stage_input.Kwargs]
        _broadcast_user_inputs_message(input_list)
        self._stage_input = StageInput(input_list[0], input_list[1])

    def _exec_dp_scatter(self):
        outputs = self.model.dp_scatter(self._stage_input)
        self._stage_input = StageInput(outputs[0], outputs[1])

    def _exec_dp_gather(self):
        outputs = self.model.dp_gather(self._stage_input)
        self._stage_input = StageInput(outputs[0], outputs[1])

    def _exec_preprocess(self):
        outputs = self.model.preprocess(*self._stage_input.Args, **self._stage_input.Kwargs)
        self._stage_input = StageInput(outputs[0], outputs[1])

    def _exec_postprocess(self):
        self._inference_output = self.model.postprocess(*self._stage_input.Args, **self._stage_input.Kwargs)

    _INSTRUCTION_MAP = {
        DefModInstruction: _exec_init,
        BroadcastInputsInstruction: _exec_broadcast_inputs,
        PreprocessInstruction: _exec_preprocess,
        DPScatterInstruction: _exec_dp_scatter,
        CallInstruction: _exec_forward,
        DPGatherInstruction: _exec_dp_gather,
        PostprocessInstruction: _exec_postprocess,
    }

    def _exec_sche(self, sche_list):
        for step in sche_list:
            if step not in self._INSTRUCTION_MAP:
                raise NotImplementedError(f"{repr(step)} not supported.")

            exec_func = MethodType(self._INSTRUCTION_MAP[step], self)
            exec_func()  # pylint: disable=E1102

    def forward(self, *args, **kwargs):
        self._stage_input = StageInput(args, kwargs)
        self._inference_output = None
        self._exec_sche(self.sche_list)
        return self._inference_output
