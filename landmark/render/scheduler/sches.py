from abc import abstractmethod
from functools import wraps
from typing import Callable

from landmark.utils import _PARALLEL_CONFIG_STR, Config, EnvSetting

from .instructions import (
    BroadcastInputsInstruction,
    CallInstruction,
    DefModInstruction,
    DPGatherInstruction,
    DPScatterInstruction,
    PipeGetResultInstruction,
    PipeSubmitInstruction,
    PostprocessInstruction,
    PreprocessInstruction,
)


class AbstractSche:
    @abstractmethod
    def steps(self):
        pass


class InferenceSche(AbstractSche):
    """
    inference scheduler on one cuda
    """

    def steps(self, rank=None):
        assert rank >= 0
        return [
            self.def_mod_step(),
            self.preprocess_step(),
            self.call_step(),
            self.postprocess_step(),
        ]

    def preprocess_step(self):
        return PreprocessInstruction

    def def_mod_step(self):
        return DefModInstruction

    def postprocess_step(self):
        return PostprocessInstruction

    def call_step(self):
        return CallInstruction


class MultiDeviceSche(InferenceSche):
    """
    Multi device scheduler
    """

    def steps(self, rank=None):
        steps = super().steps(rank)
        steps.insert(1, self.broadcast_inputs_step())
        return steps

    def broadcast_inputs_step(self):
        return BroadcastInputsInstruction


class DDPSche(MultiDeviceSche):
    """
    Data parallel scheduler
    """

    def steps(self, rank=None):
        steps = super().steps(rank)
        steps.insert(3, self.scatter_step())
        steps.insert(5, self.gather_step())
        return steps

    def scatter_step(self):
        return DPScatterInstruction

    def gather_step(self):
        return DPGatherInstruction


class DDPPipeOffloadSche(DDPSche):
    pass


class DDPPipeSche(DDPSche):
    """
    Data parallel pipe scheduler
    """

    def steps(self, rank=None):
        steps = [
            self.def_mod_step(),
            self.broadcast_inputs_step(),
            self.pipe_submit_step(),
            self.pipe_get_result_step(),
        ]
        return steps

    def pipe_submit_step(self):
        return PipeSubmitInstruction

    def pipe_get_result_step(self):
        return PipeGetResultInstruction


def steps_wrapper(fn: Callable, rank) -> Callable:
    @wraps(fn)
    def wrapped_fn():
        return fn(rank)

    return wrapped_fn


def gen_scheduler(config: Config):
    scheduler = None
    if hasattr(config, _PARALLEL_CONFIG_STR):
        dp_world_size = config.parallel_config.world_size // config.parallel_config.tp_size
        if dp_world_size > 1:
            if hasattr(config.parallel_config, "pipeline") and config.parallel_config.pipeline:
                scheduler = DDPPipeSche()
            elif not hasattr(config.parallel_config, "pipeline") or not config.parallel_config.pipeline:
                scheduler = DDPSche()
            else:
                raise NotImplementedError()
        elif config.parallel_config.world_size > 1:
            scheduler = MultiDeviceSche()
    else:
        scheduler = InferenceSche()

    scheduler.steps = steps_wrapper(scheduler.steps, EnvSetting.RANK)
    return scheduler
