from abc import ABC, abstractmethod

from torch import nn

from landmark.nerf_components.configs import BaseConfig

from .fusion import (
    _convert_components_by_kernel_runtime,
    _filter_dict_by_strings,
    _rewarp_ddp_module,
)
from .parallelize import _convert_module_by_parallel_mode


class ComponentConvertorFactory:
    """
    A factory class for creating converter instances based on the specified type.

    This class supports creating converters for parallelizing components or converting
    torch components to kernel components, among other potential future extensions.
    """

    @staticmethod
    def get_convertor(convert_type):
        """
        Creates and returns a converter instance based on the specified type.

        Args:
            convert_type (str): The type of converter to create. Expected values are
                                "parallelize" for ParallelConvertor or "runtime" for
                                RuntimeConvertor.

        Returns:
            An instance of either ParallelConvertor or RuntimeConvertor based on the
            input argument.

        Raises:
            ValueError: If an unknown convert type is specified.
        """
        if convert_type == "parallelize":
            return ParallelConvertor()
        elif convert_type == "runtime":
            return RuntimeConvertor()
        else:
            raise ValueError(f"Unknown convert type {convert_type}.")


class ComponentsConvertorBase(ABC):
    """
    An abstract base class for component converters.

    This class defines the interface for all converters, requiring the implementation
    of a convert method that transforms models according to specific configurations.
    """

    def __init__(self):
        pass

    @abstractmethod
    def convert(self, model: nn.Module, config: BaseConfig, state_dict=None, verbose: bool = True) -> nn.Module:
        """
        Converts a given model according to the specified configuration.

        Args:
            model (nn.Module): The model to be converted.
            config (BaseConfig): The configuration specifying how the conversion
                                 should be performed.
            state_dict (dict, optional): The state dictionary of the model, if available.
            verbose (bool, optional): If True, print verbose messages during conversion.

        Returns:
            nn.Module: The converted model.
        """
        pass


class ParallelConvertor(ComponentsConvertorBase):
    """
    Converts sequential components into parallel components.

    This converter specifically handles the conversion of models to utilize parallel
    processing capabilities, such as branch or channel parallelism.
    """

    def convert(self, model: nn.Module, config: BaseConfig, state_dict=None, verbose: bool = True):
        if verbose:
            print("Converting model to parallel model...")
        if config.branch_parallel:
            parallel_mode = "BranchParallel"
            plane_division = config.plane_division
            parallel_degree = plane_division[0] * plane_division[1]
        elif config.channel_parallel:
            parallel_mode = "ChannelParallel"
            parallel_degree = config.channel_parallel_size
        else:
            raise NotImplementedError("Only support branch parallel currently.")
        model = _convert_module_by_parallel_mode(
            model, parallel_mode, parallel_degree, config.mp_rank, config.mp_group, verbose
        )
        return model


class RuntimeConvertor(ComponentsConvertorBase):
    """
    Converts torch components to kernel components.

    This converter is responsible for transforming torch models into a format that
    utilizes custom kernel operations, potentially for performance optimizations.
    """

    def convert(self, model: nn.Module, config: BaseConfig, state_dict=None, verbose: bool = True):
        trans_comp = []
        if verbose:

            print("Converting torch model to fusion model...")

        if config.runtime == "Kernel":
            _convert_components_by_kernel_runtime(model, verbose, trans_comp)
            _rewarp_ddp_module(model, trans_comp)
        else:
            print(f"The [{config.runtime}] runtime is not found and fallback to torch model.")

        if state_dict is not None:
            real_state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            _filter_dict_by_strings(real_state_dict, trans_comp)

        return model
