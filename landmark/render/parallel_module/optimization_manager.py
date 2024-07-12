from torch.nn import Module

from landmark.nerf_components.components_convertor.fusion import (
    _convert_components_by_kernel_runtime,
    _filter_dict_by_strings,
)
from landmark.utils import Config


class OptimizationManager:
    """
    Inference optimization(no parallelism) manager
    """

    def __init__(self, config: Config, state_dict) -> None:
        self._config = config
        self._state_dict = state_dict

    def opti_num(self):
        return int(self._config.kernel_fusion)

    def _build_half_precision(self, model: Module):
        # TODO(liujun): half precision
        return model

    def _build_static_allocation(self, model: Module):
        return model

    def _build_kernel_fusion(self, model: Module):
        """
        Replace normal nerf model components with fused kernel components.
        """
        trans_comp = []
        _convert_components_by_kernel_runtime(model, False, trans_comp)
        for i in range(len(trans_comp)):
            while trans_comp[i].startswith("model."):
                trans_comp[i] = trans_comp[i][len("model.") :]
        real_state_dict = self._state_dict["state_dict"] if "state_dict" in self._state_dict else self._state_dict
        _filter_dict_by_strings(real_state_dict, trans_comp)
        return model

    def build(self, model: Module):
        optimizations = []
        if self._config.kernel_fusion:
            optimizations.append(self._build_kernel_fusion)
        if hasattr(self._config, "static_allocation") and self._config.static_allocation:
            optimizations.append(self._build_static_allocation)

        if len(optimizations) == 0:
            return model

        for opt in optimizations:
            model = opt(model)
        return model
