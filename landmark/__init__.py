from pathlib import Path
from typing import Dict, Union

from landmark.communicator import init_communication_context
from landmark.nerf_components.model.base_model import Model
from landmark.render import (
    InferenceEngine,
    InferenceInterface,
    InferenceModule,
    InferenceModuleInitializer,
)
from landmark.utils import (
    _OFFLOAD_CONFIG_STR,
    _PARALLEL_CONFIG_STR,
    _RANK_STR,
    _WORLD_SIZE_STR,
    Config,
    EnvSetting,
)


def _transform_to_cls(config: Union[str, Path, Config, Dict]):
    if not isinstance(config, Config) and isinstance(config, dict):
        config = Config(config)
    if isinstance(config, (str, Path)):
        config = Config.from_file(config)
    return config


def _update_global_info(parallel_config: Config):
    rank = EnvSetting.RANK
    world_size = EnvSetting.WORLD_SIZE

    if _RANK_STR not in parallel_config or parallel_config[_RANK_STR] != rank:
        parallel_config.update({_RANK_STR: rank})
    if _WORLD_SIZE_STR not in parallel_config or parallel_config[_WORLD_SIZE_STR] != world_size:
        parallel_config.update({_WORLD_SIZE_STR: world_size})
    parallel_config = Config(parallel_config)

    return parallel_config


def _launch(parallel_config: Union[str, Path, Config, Dict]):
    parallel_config = _update_global_info(parallel_config)
    init_communication_context(parallel_config)


def _infer_config_sanity_check(model_config, inference_config):
    if hasattr(model_config, "act_cache") and hasattr(inference_config, _OFFLOAD_CONFIG_STR):
        assert model_config.act_cache is False, "Offload is not supportted when act_cache is True"
    if hasattr(inference_config, _OFFLOAD_CONFIG_STR) and hasattr(inference_config, _PARALLEL_CONFIG_STR):
        assert not (
            inference_config[_PARALLEL_CONFIG_STR].tp_size > 1
        ), "Offload is not supportted when tp_size > 1, please set tp_size to 1 or disabled tensor parallel"
    if hasattr(inference_config, "half_precision"):
        assert inference_config.half_precision is False, "Half precision is not supportted yet"
    if hasattr(inference_config, "static_allocation"):
        assert inference_config.static_allocation is False, "Static Allocation is not supportted yet"


def init_inference(
    based_model_cls: Model,
    infer_model_cls: InferenceModule,
    model_config: Union[str, Path, Config, Dict],
    inference_config: Union[str, Path, Config, Dict],
):
    """
    Initialize the inference engine.

    Args:
        based_model_cls (Model): a model which is composed of the nerf components.
        infer_model_cls (InferenceModule): an inference module, which is composed of a model,
            defines the preprocess, forward and postprocess functions.
        model_config (Union[str, Path, Config, Dict]): used to define the model parameters.
        inference_config (Union[str, Path, Config, Dict]): used to define the inference engine.

    Returns:
        InferenceEngine
    """
    inference_config = _transform_to_cls(inference_config)
    model_config = _transform_to_cls(model_config)

    if _PARALLEL_CONFIG_STR in inference_config:
        _launch(inference_config[_PARALLEL_CONFIG_STR])

    _infer_config_sanity_check(model_config, inference_config)

    initializer = InferenceModuleInitializer(based_model_cls, infer_model_cls)
    model = initializer.get_infer_model(model_config, inference_config)

    engine = InferenceEngine(model, inference_config)
    return engine


def init_training():
    raise NotImplementedError()


__all__ = ["init_inference", "InferenceModule", "InferenceInterface"]
