from landmark.nerf_components.model import BaseGaussian, BaseNeRF
from landmark.nerf_components.scene import SceneManager
from landmark.utils import _OFFLOAD_CONFIG_STR


class InferenceModuleInitializer:
    """Class for initialing the user defined inference module"""

    def __init__(self, based_model_cls, infer_model_cls):
        self.based_model_cls = based_model_cls
        self.infer_model_cls = infer_model_cls

    def get_infer_model(self, model_config, inference_config):
        if model_config is not None:
            assert model_config.ckpt is not None, "should set the checkpoint file path"
            if hasattr(inference_config, "ckpt_path") and inference_config.ckpt_path is not None:
                assert inference_config.ckpt_path == model_config.ckpt
            else:
                inference_config.ckpt_path = model_config.ckpt
            kwargs = (
                self.based_model_cls.get_kwargs_from_ckpt(model_config.kwargs, model_config.device)
                if hasattr(model_config, "kwargs") and model_config.kwargs is not None
                else {}
            )
            kwargs["config"] = model_config
            if _OFFLOAD_CONFIG_STR in inference_config and issubclass(self.based_model_cls, BaseGaussian):
                assert False, "not support offload in Gaussian models"
            kwargs["scene_manager"] = SceneManager(model_config)
            if "device" in kwargs:
                kwargs["device"] = "cpu"
        else:
            kwargs = self.based_model_cls.get_kwargs()

        if issubclass(self.based_model_cls, BaseNeRF):
            model = (
                self.infer_model_cls(**kwargs) if self.infer_model_cls is not None else self.based_model_cls(**kwargs)
            )
        elif issubclass(self.based_model_cls, BaseGaussian):
            model = (
                self.infer_model_cls(**kwargs) if self.infer_model_cls is not None else self.based_model_cls(**kwargs)
            )
        else:
            raise NotImplementedError(f"Unknown support model type {self.based_model_cls}")
        return model
