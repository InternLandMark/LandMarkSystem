import os
from typing import Any, Dict, Mapping, Optional, Union

from torch import nn

from landmark.nerf_components.scene import SceneManager


def preprocess(*args, **kwargs):
    return args, kwargs


def postprocess(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0:
        return args[0]
    return tuple(list(args).append(kwargs)) if len(kwargs) > 0 else args


def _bind_empty_preprocess_and_postprocess(model):
    assert not (hasattr(model, "preprocess") or hasattr(model, "postprocess"))
    setattr(model, "preprocess", preprocess)
    setattr(model, "postprocess", postprocess)


class InferenceInterface(nn.Module):
    """
    The defination of landmark inference interface
    """

    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, *args, **kwargs):
        return args, kwargs

    def postprocess(self, *args, **kwargs):
        return args, kwargs

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class InferenceModule(InferenceInterface):
    """Abstract inference module for inference engine adaptation

    Args:
        model (Optional[nn.Module]): internal model which is composed of the nerf components
    """

    def __init__(self, model: Optional[nn.Module] = None) -> None:
        """Initializes the InferenceModule class.

        Args:
            model (Optional[nn.Module]): internal model which is composed of the nerf components. Defaults to None.
        """
        super().__init__()
        self.model = model
        self.scene_mgr = None

    @property
    def inter_model(self):
        """Returns the internal model."""
        return self.model

    @inter_model.setter
    def inter_model(self, model):
        """set the internal model.

        Args:
            model (Optional[nn.Module]): internal model which is composed of the nerf components.
        """
        self.model = model

    @property
    def scene_manager(self):
        """Returns the scene manager."""
        if self.scene_mgr is None:
            for attr_name in vars(self.model):
                attr_value = getattr(self.model, attr_name)
                if isinstance(attr_value, SceneManager):
                    self.scene_mgr = attr_value
        return self.scene_mgr

    @property
    def merge_config(self):
        """Returns the offload merge config."""
        return self.model.get_merge_config()

    def channel_last(self):
        """Set the model in channel last mode."""
        self.model.channel_last()

    def get_state_dict_from_ckpt(
        self, file_path: Union[str, os.PathLike], map_location: Union[str, Dict[str, str]]
    ) -> Mapping[str, Any]:
        """Get state_dict from the given file path.

        Args:
            file_path (Union[str, os.PathLike]): a string or os.PathLike object containing a file name.
            map_location (Union[str, Dict[str, str]]): a string or a dict specifying how to remap storage locations.

        Returns:
            Mapping[str, Any]: the model's state dict
        """
        return self.model.get_state_dict_from_ckpt(file_path, map_location)

    def load_from_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        """
        Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys in state_dict
                            match the keys returned by this module's key
        """
        self.model.load_from_state_dict(state_dict, strict)

    def preprocess(self, *args, **kwargs):
        """
        Define the engine's preprocessing logic. By default, it returns the inputs.

        Args:
            args (list): a list containing the input parameters of preprocess.
            kwargs (dict): a dict containing input parameters of preprocess.

        Returns:
            Tuple[list, dict]: containing the output of the model preprocess.
        """
        if self.model is None:
            return super().preprocess(*args, **kwargs)
        return self.model.preprocess(*args, **kwargs)

    def postprocess(self, *args, **kwargs):
        """
        Define the engine's postprocess logic. By default, it returns the inputs.

        Args:
            args (list): a list containing the input parameters of postprocess.
            kwargs (dict): a dict containing input parameters of postprocess.

        Returns:
            Tuple[list, dict]: containing the output of the model postprocess.
        """
        if self.model is None:
            return super().postprocess(*args, **kwargs)
        return self.model.postprocess(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Define the engine's forward logic. By default, it uses the forward logic of internal model.

        Args:
            args (list): a list containing the input parameters of forward.
            kwargs (dict): a dict containing input parameters of forward.

        Returns:
            Tuple[list, dict]: containing the output of the model forward.
        """
        assert self.model is not None
        return self.model(*args, **kwargs)
