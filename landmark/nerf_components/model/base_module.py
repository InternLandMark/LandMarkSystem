import os
from abc import abstractmethod
from typing import Any, Dict, Mapping, Union

from torch import nn


class BaseModule(nn.Module):
    """
    Desc:
        Base module for all model and component which need offload module and kernel module.
    """

    def __init__(self):
        super().__init__()
        self.merge_config = {}

    def set_merge_config(self, merge_config):
        for name, module in self.named_children():
            if isinstance(module, BaseModule):
                sub_merge_config = {}
                for key, value in merge_config.items():
                    if key.startswith(name):
                        sub_merge_config[key[len(name) + 1 :]] = value
                module.set_merge_config(sub_merge_config)

        if hasattr(self, "merge_config"):
            for key, value in merge_config.items():
                if key in self.merge_config:
                    self.merge_config[key] = value

    # TODO: can save the result for the next time.
    def get_merge_config(self, prefix=""):
        if hasattr(self, "merge_config"):
            merge_config = self.merge_config
        else:
            merge_config = {}

        for name, module in self.named_children():
            sub_prefix = prefix + name
            if isinstance(module, BaseModule):
                sub_merge_config = module.get_merge_config()
                for sub_key in sub_merge_config.keys():
                    new_key = sub_prefix + "." + sub_key
                    merge_config[new_key] = sub_merge_config[sub_key]

        return merge_config

    @staticmethod
    def get_kwargs() -> Mapping[str, Any]:
        """
        Get initialization parameters directly
        """
        raise NotImplementedError

    @staticmethod
    def get_kwargs_from_ckpt(
        file_path: Union[str, os.PathLike], map_location: Union[str, Dict[str, str]]
    ) -> Mapping[str, Any]:
        """
        Get kwargs from ckpt to initialize model class.

        Args:
            file_path: a string or os.PathLike object containing a file name
            map_location: a string or a dict specifying how to remap storage locations
        """
        raise NotImplementedError

    @abstractmethod
    def get_state_dict_from_ckpt(
        self, file_path: Union[str, os.PathLike], map_location: Union[str, Dict[str, str]], prefix: str = ""
    ) -> Mapping[str, Any]:
        """
        Get state_dict from the given file path.

        Args:
            file_path: a string or os.PathLike object containing a file name
            map_location: a string or a dict specifying how to remap storage locations
        """
        raise NotImplementedError

    @abstractmethod
    def load_from_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
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
        raise NotImplementedError

    def save_init_kwargs(self, local_dict):
        import inspect

        if hasattr(self.__init__, "_origin_init"):
            arg_key = inspect.getfullargspec(self.__init__._origin_init).args[1:]
        else:
            arg_key = inspect.getfullargspec(self.__init__).args[1:]

        self._init_kwargs = {}
        for key in arg_key:
            try:
                self._init_kwargs[key] = local_dict[key]
            except KeyError:
                pass

    def get_init_kwargs(self):
        return self._init_kwargs
