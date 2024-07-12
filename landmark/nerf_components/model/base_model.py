import os
from abc import abstractmethod
from typing import Any, Dict, Mapping, Union

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .base_module import BaseModule


class Model(BaseModule):
    """
    A basic model class for user-defined models, providing functionality for configuration management,
    scene management, and handling of additional keyword arguments.

    This class is designed to be a flexible foundation for building complex models, allowing for easy
    extension and customization.

    Attributes:
        config (Config): Configuration object containing model settings.
        scene_manager (SceneManager): Manager for scene information.
        _kwargs_keys (list): Keys of additional keyword arguments passed during initialization.

    Methods:
        kwargs(): Returns a dictionary of additional keyword arguments.
        state_dict(destination=None, prefix="", keep_vars=False): Returns the state dictionary of the model.
        save_kwargs(file_path): Saves the keyword arguments to a file.
    """

    def __init__(self, config=None, scene_manager=None, **kwargs):
        """
        Initializes the Model with configuration, scene manager, and additional keyword arguments.

        Parameters:
            config (Config): The configuration object for the model.
            scene_manager (SceneManager): The scene manager for handling scene information.
            **kwargs: Arbitrary keyword arguments that are saved as attributes of the model.
        """
        super().__init__()
        self.config = config
        self.scene_manager = scene_manager
        self._kwargs_keys = []
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
            self._kwargs_keys = kwargs.keys()

    def kwargs(self):
        """
        Returns a dictionary of the model's keyword arguments.

        Returns:
            dict: A dictionary containing the keyword arguments.
        """
        _kwargs = {}
        for key in self._kwargs_keys:
            value = getattr(self, key)
            _kwargs[key] = value
        return _kwargs

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Returns the state dictionary of the model, potentially modified by the destination and prefix.

        Parameters:
            destination: The destination for saving the state dictionary.
            prefix (str): A prefix to add to each key in the state dictionary.
            keep_vars (bool): Whether to keep variables that are not part of the model.

        Returns:
            dict: The modified state dictionary.
        """
        state_dict = super().state_dict(destination, prefix, keep_vars)

        if prefix == "module.":
            for key in list(state_dict.keys()):
                if key.startswith(prefix):
                    new_key = key[len(prefix) :]
                    state_dict[new_key] = state_dict.pop(key)  # pylint:disable=E1137

        for name, module in self.named_modules():
            if isinstance(module, DDP):
                for key in list(state_dict.keys()):
                    if key.startswith(f"{name}.module."):
                        new_key = f"{name}." + key[len(f"{name}.module.") :]
                        state_dict[new_key] = state_dict.pop(key)  # pylint:disable =E1137
        return state_dict

    def save_kwargs(self, file_path: Union[str, os.PathLike]):
        kwargs = self.kwargs()
        assert len(kwargs) > 0
        torch.save(kwargs, file_path)

    def save_state_dict(self, file_path: Union[str, os.PathLike]):
        """
        Saves the state dictionary of the model to a file.

        This method serializes the state dictionary of the model and saves it to the specified file path.
        It is useful for checkpointing models during training or for later restoration.

        Parameters:
            file_path (Union[str, os.PathLike]): The path to the file where the state dictionary will be saved.

        Raises:
            AssertionError: If the state dictionary is empty, indicating there is nothing to save.
        """
        state_dict = self.state_dict()
        assert len(state_dict) > 0
        torch.save(state_dict, file_path)

    def save_model(self, dir_path: Union[str, os.PathLike], rank=None):
        """
        Saves both the state dictionary and keyword arguments of the model to files in the specified directory.

        This method is a convenience function that saves the model's state dictionary and any additional
        keyword arguments to separate files within a given directory. It supports saving multiple copies
        of the model for different ranks in distributed training scenarios.

        Parameters:
            dir_path (Union[str, os.PathLike]): The directory path where the model files will be saved.
            rank (Optional[int]): The rank of the process in distributed training. If provided, the method
                                  saves the files with a suffix indicating the rank.

        Raises:
            AssertionError: If there are no keyword arguments to save, indicating an empty keyword argument list.
        """
        os.makedirs(dir_path, exist_ok=True)
        state_dict_name = "state_dict.th" if rank is None else f"state_dict-sub{rank}.th"
        kwargs_name = "kwargs.th" if rank is None else f"kwargs-sub{rank}.th"
        if len(self._kwargs_keys) > 0:
            self.save_kwargs(os.path.join(dir_path, kwargs_name))
        self.save_state_dict(os.path.join(dir_path, state_dict_name))

    def get_state_dict_from_ckpt(
        self, file_path: Union[str, os.PathLike], map_location: Union[str, Dict[str, str]]
    ) -> Mapping[str, Any]:
        """
        Get state_dict from the given file path.

        Args:
            file_path: a string or os.PathLike object containing a file name
            map_location: a string or a dict specifying how to remap storage locations
        """
        assert os.path.exists(file_path)
        state_dict = torch.load(file_path, map_location=map_location)
        return state_dict

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

        def kwargs_tensors_to_device(kwargs, device):
            # move the tensors in kwargs to target device
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    kwargs_tensors_to_device(value, device)
                elif isinstance(value, torch.Tensor):
                    kwargs[key] = value.to(device)

        assert os.path.exists(file_path)
        kwargs = torch.load(file_path, map_location=map_location)
        if "device" in kwargs:
            kwargs["device"] = map_location
        kwargs_tensors_to_device(kwargs, map_location)
        return kwargs

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
        return super().load_state_dict(state_dict, strict)


class BaseNeRF(Model):
    """Base Class for volumetric rendered NeRF model"""

    @abstractmethod
    def init_sampler(self):
        raise NotImplementedError

    @abstractmethod
    def init_field_components(self):
        raise NotImplementedError

    @abstractmethod
    def init_renderer(self):
        raise NotImplementedError

    @torch.no_grad()
    def render_all_rays(self, rays, chunk_size=None, N_samples=-1, idxs=None, app_code=None):  # pylint: disable=W0613
        all_ret = {"rgb_map": [], "depth_map": []}
        N_rays_all = rays.shape[0]
        for chunk_idx in range(N_rays_all // chunk_size + int(N_rays_all % chunk_size > 0)):
            rays_chunk = rays[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
            idxs_chunk = idxs[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
            rays_chunk = torch.cat((rays_chunk, idxs_chunk.unsqueeze(-1)), dim=-1)

            ret = self.forward(
                rays_chunk,  # [bs, 7]: ori + dir + idx
            )
            all_ret["rgb_map"].append(ret["rgb_map"])
            all_ret["depth_map"].append(ret["depth_map"])

        all_ret = {k: torch.cat(v, 0) for k, v in all_ret.items()}
        return all_ret


class BaseGaussian(Model):
    """Base Class for Gaussian based model"""

    @abstractmethod
    def init_field_components(self):
        raise NotImplementedError

    @abstractmethod
    def init_renderer(self):
        raise NotImplementedError

    def _find_gaussian_encoding(self):
        from landmark.nerf_components.model_components.fields.encodings.base_encoding import (
            BaseGaussianEncoding,
        )

        gaussian_encoding = None
        gaussian_encoding_name = ""
        for name, module in self.named_modules():
            if isinstance(module, BaseGaussianEncoding):
                gaussian_encoding = module
                gaussian_encoding_name = name
        assert gaussian_encoding is not None
        return gaussian_encoding_name, gaussian_encoding

    def save_model(self, dir_path: Union[str, os.PathLike], rank=None):
        """
        Saves the model's state and point cloud data to the specified directory.

        This method extends the base model's save functionality by also saving the point cloud data associated
        with the model. It first saves the point cloud data using the `save_ply` method and then calls the base
        class's `save_model` method to save the model's state.

        Parameters:
            dir_path (Union[str, os.PathLike]): The directory path where the model and point cloud data will be saved.
            rank (Optional[int]): The rank of the process in distributed training. If provided, the method
                                  saves the files with a suffix indicating the rank.
        """
        os.makedirs(dir_path, exist_ok=True)
        self.save_ply(dir_path, rank)
        super().save_model(dir_path, rank)

    def save_ply(self, dir_path: Union[str, os.PathLike], rank=None):
        """
        Saves the point cloud data to a PLY file in the specified directory.

        This method finds the Gaussian encoding module within the model, uses it to generate the point cloud
        data, and saves this data to a PLY file. The file name is adjusted based on the rank in distributed training.

        Parameters:
            dir_path (Union[str, os.PathLike]): The directory path where the point cloud data will be saved.
            rank (Optional[int]): The rank of the process in distributed training. If provided, the method
                                  saves the files with a suffix indicating the rank.
        """
        ply_name = "point_cloud.ply" if rank is None else f"point_cloud-sub{rank}.ply"
        file_path = os.path.join(dir_path, ply_name)
        _, gaussian_encoding = self._find_gaussian_encoding()
        gaussian_encoding.save_ply(file_path)

    def save_state_dict(self, file_path: Union[str, os.PathLike]):
        """
        Saves the model's state dictionary to a file, excluding the Gaussian encoding state.

        This method modifies the base class's `save_state_dict` method by excluding the state of the Gaussian
        encoding module from the saved state dictionary. This is useful for scenarios where the Gaussian encoding
        state does not need to be saved or restored, such as when the encoding is specific to the training process
        and not required for inference.

        Parameters:
            file_path (Union[str, os.PathLike]): The path to the file where the state dictionary will be saved.

        Note:
            This method directly modifies the state dictionary before saving it, removing any keys that start
            with the name of the Gaussian encoding module. This ensures that the saved state dictionary does not
            include the Gaussian encoding state.
        """
        name, _ = self._find_gaussian_encoding()
        state_dict = self.state_dict()
        for key, _ in list(state_dict.items()):
            if key.startswith(name):
                state_dict.pop(key)
        if len(state_dict) > 0:
            torch.save(state_dict, file_path)

    def get_state_dict_from_ckpt(
        self, file_path: Union[str, os.PathLike], map_location: Union[str, Dict[str, str]]
    ) -> Mapping[str, Any]:
        """
        Retrieves and merges the state dictionary from a checkpoint file, excluding the Gaussian encoding state.

        This method is designed to load the state dictionary from a checkpoint file, specifically excluding the
        state of the Gaussian encoding module. It then attempts to merge this state dictionary with an existing
        state dictionary saved in the same directory as the checkpoint file. If duplicate keys are found during
        the merge, it raises a ValueError.

        Parameters:
            file_path (Union[str, os.PathLike]): The path to the checkpoint file from which to load the state
            dictionary.
            map_location (Union[str, Dict[str, str]]): The location where the model should be mapped if it is
                                                       loaded on a different device than it was saved on.

        Returns:
            Mapping[str, Any]: The merged state dictionary, excluding the Gaussian encoding state.

        Raises:
            ValueError: If duplicate keys are found between the loaded state dictionary and any existing state
                        dictionary in the same directory.
        """
        name, gaussian_encoding = self._find_gaussian_encoding()
        prefix = name + "."
        ply_state_dict = gaussian_encoding.get_state_dict_from_ckpt(file_path, map_location, prefix)
        state_dict_path = os.path.join(os.path.dirname(file_path), "state_dict.th")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path)
            duplicate_keys = set(ply_state_dict.keys()) & set(state_dict.keys())
            if duplicate_keys:
                raise ValueError(f"Duplicate keys found: {duplicate_keys}")
            ply_state_dict.update(state_dict)
        return ply_state_dict

    def load_from_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """
        Loads the model's state dictionary from a given mapping, excluding the Gaussian encoding state.

        This method loads the state dictionary into the model, specifically excluding the state of the Gaussian
        encoding module. It first extracts and removes any keys related to the Gaussian encoding from the provided
        state dictionary. Then, it calls the base class's `load_from_state_dict` method to load the remaining state
        dictionary. Finally, it loads the Gaussian encoding state using its own `load_from_state_dict` method.

        Parameters:
            state_dict (Mapping[str, Any]): The state dictionary to load into the model.
            strict (bool, optional): If True, the method will raise an error if keys in the state dictionary do not
                                     match the model's keys. Defaults to True.

        Note:
            This method modifies the input `state_dict` by removing keys related to the Gaussian encoding before
            passing it to the base class's method. It also prints a warning if no keys related to the Gaussian encoding
            are found in the provided state dictionary.
        """
        name, gaussian_encoding = self._find_gaussian_encoding()
        prefix = name + "."
        gs_state_dict = {}
        for key, _ in list(state_dict.items()):
            if key.startswith(prefix):
                gs_state_dict[key[len(prefix) :]] = state_dict.pop(key)
        if len(gs_state_dict) == 0:
            print(f"WRANING: could not found any key start with {prefix} in provided state_dict")
        super().load_from_state_dict(state_dict, strict)
        gaussian_encoding.load_from_state_dict(gs_state_dict, strict)
