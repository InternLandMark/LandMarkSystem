from typing import List, Literal, Union

import torch

from .base_configclass import ConfigClass


class DatasetConfig(ConfigClass):
    """Dataset Configuration class for setting up dataset parameters.

    Args:
        partition (str): Specifies the dataset partition to use.
        datadir (str): Directory where the dataset is located.
        subfolder (list, optional): List of subfolders to include in the dataset. Defaults to None.
        dataset_name (Literal["city", "matrixcity", "blender", "Colmap"], optional):
            Name of the dataset. Defaults to "city".
        dataset_type (Literal["nerf", "gaussian"], optional): Type of the dataset. Defaults to "nerf".
        use_preprocessed_data (bool, optional): Flag to indicate if preprocessed data should be used. Defaults to False.
        processed_data_type (str, optional):
            Type of preprocessed data. Defaults to "file".
        preprocessed_dir (str, optional): Directory for preprocessed data.
            Defaults to "/cpfs01/shared/landmarks-viewer/preprocessed_data/".
        downsample_train (int, optional): Downsampling factor for training images. Defaults to 1.
        train_all_data (bool, optional): Flag to indicate if all data should be used for training. Defaults to False.
        img_uint8 (bool, optional): Flag to indicate if images are in uint8 format. Defaults to False.
        preload (bool, optional): Flag to indicate if data should be preloaded into memory. Defaults to False.
        dynamic_load (bool, optional): Flag to indicate if data should be dynamically loaded. Defaults to False.
        max_iter_each_block (int, optional): Maximum iterations per block for dynamic training. Defaults to -1.
        lb (Union[List[float], torch.Tensor], optional):
            Lower bound of the bounding box. Defaults to [0, 0, 0].
        ub (Union[List[float], torch.Tensor], optional):
            Upper bound of the bounding box. Defaults to [0, 0, 0].
        pose_rotation (Union[List[float], torch.Tensor], optional):
            Rotation part of the pose. Defaults to [0, 0, 0].
        pose_translation (Union[List[float], torch.Tensor], optional):
            Translation part of the pose. Defaults to [0, 0, 0].

    """

    partition: str = "all"
    datadir: str
    subfolder: list = None
    dataset_name: Literal["city", "matrixcity", "blender", "Colmap"] = "city"
    dataset_type: Literal["nerf", "gaussian"] = "nerf"
    use_preprocessed_data: bool = False
    processed_data_type: str = "file"
    preprocessed_dir: str = "/cpfs01/shared/landmarks-viewer/preprocessed_data/"
    downsample_train: int = 1
    train_all_data: bool = False
    img_uint8: bool = False
    preload: bool = False
    dynamic_load: bool = False
    max_iter_each_block: int = -1  # for dynamic training
    lb: Union[List[float], torch.Tensor] = [0, 0, 0]
    ub: Union[List[float], torch.Tensor] = [0, 0, 0]
    pose_rotation: Union[List[float], torch.Tensor] = [0, 0, 0]
    pose_translation: Union[List[float], torch.Tensor] = [0, 0, 0]

    def check_args(self):
        """Performs a sanity check on the dataset configuration."""

        super().check_args()

        assert (
            not self.dynamic_load or self.dataset_type == "gaussian"
        ), "dynamic_load is only supported for gaussian dataset currently."
        assert (
            not self.use_preprocessed_data or self.dataset_type == "nerf"
        ), "use_preprocessed_data is only supported for nerf dataset currently."

        if isinstance(self.lb, list):
            assert len(self.lb) == 3
        if isinstance(self.lb, torch.Tensor):
            assert self.lb.shape == (3,)
        if isinstance(self.ub, list):
            assert len(self.ub) == 3
        if isinstance(self.ub, torch.Tensor):
            assert self.ub.shape == (3,)
        if isinstance(self.pose_rotation, list):
            assert len(self.pose_rotation) == 3
        if isinstance(self.pose_rotation, torch.Tensor):
            assert self.pose_rotation.shape == (3,)
        if isinstance(self.pose_translation, list):
            assert len(self.pose_translation) == 3
        if isinstance(self.pose_translation, torch.Tensor):
            assert self.pose_translation.shape == (3,)
