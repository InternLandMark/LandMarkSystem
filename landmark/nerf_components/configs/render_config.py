from typing import List, Optional, Union

import torch
from typing_extensions import Literal

from .base_configclass import ConfigClass
from .train_config import TrainConfig


class RenderConfig(ConfigClass):
    """Render Config

    Args:
        model_name (str): The name of the model.
        expname (str): The name of the experiment.
        batch_size (int): The size of the batch for processing.
        basedir (str): The base directory for logging and saving outputs.
        add_timestamp (bool): Whether to add a timestamp to the output filenames.
        random_seed (int): The seed for random number generation to ensure reproducibility.
        debug (bool): Enables debug mode if set to True.
        N_vis (int): Number of visualizations to generate.
        ckpt (Optional[str]): Path to a checkpoint file.
        ckpt_type (Literal["full", "sub"]): Type of checkpoint, either 'full' or 'sub'.
        kwargs (Optional[str]): Additional keyword arguments.
        is_train (bool): Flag indicating whether the configuration is for training.
        distributed (bool): Flag indicating whether distributed training is enabled.
        device (Union[str, torch.device]): The computing device to use ('cuda' or 'cpu').
        DDP (bool): Flag indicating whether Distributed Data Parallel (DDP) is used.
        channel_parallel (bool): Flag indicating whether channel parallelism is used.
        branch_parallel (bool): Flag indicating whether branch parallelism is used.
        plane_division (Optional[list]): Specifies how to divide the input data into planes.
        channel_parallel_size (Optional[int]): Size of the channel parallelism.
        model_parallel_and_DDP (bool): Flag indicating whether model parallelism and DDP are combined.
        test_iterations (List[int]): List of iterations at which to perform testing.
        save_iterations (List[int]): List of iterations at which to save outputs.
        quiet (bool): If True, reduces the verbosity of the output.
        start_checkpoint (Optional[str]): Path to the starting checkpoint for resuming training.
        compute_extra_metrics (bool): Flag indicating whether to compute additional metrics.

    """

    model_name: str = None
    expname: str = None
    batch_size: int = 4096
    basedir: str = "./log"
    add_timestamp: bool = False
    random_seed: int = 20211202
    debug: bool = False
    N_vis: int = 5
    ckpt: Optional[str] = None
    ckpt_type: Literal["full", "sub"] = "full"
    kwargs: Optional[str] = None
    is_train: bool = False
    distributed: bool = False
    device: Union[str, torch.device] = torch.device("cuda")
    DDP: bool = False
    channel_parallel: bool = False
    branch_parallel: bool = False
    plane_division: Optional[list] = None
    channel_parallel_size: Optional[int] = None
    model_parallel_and_DDP: bool = False
    test_iterations: List[int] = []
    save_iterations: List[int] = []
    quiet: bool = False
    start_checkpoint: Optional[str] = None
    compute_extra_metrics: bool = False

    def from_train_config(self, train_config: TrainConfig):
        """Updates the configuration based on a given TrainConfig instance.

        Args:
            train_config (TrainConfig): The training configuration to update this configuration with.

        """
        # TODO add warning when having conflict settings (frank)
        self.model_name = train_config.model_name if self.model_name is None else self.model_name
        self.expname = train_config.expname if self.expname is None else self.expname
        self.basedir = train_config.basedir if self.basedir is None else self.basedir
        self.add_timestamp = train_config.add_timestamp if self.add_timestamp is None else self.add_timestamp
        self.random_seed = train_config.random_seed if self.random_seed is None else self.random_seed
        self.debug = train_config.debug if self.debug is None else self.debug
        self.N_vis = train_config.N_vis if self.N_vis is None else self.N_vis
        self.device = train_config.device if self.device is None else self.device
        self.quiet = train_config.quiet if self.quiet is None else self.quiet

        self.plane_division = train_config.plane_division if self.plane_division is None else self.plane_division

        return self

    def check_args(self):
        """This method ensures that the configuration is valid for rendering"""
        assert not self.is_train, "RenderConfig is not for training"
