from typing import List, Optional, Union

import torch
from typing_extensions import Literal

from .base_configclass import ConfigClass


class TrainConfig(ConfigClass):
    """
    Train Configuration class for setting up training parameters.

    Args:
        model_name (str): Name of the model to be trained.
        expname (str): Experiment name.
        start_iters (int, optional): Starting iteration number. Defaults to 0.
        n_iters (int): Total number of iterations.
        batch_size (int, optional): Batch size for training. Defaults to 4096.
        progress_refresh_rate (int, optional): Refresh rate for progress updates. Defaults to 10.
        wandb (bool, optional): Flag to indicate if Weights & Biases logging is enabled. Defaults to False.
        tensorboard (bool, optional): Flag to indicate if TensorBoard logging is enabled. Defaults to False.
        basedir (str, optional): Base directory for logging. Defaults to "./log".
        add_timestamp (bool, optional):
            Flag to indicate if timestamp should be added to log filenames. Defaults to False.
        random_seed (int, optional): Random seed. Defaults to 20211202.
        debug (bool, optional): Flag to indicate if debug mode is enabled. Defaults to False.
        N_vis (int, optional): Number of images to visualize. Defaults to 5.
        vis_every (int, optional): Iteration interval for visualization. Defaults to 10000.
        skip_save_imgs (bool, optional): Flag to indicate if saving images is skipped. Defaults to False.
        optim_dir (Optional[str], optional): Directory for optimization checkpoints. Defaults to None.
        ckpt (Optional[str], optional): Path to a specific checkpoint. Defaults to None.
        ckpt_type (Literal["full", "sub"], optional): Type of checkpoint. Defaults to "full".
        kwargs (Optional[str], optional): Additional arguments for checkpointing. Defaults to None.
        is_train (bool, optional): Flag to indicate if the configuration is for training. Defaults to True.
        device (Union[str, torch.device], optional): Device to use for training. Defaults to torch.device("cuda").
        DDP (bool, optional): Flag to indicate if Distributed Data Parallel is used. Defaults to False.
        channel_parallel (bool, optional): Flag to indicate if channel parallelism is used. Defaults to False.
        branch_parallel (bool, optional): Flag to indicate if branch parallelism is used. Defaults to False.
        plane_division (Optional[list], optional): Division of planes for parallelism. Defaults to [1, 1].
        channel_parallel_size (Optional[int], optional):
            Size for channel parallelism. Defaults to None.
        model_parallel_and_DDP (bool, optional):
            Flag to indicate if model parallelism and DDP are combined. Defaults to False.
        test_iterations (List[int], optional): Iterations for testing. Defaults to [].
        save_iterations (List[int], optional): Iterations for saving checkpoints. Defaults to [].
        start_checkpoint (Optional[str], optional): Path to the starting checkpoint. Defaults to None.
        print_timestamp (bool, optional): Flag to indicate if timestamps should be printed. Defaults to True.
    """

    model_name: str
    expname: str
    start_iters: int = 0
    n_iters: int
    batch_size: int = 4096
    progress_refresh_rate: int = 10
    wandb: bool = False
    tensorboard: bool = False
    basedir: str = "./log"
    add_timestamp: bool = False
    random_seed: int = 20211202
    debug: bool = False
    N_vis: int = 5
    vis_every: int = 10000
    skip_save_imgs: bool = False
    optim_dir: Optional[str] = None
    ckpt: Optional[str] = None
    ckpt_type: Literal["full", "sub"] = "full"
    kwargs: Optional[str] = None

    is_train: bool = True
    device: Union[str, torch.device] = torch.device("cuda")

    DDP: bool = False
    channel_parallel: bool = False
    branch_parallel: bool = False

    plane_division: Optional[list] = [1, 1]
    channel_parallel_size: Optional[int] = None
    model_parallel_and_DDP: bool = False

    # TODO for 3d gaussian temporary (frank)
    # detect_anomaly: bool = False
    test_iterations: List[int] = []
    save_iterations: List[int] = []
    start_checkpoint: Optional[str] = None

    print_timestamp: bool = True

    def check_args(self):
        """Performs a sanity check on the training configuration."""
        pass
