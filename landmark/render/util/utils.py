import torch

from landmark.nerf_components.model.base_module import BaseModule


def transform_gridnerf_model_to_channel_last(model: BaseModule, ckpt_path, new_ckpt_path=None):
    """tool to generate the channel last ckpt

    Args:
        model (BaseModule): the origin model.
        ckpt_path (string): a path which saved the state dict.
        new_ckpt_path (string): a path to save the new state dict. Defaults to None.
    """
    model.channel_last()
    if new_ckpt_path is None:
        new_ckpt_path = ckpt_path[:-3] + "_channel_last" + ".th"
    state_dict = model.state_dict()
    print(f"save channel last ckpt in {new_ckpt_path}")
    torch.save(state_dict, new_ckpt_path)
