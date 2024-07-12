from typing import Optional, Union

import torch

from .utils.tensor_dataclass import TensorDataClass


class Rays(TensorDataClass):
    """Class for rays"""

    def __init__(
        self,
        origin: torch.Tensor,
        dirs: torch.Tensor,
        camera_idx: torch.Tensor,
        near: float,
        far: float,
        device: Optional[Union[torch.device, str]] = None,
        rank: int = None,
        group=None,
        _data: Optional[torch.Tensor] = None,
    ):
        super().__init__(device, rank, group, _data)
        if _data is not None:
            assert origin is None, "origin must be None when constructing from a '_data' tensor."
            assert dirs is None, "dirs must be None when constructing from a '_data' tensor."
            assert camera_idx is None, "camera_idx must be None when constructing from a '_data' tensor."

        else:
            # shape check
            self._shape = origin.shape[:-1]
            shape_dim = len(self._shape)
            assert dirs.shape[:shape_dim] == self._shape and dirs.dim() == shape_dim + 1, (
                f"The front shape of origin and dirs must be the same, but got dirs shape: {dirs.shape[:-1]} and origin"
                f" shape: {origin.shape[:-1]}."
            )
            assert camera_idx.shape[:shape_dim] == self._shape, (
                "The front shape of origin and camera_indice must be the same, but got camera_idx shape:"
                f" {camera_idx.shape[:-1]} and origin shape: {origin.shape[:-1]}."
            )
            if camera_idx.dim() == shape_dim:
                camera_idx = camera_idx.unsqueeze(-1)

            self._data = torch.cat((origin, dirs, camera_idx), dim=-1)

        self.near = near
        self.far = far

    @property
    def origin(self):
        return self._data[..., :3]

    @property
    def dirs(self):
        return self._data[..., 3:6]

    @property
    def camera_idx(self):
        return self._data[..., 6].long()

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)

        if device == self.device:
            return self

        _data = self._data.to(device)
        return Rays(
            origin=None,
            dirs=None,
            camera_idx=None,
            near=self.near,
            far=self.far,
            _data=_data,
            rank=self.rank,
            group=self.group,
        )

    def view(self, *shape):
        _data = super().view(*shape)
        return Rays(
            origin=None,
            dirs=None,
            camera_idx=None,
            near=self.near,
            far=self.far,
            _data=_data,
            rank=self.rank,
            group=self.group,
        )

    def reshape(self, *shape):
        _data = super().reshape(*shape)
        return Rays(
            origin=None,
            dirs=None,
            camera_idx=None,
            near=self.near,
            far=self.far,
            _data=_data,
            rank=self.rank,
            group=self.group,
        )

    def __getitem__(self, item):
        _data = super().__getitem__(item)
        return Rays(
            origin=None,
            dirs=None,
            camera_idx=None,
            near=self.near,
            far=self.far,
            _data=_data,
            rank=self.rank,
            group=self.group,
        )

    def __setitem__(self, item, value):
        super().__setitem__(item, value)
