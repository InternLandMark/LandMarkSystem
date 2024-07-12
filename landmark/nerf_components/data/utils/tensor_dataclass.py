from abc import abstractmethod
from typing import Optional, Union

import torch


class TensorDataClass:
    """Tensor like data class, for Samples and Rays

    Args:
        rank: usually the rank in a branch parallel communication group
        group: usually the branch parallel communication group
    """

    def __init__(
        self,
        device: Optional[Union[torch.device, str]] = None,
        rank: int = None,
        group: torch.distributed.ProcessGroup = None,
        _data: Optional[torch.Tensor] = None,
    ):
        self._shape = None

        self.rank = rank
        self.group = group
        if device:
            if isinstance(device, str):
                device = torch.device(device)
        self._device = device

        if _data is not None:
            self._data = _data
            if not device:
                self._device = _data.device
            else:
                self._data = _data.to(device)
            self._shape = _data.shape[:-1]

        if not self._device:
            self._device = torch.device("cpu")

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def data(self):
        """Return the pure tensor data."""
        return self._data

    def data_ptr(self):
        return self._data.data_ptr()

    def __len__(self) -> int:
        """Return the total number of rays in the Rays."""
        num_rays = torch.numel(self._data) // self._data.shape[-1]

        return num_rays

    def __repr__(self) -> str:
        ...  # TODO (frank)

    @abstractmethod
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        _data = self._data.view(*shape, self._data.shape[-1])
        return _data

    @abstractmethod
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        _data = self._data.reshape(*shape, self._data.shape[-1])
        return _data

    @abstractmethod
    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = item + (slice(None, None, None),)
        _data = self._data[item]
        return _data

    @abstractmethod
    def __setitem__(self, item, value):
        if isinstance(item, tuple):
            item = item + (slice(None, None, None),)
        self._data[item] = value
