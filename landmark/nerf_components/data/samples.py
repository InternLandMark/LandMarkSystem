from typing import Any, Optional, Union

import torch
import torch.distributed as dist

from .utils.tensor_dataclass import TensorDataClass


class Samples(TensorDataClass):
    """Class for Samples"""

    def __init__(
        self,
        xyz,
        dirs,
        camera_idx: Optional[torch.tensor] = None,
        block_idx: Optional[torch.tensor] = None,
        z_vals: Optional[torch.Tensor] = None,
        device: Optional[Union[torch.device, str]] = None,
        rank: Optional[int] = None,
        group: Optional[Any] = None,
        _data: Optional[torch.Tensor] = None,
        _has_block_idx: Optional[bool] = False,
        _has_z_vals: Optional[bool] = False,
        _has_camera_idx: Optional[bool] = False,
    ):
        super().__init__(device, rank, group, _data)
        if _data is not None:
            assert xyz is None, "xyz must be None when constructing from a '_data' tensor."
            assert dirs is None, "dirs must be None when constructing from a '_data' tensor."
            assert camera_idx is None, "camera_idx must be None when constructing from a '_data' tensor."
            assert block_idx is None, "block_idx must be None when constructing from a '_data' tensor."
            assert z_vals is None, "z_vals must be None when constructing from a '_data' tensor."
            self._has_block_idx = _has_block_idx
            self._has_z_vals = _has_z_vals
            self._has_camera_idx = _has_camera_idx

        else:
            # shape check
            self._shape = xyz.shape[:-1]
            if not device:
                self._device = xyz.device
            shape_dim = len(self._shape)
            if dirs is not None:
                assert dirs.shape[:shape_dim] == self._shape and dirs.dim() == shape_dim + 1, (
                    f"The front shape of xyz and dirs must be the same, but got dirs shape: {dirs.shape[:-1]} and xyz"
                    f" shape: {xyz.shape[-1]}."
                )
            else:
                dirs = torch.zeros(*self._shape, 3, device=self._device, dtype=torch.float32)

            self._has_camera_idx = False
            if camera_idx is not None:
                assert camera_idx.shape[:shape_dim] == self._shape, (
                    "The front shape of xyz and camera_idx must be the same, but got camera_idx shape:"
                    f" {camera_idx.shape[:-1]} and xyz shape: {xyz.shape[:-1]}."
                )
                if camera_idx.dim() == shape_dim:
                    camera_idx = camera_idx.unsqueeze(-1).float()
                self._has_camera_idx = True
            else:
                camera_idx = torch.zeros(*self._shape, 1, device=self._device, dtype=torch.float32)

            self._has_block_idx = False
            if block_idx is not None:
                assert block_idx.shape[:shape_dim] == self._shape, (
                    "The front shape of xyz and block_idx must be the same, but got block_idx shape:"
                    f" {block_idx.shape[:-1]} and xyz shape: {xyz.shape[:-1]}."
                )
                if block_idx.dim() == shape_dim:
                    block_idx = block_idx.unsqueeze(-1).float()
                self._has_block_idx = True
            else:
                block_idx = torch.zeros(*self._shape, 1, device=self._device, dtype=torch.float32)
                self._has_block_idx = True

            self._has_z_vals = False
            if z_vals is not None:
                assert z_vals.shape[:shape_dim] == self._shape, (
                    "The front shape of xyz and z_vals must be the same, but got z_vals shape:"
                    f" {z_vals.shape[:shape_dim]} and xyz shape: {xyz.shape[:-1]}."
                )
                if z_vals.dim() == shape_dim:
                    z_vals = z_vals.unsqueeze(-1)
                self._has_z_vals = True
            else:
                z_vals = torch.zeros(*self._shape, 1, device=self._device, dtype=torch.float32)

            sigma = torch.zeros(*self._shape, 1, device=self._device, dtype=torch.float32)
            rgb = torch.zeros(*self._shape, 3, device=self._device, dtype=torch.float32)
            self.masks = None

            if camera_idx.device != self.device:
                camera_idx = camera_idx.to(self.device)
            if block_idx.device != self.device:
                block_idx = block_idx.to(self.device)
            if z_vals.device != self.device:
                z_vals = z_vals.to(self.device)
            if xyz.device != self.device:
                xyz = xyz.to(self.device)
            if dirs.device != self.device:
                dirs = dirs.to(self.device)

            self._data = torch.cat((sigma, rgb, camera_idx, z_vals, dirs, xyz, block_idx), dim=-1)

    @property
    def xyz(self) -> torch.Tensor:
        return self._data[..., 9:12]

    @property
    def xyzb(self) -> torch.Tensor:
        assert self._has_block_idx, "Error, the block index of each sample has not been set."
        return self._data[..., 9:13]

    @property
    def block_idx(self) -> torch.Tensor:
        if self._has_block_idx:
            return self.data[..., 12].long()
        else:
            return None

    @property
    def dirs(self) -> torch.Tensor:
        return self._data[..., 6:9]

    @property
    def z_vals(self) -> torch.Tensor:
        if self._has_z_vals:
            return self._data[..., 5]
        else:
            return None

    @property
    def camera_idx(self) -> torch.Tensor:
        if self._has_camera_idx:
            return self._data[..., 4].long()
        else:
            return None

    @property
    def sigma(self) -> torch.Tensor:
        return self._data[..., 0]

    @property
    def rgb(self) -> torch.Tensor:
        return self._data[..., 1:4]

    @property
    def dists(self):
        return torch.cat((self.z_vals[..., 1:] - self.z_vals[..., :-1], torch.zeros_like(self.z_vals[..., :1])), dim=-1)

    def __setattr__(self, key, value):
        if key == "sigma":
            if value.shape == self.shape:
                value = value.float().unsqueeze(-1)
                index = torch.tensor([0], device=self.device).expand(*self.shape, len([0]))
                self._data = self._data.scatter(-1, index, src=value)
                # self._data[..., 0] = value.float().squeeze()
            else:
                self._data[..., [0]] = value
        elif key == "rgb":
            if value.shape == self.shape:
                value = value.float().unsqueeze(-1)
                index = torch.tensor([1, 2, 3], device=self.device).expand(*self.shape, len([1, 2, 3]))
                self._data = self._data.scatter(-1, index, src=value)
            else:
                self._data[..., 1:4] = value
        elif key == "camera_idx":
            if value.shape == self.shape:
                self._data[..., [4]] = value.float().unsqueeze(-1)
            else:
                self._data[..., [4]] = value
            self._has_camera_idx = True
        elif key == "z_vals":
            if value.shape == self.shape:
                self._data[..., [5]] = value.float().unsqueeze(-1)
            else:
                self._data[..., [5]] = value
            self._has_z_vals = True
        elif key == "dirs":
            self._data[..., 6:9] = value
        elif key == "xyz":
            index = torch.tensor([9, 10, 11], device=self.device).expand(*self.shape, len([9, 10, 11]))
            self._data = self._data.scatter(-1, index, src=value)
            # self._data[..., 9:12] = value
        elif key == "block_idx":
            if value.shape == self.shape:
                value = value.float().unsqueeze(-1)
                index = torch.tensor([12], device=self.device).expand(*self.shape, len([12]))
                self._data = self._data.scatter(-1, index, src=value)
            else:
                self._data[..., 12] = value
            self._has_block_idx = True
        elif key == "xyzb":
            self._data[..., 9:13] = value
            self._has_block_idx = True
        else:
            super().__setattr__(key, value)

    def to(self, device: Union[torch.device, str]):
        if isinstance(device, str):
            device = torch.device(device)
        if device == self.device:
            return self

        _data = self._data.to(device)
        samples = Samples(
            xyz=None,
            dirs=None,
            group=self.group,
            rank=self.rank,
            _data=_data,
            _has_z_vals=self._has_z_vals,
            _has_camera_idx=self._has_camera_idx,
            _has_block_idx=self._has_block_idx,
        )
        samples.masks = self.masks
        return samples

    def view(self, *shape):
        _data = super().view(*shape)
        samples = Samples(
            xyz=None,
            dirs=None,
            rank=self.rank,
            group=self.group,
            _data=_data,
            _has_z_vals=self._has_z_vals,
            _has_camera_idx=self._has_camera_idx,
            _has_block_idx=self._has_block_idx,
        )
        # samples.masks = self.masks
        return samples

    def reshape(self, *shape):
        _data = super().reshape(*shape)
        samples = Samples(
            xyz=None,
            dirs=None,
            rank=self.rank,
            group=self.group,
            _data=_data,
            _has_z_vals=self._has_z_vals,
            _has_camera_idx=self._has_camera_idx,
            _has_block_idx=self._has_block_idx,
        )
        # samples.masks = self.masks
        return samples

    def __getitem__(self, item):
        _data = super().__getitem__(item)

        samples = Samples(
            xyz=None,
            dirs=None,
            rank=self.rank,
            group=self.group,
            _data=_data,
            _has_z_vals=self._has_z_vals,
            _has_camera_idx=self._has_camera_idx,
            _has_block_idx=self._has_block_idx,
        )
        samples._has_z_vals = self._has_z_vals
        samples._has_camera_idx = self._has_camera_idx
        samples._has_block_idx = self._has_block_idx
        if self.masks is not None:
            samples.masks = [mask[item] for mask in self.masks]
        return samples

    def __setitem__(self, item, value):
        assert isinstance(value, Samples), "Error, the value to be set must be a Samples object."
        super().__setitem__(item, value)
        self._has_z_vals = value._has_z_vals
        self._has_camera_idx = value._has_camera_idx
        self._has_block_idx = value._has_block_idx
        # self.masks = value.masks

    @property
    def mask(self):
        assert self.rank is not None
        return self.masks[self.rank]

    def sync_sigma(self):
        sample_nums = [mask.sum() for mask in self.masks]
        if self.sigma is not None:
            sigma_tensor_list = [torch.zeros(num, device=self.device) for num in sample_nums]

            with torch.no_grad():
                dist.all_gather(sigma_tensor_list, self.sigma[self.mask], group=self.group)
            assert self.rank is not None
            sigma_tensor_list[self.rank] = self.sigma[self.mask]
            for mask, validsigma in zip(self.masks, sigma_tensor_list):
                self.sigma.masked_scatter_(mask, validsigma)  # TODO may have bug
        else:
            assert False, "Error, the sigma of the samples has not been set."

    def sync_rgb(self):
        sample_nums = [mask.sum() for mask in self.masks]
        if self.rgb is not None:
            rgb_tensor_list = [torch.zeros((num, 3), device=self.device) for num in sample_nums]

            with torch.no_grad():
                dist.all_gather(rgb_tensor_list, self.rgb[self.mask], group=self.group)
            assert self.rank is not None
            rgb_tensor_list[self.rank] = self.rgb[self.mask]
            for mask, validrgb in zip(self.masks, rgb_tensor_list):
                self.rgb.masked_scatter_(mask.unsqueeze(-1).expand(*mask.shape, 3), validrgb)
        else:
            assert False, "Error, the rgb of the samples has not been set."
