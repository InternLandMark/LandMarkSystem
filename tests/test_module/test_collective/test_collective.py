# from dataclasses import dataclass

import torch

from landmark.communicator import (
    destroy_communication_context,
    init_communication_context,
)
from landmark.communicator.collective import (
    all_gather,
    all_reduce,
    broadcast,
    broadcast_object_list,
    gather,
    scatter,
)
from landmark.communicator.group_initializer import CommMode, _InterCommMode
from landmark.utils.config import Config
from landmark.utils.env import EnvSetting


class TestCollectiveOp:
    """
    Test collective op
    """

    def setup_class(self):
        self.conf = Config(
            dict(
                rank=EnvSetting.RANK,
                world_size=EnvSetting.WORLD_SIZE,
            )
        )
        self.device = f"cuda:{self.conf.rank}"
        self.tensor_size = 5
        init_communication_context(self.conf)

    def teardown_class(self):
        torch.cuda.empty_cache()
        destroy_communication_context()

    def test_allgather(self):
        gathered_tensor = torch.zeros(self.tensor_size * self.conf.world_size, device=self.device)
        tensor = torch.ones(self.tensor_size, device=self.device) * self.conf.rank
        all_gather(output_tensor=gathered_tensor, input_tensor=tensor, comm_mode=CommMode.GLOBAL)
        torch.equal(
            gathered_tensor[self.conf.rank : (self.conf.rank + 1) * self.tensor_size],
            tensor,
        )

    def test_allreduce(self):
        x = torch.rand(self.tensor_size, device=self.device)
        y = torch.rand(self.tensor_size, device=self.device)
        if self.conf.rank == 0:
            input_tensor = x
        else:
            input_tensor = y
        all_reduce(tensor=input_tensor, comm_mode=CommMode.GLOBAL)
        torch.equal(input_tensor, x + y)

    def test_scatter(self):
        x = torch.zeros(self.tensor_size, device=self.device)
        y = torch.ones(self.tensor_size, device=self.device)
        if self.conf.rank == 0:
            obj_list = [x, y]
        else:
            obj_list = None
        output_tensor = x
        scatter(tensor=output_tensor, comm_mode=CommMode.GLOBAL, scatter_list=obj_list)
        if self.conf.rank == 0:
            assert torch.equal(output_tensor, x)
        else:
            assert torch.equal(output_tensor, y)

    def test_broadcast(self):
        if self.conf.rank == 0:
            x = torch.ones(self.tensor_size, device=self.device)
        else:
            x = torch.zeros(self.tensor_size, device=self.device)
        if self.conf.rank == 0:
            assert torch.equal(x, torch.ones(self.tensor_size, device=self.device))
        else:
            assert not torch.equal(x, torch.ones(self.tensor_size, device=self.device))
        broadcast(tensor=x, comm_mode=CommMode.GLOBAL)
        assert torch.equal(x, torch.ones(self.tensor_size, device=self.device))

    def test_gather(self):
        if self.conf.rank == 0:
            x = torch.zeros(self.tensor_size, device=self.device)
            gather_list = None
        else:
            x = torch.ones(self.tensor_size, device=self.device)
            gather_list = [torch.zeros(self.tensor_size, device=self.device) for _ in range(self.conf.world_size)]
        gather(tensor=x, comm_mode=CommMode.GLOBAL, gather_list=gather_list, dst=1)
        if self.conf.rank == 1:
            assert torch.equal(gather_list[0], torch.zeros(self.tensor_size, device=self.device))
        else:
            assert gather_list is None

    def test_broadcast_object_list(self):
        if self.conf.rank == 0:
            obj_list = [(1, 2), {"abc": "dfe"}]
        else:
            obj_list = [None, None]
        broadcast_object_list(obj_list, _InterCommMode.MESSAGE_TRANSFER)
        assert obj_list == [(1, 2), {"abc": "dfe"}]

    def test_broadcast_object_list_var(self):
        if self.conf.rank == 0:
            x = (1, 2)
            y = {"abc": "dfe"}
            obj_list = [x, y]
        else:
            x = (3, 4)
            y = {"dbc": "aaa"}
            obj_list = [x, y]
        broadcast_object_list(obj_list, _InterCommMode.MESSAGE_TRANSFER)
        assert obj_list == [(1, 2), {"abc": "dfe"}]
        if self.conf.rank == 0:
            assert x == (1, 2)
            assert y == {"abc": "dfe"}
        else:
            assert x == (3, 4)
            assert y == {"dbc": "aaa"}

    # def test_broadcast_object_list_dataclass(self):
    #     @dataclass
    #     class Camera:
    #         a: int
    #         b: torch.Tensor

    #     if self.conf.rank == 0:
    #         a = 1
    #         b = torch.zeros(self.tensor_size)
    #         obj_list = [Camera(a, b)]
    #     else:
    #         a = 2
    #         b = torch.ones(self.tensor_size)
    #         obj_list = [Camera(a, b)]

    #     broadcast_object_list(obj_list, _InterCommMode.MESSAGE_TRANSFER)
    #     assert obj_list == [Camera(1, torch.zeros(self.tensor_size))]
