from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class RuntimeType(Enum):
    """
    Inference runtime types.
    """

    Kernel = "Kernel"
    Torch = "Torch"


class MergeType(str, Enum):
    """
    Compile status
    """

    @classmethod
    def check(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True

    Evenly = "Evenly"  # use `torch.chunk` to split tensors into blocks, use config to merge on specific dim
    Unevenly = "Unevenly"  # use `torch.split` to split tensors into blocks, use concat on dim 0 to merge
    List = "List"  # use list of tensors or list of modules to represent blocks, instead of a single tensor.


@dataclass
class StageInput:
    Args: Optional[Tuple] = None
    Kwargs: Optional[Dict] = None
