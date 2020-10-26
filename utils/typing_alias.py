"""
Meant to be imported as
from utils.typing_helper import *

To ease # imports for typing.
"""

__all__ = [
    "TYPE_CHECKING",
    "Any",
    "Dict",
    "DataLoader",
    "List",
    "lr_scheduler",
    "nn.Module",
    "optim",
    "SummaryWriter",
    "tupperware",
    "Tensor",
    "Tuple",
    "Union",
]


from typing import Dict, List, Any, Tuple, Union
from torch.utils.data import DataLoader
from utils.tupperware import tupperware
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor, nn, optim
import torch.optim.lr_scheduler as lr_scheduler
