import torch
from typing import List
import torch.nn as nn


def is_child(child_module: nn.Module, parent_modules: List[nn.Module]) -> bool:
    for parent_module in parent_modules:
        if child_module in parent_module.modules():
            return True

    return False


def initialize_weights(module: nn.Module, excluded_modules: List[nn.Module]):
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.Conv2d) and not is_child(submodule, excluded_modules):
            print(f"Initializing {name}")
            nn.init.xavier_uniform_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        else:
            print(f"Skipping {name}")
