from torch import nn
import torch
import numpy as np


def init_module_weights(layer):
    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    nn.init.constant_(layer.bias, 0.0)
    return layer


def take_per_row(a: torch.Tensor, start_idx: torch.Tensor, length: int):
    assert length != 0
    if length > 0:
        all_indx = start_idx.unsqueeze(1) + torch.arange(
            length, device=start_idx.device
        )
    else:
        all_indx = start_idx.unsqueeze(1) + torch.arange(
            length + 1, 1, 1, device=start_idx.device
        )
    return a[torch.arange(start_idx.shape[0]).unsqueeze(1), all_indx]


def all_equal(*t: torch.Tensor):
    assert len(t) >= 2
    pair_eqs = [torch.equal(t[i], t[i + 1]) for i in range(len(t) - 1)]

    return np.all(pair_eqs)
