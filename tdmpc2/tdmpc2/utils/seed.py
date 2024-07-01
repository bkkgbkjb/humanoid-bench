import random
import numpy as np
import torch


def seed_all(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
