import random

import numpy as np
import torch


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    if torch.backends.mps.is_available():
        device = "mps"
    return device


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
