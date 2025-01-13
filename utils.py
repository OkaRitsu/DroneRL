import torch


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    if torch.backends.mps.is_available():
        device = "mps"
    return device
