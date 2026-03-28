import torch


def is_sm10x():
    """Check if current GPU is Blackwell (SM120+) or newer."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 10


def is_hopper():
    """Check if current GPU is Hopper (SM90)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (9, 0)
