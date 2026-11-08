"""Select an available PyTorch execution device."""

import torch


def select_device(device):
    """Select MPS, CUDA, or CPU based on availability.

    Parameters
    ----------
    device : str
        Requested device value. The current implementation replaces it with
        the first available device in MPS, CUDA, CPU order.

    Returns
    -------
    torch.device
        Automatically selected execution device.
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return torch.device(device)
