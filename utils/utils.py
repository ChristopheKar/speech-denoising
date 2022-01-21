import random
import numpy as np
import torch


def set_device(verbose=True):
    """Set Torch compute device, either `cuda` or `cpu`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (verbose):
        print('Device: ', device)

    return device


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True