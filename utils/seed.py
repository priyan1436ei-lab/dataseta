"""Seed utilities for reproducible AI/ML experiments across Python, NumPy, PyTorch, and CUDA."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set seeds across all random number generators for strict reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
