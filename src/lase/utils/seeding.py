"""Random-seed control for reproducible runs."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch.

    Args:
        seed: Seed value; record it in the run's provenance.
        deterministic: If True, request deterministic PyTorch algorithms
            (slower; errors on ops without a deterministic implementation).
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global seed for third-party code
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
