"""Learned-output interface (CONVENTIONS section 6.3), with neutral defaults.

With neutral values (zero corrections, unit multipliers) the filter is exactly the
analytical baseline. Method B (learned constants) and C (GRU) produce the same type.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HeadOutputs:
    """Per-step learned quantities (batch-first)."""

    bias_drift: Tensor  # (B, 6): accel m/s^3 (0-2), gyro rad/s^2 (3-5)
    wheel_residual: Tensor  # (B, 1) m/s
    wheel_var_mult: Tensor  # (B, 1)
    gnss_var_mult: Tensor  # (B, 2) north, east

    @classmethod
    def neutral(cls, batch: int, dtype: torch.dtype = torch.float64) -> HeadOutputs:
        """Zero corrections and unit multipliers: the analytical baseline."""
        return cls(
            bias_drift=torch.zeros(batch, 6, dtype=dtype),
            wheel_residual=torch.zeros(batch, 1, dtype=dtype),
            wheel_var_mult=torch.ones(batch, 1, dtype=dtype),
            gnss_var_mult=torch.ones(batch, 2, dtype=dtype),
        )
