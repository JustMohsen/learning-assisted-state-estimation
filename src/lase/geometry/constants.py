"""Physical constants (with sources)."""

from __future__ import annotations

import math

# NCLT local-frame origin (paper Table 3), used for local gravity.
_NCLT_LAT_DEG = 42.293227
_NCLT_ALT_M = 270.0


def normal_gravity(lat_deg: float, alt_m: float) -> float:
    """WGS-84 normal gravity (Somigliana formula) with free-air correction (m/s^2)."""
    s2 = math.sin(math.radians(lat_deg)) ** 2
    g0 = 9.7803253359 * (1 + 0.00193185265241 * s2) / math.sqrt(1 - 0.00669437999013 * s2)
    return g0 - 3.086e-6 * alt_m


GRAVITY_MPS2 = normal_gravity(_NCLT_LAT_DEG, _NCLT_ALT_M)
"""Local gravity magnitude at the NCLT origin (~9.8029 m/s^2), pointing +z in NED."""
