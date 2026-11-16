"""Phase 5 pilot: method C, a small causal GRU that sets per-step noise multipliers.

Same learned roles as the method-B pilot (GPS, wheel and lateral-constraint variance
multipliers), but context-dependent instead of constant, so C vs B isolates the value of
context under the same training objective.

- **Features** (sensor-only, causal): computed on a regular grid from samples at or
  before each grid time (IMU means and vibration, wheel speeds, wheel and FOG turn rates,
  GPS recency, satellites, fix mode, GPS-vs-wheel speed consistency).
- **Network:** GRU over the feature grid (float32) and a linear head; multipliers
  ``exp(tanh(z) * log(bound))`` in ``[1/bound, bound]``. The head starts at zero, so an
  untrained network reproduces baseline A exactly.
- The filter looks up the latest grid output at or before each event time.

A separate design skeleton (``lase.learning.context_net``) exists for the full Phase 5
network; reconciling the two is a Phase 5 open item.
"""

from __future__ import annotations

import bisect
import datetime as dt
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from lase.data.calibration import load_calibration
from lase.data.events import SensorId
from lase.data.raw import RowFlag
from lase.data.session import SensorStreams
from lase.filter.learned import HeadOutputs
from lase.filter.runner import LearnedFn
from lase.parameters import Parameters
from lase.training.loop import Snapshot, run_segment
from lase.training.pilot import load_session, task_loss

D = torch.float64
N_FEATURES = 16


def _window_mean(t: np.ndarray, cs: np.ndarray, tq: np.ndarray, span_us: int) -> np.ndarray:
    """Mean of samples in (tq - span, tq] from a cumulative sum ``cs`` (len N + 1)."""
    hi = np.searchsorted(t, tq, side="right")
    lo = np.searchsorted(t, tq - span_us, side="right")
    n = (hi - lo).astype(float)
    last = np.clip(hi - 1, 0, len(t) - 1)
    mean = (cs[hi] - cs[lo]) / np.maximum(n, 1)[:, None]
    single = cs[last + 1] - cs[last]  # the latest sample, if the window is empty
    out: np.ndarray = np.where((n > 0)[:, None], mean, single)
    out[hi == 0] = 0.0
    return out


def features(streams: SensorStreams, tg: np.ndarray) -> np.ndarray:
    """Causal features (len(tg), N_FEATURES) at grid times ``tg`` (us)."""
    cal = load_calibration()
    imu, wh, gps = streams.imu, streams.wheels, streams.gps
    f = np.zeros((len(tg), N_FEATURES))
    m = imu.valid
    ti, acc, gyr = imu.t_us[m], imu.acc[m], imu.gyr[m]
    cs = np.vstack([np.zeros((1, 6)), np.cumsum(np.hstack([acc, gyr]), 0)])
    ag = _window_mean(ti, cs, tg, 100_000)
    f[:, 0:3] = ag[:, :3] / 9.81
    f[:, 3:6] = ag[:, 3:] / 0.5
    nrm = np.linalg.norm(acc, axis=1)
    cs2 = np.vstack([np.zeros((1, 2)), np.cumsum(np.c_[nrm, nrm**2], 0)])
    mm = _window_mean(ti, cs2, tg, 1_000_000)
    f[:, 6] = np.sqrt(np.maximum(mm[:, 1] - mm[:, 0] ** 2, 0.0)) / 0.05  # vibration (m/s^2)
    mw = wh.valid
    tw, left, right = wh.t_us[mw], wh.left[mw], wh.right[mw]
    csw = np.vstack([np.zeros((1, 2)), np.cumsum(np.c_[left, right], 0)])
    lr = _window_mean(tw, csw, tg, 100_000)
    f[:, 7:9] = lr / 2.0
    lr5 = _window_mean(tw, csw, tg, 500_000)
    f[:, 9] = (lr5[:, 0] - lr5[:, 1]) / cal.wheels.effective_track_width_raw_m / 0.5
    # FOG yaw rate over the trailing 0.5 s (NED sign)
    tf, ang = streams.fog.t_us, streams.fog_angle
    i1 = np.searchsorted(tf, tg, side="right") - 1
    i0 = np.searchsorted(tf, tg - 500_000, side="right") - 1
    ok = (i0 >= 0) & (i1 > i0)
    i0c, i1c = np.clip(i0, 0, None), np.clip(i1, 0, None)
    dtf = (tf[i1c] - tf[i0c]) * 1e-6
    rate = np.where(ok, (ang[i1c] - ang[i0c]) / np.maximum(dtf, 1e-3), 0.0)
    f[:, 10] = rate * cal.fog_convention.angle_sign_to_ned_yaw / 0.5
    # GPS: recency, satellites, fix mode, speed consistency with the wheels
    held = (gps.flags & np.uint16(RowFlag.GPS_HELD)) != 0
    gv = gps.valid & ~held
    tgps, ned = gps.t_us[gv], streams.gnss_ned[gv, :2]
    sats, mode = gps.num_sats[gv], gps.mode[gv]
    j = np.searchsorted(tgps, tg, side="right") - 1
    has = j >= 0
    jc = np.clip(j, 0, None)
    since = np.where(has, (tg - tgps[jc]) * 1e-6, 10.0) if len(tgps) else np.full(len(tg), 10.0)
    f[:, 11] = np.clip(since, 0, 10) / 10.0
    if len(tgps):
        f[:, 12] = np.where(has, sats[jc], 0) / 12.0
        f[:, 13] = np.where(has, mode[jc], 0) / 3.0
        jp = np.clip(j - 1, 0, None)
        two = j >= 1
        dt_g = np.maximum((tgps[jc] - tgps[jp]) * 1e-6, 1e-3)
        v_gps = np.linalg.norm(ned[jc] - ned[jp], axis=1) / dt_g
        v_wh = 0.5 * (lr5[:, 0] + lr5[:, 1]) / cal.wheels.speed_scale
        recent = two & (dt_g < 3.0) & (since < 2.0)
        f[:, 14] = np.where(recent, np.clip(v_gps - v_wh, -3, 3), 0.0) / 2.0
        f[:, 15] = (since < 1.5).astype(float)
    f[~np.isfinite(f)] = 0.0
    return f


class PilotNet(nn.Module):
    """GRU + linear head -> (T, 3) multipliers for GPS, wheel and lateral variances."""

    def __init__(self, hidden: int, bound: float) -> None:
        super().__init__()
        self.gru = nn.GRU(N_FEATURES, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.log_bound = math.log(bound)

    def forward(self, x: Tensor) -> Tensor:
        """x (T, N_FEATURES) float32 -> multipliers (T, 3) float64."""
        y, _ = self.gru(x[None])
        return torch.exp(torch.tanh(self.head(y[0])) * self.log_bound).to(D)


def make_net(p: Parameters) -> PilotNet:
    c = p.training.pilot_c
    torch.manual_seed(c.seed)
    return PilotNet(c.hidden, c.mult_bound)


def grid(t0_us: int, t1_us: int, hz: float) -> np.ndarray:
    step = int(1e6 / hz)
    out: np.ndarray = np.arange(t0_us, t1_us + step, step, dtype=np.int64)
    return out


def learned_fn_from(net: PilotNet, streams: SensorStreams, tg: np.ndarray) -> LearnedFn:
    """Per-event head outputs: the latest grid output at or before the event time."""
    x = torch.as_tensor(features(streams, tg), dtype=torch.float32)
    m = net(x)
    tl = tg.tolist()
    cache: dict[int, HeadOutputs] = {}
    z6, z1 = torch.zeros(1, 6, dtype=D), torch.zeros(1, 1, dtype=D)

    def fn(t: int, k: SensorId) -> HeadOutputs:
        j = max(bisect.bisect_right(tl, t) - 1, 0)
        h = cache.get(j)
        if h is None:
            mj = m[j]
            h = HeadOutputs(
                z6, z1, mj[1].reshape(1, 1), mj[0].expand(2).reshape(1, 2), mj[2].reshape(1, 1)
            )
            cache[j] = h
        return h

    return fn


def chunk_grad_c(args: tuple[dt.date, Snapshot, dict[str, Any], Parameters]) -> dict[str, Any]:
    """Worker: loss and parameter gradients of one chunk (gradient through the whole chunk)."""
    date, snap, state, p = args
    torch.set_num_threads(1)
    sess = load_session(date, p)
    c, w = p.training.pilot_c, p.training.windows
    net = make_net(p)
    net.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
    t_loss = snap.t_us + int(w.burn_in_s * 1e6)
    t_end = t_loss + int(w.loss_window_s * 1e6)
    tg = grid(snap.t_us - int(c.gru_warmup_s * 1e6), t_end, c.grid_hz)
    fn = learned_fn_from(net, sess.streams, tg)
    res = run_segment(
        sess.streams, sess.cfg, sess.params, snap, t_end, fn, t_grad_us=snap.t_us, t_out_us=t_loss
    )
    loss, n = task_loss(sess, res, p)
    grads = {k: np.zeros(v.shape, dtype=np.float32) for k, v in state.items()}
    if n and loss.requires_grad:
        loss.backward()  # type: ignore[no-untyped-call]
        for k, prm in net.named_parameters():
            if prm.grad is not None:
                grads[k] = prm.grad.numpy().copy()
    with torch.no_grad():
        mult = fn(t_loss, SensorId.IMU)
        mults = [
            float(mult.gnss_var_mult[0, 0]),
            float(mult.wheel_var_mult[0, 0]),
            float(mult.nhc_var_mult[0, 0]),
        ]
    return {"loss": float(loss.detach()), "grads": grads, "n": n, "mults": mults}


__all__ = [
    "N_FEATURES",
    "PilotNet",
    "chunk_grad_c",
    "features",
    "grid",
    "learned_fn_from",
    "make_net",
]
