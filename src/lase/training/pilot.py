"""Phase 4 pilot: method B with three learned noise levels (GPS, wheel, lateral constraint).

Each learned quantity is an anchored deviation: sigma = anchor * exp(delta), applied
through the filter's existing variance multipliers (``HeadOutputs``: multiplier
exp(2 delta)). delta = 0 is exactly baseline A. The loss on one chunk is the Huber task
loss over the loss window (position, wrapped heading, velocity against the dense
reference, each normalised); the anchor penalty is added once per optimiser step,
divided by the number of training chunks (a Gaussian prior on the whole training set).
"""

from __future__ import annotations

import datetime as dt
import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lase.data.calibration import load_calibration
from lase.data.reference import Reference
from lase.data.session import SensorStreams, load_inputs, load_reference_for
from lase.filter.learned import HeadOutputs
from lase.filter.params import FilterConfig, FilterParams
from lase.filter.runner import initial_condition_from_reference, initial_state
from lase.parameters import Parameters
from lase.training.loop import LoopBook, Snapshot, run_segment

D = torch.float64
LEARNED = ("gnss_sigma", "wheel_sigma", "lateral_sigma")


def learned_head(delta: Tensor) -> HeadOutputs:
    """Constant head outputs for deltas (3,) in the order of ``LEARNED``."""
    m = torch.exp(2.0 * delta)
    return HeadOutputs(
        bias_drift=torch.zeros(1, 6, dtype=D),
        wheel_residual=torch.zeros(1, 1, dtype=D),
        wheel_var_mult=m[1].reshape(1, 1),
        gnss_var_mult=m[0].expand(2).reshape(1, 2),
        nhc_var_mult=m[2].reshape(1, 1),
    )


def anchors(cfg: FilterConfig) -> dict[str, float]:
    """Baseline-A values the deltas are relative to."""
    return {
        "gnss_sigma": cfg.gnss.sigma_m,
        "wheel_sigma": cfg.wheel.sigma_mps,
        "lateral_sigma": cfg.nonholonomic.sigma_lateral_mps,
    }


def anchor_sigmas(p: Parameters) -> Tensor:
    """Prior widths (log units) of the deltas."""
    a = p.training.anchor_sigma
    return torch.tensor([a.gnss_sigma, a.wheel_sigma, a.lateral_sigma], dtype=D)


def huber(x: Tensor, delta: float) -> Tensor:
    """Elementwise Huber, quadratic below ``delta`` (0.5 x^2) and linear above."""
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * x**2, delta * (ax - 0.5 * delta))


@dataclass(frozen=True)
class Session:
    """Everything needed to run chunks of one session."""

    date: dt.date
    streams: SensorStreams
    ref: Reference
    cfg: FilterConfig
    params: FilterParams


@functools.lru_cache(maxsize=8)
def load_session(date: dt.date, p: Parameters) -> Session:
    """Load one session (cached per process)."""
    cfg = p.filter.for_date(date)
    cal = load_calibration()
    return Session(
        date,
        load_inputs(date, cal, p.data.limits),
        load_reference_for(date, p.data.node_match_tolerance_s),
        cfg,
        FilterParams.build(cal, cfg),
    )


def session_start(sess: Session) -> Snapshot:
    """Baseline-A initial condition, as in ``run_session``."""
    ref = sess.ref
    t0 = int(ref.gt.t_us[ref.gt.valid][0]) + int(sess.cfg.init.start_offset_s * 1e6)
    ic = initial_condition_from_reference(ref, t0)
    st, P = initial_state(ic, sess.streams, sess.params, sess.cfg)
    return Snapshot(ic.t_us, st, P, LoopBook(ic.t_us))


def cache_snapshots(date: dt.date, p: Parameters) -> list[Snapshot]:
    """Run baseline A over the whole session and keep snapshots every chunk spacing."""
    torch.set_num_threads(1)
    sess = load_session(date, p)
    neutral = HeadOutputs.neutral(1)
    with torch.no_grad():
        res = run_segment(
            sess.streams,
            sess.cfg,
            sess.params,
            session_start(sess),
            int(sess.streams.imu.t_us[-1]) + 1,
            lambda t, k: neutral,
            snapshot_every_s=p.training.pilot.snapshot_every_s,
        )
    return res.snapshots


def chunk_loss(
    sess: Session,
    snap: Snapshot,
    delta: Tensor,
    p: Parameters,
    burn_delta: Tensor | None = None,
) -> tuple[Tensor, int]:
    """Mean task loss over the loss window of the chunk starting at ``snap``.

    With ``training.windows.grad_through_burn_in`` the gradient also flows through the
    burn-in (the loss is still only on the loss window). ``burn_delta`` (tests only)
    fixes the burn-in values independently of ``delta``.
    """
    w, lp = p.training.windows, p.training.loss
    t_loss = snap.t_us + int(w.burn_in_s * 1e6)
    t_end = t_loss + int(w.loss_window_s * 1e6)
    head = learned_head(delta)
    burn = learned_head(burn_delta if burn_delta is not None else delta.detach())
    full = w.grad_through_burn_in and burn_delta is None
    res = run_segment(
        sess.streams,
        sess.cfg,
        sess.params,
        snap,
        t_end,
        (lambda t, k: head) if full else (lambda t, k: head if t >= t_loss else burn),
        t_grad_us=snap.t_us if full else t_loss,
        t_out_us=t_loss,
    )
    if not res.out_t_us:
        return torch.zeros((), dtype=D), 0
    tq = np.asarray(res.out_t_us, dtype=np.int64)
    p_ref, rpy_ref = sess.ref.pose_at(tq)
    v_ref, v_ok = sess.ref.velocity_w(
        tq, int(p.data.velocity_half_window_s * 1e6), p.data.max_reference_speed_mps
    )
    ok = np.isfinite(p_ref).all(1) & np.isfinite(rpy_ref[:, 2]) & v_ok
    if not ok.any():
        return torch.zeros((), dtype=D), 0
    k = torch.as_tensor(np.flatnonzero(ok))
    pe = (torch.stack(res.out_p)[k] - torch.as_tensor(p_ref[ok, :2], dtype=D)) / lp.pos_scale_m
    dy = torch.stack(res.out_yaw)[k] - torch.as_tensor(rpy_ref[ok, 2], dtype=D)
    he = torch.atan2(torch.sin(dy), torch.cos(dy)) / lp.heading_scale_rad
    ve = (torch.stack(res.out_v)[k] - torch.as_tensor(v_ref[ok, :2], dtype=D)) / lp.vel_scale_mps
    hd = lp.huber_delta
    per_sample = (
        lp.pos_weight * huber(pe, hd).sum(1)
        + lp.heading_weight * huber(he, hd)
        + lp.vel_weight * huber(ve, hd).sum(1)
    )
    return per_sample.mean(), int(ok.sum())


def chunk_grad(args: tuple[dt.date, Snapshot, list[float], Parameters]) -> dict[str, Any]:
    """Worker: loss and gradient of one chunk at the given deltas (no anchor penalty)."""
    date, snap, delta_list, p = args
    torch.set_num_threads(1)
    sess = load_session(date, p)
    delta = torch.tensor(delta_list, dtype=D, requires_grad=True)
    loss, n = chunk_loss(sess, snap, delta, p)
    if n and loss.requires_grad:
        loss.backward()  # type: ignore[no-untyped-call]
        grad = delta.grad.numpy().copy() if delta.grad is not None else np.zeros(len(delta_list))
    else:
        grad = np.zeros(len(delta_list))
    return {"loss": float(loss.detach()), "grad": grad, "n": n}


__all__ = [
    "LEARNED",
    "anchor_sigmas",
    "anchors",
    "cache_snapshots",
    "chunk_grad",
    "chunk_loss",
    "learned_head",
    "load_session",
    "session_start",
]
