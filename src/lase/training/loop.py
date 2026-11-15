"""Training-mode filter loop: the Phase 3 runner's event loop, kept differentiable.

Pilot version (Phase 4 skeleton). It mirrors ``lase.filter.runner.run_filter`` event by
event (exact-time splitting, causal FOG handling, GPS gating, re-acquisition and recovery)
with three differences:

- no NumPy conversion of tensors that may carry gradients (logs are not collected);
- gradients are switched on only from ``t_grad_us`` (burn-in before it runs without);
- it can start from a saved snapshot (state, covariance and loop bookkeeping) and save
  snapshots, so training chunks warm-start from baseline A.

Merging this with ``run_filter`` into one implementation is a Phase 4 open item (plan D3).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from lase.data.events import SensorId, build_event_index
from lase.data.raw import RowFlag
from lase.data.session import SensorStreams
from lase.filter.params import FilterConfig, FilterParams
from lase.filter.propagation import ImuInterval, propagate
from lase.filter.runner import LearnedFn, _FogHistory
from lase.filter.so3 import yaw_of
from lase.filter.state import P_, NavState
from lase.filter.updates import gate_threshold, update_gnss, update_nonholonomic, update_wheel

D = torch.float64


@dataclass
class LoopBook:
    """Loop bookkeeping carried across a snapshot (everything besides state and P)."""

    last_gnss_accept: int
    last_gnss_queued: int = -(10**18)
    gnss_rejects: int = 0
    fog_prev: float | None = None
    reacq: list[tuple[int, np.ndarray]] = field(default_factory=list)


@dataclass
class Snapshot:
    """Filter state right after the IMU sample at ``t_us`` was processed."""

    t_us: int
    state: NavState
    P: Tensor
    book: LoopBook


@dataclass
class SegmentResult:
    """Outputs of :func:`run_segment`."""

    state: NavState
    P: Tensor
    out_t_us: list[int]
    out_p: list[Tensor]  # (2,) axle-centre north/east, differentiable inside the grad window
    out_v: list[Tensor]  # (2,)
    out_yaw: list[Tensor]  # ()
    snapshots: list[Snapshot]


def _detach_state(s: NavState) -> NavState:
    return NavState(**{k: getattr(s, k).detach() for k in s.__dataclass_fields__})


def run_segment(
    streams: SensorStreams,
    cfg: FilterConfig,
    params: FilterParams,
    start: Snapshot,
    t_end_us: int,
    learned_fn: LearnedFn,
    t_grad_us: int | None = None,
    t_out_us: int | None = None,
    snapshot_every_s: float | None = None,
) -> SegmentResult:
    """Run from ``start`` to ``t_end_us``.

    Gradients are recorded only for IMU steps at or after ``t_grad_us`` (None: never).
    Outputs are collected at every IMU step at or after ``t_out_us`` (None: none).
    Snapshots are saved every ``snapshot_every_s`` seconds (None: none).
    """
    state, P = start.state, start.P
    bk = LoopBook(**{**start.book.__dict__, "reacq": list(start.book.reacq)})
    idx = build_event_index(streams).window(start.t_us, t_end_us)
    fog_hist = _FogHistory(
        streams, cfg.fog_handling.max_extrapolation_s, cfg.fog_handling.extrapolation_baseline_s
    )
    imu, wh, gps = streams.imu, streams.wheels, streams.gps
    held_bit = np.uint16(RowFlag.GPS_HELD)
    Cb = params.C_b_imu
    l_imu = params.l_imu
    g = cfg.gnss
    gate2 = gate_threshold(g.gate_prob, 2)
    res = SegmentResult(state, P, [], [], [], [], [])
    next_snap = start.t_us + int(snapshot_every_s * 1e6) if snapshot_every_s else None
    last_imu: tuple[int, Tensor, Tensor] | None = None
    pending: list[tuple[int, int, int]] = []

    def body_rate(gyr: Tensor, st: NavState) -> Tensor:
        return (gyr - st.b_gyr) @ Cb.T

    def apply_update(
        st: NavState, Pm: Tensor, t: int, sensor: int, row: int, omega: Tensor
    ) -> tuple[NavState, Tensor]:
        if sensor == SensorId.WHEEL:
            lw = learned_fn(t, SensorId.WHEEL)
            st, Pm, _ = update_wheel(
                st,
                Pm,
                torch.tensor([wh.left[row]], dtype=D),
                torch.tensor([wh.right[row]], dtype=D),
                omega,
                params,
                cfg.wheel,
                lw,
            )
            if cfg.nonholonomic.enabled:
                st, Pm, _ = update_nonholonomic(st, Pm, omega, params, cfg.nonholonomic, lw)
            return st, Pm
        z = torch.as_tensor(streams.gnss_ned[row, :2], dtype=D)[None]
        lg = learned_fn(t, SensorId.GNSS)
        gap_us = t - bk.last_gnss_accept
        if g.reacq_enabled and gap_us > g.reacq_trigger_s * 1e6:
            _, _, ud = update_gnss(st, Pm, z, params, g, lg, apply=torch.tensor([False]))
            if float(ud.nis[0]) <= gate2:
                bk.reacq.append((t, ud.innovation[0].detach().numpy()))
            else:
                bk.reacq.clear()
            bk.reacq[:] = [(tt, r) for tt, r in bk.reacq if t - tt <= g.reacq_window_s * 1e6]
            if len(bk.reacq) >= g.reacq_min_fixes:
                inn = np.array([r for _, r in bk.reacq])
                spread = float(np.max(np.linalg.norm(inn - inn.mean(0), axis=1)))
                allowed = g.reacq_max_offset_m + g.reacq_offset_growth_mps * gap_us * 1e-6
                if (
                    spread <= g.reacq_consistency_m
                    and float(np.linalg.norm(inn.mean(0))) <= allowed
                ):
                    st, Pm, ud = update_gnss(st, Pm, z, params, g, lg)
                    bk.reacq.clear()
        else:
            st, Pm, ud = update_gnss(st, Pm, z, params, g, lg)
        if bool(ud.accepted[0]):
            bk.last_gnss_accept, bk.gnss_rejects = t, 0
        else:
            bk.gnss_rejects += 1
            if (t - bk.last_gnss_accept) > g.recovery_after_s * 1e6 and (
                bk.gnss_rejects >= g.recovery_min_rejections
            ):
                Pm = Pm.clone()
                Pm[:, P_, :] = 0.0
                Pm[:, :, P_] = 0.0
                Pm[:, P_, P_] = torch.eye(3, dtype=D) * g.recovery_pos_sigma_m**2
                st, Pm, ud = update_gnss(st, Pm, z, params, g, lg)
                bk.last_gnss_accept, bk.gnss_rejects = t, 0
        return st, Pm

    for t, s, row, valid in zip(idx.t_us, idx.sensor, idx.row, idx.valid, strict=True):
        t, row = int(t), int(row)
        if s == SensorId.FOG or not valid:
            continue
        if s == SensorId.WHEEL:
            if cfg.wheel.enabled:
                pending.append((t, int(s), row))
            continue
        if s == SensorId.GNSS:
            if (
                g.enabled
                and not (g.skip_held and (gps.flags[row] & held_bit))
                and (t - bk.last_gnss_queued) >= g.min_interval_s * 1e6
            ):
                pending.append((t, int(s), row))
                bk.last_gnss_queued = t
            continue
        grad_on = t_grad_us is not None and t >= t_grad_us
        with torch.set_grad_enabled(grad_on):
            acc = torch.as_tensor(imu.acc[row], dtype=D)[None]
            gyr = torch.as_tensor(imu.gyr[row], dtype=D)[None]
            if last_imu is None or t <= last_imu[0]:
                if last_imu is None:
                    for tp, sp, rp in pending:
                        state, P = apply_update(state, P, tp, sp, rp, body_rate(gyr, state))
                    pending.clear()
                last_imu = (t, acc, gyr)
                continue
            t0, acc0, gyr0 = last_imu
            pending.sort()
            bounds = [t0] + [tp for tp, _, _ in pending if t0 < tp < t] + [t]
            for tp, sp, rp in [(a, b, c) for a, b, c in pending if a <= t0]:
                state, P = apply_update(state, P, tp, sp, rp, body_rate(gyr0, state))
            span = float(t - t0)

            def lerp(a: Tensor, b: Tensor, tq: int, t0: int = t0, span: float = span) -> Tensor:
                return a + ((tq - t0) / span) * (b - a)

            omega = body_rate(gyr0, state)
            fog_at: dict[int, float | None] = {}
            if cfg.use_fog:
                fog_at[t0] = bk.fog_prev if bk.fog_prev is not None else fog_hist.angle_at(t0, t)
                for bb in bounds[1:]:
                    fog_at[bb] = fog_hist.angle_at(bb, t)
            for a, b in itertools.pairwise(bounds):
                if b > a:
                    fa, fb = fog_at.get(a), fog_at.get(b)
                    fog_ok = fa is not None and fb is not None
                    gb = lerp(gyr0, gyr, b)
                    iv = ImuInterval(
                        dt=torch.tensor([(b - a) * 1e-6], dtype=D),
                        acc0=lerp(acc0, acc, a),
                        acc1=lerp(acc0, acc, b),
                        gyr0=lerp(gyr0, gyr, a),
                        gyr1=gb,
                        fog_dangle=torch.tensor([(fb - fa) if fog_ok else 0.0], dtype=D),  # type: ignore[operator]
                        fog_valid=torch.tensor([1.0 if fog_ok else 0.0], dtype=D),
                    )
                    state, P, _ = propagate(
                        state, P, iv, params, cfg.noise, learned_fn(b, SensorId.IMU)
                    )
                    omega = body_rate(gb, state)
                for tp, sp, rp in pending:
                    if tp == b and b < t:
                        state, P = apply_update(state, P, tp, sp, rp, omega)
            for tp, sp, rp in pending:
                if tp == t:
                    state, P = apply_update(state, P, tp, sp, rp, body_rate(gyr, state))
            pending.clear()
            last_imu = (t, acc, gyr)
            bk.fog_prev = fog_at.get(t)
            if t_out_us is not None and t >= t_out_us:
                C = state.C_wb
                w_b = body_rate(gyr, state)
                omega_l = torch.linalg.cross(w_b, l_imu[None])
                p_b = state.p_w - (C @ l_imu)
                v_b = state.v_w - (C @ omega_l[..., None])[..., 0]
                res.out_t_us.append(t)
                res.out_p.append(p_b[0, :2])
                res.out_v.append(v_b[0, :2])
                res.out_yaw.append(yaw_of(C)[0])
        if next_snap is not None and t >= next_snap:
            res.snapshots.append(
                Snapshot(
                    t,
                    _detach_state(state),
                    P.detach().clone(),
                    LoopBook(**{**bk.__dict__, "reacq": list(bk.reacq)}),
                )
            )
            next_snap += int(snapshot_every_s * 1e6)  # type: ignore[operator]
    res.state, res.P = state, P
    return res


__all__ = ["LoopBook", "SegmentResult", "Snapshot", "run_segment"]
