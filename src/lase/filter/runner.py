"""Run the ESKF over a session's causal event stream.

Updates are applied at their exact acquisition times. Wheel and GPS measurements are
buffered until the next IMU sample arrives; the IMU interval is then split at the
pending measurement times, and each sub-interval is propagated with IMU values linearly
interpolated between the two bracketing samples and FOG angles interpolated (or briefly
extrapolated) from FOG samples already received. This adds at most one IMU interval
(~20 ms) of processing latency and uses no future data. Outputs are reported at the body
reference point (axle centre) after each IMU sample is processed.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from lase.data.events import SensorId, build_event_index
from lase.data.raw import F64, I64, RowFlag
from lase.data.reference import Reference
from lase.data.session import SensorStreams
from lase.evaluation.estimate import NavEstimate
from lase.filter.learned import HeadOutputs
from lase.filter.params import FilterConfig, FilterParams
from lase.filter.propagation import ImuInterval, PropagationDiagnostics, propagate
from lase.filter.so3 import matrix_to_quat, skew, yaw_of
from lase.filter.state import BA, BF, BG, N_ERR, P_, TH, V_, NavState
from lase.filter.updates import update_gnss, update_nonholonomic, update_wheel
from lase.geometry.euler import euler_zyx_to_matrix

D = torch.float64


@dataclass(frozen=True)
class InitialCondition:
    """Common initial-pose prior: body-frame pose at ``t_us`` (from the reference)."""

    t_us: int
    p_body_w: F64  # (3,)
    rpy: F64  # (3,)


def initial_condition_from_reference(
    ref: Reference, t_us: int | None = None, start_offset_s: float = 1.0
) -> InitialCondition:
    """Reference pose at ``t_us`` (default: first valid reference time + ``start_offset_s``)."""
    t0 = int(ref.gt.t_us[ref.gt.valid][0]) + int(start_offset_s * 1e6) if t_us is None else t_us
    p, rpy = ref.pose_at(np.array([t0], dtype=np.int64))
    return InitialCondition(t0, p[0], rpy[0])


@dataclass
class UpdateLog:
    """Per-update diagnostics collected during a run (NumPy, for evaluation)."""

    t_us: list[int] = field(default_factory=list)
    innovation: list[F64] = field(default_factory=list)
    S: list[F64] = field(default_factory=list)
    nis: list[float] = field(default_factory=list)
    accepted: list[bool] = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        """Stacked arrays."""
        return {
            "t_us": np.asarray(self.t_us, dtype=np.int64),
            "innovation": np.asarray(self.innovation),
            "S": np.asarray(self.S),
            "nis": np.asarray(self.nis),
            "accepted": np.asarray(self.accepted, dtype=bool),
        }


@dataclass(frozen=True)
class RunResult:
    """Output of one filter run."""

    estimate: NavEstimate
    wheel: dict[str, np.ndarray]
    gnss: dict[str, np.ndarray]
    nonholonomic: dict[str, np.ndarray]
    biases: dict[str, np.ndarray]  # t_us, acc (N,3), gyr (N,3), fog (N,) at output times
    final_state: NavState
    final_P: Tensor
    n_imu_steps: int
    wall_time_s: float
    gnss_recoveries: int


LearnedFn = Callable[[int, SensorId], HeadOutputs]


def initial_state(
    ic: InitialCondition, streams: SensorStreams, params: FilterParams, cfg: FilterConfig
) -> tuple[NavState, Tensor]:
    """Nominal state and covariance at ``ic.t_us``: reference pose, wheel-based velocity, zero biases."""
    C = euler_zyx_to_matrix(*ic.rpy)
    p_imu = ic.p_body_w + C @ params.l_imu.numpy()
    w = streams.wheels
    k = max(int(np.searchsorted(w.t_us, ic.t_us, side="right")) - 1, 0)
    v_fwd = 0.5 * (w.left[k] + w.right[k]) / params.wheel_scale
    v_w = C @ np.array([v_fwd, 0.0, 0.0])
    state = NavState(
        p_w=torch.as_tensor(p_imu, dtype=D)[None],
        v_w=torch.as_tensor(v_w, dtype=D)[None],
        q_wb=matrix_to_quat(torch.as_tensor(C, dtype=D)[None]),
        b_acc=torch.zeros(1, 3, dtype=D),
        b_gyr=torch.zeros(1, 3, dtype=D),
        b_fog=torch.zeros(1, 1, dtype=D),
        wheel_scale=torch.full((1, 1), params.wheel_scale, dtype=D),
    )
    pr = cfg.prior
    sig = np.r_[
        [pr.pos_m] * 3,
        [pr.vel_mps] * 3,
        pr.att_rad,
        [pr.acc_bias_mps2] * 3,
        [pr.gyr_bias_radps] * 3,
        [pr.fog_bias_radps],
        [pr.wheel_scale],
    ]
    return state, torch.diag(torch.as_tensor(sig, dtype=D) ** 2)[None]


def _output_jacobian(C: Tensor, l_imu: Tensor) -> Tensor:
    """d [north, east, v_n, v_e, heading] / d error state (1, 5, 16)."""
    J = torch.zeros(1, 5, N_ERR, dtype=D)
    J[0, 0:2, P_] = torch.eye(3, dtype=D)[:2]
    J[0, 0:2, TH] = (C[0] @ skew(l_imu))[:2]  # p_body = p_imu - C l_imu
    J[0, 2:4, V_] = torch.eye(3, dtype=D)[:2]
    J[0, 4, TH] = C[0, 2, :]  # world-z component of the rotation error
    return J


class _FogHistory:
    """Causal access to the unwrapped FOG angle (valid samples only)."""

    def __init__(
        self,
        streams: SensorStreams,
        max_extrapolation_s: float = 0.05,
        extrapolation_baseline_s: float = 0.02,
    ) -> None:
        m = streams.fog.valid
        self.t = streams.fog.t_us[m]
        self.a = streams.fog_angle[m]
        self.max_extrapolate_us = int(max_extrapolation_s * 1e6)
        self.baseline_us = int(extrapolation_baseline_s * 1e6)

    def angle_at(self, t_query: int, t_now: int) -> float | None:
        """Angle at ``t_query`` using only samples with time <= ``t_now``."""
        n_avail = int(np.searchsorted(self.t, t_now, side="right"))
        if n_avail < 2:
            return None
        j = int(np.searchsorted(self.t[:n_avail], t_query, side="right")) - 1
        if j < 0:
            return None
        if j + 1 < n_avail:  # bracketed: interpolate
            t0, t1 = self.t[j], self.t[j + 1]
            w = (t_query - t0) / (t1 - t0)
            return float(self.a[j] + w * (self.a[j + 1] - self.a[j]))
        if t_query - self.t[j] > self.max_extrapolate_us:
            return None
        # Extrapolate with a rate over a minimum baseline: FOG samples can arrive ~0.1 ms
        # apart, and a two-sample rate over such a gap is wildly wrong (one-sample spikes).
        i0 = max(
            int(np.searchsorted(self.t[:n_avail], self.t[j] - self.baseline_us, side="right")) - 1,
            0,
        )
        if i0 >= j:
            return float(self.a[j])
        rate = (self.a[j] - self.a[i0]) / (self.t[j] - self.t[i0])
        return float(self.a[j] + rate * (t_query - self.t[j]))


def run_filter(
    streams: SensorStreams,
    ic: InitialCondition,
    cfg: FilterConfig,
    params: FilterParams,
    t_end_us: int | None = None,
    learned_fn: LearnedFn | None = None,
) -> RunResult:
    """Run the filter from ``ic.t_us`` to ``t_end_us`` (default: end of the IMU stream)."""
    t_start = time.perf_counter()
    state, P = initial_state(ic, streams, params, cfg)
    neutral = HeadOutputs.neutral(1)
    learned = learned_fn or (lambda t, s: neutral)
    idx = build_event_index(streams).window(ic.t_us, t_end_us or int(streams.imu.t_us[-1]) + 1)
    fog_hist = _FogHistory(
        streams, cfg.fog_handling.max_extrapolation_s, cfg.fog_handling.extrapolation_baseline_s
    )

    imu, wh, gps = streams.imu, streams.wheels, streams.gps
    held_bit = np.uint16(RowFlag.GPS_HELD)
    n_max = int((idx.sensor == SensorId.IMU).sum()) + 1
    out_t = np.zeros(n_max, dtype=np.int64)
    out_p, out_v = np.zeros((n_max, 2)), np.zeros((n_max, 2))
    out_h, out_cov = np.zeros(n_max), np.zeros((n_max, 5, 5))
    out_ba, out_bg, out_bf = np.zeros((n_max, 3)), np.zeros((n_max, 3)), np.zeros(n_max)
    out_ws = np.zeros(n_max)
    n_out = 0
    wheel_log, gnss_log, nhc_log = UpdateLog(), UpdateLog(), UpdateLog()
    last_imu: tuple[int, Tensor, Tensor] | None = None
    pending: list[tuple[int, int, int]] = []  # (time, sensor, row)
    fog_prev: float | None = (
        None  # FOG angle used at the previous boundary (keeps increments telescoping)
    )
    last_gnss_accept = ic.t_us
    last_gnss_queued = -(10**18)
    gnss_rejects = 0
    recoveries = 0
    n_steps = 0
    l_imu = params.l_imu
    Cb = params.C_b_imu

    def body_rate(gyr: Tensor, st: NavState) -> Tensor:
        return (gyr - st.b_gyr) @ Cb.T

    def apply_update(
        st: NavState, Pm: Tensor, t: int, sensor: int, row: int, omega: Tensor
    ) -> tuple[NavState, Tensor]:
        nonlocal last_gnss_accept, gnss_rejects, recoveries
        if sensor == SensorId.WHEEL:
            st, Pm, ud = update_wheel(
                st,
                Pm,
                torch.tensor([wh.left[row]], dtype=D),
                torch.tensor([wh.right[row]], dtype=D),
                omega,
                params,
                cfg.wheel,
                learned(t, SensorId.WHEEL),
            )
            if cfg.nonholonomic.enabled:
                st, Pm, nd = update_nonholonomic(
                    st, Pm, omega, params, cfg.nonholonomic, learned(t, SensorId.WHEEL)
                )
                nhc_log.t_us.append(t)
                nhc_log.innovation.append(nd.innovation[0].numpy())
                nhc_log.S.append(nd.S[0].numpy())
                nhc_log.nis.append(float(nd.nis[0]))
                nhc_log.accepted.append(bool(nd.accepted[0]))
            log = wheel_log
        else:
            z = torch.as_tensor(streams.gnss_ned[row, :2], dtype=D)[None]
            st, Pm, ud = update_gnss(st, Pm, z, params, cfg.gnss, learned(t, SensorId.GNSS))
            if bool(ud.accepted[0]):
                last_gnss_accept, gnss_rejects = t, 0
            else:
                gnss_rejects += 1
                if (
                    (t - last_gnss_accept) > cfg.gnss.recovery_after_s * 1e6
                    and gnss_rejects >= cfg.gnss.recovery_min_rejections
                ):
                    # re-acquisition: inflate position uncertainty, then retry this fix
                    Pm = Pm.clone()
                    Pm[:, P_, :] = 0.0
                    Pm[:, :, P_] = 0.0
                    Pm[:, P_, P_] = torch.eye(3, dtype=D) * cfg.gnss.recovery_pos_sigma_m**2
                    st, Pm, ud = update_gnss(st, Pm, z, params, cfg.gnss, learned(t, SensorId.GNSS))
                    recoveries += 1
                    last_gnss_accept, gnss_rejects = t, 0
            log = gnss_log
        log.t_us.append(t)
        log.innovation.append(ud.innovation[0].numpy())
        log.S.append(ud.S[0].numpy())
        log.nis.append(float(ud.nis[0]))
        log.accepted.append(bool(ud.accepted[0]))
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
                cfg.gnss.enabled
                and not (cfg.gnss.skip_held and (gps.flags[row] & held_bit))
                and (t - last_gnss_queued) >= cfg.gnss.min_interval_s * 1e6
            ):
                pending.append((t, int(s), row))
                last_gnss_queued = t
            continue
        # IMU sample: process the interval since the previous sample, split at pending updates
        acc = torch.as_tensor(imu.acc[row], dtype=D)[None]
        gyr = torch.as_tensor(imu.gyr[row], dtype=D)[None]
        if last_imu is None or t <= last_imu[0]:
            if last_imu is None:
                for (
                    tp,
                    sp,
                    rp,
                ) in pending:  # before the first IMU sample: apply at the initial state
                    state, P = apply_update(state, P, tp, sp, rp, body_rate(gyr, state))
                pending.clear()
            last_imu = (t, acc, gyr)
            continue
        t0, acc0, gyr0 = last_imu
        pending.sort()
        bounds = [t0] + [tp for tp, _, _ in pending if t0 < tp < t] + [t]
        late = [(tp, sp, rp) for tp, sp, rp in pending if tp <= t0]
        for tp, sp, rp in late:  # arrived at/before t0 (ties): apply now
            state, P = apply_update(state, P, tp, sp, rp, body_rate(gyr0, state))
        span = float(t - t0)

        def lerp(a: Tensor, b: Tensor, tq: int, t0: int = t0, span: float = span) -> Tensor:
            w = (tq - t0) / span
            return a + w * (b - a)

        omega = body_rate(gyr0, state)
        fog_at: dict[int, float | None] = {}
        if cfg.use_fog:
            fog_at[t0] = fog_prev if fog_prev is not None else fog_hist.angle_at(t0, t)
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
                state, P, _ = propagate(state, P, iv, params, cfg.noise, learned(b, SensorId.IMU))
                omega = body_rate(gb, state)
            for tp, sp, rp in pending:
                if tp == b and b < t:
                    state, P = apply_update(state, P, tp, sp, rp, omega)
        for tp, sp, rp in pending:  # measurements exactly at the new IMU time
            if tp == t:
                state, P = apply_update(state, P, tp, sp, rp, body_rate(gyr, state))
        pending.clear()
        n_steps += 1
        last_imu = (t, acc, gyr)
        fog_prev = fog_at.get(t)
        if n_steps % cfg.output_every == 0:
            C = state.C_wb
            w_b = body_rate(gyr, state)
            omega_l = torch.linalg.cross(w_b, l_imu[None])
            p_b = state.p_w - (C @ l_imu)
            v_b = state.v_w - (C @ omega_l[..., None])[..., 0]
            J = _output_jacobian(C, l_imu)
            out_t[n_out] = t
            out_p[n_out] = p_b[0, :2].numpy()
            out_v[n_out] = v_b[0, :2].numpy()
            out_h[n_out] = float(yaw_of(C)[0])
            out_cov[n_out] = (J @ P @ J.transpose(-1, -2))[0].numpy()
            out_ba[n_out] = state.b_acc[0].numpy()
            out_bg[n_out] = state.b_gyr[0].numpy()
            out_bf[n_out] = float(state.b_fog[0, 0])
            out_ws[n_out] = float(state.wheel_scale[0, 0])
            n_out += 1

    est = NavEstimate(
        cfg.name,
        out_t[:n_out],
        out_p[:n_out],
        out_v[:n_out],
        out_h[:n_out],
        out_cov[:n_out],
    )
    biases = {
        "t_us": out_t[:n_out],
        "acc": out_ba[:n_out],
        "gyr": out_bg[:n_out],
        "fog": out_bf[:n_out],
        "wheel_scale": out_ws[:n_out],
    }
    _ = (BA, BG, BF, PropagationDiagnostics)
    return RunResult(
        estimate=est,
        wheel=wheel_log.arrays(),
        gnss=gnss_log.arrays(),
        nonholonomic=nhc_log.arrays(),
        biases=biases,
        final_state=state,
        final_P=P,
        n_imu_steps=n_steps,
        wall_time_s=time.perf_counter() - t_start,
        gnss_recoveries=recoveries,
    )


__all__ = ["I64", "InitialCondition", "RunResult", "initial_condition_from_reference", "run_filter"]
