"""Training-mode loop (Phase 4 pilot): equivalence with run_filter, warm starts, gradients."""

from dataclasses import replace

import pytest
import torch

from lase.filter.learned import HeadOutputs
from lase.filter.params import FilterConfig, FilterParams
from lase.filter.runner import initial_condition_from_reference, initial_state, run_filter
from lase.parameters import Parameters
from lase.training.loop import LoopBook, Snapshot, run_segment
from lase.training.pilot import Session, chunk_loss
from tests.unit.test_runner import _clean_streams
from tests.unit.test_sim_sensors import T0

FIELDS = ("p_w", "v_w", "q_wb", "b_acc", "b_gyr", "b_fog")


@pytest.fixture(scope="module")
def setup():  # type: ignore[no-untyped-def]
    streams, ref, cal = _clean_streams(1.0)
    cfg = FilterConfig()
    params = replace(FilterParams.build(cal, cfg), wheel_scale=1.0)
    ic = initial_condition_from_reference(ref, T0 + 20_000_000)
    st, P = initial_state(ic, streams, params, cfg)
    return streams, ref, cfg, params, ic, Snapshot(ic.t_us, st, P, LoopBook(ic.t_us))


def _neutral(t: int, k: object) -> HeadOutputs:
    return HeadOutputs.neutral(1)


def test_matches_run_filter(setup) -> None:  # type: ignore[no-untyped-def]
    streams, _ref, cfg, params, ic, start = setup
    t_end = T0 + 60_000_000
    ref_run = run_filter(streams, ic, cfg, params, t_end_us=t_end)
    seg = run_segment(streams, cfg, params, start, t_end, _neutral)
    for k in FIELDS:
        assert torch.equal(getattr(seg.state, k), getattr(ref_run.final_state, k)), k
    assert torch.equal(seg.P, ref_run.final_P)


def test_warm_start_continues_the_run(setup) -> None:  # type: ignore[no-untyped-def]
    streams, _ref, cfg, params, _ic, start = setup
    t_end = T0 + 60_000_000
    full = run_segment(streams, cfg, params, start, t_end, _neutral, snapshot_every_s=15.0)
    snap = full.snapshots[0]
    cont = run_segment(streams, cfg, params, snap, t_end, _neutral)
    for k in FIELDS:
        torch.testing.assert_close(
            getattr(cont.state, k), getattr(full.state, k), rtol=0, atol=1e-9
        )


@pytest.mark.parametrize("through_burn_in", [True, False])
def test_gradient_matches_finite_differences(setup, through_burn_in: bool) -> None:  # type: ignore[no-untyped-def]
    streams, ref, cfg, params, _ic, start = setup
    p = Parameters()
    w = p.training.windows.model_copy(
        update={"burn_in_s": 5.0, "loss_window_s": 5.0, "grad_through_burn_in": through_burn_in}
    )
    p = p.model_copy(update={"training": p.training.model_copy(update={"windows": w})})
    sess = Session(streams.date, streams, ref, cfg, params)
    burn = None if through_burn_in else torch.zeros(3, dtype=torch.float64)
    x0 = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    if burn is not None:
        burn = x0.clone()
    d = x0.clone().requires_grad_(True)
    loss, n = chunk_loss(sess, start, d, p, burn_delta=burn)
    assert n > 0
    loss.backward()  # type: ignore[no-untyped-call]
    assert d.grad is not None
    eps = 1e-5
    for i in range(3):
        e = torch.zeros(3, dtype=torch.float64)
        e[i] = eps
        with torch.no_grad():
            lp, _ = chunk_loss(sess, start, x0 + e, p, burn_delta=burn)
            lm, _ = chunk_loss(sess, start, x0 - e, p, burn_delta=burn)
        fd = float(lp - lm) / (2 * eps)
        assert float(d.grad[i]) == pytest.approx(fd, rel=1e-4, abs=1e-9), i
