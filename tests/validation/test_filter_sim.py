"""Phase 3 baseline on simulated sessions (needs downloaded data for the trajectory)."""

import datetime as dt

import pytest

from lase.data.session import extracted_dir
from lase.filter.validation import run_sim_scenario

pytestmark = pytest.mark.validation
DATE = dt.date(2013, 1, 10)


@pytest.fixture(autouse=True)
def _need_data() -> None:
    if not (extracted_dir(DATE) / "ms25.csv").exists():
        pytest.skip("NCLT data not downloaded")


def test_clean_scenario_tracks_truth() -> None:
    r = run_sim_scenario("clean", DATE)
    m = r["metrics"]
    assert m["pos_rmse_m"] < 0.05
    assert m["heading_rmse_deg"] < 0.2  # includes causal-association lag
    assert m["vel_rmse_mps"] < 0.02


def test_matched_noise_scenario_is_consistent() -> None:
    r = run_sim_scenario("matched", DATE)
    assert 1.5 < r["nees_pos"].mean < 3.0  # ideal 2
    assert r["nees_pos"].coverage_95 > 0.88
    assert 0.8 < r["nis_wheel"].mean < 1.2  # ideal 1
    assert 1.6 < r["nis_gnss"].mean < 2.5  # ideal 2
