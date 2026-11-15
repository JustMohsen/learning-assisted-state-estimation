"""Smoke test of the real-data experiment runner (needs downloaded data; short segment)."""

import datetime as dt

import pytest

from lase.data.session import extracted_dir
from lase.filter.experiment import run_session
from lase.filter.params import load_filter_config

pytestmark = pytest.mark.validation
DATE = dt.date(2013, 1, 10)


@pytest.mark.parametrize("vertical", [False, True])
def test_run_session_summary_with_both_constraint_variants(vertical: bool) -> None:
    if not (extracted_dir(DATE) / "ms25.csv").exists():
        pytest.skip("NCLT data not downloaded")
    base = load_filter_config("baseline")
    cfg = base.model_copy(
        update={"nonholonomic": base.nonholonomic.model_copy(update={"vertical_enabled": vertical})}
    )
    row = run_session(DATE, cfg, "outages", max_minutes=2.0)
    assert row["nhc_nis_mean"] > 0
    assert row["wheel_accept"] > 0.5
    assert "nodes_pos_median_m" in row
