import numpy as np

from lase.filter.validation import monte_carlo_propagation


def test_propagated_covariance_matches_monte_carlo() -> None:
    r = monte_carlo_propagation(n_runs=400, duration_s=10.0, seed=0)
    # 400 runs: sampling std of a variance ratio ~ sqrt(2/400) = 7%; allow ~4 sigma.
    assert np.all(r.var_ratio > 0.72), r.var_ratio
    assert np.all(r.var_ratio < 1.3), r.var_ratio
