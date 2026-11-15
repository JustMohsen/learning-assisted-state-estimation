# Validation checks

Slower checks against independent references (simulator truth, oracle-state residuals,
Monte Carlo covariance, published baselines, ...), as described per phase in
[DEVELOPMENT_PLAN.md](../../DEVELOPMENT_PLAN.md). They are not run in CI by default.

Mark each test with `@pytest.mark.validation` and run explicitly:

    uv run pytest tests/validation

Checks that produce figures or tables write them to `reports/`.
