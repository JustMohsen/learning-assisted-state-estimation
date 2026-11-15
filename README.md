# Learning-assisted state estimation

A physics-based Kalman filter for ground-robot navigation, with a small causal neural network (GRU) that adapts the filter's normally hand-tuned parts (sensor bias drift, wheel-model error and sensor trust) to the current situation. The network is trained end to end through the filter. The project uses the NCLT dataset (Segway robot; IMU, fibre-optic gyro, wheels, consumer GPS; no cameras) and asks whether context-dependent adaptation beats the best fixed settings, especially during GPS outages.

**Status:** scope agreed (14 November 2026). No code, trained model or results yet.

## Start here

1. **[PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md):** the one-page big picture. Read this first.
2. **[PROJECT_SCOPE.md](./PROJECT_SCOPE.md):** the comprehensive scope: design, data, training, comparisons, evaluation, positioning and exclusions.

These two documents define the project. Where anything else in the repository conflicts with them, they win.

3. **[DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md):** how it gets built: phases, dependencies, repository setup and conventions, and how each part is validated against independent references.
4. **[CONVENTIONS.md](./CONVENTIONS.md):** coding rules, frames and units, state layout and interface contract. Read before writing code.

## Development setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed by uv if missing.

```bash
uv sync                          # create the environment from uv.lock
uv run pre-commit install        # enable format/lint/type-check hooks
uv run pytest                    # unit tests
cp .env.example .env             # then set LASE_DATA_DIR to the NCLT data location
```

Code lives in `src/lase/` (package `lase`), with thin CLIs in `scripts/`, configs in `configs/`, unit tests in `tests/unit/` and validation checks in `tests/validation/`. See [CONVENTIONS.md](./CONVENTIONS.md).

## Background material

| Location | Content |
|---|---|
| [research/](./research/) | Earlier research notes: NCLT sensor/reference audit, estimator mathematics, calibration, training/evaluation discussion, literature and prior-work reviews, decision log. Useful as evidence and detail; some of it describes the broader earlier scope. |
| [archive/PROJECT_SNAPSHOT.md](./archive/PROJECT_SNAPSHOT.md) | The earlier consolidated design, including the camera/vision pipeline now out of scope. Archived for reference only. |

Treat dated counts, cost estimates and unselected proposals in the background material as historical.
