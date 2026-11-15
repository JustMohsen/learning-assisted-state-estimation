# Conventions and Interface Contract

Rules every contributor, human or agent, follows so independently built parts fit together. Project context: [PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md), [PROJECT_SCOPE.md](./PROJECT_SCOPE.md), [DEVELOPMENT_PLAN.md](./DEVELOPMENT_PLAN.md).

**Changing anything here** requires a single commit that updates this file *and* every affected module together.

---

## 1. Environment and workflow

| Topic | Rule |
|---|---|
| Environment | `uv sync` creates the environment from `uv.lock`. Run everything through `uv run ...`. |
| Dependencies | Add with `uv add <pkg>` or `uv add --dev <pkg>`. Never `pip install` into the project environment. Commit `pyproject.toml` and `uv.lock` together. |
| Hooks | `uv run pre-commit install` once per clone. Hooks: ruff format, ruff lint, mypy, notebook-output stripping, basic file hygiene. |
| Branches | During initial development, commit directly to `main`. Run `uv run pre-commit run --all-files` before every push, and keep `main` green. Feature branches and pull requests are optional (e.g. for risky experiments) and may become the default later. |
| Commits | Imperative, concise subject (`Add GNSS update`, `Fix FOG unwrap sign`). One logical change per commit. |
| Data | Never committed. Data lives in `local_data/` at the repo root (gitignored except its README), or under `$LASE_DATA_DIR` if set. Access it only through `lase.utils.paths` (`data_dir()`, `nclt_dir()`), never hard-coded paths. Caches and run outputs live in gitignored directories. |
| Third-party code | Cloned into `external/` (gitignored), never copied into `src/`. Record the upstream commit used. |

## 2. Code organisation

- **Library code** lives in `src/lase/`. **Scripts** in `scripts/` are thin command-line wrappers: parse arguments, load config, call library functions.
- **Notebooks** are for exploration only. Anything reused moves into `src/lase/` with tests. Outputs are stripped on commit.
- **Functional numerical core.** Filter steps are pure functions of their inputs (section 6). They do not mutate arguments, keep hidden state or read global config. Classes are used for orchestration: runners, datasets, models (`torch.nn.Module`).
- **Configs, not constants.** Tunable values come from pydantic configs (`lase.utils.config.Config` subclasses) loaded from `configs/`. Physical constants (e.g. Earth parameters) live in one module with sources cited.
- **Style.** ruff (line length 100) formats and lints. mypy runs in strict mode on `src/`. Every public function has type hints and a docstring stating tensor shapes, frames and units.
- **Naming.** pep8 except for established mathematical symbols: `P`, `F`, `H`, `K`, `Q`, `R`, `C_wb` are allowed and preferred where they match the equations.

## 3. Numbers, tensors and time

| Topic | Rule |
|---|---|
| Units | SI throughout: m, s, rad, m/s, rad/s, m/s². Never degrees internally. |
| Time | Absolute timestamps are `int64` microseconds (`t_us`, NCLT's `utime`). Interval lengths are `float64` seconds (`dt`). Never store absolute time as float32. |
| Filter precision | All filter states, covariances and Jacobians are `torch.float64`. |
| Network precision | Learned modules are `torch.float32`. Cast explicitly at the boundary (`.to(torch.float64)`), which is differentiable. |
| Batch dimension | Every filter tensor has a **leading batch dimension `B`**, even when `B = 1`. Vectors are `(B, n)`, matrices `(B, n, m)`. |
| Device | CPU by default for the filter. Any device choice is explicit, never implicit. |
| Causality | Estimator code receives only events with `t_us <= now`. Reference data is accessible only to losses and evaluation, never to filter or network inputs. |

## 4. Frames and attitude

| Item | Convention |
|---|---|
| World frame `w` | Local North-East-Down (NED) tangent frame. The origin (lat, lon, alt) is set in the data config and recorded with every run. |
| Body frame `b` | Forward-Right-Down, attached to the robot. Navigation position and velocity are those of the **IMU origin**. Reported outputs are transformed to NCLT's body reference point. |
| Sensor frames | `imu`, `fog`, `wheel`, `gnss` (antenna). Each has an explicit, configured rotation and lever arm relative to `b`. Sensor biases are expressed in their own sensor axes. |
| Rotation naming | `C_ab` maps vectors from frame `b` to frame `a`: `v_a = C_ab @ v_b`. So `C_wb` is body-to-world. |
| Quaternions | Hamilton convention, scalar first `[w, x, y, z]`, unit norm, representing `C_wb`. |
| Attitude error | Right-multiplicative, in body axes: `C_wb_true = C_wb_est @ Exp(dtheta)`. |
| Position and vector naming | `p_w_imu` = IMU position in world. `v_w` = IMU velocity in world. `omega_b` = body angular rate. Lever arm `l_b_imu_gnss` = vector from IMU to GNSS antenna in body axes. |
| Heading | Yaw angle of `C_wb` about world Down, wrapped to `(-pi, pi]`. Heading errors are always wrapped before squaring or averaging. |
| External code | Some reference implementations (e.g. OpenVINS) use JPL quaternions and left errors. Convert at the boundary; never mix conventions inside `src/lase`. |

## 5. State layout

### 5.1 Nominal state

| Field | Shape | Units | Meaning |
|---|---|---|---|
| `p_w` | `(B, 3)` | m | IMU position, world NED |
| `v_w` | `(B, 3)` | m/s | IMU velocity, world NED |
| `q_wb` | `(B, 4)` | - | Body-to-world quaternion |
| `b_acc` | `(B, 3)` | m/s² | Accelerometer bias, IMU axes |
| `b_gyr` | `(B, 3)` | rad/s | Gyroscope bias, IMU axes |
| `b_fog` | `(B, 1)` | rad/s | FOG rate bias, FOG axis |

### 5.2 Error state (16 core coordinates, fixed order)

| Indices | Symbol | Meaning |
|---|---|---|
| 0–2 | `dp` | Position error, world |
| 3–5 | `dv` | Velocity error, world |
| 6–8 | `dtheta` | Attitude error, body (right-multiplicative) |
| 9–11 | `db_acc` | Accelerometer-bias error |
| 12–14 | `db_gyr` | Gyroscope-bias error |
| 15 | `db_fog` | FOG-bias error |

The covariance `P` is `(B, 16, 16)` in this order. Later augmentations (e.g. calibration states) are appended after index 15 and declared here when added. Index constants live in one module (`lase.filter.state`), so code never hard-codes numbers like `9:12`.

## 6. Interfaces

These are the agreed signatures. Exact class names may be refined when the modules are written, but the shape of the contract must not change.

### 6.1 Events

```text
Event:
    t_us:    int64                    # acquisition timestamp, microseconds
    sensor:  SensorId                 # IMU | FOG | WHEEL | GNSS
    data:    sensor-specific record   # SI units, sensor axes
    valid:   bool                     # False for flagged/held/out-of-range samples
```

The event stream is sorted by `t_us`. Ties use a fixed priority: propagation inputs (IMU, FOG) before updates (WHEEL, GNSS).

### 6.2 Filter steps

```text
propagate(state, P, interval, params, learned) -> (state, P, PropagationDiagnostics)
update_wheel(state, P, measurement, params, learned) -> (state, P, UpdateDiagnostics)
update_gnss(state, P, measurement, params, learned) -> (state, P, UpdateDiagnostics)
```

- `interval`: IMU and FOG data over one completed time interval, plus `dt`.
- `params`: fixed calibration and baseline noise, from config.
- `learned`: a `HeadOutputs` instance (6.3). With neutral values the filter is the analytical baseline.
- `UpdateDiagnostics`: innovation, innovation covariance `S`, NIS, accepted/rejected flag and reason. Used by evaluation and fed back (pre-update only) as GRU context.

### 6.3 Learned outputs (`HeadOutputs`)

| Field | Shape | Units | Neutral value | Bounding |
|---|---|---|---|---|
| `bias_drift` | `(B, 6)` | accel m/s³ (0–2), gyro rad/s² (3–5) | 0 | scaled `tanh` |
| `wheel_residual` | `(B, 1)` | m/s | 0 | scaled `tanh` |
| `wheel_var_mult` | `(B, 1)` | - | 1 | bounded positive, default [0.25, 25] |
| `gnss_var_mult` | `(B, 2)` | - (North, East) | 1 | bounded positive, default [0.25, 25] |

Method B (learned constants) produces the same `HeadOutputs` from learnable constant parameters. The filter cannot tell the difference, which is what makes A/B/C directly comparable.

## 7. Testing and validation

| Tier | Location | Runs | Purpose |
|---|---|---|---|
| Unit tests | `tests/unit/` | CI, every push | Fast checks of individual functions: shapes, conventions, edge cases, known small examples |
| Validation | `tests/validation/` (marked `validation`) | Explicitly: `uv run pytest tests/validation` | Checks against independent references (simulator truth, oracle residuals, Monte Carlo, published baselines), as listed per phase in the development plan. Figures and tables go to `reports/`. |

A phase is complete only when its validation checks pass, not just its unit tests.

## 8. Runs and reproducibility

- Every run creates its directory via `lase.utils.provenance.create_run_dir`, which records the resolved config, git commit and dirty flag, seed, versions and data manifest.
- Results intended for reporting come from a **clean** working tree (`dirty: false`).
- Seeds are set with `lase.utils.seeding.seed_everything`. Reported comparisons use three seeds.
- Test-day data is used only for frozen final comparisons (development plan, Phase 7).
