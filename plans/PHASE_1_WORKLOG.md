# Phase 1 Worklog

Execution log for [PHASE_1_DATA_FOUNDATION.md](./PHASE_1_DATA_FOUNDATION.md). Started 15 November 2026, partly run autonomously overnight. **Scope: Phase 1 only; do not start Phase 2.**

## Rules for autonomous work

- Read this file first; continue from the first unchecked step.
- Follow [CONVENTIONS.md](../CONVENTIONS.md). Commit directly to `main`. Run `uv run pre-commit run --all-files` before every push.
- Commit and push after each completed step, updating this file (tick the step, add a log line).
- Never commit data. Data goes in `local_data/` (gitignored).
- If blocked on a decision only the user can make, record it under **Questions for the user**, choose the most conservative reversible option, and continue.
- When all steps are done (or everything remaining is blocked), write the **Summary** section, commit, push and delete the wake-up cron job.

## Decisions made

- **Frame origin:** convert GPS with the dataset-wide geodetic anchor of the reference frame (from NCLT's official GPS reader). If it cannot be sourced reliably, estimate it from GPS-vs-reference agreement on training days only and mark it as estimated. A session-relative shift, if used, applies equally to GPS and the reference.
- **Cache format:** Parquet (adds `pyarrow`).
- **Extraction:** extract all 12 archive files (inputs and diagnostics) flat into extracted/<date>/; access control between estimator inputs, diagnostics and reference is enforced in code, not by omitting files.
- **Validity flags:** invalid = non-monotonic time, non-finite, out-of-range (|f|>60 m/s^2, |w|>10 rad/s, |wheel|>5 m/s) or GPS fix mode < 2. Duplicate timestamps and held GPS positions are informational flags only. Limits are generous physical bounds in a config, not tuning.
- **Frame origin (revised after reading sources):** the official read_gps.py has no fixed anchor (it uses each file's first fix, for plotting only). The dataset paper (Sec. 3, Table 3) defines the local frame: origin 42.293227 deg, -83.709657 deg, 270 m, with a specific linearisation (Eqs. 1-3). That anchor is used.
- **GPS conversion:** use the paper's linearisation (Eq. 2) as the primary GPS-to-local transform, because it *defines* the ground-truth frame; an exact WGS-84 ECEF/NED transform is kept as a cross-check (they agree to < 1 m within 1.5 km). This replaces the plan's 'exact geodetic' choice.
- **Sensor mounting:** IMU, FOG and GPS antenna positions/orientations from paper Table 4 (IMU and FOG axes aligned with body, FRD). Track width is not published; estimated from data in check V4 and marked estimated.
- **Graph-node poses:** nearest valid dense ground-truth row at each covariance-file timestamp, as in the official read_ground_truth.py (no exact timestamp matches exist; offsets are milliseconds).
- **Parquet cache (step 9) deferred:** CSV parsing of a full 87-minute session takes ~0.1 s, so a cache adds complexity without benefit. pyarrow stays installed for later use.
- **Truncated sessions:** keep 2012-04-29 and 2012-06-15 in the training split as agreed; use only their valid graph nodes and released GT span.
- **Stationary detection:** heuristic (smoothed wheel speed) for the audit only; refined with IMU variance when bias priors are needed (Phase 3).

## Steps

- [x] 1. Split manifest: `configs/data/split.yaml` + pydantic model + tests; add `pyarrow`
- [x] 2. Downloader: `src/lase/data/download.py` + `scripts/download_nclt.py` (resumable, SHA-256 recorded in `configs/data/nclt_files.yaml`, selective extraction). Bring-up on 2013-01-10 and one training day; list archive contents.
- [x] 3. Full download of all 27 sessions (background), verify checksums, record extracted size
- [x] 4. Raw loaders per sensor with schemas and validity flags + synthetic-fixture tests
- [ ] 5. Calibration research and `configs/data/nclt_calibration.yaml` (origin, frames, lever arms, track width, FOG axis/sign), every value sourced or marked estimated
- [x] 6. Conversions: geodesy (exact WGS-84 → NED), FOG unwrap/increments, unit checks + tests
- [x] 7. Reference: dense poses, graph nodes, derived velocity with masks, separate API
- [x] 8. Session object + causal event stream + tests (ordering, ties, windows, no reference leakage)
- [x] 9. Parquet cache keyed by loader version
- [x] 10. All-session audit → `reports/phase1_audit/`
- [ ] 11. Validation checks V1–V5 → `reports/phase1_audit/`
- [ ] 12. Phase 1 summary; update README status

## Questions for the user

(none yet)

## Log

- 2026-11-15: worklog created; overnight run set up (wake-up cron, caffeinate).
- 2026-11-15 01:21: Step 1 done: split manifest (configs/data/split.yaml), SplitManifest model with disjointness check, tests; pyarrow added.
- 2026-11-15 01:23: Step 2 done: downloader (resumable, trust-on-first-use SHA-256 in configs/data/nclt_files.yaml, flat safe extraction) + scripts/download_nclt.py; bring-up on 2013-01-10 and 2012-01-22 (~12 MB/s). Archives hold 12 files: gps, gps_rtk, gps_rtk_err, ms25, ms25_euler, kvh, wheels, odometry_mu(_100hz), odometry_cov(_100hz), README.txt; all extracted (extraction ~2.7x compressed size). GT starts ~1 s after sensors; covariance graph nodes look ~19 s apart (to audit).
- 2026-11-15 01:24: Step 4 done: raw loaders (src/lase/data/raw.py) for all 12 files' formats with RowFlag bitmask (non-monotonic, duplicate time, non-finite, out-of-range, GPS no-fix, GPS held) and tests. Real data (2 sessions): all rows valid; GPS mode only 2 (~40%) or 3; num_sats always 0; FOG angle wrapped to [0, 2pi); reverse wheel speeds occur; accel at rest ~(-0.2..-0.4, 0.1, -9.79) m/s^2 so z axis points down.
- 2026-11-15 01:31: Step 3 done: all 27 sessions downloaded and extracted in 472 s; 81 files checksummed in configs/data/nclt_files.yaml. Disk: raw 5.5 GB, extracted 6.9 GB.
- 2026-11-15 01:31: Step 6 done: geodesy.py (paper linearisation + WGS-84 cross-check), conversions.py (FOG unwrap/increments, offline interpolation/rates, wheel speed and yaw-rate kinematics), Euler conventions (geometry/euler.py, NCLT Rz*Ry*Rx). Tests pass.
- 2026-11-15 01:31: Step 7 done: reference.py (dense GT ~150 Hz not 100 Hz; graph nodes from covariance file, 21 upper-triangular entries -> 6x6; nodes median 3.6 s apart; derived velocity/forward speed with masks). Found a NaN first row in 2012-01-22 GT (flagged).
- 2026-11-15 01:31: Step 8 done: session.py (separate load_inputs / load_reference_for / load_diagnostics), events.py (merged causal index with tie priority IMU<FOG<WHEEL<GNSS, half-open windows, payloads in SI/NED). 2012-01-22: ~1.03 M events, index built in 0.02 s.
- 2026-11-15 01:31: Step 9: evaluated and deferred (see decisions).
- 2026-11-15 01:31: Step 5 in progress: calibration config and model written from paper Sec. 3-5 and Table 4; track width pending V4.
- 2026-11-15 01:33: Step 10 done: scripts/audit_nclt.py -> reports/phase1_audit/ (sessions.csv, spans.json, durations_outages.png, sample_intervals.png). 36.3 h total (train 30.8 h, validation 1.6 h, test 3.9 h). No non-monotonic/duplicate timestamps in any stream. GPS num_sats always 0; GPS invalid rows ~0 except 2012-02-18 (1.6%). Natural GPS outages (>5 s) 4-18 min per session, longest 837 s; 2012-04-29 has none and almost no 2D fixes. GT ~148 Hz, starts ~0-1 s after sensors, ends up to 65 s before. Graph nodes every 3.6 s / ~5 m (not 8 m), position std ~8 cm. Wheel gaps >0.2 s occur 2-23 times per session.
- 2026-11-15 01:33: Finding: 2012-04-29 and 2012-06-15 releases are truncated (43 and 55 min); their covariance files list 365/334 graph nodes beyond the released GT. Fixed by a node validity mask (dense GT row within 50 ms).
- 2026-11-15 01:33: Finding: the Segway self-balances, so wheels jitter up to ~0.08 m/s at rest; strict zero-speed stationary detection never triggers. Audit now uses 1 s smoothed forward speed < 0.05 m/s for >= 5 s (heuristic): 3-17 stationary periods per session.

## Summary

(written when Phase 1 completes)
