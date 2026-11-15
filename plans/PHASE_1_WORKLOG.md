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
- **FOG sign:** NED yaw = -1 x released FOG angle (V2, all sessions). Recorded as fog_convention.angle_sign_to_ned_yaw; the event payload keeps the released angle and the filter applies the sign.
- **Wheel geometry:** physical track width 0.560 m; wheel speed scale recorded as a training-median estimate (1.054) with its seasonal drift documented, not hard-coded into loaders.
- **Validation method:** scales/sign from 1-s window increments (rate regressions on 100 Hz differentiated signals were attenuated by noise, e.g. 0.86-0.90); V1 frame fit on inliers with median residuals (GPS multipath outliers dominated least squares).

## Steps

- [x] 1. Split manifest: `configs/data/split.yaml` + pydantic model + tests; add `pyarrow`
- [x] 2. Downloader: `src/lase/data/download.py` + `scripts/download_nclt.py` (resumable, SHA-256 recorded in `configs/data/nclt_files.yaml`, selective extraction). Bring-up on 2013-01-10 and one training day; list archive contents.
- [x] 3. Full download of all 27 sessions (background), verify checksums, record extracted size
- [x] 4. Raw loaders per sensor with schemas and validity flags + synthetic-fixture tests
- [x] 5. Calibration research and `configs/data/nclt_calibration.yaml` (origin, frames, lever arms, track width, FOG axis/sign), every value sourced or marked estimated
- [x] 6. Conversions: geodesy (exact WGS-84 → NED), FOG unwrap/increments, unit checks + tests
- [x] 7. Reference: dense poses, graph nodes, derived velocity with masks, separate API
- [x] 8. Session object + causal event stream + tests (ordering, ties, windows, no reference leakage)
- [x] 9. Parquet cache keyed by loader version
- [x] 10. All-session audit → `reports/phase1_audit/`
- [x] 11. Validation checks V1–V5 → `reports/phase1_audit/`
- [x] 12. Phase 1 summary; update README status

## Questions for the user

- **Wheel scale in the baseline/pilot:** the scope fixes wheel scale in the pilot, but data shows 1.044-1.085 (drifting over months). Options: (a) estimate wheel scale as a slow analytical state in the Phase 3 baseline (recommended: cheap, physically motivated, fair to all methods); (b) fixed training-median 1.054; (c) keep 1.0 and let learning absorb it. This affects baseline fairness and the scope's pilot definition.
- **Validation data volume:** validation is 97 min, ~80 of them from 2013-02-23 (2013-01-10 is 17 min). The split stays as agreed unless you want to revisit it; if checkpoint selection proves noisy, cross-validation over training days is an alternative.
- ~~**Consumer GPS mode-2 fixes** may deserve different noise than mode-3.~~ **Answered in Phase 2 (step 10):** accuracy is nearly equal (2.94 vs 2.78 m) and modes alternate almost every fix; no separate noise needed.

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
- 2026-11-15 01:38: Step 5 done: calibration config complete. Published values from paper (origin, linearisation, Table 4 mounts); estimated from data: FOG sign (-1), physical track width 0.560 m, raw-units width 0.589 m, wheel speed scale 1.054 (training medians).
- 2026-11-15 01:38: Step 11 done: crosscheck.py + scripts/validate_phase1.py + tests/validation/test_phase1.py (17 pass). V1 GPS median 2.9-5.6 m, inlier rotation <=0.32 deg, scale <=0.53%. V2 IMU scale ~+0.998, FOG scale ~-0.998 (sign reversed), lags within ~2.5 ms. V3 wheel scale 1.044-1.085 drifting upward over the year. V4 track width R2 0.96-0.99. V5 integrity passes.
- 2026-11-15 01:38: Step 12 done: reports/phase1_audit/README.md (results, answers to open questions, implications); README status updated to Phase 1 complete; local_data README documents scripted download; Phase 1 plan links to outcome.

## Review outcome (15 November 2026)

**Accepted by the user as complete.** Decisions:
- **Wheel scale:** estimate as a slow state in the Phase 3 baseline, initialised from the previous session's final estimate by date (first session: 1.0).
- **Split:** keep the agreed 22/2/3 split.
- **GPS gating:** robust gating agreed.
- **Headline metric:** decided in the Phase 2 review.

See [REVIEW_PHASES.md](./REVIEW_PHASES.md), section 4a. Clarification: "training median" = median of the per-session wheel-speed/reference-speed ratios over the 22 training-split sessions (no machine learning involved).

## Summary

**Phase 1 is complete** (15 November 2026, one overnight session). All 12 steps are done; step 9 (Parquet cache) was evaluated and deliberately deferred.

**Built** (`src/lase/data/`, 60 tests: 43 unit + 17 validation):
- `splits`, `download`: split manifest; resumable downloader with SHA-256 manifest; `scripts/download_nclt.py`
- `raw`: loaders for all 12 file formats, with per-row validity flags
- `calibration`: sourced config (`configs/data/nclt_calibration.yaml`)
- `geodesy`, `conversions`, `geometry/euler`: frames and conversions
- `reference`: dense poses, graph nodes, derived velocity
- `session`, `events`: role-separated access and a causal event stream
- `audit`, `crosscheck`: audit and validation, with `scripts/audit_nclt.py` and `scripts/validate_phase1.py`

**Data:** all 27 sessions (36.3 h) in `local_data/` (5.5 GB raw + 6.9 GB extracted), checksummed.

**Key results** (details in [reports/phase1_audit/README.md](../reports/phase1_audit/README.md)):
- Frames, timing, signs and scales are all validated against the reference.
- GPS median error is 3–5.6 m, with multipath outliers.
- **The FOG sign is reversed.** IMU and FOG timing match the reference to about 2 ms.
- **Wheel speeds read 4–8% high,** drifting over the year.
- The physical track width is 0.560 m.
- Two training sessions are truncated releases (handled).
- The Segway self-balances, which affects stationary detection.

**Decisions made autonomously:** see *Decisions made* above. The most consequential:
- using the paper's frame linearisation instead of an exact geodetic transform;
- the FOG sign;
- deferring the cache.

**Needs the user:** see *Questions for the user*. The most important is whether the baseline should estimate wheel scale.

**Not done (by design):** Phase 2 has not been started.
