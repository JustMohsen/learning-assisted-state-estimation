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

## Steps

- [x] 1. Split manifest: `configs/data/split.yaml` + pydantic model + tests; add `pyarrow`
- [x] 2. Downloader: `src/lase/data/download.py` + `scripts/download_nclt.py` (resumable, SHA-256 recorded in `configs/data/nclt_files.yaml`, selective extraction). Bring-up on 2013-01-10 and one training day; list archive contents.
- [ ] 3. Full download of all 27 sessions (background), verify checksums, record extracted size
- [ ] 4. Raw loaders per sensor with schemas and validity flags + synthetic-fixture tests
- [ ] 5. Calibration research and `configs/data/nclt_calibration.yaml` (origin, frames, lever arms, track width, FOG axis/sign), every value sourced or marked estimated
- [ ] 6. Conversions: geodesy (exact WGS-84 → NED), FOG unwrap/increments, unit checks + tests
- [ ] 7. Reference: dense poses, graph nodes, derived velocity with masks, separate API
- [ ] 8. Session object + causal event stream + tests (ordering, ties, windows, no reference leakage)
- [ ] 9. Parquet cache keyed by loader version
- [ ] 10. All-session audit → `reports/phase1_audit/`
- [ ] 11. Validation checks V1–V5 → `reports/phase1_audit/`
- [ ] 12. Phase 1 summary; update README status

## Questions for the user

(none yet)

## Log

- 2026-11-15: worklog created; overnight run set up (wake-up cron, caffeinate).
- 2026-11-15 01:21: Step 1 done: split manifest (configs/data/split.yaml), SplitManifest model with disjointness check, tests; pyarrow added.
- 2026-11-15 01:23: Step 2 done: downloader (resumable, trust-on-first-use SHA-256 in configs/data/nclt_files.yaml, flat safe extraction) + scripts/download_nclt.py; bring-up on 2013-01-10 and 2012-01-22 (~12 MB/s). Archives hold 12 files: gps, gps_rtk, gps_rtk_err, ms25, ms25_euler, kvh, wheels, odometry_mu(_100hz), odometry_cov(_100hz), README.txt; all extracted (extraction ~2.7x compressed size). GT starts ~1 s after sensors; covariance graph nodes look ~19 s apart (to audit).

## Summary

(written when Phase 1 completes)
