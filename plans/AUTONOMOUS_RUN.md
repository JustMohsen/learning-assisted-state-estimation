# Autonomous Run Control

Instructions for unattended work sessions, started 15 November 2026 (night 2). Any agent resuming work reads this file first.

## Scope for this run

1. **Phase 2** (simulator + evaluation harness): plan → implement → validate → summarise. Plan: [PHASE_2_SIM_EVAL.md](./PHASE_2_SIM_EVAL.md); log: [PHASE_2_WORKLOG.md](./PHASE_2_WORKLOG.md).
2. If Phase 2 is complete: **Phase 3** (analytical baseline): plan → implement as far as possible. Plan: `PHASE_3_BASELINE.md`; log: `PHASE_3_WORKLOG.md`.
3. **Do not start Phase 4.** It requires a review and discussion with the user first.
4. At the end (or when blocked), write or update [REVIEW_PHASES.md](./REVIEW_PHASES.md): the decisions made, the summaries, and where user input is needed, covering Phases 1, 2 and 3.

## Rules

- Continue from the first unchecked step of the current phase's worklog.
- Follow [CONVENTIONS.md](../CONVENTIONS.md). Commit directly to `main`; run `uv run pre-commit run --all-files` before every push.
- Commit and push after each completed step; tick the step and add a log line.
- Never commit data (`local_data/` is gitignored).
- Record every non-trivial choice under the worklog's *Decisions made*, and anything needing the user under *Questions for the user*. When blocked on a user decision, choose the most conservative reversible option and continue.
- Do not reopen agreed scope (see [PROJECT_SCOPE.md](../PROJECT_SCOPE.md)); raise questions instead.
- When finished or fully blocked: update `REVIEW_PHASES.md`, commit, push, then remove the wake-up cron jobs (CronList / CronDelete).
