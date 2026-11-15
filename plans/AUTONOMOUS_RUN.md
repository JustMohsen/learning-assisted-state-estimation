# Autonomous Run Control

Instructions for unattended work sessions, started 15 November 2026 (night 2). Any agent resuming work reads this file first.

## Scope for this run (revised by the user ~02:50)

1. **Phase 2** (simulator + evaluation harness): complete. Continue **hardening it thoroughly**: fit simulator error models to real data, validate across more sessions, and add an evaluation entry point. Log: [PHASE_2_WORKLOG.md](./PHASE_2_WORKLOG.md).
2. **Phase 3: planning only.** Do not implement on `main`. The draft plan [PHASE_3_BASELINE.md](./PHASE_3_BASELINE.md) is discussed with the user first. A prototype lives on branch `phase3-prototype` (do not merge).
3. **Do not start Phase 4.**
4. At the end, write or update [REVIEW_PHASES.md](./REVIEW_PHASES.md): the decisions made, summaries, and where user input is needed (Phases 1 and 2, plus the Phase 3 draft plan).

## Rules

- Continue from the first unchecked step of the current phase's worklog.
- Follow [CONVENTIONS.md](../CONVENTIONS.md). Commit directly to `main`; run `uv run pre-commit run --all-files` before every push.
- Commit and push after each completed step; tick the step and add a log line.
- Never commit data (`local_data/` is gitignored).
- Record every non-trivial choice under the worklog's *Decisions made*, and anything needing the user under *Questions for the user*. When blocked on a user decision, choose the most conservative reversible option and continue.
- Do not reopen agreed scope (see [PROJECT_SCOPE.md](../PROJECT_SCOPE.md)); raise questions instead.
- When finished or fully blocked: update `REVIEW_PHASES.md`, commit, push, then remove the wake-up cron jobs (CronList / CronDelete).
