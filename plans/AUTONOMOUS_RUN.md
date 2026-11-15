# Autonomous Run Control

Instructions for unattended work sessions, started 15 November 2026 (night 2). Any agent resuming work reads this file first.

## Scope for this run (revised again by the user in the morning)

1. **Phase 2:** complete and hardened.
2. **Phase 3: resume and finish** from the restored prototype. Continue from the first unchecked step of [PHASE_3_WORKLOG.md](./PHASE_3_WORKLOG.md). The review order afterwards is Phases 1 and 2 first, then Phase 3.
3. **Do not start Phase 4.**
4. At the end, update [REVIEW_PHASES.md](./REVIEW_PHASES.md) with the Phase 3 results and decisions.

## Rules

- Continue from the first unchecked step of the current phase's worklog.
- Follow [CONVENTIONS.md](../CONVENTIONS.md). Commit directly to `main`; run `uv run pre-commit run --all-files` before every push.
- Commit and push after each completed step; tick the step and add a log line.
- Never commit data (`local_data/` is gitignored).
- Record every non-trivial choice under the worklog's *Decisions made*, and anything needing the user under *Questions for the user*. When blocked on a user decision, choose the most conservative reversible option and continue.
- Do not reopen agreed scope (see [PROJECT_SCOPE.md](../PROJECT_SCOPE.md)); raise questions instead.
- When finished or fully blocked: update `REVIEW_PHASES.md`, commit, push, then remove the wake-up cron jobs (CronList / CronDelete).
