---
description: Execute exactly one TDD iteration from BACKLOG.md (red → green → refactor → verify → commit)
---

You are running one iteration of the CudaOCR loop-engineering protocol. Read CLAUDE.md, architecture.md (§8 testing, §9 style), and BACKLOG.md before acting.

## Procedure

1. **Select task.** Read `LOOP_STATE.md`. If it names a `CURRENT` task, resume it. Otherwise pick the first unchecked task in the active milestone of `BACKLOG.md`, write it to `LOOP_STATE.md` as `CURRENT` with `attempts: 1` (or increment `attempts` if resuming after a failed iteration).

2. **Check stop conditions FIRST — if any hold, do not work; report and end:**
   - No unchecked tasks remain in the active milestone → write `MILESTONE-COMPLETE` in LOOP_STATE.md. State clearly that the loop should stop.
   - `attempts >= 3` for the current task → move it to a `BLOCKED` section in LOOP_STATE.md with a precise description of what failed and why, clear `CURRENT`. State clearly that the loop should stop pending user input.
   - The task requires a product/architecture decision not covered by product.md/architecture.md → write it under `NEEDS-DECISION` in LOOP_STATE.md with the options and your recommendation. State clearly that the loop should stop pending user input.

3. **Red.** Write the test(s) for the task's acceptance criteria first (CPU oracle in `tests/oracles/` if it's a kernel task). Build and run to confirm the new test FAILS (or fails to compile). If it passes immediately, the test is wrong — strengthen it.

4. **Green.** Implement the minimal code to pass, following architecture.md §4–§9 (RAII, Status/CUDA_CHECK, stage contract, naming).

5. **Refactor.** Clean up under green tests. Run clang-format on changed files.

6. **Verify (verification condition).** Run `./scripts/verify.sh` (if it doesn't exist yet — early M0 — run the equivalent manually: clean build, full ctest, memcheck on touched kernels). It must exit 0.

7. **Progress (step-progression condition).** Only if verification passed:
   - Check the task off in BACKLOG.md.
   - Clear `CURRENT` and reset `attempts` in LOOP_STATE.md; append one line to its `DONE` log: task id, date, commit subject.
   - Commit everything with a scoped message (e.g. `core: add DeviceImage RAII wrapper (M0.5)`).

8. **If verification failed**: do NOT check off the task, do NOT commit broken state (you may commit the failing test on a clearly-marked WIP only if reverting would lose significant work — prefer leaving the working tree dirty). Record the failure summary in LOOP_STATE.md under `CURRENT.lastFailure`. The next iteration will retry (attempts+1).

## Hard rules

- Exactly ONE backlog task per iteration. Never start a second.
- Never weaken a test or a tolerance to get green. If the acceptance criterion seems wrong, that's a NEEDS-DECISION stop.
- Never delete or reorder checked-off backlog items.
- End your final message with exactly one of these markers so the loop result is auditable:
  `ITERATION-RESULT: DONE <task-id>` | `ITERATION-RESULT: RETRY <task-id> (attempt N/3)` | `ITERATION-RESULT: STOPPED <reason>`
