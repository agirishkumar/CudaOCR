# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read these first

- **[product.md](product.md)** — vision, scope, full feature spec, roadmap (M0–M5), audited current status, known bugs. Decides *what* to build and in what order.
- **[architecture.md](architecture.md)** — toolchain, library decisions, design principles/patterns, stage contract, error model, coding style, testing strategy, decision log. Decides *how*. Any new significant choice must be appended to its Decision Log (§11) with reasoning.
- **[BACKLOG.md](BACKLOG.md)** — ordered task queue with acceptance criteria; the unit of work for loop engineering (below).

## Project state (important)

The legacy Makefile build is **broken on this machine** and is being replaced: CUDA 13.1 removed the non-`_Ctx` NPP APIs and `sm_61` support used by the old code, `g++-9` is uninstalled, and OpenCV is not installed. Target toolchain: CMake + CUDA 13.1 + sm_86 + g++-13 + C++20. Do not "fix" the Makefile; execute the migration plan in architecture.md §12. OpenCV (CPU) must be installed via `sudo apt install libopencv-dev` before M0 completes — ask the user to run it if missing.

## Commands

Target build/test workflow (once M0 lands; until then, creating this is the work):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release        # configure
cmake --build build -j                            # build
ctest --test-dir build --output-on-failure        # all tests
ctest --test-dir build -R <TestName>              # single test (regex)
./build/tests/unit_tests --gtest_filter=Suite.Case  # single gtest case
compute-sanitizer --tool memcheck ./build/tests/unit_tests  # after any kernel change
./scripts/verify.sh                               # full verification gate
clang-format -i $(git diff --name-only -- '*.cu' '*.cuh' '*.hpp' '*.cpp')
```

## Non-negotiable workflow rules

1. **TDD always** (architecture.md §8): CPU oracle + failing GoogleTest first, then the kernel, then refactor. No GPU code lands without an oracle-comparison test.
2. **Verification gate before every commit**: clean build, 100% ctest, memcheck-clean for touched kernels, clang-format clean.
3. Custom CUDA kernels are the product; OpenCV-CUDA/NPP appear only in tests/benchmarks (architecture.md §3).
4. No naked `cudaMalloc`/`cudaFree` outside `core/`; use the RAII wrappers. All errors via `Status`/`CUDA_CHECK` — never silently return true.
5. Every fixed bug gets a named regression test before the fix.
6. One bug to never reintroduce: the Hough accumulator is `rho * theta` ints indexed `r * theta + t`; bounds must check `r < rho` (see product.md §4.2).

## Loop engineering protocol

Iterative autonomous development runs as `/loop /engineer-step` (command defined in `.claude/commands/engineer-step.md`; state in `LOOP_STATE.md`). Each iteration takes exactly one BACKLOG.md task through red→green→refactor→verify. Conditions:

- **Result condition** (iteration succeeded): the task's acceptance criteria are met, its new test(s) went red→green, and the verification gate passes.
- **Verification condition**: `./scripts/verify.sh` exits 0 — clean build, all tests pass, memcheck clean for touched `.cu` files, formatting clean.
- **Step-progression condition**: only after the result condition holds AND the work is committed may the task be checked off in BACKLOG.md and the next task started. Never work on two tasks in one iteration.
- **Stop condition**: stop the loop when (a) BACKLOG.md has no unchecked tasks in the current milestone, (b) the same task has failed 3 consecutive iterations (record the blocker in LOOP_STATE.md under `BLOCKED`), or (c) a decision requiring the user appears — record it in LOOP_STATE.md under `NEEDS-DECISION` and stop rather than guessing.
