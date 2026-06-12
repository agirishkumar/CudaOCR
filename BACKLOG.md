# BACKLOG — ordered task queue for loop engineering

Rules: tasks are executed top-to-bottom within the active milestone, one per loop iteration.
A task is checked off only when its acceptance criteria pass via `scripts/verify.sh` and the work is committed.
Acceptance criteria must be machine-checkable. Add/split tasks freely; never reorder a checked task.

Active milestone: **M2** (M0, M1 complete 2026-06-11)

## M0 — Re-foundation (build + test harness)

- [x] **M0.1 CMake skeleton + GoogleTest.** Create CMakeLists.txt (CUDA 13.1, C++20, arch 86 cache var, `-Wall -Wextra`), FetchContent GoogleTest, empty `cudaocr` static lib + `unit_tests` target with one trivial failing test; make it pass.
  *Accept:* `cmake -B build && cmake --build build -j && ctest --test-dir build` → 1/1 pass.
- [x] **M0.2 scripts/verify.sh.** Implement the verification gate (build, ctest, conditional memcheck on changed .cu, clang-format check) + `.clang-format`.
  *Accept:* `./scripts/verify.sh` exits 0 on clean tree; exits non-zero when a test is deliberately broken (verified once, then reverted).
- [x] **M0.3 Status/Result + CUDA_CHECK.** `include/cudaocr/core/status.hpp` with ErrorCode, Status, Result<T>, CUDA_CHECK macro. Unit tests for message formatting and macro behavior (mockable via a failing cudaMalloc size).
  *Accept:* tests cover Ok path, CUDA error path with file:line in message.
- [x] **M0.4 Logger v2.** Port existing Logger: add mutex, TRACE/WARN levels, `<sstream>` fix, stderr mirror ≥ WARN, injectable sink for tests. Thread-safety test (N threads × M messages, no interleaving corruption).
  *Accept:* logger tests pass incl. concurrency test.
- [x] **M0.5 core RAII wrappers.** `Stream`, `DeviceBuffer<T>`, `DeviceImage` (pitched, move-only, async upload/download with pinned staging). Tests: alloc/move/upload-download roundtrip byte-exact; memcheck clean.
  *Accept:* roundtrip test green + memcheck 0 errors.
- [x] **M0.6 Grayscale kernel (first oracle-TDD kernel).** CPU oracle (BT.601) in tests/oracles; `grayscaleKernel` grid-stride on DeviceImage. Replaces NPP path.
  *Accept:* byte-exact vs oracle on synthetic ramp + `resources/sample.png`.
- [x] **M0.7 CLI skeleton.** `apps/cudaocr_cli`: arg parsing (input path, `-o`, `--dump-stages`, `--log-level`), loads image via OpenCV CPU, runs load→grayscale→save. Integration test runs the binary on sample.png and checks output exists & matches golden.
  *Accept:* integration test green; `./build/apps/cudaocr resources/sample.png --dump-stages out/` produces grayscale image.
- [x] **M0.8 Legacy teardown.** Delete Makefile, `dev/`, duplicate/unused includes, commented-out blocks, `using namespace std`; move remaining legacy .cu logic into a `legacy/` reference folder or delete where superseded. README updated to new build instructions.
  *Accept:* build green, ctest green, `grep -r "using namespace std" src include apps` empty.

## M1 — Preprocessing custom kernels (each task = oracle + kernel + benchmark)

- [x] **M1.1 Box-filter / integral image kernel** (foundation for adaptive threshold). Oracle: OpenCV CPU `boxFilter`. *Accept:* max abs diff ≤ 1 vs oracle.
- [x] **M1.2 Adaptive mean threshold kernel** (params blockSize, C; uses M1.1). Regression vs current behavior on fixtures. *Accept:* byte-exact vs oracle.
- [x] **M1.3 Otsu binarization** (GPU 256-bin histogram + host argmax). *Accept:* threshold value == oracle's on 5 fixtures.
- [x] **M1.4 Median filter 3×3/5×5 kernel** (shared-memory tiles). Oracle: OpenCV CPU `medianBlur`. *Accept:* byte-exact.
- [x] **M1.5 Separable Gaussian kernel** (shared memory; sigma param). *Accept:* max abs diff ≤ 1 vs OpenCV CPU `GaussianBlur`.
- [x] **M1.6 Histogram equalization kernel.** *Accept:* byte-exact vs oracle.
- [x] **M1.7 Inversion detection + normalization.** *Accept:* property test — inverted fixture normalizes to same output as original.
- [x] **M1.8 Benchmark harness** (google-benchmark): each M1 kernel vs NPP `_Ctx`/OpenCV-CPU equivalent; results table generated into `benchmarks/RESULTS.md`. *Accept:* table generated; each kernel within 2× of library baseline or a follow-up optimization task is filed.
- [ ] **M1.9 Median filter optimization** (currently 27× slower than NPP, see benchmarks/RESULTS.md): shared-memory tiles + sorting network for 3×3; histogram or network approach for 5×5. *Accept:* byte-exact tests stay green; ≤2× of NPP in `./build/bench`.
- [ ] **M1.10 Gaussian optimization** (currently 2.75× of NPP): shared-memory tile per block for both passes. *Accept:* byte-exact tests stay green; ≤2× of NPP.

## M2 — Layout & segmentation (expand when M1 nears completion)

- [x] **M2.1 Hough skew detection — fixed.** Regression test `HoughAccumulatorBoundsRegression` reproducing the old OOB (bounds `r < rho`), constant-memory sin/cos, θ window ±15°, oracle = CPU Hough. *Accept:* detected angle within 0.5° of known-skew synthetic fixtures; memcheck clean.
- [x] **M2.2 GPU rotation kernel** (bilinear, replaces CPU warpAffine). *Accept:* max abs diff ≤ 2 vs OpenCV CPU `warpAffine` away from borders.
- [x] **M2.3 Horizontal projection profile + line detection.** *Accept:* line count & boundaries match ground truth on 3 fixtures.
- [x] **M2.4 Vertical profiles → word boxes.** *Accept:* word count matches ground truth fixtures.
- [ ] **M2.5 Connected-component labeling (GPU union-find) → char boxes.** *Accept:* component count matches oracle (OpenCV CPU `connectedComponents`).

## M3+ — defined in product.md §7; expand into tasks when M2 is underway.
