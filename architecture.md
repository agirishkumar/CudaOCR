# CudaOCR — Architecture & Engineering Standards

> Companion: [product.md](product.md) (what/why). This file defines **how**: toolchain, design, patterns, style, testing. Every choice carries its justification. When a new significant choice is made, append it to the Decision Log (§11).

## 1. System overview

```
                 ┌────────────────────────── host (CPU) ──────────────────────────┐
 image file ──► decode (OpenCV imread) ──► pinned staging buffer ──► [H2D async]  │
                 └───────────────────────────────────────────────────────────────┘
        ┌──────────────────────────── device (GPU), one cudaStream_t ────────────────────────────┐
        │ grayscale ► denoise ► contrast ► binarize ► deskew ► line seg ► word/char seg ► recog  │
        └────────────────────────────────────────────────────────────────────────────────────────┘
                 [D2H async] ──► CTC decode/assembly ──► formatter (txt/json/hOCR) ──► stdout/file
```

- The pipeline is a sequence of **stages**; each stage is a pure function over images/buffers on the device plus a parameter struct.
- One non-default CUDA stream owns the whole pipeline run; stages enqueue async work; synchronization happens only at stage boundaries that need readback and at the end.
- Every stage has a **CPU reference implementation (oracle)** living in the test tree, never shipped in the hot path.

## 2. Repository layout (target)

```
CudaOCR/
├── CMakeLists.txt
├── product.md / architecture.md / CLAUDE.md / BACKLOG.md
├── include/cudaocr/          # public headers (.hpp, .cuh)
│   ├── core/    (device_image.cuh, stream.cuh, status.hpp, logger.hpp, config.hpp)
│   ├── stages/  (grayscale.cuh, denoise.cuh, binarize.cuh, deskew.cuh, segment.cuh, recognize.hpp)
│   └── pipeline.hpp
├── src/                      # implementations (.cu, .cpp), mirrors include/
├── apps/cudaocr_cli/main.cpp
├── tests/
│   ├── oracles/              # CPU reference implementations
│   ├── unit/                 # per-kernel tests (oracle comparison)
│   ├── integration/          # multi-stage & end-to-end tests
│   └── fixtures/             # small deterministic images + ground truth
├── benchmarks/               # google-benchmark or CUDA-event timers, vs NPP/OpenCV-CUDA
└── resources/                # sample documents (existing)
```

Justification: `include/<project>/` namespacing is the standard installable-library layout; separating oracles into `tests/` keeps the shipped binary free of OpenCV-CUDA/duplicate logic; `apps/` vs `src/` separates the CLI from the linkable library (product.md §2: "CLI + linkable static library").

## 3. Toolchain decisions

| Choice | Decision | Justification |
|---|---|---|
| CUDA | **13.1** (installed) | Already on system; forces `_Ctx` NPP APIs and sm_75+ — aligns with modern practice. |
| GPU arch | `CMAKE_CUDA_ARCHITECTURES=86` default, cache-overridable | RTX 3060 Laptop is sm_86. Old `sm_61` is unsupported by CUDA 13. Configurable for portability (product.md §6). |
| Host compiler | **g++-13** (system default) | Supported by CUDA 13.1; g++-9 no longer installed. |
| C++ standard | **C++20** (`-std=c++20` for both host and nvcc) | Supported by g++-13 + CUDA 13.1; gives `std::span`, concepts, designated initializers. Code currently uses C++14 — migrate. |
| Build system | **CMake ≥3.24** (4.3.3 installed), replacing the Makefile | First-class CUDA language support, CTest integration (TDD needs `ctest` as the verification gate), FetchContent for GoogleTest, separable compilation handling. The Makefile is deleted once CMake builds everything. |
| Warnings | `-Wall -Wextra` host, `--Werror all-warnings` off initially, **never `-w`** | The old Makefile suppressed all warnings (`-w`); that hid the missing-include bug. Warnings are part of verification. |
| Test framework | **GoogleTest** via FetchContent + **CTest** | De-facto standard, no system install needed (none present), parametrized tests fit kernel/oracle comparisons, integrates with CTest for the loop's verification condition. |
| Sanitizer | `compute-sanitizer` (memcheck) target `make sanitize` | Would have caught the Hough out-of-bounds write immediately. Mandatory for any kernel change. |
| Formatter | `clang-format` (LLVM base, 4-space indent, 100 cols), `.clang-format` in repo | Removes style debates; loop iterations auto-format. |

### Library decisions

| Library | Role | Justification |
|---|---|---|
| **OpenCV (CPU, apt `libopencv-dev`)** | Image decode/encode in the CLI; **test oracles only** elsewhere | A CUDA-enabled OpenCV doesn't exist on this system and building 4.13+contrib from source costs hours and ~10 GB for functionality we *intend to hand-write anyway* (vision: custom kernels). CPU OpenCV from apt is minutes to install and doubles as the golden reference for TDD. |
| **OpenCV CUDA modules** | **Dropped** from the product | Replaced by custom kernels (grayscale, box/adaptive threshold, warp). They were scaffolding; the vision says scaffolding goes. |
| **NPP (`_Ctx` APIs)** | Benchmarks only, not the pipeline | Non-`_Ctx` APIs are gone in CUDA 13. Keeping NPP in benchmarks gives the "within 2× of NPP" metric (product.md §6) without making it a dependency of the product. |
| **ONNX Runtime (CUDA EP)** | Recognition Phase A only | Fastest route to end-to-end text (research consensus: ONNX → TensorRT/ORT for CRNN inference). Explicitly temporary: Phase B replaces it with custom kernels and uses Phase A as oracle. |
| **Thrust/CUB** | Allowed for reductions/scans inside stages | Ships with CUDA; writing a worse scan than CUB teaches little. Custom kernels are required where the *algorithm* is the point (filters, Hough, CC labeling), CUB is fine for plumbing (argmax, prefix sums). |
| TOML parser (toml++, header-only, FetchContent) | Config file | Tiny, no system dependency. |

**Action required on this machine:** `sudo apt install libopencv-dev` (CPU build, ~4.6.0). Everything else comes from CUDA 13.1 or FetchContent.

## 4. Design principles

1. **Oracle-first TDD.** No GPU kernel is written before (a) its CPU oracle exists in `tests/oracles/` and (b) a failing test compares them. The oracle is simple, obviously-correct, unoptimized C++.
2. **RAII for every resource.** No naked `cudaMalloc`/`cudaFree` outside `core/`. `DeviceImage`, `DeviceBuffer<T>`, `Stream`, `Event` wrappers own lifetimes. (The current code leaks the Hough buffer on early-return paths.)
3. **Fail fast, fail loud.** Every CUDA/library call goes through a checking mechanism (§7). No function "always returns true".
4. **Stages are pure & composable.** `Status run(const DeviceImage& in, DeviceImage& out, const Params&, Stream&)`. No stage reads global state; the pipeline composes them. Enables testing each in isolation.
5. **Parameters are data, not constants.** Every tunable lives in a `Params` struct with documented defaults, fed from TOML/CLI. (Fixes the hardcoded blockSize=3/C=2.0 problem.)
6. **Measure before and after.** A kernel optimization without a benchmark delta in the commit message didn't happen.
7. **The GPU does the work.** Falling back to CPU inside the pipeline (like the current `warpAffine` deskew) is a bug unless justified in the Decision Log.
8. **Small headers, explicit includes.** Include-what-you-use; no `using namespace std;` ever (current `main.cu` violates this).

## 5. Design patterns (used deliberately, and why)

| Pattern | Where | Justification |
|---|---|---|
| **Pipeline** | `Pipeline` class holds an ordered list of stages | The domain *is* a pipeline; makes `--dump-stages`, per-stage timing, and stage skipping trivial. |
| **Strategy** | Stage variants (median vs Gaussian denoise; adaptive vs Otsu binarize) behind a per-stage interface, selected by config | Variants are the product's experimental surface; swapping must not touch call sites. |
| **RAII wrappers** | `core/` (DeviceImage, Stream, Event, PinnedBuffer) | Memory/stream safety; exception-safe early returns. |
| **Builder-ish config** | `PipelineConfig` assembled from TOML + CLI overrides, validated once | One validation point; stages receive already-valid params. |
| **Template method (light)** | `StageBase` provides timing + logging around a virtual `runImpl` | Per-stage GPU timing requirement without duplicating event code 8×. |
| **Oracle/golden-master testing** | tests | The core TDD enabler for numeric GPU code. |

Anti-choices: no singleton (Logger becomes an injected/explicit handle internally; the static facade remains only at the CLI edge), no inheritance-heavy hierarchy beyond `StageBase`, no premature plugin system.

## 6. Core types & stage contract

```cpp
namespace cudaocr {

enum class ErrorCode { Ok, InvalidArgument, CudaError, NppError, IoError, ModelError };
struct Status { ErrorCode code; std::string message; /* + stage name, file:line */ };

// Pitched 2D device allocation, move-only RAII.
class DeviceImage {            // core/device_image.cuh
    void* data_;  size_t pitch_;  int width_, height_, channels_;  // 8u or 32f
    cudaStream_t allocStream_;
public:
    static Result<DeviceImage> create(int w, int h, int ch, PixelType t);
    // upload/download via PinnedBuffer, async on a Stream
};

class Stream { cudaStream_t s_; /* RAII, non-copyable */ };

struct StageTiming { float ms; };
class StageBase {
public:
    Status run(const DeviceImage& in, DeviceImage& out, Stream& st);  // times + logs
protected:
    virtual Status runImpl(const DeviceImage&, DeviceImage&, Stream&) = 0;
};
} // namespace cudaocr
```

- Everything lives in `namespace cudaocr` (currently there is no namespace — fix during migration).
- `Result<T>` = `tl::expected`-style (small vendored header or C++23-ish hand-rolled) so constructors can fail without exceptions.

## 7. Error handling & logging

- **Macros** `CUDA_CHECK(expr)` / `NPP_CHECK(expr)` convert error codes into `Status` returns carrying `cudaGetErrorString`, file:line, and current stage. Kernel launches are followed by `CUDA_CHECK(cudaGetLastError())` and, in debug builds, `cudaStreamSynchronize` for precise attribution.
- **No exceptions across the library boundary**; exceptions allowed internally for truly unrecoverable states (allocation of core wrappers) but caught at `Pipeline::run`. Justification: CUDA call sites are pervasive — `Status` returns keep error paths explicit and testable; this matches the existing code's style (bool returns) but with information attached.
- **Logger**: keep the existing timestamped-file design; add a `std::mutex`, a TRACE level, stderr mirroring above a threshold, and make it injectable. Known bug to fix: missing `<sstream>` include.

## 8. Testing strategy (TDD — the law of the repo)

**Cycle (every feature, no exceptions):**
1. **Red** — write the test first: CPU oracle in `tests/oracles/`, GoogleTest case comparing oracle vs (not yet existing / stubbed) GPU implementation on fixture images. Confirm it fails/doesn't compile.
2. **Green** — write the minimal kernel/host code to pass.
3. **Refactor** — clean up under green tests; run clang-format.
4. **Verify** — full gate (below) before commit.

**Test taxonomy**
| Layer | What | Pass criterion |
|---|---|---|
| Unit (kernel) | GPU stage vs CPU oracle on fixtures: tiny synthetic images (checkerboards, ramps, known-skew text) + crops of `resources/` | Integer ops: byte-exact. Float paths: max abs diff ≤ 1 (8u) / ≤1e-4 (32f), documented per test. |
| Unit (host) | Config parsing, Status plumbing, Logger thread-safety | exact |
| Property | Idempotence (binarize twice = once), inversion symmetry, rotation(−detect(angle)) reduces profile variance | invariant holds |
| Integration | Stage chains on real `resources/` images; golden output images checked in (regenerate via `--update-goldens` env flag) | pixel diff ≤ tolerance |
| End-to-end (M3+) | Ground-truth text fixtures | char accuracy ≥ threshold per fixture |
| Regression | One test per fixed bug, named after it (first: `HoughAccumulatorBoundsRegression`) | never crashes/regresses |

**Verification gate** (the loop's verification condition, also `scripts/verify.sh`):
1. Configure+build clean: `cmake --build build -j` → zero errors, zero *new* warnings.
2. `ctest --test-dir build --output-on-failure` → 100% pass.
3. If any `.cu` under `src/` changed: `compute-sanitizer --tool memcheck ./build/tests/unit_tests --gtest_filter=<touched stages>` → 0 errors.
4. `clang-format --dry-run --Werror` on changed files.

GPU determinism note: kernels that use `atomicAdd` on floats or reduction order must either be made deterministic for tests (fixed block config, integer atomics — the Hough accumulator is already integer) or tested with tolerance, decided per kernel and documented in the test.

## 9. Coding style

- **Naming**: `PascalCase` types, `camelCase` functions/methods, `snake_case_` private members with trailing underscore, `kPascalCase` constants, `d_`/`h_` prefixes for raw device/host pointers inside kernels and launch code only. CUDA kernels: `camelCase` + `Kernel` suffix (`grayscaleKernel`). Files: `snake_case` (`device_image.cuh`).
- **Headers**: `.cuh` only if the header contains device code/CUDA types; `.hpp` otherwise (so CPU-only TUs and tests can include them without nvcc). `#pragma once`.
- **Kernels**: grid-stride loops by default; `__restrict__` on pointer params; bounds check first statement; block sizes named constants chosen by occupancy reasoning recorded in a comment when non-obvious.
- **No magic numbers**: parameters via `Params` structs; tunables documented at declaration.
- **Comments** explain *why* and constraints (memory layout, race rationale), never narrate code.
- **Commits**: imperative, scoped (`deskew: fix accumulator OOB write (regression test included)`), benchmarks quoted when perf-relevant.
- Formatting is whatever `.clang-format` says — never hand-argued.

## 10. Performance architecture

- **Streams**: one `Stream` per pipeline run; H2D/D2H async via pinned staging buffers. Multi-image batch mode (stretch) = N streams round-robin.
- **Memory**: pitched allocations for 2D images (coalescing); a simple size-bucketed `DeviceImage` pool inside `Pipeline` to stop per-run malloc churn (added only when benchmarks justify — principle §4.6).
- **Constant memory** for small LUTs (sin/cos tables for Hough — also fixes the per-thread recompute inefficiency).
- **Shared memory** tiles for separable Gaussian and median (the failed texture-memory experiment from the README is re-attempted later as a benchmark-driven task, not redone blindly).
- **Timing**: CUDA events around every stage, reported by `--benchmark`; google-benchmark (FetchContent) harness in `benchmarks/` comparing vs NPP `_Ctx`/OpenCV-CPU.

## 11. Decision log

| # | Date | Decision | Alternatives considered | Reasoning |
|---|---|---|---|---|
| 1 | 2026-06-11 | Re-found build on CMake + CUDA 13.1 + sm_86 + g++-13 + C++20 | Patch old Makefile | Old toolchain (sm_61, g++-9, non-Ctx NPP) is unsupported/uninstalled; CMake gives CTest which TDD needs. |
| 2 | 2026-06-11 | Drop OpenCV-CUDA & NPP from the product pipeline; custom kernels, libraries as benchmarks/oracles | Build OpenCV 4.13+contrib w/ CUDA from source | Hours of build for code the vision says to hand-write; CPU OpenCV (apt) suffices for I/O + oracles. |
| 3 | 2026-06-11 | GoogleTest + CTest, oracle-comparison TDD | Catch2, doctest | GTest parametrized tests fit kernel/oracle matrices; FetchContent avoids system install; largest ecosystem. |
| 4 | 2026-06-11 | Recognition phased: ONNX Runtime CUDA EP first (Phase A), custom kernels second (Phase B) with A as oracle | Tesseract bridge; custom-first; TensorRT | Tesseract is CPU & against vision; custom-first has no ground truth to test against — ORT gives an oracle and an early end-to-end win. TensorRT adds engine-build complexity for no test benefit at this scale. |
| 5 | 2026-06-11 | Status/Result error model, no exceptions across library boundary | exceptions everywhere | Pervasive CUDA call sites; explicit error paths are testable and match CUDA's C API shape. |
| 6 | 2026-06-11 | Thrust/CUB allowed for plumbing (scan/reduce/argmax), custom kernels mandatory where the algorithm is the learning target | all-custom; all-library | Maximizes learning value per hour; CUB ships with CUDA so no dependency cost. |
| 7 | 2026-06-11 | Loop-engineering workflow with BACKLOG.md + /engineer-step + verify gate | ad-hoc prompting | See CLAUDE.md "Loop engineering"; conditions defined there. |
| 8 | 2026-06-11 | Image I/O via vendored stb_image/stb_image_write; CPU oracles hand-written in `tests/oracles/` | OpenCV CPU via apt | apt install needs sudo (unavailable in the build session); stb removes the last system dependency and hand-written oracles are simple enough to be obviously correct. Supersedes the OpenCV-for-I/O half of #2. |
| 9 | 2026-06-11 | Stages are free functions `Status f(const DeviceImage&, DeviceImage&, ..., cudaStream_t)`; `runPipeline` composes and times them with CUDA events | StageBase virtual hierarchy (§5) | Same purity/timing benefits with less ceremony at the current stage count; revisit if stages need polymorphic configs. |
| 10 | 2026-06-11 | Hough votes only on baseline (bottom-edge dark) pixels | vote on all dark pixels | All-ink voting is degenerate: diagonals cutting several thick text rows out-vote true baselines (measured ±3° bias on fixtures); baseline voting fixed it and cut atomics ~10×. |
| 11 | 2026-06-11 | Polarity normalization runs on the grayscale image before binarization | invert the binarized image | Adaptive-thresholding light-on-dark text produces halo artifacts around glyphs; flipping gray first keeps one well-understood binarization path. |
| 12 | 2026-06-11 | Determinism contract: integer/fixed-point kernels are byte-exact vs oracle; float kernels (rotate) compared by outlier fraction (≤0.5% of pixels beyond ±2) | force exactness everywhere | FMA contraction makes bitwise float reproducibility CPU-vs-GPU impractical; the fixed-point design makes every other kernel exact. |

*(Append new rows; never delete — supersede with a new row referencing the old.)*

## 12. Migration plan from current code (feeds BACKLOG.md M0)

1. CMake skeleton + GoogleTest + trivial smoke test (red→green on purpose).
2. Port `Logger` (add mutex, `<sstream>`, tests) and `core/` RAII wrappers with tests.
3. Port grayscale as the **first oracle-TDD kernel** (replacing NPP call); delete OpenCV-CUDA upload/download helpers in favor of `DeviceImage`.
4. Port denoise, binarize the same way.
5. Fix Hough: correct accumulator bounds (regression test from the recorded crash), constant-memory trig, θ-window param; then GPU rotation kernel.
6. Delete Makefile, dead code (`dev/`, duplicate includes, Thrust leftovers, commented-out blocks in main), wire CLI args.
