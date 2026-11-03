# CudaOCR — Architecture & Engineering Standards

> **How** CudaOCR is built: toolchain, design, interfaces, testing and process. What and why:
> [product.md](product.md). Every significant choice is recorded with its reasoning in the
> decision log (§11); new choices are appended there.

## 1. System overview

```
 host     input file ─► decode (stb_image) or render PDF page (poppler) ─► HostImage (8-bit)
          ─► DeviceImage upload on the pipeline's stream
 device   toGray (grayscale, 3-channel input only)
 (one     ─► preprocess list (default: median 3×3, polarity) ───────────────► G (gray page)
 non-         └─► binarizePrep list (default: empty) ─► binarizer (adaptive C=40 │ otsu)
 blocking     ─► deskewer (hough): angle from the binary page, rotate binary AND G
 stream)      ─► detector (projection): line + word boxes on the deskewed binary page
              ─► recognizer (onnx): crop/resize/normalize lines of deskewed G (line_prep kernel)
                 ─► PP-OCRv5 via ONNX Runtime (CUDA EP, device tensor in) ─► CTC probabilities
 host     decoder (ctc-greedy) ─► line text/conf + words from CTC timesteps ─► PageResult
          ─► formatter (txt │ json │ hocr) ─► stdout or <out>/<stem>.<ext>
```

- The pipeline is a sequence of **components** behind small interfaces (§6), created by name
  from a `PipelineConfig` through a registry. Underneath, the GPU work is done by free-function
  kernels in `stages/` over `DeviceImage`s plus parameters.
- One `Pipeline` owns one non-blocking CUDA stream and its device buffers (reused across
  pages); the CLI creates one `Pipeline` per run, so a batch of pages costs one model load.
- Every custom kernel has a **CPU reference implementation (oracle)** in `tests/oracles/`,
  never shipped.
- Optional dependencies stay optional: without ONNX Runtime the pipeline detects but does not
  recognize; without poppler PDFs are reported as unsupported.

## 2. Repository layout

```
CudaOCR/
├── CMakeLists.txt            # library cudaocr_lib, apps, tests, bench, install
├── README.md, product.md, architecture.md, BACKLOG.md, LOOP_STATE.md
├── include/cudaocr/          # public headers (.hpp host-only, .cuh need CUDA)
│   ├── core/                 status, cuda_check, device_buffer (DeviceBuffer, Stream), device_image,
│   │                         host_image (stb load/save), logger, device_info, pdf
│   ├── stages/               CUDA kernel entry points: grayscale, filters, binarize, enhance,
│   │                         deskew, segment (+ segment_types), line_prep
│   ├── components/           interfaces (image_stage, deskewer, text_detector, line_recognizer,
│   │                         decoder, formatter, ctc, params, stage_context) and built-ins
│   │                         (preprocess, hough_deskewer, projection_detector,
│   │                         onnx_line_recognizer, ctc_greedy_decoder, words, txt/json/hocr)
│   ├── eval/metrics.hpp      CER/WER
│   ├── types.hpp             BBox, WordResult, LineResult, PageResult
│   ├── registry.hpp          ComponentRegistry, ComponentSpec
│   ├── pipeline.hpp          PipelineConfig, Pipeline, StageImages
│   └── config.hpp            TOML <-> PipelineConfig
├── src/                      mirrors include/ (src/components/builtin.cpp registers built-ins)
├── apps/                     cudaocr_cli (cudaocr), cudaocr_eval (cudaocr-eval)
├── tests/
│   ├── oracles/              CPU reference implementations
│   ├── unit/, integration/   GoogleTest (one binary: unit_tests)
│   ├── scripts/              CLI/tooling tests run by ctest (bash; exit 77 = skipped)
│   └── fixtures/             synth/ (generated pages + GT), real/ (GT for resources/),
│                             pdf/, onnx/ (tiny Add model), eval/
├── benchmarks/               bench.cu (custom vs NPP), RESULTS.md (speed), ACCURACY.md (CER/WER)
├── docs/                     config.md, output-formats.md
├── models/                   fetched model (gitignored) + README.md model card
├── scripts/                  verify.sh, changed_files.sh, fetch_models.sh, fetch_unlv.sh,
│                             prepare_unlv.py, gen_fixtures.py, accuracy_report.sh, onnx_info.py
├── resources/                sample documents (sherlock.png is the clean reference page)
└── third_party/stb/          vendored image I/O
```

`data/` (UNLV download) and `build*/` are gitignored. `CLAUDE.md` and `.claude/` are
local-only assistant configuration.

## 3. Toolchain and dependencies

| Item | Choice | Why |
|---|---|---|
| CUDA | 13.x (developed on 13.4; driver 595.99 supports 13.2) | Newer toolkit than driver works only with native SASS embedded (decision #13). |
| GPU arch | `CMAKE_CUDA_ARCHITECTURES=86` default, set **before** `project()`, overridable | RTX 3060 Laptop; after `project()` CMake's default (75) silently wins. |
| Host compiler / standard | system g++ (15.2), **C++20** for host and device code | `std::span`, concepts, designated initializers; pinning a g++ version broke once. |
| Build | CMake ≥ 3.24, FetchContent for external sources | CTest is the verification gate; no system installs needed beyond CUDA. |
| Warnings | `-Wall -Wextra`, never `-w` | Suppressed warnings once hid a missing include. |
| Tests | GoogleTest 1.15 (FetchContent) + CTest | Parametrized oracle comparisons; ctest also runs the bash script tests. |
| Sanitizer / format | `compute-sanitizer --tool memcheck`; `clang-format` (`.clang-format`) | Both part of `scripts/verify.sh`. |

| Library | Role | Notes |
|---|---|---|
| stb_image / stb_image_write (vendored) | image decode/encode | replaced OpenCV (decision #8) |
| ONNX Runtime 1.31.1 `gpu_cuda13` (FetchContent, SHA-256) | recognizer backend; CPU EP is the oracle for M5 | optional (`CUDAOCR_WITH_ORT`); headers only in `src/components/onnx_line_recognizer.cpp` (decision #22) |
| cuDNN 9 for CUDA 13 (system) | needed by ORT's CUDA EP | `cudnn9-cuda-13` |
| PP-OCRv5 English recognition, ONNX (fetched) | recognizer weights | Apache-2.0; `models/README.md` (decision #16) |
| toml++ 3.4.0 (FetchContent, header-only) | config files | only `src/config.cpp` includes it (decision #26) |
| poppler-cpp (system `libpoppler-cpp-dev`) | PDF page rendering | optional (`CUDAOCR_WITH_PDF`), GPL-2+ (decision #28) |
| NPP `_Ctx` (CUDA) | benchmark baseline only | never in the pipeline |
| Thrust/CUB | allowed for plumbing (scans, reductions) | custom kernels where the algorithm is the point (decision #6) |
| Python 3 + Pillow | developer scripts only (fixtures, UNLV prep, test PDF) | never at build or run time |

## 4. Design principles

1. **Oracle-first TDD.** No GPU kernel without (a) a CPU oracle in `tests/oracles/` and (b) a
   failing test comparing them.
2. **RAII for every resource.** No naked `cudaMalloc`/`cudaFree` outside `core/`.
3. **Fail loud.** Every CUDA/ORT/IO call is checked and surfaces as a `Status` with context.
4. **Components are pure and composable.** They read inputs and parameters only; `Pipeline`
   composes them, so each is testable alone.
5. **Parameters are data.** Every tunable is a component parameter with a documented default,
   settable from TOML or flags.
6. **Measure before and after.** Accuracy changes go through `accuracy_report.sh` and the
   ratchets; speed changes quote a benchmark.
7. **The GPU does the pixel work.** CPU work in the pipeline is limited to decisions on small
   data (profiles, histograms, CTC decoding).
8. **Small headers, explicit includes,** no `using namespace std`.

## 5. Design patterns

| Pattern | Where | Why |
|---|---|---|
| Pipeline | `Pipeline` | The domain is a pipeline; dumps, timing and slot skipping fall out of it. |
| Strategy | component interfaces (§6) | Stages are swappable (projection → DBNet, ORT → custom CUDA). |
| Registry / factory | `ComponentRegistry`: name → factory(`Params`) | A new implementation is one class + one registration; configs refer to names. |
| RAII | `core/` wrappers, `PdfDocument` | Memory/stream safety with early returns. |
| Pimpl | `Pipeline`, `OnnxLineRecognizer`, `PdfDocument` | Public headers stay host-only and free of CUDA/ORT/poppler. |
| Timing decorator | `Pipeline` wraps each component call in CUDA events named after the component | Per-stage timing without per-component code. |
| Golden master / ratchet | `PipelineGolden`, `Accuracy.*` | Pin geometry and accuracy so refactors can't silently change them. |

No singletons beyond the logger facade and the process-lifetime ORT environment (kept alive
deliberately to avoid a crash at exit; `Ort.ExitIsClean`).

## 6. Core types and interfaces

```cpp
namespace cudaocr {
enum class ErrorCode { Ok, InvalidArgument, CudaError, IoError, InternalError };
struct Status { ErrorCode code; std::string message; bool ok() const; };
template <class T> class Result;            // value or Status; ok(), value(), take(), status()

// core/ — all transfers/fills take the caller's stream (decision #20)
template <class T> class DeviceBuffer;      // create(n), zero/upload/download(…, stream)
class Stream;                               // RAII cudaStreamNonBlocking
class DeviceImage;                          // pitched 8-bit, 1 or 3 channels; create, fromHost,
                                            // ensure(w,h,c) (reuse), copyTo, upload/download
struct HostImage;                           // interleaved 8-bit; loadImage()/saveImage() via stb
class PdfDocument;                          // open(path), pageCount(), renderPage(i, dpi) -> gray

// types.hpp — what components exchange (boxes inclusive, page pixels, deskewed frame)
struct BBox { int x0, y0, x1, y1; };        // default empty; unite(), width(), height()
struct WordResult { BBox box; std::string text; float conf; std::vector<BBox> glyphs; };
struct LineResult { BBox box; std::vector<WordResult> words; std::string text; float conf; };
struct PageResult { int width, height; float skewDeg; bool inverted;
                    std::vector<LineResult> lines; std::vector<StageTime> timings;
                    std::map<std::string, double> metrics; };   // e.g. "otsu.threshold"
}
```

### 6.1 Component interfaces (`include/cudaocr/components/`, host-only)

```cpp
class Params;   // text values parsed on read: get<int|float|double|bool|string>(key, fallback);
                // bad value -> InvalidArgument naming key + value; requireKnown() rejects typos
struct StageContext { CUstream_st* stream; PageResult* page; };  // page = metadata sink

struct ImageStage     { Status run(const DeviceImage& in, DeviceImage& out, StageContext&); };
struct Deskewer       { Result<float> estimate(const DeviceImage& binary, StageContext&);
                        Status apply(float deg, const DeviceImage& in, DeviceImage& out,
                                     uint8_t border, StageContext&); };
struct TextDetector   { Result<std::vector<LineResult>> detect(const DeviceImage& binary,
                                                               StageContext&); };   // geometry only
struct LineRecognizer { const Charset& charset() const;
                        Result<std::vector<CtcProbs>> recognize(const DeviceImage& gray,
                                                 std::span<const LineResult>, StageContext&); };
struct Decoder        { Result<DecodedLine> decode(const CtcProbs&, const Charset&); };
struct Formatter      { std::string_view extension() const;
                        Result<std::string> format(const PageResult&); };
// every interface also has name(); implementations have static create(const Params&).

class CtcProbs;      // [T x C] probabilities for one line + page geometry of the timesteps:
                     // timestep t covers columns [x0 + t*stepWidth, x0 + (t+1)*stepWidth)
struct Charset { std::vector<std::string> symbols; int blank; };   // PaddleOCR layout
struct DecodedLine { std::string text; float conf;
                     std::vector<int> timesteps, classes; std::vector<float> probs; };
```

Rules: components never call each other — only `Pipeline` composes them; data crosses
component boundaries only as these types; third-party headers (ORT, poppler, toml++) appear
only in the one `.cpp` that needs them.

### 6.2 Pipeline

```cpp
struct PipelineConfig {               // defaults = the built-in pipeline; TOML mirrors it
    ComponentSpec toGray{"grayscale"};
    std::vector<ComponentSpec> preprocess{{"median", {k=3}}, {"polarity"}};  // shared by both paths
    std::vector<ComponentSpec> binarizePrep;                  // binarizer-only stages
    ComponentSpec binarizer{"adaptive"};
    std::optional<ComponentSpec> deskewer = {"hough"};
    ComponentSpec detector{"projection"};
    std::optional<ComponentSpec> recognizer;                  // CLI default: {"onnx"}
    ComponentSpec decoder{"ctc-greedy"};
};
Result<Pipeline> Pipeline::create(const PipelineConfig&, const ComponentRegistry& = withBuiltins());
Result<PageResult> Pipeline::run(const HostImage&, StageImages* keep = nullptr);
```

`create` validates every slot once (errors name the slot: `preprocess[0] median: k must be 3
or 5`). `run` reuses device buffers, times each component, deskews the gray page with the
binary page's angle, and replaces detector words with recognized words when the recognizer
reports timestep geometry.

### 6.3 Built-in components

| Name | Interface | Params (default) |
|---|---|---|
| `grayscale` | ImageStage | — (1-channel input passes through) |
| `equalize` | ImageStage | — |
| `median` | ImageStage | `k` ∈ {3, 5} (3) |
| `gaussian` | ImageStage | `ksize` odd ≥ 3 (5), `sigma` ≥ 0 (0 = from ksize) |
| `polarity` | ImageStage | — (sets `page.inverted`) |
| `adaptive` | ImageStage | `blockSize` odd ≥ 3 (31), `C` (40; decision #25) |
| `otsu` | ImageStage | — (records `otsu.threshold`) |
| `hough` | Deskewer | `maxAngle` (0, 45] (15), `step` (0, maxAngle] (0.25); \|angle\| < step/2 → copy |
| `projection` | TextDetector | `minLineHeight` ≥ 1 (3) |
| `onnx` | LineRecognizer | `model`, `dict` (see `defaultModelDir()`), `ep` cuda\|cpu (cuda), `maxBatch` (16), `padFrac` (0.1), `maxWidth` (6144, 0 = none), `tf32` (false) |
| `ctc-greedy` | Decoder | — |
| `txt`, `json`, `hocr` | Formatter | — ([docs/output-formats.md](docs/output-formats.md)) |

## 7. Error handling and logging

- `CUDAOCR_CHECK(expr)` / `CUDAOCR_CHECK_LAUNCH(name)` turn CUDA errors into `Status` with the
  expression or kernel name and file:line; ORT and poppler exceptions/failures are caught at
  the component boundary and converted the same way.
- **No exceptions cross the library boundary.**
- `Logger`: static, thread-safe facade with TRACE…ERROR levels, an injectable sink (tests),
  stderr mirroring at WARN and above; the CLI writes `cudaocr.log` only at `--log-level
  debug` or lower. Pipeline stage failures are logged with the stage name.
- The CLI reports per-page failures with the input path and keeps going; exit code = worst
  outcome (2 runtime, 1 bad input/config, 3 no text, 0).

## 8. Testing strategy

**Cycle for every change:** red (test first; a failing oracle comparison for kernels) →
green → refactor → verify (`scripts/verify.sh`). Every fixed bug gets a named regression test
first.

| Layer | What | Criterion |
|---|---|---|
| Kernel unit tests | each `stages/` kernel vs its CPU oracle on synthetic images and fixtures | integer kernels byte-exact; float paths documented tolerance (rotation: ≤ 0.5 % of pixels beyond ±2; line prep ≤ 1e-4) |
| Host unit tests | Status, Params, registry, config round trip, CER/WER, CTC decoding, word boxes, formatters (exact output) | exact |
| Component tests | adapters == the free functions they wrap; recognizer on both EPs (CPU vs CUDA ≤ 1e-3) | exact / tolerance |
| Golden | `PipelineGolden`: counts + FNV digest of every box on 18 fixture/config cases | exact; intentional changes regenerate rows with a reason (decision #21) |
| Integration | synthetic fixtures (line counts, skew, polarity), gray deskew alignment, end-to-end CER | thresholds documented in the tests |
| Accuracy ratchets | `Accuracy.*`: clean synthetic, all synthetic, sherlock, real scans, UNLV | measured CER + 1 point |
| CLI / tooling | bash scripts under `tests/scripts/` (text/JSON/hOCR output, config precedence, multi-page, PDF, install, fixture determinism, changed-files helper) | exit 0; 77 = skipped |
| Regression | one named test per fixed bug (product.md §6) | never regresses |

Model-dependent tests skip (gtest `GTEST_SKIP`, scripts exit 77) when `models/` is empty;
UNLV ones skip without `data/unlv`; PDF ones without poppler.

**Verification gate** (`scripts/verify.sh`; `--full` memchecks everything, `--all-format`
checks every source):
1. configure + build without errors;
2. `ctest` 100 % pass;
3. `compute-sanitizer --tool memcheck` over the unit tests if any `.cu`/`.cuh` changed;
4. `clang-format --dry-run --Werror` on changed sources.

"Changed" = everything the branch touches vs its merge-base with `origin/main` (committed,
staged, unstaged, untracked; `scripts/changed_files.sh`).

## 9. Coding style

- `PascalCase` types, `camelCase` functions, `snake_case_` private members, `kPascalCase`
  constants; kernels `camelCase` + `Kernel`; files `snake_case`.
- `.cuh` only for headers that need CUDA; everything else `.hpp` so host-only code and tests
  compile without nvcc. `#pragma once`.
- Kernels: bounds check first, `__restrict__` pointers, named block sizes.
- Comments explain *why* and constraints, with measurements where a number was chosen.
- Commits: imperative and scoped (`deskew: …`), no AI attribution trailers. Formatting is
  whatever `.clang-format` says.

## 10. Performance

- **Streams:** one non-blocking stream per `Pipeline`; every transfer and fill is ordered on it
  (decision #20, guarded by `StreamOrdering.*`).
- **Memory:** pitched images; `Pipeline` reuses its device buffers across pages
  (`DeviceImage::ensure`); line widths capped at 6144 px so attention memory stays bounded.
- **Kernels:** constant memory for Hough trig tables and Gaussian weights; separable box mean
  (8× faster than NPP); median and Gaussian still use global loads (M6.1, M6.2).
- **Recognition:** lines sorted by width and batched (16), the GPU batch is passed to ORT as a
  device tensor, TF32 off for exactness (decision #24).
- **Measured** (RTX 3060 Laptop, warm, 33-line page): preprocessing + deskew + detection
  ≈ 3 ms, recognition ≈ 370 ms (M6.7), first page per process +0.5 s start-up; batch mode is
  ~6× faster than one process per page. Kernel numbers: `benchmarks/RESULTS.md`.

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
| 13 | 2026-11-02 | Set `CMAKE_CUDA_ARCHITECTURES` default before `project()`; always embed native SASS for the target GPU | rely on PTX JIT; pin toolkit to 13.1 | The default was silently 75 since June; JIT masked it until toolkit 13.4 > driver 13.2 made JIT fail (33/59 tests). Native SASS is faster to load and immune to toolkit/driver skew. |
| 14 | 2026-11-02 | Product priority reframed: functional → modular → custom; not competing with SOTA | keep "custom kernels first" ordering | User direction. Recognition (M3) now precedes median/Gaussian optimization and CCL, which move to M6. |
| 15 | 2026-11-02 | Component interfaces + name registry (§6.1); supersedes #9 for composition (kernels stay free functions underneath) | keep hard-wired `runPipeline` | Swapping detector/recognizer/binarizer must be a config change; free-function kernels remain the tested implementation layer. |
| 16 | 2026-11-02 | M3 recognizer = PP-OCRv5 English rec model via ONNX Runtime ≥1.27 `gpu_cuda13`; line-level CTC, so char segmentation leaves the functional path | train our own CRNN; docTR word-level models; Tesseract | Pretrained, small, line-level, permissive license, ONNX available; ORT 1.27 ships CUDA 13 builds. Training a model is out of scope for "make it functional". |
| 17 | 2026-11-02 | Ground truth from synthetic rendering (Pillow, seeded) + small real-scan subsets (UNLV/ISRI) | real datasets only | Synthetic gives exact GT, deterministic tests and no licensing questions; real scans keep the numbers honest. |
| 18 | 2026-11-02 | One branch + one PR per backlog task; user merges; branches deleted after merge; `CLAUDE.md`/`.claude/` gitignored; no AI co-author trailers | commit straight to `main` | User direction; PRs give a reviewable unit per task and a clean history (§13). |
| 19 | 2026-11-02 | Interface details (refines #15): detectors return geometry-only `LineResult`s (no separate `LineRegion` type); deskew is its own `Deskewer` interface (estimate once, apply to binary and gray); stages write page metadata through `StageContext::page`; recognizers emit CTC *probabilities* (`CtcProbs`); `Params` stores text and parses on read | `LineRegion` type; deskew as an `ImageStage`; side-output structs per stage; typed `std::variant` params | Fewer types to keep in sync; deskew genuinely needs two outputs from one estimate; a page sink keeps stage signatures uniform; text params match their CLI/TOML origin and give one uniform error for bad values. |
| 20 | 2026-11-02 | All device transfers/fills are ordered on the caller's stream (`zero/upload/download/fromHost(…, stream)`) | make pipeline streams blocking; sync the device around transfers | Found as a real race (MOD.4). Blocking streams would hide it but serialize against any default-stream work; explicit stream arguments keep ordering visible and enable future multi-stream batching. |
| 21 | 2026-11-02 | Replace-and-pin refactors: prove new == old directly, then freeze old outputs as a golden table (counts + FNV digest of all boxes) before deleting the old code | keep the old implementation as a permanent oracle | The direct comparison found a real bug; the golden table keeps the guarantee without carrying dead code. Intentional behavior changes must regenerate the table and say why. |
| 22 | 2026-11-02 | ORT pinned to 1.31.1 `gpu_cuda13` via FetchContent (URL + SHA-256 from the GitHub release digest), imported as `onnxruntime::onnxruntime`; `CUDAOCR_ORT_ROOT` to reuse an extracted tree; `CUDAOCR_WITH_ORT=OFF` builds the core without it | system install; vendoring binaries in git; Python `onnxruntime` | Reproducible and verifiable without sudo or a 233 MB blob in history; OFF keeps the GPU core buildable anywhere. Verified on driver 595.99 (CUDA 13.2) + toolkit 13.4 + cuDNN 9.27: CUDA EP runs with CPU fallback disabled. |
| 23 | 2026-11-02 | Line crops: pad each detected box by 0.10 × its height on every side (off-page = white), bilinear to H=48 keeping aspect, coordinates in double inside the kernel | 0, 0.05, 0.15, 0.2, 0.3 padding; float coordinates | Sweep over the 6 synthetic pages (detected boxes → GPU crops → PP-OCRv5 on ORT-CPU → greedy CTC): CER 0.37 / 0.41 / **0.25** / 1.52 / 2.46 / 9.96 %; larger pads pull in neighbouring lines (54 % CER on the −4° page at 0.3). Float coordinates risk ~2e-4 error on 1200-px lines vs the 1e-4 oracle bound; double costs nothing measurable here (max diff vs oracle 3e-7). |
| 24 | 2026-11-02 | `onnx` recognizer: TF32 off by default on the CUDA EP; lines sorted by width before batching; GPU batch handed to ORT as a device tensor; per-line output truncated to ⌊outW/8⌋ | ORT defaults (TF32 on); input order batching; host round-trip | TF32 moved probabilities by up to 0.065 vs the CPU EP (1.8e-5 with it off) — correctness first, `tf32=true` stays available. Width sorting minimises white padding; device input avoids a D2H+H2D copy per batch; truncation keeps padding columns out of decoding. |
| 25 | 2026-11-02 | Default adaptive threshold C = 40 (was 10); median stays shared by binarization and recognition; new `binarizePrep` slot for binarizer-only stages (empty by default) | C ∈ {10, 20, 25, 30, 35, 50, 60}, blockSize 21/41/51, Otsu, median 5, median moved to `binarizePrep` | Sweep with `accuracy_report.sh` (CER): C=10 synth-ci 9.17 / real 23.45 / large 19.89 %; Otsu 0.32 / 15.95; C=30 0.28 / 9.04 / 0.28; **C=40 0.35 / 5.01 / 0.21**; C=50 0.32 / 3.24 / 0.21; C=60 large 0.22 but a low-contrast probe (text 170 on paper 235) collapses to 99 % (C ≤ 50: ≤ 0.7 %). C=40 keeps a 25-level margin for faint print; C=50's extra gain comes from one stained scan. Median 5: sherlock 48 %. Median moved to `binarizePrep` (recognizer reads unfiltered gray): sherlock 0.89 → 2.98 %, −4° page 0 → 4.46 % — the median helps recognition here, contradicting a single no-ground-truth observation on sample.png (M3.9). |
| 26 | 2026-11-02 | Config = TOML via toml++ 3.4.0 (header-only, MIT, FetchContent, SHA-256 pinned, only `src/config.cpp` includes it); file mirrors `PipelineConfig` (component = inline table with `name` + params, `false` disables optional slots); CLI = defaults → `--config` → explicit flags only; canonical `toToml` makes `--print-config` round-trip | JSON/YAML config; flags rebuilding the whole config; hand-written parser | TOML was planned (§3) and is comment-friendly; mirroring the component model means a new component needs no config code; explicit-only overrides make the file authoritative; round trip is testable (`parse(print(c)) == c`). |
| 27 | 2026-11-02 | Real-scan evaluation on UNLV/ISRI bus.3B + doe3.3B: seeded 40-page sample, scored **zoned** (each ground-truth zone OCRed separately, joined in zone order — the ISRI protocol) and **full page**; GT `~` (undeterminable character) = one-character wildcard (`cudaocr-eval --ref-wildcard`); data fetched by script, never committed or quoted (UNLV licence) | full page only; mag/news sets; strip `~` | Zoned isolates recognition + line detection from layout (6.84 % vs 19.64 % shows where the remaining error is); mag/news are multi-column, out of the current detector's scope; stripping `~` would count every undeterminable character as an insertion. The wildcard is our approximation of ISRI's convention. |
| 28 | 2026-11-02 | PDF input via poppler-cpp (system `libpoppler-cpp-dev`, pkg-config, optional `CUDAOCR_WITH_PDF`, linked PRIVATE into the library behind `PdfDocument`); pages rendered in-process to 8-bit gray at `--pdf-dpi` (300) and fed through the multi-page path as `<stem>-p001` … | shell out to `pdftoppm`; MuPDF (AGPL); pdfium prebuilt | In-process rendering needs no temp files or run-time tools; poppler-cpp is GPL-2+ like the system poppler (fine for this project, noted for redistribution), packaged on Ubuntu, and `format_gray8` renders straight to our input format. Measured on the fixture: PDF CER 0.48 % vs 0.24 % on the source PNGs (JPEG embedding + poppler resampling, ~7 gray levels mean difference — identical to pdftoppm). |

*(Append new rows; never delete — supersede with a new row referencing the old.)*

## 12. Git & branching workflow

1. **Start a task:** `git switch main && git pull --ff-only`, then `git switch -c task/<id>-<slug>` (e.g. `task/W.1-cuda-arch`). One BACKLOG task per branch.
2. **Work:** red → green → refactor (§8). Commit as often as useful on the branch; messages imperative and scoped (`build: set CUDA arch before project() (W.1)`).
3. **Verify:** `./scripts/verify.sh` must exit 0. Check the task off in BACKLOG.md and update LOOP_STATE.md *in the same branch*.
4. **Publish:** `git push -u origin <branch>` then `gh pr create --base main` with: summary, acceptance criteria → evidence (test names, verify output, benchmark deltas).
5. **Merge:** the user reviews and merges (squash). Nobody enables auto-merge or merges on the user's behalf unless asked.
6. **Clean up after merge:** `git switch main && git pull --ff-only && git branch -d <branch> && git push origin --delete <branch>` (skip the remote delete if GitHub already removed it) `&& git fetch --prune`.
7. **No stacking:** the next task starts from the updated `main` only after the previous PR is merged. If it isn't merged yet, stop and report `AWAITING-MERGE`.
8. **Commit/PR text carries no AI attribution trailers** (no `Co-Authored-By: Claude…`, no "Generated with…" line).
