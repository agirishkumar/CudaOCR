# CudaOCR — Architecture & Engineering Standards

> **How** CudaOCR is built: toolchain, design, interfaces, testing and process. What and why:
> [product.md](product.md). Every significant choice is recorded with its reasoning in
> [decisions.md](decisions.md) ("decision #N" below); new choices are appended there.

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
├── README.md, LICENSE
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
├── benchmarks/               bench.cu (custom kernels vs NPP), ACCURACY.md (generated CER/WER report)
├── docs/                     product.md, architecture.md, decisions.md, backlog.md,
│                             config.md, output-formats.md
├── models/                   fetched model (gitignored) + README.md model card
├── scripts/                  verify.sh, changed_files.sh, fetch_models.sh, fetch_unlv.sh,
│                             prepare_unlv.py, gen_fixtures.py, accuracy_report.sh, onnx_info.py
├── resources/                redistributable sample pages (+ README.md with sources/licences)
└── third_party/stb/          vendored image I/O
```

`data/` (UNLV download) and `build*/` are gitignored. `CLAUDE.md`, `.claude/` and
`LOOP_STATE.md` are local-only assistant configuration and state.

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
| `txt`, `json`, `hocr` | Formatter | — ([output-formats.md](output-formats.md)) |

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
| Regression | one named test per fixed bug ([product.md](product.md) §6) | never regresses |

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
  ~6× faster than one process per page. Kernel numbers vs NPP: `./build/bench` and the README
  results table.

## 11. Git & branching workflow

1. **Start a task:** `git switch main && git pull --ff-only`, then `git switch -c task/<id>-<slug>` (e.g. `task/W.1-cuda-arch`). One BACKLOG task per branch.
2. **Work:** red → green → refactor (§8). Commit as often as useful on the branch; messages imperative and scoped (`build: set CUDA arch before project() (W.1)`).
3. **Verify:** `./scripts/verify.sh` must exit 0. Check the task off in [backlog.md](backlog.md) in the same branch.
4. **Publish:** `git push -u origin <branch>` then `gh pr create --base main` with: summary, acceptance criteria → evidence (test names, verify output, benchmark deltas).
5. **Merge:** the user reviews and merges (squash). Nobody enables auto-merge or merges on the user's behalf unless asked.
6. **Clean up after merge:** `git switch main && git pull --ff-only && git branch -d <branch> && git push origin --delete <branch>` (skip the remote delete if GitHub already removed it) `&& git fetch --prune`.
7. **No stacking:** the next task starts from the updated `main` only after the previous PR is merged. If it isn't merged yet, stop and report `AWAITING-MERGE`.
8. **Commit/PR text carries no AI attribution trailers** (no `Co-Authored-By: Claude…`, no "Generated with…" line).
