# CudaOCR — Product Definition

> Companion document: [architecture.md](architecture.md) (how we build it). This file defines **what** we build and **why**.
> Status date: 2026-11-02 (previous audit 2026-06-11).

## 1. Vision

CudaOCR is a **GPU-first optical character recognition engine written in CUDA/C++**: a CLI and linkable library that takes a printed-document image and emits the recognized text with positions and confidences.

Priorities, in order:

1. **Functional.** Image in, correct text out, on ordinary printed documents. A working end-to-end path beats a perfect half-pipeline.
2. **Modular.** Every stage — preprocessing filters, binarizer, deskew, text detector, line recognizer, decoder, output formatter — sits behind a small interface and is selected by name from config. Any one of them can later be optimized, replaced (e.g. projection profiles → a learned detector), or re-implemented (ONNX Runtime → hand-written CUDA) without touching its neighbours.
3. **Custom CUDA where it teaches something.** Hand-written kernels are the learning/portfolio goal; libraries (NPP, OpenCV, ONNX Runtime) are allowed as baselines, oracles, or temporary implementations behind an interface.

**Non-goal: state of the art.** We are not competing with VLM-based OCR (Qwen-VL-class, GOT-OCR, olmOCR) or full document-understanding stacks. Target quality is "Tesseract-class on clean printed pages"; target strength is a fast, dependency-light, well-tested GPU pipeline whose parts can be swapped.

## 2. What CudaOCR is NOT

- Not a Tesseract/PaddleOCR wrapper (though a PaddleOCR-trained *model* may run inside our pipeline — see §5 Stage 5).
- Not a scene-text reader (street signs, photos of screens). Scope is **printed document images**: scans, screenshots, PDFs rendered to images.
- Not a handwriting recognizer.
- Not a service/server. CLI + static library. (Server wrapper is a stretch goal.)
- Not a layout-understanding system (tables, math, forms → key/value) — out of scope for v1.

## 3. Users & use cases

| User | Use case |
|---|---|
| Primary: the author | Learn CUDA deeply; portfolio artifact with measurable benchmarks |
| Developer integrating the lib | Batch-digitize printed documents on an NVIDIA GPU without heavyweight dependencies |
| Recruiter/reviewer | Read clean, tested, benchmarked CUDA code |

## 4. Current status (audited 2026-11-02)

### 4.1 Done

| Area | What exists | Evidence |
|---|---|---|
| Build & harness (M0) | CMake + CUDA 13 + C++20, GoogleTest/CTest, `scripts/verify.sh` gate (build, tests, memcheck, clang-format), `Status`/`Result<T>`/`CUDA_CHECK`, thread-safe logger, RAII `DeviceBuffer`/`DeviceImage`/`Stream`, vendored stb image I/O, CLI. Legacy Makefile removed. | 59 tests |
| Preprocessing (M1.1–M1.8) | Custom kernels: grayscale (BT.601), separable box mean, adaptive mean threshold, Otsu (GPU histogram), median 3×3/5×5, separable Gaussian, histogram equalization, polarity detection + invert. Each byte-exact (or documented tolerance) vs a hand-written CPU oracle. Benchmark harness vs NPP `_Ctx` → `benchmarks/RESULTS.md`. | grayscale 1.2× NPP; box mean **7.8× faster** than NPP |
| Layout (M2.1–M2.4) | Hough skew detection (baseline-pixel voting, constant-memory trig, ±15° window, accumulator-bounds regression test), GPU bilinear rotation, row-projection line detection, column-projection word and char boxes. | 35 lines / 348 words on `resources/sample.png`, ~3.6 ms GPU total |
| CLI | `cudaocr <img> [-o out.json] [--dump-stages dir] [--otsu] [--block-size] [--C] [--median] [--equalize] [--no-deskew] [--benchmark] [--log-level]`; JSON of boxes; overlay PNG; `install` target. | |

### 4.2 Not done

- ~~Recognition~~ — done in M3.7–M3.9 (`onnx` recognizer + `ctc-greedy` decoder + `txt` formatter; CLI prints text).
- Text output formats (txt/hOCR), config file, accuracy measurement (no ground-truth harness, no CER/WER numbers).
- Performance follow-ups: median is 27× slower than NPP (M1.9 → M6), Gaussian 2.75× (M1.10 → M6).
- GPU connected-component labeling (M2.5 → M6; not needed for line-level recognition, see §5 Stage 4).

### 4.3 Known bugs & issues

1. ~~**CUDA arch default never applies.**~~ *Fixed in W.1 (regression test `CudaArch.CudaArchMatchesDevice`).* `CMakeLists.txt` sets `CMAKE_CUDA_ARCHITECTURES 86` *after* `project()`, by which point CMake has already defined it from the compiler default (75). Binaries contain sm_75 SASS + compute_75 PTX, so on the sm_86 GPU every kernel is PTX-JIT-compiled. Since the toolkit moved to 13.4 while the driver supports 13.2, JIT fails ("PTX was compiled with an unsupported toolchain") and 33/59 tests fail. Confirmed fix: configuring with `-DCMAKE_CUDA_ARCHITECTURES=86` → 59/59 pass. Backlog W.1.
2. ~~**`verify.sh` diffs against `HEAD`**~~ *Fixed in W.2 (`scripts/changed_files.sh`, test `Scripts.ChangedFiles`).*, so on a task branch with committed work the memcheck/format steps see no changes. Must diff against the merge-base with `main`. Backlog W.2.
3. **Character segmentation under-splits**: 903 chars / 348 words ≈ 2.6 chars/word on the sample (English averages ~4.7) — touching glyphs merge under the 1-blank-column rule. Low impact: line-level CTC recognition does not need char boxes.
4. ~~**Deskew only rotates the binary image.**~~ *Fixed in M3.3:* the same angle is applied to the polarity-normalized gray page (`StageImages::deskewedGray`, `05_deskewed_gray.png`).
5. **Projection-profile layout assumes a single text column.** Multi-column pages, tables, and figures interleave lines. Accepted for v1; the `TextDetector` interface lets a learned detector (DBNet) replace it in M6.
6. Toolchain drift since June: system now has CUDA toolkit 13.4 (13.1 also installed), driver 595.99 (CUDA 13.2), default host compiler g++ 15.2 (g++-13 removed). The build works with these once #1 is fixed.
7. ~~**Default-stream transfers raced pipeline kernels.**~~ *Fixed in MOD.4.* Pipeline streams are non-blocking, but uploads/downloads/zero-fills used the default stream, so e.g. the polarity histogram could read a half-uploaded image (~1 in 3 runs) and pipeline tests were order-dependent. Regression tests: `StreamOrdering.*`.
8. ~~**Default thresholding is weak on noisy scans**~~ *Mostly fixed in M3.11* (adaptive C 10 → 40, decision #25): noisy synthetic pages 99 % → ≤ 0.8 % CER, `noisy_text2` 98 % → 0.7 %, real-scan aggregate 23.5 % → 5.0 %. Remaining: coffee-stained `noisy_text` 30 %, speckled `vigorous-before` → M6.8 (`noisy_text2.png`: 2 lines with defaults vs 9 with `--otsu`). Backlog M3.11.
9. ~~**Descender slivers split into separate lines.**~~ *Fixed in M3.1.* Below the baseline only a few descender strokes carry ink, so the row count dipped under the ink threshold and descender bottoms became 5–6 px "lines" (10 lines on an 8-line page). `detectLines` now merges slivers (< 1/3 typical height, within 1/3 line) into the adjacent line. Regression tests: `DetectLines.Merges*`, `DetectLines.DoesNotSwallowShortLineBetweenNoiseBlobs`.
10. ~~**Exit-time heap corruption with the CUDA EP.**~~ *Fixed in M3.10:* a static `Ort::Env` destroyed during static destruction (after the CUDA EP) intermittently aborted the process (~15 % of runs of the CUDA smoke test). ORT Env/Session singletons are now process-lifetime (never destroyed). Regression: `Ort.ExitIsClean` (25 fresh processes).
11. ~~**Thin rules exhausted GPU memory.**~~ *Fixed in M4.6:* on UNLV pages, 3–6 px horizontal rules were detected as lines, resized to ~30,000 px wide, and the recognizer's O(T²) attention asked ONNX Runtime for up to 3.8 GB, failing 7/40 pages. `LinePrepParams::maxWidth` (6144, also the `onnx` `maxWidth` param) caps it. Regressions: `PlanLineCrops.MaxWidthCapsDegenerateAspectRatios`, `OnnxRecognizer.DegenerateThinLineIsCappedNotFatal`.

## 5. Full product specification (features by pipeline stage)

Each stage is a **component** implementing a small interface (architecture.md §6), with: a GPU (or library) implementation, a CPU oracle used by tests where meaningful, parameters from config, and per-stage timing.

### Stage 0 — CLI & I/O
- `cudaocr <input...> [-o out] [--format txt|json|hocr] [--config file.toml] [--dump-stages dir] [--log-level ...] [--benchmark]`
- Load PNG/JPEG/BMP/TGA via vendored stb_image; decode errors reported with path and reason.
- Directory or multiple inputs → one output per page (M4).
- Exit codes: 0 success, 1 bad input, 2 GPU/runtime error, 3 no text found.

### Stage 1 — Preprocessing (`ImageStage` components, all custom kernels) — **done**
Grayscale, equalize, median/Gaussian denoise, polarity normalization, adaptive/Otsu binarization. Configured as an ordered list.

### Stage 2 — Deskew (`ImageStage`) — **done for binary; gray path in M3.3**
Hough skew estimate on the binary image; the same angle rotates both binary (for detection) and gray (for recognition).

### Stage 3 — Text detection (`TextDetector`)
- **v1 (done, to be wrapped):** `projection` — row projection → line bands; column projection → word boxes.
- **M6 swap-in:** `dbnet-onnx` — PP-OCR detection model (DBNet) via ONNX Runtime; handles multi-column/complex layouts. Same output type.
- Output: `std::vector<LineRegion>` (box, optional polygon, reading-order index).

### Stage 4 — Word & character segmentation
- Word boxes come from the detector (projection gaps) and/or from CTC timestep positions after recognition.
- Character boxes are **optional** (debug overlay only). GPU connected-component labeling is an M6 learning task, not on the functional path.

### Stage 5 — Line recognition (`LineRecognizer` + `Decoder`)
- **M3 (functional):** `onnx` recognizer running the **PP-OCRv5 English recognition model** (`en_PP-OCRv5_mobile_rec`, 7.8 MB, 1.94M params, 438 CTC classes) through **ONNX Runtime 1.31.1** (`gpu_cuda13` build), execution provider selectable (`cuda` | `cpu`). Input contract measured in M3.5 (`models/README.md`): tight line crop → height 48, aspect kept, 3 channels, `(v/255−0.5)/0.5`; output `[N, ⌊W/8⌋, 438]` probabilities.
- **M5 (custom):** `cuda` recognizer — hand-written kernels for the same model's ops (conv/depthwise conv with folded BN, hardswish, attention/matmul, layernorm, linear, softmax), loading the same weights. ORT-CPU logits are the oracle.
- **Decoder:** CTC greedy (M3); beam search + optional wordlist (M6).

### Stage 6 — Post-processing
- Assemble lines → page text in reading order; per-line/word confidence from softmax.
- Optional dictionary correction (M6).

### Stage 7 — Output (`Formatter`)
- `txt` (M3), `json` with boxes + text + confidences (M4), `hocr` (M4).

### Cross-cutting
- **Component registry + config:** every component is created by name with a parameter map; config from TOML file + CLI overrides (M4). No hardcoded constants.
- **Per-stage GPU timing** via CUDA events, printed with `--benchmark`.
- **Thread-safe logging** TRACE…ERROR. **Deterministic mode** for tests.
- **Accuracy harness:** CER/WER evaluation tool over a directory of image + ground-truth pairs (M3).

## 6. Non-functional requirements

| Requirement | Target |
|---|---|
| Correctness | Every custom kernel matches its CPU oracle: byte-exact for integer/fixed-point ops, documented tolerance for float paths (architecture.md §8) |
| Accuracy (M3) | **CER ≤ 5%** on the clean synthetic 300-DPI fixture set; non-empty, human-readable text on `resources/` scans. Numbers recorded in `benchmarks/ACCURACY.md`. |
| Accuracy (M4+) | CER ≤ 10% on a small real-scan set (e.g. UNLV/ISRI-style pages, see §9) — measure first, then set a ratchet |
| Performance | ≤ 150 ms end-to-end for a 2500×3500 page incl. recognition (ORT CUDA EP, excluding model load/warm-up) on RTX 3060 Laptop |
| Memory | Peak GPU memory ≤ 2 GB for a 4K page (6 GB card must never OOM) |
| Reliability | memcheck clean; all CUDA/ORT errors surfaced as `Status` with stage context |
| Portability | Any sm_75+ GPU with CUDA 13.x; arch configurable; core library builds **without** ONNX Runtime (`CUDAOCR_WITH_ORT=OFF`) |
| Modularity | Adding a new implementation of a component = new class + one registry line + tests; zero edits to `Pipeline` |

## 7. Roadmap & milestones

Ordering principle: **functional end-to-end first, then usable, then custom/optimized.** Optimizations and swap-ins that don't change functionality are deferred to M6.

| Milestone | Goal | Status |
|---|---|---|
| **M0** Re-foundation | CMake/CUDA 13/C++20 build, test harness, core RAII, logger, CLI | ✅ done |
| **M1** Preprocessing | Custom preprocessing kernels + oracles + NPP benchmarks | ✅ done (M1.9/M1.10 perf → M6) |
| **M2** Layout | Hough deskew, GPU rotate, projection line/word segmentation | ✅ done (M2.5 CCL → M6) |
| **W** Workflow fixes | Fix CUDA arch default; branch-aware `verify.sh` | ✅ done |
| **MOD** Modular core | Result types, component interfaces, registry, Pipeline built from config; existing stages wrapped with zero behavior change | ✅ done |
| **M3** Functional OCR | Ground-truth fixtures + CER tool, ORT dependency, line crop/normalize kernel, ONNX recognizer, CTC decoder, txt output, accuracy baseline | ✅ done (CER: synthetic 0.2–0.35 %, real scans 5.0 %, sherlock 0.94 %) |
| **M4** Usable product | JSON/hOCR output, TOML config, multi-page input, GitHub Actions compile/format CI, README results tables | M4.1–M4.4 ✅; M4.5 CI, M4.6 real-scan set ⏳ |
| **M5** Custom recognizer | Hand-written CUDA inference for the recognition model; ORT as oracle | ⏳ |
| **M6** Optimize & upgrade | Median/Gaussian optimization, DBNet detector swap-in, GPU CCL, beam search, dictionary correction, multi-stream batching | backlog |

Detailed, machine-checkable tasks live in [BACKLOG.md](BACKLOG.md).

## 8. Success metrics

1. `ctest` green and `verify.sh` passing on every merged PR.
2. `cudaocr resources/sample.png` prints readable text (M3).
3. Accuracy table (CER/WER per fixture set) in `benchmarks/ACCURACY.md` and README (M3/M4).
4. Benchmark table: custom kernel vs library baseline (exists; extended per stage).
5. Swap test: switching `detector`/`recognizer`/`binarizer` in config changes behavior without code edits (MOD/M4).

## 9. Evaluation data & infrastructure

### Data (in order of use)
| Source | What | Use |
|---|---|---|
| **Synthetic (generated in-repo)** | `scripts/gen_fixtures.py` renders known text with system fonts (DejaVu, Liberation) via Pillow; optional skew, noise, blur, inversion, DPI | Primary TDD/CER fixtures: exact ground truth, deterministic seeds, no licensing issues. Small set committed; larger set generated on demand. |
| `resources/` scans | 5 existing real images, no ground truth | Smoke tests ("produces non-empty readable text"); hand-transcribe 1–2 pages for a real-scan CER number |
| **UNLV / ISRI dataset** | ~2,900 scanned printed pages (magazines, letters, reports) with ground-truth text; the set Tesseract was historically evaluated on | **In use (M4.6):** `scripts/fetch_unlv.sh` → 40-page seeded sample of bus.3B + doe3.3B (300 dpi bitonal); © UNLV, not redistributable — never committed |
| FUNSD | 199 scanned forms with word boxes + text | Detection/word-box sanity on noisy scans (M4/M6) |
| PubLayNet / DocLayNet | Layout annotations | Only if/when multi-column layout (M6) is attempted |

### Models
- **PP-OCRv5 English recognition** ONNX + `dict.txt` (Apache-2.0 PaddleOCR weights; community ONNX export on Hugging Face) — fetched by `scripts/fetch_models.sh` into `models/` (gitignored), SHA-256 pinned.
- PP-OCRv5 detection (DBNet) ONNX — same source, for the M6 detector swap-in.

### Compute / servers
- **Local RTX 3060 Laptop (sm_86, 6 GB) is sufficient** for every milestone; no server is required.
- **CI:** GitHub-hosted runners have no GPU. Plan (M4): a GitHub Actions job that installs the CUDA toolkit and does a compile-only build + clang-format check on every PR; GPU tests stay in `verify.sh` locally. Optional later: a self-hosted runner on this machine to run GPU tests on PRs.
- **Optional:** rent a GPU by the hour (RunPod, Vast.ai, Lambda) only to check portability on another arch (sm_75/sm_89); a Hugging Face Space or Modal for a public demo is a stretch goal.
