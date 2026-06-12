# CudaOCR — Product Definition

> Companion document: [architecture.md](architecture.md) (how we build it). This file defines **what** we build and **why**.
> Status date: 2026-06-11.

## 1. Vision

CudaOCR is a **GPU-first optical character recognition engine written from scratch in CUDA/C++**, where every stage of the classical OCR pipeline — preprocessing, layout analysis, segmentation, and recognition — runs on the GPU through hand-written CUDA kernels.

It is simultaneously:

1. **A working product**: a CLI tool that takes a document image and emits the recognized text with positional metadata and confidence scores.
2. **A learning & portfolio vehicle**: demonstrating mastery of CUDA kernel design, GPU memory management, streams, and performance engineering. Off-the-shelf GPU libraries (OpenCV CUDA, NPP) are used only as *benchmarks to beat*, not as the implementation.

**Guiding principle:** if a stage can be a custom kernel, it is a custom kernel. Library calls are temporary scaffolding or comparison baselines.

## 2. What CudaOCR is NOT

- Not a Tesseract/PaddleOCR wrapper.
- Not a general-purpose scene-text reader (street signs, memes). Scope is **printed document images**: scans, screenshots, PDFs rendered to images.
- Not (initially) a handwriting recognizer.
- Not a service/server. It is a CLI + linkable static library. (A server wrapper is a stretch goal.)

## 3. Users & use cases

| User | Use case |
|---|---|
| Primary: the author | Learn CUDA deeply; portfolio artifact with measurable benchmarks |
| Developer integrating the lib | Batch-digitize printed documents on an NVIDIA GPU without heavyweight dependencies |
| Recruiter/reviewer | Read clean, tested, benchmarked CUDA code |

## 4. Current status (audited 2026-06-11)

### 4.1 Done (works conceptually, but no longer compiles — see 4.3)

| Stage | Implementation | Notes |
|---|---|---|
| Image acquisition | `loadImageFromPath`, `uploadImageToGPU` (OpenCV) | Works |
| Grayscale | `convertToGrayscaleNPP` (BT.601 coefficients) + OpenCV CUDA variant | Works |
| Denoising | NPP median filter + NPP Gaussian (3/5/7) | Median called with kernelSize=1 (no-op) in main |
| Binarization | Custom adaptive threshold via OpenCV CUDA box filter + compare | Hardcoded blockSize=3, C=2.0 |
| Skew detection | Custom `houghTransform` CUDA kernel + accumulator argmax | **Crashes** (see 4.2) |
| Skew correction | CPU `cv::warpAffine` | Defeats GPU purpose |
| Logging | `Logger` static class, timestamped file log | Not thread-safe |

### 4.2 Known bugs & issues (from code audit)

1. **Illegal memory access in `houghTransform`** (`src/page_analysis.cu`): the bounds check is `r >= 0 && r < rho * 2` but the accumulator is allocated with `rho * theta` ints and indexed `r * theta + t`. Any `r >= rho` writes out of bounds. This is the crash recorded in `ocr_pipeline.log`.
2. **Output quality**: README reports excess noise and a blackened bottom region in preprocessed output (likely related to #1 and to the kernelSize=1 median call).
3. `utils.cu` uses `std::stringstream` without `#include <sstream>`.
4. Duplicated include block in `page_analysis.cu`; unused Thrust includes.
5. `using namespace std;` in `main.cu`.
6. No tests of any kind; no CI.
7. All parameters hardcoded; no CLI argument parsing (input path baked into `config.h`).
8. `downloadImageFromGPU` always returns true (no error handling).
9. Texture-memory experiments failed and were removed (README) — revisit later as an optimization task.

### 4.3 Build is broken on current system (forced migration)

- CUDA **13.1** removed every non-`_Ctx` NPP API the code calls (`nppiColorToGray_8u_C3C1R`, `nppiFilterGauss_8u_C1R`, `nppiFilterMedian_8u_C1R`, …).
- CUDA 13 dropped `sm_61`; the GPU is an RTX 3060 Laptop (**sm_86**, 6 GB).
- Makefile pins `g++-9`, which is not installed (g++-11/12/13 are).
- OpenCV is not installed (no apt package, no source build, no CUDA modules).

**Conclusion: the pipeline must be re-founded on the current toolchain. Decisions in architecture.md §3.**

## 5. Full product specification (features by pipeline stage)

Each stage is a module with: a GPU implementation, a CPU **reference implementation (oracle)** used by tests, configurable parameters, and per-stage timing. Stage contracts (types, error model) are in architecture.md §6.

### Stage 0 — CLI & I/O
- `cudaocr <input> [-o out.txt] [--format txt|json|hocr] [--dump-stages <dir>] [--config <toml>] [--log-level ...] [--benchmark]`
- Load PNG/JPEG/TIFF/BMP via OpenCV `imread` (CPU); decode errors reported with file path and reason.
- `--dump-stages` writes each intermediate image for visual debugging (replaces today's commented-out blocks in `main.cu`).
- Exit codes: 0 success, 1 bad input, 2 GPU/runtime error, 3 no text found.

### Stage 1 — Preprocessing (all custom kernels)
| Feature | Detail |
|---|---|
| Grayscale | BT.601 weighted kernel (replaces NPP). Param: coefficient set. |
| Denoise | Median 3×3/5×5 kernel; Gaussian (separable, shared-memory) kernel. Param: method, kernel size, sigma. |
| Contrast | Global histogram equalization; CLAHE (tiled) as v1.1. |
| Binarization | Adaptive mean threshold (integral-image based, so blockSize can be large); Otsu (GPU histogram + CPU argmax over 256 bins). Params: method, blockSize, C. |
| Inversion detection | Decide dark-on-light vs light-on-dark from histogram; normalize to black text on white. |

### Stage 2 — Page layout analysis
| Feature | Detail |
|---|---|
| Skew detection | Fixed Hough kernel (correct accumulator bounds, cached sin/cos in constant memory, restricted θ range ±15° around horizontal for speed/robustness). Output: angle in degrees. |
| Skew correction | **GPU** affine rotation kernel with bilinear sampling (replaces CPU `warpAffine`). |
| Region/zone detection (v1.1) | Connected components / projection profiles to find text blocks vs whitespace; multi-column ordering. |

### Stage 3 — Text line detection
- Horizontal projection-profile kernel (row-sum reduction) → valley detection → line boundaries.
- Output: list of `LineBox {y0, y1, baseline}`.
- Robust to ±0.5° residual skew.

### Stage 4 — Word & character segmentation
- Per-line vertical projection profile → word gaps (gap threshold estimated from gap histogram).
- Character segmentation: connected-component labeling (GPU union-find) within words; merge/split heuristics for touching characters.
- Output: `WordBox`, `CharBox` hierarchies.

### Stage 5 — Recognition (phased — see roadmap)
- **Phase A (MVP)**: pretrained CRNN/CTC ONNX model executed via ONNX Runtime CUDA EP on cropped line images. Gets end-to-end text out quickly; establishes accuracy baseline.
- **Phase B (the point of the project)**: replace ONNX Runtime with **hand-written inference kernels** (im2col/implicit GEMM conv, BiLSTM or conv-only recognizer, CTC greedy + beam decode) loading the same exported weights. Must match Phase A output within tolerance — Phase A becomes the oracle.

### Stage 6 — Post-processing
- CTC decode → UTF-8 text assembly with line/word structure preserved.
- Confidence per character/word/line (from softmax probabilities).
- Optional dictionary correction (edit-distance ≤ 1 against a wordlist) — v1.2.

### Stage 7 — Output formatting
- `txt` (plain text), `json` (boxes + confidences), `hOCR` (HTML standard for OCR) — v1.1 for hOCR.

### Cross-cutting features
- **Config file** (TOML) + CLI overrides for every stage parameter; no hardcoded constants.
- **Per-stage GPU timing** (CUDA events) printed with `--benchmark`; comparison table vs OpenCV-CUDA/NPP equivalents where they exist.
- **Thread-safe logging** with levels TRACE/DEBUG/INFO/WARN/ERROR.
- **Deterministic mode** for tests (fixed seeds, synchronous execution).

## 6. Non-functional requirements

| Requirement | Target |
|---|---|
| Correctness | Every GPU kernel matches its CPU oracle: exact for integer ops, ≤1 ULP-equivalent tolerance for float paths (see architecture.md §8) |
| Accuracy (end-to-end, MVP) | ≥90% character accuracy on clean 300-DPI printed scans (test set in `resources/` + generated set) |
| Performance | ≤150 ms end-to-end for a 2500×3500 px page on RTX 3060 (excluding model warm-up); each custom kernel within 2× of NPP/OpenCV-CUDA equivalent, with a benchmark proving it |
| Memory | Peak GPU memory ≤2 GB for a 4K-resolution page (6 GB card must never OOM) |
| Reliability | No leaks (`compute-sanitizer --tool memcheck` clean); all CUDA errors surfaced with stage context |
| Portability | Builds on any sm_75+ GPU with CUDA 12.4+ (arch flags configurable); primary target sm_86 |

## 7. Roadmap & milestones

**M0 — Re-foundation (build + test harness)**
CMake build on CUDA 13.1/sm_86/g++-13, GoogleTest + CTest wired, CI-style `make check` target, logger made thread-safe, existing code migrated & compiling, Hough bug fixed under a regression test.

**M1 — Preprocessing, custom & tested**
All Stage-1 features as custom kernels with CPU oracles and benchmarks; NPP/OpenCV-CUDA usage removed from the pipeline (kept only in benchmark targets, if installed).

**M2 — Layout & segmentation**
GPU skew correct + line/word/char segmentation; visual debug overlays via `--dump-stages`.

**M3 — MVP recognition (Phase A)**
ONNX Runtime CUDA EP + pretrained CRNN; end-to-end text output; accuracy harness with ground-truth fixtures.

**M4 — Custom recognition (Phase B)**
Hand-written conv/GEMM/CTC kernels replacing ORT; Phase A output as oracle.

**M5 — Productization**
Config file, JSON/hOCR output, benchmark suite report, README with results table.

**Stretch**: batch/multi-page throughput mode with streams, texture-memory filter variants (retry of the removed experiment), CLAHE, dictionary correction, Python bindings.

## 8. Success metrics

1. `ctest` green at every commit (TDD discipline — see architecture.md §8).
2. Accuracy table in README: char/word accuracy per test image.
3. Benchmark table: custom kernel vs library baseline, ms and speedup.
4. End-to-end demo: `cudaocr resources/sample.png` prints correct text.
