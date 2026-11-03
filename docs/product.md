# CudaOCR — Product Definition

> **What** we build and **why**. How it is built: [architecture.md](architecture.md) (choices:
> [decisions.md](decisions.md)). Open work: [backlog.md](backlog.md). Status as of 2026-11-02 (milestones M0–M4 complete).

## 1. Vision

CudaOCR is a **GPU-first OCR engine for printed documents**, written in CUDA/C++: a CLI and
a linkable library that take document images or scanned PDFs and produce text with positions
and confidences.

Priorities, in order:

1. **Functional.** Image in, correct text out, on ordinary printed documents.
2. **Modular.** Every stage — preprocessing filters, binarizer, deskew, text detector, line
   recognizer, decoder, output formatter — sits behind a small interface and is chosen by
   name from config, so any one can be optimized, replaced (projection profiles → a learned
   detector) or re-implemented (ONNX Runtime → hand-written CUDA) without touching the rest.
3. **Custom CUDA where it teaches something.** Hand-written kernels are the learning and
   portfolio goal; libraries (NPP, ONNX Runtime) serve as baselines, oracles, or
   implementations behind an interface.

**Non-goal: state of the art.** CudaOCR does not compete with VLM-based OCR or document-
understanding stacks. Target: Tesseract-class accuracy on clean printed pages from a fast,
dependency-light, well-tested GPU pipeline.

## 2. Scope

| In scope | Out of scope |
|---|---|
| Printed documents: scans, screenshots, scanned PDFs | Handwriting, scene text (signs, photos) |
| English / Latin script (the model's 436-character dictionary) | Other scripts (possible by swapping the model + dictionary) |
| Single-column reading order | Layout understanding: tables, forms, multi-column ordering (M6) |
| CLI + static library | Server/service, GUI |

## 3. Users

| User | Use case |
|---|---|
| The author | Learn CUDA deeply; a portfolio project with measurable results |
| A developer | Batch-digitize printed documents on an NVIDIA GPU |
| A reviewer | Read clean, tested, benchmarked CUDA code |

## 4. Current capabilities

| Area | What it does | Where |
|---|---|---|
| Input | PNG/JPEG/BMP/TGA/GIF/PNM (stb), **PDF** pages rendered in-process at `--pdf-dpi` (poppler-cpp, optional); files and directories; one pipeline and one model load per run | `apps/cudaocr_cli`, `core/pdf` |
| Preprocessing (custom CUDA) | BT.601 grayscale, median 3×3/5×5, separable Gaussian, histogram equalization, polarity detection + inversion | `stages/` |
| Binarization (custom CUDA) | adaptive mean threshold (separable box mean; default 31 px, C = 40), Otsu | `stages/binarize` |
| Deskew (custom CUDA) | Hough skew estimate on baseline pixels (±15°, 0.25° steps), bilinear rotation of both the binary and the gray page | `stages/deskew` |
| Line detection (custom CUDA) | row/column projection profiles → line and word boxes; descender/i-dot slivers merged | `stages/segment` |
| Line preparation (custom CUDA) | crop each line (+10 % padding), resize to 48 px, normalize to the model's input, batch (width-capped) | `stages/line_prep` |
| Recognition | PP-OCRv5 English (`en_PP-OCRv5_mobile_rec`) via ONNX Runtime 1.31.1, CUDA or CPU execution provider; lines batched by width | `components/onnx_line_recognizer` |
| Decoding | CTC greedy with confidences; word boxes from CTC timesteps | `components/ctc_greedy_decoder`, `components/words` |
| Output | txt, JSON (schema `cudaocr.page/1`), hOCR 1.2 | `components/*_formatter`, [output-formats.md](output-formats.md) |
| Configuration | TOML config mirroring the component model; `--print-config` round-trips; flags override | `config`, [config.md](config.md) |
| Evaluation | `cudaocr-eval` (CER/WER, UTF-8, wildcard), `accuracy_report.sh`, synthetic fixture generator, UNLV/ISRI fetcher, accuracy ratchets in ctest | `eval/`, `scripts/` |
| Diagnostics | `-v` summaries, `--benchmark` per-stage GPU timings, `--dump-stages` images + overlay | CLI |
| Install | `cmake --install` produces a self-contained prefix (binaries, ONNX Runtime libraries, model) | `CMakeLists.txt` |

## 5. Quality and performance (measured)

| Requirement | Target | Status |
|---|---|---|
| Kernel correctness | every custom kernel matches its CPU oracle (byte-exact integer ops; documented float tolerance) | ✅ all kernels; memcheck clean |
| Accuracy, synthetic clean pages | CER ≤ 5 % | ✅ 0.21 % (50 pages) |
| Accuracy, real scans | CER ≤ 10 % | ✅ 4.75 % on `resources/` (3 scans), 6.84 % on UNLV zoned (40 pages); ❌ 19.64 % UNLV full page (layout) |
| End-to-end latency | ≤ 150 ms per page | ❌ ≈ 370 ms warm on a dense 33-line page (recognizer ≈ 99 %; GPU preprocessing + layout ≈ 3 ms) |
| Kernel speed | each custom kernel within 2× of NPP | ✅ grayscale 1.2×, box mean 8× faster; ❌ Gaussian 2.6×, median 5×5 27× (M6) |
| GPU memory | never OOM on a 6 GB card | ✅ line width capped (degenerate lines once requested 3.8 GB) |
| Reliability | all CUDA/ORT/IO errors surfaced as `Status` with context; bad pages don't stop a batch | ✅ |
| Portability | sm_75+ with CUDA 13.x; builds without ONNX Runtime or poppler | ✅ |
| Modularity | a new implementation = one class + one registry line; no `Pipeline` edits | ✅ |

Numbers: [benchmarks/ACCURACY.md](../benchmarks/ACCURACY.md) (generated) and the
[README results](../README.md#results).

## 6. Known limitations

1. **Layout assumes one column.** Projection profiles read top to bottom; multi-column pages,
   tables, headers/footers and figures interleave (UNLV full page 19.6 % vs zoned 6.8 %).
   → M6.3 (DBNet detector), M6.9.
2. **Stains and speckles.** Coffee-stained `noisy_text.png` ≈ 30 % CER; isolated specks can be
   read as characters. → M6.8.
3. **Recognizer throughput** ≈ 0.3–0.4 s per dense page. → M6.7.
4. **Character boxes** (detection-only JSON `glyphs`) come from column gaps and merge touching
   glyphs; recognized words don't need them. → M6.4 (connected components).
5. **Median 5×5 and Gaussian** kernels are slower than NPP. → M6.1, M6.2.

Bugs fixed so far each have a named regression test:

| Bug | Regression test |
|---|---|
| Hough accumulator out-of-bounds write (legacy code) | `DetectSkew.HoughAccumulatorBoundsRegression` |
| CUDA arch defaulted to sm_75 → PTX JIT refused on newer toolkits | `CudaArch.CudaArchMatchesDevice` |
| Default-stream transfers raced pipeline kernels | `StreamOrdering.*` |
| Descender slivers split into separate lines | `DetectLines.Merges*`, `DetectLines.DoesNotSwallowShortLineBetweenNoiseBlobs` |
| Static ORT environment destroyed after the CUDA EP → crash at exit | `Ort.ExitIsClean` |
| Thin rules detected as lines exhausted GPU memory | `PlanLineCrops.MaxWidthCapsDegenerateAspectRatios`, `OnnxRecognizer.DegenerateThinLineIsCappedNotFatal` |
| Installed binary could not find ONNX Runtime or the model | `Install.SelfContained` |

## 7. Roadmap

| Milestone | Goal | Status |
|---|---|---|
| M0 Re-foundation | CMake/CUDA 13/C++20 build, test harness, RAII core, CLI | ✅ |
| M1 Preprocessing | custom kernels + oracles + NPP benchmarks | ✅ |
| M2 Layout | Hough deskew, GPU rotation, projection line/word detection | ✅ |
| MOD Modular core | result types, component interfaces, registry, config-built pipeline | ✅ |
| M3 Functional OCR | fixtures + CER tool, ONNX recognizer, CTC decoder, text output, tuned defaults | ✅ |
| M4 Usable product | JSON/hOCR, TOML config, multi-page, PDF input, real-scan accuracy | ✅ (CI deferred) |
| **M5** Custom recognizer | hand-written CUDA inference for the recognition model, ONNX Runtime as oracle | next candidate |
| **M6** Optimize & upgrade | DBNet detector, full-page layout, stain robustness, recognizer throughput, kernel optimization, beam search | backlog |

## 8. Evaluation data

| Source | What | Use |
|---|---|---|
| Synthetic (`scripts/gen_fixtures.py`) | seeded renders of known text with DejaVu/Liberation fonts: clean, skewed, noisy, inverted; 10 committed, 50 on demand | unit/integration tests, CER ratchets |
| `resources/` + `tests/fixtures/real/` | 3 hand-transcribed real pages (public-domain book page; two CC BY 4.0 NoisyOffice-style scans, stained and crumpled) — sources in `resources/README.md` | real-scan CER ratchet |
| UNLV/ISRI (`scripts/fetch_unlv.sh`) | 40-page seeded sample of business letters + DOE reports, 300 dpi bitonal, with zone files | zoned and full-page CER; © UNLV, never committed |
| `tests/fixtures/pdf/` | 2-page JPEG-embedded PDF of two synthetic pages | PDF input tests |

Model: PP-OCRv5 English recognition, ONNX export (Apache-2.0), fetched by
`scripts/fetch_models.sh`; model card [models/README.md](../models/README.md).
Compute: a single RTX 3060 Laptop (sm_86, 6 GB) covers everything; GitHub-hosted CI has no
GPU, so `scripts/verify.sh` is the local gate for every PR.
