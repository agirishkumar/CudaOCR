# BACKLOG — ordered task queue

Rules
- Tasks run top-to-bottom within the active milestone, **one task = one branch = one PR** (workflow: architecture.md §13).
- A task is checked off (in its own PR) only when its acceptance criteria pass via `scripts/verify.sh`.
- Acceptance criteria must be machine-checkable. Add/split tasks freely; never reorder or delete a checked task.
- Unchecked tasks may be moved between milestones; leave a "→ moved to" note.

Active milestone: **MOD** (MOD.2 next) → then **M3**

## M0 — Re-foundation ✅ (2026-06-11)

- [x] **M0.1 CMake skeleton + GoogleTest.**
- [x] **M0.2 scripts/verify.sh.**
- [x] **M0.3 Status/Result + CUDA_CHECK.**
- [x] **M0.4 Logger v2.**
- [x] **M0.5 core RAII wrappers.**
- [x] **M0.6 Grayscale kernel (first oracle-TDD kernel).**
- [x] **M0.7 CLI skeleton.**
- [x] **M0.8 Legacy teardown.**

## M1 — Preprocessing custom kernels ✅ (2026-06-11)

- [x] **M1.1 Box-filter / integral image kernel.**
- [x] **M1.2 Adaptive mean threshold kernel.**
- [x] **M1.3 Otsu binarization.**
- [x] **M1.4 Median filter 3×3/5×5 kernel.**
- [x] **M1.5 Separable Gaussian kernel.**
- [x] **M1.6 Histogram equalization kernel.**
- [x] **M1.7 Inversion detection + normalization.**
- [x] **M1.8 Benchmark harness** → `benchmarks/RESULTS.md`.
- M1.9 Median optimization → moved to M6.1
- M1.10 Gaussian optimization → moved to M6.2

## M2 — Layout & segmentation ✅ (2026-06-11)

- [x] **M2.1 Hough skew detection — fixed** (`HoughAccumulatorBoundsRegression`).
- [x] **M2.2 GPU rotation kernel.**
- [x] **M2.3 Horizontal projection profile + line detection.**
- [x] **M2.4 Vertical profiles → word boxes.**
- M2.5 GPU connected-component labeling → moved to M6.4 (not needed for line-level CTC recognition)

## W — Workflow fixes (do first; they unblock the gate)

- [x] **W.1 CUDA arch default regression.** Set the `CMAKE_CUDA_ARCHITECTURES` default *before* `project()` (cache-overridable). Add regression test `CudaArchMatchesDevice` asserting a library kernel's `cudaFuncAttributes.binaryVersion` equals the device's compute capability (i.e. native SASS present, no PTX JIT).
  *Accept:* on a fresh `cmake -B build` (no `-D` flags) `ctest` → 100% pass on CUDA 13.4 + driver 13.2; `cuobjdump --list-elf build/libcudaocr_lib.a` shows `sm_86`; new test fails if arch is forced to 75.
- [x] **W.2 Branch-aware verify gate.** `verify.sh` computes changed files as `git diff --name-only $(git merge-base HEAD origin/main)` ∪ working tree, so memcheck/format run on everything the PR touches; add a `--full` flag (memcheck all tests regardless).
  *Accept:* on a branch with a committed `.cu` change and clean tree, `verify.sh` runs memcheck and format on that file; on `main` with clean tree it skips them.

- [x] **W.3 Format the whole tree once.** 16 files predate clang-format enforcement (found by the W.2 gate during MOD.1); format them in one whitespace-only PR so later task diffs stay clean. Add `verify.sh --all-format` to check every tracked source.
  *Accept:* `git diff -w` of the PR shows only line re-wrapping; `verify.sh --all-format` passes; all tests green.

## MOD — Modular core (no behavior change; makes every later task a drop-in)

- [x] **MOD.1 Result types.** `include/cudaocr/types.hpp` (host-only): `BBox`, `WordResult{box, text, conf}`, `LineResult{box, words, text, conf}`, `PageResult{lines, skewDeg, inverted, timings}`. Conversion from existing `TextLine`.
  *Accept:* unit tests for construction/conversion; header compiles in a `.cpp` TU without nvcc.
- [x] **MOD.2 Component interfaces.** `include/cudaocr/components/`: `ImageStage` (DeviceImage→DeviceImage), `TextDetector` (binary DeviceImage→`vector<LineRegion>`), `LineRecognizer` (gray DeviceImage + regions → `Logits`), `Decoder` (`Logits`→text+conf), `Formatter` (`PageResult`→string). Each with `name()` and `Status`-returning methods; a `Params` key/value map type with typed getters + validation errors.
  *Accept:* interface headers compile; `Params` unit tests (missing key → default, bad type → `InvalidArgument`).
- [x] **MOD.3 Wrap existing stages.** `ImageStage`s: `GrayscaleStage`, `EqualizeStage`, `MedianStage`, `GaussianStage`, `PolarityStage` (sets `page->inverted`), `AdaptiveBinarizer`, `OtsuBinarizer`; `Deskewer`: `HoughDeskewer`; `TextDetector`: `ProjectionDetector` (returns `toLineResults(...)`). Each takes `Params` and validates with `requireKnown`. Thin adapters over existing kernels — kernels untouched.
  *Accept:* per-adapter tests: output identical to calling the underlying free function.
- [ ] **MOD.4 Registry + config-built Pipeline.** `ComponentRegistry` (name → factory); `PipelineConfig{preprocess: [ {name, params} ], binarizer, deskew, detector, recognizer?, decoder?, formatter}`; `Pipeline::create(cfg)` validates once; `Pipeline::run(HostImage) → PageResult` with per-component CUDA-event timing; CLI flags map onto the config.
  *Accept:* golden-equivalence test — new `Pipeline` reproduces `runPipeline` lines/words/skew exactly on all existing pipeline fixtures; unknown component name → `InvalidArgument` naming it; then `runPipeline` is deleted and all tests stay green.

## M3 — Functional OCR (text out)

- [ ] **M3.1 Synthetic ground-truth fixtures.** `scripts/gen_fixtures.py` (Python + Pillow, seeded): renders pages/lines of known text with DejaVu/Liberation fonts at 300 DPI; variants: clean, skew ±5°, gaussian noise, inverted. Writes `*.png` + `*.gt.txt`. Commit a small set (≤ 2 MB) under `tests/fixtures/synth/`.
  *Accept:* script is deterministic (same seed → identical bytes); existing segmentation finds the expected line count on the committed clean pages.
- [ ] **M3.2 CER/WER evaluation module + tool.** `src/eval/` Levenshtein-based CER/WER (host C++), `cudaocr-eval <pred_dir> <gt_dir>` → per-file + aggregate table (markdown).
  *Accept:* unit tests on known pairs (identical → 0, one substitution, insert/delete, empty strings, UTF-8 multi-byte).
- [ ] **M3.3 Gray deskew path.** Pipeline keeps the deskewed gray image (same angle as binary) for recognition; `--dump-stages` writes it.
  *Accept:* test — rotating gray and binary with the detected angle yields aligned images (binary of rotated gray ≈ rotated binary within documented tolerance).
- [ ] **M3.4 ONNX Runtime dependency.** CMake option `CUDAOCR_WITH_ORT` (default ON): download prebuilt `onnxruntime-linux-x64-gpu_cuda13-<ver>.tgz` (≥1.27, SHA-256 pinned) via FetchContent; imported target; core lib still builds with it OFF.
  *Accept:* smoke test runs a tiny committed ONNX model (e.g. Add/Identity) on CPU EP and CUDA EP and gets the expected output; `-DCUDAOCR_WITH_ORT=OFF` build + ctest green.
- [ ] **M3.5 Model fetch + model card.** `scripts/fetch_models.sh` downloads PP-OCRv5 English `rec.onnx` + `dict.txt` into `models/` (gitignored), verifies SHA-256. Document in `models/README.md`: input name/shape/normalization, output shape, dictionary size + blank index, ONNX op list (sizes M5).
  *Accept:* a test (skipped with a clear message if models absent) loads the model and asserts the documented I/O shapes.
- [ ] **M3.6 Line crop + normalize kernel.** `LinePreprocessor`: crop each `LineRegion` from the deskewed gray image (with padding), bilinear-resize to model height keeping aspect, pad to batch max width, write float NCHW with the model's normalization. CPU oracle.
  *Accept:* max abs diff ≤ 1e-4 vs oracle on fixture lines; memcheck clean.
- [ ] **M3.7 CTC greedy decoder.** `CtcGreedyDecoder` implements `Decoder`: argmax per timestep, collapse repeats, drop blank, map via dictionary; confidence = mean max-prob of emitted chars; keeps per-char timestep for later word-box alignment.
  *Accept:* unit tests on hand-built logits (repeats, blanks between repeats, all-blank → empty, unknown index → error).
- [ ] **M3.8 OnnxLineRecognizer.** Implements `LineRecognizer`; batches lines (configurable max batch/width), EP from params (`cuda`|`cpu`), session created once per Pipeline.
  *Accept:* on a rendered single-line fixture ("The quick brown fox jumps over the lazy dog") greedy-decoded text matches exactly on both EPs; CPU and CUDA logits agree within 1e-3.
- [ ] **M3.9 End-to-end text + txt formatter.** Wire recognizer+decoder into `Pipeline`; `TxtFormatter`; CLI prints text to stdout or `-o file.txt`; exit code 3 when no text.
  *Accept:* integration test — CER ≤ 5% on committed clean synthetic pages; `cudaocr resources/sample.png` exits 0 with non-empty output.
- [ ] **M3.10 Accuracy baseline report.** Run eval on all synthetic variants + 1–2 hand-transcribed `resources/` pages; write `benchmarks/ACCURACY.md`; add a ratchet test that fails if clean-synthetic CER regresses by > 1 point.
  *Accept:* report committed; ratchet test green.

## M4 — Usable product

- [ ] **M4.1 JSON formatter** (page → lines → words with boxes, text, confidences; schema documented). Word boxes from CTC timestep → x-coordinate mapping.
- [ ] **M4.2 hOCR formatter** (validates as well-formed XHTML; `ocr_line`/`ocrx_word` with `bbox`).
- [ ] **M4.3 TOML config** (toml++ via FetchContent): full `PipelineConfig` from file, CLI overrides on top; `--print-config`.
- [ ] **M4.4 Multi-page input** (directory / multiple paths; reuse one Pipeline + ORT session; per-page outputs).
- [ ] **M4.5 GitHub Actions CI** (CUDA toolkit container, compile-only build with `CUDAOCR_WITH_ORT=ON`, clang-format check, host-only tests).
- [ ] **M4.6 Real-scan accuracy** (20–50 UNLV/ISRI pages via a fetch script, not committed) → table in `benchmarks/ACCURACY.md`; README results section.

## M5 — Custom CUDA recognizer (ORT becomes the oracle)

Expand into per-op tasks after M3.5 documents the model's op list. Expected shape:
- [ ] M5.1 Weight loader (ONNX initializers → device buffers; BN folded into conv).
- [ ] M5.2 Conv2d / depthwise conv kernels (implicit GEMM), oracle = ORT-CPU single-op models.
- [ ] M5.3 Activations (hardswish, GELU/ReLU), layernorm, linear/matmul, softmax kernels.
- [ ] M5.4 Attention block (if present in the model).
- [ ] M5.5 `CudaLineRecognizer` assembling the graph; registered as `recognizer = "cuda"`.
  *Accept (M5 overall):* logits within 1e-3 of ORT-CPU on fixtures; same decoded text; benchmark vs ORT CUDA EP.

## M6 — Optimize & upgrade (swap-ins; none change the interfaces)

- [ ] **M6.1 Median filter optimization** (was M1.9; 27× slower than NPP): shared-memory tiles + sorting network. *Accept:* byte-exact tests green; ≤ 2× NPP.
- [ ] **M6.2 Gaussian optimization** (was M1.10; 2.75× NPP): shared-memory tiles. *Accept:* byte-exact; ≤ 2× NPP.
- [ ] **M6.3 DBNet detector swap-in** (`detector = "dbnet-onnx"`, PP-OCRv5 det model): handles multi-column pages. *Accept:* reading-order-correct text on a 2-column synthetic fixture where `projection` fails.
- [ ] **M6.4 GPU connected-component labeling** (was M2.5): union-find; char boxes for debug overlay. *Accept:* component count == CPU oracle.
- [ ] **M6.5 CTC beam search + optional wordlist.**
- [ ] **M6.6 Multi-stream batch throughput** for M4.4.
