# BACKLOG — task queue

Rules
- One task = one branch (`task/<id>-<slug>`) = one PR, unless the user asks to batch several
  (workflow: [architecture.md](architecture.md) §11). A task is checked off in its own PR once its acceptance
  criteria pass `scripts/verify.sh`.
- Acceptance criteria are machine-checkable. Tasks may be added, split or moved between
  milestones; completed work is summarized below, the detail lives in the PRs and git history.

## Done (M0–M4, 2026-06-11 → 2026-11-02)

| Milestone | Delivered | PRs |
|---|---|---|
| M0 Re-foundation | CMake/CUDA 13/C++20 build, GoogleTest/CTest, `verify.sh`, Status/Result, logger, RAII GPU wrappers, grayscale kernel, CLI, legacy teardown | initial commits |
| M1 Preprocessing | box mean, adaptive threshold, Otsu, median, Gaussian, equalization, polarity kernels + CPU oracles; NPP benchmark harness | initial commits |
| M2 Layout | Hough skew detection (bounds-checked), GPU rotation, projection line and word detection | initial commits |
| W Workflow | CUDA arch default fix (W.1), branch-aware verify gate (W.2), whole-tree formatting (W.3) | #1, #2, #4 |
| MOD Modular core | result types (MOD.1), component interfaces + Params (MOD.2), built-in components (MOD.3), registry + config-built Pipeline, stream-race fix (MOD.4) | #3, #5, #6, #7 |
| M3 Functional OCR | synthetic fixtures (M3.1), CER/WER tool (M3.2), gray deskew (M3.3), ONNX Runtime (M3.4), model fetch + card (M3.5), line-prep kernel (M3.6), CTC decoder (M3.7), ONNX recognizer (M3.8), end-to-end text (M3.9), accuracy report + ratchets (M3.10), tuned defaults (M3.11) | #8–#14 |
| M4 Usable product | JSON + word boxes (M4.1), hOCR (M4.2), TOML config (M4.3), multi-page input (M4.4), UNLV real-scan accuracy (M4.6), PDF input (M4.7); self-contained install | #15–#17 |

Deferred: **M4.5 GitHub Actions CI** (user decision 2026-11-02) — hosted runners have no
GPU, so CI could only compile and format-check; `scripts/verify.sh` gates every PR locally.
Revisit when there are outside contributors.

## M5 — Custom CUDA recognizer (ONNX Runtime becomes the oracle)

The model's op list is in [models/README.md](../models/README.md) (25 op types: Conv, BatchNormalization,
HardSwish/HardSigmoid, MatMul, Softmax, decomposed LayerNorm, pooling, elementwise).
- [ ] **M5.1 Weight loader:** ONNX `Constant` tensors → device buffers; fold BatchNormalization into conv.
- [ ] **M5.2 Conv kernels:** Conv2d and depthwise conv (implicit GEMM); oracle = ORT-CPU single-op models.
- [ ] **M5.3 Elementwise + normalization kernels:** HardSwish/HardSigmoid/ReLU/Sigmoid, LayerNorm, pooling, Softmax.
- [ ] **M5.4 MatMul / attention block.**
- [ ] **M5.5 `CudaLineRecognizer`** assembling the graph, registered as `recognizer = "cuda"`.
  *Accept (M5):* probabilities within 1e-3 of ORT-CPU on the fixtures; identical decoded text;
  accuracy ratchets green; benchmark vs the ORT CUDA EP.

## M6 — Optimize & upgrade (swap-ins; interfaces unchanged)

Ordered by value for real documents:
- [ ] **M6.3 DBNet detector swap-in** (`detector = "dbnet-onnx"`, PP-OCRv5 detection model): multi-column pages. *Accept:* reading-order-correct text on a 2-column synthetic fixture where `projection` fails; ratchets green.
- [ ] **M6.9 Full-page layout on real documents.** UNLV full page 19.64 % vs zoned 6.84 % CER: zone reading order, tables, headers/footers, rules. Candidates: M6.3, column/zone segmentation, rule removal before detection. *Accept:* unlv-page within 5 points of unlv-zoned.
- [ ] **M6.8 Stained and speckled scans.** `noisy_text.png` 30.4 % CER (coffee stains; `--C 50` gives 17 %), isolated specks read as characters. First add a redistributable speckled fixture (e.g. a seeded speckle variant in `gen_fixtures.py`; the original speckled sample was removed for licensing). Candidates: Sauvola/Wolf thresholding, connected-component speck filter (M6.4), confidence gating of tiny lines. *Accept:* real-scan CER ≤ 2 % and speckle-fixture CER ≤ 5 % without regressing the clean ratchets.
- [ ] **M6.7 Recognizer throughput.** ≈ 370 ms warm on `sherlock.png` (33 lines); CPU EP ≈ 8× slower; maxBatch 8–16 best; TF32 ≈ 3 %; cuDNN algo-search modes no help. Profile with Nsight Systems first; candidates: IoBinding with preallocated outputs, CUDA graphs, FP16 model, TensorRT EP, a height cap for very small text. *Accept:* ≤ 150 ms warm on sherlock.png with CER unchanged.
- [ ] **M6.1 Median 5×5 optimization** (27× NPP): shared-memory tiles + sorting network. *Accept:* byte-exact tests green; ≤ 2× NPP.
- [ ] **M6.2 Gaussian optimization** (2.6× NPP): shared-memory tiles. *Accept:* byte-exact; ≤ 2× NPP.
- [ ] **M6.4 GPU connected-component labeling** (union-find): glyph boxes and speck filtering. *Accept:* component count == CPU oracle.
- [ ] **M6.5 CTC beam search + optional wordlist.** *Accept:* CER not worse than greedy on all sets; better on at least one.
- [ ] **M6.6 Multi-stream batch throughput:** overlap pages across streams in multi-page runs. *Accept:* ≥ 1.3× pages/s on the 50-page synthetic set with identical outputs.
