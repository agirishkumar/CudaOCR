# LOOP_STATE — machine state for /engineer-step iterations

Active milestone: M4 (M4.5 deferred by user; next after M4.6: M4.7 PDF input)

CURRENT: (none)
branch: task/M4.6-real-scans
pr: (none)
attempts: 0
lastFailure: (none)

## DONE
- M0.1–M0.8, 2026-06-11, "M0: re-foundation — CMake/CUDA 13.1, core RAII, logger, CLI, legacy teardown"
- M1.1–M1.8, 2026-06-11, "M1: preprocessing kernels (box/median/gauss/adaptive/otsu/equalize/invert) + bench"
- M2.1–M2.4, 2026-06-11, "M2: hough deskew (baseline voting, bounds regression), GPU rotate, line/word segmentation (char boxes via column gaps; CCL not done)"
- docs, 2026-11-02, "docs: re-audit status, refine roadmap (W, MOD, M3–M6), branch-per-task workflow" (committed directly to main, pre-workflow)
- W.1, 2026-11-02, "build: set CUDA arch default before project() (W.1)" — PR #1 merged
- W.2, 2026-11-02, "verify: check everything the branch changes, not just the working tree (W.2)" — PR #2 merged
- MOD.1, 2026-11-02, "types: add host-only page/line/word result types (MOD.1)" — PR #3 merged
- W.3, 2026-11-02, "style: clang-format the whole tree; add verify.sh --all-format (W.3)" — PR #4 merged
- MOD.2, 2026-11-02, "components: add stage interfaces and Params (MOD.2)" — PR #5 merged
- MOD.3, 2026-11-02, "components: wrap existing kernels as built-in components (MOD.3)" — PR #6 merged
- MOD.4, 2026-11-02, "pipeline: registry + config-built Pipeline; fix stream race; pin legacy output (MOD.4)" — PR #7 merged
- M3.1, 2026-11-02, "fixtures: synthetic ground-truth generator; merge descender slivers (M3.1)" — PR #8 merged
- M3.2, 2026-11-02, "eval: CER/WER metrics and cudaocr-eval tool (M3.2)" — PR #9 merged
- M3.3, 2026-11-02, "pipeline: deskew the gray page with the binary page's angle (M3.3)" — PR #10 merged
- M3.4, 2026-11-02, "build: add pinned ONNX Runtime 1.31.1 (CUDA 13) dependency (M3.4)" — PR #11 merged
- M3.5, 2026-11-02, "models: pinned PP-OCRv5 English recognizer fetch + model card (M3.5)" — PR #12 merged
- M3.6, 2026-11-02, "stages: GPU line crop/resize/normalize for the recognizer (M3.6)" — PR #13 merged
- M3.7, 2026-11-02, "components: CTC greedy decoder (M3.7)"
- M3.8, 2026-11-02, "components: ONNX Runtime line recognizer (M3.8)"
- M3.9, 2026-11-02, "pipeline: end-to-end text, txt formatter, CLI text output (M3.9)"
- M3.10, 2026-11-02, "benchmarks: accuracy baseline, real-scan ground truth, ratchets; fix ORT exit crash (M3.10)"
- M3.11, 2026-11-02, "pipeline: adaptive C 40 default, binarizePrep slot, new ratchets (M3.11)" — PR #14 merged
- M4.1, 2026-11-02, "output: JSON schema cudaocr.page/1 with recognized word boxes (M4.1)"
- M4.2, 2026-11-02, "output: hOCR formatter (M4.2)"
- M4.3, 2026-11-02, "config: TOML pipeline config, --config/--print-config (M4.3)"
- M4.4, 2026-11-02, "cli: multi-page input with one pipeline per run (M4.4)" — PR #15 merged
- M4.6, 2026-11-02, "benchmarks: UNLV/ISRI real-scan accuracy; cap line width (M4.6)"

## AWAITING-MERGE
- PR #16 https://github.com/agirishkumar/CudaOCR/pull/16 (task/M4.6-real-scans)

## BLOCKED
(none)

## NEEDS-DECISION
(none)
