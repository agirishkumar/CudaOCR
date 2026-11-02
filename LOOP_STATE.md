# LOOP_STATE — machine state for /engineer-step iterations

Active milestone: M3 (next unchecked task: M3.5 model fetch + model card)

CURRENT: (none)
branch: (none)
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
- M3.4, 2026-11-02, "build: add pinned ONNX Runtime 1.31.1 (CUDA 13) dependency (M3.4)"

## AWAITING-MERGE
- PR #11 https://github.com/agirishkumar/CudaOCR/pull/11 (task/M3.4-onnxruntime)

## BLOCKED
(none)

## NEEDS-DECISION
(none)
