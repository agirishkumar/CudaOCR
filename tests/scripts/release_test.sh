#!/usr/bin/env bash
# Release properties (v0.1.0): --version output, and the CLI does not need the
# CUDA runtime shared library (cudart is linked statically), so it starts on a
# machine with only the NVIDIA driver; GPU recognition additionally needs the
# CUDA 13 math libraries + cuDNN, which ONNX Runtime's CUDA provider loads.
set -uo pipefail
CLI="$1" EVAL="$2"
fail=0
v=$("$CLI" --version) || { echo "FAIL - --version exit"; fail=1; }
echo "$v" | grep -qE '^cudaocr 0\.1\.0' && echo "ok   - $v" | head -1 || { echo "FAIL - version: $v"; fail=1; }
echo "$v" | grep -qE 'recognizer: (onnxruntime|none)' && echo "$v" | grep -qE 'pdf: (poppler|none)' \
    && echo "ok   - --version reports optional features" || { echo "FAIL - features: $v"; fail=1; }
"$EVAL" --version | grep -qE '^cudaocr-eval 0\.1\.0' && echo "ok   - cudaocr-eval --version" || { echo "FAIL - eval version"; fail=1; }
if readelf -d "$CLI" | grep -q 'libcudart'; then echo "FAIL - cudaocr needs libcudart.so (link cudart statically)"; fail=1
else echo "ok   - no libcudart.so dependency"; fi
exit $fail
