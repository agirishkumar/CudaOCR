#!/usr/bin/env bash
# Regression: `cmake --install` produced a cudaocr that could not start
# (libonnxruntime.so.1 not found) and only found the model inside the source
# tree. Install into a temp prefix and OCR from another directory.
set -uo pipefail
BUILD="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "$ROOT/models/rec.onnx" ] || { echo "SKIP: models not fetched"; exit 77; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cmake --install "$BUILD" --prefix "$TMP/prefix" >/dev/null || { echo "FAIL: install"; exit 1; }
cp "$ROOT/tests/fixtures/synth/line_pangram_sans.png" "$TMP/page.png"
# No CUDAOCR_MODEL_DIR, no source tree on the path: everything must come from the prefix.
out=$(cd "$TMP" && env -u CUDAOCR_MODEL_DIR "$TMP/prefix/bin/cudaocr" page.png 2>"$TMP/err")
rc=$?
want="$(cat "$ROOT/tests/fixtures/synth/line_pangram_sans.gt.txt")"
if [ $rc -eq 0 ] && [ "$out" == "$want" ]; then
    echo "ok   - installed cudaocr runs from another directory and reads the page"
else
    echo "FAIL - rc=$rc out=[$out]"; cat "$TMP/err"; exit 1
fi
grep -q "share/cudaocr/models" <(cd "$TMP" && "$TMP/prefix/bin/cudaocr" --print-config -v 2>&1; ls -R "$TMP/prefix") \
    && echo "ok   - model installed under share/cudaocr/models" || { echo "FAIL - model not installed"; exit 1; }
