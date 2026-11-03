#!/usr/bin/env bash
# Accuracy ratchet: OCR the given fixtures with the CLI and fail if aggregate CER
# exceeds MAX_CER (a fraction). Skips (77) without the recognition model.
#
#   accuracy_ratchet.sh <cudaocr> <cudaocr-eval> <MAX_CER> <img_dir> <gt_dir> <stem>... [-- flags]
set -uo pipefail
CLI="$1" EVAL="$2" MAX="$3" IMGS="$4" GTS="$5"
shift 5
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "${CUDAOCR_MODEL_DIR:-$ROOT/models}/rec.onnx" ] || { echo "SKIP: models not fetched"; exit 77; }
STEMS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do STEMS+=("$1"); shift; done
[ "${1:-}" == "--" ] && shift
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/pred" "$TMP/gt"
IMAGES=()
for s in "${STEMS[@]}"; do
    cp "$GTS/$s.gt.txt" "$TMP/gt/"
    IMAGES+=("$IMGS/$s.png")
done
"$CLI" "${IMAGES[@]}" -o "$TMP/pred/" "$@" 2>/dev/null || [ $? -eq 3 ]
"$EVAL" "$TMP/pred" "$TMP/gt" --max-cer "$MAX"
