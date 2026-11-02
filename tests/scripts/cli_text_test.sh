#!/usr/bin/env bash
# CLI text output (M3.9): stdout is exactly the recognized text, -o picks the
# format from the extension, nothing found -> exit 3. Skips without the model.
set -uo pipefail
CLI="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS="${CUDAOCR_MODEL_DIR:-$ROOT/models}"
[ -f "$MODELS/rec.onnx" ] || { echo "SKIP: models not fetched (scripts/fetch_models.sh)"; exit 77; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0
check() { if [ "$2" == "$3" ]; then echo "ok   - $1"; else echo "FAIL - $1"; echo "  want: [$2]"; echo "  got:  [$3]"; fail=1; fi; }

FIX="$ROOT/tests/fixtures/synth/line_pangram_sans"
check "stdout is the text" "$(cat "$FIX.gt.txt")" "$("$CLI" "$FIX.png" 2>/dev/null)"
check "cpu EP gives the same text" "$(cat "$FIX.gt.txt")" "$("$CLI" "$FIX.png" --ep cpu 2>/dev/null)"

"$CLI" "$FIX.png" -o "$TMP/out.txt" 2>/dev/null
check "-o file.txt writes text" "$(cat "$FIX.gt.txt")" "$(cat "$TMP/out.txt")"

"$CLI" "$FIX.png" -o "$TMP/out.json" 2>/dev/null
grep -q '"text": "The quick brown fox jumps over the lazy dog.\\n"' "$TMP/out.json" \
    && echo "ok   - -o file.json writes JSON with text" || { echo "FAIL - json text"; fail=1; }

"$CLI" "$ROOT/tests/fixtures/blank.png" >/dev/null 2>&1
check "blank page exits 3" "3" "$?"
"$CLI" "$FIX.png" --model-dir /nonexistent >/dev/null 2>&1
check "missing model is a config error (exit 1)" "1" "$?"
exit $fail
