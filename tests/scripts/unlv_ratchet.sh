#!/usr/bin/env bash
# UNLV accuracy ratchet (M4.6): baseline + 1 CER point for the default sample
# (scripts/fetch_unlv.sh: 40 pages, seed 1). Skips (77) without the model or the
# data, or when a different sample was prepared (thresholds would not apply).
#   unlv_ratchet.sh <cudaocr> <cudaocr-eval> <max_zoned> <max_page>
set -uo pipefail
CLI="$1" EVAL="$2" MAX_ZONED="$3" MAX_PAGE="$4"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
U="$ROOT/data/unlv"
[ -f "${CUDAOCR_MODEL_DIR:-$ROOT/models}/rec.onnx" ] || { echo "SKIP: models not fetched"; exit 77; }
[ "$(cat "$U/sample.txt" 2>/dev/null)" == "40 pages (194 zones), seed 1" ] ||
    { echo "SKIP: default UNLV sample not prepared (scripts/fetch_unlv.sh)"; exit 77; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0

"$CLI" "$U/zones" -o "$TMP/z/" 2>/dev/null || [ $? -eq 3 ]
mkdir -p "$TMP/zoned"
for list in "$U/zones"/*.zones; do
    page="$(basename "$list" .zones)"
    : >"$TMP/zoned/$page.txt"
    while read -r stem; do
        [ -n "$stem" ] && [ -f "$TMP/z/$stem.txt" ] && cat "$TMP/z/$stem.txt" >>"$TMP/zoned/$page.txt"
    done <"$list"
done
echo "zoned:"
"$EVAL" "$TMP/zoned" "$U/page" --ref-wildcard '~' --max-cer "$MAX_ZONED" | tail -1 || fail=1

"$CLI" "$U/page" -o "$TMP/page/" 2>/dev/null || [ $? -eq 3 ] || { echo "FAIL: page run errored"; fail=1; }
echo "page:"
"$EVAL" "$TMP/page" "$U/page" --ref-wildcard '~' --max-cer "$MAX_PAGE" | tail -1 || fail=1
exit $fail
