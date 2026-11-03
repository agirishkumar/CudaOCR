#!/usr/bin/env bash
# CLI PDF input (M4.7). Skips without the model or without PDF support.
set -uo pipefail
CLI="$1" EVAL="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "${CUDAOCR_MODEL_DIR:-$ROOT/models}/rec.onnx" ] || { echo "SKIP: models not fetched"; exit 77; }
PDF="$ROOT/tests/fixtures/pdf/two_pages.pdf"
SYN="$ROOT/tests/fixtures/synth"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
if "$CLI" "$PDF" --no-recognize >/dev/null 2>"$TMP/err" || [ $? -eq 3 ]; then :; else
    grep -q "built without PDF support" "$TMP/err" && { echo "SKIP: built without PDF support"; exit 77; }
fi
fail=0
ok() { echo "ok   - $1"; }
bad() { echo "FAIL - $1"; fail=1; }
cer() { "$EVAL" "$1" "$2" | tail -1 | awk -F'|' '{gsub(/[ *%]/, "", $3); print $3}'; }

"$CLI" "$PDF" -o "$TMP/pdf/" 2>/dev/null; rc=$?
[ $rc -eq 0 ] && [ -f "$TMP/pdf/two_pages-p001.txt" ] && [ -f "$TMP/pdf/two_pages-p002.txt" ] \
    && ok "2-page PDF -> two_pages-p001/p002, exit 0" || bad "pdf outputs ($rc)"

# Same text as OCR on the source PNGs (acceptance: within 0.5 CER points).
mkdir -p "$TMP/gt" "$TMP/png" "$TMP/pdfjoined"
cp "$ROOT/tests/fixtures/pdf/two_pages.gt.txt" "$TMP/gt/two_pages.gt.txt"
cat "$TMP/pdf/two_pages-p001.txt" "$TMP/pdf/two_pages-p002.txt" > "$TMP/pdfjoined/two_pages.txt"
"$CLI" "$SYN/page_sans_clean.png" "$SYN/line_pangram_sans.png" -o "$TMP/pngout/" 2>/dev/null
cat "$TMP/pngout/page_sans_clean.txt" "$TMP/pngout/line_pangram_sans.txt" > "$TMP/png/two_pages.txt"
pdfCer=$(cer "$TMP/pdfjoined" "$TMP/gt"); pngCer=$(cer "$TMP/png" "$TMP/gt")
awk -v a="$pdfCer" -v b="$pngCer" 'BEGIN { exit !(a <= b + 0.5) }' \
    && ok "PDF CER ${pdfCer}% vs source PNGs ${pngCer}% (within 0.5 pt)" || bad "PDF CER ${pdfCer}% vs ${pngCer}%"

out=$("$CLI" "$PDF" 2>/dev/null)
[ "$(printf '%s' "$out" | tr -cd '\f' | wc -c)" -eq 1 ] && ok "stdout: pages separated by one form feed" || bad "stdout"

"$CLI" "$PDF" --pdf-dpi 150 -o "$TMP/lo/" 2>/dev/null && [ -s "$TMP/lo/two_pages-p002.txt" ] \
    && ok "--pdf-dpi 150 works" || bad "--pdf-dpi 150"
for v in 0 -1 abc 9999; do
    "$CLI" "$PDF" --pdf-dpi "$v" >/dev/null 2>&1
    [ $? -eq 1 ] || bad "--pdf-dpi $v should be rejected"
done
ok "--pdf-dpi validation"

mkdir -p "$TMP/mixed" && cp "$PDF" "$SYN/line_punct_mono.png" "$TMP/mixed/" && echo "not a pdf" > "$TMP/mixed/broken.pdf"
err=$("$CLI" "$TMP/mixed" -o "$TMP/mixed_out" 2>&1 >/dev/null); rc=$?
[ $rc -eq 1 ] && echo "$err" | grep -q broken.pdf && [ -f "$TMP/mixed_out/two_pages-p002.txt" ] \
    && [ -f "$TMP/mixed_out/line_punct_mono.txt" ] \
    && ok "directory with PDF + PNG + broken PDF: others processed, exit 1" || { bad "mixed dir ($rc)"; echo "$err"; }
exit $fail
