#!/usr/bin/env bash
# CLI multi-page input (M4.4). Skips without the model.
set -uo pipefail
CLI="$1" EVAL="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -f "${CUDAOCR_MODEL_DIR:-$ROOT/models}/rec.onnx" ] || { echo "SKIP: models not fetched"; exit 77; }
SYN="$ROOT/tests/fixtures/synth"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0
ok() { echo "ok   - $1"; }
bad() { echo "FAIL - $1"; fail=1; }

"$CLI" "$SYN" -o "$TMP/out" 2>/dev/null; rc=$?
n=$(ls "$TMP/out"/*.txt 2>/dev/null | wc -l)
[ $rc -eq 0 ] && [ "$n" -eq 10 ] && ok "directory -> 10 per-page files, exit 0" || bad "directory ($rc, $n files)"
"$EVAL" "$TMP/out" "$SYN" --max-cer 0.0135 >/dev/null && ok "multi-page CER within the synth ratchet" || bad "CER"

"$CLI" "$SYN/page_sans_clean.png" "$SYN/line_pangram_sans.png" -o "$TMP/js" --format json 2>/dev/null
[ -f "$TMP/js/page_sans_clean.json" ] && [ -f "$TMP/js/line_pangram_sans.json" ] && ok "--format json per page" || bad "json files"

out=$("$CLI" "$SYN/line_pangram_sans.png" "$SYN/line_digits_serif.png" 2>/dev/null)
want="$(cat "$SYN/line_pangram_sans.gt.txt")"$'\n\f'"$(cat "$SYN/line_digits_serif.gt.txt")"
[ "$out" == "$want" ] && ok "stdout pages separated by form feed" || { bad "stdout concat"; printf '%q\n' "$out"; }

"$CLI" "$SYN/line_pangram_sans.png" "$SYN/line_digits_serif.png" --format json >/dev/null 2>&1
[ $? -eq 1 ] && ok "json to stdout with several pages is rejected" || bad "json stdout"

mkdir -p "$TMP/mixed" && cp "$SYN/line_pangram_sans.png" "$SYN/line_punct_mono.png" "$TMP/mixed/"
echo "not an image" > "$TMP/mixed/broken.png"
err=$("$CLI" "$TMP/mixed" -o "$TMP/mixed_out" 2>&1 >/dev/null); rc=$?
[ $rc -eq 1 ] && echo "$err" | grep -q broken.png && [ -f "$TMP/mixed_out/line_punct_mono.txt" ] \
    && ok "bad page reported, others processed, exit 1" || { bad "bad page ($rc)"; echo "$err"; }

mkdir -p "$TMP/d1" "$TMP/d2" && cp "$SYN/line_pangram_sans.png" "$TMP/d1/a.png" && cp "$SYN/line_digits_serif.png" "$TMP/d2/a.png"
"$CLI" "$TMP/d1/a.png" "$TMP/d2/a.png" -o "$TMP/dup" >/dev/null 2>&1
[ $? -eq 1 ] && [ ! -e "$TMP/dup/a.txt" ] && ok "colliding output names rejected up front" || bad "collision"

"$CLI" "$SYN/line_pangram_sans.png" -o "$TMP/single/" 2>/dev/null
[ -f "$TMP/single/line_pangram_sans.txt" ] && ok "single page with -o dir/ writes into the dir" || bad "single to dir"

# One pipeline/session for all pages: 4 pages in one run vs 4 separate runs.
ms() { date +%s%3N; }
t0=$(ms); "$CLI" "$SYN"/page_*_clean.png "$SYN/page_sans_noise.png" -o "$TMP/t1" 2>/dev/null; t1=$(ms)
for f in "$SYN"/page_*_clean.png "$SYN/page_sans_noise.png"; do "$CLI" "$f" >/dev/null 2>&1; done; t2=$(ms)
one=$((t1 - t0)); sep=$((t2 - t1))
[ $((one * 10)) -lt $((sep * 6)) ] && ok "one run ${one} ms < 0.6 x separate runs ${sep} ms" || bad "no speed-up (${one} vs ${sep} ms)"
exit $fail
