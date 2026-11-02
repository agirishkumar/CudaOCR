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

python3 - "$TMP/out.json" <<'PY' && echo "ok   - JSON parses and follows cudaocr.page/1" || { echo "FAIL - JSON schema"; fail=1; }
import json, sys
d = json.load(open(sys.argv[1]))
assert d["schema"] == "cudaocr.page/1", d["schema"]
for k in ("width", "height", "skewDeg", "inverted", "text", "metrics", "timingsMs", "lines"):
    assert k in d, k
line = d["lines"][0]
assert len(line["bbox"]) == 4 and line["text"] and 0 < line["conf"] <= 1
words = [w["text"] for w in line["words"]]
assert words == "The quick brown fox jumps over the lazy dog.".split(), words
for w in line["words"]:
    x0, y0, x1, y1 = w["bbox"]
    assert line["bbox"][0] <= x0 <= x1 <= line["bbox"][2], w
PY

"$CLI" "$ROOT/tests/fixtures/synth/page_sans_clean.png" -o "$TMP/page.hocr" 2>/dev/null
"$CLI" "$ROOT/tests/fixtures/synth/page_sans_clean.png" -o "$TMP/page.json" 2>/dev/null
python3 - "$TMP/page.hocr" "$TMP/page.json" <<'PY' && echo "ok   - hOCR is well-formed XHTML matching the JSON" || { echo "FAIL - hOCR"; fail=1; }
import json, sys, xml.etree.ElementTree as ET
ns = {"h": "http://www.w3.org/1999/xhtml"}
root = ET.parse(sys.argv[1]).getroot()  # raises on malformed XML
page = json.load(open(sys.argv[2]))
lines = root.findall(".//h:span[@class='ocr_line']", ns)
assert len(lines) == len(page["lines"]), (len(lines), len(page["lines"]))
for hl, jl in zip(lines, page["lines"]):
    words = hl.findall("h:span[@class='ocrx_word']", ns)
    assert [w.text for w in words] == [w["text"] for w in jl["words"]]
    for w, jw in zip(words, jl["words"]):
        x0, y0, x1, y1 = jw["bbox"]
        assert w.get("title").startswith(f"bbox {x0} {y0} {x1 + 1} {y1 + 1}; x_wconf "), w.get("title")
PY

"$CLI" "$ROOT/tests/fixtures/blank.png" >/dev/null 2>&1
check "blank page exits 3" "3" "$?"
"$CLI" "$FIX.png" --model-dir /nonexistent >/dev/null 2>&1
check "missing model is a config error (exit 1)" "1" "$?"
exit $fail
