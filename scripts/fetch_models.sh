#!/usr/bin/env bash
# Download the recognition model into models/ (gitignored) and verify SHA-256.
# Idempotent: files already present with the right hash are not re-downloaded.
#
#   scripts/fetch_models.sh [dest_dir]     # default: <repo>/models
#
# Source: Hugging Face monkt/paddleocr-onnx (Apache-2.0), an ONNX export of
# PaddleOCR's en_PP-OCRv5_mobile_rec, pinned to an exact repo revision.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/models}"
REV=7b02d0a30a07ba2b92ad1ff5a8941ae2c633de65
BASE="https://huggingface.co/monkt/paddleocr-onnx/resolve/$REV/languages/english"

# local name | remote path | sha256
FILES=(
    "rec.onnx|rec.onnx|4e16deb22c4da6468bdca539b2cd3c8687825538b67109177c47d359ab994cd7"
    "dict.txt|dict.txt|e025a66d31f327ba0c232e03f407ae8d105e1e709e7ccb3f408aa778c24e70d6"
)

mkdir -p "$DEST"
for entry in "${FILES[@]}"; do
    IFS='|' read -r name remote sha <<<"$entry"
    out="$DEST/$name"
    if [ -f "$out" ] && echo "$sha  $out" | sha256sum -c --status; then
        echo "ok      $name (cached)"
        continue
    fi
    tmp="$out.part"
    curl -sSfL --retry 3 -o "$tmp" "$BASE/$remote"
    if ! echo "$sha  $tmp" | sha256sum -c --status; then
        echo "FAIL    $name: SHA-256 mismatch (got $(sha256sum "$tmp" | cut -d' ' -f1))" >&2
        rm -f "$tmp"
        exit 1
    fi
    mv "$tmp" "$out"
    echo "fetched $name ($(stat -c %s "$out") bytes)"
done
