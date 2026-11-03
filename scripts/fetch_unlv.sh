#!/usr/bin/env bash
# Fetch and prepare a real-scan evaluation sample from the UNLV/ISRI OCR test data
# (300 dpi bitonal scans with ground truth, as used by Tesseract's UNLV tests).
#
#   scripts/fetch_unlv.sh [--sample N] [--seed S]     # default: 40 pages, seed 1
#
# Writes data/unlv/ (gitignored): download/ (tarballs, SHA-256 verified),
# raw/ (extracted), page/<id>.png + <id>.gt.txt, zones/ (per-zone crops).
#
# LICENSE: the data is (c) 1996 Regents of the University of Nevada and "may not
# be reproduced in whole or in part without written permission from UNLV". It is
# downloaded for local evaluation only and must never be committed or quoted;
# only aggregate scores go into the repo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/unlv"
SAMPLE=40
SEED=1
while [ $# -gt 0 ]; do
    case "$1" in
        --sample) SAMPLE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
BASE="https://sourceforge.net/projects/isri-ocr-evaluation-tools-alt/files"
# set | sha256 (business letters and DOE reports: mostly single-column)
SETS=(
    "bus.3B|4b7268f095f7fa4bc190ee15ee61e848342d6f8b21f2054b0e9a8a49875a6b3f"
    "doe3.3B|dc5e401217c7201a7e70630f505f6c1b1781878cda2a6f5461670cbf1788893e"
)
mkdir -p "$DATA/download" "$DATA/raw"
for entry in "${SETS[@]}"; do
    IFS='|' read -r set sha <<<"$entry"
    tgz="$DATA/download/$set.tar.gz"
    if ! { [ -f "$tgz" ] && echo "$sha  $tgz" | sha256sum -c --status; }; then
        echo "downloading $set ..."
        curl -sSfL --retry 3 -o "$tgz.part" "$BASE/$set.tar.gz/download"
        echo "$sha  $tgz.part" | sha256sum -c --status ||
            { echo "FAIL $set: SHA-256 mismatch" >&2; rm -f "$tgz.part"; exit 1; }
        mv "$tgz.part" "$tgz"
    fi
    [ -d "$DATA/raw/$set" ] || tar xzf "$tgz" -C "$DATA/raw"
done
python3 "$ROOT/scripts/prepare_unlv.py" --raw "$DATA/raw" --out "$DATA" \
    --sets bus.3B doe3.3B --sample "$SAMPLE" --seed "$SEED"
