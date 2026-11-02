#!/usr/bin/env bash
# scripts/gen_fixtures.py must be byte-deterministic, and the committed fixture
# set must be exactly what the script produces (so it can always be regenerated).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GEN="$ROOT/scripts/gen_fixtures.py"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

python3 "$GEN" --out "$TMP/a" --seed 7 >/dev/null; rc=$?
[ $rc -eq 77 ] && exit 77  # fonts missing on this machine -> skipped
[ $rc -ne 0 ] && { echo "FAIL - generator exited $rc"; exit 1; }
python3 "$GEN" --out "$TMP/b" --seed 7 >/dev/null || exit 1
if diff -r "$TMP/a" "$TMP/b" >/dev/null; then echo "ok   - same seed -> identical bytes"
else echo "FAIL - same seed produced different files"; exit 1; fi

python3 "$GEN" --out "$TMP/c" --seed 8 >/dev/null || exit 1
if cmp -s "$TMP/a/page_sans_clean.gt.txt" "$TMP/c/page_sans_clean.gt.txt"; then
    echo "FAIL - different seed produced the same text"; exit 1
else echo "ok   - different seed -> different text"; fi

python3 "$GEN" --check "$ROOT/tests/fixtures/synth" || { echo "FAIL - committed set is stale"; exit 1; }
echo "ok   - committed set regenerates exactly"
