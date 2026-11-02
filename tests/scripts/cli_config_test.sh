#!/usr/bin/env bash
# CLI config (M4.3): --print-config round trip, file < flag precedence, errors.
# Needs no model: every run here is detection-only or --print-config.
set -uo pipefail
CLI="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0
ok() { echo "ok   - $1"; }
bad() { echo "FAIL - $1"; fail=1; }

"$CLI" --print-config > "$TMP/a.toml" || bad "--print-config exits 0"
"$CLI" --config "$TMP/a.toml" --print-config > "$TMP/b.toml"
cmp -s "$TMP/a.toml" "$TMP/b.toml" && ok "print -> load -> print is a fixed point" || bad "round trip"

cat > "$TMP/c.toml" <<'TOML'
binarizer = { name = "adaptive", C = 35 }
recognizer = false
TOML
"$CLI" --config "$TMP/c.toml" --print-config | grep -q 'binarizer = { name = "adaptive", C = 35 }' \
    && ok "file value used when no flag" || bad "file value"
"$CLI" --config "$TMP/c.toml" --C 20 --print-config | grep -q 'binarizer = { name = "adaptive", C = 20 }' \
    && ok "explicit flag overrides the file" || bad "flag override"
"$CLI" --config "$TMP/c.toml" --print-config | grep -q '^recognizer = false$' \
    && ok "recognizer = false honoured" || bad "recognizer = false"

out=$("$CLI" "$ROOT/resources/sample.png" --config "$TMP/c.toml" -v 2>&1 >/dev/null)
echo "$out" | grep -q "lines: 35" && ok "config drives a real run (detection only)" || { bad "config run"; echo "$out"; }

printf '\n\nbinariser = { name = "otsu" }\n' > "$TMP/typo.toml"
msg=$("$CLI" --config "$TMP/typo.toml" --print-config 2>&1 >/dev/null); rc=$?
[ $rc -eq 1 ] && echo "$msg" | grep -q "typo.toml:3" && echo "$msg" | grep -q binariser \
    && ok "unknown key: exit 1 with file:line" || { bad "unknown key"; echo "$msg"; }

msg=$("$CLI" --config /nonexistent.toml --print-config 2>&1); rc=$?
[ $rc -eq 1 ] && echo "$msg" | grep -q "/nonexistent.toml" && ok "missing file: exit 1" || bad "missing file"

"$CLI" --print-config --otsu --C 20 >/dev/null 2>&1
[ $? -eq 1 ] && ok "--otsu with --C is rejected" || bad "otsu/C conflict"
exit $fail
