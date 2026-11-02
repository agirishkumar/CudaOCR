#!/usr/bin/env bash
# Tests scripts/changed_files.sh — the file list verify.sh uses to decide what to
# memcheck and clang-format. Runs against a throwaway git repo.
set -uo pipefail
SCRIPT="$(cd "$(dirname "$0")/../.." && pwd)/scripts/changed_files.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"

fails=0
g() { git -c user.name=t -c user.email=t@t "$@" >/dev/null 2>&1; }
expect() {  # expect <description> <expected newline-joined> <actual>
    if [ "$2" == "$3" ]; then echo "ok   - $1"; else
        echo "FAIL - $1"; echo "       expected: [$2]"; echo "       actual:   [$3]"; fails=$((fails + 1)); fi
}
run() { CUDAOCR_BASE_REF=main "$SCRIPT" '*.cu' '*.hpp'; }

g init -b main
mkdir -p src && echo a > src/a.cu && echo c > src/c.cu && echo d > doc.md
g add -A && g commit -m base

expect "main, clean tree -> nothing" "" "$(run)"

echo a2 > src/a.cu
expect "main, uncommitted edit -> listed" "src/a.cu" "$(run)"
g checkout -- src/a.cu

g switch -c task/x
echo a3 > src/a.cu && echo d2 > doc.md && g commit -am "edit"
expect "branch, committed edit, clean tree -> listed" "src/a.cu" "$(run)"

echo b > src/b.hpp
expect "untracked matching file -> listed" "$(printf 'src/a.cu\nsrc/b.hpp')" "$(run)"
rm src/b.hpp

g rm -q src/c.cu && g commit -m "delete"
expect "deleted file -> not listed" "src/a.cu" "$(run)"

expect "missing base ref -> falls back to HEAD (clean -> nothing)" "" \
    "$(CUDAOCR_BASE_REF=no/such/ref "$SCRIPT" '*.cu')"

[ "$fails" -eq 0 ] && echo "all passed" || { echo "$fails failed"; exit 1; }
