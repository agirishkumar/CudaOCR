#!/usr/bin/env bash
# CudaOCR verification gate — the loop's verification condition.
# Exit 0 = iteration may progress; non-zero = retry.
# Usage: verify.sh [--full] [--all-format]
#   --full        memcheck even when no kernel files changed
#   --all-format  clang-format check every tracked source, not just changed ones
# Changed files are everything the branch touches vs its merge-base with
# origin/main (see scripts/changed_files.sh), so committed work on a task branch
# is still checked.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

FULL=0
ALL_FORMAT=0
for arg in "$@"; do
    case "$arg" in
        --full) FULL=1 ;;
        --all-format) ALL_FORMAT=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

echo "== [1/4] configure + build =="
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)"

echo "== [2/4] tests =="
ctest --test-dir build --output-on-failure

echo "== [3/4] memcheck (if .cu/.cuh changed vs branch base, or --full) =="
SANITIZER="${SANITIZER:-/usr/local/cuda/bin/compute-sanitizer}"
KERNELS=$(scripts/changed_files.sh '*.cu' '*.cuh')
if [ "$FULL" -eq 1 ] || [ -n "$KERNELS" ]; then
    [ -n "$KERNELS" ] && echo "changed kernel files:" $KERNELS
    if [ -x "$SANITIZER" ]; then
        # --report-api-errors no: the CudaCheck test triggers an API error on purpose;
        # memory violations are still fully reported.
        "$SANITIZER" --tool memcheck --report-api-errors no --error-exitcode 1 \
            ./build/tests/unit_tests
    else
        echo "WARNING: compute-sanitizer not found; skipping" >&2
    fi
else
    echo "no kernel changes — skipped"
fi

echo "== [4/4] formatting =="
SOURCES=('*.cu' '*.cuh' '*.hpp' '*.cpp' '*.h')
if [ "$ALL_FORMAT" -eq 1 ]; then
    CHANGED=$(git ls-files -- "${SOURCES[@]}" ':!third_party')
else
    CHANGED=$(scripts/changed_files.sh "${SOURCES[@]}")
fi
if [ -n "$CHANGED" ] && command -v clang-format >/dev/null; then
    echo "checking $(echo "$CHANGED" | wc -l) file(s)"
    # shellcheck disable=SC2086
    clang-format --dry-run --Werror $CHANGED
else
    echo "nothing to check / clang-format missing"
fi

echo "VERIFY: PASS"
