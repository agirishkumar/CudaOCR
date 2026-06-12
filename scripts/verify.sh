#!/usr/bin/env bash
# CudaOCR verification gate — the loop's verification condition.
# Exit 0 = iteration may progress; non-zero = retry.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "== [1/4] configure + build =="
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)"

echo "== [2/4] tests =="
ctest --test-dir build --output-on-failure

echo "== [3/4] memcheck (only if .cu/.cuh changed vs HEAD) =="
SANITIZER="${SANITIZER:-/usr/local/cuda/bin/compute-sanitizer}"
if git diff HEAD --name-only -- '*.cu' '*.cuh' | grep -q . ; then
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
CHANGED=$(git diff HEAD --name-only -- '*.cu' '*.cuh' '*.hpp' '*.cpp' '*.h' || true)
if [ -n "$CHANGED" ] && command -v clang-format >/dev/null; then
    # shellcheck disable=SC2086
    clang-format --dry-run --Werror $CHANGED
else
    echo "nothing to check / clang-format missing"
fi

echo "VERIFY: PASS"
