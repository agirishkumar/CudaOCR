#!/usr/bin/env bash
# Lists existing files matching the given pathspecs that differ from the branch's
# base: committed-on-branch + staged + unstaged changes + untracked files.
# Base = merge-base of HEAD and $CUDAOCR_BASE_REF (default origin/main); falls
# back to HEAD when that ref doesn't exist. Deleted files are omitted.
# Usage: changed_files.sh '*.cu' '*.hpp' ...
set -euo pipefail
base_ref="${CUDAOCR_BASE_REF:-origin/main}"
if git rev-parse -q --verify "$base_ref^{commit}" >/dev/null; then
    base=$(git merge-base HEAD "$base_ref")
else
    base=HEAD
fi
{
    git diff --name-only --diff-filter=d "$base" -- "$@"
    git ls-files --others --exclude-standard -- "$@"
} | sort -u
