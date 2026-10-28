#!/usr/bin/env bash
# Project-local installation only. No system package or driver changes.
set -euo pipefail
RCS_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONPATH PYTHONHOME
exec "${RCS_BOOTSTRAP_PYTHON:-python3}" "$RCS_REPO_ROOT/scripts/bootstrap_runtime.py" "$@"
