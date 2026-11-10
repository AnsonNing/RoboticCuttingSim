#!/usr/bin/env bash
# Pytest for the Torch venv without installing into it (plan decision D-09).
# Copies the pure-Python pytest packages of the project .venv (same Python 3.12)
# into an overlay directory used via PYTHONPATH. .deps venvs stay untouched.
#   bash scripts/rl_ready/make_pytest_overlay.sh [overlay_dir]
#   PYTHONPATH=<overlay_dir>:src .deps/isaacsim-venv/bin/python -m pytest -p no:cacheprovider tests/...
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OV=${1:-/home/ningan/rcs_rl_data/pytest_overlay_py312}
SP=$("$ROOT/.venv/bin/python" -c "import pytest,os;print(os.path.dirname(os.path.dirname(pytest.__file__)))")
mkdir -p "$OV"
for name in _pytest pytest iniconfig pluggy pygments py.py; do cp -r "$SP/$name" "$OV/"; done
cp -r "$SP"/pytest-*.dist-info "$SP"/iniconfig-*.dist-info "$SP"/pluggy-*.dist-info "$SP"/pygments-*.dist-info "$OV/"
PYTHONPATH="$OV" "$ROOT/.deps/isaacsim-venv/bin/python" -m pytest --version
