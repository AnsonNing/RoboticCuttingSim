#!/usr/bin/env bash
# Isolated pinned Isaac Lab3.0, with its own usd-exchange provider.
set -euo pipefail
RCS_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RCS_LAB_OPTIONS=()
if [[ -n "${RCS_LAB_PYTHON:-}" ]]; then
  RCS_LAB_OPTIONS+=(--lab-env "$(dirname -- "$(dirname -- "$RCS_LAB_PYTHON")")")
fi
if [[ -n "${RCS_LAB_SOURCE:-}" ]]; then
  RCS_LAB_OPTIONS+=(--lab-source "$RCS_LAB_SOURCE")
fi
exec bash "$RCS_REPO_ROOT/scripts/bootstrap.sh" --runtime isaac_lab "${RCS_LAB_OPTIONS[@]}" "$@"
