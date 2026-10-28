#!/usr/bin/env bash
set -euo pipefail
RCS_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$RCS_REPO_ROOT/scripts/bootstrap.sh" --runtime isaac_sim "$@"
