#!/usr/bin/env bash
# Isolated, pinned Isaac Lab 3.0 kit-less research runtime; no training packages.
set -euo pipefail
cd "$(dirname "$0")/.."
RCS_LAB_SOURCE="${RCS_LAB_SOURCE:-$PWD/.deps/IsaacLab}"
RCS_LAB_PYTHON="${RCS_LAB_PYTHON:-$PWD/.deps/isaaclab-venv/bin/python}"
RCS_LAB_SHA=e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4
test "$(git -C "$RCS_LAB_SOURCE" rev-parse HEAD)" = "$RCS_LAB_SHA"
"$RCS_LAB_PYTHON" -m pip install --no-cache-dir --timeout 120 --retries 8 \
  torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
# usd-exchange is the single USD provider required by this Lab release.
# Do not install usd-core into this environment, or newton[importers].
"$RCS_LAB_PYTHON" -m pip install --no-cache-dir --timeout 120 --retries 8 \
  'newton[sim]==1.6.1' warp-lang==1.17.0 usd-exchange==3.0.1 \
  newton-usd-schemas==0.6.0 mujoco==3.12.0 mujoco-warp==3.12.0 \
  'numpy>=2' scipy PyYAML lazy_loader prettytable psutil packaging \
  'gymnasium>=1.2' 'pyglet>=2.1.6,<3' trimesh matplotlib Pillow \
  requests 'urllib3>=2.8' filelock omniverseclient==2.74.0 \
  rich tqdm typing-extensions hydra-core h5py imageio imageio-ffmpeg
for RCS_LAB_PACKAGE in isaaclab isaaclab_newton isaaclab_ov isaaclab_physx isaaclab_visualizers isaaclab_assets isaaclab_ppisp isaaclab_contrib; do
  "$RCS_LAB_PYTHON" -m pip install --no-deps -e "$RCS_LAB_SOURCE/source/$RCS_LAB_PACKAGE"
done
"$RCS_LAB_PYTHON" -m pip install --no-deps -e .
"$RCS_LAB_PYTHON" -m pip freeze
