#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source_dir="$PWD/.deps/CRESSim-MPM"
build_dir="$PWD/.deps/cressim-build"
cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_STATIC=OFF -DENABLE_EXAMPLES=OFF -DENABLE_TESTS=OFF -DENABLE_DEBUG_LOGGER=ON
cmake --build "$build_dir" -j 4
g++ -std=c++17 -O2 scripts/cressim_probe.cpp \
  -I/usr/local/cuda-12.8/include \
  -I"$source_dir/include/engine" -I"$source_dir/include/solver" \
  -I"$source_dir/include/core" -I"$source_dir/include/math" \
  -I"$source_dir/include/utils" -I"$source_dir/include/geometry_utils" \
  -L"$build_dir/bin" -Wl,-rpath,"$build_dir/bin" -lcrmpm_engine \
  -o "$build_dir/cressim_probe"
