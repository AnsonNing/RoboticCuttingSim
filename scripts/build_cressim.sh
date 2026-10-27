#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source_dir="$PWD/.deps/CRESSim-MPM"
build_dir="$PWD/.deps/cressim-build"
for patch_file in "$PWD/scripts/patches/cressim_si_mls.patch" "$PWD/scripts/patches/cressim_math_layout.patch"; do
  if git -C "$source_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "CRESSim patch already applied: $(basename "$patch_file")"
  elif git -C "$source_dir" apply --check "$patch_file"; then
    git -C "$source_dir" apply "$patch_file"
  else
    echo "CRESSim source differs from tested patch context: $patch_file" >&2
    exit 1
  fi
done
mkdir -p "$build_dir"
g++ -std=c++17 -O2 scripts/cressim_low_level_verify.cpp \
  -I"$source_dir/include/math" -I"$source_dir/include/core" -I"$source_dir/include/utils" \
  -I"$source_dir/include/engine" -I"$source_dir/include/geometry_utils" \
  -I"$source_dir/include/solver" -I"$source_dir/src/solver" \
  -o "$build_dir/cressim_low_level_verify"
"$build_dir/cressim_low_level_verify"
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
