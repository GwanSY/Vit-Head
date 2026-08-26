#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
build_root="${BUILD_ROOT:-$repo/build_vit65_a0}"
sci_build="$build_root/sci"
cmake_bin="${CMAKE_BIN:-cmake}"
jobs="${JOBS:-$(nproc)}"
cuda_arch="${CUDA_ARCH:-80}"
cuda_compiler="${CUDA_COMPILER:-/usr/local/cuda-12.3/bin/nvcc}"
cuda_root="${CUDAToolkit_ROOT:-$(cd "$(dirname "$cuda_compiler")/.." && pwd -P)}"

case "$build_root" in "$repo"/*) ;; *)
  echo "VIT65_A0_BUILD status=FAIL reason=BUILD_ROOT_outside_OURS" >&2
  exit 2
esac
[[ -x "$cuda_compiler" ]] || {
  echo "VIT65_A0_BUILD status=BLOCKED reason=nvcc_missing path=$cuda_compiler" >&2
  exit 2
}
[[ -f "$cuda_root/include/cuda.h" ]] || {
  echo "VIT65_A0_BUILD status=BLOCKED reason=cuda_headers_missing root=$cuda_root" >&2
  exit 2
}

"$cmake_bin" -S "$repo/SCI" -B "$sci_build" \
  -DUSE_HE_GPU=ON \
  -DNO_REVEAL_OUTPUT=OFF \
  -DBUILD_TESTS=ON \
  -DBUILD_NETWORKS=OFF \
  -DSCI_EIGEN_INCLUDE_DIR="$repo/SCI/extern/eigen" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$cuda_compiler" \
  -DCUDAToolkit_ROOT="$cuda_root" \
  -DCUDA_TOOLKIT_ROOT_DIR="$cuda_root" \
  -DCMAKE_CUDA_FLAGS="-arch=sm_$cuda_arch"
"$cmake_bin" --build "$sci_build" --target ckks_vit65_a0 ckks_vit65_hetero \
  --parallel "$jobs"

binary="$sci_build/bin/ckks_vit65_a0"
hetero_binary="$sci_build/bin/ckks_vit65_hetero"
[[ -x "$binary" ]] || {
  echo "VIT65_A0_BUILD status=FAIL reason=missing_binary path=$binary" >&2
  exit 1
}
[[ -x "$hetero_binary" ]] || {
  echo "VIT65_A0_BUILD status=FAIL reason=missing_hetero_binary path=$hetero_binary" >&2
  exit 1
}
echo "VIT65_A0_BUILD status=PASS backend=phantom_gpu cuda_arch=$cuda_arch binary=$binary hetero_binary=$hetero_binary"
