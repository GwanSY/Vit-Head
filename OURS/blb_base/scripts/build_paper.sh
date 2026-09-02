#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [cpu|gpu|all]" >&2
  exit 2
fi

backend=${1:-all}
if [[ "$backend" != cpu && "$backend" != gpu && "$backend" != all ]]; then
  echo "backend must be cpu, gpu, or all" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
jobs=${JOBS:-64}
cuda_root=${CUDA_ROOT:-${CUDAToolkit_ROOT:-/usr/local/cuda-12.3}}

build_cpu() {
  cmake -S "$root/cpu/SCI" -B "$root/build/cpu" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DBUILD_NETWORKS=OFF \
    -DNO_REVEAL_OUTPUT=ON \
    -DSEAL_DIR=/home/CKKS-MPC/SCI/extern/SEAL/build/cmake \
    -DEigen3_DIR=/usr/local/share/eigen3/cmake
  cmake --build "$root/build/cpu" \
    --target ckks_bert_large_main --parallel "$jobs"
}

build_gpu() {
  if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
    echo "CUDA toolkit not found at $cuda_root" >&2
    exit 2
  fi
  cmake -S "$root/gpu/SCI" -B "$root/build/gpu" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DBUILD_NETWORKS=OFF \
    -DNO_REVEAL_OUTPUT=ON \
    -DUSE_HE_GPU=ON \
    -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
    -DCUDAToolkit_ROOT="$cuda_root" \
    -DCMAKE_CUDA_FLAGS=-arch=sm_80 \
    -DEigen3_DIR=/usr/local/share/eigen3/cmake
  cmake --build "$root/build/gpu" \
    --target ckks_bert_large_main --parallel "$jobs"
}

case "$backend" in
  cpu) build_cpu ;;
  gpu) build_gpu ;;
  all)
    build_cpu
    build_gpu
    ;;
esac
