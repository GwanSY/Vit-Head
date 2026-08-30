#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
jobs=${JOBS:-64}
cuda_root=${CUDA_ROOT:-${CUDAToolkit_ROOT:-/usr/local/cuda-12.3}}

if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
  echo "CUDA toolkit not found at $cuda_root" >&2
  exit 2
fi

# One canonical GPU-enabled build provides both audited runtime selections:
# SEAL/HOST for CPU and Phantom/DEVICE for GPU. VOLE is enabled by the
# unmodified SCI top-level build definition.
cmake -S "$root/gpu/SCI" -B "$root/build/gpu" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DBUILD_NETWORKS=OFF \
  -DNO_REVEAL_OUTPUT=OFF \
  -DUSE_HE_GPU=ON \
  -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
  -DCUDAToolkit_ROOT="$cuda_root" \
  -DCMAKE_CUDA_FLAGS=-arch=sm_80 \
  -DEigen3_DIR=/usr/local/share/eigen3/cmake

# Phantom is installed into the build tree when SCI is first configured.
# Rebuild that vendored dependency here so ViT-only additive encoder APIs do
# not leave a stale installed header/library on incremental builds. Existing
# Phantom encode callers (including BERT) keep their original checked path.
if [[ -d "$root/build/gpu/thirdparty/phantom-build" ]]; then
  cmake --build "$root/build/gpu/thirdparty/phantom-build" \
    --target Phantom --parallel "$jobs"
  cmake --install "$root/build/gpu/thirdparty/phantom-build"
fi

cmake --build "$root/build/gpu" \
  --target ckks_vit_b16_main --parallel "$jobs"

test -x "$root/build/gpu/bin/ckks_vit_b16_main"
echo "BLB_VIT_BUILD status=PASS backend=SEAL_HOST+Phantom_DEVICE ot=VOLE threads=$jobs"
