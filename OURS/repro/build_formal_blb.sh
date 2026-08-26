#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
build_root="${BUILD_ROOT:-$repo/build_blb_formal}"
sci_build="$build_root/sci"
cmake_bin="${CMAKE_BIN:-cmake}"
jobs="${JOBS:-$(nproc)}"
use_he_gpu="${USE_HE_GPU:-ON}"
cuda_arch="${CUDA_ARCH:-70}"
cuda_compiler="${CUDA_COMPILER:-}"
cuda_root="${CUDAToolkit_ROOT:-${CUDA_ROOT:-}}"
local_gmp_root="$repo/build_deps/gmp/usr"

case "$build_root" in "$repo"/*) ;; *)
  echo "BLB_FORMAL_BUILD status=FAIL reason=BUILD_ROOT_outside_BLB" >&2
  exit 2
esac

gmp_args=()
if [[ -f "$local_gmp_root/include/x86_64-linux-gnu/gmp.h" ]]; then
  gmp_args+=(
    "-DGMP_INCLUDE_DIR=$local_gmp_root/include/x86_64-linux-gnu"
    "-DGMP_LIBRARIES=$local_gmp_root/lib/x86_64-linux-gnu/libgmp.a"
    "-DGMPXX_LIBRARIES=$local_gmp_root/lib/x86_64-linux-gnu/libgmpxx.a"
  )
fi

gpu_args=()
if [[ "$use_he_gpu" == "ON" ]]; then
  if [[ -z "$cuda_compiler" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
      cuda_compiler="$(readlink -f "$(command -v nvcc)")"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
      cuda_compiler=/usr/local/cuda/bin/nvcc
    else
      cuda_compiler="$(find /usr/local -maxdepth 4 -type f -path '*/bin/nvcc' -perm -111 -print 2>/dev/null | sort -V | tail -n 1 || true)"
    fi
  fi
  if [[ -n "$cuda_compiler" ]]; then
    cuda_compiler="$(readlink -f "$cuda_compiler")"
  fi
  if [[ -z "$cuda_compiler" || ! -x "$cuda_compiler" ]]; then
    echo "BLB_FORMAL_BUILD status=BLOCKED reason=nvcc_missing" >&2
    echo "Set CUDA_COMPILER=/path/to/nvcc or CUDA_ROOT=/path/to/cuda-toolkit." >&2
    exit 2
  fi
  if [[ -z "$cuda_root" ]]; then
    cuda_root="$(cd "$(dirname "$cuda_compiler")/.." && pwd -P)"
  fi
  if [[ ! -f "$cuda_root/include/cuda.h" ]]; then
    echo "BLB_FORMAL_BUILD status=BLOCKED reason=cuda_headers_missing root=$cuda_root" >&2
    exit 2
  fi
  export CUDA_HOME="$cuda_root"
  export CUDA_PATH="$cuda_root"
  export CUDAToolkit_ROOT="$cuda_root"
  export PATH="$cuda_root/bin:$PATH"
  gpu_args+=(
    "-DCMAKE_CUDA_COMPILER=$cuda_compiler"
    "-DCUDAToolkit_ROOT=$cuda_root"
    "-DCUDA_TOOLKIT_ROOT_DIR=$cuda_root"
    "-DCMAKE_CUDA_FLAGS=-arch=sm_$cuda_arch"
  )
fi

"$cmake_bin" -S "$repo/SCI" -B "$sci_build" \
  -DUSE_HE_GPU="$use_he_gpu" \
  -DNO_REVEAL_OUTPUT=OFF \
  -DBUILD_TESTS=ON \
  -DBUILD_NETWORKS=OFF \
  -DSCI_EIGEN_INCLUDE_DIR="$repo/SCI/extern/eigen" \
  -DCMAKE_BUILD_TYPE=Release \
  "${gpu_args[@]}" \
  "${gmp_args[@]}"

"$cmake_bin" --build "$sci_build" --target ckks_bert_large_main \
  --parallel "$jobs"

binary="$sci_build/bin/ckks_bert_large_main"
if [[ ! -x "$binary" ]]; then
  echo "BLB_FORMAL_BUILD status=FAIL reason=missing_binary path=$binary" >&2
  exit 1
fi

# Record the libc used for this binary. The official artifact image has an
# older glibc than many hosts, so a host-built binary must never be silently
# reused inside the container.
build_glibc="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
if [[ -z "$build_glibc" ]]; then
  echo "BLB_FORMAL_BUILD status=FAIL reason=glibc_version_unavailable" >&2
  exit 1
fi
printf 'schema=blb.build-environment.v1\nglibc=%s\n' "$build_glibc" \
  > "$sci_build/.blb_build_environment"

echo "BLB_FORMAL_BUILD status=PASS formal_ready=no he_gpu=$use_he_gpu cuda_arch=$cuda_arch glibc=$build_glibc binary=$binary"
echo "BLB_FORMAL_BUILD note=build_success_is_not_reproduction_success"
