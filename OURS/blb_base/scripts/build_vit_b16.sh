#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
jobs=${JOBS:-64}
cuda_root=${CUDA_ROOT:-${CUDAToolkit_ROOT:-/usr/local/cuda-12.3}}
cmake_bin=${CMAKE_BIN:-cmake}
build_dir=${OURS_VIT_BUILD_DIR:-$root/build/gpu}
eigen_build=${OURS_VIT_EIGEN_BUILD_DIR:-$root/build/eigen-config}

# SCI's Phantom backend uses FindCUDAToolkit, which requires CMake >= 3.17.
# The host image still exposes CMake 3.16 as /usr/bin/cmake, while the
# successful BLB build was configured with a newer pip CMake.  Keep the
# override explicit and use the known local fallback only when the default is
# too old.
cmake_version=$($cmake_bin --version | awk 'NR == 1 {print $3}')
cmake_major=${cmake_version%%.*}
cmake_minor=${cmake_version#*.}
cmake_minor=${cmake_minor%%.*}
if (( cmake_major < 3 || (cmake_major == 3 && cmake_minor < 17) )); then
  local_cmake=/home/dm/anaconda3/envs/compare_heal/bin/cmake
  if [[ -x "$local_cmake" ]]; then
    cmake_bin=$local_cmake
  else
    echo "CMake >= 3.17 is required; set CMAKE_BIN to a compatible binary" >&2
    exit 2
  fi
fi

if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
  echo "CUDA toolkit not found at $cuda_root" >&2
  exit 2
fi

# Configure the vendored Eigen source in the OURS build tree.  This avoids the
# donor container's stale /usr/local Eigen package, whose exported include
# path points at /workspace/Pencil.
"$cmake_bin" -S "$root/gpu/SCI/extern/eigen" -B "$eigen_build" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DBUILD_TESTING=OFF

# One canonical GPU-enabled build provides both audited runtime selections:
# SEAL/HOST for CPU and Phantom/DEVICE for GPU. VOLE is enabled by the
# unmodified SCI top-level build definition.
"$cmake_bin" -S "$root/gpu/SCI" -B "$build_dir" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DBUILD_NETWORKS=OFF \
  -DNO_REVEAL_OUTPUT=OFF \
  -DUSE_HE_GPU=ON \
  -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
  -DCUDAToolkit_ROOT="$cuda_root" \
  -DCMAKE_CUDA_FLAGS=-arch=sm_80 \
  -DEigen3_DIR="$eigen_build"

# Phantom is installed into the build tree when SCI is first configured.
# Rebuild that vendored dependency here so ViT-only additive encoder APIs do
# not leave a stale installed header/library on incremental builds. Existing
# Phantom encode callers (including BERT) keep their original checked path.
if [[ -d "$build_dir/thirdparty/phantom-build" ]]; then
  "$cmake_bin" --build "$build_dir/thirdparty/phantom-build" \
    --target Phantom --parallel "$jobs"
  "$cmake_bin" --install "$build_dir/thirdparty/phantom-build"
fi

"$cmake_bin" --build "$build_dir" \
  --target ckks_vit_b16_main ours_vit_b16_main ours_vit_hetero_plan_test \
           ours_vit_a0_conversion_selftest ours_vit_conversion_test \
  --parallel "$jobs"

test -x "$build_dir/bin/ckks_vit_b16_main"
test -x "$build_dir/bin/ours_vit_b16_main"
test -x "$build_dir/bin/ours_vit_hetero_plan_test"
test -x "$build_dir/bin/ours_vit_a0_conversion_selftest"
test -x "$build_dir/bin/ours_vit_conversion_test"
echo "OURS_VIT_BUILD status=PASS targets=A0+OURS+PLAN_TEST+CONVERSION_TEST backend=SEAL_HOST+Phantom_DEVICE ot=VOLE threads=$jobs cmake=$cmake_bin build_dir=$build_dir"
