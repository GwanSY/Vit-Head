#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${BUILD_ROOT:-$repo/build_vit_smoke_e2e}"
sci_build="$build_root/sci"
adapter_build="$build_root/adapter"
cmake_bin="${CMAKE_BIN:-cmake}"
jobs="${JOBS:-$(nproc)}"
local_gmp_root="$repo/build_deps/gmp/usr"
if [[ -z "${BLB_GMP_ROOT:-}" && -f "$local_gmp_root/include/x86_64-linux-gnu/gmp.h" ]]; then
  BLB_GMP_ROOT="$local_gmp_root"
fi

if [[ "$build_root" != "$repo"/* ]]; then
  echo "BUILD_RESULT status=FAIL reason=BUILD_ROOT_must_be_inside_repo" >&2
  exit 2
fi
if ! command -v "$cmake_bin" >/dev/null 2>&1; then
  echo "BUILD_RESULT status=FAIL reason=cmake_not_found value=$cmake_bin" >&2
  exit 2
fi

sci_args=(
  -S "$repo/SCI" \
  -B "$sci_build" \
  -DUSE_HE_GPU=OFF \
  -DNO_REVEAL_OUTPUT=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_NETWORKS=OFF \
  -DSCI_EIGEN_INCLUDE_DIR="$repo/SCI/extern/eigen" \
  -DCMAKE_INSTALL_PREFIX="$sci_build/install" \
  -DCMAKE_BUILD_TYPE=Release
)
if [[ -n "${BLB_GMP_ROOT:-}" ]]; then
  gmp_include="$BLB_GMP_ROOT/include"
  if [[ -f "$BLB_GMP_ROOT/include/x86_64-linux-gnu/gmp.h" ]]; then
    gmp_include="$BLB_GMP_ROOT/include/x86_64-linux-gnu"
  fi
  gmp_lib_dir="$BLB_GMP_ROOT/lib"
  if [[ -d "$BLB_GMP_ROOT/lib/x86_64-linux-gnu" ]]; then
    gmp_lib_dir="$BLB_GMP_ROOT/lib/x86_64-linux-gnu"
  fi
  sci_args+=(
    "-DGMP_INCLUDE_DIR=$gmp_include"
    "-DGMP_LIBRARIES=$gmp_lib_dir/libgmp.a"
    "-DGMPXX_LIBRARIES=$gmp_lib_dir/libgmpxx.a"
  )
fi
"$cmake_bin" "${sci_args[@]}"

"$cmake_bin" --build "$sci_build" \
  --target SCI-FloatingPoint SCI-Math SCI-BuildingBlocks SCI-LinearOT \
  --parallel "$jobs"

adapter_args=(
  -S "$repo/vit_adapter"
  -B "$adapter_build"
  -DBLB_SCI_ROOT="$repo/SCI"
  -DBLB_SCI_BUILD="$sci_build"
  -DBLB_SEAL_DIR="$sci_build/extern/seal/lib/cmake/SEAL-4.1"
  -DCMAKE_BUILD_TYPE=Release
)
if [[ -n "${BLB_GMP_ROOT:-}" ]]; then
  adapter_args+=("-DBLB_GMP_ROOT=$BLB_GMP_ROOT")
fi
"$cmake_bin" "${adapter_args[@]}"
"$cmake_bin" --build "$adapter_build" --target vit_smoke_e2e \
  --parallel "$jobs"

binary="$adapter_build/bin/vit_smoke_e2e"
if [[ ! -x "$binary" ]]; then
  echo "BUILD_RESULT status=FAIL missing_binary=$binary" >&2
  exit 1
fi
echo "VIT_SMOKE_BUILD status=PASS formal_result=no binary=$binary build_root=$build_root"
