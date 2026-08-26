#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/SCI/build_vit}"
DEPS_DIR="$BUILD_DIR/deps"
SEAL_SOURCE="$DEPS_DIR/seal-src"
SEAL_BUILD="$DEPS_DIR/seal-build"
SEAL_INSTALL="$DEPS_DIR/seal-install"
EIGEN_BUILD="$DEPS_DIR/eigen-build"
EIGEN_INSTALL="$DEPS_DIR/eigen-install"
JOBS="${JOBS:-4}"
TARGET="${TARGET:-BOLT_VIT}"
EXTRA_CMAKE_ARGS=("$@")

# SCI's bundled FindGMP.cmake does not search Conda prefixes by itself.
# Prefer explicit caller values; otherwise use the active Conda GMP when it
# is complete.  System libgmp-dev installations continue to be auto-found.
has_gmp_override=0
for argument in "${EXTRA_CMAKE_ARGS[@]}"; do
  if [[ "$argument" == -DGMP_* ]]; then
    has_gmp_override=1
    break
  fi
done
if [[ "$has_gmp_override" == 0 && -n "${CONDA_PREFIX:-}" && \
      -f "$CONDA_PREFIX/include/gmp.h" && \
      -f "$CONDA_PREFIX/lib/libgmp.so" && \
      -f "$CONDA_PREFIX/lib/libgmpxx.so" ]]; then
  EXTRA_CMAKE_ARGS+=(
    "-DGMP_INCLUDE_DIR=$CONDA_PREFIX/include"
    "-DGMP_LIBRARIES=$CONDA_PREFIX/lib/libgmp.so"
    "-DGMPXX_LIBRARIES=$CONDA_PREFIX/lib/libgmpxx.so"
  )
fi

for command in cmake c++ make; do
  command -v "$command" >/dev/null || {
    echo "missing build command: $command" >&2
    exit 2
  }
done

test -f "$ROOT/SCI/extern/SEAL/CMakeLists.txt" || {
  echo "missing SCI/extern/SEAL; run: git submodule update --init --recursive" >&2
  exit 2
}
test -f "$ROOT/SCI/extern/eigen/CMakeLists.txt" || {
  echo "missing SCI/extern/eigen; run: git submodule update --init --recursive" >&2
  exit 2
}

cmake -E make_directory "$DEPS_DIR"

if [[ ! -f "$SEAL_INSTALL/lib/cmake/SEAL-4.1/SEALConfig.cmake" ]]; then
  if [[ ! -f "$SEAL_SOURCE/CMakeLists.txt" ]]; then
    cmake -E copy_directory "$ROOT/SCI/extern/SEAL" "$SEAL_SOURCE"
  fi
  # Some distributed source bundles contain this generated in-source header.
  # The isolated out-of-source configuration below must create its own copy.
  cmake -E remove -f "$SEAL_SOURCE/native/src/seal/util/config.h"
  cmake -S "$SEAL_SOURCE" -B "$SEAL_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$SEAL_INSTALL" \
    -DSEAL_BUILD_DEPS=OFF \
    -DSEAL_BUILD_EXAMPLES=OFF \
    -DSEAL_BUILD_TESTS=OFF \
    -DSEAL_USE_MSGSL=OFF \
    -DSEAL_USE_ZLIB=OFF \
    -DSEAL_USE_ZSTD=OFF
  cmake --build "$SEAL_BUILD" --target install --parallel "$JOBS"
fi

if [[ ! -f "$EIGEN_INSTALL/share/eigen3/cmake/Eigen3Config.cmake" ]]; then
  cmake -S "$ROOT/SCI/extern/eigen" -B "$EIGEN_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$EIGEN_INSTALL" \
    -DBUILD_TESTING=OFF
  cmake --build "$EIGEN_BUILD" --target install --parallel "$JOBS"
fi

cmake -S "$ROOT/SCI" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DBUILD_NETWORKS=OFF \
  -DSCI_SEAL_ROOT="$SEAL_INSTALL" \
  -DEigen3_DIR="$EIGEN_INSTALL/share/eigen3/cmake" \
  "${EXTRA_CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --target "$TARGET" --parallel "$JOBS"

echo "built: $BUILD_DIR/bin/$TARGET"
