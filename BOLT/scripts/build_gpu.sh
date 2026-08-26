#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT/.." && pwd)"
TARGET="${TARGET:-BOLT_BERT_GPU}"
BUILD_DIR="${BUILD_DIR:-$ROOT/SCI/build_gpu}"
PHANTOM_DIR="${BOLT_PHANTOM_DIR:-$WORKSPACE_ROOT/BLB/build_vit_b16_gpu/sci/extern/phantom/lib/cmake/phantom}"
JOBS="${JOBS:-4}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-80}"

case "$TARGET" in
  BOLT_BERT_GPU|BOLT_VIT_GPU) ;;
  *)
    echo "usage: TARGET=BOLT_BERT_GPU|BOLT_VIT_GPU $0" >&2
    exit 2
    ;;
esac

if [[ ! -s "$PHANTOM_DIR/PhantomConfig.cmake" ]]; then
  echo "missing BLB Phantom package: $PHANTOM_DIR" >&2
  echo "build it first with: USE_HE_GPU=ON BLB_VIT_USE_GPU=ON BUILD_ROOT=$WORKSPACE_ROOT/BLB/build_vit_b16_gpu bash $WORKSPACE_ROOT/BLB/repro/build_vit_b16.sh" >&2
  exit 2
fi

CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-}"
if [[ -z "$CUDA_COMPILER" ]]; then
  CUDA_COMPILER="$(command -v nvcc || true)"
fi
if [[ -z "$CUDA_COMPILER" || ! -x "$CUDA_COMPILER" ]]; then
  echo "missing nvcc; set CMAKE_CUDA_COMPILER=/path/to/nvcc" >&2
  exit 2
fi
CUDA_COMPILER="$(readlink -f "$CUDA_COMPILER")"
CUDA_ROOT="${CUDAToolkit_ROOT:-$(cd "$(dirname "$CUDA_COMPILER")/.." && pwd -P)}"

echo "BOLT_BUILD target=$TARGET jobs=$JOBS backend=phantom_bfv gpu=0"
TARGET="$TARGET" BUILD_DIR="$BUILD_DIR" JOBS="$JOBS" \
  CUDA_HOME="$CUDA_ROOT" CUDAToolkit_ROOT="$CUDA_ROOT" \
  bash "$ROOT/scripts/build_vit.sh" \
  -DBOLT_USE_HE_GPU=ON \
  -DBOLT_PHANTOM_DIR="$PHANTOM_DIR" \
  -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  "$@"
