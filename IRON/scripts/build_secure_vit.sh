#!/usr/bin/env bash
set -euo pipefail

IRON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$IRON_ROOT/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$IRON_ROOT/build/secure_vit}"
BLB_ROOT="${BLB_ROOT:-$WORKSPACE_ROOT/BLB}"
SCI_ROOT="${SCI_ROOT:-$BLB_ROOT/SCI}"
SCI_BUILD="${SCI_BUILD:-$BLB_ROOT/build_vit_b16/sci}"
BOLT_SCI_BUILD="${BOLT_SCI_BUILD:-$WORKSPACE_ROOT/BOLT/SCI/build_vit}"
JOBS="${JOBS:-4}"
TARGET="${TARGET:-IRON_VIT}"

case "$TARGET" in
  IRON_VIT|IRON_VIT_GPU) ;;
  *)
    echo "usage: TARGET=IRON_VIT|IRON_VIT_GPU $0" >&2
    exit 2
    ;;
esac

for command in cmake c++; do
  command -v "$command" >/dev/null || {
    echo "missing build command: $command" >&2
    exit 2
  }
done

# IRON keeps its compact ViT-B/16 adapter and BFV+MPC protocol in both
# variants.  The GPU variant only changes the HE evaluator to Phantom/CUDA;
# the MPC side, wire format, model ownership and fail-closed checks remain
# identical to the CPU/SEAL build.
if [[ "$TARGET" == IRON_VIT_GPU ]]; then
  BLB_GPU_BUILD="${BLB_GPU_BUILD:-$BLB_ROOT/build_vit_b16_gpu}"
  USE_HE_GPU=ON BLB_VIT_USE_GPU=ON BUILD_ROOT="$BLB_GPU_BUILD" JOBS="$JOBS" \
    bash "$BLB_ROOT/repro/build_vit_b16.sh" "$@"
  mkdir -p "$BUILD_DIR/bin"
  ln -sfn "$BLB_GPU_BUILD/adapter/bin/blb_vit_b16" \
    "$BUILD_DIR/bin/IRON_VIT_GPU"
  ln -sfn "$BLB_GPU_BUILD/adapter/bin/blb_vit_b16_validate" \
    "$BUILD_DIR/bin/IRON_VIT_GPU_VALIDATE"
  ln -sfn "$BLB_GPU_BUILD/adapter/bin/blb_vit_model_inspect" \
    "$BUILD_DIR/bin/IRON_VIT_GPU_MODEL_INSPECT"
  echo "built: $BUILD_DIR/bin/IRON_VIT_GPU"
  exit 0
fi

# Reuse the already validated SCI/BOLT core.  If a fresh checkout has not
# built it yet, use the existing BOLT build entry point; this does not alter
# any BERT or cryptographic implementation.
if [[ ! -f "$SCI_BUILD/lib/libSCI-FloatingPoint.a" ||
      ! -f "$SCI_BUILD/lib/libSCI-LinearHE.a" ]]; then
  BUILD_DIR="$SCI_BUILD" TARGET=BOLT_VIT JOBS="$JOBS" \
    bash "$WORKSPACE_ROOT/BOLT/scripts/build_vit.sh" "$@"
fi

SEAL_DIR="${SCI_SEAL_DIR:-$SCI_BUILD/extern/seal/lib/cmake/SEAL-4.1}"
[[ -f "$SEAL_DIR/SEALConfig.cmake" ]] || {
  echo "missing SEAL package: $SEAL_DIR" >&2
  echo "build the BOLT SCI tree before building IRON ViT" >&2
  exit 2
}

cmake -S "$BLB_ROOT/vit_b16" -B "$BUILD_DIR/adapter" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBLB_SCI_ROOT="$SCI_ROOT" \
  -DBLB_SCI_BUILD="$SCI_BUILD" \
  -DBLB_SEAL_DIR="$SEAL_DIR" \
  -DBLB_GMP_ROOT="${BLB_GMP_ROOT:-${CONDA_PREFIX:-}}"
cmake --build "$BUILD_DIR/adapter" \
  --target blb_vit_b16 blb_vit_b16_validate \
           blb_vit_model_inspect blb_vit_signed_conversion_test \
           blb_vit_compat_conversion_test blb_vit_compat_linear_test \
           blb_vit_compat_packing_test blb_vit_compat_attention_test \
           blb_vit_softmax_197_test --parallel "$JOBS"

# The original IRON smoke harness uses a deliberately tiny non-B/16 fixture.
# Keep its pre-existing generic ViT graph available under a distinct name;
# the standard 224/16 runner below selects the compact adapter explicitly.
if [[ ! -x "$BOLT_SCI_BUILD/bin/IRON_VIT" ]]; then
  BUILD_DIR="$BOLT_SCI_BUILD" TARGET=IRON_VIT JOBS="$JOBS" \
    bash "$WORKSPACE_ROOT/BOLT/scripts/build_vit.sh"
fi

# Keep IRON's public executable names and output directory stable while the
# implementation is supplied by the reviewed compact ViT adapter.  The
# production link never enables validation logits or intermediate observers.
mkdir -p "$BUILD_DIR/bin"
ln -sfn "$BUILD_DIR/adapter/bin/blb_vit_b16" \
  "$BUILD_DIR/bin/IRON_VIT"
ln -sfn "$BOLT_SCI_BUILD/bin/IRON_VIT" \
  "$BUILD_DIR/bin/IRON_VIT_GENERIC"
ln -sfn "$BUILD_DIR/adapter/bin/blb_vit_b16_validate" \
  "$BUILD_DIR/bin/IRON_VIT_VALIDATE"
ln -sfn "$BUILD_DIR/adapter/bin/blb_vit_model_inspect" \
  "$BUILD_DIR/bin/IRON_VIT_MODEL_INSPECT"
echo "built: $BUILD_DIR/bin/IRON_VIT"
