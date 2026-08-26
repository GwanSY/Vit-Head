#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set to the physical GPU used for this run" >&2
  echo "Example: CUDA_VISIBLE_DEVICES=1 $0 <image> <output_dir> <port>" >&2
  exit 2
fi
echo "BOLT_VIT_GPU_DEVICE_SELECTION=CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
export BINARY="${BINARY:-$ROOT/SCI/build_gpu/bin/BOLT_VIT_GPU}"
OUTPUT_DIR="${2:-$ROOT/outputs/vit_b16_e2e}"

"$ROOT/scripts/run_prepared_vit_b16.sh" "$@"
grep -q '^BOLT_HE_BACKEND backend=phantom_bfv device=0 protocol=BOLT$' \
  "$OUTPUT_DIR/alice.log"
echo "BOLT_VIT_GPU_BACKEND=PASS logical_device=0 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
