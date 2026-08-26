#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  echo "BOLT GPU experiments are restricted to physical GPU 0; unset CUDA_VISIBLE_DEVICES or set it to 0" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=0
export BINARY="${BINARY:-$ROOT/SCI/build_gpu/bin/BOLT_BERT_GPU}"
export OUTPUT="${OUTPUT:-$ROOT/outputs/bert_gpu_e2e}"
export BOLT_GPU_RESIDENT="${BOLT_GPU_RESIDENT:-0}"

if [[ "$BOLT_GPU_RESIDENT" == "0" ]]; then
  BOLT_GPU_RESIDENCY_NAME="off"
else
  BOLT_GPU_RESIDENCY_NAME="on"
fi
echo "BOLT_BERT_GPU_RESIDENCY=$BOLT_GPU_RESIDENCY_NAME"

"$ROOT/scripts/run_bert_e2e.sh" "$@"
grep -q "^BOLT_HE_BACKEND backend=phantom_bfv device=0 protocol=BOLT residency=$BOLT_GPU_RESIDENCY_NAME$" \
  "$OUTPUT/alice.log"
echo "BOLT_BERT_GPU_BACKEND=PASS device=0"
