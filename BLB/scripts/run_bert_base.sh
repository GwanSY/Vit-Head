#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 cpu|gpu OUTPUT_DIR PORT [ALICE_GPU_ID [BOB_GPU_ID]]" >&2
  exit 2
fi

backend=$1
output_dir=$2
port=$3
alice_gpu_id=${4:-}
bob_gpu_id=${5:-$alice_gpu_id}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

if [[ "$output_dir" != /* ]]; then
  output_dir="$root/$output_dir"
fi

case "$backend" in
  cpu)
    binary="$root/build/cpu/bin/ckks_bert_large_main"
    exec "$root/scripts/run_pair_once.sh" \
      cpu "$binary" "$output_dir" "$port"
    ;;
  gpu)
    if [[ -z "$alice_gpu_id" ]]; then
      echo "GPU IDs are required for the gpu backend" >&2
      exit 2
    fi
    binary="$root/build/gpu/bin/ckks_bert_large_main"
    exec "$root/scripts/run_pair_once.sh" \
      gpu "$binary" "$output_dir" "$port" \
      "$alice_gpu_id" "$bob_gpu_id"
    ;;
  *)
    echo "backend must be cpu or gpu" >&2
    exit 2
    ;;
esac

