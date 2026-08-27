#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_DIR [REFERENCE_JSON]" >&2
  exit 2
fi
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
run_dir=$1
if [[ "$run_dir" != /* ]]; then run_dir="$root/$run_dir"; fi
reference=${2:-$run_dir/plain_fixed_secure_validation.json}
if [[ "$reference" != /* ]]; then reference="$root/$reference"; fi

python3 "$root/vit_b16/scripts/verify_vit_manifest.py" \
  --model "$root/checkpoints/vit_b16/bolt_model.txt" \
  --manifest "$root/checkpoints/vit_b16/manifest.json"
cmp "$run_dir/secure_logits_party1.txt" "$run_dir/secure_logits_party2.txt"
cmp "$run_dir/prediction_party1.txt" "$run_dir/prediction_party2.txt"
python3 "$root/vit_b16/scripts/vit_reference.py" \
  --model "$root/checkpoints/vit_b16/bolt_model.txt" \
  --input "$root/inputs/vit_b16/bob_image.txt" \
  --secure-prediction "$run_dir/prediction_party1.txt" \
  --secure-logits "$run_dir/secure_logits_party1.txt" \
  --output "$reference"
echo "BLB_VIT_VALIDATION status=PASS report=$reference"
