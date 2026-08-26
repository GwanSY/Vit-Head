#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COUNT="${COUNT:-10}"
BASE_SEED="${BASE_SEED:-20260810}"
BASE_PORT="${BASE_PORT:-48000}"
SUMMARY="${SUMMARY:-$ROOT/outputs/vit_seed_regression.csv}"

mkdir -p "$(dirname "$SUMMARY")"
echo "seed,layers,expected_prediction,secure_prediction,status" >"$SUMMARY"
for ((case_id=0; case_id<COUNT; ++case_id)); do
  seed=$((BASE_SEED + case_id))
  fixture="$ROOT/fixtures/vit_seed_$seed"
  output="$ROOT/outputs/vit_seed_$seed"
  python3 "$ROOT/scripts/generate_vit_fixture.py" \
    --out-dir "$fixture" --seed "$seed" --layers 1
  # IOPack reserves p, p+50 and p+100. Space cases apart so a prior socket in
  # TIME_WAIT cannot overlap the next case's auxiliary channels.
  FIXTURE="$fixture" OUTPUT="$output" PORT=$((BASE_PORT + case_id * 200)) \
    CHECK_EXPECTED=0 \
    bash "$ROOT/scripts/run_vit_e2e.sh"
  expected="$(awk '{print $2}' "$fixture/expected_prediction.txt")"
  actual="$(awk '{print $2}' "$output/prediction.txt")"
  status=PASS
  [[ "$expected" == "$actual" ]] || status=FAIL
  echo "$seed,1,$expected,$actual,$status" >>"$SUMMARY"
done

if grep -q ',FAIL$' "$SUMMARY"; then
  echo "BOLT_VIT_TEN_SEED_FINAL_CLASS_MATCH=FAIL"
  exit 2
fi
echo "BOLT_VIT_TEN_SEED_FINAL_CLASS_MATCH=PASS cases=$COUNT"
