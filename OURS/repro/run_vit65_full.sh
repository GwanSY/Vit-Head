#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
binary="${BINARY:-$repo/build_vit65_compare/sci/bin/ckks_vit65_full}"
model="${MODEL:-$repo/fixtures/vit65_full}"
port="${PORT:-27400}"
gpu="${GPU:-2}"
output="${OUTPUT:-$repo/outputs/vit65_full_gpu${gpu}}"
blocks="${BLOCKS:-7}"
low_memory="${LOW_MEMORY:-0}"
min_free_mib="${MIN_FREE_MIB:-12000}"
binary_dir="$(cd "$(dirname "$binary")" && pwd -P)"
phantom_lib="$(cd "$binary_dir/../extern/phantom/lib" 2>/dev/null && pwd -P || true)"

fail() { echo "VIT65_FULL_RUN status=FAIL reason=$*" >&2; exit 1; }
[[ -x "$binary" ]] || fail "missing_binary path=$binary"
[[ -s "$model/manifest.json" ]] || fail "missing_manifest path=$model"
[[ -s "$phantom_lib/libPhantom.so" ]] || fail "missing_Phantom_runtime"
[[ "$gpu" =~ ^[0-9]+$ ]] || fail "GPU_must_be_a_nonnegative_index"
[[ "$blocks" =~ ^[1-7]$ ]] || fail "BLOCKS_must_be_1_to_7"
[[ "$low_memory" =~ ^[01]$ ]] || fail "LOW_MEMORY_must_be_0_or_1"
[[ "$min_free_mib" =~ ^[0-9]+$ ]] || fail "MIN_FREE_MIB_must_be_nonnegative"
if ! [[ "$port" =~ ^[0-9]+$ ]] || ((port < 1024 || port > 28000)); then
  fail "PORT_must_be_between_1024_and_28000"
fi
free_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
[[ "$free_mib" =~ ^[0-9]+$ ]] || fail "GPU_memory_query_failed gpu=$gpu"
if ((free_mib < min_free_mib)); then
  fail "GPU_insufficient_free_memory gpu=$gpu free_mib=$free_mib required_mib=$min_free_mib"
fi

mkdir -p "$output"
runtime_ld_path="$phantom_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
common=(model="$model" blocks="$blocks")
alice=(env CUDA_VISIBLE_DEVICES="$gpu" BLB_PREPROCESS_LAYERS="$blocks" VIT65_LOW_MEMORY="$low_memory" LD_LIBRARY_PATH="$runtime_ld_path" \
  "$binary" r=1 p="$port" "${common[@]}")
bob=(env CUDA_VISIBLE_DEVICES="$gpu" BLB_PREPROCESS_LAYERS="$blocks" VIT65_LOW_MEMORY="$low_memory" LD_LIBRARY_PATH="$runtime_ld_path" \
  "$binary" r=2 p="$port" "${common[@]}")

alice_pid=""
bob_pid=""
cleanup() {
  for task_pid in "$alice_pid" "$bob_pid"; do
    if [[ -n "$task_pid" ]] && kill -0 "$task_pid" 2>/dev/null; then
      kill "$task_pid" 2>/dev/null || true
      wait "$task_pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM
echo "VIT65_FULL_RUN status=START backend=phantom_gpu gpu=$gpu blocks=$blocks low_memory=$low_memory port=$port"
"${alice[@]}" >"$output/alice.log" 2>&1 &
alice_pid=$!
sleep 1
set +e
"${bob[@]}" >"$output/bob.log" 2>&1 &
bob_pid=$!
alice_status=0
bob_status=0
while kill -0 "$alice_pid" 2>/dev/null && kill -0 "$bob_pid" 2>/dev/null; do
  sleep 1
done
if ! kill -0 "$alice_pid" 2>/dev/null; then
  wait "$alice_pid"
  alice_status=$?
  if ((alice_status != 0)) && kill -0 "$bob_pid" 2>/dev/null; then
    kill "$bob_pid" 2>/dev/null || true
  fi
  wait "$bob_pid"
  bob_status=$?
else
  wait "$bob_pid"
  bob_status=$?
  if ((bob_status != 0)) && kill -0 "$alice_pid" 2>/dev/null; then
    kill "$alice_pid" 2>/dev/null || true
  fi
  wait "$alice_pid"
  alice_status=$?
fi
set -e
alice_pid=""
bob_pid=""
trap - EXIT INT TERM
if ((alice_status != 0 || bob_status != 0)); then
  tail -80 "$output/alice.log" >&2 || true
  tail -80 "$output/bob.log" >&2 || true
  fail "party_exit alice=$alice_status bob=$bob_status"
fi
grep -E 'VIT65_FULL(_SMOKE)? status=PASS' "$output/alice.log" \
  "$output/bob.log" >"$output/result.txt" || fail "missing_PASS_marker"
echo "VIT65_FULL_RUN status=PASS output=$output"
