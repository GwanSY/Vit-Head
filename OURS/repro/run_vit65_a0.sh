#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
protocol="${PROTOCOL:-a0}"
case "$protocol" in
  a0)
    default_binary="$repo/build_vit65_a0/sci/bin/ckks_vit65_a0"
    metric_prefix="A0"
    ;;
  hetero)
    default_binary="$repo/build_vit65_compare/sci/bin/ckks_vit65_hetero"
    metric_prefix="HETERO"
    ;;
  *)
    echo "VIT65_RUN status=FAIL reason=PROTOCOL_must_be_a0_or_hetero" >&2
    exit 1
    ;;
esac
binary="${BINARY:-$default_binary}"
model="${MODEL:-$repo/fixtures/vit65_a0}"
output="${OUTPUT:-$repo/outputs/vit65_a0/performance}"
port="${PORT:-20000}"
alice_gpu="${ALICE_GPU:-0}"
bob_gpu="${BOB_GPU:-1}"
trace="${TRACE:-0}"
python_bin="${PYTHON:-python3}"
binary_dir="$(cd "$(dirname "$binary")" && pwd -P)"
phantom_lib="$(cd "$binary_dir/../extern/phantom/lib" 2>/dev/null && pwd -P || true)"

fail() { echo "VIT65_A0_RUN status=FAIL reason=$*" >&2; exit 1; }
[[ -x "$binary" ]] || fail "missing_binary path=$binary"
[[ -n "$phantom_lib" && -s "$phantom_lib/libPhantom.so" ]] || \
  fail "missing_phantom_runtime binary=$binary expected=$binary_dir/../extern/phantom/lib"
[[ -s "$model/manifest.json" ]] || fail "missing_fixture_manifest path=$model"
[[ "$trace" == "0" || "$trace" == "1" ]] || fail "TRACE_must_be_0_or_1"
# The donor creates ports p+[0,31], p+[50,81], and p+[100,131].  Keeping the
# base below Linux's usual ephemeral range prevents source-port collisions.
if ! [[ "$port" =~ ^[0-9]+$ ]] || ((port < 1024 || port > 28000)); then
  fail "PORT_must_be_between_1024_and_28000"
fi
mkdir -p "$output"

common=(model="$model" backend=gpu)
if [[ "$trace" == "1" ]]; then
  mkdir -p "$output/trace"
  common+=(trace_dir="$output/trace")
fi
runtime_ld_path="$phantom_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
alice=(env CUDA_VISIBLE_DEVICES="$alice_gpu" LD_LIBRARY_PATH="$runtime_ld_path" \
  "$binary" r=1 p="$port" "${common[@]}")
bob=(env CUDA_VISIBLE_DEVICES="$bob_gpu" LD_LIBRARY_PATH="$runtime_ld_path" \
  "$binary" r=2 p="$port" "${common[@]}")

alice_pid=""
cleanup() {
  if [[ -n "$alice_pid" ]] && kill -0 "$alice_pid" 2>/dev/null; then
    kill "$alice_pid" 2>/dev/null || true
    wait "$alice_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "VIT65_A0_RUN status=START protocol=$protocol backend=phantom_gpu logical_tokens=65 physical_tokens=128 port=$port trace=$trace"
"${alice[@]}" >"$output/alice.log" 2>&1 &
alice_pid=$!
sleep 1
set +e
"${bob[@]}" >"$output/bob.log" 2>&1
bob_status=$?
wait "$alice_pid"
alice_status=$?
set -e
alice_pid=""
trap - EXIT INT TERM
((alice_status == 0 && bob_status == 0)) || \
  fail "party_exit alice=$alice_status bob=$bob_status"

"$python_bin" "$repo/repro/summarize_vit65_a0.py" \
  --alice "$output/alice.log" --bob "$output/bob.log" \
  --prefix "$metric_prefix" --output "$output/metrics.json"
if [[ "$trace" == "1" ]]; then
  "$python_bin" "$repo/repro/verify_vit65_a0.py" \
    --fixture "$model" --trace "$output/trace" \
    --protocol "$protocol" \
    --output "$output/correctness.json"
fi
echo "VIT65_A0_RUN status=PASS output=$output"
