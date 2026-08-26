#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_build_dir="${BUILD_DIR:-$repo/build_blb_formal/sci}"
binary="${BLB_BINARY:-$default_build_dir/bin/ckks_bert_large_main}"
output="${OUTPUT:-$repo/outputs/bert_layer0_formal_$(date +%Y%m%d_%H%M%S)}"
# Keep the 65-listener block above Linux's usual ephemeral client range.  If
# it overlaps, BOB's successive loopback connects can occupy a future ALICE
# listener port between the up-front check and bind.
port="${PORT:-62000}"
gpu_id="${GPU_ID:-0}"
backend="${BACKEND:-cpu}"
run_timeout="${RUN_TIMEOUT:-60m}"
model_path="${MODEL_PATH:-$repo/checkpoints/official-blb-mrpc/weights_txt}"
oracle_dir="${BLB_ORACLE_DIR:-$model_path/oracle_shares}"
runtime_paths=(
  "$default_build_dir/extern/phantom/lib"
  "$default_build_dir/extern/seal/lib"
)
for cuda_root_candidate in "${CUDA_ROOT:-}" "${CUDA_HOME:-}" "${CUDAToolkit_ROOT:-}"; do
  if [[ -n "$cuda_root_candidate" ]]; then
    [[ -d "$cuda_root_candidate/lib64" ]] && runtime_paths+=("$cuda_root_candidate/lib64")
    [[ -d "$cuda_root_candidate/targets/x86_64-linux/lib" ]] && \
      runtime_paths+=("$cuda_root_candidate/targets/x86_64-linux/lib")
  fi
done
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  runtime_paths+=("$LD_LIBRARY_PATH")
fi
blb_runtime_path="$(IFS=:; printf '%s' "${runtime_paths[*]}")"

case "$binary" in "$repo"/*) ;; *)
  echo "error: BLB_BINARY must be inside $repo" >&2; exit 2 ;;
esac
case "$output" in "$repo"/*) ;; *)
  echo "error: OUTPUT must be inside $repo" >&2; exit 2 ;;
esac
case "$model_path" in "$repo"/*) ;; *)
  echo "error: MODEL_PATH must be inside $repo" >&2; exit 2 ;;
esac
case "$oracle_dir" in "$repo"/*) ;; *)
  echo "error: BLB_ORACLE_DIR must be inside $repo" >&2; exit 2 ;;
esac

if [[ ! -x "$binary" ]]; then
  echo "error: BLB binary is missing or not executable: $binary" >&2
  exit 1
fi
build_environment="$default_build_dir/.blb_build_environment"
if [[ ! -f "$build_environment" ]]; then
  echo "SINGLE_LAYER_FORMAL status=BLOCKED reason=build_environment_stamp_missing path=$build_environment" >&2
  echo "Rebuild this BUILD_DIR inside the current host/container with repro/build_formal_blb.sh." >&2
  exit 2
fi
build_glibc="$(sed -n 's/^glibc=//p' "$build_environment")"
runtime_glibc="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
if [[ -z "$build_glibc" || -z "$runtime_glibc" || "$build_glibc" != "$runtime_glibc" ]]; then
  echo "SINGLE_LAYER_FORMAL status=BLOCKED reason=glibc_build_runtime_mismatch built=${build_glibc:-unknown} runtime=${runtime_glibc:-unknown}" >&2
  echo "Do not reuse a host build in the official container; rebuild into a container-only BUILD_ROOT." >&2
  exit 2
fi
if [[ ! -d "$model_path" ]] || ! find "$model_path" -type f -print -quit | grep -q .; then
  echo "SINGLE_LAYER_FORMAL status=BLOCKED reason=missing_BLB_local_checkpoint path=$model_path" >&2
  exit 2
fi
if ! /usr/bin/python3 "$repo/repro/verify_bert_manifest.py" "$model_path" --layer 0; then
  echo "SINGLE_LAYER_FORMAL status=BLOCKED reason=manifest_verification_failed" >&2
  exit 2
fi
if ! /usr/bin/python3 "$repo/repro/verify_bert_oracle.py" "$model_path" "$oracle_dir"; then
  echo "SINGLE_LAYER_FORMAL status=BLOCKED reason=three_oracle_manifest_verification_failed" >&2
  exit 2
fi

if ! [[ "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65471 )); then
  echo "error: PORT must be an integer between 1024 and 65471" >&2
  exit 1
fi
case "$backend" in
  cpu|host) backend="cpu" ;;
  gpu|device) backend="gpu" ;;
  *) echo "error: BACKEND must be cpu or gpu" >&2; exit 1 ;;
esac
if [[ -r /proc/sys/net/ipv4/ip_local_port_range ]]; then
  read -r ephemeral_low ephemeral_high < /proc/sys/net/ipv4/ip_local_port_range
  if [[ "$ephemeral_low" =~ ^[0-9]+$ && "$ephemeral_high" =~ ^[0-9]+$ ]] &&
     (( port <= ephemeral_high && port + 64 >= ephemeral_low )); then
    echo "error: PORT..PORT+64 overlaps the local ephemeral range ${ephemeral_low}-${ephemeral_high}; choose a base above it" >&2
    exit 1
  fi
fi

if command -v ss >/dev/null 2>&1; then
  # NetIO uses PORT and NonLinear constructs 64 IOPacks on PORT+1..PORT+64.
  # Checking only two sockets allowed a run to spend minutes generating CKKS
  # keys before failing on an occupied nonlinear port.
  for ((port_offset = 0; port_offset <= 64; ++port_offset)); do
    candidate="$((port + port_offset))"
    # Check every TCP state, not only LISTEN. SCI does not enable SO_REUSEADDR,
    # so a recently closed/TIME_WAIT socket can also make bind fail.
    if ss -H -tan "sport = :$candidate" | grep -q .; then
      echo "error: port $candidate has an active or recently closed TCP socket; choose another PORT" >&2
      exit 1
    fi
  done
fi

if ! mkdir -p "$output" 2>/dev/null || [[ ! -w "$output" ]]; then
  echo "SINGLE_LAYER_FORMAL status=BLOCKED reason=output_not_writable path=$output" >&2
  echo "Choose a new writable OUTPUT directory inside BLB." >&2
  exit 2
fi
cd "$repo"
alice_log="$output/alice.log"
bob_log="$output/bob.log"
alice_pid=""
bob_pid=""

cleanup() {
  for child_pid in "$alice_pid" "$bob_pid"; do
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
      kill "$child_pid" 2>/dev/null || true
      wait "$child_pid" 2>/dev/null || true
    fi
  done
  alice_pid=""
  bob_pid=""
}

abort_run() {
  trap - INT TERM HUP TSTP
  cleanup
  echo "SINGLE_LAYER_FORMAL status=BLOCKED reason=interrupted" >&2
  exit 130
}

trap cleanup EXIT
trap abort_run INT TERM HUP TSTP

echo "SINGLE_LAYER_FORMAL status=START layer=0 model_layers=12 backend=$backend port=$port binary=$binary output=$output"

set +e
CUDA_VISIBLE_DEVICES="$gpu_id" \
  timeout "$run_timeout" \
  env BLB_ORACLE_DIR="$oracle_dir" BLB_ORACLE_MANIFEST_VERIFIED=1 LD_LIBRARY_PATH="$blb_runtime_path" "$binary" r=1 p="$port" ip=127.0.0.1 model="$model_path" backend="$backend" manifest_verified=1 layers=1 \
  >"$alice_log" 2>&1 &
alice_pid=$!

for _ in {1..100}; do
  if ! kill -0 "$alice_pid" 2>/dev/null; then
    break
  fi
  if command -v ss >/dev/null 2>&1 &&
     ss -H -ltn "sport = :$port" | grep -q .; then
    break
  fi
  sleep 0.1
done

CUDA_VISIBLE_DEVICES="$gpu_id" \
  timeout "$run_timeout" \
  env BLB_ORACLE_DIR="$oracle_dir" BLB_ORACLE_MANIFEST_VERIFIED=1 LD_LIBRARY_PATH="$blb_runtime_path" "$binary" r=2 p="$port" ip=127.0.0.1 model="$model_path" backend="$backend" manifest_verified=1 layers=1 \
  >"$bob_log" 2>&1 &
bob_pid=$!
wait "$bob_pid"
bob_status=$?
bob_pid=""

wait "$alice_pid"
alice_status=$?
alice_pid=""
set -e

if [[ "$alice_status" != "0" || "$bob_status" != "0" ]]; then
  echo "SINGLE_LAYER_FORMAL status=BLOCKED alice_status=$alice_status bob_status=$bob_status"
  echo "--- ALICE log (tail) ---"
  tail -n 40 "$alice_log"
  echo "--- BOB log (tail) ---"
  tail -n 40 "$bob_log"
  exit 1
fi

/usr/bin/python3 "$repo/repro/accept_single_layer.py" "$output"
echo "SINGLE_LAYER_FORMAL status=PASS output=$output"
