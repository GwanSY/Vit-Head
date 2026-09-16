#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "usage: $0 cpu|gpu BINARY OUTPUT_DIR PORT [ALICE_GPU_ID [BOB_GPU_ID]]" >&2
  exit 2
fi

backend=$1
binary=$2
output_dir=$3
port=$4
alice_gpu_id=${5:-}
bob_gpu_id=${6:-$alice_gpu_id}

if [[ "$backend" != cpu && "$backend" != gpu ]]; then
  echo "backend must be cpu or gpu" >&2
  exit 2
fi
if [[ "$backend" == gpu && -z "$alice_gpu_id" ]]; then
  echo "ALICE_GPU_ID is required for the gpu backend" >&2
  exit 2
fi
if [[ ! -x "$binary" ]]; then
  echo "binary is not executable: $binary" >&2
  exit 2
fi
run_dir=$(dirname "$(dirname "$binary")")

mkdir -p "$output_dir"
alice_pid=
bob_pid=
monitor_pid=
cleanup() {
  if [[ -n "$alice_pid" ]]; then kill -TERM -- "-$alice_pid" 2>/dev/null || true; fi
  if [[ -n "$bob_pid" ]]; then kill -TERM -- "-$bob_pid" 2>/dev/null || true; fi
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
  tc qdisc del dev lo root 2>/dev/null || true
  tc qdisc show dev lo > "$output_dir/qdisc_after.txt" 2>&1 || true
}
trap cleanup EXIT INT TERM

tc qdisc show dev lo > "$output_dir/qdisc_before.txt"
if ! grep -q '^qdisc noqueue ' "$output_dir/qdisc_before.txt"; then
  echo "refusing to replace a pre-existing loopback qdisc" >&2
  exit 3
fi

# Paper LAN: 1 Gbps and 0.3 ms RTT. Netem delay is one-way, hence 0.15 ms.
tc qdisc replace dev lo root netem limit 100000 delay 0.15ms rate 1gbit
tc qdisc show dev lo > "$output_dir/qdisc_active.txt"

{
  date --iso-8601=ns
  echo "backend=$backend"
  echo "binary=$binary"
  echo "run_dir=$run_dir"
  echo "binary_sha256=$(sha256sum "$binary" | awk '{print $1}')"
  echo "port=$port"
  echo "alice_cpus=0-47,96-111"
  echo "bob_cpus=48-95,144-159"
  echo "threads=64"
  if [[ "$backend" == gpu ]]; then
    echo "alice_gpu_id=$alice_gpu_id"
    echo "bob_gpu_id=$bob_gpu_id"
    nvidia-smi -i "$alice_gpu_id" --query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu --format=csv,noheader
    if [[ "$bob_gpu_id" != "$alice_gpu_id" ]]; then
      nvidia-smi -i "$bob_gpu_id" --query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu --format=csv,noheader
    fi
  fi
  lscpu | grep -E 'Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core'
} > "$output_dir/environment.txt"

alice_env=(env OMP_NUM_THREADS=64 OMP_DYNAMIC=FALSE)
bob_env=(env OMP_NUM_THREADS=64 OMP_DYNAMIC=FALSE)
if [[ "$backend" == gpu ]]; then
  alice_env+=(CUDA_VISIBLE_DEVICES="$alice_gpu_id")
  bob_env+=(CUDA_VISIBLE_DEVICES="$bob_gpu_id")
fi

cd "$run_dir"
date --iso-8601=ns > "$output_dir/start.txt"
setsid /usr/bin/time -v -o "$output_dir/alice.resource.txt" \
  timeout --signal=TERM --kill-after=10s 900s \
  taskset -c 0-47,96-111 "${alice_env[@]}" \
  "$binary" r=1 p="$port" nt=64 > "$output_dir/alice.log" 2>&1 &
alice_pid=$!

sleep 1

setsid /usr/bin/time -v -o "$output_dir/bob.resource.txt" \
  timeout --signal=TERM --kill-after=10s 900s \
  taskset -c 48-95,144-159 "${bob_env[@]}" \
  "$binary" r=2 p="$port" nt=64 ip=127.0.0.1 > "$output_dir/bob.log" 2>&1 &
bob_pid=$!

# Capture kernel byte counters at the first flushed "begin test" marker.
# This independently checks the program's own communication accounting.
(
  while kill -0 "$alice_pid" 2>/dev/null && kill -0 "$bob_pid" 2>/dev/null; do
    if grep -q '^begin test$' "$output_dir/alice.log" 2>/dev/null || \
       grep -q '^begin test$' "$output_dir/bob.log" 2>/dev/null; then
      date --iso-8601=ns > "$output_dir/online_start.txt"
      tc -s qdisc show dev lo > "$output_dir/qdisc_online_start.txt"
      exit 0
    fi
    sleep 0.02
  done
  exit 1
) &
monitor_pid=$!

set +e
wait "$alice_pid"
alice_rc=$?
alice_pid=
if [[ $alice_rc -ne 0 && -n "$bob_pid" ]]; then
  kill -TERM -- "-$bob_pid" 2>/dev/null || true
fi
wait "$bob_pid"
bob_rc=$?
bob_pid=
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=

tc -s qdisc show dev lo > "$output_dir/qdisc_online_end.txt"
set -e

date --iso-8601=ns > "$output_dir/end.txt"
{
  echo "alice_rc=$alice_rc"
  echo "bob_rc=$bob_rc"
} > "$output_dir/status.txt"

if [[ $alice_rc -ne 0 || $bob_rc -ne 0 ]]; then
  exit 1
fi
