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
binary="$root/build/gpu/bin/ckks_vit_b16_main"
model=${BLB_VIT_MODEL:-$root/checkpoints/vit_b16/bolt_model.txt}
input=${BLB_VIT_INPUT:-$root/inputs/vit_b16/bob_image.txt}
stop_after_block=${BLB_VIT_STOP_AFTER_BLOCK:-12}
timeout_s=${BLB_VIT_TIMEOUT_SECONDS:-21600}
full_pause_file=${BLB_VIT_FULL_PAUSE_FILE:-/tmp/blb_vit_full_paused}

if [[ "$output_dir" != /* ]]; then output_dir="$root/$output_dir"; fi
if [[ "$backend" != cpu && "$backend" != gpu ]]; then
  echo "backend must be cpu or gpu" >&2
  exit 2
fi
if [[ "$backend" == gpu && -z "$alice_gpu_id" ]]; then
  echo "GPU IDs are required for the gpu backend" >&2
  exit 2
fi
if [[ "$stop_after_block" -eq 12 && -e "$full_pause_file" ]]; then
  echo "full 12-block run is paused while single-block performance is under diagnosis: $full_pause_file" >&2
  exit 4
fi
for path in "$binary" "$model" "$input"; do
  if [[ ! -e "$path" ]]; then echo "missing required path: $path" >&2; exit 2; fi
done

mkdir -p "$output_dir"
alice_pid=
bob_pid=
qdisc_owned=0
cleanup() {
  if [[ -n "$alice_pid" ]]; then kill -TERM -- "-$alice_pid" 2>/dev/null || true; fi
  if [[ -n "$bob_pid" ]]; then kill -TERM -- "-$bob_pid" 2>/dev/null || true; fi
  if [[ "$qdisc_owned" == 1 ]]; then
    tc qdisc del dev lo root 2>/dev/null || true
  fi
  tc qdisc show dev lo > "$output_dir/qdisc_after.txt" 2>&1 || true
}
trap cleanup EXIT INT TERM

tc qdisc show dev lo > "$output_dir/qdisc_before.txt"
if ! grep -q '^qdisc noqueue ' "$output_dir/qdisc_before.txt"; then
  echo "refusing to replace a pre-existing loopback qdisc" >&2
  exit 3
fi
# 0.3 ms RTT means 0.15 ms delay in each loopback direction.
tc qdisc replace dev lo root netem limit 100000 delay 0.15ms rate 1gbit
qdisc_owned=1
tc qdisc show dev lo > "$output_dir/qdisc_active.txt"

{
  date --iso-8601=ns
  echo "backend=$backend"
  echo "binary=$binary"
  echo "binary_sha256=$(sha256sum "$binary" | awk '{print $1}')"
  echo "model=$model"
  echo "model_sha256=$(sha256sum "$model" | awk '{print $1}')"
  echo "input=$input"
  echo "input_sha256=$(sha256sum "$input" | awk '{print $1}')"
  echo "logical_tokens=197"
  echo "physical_tokens=256"
  echo "transformer_blocks=$stop_after_block"
  echo "network=1Gbps/0.3ms_RTT"
  echo "alice_cpus=0-47,96-111"
  echo "bob_cpus=48-95,144-159"
  echo "threads_per_party=64"
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

date --iso-8601=ns > "$output_dir/start.txt"
setsid /usr/bin/time -v -o "$output_dir/alice.resource.txt" \
  timeout --foreground --signal=TERM --kill-after=30s "${timeout_s}s" \
  taskset -c 0-47,96-111 "${alice_env[@]}" \
  "$binary" --party 1 --port "$port" --address 127.0.0.1 \
  --model "$model" --input "$input" --output "$output_dir" \
  --backend "$backend" --stop-after-block "$stop_after_block" \
  > "$output_dir/alice.log" 2>&1 &
alice_pid=$!

sleep 1
setsid /usr/bin/time -v -o "$output_dir/bob.resource.txt" \
  timeout --foreground --signal=TERM --kill-after=30s "${timeout_s}s" \
  taskset -c 48-95,144-159 "${bob_env[@]}" \
  "$binary" --party 2 --port "$port" --address 127.0.0.1 \
  --model "$model" --input "$input" --output "$output_dir" \
  --backend "$backend" --stop-after-block "$stop_after_block" \
  > "$output_dir/bob.log" 2>&1 &
bob_pid=$!

set +e
wait "$alice_pid"; alice_rc=$?; alice_pid=
if [[ $alice_rc -ne 0 && -n "$bob_pid" ]]; then kill -TERM -- "-$bob_pid" 2>/dev/null || true; fi
wait "$bob_pid"; bob_rc=$?; bob_pid=
set -e

tc -s qdisc show dev lo > "$output_dir/qdisc_end.txt"
date --iso-8601=ns > "$output_dir/end.txt"
printf 'alice_rc=%s\nbob_rc=%s\n' "$alice_rc" "$bob_rc" > "$output_dir/status.txt"
if [[ $alice_rc -ne 0 || $bob_rc -ne 0 ]]; then exit 1; fi

python3.7 "$root/vit_b16/scripts/summarize_blb_vit_baseline.py" \
  --run-dir "$output_dir"
echo "BLB_VIT_RUN status=PASS backend=$backend output=$output_dir"
