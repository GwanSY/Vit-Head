#!/usr/bin/env bash
set -euo pipefail

backend="gpu"
shape_network=0
output_dir="outputs/bumblebee-bert"
extra_args=()

while (($#)); do
  case "$1" in
    --backend)
      backend="${2:?--backend requires gpu or cpu}"
      shift 2
      ;;
    --shape-network)
      shape_network=1
      shift
      ;;
    --output-dir)
      output_dir="${2:?--output-dir requires a path}"
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ "$backend" != "gpu" && "$backend" != "cpu" ]]; then
  echo "error: --backend must be gpu or cpu" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"
config="$repo_root/examples/python/llm/flax_bert/2pc.json"
bazel_bin="${BAZEL_BIN:-/usr/local/bin/bazel-6.5.0}"
bazel_args=(--batch)
build_args=(-c opt --jobs="${BAZEL_JOBS:-128}")

export JAX_PLATFORMS=cpu
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export SPU_BB_SET_IEQUAL_BITS="${SPU_BB_SET_IEQUAL_BITS:-16}"
export SPU_BB_ENABLE_MMUL_PACK="${SPU_BB_ENABLE_MMUL_PACK:-1}"
export SPU_BUMBLEBEE_HE_BACKEND="$backend"

if [[ "$backend" == "gpu" ]]; then
  command -v nvidia-smi >/dev/null || {
    echo "error: GPU backend requested but nvidia-smi is unavailable; no CPU fallback" >&2
    exit 1
  }
  [[ -r /opt/openbumblebee-phantom/lib/libPhantom.so ]] || {
    echo "error: /opt/openbumblebee-phantom/lib/libPhantom.so is unavailable; no CPU fallback" >&2
    exit 1
  }
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    selected_gpu="$({ nvidia-smi --query-gpu=index,memory.free \
      --format=csv,noheader,nounits || true; } | sort -t, -k2,2nr | head -n1 | cut -d, -f1 | tr -d ' ')"
    [[ -n "$selected_gpu" ]] || {
      echo "error: GPU backend requested but no CUDA GPU is visible; no CPU fallback" >&2
      exit 1
    }
    export CUDA_VISIBLE_DEVICES="$selected_gpu"
  fi
  build_args+=(--config=gpu)
  export LD_LIBRARY_PATH="/opt/openbumblebee-phantom/lib:${LD_LIBRARY_PATH:-}"
  echo "OpenBumbleBee backend=gpu CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
  export CUDA_VISIBLE_DEVICES=""
  echo "OpenBumbleBee backend=cpu (original SEAL path)"
fi

node_pids=()
cleanup() {
  for node_pid in "${node_pids[@]}"; do
    kill "$node_pid" 2>/dev/null || true
    wait "$node_pid" 2>/dev/null || true
  done
  if ((shape_network)); then
    tc qdisc del dev lo root 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if ((shape_network)); then
  command -v tc >/dev/null || {
    echo "error: --shape-network requires tc" >&2
    exit 1
  }
  # Colocated 2PC traverses loopback twice per ping. 0.15 ms one-way gives
  # approximately 0.3 ms RTT, while the rate limiter is exactly 1 Gbit/s.
  tc qdisc replace dev lo root netem delay 0.15ms rate 1gbit
fi

"$bazel_bin" "${bazel_args[@]}" build "${build_args[@]}" \
  //examples/python/utils:nodectl \
  //examples/python/llm/flax_bert

runfiles="$repo_root/bazel-bin/examples/python/llm/flax_bert/flax_bert.runfiles"
export PYTHONPATH="$runfiles:$runfiles/spulib${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output_dir"
stamp="$(date +%Y%m%d-%H%M%S)"
node0_log="$output_dir/node0-$backend-$stamp.log"
node1_log="$output_dir/node1-$backend-$stamp.log"
driver_log="$output_dir/bert-$backend-$stamp.log"

# Start ranks as independent processes. `nodectl up` forks after importing JAX,
# which is unsafe once a CUDA runtime may be initialized.
stdbuf -oL -eL bazel-bin/examples/python/utils/nodectl \
  --config "$config" start --node_id node:0 >"$node0_log" 2>&1 &
node_pids+=("$!")
stdbuf -oL -eL bazel-bin/examples/python/utils/nodectl \
  --config "$config" start --node_id node:1 >"$node1_log" 2>&1 &
node_pids+=("$!")
sleep 3
for node_pid in "${node_pids[@]}"; do
  if ! kill -0 "$node_pid" 2>/dev/null; then
    echo "error: an SPU node exited during startup; see $node0_log and $node1_log" >&2
    exit 1
  fi
done

bazel-bin/examples/python/llm/flax_bert/flax_bert \
  --backend "$backend" --config "$config" --max-length 128 \
  "${extra_args[@]}" 2>&1 | tee "$driver_log"

echo "driver log: $driver_log"
echo "node logs: $node0_log $node1_log"
