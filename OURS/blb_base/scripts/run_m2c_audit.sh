#!/usr/bin/env bash
# 隔离的 M2C 对照；BERT 保留供体五块测试入口，不冒充完整模型验证。
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
output=${1:?usage: bash scripts/run_m2c_audit.sh NEW_OUTPUT_DIR [GPU0 GPU1]}
gpu0=${2:-0}
gpu1=${3:-1}
[[ "$gpu0" =~ ^[0-9]+$ && "$gpu1" =~ ^[0-9]+$ && "$gpu0" != "$gpu1" ]]
[[ "$output" = /* ]] || output="$root/$output"
[[ "$output" == "$root/outputs/"* && ! -e "$output" ]] || { echo 'output must be a new directory under blb_base/outputs' >&2; exit 2; }
image=${OURS_DOCKER_IMAGE:-ours:restored-fast-20260903d}
container=/workspace/OURS/${output#"$root/"}
weights=$(readlink -f "$root/../../BLB/dataset/quantize/mrpc/weights_txt")
[[ -f "$weights/bert.encoder.layer.0.attention.self.query.weight.txt" ]]
cd "$root"
python3 -B scripts/prepare_m2c_audit.py --output "$output/bert"
python3 -B scripts/prepare_m2c_audit.py --vit --output "$output/vit"
for model in bert vit; do
  docker run --rm --network=none \
    --mount "type=bind,src=$root,dst=/workspace/OURS,readonly" \
    --mount "type=bind,src=$output,dst=$container" \
    --mount type=bind,src=/usr/local/cuda-12.3,dst=/usr/local/cuda-12.3,readonly \
    --entrypoint bash "$image" "$container/$model/build.sh" > "$output/build_$model.log" 2>&1
done
mkdir -p "$output/dataset/quantize/mrpc/weights_txt"
cp "$root/../../BLB/scripts/run_pair_once.sh" "$output/run_pair.sh"
docker run --rm --gpus all --network=none --cap-add=NET_ADMIN --shm-size=16g \
  --mount "type=bind,src=$root,dst=/workspace/OURS,readonly" \
  --mount "type=bind,src=$output,dst=$container" \
  --mount "type=bind,src=$weights,dst=$container/dataset/quantize/mrpc/weights_txt,readonly" \
  --mount type=bind,src=/usr/local/cuda-12.3,dst=/usr/local/cuda-12.3,readonly \
  -e LD_LIBRARY_PATH=/workspace/OURS/build/container-gpu/extern/phantom/lib:/usr/local/cuda-12.3/lib64 \
  --entrypoint bash "$image" "$container/run_pair.sh" gpu \
  "$container/bert/build/bin/ckks_bert_large_main" "$container/bert/run" 21103 "$gpu0" "$gpu1" \
  > "$output/bert_launcher.log" 2>&1
OURS_VIT_PHYSICAL_GPUS="$gpu0,$gpu1" \
OURS_VIT_MPC_WORKERS=64 \
OURS_VIT_BINARY_OVERRIDE="$output/vit/build/bin/ours_vit_b16_main" \
bash scripts/run_ours_container.sh B ours_vit_b16/experiments/all_softmax_head_plan.txt \
  "$output/vit/run" 2 block > "$output/vit_launcher.log" 2>&1
python3 -B scripts/report_m2c_audit.py --output "$output" --bert bert --vit vit > "$output/report_stdout.json"
