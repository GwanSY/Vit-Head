#!/usr/bin/env bash
# 作者原入口：BERT-base 128 的合成单层，不是完整12层精度推理。
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
output=${1:?usage: bash scripts/run_migration_bert.sh OUTPUT_DIR GPU1 GPU2}
gpu1=${2:?GPU1 required}
gpu2=${3:?GPU2 required}
[[ "$gpu1" =~ ^[0-9]+$ && "$gpu2" =~ ^[0-9]+$ && "$gpu1" != "$gpu2" ]]
[[ "$output" = /* ]] || output="$root/$output"
[[ ! -e "$output" ]] || { echo 'output already exists' >&2; exit 2; }
for gpu in "$gpu1" "$gpu2"; do
  read -r free util < <(nvidia-smi -i "$gpu" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
  (( free >= 70000 && util <= 5 )) || { echo "GPU $gpu busy" >&2; exit 2; }
done
image=${OURS_DOCKER_IMAGE:-ours:restored-fast-20260903d}
weights=$(readlink -f "$root/../../BLB/dataset/quantize/mrpc/weights_txt")
[[ -f "$weights/bert.encoder.layer.0.attention.self.query.weight.txt" ]]
mkdir -p "$output/bert_native"
mkdir -p "$output/build/container-gpu/bin" "$output/dataset/quantize/mrpc/weights_txt"
ln -s /workspace/OURS/build/container-gpu/bin/ckks_bert_large_main "$output/build/container-gpu/bin/ckks_bert_large_main"
cp "$root/../../BLB/scripts/run_pair_once.sh" "$output/run_pair_once.sh"
sha256sum "$root/build/container-gpu/bin/ours_vit_b16_main" > "$output/vit_binary_before.sha256"
docker image inspect "$image" --format '{{.Id}}' > "$output/image_id.txt"
# 只构建作者原有 BERT target，不重建/替换冻结的 ViT 程序。
docker run --rm --runtime=runc --network=none -e NVIDIA_VISIBLE_DEVICES=void \
  --mount "type=bind,src=$root,dst=/workspace/OURS,readonly" \
  --mount "type=bind,src=$root/build/container-gpu,dst=/workspace/OURS/build/container-gpu" \
  --mount type=bind,src=/usr/local/cuda-12.3,dst=/usr/local/cuda-12.3,readonly \
  --entrypoint bash "$image" -lc \
  '$(sed -n "s/^CMAKE_COMMAND:INTERNAL=//p" /workspace/OURS/build/container-gpu/CMakeCache.txt) --build /workspace/OURS/build/container-gpu --target ckks_bert_large_main --parallel 8' \
  > "$output/build.log" 2>&1
docker run --rm --gpus all --network=none --cap-add=NET_ADMIN --shm-size=16g \
  --mount "type=bind,src=$root,dst=/workspace/OURS,readonly" \
  --mount "type=bind,src=$output,dst=/audit" \
  --mount "type=bind,src=$weights,dst=/audit/dataset/quantize/mrpc/weights_txt,readonly" \
  --mount type=bind,src=/usr/local/cuda-12.3,dst=/usr/local/cuda-12.3,readonly \
  -e LD_LIBRARY_PATH=/workspace/OURS/build/container-gpu/extern/phantom/lib:/usr/local/cuda-12.3/lib64 \
  --entrypoint bash "$image" -ec \
  'cd /audit/build/container-gpu; test -r ../../dataset/quantize/mrpc/weights_txt/bert.encoder.layer.0.attention.self.query.weight.txt; exec bash /audit/run_pair_once.sh gpu /audit/build/container-gpu/bin/ckks_bert_large_main /audit/bert_native 21003 "$1" "$2"' \
  migration-bert "$gpu1" "$gpu2"
sha256sum --check "$output/vit_binary_before.sha256"
