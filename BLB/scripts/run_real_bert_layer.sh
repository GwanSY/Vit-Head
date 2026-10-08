#!/usr/bin/env bash
# Real BERT layer-0 numerical gate; no backup tree or synthetic donor entry.
set -euo pipefail
blb_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
ours_root="$blb_root/../OURS/blb_base"
ours_root=$(cd "$ours_root" && pwd -P)
out=${1:?usage: run_real_bert_layer.sh NEW_OUTPUT_UNDER_OURS_OUTPUTS GPU0 GPU1}
gpu0=${2:?GPU0 required}
gpu1=${3:?GPU1 required}
[[ "$gpu0" =~ ^[0-9]+$ && "$gpu1" =~ ^[0-9]+$ && "$gpu0" != "$gpu1" ]]
[[ "$out" = /* ]] || out="$ours_root/$out"
[[ "$out" == "$ours_root/outputs/"* && ! -e "$out" ]]
image=${OURS_DOCKER_IMAGE:-ours:restored-fast-20260903d}
python3 -B "$blb_root/scripts/validate_real_bert_layer.py" --check-inputs
python3 -B "$blb_root/scripts/prepare_real_bert_layer.py" --output "$out"
container_out="/workspace/OURS/${out#"$ours_root/"}"
docker run --rm --network=none \
  --mount "type=bind,src=$ours_root,dst=/workspace/OURS,readonly" \
  --mount "type=bind,src=$out,dst=$container_out" \
  --mount type=bind,src=/usr/local/cuda-12.3,dst=/usr/local/cuda-12.3,readonly \
  --entrypoint bash "$image" "$container_out/build.sh" > "$out/build.log" 2>&1
docker run --rm --gpus all --network=none --cap-add=NET_ADMIN --shm-size=16g \
  --mount "type=bind,src=$ours_root,dst=/workspace/OURS,readonly" \
  --mount "type=bind,src=$out,dst=$container_out" \
  --mount "type=bind,src=$blb_root,dst=/workspace/BLB,readonly" \
  --mount type=bind,src=/usr/local/cuda-12.3,dst=/usr/local/cuda-12.3,readonly \
  -e BLB_REAL_MODEL=/workspace/BLB/checkpoints/official-blb-mrpc/weights_txt \
  -e "BLB_REAL_OUTPUT=$container_out/run" -e OURS_VIT_MPC_WORKERS=64 \
  -e LD_LIBRARY_PATH=/workspace/OURS/build/container-gpu/extern/phantom/lib:/usr/local/cuda-12.3/lib64 \
  --entrypoint bash "$image" /workspace/BLB/scripts/run_pair_once.sh gpu \
  "$container_out/build/bin/ours_vit_b16_main" "$container_out/run" 22103 "$gpu0" "$gpu1" \
  > "$out/run.log" 2>&1
OPENBLAS_NUM_THREADS=4 python3 -B "$blb_root/scripts/validate_real_bert_layer.py" \
  --run "$out/run" > "$out/validation_stdout.json"
echo "REAL_BERT_LAYER_GATE status=PASS output=$out"
