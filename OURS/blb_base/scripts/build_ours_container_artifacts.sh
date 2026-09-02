#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
base_image=${OURS_BASE_IMAGE:-crpi-hxmgk7l66ku6bx2x.cn-hangzhou.personal.cr.aliyuncs.com/tianshixu/ppdl:blbv1}
jobs=${JOBS:-16}
container_name="ours-build-$(date +%Y%m%d-%H%M%S)-$$"

docker run --rm --pull=never --name "$container_name" --network=none \
  --mount "type=bind,src=$root,dst=/workspace/OURS" \
  --mount "type=bind,src=/usr/local/cuda-12.3,dst=/usr/local/cuda-12.3,readonly" \
  --workdir /workspace/OURS "$base_image" sh -lc \
  "env JOBS=$jobs \
OURS_VIT_BUILD_DIR=/workspace/OURS/build/container-gpu \
OURS_VIT_EIGEN_BUILD_DIR=/workspace/OURS/build/container-eigen \
CUDA_ROOT=/usr/local/cuda-12.3 \
CMAKE_BIN=/usr/local/bin/cmake \
scripts/build_vit_b16.sh && \
chown -R $(id -u):$(id -g) /workspace/OURS/build/container-gpu \
  /workspace/OURS/build/container-eigen"

echo "OURS_CONTAINER_ARTIFACT_BUILD status=PASS build_dir=$root/build/container-gpu"
