#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
image=${OURS_DOCKER_IMAGE:-ours}

if [[ "$image" != ours && "$image" != ours:* ]]; then
  echo "OURS_DOCKER_IMAGE must be ours or an ours:<tag> name" >&2
  exit 2
fi

host_binary="$root/build/container-gpu/bin/ours_vit_b16_main"
if [[ ! -x "$host_binary" ]]; then
  echo "missing compiled OURS binary: $host_binary" >&2
  exit 2
fi
host_binary_sha=$(sha256sum "$host_binary" | awk '{print $1}')

# The legacy Docker builder has previously reused a stale COPY layer even
# after the executable changed. Formal experiment provenance requires the
# image payload to be byte-identical to the just-built host artifact.
docker build --no-cache --pull=false --file "$root/docker/Dockerfile.ours" \
  --tag "$image" "$root"
image_binary_sha=$(docker run --rm --network=none "$image" \
  sha256sum /workspace/OURS/build/gpu/bin/ours_vit_b16_main |
  awk '{print $1}')
if [[ "$image_binary_sha" != "$host_binary_sha" ]]; then
  echo "image binary hash mismatch: host=$host_binary_sha image=$image_binary_sha" >&2
  exit 3
fi
echo "OURS_IMAGE_BINARY_BINDING status=PASS sha256=$host_binary_sha"
echo "OURS_IMAGE_BUILD status=PASS image=$image"
