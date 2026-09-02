#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
image=${OURS_DOCKER_IMAGE:-ours}

if [[ "$image" != ours && "$image" != ours:* ]]; then
  echo "OURS_DOCKER_IMAGE must be ours or an ours:<tag> name" >&2
  exit 2
fi

docker build --pull=false --file "$root/docker/Dockerfile.ours" \
  --tag "$image" "$root"
echo "OURS_IMAGE_BUILD status=PASS image=$image"

