#!/usr/bin/env bash
set -euo pipefail

# Canonical shared build entry.  TARGET selects the model or test binary;
# build_vit.sh remains the implementation for backward compatibility.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${TARGET:-}" ]]; then
  echo "usage: TARGET=BOLT_BERT|BOLT_VIT|BOLT_BERT_GPU|BOLT_VIT_GPU JOBS=4 scripts/build.sh" >&2
  exit 2
fi

if [[ "$TARGET" == *_GPU ]]; then
  exec "$ROOT/scripts/build_gpu.sh" "$@"
fi

echo "BOLT_BUILD target=$TARGET jobs=${JOBS:-4} backend=cpu"
exec "$ROOT/scripts/build_vit.sh" "$@"
