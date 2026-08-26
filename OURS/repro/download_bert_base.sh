#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
raw_dir="$repo/checkpoints/bert-base/raw"
mkdir -p "$raw_dir"

base_url="https://huggingface.co/google-bert/bert-base-uncased/resolve/main"
curl -fL --retry 3 --connect-timeout 20 \
  -o "$raw_dir/config.json" "$base_url/config.json"
curl -fL --retry 3 --connect-timeout 20 \
  -o "$raw_dir/model.safetensors" "$base_url/model.safetensors"

expected="68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
actual="$(sha256sum "$raw_dir/model.safetensors" | awk '{print $1}')"
if [[ "$actual" != "$expected" ]]; then
  echo "CHECKPOINT status=FAIL reason=sha256_mismatch expected=$expected actual=$actual" >&2
  exit 2
fi

echo "CHECKPOINT status=PASS model=google-bert/bert-base-uncased sha256=$actual"
