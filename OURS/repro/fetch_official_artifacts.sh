#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
artifact_dir="$repo/artifacts/zenodo"
fetch_image="${FETCH_IMAGE:-1}"
mkdir -p "$artifact_dir"

download_resumable() {
  local url="$1"
  local output="$2"
  local expected="$3"
  if [[ -f "$output" ]] &&
     [[ "$(md5sum "$output" | awk '{print $1}')" == "$expected" ]]; then
    echo "ZENODO_FETCH item=$(basename "$output") status=PASS source=cache md5=$expected bytes=$(stat -c %s "$output")"
    return
  fi
  echo "ZENODO_FETCH item=$(basename "$output") status=START"
  curl -fL --retry 5 --retry-delay 2 --connect-timeout 30 \
    --continue-at - --output "$output" "$url"
  verify_md5 "$output" "$expected"
}

download_fresh() {
  local url="$1"
  local output="$2"
  echo "ZENODO_FETCH item=$(basename "$output") status=START"
  curl -fL --retry 5 --retry-delay 2 --connect-timeout 30 \
    --output "$output" "$url"
  echo "ZENODO_FETCH item=$(basename "$output") status=PASS bytes=$(stat -c %s "$output")"
}

verify_md5() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(md5sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "ZENODO_FETCH item=$(basename "$path") status=FAIL reason=md5 expected=$expected actual=$actual" >&2
    exit 2
  fi
  echo "ZENODO_FETCH item=$(basename "$path") status=PASS md5=$actual bytes=$(stat -c %s "$path")"
}

download_resumable \
  "https://zenodo.org/api/records/15590214/files/CKKS-MPC-main.zip/content" \
  "$artifact_dir/CKKS-MPC-main.zip" \
  "a6a84554a55204689ccb563001595c6c"

download_resumable \
  "https://zenodo.org/api/records/15627952/files/0001-more-flexible.patch/content" \
  "$artifact_dir/0001-more-flexible.patch" \
  "b5401f8f59b935cefea4d53f2117bdef"

download_fresh "https://zenodo.org/api/records/15621724" \
  "$artifact_dir/record-15621724.json"
download_fresh "https://zenodo.org/api/records/15627952" \
  "$artifact_dir/record-15627952.json"

if [[ "$fetch_image" == "1" ]]; then
  download_resumable \
    "https://zenodo.org/api/records/15621724/files/blb.tar.gz/content" \
    "$artifact_dir/blb.tar.gz" \
    "86393729bc5530f43e95d754138d51e7"
else
  echo "ZENODO_FETCH item=blb.tar.gz status=SKIPPED reason=FETCH_IMAGE_not_1"
fi

echo "ZENODO_FETCH status=PASS root=$artifact_dir"
