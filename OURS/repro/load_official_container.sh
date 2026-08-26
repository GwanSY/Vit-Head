#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
archive="$repo/artifacts/zenodo/blb.tar.gz"
expected="86393729bc5530f43e95d754138d51e7"

if [[ ! -f "$archive" ]]; then
  echo "BLB_CONTAINER_LOAD status=BLOCKED reason=missing_archive path=$archive" >&2
  exit 2
fi
actual="$(md5sum "$archive" | awk '{print $1}')"
if [[ "$actual" != "$expected" ]]; then
  echo "BLB_CONTAINER_LOAD status=FAIL reason=md5 expected=$expected actual=$actual" >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "BLB_CONTAINER_LOAD status=BLOCKED reason=docker_daemon_permission" >&2
  echo "Run: sudo usermod -aG docker \"\$USER\"; then log out and back in." >&2
  exit 2
fi

gzip -dc "$archive" | docker load
echo "BLB_CONTAINER_LOAD status=PASS source=$archive"
