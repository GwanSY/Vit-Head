#!/usr/bin/env bash
set -euo pipefail

# Robust downloader for the official ILSVRC2012 validation archive.  A failed
# transfer returns to this outer loop so every new curl process derives its
# Range offset from the actual on-disk length.  This avoids curl's in-process
# retry behavior overwriting a partially downloaded multi-gigabyte file.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-$ROOT/datasets/imagenet2012}"
URL="${IMAGENET_VAL_URL:-https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar}"
DEST="$DATA_ROOT/ILSVRC2012_img_val.tar"
PART="$DEST.download"
EXPECTED_BYTES=6744924160
EXPECTED_MD5=29b22e2961454d5413ddabcf34fc5622

log() {
  printf '[ImageNet val %s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

fail() {
  log "ERROR: $*" >&2
  exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v md5sum >/dev/null 2>&1 || fail "md5sum is required"
command -v tar >/dev/null 2>&1 || fail "tar is required"
mkdir -p "$DATA_ROOT"

if [[ -e "$DEST" ]]; then
  dest_bytes="$(stat -c '%s' "$DEST")"
  if [[ "$dest_bytes" -eq "$EXPECTED_BYTES" ]]; then
    dest_md5="$(md5sum "$DEST" | awk '{print $1}')"
    if [[ "$dest_md5" == "$EXPECTED_MD5" ]]; then
      log "archive is already complete and verified: $DEST"
      exit 0
    fi
  fi
  fail "destination exists but is not the verified official archive: $DEST"
fi

attempt=0
while true; do
  current_bytes=0
  if [[ -e "$PART" ]]; then
    current_bytes="$(stat -c '%s' "$PART")"
  fi
  if [[ "$current_bytes" -gt "$EXPECTED_BYTES" ]]; then
    fail "partial file exceeds official length: $current_bytes > $EXPECTED_BYTES"
  fi
  if [[ "$current_bytes" -eq "$EXPECTED_BYTES" ]]; then
    break
  fi

  attempt=$((attempt + 1))
  log "attempt $attempt: resuming at byte $current_bytes of $EXPECTED_BYTES"
  curl_status=0
  curl -fL --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
    --continue-at - --output "$PART" "$URL" || curl_status=$?

  next_bytes="$(stat -c '%s' "$PART")"
  if [[ "$next_bytes" -gt "$EXPECTED_BYTES" ]]; then
    fail "server response produced an oversized file: $next_bytes"
  fi
  if [[ "$next_bytes" -eq "$EXPECTED_BYTES" ]]; then
    break
  fi
  if [[ "$next_bytes" -le "$current_bytes" ]]; then
    fail "attempt $attempt made no progress (curl status $curl_status)"
  fi
  log "connection ended with status $curl_status at byte $next_bytes; retrying in 5s"
  sleep 5
done

actual_md5="$(md5sum "$PART" | awk '{print $1}')"
[[ "$actual_md5" == "$EXPECTED_MD5" ]] || \
  fail "MD5 mismatch: $actual_md5 != $EXPECTED_MD5"

member_count="$(tar -tf "$PART" | wc -l)"
[[ "$member_count" -eq 50000 ]] || \
  fail "tar member count mismatch: $member_count != 50000"

mv "$PART" "$DEST"
log "DOWNLOAD=PASS bytes=$EXPECTED_BYTES md5=$EXPECTED_MD5 members=$member_count"
log "archive: $DEST"
