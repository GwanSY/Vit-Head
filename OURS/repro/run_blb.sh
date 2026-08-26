#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "NOTICE: repro/run_blb.sh only runs vit_smoke_e2e; formal_result=no" >&2
exec bash "$repo/repro/run_vit_e2e.sh" "$@"
