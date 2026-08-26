#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo/repro/run_vit_e2e.sh" "$@"
