#!/usr/bin/env bash
set -euo pipefail

# IRON-facing multi-seed harness.  The fixture generator and reference are
# shared with the reviewed BLB adapter; BINARY is explicitly overridden so
# every secure run exercises the IRON build output.
IRON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$IRON_ROOT/.." && pwd)"
BLB_ROOT="${BLB_ROOT:-$WORKSPACE_ROOT/BLB}"
if [[ "${VALIDATION_LOGITS:-0}" == "1" || "${DEBUG_LOGITS:-0}" == "1" ]]; then
  default_binary="$IRON_ROOT/build/secure_vit/bin/IRON_VIT_GENERIC"
else
  default_binary="$IRON_ROOT/build/secure_vit/bin/IRON_VIT_GENERIC"
fi
BINARY="${BINARY:-$default_binary}"
FIXTURE_ROOT="${FIXTURE_ROOT:-$IRON_ROOT/fixtures/vit_b16_seed_regression}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$IRON_ROOT/outputs/vit_b16_seed_regression}"
PYTHON="${PYTHON:-$WORKSPACE_ROOT/BOLT/.venv-vit-b16/bin/python}"

[[ -x "$BINARY" ]] || { echo "missing IRON ViT binary: $BINARY" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 2; }

BINARY="$BINARY" FIXTURE_ROOT="$FIXTURE_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
  PYTHON="$PYTHON" \
  bash "$BLB_ROOT/repro/run_vit_seed_regression.sh"
