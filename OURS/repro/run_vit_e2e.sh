#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${VIT_SMOKE_BINARY:-$repo/build_vit_smoke_e2e/adapter/bin/vit_smoke_e2e}"
fixture="${FIXTURE:-$repo/fixtures/vit_e2e_tiny}"
output="${OUTPUT:-$repo/outputs/vit_smoke_e2e_$(date +%Y%m%d_%H%M%S)}"
port="${PORT:-48000}"
run_timeout="${RUN_TIMEOUT:-20m}"
blb_debug="${BLB_DEBUG:-0}"
blb_warmup="${BLB_WARMUP:-0}"

if ! [[ "$port" =~ ^[0-9]+$ ]] || ((port < 1024 || port > 65435)); then
  echo "error: PORT must be an integer in [1024,65435]" >&2
  exit 2
fi
if ! [[ "$blb_debug" =~ ^[01]$ && "$blb_warmup" =~ ^[01]$ ]]; then
  echo "error: BLB_DEBUG and BLB_WARMUP must be 0 or 1" >&2
  exit 2
fi

case "$binary" in "$repo"/*) ;; *)
  echo "error: VIT_SMOKE_BINARY must be inside $repo" >&2; exit 2 ;;
esac
case "$fixture" in "$repo"/*) ;; *)
  echo "error: FIXTURE must be inside $repo" >&2; exit 2 ;;
esac
case "$output" in "$repo"/*) ;; *)
  echo "error: OUTPUT must be inside $repo" >&2; exit 2 ;;
esac
if [[ ! -x "$binary" ]]; then
  echo "error: missing binary: $binary" >&2
  echo "run: CMAKE_BIN=<cmake> bash $repo/repro/build_blb.sh" >&2
  exit 2
fi
if [[ ! -f "$fixture/alice_model.txt" || ! -f "$fixture/bob_input.txt" ]]; then
  python3 "$repo/repro/generate_vit_e2e_fixture.py" --out-dir "$fixture"
fi
mkdir -p "$output"
echo "VIT_SMOKE_E2E status=START formal_result=no reason=protocol_chain_only"

alice=(env -u LD_LIBRARY_PATH "$binary" r=1 p="$port" model="$fixture/alice_model.txt"
       variant=blb he=packed output="$output/secure" debug="$blb_debug"
       warmup="$blb_warmup")
bob=(env -u LD_LIBRARY_PATH "$binary" r=2 p="$port" ip=127.0.0.1
     input="$fixture/bob_input.txt" variant=blb he=packed
     output="$output/secure" debug="$blb_debug" warmup="$blb_warmup")
if [[ "$run_timeout" != "0" ]]; then
  alice=(timeout --foreground --kill-after=5s "$run_timeout" "${alice[@]}")
  bob=(timeout --foreground --kill-after=5s "$run_timeout" "${bob[@]}")
fi

alice_pid=""
cleanup() {
  if [[ -n "$alice_pid" ]] && kill -0 "$alice_pid" 2>/dev/null; then
    kill "$alice_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
"${alice[@]}" >"$output/alice.log" 2>&1 &
alice_pid=$!
sleep 0.3
set +e
"${bob[@]}" >"$output/bob.log" 2>&1
bob_status=$?
wait "$alice_pid"
alice_status=$?
set -e
alice_pid=""
trap - EXIT INT TERM
if ((alice_status != 0 || bob_status != 0)); then
  echo "VIT_SMOKE_E2E status=FAIL alice=$alice_status bob=$bob_status" >&2
  tail -80 "$output/alice.log" >&2 || true
  tail -80 "$output/bob.log" >&2 || true
  exit 1
fi
cat "$output/bob.log"
if [[ ! -s "$output/secure_vit_smoke.txt" ]] ||
   ! grep -q 'VIT_SMOKE_RESULT status=PASS' "$output/bob.log"; then
  echo "VIT_SMOKE_E2E status=FAIL reason=missing_secure_completion_marker" >&2
  exit 1
fi

plain="$output/plain_vit_smoke.txt"
python3 "$repo/repro/plain_vit_e2e.py" \
  --model "$fixture/alice_model.txt" \
  --input "$fixture/bob_input.txt" \
  --output "$plain"
python3 - "$plain" "$output/secure_vit_smoke.txt" "${ATOL:-0.15}" <<'PY'
import sys
import numpy as np
plain = np.loadtxt(sys.argv[1]).reshape(-1)
secure = np.loadtxt(sys.argv[2]).reshape(-1)
atol = float(sys.argv[3])
if plain.shape != secure.shape or not np.isfinite(secure).all():
    raise SystemExit("VIT_SMOKE_VERIFY status=FAIL reason=shape_or_nonfinite")
error = np.abs(plain - secure)
print(f"VIT_SMOKE_VERIFY max_abs={error.max():.8g} mean_abs={error.mean():.8g} mse={np.mean((plain-secure)**2):.8g} atol={atol}")
if error.max() > atol or plain.argmax() != secure.argmax():
    raise SystemExit("VIT_SMOKE_VERIFY status=FAIL reason=logit_or_class_mismatch")
print(f"VIT_SMOKE_VERIFY status=PASS formal_result=no predicted_class={int(secure.argmax())}")
PY
echo "VIT_SMOKE_E2E status=PASS formal_result=no output=$output"
