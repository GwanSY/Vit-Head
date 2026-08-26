#!/usr/bin/env bash
set -euo pipefail

# Run the faithful original-BLB CKKS/MPC ViT-B/16 baseline.  This is the sole
# ViT secure entry point; it enables the existing local truncation mode and
# does not select any heterogeneous policy.  Set BLB_BUILD to the SCI build
# containing ckks_vit_b16_main.

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

BLB_BUILD=${BLB_BUILD:-"${repo}/build_blb_formal/sci"}
BLB_BINARY=${BLB_BINARY:-"${BLB_BUILD}/bin/ckks_vit_b16_main"}
BLB_MODEL=${BLB_MODEL:-"${repo}/checkpoints/vit_b16/bolt_model.txt"}
BLB_INPUT=${BLB_INPUT:-"${repo}/outputs/vit_b16_gpu/bob_image.txt"}
BLB_OUTPUT=${BLB_OUTPUT:-"${repo}/outputs/blb_vit_ckks_gpu2_lan"}
BLB_ADDRESS=${BLB_ADDRESS:-127.0.0.1}
BLB_PORT=${BLB_PORT:-8000}
BLB_DEVICE=${BLB_DEVICE:-2}
BLB_STOP_AFTER_BLOCK=${BLB_STOP_AFTER_BLOCK:-12}
# OT selection is compile-time.  The formal build wrapper defaults this to
# VOLE; set USE_VOLE=OFF when producing an IKNP-compatible binary.
BLB_USE_VOLE=${BLB_USE_VOLE:-ON}

cache_file="${BLB_BUILD}/CMakeCache.txt"
if [[ -f "${cache_file}" ]]; then
  cached_use_vole="$(sed -n 's/^USE_VOLE:BOOL=//p' "${cache_file}" | head -n 1)"
  expected_use_vole="${BLB_USE_VOLE^^}"
  if [[ -z "${cached_use_vole}" || "${cached_use_vole}" != "${expected_use_vole}" ]]; then
    cached_use_vole="${cached_use_vole:-UNSET}"
    echo "BLB_VIT_RUN status=FAIL reason=ot_mode_mismatch expected=${expected_use_vole} cached=${cached_use_vole}" >&2
    if [[ "${cached_use_vole}" == "UNSET" ]]; then
      echo "Reconfigure/rebuild with USE_VOLE=${expected_use_vole}; this cache has no OT mode marker." >&2
    else
      echo "Rebuild with USE_VOLE=${expected_use_vole}, or set BLB_USE_VOLE=${cached_use_vole} to run the existing binary." >&2
    fi
    exit 2
  fi
fi

if [[ ! -x "${BLB_BINARY}" ]]; then
  echo "missing ${BLB_BINARY}; configure/build the SCI target ckks_vit_b16_main first" >&2
  exit 2
fi
mkdir -p "${BLB_OUTPUT}"

export CUDA_VISIBLE_DEVICES=${BLB_DEVICE}
export LD_LIBRARY_PATH="${BLB_BUILD}/extern/phantom/lib:${BLB_BUILD}/extern/seal/lib:/usr/local/cuda-12.3/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

echo "BLB_VIT_RUN protocol=original_blb_ckks_mpc device=${BLB_DEVICE} ot_default=${BLB_USE_VOLE}"
echo "BLB_VIT_RUN network=LAN_contract address=${BLB_ADDRESS} port=${BLB_PORT}"
echo "BLB_VIT_RUN stop_after_block=${BLB_STOP_AFTER_BLOCK}"

"${BLB_BINARY}" --party 1 --port "${BLB_PORT}" --address "${BLB_ADDRESS}" \
  --model "${BLB_MODEL}" --input "${BLB_INPUT}" --output "${BLB_OUTPUT}" \
  --backend gpu --stop-after-block "${BLB_STOP_AFTER_BLOCK}" \
  >"${BLB_OUTPUT}/alice.log" 2>&1 &
alice_pid=$!

"${BLB_BINARY}" --party 2 --port "${BLB_PORT}" --address "${BLB_ADDRESS}" \
  --model "${BLB_MODEL}" --input "${BLB_INPUT}" --output "${BLB_OUTPUT}" \
  --backend gpu --stop-after-block "${BLB_STOP_AFTER_BLOCK}" \
  >"${BLB_OUTPUT}/bob.log" 2>&1 &
bob_pid=$!

cleanup() {
  for pid in "${alice_pid:-}" "${bob_pid:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

alice_status=0
bob_status=0
wait "${alice_pid}" || alice_status=$?
if [[ "${alice_status}" -ne 0 ]]; then
  kill "${bob_pid}" 2>/dev/null || true
fi
wait "${bob_pid}" || bob_status=$?
if [[ "${bob_status}" -ne 0 ]]; then
  kill "${alice_pid}" 2>/dev/null || true
fi

if [[ "${alice_status}" -ne 0 || "${bob_status}" -ne 0 ]]; then
  echo "BLB_VIT_RUN status=FAILED alice=${alice_status} bob=${bob_status}" >&2
  exit 1
fi

python3 "${repo}/vit_b16/scripts/summarize_blb_vit_baseline.py" \
  --run-dir "${BLB_OUTPUT}"
echo "BLB_VIT_RUN status=PASS summary=${BLB_OUTPUT}/baseline_summary.md"
