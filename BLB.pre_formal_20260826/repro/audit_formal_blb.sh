#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
expected_host="${BLB_EXPECTED_HOST:-/home/dm/mind/Vit-Head/BLB.pre_formal_20260826}"
expected_container="/workspace/BLB"
bert_src="$repo/SCI/tests/bert_large_bolt"
bert_checkpoint="${BERT_CHECKPOINT:-$repo/checkpoints/official-blb-mrpc/weights_txt}"
blocked=0

# The official image does not ship ripgrep.  Keep the audit usable there
# without adding another project dependency; on the host, native rg remains
# the fast path.
if ! command -v rg >/dev/null 2>&1; then
  rg() {
    local quiet=0
    local line_number=0
    local pattern
    local -a excludes=()
    local -a paths=()
    local -a grep_args=(-R -E)

    while (($#)); do
      case "$1" in
        -q) quiet=1; shift ;;
        -n) line_number=1; shift ;;
        --glob)
          case "${2:-}" in
            '!**/CMakeFiles/**') excludes+=(--exclude-dir=CMakeFiles) ;;
            '!**/Makefile') excludes+=(--exclude=Makefile) ;;
            '!**/cmake_install.cmake') excludes+=(--exclude=cmake_install.cmake) ;;
          esac
          shift 2 ;;
        --) shift; break ;;
        -*) shift ;;
        *) break ;;
      esac
    done

    while (($#)); do
      case "$1" in
        --glob)
          case "${2:-}" in
            '!**/CMakeFiles/**') excludes+=(--exclude-dir=CMakeFiles) ;;
            '!**/Makefile') excludes+=(--exclude=Makefile) ;;
            '!**/cmake_install.cmake') excludes+=(--exclude=cmake_install.cmake) ;;
          esac
          shift 2 ;;
        -q|-n) shift ;;
        -*) shift ;;
        *)
          if [[ -z "${pattern:-}" ]]; then
            pattern="$1"
          else
            paths+=("$1")
          fi
          shift ;;
      esac
    done
    : "${pattern:?grep pattern missing}"
    if ((${#paths[@]} == 0)); then
      paths=(.)
    fi
    if ((quiet)); then
      grep_args+=(-q)
    fi
    if ((line_number)); then
      grep_args+=(-n)
    fi
    grep "${grep_args[@]}" "${excludes[@]}" "$pattern" "${paths[@]}"
  }
fi

pass() { echo "BLB_FORMAL_AUDIT item=$1 status=PASS${2:+ detail=$2}"; }
block() { echo "BLB_FORMAL_AUDIT item=$1 status=BLOCKED${2:+ detail=$2}"; blocked=1; }

if [[ "$repo" == "$expected_host" || "$repo" == "$expected_container" ]]; then
  pass repo_root "$repo"
else
  block repo_root "expected=${expected_host} or ${expected_container} actual=$repo"
fi

for path in "$bert_checkpoint"; do
  case "$path" in
    "$repo"/*) ;;
    *) block checkpoint_boundary "outside_repo=$path" ;;
  esac
done

placeholder_count="$(
  { rg -n 'fake weight|fake_vec|暂时令r=0|fake实现|simulate residual' \
      "$bert_src/ckks_bert.cpp" "$bert_src/ckks_bert_basic.cpp" \
      "$bert_src/linear_ckks.cpp" || true; } | wc -l | tr -d ' '
)"
if [[ "$placeholder_count" == "0" ]]; then
  pass placeholder_gate count=0
else
  block placeholder_gate "count=$placeholder_count"
fi

external_refs="$(
  rg -n '/ezpc_dir|/home/CKKS-MPC|\.\./\.\./dataset' \
    "$bert_src" \
    --glob '!**/CMakeFiles/**' \
    --glob '!**/Makefile' \
    --glob '!**/cmake_install.cmake' || true
)"
if [[ -z "$external_refs" ]]; then
  pass source_boundary no_external_project_paths
  mkdir -p "$repo/outputs"
  : >"$repo/outputs/formal_audit_external_paths.log"
else
  block source_boundary "see=$repo/outputs/formal_audit_external_paths.log"
  mkdir -p "$repo/outputs"
  printf '%s\n' "$external_refs" >"$repo/outputs/formal_audit_external_paths.log"
fi

for symbol in \
  bert_cipher_plain_bsgs \
  bert_cipher_cipher_cross_packing \
  softmax_v \
  layer_norm_affine_ckks \
  ckks_to_mpc \
  mpc_to_ckks \
  field_to_ring \
  ring_to_field; do
  if rg -q "$symbol" "$bert_src"; then
    pass "symbol_$symbol" present
  else
    block "symbol_$symbol" missing
  fi
done

if rg -q '#define ATTENTION_LAYERS 1' "$bert_src/linear_ckks.h"; then
  pass bert_execution_depth "layers=1_pending_single_layer_acceptance_model_metadata=12"
else
  block bert_execution_depth "expected=1_until_acceptance_actual=$(sed -n 's/^#define ATTENTION_LAYERS //p' "$bert_src/linear_ckks.h")"
fi

if [[ -d "$bert_checkpoint" ]] && find "$bert_checkpoint" -type f -print -quit | grep -q .; then
  pass bert_checkpoint "$bert_checkpoint"
else
  block bert_checkpoint "missing=$bert_checkpoint"
fi
if rg -q 'sample_poly_uniform' "$bert_src/ckks_bert_basic.cpp" &&
   rg -q 'field_to_ring_calls\+\+' "$bert_src/ckks_bert_basic.cpp"; then
  pass algorithm1_ckks_to_mpc "uniform_current_RNS_ring_and_real_field_to_ring"
else
  block algorithm1_ckks_to_mpc missing_secure_conversion
fi

for pp in pp_1 pp_2 pp_3 pp_4; do
  if rg -q "${pp}\[i\] = params_preprocessing" "$bert_src/linear_ckks.cpp"; then
    pass "preprocess_$pp" enabled
  else
    block "preprocess_$pp" disabled
  fi
done

metric_missing=0
for metric in STAGE_MSE ROTATION_COUNT CKKS_TO_MPC_COUNT MPC_TO_CKKS_COUNT TRUNCATION_COUNT; do
  if rg -q "$metric" "$bert_src"; then
    pass "metric_$metric" present
  else
    block "metric_$metric" missing
    metric_missing=1
  fi
done

if rg -q 'constexpr bool kFormalLayer0InputReady = false' "$bert_src/ckks_bert_large_main.cpp" &&
   rg -q 'FORMAL_GATE status=BLOCKED reason=real_layer0_input_not_implemented' "$bert_src/ckks_bert_large_main.cpp"; then
  pass formal_input_fail_closed compile_time_gate_before_network
elif rg -q 'constexpr bool kFormalLayer0InputReady = true' "$bert_src/ckks_bert_large_main.cpp" &&
     rg -q 'read_signed_tensor' "$bert_src/ckks_bert_large_main.cpp" &&
     rg -q 'attention_from_real_input' "$bert_src/ckks_bert_large_main.cpp" &&
     rg -q 'raw_input' "$bert_src/ckks_bert.cpp" &&
     rg -q 'sci::BOB' "$bert_src/ckks_bert_large_main.cpp"; then
  pass formal_input_fail_closed real_layer0_input_wired_and_shape_checked
else
  block formal_input_fail_closed missing_non_bypassable_gate
fi

if rg -q 'constexpr bool kFormalThreeOracleReady = true' "$bert_src/ckks_bert_large_main.cpp" &&
   rg -q 'load_oracle_references' "$bert_src/ckks_bert_large_main.cpp" &&
   rg -q 'secure_stage_mse' "$bert_src/ckks_bert_large_main.cpp" &&
   rg -q 'three-oracle manifest is missing' "$bert_src/ckks_bert_large_main.cpp"; then
  pass three_oracle_fail_closed runtime_manifest_gate_before_network
elif rg -q 'constexpr bool kFormalThreeOracleReady = false' "$bert_src/ckks_bert_large_main.cpp" &&
     rg -q 'FORMAL_GATE status=BLOCKED reason=three_oracle_not_implemented' "$bert_src/ckks_bert_large_main.cpp"; then
  pass three_oracle_fail_closed compile_time_gate_before_network
else
  block three_oracle_fail_closed missing_non_bypassable_gate
fi

if rg -q 'FixArray (w|b) = fpmath->fix->input\(party, x_flat_avg.size, 1' \
    "$bert_src/ckks_bert_basic.cpp"; then
  block layernorm_real_parameters placeholder_weight_or_bias_detected
else
  pass layernorm_real_parameters missing_parameters_are_rejected
fi

if [[ ! -e "$repo/vit_adapter" && ! -e "$repo/repro/run_vit_e2e.sh" ]]; then
  pass smoke_path_removed no_independent_vit_entrypoint
else
  block smoke_path_removed obsolete_independent_vit_entrypoint_present
fi

if [[ "$blocked" == "0" ]]; then
  echo "BLB_FORMAL_AUDIT status=PASS"
else
  echo "BLB_FORMAL_AUDIT status=BLOCKED report=$repo/docs/BLB_FORMAL_AUDIT_ZH.md"
  exit 2
fi
