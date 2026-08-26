#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
phantom="$repo/SCI/extern/phantom"
expected_blb="d9af284b62a2dad398d2bf4928adae3ddd6cf3b2"
expected_phantom="d09c110d2d3a48a45fe1796d8ab6f0d1dabd6b15"

fail=0

check_equal() {
  local label="$1"
  local actual="$2"
  local expected="$3"
  if [[ "$actual" == "$expected" ]]; then
    echo "SOURCE_CHECK item=$label status=PASS value=$actual"
  else
    echo "SOURCE_CHECK item=$label status=FAIL expected=$expected actual=$actual"
    fail=1
  fi
}

check_pattern() {
  local label="$1"
  local pattern="$2"
  local path="$3"
  if grep -Eq -- "$pattern" "$path"; then
    echo "SOURCE_CHECK item=$label status=PASS"
  else
    echo "SOURCE_CHECK item=$label status=FAIL path=$path"
    fail=1
  fi
}

check_allowed_diff() {
  local checkout="$1"
  local label="$2"
  shift 2
  local modified unexpected=0 path allowed
  modified="$(git -C "$checkout" diff --name-only HEAD -- .)"
  if [[ -z "$modified" ]]; then
    echo "SOURCE_CHECK item=$label status=PASS detail=clean"
    return
  fi
  while IFS= read -r path; do
    allowed=0
    for expected in "$@"; do
      if [[ "$path" == "$expected" ]]; then
        allowed=1
        break
      fi
    done
    if [[ "$allowed" == "0" ]]; then
      echo "SOURCE_CHECK item=$label status=FAIL unexpected_modified=$path"
      unexpected=1
    fi
  done <<<"$modified"
  if [[ "$unexpected" == "0" ]]; then
    echo "SOURCE_CHECK item=$label status=PASS detail=known_build_portability_patch"
  else
    fail=1
  fi
}

check_equal "blb_commit" "$(git -C "$repo" rev-parse HEAD)" "$expected_blb"
check_allowed_diff "$repo" "blb_tracked_tree" \
  ".gitignore" \
  "README.md" \
  "SCI/cmake/SCIConfig.cmake" \
  "SCI/src/BuildingBlocks/truncation.cpp" \
  "SCI/src/BuildingBlocks/truncation.h" \
  "SCI/src/CMakeLists.txt" \
  "SCI/src/GC/aes_opt.h" \
  "SCI/src/GC/mitccrh.h" \
  "SCI/src/LinearHE/CMakeLists.txt" \
  "SCI/src/LinearHE/evaluator-wrapper.cpp" \
  "SCI/src/LinearHE/evaluator-wrapper.h" \
  "SCI/src/LinearHE/unified-he.cpp" \
  "SCI/src/LinearHE/unified-he.h" \
  "SCI/tests/bert_large_bolt/bert_utils.cpp" \
  "SCI/tests/bert_large_bolt/bert_utils.h" \
  "SCI/tests/bert_large_bolt/bolt_bert_large.cpp" \
  "SCI/tests/bert_large_bolt/ckks_bert.cpp" \
  "SCI/tests/bert_large_bolt/ckks_bert.h" \
  "SCI/tests/bert_large_bolt/ckks_bert_basic.cpp" \
  "SCI/tests/bert_large_bolt/ckks_bert_large_main.cpp" \
  "SCI/tests/bert_large_bolt/he.cpp" \
  "SCI/tests/bert_large_bolt/he.h" \
  "SCI/tests/bert_large_bolt/linear_ckks.cpp" \
  "SCI/tests/bert_large_bolt/linear_ckks.h" \
  "SCI/extern/phantom"

if [[ ! -d "$phantom/.git" ]]; then
  echo "SOURCE_CHECK item=phantom_checkout status=FAIL path=$phantom"
  fail=1
else
  check_equal "phantom_commit" \
    "$(git -C "$phantom" rev-parse HEAD)" "$expected_phantom"
  check_allowed_diff "$phantom" "phantom_tracked_tree" \
    "CMakeLists.txt" \
    "src/CMakeLists.txt"
fi

check_pattern "gpu_build_option" \
  'option\(USE_HE_GPU "Use HE-GPU backend" ON\)' \
  "$repo/SCI/CMakeLists.txt"
check_pattern "audited_host_entrypoint" \
  'Bert_ckks bert_ckks\(party, port, address, LOCATION::HOST' \
  "$repo/SCI/tests/bert_large_bolt/ckks_bert_large_main.cpp"
check_pattern "phantom_context" \
  'PhantomContext' \
  "$repo/SCI/tests/bert_large_bolt/he.h"

placeholder_count="$(
  { grep -En \
      'fake weight|fake_vec|暂时令r=0' \
      "$repo/SCI/tests/bert_large_bolt/linear_ckks.cpp" \
      "$repo/SCI/tests/bert_large_bolt/ckks_bert.cpp" \
      "$repo/SCI/tests/bert_large_bolt/ckks_bert_basic.cpp" || true; } |
    wc -l |
    tr -d ' '
)"

if [[ "$placeholder_count" == "0" ]]; then
  echo "CORRECTNESS_GATE status=PASS placeholders=0"
else
  echo "CORRECTNESS_GATE status=BLOCKED placeholders=$placeholder_count"
  echo "CORRECTNESS_GATE reason=formal_source_contains_rejected_placeholder_pattern"
fi

if [[ "$fail" != "0" ]]; then
  exit 1
fi

if [[ "${REQUIRE_CORRECTNESS:-0}" == "1" && "$placeholder_count" != "0" ]]; then
  exit 2
fi
