#!/usr/bin/env python3
"""Lightweight regression for the formal BERT public layout adapters.

This mirrors the index equations in ckks_bert.cpp and checks them against
the original BOLT ``plain_col_packing_postprocess(..., false, data)`` map.
It contains no model values and performs no protocol operation.
"""

from __future__ import annotations

import re
from pathlib import Path


def token_to_feature(values: list[int], tokens: int, features: int) -> list[int]:
    output = [0] * len(values)
    for token in range(tokens):
        for feature in range(features):
            output[feature * tokens + token] = values[token * features + feature]
    return output


def feature_to_token(values: list[int], tokens: int, features: int) -> list[int]:
    output = [0] * len(values)
    for feature in range(features):
        for token in range(tokens):
            output[token * features + feature] = values[feature * tokens + token]
    return output


def token_to_bsgs_output(
    values: list[int], tokens: int, features: int, slots: int
) -> list[int]:
    output = [0] * len(values)
    features_per_half = slots // (2 * tokens)
    for ciphertext in range(len(values) // slots):
        for slot_index in range(slots):
            token = slot_index % tokens
            upper_half = slot_index >= slots // 2
            local_feature = (slot_index % (slots // 2)) // tokens
            feature = (
                ciphertext * features_per_half
                + local_feature
                + (features // 2 if upper_half else 0)
            )
            output[ciphertext * slots + slot_index] = values[
                token * features + feature
            ]
    return output


def original_bolt_postprocess_false(
    values: list[int], tokens: int, features: int, slots: int
) -> list[int]:
    """Exact index map from Linear::plain_col_packing_postprocess(false)."""
    output = [0] * len(values)
    for ciphertext in range(len(values) // slots):
        for slot_index in range(slots):
            feature = slot_index // tokens
            token = slot_index % tokens
            if slot_index >= slots // 2:
                feature -= slots // tokens // 2
                feature += features // 2
            feature += ciphertext * slots // tokens // 2
            output[token * features + feature] = values[
                ciphertext * slots + slot_index
            ]
    return output


def check(tokens: int, features: int, slots: int) -> None:
    total = tokens * features
    source = list(range(total))
    column = token_to_feature(source, tokens, features)
    assert feature_to_token(column, tokens, features) == source
    packed = token_to_bsgs_output(source, tokens, features, slots)
    assert sorted(packed) == source
    assert (
        original_bolt_postprocess_false(packed, tokens, features, slots)
        == source
    )
    print(
        "LAYOUT_ADAPTER_TEST status=PASS"
        f" tokens={tokens} features={features} slots={slots}"
        " transpose_roundtrip=1 bsgs_roundtrip=1 permutation_coverage=1"
    )


def check_real_pre_gelu_wiring() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "SCI/tests/bert_large_bolt/ckks_bert.cpp"
    ).read_text(encoding="utf-8")
    required = (
        "pre_gelu_cts[i] = linear3_output[i].hcipher();",
        "prepare_host_conversion_from_actual_scale(\n"
        "            *this, pre_gelu_cts, kFormalConversionScaleBits, he,",
        "mask_ckks_to_mpc_inplace(pre_gelu_cts, pre_gelu_pts, he);",
        "vec_pre_gelu = ckks_to_mpc(",
        "x_array = bsgs_output_to_token_major(\n"
        "        pre_gelu_packed, linear.data_lin3, \"ffn1_pre_gelu_to_mpc\");",
    )
    for snippet in required:
        assert snippet in source, f"missing real pre-GeLU wiring: {snippet!r}"

    snapshot = source.index("FORMAL_PRE_GELU_SOURCE status=PASS")
    polynomial = source.index(
        "eval_gelu_poly(gelu_output, linear3_output", snapshot
    )
    masked_export = source.index(
        "mask_ckks_to_mpc_inplace(pre_gelu_cts, pre_gelu_pts, he);",
        polynomial,
    )
    assert snapshot < polynomial < masked_export

    forbidden = (
        r"x_array\s*=\s*fpmath->fix->input\s*\(\s*party\s*,\s*gelu_input_size",
        r"FixArray\s+x_flat_avg_packed\s*=\s*fpmath->fix->input",
        r"sigma_flat\s*=\s*fpmath->fix->input\s*\(\s*party",
        r"FixArray\s+ffn_residual_column\s*=\s*fpmath->fix->input",
    )
    for pattern in forbidden:
        assert re.search(pattern, source) is None, (
            f"synthetic/re-inputed local output buffer remains: {pattern}"
        )
    print(
        "PRE_GELU_WIRING_TEST status=PASS real_linear3_ciphertext=1"
        " masked_algorithm1=1 bsgs_to_token_major=1"
        " constant_placeholder=0 local_share_reinput=0"
    )


def main() -> int:
    # he2 linear2 output, he3 linear3 output, and he4 linear4 output.
    check(128, 768, 16384)
    check(128, 3072, 16384)
    check(128, 768, 8192)
    check_real_pre_gelu_wiring()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
