#!/usr/bin/env python3
"""Strict acceptance gate for the formal BLB layer-0 run logs."""

import argparse
import math
import re
from pathlib import Path


STAGES = (
    "fgf1_ln_attention",
    "fgf2_softmax",
    "fgf3_softmax_v_ln",
    "fgf4_ln_ffn_gelu",
    "fgf5_ffn2_ln",
)
METRICS = (
    "ROTATION_COUNT",
    "CT_CT_MULTIPLY_COUNT",
    "CT_PT_MULTIPLY_COUNT",
    "CKKS_TO_MPC_COUNT",
    "MPC_TO_CKKS_COUNT",
    "TRUNCATION_COUNT",
    "COMMUNICATION_BYTES",
    "END_TO_END_LATENCY_SECONDS",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()
    texts = []
    for name in ("alice.log", "bob.log"):
        path = args.log_dir / name
        if not path.is_file():
            raise RuntimeError(f"missing party log: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        if "formal_result=no" in text:
            raise RuntimeError(f"non-formal result marker found in {path}")
        texts.append(text)
    combined = "\n".join(texts)
    for stage in STAGES:
        match = re.search(
            rf"ORACLE_STAGE name={stage} .*?A_B_MSE=([0-9.eE+-]+) "
            rf".*?B_C_MSE=([0-9.eE+-]+) "
            rf"B_C_MAX_ABS=([0-9.eE+-]+) B_C_MAE=([0-9.eE+-]+)",
            combined,
        )
        if not match:
            raise RuntimeError(f"missing three-oracle result for {stage}")
        fixed_secure = float(match.group(2))
        max_abs = float(match.group(3))
        mae = float(match.group(4))
        if not all(math.isfinite(value) and value >= 0.0
                   for value in (fixed_secure, max_abs, mae)):
            raise RuntimeError(f"{stage} contains an invalid secure aggregate")
        if fixed_secure >= 1e-11:
            raise RuntimeError(f"{stage} fixed-clear/secure MSE={fixed_secure} >= 1e-11")
        tolerance = max(1e-18, max_abs * 1e-12)
        if mae > max_abs + tolerance or fixed_secure > max_abs * max_abs + tolerance:
            raise RuntimeError(f"{stage} aggregate metrics violate max-error bounds")

        aggregate = re.search(
            rf"ORACLE_AGGREGATE name={stage} accumulator_ell=63 "
            rf"aggregation_blocks=1 max_abs_raw=([0-9]+) "
            rf"max_square_safe_raw=([0-9]+) "
            rf"B_C_MAX_ABS=([0-9.eE+-]+) B_C_MAE=([0-9.eE+-]+) "
            rf"square_safe=1 opened_scalars=([0-9]+) "
            rf"intermediate_tensor_opened=0",
            combined,
        )
        if not aggregate:
            raise RuntimeError(f"missing scalar-only aggregate diagnostic for {stage}")
        max_abs_raw = int(aggregate.group(1))
        max_square_safe_raw = int(aggregate.group(2))
        aggregate_max_abs = float(aggregate.group(3))
        aggregate_mae = float(aggregate.group(4))
        opened_scalars = int(aggregate.group(5))
        if opened_scalars != 3 or max_abs_raw > max_square_safe_raw:
            raise RuntimeError(f"{stage} aggregate scalar metadata is invalid")
        if not math.isclose(aggregate_max_abs, max_abs,
                            rel_tol=1e-12, abs_tol=1e-18):
            raise RuntimeError(f"{stage} max-absolute diagnostics disagree")
        if not math.isclose(aggregate_mae, mae, rel_tol=1e-12, abs_tol=1e-18):
            raise RuntimeError(f"{stage} MAE diagnostics disagree")
        expected_max_abs = max_abs_raw / float(1 << 12)
        if not math.isclose(max_abs, expected_max_abs, rel_tol=1e-12, abs_tol=1e-18):
            raise RuntimeError(f"{stage} max-absolute scalar/raw mismatch")
    for metric in METRICS:
        if not re.search(rf"^{metric} value=", combined, re.MULTILINE):
            raise RuntimeError(f"missing metric: {metric}")
    if not re.search(
        r"FINAL_LAYERNORM status=PASS real_ln2_parameters=1 ckks_affine=1",
        combined,
    ):
        raise RuntimeError("final real LayerNorm CKKS affine was not completed")
    if "SINGLE_LAYER_SECURITY status=PASS" not in combined:
        raise RuntimeError("single-layer security gate did not pass")
    print("SINGLE_LAYER_ACCEPTANCE status=PASS figure10_blocks=5 mse_limit=1e-11")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SINGLE_LAYER_ACCEPTANCE status=BLOCKED reason={exc}")
        raise SystemExit(2)
