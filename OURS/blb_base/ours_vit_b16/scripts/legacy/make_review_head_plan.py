#!/usr/bin/env python3
"""Generate explicit A1--A4 review plans without calibration or inference.

This utility is intentionally separate from the A2 selector.  Its output is
only for path wiring tests: A1 keeps original head order; A2--A4 group physical
heads and record the synchronized original-head permutation.  Standalone A3
tests the real grouped Softmax/ReLU conversion path.  A4 adds native CKKS
Linear heads; it is the first complete stage that can consume those heads.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path


PROTOCOLS = ("softmax_mpc", "relu_norm_mpc", "he_linear")
LAYERS = 12
HEADS = 12


def assignment_for_original_head(head: int, pattern: str) -> tuple[str, float]:
    if pattern == "mixed":
        protocol = PROTOCOLS[head % len(PROTOCOLS)]
    elif pattern == "softmax11_relu1":
        protocol = "relu_norm_mpc" if head == HEADS - 1 else "softmax_mpc"
    else:
        protocol = {"softmax": "softmax_mpc", "relu": "relu_norm_mpc",
                    "linear": "he_linear"}[pattern]
    # A deterministic public coefficient for wiring tests only.  A production
    # value must come from calibrate_heterogeneous_heads.py.
    alpha = 1.0 / 197.0 if protocol == "he_linear" else 1.0
    return protocol, alpha


def physical_order(ablation: str, pattern: str) -> list[int]:
    if ablation == "a1" or pattern != "mixed":
        return list(range(HEADS))
    return sorted(range(HEADS),
                  key=lambda head: PROTOCOLS.index(
                      assignment_for_original_head(head, pattern)[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=("a1", "a2", "a3", "a4"),
                        required=True)
    parser.add_argument("--pattern", choices=(
        "mixed", "softmax", "relu", "linear", "softmax11_relu1"),
        default="mixed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    has_linear = any(assignment_for_original_head(head, args.pattern)[0] ==
                     "he_linear" for head in range(HEADS))
    if args.ablation == "a3" and has_linear:
        parser.error("standalone A3 accepts only MPC paths; use A4 for native "
                     "CKKS Linear heads")
    if args.ablation == "a4" and not has_linear:
        parser.error("A4 requires at least one native CKKS Linear head")
    order = physical_order(args.ablation, args.pattern)
    lines = [
        "OURS_VIT_HEAD_PLAN_V1",
        "layers=12,heads=12,logical_tokens=197,physical_tokens=256",
        "layer,physical_head,original_head,protocol,alpha",
    ]
    for layer in range(LAYERS):
        for physical_head, original_head in enumerate(order):
            protocol, alpha = assignment_for_original_head(original_head,
                                                            args.pattern)
            lines.append(
                f"{layer},{physical_head},{original_head},{protocol},{alpha:.17g}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OURS_REVIEW_HEAD_PLAN status=PASS ablation={args.ablation.upper()} "
          f"pattern={args.pattern}")
    print(f"OURS_REVIEW_HEAD_PLAN_FILE={args.output}")


if __name__ == "__main__":
    main()
