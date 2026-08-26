#!/usr/bin/env python3
"""Compare phase-A PyTorch logits with the canonical floating reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def logits(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("logits")
    if values is None and "clear" in data:
        values = data["clear"].get("logits")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} does not contain a logits array")
    return [float(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    lhs, rhs = logits(args.pytorch), logits(args.canonical)
    if len(lhs) != len(rhs):
        raise ValueError(f"logit length mismatch: {len(lhs)} != {len(rhs)}")
    maximum = max(abs(a - b) for a, b in zip(lhs, rhs))
    lhs_top1 = max(range(len(lhs)), key=lhs.__getitem__)
    rhs_top1 = max(range(len(rhs)), key=rhs.__getitem__)
    passed = maximum <= args.atol and lhs_top1 == rhs_top1
    print(f"VIT_FRAMEWORK_REFERENCE_MAX_ABS_ERROR={maximum:.12g}")
    print(f"VIT_FRAMEWORK_REFERENCE_TOP1={lhs_top1}")
    print(f"VIT_CANONICAL_REFERENCE_TOP1={rhs_top1}")
    print(f"VIT_FRAMEWORK_REFERENCE_MATCH={'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
