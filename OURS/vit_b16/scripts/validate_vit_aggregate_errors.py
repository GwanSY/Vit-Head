#!/usr/bin/env python3
"""Fail-closed schema and norm checks for ViT aggregate-only evidence.

This validator is intentionally separate from ``vit_reference.py`` so the
offline numerical Graph can remain immutably SHA-bound by the oracle manifest
while report/parser hardening evolves independently.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Mapping


EXPECTED_FIELDS = [
    "stage", "elements", "observer_ell", "observer_scale",
    "clear_quantization_bound", "fixed_secure_max_abs",
    "fixed_secure_mean_abs", "fixed_secure_rmse", "clear_secure_max_abs",
    "clear_secure_mean_abs", "clear_secure_rmse", "time_ms", "bytes",
    "logical_rounds", "rmse_square_scale", "intermediate_reconstruction",
    "rmse_split_bits", "rmse_accumulator", "status",
]


def expected_stages(blocks: int) -> list[tuple[str, int]]:
    stages = [("patch_projection", 196 * 768)]
    for block in range(blocks):
        stages.extend((
            (f"block_{block}.attention_residual", 197 * 768),
            (f"block_{block}.ffn_residual", 197 * 768),
        ))
    stages.extend((("cls", 768), ("logits", 1000)))
    return stages


def metric(raw: Mapping[str | None, str | list[str] | None], prefix: str,
           stage: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in ("max_abs", "mean_abs", "rmse"):
        value = float(raw[f"{prefix}_{name}"])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{stage}.{prefix}.{name} is invalid")
        values[name] = value
    if (values["max_abs"] == 0.0 and
            (values["mean_abs"] != 0.0 or values["rmse"] != 0.0)):
        raise ValueError(
            f"{stage}.{prefix}: zero maximum requires zero mean and RMSE")
    tolerance = 16.0 * max(
        math.ulp(values["max_abs"]), math.ulp(values["mean_abs"]),
        math.ulp(values["rmse"]))
    if (values["mean_abs"] > values["max_abs"] + tolerance or
            values["rmse"] < values["mean_abs"] - tolerance or
            values["rmse"] > values["max_abs"] + tolerance):
        raise ValueError(
            f"{stage}.{prefix} violates mean_abs <= RMSE <= max_abs")
    return values


def validate(path: Path, blocks: int) -> int:
    expected = expected_stages(blocks)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_FIELDS:
            raise ValueError("secure aggregate CSV header/schema mismatch")
        rows = list(reader)
    if len(rows) != len(expected):
        raise ValueError(
            f"secure aggregate stage count mismatch: {len(rows)} != {len(expected)}")
    for index, (raw, (expected_name, expected_elements)) in enumerate(
            zip(rows, expected, strict=True)):
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError(f"secure aggregate row {index} has invalid width")
        name = raw["stage"]
        elements = int(raw["elements"])
        ell = int(raw["observer_ell"])
        scale = int(raw["observer_scale"])
        square_scale = int(raw["rmse_square_scale"])
        split_bits = int(raw["rmse_split_bits"])
        quantization = float(raw["clear_quantization_bound"])
        time_ms = float(raw["time_ms"])
        byte_count = int(raw["bytes"])
        logical_rounds = int(raw["logical_rounds"])
        if name != expected_name or elements != expected_elements:
            raise ValueError(
                f"secure aggregate stage mismatch at row {index}: {name}")
        if (ell != 43 or scale != 18 or square_scale != 36 or
                split_bits != (elements - 1).bit_length() or
                quantization != 0.5 / (1 << scale) or
                raw["rmse_accumulator"] != "exact_canonical_two_limb" or
                raw["intermediate_reconstruction"] != "false" or
                raw["status"] != "PASS"):
            raise ValueError(f"secure aggregate metadata mismatch at {name}")
        if (not math.isfinite(time_ms) or time_ms < 0.0 or byte_count < 0 or
                logical_rounds < 0):
            raise ValueError(f"secure aggregate accounting is invalid at {name}")
        metric(raw, "fixed_secure", name)
        metric(raw, "clear_secure", name)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--blocks", type=int, choices=range(1, 13), required=True)
    args = parser.parse_args()
    rows = validate(args.input, args.blocks)
    print("VIT_SECURE_AGGREGATE_SCHEMA=PASS "
          f"rows={rows} blocks={args.blocks} input={args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
