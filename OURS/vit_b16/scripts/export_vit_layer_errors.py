#!/usr/bin/env python3
"""Export the 27 privacy-preserving ViT validation error summaries.

The input reference report contains clear/fixed comparisons computed offline
and secure comparisons opened only as aggregate scalars.  This script never
loads, reconstructs, or writes a secure intermediate tensor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REFERENCE_FORMAT = "BOLT_VIT_REFERENCE_REPORT_V1"


def expected_stages() -> list[str]:
    stages = ["patch_projection"]
    for block in range(12):
        stages.extend((
            f"block_{block}.attention_residual",
            f"block_{block}.ffn_residual",
        ))
    stages.extend(("cls", "logits"))
    return stages


def require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def require_metric(value: Any, field: str) -> dict[str, float | int]:
    row = require_object(value, field)
    result: dict[str, float | int] = {}
    elements = row.get("elements")
    if isinstance(elements, bool) or not isinstance(elements, int) or elements <= 0:
        raise ValueError(f"{field}.elements must be a positive integer")
    result["elements"] = elements
    for name in ("max_abs", "mean_abs", "rmse"):
        raw = row.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{field}.{name} must be numeric")
        number = float(raw)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{field}.{name} must be finite and nonnegative")
        result[name] = number
    if (result["max_abs"] == 0.0 and
            (result["mean_abs"] != 0.0 or result["rmse"] != 0.0)):
        raise ValueError(f"{field}: zero maximum requires zero mean and RMSE")
    tolerance = 16.0 * max(
        math.ulp(float(result["max_abs"])),
        math.ulp(float(result["mean_abs"])),
        math.ulp(float(result["rmse"])))
    if (result["mean_abs"] > result["max_abs"] + tolerance or
            result["rmse"] < result["mean_abs"] - tolerance or
            result["rmse"] > result["max_abs"] + tolerance):
        raise ValueError(f"{field} violates mean_abs <= RMSE <= max_abs")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.reference_report.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("format") != REFERENCE_FORMAT:
        raise ValueError("unexpected reference report format")
    if report.get("execution", {}).get("executed_blocks") != 12:
        raise ValueError("a complete 12-block report is required")
    if report.get("secure", {}).get("layerwise_error_status") != "AGGREGATE_ONLY":
        raise ValueError("secure layerwise errors must be AGGREGATE_ONLY")
    if report.get("secure", {}).get("intermediate_reconstruction") is not False:
        raise ValueError("secure intermediate reconstruction must be false")

    operators = report.get("operators")
    if not isinstance(operators, list):
        raise ValueError("operators must be an array")
    secure_operators: dict[str, dict[str, Any]] = {}
    for index, raw_operator in enumerate(operators):
        operator = require_object(raw_operator, f"operators[{index}]")
        secure = operator.get("secure")
        # Non-observed operators deliberately carry a NOT_AVAILABLE marker.
        # Only rows with the aggregate fixed/secure comparison are exportable.
        if not isinstance(secure, dict) or "fixed_secure" not in secure:
            continue
        name = operator.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"operators[{index}].name must be nonempty")
        if name in secure_operators:
            raise ValueError(f"duplicate secure operator: {name}")
        secure_operators[name] = operator

    stages = expected_stages()
    if set(secure_operators) != set(stages):
        missing = sorted(set(stages) - set(secure_operators))
        extra = sorted(set(secure_operators) - set(stages))
        raise ValueError(f"secure stage mismatch: missing={missing}, extra={extra}")

    fieldnames = [
        "stage", "shape", "observed_shape", "observation_scope",
        "scale", "bit_length", "packing", "elements",
        "observer_ell", "observer_scale", "rmse_square_scale",
        "rmse_split_bits", "rmse_accumulator",
        "clear_fixed_max_abs", "clear_fixed_mean_abs", "clear_fixed_rmse",
        "fixed_secure_max_abs", "fixed_secure_mean_abs", "fixed_secure_rmse",
        "clear_secure_max_abs", "clear_secure_mean_abs", "clear_secure_rmse",
        "intermediate_reconstruction", "fixed_secure_acceptance",
        "clear_fixed_numeric_acceptance",
    ]
    rows: list[dict[str, Any]] = []
    for stage in stages:
        operator = secure_operators[stage]
        clear_fixed = require_metric(operator.get("clear_fixed"),
                                     f"{stage}.clear_fixed")
        secure = require_object(operator.get("secure"), f"{stage}.secure")
        if secure.get("intermediate_reconstruction") is not False:
            raise ValueError(f"{stage} reconstructed an intermediate tensor")
        fixed_secure = require_metric(secure.get("fixed_secure"),
                                      f"{stage}.fixed_secure")
        clear_secure = require_metric(secure.get("clear_secure"),
                                      f"{stage}.clear_secure")
        elements = clear_fixed["elements"]
        if fixed_secure["elements"] != elements or clear_secure["elements"] != elements:
            raise ValueError(f"{stage} comparison element counts disagree")
        expected_split = (int(elements) - 1).bit_length()
        if (secure.get("observer_ell") != 43 or
                secure.get("observer_scale") != 18 or
                secure.get("rmse_square_scale") != 36 or
                secure.get("rmse_split_bits") != expected_split or
                secure.get("rmse_accumulator") !=
                "exact_canonical_two_limb"):
            raise ValueError(f"{stage} secure aggregate metadata mismatch")
        shape = operator.get("shape")
        if (not isinstance(shape, list) or not shape or
                any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                    for v in shape)):
            raise ValueError(f"{stage}.shape must contain positive integers")
        physical_elements = math.prod(shape)
        if physical_elements == elements:
            observed_shape = shape
            observation_scope = "all_elements"
        elif (len(shape) == 2 and elements % shape[1] == 0 and
              elements < physical_elements):
            observed_shape = [elements // shape[1], shape[1]]
            observation_scope = "valid_prefix_only"
        else:
            raise ValueError(
                f"{stage} cannot map {elements} observed values to shape {shape}")
        rows.append({
            "stage": stage,
            "shape": "[" + ",".join(str(v) for v in shape) + "]",
            "observed_shape": (
                "[" + ",".join(str(v) for v in observed_shape) + "]"),
            "observation_scope": observation_scope,
            "scale": operator.get("scale"),
            "bit_length": operator.get("bit_length"),
            "packing": operator.get("packing"),
            "elements": elements,
            "observer_ell": secure["observer_ell"],
            "observer_scale": secure["observer_scale"],
            "rmse_square_scale": secure["rmse_square_scale"],
            "rmse_split_bits": secure["rmse_split_bits"],
            "rmse_accumulator": secure["rmse_accumulator"],
            "clear_fixed_max_abs": clear_fixed["max_abs"],
            "clear_fixed_mean_abs": clear_fixed["mean_abs"],
            "clear_fixed_rmse": clear_fixed["rmse"],
            "fixed_secure_max_abs": fixed_secure["max_abs"],
            "fixed_secure_mean_abs": fixed_secure["mean_abs"],
            "fixed_secure_rmse": fixed_secure["rmse"],
            "clear_secure_max_abs": clear_secure["max_abs"],
            "clear_secure_mean_abs": clear_secure["mean_abs"],
            "clear_secure_rmse": clear_secure["rmse"],
            "intermediate_reconstruction": "false",
            "fixed_secure_acceptance": (
                "PASS_ATOL_0" if fixed_secure["max_abs"] == 0.0
                else "FAIL_ATOL_0"),
            "clear_fixed_numeric_acceptance": (
                "NOT_EVALUATED_NO_NUMERIC_THRESHOLD"),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"VIT_LAYER_ERROR_EXPORT=PASS rows={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
