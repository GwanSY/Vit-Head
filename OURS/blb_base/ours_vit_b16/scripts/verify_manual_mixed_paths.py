#!/usr/bin/env python3
"""Public fixed-clear oracle for A1's three paths and mixed head execution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import vit_reference as vr


PROTOCOLS = ("softmax_mpc", "relu_norm_mpc", "he_linear")


def manual_plan(layers: int, heads: int) -> list[list[tuple[str, float]]]:
    return [[(PROTOCOLS[head % 3],
              1.0 / 197.0 if head % 3 == 2 else 1.0)
             for head in range(heads)] for _ in range(layers)]


def expected_weights(score: np.ndarray, protocol: str, alpha: float,
                     fixed: bool) -> np.ndarray:
    if protocol == "softmax_mpc":
        return vr.fixed_softmax(score) if fixed else vr.softmax(score)
    if protocol == "relu_norm_mpc":
        if fixed:
            return vr.fixed_relu_normalization(score)
        positive = np.maximum(score, 0.0)
        denominator = positive.sum(axis=-1, keepdims=True, dtype=np.float64)
        result = np.divide(positive, denominator, out=np.zeros_like(positive),
                           where=denominator != 0.0)
        return result
    if protocol == "he_linear":
        return (vr.fixed_linear_scaling(score, alpha) if fixed else
                alpha * score)
    raise ValueError(f"unknown protocol {protocol}")


def metric(expected: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    delta = np.asarray(actual) - np.asarray(expected)
    squared = float(np.sum(delta * delta, dtype=np.float64))
    reference_squared = float(np.sum(
        np.asarray(expected) * np.asarray(expected), dtype=np.float64))
    return {
        "elements": int(delta.size),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "relative_l2": (math.sqrt(squared / reference_squared)
                        if reference_squared > 0.0 else
                        (0.0 if squared == 0.0 else math.inf)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-limit", type=int, default=1)
    args = parser.parse_args()
    model = vr.load_model(args.model)
    pixels = vr.load_input(args.input)
    layers = int(model["layers"])
    heads = int(model["heads"])
    plan = manual_plan(layers, heads)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for oracle, fixed, tolerance in (("fp32", False, 1e-12),
                                     ("fixed_clear", True, 0.0)):
        _, records = vr.Graph(
            model, fixed=fixed, physical_tokens=256,
            block_limit=args.block_limit, head_plan=plan).run(pixels)
        for layer in range(args.block_limit):
            scores = records[f"block_{layer}.qk"]["value"]
            actual = records[f"block_{layer}.softmax"]["value"]
            for head, (protocol, alpha) in enumerate(plan[layer]):
                expected = expected_weights(
                    scores[head, :197, :197], protocol, alpha, fixed)
                observed = actual[head, :197, :197]
                row = {"oracle": oracle, "layer": layer, "head": head,
                       "protocol": protocol, **metric(expected, observed)}
                padded_key_max = float(np.max(
                    np.abs(actual[head, :, 197:]), initial=0.0))
                row["padded_key_max_abs"] = padded_key_max
                rows.append(row)
                if row["max_abs"] > tolerance or padded_key_max != 0.0:
                    failures.append(
                        f"{oracle}:layer={layer}:head={head}:path={protocol}")
    counts = {protocol: sum(row["protocol"] == protocol for row in rows)
              for protocol in PROTOCOLS}
    if any(count == 0 for count in counts.values()):
        failures.append("manual plan did not execute all three paths")
    report = {
        "format": "OURS_VIT_A1_MANUAL_MIXED_ORACLE_V1",
        "status": "PASS" if not failures else "FAIL",
        "model": str(args.model.resolve()),
        "input": str(args.input.resolve()),
        "block_limit": args.block_limit,
        "physical_tokens": 256,
        "logical_tokens": 197,
        "path_counts": counts,
        "failures": failures,
        "metrics": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"OURS_A1_MANUAL_MIXED_ORACLE status={report['status']}")
    print(f"OURS_A1_MANUAL_MIXED_REPORT={args.output}")
    if failures:
        raise SystemExit("A1 manual mixed oracle failed")


if __name__ == "__main__":
    main()
