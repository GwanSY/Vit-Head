#!/usr/bin/env python3
"""Plaintext oracle for synchronized WQ/WK/WV/WO head reordering."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import vit_reference as vr


def load_order(path: Path, layers: int, heads: int) -> list[list[int]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) < 3 or lines[0] != "OURS_VIT_HEAD_PLAN_V1" or lines[1] != (
            "layers=12,heads=12,logical_tokens=197,physical_tokens=256"):
        raise ValueError("unsupported head plan header")
    reader = csv.DictReader(lines[2:])
    required = {"layer", "physical_head", "original_head", "protocol", "alpha"}
    if set(reader.fieldnames or ()) != required:
        raise ValueError("head plan columns changed")
    order = [[-1 for _ in range(heads)] for _ in range(layers)]
    for row in reader:
        layer = int(row["layer"])
        physical = int(row["physical_head"])
        original = int(row["original_head"])
        if not (0 <= layer < layers and 0 <= physical < heads and
                0 <= original < heads) or order[layer][physical] != -1:
            raise ValueError("invalid/duplicate head-plan row")
        order[layer][physical] = original
    for layer, row in enumerate(order):
        if sorted(row) != list(range(heads)):
            raise ValueError(f"layer {layer} original_head is not a permutation")
    return order


def reordered_model(model: dict[str, Any], order: list[list[int]]) -> dict[str, Any]:
    dim = int(model["dim"])
    heads = int(model["heads"])
    head_dim = dim // heads
    result = model.copy()
    result["blocks"] = []
    for layer, source in enumerate(model["blocks"]):
        block = source.copy()
        permutation = order[layer]
        for name in ("wq", "wk", "wv"):
            weight = source[name].reshape(heads, head_dim, dim)
            block[name] = weight[permutation].reshape(dim, dim).copy()
        for name in ("q_bias", "k_bias", "v_bias"):
            bias = source[name].reshape(heads, head_dim)
            block[name] = bias[permutation].reshape(dim).copy()
        # WO is output-major. Its input-feature blocks must follow the same
        # physical head order as the newly concatenated attention context.
        wo = source["wo"].reshape(dim, heads, head_dim)
        block["wo"] = wo[:, permutation, :].reshape(dim, dim).copy()
        result["blocks"].append(block)
    return result


def restore_original_heads(value: np.ndarray, order: list[int],
                           heads: int) -> np.ndarray:
    array = np.asarray(value)
    restored = np.empty_like(array)
    if array.ndim == 3 and array.shape[0] == heads:
        for physical, original in enumerate(order):
            restored[original] = array[physical]
        return restored
    if array.ndim == 2 and array.shape[1] % heads == 0:
        head_dim = array.shape[1] // heads
        source = array.reshape(array.shape[0], heads, head_dim)
        target = restored.reshape(array.shape[0], heads, head_dim)
        for physical, original in enumerate(order):
            target[:, original, :] = source[:, physical, :]
        return restored
    raise ValueError(f"unsupported head tensor shape {array.shape}")


def metric(expected: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    difference = np.asarray(actual) - np.asarray(expected)
    return {
        "elements": int(difference.size),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "mse": float(np.mean(difference * difference, dtype=np.float64)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--head-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-limit", type=int, default=12)
    parser.add_argument("--fp32-tolerance", type=float, default=1e-10)
    parser.add_argument("--fixed-tolerance", type=float, default=0.0)
    args = parser.parse_args()

    model = vr.load_model(args.model)
    pixels = vr.load_input(args.input)
    layers = int(model["layers"])
    heads = int(model["heads"])
    order = load_order(args.head_plan, layers, heads)
    changed = reordered_model(model, order)
    report_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for oracle, fixed, tolerance in (
            ("fp32", False, args.fp32_tolerance),
            ("fixed", True, args.fixed_tolerance)):
        expected_logits, expected = vr.Graph(
            model, fixed=fixed, block_limit=args.block_limit).run(pixels)
        actual_logits, actual = vr.Graph(
            changed, fixed=fixed, block_limit=args.block_limit).run(pixels)
        for layer in range(args.block_limit):
            for suffix in ("q", "k", "v", "qk", "softmax",
                           "attention_context"):
                name = f"block_{layer}.{suffix}"
                observed = restore_original_heads(
                    actual[name]["value"], order[layer], heads)
                row = {"oracle": oracle, "stage": name,
                       **metric(expected[name]["value"], observed)}
                report_rows.append(row)
                if row["max_abs"] > tolerance:
                    failures.append(f"{oracle}:{name}:{row['max_abs']}")
            name = f"block_{layer}.output_projection"
            row = {"oracle": oracle, "stage": name,
                   **metric(expected[name]["value"], actual[name]["value"])}
            report_rows.append(row)
            if row["max_abs"] > tolerance:
                failures.append(f"{oracle}:{name}:{row['max_abs']}")
        logits_row = {"oracle": oracle, "stage": "logits",
                      **metric(expected_logits, actual_logits)}
        report_rows.append(logits_row)
        if logits_row["max_abs"] > tolerance:
            failures.append(f"{oracle}:logits:{logits_row['max_abs']}")

    report = {
        "format": "OURS_VIT_HEAD_REORDER_ORACLE_V1",
        "status": "PASS" if not failures else "FAIL",
        "model": str(args.model.resolve()),
        "input": str(args.input.resolve()),
        "head_plan": str(args.head_plan.resolve()),
        "block_limit": args.block_limit,
        "fp32_tolerance": args.fp32_tolerance,
        "fixed_tolerance": args.fixed_tolerance,
        "failures": failures,
        "stages": report_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"OURS_HEAD_REORDER_ORACLE status={report['status']}")
    print(f"OURS_HEAD_REORDER_REPORT={args.output}")
    if failures:
        raise SystemExit("head reorder oracle failed")


if __name__ == "__main__":
    main()
