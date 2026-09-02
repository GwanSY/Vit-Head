#!/usr/bin/env python3
"""Build a fixed sparse A4 plan from measured per-head disturbances.

This is an explicit wiring/measurement-plan generator, not the A2 cost
selector.  For each requested layer it chooses an even number of original
heads with the smallest worst-case (FP32/fixed) Linear relative-L2
disturbances, moves them into one final contiguous physical Linear group, and
keeps every other head on Softmax.  The even-size constraint matches BLB's
two-head half/complex packing so a Linear group can remove complete QK/V
conversion pairs.  Inference never reads the calibration CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


LAYERS = 12
HEADS = 12


def parse_layers(value: str) -> set[int]:
    if value == "all":
        return set(range(LAYERS))
    result = {int(item) for item in value.split(",") if item}
    if not result or any(layer < 0 or layer >= LAYERS for layer in result):
        raise argparse.ArgumentTypeError(
            "linear layers must be 'all' or comma-separated indices in [0,11]")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-errors", type=Path, required=True)
    parser.add_argument("--linear-layers", type=parse_layers, default={0})
    parser.add_argument("--linear-heads-per-layer", type=int, default=2)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    args = parser.parse_args()
    if (args.linear_heads_per_layer < 2 or
            args.linear_heads_per_layer > HEADS or
            args.linear_heads_per_layer % 2 != 0):
        parser.error("--linear-heads-per-layer must be even and in [2,12]")

    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    with args.candidate_errors.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["protocol"] == "he_linear":
                grouped[(int(row["layer"]), int(row["head"]))].append(row)

    selected: dict[int, list[dict[str, float | int]]] = {}
    for layer in sorted(args.linear_layers):
        candidates = []
        for head in range(HEADS):
            observations = grouped[(layer, head)]
            if not observations:
                raise ValueError(f"missing Linear calibration for layer={layer} head={head}")
            alphas = {float(row["alpha"]) for row in observations}
            if len(alphas) != 1:
                raise ValueError(f"inconsistent alpha for layer={layer} head={head}")
            candidates.append((
                max(float(row["relative_l2"]) for row in observations),
                head,
                alphas.pop(),
            ))
        candidates.sort()
        selected[layer] = [
            {
                "original_head": head,
                "alpha": alpha,
                "max_relative_l2_across_oracles": disturbance,
            }
            for disturbance, head, alpha in
            candidates[:args.linear_heads_per_layer]
        ]

    lines = [
        "OURS_VIT_HEAD_PLAN_V1",
        "layers=12,heads=12,logical_tokens=197,physical_tokens=256",
        "layer,physical_head,original_head,protocol,alpha",
    ]
    physical_orders: list[list[int]] = []
    for layer in range(LAYERS):
        linear_by_head = {
            int(item["original_head"]): item
            for item in selected.get(layer, [])
        }
        linear_heads = sorted(linear_by_head)
        order = [head for head in range(HEADS) if head not in linear_by_head]
        order.extend(linear_heads)
        physical_orders.append(order)
        for physical, original in enumerate(order):
            if original in linear_by_head:
                protocol = "he_linear"
                alpha = float(linear_by_head[original]["alpha"])
            else:
                protocol = "softmax_mpc"
                alpha = 1.0
            lines.append(
                f"{layer},{physical},{original},{protocol},{alpha:.17g}")

    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit = {
        "format": "OURS_A4_SPARSE_REVIEW_PLAN_V2",
        "purpose": "real_A4_wiring_and_cost_measurement_not_final_A2_selection",
        "candidate_errors": str(args.candidate_errors.resolve()),
        "linear_layers": sorted(args.linear_layers),
        "linear_heads_per_layer": args.linear_heads_per_layer,
        "packing_constraint": "complete_two_head_half_complex_pairs",
        "selected": selected,
        "physical_head_order": physical_orders,
        "inference_dynamic_selection": False,
        "mpc_linear_surrogate": False,
    }
    args.output_audit.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("OURS_A4_SPARSE_REVIEW_PLAN status=PASS "
          f"linear_layers={','.join(map(str, sorted(args.linear_layers)))} "
          f"linear_heads_per_layer={args.linear_heads_per_layer}")
    print(f"OURS_A4_SPARSE_REVIEW_PLAN_FILE={args.output_plan}")


if __name__ == "__main__":
    main()
