#!/usr/bin/env python3
"""Build independent fixed-budget and variable-budget ViT head plans.

This file deliberately does not select an ``Ours`` plan.  It creates the
candidate assignments that are later selected from directly measured
complete-attention costs by ``select_two_budget_plans.py``.

The fixed cohort holds (Softmax, ReLU, Linear) counts at 4/4/4.  The variable
cohort contains several explicitly enumerated count vectors; its selector is
free to choose a different vector for every Transformer block.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from calibrate_heterogeneous_heads import grouped_original_order, write_plan


PROTOCOLS = ("softmax_mpc", "relu_norm_mpc", "he_linear")
FIXED_COUNTS = (4, 4, 4)
DEFAULT_RANDOM_SEEDS = (20260911, 20260912, 20260913, 20260914, 20260915)


def read_candidates(path: Path) -> tuple[dict[tuple[int, int, str], float],
                                             dict[tuple[int, int], float]]:
    """Return worst-oracle disturbance and fitted Linear alpha."""
    observations: dict[tuple[int, int, str], list[float]] = {}
    alphas: dict[tuple[int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["layer"]), int(row["head"]), row["protocol"])
            observations.setdefault(key, []).append(float(row["relative_l2"]))
            if row["protocol"] == "he_linear":
                alpha = float(row["alpha"])
                previous = alphas.setdefault(key[:2], alpha)
                if abs(previous - alpha) > 1e-12:
                    raise ValueError(f"inconsistent Linear alpha for {key[:2]}")
    disturbance = {key: max(values) for key, values in observations.items()}
    expected = {(layer, head, protocol)
                for layer in range(12) for head in range(12)
                for protocol in PROTOCOLS}
    if set(disturbance) != expected:
        missing = sorted(expected - set(disturbance))[:5]
        raise ValueError(f"candidate errors are incomplete; missing {missing}")
    return disturbance, alphas


def sensitivity_assignment(
        counts: tuple[int, int, int],
        disturbance: dict[tuple[int, int, str], float],
        alphas: dict[tuple[int, int], float],
        *, layer: int) -> list[tuple[str, float]]:
    """Assign the least-disturbing available heads to each non-Softmax path."""
    softmax, relu, linear = counts
    if softmax + relu + linear != 12:
        raise ValueError(f"counts do not sum to 12: {counts}")
    entries = [("softmax_mpc", 1.0) for _ in range(12)]
    unused = set(range(12))
    # Linear is assigned first because it normally has the strictest
    # sensitivity profile; ReLU then uses the remaining heads.
    for protocol, count in (("he_linear", linear),
                            ("relu_norm_mpc", relu)):
        selected = sorted(
            unused,
            key=lambda head: (disturbance[(layer, head, protocol)], head),
        )[:count]
        for head in selected:
            alpha = alphas[(layer, head)] if protocol == "he_linear" else 1.0
            entries[head] = (protocol, alpha)
            unused.remove(head)
    if len(unused) != softmax:
        raise AssertionError("head assignment count mismatch")
    return entries


def fixed_random_assignment(seed: int, alphas: dict[tuple[int, int], float],
                            layer: int) -> list[tuple[str, float]]:
    rng = random.Random(seed + layer * 1000003)
    order = list(range(12))
    rng.shuffle(order)
    entries = [("softmax_mpc", 1.0) for _ in range(12)]
    cursor = 0
    for protocol, count in zip(PROTOCOLS[1:], FIXED_COUNTS[1:]):
        for head in order[cursor:cursor + count]:
            alpha = alphas[(layer, head)] if protocol == "he_linear" else 1.0
            entries[head] = (protocol, alpha)
        cursor += count
    return entries


def fixed_contiguous_assignment(alphas: dict[tuple[int, int], float],
                                layer: int) -> list[tuple[str, float]]:
    entries: list[tuple[str, float]] = []
    for head in range(12):
        if head < 4:
            entries.append(("softmax_mpc", 1.0))
        elif head < 8:
            entries.append(("relu_norm_mpc", 1.0))
        else:
            entries.append(("he_linear", alphas[(layer, head)]))
    return entries


def count_plan(plan: list[list[tuple[str, float]]]) -> list[dict[str, int]]:
    return [dict(Counter(protocol for protocol, _ in layer)) for layer in plan]


def plan_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit_plan(output_dir: Path, name: str, ablation: str,
              budget_group: str, plan: list[list[tuple[str, float]]],
              manifest: dict[str, object], description: str) -> None:
    filename = name.lower().replace(" ", "_").replace("+", "_") + ".txt"
    path = output_dir / filename
    write_plan(path, plan, grouped_original_order(plan))
    info = {
        "path": str(path.resolve()),
        "sha256": plan_hash(path),
        "name": name,
        "ablation": ablation,
        "budget_group": budget_group,
        "description": description,
        "layer_counts": count_plan(plan),
        "global_counts": dict(Counter(
            protocol for layer in plan for protocol, _ in layer)),
    }
    manifest["plans"][name] = info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-errors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seeds", type=int, nargs="+",
                        default=DEFAULT_RANDOM_SEEDS)
    args = parser.parse_args()
    if len(args.random_seeds) < 5 or len(set(args.random_seeds)) != len(args.random_seeds):
        raise ValueError("fixed cohort requires at least five unique random seeds")
    disturbance, alphas = read_candidates(args.candidate_errors)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "OURS_VIT_TWO_BUDGET_PLAN_SET_V1",
        "model": "ViT-B/16",
        "layers": 12,
        "heads": 12,
        "candidate_errors": str(args.candidate_errors.resolve()),
        "candidate_errors_sha256": plan_hash(args.candidate_errors),
        "fixed_budget": {
            "counts_per_layer": {
                "softmax_mpc": FIXED_COUNTS[0],
                "relu_norm_mpc": FIXED_COUNTS[1],
                "he_linear": FIXED_COUNTS[2],
            },
            "purpose": "same path counts; compare head assignment only",
            "selection": "Cost-only and Ours are selected from measured fixed candidates after cost collection",
        },
        "variable_budget": {
            "purpose": "allow each layer to select a measured count vector",
            "candidate_count_vectors": [
                {"name": "12S", "counts": [12, 0, 0], "ablation": "a3"},
                {"name": "10S2L", "counts": [10, 0, 2], "ablation": "a4"},
                {"name": "8S4L", "counts": [8, 0, 4], "ablation": "a4"},
                {"name": "8S4R", "counts": [8, 4, 0], "ablation": "a3"},
                {"name": "6S4R2L", "counts": [6, 4, 2], "ablation": "a4"},
                {"name": "4S4R4L", "counts": [4, 4, 4], "ablation": "a4"},
                {"name": "4S8R", "counts": [4, 8, 0], "ablation": "a3"},
            ],
            "selection": "minimum direct complete-assignment latency subject to per-head disturbance threshold",
        },
        "plans": {},
    }
    plans = manifest["plans"]
    assert isinstance(plans, dict)

    all_softmax = [[("softmax_mpc", 1.0) for _ in range(12)]
                   for _ in range(12)]
    emit_plan(args.output_dir, "All-Softmax", "a3", "reference",
              all_softmax, manifest, "12 Softmax heads in every block")

    sensitivity_fixed = [
        sensitivity_assignment(FIXED_COUNTS, disturbance, alphas, layer=layer)
        for layer in range(12)
    ]
    emit_plan(args.output_dir, "Fixed-Sensitivity-only", "a4", "fixed_candidate",
              sensitivity_fixed, manifest,
              "fixed 4S+4R+4L; lowest measured disturbance heads")

    contiguous = [fixed_contiguous_assignment(alphas, layer)
                  for layer in range(12)]
    emit_plan(args.output_dir, "Fixed-Contiguous", "a4", "fixed_candidate",
              contiguous, manifest,
              "fixed 4S+4R+4L; deterministic contiguous original-head assignment")

    for seed in args.random_seeds:
        plan = [fixed_random_assignment(seed, alphas, layer)
                for layer in range(12)]
        emit_plan(args.output_dir, f"Fixed-Random-{seed}", "a4",
                  "fixed_candidate", plan, manifest,
                  f"fixed 4S+4R+4L; random seed {seed}")

    variable_vectors = (
        ("Variable-12S", (12, 0, 0), "a3"),
        ("Variable-10S2L", (10, 0, 2), "a4"),
        ("Variable-8S4L", (8, 0, 4), "a4"),
        ("Variable-8S4R", (8, 4, 0), "a3"),
        ("Variable-6S4R2L", (6, 4, 2), "a4"),
        ("Variable-4S4R4L", (4, 4, 4), "a4"),
        ("Variable-4S8R", (4, 8, 0), "a3"),
    )
    for name, counts, ablation in variable_vectors:
        plan = [sensitivity_assignment(counts, disturbance, alphas, layer=layer)
                for layer in range(12)]
        emit_plan(args.output_dir, name, ablation, "variable_candidate", plan,
                  manifest, f"sensitivity-ranked assignment for count vector {counts}")

    manifest_path = args.output_dir / "two_budget_plan_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"OURS_TWO_BUDGET_PLANS status=PASS output={manifest_path}")
    for name, info in plans.items():
        print(f"OURS_TWO_BUDGET_PLAN name={name} counts={info['global_counts']} "
              f"ablation={info['ablation']} sha256={info['sha256']}")


if __name__ == "__main__":
    main()
