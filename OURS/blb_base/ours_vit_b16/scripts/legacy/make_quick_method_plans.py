#!/usr/bin/env python3
"""Create the quick, fixed-count P1 runtime plans.

This helper is intentionally separate from the formal P1 planner.  The
formal planner requires directly measured complete assignments for Cost-only;
the quick runtime cohort only has the measured per-path/group curve.  It uses
that curve for a deterministic, head-symmetric Cost-only tie-break and records
the limitation in the manifest.  The resulting plans are still useful for a
same-input/same-hardware block-0 runtime comparison, but must not be presented
as the final cost-aware P1 selection evidence.
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
PATH_LABELS = {
    "Softmax": "softmax_mpc",
    "ReLU": "relu_norm_mpc",
    "Linear": "he_linear",
}


def read_plan(path: Path, layers: int = 12, heads: int = 12):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 3 or not lines[0].startswith("OURS_VIT_HEAD_PLAN_V1"):
        raise ValueError(f"invalid head plan header: {path}")
    reader = csv.DictReader(lines[2:])
    plan = [[("", 1.0) for _ in range(heads)] for _ in range(layers)]
    for row in reader:
        layer = int(row["layer"])
        head = int(row["original_head"])
        if not (0 <= layer < layers and 0 <= head < heads):
            raise ValueError(f"head plan index out of range: {path}")
        plan[layer][head] = (row["protocol"], float(row["alpha"]))
    if any(protocol not in PROTOCOLS for layer in plan
           for protocol, _ in layer):
        raise ValueError(f"incomplete or invalid head plan: {path}")
    return plan


def read_disturbance(path: Path):
    values = {}
    alphas = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["layer"]), int(row["head"]), row["protocol"])
            values.setdefault(key, {})[row["oracle"]] = float(row["relative_l2"])
            alphas[key] = float(row["alpha"])
    score = {key: max(oracles.values()) for key, oracles in values.items()}
    return score, alphas


def read_group_costs(path: Path):
    costs = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["heads_group"]) != 4:
                continue
            protocol = PATH_LABELS.get(row["path"])
            if protocol is None:
                continue
            costs[protocol] = {
                "latency_s": float(row["online_latency_s"]),
                "communication_mb": float(row["communication_mb"]),
                "rounds": int(float(row["rounds"])),
                "source_run": row["source_run"],
            }
    if set(costs) != set(PROTOCOLS):
        raise ValueError(f"missing measured group-4 path costs in {path}")
    return costs


def counts(plan):
    return [Counter(protocol for protocol, _ in layer) for layer in plan]


def random_plan(ours, alphas, seed):
    rng = random.Random(seed)
    result = []
    for layer, target in enumerate(counts(ours)):
        entries = [("softmax_mpc", 1.0) for _ in ours[layer]]
        heads = list(range(len(entries)))
        rng.shuffle(heads)
        cursor = 0
        for protocol in ("he_linear", "relu_norm_mpc"):
            for head in heads[cursor:cursor + target[protocol]]:
                alpha = (alphas[(layer, head, protocol)]
                         if protocol == "he_linear" else 1.0)
                entries[head] = (protocol, alpha)
            cursor += target[protocol]
        result.append(entries)
    return result


def sensitivity_plan(ours, disturbance, alphas):
    result = []
    for layer, target in enumerate(counts(ours)):
        entries = [("softmax_mpc", 1.0) for _ in ours[layer]]
        unused = set(range(len(entries)))
        # Assign the lowest-disturbance heads to the non-Softmax paths.  The
        # fixed tie-break makes the plan reproducible across machines.
        for protocol in ("he_linear", "relu_norm_mpc"):
            selected = sorted(
                unused,
                key=lambda head: (disturbance[(layer, head, protocol)], head),
            )[:target[protocol]]
            for head in selected:
                alpha = (alphas[(layer, head, protocol)]
                         if protocol == "he_linear" else 1.0)
                entries[head] = (protocol, alpha)
                unused.remove(head)
        result.append(entries)
    return result


def cost_only_plan(ours, alphas, path_costs):
    # The available measurements are group-level and head-symmetric, so they
    # cannot distinguish individual heads.  Use measured group cost to order
    # the paths, then use original-head order as the explicit tie-break.
    path_order = sorted(
        PROTOCOLS,
        key=lambda protocol: (path_costs[protocol]["latency_s"], protocol),
    )
    result = []
    for layer, target in enumerate(counts(ours)):
        entries = [("softmax_mpc", 1.0) for _ in ours[layer]]
        cursor = 0
        for protocol in path_order:
            for head in range(cursor, cursor + target[protocol]):
                alpha = (alphas[(layer, head, protocol)]
                         if protocol == "he_linear" else 1.0)
                entries[head] = (protocol, alpha)
            cursor += target[protocol]
        result.append(entries)
    return result, path_order


def plan_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-plan", type=Path, required=True)
    parser.add_argument("--candidate-errors", type=Path, required=True)
    parser.add_argument("--path-costs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seeds", type=int, nargs="+",
                        default=(20260903, 20260904, 20260905, 20260906,
                                 20260907))
    args = parser.parse_args()
    if len(args.random_seeds) < 5 or len(set(args.random_seeds)) != len(args.random_seeds):
        raise ValueError("quick plan manifest requires five unique random seeds")

    ours = read_plan(args.ours_plan)
    disturbance, alphas = read_disturbance(args.candidate_errors)
    path_costs = read_group_costs(args.path_costs)
    layers, heads = 12, 12
    all_softmax = [[("softmax_mpc", 1.0) for _ in range(heads)]
                   for _ in range(layers)]
    cost_only, cost_order = cost_only_plan(ours, alphas, path_costs)
    plans = {
        "All-Softmax": all_softmax,
        "Sensitivity-only": sensitivity_plan(ours, disturbance, alphas),
        "Cost-only": cost_only,
        "Ours": ours,
    }
    for seed in args.random_seeds:
        plans[f"Random-{seed}"] = random_plan(ours, alphas, seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "OURS_VIT_QUICK_P1_PLAN_SET_V1",
        "scope": "single_image_block0_runtime",
        "source_ours_plan": str(args.ours_plan.resolve()),
        "source_candidate_errors": str(args.candidate_errors.resolve()),
        "source_path_costs": str(args.path_costs.resolve()),
        "random_seeds": list(args.random_seeds),
        "cost_only_interpretation": (
            "group-level measured path-cost ordering with original-head tie-break; "
            "provisional until complete-assignment Cost-only measurements exist"
        ),
        "path_costs_group4": path_costs,
        "cost_order_fast_to_slow": cost_order,
        "plans": {},
    }
    for name, plan in plans.items():
        filename = name.lower().replace("-", "_") + ".txt"
        path = args.output_dir / filename
        write_plan(path, plan, grouped_original_order(plan))
        layer_counts = [dict(counts(plan)[layer]) for layer in range(layers)]
        manifest["plans"][name] = {
            "path": str(path.resolve()),
            "sha256": plan_hash(path),
            "layer_counts": layer_counts,
            "global_counts": dict(sum(counts(plan), Counter())),
        }
    (args.output_dir / "quick_p1_plan_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"OURS_QUICK_P1_PLANS status=PASS output={args.output_dir}")
    print(f"OURS_QUICK_COST_ORDER fast_to_slow={','.join(cost_order)}")
    for name, info in manifest["plans"].items():
        print(f"OURS_QUICK_P1_PLAN name={name} counts={info['global_counts']} "
              f"sha256={info['sha256']}")


if __name__ == "__main__":
    main()
