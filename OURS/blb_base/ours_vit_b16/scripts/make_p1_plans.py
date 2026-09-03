#!/usr/bin/env python3
"""Build fixed P1 comparison plans with the Ours counts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from calibrate_heterogeneous_heads import grouped_original_order, write_plan


PROTOCOLS = ("softmax_mpc", "relu_norm_mpc", "he_linear")


def read_plan(path: Path, layers: int, heads: int):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines[2:])
    plan = [[("", 1.0) for _ in range(heads)] for _ in range(layers)]
    for row in reader:
        plan[int(row["layer"])][int(row["original_head"])] = (
            row["protocol"], float(row["alpha"]))
    if any(protocol not in PROTOCOLS for layer in plan
           for protocol, _ in layer):
        raise ValueError(f"invalid plan: {path}")
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


def read_costs(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    return rows


def counts(plan):
    return [Counter(protocol for protocol, _ in layer) for layer in plan]


def fingerprint(plan):
    return tuple((protocol, round(alpha, 15)) for layer in plan
                 for protocol, alpha in layer)


def sensitivity_plan(ours, disturbance, alphas):
    result = []
    for layer, target in enumerate(counts(ours)):
        entries = [("softmax_mpc", 1.0) for _ in ours[layer]]
        unused = set(range(len(entries)))
        for protocol in ("he_linear", "relu_norm_mpc"):
            heads = sorted(
                unused,
                key=lambda head: (disturbance[(layer, head, protocol)], head),
            )[:target[protocol]]
            for head in heads:
                alpha = alphas[(layer, head, protocol)] if protocol == "he_linear" else 1.0
                entries[head] = (protocol, alpha)
                unused.remove(head)
        result.append(entries)
    return result


def random_plan(ours, alphas, seed):
    rng = random.Random(seed)
    result = []
    for layer, target in enumerate(counts(ours)):
        entries = [("softmax_mpc", 1.0) for _ in ours[layer]]
        unused = list(range(len(entries)))
        rng.shuffle(unused)
        cursor = 0
        for protocol in ("he_linear", "relu_norm_mpc"):
            for head in unused[cursor:cursor + target[protocol]]:
                alpha = alphas[(layer, head, protocol)] if protocol == "he_linear" else 1.0
                entries[head] = (protocol, alpha)
            cursor += target[protocol]
        result.append(entries)
    return result


def cost_only_plan(ours, costs, alphas):
    result = []
    for layer, target in enumerate(counts(ours)):
        candidates = [row for row in costs if int(row["layer"]) == layer and
                      (int(row["softmax_heads"]), int(row["relu_heads"]),
                       int(row["linear_heads"])) ==
                      (target["softmax_mpc"], target["relu_norm_mpc"],
                       target["he_linear"])]
        if not candidates:
            raise ValueError(f"no directly measured cost row for layer {layer}")
        row = min(candidates, key=lambda item: (float(item["latency_ms"]),
                                                 int(item["communication_bytes"])))
        protocols = row["protocols_original"].split(";")
        entries = []
        for head, protocol in enumerate(protocols):
            alpha = alphas[(layer, head, protocol)] if protocol == "he_linear" else 1.0
            entries.append((protocol, alpha))
        result.append(entries)
    return result


def plan_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-plan", type=Path, required=True)
    parser.add_argument("--candidate-errors", type=Path, required=True)
    parser.add_argument("--assignment-costs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int,
                        help="deprecated single-seed compatibility option")
    parser.add_argument("--random-seeds", type=int, nargs="+",
                        default=(20260902, 20260903, 20260904, 20260905,
                                 20260906))
    args = parser.parse_args()

    layers, heads = 12, 12
    ours = read_plan(args.ours_plan, layers, heads)
    disturbance, alphas = read_disturbance(args.candidate_errors)
    costs = read_costs(args.assignment_costs)
    random_seeds = ([args.random_seed] if args.random_seed is not None
                    else list(args.random_seeds))
    if len(random_seeds) < 5 and args.random_seed is None:
        raise ValueError("formal P1 requires at least five random seeds")
    if len(set(random_seeds)) != len(random_seeds):
        raise ValueError("random seeds must be unique")
    plans = {
        "All-Softmax": [[("softmax_mpc", 1.0) for _ in range(heads)]
                        for _ in range(layers)],
        "Sensitivity-only": sensitivity_plan(ours, disturbance, alphas),
        "Cost-only": cost_only_plan(ours, costs, alphas),
        "Ours": ours,
    }
    for seed in random_seeds:
        plans[f"Random-{seed}"] = random_plan(ours, alphas, seed)
    ours_fingerprint = fingerprint(ours)
    for name in ("Sensitivity-only", "Cost-only"):
        if fingerprint(plans[name]) == ours_fingerprint:
            raise ValueError(
                f"formal P1 refused: {name} is identical to Ours; choose a "
                "budget/cost cohort that produces an independent baseline")
    random_fingerprints = {
        fingerprint(plans[f"Random-{seed}"]) for seed in random_seeds
    }
    if len(random_fingerprints) != len(random_seeds):
        raise ValueError(
            "formal P1 refused: random seeds produced duplicate assignments")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "OURS_VIT_P1_PLAN_SET_V2",
                "random_seeds": random_seeds,
                "fairness_scope": {
                    "same_path_counts": ["Random-*", "Sensitivity-only",
                                         "Cost-only", "Ours"],
                    "all_softmax_role": "reference_not_same_count_budget",
                },
                "source_ours_plan": str(args.ours_plan.resolve()), "plans": {}}
    for name, plan in plans.items():
        filename = name.lower().replace("-", "_") + ".txt"
        path = args.output_dir / filename
        write_plan(path, plan, grouped_original_order(plan))
        manifest["plans"][name] = {
            "path": str(path.resolve()),
            "sha256": plan_hash(path),
            "layer_counts": [dict(counts(plan)[layer]) for layer in range(layers)],
            "global_counts": dict(sum(counts(plan), Counter())),
        }
    (args.output_dir / "p1_plan_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OURS_P1_PLANS status=PASS output={args.output_dir}")
    for name, info in manifest["plans"].items():
        print(f"OURS_P1_PLAN name={name} counts={info['global_counts']} "
              f"sha256={info['sha256']}")


if __name__ == "__main__":
    main()
