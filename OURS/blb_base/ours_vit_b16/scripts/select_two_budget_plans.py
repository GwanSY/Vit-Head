#!/usr/bin/env python3
"""Select fixed- and variable-budget plans from measured assignments.

The selector is intentionally fail-closed: it never estimates a complete
assignment from per-head or independent-path costs.  Every selected layer is
backed by a directly measured ``complete_attention_assignment`` row with the
same binary, image, network and GPU topology.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from calibrate_heterogeneous_heads import grouped_original_order, write_plan


PROTOCOLS = ("softmax_mpc", "relu_norm_mpc", "he_linear")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_plan(path: Path) -> list[list[tuple[str, float]]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if lines[:2] != [
            "OURS_VIT_HEAD_PLAN_V1",
            "layers=12,heads=12,logical_tokens=197,physical_tokens=256"]:
        raise ValueError(f"invalid head plan header: {path}")
    plan = [[("", 0.0) for _ in range(12)] for _ in range(12)]
    for row in csv.DictReader(lines[2:]):
        layer = int(row["layer"])
        physical = int(row["physical_head"])
        original = int(row["original_head"])
        if not (0 <= layer < 12 and 0 <= physical < 12 and 0 <= original < 12):
            raise ValueError(f"head plan index out of range: {path}")
        if plan[layer][original][0]:
            raise ValueError(f"duplicate original head in {path}")
        plan[layer][original] = (row["protocol"], float(row["alpha"]))
    if any(protocol not in PROTOCOLS for layer in plan for protocol, _ in layer):
        raise ValueError(f"incomplete or invalid head plan: {path}")
    return plan


def read_candidates(path: Path) -> dict[tuple[int, int, str], float]:
    values: dict[tuple[int, int, str], list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["layer"]), int(row["head"]), row["protocol"])
            values.setdefault(key, []).append(float(row["relative_l2"]))
    expected = {(layer, head, protocol)
                for layer in range(12) for head in range(12)
                for protocol in PROTOCOLS}
    if set(values) != expected:
        raise ValueError("candidate error CSV does not cover all heads/protocols")
    return {key: max(items) for key, items in values.items()}


def plan_counts(plan: list[list[tuple[str, float]]]) -> list[dict[str, int]]:
    return [dict(Counter(protocol for protocol, _ in layer)) for layer in plan]


def layer_disturbance(plan: list[tuple[str, float]], layer: int,
                      disturbance: dict[tuple[int, int, str], float]) -> dict[str, float]:
    values = [disturbance[(layer, head, protocol)]
              for head, (protocol, _) in enumerate(plan)
              if protocol != "softmax_mpc"]
    return {
        "max": max(values, default=0.0),
        "sum": sum(values),
        "mean": sum(values) / len(values) if values else 0.0,
        "non_softmax_heads": len(values),
    }


def load_costs(path: Path) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty assignment-cost CSV: {path}")
    required = {"layer", "latency_ms", "communication_bytes", "rounds",
                "plan_sha256", "measurement_scope", "cost_unit", "image_id",
                "binary_sha256", "network_gbps", "rtt_ms", "backend",
                "gpu_topology", "ablation", "softmax_heads", "relu_heads",
                "linear_heads"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"assignment-cost CSV missing {sorted(missing)}")
    provenance_sets = {
        field: {row[field] for row in rows}
        for field in ("image_id", "binary_sha256", "network_gbps", "rtt_ms",
                      "backend", "gpu_topology")
    }
    if any(len(values) != 1 for values in provenance_sets.values()):
        raise ValueError(f"cost rows do not share one provenance: {provenance_sets}")
    for row in rows:
        if (row["measurement_scope"] != "complete_attention_assignment" or
                row["cost_unit"] != "direct_assignment_not_per_head" or
                row["network_gbps"] != "1.0" or row["rtt_ms"] != "0.3" or
                row["backend"].lower() != "phantom_gpu" or
                row["gpu_topology"] != "separate_physical_gpus"):
            raise ValueError("ineligible assignment cost row")
    # JSON manifests need deterministic scalar/list values rather than the
    # internal sets used for the invariant check.
    provenance = {
        field: sorted(values) for field, values in provenance_sets.items()
    }
    return rows, provenance


def cost_key(row: dict[str, str]) -> tuple[float, int, int, str]:
    return (float(row["latency_ms"]), int(row["communication_bytes"]),
            int(row["rounds"]), row["plan_sha256"])


def copy_plan(output_dir: Path, name: str,
              plan: list[list[tuple[str, float]]], info: dict[str, object],
              manifest: dict[str, object], description: str) -> None:
    path = output_dir / (name.lower().replace(" ", "_").replace("+", "_") + ".txt")
    write_plan(path, plan, grouped_original_order(plan))
    counts = plan_counts(plan)
    manifest["plans"][name] = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "name": name,
        "ablation": info["ablation"],
        "budget_group": info["budget_group"],
        "description": description,
        "layer_counts": counts,
        "global_counts": dict(Counter(protocol for layer in plan
                                        for protocol, _ in layer)),
        "selection": info.get("selection"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-errors", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-head-relative-l2", type=float, default=1.0)
    args = parser.parse_args()
    source = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    source_plans = source.get("plans", {})
    if not isinstance(source_plans, dict):
        raise ValueError("candidate manifest has no plans")
    disturbance = read_candidates(args.candidate_errors)
    cost_rows, provenance = load_costs(args.costs)
    plans: dict[str, list[list[tuple[str, float]]]] = {}
    infos: dict[str, dict[str, object]] = {}
    for name, info in source_plans.items():
        if not isinstance(info, dict):
            continue
        plans[name] = read_plan(Path(info["path"]))
        infos[name] = info
    # All-Softmax/Variable-12S and Fixed-Sensitivity/Variable-4S4R4L can
    # intentionally be the same assignment.  One measured row must serve
    # every manifest alias; it must not be treated as a duplicate experiment.
    by_hash: dict[str, list[str]] = {}
    for name, info in source_plans.items():
        by_hash.setdefault(sha256(Path(info["path"])), []).append(name)
    rows_by_plan_layer: dict[tuple[str, int], dict[str, str]] = {}
    for row in cost_rows:
        names = by_hash.get(row["plan_sha256"], [])
        if not names:
            continue
        for name in names:
            key = (name, int(row["layer"]))
            if key in rows_by_plan_layer:
                raise ValueError(f"duplicate cost row for {name} layer {key[1]}")
            rows_by_plan_layer[key] = row

    fixed_names = [name for name, info in infos.items()
                   if info.get("budget_group") == "fixed_candidate"]
    variable_names = [name for name, info in infos.items()
                      if info.get("budget_group") == "variable_candidate"]
    if not fixed_names or not variable_names:
        raise ValueError("candidate manifest lacks fixed or variable candidates")

    def fixed_score(name: str) -> tuple[float, int, int, str]:
        rows = [row for (candidate, _layer), row in rows_by_plan_layer.items()
                if candidate == name]
        if not rows:
            raise ValueError(f"no direct cost row for fixed candidate {name}")
        # Fixed calibration runs may be block-0 only.  Use their measured
        # assignment row directly; full-12 rows are averaged over layers.
        return (sum(float(row["latency_ms"]) for row in rows) / len(rows),
                sum(int(row["communication_bytes"]) for row in rows) // len(rows),
                sum(int(row["rounds"]) for row in rows) // len(rows),
                name)

    fixed_cost_name = min(fixed_names, key=fixed_score)
    fixed_eligible = []
    fixed_audit = []
    for name in fixed_names:
        plan = plans[name]
        disturbances = [layer_disturbance(plan[layer], layer, disturbance)
                        for layer in range(12)]
        max_disturbance = max(item["max"] for item in disturbances)
        eligible = max_disturbance <= args.max_head_relative_l2
        if eligible:
            fixed_eligible.append(name)
        fixed_audit.append({
            "candidate": name,
            "score": fixed_score(name),
            "eligible_for_ours": eligible,
            "max_head_relative_l2": max_disturbance,
            "layer_disturbance": disturbances,
        })
    if not fixed_eligible:
        raise ValueError("no fixed candidate satisfies the Ours sensitivity threshold")
    fixed_ours_name = min(fixed_eligible, key=fixed_score)

    variable_audit: list[dict[str, object]] = []
    variable_ours_layers: list[list[tuple[str, float]]] = []
    variable_cost_layers: list[list[tuple[str, float]]] = []
    variable_selected: list[dict[str, object]] = []
    for layer in range(12):
        candidates = []
        for name in variable_names:
            row = rows_by_plan_layer.get((name, layer))
            if row is None:
                continue
            sensitivity = layer_disturbance(plans[name][layer], layer, disturbance)
            item = {
                "candidate": name,
                "row": row,
                "sensitivity": sensitivity,
                "eligible_for_ours": sensitivity["max"] <= args.max_head_relative_l2,
            }
            candidates.append(item)
        if not candidates:
            raise ValueError(f"no directly measured variable candidate for layer {layer}")
        cost_choice = min(candidates, key=lambda item: cost_key(item["row"]))
        eligible = [item for item in candidates if item["eligible_for_ours"]]
        if not eligible:
            raise ValueError(f"no eligible variable candidate for layer {layer}")
        ours_choice = min(eligible, key=lambda item: cost_key(item["row"]))
        variable_cost_layers.append(plans[cost_choice["candidate"]][layer])
        variable_ours_layers.append(plans[ours_choice["candidate"]][layer])
        variable_selected.append({
            "layer": layer,
            "cost_only": {
                "candidate": cost_choice["candidate"],
                "counts": plan_counts([plans[cost_choice["candidate"]][layer]])[0],
                "cost": cost_choice["row"],
                "sensitivity": cost_choice["sensitivity"],
            },
            "ours": {
                "candidate": ours_choice["candidate"],
                "counts": plan_counts([plans[ours_choice["candidate"]][layer]])[0],
                "cost": ours_choice["row"],
                "sensitivity": ours_choice["sensitivity"],
            },
            "candidates": [{
                "candidate": item["candidate"],
                "counts": plan_counts([plans[item["candidate"]][layer]])[0],
                "cost": item["row"],
                "sensitivity": item["sensitivity"],
                "eligible_for_ours": item["eligible_for_ours"],
            } for item in sorted(candidates, key=lambda item: cost_key(item["row"]))],
        })
    variable_audit = variable_selected

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_manifest: dict[str, object] = {
        "format": "OURS_VIT_TWO_BUDGET_SELECTED_PLAN_SET_V1",
        "candidate_manifest": str(args.candidate_manifest.resolve()),
        "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "candidate_errors": str(args.candidate_errors.resolve()),
        "candidate_errors_sha256": sha256(args.candidate_errors),
        "direct_costs": str(args.costs.resolve()),
        "direct_costs_sha256": sha256(args.costs),
        "cost_provenance": provenance,
        "max_head_relative_l2": args.max_head_relative_l2,
        "fixed_selection": {
            "cost_only_candidate": fixed_cost_name,
            "ours_candidate": fixed_ours_name,
            "candidate_audit": fixed_audit,
            "cost_unit": "direct_complete_attention_assignment",
        },
        "variable_selection": {
            "cost_unit": "direct_complete_attention_assignment_per_layer",
            "layer_selections": variable_audit,
        },
        "plans": {},
    }

    # Keep the reference and all fixed-budget controls in the final manifest.
    copy_plan(args.output_dir, "All-Softmax", plans["All-Softmax"],
              {"ablation": "a3", "budget_group": "reference"},
              final_manifest, "12 Softmax heads in every block")
    for name in fixed_names:
        copy_plan(args.output_dir, name, plans[name], infos[name],
                  final_manifest, str(infos[name].get("description", "")))
    copy_plan(args.output_dir, "Fixed-Cost-only", plans[fixed_cost_name],
              {"ablation": "a4", "budget_group": "fixed_selected",
               "selection": fixed_cost_name},
              final_manifest, "minimum measured fixed-budget assignment cost")
    copy_plan(args.output_dir, "Fixed-Ours", plans[fixed_ours_name],
              {"ablation": "a4", "budget_group": "fixed_selected",
               "selection": fixed_ours_name},
              final_manifest, "minimum measured fixed-budget cost among sensitivity-eligible assignments")
    variable_cost_ablation = "a4" if any(
        protocol == "he_linear" for layer in variable_cost_layers
        for protocol, _ in layer) else "a3"
    copy_plan(args.output_dir, "Variable-Cost-only", variable_cost_layers,
              {"ablation": variable_cost_ablation, "budget_group": "variable_selected",
               "selection": "per-layer minimum measured cost"},
              final_manifest, "per-layer minimum direct cost without sensitivity filtering")
    variable_ablation = "a4" if any(
        protocol == "he_linear" for layer in variable_ours_layers
        for protocol, _ in layer) else "a3"
    copy_plan(args.output_dir, "Variable-Ours", variable_ours_layers,
              {"ablation": variable_ablation, "budget_group": "variable_selected",
               "selection": "per-layer minimum measured cost under sensitivity threshold"},
              final_manifest, "per-layer cost minimum under the declared sensitivity threshold")

    audit_path = args.output_dir / "two_budget_selection_audit.json"
    audit_path.write_text(json.dumps({
        "format": "OURS_VIT_TWO_BUDGET_SELECTION_AUDIT_V1",
        "fixed": fixed_audit,
        "variable": variable_audit,
        "fixed_cost_only": fixed_cost_name,
        "fixed_ours": fixed_ours_name,
        "max_head_relative_l2": args.max_head_relative_l2,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_path = args.output_dir / "two_budget_selected_manifest.json"
    final_manifest["selection_audit"] = str(audit_path.resolve())
    final_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    print(f"OURS_TWO_BUDGET_SELECTION status=PASS output={final_path}")
    print(f"OURS_TWO_BUDGET_FIXED cost_only={fixed_cost_name} ours={fixed_ours_name}")
    print("OURS_TWO_BUDGET_VARIABLE")
    for item in variable_selected:
        print(f"layer={item['layer']} cost_only={item['cost_only']['candidate']} "
              f"ours={item['ours']['candidate']}")


if __name__ == "__main__":
    main()
