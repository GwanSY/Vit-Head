#!/usr/bin/env python3
"""Cost-aware greedy selector for fixed head assignments.

The selector consumes measured group rows and calibration rows.  It only
considers one Head transition at a time; it never enumerates assignments.  A
model-level metric file is required before a transition can be accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PATHS = ("softmax", "relu", "linear")
HEADS = 12
LAYERS = 12


def read_a0_status(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "PASS" in text and "FAIL" not in text:
        return "PASS"
    return "FAIL"


def load_candidates(path: Path) -> dict[tuple[int, int, str], dict[str, float]]:
    result: dict[tuple[int, int, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["layer"]), int(row["head"]), row["protocol"])
            result.setdefault(key, {})[row["oracle"]] = float(row["relative_l2"])
            if row["protocol"] == "he_linear":
                result[key]["alpha"] = float(row["alpha"])
    return result


def load_costs(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[(row["path"].lower(), int(row["heads_group"]))] = row
    return result


def group_cost(costs: dict[tuple[str, int], dict[str, str]], counts: dict[str, int]) -> float | None:
    total = 0.0
    for path, count in counts.items():
        if count == 0:
            continue
        row = costs.get((path, count))
        if row is None or row["online_latency_s"] == "N/A":
            return None
        total += float(row["online_latency_s"])
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0-status", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disturbance-threshold", type=float, default=0.25)
    parser.add_argument("--metric-json", type=Path,
                        help="model-level logits/accuracy gate keyed by step")
    args = parser.parse_args()

    trace: dict[str, object] = {
        "format": "OURS_GREEDY_HEAD_SELECTOR_V1",
        "status": "BLOCKED_A0",
        "enumeration": False,
        "model_metric_gate": "required",
        "steps": [],
    }
    if read_a0_status(args.a0_status) != "PASS":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(trace, indent=2) + "\n",
                               encoding="utf-8")
        print("OURS_GREEDY_SELECTOR status=BLOCKED_A0")
        print(f"OURS_GREEDY_SELECTOR_TRACE={args.output}")
        return

    candidates = load_candidates(args.calibration)
    costs = load_costs(args.costs)
    metrics = (json.loads(args.metric_json.read_text(encoding="utf-8"))
               if args.metric_json else {})
    assignment = [["softmax"] * HEADS for _ in range(LAYERS)]
    counts = {"softmax": HEADS, "relu": 0, "linear": 0}
    current_cost = group_cost(costs, counts)
    if current_cost is None:
        trace["status"] = "BLOCKED_COST_GRID"
    else:
        step = 0
        while True:
            best = None
            for layer in range(LAYERS):
                for head in range(HEADS):
                    source = assignment[layer][head]
                    if source != "softmax":
                        continue
                    for target in ("relu", "linear"):
                        key = (layer, head, f"{target}_norm_mpc"
                               if target == "relu" else "he_linear")
                        candidate = candidates.get(key)
                        if not candidate:
                            continue
                        disturbance = max(candidate.get("fp32", 0.0),
                                          candidate.get("fixed", 0.0))
                        if disturbance > args.disturbance_threshold:
                            continue
                        next_counts = dict(counts)
                        next_counts["softmax"] -= 1
                        next_counts[target] += 1
                        next_cost = group_cost(costs, next_counts)
                        if next_cost is None:
                            continue
                        saved = current_cost - next_cost
                        if saved <= 0.0:
                            continue
                        score = saved / max(disturbance, 1e-12)
                        item = (score, layer, head, target, disturbance,
                                saved, next_cost)
                        if best is None or item[0] > best[0]:
                            best = item
            if best is None:
                break
            _, layer, head, target, disturbance, saved, next_cost = best
            step += 1
            metric = metrics.get(str(step))
            accepted = bool(metric and metric.get("pass", False))
            trace["steps"].append({
                "step": step, "layer": layer, "head": head,
                "from": "softmax", "to": target,
                "disturbance": disturbance, "cost_saved": saved,
                "score": best[0], "model_metric": metric or "N/A",
                "accepted": accepted,
            })
            if not accepted:
                break
            assignment[layer][head] = target
            counts["softmax"] -= 1
            counts[target] += 1
            current_cost = next_cost
        trace["status"] = "PASS" if trace["steps"] else "NO_ACCEPTED_STEP"
        trace["counts"] = counts
        trace["assignment"] = assignment

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2) + "\n",
                           encoding="utf-8")
    print(f"OURS_GREEDY_SELECTOR status={trace['status']}")
    print(f"OURS_GREEDY_SELECTOR_TRACE={args.output}")
    if trace["status"] != "PASS":
        return
    assignment_path = args.output.parent / "head_assignment.json"
    assignment_path.write_text(json.dumps({
        "format": "OURS_HEAD_ASSIGNMENT_V1", "dynamic_selection": False,
        "assignment": assignment,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"OURS_GREEDY_SELECTOR_ASSIGNMENT={assignment_path}")


if __name__ == "__main__":
    main()
