#!/usr/bin/env python3
"""Run the deliberately small *true clear-model* oracle.

The production allocator never enumerates all head assignments.  This helper
is the exception used only for an audit on one layer and four to six heads:
it enumerates 3**H assignments, runs the complete clear ViT for every
assignment on the fixed calibration set, and compares that Pareto set with
the count-state DP's one proxy-best assignment per composition.

No fixed graph, MPC process, or CKKS process is started here.  Cost coordinates
come from the existing measured anchors or the explicitly labelled offline
composition model; the accuracy/risk coordinate is from a real full clear
ViT evaluation, not a sum of isolated-head errors.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import offline_cost_aware_head_allocator as base  # noqa: E402
from offline_allocator_experiment import (  # noqa: E402
    SearchCompositionCostModel,
    clear_baseline_batch,
    close_pool,
    create_pool,
    fit_cost_models,
    load_probe_points,
    read_csv,
    state_from_plan,
)


SOFTMAX, RELU, LINEAR = base.PROTOCOLS
PROTOCOLS = base.PROTOCOLS


def parse_heads(value: str) -> list[int]:
    heads = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(heads) < 4 or len(heads) > 6:
        raise ValueError("oracle must contain 4--6 heads")
    if len(set(heads)) != len(heads):
        raise ValueError("oracle heads must be distinct")
    if any(head < 0 or head >= base.MODEL_HEADS for head in heads):
        raise ValueError("oracle head is outside ViT-B/16")
    return heads


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plan_for_combo(layer: int, heads: list[int], combo: tuple[str, ...],
                   alphas: list[list[float]]) -> tuple[tuple[tuple[str, float], ...], ...]:
    plan = [[(SOFTMAX, 1.0) for _ in range(base.MODEL_HEADS)]
            for _ in range(base.MODEL_LAYERS)]
    for head, protocol in zip(heads, combo):
        plan[layer][head] = (
            protocol, alphas[layer][head] if protocol == LINEAR else 1.0)
    return tuple(tuple(entries) for entries in plan)


def combo_counts(heads: list[int], combo: tuple[str, ...]) -> tuple[int, int, int]:
    counts = [base.MODEL_HEADS - len(heads), 0, 0]
    for protocol in combo:
        if protocol == SOFTMAX:
            counts[0] += 1
        elif protocol == RELU:
            counts[1] += 1
        elif protocol == LINEAR:
            counts[2] += 1
        else:
            raise ValueError(f"unknown protocol: {protocol}")
    return tuple(counts)  # type: ignore[return-value]


def dp_oracle(layer: int, heads: list[int], candidates: dict[tuple[int, int, str], dict[str, Any]],
              alphas: list[list[float]]) -> tuple[
                  dict[tuple[int, int, int], tuple[float, tuple[str, ...]]], int, int]:
    """Keep one minimum isolated-proxy path per exact count state.

    Unlike the legacy measured-anchor audit, this DP keeps every count state
    because the SearchCompositionCostModel can label unmeasured compositions
    as predictions.  The later clear evaluation decides the *true* Pareto
    coordinates for the returned DP paths.
    """
    states: dict[tuple[int, int, int], tuple[float, tuple[str, ...]]] = {
        (0, 0, 0): (0.0, ())}
    transitions = 0
    max_states = 1
    for head in heads:
        next_states: dict[tuple[int, int, int], tuple[float, tuple[str, ...]]] = {}
        for used, (score, path_names) in states.items():
            for protocol in PROTOCOLS:
                transitions += 1
                increment = ((1, 0, 0) if protocol == SOFTMAX else
                             (0, 1, 0) if protocol == RELU else (0, 0, 1))
                counts = tuple(used[index] + increment[index]
                               for index in range(3))
                if protocol == SOFTMAX:
                    increment_score = 0.0
                else:
                    row = candidates[(layer, head, protocol)]
                    increment_score = float(row["proxy_score"])
                candidate = (score + increment_score,
                             path_names + (protocol,))
                old = next_states.get(counts)
                if old is None or candidate < old:
                    next_states[counts] = candidate
        states = next_states
        max_states = max(max_states, len(states))
    fixed_softmax = base.MODEL_HEADS - len(heads)
    result = {
        (used[0] + fixed_softmax, used[1], used[2]): value
        for used, value in states.items()
    }
    return result, transitions, max_states


def risk(metrics: dict[str, Any], logit_weight: float) -> float:
    return (max(0.0, float(metrics["loss_delta"])) +
            logit_weight * float(metrics["relative_l2"]))


def point(entry: dict[str, Any]) -> tuple[float, float, float]:
    return (float(entry["actual_risk_score"]),
            float(entry["predicted_latency_ms"]),
            float(entry["predicted_communication_bytes"]))


def pareto(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in entries:
        if not any(base.dominates(point(other), point(entry))
                   for other in entries if other is not entry):
            result.append(entry)
    return sorted(result, key=lambda item: (
        point(item), base.composition_name(item["counts"]),
        item["assignment_id"]))


def cost_source(cost: dict[str, Any]) -> str:
    layers = cost.get("layers", [])
    if not layers:
        return "unknown"
    observations = [int(layer.get("measured_observation_count", 0))
                    for layer in layers]
    return ("direct_measured_anchor" if all(value > 0 for value in observations)
            else "composition_model_prediction")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--preprocess-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--isolated-candidates", type=Path, required=True)
    parser.add_argument("--allocator-costs", type=Path, required=True)
    parser.add_argument("--supplemental-costs", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--heads", type=str, default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--logit-weight", type=float, default=0.25)
    parser.add_argument("--cost-ridge", type=float, default=1e-3)
    args = parser.parse_args()

    if args.layer < 0 or args.layer >= base.MODEL_LAYERS:
        raise ValueError("oracle layer is outside ViT-B/16")
    if args.workers <= 0 or args.batch_size <= 0:
        raise ValueError("workers and batch-size must be positive")
    heads = parse_heads(args.heads)

    model = base.vr.load_model(args.model)
    if (int(model["layers"]) != base.MODEL_LAYERS or
            int(model["heads"]) != base.MODEL_HEADS or
            int(model["tokens"]) != base.MODEL_TOKENS):
        raise ValueError("oracle requires the audited ViT-B/16 model")
    preprocess = base.load_preprocess(args.preprocess_manifest, model)
    calibration_rows = read_csv(args.calibration_manifest)
    samples = base.read_manifest(
        args.calibration_manifest, args.image_root, len(calibration_rows))
    if not samples:
        raise ValueError("calibration manifest is empty")
    pixels = base.load_pixels_for_samples(samples, preprocess)
    labels = [sample.label for sample in samples]

    isolated_rows = read_csv(args.isolated_candidates)
    isolated = base.load_candidate_index(isolated_rows)
    alphas = base.alphas_from_isolated(
        isolated_rows, base.MODEL_LAYERS, base.MODEL_HEADS)
    existing, train, _heldout, _missing_train, _missing_heldout = load_probe_points(
        args.allocator_costs, args.supplemental_costs, args.probe_manifest,
        allow_missing=True)
    measured = base.MeasuredCompositionCostModel(
        args.allocator_costs, base.MODEL_LAYERS, base.MODEL_HEADS)
    latency_model, communication_model = fit_cost_models(
        existing + train, args.cost_ridge)
    search_cost_model = SearchCompositionCostModel(
        measured, existing + train, latency_model, communication_model)

    combos = list(itertools.product(PROTOCOLS, repeat=len(heads)))
    plans = {
        f"exhaustive_{index:03d}": plan_for_combo(
            args.layer, heads, combo, alphas)
        for index, combo in enumerate(combos)
    }
    states = {
        name: state_from_plan(plan, isolated, search_cost_model)
        for name, plan in plans.items()
    }
    validation_cache = args.output.with_name(
        args.output.stem + ".validation_metrics.json")
    validation_rows: list[dict[str, Any]] | None = None
    if validation_cache.is_file():
        cached = json.loads(validation_cache.read_text(encoding="utf-8"))
        if (cached.get("layer") == args.layer and
                cached.get("heads") == heads and
                cached.get("samples") == len(samples)):
            validation_rows = cached.get("rows")
            print(f"OURS_TRUE_ORACLE_VALIDATE reuse={len(validation_rows or [])}/"
                  f"{len(plans)}", flush=True)
    if validation_rows is None:
        baseline = clear_baseline_batch(model, pixels, args.batch_size)
        pool = create_pool(model, pixels, labels, pixels, labels,
                           args.workers, args.logit_weight, args.batch_size)
        try:
            validation_rows = base.run_validation(
                model, pixels, labels, baseline, states, args.workers, pool,
                args.batch_size, progress_every=max(1, len(plans) // 9))
        finally:
            close_pool(pool)
        validation_cache.write_text(json.dumps({
            "layer": args.layer,
            "heads": heads,
            "samples": len(samples),
            "rows": validation_rows,
        }, sort_keys=True) + "\n", encoding="utf-8")
    validation_by_name = {row["candidate"]: row for row in validation_rows}
    if set(validation_by_name) != set(plans):
        raise ValueError("true oracle validation is incomplete")

    exhaustive_entries: list[dict[str, Any]] = []
    for index, combo in enumerate(combos):
        name = f"exhaustive_{index:03d}"
        state = states[name]
        metrics = validation_by_name[name]
        cost = search_cost_model.predict_plan(plans[name])
        exhaustive_entries.append({
            "assignment_id": name,
            "combo": ";".join(combo),
            "counts": combo_counts(heads, combo),
            "plan": plans[name],
            "state": state,
            "metrics": metrics,
            "cost": cost,
            "actual_risk_score": risk(metrics, args.logit_weight),
            "predicted_latency_ms": cost["latency_ms"],
            "predicted_communication_bytes": cost["communication_bytes"],
            "cost_source": cost_source(cost),
        })

    dp_values, transitions, max_states = dp_oracle(
        args.layer, heads, isolated, alphas)
    dp_entries: list[dict[str, Any]] = []
    for index, (counts, (proxy, selected_combo)) in enumerate(sorted(dp_values.items())):
        plan = plan_for_combo(args.layer, heads, selected_combo, alphas)
        plan_key = base.plan_key(plan)
        match = next(entry for entry in exhaustive_entries
                     if base.plan_key(entry["plan"]) == plan_key)
        dp_entries.append({**match, "assignment_id": f"dp_{index:03d}",
                           "dp_proxy_score": proxy})

    exhaustive_pareto = pareto(exhaustive_entries)
    dp_pareto = pareto(dp_entries)
    exhaustive_by_counts: dict[tuple[int, int, int], float] = {}
    for entry in exhaustive_entries:
        counts = entry["counts"]
        exhaustive_by_counts[counts] = min(
            exhaustive_by_counts.get(counts, math.inf),
            float(entry["actual_risk_score"]))
    same_count_gaps = [
        float(entry["actual_risk_score"]) - exhaustive_by_counts[entry["counts"]]
        for entry in dp_entries
    ]
    dp_plan_keys = {base.plan_key(entry["plan"]) for entry in dp_pareto}
    exhaustive_plan_keys = {base.plan_key(entry["plan"])
                            for entry in exhaustive_pareto}
    overlap = len(dp_plan_keys & exhaustive_plan_keys)
    covered_exhaustive = sum(
        1 for entry in exhaustive_pareto
        if entry["counts"] in {item["counts"] for item in dp_entries})
    pareto_coverage_percent = (
        covered_exhaustive / len(exhaustive_pareto) * 100.0
        if exhaustive_pareto else 100.0)
    assignment_overlap_percent = (
        overlap / len(exhaustive_pareto) * 100.0
        if exhaustive_pareto else 100.0)
    summary = {
        "row_type": "summary",
        "method": "DP_vs_true_exhaustive_clear_model",
        "dataset": "fixed_calibration",
        "layer": args.layer,
        "head_subset": ";".join(map(str, heads)),
        "enumerated_assignments": len(combos),
        "feasible_assignments": len(combos),
        "dp_count_states": len(dp_entries),
        "dp_transition_count": transitions,
        "dp_max_states": max_states,
        "dp_pareto_points": len(dp_pareto),
        "exhaustive_pareto_points": len(exhaustive_pareto),
        "dp_exact_pareto_overlap": overlap,
        "dp_pareto_coverage_percent": pareto_coverage_percent,
        "assignment_overlap_percent": assignment_overlap_percent,
        "exhaustive_pareto_coverage_by_dp_composition": covered_exhaustive,
        "same_count_mean_actual_risk_gap": (
            sum(same_count_gaps) / len(same_count_gaps)
            if same_count_gaps else 0.0),
        "same_count_max_actual_risk_gap": max(same_count_gaps, default=0.0),
        "pareto_set_equal": dp_plan_keys == exhaustive_plan_keys,
        "oracle_runtime_accuracy": "full_clear_vit",
        "oracle_cost_axis": "measured_anchor_else_composition_model_prediction",
        "comparison_scope": (
            "one layer; 4--6 heads only; exhaustive 3^H clear assignments; "
            "no full-model brute-force allocator search"),
    }
    rows: list[dict[str, Any]] = [summary]
    fields = list(summary)
    fields.extend([
        "assignment_id", "combo", "composition", "actual_accuracy_percent",
        "actual_accuracy_drop_pp", "actual_validation_loss", "actual_loss_delta",
        "actual_relative_l2", "actual_risk_score", "dp_proxy_score",
        "predicted_latency_ms", "predicted_communication_bytes", "cost_source",
        "same_count_true_risk_gap", "is_in_other_pareto",
    ])
    for method, entries, other in (
            ("DP", dp_pareto, exhaustive_plan_keys),
            ("exhaustive_true", exhaustive_pareto, dp_plan_keys)):
        for entry in entries:
            metrics = entry["metrics"]
            rows.append({
                "row_type": "pareto_point",
                "method": method,
                "dataset": "fixed_calibration",
                "layer": args.layer,
                "head_subset": ";".join(map(str, heads)),
                "assignment_id": entry["assignment_id"],
                "combo": entry["combo"],
                "composition": base.composition_name(entry["counts"]),
                "actual_accuracy_percent": float(metrics["candidate_accuracy"]) * 100.0,
                "actual_accuracy_drop_pp": float(metrics["accuracy_drop"]) * 100.0,
                "actual_validation_loss": metrics["validation_loss"],
                "actual_loss_delta": metrics["loss_delta"],
                "actual_relative_l2": metrics["relative_l2"],
                "actual_risk_score": entry["actual_risk_score"],
                "dp_proxy_score": entry.get("dp_proxy_score", ""),
                "predicted_latency_ms": entry["predicted_latency_ms"],
                "predicted_communication_bytes": entry["predicted_communication_bytes"],
                "cost_source": entry["cost_source"],
                "same_count_true_risk_gap": (
                    float(entry["actual_risk_score"]) -
                    exhaustive_by_counts[entry["counts"]]),
                "is_in_other_pareto": base.plan_key(entry["plan"]) in other,
                "enumerated_assignments": len(combos),
                "feasible_assignments": len(combos),
                "dp_count_states": len(dp_entries),
                "dp_transition_count": transitions,
                "dp_max_states": max_states,
                "dp_pareto_points": len(dp_pareto),
                "exhaustive_pareto_points": len(exhaustive_pareto),
                "dp_exact_pareto_overlap": overlap,
                "dp_pareto_coverage_percent": pareto_coverage_percent,
                "assignment_overlap_percent": assignment_overlap_percent,
                "exhaustive_pareto_coverage_by_dp_composition": covered_exhaustive,
                "same_count_mean_actual_risk_gap": summary[
                    "same_count_mean_actual_risk_gap"],
                "same_count_max_actual_risk_gap": summary[
                    "same_count_max_actual_risk_gap"],
                "pareto_set_equal": summary["pareto_set_equal"],
                "oracle_runtime_accuracy": summary["oracle_runtime_accuracy"],
                "oracle_cost_axis": summary["oracle_cost_axis"],
                "comparison_scope": summary["comparison_scope"],
            })
    # Preserve the summary-first layout and all fields in both row types.
    fields = list(dict.fromkeys(fields + [
        "dp_count_states", "exhaustive_pareto_coverage_by_dp_composition",
        "oracle_runtime_accuracy", "oracle_cost_axis", "comparison_scope",
    ]))
    write_csv(args.output, fields, rows)

    report_path = args.output.parent / "allocator_experiment_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["oracle"] = {
            "layer": args.layer,
            "head_subset": heads,
            "enumerated_assignments": len(combos),
            "comparison_file": str(args.output.resolve()),
            "mode": "true_full_clear_vit_exhaustive_vs_count_dp",
            "dataset": "fixed_calibration",
            "dp_transition_count": transitions,
            "dp_max_states": max_states,
            "dp_pareto_points": len(dp_pareto),
            "exhaustive_pareto_points": len(exhaustive_pareto),
            "dp_exact_pareto_overlap": overlap,
            "same_count_mean_actual_risk_gap": summary[
                "same_count_mean_actual_risk_gap"],
            "same_count_max_actual_risk_gap": summary[
                "same_count_max_actual_risk_gap"],
            "pareto_set_equal": summary["pareto_set_equal"],
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print(json.dumps({
        "status": "PASS_TRUE_ORACLE",
        "enumerated_assignments": len(combos),
        "dp_count_states": len(dp_entries),
        "dp_pareto_points": len(dp_pareto),
        "exhaustive_pareto_points": len(exhaustive_pareto),
        "dp_exact_pareto_overlap": overlap,
        "dp_pareto_coverage_percent": pareto_coverage_percent,
        "assignment_overlap_percent": assignment_overlap_percent,
        "same_count_mean_actual_risk_gap": summary[
            "same_count_mean_actual_risk_gap"],
        "same_count_max_actual_risk_gap": summary[
            "same_count_max_actual_risk_gap"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
