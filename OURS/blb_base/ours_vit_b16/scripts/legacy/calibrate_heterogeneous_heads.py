#!/usr/bin/env python3
"""Public-data, no-training calibration for ViT head protocols.

The script measures per-head attention-context relative-L2 disturbance for
Softmax-MPC, ReLU-Norm-MPC, and linear scaling. Formal cost selection compares
exact complete 12-Head assignments measured through the full A3/A4 attention
boundary. It never divides a packed run per Head, adds independent path rows,
copies one layer's timing, or extrapolates a fixed BLB remainder. Linear is
eligible only in a real one-level A4 CKKS run; the A1/A2 MPC semantic surrogate
is excluded. The assignment is fixed before inference and rerun through the
FP32/fixed-clear global guard.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import vit_reference as vr


PROTOCOLS = ("softmax_mpc", "relu_norm_mpc", "he_linear")
COST_COLUMNS = {
    "layer", "protocol", "latency_ms", "he_ms", "mpc_ms",
    "conversion_ms", "communication_bytes", "rounds", "head_count",
    "packing_utilization", "multiplicative_depth", "ct_ct_multiply",
    "rotations", "relinearization", "rescale", "measured", "backend",
    "network_gbps", "rtt_ms", "execution_domain", "cost_stage",
    "measurement_scope", "conversion_scope", "cost_scope", "gpu_topology",
    "selection_eligible",
}

ASSIGNMENT_COST_COLUMNS = {
    "layer", "ablation", "softmax_heads", "relu_heads", "linear_heads",
    "latency_ms", "he_instrumented_ms", "conversion_instrumented_ms",
    "communication_bytes", "rounds", "protocol_groups", "mpc_group_calls",
    "native_linear_groups", "ckks_to_mpc", "mpc_to_ckks",
    "he_ct_pt_multiply", "he_ct_ct_multiply", "he_rotations", "he_rescales",
    "measurement_scope", "cost_unit", "network_gbps", "rtt_ms", "backend",
    "gpu_topology", "image_id", "binary_sha256", "plan_sha256",
    "protocols_original", "alphas_original", "physical_protocols",
    "original_order", "a4_data_level_extension", "surrogate",
    "dynamic_selection", "source_run",
}

# Legacy A2 diagnostic helpers below retain the historical baseline constants
# for reading old reports. The formal CLI rejects that format and never emits
# an E2E estimate from them.
BLB_BASELINE_E2E_MS = 498_104.0
BLB_BASELINE_NORMALIZATION_MS = 97_486.694
BLB_BASELINE_NON_NORMALIZATION_MS = (
    BLB_BASELINE_E2E_MS - BLB_BASELINE_NORMALIZATION_MS)
TARGET_E2E_MS = 480_000.0


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int | None


@dataclass(frozen=True)
class Cost:
    latency_ms: float
    he_ms: float
    mpc_ms: float
    conversion_ms: float
    communication_bytes: int
    rounds: int
    head_count: int
    packing_utilization: float
    multiplicative_depth: int
    ct_ct_multiply: int
    rotations: int
    relinearization: int
    rescale: int
    execution_domain: str
    cost_stage: str
    measurement_scope: str
    conversion_scope: str
    cost_scope: str
    gpu_topology: str


@dataclass(frozen=True)
class AssignmentCost:
    layer: int
    ablation: str
    protocols: tuple[str, ...]
    latency_ms: float
    he_ms: float
    conversion_ms: float
    communication_bytes: int
    rounds: int
    protocol_groups: int
    mpc_group_calls: int
    native_linear_groups: int
    ckks_to_mpc: int
    mpc_to_ckks: int
    ct_pt_multiply: int
    ct_ct_multiply: int
    rotations: int
    rescales: int
    a4_data_level_extension: int
    image_id: str
    binary_sha256: str
    plan_sha256: str
    source_run: str


def load_assignment_costs(
        path: Path, layers: int, heads: int,
        max_he_depth: int) -> list[AssignmentCost]:
    """Load only whole, directly measured A3/A4 attention assignments."""
    result: list[AssignmentCost] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = ASSIGNMENT_COST_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"assignment-cost CSV missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            layer = int(row["layer"])
            ablation = row["ablation"].upper()
            protocols = tuple(row["protocols_original"].split(";"))
            physical_protocols = tuple(row["physical_protocols"].split(";"))
            original_order = tuple(int(value) for value in
                                   row["original_order"].split(";"))
            if (layer < 0 or layer >= layers or ablation not in {"A3", "A4"} or
                    len(protocols) != heads or len(physical_protocols) != heads or
                    sorted(original_order) != list(range(heads)) or
                    any(protocol not in PROTOCOLS for protocol in protocols)):
                raise ValueError(f"invalid measured assignment at line {line}")
            expected_physical = tuple(protocols[index]
                                      for index in original_order)
            if expected_physical != physical_protocols:
                raise ValueError(
                    f"physical/original Head mapping mismatch at line {line}")
            groups = 1 + sum(
                physical_protocols[index] != physical_protocols[index - 1]
                for index in range(1, heads))
            for protocol in PROTOCOLS:
                positions = [index for index, value in
                             enumerate(physical_protocols)
                             if value == protocol]
                if positions and positions != list(
                        range(positions[0], positions[-1] + 1)):
                    raise ValueError(
                        f"protocol is not one contiguous group at line {line}")
            counts = {protocol: protocols.count(protocol)
                      for protocol in PROTOCOLS}
            if (counts["softmax_mpc"] != int(row["softmax_heads"]) or
                    counts["relu_norm_mpc"] != int(row["relu_heads"]) or
                    counts["he_linear"] != int(row["linear_heads"]) or
                    groups != int(row["protocol_groups"])):
                raise ValueError(
                    f"measured assignment counts disagree at line {line}")
            extension = int(row["a4_data_level_extension"])
            if extension != (1 if ablation == "A4" else 0):
                raise ValueError(
                    f"assignment does not use the active one-level chain at line {line}")
            if (ablation == "A3" and counts["he_linear"]):
                raise ValueError(f"A3 row contains native Linear at line {line}")
            if counts["he_linear"] and (ablation != "A4" or extension != 1):
                raise ValueError(f"Linear row is not one-level A4 at line {line}")
            if extension > max_he_depth:
                continue
            if (row["measurement_scope"] != "complete_attention_assignment" or
                    row["cost_unit"] != "direct_assignment_not_per_head" or
                    row["backend"].lower() != "phantom_gpu" or
                    row["gpu_topology"] != "separate_physical_gpus" or
                    row["surrogate"] != "0" or
                    row["dynamic_selection"] != "0" or
                    not math.isclose(float(row["network_gbps"]), 1.0,
                                     rel_tol=0.0, abs_tol=1e-9) or
                    not math.isclose(float(row["rtt_ms"]), 0.3,
                                     rel_tol=0.0, abs_tol=1e-9)):
                raise ValueError(
                    f"ineligible assignment provenance at line {line}")
            if (not row["image_id"].startswith("sha256:") or
                    len(row["binary_sha256"]) != 64 or
                    len(row["plan_sha256"]) != 64):
                raise ValueError(
                    f"assignment lacks immutable hashes at line {line}")
            cost = AssignmentCost(
                layer=layer, ablation=ablation, protocols=protocols,
                latency_ms=float(row["latency_ms"]),
                he_ms=float(row["he_instrumented_ms"]),
                conversion_ms=float(row["conversion_instrumented_ms"]),
                communication_bytes=int(row["communication_bytes"]),
                rounds=int(row["rounds"]),
                protocol_groups=int(row["protocol_groups"]),
                mpc_group_calls=int(row["mpc_group_calls"]),
                native_linear_groups=int(row["native_linear_groups"]),
                ckks_to_mpc=int(row["ckks_to_mpc"]),
                mpc_to_ckks=int(row["mpc_to_ckks"]),
                ct_pt_multiply=int(row["he_ct_pt_multiply"]),
                ct_ct_multiply=int(row["he_ct_ct_multiply"]),
                rotations=int(row["he_rotations"]),
                rescales=int(row["he_rescales"]),
                a4_data_level_extension=extension,
                image_id=row["image_id"],
                binary_sha256=row["binary_sha256"],
                plan_sha256=row["plan_sha256"],
                source_run=row["source_run"],
            )
            numeric = (cost.latency_ms, cost.he_ms, cost.conversion_ms)
            if (any(not math.isfinite(value) or value < 0.0
                    for value in numeric) or cost.communication_bytes < 0 or \
                    cost.rounds < 0):
                raise ValueError(f"invalid assignment cost at line {line}")
            result.append(cost)
    identities = [(cost.layer, cost.plan_sha256, cost.ablation)
                  for cost in result]
    if not result or len(identities) != len(set(identities)):
        raise ValueError("assignment-cost CSV is empty or contains duplicates")
    return result

def samples_from_args(args: argparse.Namespace) -> list[Sample]:
    result = [Sample(path, None) for path in args.input]
    if args.manifest:
        with args.manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "input" not in reader.fieldnames:
                raise ValueError("calibration manifest requires an input column")
            for row in reader:
                path = Path(row["input"])
                if not path.is_absolute():
                    path = args.manifest.parent / path
                label_text = (row.get("label") or "").strip()
                result.append(Sample(path.resolve(),
                                     int(label_text) if label_text else None))
    if not result:
        raise ValueError("provide --input or --manifest calibration data")
    for sample in result:
        if not sample.path.is_file():
            raise ValueError(f"missing calibration input: {sample.path}")
    return result


def load_costs(
        path: Path, layers: int, max_he_depth: int
) -> dict[tuple[int, str, int], Cost]:
    """Load selection-eligible measured group costs.

    A2 Linear rows are useful wiring diagnostics, but they execute in MPC and
    are never a performance proxy for A4.  Such rows remain in the source CSV
    with selection_eligible=0 and are skipped here.
    """
    del max_he_depth  # Enforced when a real A4 CKKS row is considered.
    result: dict[tuple[int, str, int], Cost] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = COST_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"cost CSV missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            layer = int(row["layer"])
            protocol = row["protocol"].strip()
            if layer < 0 or layer >= layers or protocol not in PROTOCOLS:
                raise ValueError(f"invalid layer/protocol in cost CSV line {line}")
            if row["measured"].strip().lower() not in {"1", "true", "yes"}:
                continue
            if row["selection_eligible"].strip().lower() not in {
                    "1", "true", "yes"}:
                continue
            if row["backend"].strip().lower() not in {
                    "phantom_gpu", "gpu_phantom", "phantom_device"}:
                raise ValueError(f"non-Phantom cost row at line {line}")
            network_gbps = float(row["network_gbps"])
            rtt_ms = float(row["rtt_ms"])
            if (not math.isclose(network_gbps, 1.0, rel_tol=0.0,
                                 abs_tol=1e-9) or
                    not math.isclose(rtt_ms, 0.3, rel_tol=0.0,
                                     abs_tol=1e-9)):
                raise ValueError(
                    f"cost row at line {line} is not measured at "
                    "1 Gbps / 0.3 ms RTT")
            execution_domain = row["execution_domain"].strip().lower()
            expected_domains = (
                {"mpc_ckks_hybrid"} if protocol != "he_linear" else {"ckks"})
            if execution_domain not in expected_domains:
                raise ValueError(
                    f"invalid execution_domain for {protocol} at line {line}")
            cost_stage = row["cost_stage"].strip().upper()
            if cost_stage not in {"A2", "A4"}:
                raise ValueError(f"invalid cost_stage at line {line}")
            if protocol == "he_linear" and cost_stage != "A4":
                raise ValueError(
                    f"HE-linear selection cost must come from A4 at line {line}")
            if protocol != "he_linear" and cost_stage != "A2":
                raise ValueError(
                    f"{protocol} normalization cost must come from A2 at line {line}")
            measurement_scope = row["measurement_scope"].strip()
            conversion_scope = row["conversion_scope"].strip()
            cost_scope = row["cost_scope"].strip()
            gpu_topology = row["gpu_topology"].strip()
            if cost_stage == "A2":
                if measurement_scope not in {
                        "direct_layer_normalization_only",
                        "shape_reuse_normalization_only"}:
                    raise ValueError(
                        f"invalid A2 measurement_scope at line {line}")
                if (conversion_scope != "excluded_deferred_to_A3" or
                        cost_scope != "a2_normalization_only"):
                    raise ValueError(
                        f"A2 cost provenance is ambiguous at line {line}")
                if gpu_topology != "separate_physical_gpus":
                    raise ValueError(
                        f"A2 GPU topology mismatch at line {line}")
            cost = Cost(
                latency_ms=float(row["latency_ms"]),
                he_ms=float(row["he_ms"]),
                mpc_ms=float(row["mpc_ms"]),
                conversion_ms=float(row["conversion_ms"]),
                communication_bytes=int(row["communication_bytes"]),
                rounds=int(row["rounds"]),
                head_count=int(row["head_count"]),
                packing_utilization=float(row["packing_utilization"]),
                multiplicative_depth=int(row["multiplicative_depth"]),
                ct_ct_multiply=int(row["ct_ct_multiply"]),
                rotations=int(row["rotations"]),
                relinearization=int(row["relinearization"]),
                rescale=int(row["rescale"]),
                execution_domain=execution_domain,
                cost_stage=cost_stage,
                measurement_scope=measurement_scope,
                conversion_scope=conversion_scope,
                cost_scope=cost_scope,
                gpu_topology=gpu_topology,
            )
            numeric = (cost.latency_ms, cost.he_ms, cost.mpc_ms,
                       cost.conversion_ms,
                       cost.packing_utilization)
            if any(not math.isfinite(value) or value < 0.0 for value in numeric):
                raise ValueError(f"invalid numeric cost at line {line}")
            if (cost.communication_bytes < 0 or cost.rounds < 0 or
                    cost.head_count <= 0 or cost.head_count > 12):
                raise ValueError(f"invalid count cost at line {line}")
            if cost.packing_utilization > 1.0:
                raise ValueError(f"packing utilization exceeds one at line {line}")
            key = (layer, protocol, cost.head_count)
            if key in result:
                raise ValueError(f"duplicate cost row for {key}")
            result[key] = cost
    if not result:
        raise ValueError("cost CSV has no selection-eligible measured group rows")
    return result


def error_metrics(reference: list[np.ndarray], candidate: list[np.ndarray]) -> dict[str, float | int]:
    if len(reference) != len(candidate):
        raise ValueError("error metric observation count mismatch")
    count = 0
    squared = 0.0
    reference_squared = 0.0
    maximum = 0.0
    for expected, actual in zip(reference, candidate):
        expected_array = np.asarray(expected)
        difference = np.asarray(actual) - expected_array
        count += difference.size
        squared += float(np.sum(difference * difference, dtype=np.float64))
        reference_squared += float(np.sum(
            expected_array * expected_array, dtype=np.float64))
        maximum = max(maximum, float(np.max(np.abs(difference), initial=0.0)))
    l2 = math.sqrt(squared)
    reference_l2 = math.sqrt(reference_squared)
    relative_l2 = (l2 / reference_l2 if reference_l2 > 0.0
                   else (0.0 if l2 == 0.0 else math.inf))
    return {"elements": count, "max_abs": maximum,
            "mse": squared / count if count else 0.0,
            "l2": l2, "reference_l2": reference_l2,
            "relative_l2": relative_l2}


def derive_contexts(records: dict[str, dict[str, Any]], layer: int,
                    logical: int, heads: int, fixed: bool) -> list[dict[str, np.ndarray]]:
    score = records[f"block_{layer}.qk"]["value"][:, :logical, :logical]
    probability = records[f"block_{layer}.softmax"]["value"][:, :logical, :logical]
    value = records[f"block_{layer}.v"]["value"][:logical]
    head_dim = value.shape[-1] // heads
    values = value.reshape(logical, heads, head_dim).transpose(1, 0, 2)
    result: list[dict[str, np.ndarray]] = []
    for head in range(heads):
        if fixed:
            p_int = vr.round_int(probability[head], vr.ACT_SCALE)
            v_int = vr.round_int(values[head], 11)
            raw = vr.matmul_integer(p_int, v_int)
            target = (vr._floor_shift(raw.astype(np.int64), 11)
                      .astype(np.float64) / float(1 << vr.ACT_SCALE))
        else:
            target = probability[head] @ values[head]
        if fixed:
            relu_weight = vr.fixed_relu_normalization(score[head])
            relu_raw = vr.matmul_integer(
                vr.round_int(relu_weight, vr.ACT_SCALE),
                vr.round_int(values[head], 11)).astype(np.int64)
            relu_context = (vr._floor_shift(relu_raw, 11).astype(np.float64) /
                            float(1 << vr.ACT_SCALE))
        else:
            positive = np.maximum(score[head], 0.0)
            denominator = positive.sum(axis=-1, keepdims=True,
                                       dtype=np.float64)
            relu_weight = np.divide(positive, denominator,
                                    out=np.zeros_like(positive),
                                    where=denominator != 0.0)
            relu_context = relu_weight @ values[head]
        result.append({"target": target, "relu": relu_context,
                       "score": score[head].copy(), "value": values[head].copy()})
    return result


def linear_context(observation: dict[str, np.ndarray], alpha: float,
                   fixed: bool) -> np.ndarray:
    if not fixed:
        return (alpha * observation["score"]) @ observation["value"]
    weights = vr.fixed_linear_scaling(observation["score"], alpha)
    raw = vr.matmul_integer(vr.round_int(weights, vr.ACT_SCALE),
                            vr.round_int(observation["value"], 11)).astype(np.int64)
    return (vr._floor_shift(raw, 11).astype(np.float64) /
            float(1 << vr.ACT_SCALE))


def evaluate_plan(model: dict[str, Any], samples: list[Sample],
                  plan: list[list[tuple[str, float]]]) -> dict[str, Any]:
    fp_logits: list[np.ndarray] = []
    fixed_logits: list[np.ndarray] = []
    for sample in samples:
        pixels = vr.load_input(sample.path)
        fp_logits.append(vr.Graph(model, fixed=False, head_plan=plan).run(pixels)[0])
        fixed_logits.append(vr.Graph(model, fixed=True, head_plan=plan).run(pixels)[0])
    return {"fp_logits": fp_logits, "fixed_logits": fixed_logits}


def classification_metrics(samples: list[Sample], baseline: dict[str, Any],
                           candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_top1 = [int(np.argmax(row)) for row in baseline["fp_logits"]]
    candidate_top1 = [int(np.argmax(row)) for row in candidate["fp_logits"]]
    agreement = sum(a == b for a, b in zip(baseline_top1, candidate_top1)) / len(samples)
    labeled = [(index, sample.label) for index, sample in enumerate(samples)
               if sample.label is not None]
    base_accuracy = None
    candidate_accuracy = None
    if labeled:
        base_accuracy = sum(baseline_top1[i] == label for i, label in labeled) / len(labeled)
        candidate_accuracy = sum(candidate_top1[i] == label for i, label in labeled) / len(labeled)
    return {
        "top1_agreement": agreement,
        "baseline_accuracy": base_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_drop": ((base_accuracy - candidate_accuracy)
                          if labeled else None),
        "fp32_logits": error_metrics(baseline["fp_logits"], candidate["fp_logits"]),
        "fixed_logits": error_metrics(baseline["fixed_logits"],
                                      candidate["fixed_logits"]),
    }


def write_plan(path: Path, plan: list[list[tuple[str, float]]],
               original_order: list[list[int]] | None = None) -> None:
    if original_order is None:
        original_order = [list(range(len(entries))) for entries in plan]
    if len(original_order) != len(plan):
        raise ValueError("head order layer count does not match plan")
    with path.open("w", encoding="utf-8") as output:
        output.write("OURS_VIT_HEAD_PLAN_V1\n")
        output.write("layers=12,heads=12,logical_tokens=197,physical_tokens=256\n")
        output.write("layer,physical_head,original_head,protocol,alpha\n")
        for layer, entries in enumerate(plan):
            order = original_order[layer]
            if sorted(order) != list(range(len(entries))):
                raise ValueError(f"layer {layer} head order is not a permutation")
            for physical_head, original_head in enumerate(order):
                protocol, alpha = entries[original_head]
                output.write(
                    f"{layer},{physical_head},{original_head},{protocol},"
                    f"{alpha:.17g}\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cost_json(cost: Cost) -> dict[str, Any]:
    return {
        "latency_ms": cost.latency_ms,
        "he_ms": cost.he_ms,
        "mpc_ms": cost.mpc_ms,
        "conversion_ms": cost.conversion_ms,
        "communication_bytes": cost.communication_bytes,
        "rounds": cost.rounds,
        "group_head_count": cost.head_count,
        "cost_unit": "measured_protocol_group_not_per_head_extrapolation",
        "packing_utilization": cost.packing_utilization,
        "multiplicative_depth": cost.multiplicative_depth,
        "ct_ct_multiply": cost.ct_ct_multiply,
        "rotations": cost.rotations,
        "relinearization": cost.relinearization,
        "rescale": cost.rescale,
        "execution_domain": cost.execution_domain,
        "cost_stage": cost.cost_stage,
        "measurement_scope": cost.measurement_scope,
        "conversion_scope": cost.conversion_scope,
        "cost_scope": cost.cost_scope,
        "gpu_topology": cost.gpu_topology,
    }


def assignment_cost_json(cost: AssignmentCost) -> dict[str, Any]:
    return {
        "latency_ms": cost.latency_ms,
        "he_instrumented_ms": cost.he_ms,
        "conversion_instrumented_ms": cost.conversion_ms,
        "communication_bytes": cost.communication_bytes,
        "rounds": cost.rounds,
        "protocol_groups": cost.protocol_groups,
        "mpc_group_calls": cost.mpc_group_calls,
        "native_linear_groups": cost.native_linear_groups,
        "ckks_to_mpc": cost.ckks_to_mpc,
        "mpc_to_ckks": cost.mpc_to_ckks,
        "he_ct_pt_multiply": cost.ct_pt_multiply,
        "he_ct_ct_multiply": cost.ct_ct_multiply,
        "he_rotations": cost.rotations,
        "he_rescales": cost.rescales,
        "a4_data_level_extension": cost.a4_data_level_extension,
        "ablation": cost.ablation,
        "image_id": cost.image_id,
        "binary_sha256": cost.binary_sha256,
        "plan_sha256": cost.plan_sha256,
        "source_run": cost.source_run,
        "measurement_scope": "complete_attention_assignment",
        "cost_unit": "direct_assignment_not_per_head",
    }


def select_measured_assignments(
        layers: int, heads: int, alphas: list[list[float]],
        costs: list[AssignmentCost],
        candidate_metrics: dict[tuple[str, int, int, str], dict[str, Any]],
        max_head_relative_l2: float
) -> tuple[list[list[tuple[str, float]]], list[dict[str, Any]],
           dict[int, AssignmentCost]]:
    """Choose only among exact, whole assignments observed on both GPUs."""
    plan = [[("softmax_mpc", 1.0) for _ in range(heads)]
            for _ in range(layers)]
    decisions: list[dict[str, Any]] = []
    selected: dict[int, AssignmentCost] = {}
    for layer in range(layers):
        eligible: list[AssignmentCost] = []
        rejected: list[dict[str, Any]] = []
        for cost in (entry for entry in costs if entry.layer == layer):
            violations: list[dict[str, Any]] = []
            for head, protocol in enumerate(cost.protocols):
                disturbance = max(
                    float(candidate_metrics[("fp32", layer, head, protocol)]
                          ["relative_l2"]),
                    float(candidate_metrics[("fixed", layer, head, protocol)]
                          ["relative_l2"]),
                )
                if disturbance > max_head_relative_l2:
                    violations.append({
                        "head": head, "protocol": protocol,
                        "relative_l2": disturbance,
                    })
            if violations:
                rejected.append({
                    "plan_sha256": cost.plan_sha256,
                    "reason": "relative_l2_threshold",
                    "violations": violations,
                })
            else:
                eligible.append(cost)
        if not eligible:
            raise ValueError(
                f"no exact directly measured assignment is eligible for "
                f"layer={layer}; measured candidates cannot be synthesized")
        choice = min(
            eligible,
            key=lambda cost: (cost.latency_ms, cost.communication_bytes,
                              cost.rounds, cost.plan_sha256))
        selected[layer] = choice
        for head, protocol in enumerate(choice.protocols):
            alpha = alphas[layer][head] if protocol == "he_linear" else 1.0
            plan[layer][head] = (protocol, alpha)
            decisions.append({
                "layer": layer,
                "head": head,
                "selected_protocol": protocol,
                "alpha": alpha,
                "selected_complete_assignment":
                    assignment_cost_json(choice),
                "selection_unit": "whole_measured_assignment",
                "rejected_complete_assignments": rejected,
            })
    return plan, decisions, selected


def measured_assignment_cost_summary(
        selected: dict[int, AssignmentCost]) -> dict[str, Any]:
    ordered = [selected[layer] for layer in sorted(selected)]
    return {
        "selection_objective_latency_ms":
            sum(cost.latency_ms for cost in ordered),
        "communication_bytes":
            sum(cost.communication_bytes for cost in ordered),
        "rounds": sum(cost.rounds for cost in ordered),
        "scope": "sum_of_directly_measured_per_layer_selection_units",
        "is_end_to_end_prediction": False,
        "under_eight_minutes": None,
        "requires_full_12_layer_secure_run": True,
        "per_head_extrapolation": False,
        "non_attention_fixed_cost_extrapolation": False,
        "layers": [
            {"layer": cost.layer, "cost": assignment_cost_json(cost)}
            for cost in ordered
        ],
    }


def select_protocols(
        layers: int, heads: int, alphas: list[list[float]],
        costs: dict[tuple[int, str, int], Cost],
        candidate_metrics: dict[tuple[str, int, int, str], dict[str, Any]],
        max_head_relative_l2: float,
        max_he_depth: int) -> tuple[list[list[tuple[str, float]]],
                                    list[dict[str, Any]]]:
    """Select the least-cost measured grouping from public offline data.

    Runtime executes one contiguous group for each selected protocol.  The
    objective therefore uses the directly measured C(protocol, group size)
    rows.  It never assumes that packed MPC/CKKS work scales linearly per head.
    """
    raise RuntimeError(
        "retired additive protocol-group selector; use "
        "select_measured_assignments with complete A3/A4 rows")
    plan = [[("softmax_mpc", 1.0) for _ in range(heads)]
            for _ in range(layers)]
    decisions: list[dict[str, Any]] = []
    protocol_index = {protocol: index
                      for index, protocol in enumerate(PROTOCOLS)}
    for layer in range(layers):
        per_head_eligible: list[list[str]] = []
        per_head_rejected: list[list[dict[str, Any]]] = []
        for head in range(heads):
            eligible: list[str] = []
            rejected: list[dict[str, Any]] = []
            for protocol in PROTOCOLS:
                protocol_costs = [
                    cost for (cost_layer, cost_protocol, _), cost
                    in costs.items()
                    if cost_layer == layer and cost_protocol == protocol
                ]
                if not protocol_costs:
                    rejected.append({
                        "protocol": protocol,
                        "reason": ("a4_ckks_cost_unavailable"
                                   if protocol == "he_linear" else
                                   "measured_group_cost_unavailable"),
                    })
                    continue
                if (protocol == "he_linear" and all(
                        cost.multiplicative_depth > max_he_depth
                        for cost in protocol_costs)):
                    rejected.append({"protocol": protocol,
                                     "reason": "he_depth"})
                    continue
                fp32_relative_l2 = float(candidate_metrics[
                    ("fp32", layer, head, protocol)]["relative_l2"])
                fixed_relative_l2 = float(candidate_metrics[
                    ("fixed", layer, head, protocol)]["relative_l2"])
                disturbance = max(fp32_relative_l2, fixed_relative_l2)
                if disturbance > max_head_relative_l2:
                    rejected.append({
                        "protocol": protocol,
                        "reason": "relative_l2_threshold",
                        "relative_l2": disturbance,
                    })
                    continue
                eligible.append(protocol)
            if not eligible:
                raise ValueError(
                    f"no eligible protocol for layer={layer},head={head}; "
                    "Softmax cost/metric input is invalid")
            per_head_eligible.append(eligible)
            per_head_rejected.append(rejected)

        # State is the number of heads assigned to each protocol.  Retaining a
        # single assignment per state is sufficient because group cost depends
        # only on the public protocol counts, while all assignments reaching a
        # state have already passed the per-head disturbance guard.
        states: dict[tuple[int, ...], list[str]] = {
            tuple(0 for _ in PROTOCOLS): []
        }
        for eligible in per_head_eligible:
            next_states: dict[tuple[int, ...], list[str]] = {}
            for counts, assignment in states.items():
                for protocol in eligible:
                    updated = list(counts)
                    updated[protocol_index[protocol]] += 1
                    key = tuple(updated)
                    next_states.setdefault(key, assignment + [protocol])
            states = next_states

        measured_candidates: list[
            tuple[tuple[float, int, int, tuple[int, ...]],
                  list[str], dict[str, Cost]]
        ] = []
        for counts, assignment in states.items():
            group_costs: dict[str, Cost] = {}
            valid = True
            for protocol, count in zip(PROTOCOLS, counts):
                if count == 0:
                    continue
                cost = costs.get((layer, protocol, count))
                if cost is None or (protocol == "he_linear" and
                                    cost.multiplicative_depth > max_he_depth):
                    valid = False
                    break
                group_costs[protocol] = cost
            if not valid:
                continue
            objective = (
                sum(cost.latency_ms for cost in group_costs.values()),
                sum(cost.communication_bytes for cost in group_costs.values()),
                sum(cost.rounds for cost in group_costs.values()),
                counts,
            )
            measured_candidates.append((objective, assignment, group_costs))
        if not measured_candidates:
            available = sorted(
                (protocol, count) for cost_layer, protocol, count in costs
                if cost_layer == layer)
            raise ValueError(
                f"no complete measured grouping for layer={layer}; "
                f"available group costs={available}")
        objective, assignment, selected_group_costs = min(
            measured_candidates, key=lambda entry: entry[0])

        selected_counts = {
            protocol: assignment.count(protocol) for protocol in PROTOCOLS
        }
        for head, protocol in enumerate(assignment):
            alpha = alphas[layer][head] if protocol == "he_linear" else 1.0
            plan[layer][head] = (protocol, alpha)
            decisions.append({
                "layer": layer,
                "head": head,
                "selected_protocol": protocol,
                "alpha": alpha,
                "eligible_protocols": per_head_eligible[head],
                "rejected_protocols": per_head_rejected[head],
                "selected_group_head_count": selected_counts[protocol],
                "selected_group_cost": cost_json(
                    selected_group_costs[protocol]),
                "layer_group_objective": {
                    "latency_ms": objective[0],
                    "communication_bytes": objective[1],
                    "rounds": objective[2],
                },
            })
    return plan, decisions


def plan_cost_summary(
        plan: list[list[tuple[str, float]]],
        costs: dict[tuple[int, str, int], Cost]) -> dict[str, Any]:
    raise RuntimeError(
        "retired BLB-remainder E2E extrapolation; run the selected full "
        "12-layer secure graph")
    normalization_latency_ms = 0.0
    communication_bytes = 0
    rounds = 0
    layer_groups: list[dict[str, Any]] = []
    for layer, entries in enumerate(plan):
        groups: list[dict[str, Any]] = []
        for protocol in PROTOCOLS:
            count = sum(entry[0] == protocol for entry in entries)
            if count == 0:
                continue
            cost = costs[(layer, protocol, count)]
            normalization_latency_ms += cost.latency_ms
            communication_bytes += cost.communication_bytes
            rounds += cost.rounds
            groups.append({
                "protocol": protocol,
                "head_count": count,
                "cost": cost_json(cost),
            })
        layer_groups.append({"layer": layer, "groups": groups})
    estimated_e2e_ms = (BLB_BASELINE_NON_NORMALIZATION_MS +
                        normalization_latency_ms)
    return {
        "estimator": (
            "BLB measured 498.104 s E2E minus its measured 97.486694 s "
            "normalization plus directly measured selected group costs"),
        "baseline_e2e_ms": BLB_BASELINE_E2E_MS,
        "baseline_normalization_ms": BLB_BASELINE_NORMALIZATION_MS,
        "baseline_non_normalization_ms": BLB_BASELINE_NON_NORMALIZATION_MS,
        "selected_normalization_ms": normalization_latency_ms,
        "estimated_e2e_ms": estimated_e2e_ms,
        "estimated_e2e_minutes": estimated_e2e_ms / 60_000.0,
        "estimated_speedup_percent": (
            (BLB_BASELINE_E2E_MS - estimated_e2e_ms) /
            BLB_BASELINE_E2E_MS * 100.0),
        "target_e2e_ms": TARGET_E2E_MS,
        "under_eight_minutes": estimated_e2e_ms < TARGET_E2E_MS,
        "normalization_communication_bytes": communication_bytes,
        "normalization_rounds": rounds,
        "scope": "estimate_only_until_formal_full_12_layer_run",
        "layer_groups": layer_groups,
    }


def grouped_original_order(
        plan: list[list[tuple[str, float]]]) -> list[list[int]]:
    """Return the public physical order with one contiguous run per path."""
    protocol_rank = {name: rank for rank, name in enumerate(PROTOCOLS)}
    return [
        sorted(range(len(entries)),
               key=lambda head: protocol_rank[entries[head][0]])
        for entries in plan
    ]


def summarize_candidates(rows: list[dict[str, Any]], layers: int,
                         heads: int) -> dict[str, Any]:
    """Compact review metrics; the candidate CSV remains the source of truth."""
    thresholds = (0.25, 0.30, 0.35, 0.40, 0.50, 1.00)
    summary: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        oracle_summary: dict[str, Any] = {}
        for oracle in ("fp32", "fixed"):
            values = np.asarray([
                float(row["relative_l2"]) for row in rows
                if row["protocol"] == protocol and row["oracle"] == oracle
            ], dtype=np.float64)
            oracle_summary[oracle] = {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
        per_head_max = []
        for layer in range(layers):
            for head in range(heads):
                per_head_max.append(max(
                    float(row["relative_l2"]) for row in rows
                    if (row["protocol"] == protocol and
                        int(row["layer"]) == layer and
                        int(row["head"]) == head)))
        oracle_summary["max_across_oracles"] = {
            "min": min(per_head_max),
            "median": float(np.median(per_head_max)),
            "max": max(per_head_max),
            "eligible_head_counts_by_review_threshold": {
                f"{threshold:.2f}": sum(value <= threshold
                                        for value in per_head_max)
                for threshold in thresholds
            },
        }
        summary[protocol] = oracle_summary
    return summary


def write_assignment_json(
        path: Path, model_path: Path, samples: list[Sample],
        plan: list[list[tuple[str, float]]],
        original_order: list[list[int]],
        costs: dict[tuple[int, str, int], Cost],
        candidate_metrics: dict[tuple[str, int, int, str], dict[str, Any]],
        max_head_relative_l2: float, final_metrics: dict[str, Any],
        global_guard_passed: bool, runtime_plan_path: Path,
        cost_source_path: Path) -> None:
    raise RuntimeError(
        "retired group-cost assignment format; use "
        "write_direct_assignment_json")
    layers: list[dict[str, Any]] = []
    for layer, order in enumerate(original_order):
        selected_counts = {
            protocol: sum(entry[0] == protocol for entry in plan[layer])
            for protocol in PROTOCOLS
        }
        heads: list[dict[str, Any]] = []
        for physical_head, original_head in enumerate(order):
            protocol, alpha = plan[layer][original_head]
            fp32 = candidate_metrics[("fp32", layer, original_head, protocol)]
            fixed = candidate_metrics[("fixed", layer, original_head, protocol)]
            heads.append({
                "physical_head": physical_head,
                "original_head": original_head,
                "protocol": protocol,
                "alpha": alpha,
                "disturbance": {
                    "definition": "relative_l2(attention_head_context,softmax_reference)",
                    "fp32": fp32,
                    "fixed_clear": fixed,
                    "threshold": max_head_relative_l2,
                    "status": "PASS" if max(float(fp32["relative_l2"]),
                                               float(fixed["relative_l2"])) <=
                                           max_head_relative_l2 else "FAIL",
                },
                "measured_group_cost": cost_json(
                    costs[(layer, protocol, selected_counts[protocol])]),
            })
        layers.append({"layer": layer, "heads": heads})
    cost_summary = plan_cost_summary(plan, costs)
    has_a4_linear_cost = any(
        protocol == "he_linear" for _, protocol, _ in costs)
    payload = {
        "format": "OURS_VIT_HEAD_ASSIGNMENT_V2",
        "status": "PASS" if global_guard_passed else "FAIL_GLOBAL_GUARD",
        "model": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "layers": 12,
            "heads": 12,
            "logical_tokens": 197,
            "physical_tokens": 256,
        },
        "calibration_samples": [
            {"input": str(sample.path.resolve()), "label": sample.label}
            for sample in samples
        ],
        "selection": {
            "reference": "softmax_mpc",
            "head_disturbance_metric":
                "relative_l2(attention_head_context,softmax_reference)",
            "max_head_relative_l2": max_head_relative_l2,
            "cost_objective":
                "minimum sum of directly measured protocol-group latency; communication then rounds as ties",
            "per_head_linear_extrapolation": False,
            "a1_a2_mpc_linear_surrogate_selection_eligible": False,
            "a4_ckks_linear_cost_available": has_a4_linear_cost,
            "network_gbps": 1.0,
            "rtt_ms": 0.3,
            "cost_scope": sorted({cost.cost_scope for cost in costs.values()}),
            "conversion_scope": sorted({cost.conversion_scope
                                         for cost in costs.values()}),
            "gpu_topology": sorted({cost.gpu_topology
                                     for cost in costs.values()}),
            "private_dynamic_branch": False,
            "selected_offline_before_inference": True,
        },
        "cost_source": {
            "path": str(cost_source_path.resolve()),
            "sha256": sha256_file(cost_source_path),
        },
        "runtime_plan": str(runtime_plan_path.resolve()),
        "blb_anchored_cost_summary": cost_summary,
        "final_metrics": final_metrics,
        "layers": layers,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def write_direct_assignment_json(
        path: Path, model_path: Path, samples: list[Sample],
        plan: list[list[tuple[str, float]]],
        original_order: list[list[int]],
        selected_costs: dict[int, AssignmentCost],
        candidate_metrics: dict[tuple[str, int, int, str], dict[str, Any]],
        max_head_relative_l2: float, final_metrics: dict[str, Any],
        global_guard_passed: bool, runtime_plan_path: Path,
        cost_source_path: Path) -> None:
    layers: list[dict[str, Any]] = []
    for layer, order in enumerate(original_order):
        heads: list[dict[str, Any]] = []
        for physical_head, original_head in enumerate(order):
            protocol, alpha = plan[layer][original_head]
            fp32 = candidate_metrics[("fp32", layer, original_head, protocol)]
            fixed = candidate_metrics[("fixed", layer, original_head, protocol)]
            heads.append({
                "physical_head": physical_head,
                "original_head": original_head,
                "protocol": protocol,
                "alpha": alpha,
                "disturbance": {
                    "definition":
                        "relative_l2(attention_head_context,softmax_reference)",
                    "fp32": fp32,
                    "fixed_clear": fixed,
                    "threshold": max_head_relative_l2,
                    "status": "PASS",
                },
            })
        layers.append({
            "layer": layer,
            "measured_complete_assignment_cost":
                assignment_cost_json(selected_costs[layer]),
            "heads": heads,
        })
    payload = {
        "format": "OURS_VIT_HEAD_ASSIGNMENT_V3_DIRECT_COST",
        "status": "PASS" if global_guard_passed else "FAIL_GLOBAL_GUARD",
        "model": {
            "path": str(model_path.resolve()),
            "sha256": sha256_file(model_path),
            "layers": 12, "heads": 12,
            "logical_tokens": 197, "physical_tokens": 256,
        },
        "calibration_samples": [
            {"input": str(sample.path.resolve()), "label": sample.label}
            for sample in samples
        ],
        "selection": {
            "reference": "softmax_mpc",
            "max_head_relative_l2": max_head_relative_l2,
            "cost_objective":
                "minimum latency among exact directly measured complete attention assignments",
            "cost_unit": "complete_attention_assignment",
            "per_head_linear_extrapolation": False,
            "protocol_group_additive_extrapolation": False,
            "non_attention_fixed_cost_extrapolation": False,
            "network_gbps": 1.0, "rtt_ms": 0.3,
            "private_dynamic_branch": False,
            "selected_offline_before_inference": True,
            "eight_minute_verdict_source":
                "required_full_12_layer_secure_end_to_end_run",
        },
        "cost_source": {
            "path": str(cost_source_path.resolve()),
            "sha256": sha256_file(cost_source_path),
        },
        "runtime_plan": str(runtime_plan_path.resolve()),
        "measured_assignment_cost_summary":
            measured_assignment_cost_summary(selected_costs),
        "final_metrics": final_metrics,
        "layers": layers,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--costs", type=Path,
        help=("retired normalization-group format; retained only to emit a "
              "fail-closed migration error"))
    parser.add_argument(
        "--assignment-costs", type=Path,
        help=("direct complete-attention assignment rows collected by "
              "collect_attention_assignment_costs.py"))
    parser.add_argument("--max-he-depth", type=int, default=2)
    parser.add_argument("--max-head-relative-l2", type=float,
                        help="per-head attention-context disturbance threshold")
    parser.add_argument("--max-logit-max-abs", type=float)
    parser.add_argument("--max-logit-mse", type=float)
    parser.add_argument("--min-top1-agreement", type=float, default=1.0)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.0)
    args = parser.parse_args()

    samples = samples_from_args(args)
    model = vr.load_model(args.model)
    layers = int(model["layers"])
    heads = int(model["heads"])
    if layers != 12 or heads != 12 or int(model["tokens"]) != 197:
        raise ValueError("calibration requires the audited ViT-B/16 12x12/197 model")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    observations = {
        mode: [[[] for _ in range(heads)] for _ in range(layers)]
        for mode in ("fp32", "fixed")
    }
    baseline = {"fp_logits": [], "fixed_logits": []}
    for sample in samples:
        pixels = vr.load_input(sample.path)
        for mode, fixed in (("fp32", False), ("fixed", True)):
            graph = vr.Graph(model, fixed=fixed)
            logits, records = graph.run(pixels)
            baseline["fixed_logits" if fixed else "fp_logits"].append(logits)
            for layer in range(layers):
                derived = derive_contexts(records, layer, int(model["tokens"]),
                                          heads, fixed)
                for head in range(heads):
                    observations[mode][layer][head].append(derived[head])

    rows: list[dict[str, Any]] = []
    alphas = [[1.0 for _ in range(heads)] for _ in range(layers)]
    for layer in range(layers):
        for head in range(heads):
            fp = observations["fp32"][layer][head]
            numerator = sum(float(np.sum((item["score"] @ item["value"]) *
                                          item["target"], dtype=np.float64))
                            for item in fp)
            denominator = sum(float(np.sum((item["score"] @ item["value"]) ** 2,
                                            dtype=np.float64)) for item in fp)
            alpha = numerator / denominator if denominator > 0.0 else 0.0
            alphas[layer][head] = alpha
            for mode, fixed in (("fp32", False), ("fixed", True)):
                obs = observations[mode][layer][head]
                target = [item["target"] for item in obs]
                relu = [item["relu"] for item in obs]
                linear = [linear_context(item, alpha, fixed) for item in obs]
                for protocol, candidate in (("softmax_mpc", target),
                                            ("relu_norm_mpc", relu),
                                            ("he_linear", linear)):
                    rows.append({"layer": layer, "head": head,
                                 "protocol": protocol,
                                 "alpha": alpha if protocol == "he_linear" else 1.0,
                                 "oracle": mode, **error_metrics(target, candidate)})

    candidate_csv = args.output_dir / "head_candidate_errors.csv"
    with candidate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report: dict[str, Any] = {
        "format": "OURS_VIT_HEAD_CALIBRATION_V1",
        "training_or_backpropagation": False,
        "private_dynamic_branch": False,
        "model": str(args.model.resolve()),
        "samples": [{"input": str(row.path.resolve()), "label": row.label}
                    for row in samples],
        "disturbance": {
            "reference": "softmax_mpc",
            "tensor": "attention_head_context",
            "metric": "relative_l2",
            "formula": "l2(candidate-reference)/l2(reference)",
            "oracles": ["fp32", "fixed_clear"],
        },
        "linear_alpha_fit":
            "offline least_squares(score@V, softmax(score)@V) on fp32 calibration observations",
        "candidate_summary": summarize_candidates(rows, layers, heads),
        "candidate_errors": str(candidate_csv.resolve()),
        "selection_status": "CANDIDATES_ONLY",
    }

    if args.costs:
        raise ValueError(
            "--costs normalization-group selection is retired: it adds "
            "separate path costs and extrapolates a fixed BLB remainder. "
            "Use --assignment-costs from real A3/A4 complete-attention runs")

    if args.assignment_costs:
        if len(samples) < 4:
            raise ValueError("final assignment requires at least four public calibration samples")
        if (args.max_head_relative_l2 is None or
                not math.isfinite(args.max_head_relative_l2) or
                args.max_head_relative_l2 < 0.0):
            raise ValueError(
                "cost-based selection requires a finite non-negative "
                "--max-head-relative-l2")
        if args.max_logit_max_abs is None or args.max_logit_mse is None:
            raise ValueError(
                "cost-based selection requires both logit error thresholds")
        costs = load_assignment_costs(
            args.assignment_costs, layers, heads, args.max_he_depth)
        candidate_metrics = {
            (str(row["oracle"]), int(row["layer"]), int(row["head"]),
             str(row["protocol"])): row for row in rows
        }
        plan, decisions, selected_costs = select_measured_assignments(
            layers, heads, alphas, costs, candidate_metrics,
            args.max_head_relative_l2)
        # The secure layout is public and fixed before inference. Keep each
        # protocol in one contiguous run; original_head records the matching
        # synchronized WQ/WK/WV/WO permutation.
        original_order = grouped_original_order(plan)
        plan_path = args.output_dir / "head_plan.txt"
        write_plan(plan_path, plan, original_order)
        final_eval = evaluate_plan(model, samples, plan)
        final_metrics = classification_metrics(samples, baseline, final_eval)
        accuracy_ok = (final_metrics["accuracy_drop"] is None or
                       final_metrics["accuracy_drop"] <= args.max_accuracy_drop)
        global_guard_passed = (
            final_metrics["top1_agreement"] >= args.min_top1_agreement and
            final_metrics["fp32_logits"]["max_abs"] <=
                args.max_logit_max_abs and
            final_metrics["fp32_logits"]["mse"] <= args.max_logit_mse and
            final_metrics["fixed_logits"]["max_abs"] <=
                args.max_logit_max_abs and
            final_metrics["fixed_logits"]["mse"] <= args.max_logit_mse and
            accuracy_ok)
        assignment_path = args.output_dir / "head_assignment.json"
        write_direct_assignment_json(
            assignment_path, args.model, samples, plan, original_order,
            selected_costs,
            candidate_metrics, args.max_head_relative_l2, final_metrics,
            global_guard_passed, plan_path, args.assignment_costs)
        direct_cost_summary = measured_assignment_cost_summary(selected_costs)
        report.update({
            "selection_status": ("PASS" if global_guard_passed else
                                 "FAIL_GLOBAL_GUARD"),
            "cost_source": str(args.assignment_costs.resolve()),
            "cost_source_requirement":
                "exact complete A3/A4 attention assignments measured on separate Phantom GPUs at 1 Gbps / 0.3 ms RTT; includes selective conversion, MPC groups, AV, WO, and genuine output conversion",
            "measured_assignment_cost_summary": direct_cost_summary,
            "end_to_end_estimate_removed": True,
            "eight_minute_verdict":
                "pending_full_12_layer_secure_end_to_end_run",
            "head_plan": str(plan_path.resolve()),
            "head_assignment": str(assignment_path.resolve()),
            "decisions": decisions,
            "physical_head_order": original_order,
            "final_metrics": final_metrics,
            "global_guard_passed": global_guard_passed,
        })

    report_path = args.output_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"OURS_HEAD_CALIBRATION status={report['selection_status']}")
    print(f"OURS_HEAD_CANDIDATES={candidate_csv}")
    if "head_plan" in report:
        print(f"OURS_HEAD_PLAN={report['head_plan']}")
    if "head_assignment" in report:
        print(f"OURS_HEAD_ASSIGNMENT={report['head_assignment']}")
    print(f"OURS_HEAD_CALIBRATION_REPORT={report_path}")
    if report["selection_status"] == "FAIL_GLOBAL_GUARD":
        raise SystemExit("selected assignment failed the global calibration guard")


if __name__ == "__main__":
    main()
