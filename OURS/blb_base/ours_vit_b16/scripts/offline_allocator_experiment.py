#!/usr/bin/env python3
"""Offline validation suite for the cost-aware ViT head allocator.

The suite deliberately sits above :mod:`offline_cost_aware_head_allocator`.
It reuses its clear reference, Linear-alpha calibration, layerwise DP and
measured complete-assignment lookup, but adds experiment bookkeeping that is
needed to validate the allocator itself:

* fixed public calibration and held-out manifests;
* a small, explicitly split composition-cost regression experiment;
* Accuracy-only, Cost-only, Balanced and Performance-first ablations;
* matched random-head baselines; and
* a 4--6-head exhaustive oracle that is local to one layer only.

No secure process is started here.  The only secure artifacts consumed by
this file are already collected, directly measured benchmark CSV rows.  A
one-block composition probe is represented as a cost-model observation, not
as a twelve-layer runtime estimate.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import offline_cost_aware_head_allocator as base  # noqa: E402


POLICY_ORDER = (
    "All-Softmax", "Accuracy-only", "Cost-only", "Balanced",
    "Performance-first",
)
ABLATION_POLICY_ORDER = (
    "Accuracy-only", "Cost-only", "Balanced", "Performance-first",
)
PROTOCOLS = base.PROTOCOLS
SOFTMAX = PROTOCOLS[0]
RELU = PROTOCOLS[1]
LINEAR = PROTOCOLS[2]
SEARCH_COMPOSITIONS = tuple(
    (softmax, relu, base.MODEL_HEADS - softmax - relu)
    for softmax in range(base.MODEL_HEADS + 1)
    for relu in range(base.MODEL_HEADS - softmax + 1)
)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compact_float(value: float) -> float:
    return round(float(value), 12)


def make_fixed_manifest(source: Path, output: Path, start: int,
                        count: int) -> list[dict[str, str]]:
    """Copy a deterministic public slice into an auditable manifest."""
    raw: list[list[str]] = []
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or all(not cell.strip() for cell in row):
                continue
            if row[0].strip().lower() in {"input", "image", "image_path", "path"}:
                continue
            if len(row) != 2:
                raise ValueError(f"public manifest row must have path,label: {source}")
            raw.append([row[0].strip(), row[1].strip()])
    if start < 0 or count <= 0 or start + count > len(raw):
        raise ValueError(f"manifest slice [{start},{start + count}) exceeds {source}")
    selected = [{"path": row[0], "label": row[1]}
                for row in raw[start:start + count]]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(selected)
    return selected


def assert_disjoint_manifests(calibration: list[dict[str, str]],
                              heldout: list[dict[str, str]]) -> None:
    """Fail closed if a public image appears in both fixed data splits."""
    calibration_paths = {row["path"] for row in calibration}
    heldout_paths = {row["path"] for row in heldout}
    overlap = sorted(calibration_paths & heldout_paths)
    if overlap:
        raise ValueError(
            "fixed calibration/held-out manifests overlap: "
            f"{overlap[:5]}")
    if len(calibration_paths) != len(calibration):
        raise ValueError("fixed calibration manifest contains duplicate paths")
    if len(heldout_paths) != len(heldout):
        raise ValueError("fixed held-out manifest contains duplicate paths")


def write_sensitivity_outputs(output_dir: Path,
                              isolated: list[dict[str, Any]],
                              alphas: list[list[float]],
                              logit_weight: float) -> None:
    """Write detailed and matrix-form 12x12 per-Head risk tables.

    The heatmap value is exactly the internal allocator risk used by the
    layerwise DP: non-negative validation-loss delta plus the configured
    relative-logit-deviation tie-breaker.  Raw loss/L2/accuracy fields remain
    in the detailed table so the heatmap is auditable rather than a hidden
    normalization.
    """
    by_key = {(int(row["layer"]), int(row["head"]), str(row["protocol"])): row
              for row in isolated}
    detail: list[dict[str, Any]] = []
    for layer in range(base.MODEL_LAYERS):
        for head in range(base.MODEL_HEADS):
            relu = by_key[(layer, head, RELU)]
            linear = by_key[(layer, head, LINEAR)]
            detail.append({
                "layer": layer,
                "head": head,
                "alpha_h": float(alphas[layer][head]),
                "relu_loss_delta": float(relu["loss_delta"]),
                "relu_relative_l2": float(relu["relative_l2"]),
                "relu_risk": float(relu["proxy_score"]),
                "relu_accuracy_drop_pp": float(relu["accuracy_drop"]) * 100.0,
                "relu_mse": float(relu["mse"]),
                "relu_max_abs": float(relu["max_abs"]),
                "linear_loss_delta": float(linear["loss_delta"]),
                "linear_relative_l2": float(linear["relative_l2"]),
                "linear_risk": float(linear["proxy_score"]),
                "linear_accuracy_drop_pp": float(linear["accuracy_drop"]) * 100.0,
                "linear_mse": float(linear["mse"]),
                "linear_max_abs": float(linear["max_abs"]),
            })
    write_csv(output_dir / "head_sensitivity.csv", list(detail[0]), detail)

    matrix_fields = ["layer"] + [f"head_{head}" for head in range(base.MODEL_HEADS)]
    relu_matrix = []
    linear_matrix = []
    for path_name, risk_field, target in (
            ("relu_norm_mpc", "relu_risk", relu_matrix),
            ("he_linear", "linear_risk", linear_matrix)):
        for layer in range(base.MODEL_LAYERS):
            values = [row[risk_field] for row in detail if row["layer"] == layer]
            target.append({"layer": layer, **{
                f"head_{head}": values[head] for head in range(base.MODEL_HEADS)}})
    write_csv(output_dir / "sensitivity_relu_heatmap.csv", matrix_fields, relu_matrix)
    write_csv(output_dir / "sensitivity_linear_heatmap.csv", matrix_fields, linear_matrix)

    # A single file is convenient for plotting libraries that expect a
    # path/category column; each path block is independently 12x12.
    heatmap_rows = []
    for path_name, matrix in (("relu_norm_mpc", relu_matrix),
                              ("he_linear", linear_matrix)):
        heatmap_rows.extend({"path": path_name, **row} for row in matrix)
    write_csv(output_dir / "sensitivity_heatmap.csv",
              ["path"] + matrix_fields, heatmap_rows)
    (output_dir / "sensitivity_metadata.json").write_text(
        json.dumps({
            "format": "OURS_HEAD_SENSITIVITY_V1",
            "shape_per_path": [base.MODEL_LAYERS, base.MODEL_HEADS],
            "entries": base.MODEL_LAYERS * base.MODEL_HEADS,
            "paths": [RELU, LINEAR],
            "risk_metric": "max(0, validation_loss_delta) + "
                           f"{logit_weight:g} * relative_logit_l2",
            "calibration_only": True,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clear_baseline_batch(model: dict[str, Any], pixels: list[np.ndarray],
                         batch_size: int) -> list[np.ndarray]:
    """Evaluate the clear all-Softmax baseline in small batches.

    This is an offline-only throughput helper.  ``Graph.run_batch`` is a
    separate clear reference implementation and is never available to the
    fixed/secure graph.  Keeping the batch bounded avoids materializing the
    whole validation tensor while removing the unnecessary per-image Python
    graph overhead from the held-out baseline pass.
    """
    logits: list[np.ndarray] = []
    for start in range(0, len(pixels), batch_size):
        batch = np.stack(pixels[start:start + batch_size])
        values, _ = base.vr.Graph(model, fixed=False,
                                  record_values=False).run_batch(batch)
        logits.extend(np.asarray(values))
    return logits


@dataclass(frozen=True)
class ProbeCostPoint:
    composition: tuple[int, int, int]
    latency_ms: float
    communication_bytes: int
    source: str
    split: str
    layer: int
    plan_sha256: str
    # These are carried for audit/reporting only.  They are not converted to
    # per-Head costs and are not used as a surrogate secure-runtime result.
    overhead: dict[str, Any] = field(default_factory=dict)


OVERHEAD_FIELDS = (
    "rounds", "protocol_groups", "mpc_group_calls", "native_linear_groups",
    "ckks_to_mpc", "mpc_to_ckks", "he_instrumented_ms",
    "conversion_instrumented_ms", "he_ct_pt_multiply", "he_ct_ct_multiply",
    "he_rotations", "he_rescales",
)
OVERHEAD_INTEGER_FIELDS = {
    field_name for field_name in OVERHEAD_FIELDS
    if field_name not in {"he_instrumented_ms", "conversion_instrumented_ms"}
}


def parse_overhead(row: dict[str, str]) -> dict[str, Any]:
    """Parse benchmark overhead counters without making them cost estimates."""
    result: dict[str, Any] = {}
    for field_name in OVERHEAD_FIELDS:
        value = row.get(field_name, "").strip()
        if value == "":
            result[field_name] = ""
            continue
        try:
            parsed: Any = (int(value) if field_name in OVERHEAD_INTEGER_FIELDS
                           else float(value))
        except ValueError as error:
            raise ValueError(
                f"invalid benchmark overhead {field_name}={value!r}") from error
        if not math.isfinite(float(parsed)) or parsed < 0:
            raise ValueError(
                f"benchmark overhead must be finite and non-negative: {field_name}")
        result[field_name] = parsed
    return result


def require_unpacked_probe_provenance(rows: list[dict[str, str]], path: Path) -> None:
    """Reject packed or legacy cost rows from this head-only experiment.

    The older complete-assignment benchmark predates the explicit grouping
    provenance columns.  It remains a real benchmark artifact, but it is not
    interchangeable with the current no-Head-packing cohort.  Failing here
    prevents an accidental mixture of packed and unpacked costs.
    """
    required = {"attention_packing", "heterogeneous_head_grouping"}
    fieldnames = set(rows[0]) if rows else set()
    missing = sorted(required - fieldnames)
    if missing:
        raise ValueError(
            f"cost CSV lacks current no-group provenance {missing}: {path}")
    invalid = [
        row.get("plan_sha256", "") for row in rows
        if row.get("attention_packing") != "unpacked" or
        row.get("heterogeneous_head_grouping") != "0"
    ]
    if invalid:
        raise ValueError(
            "cost CSV contains packed/grouped rows; strict head-only validation "
            f"requires attention_packing=unpacked and heterogeneous_head_grouping=0: {path}")


def overhead_for_points(points: list[ProbeCostPoint]) -> dict[str, Any]:
    """Median measured counters for a composition's direct benchmark rows."""
    result: dict[str, Any] = {}
    for field_name in OVERHEAD_FIELDS:
        values = [point.overhead[field_name] for point in points
                  if point.overhead.get(field_name, "") != ""]
        if not values:
            result[field_name] = ""
        elif field_name in OVERHEAD_INTEGER_FIELDS:
            result[field_name] = int(statistics.median(values))
        else:
            result[field_name] = float(statistics.median(values))
    return result


def parse_composition(value: str) -> tuple[int, int, int]:
    text = value.strip()
    for token in ("S", "R", "L"):
        text = text.replace(token, ",")
    parts = [part for part in text.split(",") if part != ""]
    if len(parts) != 3:
        raise ValueError(f"invalid composition: {value}")
    counts = tuple(int(part) for part in parts)
    if any(count < 0 for count in counts) or sum(counts) != base.MODEL_HEADS:
        raise ValueError(f"composition must be non-negative and sum to 12: {value}")
    return counts  # type: ignore[return-value]


def load_probe_points(source_costs: Path, supplemental_costs: Path,
                      probe_manifest: Path, allow_missing: bool = False
                      ) -> tuple[list[ProbeCostPoint], list[ProbeCostPoint],
                                 list[ProbeCostPoint], set[tuple[int, int, int]],
                                 set[tuple[int, int, int]]]:
    """Load direct observations and split new probes without leakage.

    Existing complete-assignment rows are retained for every layer so the
    composition model can learn layer baselines.  The new probe suite is a
    one-block measurement and therefore contributes only its layer-0 row.
    Both files are passed through the strict benchmark loader; arbitrary CSV
    numbers cannot silently become a cost observation.
    """
    probe_rows = read_csv(probe_manifest)
    if not probe_rows:
        raise ValueError("cost/probe inputs must not be empty")
    split_by_plan: dict[str, tuple[tuple[int, int, int], str]] = {}
    for row in probe_rows:
        plan = Path(row["plan"])
        if not plan.is_file():
            raise FileNotFoundError(plan)
        plan_hash = base.sha256_file(plan)
        composition = parse_composition(row["composition"])
        split = row["split"].strip()
        if split not in {"train", "heldout"}:
            raise ValueError(f"probe split must be train or heldout: {row}")
        old = split_by_plan.get(plan_hash)
        if old is not None and old != (composition, split):
            raise ValueError(f"probe plan has inconsistent split/composition: {plan}")
        split_by_plan[plan_hash] = (composition, split)

    source_model = base.MeasuredCompositionCostModel(
        source_costs, base.MODEL_LAYERS, base.MODEL_HEADS)
    with source_costs.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    require_unpacked_probe_provenance(source_rows, source_costs)
    source_overhead = {
        (int(row["layer"]), row["plan_sha256"]): parse_overhead(row)
        for row in source_rows
    }
    existing = [
        ProbeCostPoint(
            composition=counts,
            latency_ms=observation.latency_ms,
            communication_bytes=observation.communication_bytes,
            source=observation.source_run,
            split="train_existing",
            layer=layer,
            plan_sha256=observation.plan_sha256,
            overhead=source_overhead.get((layer, observation.plan_sha256), {}),
        )
        for (layer, counts), observations in source_model.observations.items()
        for observation in observations
    ]
    if not existing:
        raise ValueError("existing cost CSV has no complete assignment rows")

    train: list[ProbeCostPoint] = []
    heldout: list[ProbeCostPoint] = []
    existing_keys = {(point.layer, point.plan_sha256, point.source)
                     for point in existing}
    if supplemental_costs.is_file():
        supplemental_model = base.MeasuredCompositionCostModel(
            supplemental_costs, base.MODEL_LAYERS, base.MODEL_HEADS)
        with supplemental_costs.open(newline="", encoding="utf-8") as handle:
            supplemental_rows = list(csv.DictReader(handle))
        require_unpacked_probe_provenance(supplemental_rows, supplemental_costs)
        supplemental_overhead = {
            (int(row["layer"]), row["plan_sha256"]): parse_overhead(row)
            for row in supplemental_rows
        }
        source_binaries = {observation.binary_sha256
                           for observations in source_model.observations.values()
                           for observation in observations}
        supplemental_binaries = {observation.binary_sha256
                                 for observations in supplemental_model.observations.values()
                                 for observation in observations}
        if source_binaries != supplemental_binaries:
            raise ValueError(
                "supplemental probes use a different secure binary; refusing to mix "
                f"source={sorted(source_binaries)} supplemental={sorted(supplemental_binaries)}")
        seen_supplemental: set[str] = set()
        for (layer, counts), observations in supplemental_model.observations.items():
            if layer != 0:
                raise ValueError(
                    "composition probes must be one-block layer-0 observations")
            for observation in observations:
                plan_hash = observation.plan_sha256
                if plan_hash in seen_supplemental:
                    raise ValueError(f"duplicate supplemental plan row: {plan_hash}")
                seen_supplemental.add(plan_hash)
                expected = split_by_plan.get(plan_hash)
                if expected is None:
                    raise ValueError(
                        f"supplemental row is absent from probe manifest: {plan_hash}")
                if counts != expected[0]:
                    raise ValueError(
                        f"probe composition disagrees with measured row: {plan_hash}")
                # A shared input file may serve as the initial training source
                # and the probe table. Count each real run only once.
                if (layer, plan_hash, observation.source_run) in existing_keys:
                    continue
                point = ProbeCostPoint(
                    composition=counts,
                    latency_ms=observation.latency_ms,
                    communication_bytes=observation.communication_bytes,
                    source=observation.source_run,
                    split=expected[1],
                    layer=layer,
                    plan_sha256=plan_hash,
                    overhead=supplemental_overhead.get((layer, plan_hash), {}),
                )
                (train if expected[1] == "train" else heldout).append(point)

    expected_train = {
        parse_composition(row["composition"])
        for row in probe_rows if row["split"] == "train"
    }
    expected_heldout = {
        parse_composition(row["composition"])
        for row in probe_rows if row["split"] == "heldout"
    }
    if expected_train & expected_heldout:
        raise ValueError("probe manifest reuses a composition across train/held-out")
    existing_compositions = {point.composition for point in existing}
    if existing_compositions & expected_heldout:
        raise ValueError(
            "strict held-out composition is already present in the existing benchmark: "
            f"{[base.composition_name(value) for value in sorted(existing_compositions & expected_heldout)]}")
    train_compositions = {point.composition for point in train}
    # Existing complete-assignment observations are also training data for
    # the composition model.  Count only the ones that correspond to the
    # manifest's intended training split; held-out compositions were rejected
    # just above and must never enter this set.
    train_compositions |= existing_compositions & expected_train
    heldout_compositions = {point.composition for point in heldout}
    missing_train = expected_train - train_compositions
    missing_heldout = expected_heldout - heldout_compositions
    if train_compositions & heldout_compositions:
        raise ValueError("held-out composition overlaps a cost-model training composition")
    extra_supplemental = set(point.composition for point in train + heldout) - (
        expected_train | expected_heldout)
    if extra_supplemental:
        raise ValueError("supplemental cost contains a composition absent from probe manifest")
    if (missing_train or missing_heldout) and not allow_missing:
        raise ValueError(
            "real composition probes are incomplete; missing train="
            f"{[base.composition_name(value) for value in sorted(missing_train)]} "
            "heldout="
            f"{[base.composition_name(value) for value in sorted(missing_heldout)]}")
    return existing, train, heldout, missing_train, missing_heldout


def cost_features(counts: tuple[int, int, int]) -> np.ndarray:
    """Feature map for Cost(S,R,L): group sizes plus coexistence terms.

    The three path counts are retained as normalized group sizes.  Indicators
    and pair products model the measured group/coexistence overhead; this is a
    composition model, not a sum of independent Head timings.
    """
    s, r, l = (float(value) / base.MODEL_HEADS for value in counts)
    has_s, has_r, has_l = (float(value > 0) for value in counts)
    return np.asarray([
        1.0, s, r, l,
        has_s + has_r + has_l,
        has_s * has_r, has_s * has_l, has_r * has_l,
        has_s * has_r * has_l,
        s * r, s * l, r * l,
    ], dtype=np.float64)


def layer_cost_features(layer: int, counts: tuple[int, int, int]) -> np.ndarray:
    """Composition features plus a categorical layer baseline.

    The benchmark reports one complete attention-assignment unit per layer.
    Layer indicators absorb the layer-specific baseline without copying a
    measured cost to another layer.  Composition effects remain shared, and
    the group/coexistence features above keep the model non-additive in S/R/L.
    """
    if layer < 0 or layer >= base.MODEL_LAYERS:
        raise ValueError(f"layer outside ViT-B/16: {layer}")
    return np.concatenate([
        cost_features(counts),
        np.asarray([float(layer == index)
                    for index in range(1, base.MODEL_LAYERS)],
                   dtype=np.float64),
    ])


class LayerCompositionCostRegressor:
    """Ridge fit over measured complete-assignment observations only.

    Fitting happens after the train/held-out split.  The intercept is not
    regularized; non-intercept features are standardized from train points.
    Duplicate observations at one ``(layer, composition)`` are reduced by
    their median.  No missing cost is converted into a per-Head estimate.
    Predictions are clamped at zero solely to preserve physical cost units.
    """

    COMPOSITION_FEATURE_NAMES = (
        "intercept", "S_group_size_norm", "R_group_size_norm",
        "L_group_size_norm", "active_group_count", "S_and_R",
        "S_and_L", "R_and_L", "S_and_R_and_L", "S_times_R",
        "S_times_L", "R_times_L",
    )
    FEATURE_NAMES = COMPOSITION_FEATURE_NAMES + tuple(
        f"layer_{layer}_indicator" for layer in range(1, base.MODEL_LAYERS))

    def __init__(self, ridge: float = 1e-3) -> None:
        if not math.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        self.ridge = ridge
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None
        self.training_compositions: tuple[tuple[int, int, int], ...] = ()
        self.training_observation_keys: tuple[
            tuple[int, tuple[int, int, int]], ...] = ()

    def fit(self, points: list[ProbeCostPoint], target: str) -> None:
        if target not in {"latency_ms", "communication_bytes"}:
            raise ValueError(f"unsupported target: {target}")
        grouped: dict[tuple[int, tuple[int, int, int]], list[ProbeCostPoint]] = {}
        for point in points:
            grouped.setdefault((point.layer, point.composition), []).append(point)
        if len(grouped) < 4:
            raise ValueError("composition regressor needs at least four train observations")
        if len({counts for _, counts in grouped}) < 4:
            raise ValueError("composition regressor needs at least four train compositions")
        keys = sorted(grouped)
        raw = np.vstack([layer_cost_features(layer, counts)
                         for layer, counts in keys])
        self.mean = np.zeros(raw.shape[1], dtype=np.float64)
        self.scale = np.ones(raw.shape[1], dtype=np.float64)
        if raw.shape[1] > 1:
            self.mean[1:] = np.mean(raw[:, 1:], axis=0)
            self.scale[1:] = np.std(raw[:, 1:], axis=0)
            self.scale[1:] = np.where(self.scale[1:] > 1e-12,
                                      self.scale[1:], 1.0)
        design = (raw - self.mean) / self.scale
        values = np.asarray([
            statistics.median(
                getattr(point, target) for point in grouped[key])
            for key in keys
        ], dtype=np.float64)
        regularizer = np.eye(design.shape[1], dtype=np.float64) * self.ridge
        regularizer[0, 0] = 0.0
        self.coefficients = np.linalg.solve(
            design.T @ design + regularizer, design.T @ values)
        self.training_compositions = tuple(sorted({counts for _, counts in keys}))
        self.training_observation_keys = tuple(keys)

    def predict(self, layer: int, counts: tuple[int, int, int]) -> float:
        if self.mean is None or self.scale is None or self.coefficients is None:
            raise RuntimeError("regressor is not fitted")
        value = ((layer_cost_features(layer, counts) - self.mean) /
                 self.scale) @ self.coefficients
        return max(0.0, float(value))

    def metadata(self) -> dict[str, Any]:
        if self.coefficients is None or self.mean is None or self.scale is None:
            raise RuntimeError("regressor is not fitted")
        return {
            "ridge": self.ridge,
            "feature_names": list(self.FEATURE_NAMES),
            "coefficients_standardized": [float(value) for value in self.coefficients],
            "feature_mean": [float(value) for value in self.mean],
            "feature_scale": [float(value) for value in self.scale],
            "training_compositions": [base.composition_name(counts)
                                       for counts in self.training_compositions],
            "training_observation_keys": [
                {"layer": layer, "composition": base.composition_name(counts)}
                for layer, counts in self.training_observation_keys],
        }


# Keep the old name importable for small external notebooks; new code uses the
# layer-aware implementation explicitly.
CompositionCostRegressor = LayerCompositionCostRegressor


def fit_cost_models(points: list[ProbeCostPoint],
                    ridge: float
                    ) -> tuple[LayerCompositionCostRegressor,
                               LayerCompositionCostRegressor]:
    latency_model = LayerCompositionCostRegressor(ridge)
    communication_model = LayerCompositionCostRegressor(ridge)
    latency_model.fit(points, "latency_ms")
    communication_model.fit(points, "communication_bytes")
    return latency_model, communication_model


class SearchCompositionCostModel:
    """Hybrid real-anchor/model-prediction cost view used only offline.

    Existing complete benchmark rows and measured training probes are returned
    as direct observations.  All other compositions are explicitly marked as
    composition-model predictions.  Held-out observations are never supplied
    to this object, so selecting a policy cannot leak the held-out split.
    """

    def __init__(self, measured: base.MeasuredCompositionCostModel,
                 training_points: list[ProbeCostPoint],
                 latency_model: LayerCompositionCostRegressor,
                 communication_model: LayerCompositionCostRegressor) -> None:
        self.measured = measured
        self.latency_model = latency_model
        self.communication_model = communication_model
        self.training_observations: dict[
            tuple[int, tuple[int, int, int]], list[ProbeCostPoint]] = {}
        for point in training_points:
            self.training_observations.setdefault(
                (point.layer, point.composition), []).append(point)

    @property
    def available_compositions(self) -> set[tuple[int, int, int]]:
        return (self.measured.available_compositions |
                {point.composition for points in self.training_observations.values()
                 for point in points})

    def missing_compositions(self) -> list[tuple[int, int, int]]:
        all_compositions = {
            (softmax, relu, base.MODEL_HEADS - softmax - relu)
            for softmax in range(base.MODEL_HEADS + 1)
            for relu in range(base.MODEL_HEADS - softmax + 1)
        }
        return sorted(all_compositions - self.available_compositions)

    def predict_layer(self, layer: int,
                      counts: tuple[int, int, int]) -> base.CostPrediction:
        direct = self.measured.predict_layer(layer, counts)
        if direct is not None:
            return direct
        points = self.training_observations.get((layer, counts), [])
        if points:
            return base.CostPrediction(
                layer=layer, counts=counts,
                latency_ms=statistics.median(point.latency_ms for point in points),
                communication_bytes=int(statistics.median(
                    point.communication_bytes for point in points)),
                observation_count=len(points),
                source_runs=tuple(sorted({point.source for point in points})),
            )
        return base.CostPrediction(
            layer=layer, counts=counts,
            latency_ms=self.latency_model.predict(layer, counts),
            communication_bytes=int(round(
                self.communication_model.predict(layer, counts))),
            observation_count=0,
            source_runs=("composition_model_ridge",),
        )

    def predict_plan(self, plan: tuple[tuple[tuple[str, float], ...], ...]
                     ) -> dict[str, Any]:
        layer_costs = [self.predict_layer(layer, counts)
                       for layer, counts in enumerate(base.counts_for_plan(plan))]
        return {
            "latency_ms": sum(row.latency_ms for row in layer_costs),
            "communication_bytes": sum(row.communication_bytes
                                        for row in layer_costs),
            "cost_unit": "sum_of_12_layer_composition_cost_predictions",
            "not_end_to_end_secure_runtime": True,
            "not_per_head_extrapolation": True,
            "search_only_cost_model": True,
            "layers": [base.cost_prediction_json(row) for row in layer_costs],
        }


def rank_accuracy(predicted: list[float], measured: list[float]) -> float:
    return ranking_stats(predicted, measured)[0]


def ranking_stats(predicted: list[float], measured: list[float]
                  ) -> tuple[float, int, int]:
    """Return strict pairwise ranking accuracy and integer correct-pair count.

    A predicted tie is not counted as correct against a strict measured order.
    This is deliberately stricter than awarding half credit for a tie, because
    the cost gate is based on the six ordered pairs of four held-out
    compositions.
    """
    if len(predicted) != len(measured) or len(predicted) < 2:
        return 1.0, 0, 0
    correct = 0
    pairs = 0
    for left, right in itertools.combinations(range(len(predicted)), 2):
        pred_delta = predicted[left] - predicted[right]
        measured_delta = measured[left] - measured[right]
        pred_sign = 0 if abs(pred_delta) <= 1e-12 else (1 if pred_delta > 0 else -1)
        measured_sign = (0 if abs(measured_delta) <= 1e-12
                         else (1 if measured_delta > 0 else -1))
        if pred_sign == measured_sign:
            correct += 1
        pairs += 1
    return (correct / pairs if pairs else 1.0), correct, pairs


def cost_model_holdout_rows(existing: list[ProbeCostPoint],
                            train_probe: list[ProbeCostPoint],
                            heldout: list[ProbeCostPoint],
                            expected_heldout: set[tuple[int, int, int]],
                            missing_train: set[tuple[int, int, int]],
                            ridge: float
                            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_points = existing + train_probe
    grouped: dict[tuple[int, int, int], list[ProbeCostPoint]] = {}
    for point in heldout:
        grouped.setdefault(point.composition, []).append(point)
    compositions = sorted(expected_heldout)

    # A strict measurement run may be stopped before enough real composition
    # observations exist to fit even the search-only model.  Preserve that
    # fact in the requested holdout table instead of turning a missing fit
    # into fabricated predictions.  The normal path below is unchanged when
    # there are enough real training observations.
    try:
        latency_model, communication_model = fit_cost_models(train_points, ridge)
    except ValueError as error:
        training_names = ";".join(
            base.composition_name(counts)
            for counts in sorted({point.composition for point in train_points}))
        missing_heldout = set(compositions) - set(grouped)
        detail: list[dict[str, Any]] = []
        for counts in compositions:
            points = grouped.get(counts, [])
            measured_overhead = overhead_for_points(points) if points else {
                field_name: "" for field_name in OVERHEAD_FIELDS
            }
            if points:
                status = "MEASURED_NO_PREDICTION_INSUFFICIENT_REAL_TRAIN"
                measured_latency = statistics.median(
                    point.latency_ms for point in points)
                measured_communication = int(statistics.median(
                    point.communication_bytes for point in points))
                plan_sha256 = ";".join(sorted(
                    point.plan_sha256 for point in points))
            else:
                status = "NOT_MEASURED_NO_REAL_HELDOUT"
                measured_latency = ""
                measured_communication = ""
                plan_sha256 = ""
            detail.append({
                "row_type": "missing_measurement" if not points else "composition",
                "status": status,
                "split": "strict_heldout",
                "composition": base.composition_name(counts),
                "S_heads": counts[0], "R_heads": counts[1], "L_heads": counts[2],
                "train_observation_count": len(train_points),
                "heldout_observation_count": len(points),
                "measured_latency_ms": measured_latency,
                "predicted_latency_ms": "",
                "latency_ape_percent": "",
                "measured_communication_bytes": measured_communication,
                "predicted_communication_bytes": "",
                "communication_ape_percent": "",
                **{f"measured_{field_name}": measured_overhead[field_name]
                   for field_name in OVERHEAD_FIELDS},
                "heldout_plan_sha256": plan_sha256,
                "uses_heldout_in_fit": False,
                "cost_model_fit_status": "NOT_FIT_INSUFFICIENT_REAL_TRAIN",
                "cost_model_fit_error": str(error),
            })
        summary = {
            "row_type": "summary",
            "status": "INCOMPLETE_INSUFFICIENT_REAL_TRAIN",
            "split": "strict_heldout",
            "composition": "SUMMARY",
            "S_heads": "", "R_heads": "", "L_heads": "",
            "train_observation_count": len(train_points),
            "heldout_observation_count": len(heldout),
            "measured_latency_ms": "", "predicted_latency_ms": "",
            "latency_ape_percent": "", "latency_mape_percent": "NA",
            "measured_communication_bytes": "",
            "predicted_communication_bytes": "",
            "communication_ape_percent": "", "communication_mape_percent": "NA",
            "latency_ranking_accuracy": "NA",
            "communication_ranking_accuracy": "NA",
            "latency_ranking_correct_pairs": 0,
            "communication_ranking_correct_pairs": 0,
            "ranking_pairs": 0, "ranking_total_pairs": 0,
            "cost_model_gate_pass": False,
            "cost_model_gate_reason": "insufficient_real_training_observations",
            "cost_model_gate_latency_mape_limit_percent": 15.0,
            "cost_model_gate_min_latency_correct_pairs": 5,
            **{f"measured_{field_name}": "" for field_name in OVERHEAD_FIELDS},
            "heldout_plan_sha256": "",
            "training_compositions": training_names,
            "heldout_compositions": ";".join(
                base.composition_name(counts) for counts in compositions),
            "measured_heldout_compositions": ";".join(
                base.composition_name(counts)
                for counts in sorted(set(grouped))),
            "missing_train_compositions": ";".join(
                base.composition_name(counts) for counts in sorted(missing_train)),
            "missing_heldout_compositions": ";".join(
                base.composition_name(counts) for counts in sorted(missing_heldout)),
            "uses_heldout_in_fit": False,
            "latency_model": "", "communication_model": "",
            "cost_model_fit_status": "NOT_FIT_INSUFFICIENT_REAL_TRAIN",
            "cost_model_fit_error": str(error),
        }
        for row in detail:
            row.update({
                "latency_mape_percent": "NA",
                "communication_mape_percent": "NA",
                "latency_ranking_accuracy": "NA",
                "communication_ranking_accuracy": "NA",
                "latency_ranking_correct_pairs": 0,
                "communication_ranking_correct_pairs": 0,
                "ranking_pairs": 0,
                "ranking_total_pairs": 0,
                "cost_model_gate_pass": False,
                "cost_model_gate_reason": "insufficient_real_training_observations",
            })
        return detail + [summary], {
            "training_observations": len(train_points),
            "training_compositions": sorted({
                base.composition_name(point.composition)
                for point in train_points}),
            "heldout_compositions": [
                base.composition_name(counts) for counts in compositions],
            "measured_heldout_compositions": [
                base.composition_name(counts) for counts in sorted(set(grouped))],
            "missing_train_compositions": [
                base.composition_name(counts) for counts in sorted(missing_train)],
            "missing_heldout_compositions": [
                base.composition_name(counts) for counts in sorted(missing_heldout)],
            "status": "INCOMPLETE_INSUFFICIENT_REAL_TRAIN",
            "latency_mape_percent": "NA",
            "communication_mape_percent": "NA",
            "latency_ranking_accuracy": "NA",
            "communication_ranking_accuracy": "NA",
            "latency_ranking_correct_pairs": 0,
            "communication_ranking_correct_pairs": 0,
            "ranking_pairs": 0,
            "ranking_total_pairs": 0,
            "cost_model_gate_pass": False,
            "cost_model_gate_reason": "insufficient_real_training_observations",
            "cost_model_gate_latency_mape_limit_percent": 15.0,
            "cost_model_gate_min_latency_correct_pairs": 5,
            "uses_heldout_in_fit": False,
            "cost_model_fit_status": "NOT_FIT_INSUFFICIENT_REAL_TRAIN",
            "cost_model_fit_error": str(error),
        }

    detail: list[dict[str, Any]] = []
    measured_latency: list[float] = []
    predicted_latency: list[float] = []
    measured_comm: list[float] = []
    predicted_comm: list[float] = []
    for counts in compositions:
        points = grouped.get(counts, [])
        if not points:
            detail.append({
                "row_type": "missing_measurement",
                "status": "NOT_MEASURED_NO_REAL_HELDOUT",
                "split": "strict_heldout",
                "composition": base.composition_name(counts),
                "S_heads": counts[0], "R_heads": counts[1], "L_heads": counts[2],
                "train_observation_count": len(train_points),
                "heldout_observation_count": 0,
                "measured_latency_ms": "", "predicted_latency_ms": latency_model.predict(0, counts),
                "latency_ape_percent": "",
                "measured_communication_bytes": "",
                "predicted_communication_bytes": int(round(communication_model.predict(0, counts))),
                "communication_ape_percent": "",
                **{f"measured_{field_name}": ""
                   for field_name in OVERHEAD_FIELDS},
                "heldout_plan_sha256": "",
                "uses_heldout_in_fit": False,
            })
            continue
        measured_l = statistics.median(point.latency_ms for point in points)
        measured_c = statistics.median(point.communication_bytes for point in points)
        predicted_l = latency_model.predict(0, counts)
        predicted_c = communication_model.predict(0, counts)
        measured_latency.append(measured_l)
        predicted_latency.append(predicted_l)
        measured_comm.append(measured_c)
        predicted_comm.append(predicted_c)
        measured_overhead = overhead_for_points(points)
        detail.append({
            "row_type": "composition",
            "status": "MEASURED",
            "split": "strict_heldout",
            "composition": base.composition_name(counts),
            "S_heads": counts[0], "R_heads": counts[1], "L_heads": counts[2],
            "train_observation_count": len(train_points),
            "heldout_observation_count": len(points),
            "measured_latency_ms": measured_l,
            "predicted_latency_ms": predicted_l,
            "latency_ape_percent": (abs(predicted_l - measured_l) / measured_l * 100.0
                                     if measured_l else 0.0),
            "measured_communication_bytes": int(measured_c),
            "predicted_communication_bytes": int(round(predicted_c)),
            "communication_ape_percent": (abs(predicted_c - measured_c) / measured_c * 100.0
                                           if measured_c else 0.0),
            **{f"measured_{field_name}": measured_overhead[field_name]
               for field_name in OVERHEAD_FIELDS},
            "heldout_plan_sha256": ";".join(sorted(point.plan_sha256 for point in points)),
            "uses_heldout_in_fit": False,
        })
    measured_detail = [row for row in detail if row["status"] == "MEASURED"]
    latency_mape = (sum(row["latency_ape_percent"] for row in measured_detail) /
                    len(measured_detail) if measured_detail else "NA")
    comm_mape = (sum(row["communication_ape_percent"] for row in measured_detail) /
                 len(measured_detail) if measured_detail else "NA")
    latency_rank, latency_correct, latency_pairs = ranking_stats(
        predicted_latency, measured_latency)
    comm_rank, comm_correct, comm_pairs = ranking_stats(
        predicted_comm, measured_comm)
    if len(measured_latency) < 2:
        latency_rank = "NA"
    if len(measured_comm) < 2:
        comm_rank = "NA"
    measured_count = len(measured_detail)
    expected_count = len(compositions)
    latency_mape_pass = (isinstance(latency_mape, (int, float)) and
                         float(latency_mape) <= 15.0)
    ranking_pass = latency_correct >= 5 and latency_pairs >= 6
    complete_holdout = (expected_count == 4 and measured_count == expected_count)
    cost_model_gate_pass = (not missing_train and complete_holdout and
                            latency_mape_pass and ranking_pass)
    if not complete_holdout:
        cost_gate_reason = "requires_four_measured_heldout_compositions"
    elif missing_train:
        cost_gate_reason = "missing_measured_training_composition"
    elif not latency_mape_pass:
        cost_gate_reason = "latency_mape_above_15_percent"
    elif not ranking_pass:
        cost_gate_reason = "fewer_than_five_of_six_latency_rankings_correct"
    else:
        cost_gate_reason = "pass"
    summary = {
        "row_type": "summary",
        "status": ("PASS" if cost_model_gate_pass else
                    "FAIL_COST_MODEL_GATE" if complete_holdout else
                    "INCOMPLETE_MISSING_REAL_MEASUREMENT"),
        "split": "strict_heldout",
        "composition": "SUMMARY",
        "S_heads": "", "R_heads": "", "L_heads": "",
        "train_observation_count": len(train_points),
        "heldout_observation_count": len(heldout),
        "measured_latency_ms": "", "predicted_latency_ms": "",
        "latency_ape_percent": "",
        "measured_communication_bytes": "",
        "predicted_communication_bytes": "",
        "communication_ape_percent": "",
        "latency_mape_percent": latency_mape,
        "communication_mape_percent": comm_mape,
        "latency_ranking_accuracy": latency_rank,
        "communication_ranking_accuracy": comm_rank,
        "latency_ranking_correct_pairs": latency_correct,
        "communication_ranking_correct_pairs": comm_correct,
        "ranking_pairs": latency_pairs,
        "ranking_total_pairs": latency_pairs,
        "cost_model_gate_pass": cost_model_gate_pass,
        "cost_model_gate_reason": cost_gate_reason,
        "cost_model_gate_latency_mape_limit_percent": 15.0,
        "cost_model_gate_min_latency_correct_pairs": 5,
        "training_compositions": ";".join(
            base.composition_name(counts) for counts in latency_model.training_compositions),
        "heldout_compositions": ";".join(base.composition_name(counts)
                                           for counts in compositions),
        "measured_heldout_compositions": ";".join(
            str(row["composition"]) for row in measured_detail),
        "missing_train_compositions": ";".join(
            base.composition_name(counts) for counts in sorted(missing_train)),
        "missing_heldout_compositions": ";".join(
            base.composition_name(counts)
            for counts in sorted(set(compositions) - set(grouped))),
        "uses_heldout_in_fit": False,
        "latency_model": json.dumps(latency_model.metadata(), sort_keys=True),
        "communication_model": json.dumps(communication_model.metadata(), sort_keys=True),
    }
    for row in detail:
        row.update({
            "latency_mape_percent": latency_mape,
            "communication_mape_percent": comm_mape,
            "latency_ranking_accuracy": latency_rank,
            "communication_ranking_accuracy": comm_rank,
            "latency_ranking_correct_pairs": latency_correct,
            "communication_ranking_correct_pairs": comm_correct,
            "ranking_pairs": latency_pairs,
            "ranking_total_pairs": latency_pairs,
            "cost_model_gate_pass": cost_model_gate_pass,
            "cost_model_gate_reason": cost_gate_reason,
        })
    return detail + [summary], {
        "training_observations": len(train_points),
        "training_compositions": [base.composition_name(counts)
                                   for counts in latency_model.training_compositions],
        "heldout_compositions": [base.composition_name(counts) for counts in compositions],
        "measured_heldout_compositions": [row["composition"]
                                           for row in measured_detail],
        "missing_train_compositions": [base.composition_name(counts)
                                        for counts in sorted(missing_train)],
        "missing_heldout_compositions": [
            base.composition_name(counts)
            for counts in sorted(set(compositions) - set(grouped))],
        "status": ("PASS" if cost_model_gate_pass else
                    "FAIL_COST_MODEL_GATE" if complete_holdout else
                    "INCOMPLETE_MISSING_REAL_MEASUREMENT"),
        "latency_mape_percent": latency_mape,
        "communication_mape_percent": comm_mape,
        "latency_ranking_accuracy": latency_rank,
        "communication_ranking_accuracy": comm_rank,
        "latency_ranking_correct_pairs": latency_correct,
        "communication_ranking_correct_pairs": comm_correct,
        "ranking_pairs": latency_pairs,
        "ranking_total_pairs": latency_pairs,
        "cost_model_gate_pass": cost_model_gate_pass,
        "cost_model_gate_reason": cost_gate_reason,
        "cost_model_gate_latency_mape_limit_percent": 15.0,
        "cost_model_gate_min_latency_correct_pairs": 5,
        "uses_heldout_in_fit": False,
        "latency_model": latency_model.metadata(),
        "communication_model": communication_model.metadata(),
    }


def state_from_plan(plan: tuple[tuple[tuple[str, float], ...], ...],
                    isolated: dict[tuple[int, int, str], dict[str, Any]],
                    cost_model: Any
                    ) -> base.ModelState:
    proxy = 0.0
    loss = 0.0
    relative = 0.0
    for layer, entries in enumerate(plan):
        for head, (protocol, _) in enumerate(entries):
            if protocol == SOFTMAX:
                continue
            row = isolated[(layer, head, protocol)]
            proxy += float(row["proxy_score"])
            loss += max(0.0, float(row["loss_delta"]))
            relative += float(row["relative_l2"])
    cost = cost_model.predict_plan(plan)
    if cost is None:
        raise ValueError("plan has no exact measured complete-assignment cost")
    return base.ModelState(
        proxy_score=proxy,
        proxy_loss_delta=loss,
        proxy_relative_l2=relative,
        latency_ms=float(cost["latency_ms"]),
        communication_bytes=int(cost["communication_bytes"]),
        plan=plan,
    )


def randomize_plan(plan: tuple[tuple[tuple[str, float], ...], ...],
                   alphas: list[list[float]], seed: int
                   ) -> tuple[tuple[tuple[str, float], ...], ...]:
    result: list[tuple[tuple[str, float], ...]] = []
    for layer, entries in enumerate(plan):
        counts = base.counts_for_plan((entries,))[0]
        protocols = ([SOFTMAX] * counts[0] + [RELU] * counts[1] + [LINEAR] * counts[2])
        order = list(range(base.MODEL_HEADS))
        random.Random(seed + layer * 1000003).shuffle(order)
        layer_entries: list[tuple[str, float]] = [(SOFTMAX, 1.0)] * base.MODEL_HEADS
        for head, protocol in zip(order, protocols):
            layer_entries[head] = (
                protocol, alphas[layer][head] if protocol == LINEAR else 1.0)
        result.append(tuple(layer_entries))
    return tuple(result)


def plan_composition_key(
        plan: tuple[tuple[tuple[str, float], ...], ...]
        ) -> tuple[tuple[int, int, int], ...]:
    """Return the per-layer composition signature preserved by randomization."""
    return tuple(base.counts_for_plan(plan))


def global_counts(plan: tuple[tuple[tuple[str, float], ...], ...]
                  ) -> tuple[int, int, int]:
    counts = plan_composition_key(plan)
    return tuple(sum(layer[index] for layer in counts) for index in range(3))


def composition_key_text(key: tuple[tuple[int, int, int], ...]) -> str:
    return ";".join(base.composition_name(counts) for counts in key)


def write_random_head_assignment_stats(
        path: Path, reference_policy: str, reference_budget_pp: float,
        reference_state: base.ModelState,
        study_states: dict[str, base.ModelState],
        study_seeds: dict[str, int],
        heldout_by_name: dict[str, dict[str, Any]],
        ours_validation: dict[str, Any]) -> dict[str, Any]:
    """Write the configured matched-composition trials plus a summary row."""
    if not study_states:
        raise ValueError("random study must contain at least one assignment")
    reference_key = plan_composition_key(reference_state.plan)
    if any(plan_composition_key(state.plan) != reference_key
           for state in study_states.values()):
        raise ValueError("random study changed a per-layer S/R/L composition")
    ordered = sorted(study_states, key=lambda name: int(name.rsplit("_", 1)[-1]))
    accuracies = [float(heldout_by_name[name]["candidate_accuracy"]) * 100.0
                  for name in ordered]
    ours_accuracy = float(ours_validation["candidate_accuracy"]) * 100.0
    mean_accuracy = statistics.mean(accuracies)
    std_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    global_s, global_r, global_l = global_counts(reference_state.plan)
    composition_text = composition_key_text(reference_key)
    rows: list[dict[str, Any]] = []
    fields = [
        "row_type", "reference_policy", "reference_budget_pp",
        "reference_plan_sha256", "composition_key", "trial", "seed",
        "plan_sha256", "S_heads", "R_heads", "L_heads",
        "heldout_accuracy_percent", "heldout_accuracy_drop_pp",
        "random_accuracy_mean_percent", "random_accuracy_std_percent",
        "random_accuracy_min_percent", "random_accuracy_max_percent",
        "ours_accuracy_percent", "ours_accuracy_drop_pp",
        "random_minus_ours_pp", "predicted_latency_ms",
        "predicted_communication_bytes", "same_layer_compositions",
        "std_definition", "status",
    ]
    for name in ordered:
        state = study_states[name]
        validation = heldout_by_name[name]
        rows.append({
            "row_type": "trial",
            "reference_policy": reference_policy,
            "reference_budget_pp": reference_budget_pp,
            "reference_plan_sha256": base.plan_key(reference_state.plan),
            "composition_key": composition_text,
            "trial": int(name.rsplit("_", 1)[-1]) + 1,
            "seed": study_seeds[name],
            "plan_sha256": base.plan_key(state.plan),
            "S_heads": global_s, "R_heads": global_r, "L_heads": global_l,
            "heldout_accuracy_percent": float(validation["candidate_accuracy"]) * 100.0,
            "heldout_accuracy_drop_pp": float(validation["accuracy_drop"]) * 100.0,
            "ours_accuracy_percent": ours_accuracy,
            "ours_accuracy_drop_pp": float(ours_validation["accuracy_drop"]) * 100.0,
            "random_minus_ours_pp": (
                float(validation["candidate_accuracy"]) * 100.0 - ours_accuracy),
            "predicted_latency_ms": state.latency_ms,
            "predicted_communication_bytes": state.communication_bytes,
            "same_layer_compositions": True,
            "std_definition": "sample_std_over_trials",
            "status": "MEASURED_CLEAR_HELDOUT",
        })
    rows.append({
        "row_type": "summary",
        "reference_policy": reference_policy,
        "reference_budget_pp": reference_budget_pp,
        "reference_plan_sha256": base.plan_key(reference_state.plan),
        "composition_key": composition_text,
        "trial": "SUMMARY",
        "seed": "",
        "plan_sha256": "",
        "S_heads": global_s, "R_heads": global_r, "L_heads": global_l,
        "heldout_accuracy_percent": "",
        "heldout_accuracy_drop_pp": "",
        "random_accuracy_mean_percent": mean_accuracy,
        "random_accuracy_std_percent": std_accuracy,
        "random_accuracy_min_percent": min(accuracies),
        "random_accuracy_max_percent": max(accuracies),
        "ours_accuracy_percent": ours_accuracy,
        "ours_accuracy_drop_pp": float(ours_validation["accuracy_drop"]) * 100.0,
        "random_minus_ours_pp": mean_accuracy - ours_accuracy,
        "predicted_latency_ms": reference_state.latency_ms,
        "predicted_communication_bytes": reference_state.communication_bytes,
        "same_layer_compositions": True,
        "std_definition": "sample_std_over_trials",
        "status": "PASS",
    })
    write_csv(path, fields, rows)
    return {
        "reference_policy": reference_policy,
        "reference_budget_pp": reference_budget_pp,
        "trials": len(ordered),
        "composition_key": composition_text,
        "S_heads": global_s, "R_heads": global_r, "L_heads": global_l,
        "mean_accuracy_percent": mean_accuracy,
        "std_accuracy_percent": std_accuracy,
        "min_accuracy_percent": min(accuracies),
        "max_accuracy_percent": max(accuracies),
        "ours_accuracy_percent": ours_accuracy,
        "ours_accuracy_drop_pp": float(ours_validation["accuracy_drop"]) * 100.0,
        "mean_minus_ours_pp": mean_accuracy - ours_accuracy,
        "same_layer_compositions": True,
        "std_definition": "sample_std_over_trials",
    }


def actual_pareto(states: dict[str, base.ModelState],
                  validations: dict[str, dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    names = list(states)
    for name in names:
        point = (max(0.0, float(validations[name]["accuracy_drop"])) * 100.0,
                 states[name].latency_ms)
        dominated = False
        for other in names:
            if other == name:
                continue
            other_point = (max(0.0, float(validations[other]["accuracy_drop"])) * 100.0,
                           states[other].latency_ms)
            if base.dominates(other_point + (0.0,), point + (0.0,)):
                dominated = True
                break
        if not dominated:
            result.add(name)
    return result


def select_policy(policy: str, budget_pp: float,
                  states: dict[str, base.ModelState],
                  calibration: dict[str, dict[str, Any]]) -> str:
    if policy == "All-Softmax":
        return min(states, key=lambda name: (
            0 if base.counts_for_plan(states[name].plan)[0] == (12, 0, 0) else 1,
            states[name].latency_ms, name))
    eligible = [name for name, row in calibration.items()
                if float(row["accuracy_drop"]) * 100.0 <= budget_pp + 1e-9]
    if not eligible:
        raise ValueError(f"no calibration-eligible candidate for {policy} budget={budget_pp}")
    if policy == "Accuracy-only":
        return min(eligible, key=lambda name: (
            max(0.0, float(calibration[name]["loss_delta"])),
            max(0.0, float(calibration[name]["relative_l2"])),
            states[name].proxy_score, name))
    if policy == "Cost-only":
        return min(states, key=lambda name: (
            states[name].latency_ms, states[name].communication_bytes, name))
    if policy == "Performance-first":
        return min(eligible, key=lambda name: (
            states[name].latency_ms, states[name].communication_bytes,
            max(0.0, float(calibration[name]["accuracy_drop"])), name))
    # Balanced uses only candidates that satisfy the calibration budget.  Its
    # objective is a normalized clear-risk signal and measured-cost lookup.
    min_latency = min(states[name].latency_ms for name in eligible)
    max_latency = max(states[name].latency_ms for name in eligible)
    min_risk = min(max(0.0, float(calibration[name]["loss_delta"]))
                   + 0.25 * max(0.0, float(calibration[name]["relative_l2"]))
                   for name in eligible)
    max_risk = max(max(0.0, float(calibration[name]["loss_delta"]))
                   + 0.25 * max(0.0, float(calibration[name]["relative_l2"]))
                   for name in eligible)
    latency_span = max(max_latency - min_latency, 1e-12)
    risk_span = max(max_risk - min_risk, 1e-12)
    return min(eligible, key=lambda name: (
        0.5 * (states[name].latency_ms - min_latency) / latency_span +
        0.5 * ((max(0.0, float(calibration[name]["loss_delta"])) +
                0.25 * max(0.0, float(calibration[name]["relative_l2"])) -
                min_risk) / risk_span),
        states[name].latency_ms, name))


def restricted_dp_oracle(layer: int, head_subset: list[int],
                         candidates: dict[tuple[int, int, str], dict[str, Any]],
                         alphas: list[list[float]],
                         cost_model: base.MeasuredCompositionCostModel
                         ) -> tuple[dict[tuple[int, int, int], tuple[float, tuple[str, ...]]],
                                    int, int]:
    fixed_softmax = base.MODEL_HEADS - len(head_subset)
    states: dict[tuple[int, int, int], tuple[float, tuple[str, ...]]] = {
        (0, 0, 0): (0.0, ())}
    transitions = 0
    max_states = 1
    for head in head_subset:
        next_states: dict[tuple[int, int, int], tuple[float, tuple[str, ...]]] = {}
        for used, (score, path_names) in states.items():
            for protocol in PROTOCOLS:
                transitions += 1
                increment = (1, 0, 0) if protocol == SOFTMAX else (
                    (0, 1, 0) if protocol == RELU else (0, 0, 1))
                counts = tuple(used[index] + increment[index] for index in range(3))
                row = candidates.get((layer, head, protocol))
                cost = 0.0 if protocol == SOFTMAX else float(row["proxy_score"])
                new = (score + cost, path_names + (protocol,))
                old = next_states.get(counts)
                if old is None or new < old:
                    next_states[counts] = new
        states = next_states
        max_states = max(max_states, len(states))
    by_counts: dict[tuple[int, int, int], tuple[float, tuple[str, ...]]] = {}
    for selected_counts, value in states.items():
        counts = (selected_counts[0] + fixed_softmax,
                  selected_counts[1], selected_counts[2])
        if cost_model.predict_layer(layer, counts) is not None:
            by_counts[counts] = value
    return by_counts, transitions, max_states


def exhaustive_oracle(layer: int, head_subset: list[int],
                      candidates: dict[tuple[int, int, str], dict[str, Any]],
                      cost_model: base.MeasuredCompositionCostModel
                      ) -> tuple[dict[tuple[int, int, int], tuple[float, tuple[str, ...]]],
                                 int, int, float]:
    start = time.perf_counter()
    result: dict[tuple[int, int, int], tuple[float, tuple[str, ...]]] = {}
    feasible = 0
    for combo in itertools.product(PROTOCOLS, repeat=len(head_subset)):
        counts = [base.MODEL_HEADS - len(head_subset), 0, 0]
        score = 0.0
        for head, protocol in zip(head_subset, combo):
            if protocol == SOFTMAX:
                counts[0] += 1
            elif protocol == RELU:
                counts[1] += 1
            elif protocol == LINEAR:
                counts[2] += 1
            if protocol != SOFTMAX:
                score += float(candidates[(layer, head, protocol)]["proxy_score"])
        counts_tuple = tuple(counts)  # type: ignore[assignment]
        if cost_model.predict_layer(layer, counts_tuple) is None:
            continue
        feasible += 1
        old = result.get(counts_tuple)
        candidate = (score, combo)
        if old is None or candidate < old:
            result[counts_tuple] = candidate
    return result, 3 ** len(head_subset), feasible, (time.perf_counter() - start) * 1000.0


def oracle_rows(layer: int, head_subset: list[int],
                candidates: dict[tuple[int, int, str], dict[str, Any]],
                alphas: list[list[float]],
                cost_model: base.MeasuredCompositionCostModel
                ) -> list[dict[str, Any]]:
    dp, transitions, max_dp_states = restricted_dp_oracle(
        layer, head_subset, candidates, alphas, cost_model)
    exhaustive, enumerated, feasible_assignments, exhaustive_ms = exhaustive_oracle(
        layer, head_subset, candidates, cost_model)

    def point_map(values: dict[tuple[int, int, int], tuple[float, tuple[str, ...]]]
                  ) -> dict[tuple[tuple[int, int, int], float, float, int], tuple[str, ...]]:
        result = {}
        for counts, (score, _) in values.items():
            cost = cost_model.predict_layer(layer, counts)
            assert cost is not None
            result[(counts, round(score, 12), round(cost.latency_ms, 6),
                    cost.communication_bytes)] = ()
        return result

    def pareto(values: dict[tuple[int, int, int], tuple[float, tuple[str, ...]]]
                ) -> dict[tuple[int, int, int], tuple[float, float, int]]:
        points = {}
        for counts, (score, _) in values.items():
            cost = cost_model.predict_layer(layer, counts)
            assert cost is not None
            points[counts] = (score, cost.latency_ms, cost.communication_bytes)
        kept = {}
        for counts, point in points.items():
            if not any(base.dominates(other, point)
                       for other_counts, other in points.items()
                       if other_counts != counts):
                kept[counts] = point
        return kept

    dp_pareto = pareto(dp)
    exhaustive_pareto = pareto(exhaustive)
    dp_keys = {(counts, round(point[0], 12), round(point[1], 6), point[2])
               for counts, point in dp_pareto.items()}
    exhaustive_keys = {(counts, round(point[0], 12), round(point[1], 6), point[2])
                       for counts, point in exhaustive_pareto.items()}
    equal = dp_keys == exhaustive_keys
    rows: list[dict[str, Any]] = []
    rows.append({
        "row_type": "summary", "method": "DP_vs_exhaustive",
        "layer": layer, "head_subset": ";".join(map(str, head_subset)),
        "enumerated_assignments": enumerated,
        "feasible_assignments": feasible_assignments,
        "dp_transition_count": transitions,
        "dp_max_states": max_dp_states,
        "dp_pareto_points": len(dp_pareto),
        "exhaustive_pareto_points": len(exhaustive_pareto),
        "pareto_set_equal": equal,
        "dp_exhaustive_proxy_gap": max(
            abs(dp[counts][0] - exhaustive[counts][0])
            for counts in dp.keys() & exhaustive.keys()) if dp else "",
        "exhaustive_time_ms": exhaustive_ms,
        "comparison_scope": "layer-0 measured-composition subset; additive proxy; no full-model brute force",
    })
    for method, values, pareto_values in (
            ("DP", dp, dp_pareto), ("exhaustive", exhaustive, exhaustive_pareto)):
        for counts, point in sorted(pareto_values.items()):
            rows.append({
                "row_type": "pareto_point", "method": method,
                "layer": layer,
                "head_subset": ";".join(map(str, head_subset)),
                "composition": base.composition_name(counts),
                "proxy_score": point[0], "predicted_latency_ms": point[1],
                "predicted_communication_bytes": point[2],
                "is_in_other_pareto": (
                    (counts in exhaustive_pareto) if method == "DP"
                    else (counts in dp_pareto)),
                "enumerated_assignments": enumerated,
                "feasible_assignments": feasible_assignments,
                "dp_transition_count": transitions,
                "dp_max_states": max_dp_states,
                "dp_pareto_points": len(dp_pareto),
                "exhaustive_pareto_points": len(exhaustive_pareto),
                "pareto_set_equal": equal,
                "comparison_scope": "layer-0 measured-composition subset; additive proxy; no full-model brute force",
            })
    return rows


def validation_rows_for(states: dict[str, base.ModelState],
                        validations: dict[str, dict[str, Any]],
                        calibration: dict[str, dict[str, Any]],
                        pareto_names: set[str], budget_pp: float,
                        selected: dict[str, str],
                        random_validations: dict[str, dict[str, Any]],
                        random_states: dict[str, base.ModelState],
                        random_seeds: dict[str, int]
                        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    def common(name: str, state: base.ModelState,
               holdout: dict[str, Any], cal: dict[str, Any]) -> dict[str, Any]:
        counts = base.counts_for_plan(state.plan)
        global_counts = tuple(sum(item[index] for item in counts)
                              for index in range(3))
        return {
            "candidate": name,
            "plan_sha256": base.plan_key(state.plan),
            "S_heads": global_counts[0], "R_heads": global_counts[1],
            "L_heads": global_counts[2],
            "layer_compositions": ";".join(base.composition_name(item) for item in counts),
            "calibration_accuracy_percent": float(cal["candidate_accuracy"]) * 100.0,
            "calibration_accuracy_drop_pp": float(cal["accuracy_drop"]) * 100.0,
            "heldout_accuracy_percent": float(holdout["candidate_accuracy"]) * 100.0,
            "heldout_accuracy_drop_pp": float(holdout["accuracy_drop"]) * 100.0,
            "validation_loss": float(holdout["validation_loss"]),
            "loss_delta": float(holdout["loss_delta"]),
            "relative_l2": float(holdout["relative_l2"]),
            "predicted_latency_ms": state.latency_ms,
            "predicted_communication_bytes": state.communication_bytes,
            "proxy_score": state.proxy_score,
        }
    for name, state in sorted(states.items()):
        row = common(name, state, validations[name], calibration[name])
        row.update({
            "row_type": "candidate_pool", "budget_pp": "",
            "policy": "candidate_pool", "selected": False,
            "calibration_budget_pass": "", "heldout_budget_pass": "",
            "accuracy_latency_pareto": name in pareto_names,
        })
        policy_rows.append(row)
    for policy in POLICY_ORDER:
        name = selected[policy]
        state = states[name]
        row = common(name, state, validations[name], calibration[name])
        calibration_pass = float(calibration[name]["accuracy_drop"]) * 100.0 <= budget_pp + 1e-9
        heldout_pass = float(validations[name]["accuracy_drop"]) * 100.0 <= budget_pp + 1e-9
        row.update({
            "row_type": "selected_policy", "budget_pp": budget_pp,
            "policy": policy, "selected": True,
            "calibration_budget_pass": calibration_pass,
            "heldout_budget_pass": heldout_pass,
            "accuracy_latency_pareto": name in pareto_names,
        })
        policy_rows.append(row)

        random_name = f"random::{policy}::{budget_pp:g}"
        random_state = random_states[random_name]
        random_row = common(random_name, random_state,
                            random_validations[random_name]["heldout"],
                            random_validations[random_name]["calibration"])
        baseline_rows.append({
            "budget_pp": budget_pp,
            "reference_policy": policy,
            "variant": "allocator",
            "random_seed": "",
            **row,
            "same_layer_compositions": True,
        })
        baseline_rows.append({
            "budget_pp": budget_pp,
            "reference_policy": policy,
            "variant": "random_head_baseline",
            "random_seed": random_seeds[random_name],
            **random_row,
            "same_layer_compositions": (
                base.counts_for_plan(random_state.plan) == base.counts_for_plan(state.plan)),
        })
    return policy_rows, baseline_rows


def create_pool(model: dict[str, Any], calibration_pixels: list[np.ndarray],
                calibration_labels: list[int], validation_pixels: list[np.ndarray],
                validation_labels: list[int], workers: int,
                logit_weight: float, batch_size: int) -> Any | None:
    if workers <= 1:
        return None
    context = base.mp.get_context("fork")
    return context.Pool(
        workers,
        initializer=base.candidate_worker_init,
        initargs=(model, calibration_pixels, calibration_labels,
                  validation_pixels, validation_labels, logit_weight, batch_size),
    )


def close_pool(pool: Any | None) -> None:
    if pool is not None:
        pool.close()
        pool.join()


def assignment_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def write_blocked_result_tables(output_dir: Path, status: str,
                                reason: str) -> None:
    """Create auditable empty result tables when the cost gate is incomplete.

    A failed/incomplete real-cost gate must not be turned into a successful
    clear allocator run by filling measured columns with model predictions.
    Keeping the exact downstream schemas, but no data rows, makes the
    distinction machine-readable and satisfies the experiment contract that
    all five result-table paths exist.
    """
    schemas = {
        "pareto_policies.csv": [
            "row_type", "budget_pp", "policy", "candidate", "plan_sha256",
            "selected", "accuracy_latency_pareto", "calibration_budget_pass",
            "heldout_budget_pass", "S_heads", "R_heads", "L_heads",
            "clear_accuracy_percent", "accuracy_drop_pp",
            "calibration_accuracy_percent", "calibration_accuracy_drop_pp",
            "heldout_accuracy_percent", "heldout_accuracy_drop_pp",
            "validation_loss", "loss_delta", "relative_l2", "proxy_score",
            "predicted_latency_ms", "predicted_communication_bytes",
            "layer_compositions",
        ],
        "allocator_baselines.csv": [
            "budget_pp", "reference_policy", "variant", "random_seed",
            "row_type", "candidate", "plan_sha256", "selected",
            "same_layer_compositions", "S_heads", "R_heads", "L_heads",
            "clear_accuracy_percent", "accuracy_drop_pp",
            "calibration_accuracy_percent", "calibration_accuracy_drop_pp",
            "heldout_accuracy_percent", "heldout_accuracy_drop_pp",
            "validation_loss", "loss_delta", "relative_l2", "proxy_score",
            "predicted_latency_ms", "predicted_communication_bytes",
            "layer_compositions",
        ],
        "small_oracle_comparison.csv": [
            "row_type", "method", "layer", "head_subset", "composition",
            "proxy_score", "predicted_latency_ms",
            "predicted_communication_bytes", "is_in_other_pareto",
            "enumerated_assignments", "feasible_assignments",
            "dp_transition_count", "dp_max_states", "dp_pareto_points",
            "exhaustive_pareto_points", "pareto_set_equal",
            "dp_exhaustive_proxy_gap", "exhaustive_time_ms",
            "comparison_scope",
        ],
        "secure_candidates.csv": [
            "candidate", "source_policy", "matched_random_reference", "ablation",
            "head_plan", "config_json", "plan_sha256", "S_heads", "R_heads",
            "L_heads", "clear_accuracy_percent", "accuracy_drop_pp",
            "calibration_accuracy_percent", "calibration_accuracy_drop_pp",
            "predicted_latency_ms", "predicted_communication_bytes",
            "accuracy_budget_pp", "secure_inference_run", "status",
        ],
        "selection_report.csv": [
            "policy", "S_heads", "R_heads", "L_heads", "clear_accuracy",
            "accuracy_drop", "predicted_secure_latency", "predicted_comm",
            "accuracy_dataset", "secure_inference_run",
        ],
    }
    for filename, fields in schemas.items():
        write_csv(output_dir / filename, fields, [])
    (output_dir / "blocked_result_tables.json").write_text(
        json.dumps({
            "status": status,
            "reason": reason,
            "rows_written": 0,
            "measured_values_fabricated": False,
            "tables": sorted(schemas),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--preprocess-manifest", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--allocator-costs", type=Path, required=True,
                        help="existing exact complete-assignment CSV for allocator search")
    parser.add_argument("--supplemental-costs", type=Path, required=True,
                        help="new one-block direct composition probe CSV")
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--allow-missing-probes", action="store_true",
                        help="emit explicit NOT_MEASURED rows instead of fitting a complete holdout")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-count", type=int, default=1000,
                        help="fixed clear calibration size; accuracy resolution is 100/N pp")
    parser.add_argument("--calibration-start", type=int, default=0)
    parser.add_argument("--heldout-count", type=int, default=1000,
                        help="fixed disjoint clear held-out size")
    parser.add_argument("--heldout-start", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="small clear-reference batch size in each worker")
    parser.add_argument("--max-frontier", type=int, default=1024)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--logit-weight", type=float, default=0.25)
    parser.add_argument("--cost-ridge", type=float, default=1e-3)
    parser.add_argument("--oracle-layer", type=int, default=0)
    parser.add_argument("--oracle-heads", type=str, default="0,1,2,3")
    parser.add_argument("--random-seed", type=int, default=20260905)
    parser.add_argument("--random-repeats", type=int, default=20,
                        help="number of matched-composition random assignments")
    parser.add_argument("--final-budget-pp", type=float, default=None,
                        help="budget for legacy comparison tables; defaults to max(budgets-pp)")
    parser.add_argument("--budgets-pp", type=float, nargs="+",
                        default=[0.1, 0.5, 1.0])
    args = parser.parse_args()

    if args.workers <= 0 or args.batch_size <= 0 or args.max_frontier <= 0 or args.max_candidates <= 0:
        raise ValueError("workers/frontier/candidate limits must be positive")
    if args.calibration_count <= 0 or args.heldout_count <= 0:
        raise ValueError("calibration-count and heldout-count must be positive")
    if args.random_repeats <= 0:
        raise ValueError("random-repeats must be positive")
    if len(set(args.budgets_pp)) != len(args.budgets_pp) or any(
            not math.isfinite(value) or value < 0.0 for value in args.budgets_pp):
        raise ValueError("budgets must be unique, finite and non-negative")
    if args.final_budget_pp is None:
        args.final_budget_pp = max(args.budgets_pp)
    if not math.isfinite(args.final_budget_pp) or args.final_budget_pp < 0.0:
        raise ValueError("final budget must be finite and non-negative")
    if args.oracle_layer < 0 or args.oracle_layer >= base.MODEL_LAYERS:
        raise ValueError("oracle layer is outside ViT-B/16")
    head_subset = [int(value) for value in args.oracle_heads.split(",") if value.strip()]
    if len(head_subset) < 4 or len(head_subset) > 6 or len(set(head_subset)) != len(head_subset):
        raise ValueError("oracle must select 4--6 distinct heads")
    if any(head < 0 or head >= base.MODEL_HEADS for head in head_subset):
        raise ValueError("oracle head is outside ViT-B/16")
    if args.final_budget_pp not in args.budgets_pp:
        raise ValueError("final budget must be included in --budgets-pp")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_manifest = output_dir / "fixed_calibration.csv"
    heldout_manifest = output_dir / "fixed_heldout.csv"
    calibration_rows = make_fixed_manifest(
        args.public_manifest, calibration_manifest,
        args.calibration_start, args.calibration_count)
    heldout_rows = make_fixed_manifest(
        args.public_manifest, heldout_manifest,
        args.heldout_start, args.heldout_count)
    assert_disjoint_manifests(calibration_rows, heldout_rows)

    (existing_points, train_probe_points, heldout_points, missing_train,
     missing_heldout) = load_probe_points(
        args.allocator_costs, args.supplemental_costs, args.probe_manifest,
        args.allow_missing_probes)
    expected_heldout = {
        parse_composition(row["composition"])
        for row in read_csv(args.probe_manifest)
        if row["split"] == "heldout"
    }
    holdout_cost_rows, holdout_cost_meta = cost_model_holdout_rows(
        existing_points, train_probe_points, heldout_points, expected_heldout,
        missing_train, args.cost_ridge)
    train_cost_rows = []
    for point in existing_points + train_probe_points + heldout_points:
        train_cost_rows.append({
            "split": point.split,
            "composition": base.composition_name(point.composition),
            "S_heads": point.composition[0], "R_heads": point.composition[1],
            "L_heads": point.composition[2], "layer": point.layer,
            "latency_ms": point.latency_ms,
            "communication_bytes": point.communication_bytes,
            "source": point.source, "plan_sha256": point.plan_sha256,
            "used_in_fit": point.split != "strict_heldout",
            **point.overhead,
        })
    write_csv(output_dir / "cost_model_train.csv", [
        "split", "composition", "S_heads", "R_heads", "L_heads", "layer",
        "latency_ms", "communication_bytes", "source", "plan_sha256",
        "used_in_fit", *OVERHEAD_FIELDS], train_cost_rows)
    holdout_cost_fields = [
        "row_type", "status", "split", "composition", "S_heads", "R_heads", "L_heads",
        "train_observation_count", "heldout_observation_count",
        "measured_latency_ms", "predicted_latency_ms", "latency_ape_percent",
        "latency_mape_percent", "measured_communication_bytes",
        "predicted_communication_bytes", "communication_ape_percent",
        "communication_mape_percent", "latency_ranking_accuracy",
        "communication_ranking_accuracy", "latency_ranking_correct_pairs",
        "communication_ranking_correct_pairs", "ranking_pairs",
        "ranking_total_pairs", "cost_model_gate_pass", "cost_model_gate_reason",
        "cost_model_gate_latency_mape_limit_percent",
        "cost_model_gate_min_latency_correct_pairs",
        *(f"measured_{field_name}" for field_name in OVERHEAD_FIELDS),
        "heldout_plan_sha256", "training_compositions", "heldout_compositions",
        "measured_heldout_compositions", "missing_train_compositions",
        "missing_heldout_compositions",
        "uses_heldout_in_fit", "latency_model", "communication_model",
        "cost_model_fit_status", "cost_model_fit_error",
    ]
    write_csv(output_dir / "cost_model_holdout.csv", holdout_cost_fields,
              holdout_cost_rows)
    # Keep the descriptive validation name as an alias.  Both files contain
    # exactly the same rows; neither can be mistaken for an independent
    # measurement set.
    write_csv(output_dir / "cost_model_validation.csv", holdout_cost_fields,
              holdout_cost_rows)
    if not holdout_cost_meta.get("cost_model_gate_pass", False):
        gate_status = holdout_cost_meta.get(
            "status", "FAIL_COST_MODEL_GATE")
        gate_reason = holdout_cost_meta.get(
            "cost_model_gate_reason", "unknown")
        write_blocked_result_tables(output_dir, gate_status, gate_reason)
        gate_report = {
            "status": gate_status,
            "cost_model_gate": holdout_cost_meta,
            "cost_model_holdout": str(
                (output_dir / "cost_model_holdout.csv").resolve()),
            "cost_model_validation": str(
                (output_dir / "cost_model_validation.csv").resolve()),
            "blocked_result_tables": str(
                (output_dir / "blocked_result_tables.json").resolve()),
            "allocator_costs": str(args.allocator_costs.resolve()),
            "supplemental_costs": str(args.supplemental_costs.resolve()),
            "probe_manifest": str(args.probe_manifest.resolve()),
            "calibration_count": args.calibration_count,
            "heldout_count": args.heldout_count,
            "calibration_heldout_disjoint": True,
            "secure_inference_run": False,
            "allocator_run": False,
            "next_action": "fix_cost_model_before_clear_pareto_or_secure_plans",
        }
        (output_dir / "cost_model_gate_report.json").write_text(
            json.dumps(gate_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(
            "cost model gate failed; no Pareto allocator or secure plans generated: "
            f"{gate_reason}")
        print("Missing training compositions: " + ", ".join(
            holdout_cost_meta.get("missing_train_compositions", [])))
        print("Missing held-out compositions: " + ", ".join(
            holdout_cost_meta.get("missing_heldout_compositions", [])))
        raise SystemExit(3)

    model = base.vr.load_model(args.model)
    if (int(model["layers"]) != base.MODEL_LAYERS or
            int(model["heads"]) != base.MODEL_HEADS or
            int(model["tokens"]) != base.MODEL_TOKENS):
        raise ValueError("experiment requires the audited ViT-B/16 12x12/197 model")
    preprocess = base.load_preprocess(args.preprocess_manifest, model)
    calibration_samples = base.read_manifest(calibration_manifest, args.image_root,
                                             args.calibration_count)
    heldout_samples = base.read_manifest(heldout_manifest, args.image_root,
                                         args.heldout_count)
    calibration_pixels = base.load_pixels_for_samples(calibration_samples, preprocess)
    heldout_pixels = base.load_pixels_for_samples(heldout_samples, preprocess)
    calibration_labels = [sample.label for sample in calibration_samples]
    heldout_labels = [sample.label for sample in heldout_samples]

    cost_model = base.MeasuredCompositionCostModel(
        args.allocator_costs, base.MODEL_LAYERS, base.MODEL_HEADS)
    if not set(base.BENCHMARK_COMPOSITIONS).issubset(cost_model.available_compositions):
        raise ValueError("allocator cost CSV is missing an existing exact search composition")
    latency_cost_model, communication_cost_model = fit_cost_models(
        existing_points + train_probe_points, args.cost_ridge)
    search_cost_model = SearchCompositionCostModel(
        cost_model, existing_points + train_probe_points,
        latency_cost_model, communication_cost_model)

    # Fork all worker pools before the parent performs any BLAS call.  One
    # pool owns calibration validation, one owns held-out validation; neither
    # pool invokes a secure binary.
    calibration_pool = create_pool(
        model, calibration_pixels, calibration_labels,
        calibration_pixels, calibration_labels, args.workers, args.logit_weight,
        args.batch_size)
    heldout_pool: Any | None = None
    try:
        calibration_baseline: list[np.ndarray] = []
        calibration_records: list[dict[str, dict[str, Any]]] = []
        for pixels in calibration_pixels:
            logits, records = base.vr.Graph(model, fixed=False).run(pixels)
            calibration_baseline.append(logits)
            calibration_records.append(records)
        alphas = base.fit_linear_alphas(
            calibration_records, base.MODEL_LAYERS, base.MODEL_HEADS)
        isolated_fields = [
            "layer", "head", "protocol", "alpha", "validation_loss",
            "baseline_validation_loss", "loss_delta", "mse", "max_abs",
            "relative_l2", "top1_agreement", "baseline_accuracy",
            "candidate_accuracy", "accuracy_drop", "proxy_score", "elements"]
        isolated_cache = output_dir / "isolated_candidate_metrics.csv"
        isolated_config_path = output_dir / "isolated_candidate_metrics.config.json"
        isolated_config = {
            "model_sha256": base.sha256_file(args.model),
            "preprocess_sha256": base.sha256_file(args.preprocess_manifest),
            "calibration_manifest_sha256": base.sha256_file(calibration_manifest),
            "image_root": str(args.image_root.resolve()),
            "logit_weight": args.logit_weight,
            "allocator_sha256": base.sha256_file(Path(base.__file__)),
            "reference_sha256": base.sha256_file(Path(base.vr.__file__)),
        }
        isolated: list[dict[str, Any]] | None = None
        if (isolated_cache.exists() and isolated_config_path.exists() and
                json.loads(isolated_config_path.read_text()) == isolated_config):
            with isolated_cache.open(newline="") as handle:
                cached_rows = list(csv.DictReader(handle))
            expected_keys = {
                (layer, head, protocol)
                for layer in range(base.MODEL_LAYERS)
                for head in range(base.MODEL_HEADS)
                for protocol in (RELU, LINEAR)
            }
            cached_keys = {
                (int(row["layer"]), int(row["head"]), row["protocol"])
                for row in cached_rows
                if row.get("layer") is not None and row.get("head") is not None
            }
            if len(cached_rows) == len(expected_keys) and cached_keys == expected_keys:
                isolated = cached_rows
                print(f"OURS_CLEAR_ISOLATED reuse={len(isolated)}/{len(expected_keys)}")
        if isolated is None:
            isolated = base.run_isolated_candidates(
                model, calibration_pixels, calibration_labels,
                calibration_baseline, alphas, args.workers, args.logit_weight,
                calibration_pool, args.batch_size, progress_every=12)
            isolated.sort(key=lambda row: (int(row["layer"]), int(row["head"]),
                                           str(row["protocol"])))
            write_csv(isolated_cache, isolated_fields, isolated)
            isolated_config_path.write_text(json.dumps(isolated_config, sort_keys=True) + "\n")
        isolated_index = base.load_candidate_index(isolated)
        write_sensitivity_outputs(
            output_dir, isolated, alphas, args.logit_weight)

        layer_options: list[list[base.LayerOption]] = []
        layer_summary: list[dict[str, Any]] = []
        for layer in range(base.MODEL_LAYERS):
            options: list[base.LayerOption] = []
            for counts in SEARCH_COMPOSITIONS:
                cost = search_cost_model.predict_layer(layer, counts)
                paths, score = base.min_assignment_for_counts(
                    layer, counts, isolated_index, alphas)
                options.append(base.option_from_paths(
                    layer, counts, paths, score, isolated_index, cost))
            pruned = base.pareto_layer_options(options)
            if not pruned:
                raise ValueError(f"layer {layer} has no measured Pareto option")
            layer_options.append(pruned)
            layer_summary.append({
                "layer": layer,
                "search_options": len(options),
                "pareto_options": len(pruned),
                "compositions": [base.composition_name(option.counts)
                                 for option in pruned],
            })
        frontier, frontier_truncated = base.combine_model_frontier(
            layer_options, args.max_frontier)
        pool_states = base.candidate_pool(
            frontier, layer_options, args.max_candidates, SEARCH_COMPOSITIONS)
        state_by_name = {
            f"candidate_{index:03d}": state
            for index, state in enumerate(pool_states)
        }
        all_softmax = base.uniform_composition_state(layer_options, (12, 0, 0))
        if all_softmax is None:
            raise ValueError("all-softmax state is unavailable")
        baseline_name = next((name for name, state in state_by_name.items()
                              if base.plan_key(state.plan) == base.plan_key(all_softmax.plan)), None)
        if baseline_name is None:
            baseline_name = "candidate_baseline"
            state_by_name[baseline_name] = all_softmax

        calibration_valid_baseline = calibration_baseline
        calibration_validation_rows = base.run_validation(
            model, calibration_pixels, calibration_labels,
            calibration_valid_baseline, state_by_name, args.workers,
            calibration_pool, args.batch_size, progress_every=1)
        calibration_by_name = {
            row["candidate"]: row for row in calibration_validation_rows}
        if set(calibration_by_name) != set(state_by_name):
            raise ValueError("calibration candidate validation is incomplete")

        selected_by_budget: dict[float, dict[str, str]] = {}
        for budget in args.budgets_pp:
            selected_by_budget[budget] = {
                policy: select_policy(policy, budget, state_by_name, calibration_by_name)
                for policy in POLICY_ORDER
            }

        # Build one matched random baseline per unique per-layer composition.
        # The 35 policy/budget aliases below may therefore share a single
        # clear evaluation; no random result is used in DP or policy selection.
        random_states: dict[str, base.ModelState] = {}
        random_seeds: dict[str, int] = {}
        random_alias_to_unique: dict[str, str] = {}
        random_unique_states: dict[str, base.ModelState] = {}
        random_unique_seeds: dict[str, int] = {}
        random_unique_by_composition: dict[
            tuple[tuple[int, int, int], ...], str] = {}
        random_state_counter = 0
        for budget in args.budgets_pp:
            for policy in POLICY_ORDER:
                name = selected_by_budget[budget][policy]
                source_state = state_by_name[name]
                composition_key = plan_composition_key(source_state.plan)
                unique_name = random_unique_by_composition.get(composition_key)
                if unique_name is None:
                    unique_name = f"random_composition_{len(random_unique_states):03d}"
                    seed = args.random_seed + random_state_counter
                    random_state_counter += 1
                    random_unique_by_composition[composition_key] = unique_name
                    random_unique_seeds[unique_name] = seed
                    random_unique_states[unique_name] = state_from_plan(
                        randomize_plan(source_state.plan, alphas, seed),
                        isolated_index, search_cost_model)
                random_name = f"random::{policy}::{budget:g}"
                random_alias_to_unique[random_name] = unique_name
                random_states[random_name] = random_unique_states[unique_name]
                random_seeds[random_name] = random_unique_seeds[unique_name]

        # The final random study is tied to the final Performance-first plan.
        # It has exactly the same per-layer S/R/L counts as Ours and is
        # evaluated independently on held-out data.  Only trial 1 is also
        # evaluated on calibration, because the study statistic is held-out
        # accuracy, not another selection signal.
        random_reference = "Performance-first"
        reference_name = selected_by_budget[args.final_budget_pp][random_reference]
        reference_state = state_by_name[reference_name]
        random_study_states: dict[str, base.ModelState] = {}
        random_study_seeds: dict[str, int] = {}
        for trial in range(args.random_repeats):
            study_name = f"random_study_{trial:03d}"
            seed = args.random_seed + 1_000_000 + trial
            random_study_seeds[study_name] = seed
            random_study_states[study_name] = state_from_plan(
                randomize_plan(reference_state.plan, alphas, seed),
                isolated_index, search_cost_model)

        random_calibration_eval_states = dict(random_unique_states)
        random_calibration_eval_states["random_study_000"] = random_study_states[
            "random_study_000"]
        random_calibration_rows = base.run_validation(
            model, calibration_pixels, calibration_labels,
            calibration_valid_baseline, random_calibration_eval_states, args.workers,
            calibration_pool, args.batch_size, progress_every=1)
        random_calibration_by_name = {
            row["candidate"]: row for row in random_calibration_rows}
        if set(random_calibration_by_name) != set(random_calibration_eval_states):
            raise ValueError("calibration random-baseline validation is incomplete")
        close_pool(calibration_pool)
        calibration_pool = None

        # Hold-out evaluation is run in a separate pool only after all policy
        # decisions and random seeds have been fixed.  This also keeps the
        # large calibration and held-out worker arrays from co-residing.
        heldout_pool = create_pool(
            model, calibration_pixels, calibration_labels,
            heldout_pixels, heldout_labels, args.workers, args.logit_weight,
            args.batch_size)
        heldout_baseline = clear_baseline_batch(
            model, heldout_pixels, args.batch_size)
        heldout_validation_rows = base.run_validation(
            model, heldout_pixels, heldout_labels, heldout_baseline,
            state_by_name, args.workers, heldout_pool, args.batch_size,
            progress_every=1)
        random_heldout_eval_states = dict(random_unique_states)
        random_heldout_eval_states.update(random_study_states)
        random_heldout_rows = base.run_validation(
            model, heldout_pixels, heldout_labels, heldout_baseline,
            random_heldout_eval_states, args.workers, heldout_pool, args.batch_size,
            progress_every=1)
        heldout_by_name = {row["candidate"]: row for row in heldout_validation_rows}
        random_heldout_by_name = {row["candidate"]: row for row in random_heldout_rows}
        if set(heldout_by_name) != set(state_by_name) or \
                set(random_heldout_by_name) != set(random_heldout_eval_states):
            raise ValueError("held-out candidate validation is incomplete")
        random_validations = {
            name: {"calibration": random_calibration_by_name[
                       random_alias_to_unique[name]],
                   "heldout": random_heldout_by_name[
                       random_alias_to_unique[name]]}
            for name in random_states
        }

        # The actual Accuracy--Latency Pareto mark is evaluated on held-out
        # Accuracy and exact predicted composition cost.  Policy eligibility
        # remains calibration-only, so the held-out set is not used to tune a
        # policy budget.
        pareto_names = actual_pareto(state_by_name, heldout_by_name)
        policy_rows: list[dict[str, Any]] = []
        baseline_rows: list[dict[str, Any]] = []
        for budget in args.budgets_pp:
            rows, baselines = validation_rows_for(
                state_by_name, heldout_by_name, calibration_by_name,
                pareto_names, budget, selected_by_budget[budget],
                random_validations, random_states, random_seeds)
            policy_rows.extend(rows)
            baseline_rows.extend(baselines)

        policy_fields = [
            "row_type", "budget_pp", "policy", "candidate", "plan_sha256",
            "selected", "accuracy_latency_pareto", "calibration_budget_pass",
            "heldout_budget_pass", "S_heads", "R_heads", "L_heads",
            "clear_accuracy_percent", "accuracy_drop_pp",
            "calibration_accuracy_percent", "calibration_accuracy_drop_pp",
            "heldout_accuracy_percent", "heldout_accuracy_drop_pp",
            "validation_loss", "loss_delta", "relative_l2", "proxy_score",
            "predicted_latency_ms", "predicted_communication_bytes",
            "layer_compositions",
        ]
        # Add compatibility aliases used in the previous allocator reports.
        for row in policy_rows:
            row["clear_accuracy_percent"] = row["heldout_accuracy_percent"]
            row["accuracy_drop_pp"] = row["heldout_accuracy_drop_pp"]
        write_csv(output_dir / "pareto_policies.csv", policy_fields, policy_rows)

        baseline_fields = [
            "budget_pp", "reference_policy", "variant", "random_seed",
            "row_type", "candidate", "plan_sha256", "selected",
            "same_layer_compositions", "S_heads", "R_heads", "L_heads",
            "clear_accuracy_percent", "accuracy_drop_pp",
            "calibration_accuracy_percent", "calibration_accuracy_drop_pp",
            "heldout_accuracy_percent", "heldout_accuracy_drop_pp",
            "validation_loss", "loss_delta", "relative_l2", "proxy_score",
            "predicted_latency_ms", "predicted_communication_bytes",
            "layer_compositions",
        ]
        write_csv(output_dir / "allocator_baselines.csv", baseline_fields, baseline_rows)

        oracle = oracle_rows(args.oracle_layer, head_subset, isolated_index,
                             alphas, cost_model)
        oracle_fields = [
            "row_type", "method", "layer", "head_subset", "composition",
            "proxy_score", "predicted_latency_ms",
            "predicted_communication_bytes", "is_in_other_pareto",
            "enumerated_assignments", "feasible_assignments",
            "dp_transition_count", "dp_max_states", "dp_pareto_points",
            "exhaustive_pareto_points", "pareto_set_equal",
            "dp_exhaustive_proxy_gap", "exhaustive_time_ms",
            "comparison_scope",
        ]
        write_csv(output_dir / "small_oracle_comparison.csv", oracle_fields, oracle)

        random_stats_summary = write_random_head_assignment_stats(
            output_dir / "random_head_assignment_stats.csv",
            random_reference, args.final_budget_pp, reference_state,
            random_study_states, random_study_seeds, random_heldout_by_name,
            heldout_by_name[reference_name])

        # Materialize only the five requested secure candidates.  Cost-only is
        # intentionally an ablation row only; it receives no secure plan.
        final_selected = selected_by_budget[args.final_budget_pp]
        final_states: dict[str, base.ModelState] = {
            "All-Softmax": all_softmax,
            "Accuracy-only": state_by_name[final_selected["Accuracy-only"]],
            "Balanced": state_by_name[final_selected["Balanced"]],
            "Performance-first": state_by_name[final_selected["Performance-first"]],
            "Random": random_study_states["random_study_000"],
        }
        final_validation: dict[str, dict[str, Any]] = {
            "All-Softmax": heldout_by_name[baseline_name],
            "Accuracy-only": heldout_by_name[final_selected["Accuracy-only"]],
            "Balanced": heldout_by_name[final_selected["Balanced"]],
            "Performance-first": heldout_by_name[final_selected["Performance-first"]],
            "Random": random_heldout_by_name["random_study_000"],
        }
        final_calibration: dict[str, dict[str, Any]] = {
            "All-Softmax": calibration_by_name[baseline_name],
            "Accuracy-only": calibration_by_name[final_selected["Accuracy-only"]],
            "Balanced": calibration_by_name[final_selected["Balanced"]],
            "Performance-first": calibration_by_name[final_selected["Performance-first"]],
            "Random": random_calibration_by_name["random_study_000"],
        }
        final_documents: dict[str, dict[str, Any]] = {}
        search_meta = {
            "algorithm": "layerwise_exact_count_dp_then_model_pareto_dp",
            "whole_model_bruteforce": False,
            "per_layer_head_state_space": "prefix counts (S,R,L), 3 transitions/head",
            "calibration_samples": len(calibration_samples),
            "heldout_samples": len(heldout_samples),
            "fixed_split_disjoint": True,
            "validation_samples": len(heldout_samples),
            "selection_accuracy_set": "fixed_calibration",
            "final_accuracy_set": "fixed_heldout",
            "available_measured_compositions": [
                base.composition_name(counts)
                for counts in sorted(cost_model.available_compositions)],
            "available_training_probe_compositions": [
                base.composition_name(counts)
                for counts in sorted(search_cost_model.available_compositions -
                                     cost_model.available_compositions)],
            "search_composition_grid_size": len(SEARCH_COMPOSITIONS),
            "search_cost_mode": (
                "direct_measured_anchor_else_layer_aware_composition_model"),
            "relu_forced": False,
            "head_grouping_enabled": False,
            "cost_model_training": latency_cost_model.metadata(),
            "cost_model_gate": holdout_cost_meta,
            "composition_probe_manifest": str(args.probe_manifest.resolve()),
            "composition_probe_manifest_sha256": base.sha256_file(args.probe_manifest),
            "layer_pareto_summary": layer_summary,
            "model_frontier_size": len(frontier),
            "model_frontier_truncated": frontier_truncated,
            "candidate_pool_size": len(state_by_name),
            "candidate_pool_max": args.max_candidates,
            "random_baseline_unique_compositions": len(random_unique_states),
            "random_study": random_stats_summary,
            "random_head_assignment_stats": str(
                (output_dir / "random_head_assignment_stats.csv").resolve()),
            "budgets_pp": args.budgets_pp,
            "final_budget_pp": args.final_budget_pp,
            "random_seed": args.random_seed,
            "selected_by_budget": {
                str(budget): selected_by_budget[budget] for budget in args.budgets_pp},
            "isolated_candidate_metrics": str(
                (output_dir / "isolated_candidate_metrics.csv").resolve()),
            "pareto_policies": str((output_dir / "pareto_policies.csv").resolve()),
        }
        # Export one minimum-cost eligible plan for every requested budget.
        # The legacy comparison budget only selects the optional ablation
        # tables below; it must not discard the other deployment plans.
        budget_rows = []
        first_budget_by_hash = {}
        for budget in args.budgets_pp:
            name = selected_by_budget[budget]["Performance-first"]
            state = state_by_name[name]
            budget_dir = output_dir / "budgets" / str(float(budget))
            budget_dir.mkdir(parents=True, exist_ok=True)
            plan_path = budget_dir / "head_plan.txt"
            base.write_runtime_plan(plan_path, state.plan)
            document = base.output_assignment(
                budget_dir / "head_assignment.json", "Performance-first",
                budget, state, heldout_by_name[name],
                search_cost_model.predict_plan(state.plan), alphas,
                args.model, args.model_manifest, calibration_manifest,
                heldout_manifest, args.allocator_costs, plan_path,
                search_cost_model.missing_compositions(),
                {**search_meta, "selected_candidate": name})
            plan_hash = document["search"]["selected_plan_sha256"]
            canonical_budget = first_budget_by_hash.setdefault(plan_hash, budget)
            budget_rows.append({
                "budget_pp": budget,
                "calibration_accuracy_drop_pp": calibration_by_name[name]["accuracy_drop"] * 100,
                "heldout_accuracy_drop_pp": heldout_by_name[name]["accuracy_drop"] * 100,
                "heldout_budget_pass": document["status"] == "PASS",
                "predicted_attention_latency_ms": state.latency_ms,
                "predicted_communication_bytes": state.communication_bytes,
                "assignment": str((budget_dir / "head_assignment.json").resolve()),
                "plan_sha256": plan_hash,
                "same_plan_as_budget_pp": canonical_budget,
                "secure_inference_run": False,
            })
        write_csv(output_dir / "budget_candidates.csv", list(budget_rows[0]), budget_rows)

        for policy in ("All-Softmax", "Random", "Accuracy-only",
                       "Balanced", "Performance-first"):
            state = final_states[policy]
            cost = search_cost_model.predict_plan(state.plan)
            plan_path = output_dir / f"{assignment_name(policy)}.head_plan.txt"
            base.write_runtime_plan(plan_path, state.plan)
            document = base.output_assignment(
                output_dir / f"{assignment_name(policy)}.json",
                policy, args.final_budget_pp, state, final_validation[policy],
                cost, alphas, args.model, args.model_manifest,
                calibration_manifest, heldout_manifest, args.allocator_costs,
                plan_path, search_cost_model.missing_compositions(),
                {**search_meta, "selected_candidate": policy,
                 "matched_random_reference": random_reference if policy == "Random" else ""})
            document["selection_dataset"] = "fixed_calibration"
            document["final_accuracy_dataset"] = "fixed_heldout"
            document["cost_model_holdout"] = str(
                (output_dir / "cost_model_holdout.csv").resolve())
            document["cost_model_holdout_status"] = holdout_cost_meta.get(
                "status", "UNKNOWN")
            (output_dir / f"{assignment_name(policy)}.json").write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            final_documents[policy] = document

        secure_rows: list[dict[str, Any]] = []
        for policy in ("All-Softmax", "Random", "Accuracy-only",
                       "Balanced", "Performance-first"):
            document = final_documents[policy]
            state = final_states[policy]
            counts = base.counts_for_plan(state.plan)
            global_counts = tuple(sum(item[index] for item in counts)
                                  for index in range(3))
            secure_rows.append({
                "candidate": policy,
                "source_policy": (random_reference if policy == "Random" else policy),
                "matched_random_reference": (random_reference if policy == "Random" else ""),
                "ablation": document["runtime"]["ablation"],
                "head_plan": document["runtime"]["head_plan"],
                "config_json": str((output_dir / f"{assignment_name(policy)}.json").resolve()),
                "plan_sha256": document["search"]["selected_plan_sha256"],
                "S_heads": global_counts[0], "R_heads": global_counts[1],
                "L_heads": global_counts[2],
                "clear_accuracy_percent": final_validation[policy]["candidate_accuracy"] * 100.0,
                "accuracy_drop_pp": final_validation[policy]["accuracy_drop"] * 100.0,
                "calibration_accuracy_percent": final_calibration[policy]["candidate_accuracy"] * 100.0,
                "calibration_accuracy_drop_pp": final_calibration[policy]["accuracy_drop"] * 100.0,
                "predicted_latency_ms": document["predicted_cost"]["latency_ms"],
                "predicted_communication_bytes": document["predicted_cost"]["communication_bytes"],
                "accuracy_budget_pp": args.final_budget_pp,
                "secure_inference_run": False,
                "status": "READY_NOT_RUN",
            })
        unique_by_hash: dict[str, dict[str, Any]] = {}
        unique_rows: list[dict[str, Any]] = []
        for row in secure_rows:
            plan_hash = str(row["plan_sha256"])
            canonical = unique_by_hash.get(plan_hash)
            if canonical is None:
                unique_id = f"secure_plan_{len(unique_rows):03d}"
                row["unique_plan_id"] = unique_id
                row["candidate_labels"] = row["candidate"]
                row["duplicate_of"] = ""
                unique_row = dict(row)
                unique_by_hash[plan_hash] = unique_row
                unique_rows.append(unique_row)
            else:
                row["unique_plan_id"] = canonical["unique_plan_id"]
                row["candidate_labels"] = ""
                row["duplicate_of"] = canonical["candidate"]
                row["status"] = "ALIAS_DUPLICATE_NOT_RUN"
                canonical["candidate_labels"] = (
                    str(canonical["candidate_labels"]) + ";" + str(row["candidate"]))
        write_csv(output_dir / "secure_candidates.csv", list(secure_rows[0]), secure_rows)
        write_csv(output_dir / "unique_secure_plans.csv", [
            "unique_plan_id", "candidate_labels", "candidate", "head_plan",
            "config_json", "plan_sha256", "S_heads", "R_heads", "L_heads",
            "clear_accuracy_percent", "accuracy_drop_pp",
            "predicted_latency_ms", "predicted_communication_bytes",
            "accuracy_budget_pp", "secure_inference_run", "status",
        ], unique_rows)

        # Compatibility summary for the previous allocator report format.
        # Unlike the old three-policy table, this contains the five fixed
        # candidates that are actually eligible for the next secure-plan
        # review, and its accuracy columns are explicitly held-out results.
        selection_rows: list[dict[str, Any]] = []
        for row in secure_rows:
            selection_rows.append({
                "policy": row["candidate"],
                "S_heads": row["S_heads"],
                "R_heads": row["R_heads"],
                "L_heads": row["L_heads"],
                "clear_accuracy": row["clear_accuracy_percent"],
                "accuracy_drop": row["accuracy_drop_pp"],
                "predicted_secure_latency": row["predicted_latency_ms"],
                "predicted_comm": row["predicted_communication_bytes"],
                "accuracy_dataset": "fixed_heldout",
                "secure_inference_run": False,
            })
        write_csv(output_dir / "selection_report.csv", [
            "policy", "S_heads", "R_heads", "L_heads", "clear_accuracy",
            "accuracy_drop", "predicted_secure_latency", "predicted_comm",
            "accuracy_dataset", "secure_inference_run",
        ], selection_rows)

        default_document = dict(final_documents["Accuracy-only"])
        default_document["default_policy"] = "Accuracy-only"
        default_document["available_policy_configs"] = {
            policy: str((output_dir / f"{assignment_name(policy)}.json").resolve())
            for policy in final_documents
        }
        (output_dir / "head_assignment.json").write_text(
            json.dumps(default_document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

        final_drop_pass = {
            policy: float(final_validation[policy]["accuracy_drop"]) * 100.0
            <= args.final_budget_pp + 1e-9
            for policy in final_validation
        }
        primary_policies = ("All-Softmax", "Accuracy-only", "Balanced",
                            "Performance-first")
        primary_drop_pass = {
            policy: final_drop_pass[policy] for policy in primary_policies
        }
        report = {
            "format": "OURS_OFFLINE_ALLOCATOR_EXPERIMENT_REPORT_V2",
            "status": (
                "PASS" if all(primary_drop_pass.values()) and
                not missing_train and not missing_heldout and
                holdout_cost_meta.get("cost_model_gate_pass") is True and
                holdout_cost_meta.get("uses_heldout_in_fit") is False
                else ("INCOMPLETE_MISSING_REAL_MEASUREMENT"
                      if missing_train or missing_heldout
                      else "FAIL_FINAL_HELDOUT_BUDGET")),
            "secure_inference_run": False,
            "protocol_sources_modified": False,
            "model": str(args.model.resolve()),
            "model_sha256": base.sha256_file(args.model),
            "calibration_manifest": str(calibration_manifest.resolve()),
            "heldout_manifest": str(heldout_manifest.resolve()),
            "calibration_samples": len(calibration_samples),
            "heldout_samples": len(heldout_samples),
            "fixed_split_disjoint": True,
            "head_grouping_enabled": False,
            "relu_forced": False,
            "allocator_cost_source": str(args.allocator_costs.resolve()),
            "supplemental_cost_source": str(args.supplemental_costs.resolve()),
            "composition_probe_manifest": str(args.probe_manifest.resolve()),
            "composition_probe_manifest_sha256": base.sha256_file(args.probe_manifest),
            "cost_model": {
                "formula": (
                    "C_l/B_l = ridge([1,S/12,R/12,L/12,active_groups,SR,SL,RL,SRL,"
                    "(S/12)(R/12),(S/12)(L/12),(R/12)(L/12),layer_1..layer_11])"),
                "one_block_probe_scope": True,
                "uses_per_head_addition": False,
                "uses_heldout_in_fit": False,
                "search_uses_predictions_for_unmeasured_compositions": True,
                "training_probe_compositions": [
                    base.composition_name(value)
                    for value in sorted(search_cost_model.available_compositions -
                                        cost_model.available_compositions)],
                "latency_model": latency_cost_model.metadata(),
                "communication_model": communication_cost_model.metadata(),
                "holdout": holdout_cost_meta,
            },
            "missing_real_probe_compositions": {
                "train": [base.composition_name(value)
                           for value in sorted(missing_train)],
                "heldout": [base.composition_name(value)
                             for value in sorted(missing_heldout)],
            },
            "budgets_pp": args.budgets_pp,
            "final_budget_pp": args.final_budget_pp,
            "final_candidate_budget_pass": final_drop_pass,
            "primary_candidate_budget_pass": primary_drop_pass,
            "random_study": random_stats_summary,
            "selected_by_budget": {
                str(budget): selected_by_budget[budget] for budget in args.budgets_pp},
            "ablations": {
                "accuracy_only": "calibration risk only; no cost objective",
                "cost_only": "cost-model predicted latency only; ignores accuracy budget",
                "balanced": "equal normalized calibration risk and measured latency",
                "performance_first": "lowest cost-model predicted latency among calibration-eligible candidates",
            },
            "oracle": {
                "layer": args.oracle_layer,
                "head_subset": head_subset,
                "enumerated_assignments": 3 ** len(head_subset),
                "comparison_file": str((output_dir / "small_oracle_comparison.csv").resolve()),
            },
            "files": {
                "cost_model_holdout": str((output_dir / "cost_model_holdout.csv").resolve()),
                "cost_model_train": str((output_dir / "cost_model_train.csv").resolve()),
                "pareto_policies": str((output_dir / "pareto_policies.csv").resolve()),
                "allocator_baselines": str((output_dir / "allocator_baselines.csv").resolve()),
                "small_oracle_comparison": str((output_dir / "small_oracle_comparison.csv").resolve()),
                "secure_candidates": str((output_dir / "secure_candidates.csv").resolve()),
                "unique_secure_plans": str((output_dir / "unique_secure_plans.csv").resolve()),
                "selection_report": str((output_dir / "selection_report.csv").resolve()),
                "sensitivity_detail": str((output_dir / "head_sensitivity.csv").resolve()),
                "sensitivity_relu_heatmap": str((output_dir / "sensitivity_relu_heatmap.csv").resolve()),
                "sensitivity_linear_heatmap": str((output_dir / "sensitivity_linear_heatmap.csv").resolve()),
                "random_head_assignment_stats": str((output_dir / "random_head_assignment_stats.csv").resolve()),
                "default_assignment": str((output_dir / "head_assignment.json").resolve()),
            },
            "secure_next_step": {
                "status": "NOT_RUN",
                "required_candidates": [
                    row["unique_plan_id"] for row in unique_rows],
                "candidate_labels": [
                    row["candidate_labels"] for row in unique_rows],
                "note": "Only unique fixed plans may be scheduled for later secure validation; no secure process was launched by this experiment.",
            },
        }
        (output_dir / "allocator_experiment_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"OURS_OFFLINE_ALLOCATOR_EXPERIMENT status={report['status']}")
        print(f"OURS_OFFLINE_ALLOCATOR_EXPERIMENT_REPORT={output_dir / 'allocator_experiment_report.json'}")
    finally:
        close_pool(calibration_pool)
        close_pool(heldout_pool)


if __name__ == "__main__":
    main()
