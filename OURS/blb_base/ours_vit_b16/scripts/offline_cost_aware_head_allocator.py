#!/usr/bin/env python3
"""Offline, cost-aware assignment of heterogeneous ViT attention heads.

This module is deliberately an offline selector.  It runs the public clear
reference during calibration/validation, reads only directly measured
complete-attention assignment rows, and writes an immutable JSON assignment
plus the fixed ``OURS_VIT_HEAD_PLAN_V1`` materialization consumed by the
existing secure runtime.

There are two important fail-closed properties here:

* a cost point is available only when the benchmark contains that exact
  ``(layer, softmax_heads, relu_heads, linear_heads)`` complete assignment;
  no per-head scaling, interpolation, or path-order assumption is used;
* the search is dynamic-programming/Pareto based.  It never enumerates the
  3**144 whole-model assignments.  The only model-level approximation is the
  additive ranking proxy built from isolated full-ViT calibration candidates;
  every emitted policy is then run through the complete clear ViT on the
  validation set before it is accepted.

No secure inference is launched by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

import vit_reference as vr


PROTOCOLS = ("softmax_mpc", "relu_norm_mpc", "he_linear")
PROTOCOL_ORDER = {name: index for index, name in enumerate(PROTOCOLS)}
MODEL_LAYERS = 12
MODEL_HEADS = 12
MODEL_TOKENS = 197
PHYSICAL_TOKENS = 256

# These are the exact composition points present in the current complete
# assignment benchmark.  The loader derives the available set from the CSV;
# this tuple is only used to make the experiment's intended candidate grid
# explicit and deterministic.
BENCHMARK_COMPOSITIONS = (
    (12, 0, 0),
    (10, 0, 2),
    (8, 0, 4),
    (8, 4, 0),
    (6, 4, 2),
    (4, 8, 0),
    (4, 4, 4),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def composition_name(counts: tuple[int, int, int]) -> str:
    softmax, relu, linear = counts
    return f"{softmax}S{relu}R{linear}L"


def counts_for_plan(plan: tuple[tuple[tuple[str, float], ...], ...]
                    ) -> list[tuple[int, int, int]]:
    result = []
    for layer in plan:
        counts = tuple(sum(protocol == path for protocol, _ in layer)
                       for path in PROTOCOLS)
        result.append((int(counts[0]), int(counts[1]), int(counts[2])))
    return result


def stable_logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return (maximum + np.log(np.sum(np.exp(values - maximum), axis=axis,
                                    keepdims=True))).squeeze(axis)


def cross_entropy(logits: list[np.ndarray], labels: list[int]) -> float:
    if len(logits) != len(labels) or not logits:
        raise ValueError("cross entropy requires non-empty aligned observations")
    total = 0.0
    for row, label in zip(logits, labels):
        if label < 0 or label >= row.shape[-1]:
            raise ValueError(f"label outside classifier range: {label}")
        total += float(stable_logsumexp(np.asarray(row, dtype=np.float64)) -
                       np.asarray(row, dtype=np.float64)[label])
    return total / len(labels)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


@dataclass(frozen=True)
class CostObservation:
    layer: int
    counts: tuple[int, int, int]
    latency_ms: float
    communication_bytes: int
    source_run: str
    image_id: str
    binary_sha256: str
    plan_sha256: str


@dataclass(frozen=True)
class CostPrediction:
    layer: int
    counts: tuple[int, int, int]
    latency_ms: float
    communication_bytes: int
    observation_count: int
    source_runs: tuple[str, ...]


class MeasuredCompositionCostModel:
    """Sparse lookup model over complete measured attention assignments.

    For a layer ``l`` and measured composition ``c=(S,R,L)``:

        C_l(c) = median { measured complete-assignment latency_ms }
        B_l(c) = median { measured complete-assignment communication_bytes }

    A whole clear-model prediction is the sum of 12 *complete attention
    assignment units*, one per layer.  This is not a sum of per-head costs and
    is not presented as an end-to-end secure runtime measurement.
    """

    REQUIRED = {
        "layer", "version", "softmax_heads", "relu_heads", "linear_heads",
        "latency_ms", "communication_bytes", "measurement_scope", "cost_unit",
        "network_gbps", "rtt_ms", "backend", "gpu_topology", "image_id",
        "binary_sha256", "plan_sha256", "protocols_original",
        "physical_protocols", "original_order", "surrogate",
        "dynamic_selection", "source_run",
    }

    def __init__(self, path: Path, layers: int, heads: int) -> None:
        self.path = path
        self.layers = layers
        self.heads = heads
        self.observations: dict[tuple[int, tuple[int, int, int]],
                                list[CostObservation]] = {}
        self.cohort_metadata: tuple[str, str, str] | None = None
        self._load()

    def _load(self) -> None:
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = self.REQUIRED - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"complete assignment cost CSV missing columns: {sorted(missing)}")
            for line, row in enumerate(reader, 2):
                def fail(reason: str) -> None:
                    raise ValueError(f"invalid cost row line {line}: {reason}")

                try:
                    layer = int(row["layer"])
                    counts = (int(row["softmax_heads"]),
                              int(row["relu_heads"]),
                              int(row["linear_heads"]))
                except (TypeError, ValueError) as error:
                    fail(f"invalid layer/count: {error}")
                if not 0 <= layer < self.layers:
                    fail("layer outside model")
                if any(count < 0 for count in counts) or sum(counts) != self.heads:
                    fail("counts must be non-negative and sum to model heads")
                if row["measurement_scope"] != "complete_attention_assignment":
                    fail("row is not a complete attention assignment")
                if row["cost_unit"] != "direct_assignment_not_per_head":
                    fail("row is not a direct complete-assignment unit")
                if row["backend"].strip().lower() not in {
                        "phantom_gpu", "gpu_phantom", "phantom_device"}:
                    fail("benchmark backend is not Phantom GPU")
                try:
                    if not math.isclose(float(row["network_gbps"]), 1.0,
                                        rel_tol=0.0, abs_tol=1e-9):
                        fail("network is not 1 Gbps")
                    if not math.isclose(float(row["rtt_ms"]), 0.3,
                                        rel_tol=0.0, abs_tol=1e-9):
                        fail("RTT is not 0.3 ms")
                except ValueError as error:
                    fail(f"invalid network metadata: {error}")
                topology = row["gpu_topology"].strip()
                if topology not in {"separate_physical_gpus",
                                    "shared_physical_gpu"}:
                    fail("unknown GPU topology")
                if row["surrogate"].strip() != "0":
                    fail("surrogate cost is not permitted")
                if row["dynamic_selection"].strip() != "0":
                    fail("dynamic selection cost is not permitted")
                if not row["image_id"].startswith("sha256:"):
                    fail("image_id is not immutable")
                if (len(row["binary_sha256"]) != 64 or
                        len(row["plan_sha256"]) != 64):
                    fail("binary/plan hash is not a SHA-256")
                cohort = (topology, row["image_id"], row["binary_sha256"])
                if self.cohort_metadata is None:
                    self.cohort_metadata = cohort
                elif self.cohort_metadata != cohort:
                    fail(
                        "cost CSV mixes GPU topology, image, or binary versions: "
                        f"expected={self.cohort_metadata} actual={cohort}")
                original_protocols = row["protocols_original"].split(";")
                physical_protocols = row["physical_protocols"].split(";")
                if (len(original_protocols) != self.heads or
                        len(physical_protocols) != self.heads):
                    fail("protocol vectors do not cover all heads")
                actual_counts = tuple(original_protocols.count(path)
                                      for path in PROTOCOLS)
                if actual_counts != counts:
                    fail("composition counts disagree with protocols_original")
                if any(path not in PROTOCOLS for path in original_protocols +
                       physical_protocols):
                    fail("unknown protocol in assignment metadata")
                try:
                    latency = float(row["latency_ms"])
                    communication = int(row["communication_bytes"])
                except (TypeError, ValueError) as error:
                    fail(f"invalid numeric cost: {error}")
                if not math.isfinite(latency) or latency < 0.0:
                    fail("latency must be finite and non-negative")
                if communication < 0:
                    fail("communication must be non-negative")
                observation = CostObservation(
                    layer=layer,
                    counts=counts,
                    latency_ms=latency,
                    communication_bytes=communication,
                    source_run=row["source_run"],
                    image_id=row["image_id"],
                    binary_sha256=row["binary_sha256"],
                    plan_sha256=row["plan_sha256"],
                )
                self.observations.setdefault((layer, counts), []).append(
                    observation)
        if not self.observations:
            raise ValueError("complete assignment cost CSV has no usable rows")

    @property
    def available_compositions(self) -> set[tuple[int, int, int]]:
        return {counts for _, counts in self.observations}

    def missing_compositions(self) -> list[tuple[int, int, int]]:
        all_compositions = {
            (softmax, relu, MODEL_HEADS - softmax - relu)
            for softmax in range(MODEL_HEADS + 1)
            for relu in range(MODEL_HEADS - softmax + 1)
        }
        return sorted(all_compositions - self.available_compositions)

    def predict_layer(self, layer: int,
                      counts: tuple[int, int, int]) -> CostPrediction | None:
        rows = self.observations.get((layer, counts))
        if not rows:
            return None
        return CostPrediction(
            layer=layer,
            counts=counts,
            latency_ms=statistics.median(row.latency_ms for row in rows),
            communication_bytes=int(statistics.median(
                row.communication_bytes for row in rows)),
            observation_count=len(rows),
            source_runs=tuple(sorted({row.source_run for row in rows})),
        )

    def predict_plan(self, plan: tuple[tuple[tuple[str, float], ...], ...]
                     ) -> dict[str, Any] | None:
        layer_counts = counts_for_plan(plan)
        layer_costs: list[CostPrediction] = []
        for layer, counts in enumerate(layer_counts):
            prediction = self.predict_layer(layer, counts)
            if prediction is None:
                return None
            layer_costs.append(prediction)
        return {
            "latency_ms": sum(row.latency_ms for row in layer_costs),
            "communication_bytes": sum(row.communication_bytes
                                        for row in layer_costs),
            "cost_unit": "sum_of_12_direct_complete_attention_assignment_units",
            "not_end_to_end_secure_runtime": True,
            "not_per_head_extrapolation": True,
            "layers": [cost_prediction_json(row) for row in layer_costs],
        }

    def validation_rows(self) -> list[dict[str, Any]]:
        """Make a consistency table against complete measured source runs.

        Most compositions have one complete 12-layer source run.  The 4S/4R/4L
        point also has additional layer-0 repeats, which are retained in the
        model median but are not incorrectly treated as complete model runs in
        this table.
        """
        result = []
        for counts in sorted(self.available_compositions):
            by_source: dict[str, list[CostObservation]] = {}
            for (layer, row_counts), rows in self.observations.items():
                if row_counts != counts:
                    continue
                for row in rows:
                    by_source.setdefault(row.source_run, []).append(row)
            complete = [rows for rows in by_source.values()
                        if {row.layer for row in rows} == set(range(self.layers))]
            if not complete:
                # This should only happen if a future benchmark is supplied
                # with partial rows.  It is a missing validation point, not a
                # value to synthesize.
                continue
            prediction = sum(self.predict_layer(layer, counts).latency_ms
                              for layer in range(self.layers)
                              if self.predict_layer(layer, counts) is not None)
            # Keep the validation comparable across compositions: when more
            # than one complete run exists, compare to its median total.
            measured_totals = [sum(row.latency_ms for row in rows)
                               for rows in complete]
            measured = statistics.median(measured_totals)
            error = ((prediction - measured) / measured * 100.0
                     if measured else 0.0)
            result.append({
                "composition": composition_name(counts),
                "predicted_latency": prediction,
                "measured_latency": measured,
                "error_percent": error,
                "complete_source_runs": len(complete),
                "validation_scope": "complete_source_run_consistency",
            })
        return result


def cost_prediction_json(prediction: CostPrediction) -> dict[str, Any]:
    return {
        "layer": prediction.layer,
        "composition": composition_name(prediction.counts),
        "S_heads": prediction.counts[0],
        "R_heads": prediction.counts[1],
        "L_heads": prediction.counts[2],
        "predicted_latency_ms": prediction.latency_ms,
        "predicted_comm_bytes": prediction.communication_bytes,
        "measured_observation_count": prediction.observation_count,
        "source_runs": list(prediction.source_runs),
        "cost_unit": ("direct_complete_attention_assignment"
                       if prediction.observation_count > 0
                       else "composition_model_prediction"),
    }


def read_manifest(path: Path, image_root: Path, limit: int) -> list[Sample]:
    rows: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.reader(handle))
    for index, row in enumerate(raw_rows):
        if not row or all(not cell.strip() for cell in row):
            continue
        if index == 0 and row[0].strip().lower() in {
                "input", "image", "image_path", "path"}:
            continue
        if len(row) != 2:
            raise ValueError(f"manifest row {index + 1} must have path,label")
        path_value = Path(row[0].strip())
        resolved = (path_value if path_value.is_absolute()
                    else image_root / path_value).resolve()
        try:
            label = int(row[1])
        except ValueError as error:
            raise ValueError(f"manifest row {index + 1} label is not integer") from error
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        rows.append(Sample(resolved, label))
    if limit <= 0:
        raise ValueError("manifest limit must be positive")
    rows = rows[:limit]
    if len(rows) != limit:
        raise ValueError(f"manifest has {len(rows)} usable rows, expected {limit}")
    return rows


def load_preprocess(path: Path, model: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prep = payload.get("preprocessing", {})
    size = prep.get("resize", prep.get("size", model["image_size"]))
    if isinstance(size, dict):
        size = size.get("height", size.get("shortest_edge"))
    mean = prep.get("mean", prep.get("image_mean", [0.485, 0.456, 0.406]))
    std = prep.get("std", prep.get("image_std", [0.229, 0.224, 0.225]))
    if int(size) != int(model["image_size"]):
        raise ValueError("preprocessing size disagrees with model manifest")
    if len(mean) != 3 or len(std) != 3 or any(float(value) <= 0 for value in std):
        raise ValueError("preprocessing mean/std is invalid")
    return {"size": int(size), "mean": np.asarray(mean, dtype=np.float64),
            "std": np.asarray(std, dtype=np.float64)}


def load_pixels(path: Path, preprocess: dict[str, Any]) -> np.ndarray:
    if path.suffix.lower() == ".txt":
        return vr.load_input(path)
    image = Image.open(path).convert("RGB").resize(
        (preprocess["size"], preprocess["size"]), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float64) / 255.0
    return ((array - preprocess["mean"]) / preprocess["std"]).transpose(2, 0, 1)


def load_pixels_for_samples(samples: list[Sample],
                            preprocess: dict[str, Any]) -> list[np.ndarray]:
    return [load_pixels(sample.path, preprocess) for sample in samples]


def attention_observation(records: dict[str, dict[str, Any]], layer: int,
                          head: int, fixed: bool = False
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = records[f"block_{layer}.qk"]["value"][head, :MODEL_TOKENS,
                                                     :MODEL_TOKENS]
    probability = records[f"block_{layer}.softmax"]["value"][head,
                                                              :MODEL_TOKENS,
                                                              :MODEL_TOKENS]
    value = records[f"block_{layer}.v"]["value"][:MODEL_TOKENS]
    head_dim = value.shape[-1] // MODEL_HEADS
    values = value.reshape(MODEL_TOKENS, MODEL_HEADS, head_dim)[:, head, :]
    if fixed:
        target_raw = vr.matmul_integer(vr.round_int(probability, vr.ACT_SCALE),
                                        vr.round_int(values, 11))
        target = (vr._floor_shift(target_raw.astype(np.int64), 11).astype(
            np.float64) / float(1 << vr.ACT_SCALE))
    else:
        target = probability @ values
    return score.copy(), values.copy(), target.copy()


def fit_linear_alphas(records: list[dict[str, dict[str, Any]]],
                      layers: int, heads: int) -> list[list[float]]:
    alphas = [[0.0 for _ in range(heads)] for _ in range(layers)]
    for layer in range(layers):
        for head in range(heads):
            numerator = 0.0
            denominator = 0.0
            for sample_records in records:
                score, values, target = attention_observation(
                    sample_records, layer, head, fixed=False)
                product = score @ values
                numerator += float(np.sum(product * target, dtype=np.float64))
                denominator += float(np.sum(product * product, dtype=np.float64))
            if denominator <= 0.0:
                raise ValueError(f"cannot fit alpha for layer={layer},head={head}")
            alpha = numerator / denominator
            if not math.isfinite(alpha):
                raise ValueError(f"non-finite alpha for layer={layer},head={head}")
            alphas[layer][head] = alpha
    return alphas


def compare_logits(reference: list[np.ndarray], candidate: list[np.ndarray],
                   labels: list[int]) -> dict[str, Any]:
    if len(reference) != len(candidate) or len(reference) != len(labels):
        raise ValueError("logit observations are not aligned")
    differences = [np.asarray(actual) - np.asarray(expected)
                   for expected, actual in zip(reference, candidate)]
    squared = sum(float(np.sum(delta * delta, dtype=np.float64))
                  for delta in differences)
    ref_squared = sum(float(np.sum(np.asarray(row) * np.asarray(row),
                                   dtype=np.float64)) for row in reference)
    elements = sum(delta.size for delta in differences)
    reference_l2 = math.sqrt(ref_squared)
    relative_l2 = (math.sqrt(squared) / reference_l2
                   if reference_l2 else 0.0)
    baseline_top1 = [int(np.argmax(row)) for row in reference]
    candidate_top1 = [int(np.argmax(row)) for row in candidate]
    baseline_accuracy = sum(top == label for top, label in zip(
        baseline_top1, labels)) / len(labels)
    candidate_accuracy = sum(top == label for top, label in zip(
        candidate_top1, labels)) / len(labels)
    reference_loss = cross_entropy(reference, labels)
    candidate_loss = cross_entropy(candidate, labels)
    loss_delta = candidate_loss - reference_loss
    max_abs = max((float(np.max(np.abs(delta), initial=0.0))
                   for delta in differences), default=0.0)
    mse = squared / elements if elements else 0.0
    return {
        "validation_loss": candidate_loss,
        "baseline_validation_loss": reference_loss,
        "loss_delta": loss_delta,
        "mse": mse,
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "top1_agreement": sum(a == b for a, b in zip(
            baseline_top1, candidate_top1)) / len(labels),
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_drop": baseline_accuracy - candidate_accuracy,
        "elements": elements,
    }


# Worker state is populated through fork.  It avoids serializing the large
# model for every isolated head candidate while keeping all selection work
# local and offline.
_CANDIDATE_MODEL: dict[str, Any] | None = None
_CANDIDATE_PIXELS: list[np.ndarray] = []
_CANDIDATE_LABELS: list[int] = []
_CANDIDATE_LOGIT_WEIGHT = 0.25
_CANDIDATE_BATCH_SIZE = 1
_VALIDATE_BATCH_SIZE = 1


def candidate_worker_init(model: dict[str, Any], pixels: list[np.ndarray],
                          labels: list[int], validation_pixels: list[np.ndarray],
                          validation_labels: list[int], logit_weight: float,
                          batch_size: int = 1) -> None:
    global _CANDIDATE_MODEL, _CANDIDATE_PIXELS, _CANDIDATE_LABELS
    global _VALIDATE_MODEL, _VALIDATE_PIXELS, _VALIDATE_LABELS
    global _CANDIDATE_LOGIT_WEIGHT, _CANDIDATE_BATCH_SIZE
    _CANDIDATE_MODEL = model
    _CANDIDATE_PIXELS = pixels
    _CANDIDATE_LABELS = labels
    _VALIDATE_MODEL = model
    _VALIDATE_PIXELS = validation_pixels
    _VALIDATE_LABELS = validation_labels
    _VALIDATE_BASELINE.clear()
    _CANDIDATE_LOGIT_WEIGHT = logit_weight
    if batch_size <= 0:
        raise ValueError("candidate batch size must be positive")
    _CANDIDATE_BATCH_SIZE = int(batch_size)


def candidate_worker(task: tuple[int, int, str, float, list[np.ndarray]]) -> dict[str, Any]:
    layer, head, protocol, alpha, baseline = task
    if _CANDIDATE_MODEL is None:
        raise RuntimeError("candidate worker is not initialized")
    plan = [[("softmax_mpc", 1.0) for _ in range(MODEL_HEADS)]
            for _ in range(MODEL_LAYERS)]
    plan[layer][head] = (protocol, alpha)
    if _CANDIDATE_BATCH_SIZE <= 1:
        logits = [vr.Graph(_CANDIDATE_MODEL, fixed=False,
                           head_plan=plan, record_values=False).run(pixels)[0]
                  for pixels in _CANDIDATE_PIXELS]
    else:
        logits = []
        for start in range(0, len(_CANDIDATE_PIXELS), _CANDIDATE_BATCH_SIZE):
            batch = np.stack(_CANDIDATE_PIXELS[
                start:start + _CANDIDATE_BATCH_SIZE])
            batch_logits, _ = vr.Graph(
                _CANDIDATE_MODEL, fixed=False, head_plan=plan,
                record_values=False).run_batch(batch)
            logits.extend(batch_logits)
    metrics = compare_logits(baseline, logits, _CANDIDATE_LABELS)
    # Loss is the primary search signal; relative logit deviation provides a
    # stable tie-breaker when the tiny calibration set has unchanged top-1.
    metrics["proxy_score"] = (max(0.0, float(metrics["loss_delta"])) +
                               _CANDIDATE_LOGIT_WEIGHT *
                               float(metrics["relative_l2"]))
    return {"layer": layer, "head": head, "protocol": protocol,
            "alpha": alpha, **metrics}


def run_isolated_candidates(model: dict[str, Any], pixels: list[np.ndarray],
                            labels: list[int], baseline: list[np.ndarray],
                            alphas: list[list[float]], workers: int,
                            logit_weight: float,
                            pool: Any | None = None,
                            batch_size: int = 1,
                            progress_every: int = 0) -> list[dict[str, Any]]:
    tasks = [(layer, head, protocol,
              alphas[layer][head] if protocol == "he_linear" else 1.0,
              baseline)
             for layer in range(MODEL_LAYERS)
             for head in range(MODEL_HEADS)
             for protocol in ("relu_norm_mpc", "he_linear")]
    if workers <= 1:
        candidate_worker_init(model, pixels, labels, [], [], logit_weight,
                              batch_size)
        result = []
        for index, task in enumerate(tasks, 1):
            result.append(candidate_worker(task))
            if progress_every > 0 and index % progress_every == 0:
                print(f"OURS_CLEAR_ISOLATED progress={index}/{len(tasks)}",
                      flush=True)
        return result
    if pool is not None:
        result = []
        for index, row in enumerate(
                pool.imap_unordered(candidate_worker, tasks, chunksize=1), 1):
            result.append(row)
            if progress_every > 0 and index % progress_every == 0:
                print(f"OURS_CLEAR_ISOLATED progress={index}/{len(tasks)}",
                      flush=True)
        return result
    # This fallback is safe for library callers that already performed NumPy
    # work before invoking the function.  The CLI creates a fork pool earlier
    # (before its first Graph call) to avoid the expensive model pickling of
    # spawn while retaining the same BLAS safety.
    context = mp.get_context("spawn")
    with context.Pool(workers, initializer=candidate_worker_init,
                      initargs=(model, pixels, labels, [], [], logit_weight,
                                batch_size)) as worker_pool:
        result = []
        for index, row in enumerate(
                worker_pool.imap_unordered(candidate_worker, tasks, chunksize=1), 1):
            result.append(row)
            if progress_every > 0 and index % progress_every == 0:
                print(f"OURS_CLEAR_ISOLATED progress={index}/{len(tasks)}",
                      flush=True)
        return result


def load_candidate_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[int, int, str],
                                                                  dict[str, Any]]:
    result = {}
    for row in rows:
        key = (int(row["layer"]), int(row["head"]), str(row["protocol"]))
        if key in result:
            raise ValueError(f"duplicate isolated candidate {key}")
        result[key] = row
    return result


def alphas_from_isolated(rows: Iterable[dict[str, Any]], layers: int,
                         heads: int) -> list[list[float]]:
    result = [[math.nan for _ in range(heads)] for _ in range(layers)]
    for row in rows:
        if row["protocol"] != "he_linear":
            continue
        layer = int(row["layer"])
        head = int(row["head"])
        result[layer][head] = float(row["alpha"])
    if any(not math.isfinite(value) for layer in result for value in layer):
        raise ValueError("isolated candidate cache does not contain all Linear alphas")
    return result


@dataclass(frozen=True)
class LayerOption:
    layer: int
    counts: tuple[int, int, int]
    paths: tuple[tuple[str, float], ...]
    proxy_score: float
    proxy_loss_delta: float
    proxy_relative_l2: float
    cost: CostPrediction


def protocol_score(row: dict[str, Any] | None, protocol: str) -> float:
    if protocol == "softmax_mpc":
        return 0.0
    if row is None:
        raise ValueError(f"missing isolated candidate for {protocol}")
    return float(row["proxy_score"])


def min_assignment_for_counts(layer: int, counts: tuple[int, int, int],
                              candidates: dict[tuple[int, int, str],
                                               dict[str, Any]],
                              alphas: list[list[float]]) -> tuple[tuple[tuple[str, float], ...], float]:
    """Head-wise DP for one exact S/R/L composition.

    The state is the number of each path used after processing a prefix of the
    12 heads.  This is O(H * S * R * L * 3), not an enumeration of 3**H
    assignments, and it returns the concrete original-head assignment.
    """
    target_s, target_r, target_l = counts
    states: dict[tuple[int, int, int], tuple[float, tuple[str, ...]]] = {
        (0, 0, 0): (0.0, ())}
    for head in range(MODEL_HEADS):
        next_states: dict[tuple[int, int, int], tuple[float, tuple[str, ...]]] = {}
        for used, (score, paths) in states.items():
            for protocol, increment in (("softmax_mpc", (1, 0, 0)),
                                         ("relu_norm_mpc", (0, 1, 0)),
                                         ("he_linear", (0, 0, 1))):
                candidate_counts = tuple(used[index] + increment[index]
                                         for index in range(3))
                if any(candidate_counts[index] > counts[index]
                       for index in range(3)):
                    continue
                remaining_heads = MODEL_HEADS - head - 1
                if sum(counts[index] - candidate_counts[index]
                       for index in range(3)) != remaining_heads:
                    continue
                row = candidates.get((layer, head, protocol))
                next_score = score + protocol_score(row, protocol)
                next_paths = paths + (protocol,)
                old = next_states.get(candidate_counts)
                if old is None or (next_score, next_paths) < old:
                    next_states[candidate_counts] = (next_score, next_paths)
        states = next_states
    selected = states.get(counts)
    if selected is None:
        raise ValueError(f"no DP assignment for layer={layer}, counts={counts}")
    score, path_names = selected
    return tuple((protocol,
                  alphas[layer][head] if protocol == "he_linear" else 1.0)
                 for head, protocol in enumerate(path_names)), score


def option_from_paths(layer: int, counts: tuple[int, int, int],
                      paths: tuple[tuple[str, float], ...], score: float,
                      candidates: dict[tuple[int, int, str], dict[str, Any]],
                      cost: CostPrediction) -> LayerOption:
    loss = 0.0
    relative = 0.0
    for head, (protocol, _) in enumerate(paths):
        if protocol == "softmax_mpc":
            continue
        row = candidates[(layer, head, protocol)]
        loss += max(0.0, float(row["loss_delta"]))
        relative += float(row["relative_l2"])
    return LayerOption(layer, counts, paths, score, loss, relative, cost)


def dominates(left: tuple[float, float, float],
              right: tuple[float, float, float]) -> bool:
    return (all(a <= b + 1e-12 for a, b in zip(left, right)) and
            any(a < b - 1e-12 for a, b in zip(left, right)))


def pareto_layer_options(options: list[LayerOption]) -> list[LayerOption]:
    result = []
    for option in options:
        point = (option.proxy_score, option.cost.latency_ms,
                 float(option.cost.communication_bytes))
        if any(dominates((other.proxy_score, other.cost.latency_ms,
                          float(other.cost.communication_bytes)), point)
               for other in options if other is not option):
            continue
        result.append(option)
    return sorted(result, key=lambda item: (item.proxy_score,
                                             item.cost.latency_ms,
                                             item.counts))


@dataclass(frozen=True)
class ModelState:
    proxy_score: float
    proxy_loss_delta: float
    proxy_relative_l2: float
    latency_ms: float
    communication_bytes: int
    plan: tuple[tuple[tuple[str, float], ...], ...]


def model_point(state: ModelState) -> tuple[float, float, float]:
    return (state.proxy_score, state.latency_ms,
            float(state.communication_bytes))


def exact_pareto_states(states: list[ModelState], max_frontier: int
                        ) -> tuple[list[ModelState], bool]:
    """Pareto prune a 3D frontier, then deterministically cap its size.

    The first pass uses a Fenwick minimum query after sorting by proxy score;
    therefore a state is removed only if an already processed state is no
    worse in all three search dimensions.  A cap is a practical guard for a
    sparse benchmark grid, not a cost extrapolation.  The report records if
    that cap was used.
    """
    if max_frontier <= 0:
        raise ValueError("max_frontier must be positive")
    dedup: dict[tuple[float, float, int], ModelState] = {}
    for state in states:
        key = (round(state.proxy_score, 12), round(state.latency_ms, 6),
               state.communication_bytes)
        old = dedup.get(key)
        if old is None or state.plan < old.plan:
            dedup[key] = state
    ordered = sorted(dedup.values(), key=lambda item: model_point(item))
    comm_values = sorted({state.communication_bytes for state in ordered})
    tree = [math.inf] * (len(comm_values) + 1)

    def query(index: int) -> float:
        result = math.inf
        while index:
            result = min(result, tree[index])
            index -= index & -index
        return result

    def update(index: int, value: float) -> None:
        while index < len(tree):
            tree[index] = min(tree[index], value)
            index += index & -index

    kept: list[ModelState] = []
    for state in ordered:
        index = (math.inf if not comm_values else
                 next_power_index(comm_values, state.communication_bytes))
        if query(index) <= state.latency_ms + 1e-9:
            continue
        kept.append(state)
        update(index, state.latency_ms)
    truncated = False
    if len(kept) > max_frontier:
        truncated = True
        ranked = sorted(kept, key=model_point)
        indexes = {0, len(ranked) - 1}
        for index in range(max_frontier - 2):
            indexes.add(round(index * (len(ranked) - 1) /
                             max(1, max_frontier - 3)))
        kept = [ranked[index] for index in sorted(indexes)]
        # The index set can be one larger because of rounding; trim by the
        # stable Pareto ordering without changing the extremes.
        kept = kept[:max_frontier]
    return kept, truncated


def next_power_index(values: list[int], value: int) -> int:
    # values is sorted and value is guaranteed to be present.
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < value:
            low = middle + 1
        else:
            high = middle
    return low + 1


def combine_model_frontier(layer_options: list[list[LayerOption]],
                           max_frontier: int) -> tuple[list[ModelState], bool]:
    states = [ModelState(0.0, 0.0, 0.0, 0.0, 0, ())]
    truncated_any = False
    for options in layer_options:
        expanded = []
        for state in states:
            for option in options:
                expanded.append(ModelState(
                    proxy_score=state.proxy_score + option.proxy_score,
                    proxy_loss_delta=state.proxy_loss_delta + option.proxy_loss_delta,
                    proxy_relative_l2=state.proxy_relative_l2 + option.proxy_relative_l2,
                    latency_ms=state.latency_ms + option.cost.latency_ms,
                    communication_bytes=(state.communication_bytes +
                                         option.cost.communication_bytes),
                    plan=state.plan + (option.paths,),
                ))
        states, truncated = exact_pareto_states(expanded, max_frontier)
        truncated_any = truncated_any or truncated
    return states, truncated_any


def uniform_composition_state(layer_options: list[list[LayerOption]],
                               counts: tuple[int, int, int]) -> ModelState | None:
    chosen = []
    for options in layer_options:
        matches = [option for option in options if option.counts == counts]
        if not matches:
            return None
        chosen.append(min(matches, key=lambda option: (
            option.proxy_score, option.paths)))
    return ModelState(
        proxy_score=sum(option.proxy_score for option in chosen),
        proxy_loss_delta=sum(option.proxy_loss_delta for option in chosen),
        proxy_relative_l2=sum(option.proxy_relative_l2 for option in chosen),
        latency_ms=sum(option.cost.latency_ms for option in chosen),
        communication_bytes=sum(option.cost.communication_bytes
                                for option in chosen),
        plan=tuple(option.paths for option in chosen),
    )


def plan_key(plan: tuple[tuple[tuple[str, float], ...], ...]) -> str:
    payload = []
    for layer in plan:
        payload.append([
            {"protocol": protocol, "alpha": float(alpha)}
            for protocol, alpha in layer])
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def candidate_pool(frontier: list[ModelState],
                   layer_options: list[list[LayerOption]],
                   max_candidates: int,
                   composition_grid: tuple[tuple[int, int, int], ...] =
                   BENCHMARK_COMPOSITIONS) -> list[ModelState]:
    states = list(frontier)
    states.extend(state for state in (
        uniform_composition_state(layer_options, counts)
        for counts in composition_grid)
        if state is not None)
    all_softmax = uniform_composition_state(layer_options, (12, 0, 0))
    if all_softmax is None:
        raise ValueError("benchmark/candidate grid lacks all-softmax")
    states.append(all_softmax)
    dedup: dict[str, ModelState] = {}
    for state in states:
        dedup.setdefault(plan_key(state.plan), state)
    states = list(dedup.values())
    if len(states) <= max_candidates:
        return sorted(states, key=model_point)
    selected: dict[str, ModelState] = {}
    for state in (min(states, key=lambda item: item.proxy_score),
                  min(states, key=lambda item: item.latency_ms),
                  min(states, key=lambda item: item.communication_bytes),
                  all_softmax):
        selected[plan_key(state.plan)] = state
    ranked = sorted(states, key=model_point)
    for index in range(max_candidates):
        ranked_index = round(index * (len(ranked) - 1) /
                             max(1, max_candidates - 1))
        state = ranked[ranked_index]
        selected.setdefault(plan_key(state.plan), state)
    return sorted(selected.values(), key=model_point)[:max_candidates]


_VALIDATE_MODEL: dict[str, Any] | None = None
_VALIDATE_PIXELS: list[np.ndarray] = []
_VALIDATE_LABELS: list[int] = []


def validate_worker_init(model: dict[str, Any], pixels: list[np.ndarray],
                         labels: list[int], batch_size: int = 1) -> None:
    global _VALIDATE_MODEL, _VALIDATE_PIXELS, _VALIDATE_LABELS
    global _VALIDATE_BATCH_SIZE
    _VALIDATE_MODEL = model
    _VALIDATE_PIXELS = pixels
    _VALIDATE_LABELS = labels
    if batch_size <= 0:
        raise ValueError("validation batch size must be positive")
    _VALIDATE_BATCH_SIZE = int(batch_size)


def validate_worker(task: tuple[str, tuple[tuple[tuple[str, float], ...], ...],
                                 list[np.ndarray]]
                    ) -> dict[str, Any]:
    name, plan, baseline = task
    if _VALIDATE_MODEL is None:
        raise RuntimeError("validation worker is not initialized")
    if _VALIDATE_BATCH_SIZE <= 1:
        logits = [vr.Graph(_VALIDATE_MODEL, fixed=False, head_plan=plan,
                           record_values=False).run(pixels)[0]
                  for pixels in _VALIDATE_PIXELS]
    else:
        logits = []
        for start in range(0, len(_VALIDATE_PIXELS), _VALIDATE_BATCH_SIZE):
            batch = np.stack(_VALIDATE_PIXELS[
                start:start + _VALIDATE_BATCH_SIZE])
            batch_logits, _ = vr.Graph(
                _VALIDATE_MODEL, fixed=False, head_plan=plan,
                record_values=False).run_batch(batch)
            logits.extend(batch_logits)
    return {"candidate": name, **compare_logits(
        baseline, logits, _VALIDATE_LABELS)}


_VALIDATE_BASELINE: list[np.ndarray] = []


def run_validation(model: dict[str, Any], pixels: list[np.ndarray],
                   labels: list[int], baseline: list[np.ndarray],
                   states: dict[str, ModelState], workers: int
                   , pool: Any | None = None,
                   batch_size: int = 1,
                   progress_every: int = 0) -> list[dict[str, Any]]:
    tasks = [(name, state.plan, baseline) for name, state in states.items()]
    if workers <= 1:
        validate_worker_init(model, pixels, labels, batch_size)
        result = []
        for index, task in enumerate(tasks, 1):
            result.append(validate_worker(task))
            if progress_every > 0 and index % progress_every == 0:
                print(f"OURS_CLEAR_VALIDATE progress={index}/{len(tasks)}",
                      flush=True)
        return result
    if pool is not None:
        result = []
        for index, row in enumerate(
                pool.imap_unordered(validate_worker, tasks, chunksize=1), 1):
            result.append(row)
            if progress_every > 0 and index % progress_every == 0:
                print(f"OURS_CLEAR_VALIDATE progress={index}/{len(tasks)}",
                      flush=True)
        return result
    context = mp.get_context("spawn")
    with context.Pool(workers, initializer=validate_worker_init,
                      initargs=(model, pixels, labels, batch_size)) as worker_pool:
        result = []
        for index, row in enumerate(
                worker_pool.imap_unordered(validate_worker, tasks, chunksize=1), 1):
            result.append(row)
            if progress_every > 0 and index % progress_every == 0:
                print(f"OURS_CLEAR_VALIDATE progress={index}/{len(tasks)}",
                      flush=True)
        return result


def physical_order(plan_layer: tuple[tuple[str, float], ...]) -> list[int]:
    return sorted(range(len(plan_layer)), key=lambda head: (
        PROTOCOL_ORDER[plan_layer[head][0]], head))


def write_runtime_plan(path: Path,
                       plan: tuple[tuple[tuple[str, float], ...], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("OURS_VIT_HEAD_PLAN_V1\n")
        handle.write("layers=12,heads=12,logical_tokens=197,physical_tokens=256\n")
        handle.write("layer,physical_head,original_head,protocol,alpha\n")
        for layer, entries in enumerate(plan):
            order = physical_order(entries)
            for physical, original in enumerate(order):
                protocol, alpha = entries[original]
                handle.write(f"{layer},{physical},{original},{protocol},"
                             f"{alpha:.17g}\n")


def validate_runtime_plan(plan: tuple[tuple[tuple[str, float], ...], ...]) -> None:
    if len(plan) != MODEL_LAYERS or any(len(layer) != MODEL_HEADS for layer in plan):
        raise ValueError("assignment does not cover 12x12 heads")
    for layer, entries in enumerate(plan):
        order = physical_order(entries)
        if sorted(order) != list(range(MODEL_HEADS)):
            raise ValueError(f"layer {layer} physical order is not a permutation")
        seen: set[str] = set()
        previous = None
        for original in order:
            protocol, alpha = entries[original]
            if protocol not in PROTOCOLS or not math.isfinite(alpha):
                raise ValueError(f"invalid runtime entry layer={layer},head={original}")
            if protocol != "he_linear" and abs(alpha - 1.0) > 1e-12:
                raise ValueError("MPC path alpha must equal 1")
            if previous != protocol:
                if protocol in seen:
                    raise ValueError(f"protocol group is non-contiguous in layer {layer}")
                seen.add(protocol)
                previous = protocol


def output_assignment(path: Path, policy: str, budget_pp: float,
                      state: ModelState, validation: dict[str, Any],
                      cost: dict[str, Any], alphas: list[list[float]],
                      model_path: Path, model_manifest: Path,
                      calibration_manifest: Path, validation_manifest: Path,
                      cost_path: Path, runtime_plan: Path,
                      missing_compositions: list[tuple[int, int, int]],
                      search_meta: dict[str, Any]) -> dict[str, Any]:
    plan = state.plan
    validate_runtime_plan(plan)
    rows = []
    for layer, entries in enumerate(plan):
        order = physical_order(entries)
        physical_by_original = {original: physical
                                for physical, original in enumerate(order)}
        for head, (protocol, alpha) in enumerate(entries):
            rows.append({
                "layer": layer,
                "head": head,
                "selected_path": protocol,
                "alpha_h": float(alpha),
                "physical_head": physical_by_original[head],
            })
    counts = counts_for_plan(plan)
    # Accuracy is evaluated on a finite labelled set; serialize the percentage
    # point value after rounding so a mathematically exact one-sample drop is
    # not printed as 2.0000000000000018.
    actual_drop_pp = round(float(validation["accuracy_drop"] * 100.0), 12)
    budget_pass = actual_drop_pp <= budget_pp + 1e-9
    document = {
        "format": "OURS_OFFLINE_COST_AWARE_HEAD_ASSIGNMENT_V1",
        "status": "PASS" if budget_pass else "FAIL_ACCURACY_BUDGET",
        "policy": policy,
        "accuracy_budget_pp": budget_pp,
        "model": "ViT-B/16",
        "model_artifact": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "model_manifest": str(model_manifest.resolve()),
        "calibration": {
            "manifest": str(calibration_manifest.resolve()),
            "samples": search_meta["calibration_samples"],
            "internal_metric": "validation_cross_entropy_delta_primary_plus_relative_logit_l2",
            "linear_alpha_fit": "least_squares((QK_score @ V), (Softmax_weight @ V))",
            "training_or_backpropagation": False,
        },
        "validation": {
            "manifest": str(validation_manifest.resolve()),
            "samples": search_meta["validation_samples"],
            "clear_accuracy_percent": validation["candidate_accuracy"] * 100.0,
            "baseline_clear_accuracy_percent": validation["baseline_accuracy"] * 100.0,
            "accuracy_drop_pp": actual_drop_pp,
            "validation_loss": validation["validation_loss"],
            "loss_delta": validation["loss_delta"],
            "top1_agreement": validation["top1_agreement"],
            "logit_relative_l2": validation["relative_l2"],
            "full_plaintext_vit": True,
        },
        "assignment": rows,
        "path_counts": {
            "softmax_mpc": sum(row["selected_path"] == "softmax_mpc" for row in rows),
            "relu_norm_mpc": sum(row["selected_path"] == "relu_norm_mpc" for row in rows),
            "he_linear": sum(row["selected_path"] == "he_linear" for row in rows),
        },
        "layer_compositions": [
            {"layer": layer, "S_heads": row[0], "R_heads": row[1],
             "L_heads": row[2], "composition": composition_name(row)}
            for layer, row in enumerate(counts)
        ],
        "predicted_cost": {
            **cost,
            "latency_unit": "ms",
            "communication_unit": "bytes",
            "source": str(cost_path.resolve()),
            "source_sha256": sha256_file(cost_path),
            "is_final_secure_runtime": False,
        },
        "search": {
            **search_meta,
            "selected_proxy_score": state.proxy_score,
            "selected_proxy_loss_delta": state.proxy_loss_delta,
            "selected_proxy_relative_l2": state.proxy_relative_l2,
            "selected_plan_sha256": sha256_file(runtime_plan),
            "selected_runtime_plan": str(runtime_plan.resolve()),
        },
        "missing_cost_compositions": [composition_name(row)
                                       for row in missing_compositions],
        "online_invariants": {
            "fixed_public_assignment": True,
            "dynamic_selection": False,
            "private_dynamic_branch": False,
            "mpc_reference": 0,
            "client_decrypt_reencrypt": 0,
            "softmax_path": "softmax_mpc",
            "relu_path": "relu_norm_mpc",
            "linear_path": "native_ckks_he_linear",
            "plaintext_search_at_inference": False,
            "secure_inference_run": False,
        },
        "runtime": {
            "version": search_meta.get("runtime_version", "HA"),
            "head_plan": str(runtime_plan.resolve()),
            "head_plan_format": "OURS_VIT_HEAD_PLAN_V1",
        },
    }
    from version_matrix import plan_binding
    from experiment_contract import CONTRACT, CONTRACT_SHA256
    document["experiment_contract"] = CONTRACT
    document["experiment_contract_sha256"] = CONTRACT_SHA256
    document["runtime"].update(plan_binding(runtime_plan))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return document


def select_policy(policy: str, budget_pp: float,
                  states: dict[str, ModelState],
                  validations: dict[str, dict[str, Any]]) -> str:
    eligible = [name for name, row in validations.items()
                if float(row["accuracy_drop"]) * 100.0 <= budget_pp + 1e-9]
    if not eligible:
        raise ValueError(f"candidate pool has no eligible {policy} plan")
    if policy == "accuracy_first":
        return min(eligible, key=lambda name: (
            max(0.0, float(validations[name]["loss_delta"])),
            max(0.0, float(validations[name]["accuracy_drop"])),
            states[name].proxy_score,
            states[name].latency_ms,
            name))
    if policy == "performance_first":
        return min(eligible, key=lambda name: (
            states[name].latency_ms,
            max(0.0, float(validations[name]["accuracy_drop"])),
            states[name].proxy_score,
            name))
    # Balanced keeps equal weight on normalized predicted latency and the
    # validation loss/logit deviation signal, while still respecting the
    # actual validation accuracy budget.
    max_latency = max(states[name].latency_ms for name in eligible)
    min_latency = min(states[name].latency_ms for name in eligible)
    latency_span = max(max_latency - min_latency, 1e-12)
    max_proxy = max(states[name].proxy_score for name in eligible)
    min_proxy = min(states[name].proxy_score for name in eligible)
    proxy_span = max(max_proxy - min_proxy, 1e-12)
    return min(eligible, key=lambda name: (
        0.5 * (states[name].latency_ms - min_latency) / latency_span +
        0.5 * (states[name].proxy_score - min_proxy) / proxy_span,
        states[name].latency_ms, name))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--preprocess-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True,
                        help="direct complete-attention assignment CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-limit", type=int, default=4)
    parser.add_argument("--validation-limit", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-frontier", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--reuse-isolated-candidates", action="store_true",
                        help="reuse a complete isolated_candidate_metrics.csv in output-dir")
    parser.add_argument("--logit-weight", type=float, default=0.25)
    parser.add_argument("--accuracy-first-budget-pp", type=float, default=0.5)
    parser.add_argument("--balanced-budget-pp", type=float, default=1.0)
    parser.add_argument("--performance-first-budget-pp", type=float, default=2.0)
    args = parser.parse_args()

    if args.workers <= 0 or args.max_frontier <= 0 or args.max_candidates <= 0:
        raise ValueError("workers/frontier/candidate limits must be positive")
    if not math.isfinite(args.logit_weight) or args.logit_weight < 0.0:
        raise ValueError("logit weight must be finite and non-negative")
    budgets = {
        "accuracy_first": args.accuracy_first_budget_pp,
        "balanced": args.balanced_budget_pp,
        "performance_first": args.performance_first_budget_pp,
    }
    if any(not math.isfinite(value) or value < 0.0 for value in budgets.values()):
        raise ValueError("accuracy budgets must be finite and non-negative")

    model = vr.load_model(args.model)
    if (int(model["layers"]) != MODEL_LAYERS or
            int(model["heads"]) != MODEL_HEADS or
            int(model["tokens"]) != MODEL_TOKENS):
        raise ValueError("allocator requires the audited ViT-B/16 12x12/197 model")
    preprocess = load_preprocess(args.preprocess_manifest, model)
    calibration_samples = read_manifest(args.calibration_manifest,
                                         args.image_root,
                                         args.calibration_limit)
    validation_samples = read_manifest(args.validation_manifest, args.image_root,
                                       args.validation_limit)
    cost_model = MeasuredCompositionCostModel(args.costs, MODEL_LAYERS,
                                              MODEL_HEADS)
    if not set(BENCHMARK_COMPOSITIONS).issubset(cost_model.available_compositions):
        missing_grid = sorted(set(BENCHMARK_COMPOSITIONS) -
                              cost_model.available_compositions)
        raise ValueError("benchmark is missing intended measured compositions: "
                         f"{[composition_name(row) for row in missing_grid]}")

    calibration_pixels = load_pixels_for_samples(calibration_samples, preprocess)
    validation_pixels = load_pixels_for_samples(validation_samples, preprocess)
    calibration_labels = [sample.label for sample in calibration_samples]
    validation_labels = [sample.label for sample in validation_samples]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fork before the first Graph/BLAS call in the parent.  The workers remain
    # idle while the parent computes its baselines and are then reused for
    # both candidate and final validation passes.  This avoids both BLAS fork
    # deadlocks and repeatedly pickling the 1+ GB canonical model.
    worker_pool = None
    if args.workers > 1:
        context = mp.get_context("fork")
        worker_pool = context.Pool(
            args.workers, initializer=candidate_worker_init,
            initargs=(model, calibration_pixels, calibration_labels,
                      validation_pixels, validation_labels, args.logit_weight))

    isolated_cache = output_dir / "isolated_candidate_metrics.csv"
    if args.reuse_isolated_candidates:
        if not isolated_cache.is_file():
            raise ValueError(f"isolated candidate cache is missing: {isolated_cache}")
        with isolated_cache.open(newline="", encoding="utf-8") as handle:
            isolated = list(csv.DictReader(handle))
        if len(isolated) != MODEL_LAYERS * MODEL_HEADS * 2:
            raise ValueError("isolated candidate cache does not contain 288 rows")
        alphas = alphas_from_isolated(isolated, MODEL_LAYERS, MODEL_HEADS)
        calibration_baseline = []
    else:
        calibration_baseline = []
        calibration_records: list[dict[str, dict[str, Any]]] = []
        for pixels in calibration_pixels:
            logits, records = vr.Graph(model, fixed=False).run(pixels)
            calibration_baseline.append(logits)
            calibration_records.append(records)
        alphas = fit_linear_alphas(calibration_records, MODEL_LAYERS, MODEL_HEADS)
        isolated = run_isolated_candidates(
            model, calibration_pixels, calibration_labels, calibration_baseline,
            alphas, args.workers, args.logit_weight, worker_pool)
        isolated.sort(key=lambda row: (int(row["layer"]), int(row["head"]),
                                       str(row["protocol"])))
    candidates = load_candidate_index(isolated)

    write_csv(output_dir / "isolated_candidate_metrics.csv", [
        "layer", "head", "protocol", "alpha", "validation_loss",
        "baseline_validation_loss", "loss_delta", "mse", "max_abs",
        "relative_l2", "top1_agreement", "baseline_accuracy",
        "candidate_accuracy", "accuracy_drop", "proxy_score", "elements"],
        isolated)

    layer_options: list[list[LayerOption]] = []
    layer_prune_summary = []
    for layer in range(MODEL_LAYERS):
        options = []
        for counts in BENCHMARK_COMPOSITIONS:
            cost = cost_model.predict_layer(layer, counts)
            if cost is None:
                continue
            paths, score = min_assignment_for_counts(layer, counts, candidates,
                                                     alphas)
            options.append(option_from_paths(layer, counts, paths, score,
                                             candidates, cost))
        pruned = pareto_layer_options(options)
        if not pruned:
            raise ValueError(f"no Pareto layer options for layer {layer}")
        layer_options.append(pruned)
        layer_prune_summary.append({
            "layer": layer,
            "measured_options": len(options),
            "pareto_options": len(pruned),
            "removed_dominated_options": len(options) - len(pruned),
            "compositions": [composition_name(option.counts)
                             for option in pruned],
        })

    frontier, frontier_truncated = combine_model_frontier(
        layer_options, args.max_frontier)
    pool = candidate_pool(frontier, layer_options, args.max_candidates)
    state_by_name = {f"candidate_{index:03d}": state
                     for index, state in enumerate(pool)}
    # Make the all-softmax baseline name stable for reports and policy fallback.
    baseline_state = uniform_composition_state(layer_options, (12, 0, 0))
    if baseline_state is None:
        raise ValueError("cannot construct all-softmax state")
    baseline_name = next((name for name, state in state_by_name.items()
                          if plan_key(state.plan) == plan_key(baseline_state.plan)),
                         None)
    if baseline_name is None:
        baseline_name = "candidate_baseline"
        state_by_name[baseline_name] = baseline_state

    validation_baseline = [vr.Graph(model, fixed=False).run(pixels)[0]
                           for pixels in validation_pixels]
    validations = run_validation(model, validation_pixels, validation_labels,
                                 validation_baseline, state_by_name,
                                 args.workers, worker_pool)
    if worker_pool is not None:
        worker_pool.close()
        worker_pool.join()
        worker_pool = None
    validation_by_name = {row["candidate"]: row for row in validations}
    for name in state_by_name:
        if name not in validation_by_name:
            raise ValueError(f"missing full plaintext validation for {name}")

    # Persist the actual candidate pool so policy decisions are auditable and
    # no hidden candidate is silently chosen after this point.
    candidate_rows = []
    for name, state in sorted(state_by_name.items()):
        row = validation_by_name[name]
        cost = cost_model.predict_plan(state.plan)
        if cost is None:
            raise ValueError(f"candidate {name} has an unmeasured cost point")
        candidate_rows.append({
            "candidate": name,
            "plan_sha256": plan_key(state.plan),
            "S_heads": sum(count[0] for count in counts_for_plan(state.plan)),
            "R_heads": sum(count[1] for count in counts_for_plan(state.plan)),
            "L_heads": sum(count[2] for count in counts_for_plan(state.plan)),
            "clear_accuracy_percent": row["candidate_accuracy"] * 100.0,
            "accuracy_drop_pp": row["accuracy_drop"] * 100.0,
            "validation_loss": row["validation_loss"],
            "loss_delta": row["loss_delta"],
            "top1_agreement": row["top1_agreement"],
            "logit_relative_l2": row["relative_l2"],
            "proxy_score": state.proxy_score,
            "predicted_secure_latency_ms": cost["latency_ms"],
            "predicted_comm_bytes": cost["communication_bytes"],
            "layer_compositions": ";".join(composition_name(count)
                                             for count in counts_for_plan(state.plan)),
            "baseline": name == baseline_name,
        })
    write_csv(output_dir / "candidate_validation.csv", list(candidate_rows[0]),
              candidate_rows)

    selected_names = {
        policy: select_policy(policy, budget, state_by_name, validation_by_name)
        for policy, budget in budgets.items()
    }

    search_meta = {
        "algorithm": "layerwise_exact_count_dp_then_model_pareto_dp",
        "whole_model_bruteforce": False,
        "per_layer_head_state_space": "prefix counts (S,R,L), 3 transitions/head",
        "available_measured_compositions": [
            composition_name(row) for row in sorted(cost_model.available_compositions)],
        "layer_pareto_summary": layer_prune_summary,
        "model_frontier_size": len(frontier),
        "model_frontier_truncated": frontier_truncated,
        "candidate_pool_size": len(state_by_name),
        "candidate_pool_max": args.max_candidates,
        "calibration_samples": len(calibration_samples),
        "validation_samples": len(validation_samples),
        "logit_weight": args.logit_weight,
        "isolated_candidate_metrics": str(
            (output_dir / "isolated_candidate_metrics.csv").resolve()),
        "candidate_validation": str(
            (output_dir / "candidate_validation.csv").resolve()),
        "selected_policy_candidates": selected_names,
    }

    assignment_documents = {}
    for policy, budget in budgets.items():
        name = selected_names[policy]
        state = state_by_name[name]
        validation = validation_by_name[name]
        cost = cost_model.predict_plan(state.plan)
        if cost is None:
            raise ValueError(f"selected policy {policy} has no measured cost")
        runtime_plan = output_dir / f"{policy}.head_plan.txt"
        write_runtime_plan(runtime_plan, state.plan)
        document = output_assignment(
            output_dir / f"{policy}.json", policy, budget, state, validation,
            cost, alphas, args.model, args.model_manifest,
            args.calibration_manifest, args.validation_manifest, args.costs,
            runtime_plan, cost_model.missing_compositions(),
            {**search_meta, "selected_candidate": name})
        assignment_documents[policy] = document

    # The singular conventional artifact is an explicit deployment default,
    # not a second search result.  It is the accuracy-first fixed assignment.
    default_document = dict(assignment_documents["accuracy_first"])
    default_document["default_policy"] = "accuracy_first"
    default_document["available_policy_configs"] = {
        policy: str((output_dir / f"{policy}.json").resolve())
        for policy in budgets
    }
    (output_dir / "head_assignment.json").write_text(
        json.dumps(default_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    minimum_candidates = []
    seen_plan_hashes: set[str] = set()
    for policy in budgets:
        document = assignment_documents[policy]
        plan_hash = document["search"]["selected_plan_sha256"]
        if plan_hash in seen_plan_hashes:
            continue
        seen_plan_hashes.add(plan_hash)
        minimum_candidates.append({
            "policy": policy,
            "config": str((output_dir / f"{policy}.json").resolve()),
            "runtime_plan": document["runtime"]["head_plan"],
            "version": document["runtime"]["version"],
            "plan_sha256": plan_hash,
        })

    selection_rows = []
    for policy, budget in budgets.items():
        name = selected_names[policy]
        document = assignment_documents[policy]
        validation = document["validation"]
        cost = document["predicted_cost"]
        counts = document["path_counts"]
        selection_rows.append({
            "policy": policy,
            "S_heads": counts["softmax_mpc"],
            "R_heads": counts["relu_norm_mpc"],
            "L_heads": counts["he_linear"],
            "clear_accuracy": validation["clear_accuracy_percent"],
            "accuracy_drop": validation["accuracy_drop_pp"],
            "predicted_secure_latency": cost["latency_ms"],
            "predicted_comm": cost["communication_bytes"],
        })
    write_csv(output_dir / "selection_report.csv", [
        "policy", "S_heads", "R_heads", "L_heads", "clear_accuracy",
        "accuracy_drop", "predicted_secure_latency", "predicted_comm"],
        selection_rows)

    validation_rows = cost_model.validation_rows()
    write_csv(output_dir / "cost_model_validation.csv", [
        "composition", "predicted_latency", "measured_latency", "error_percent"],
        [{key: row[key] for key in (
            "composition", "predicted_latency", "measured_latency", "error_percent")}
         for row in validation_rows])
    write_csv(output_dir / "missing_cost_points.csv", [
        "composition", "S_heads", "R_heads", "L_heads", "status"],
        [{"composition": composition_name(counts), "S_heads": counts[0],
          "R_heads": counts[1], "L_heads": counts[2],
          "status": "NOT_MEASURED_NO_ASSIGNMENT_COST"}
         for counts in cost_model.missing_compositions()])

    report = {
        "format": "OURS_OFFLINE_COST_AWARE_HEAD_ALLOCATOR_REPORT_V1",
        "status": "PASS" if all(
            document["status"] == "PASS"
            for document in assignment_documents.values()) else "FAIL",
        "secure_inference_run": False,
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "model_manifest": str(args.model_manifest.resolve()),
        "calibration_manifest": str(args.calibration_manifest.resolve()),
        "validation_manifest": str(args.validation_manifest.resolve()),
        "cost_source": str(args.costs.resolve()),
        "cost_source_sha256": sha256_file(args.costs),
        "cost_model": {
            "kind": "sparse_measured_complete_assignment_lookup",
            "formula": "C_l(S,R,L)=median(measured complete assignment latency_ms); C_model=sum_l C_l",
            "communication_formula": "B_l(S,R,L)=median(measured complete assignment bytes); B_model=sum_l B_l",
            "uses_per_head_addition": False,
            "uses_interpolation": False,
            "available_compositions": [composition_name(row)
                                        for row in sorted(cost_model.available_compositions)],
            "missing_compositions": [composition_name(row)
                                      for row in cost_model.missing_compositions()],
            "validation_file": str((output_dir / "cost_model_validation.csv").resolve()),
        },
        "budgets_pp": budgets,
        "selected": {
            policy: {
                "candidate": selected_names[policy],
                "config": str((output_dir / f"{policy}.json").resolve()),
                "status": assignment_documents[policy]["status"],
            }
            for policy in budgets
        },
        "files": {
            "default_assignment": str((output_dir / "head_assignment.json").resolve()),
            "selection_report": str((output_dir / "selection_report.csv").resolve()),
            "candidate_validation": str((output_dir / "candidate_validation.csv").resolve()),
            "isolated_candidate_metrics": str((output_dir / "isolated_candidate_metrics.csv").resolve()),
            "missing_cost_points": str((output_dir / "missing_cost_points.csv").resolve()),
        },
        "search": search_meta,
        "next_step": {
            "required": "real secure validation only; no full 12-layer secure inference was run",
            "minimum_candidates": minimum_candidates,
            "deduplicate_by_plan_sha256": True,
        },
    }
    (output_dir / "allocator_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OURS_OFFLINE_HEAD_ALLOCATOR status={report['status']}")
    for policy in budgets:
        row = next(item for item in selection_rows if item["policy"] == policy)
        print(f"OURS_OFFLINE_HEAD_ALLOCATOR policy={policy} "
              f"accuracy={row['clear_accuracy']:.4f}% "
              f"drop_pp={row['accuracy_drop']:.4f} "
              f"latency_ms={row['predicted_secure_latency']:.3f}")
    print(f"OURS_OFFLINE_HEAD_ALLOCATOR_REPORT={output_dir / 'allocator_report.json'}")


if __name__ == "__main__":
    main()
