#!/usr/bin/env python3
"""Fail-closed acceptance summary for the BLB ViT-B/16 artifacts.

The tool is deliberately read-only with respect to every input artifact.  It
only writes the explicitly requested ``--output`` JSON file.  In particular,
operator error-row ``status`` strings are *not* interpreted as numerical
threshold decisions: numerical fields are validated and reported directly,
while the acceptance conditions below are recomputed from the underlying
top-1 values, logits, padding comparisons, BFV ranges, and accounting rows.

Exit codes are suitable for a release gate:

* 0: every requested stage, including full 12-block secure inference, passed;
* 1: at least one supplied/required artifact failed validation;
* 2: required artifacts passed, but full 12-block secure inference or its
  aggregate-only intermediate-error validation was not run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SUMMARY_FORMAT = "BLB_VIT_B16_ACCEPTANCE_SUMMARY_V1"
REFERENCE_FORMAT = "BOLT_VIT_REFERENCE_REPORT_V1"
PYTORCH_FORMAT = "BOLT_VIT_PYTORCH_REFERENCE_V1"
BFV_AUDIT_FORMAT = "BLB_VIT_BFV_RANGE_AUDIT_V1"
SEED_FORMAT = "BOLT_VIT_SEED_DIAGNOSTICS_V1"
IMAGENET_FORMAT = "BOLT_VIT_B16_PYTORCH_IMAGENET_EVAL_V1"
PLAIN_MODULUS = 536_903_681
CENTERED_HALF = (PLAIN_MODULUS - 1) // 2
UNAVAILABLE_LAYER = "NOT_AVAILABLE_NO_SECURE_INTERMEDIATE_OBSERVER"
REQUIRED_BREAKDOWN_CATEGORIES = {
    "conversion", "he", "local_or_public", "mpc", "runtime_overhead"
}
REQUIRED_PERFORMANCE_PHASES = {
    "bfv_context_and_secret_key_generation",
    "evaluation_public_keys",
    "base_ot_and_mpc_setup",
    "public_model_metadata_exchange",
    "offline_preprocessing_checkpoint_parse",
    "online_inference",
    "final_reveal",
    "end_to_end_inference",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class AcceptanceError(RuntimeError):
    """An artifact cannot establish the claimed acceptance property."""


def _reject_json_constant(value: str) -> None:
    raise AcceptanceError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AcceptanceError(f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise AcceptanceError(f"JSON root must be an object: {path}")
    return value


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcceptanceError(f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise AcceptanceError(f"{field} must be an array")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise AcceptanceError(f"{field} must be a nonempty string")
    return value


def _integer(value: Any, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AcceptanceError(f"{field} must be an exact integer")
    if minimum is not None and value < minimum:
        raise AcceptanceError(f"{field} must be >= {minimum}")
    return value


def _number(value: Any, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AcceptanceError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise AcceptanceError(f"{field} must be >= {minimum}")
    return result


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise AcceptanceError(f"{field} must be boolean")
    return value


def _shape(value: Any, field: str) -> list[int]:
    result = [_integer(item, field, 1) for item in _array(value, field)]
    if not result:
        raise AcceptanceError(f"{field} cannot be empty")
    return result


def _sha256(value: Any, field: str) -> str:
    digest = _string(value, field)
    if SHA256_RE.fullmatch(digest) is None:
        raise AcceptanceError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise AcceptanceError(f"{field}: expected {expected!r}, got {actual!r}")


def _require_close(actual: float, expected: float, field: str,
                   tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AcceptanceError(
            f"{field}: expected {expected:.12g}, got {actual:.12g}")


def _argmax(values: Sequence[float]) -> int:
    if not values:
        raise AcceptanceError("cannot compute top-1 of empty logits")
    return max(range(len(values)), key=values.__getitem__)


def _logits(section: Mapping[str, Any], field: str,
            classes: int = 1000) -> tuple[list[float], int]:
    raw = _array(section.get("logits"), f"{field}.logits")
    if len(raw) != classes:
        raise AcceptanceError(
            f"{field}.logits must contain {classes} classes, got {len(raw)}")
    values = [_number(item, f"{field}.logits[{index}]")
              for index, item in enumerate(raw)]
    top1 = _integer(section.get("top1"), f"{field}.top1", 0)
    if top1 >= classes:
        raise AcceptanceError(f"{field}.top1 is outside the classifier range")
    _require_equal(top1, _argmax(values), f"{field}.top1 recomputation")
    return values, top1


def _error_metrics(value: Any, field: str) -> dict[str, float | int]:
    """Validate numerical evidence without consulting error-row ``status``."""
    row = _object(value, field)
    result = {
        "elements": _integer(row.get("elements"), f"{field}.elements", 1),
        "max_abs": _number(row.get("max_abs"), f"{field}.max_abs", 0.0),
        "mean_abs": _number(row.get("mean_abs"), f"{field}.mean_abs", 0.0),
        "rmse": _number(row.get("rmse"), f"{field}.rmse", 0.0),
    }
    if (result["max_abs"] == 0.0 and
            (result["mean_abs"] != 0.0 or result["rmse"] != 0.0)):
        raise AcceptanceError(
            f"{field}: zero maximum requires zero mean and RMSE")
    tolerance = 16.0 * max(
        math.ulp(result["max_abs"]), math.ulp(result["mean_abs"]),
        math.ulp(result["rmse"]))
    if (result["mean_abs"] > result["max_abs"] + tolerance or
            result["rmse"] < result["mean_abs"] - tolerance or
            result["rmse"] > result["max_abs"] + tolerance):
        raise AcceptanceError(
            f"{field} violates mean_abs <= rmse <= max_abs")
    return result


def _expected_operator_names(executed_blocks: int) -> list[str]:
    names = ["patchify", "patch_projection", "cls_position"]
    suffixes = (
        "ln1", "q", "k", "v", "qk", "softmax", "attention_context",
        "attention_residual", "ln2", "fc1", "gelu", "ffn_residual",
    )
    names.extend(f"block_{block}.{suffix}"
                 for block in range(executed_blocks) for suffix in suffixes)
    names.extend(("final_ln", "cls", "logits"))
    return names


def _validate_model(model: Any, field: str = "model") -> str:
    value = _object(model, field)
    expected = {
        "channels": 3,
        "classes": 1000,
        "dim": 768,
        "ff_dim": 3072,
        "heads": 12,
        "image_size": 224,
        "layers": 12,
        "patch_size": 16,
        "tokens": 197,
    }
    for key, required in expected.items():
        _require_equal(value.get(key), required, f"{field}.{key}")
    _require_equal(value.get("qkv_bias"), True, f"{field}.qkv_bias")
    _require_close(_number(value.get("layer_norm_epsilon"),
                           f"{field}.layer_norm_epsilon", 0.0),
                   1e-12, f"{field}.layer_norm_epsilon", 0.0)
    _string(value.get("source_id"), f"{field}.source_id")
    _string(value.get("source_revision"), f"{field}.source_revision")
    return _sha256(value.get("source_sha256"), f"{field}.source_sha256")


def _validate_padding(report: Mapping[str, Any], field: str) -> None:
    packing = _object(report.get("packing"), f"{field}.packing")
    _require_equal(packing.get("logical_tokens"), 197,
                   f"{field}.packing.logical_tokens")
    _require_equal(packing.get("physical_tokens"), 256,
                   f"{field}.packing.physical_tokens")
    _require_equal(packing.get("valid_positions"), [0, 196],
                   f"{field}.packing.valid_positions")
    _require_equal(packing.get("padding_positions"), [197, 255],
                   f"{field}.packing.padding_positions")

    padding = _object(report.get("padding_invariance"),
                      f"{field}.padding_invariance")
    fixed_top1 = _integer(_object(report.get("fixed"),
                                 f"{field}.fixed").get("top1"),
                          f"{field}.fixed.top1", 0)
    _require_equal(padding.get("logical_fixed_top1"), fixed_top1,
                   f"{field}.padding_invariance.logical_fixed_top1")
    _require_equal(padding.get("physical_fixed_top1"), fixed_top1,
                   f"{field}.padding_invariance.physical_fixed_top1")
    _require_close(_number(padding.get("max_abs_valid_prefix_difference"),
                           f"{field}.padding max_abs", 0.0),
                   0.0, f"{field}.padding max_abs", 0.0)
    comparisons = _array(padding.get("operators"),
                         f"{field}.padding_invariance.operators")
    if not comparisons:
        raise AcceptanceError(f"{field}.padding_invariance.operators is empty")
    for index, entry in enumerate(comparisons):
        item = _object(entry, f"{field}.padding operator[{index}]")
        _string(item.get("name"), f"{field}.padding operator[{index}].name")
        metrics = _error_metrics(item.get("comparison"),
                                 f"{field}.padding operator[{index}].comparison")
        _require_close(float(metrics["max_abs"]), 0.0,
                       f"{field}.padding operator[{index}].max_abs", 0.0)


def _validate_reference(path: Path, executed_blocks: int,
                        checkpoint_sha256: str | None = None) -> dict[str, Any]:
    report = _load_json(path)
    _require_equal(report.get("format"), REFERENCE_FORMAT, f"{path}.format")
    source_sha = _validate_model(report.get("model"), f"{path}.model")
    if checkpoint_sha256 is not None:
        _require_equal(source_sha, checkpoint_sha256,
                       f"{path}.model.source_sha256")

    execution = _object(report.get("execution"), f"{path}.execution")
    _require_equal(execution.get("checkpoint_blocks"), 12,
                   f"{path}.execution.checkpoint_blocks")
    _require_equal(execution.get("executed_blocks"), executed_blocks,
                   f"{path}.execution.executed_blocks")
    _require_equal(execution.get("validation_block_limit"),
                   executed_blocks != 12,
                   f"{path}.execution.validation_block_limit")

    fixed_point = _object(report.get("fixed_point"), f"{path}.fixed_point")
    for key, expected in (("ell", 37), ("activation_scale", 12),
                          ("weight_scale", 12), ("gelu_input_scale", 11),
                          ("gelu_output_scale", 4)):
        _require_equal(fixed_point.get(key), expected,
                       f"{path}.fixed_point.{key}")

    clear_logits, clear_top1 = _logits(
        _object(report.get("clear"), f"{path}.clear"), f"{path}.clear")
    fixed_section = _object(report.get("fixed"), f"{path}.fixed")
    fixed_logits, fixed_top1 = _logits(fixed_section, f"{path}.fixed")
    _require_equal(fixed_section.get("top1_matches_clear"),
                   fixed_top1 == clear_top1,
                   f"{path}.fixed.top1_matches_clear")
    if fixed_top1 != clear_top1:
        raise AcceptanceError(f"{path}: fixed top-1 differs from clear top-1")

    operators = _array(report.get("operators"), f"{path}.operators")
    expected_names = _expected_operator_names(executed_blocks)
    actual_names: list[str] = []
    fixed_clear_max = 0.0
    for index, raw in enumerate(operators):
        row = _object(raw, f"{path}.operators[{index}]")
        name = _string(row.get("name"), f"{path}.operators[{index}].name")
        actual_names.append(name)
        _shape(row.get("shape"), f"{path}.{name}.shape")
        _require_equal(row.get("bit_length"), 37, f"{path}.{name}.bit_length")
        _integer(row.get("scale"), f"{path}.{name}.scale", 0)
        _string(row.get("packing"), f"{path}.{name}.packing")
        clear_meta = _object(row.get("clear"), f"{path}.{name}.clear")
        _shape(clear_meta.get("shape"), f"{path}.{name}.clear.shape")
        _require_equal(clear_meta.get("bit_length"), 64,
                       f"{path}.{name}.clear.bit_length")
        _string(clear_meta.get("packing"), f"{path}.{name}.clear.packing")
        _string(clear_meta.get("scale"), f"{path}.{name}.clear.scale")
        metrics = _error_metrics(row.get("clear_fixed"),
                                 f"{path}.{name}.clear_fixed")
        fixed_clear_max = max(fixed_clear_max, float(metrics["max_abs"]))
    _require_equal(actual_names, expected_names, f"{path}.operator graph")
    _validate_padding(report, str(path))

    return {
        "report": report,
        "source_sha256": source_sha,
        "clear_logits": clear_logits,
        "fixed_logits": fixed_logits,
        "clear_top1": clear_top1,
        "fixed_top1": fixed_top1,
        "fixed_clear_operator_max_abs": fixed_clear_max,
        "operators": len(operators),
    }


def _check_pytorch(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    _require_equal(report.get("format"), PYTORCH_FORMAT, f"{path}.format")
    _require_equal(report.get("architecture"), "ViTForImageClassification",
                   f"{path}.architecture")
    _require_equal(report.get("random_weights"), False, f"{path}.random_weights")
    digest = _sha256(report.get("checkpoint_sha256"),
                     f"{path}.checkpoint_sha256")
    logits, top1 = _logits(report, str(path))
    return {
        "status": "PASS",
        "checkpoint_sha256": digest,
        "top1": top1,
        "classes": len(logits),
        "random_weights": False,
    }


def _check_full_reference(path: Path, pytorch_path: Path,
                          pytorch_clear_atol: float) -> dict[str, Any]:
    pytorch = _load_json(pytorch_path)
    _require_equal(pytorch.get("format"), PYTORCH_FORMAT,
                   f"{pytorch_path}.format")
    checkpoint_sha = _sha256(pytorch.get("checkpoint_sha256"),
                             f"{pytorch_path}.checkpoint_sha256")
    pytorch_logits, pytorch_top1 = _logits(pytorch, str(pytorch_path))
    reference = _validate_reference(path, 12, checkpoint_sha)
    clear_diff = max(abs(left - right) for left, right in
                     zip(pytorch_logits, reference["clear_logits"]))
    if clear_diff > pytorch_clear_atol:
        raise AcceptanceError(
            f"PyTorch/clear max_abs {clear_diff:.12g} exceeds "
            f"{pytorch_clear_atol:.12g}")
    _require_equal(reference["clear_top1"], pytorch_top1,
                   "PyTorch/clear top-1")
    logits_fixed_clear = max(
        abs(left - right) for left, right in
        zip(reference["clear_logits"], reference["fixed_logits"]))
    return {
        "status": "PASS",
        "executed_blocks": 12,
        "operators": reference["operators"],
        "clear_top1": reference["clear_top1"],
        "fixed_top1": reference["fixed_top1"],
        "pytorch_clear_max_abs": clear_diff,
        "pytorch_clear_atol": pytorch_clear_atol,
        "fixed_clear_logits_max_abs": logits_fixed_clear,
        "fixed_clear_operator_max_abs":
            reference["fixed_clear_operator_max_abs"],
        "padding_max_abs": 0.0,
        "error_row_status_used_as_threshold": False,
    }


def _expected_bfv_rows() -> dict[str, tuple[str, str]]:
    expected: dict[str, tuple[str, str]] = {
        "patch_projection": ("linear", "[196,768]")
    }
    specs = (
        ("ln1_affine", "affine", "[256,768]"),
        ("q_projection", "linear", "[256,768]"),
        ("k_projection", "linear", "[256,768]"),
        ("v_projection", "linear", "[256,768]"),
        ("qk", "qk", "[12,256,256]"),
        ("softmax_v", "sv", "[12,256,64]"),
        ("output_projection", "linear", "[256,768]"),
        ("ln2_affine", "affine", "[256,768]"),
        ("fc1", "linear", "[256,3072]"),
        ("fc2", "linear", "[256,768]"),
    )
    for block in range(12):
        for suffix, kind, shape in specs:
            expected[f"block_{block}.{suffix}"] = (kind, shape)
    expected["final_ln_affine"] = ("affine", "[256,768]")
    expected["classification_head"] = ("linear", "[1,1000]")
    return expected


def _check_bfv_audit(path: Path) -> dict[str, Any]:
    audit = _load_json(path)
    _require_equal(audit.get("format"), BFV_AUDIT_FORMAT, f"{path}.format")
    _require_equal(audit.get("plain_modulus"), PLAIN_MODULUS,
                   f"{path}.plain_modulus")
    _require_equal(audit.get("centered_interval"),
                   [-CENTERED_HALF, CENTERED_HALF],
                   f"{path}.centered_interval")
    coverage = _object(audit.get("coverage"), f"{path}.coverage")
    _require_equal(coverage.get("complete"), True, f"{path}.coverage.complete")
    _require_equal(coverage.get("missing"), [], f"{path}.coverage.missing")

    rows = _array(audit.get("accumulators"), f"{path}.accumulators")
    expected = _expected_bfv_rows()
    _require_equal(len(rows), len(expected), f"{path}.accumulator count")
    _require_equal(coverage.get("expected"), len(expected),
                   f"{path}.coverage.expected")
    _require_equal(coverage.get("observed"), len(expected),
                   f"{path}.coverage.observed")
    seen: set[str] = set()
    minimum_headroom: int | None = None
    peak_utilization = 0.0
    for index, raw in enumerate(rows):
        row = _object(raw, f"{path}.accumulators[{index}]")
        name = _string(row.get("name"), f"{path}.accumulators[{index}].name")
        if name in seen:
            raise AcceptanceError(f"{path}: duplicate BFV accumulator {name}")
        seen.add(name)
        if name not in expected:
            raise AcceptanceError(f"{path}: unexpected BFV accumulator {name}")
        required_kind, required_shape = expected[name]
        _require_equal(row.get("kind"), required_kind, f"{path}.{name}.kind")
        _require_equal(row.get("shape"), required_shape, f"{path}.{name}.shape")
        raw_min = _integer(row.get("raw_min"), f"{path}.{name}.raw_min")
        raw_max = _integer(row.get("raw_max"), f"{path}.{name}.raw_max")
        if raw_min > raw_max:
            raise AcceptanceError(f"{path}.{name}: raw_min exceeds raw_max")
        if raw_min < -CENTERED_HALF or raw_max > CENTERED_HALF:
            raise AcceptanceError(f"{path}.{name}: centered BFV range would wrap")
        _require_equal(row.get("modular_wrap"), False,
                       f"{path}.{name}.modular_wrap")
        headroom = min(raw_min + CENTERED_HALF,
                       CENTERED_HALF - raw_max)
        minimum_headroom = (headroom if minimum_headroom is None
                            else min(minimum_headroom, headroom))
        peak_utilization = max(
            peak_utilization,
            max(abs(raw_min), abs(raw_max)) / CENTERED_HALF)
    _require_equal(seen, set(expected), f"{path}.BFV graph coverage")
    _require_equal(audit.get("failed_accumulators"), [],
                   f"{path}.failed_accumulators")
    _require_equal(audit.get("first_failed_accumulator"), None,
                   f"{path}.first_failed_accumulator")
    return {
        "status": "PASS",
        "plain_modulus": PLAIN_MODULUS,
        "centered_interval": [-CENTERED_HALF, CENTERED_HALF],
        "expected_accumulators": len(expected),
        "observed_accumulators": len(rows),
        "minimum_headroom": minimum_headroom,
        "peak_utilization": peak_utilization,
        "range_recomputed_from_raw_min_max": True,
    }


def _csv_rows(path: Path, required_fields: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise AcceptanceError(f"missing CSV artifact: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise AcceptanceError(f"CSV has no header: {path}")
            missing = set(required_fields) - set(reader.fieldnames)
            if missing:
                raise AcceptanceError(
                    f"{path}: missing CSV fields {sorted(missing)}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise AcceptanceError(f"cannot read CSV artifact {path}: {error}") from error
    if not rows:
        raise AcceptanceError(f"CSV has no data rows: {path}")
    return rows


def _csv_float(value: str | None, field: str) -> float:
    try:
        result = float(value if value is not None else "")
    except ValueError as error:
        raise AcceptanceError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise AcceptanceError(f"{field} must be finite")
    return result


def _csv_int(value: str | None, field: str) -> int:
    try:
        return int(value if value is not None else "")
    except ValueError as error:
        raise AcceptanceError(f"{field} must be an exact integer") from error


def _performance(path: Path) -> dict[str, dict[str, float | int]]:
    rows = _csv_rows(path, (
        "phase", "time_ms", "bidirectional_bytes", "logical_rounds", "status"))
    result: dict[str, dict[str, float | int]] = {}
    for index, row in enumerate(rows):
        phase = _string(row.get("phase"), f"{path}.row[{index}].phase")
        if phase in result:
            raise AcceptanceError(f"{path}: duplicate performance phase {phase}")
        _require_equal(row.get("status"), "PASS", f"{path}.{phase}.status")
        time_ms = _csv_float(row.get("time_ms"), f"{path}.{phase}.time_ms")
        byte_count = _csv_int(row.get("bidirectional_bytes"),
                              f"{path}.{phase}.bidirectional_bytes")
        rounds = _csv_int(row.get("logical_rounds"),
                          f"{path}.{phase}.logical_rounds")
        if min(time_ms, byte_count, rounds) < 0:
            raise AcceptanceError(f"{path}.{phase}: negative accounting value")
        result[phase] = {
            "time_ms": time_ms,
            "bidirectional_bytes": byte_count,
            "logical_rounds": rounds,
        }
    _require_equal(set(result), REQUIRED_PERFORMANCE_PHASES,
                   f"{path}.performance phases")
    if result["online_inference"]["time_ms"] <= 0:
        raise AcceptanceError(f"{path}: online inference time must be positive")
    if (result["end_to_end_inference"]["time_ms"] <
            result["online_inference"]["time_ms"]):
        raise AcceptanceError(f"{path}: end-to-end time is shorter than online time")
    return result


def _breakdown(path: Path,
               online: Mapping[str, float | int]) -> dict[str, Any]:
    required = (
        "category", "time_ms", "bidirectional_bytes", "logical_rounds",
        "ciphertexts", "rotations", "multiply_plain", "ct_ct_multiply",
        "relinearizations", "mod_switches", "noise_floodings", "status")
    rows = _csv_rows(path, required)
    categories: dict[str, dict[str, float | int]] = {}
    integer_fields = required[2:-1]
    for index, row in enumerate(rows):
        category = _string(row.get("category"),
                           f"{path}.row[{index}].category")
        if category in categories:
            raise AcceptanceError(f"{path}: duplicate breakdown category {category}")
        _require_equal(row.get("status"), "PASS", f"{path}.{category}.status")
        values: dict[str, float | int] = {
            "time_ms": _csv_float(row.get("time_ms"),
                                  f"{path}.{category}.time_ms")
        }
        for field in integer_fields:
            values[field] = _csv_int(row.get(field), f"{path}.{category}.{field}")
        if any(float(value) < 0 for value in values.values()):
            raise AcceptanceError(f"{path}.{category}: negative accounting value")
        categories[category] = values
    _require_equal(set(categories), REQUIRED_BREAKDOWN_CATEGORIES,
                   f"{path}.breakdown categories")

    time_sum = sum(float(row["time_ms"]) for row in categories.values())
    bytes_sum = sum(int(row["bidirectional_bytes"]) for row in categories.values())
    rounds_sum = sum(int(row["logical_rounds"]) for row in categories.values())
    online_time = float(online["time_ms"])
    time_tolerance = max(1e-3, abs(online_time) * 1e-9)
    _require_close(time_sum, online_time, f"{path}.time breakdown",
                   time_tolerance)
    _require_equal(bytes_sum, int(online["bidirectional_bytes"]),
                   f"{path}.byte breakdown")
    _require_equal(rounds_sum, int(online["logical_rounds"]),
                   f"{path}.round breakdown")
    return {
        "categories": categories,
        "sum": {
            "time_ms": time_sum,
            "bidirectional_bytes": bytes_sum,
            "logical_rounds": rounds_sum,
        },
        "equals_online_inference": True,
    }


def _validate_secure_final(reference: Mapping[str, Any], field: str,
                           secure_fixed_atol: float) -> dict[str, Any]:
    fixed_logits, fixed_top1 = _logits(
        _object(reference.get("fixed"), f"{field}.fixed"), f"{field}.fixed")
    secure = _object(reference.get("secure"), f"{field}.secure")
    secure_logits, secure_top1 = _logits(secure, f"{field}.secure")
    _require_equal(secure_top1, fixed_top1, f"{field}.secure/fixed top-1")
    _require_equal(secure.get("top1_matches_fixed"), True,
                   f"{field}.secure.top1_matches_fixed")
    max_abs = max(abs(left - right)
                  for left, right in zip(secure_logits, fixed_logits))
    declared = _error_metrics(secure.get("fixed_logits_error"),
                              f"{field}.secure.fixed_logits_error")
    _require_close(float(declared["max_abs"]), max_abs,
                   f"{field}.secure.fixed_logits_error.max_abs")
    if max_abs > secure_fixed_atol:
        raise AcceptanceError(
            f"{field}: secure/fixed logits max_abs {max_abs:.12g} exceeds "
            f"{secure_fixed_atol:.12g}")
    return {
        "fixed_top1": fixed_top1,
        "secure_top1": secure_top1,
        "secure_fixed_logits_max_abs": max_abs,
        "secure_fixed_atol": secure_fixed_atol,
    }


def _check_secure_run(reference_path: Path, performance_path: Path,
                      breakdown_path: Path, executed_blocks: int,
                      checkpoint_sha256: str,
                      secure_fixed_atol: float) -> dict[str, Any]:
    validated = _validate_reference(reference_path, executed_blocks,
                                    checkpoint_sha256)
    secure = _validate_secure_final(validated["report"], str(reference_path),
                                    secure_fixed_atol)
    performance = _performance(performance_path)
    online = performance["online_inference"]
    breakdown = _breakdown(breakdown_path, online)
    e2e = performance["end_to_end_inference"]
    return {
        "status": "PASS",
        "executed_blocks": executed_blocks,
        "operators": validated["operators"],
        "fixed_top1": secure["fixed_top1"],
        "secure_top1": secure["secure_top1"],
        "secure_fixed_logits_max_abs":
            secure["secure_fixed_logits_max_abs"],
        "secure_fixed_atol": secure_fixed_atol,
        "padding_max_abs": 0.0,
        "online": online,
        "end_to_end": e2e,
        "backend_breakdown": breakdown,
        "intermediate_reconstruction": False,
    }


def _check_seed_diagnostics(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    _require_equal(report.get("format"), SEED_FORMAT, f"{path}.format")
    rows = _array(report.get("rows"), f"{path}.rows")
    cases = _integer(report.get("cases"), f"{path}.cases", 3)
    _require_equal(len(rows), cases, f"{path}.row count")
    _require_equal(report.get("secure_intermediate_reconstruction"), False,
                   f"{path}.secure_intermediate_reconstruction")
    _require_equal(report.get("first_inconsistent_layer"), UNAVAILABLE_LAYER,
                   f"{path}.first_inconsistent_layer")
    seeds: set[int] = set()
    matches = 0
    fixed_clear_errors: list[float] = []
    secure_fixed_errors: list[float] = []
    for index, raw in enumerate(rows):
        row = _object(raw, f"{path}.rows[{index}]")
        seed = _integer(row.get("seed"), f"{path}.rows[{index}].seed")
        if seed in seeds:
            raise AcceptanceError(f"{path}: duplicate seed {seed}")
        seeds.add(seed)
        fixed_top1 = _integer(row.get("fixed_top1"),
                              f"{path}.rows[{index}].fixed_top1", 0)
        secure_top1 = _integer(row.get("secure_top1"),
                               f"{path}.rows[{index}].secure_top1", 0)
        declared_match = _boolean(row.get("secure_fixed_top1_match"),
                                  f"{path}.rows[{index}].top1 match")
        _require_equal(declared_match, secure_top1 == fixed_top1,
                       f"{path}.rows[{index}].top1 match recomputation")
        if not declared_match:
            raise AcceptanceError(f"{path}: seed {seed} secure/fixed top-1 differs")
        matches += 1
        _require_equal(row.get("padding_invariance"), "PASS",
                       f"{path}.rows[{index}].padding_invariance")
        _require_equal(row.get("first_inconsistent_layer"), UNAVAILABLE_LAYER,
                       f"{path}.rows[{index}].first_inconsistent_layer")
        fixed_clear_errors.append(_number(
            row.get("fixed_clear_max_abs"),
            f"{path}.rows[{index}].fixed_clear_max_abs", 0.0))
        secure_error = row.get("secure_fixed_logits_max_abs")
        if secure_error != "NOT_AVAILABLE":
            secure_fixed_errors.append(_number(
                secure_error,
                f"{path}.rows[{index}].secure_fixed_logits_max_abs", 0.0))
    _require_equal(report.get("secure_fixed_top1_matches"), matches,
                   f"{path}.secure_fixed_top1_matches")
    rate = matches / cases
    _require_close(_number(report.get("secure_fixed_top1_consistency_rate"),
                           f"{path}.consistency rate"),
                   rate, f"{path}.consistency rate")
    if rate != 1.0:
        raise AcceptanceError(f"{path}: multi-seed top-1 rate is not 1.0")
    _require_close(_number(report.get("max_abs_error_fixed_clear"),
                           f"{path}.max_abs_error_fixed_clear", 0.0),
                   max(fixed_clear_errors), f"{path}.max fixed/clear error")
    declared_secure_max = report.get("max_abs_error_secure_fixed_logits")
    if secure_fixed_errors:
        _require_close(_number(declared_secure_max,
                               f"{path}.max secure/fixed error", 0.0),
                       max(secure_fixed_errors),
                       f"{path}.max secure/fixed error")
    elif declared_secure_max is not None:
        raise AcceptanceError(
            f"{path}.max_abs_error_secure_fixed_logits must be null when unavailable")
    return {
        "status": "PASS",
        "cases": cases,
        "secure_fixed_top1_matches": matches,
        "secure_fixed_top1_consistency_rate": rate,
        "max_abs_error_fixed_clear": max(fixed_clear_errors),
        "max_abs_error_secure_fixed_logits": (
            max(secure_fixed_errors) if secure_fixed_errors else None),
        "secure_error_scope": report.get("secure_error_scope"),
        "first_inconsistent_layer": UNAVAILABLE_LAYER,
        "intermediate_reconstruction": False,
    }


def _check_imagenet(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    _require_equal(report.get("format"), IMAGENET_FORMAT, f"{path}.format")
    _require_equal(report.get("random_weights"), False, f"{path}.random_weights")
    samples = _integer(report.get("samples"), f"{path}.samples", 1)
    _require_equal(samples, 50_000, f"{path}.samples")
    top1_correct = _integer(report.get("top1_correct"),
                            f"{path}.top1_correct", 0)
    top5_correct = _integer(report.get("top5_correct"),
                            f"{path}.top5_correct", 0)
    if not (top1_correct <= top5_correct <= samples):
        raise AcceptanceError(f"{path}: invalid top-1/top-5 counts")
    top1 = _number(report.get("top1"), f"{path}.top1", 0.0)
    top5 = _number(report.get("top5"), f"{path}.top5", 0.0)
    _require_close(top1, top1_correct / samples, f"{path}.top1")
    _require_close(top5, top5_correct / samples, f"{path}.top5")
    seconds = _number(report.get("evaluation_seconds"),
                      f"{path}.evaluation_seconds", 0.0)
    if seconds <= 0:
        raise AcceptanceError(f"{path}.evaluation_seconds must be positive")
    images_per_second = _number(report.get("images_per_second"),
                                f"{path}.images_per_second", 0.0)
    _require_close(images_per_second, samples / seconds,
                   f"{path}.images_per_second", 1e-9)
    return {
        "status": "PASS",
        "samples": samples,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top1": top1,
        "top5": top5,
        "evaluation_seconds": seconds,
        "random_weights": False,
    }


def _check_intermediate_privacy(report_paths: Sequence[Path],
                                seed_path: Path,
                                secure_fixed_atol: float) -> dict[str, Any]:
    verified: list[str] = []
    intermediate_rows = 0
    aggregate_rows = 0
    full12_aggregate_rows = 0
    first_nonzero_stage: str | None = None
    for path in report_paths:
        report = _load_json(path)
        _require_equal(report.get("format"), REFERENCE_FORMAT, f"{path}.format")
        secure_root = _object(report.get("secure"), f"{path}.secure")
        _require_equal(secure_root.get("intermediate_reconstruction"), False,
                       f"{path}.secure.intermediate_reconstruction")
        layerwise_status = secure_root.get("layerwise_error_status")
        if layerwise_status not in {"NOT_AVAILABLE", "AGGREGATE_ONLY"}:
            raise AcceptanceError(
                f"{path}.secure.layerwise_error_status is invalid")
        execution = _object(report.get("execution"), f"{path}.execution")
        blocks = _integer(execution.get("executed_blocks"),
                          f"{path}.execution.executed_blocks", 1)
        expected_aggregates = {
            "patch_projection", "cls", "logits",
            *(f"block_{layer}.{suffix}"
              for layer in range(blocks)
              for suffix in ("attention_residual", "ffn_residual")),
        }
        seen_aggregates: set[str] = set()
        for index, raw in enumerate(_array(report.get("operators"),
                                           f"{path}.operators")):
            row = _object(raw, f"{path}.operators[{index}]")
            name = _string(row.get("name"), f"{path}.operators[{index}].name")
            secure = _object(row.get("secure"), f"{path}.{name}.secure")
            if (layerwise_status == "AGGREGATE_ONLY" and
                    name in expected_aggregates):
                _require_equal(secure.get("status"), "PASS",
                               f"{path}.{name}.secure.status")
                _require_equal(secure.get("intermediate_reconstruction"), False,
                               f"{path}.{name}.intermediate_reconstruction")
                fixed_secure = _error_metrics(
                    secure.get("fixed_secure"),
                    f"{path}.{name}.fixed_secure")
                clear_secure = _error_metrics(
                    secure.get("clear_secure"),
                    f"{path}.{name}.clear_secure")
                clear_fixed = _error_metrics(
                    row.get("clear_fixed"), f"{path}.{name}.clear_fixed")
                if fixed_secure["max_abs"] > secure_fixed_atol:
                    if first_nonzero_stage is None:
                        first_nonzero_stage = name
                    raise AcceptanceError(
                        f"{path}.{name}.fixed_secure.max_abs exceeds "
                        f"secure_fixed_atol={secure_fixed_atol}")
                quantization = _number(
                    secure.get("clear_quantization_bound"),
                    f"{path}.{name}.clear_quantization_bound", 0.0)
                if quantization <= 0.0:
                    raise AcceptanceError(
                        f"{path}.{name}.clear_quantization_bound must be positive")
                _require_equal(
                    secure.get("rmse_accumulator"),
                    "exact_canonical_two_limb",
                    f"{path}.{name}.rmse_accumulator")
                _require_equal(secure.get("rmse_square_scale"), 36,
                               f"{path}.{name}.rmse_square_scale")
                expected_split = (fixed_secure["elements"] - 1).bit_length()
                _require_equal(secure.get("rmse_split_bits"), expected_split,
                               f"{path}.{name}.rmse_split_bits")
                # If secure matches fixed within the configured infinity-norm
                # tolerance, the clear/secure and clear/fixed norm summaries
                # can differ only by that error plus clear-oracle quantization.
                cross_bound = quantization + fixed_secure["max_abs"]
                for metric in ("max_abs", "mean_abs", "rmse"):
                    if abs(clear_secure[metric] - clear_fixed[metric]) > \
                            cross_bound + 1e-12 * max(
                                1.0, clear_secure[metric], clear_fixed[metric]):
                        raise AcceptanceError(
                            f"{path}.{name}.{metric} exceeds the "
                            "clear-oracle quantization cross-check")
                seen_aggregates.add(name)
                aggregate_rows += 1
                continue
            if name == "logits":
                if "intermediate_reconstruction" in secure:
                    _require_equal(secure["intermediate_reconstruction"], False,
                                   f"{path}.logits.intermediate_reconstruction")
                continue
            _require_equal(secure.get("status"), "NOT_AVAILABLE",
                           f"{path}.{name}.secure.status")
            _require_equal(secure.get("max_abs"), None,
                           f"{path}.{name}.secure.max_abs")
            _require_equal(secure.get("rmse"), None,
                           f"{path}.{name}.secure.rmse")
            # Fail closed if a future schema adds a numeric intermediate
            # observer under a different key without updating this gate.
            for key, value in secure.items():
                if key not in {"status", "max_abs", "rmse", "reason"} and \
                        isinstance(value, (int, float)) and not isinstance(value, bool):
                    raise AcceptanceError(
                        f"{path}.{name}.secure.{key} unexpectedly contains a number")
            intermediate_rows += 1
        if layerwise_status == "AGGREGATE_ONLY" and \
                seen_aggregates != expected_aggregates:
            raise AcceptanceError(
                f"{path} aggregate stage coverage mismatch")
        if layerwise_status == "AGGREGATE_ONLY" and blocks == 12:
            full12_aggregate_rows += len(seen_aggregates)
        verified.append(str(path))

    seeds = _load_json(seed_path)
    _require_equal(seeds.get("secure_intermediate_reconstruction"), False,
                   f"{seed_path}.secure_intermediate_reconstruction")
    _require_equal(seeds.get("first_inconsistent_layer"), UNAVAILABLE_LAYER,
                   f"{seed_path}.first_inconsistent_layer")
    verified.append(str(seed_path))
    full12_complete = full12_aggregate_rows == 27
    return {
        "status": "PASS" if full12_complete else "NOT_AVAILABLE",
        "reason": (
            "complete 27-stage aggregate-only secure error metrics; tensors were never reconstructed"
            if full12_complete else
            "complete 12-block aggregate secure errors were not supplied"),
        "verified_artifacts": verified,
        "intermediate_operator_rows": intermediate_rows,
        "aggregate_operator_rows": aggregate_rows,
        "full12_aggregate_operator_rows": full12_aggregate_rows,
        "intermediate_reconstruction": False,
        "first_inconsistent_layer": (
            first_nonzero_stage if full12_complete and first_nonzero_stage
            else ("NONE_EXACT_MATCH" if full12_complete else UNAVAILABLE_LAYER)),
        "first_nonzero_stage": first_nonzero_stage,
    }


def _check_run_metadata(path: Path, checkpoint_sha256: str) -> None:
    metadata = _load_json(path)
    model = _object(metadata.get("model"), f"{path}.model")
    expected_model = {
        "pre_norm": True, "layers": 12, "logical_tokens": 197,
        "physical_tokens": 256, "dim": 768, "heads": 12,
        "head_dim": 64, "ff_dim": 3072, "classes": 1000,
    }
    for key, expected in expected_model.items():
        _require_equal(model.get(key), expected, f"{path}.model.{key}")
    execution = _object(metadata.get("execution"), f"{path}.execution")
    _require_equal(execution.get("checkpoint_blocks"), 12,
                   f"{path}.execution.checkpoint_blocks")
    _require_equal(execution.get("executed_blocks"), 12,
                   f"{path}.execution.executed_blocks")
    _require_equal(execution.get("validation_block_limit"), False,
                   f"{path}.execution.validation_block_limit")
    checkpoint = _object(metadata.get("checkpoint"), f"{path}.checkpoint")
    _require_equal(checkpoint.get("sha256"), checkpoint_sha256,
                   f"{path}.checkpoint.sha256")
    fail_closed = _object(metadata.get("fail_closed"), f"{path}.fail_closed")
    for key in ("scalar_ciphertext", "generic_ot_matmul", "plain_matmul",
                "intermediate_reconstruction", "silent_fallback"):
        _require_equal(fail_closed.get(key), False, f"{path}.fail_closed.{key}")


def _check_full12(directory: Path | None, checkpoint_sha256: str,
                  secure_fixed_atol: float) -> dict[str, Any]:
    if directory is None:
        return {
            "status": "NOT_RUN",
            "reason": "--full12-secure-dir was not supplied",
        }
    if not directory.is_dir():
        raise AcceptanceError(f"full-12 secure directory is missing: {directory}")
    reference = directory / "reference_report.json"
    performance = directory / "performance.csv"
    breakdown = directory / "backend_breakdown.csv"
    metadata = directory / "run_metadata.json"
    result = _check_secure_run(reference, performance, breakdown, 12,
                               checkpoint_sha256, secure_fixed_atol)
    _check_run_metadata(metadata, checkpoint_sha256)
    result["run_metadata"] = str(metadata)
    return result


def _capture(function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return function()
    except Exception as error:
        # Artifact/schema failures must still produce a machine-readable FAIL
        # summary.  KeyboardInterrupt/SystemExit remain unsuppressed because
        # they derive directly from BaseException, not Exception.
        return {"status": "FAIL", "reason": str(error)}


def _detail(check: Mapping[str, Any]) -> str:
    status = check.get("status")
    if status in {"FAIL", "NOT_RUN", "NOT_AVAILABLE"}:
        return str(check.get("reason", ""))
    fields = []
    for key in ("top1", "clear_top1", "fixed_top1", "secure_top1",
                "cases", "samples", "expected_accumulators"):
        if key in check:
            fields.append(f"{key}={check[key]}")
    if "end_to_end" in check:
        fields.append(f"time_ms={check['end_to_end']['time_ms']}")
        fields.append(f"bytes={check['end_to_end']['bidirectional_bytes']}")
    return " ".join(fields)


def _validate_output_path(output: Path, inputs: Sequence[Path]) -> None:
    output_resolved = output.resolve(strict=False)
    for path in inputs:
        if output_resolved == path.resolve(strict=False):
            raise AcceptanceError(
                f"--output must not overwrite an input artifact: {path}")
    if output.exists() and output.is_dir():
        raise AcceptanceError(f"--output names a directory: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-reference", type=Path, required=True)
    parser.add_argument("--full-reference", type=Path, required=True)
    parser.add_argument("--bfv-audit", type=Path, required=True)
    parser.add_argument("--block1-reference", type=Path, required=True)
    parser.add_argument("--block1-performance", type=Path, required=True)
    parser.add_argument("--block1-breakdown", type=Path, required=True)
    parser.add_argument("--seed-diagnostics", type=Path, required=True)
    parser.add_argument("--imagenet50k-report", type=Path, required=True)
    parser.add_argument("--full12-secure-dir", type=Path,
                        help="optional completed real 12-block secure output directory")
    parser.add_argument("--pytorch-clear-atol", type=float, default=1e-4)
    parser.add_argument("--secure-fixed-atol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.pytorch_clear_atol) or args.pytorch_clear_atol < 0:
        parser.error("--pytorch-clear-atol must be finite and nonnegative")
    if not math.isfinite(args.secure_fixed_atol) or args.secure_fixed_atol < 0:
        parser.error("--secure-fixed-atol must be finite and nonnegative")

    input_paths = [
        args.pytorch_reference, args.full_reference, args.bfv_audit,
        args.block1_reference, args.block1_performance, args.block1_breakdown,
        args.seed_diagnostics, args.imagenet50k_report,
    ]
    if args.full12_secure_dir is not None:
        input_paths.extend(args.full12_secure_dir / name for name in (
            "reference_report.json", "performance.csv",
            "backend_breakdown.csv", "run_metadata.json"))
    try:
        _validate_output_path(args.output, input_paths)
    except AcceptanceError as error:
        parser.error(str(error))

    pytorch = _capture(lambda: _check_pytorch(args.pytorch_reference))
    checkpoint_sha = pytorch.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str):
        # This sentinel cannot accidentally match a real checkpoint digest and
        # makes dependent checks fail closed while preserving all diagnostics.
        checkpoint_sha = "INVALID_PYTORCH_CHECKPOINT"

    checks: dict[str, dict[str, Any]] = {
        "pytorch_reference": pytorch,
        "full_clear_fixed_reference": _capture(lambda: _check_full_reference(
            args.full_reference, args.pytorch_reference, args.pytorch_clear_atol)),
        "bfv_range_audit": _capture(lambda: _check_bfv_audit(args.bfv_audit)),
        "real_secure_block1": _capture(lambda: _check_secure_run(
            args.block1_reference, args.block1_performance,
            args.block1_breakdown, 1, checkpoint_sha, args.secure_fixed_atol)),
        "seed_regression": _capture(
            lambda: _check_seed_diagnostics(args.seed_diagnostics)),
        "imagenet50k": _capture(
            lambda: _check_imagenet(args.imagenet50k_report)),
        "full12_secure": _capture(lambda: _check_full12(
            args.full12_secure_dir, checkpoint_sha, args.secure_fixed_atol)),
    }
    privacy_paths = [args.full_reference, args.block1_reference]
    if (args.full12_secure_dir is not None and
            (args.full12_secure_dir / "reference_report.json").is_file()):
        privacy_paths.append(args.full12_secure_dir / "reference_report.json")
    checks["secure_intermediate_errors"] = _capture(
        lambda: _check_intermediate_privacy(privacy_paths,
                                            args.seed_diagnostics,
                                            args.secure_fixed_atol))

    order = (
        "pytorch_reference", "full_clear_fixed_reference", "bfv_range_audit",
        "real_secure_block1", "seed_regression", "imagenet50k",
        "secure_intermediate_errors", "full12_secure",
    )
    failures = [name for name in order if checks[name]["status"] == "FAIL"]
    not_run = [name for name in order
               if checks[name]["status"] in {"NOT_RUN", "NOT_AVAILABLE"}]
    overall = "FAIL" if failures else ("NOT_RUN" if not_run else "PASS")
    summary = {
        "format": SUMMARY_FORMAT,
        "scope": {
            "decision": "ViT-B/16 artifacts supplied to this command only",
            "excluded_from_overall_status": {
                "bert_dynamic_regression": "OUTSIDE_SCOPE",
                "real_checkpoint_full12_multi_image_seed_regression": "NOT_RUN",
                "imagenet50k_fixed_point_accuracy": "NOT_RUN",
                "imagenet50k_secure_accuracy": "NOT_RUN",
                "clear_fixed_layerwise_numeric_threshold": "NOT_DEFINED",
                "validation_oracle_physical_role_isolation": "DEPLOYMENT_REQUIRED",
            },
        },
        "overall_status": overall,
        "inputs": {
            "pytorch_reference": str(args.pytorch_reference),
            "full_reference": str(args.full_reference),
            "bfv_audit": str(args.bfv_audit),
            "block1_reference": str(args.block1_reference),
            "block1_performance": str(args.block1_performance),
            "block1_breakdown": str(args.block1_breakdown),
            "seed_diagnostics": str(args.seed_diagnostics),
            "imagenet50k_report": str(args.imagenet50k_report),
            "full12_secure_dir": (str(args.full12_secure_dir)
                                  if args.full12_secure_dir is not None else None),
        },
        "policy": {
            "pytorch_clear_atol": args.pytorch_clear_atol,
            "secure_fixed_atol": args.secure_fixed_atol,
            "operator_error_row_status_used_as_threshold": False,
            "secure_intermediate_error_requirement": (
                "AGGREGATE_ONLY_OR_EXPLICIT_NOT_AVAILABLE"),
            "breakdown_must_sum_to_online_inference": True,
        },
        "checks": {name: checks[name] for name in order},
        "failures": failures,
        "not_run": not_run,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")

    for name in order:
        label = name.upper()
        detail = _detail(checks[name])
        print(f"{label}={checks[name]['status']}" +
              (f" {detail}" if detail else ""))
    print(f"OVERALL={overall}")
    print(f"ACCEPTANCE_JSON={args.output}")
    if overall == "FAIL":
        return 1
    if overall == "NOT_RUN":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
