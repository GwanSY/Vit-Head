#!/usr/bin/env python3
"""Fail-closed BFV centered-range audit for ViT fixed-reference traces.

The secure BFV adapter evaluates integer accumulators modulo the plaintext
modulus.  A post-modulus value can look like a small numerical error even when
the mathematical accumulator wrapped.  This tool therefore accepts *only*
explicit, pre-modulus integer accumulators (or their exact integer min/max
statistics) and requires every expected linear/QK/Softmax-V/affine
accumulator to lie in the
centered interval for ``p = 536903681``.

``vit_reference.py --trace-dir DIR`` writes the auditable JSON/NPZ pair.
Its ordinary ``BOLT_VIT_REFERENCE_REPORT_V1`` output remains a
post-truncation numerical report and intentionally produces FAIL if supplied
here.  A range-auditable trace uses this schema::

    {
      "format": "BLB_VIT_FIXED_TRACE_V1",
      "plain_modulus": 536903681,
      "expected_accumulators": ["patch_projection", "block_0.qk"],
      "accumulators": [
        {
          "name": "patch_projection",
          "kind": "linear",
          "domain": "integer_raw_pre_modulus",
          "shape": [196, 768],
          "npz_key": "raw/patch_projection",
          "accumulator_scale": 24
        },
        {
          "name": "block_0.qk",
          "kind": "qk",
          "domain": "integer_raw_pre_modulus",
          "shape": [12, 256, 256],
          "raw_min": -123,
          "raw_max": 456,
          "elements": 786432,
          "accumulator_scale": 20
        }
      ],
      "arrays_file": "fixed_trace.npz"
    }

An NPZ may instead embed the same JSON object as a scalar UTF-8 string under
``__metadata__``.  NPZ accumulator arrays must have a signed-integer dtype;
float and unsigned/modular arrays are rejected rather than interpreted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PLAIN_MODULUS = 536_903_681
CENTERED_HALF = (PLAIN_MODULUS - 1) // 2
TRACE_FORMATS = {"BLB_VIT_FIXED_TRACE_V1", "BOLT_VIT_FIXED_TRACE_V1"}
AUDIT_FORMAT = "BLB_VIT_BFV_RANGE_AUDIT_V1"
RAW_DOMAINS = {"integer_raw_pre_modulus", "signed_integer_pre_modulus"}
KINDS = {"linear", "qk", "sv", "affine"}


class AuditInputError(RuntimeError):
    """The trace cannot establish the required pre-modulus range claim."""


@dataclass(frozen=True)
class RangeRow:
    name: str
    kind: str
    shape: str
    elements: int
    accumulator_scale: int | None
    raw_min: int
    raw_max: int
    centered_lower: int
    centered_upper: int
    peak_abs: int
    lower_headroom: int
    upper_headroom: int
    minimum_headroom: int
    utilization: float
    magnitude_headroom_bits: float | None
    modular_wrap: bool
    status: str
    reason: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.int64)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _declared_sha256(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise AuditInputError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AuditInputError(f"{field} must be an exact integer")
    return int(value)


def _shape(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AuditInputError(f"{field} must be an integer shape")
    result = tuple(_integer(item, field) for item in value)
    if not result or any(item <= 0 for item in result):
        raise AuditInputError(f"{field} must contain positive dimensions")
    return result


def _shape_text(shape: Sequence[int]) -> str:
    return "[" + ",".join(str(item) for item in shape) + "]"


def _metadata_from_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if "__metadata__" not in archive.files:
            sidecar = path.with_suffix(".json")
            if not sidecar.is_file():
                raise AuditInputError(
                    "NPZ trace needs scalar __metadata__ JSON or a same-stem JSON sidecar")
            return json.loads(sidecar.read_text(encoding="utf-8"))
        encoded = archive["__metadata__"]
        if encoded.ndim != 0:
            raise AuditInputError("NPZ __metadata__ must be a scalar JSON string")
        payload = encoded.item()
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if not isinstance(payload, str):
            raise AuditInputError("NPZ __metadata__ must contain UTF-8 JSON")
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise AuditInputError("NPZ __metadata__ root must be an object")
        return parsed


def _load_metadata(trace_path: Path) -> tuple[dict[str, Any], Path | None]:
    if trace_path.suffix.lower() == ".npz":
        return _metadata_from_npz(trace_path), trace_path
    if trace_path.suffix.lower() != ".json":
        raise AuditInputError("trace must be JSON or NPZ")
    parsed = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise AuditInputError("trace JSON root must be an object")
    arrays_name = parsed.get("arrays_file", parsed.get("npz_file"))
    arrays_path: Path | None = None
    if arrays_name is not None:
        if not isinstance(arrays_name, str) or Path(arrays_name).name != arrays_name:
            raise AuditInputError("arrays_file must be a basename next to the JSON trace")
        arrays_path = trace_path.parent / arrays_name
        if not arrays_path.is_file():
            raise AuditInputError(f"trace arrays file is missing: {arrays_path}")
    return parsed, arrays_path


def _entries(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("accumulators")
    if raw is None:
        fmt = metadata.get("format", "unknown")
        if fmt == "BOLT_VIT_REFERENCE_REPORT_V1":
            raise AuditInputError(
                "vit_reference report has no pre-modulus raw accumulators; "
                "post-truncation operators cannot prove BFV headroom")
        raise AuditInputError("trace has no accumulators list")
    if isinstance(raw, Mapping):
        result: list[dict[str, Any]] = []
        for name, value in raw.items():
            if not isinstance(value, Mapping):
                raise AuditInputError(f"accumulator {name!r} metadata must be an object")
            entry = dict(value)
            entry.setdefault("name", name)
            result.append(entry)
        return result
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise AuditInputError("accumulators must be a list or object")
    if not all(isinstance(item, Mapping) for item in raw):
        raise AuditInputError("every accumulator entry must be an object")
    return [dict(item) for item in raw]


def _expected_specs(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = metadata.get("expected_accumulators")
    if expected is None and isinstance(metadata.get("coverage"), Mapping):
        expected = metadata["coverage"].get("expected_accumulators")
    if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)):
        raise AuditInputError(
            "trace must enumerate expected_accumulators; observed entries alone "
            "cannot prove that every linear/QK/Softmax-V/affine accumulator "
            "was audited")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(expected):
        name = item.get("name") if isinstance(item, Mapping) else item
        if not isinstance(name, str) or not name:
            raise AuditInputError(f"expected_accumulators[{index}] has no valid name")
        specification = dict(item) if isinstance(item, Mapping) else {"name": name}
        result.append(specification)
    names = [item["name"] for item in result]
    if len(set(names)) != len(names):
        raise AuditInputError("expected_accumulators contains duplicate names")
    if not result:
        raise AuditInputError("expected_accumulators cannot be empty")
    return result


def _validate_expected_spec(specification: Mapping[str, Any],
                            entry: Mapping[str, Any]) -> None:
    name = specification["name"]
    if "kind" in specification and specification["kind"] != entry.get("kind"):
        raise AuditInputError(f"{name}: observed kind differs from expected graph")
    if "shape" in specification:
        expected_shape = _shape(specification["shape"], f"expected {name}.shape")
        actual_shape = _shape(entry.get("shape"), f"{name}.shape")
        if actual_shape != expected_shape:
            raise AuditInputError(f"{name}: observed shape differs from expected graph")
    if "accumulator_scale" in specification:
        expected_scale = _integer(specification["accumulator_scale"],
                                  f"expected {name}.accumulator_scale")
        actual_scale = _integer(entry.get("accumulator_scale"),
                                f"{name}.accumulator_scale")
        if actual_scale != expected_scale:
            raise AuditInputError(
                f"{name}: observed accumulator scale differs from expected graph")


def _validate_coverage(metadata: Mapping[str, Any], expected: int,
                       observed: int) -> None:
    coverage = metadata.get("coverage")
    if coverage is None:
        return
    if not isinstance(coverage, Mapping):
        raise AuditInputError("coverage must be an object")
    if coverage.get("complete") is not True:
        raise AuditInputError("coverage.complete must be true")
    for field, actual in (("expected", expected), ("observed", observed)):
        if field in coverage and _integer(coverage[field], f"coverage.{field}") != actual:
            raise AuditInputError(f"coverage.{field} does not match trace contents")
    for field in ("missing", "unexpected"):
        if field in coverage and coverage[field] != []:
            raise AuditInputError(f"coverage.{field} must be empty for a complete trace")


def _json_integer_values(value: Any) -> Iterable[int]:
    if isinstance(value, list):
        for item in value:
            yield from _json_integer_values(item)
        return
    yield _integer(value, "raw accumulator value")


def _array_stats(array: np.ndarray, name: str) -> tuple[int, int, int, tuple[int, ...]]:
    if array.dtype.kind != "i":
        raise AuditInputError(
            f"{name}: NPZ raw accumulator must use a signed-integer dtype; "
            f"got {array.dtype} (float or modular/unsigned data is forbidden)")
    if array.size == 0:
        raise AuditInputError(f"{name}: raw accumulator array is empty")
    return int(array.min()), int(array.max()), int(array.size), tuple(int(x) for x in array.shape)


def _entry_stats(entry: Mapping[str, Any], arrays: Any) -> tuple[int, int, int, tuple[int, ...]]:
    name = entry["name"]
    declared_shape = _shape(entry.get("shape"), f"{name}.shape")
    declared_elements = math.prod(declared_shape)
    npz_key = entry.get("npz_key")
    if npz_key is not None:
        if arrays is None:
            raise AuditInputError(f"{name}: npz_key is present but no NPZ arrays file was supplied")
        if not isinstance(npz_key, str) or not npz_key:
            raise AuditInputError(f"{name}: npz_key must be a non-empty string")
        if npz_key not in arrays.files:
            raise AuditInputError(f"{name}: NPZ key is missing: {npz_key}")
        array = arrays[npz_key]
        minimum, maximum, elements, actual_shape = _array_stats(array, name)
        if actual_shape != declared_shape:
            raise AuditInputError(
                f"{name}: NPZ shape {actual_shape} != declared shape {declared_shape}")
        if "dtype" in entry and entry["dtype"] != str(array.dtype):
            raise AuditInputError(
                f"{name}: NPZ dtype {array.dtype} != declared dtype {entry['dtype']!r}")
        for field, actual in (("raw_min", minimum), ("raw_max", maximum),
                              ("elements", elements)):
            if field in entry and _integer(entry[field], f"{name}.{field}") != actual:
                raise AuditInputError(f"{name}: declared {field} does not match NPZ data")
        if "raw_sha256" in entry:
            declared = _declared_sha256(entry["raw_sha256"],
                                       f"{name}.raw_sha256")
            if _sha256_array(array) != declared:
                raise AuditInputError(f"{name}: raw accumulator SHA-256 mismatch")
        return minimum, maximum, elements, actual_shape

    values = entry.get("raw", entry.get("raw_values"))
    if values is not None:
        iterator = iter(_json_integer_values(values))
        try:
            first = next(iterator)
        except StopIteration as error:
            raise AuditInputError(f"{name}: raw accumulator values are empty") from error
        minimum = maximum = first
        elements = 1
        for item in iterator:
            minimum = min(minimum, item)
            maximum = max(maximum, item)
            elements += 1
        if elements != declared_elements:
            raise AuditInputError(
                f"{name}: raw value count {elements} != shape product {declared_elements}")
        return minimum, maximum, elements, declared_shape

    if "raw_min" not in entry or "raw_max" not in entry:
        raise AuditInputError(
            f"{name}: requires npz_key, raw integer values, or exact raw_min/raw_max")
    minimum = _integer(entry["raw_min"], f"{name}.raw_min")
    maximum = _integer(entry["raw_max"], f"{name}.raw_max")
    elements = _integer(entry.get("elements"), f"{name}.elements")
    if minimum > maximum:
        raise AuditInputError(f"{name}: raw_min exceeds raw_max")
    if elements != declared_elements:
        raise AuditInputError(
            f"{name}: elements {elements} != shape product {declared_elements}")
    return minimum, maximum, elements, declared_shape


def _row(entry: Mapping[str, Any], arrays: Any) -> RangeRow:
    name = entry.get("name")
    kind = entry.get("kind")
    domain = entry.get("domain")
    if not isinstance(name, str) or not name:
        raise AuditInputError("accumulator entry has no valid name")
    if kind not in KINDS:
        raise AuditInputError(
            f"{name}: kind must be linear, qk, sv, or affine")
    if domain not in RAW_DOMAINS:
        raise AuditInputError(
            f"{name}: domain must explicitly be integer_raw_pre_modulus; "
            "modular/post-truncation values are not range evidence")
    scale_value = entry.get("accumulator_scale")
    scale = None if scale_value is None else _integer(scale_value, f"{name}.accumulator_scale")
    minimum, maximum, elements, shape = _entry_stats(entry, arrays)
    lower_headroom = minimum + CENTERED_HALF
    upper_headroom = CENTERED_HALF - maximum
    minimum_headroom = min(lower_headroom, upper_headroom)
    peak_abs = max(abs(minimum), abs(maximum))
    utilization = float(peak_abs) / float(CENTERED_HALF)
    magnitude_headroom_bits = (
        None if peak_abs == 0 else
        math.log2(CENTERED_HALF) - math.log2(peak_abs))
    wrap = minimum < -CENTERED_HALF or maximum > CENTERED_HALF
    return RangeRow(
        name=name,
        kind=kind,
        shape=_shape_text(shape),
        elements=elements,
        accumulator_scale=scale,
        raw_min=minimum,
        raw_max=maximum,
        centered_lower=-CENTERED_HALF,
        centered_upper=CENTERED_HALF,
        peak_abs=peak_abs,
        lower_headroom=lower_headroom,
        upper_headroom=upper_headroom,
        minimum_headroom=minimum_headroom,
        utilization=utilization,
        magnitude_headroom_bits=magnitude_headroom_bits,
        modular_wrap=wrap,
        status="FAIL" if wrap else "PASS",
        reason=(
            "raw_accumulator_exceeds_centered_plaintext_interval_modular_wrap"
            if wrap else "within_centered_plaintext_interval"),
    )


def audit(trace_path: Path) -> tuple[dict[str, Any], list[RangeRow]]:
    metadata, arrays_path = _load_metadata(trace_path)
    fmt = metadata.get("format")
    if fmt not in TRACE_FORMATS:
        if fmt == "BOLT_VIT_REFERENCE_REPORT_V1":
            raise AuditInputError(
                "vit_reference report has no pre-modulus raw accumulators; "
                "generate a fixed trace JSON/NPZ instead")
        raise AuditInputError(f"unsupported fixed trace format: {fmt!r}")
    modulus = _integer(metadata.get("plain_modulus"), "plain_modulus")
    if modulus != PLAIN_MODULUS:
        raise AuditInputError(
            f"plain_modulus must be exactly {PLAIN_MODULUS}, got {modulus}")
    if "arrays_sha256" in metadata:
        if arrays_path is None:
            raise AuditInputError("arrays_sha256 is present but no arrays file was supplied")
        declared_archive_hash = _declared_sha256(metadata["arrays_sha256"],
                                                 "arrays_sha256")
        if _sha256(arrays_path) != declared_archive_hash:
            raise AuditInputError("BFV accumulator archive SHA-256 mismatch")
    expected_specs = _expected_specs(metadata)
    expected = [item["name"] for item in expected_specs]
    entries = _entries(metadata)
    names = [entry.get("name") for entry in entries]
    if any(not isinstance(name, str) or not name for name in names):
        raise AuditInputError("every accumulator entry must have a valid name")
    if len(set(names)) != len(names):
        raise AuditInputError("accumulator entries contain duplicate names")
    missing = [name for name in expected if name not in set(names)]
    if missing:
        raise AuditInputError("expected accumulators are missing: " + ", ".join(missing))
    unexpected = [name for name in names if name not in set(expected)]
    if unexpected:
        raise AuditInputError(
            "unexpected accumulators are not covered by the expected graph: " +
            ", ".join(unexpected))
    entry_by_name = {entry["name"]: entry for entry in entries}
    ordered_entries = [entry_by_name[name] for name in expected]
    for specification, entry in zip(expected_specs, ordered_entries):
        _validate_expected_spec(specification, entry)
    _validate_coverage(metadata, len(expected), len(ordered_entries))

    arrays = np.load(arrays_path, allow_pickle=False) if arrays_path else None
    try:
        rows = [_row(entry, arrays) for entry in ordered_entries]
    finally:
        if arrays is not None:
            arrays.close()
    failed = [row.name for row in rows if row.status != "PASS"]
    report = {
        "format": AUDIT_FORMAT,
        "status": "FAIL" if failed else "PASS",
        "source_trace": str(trace_path),
        "source_trace_sha256": _sha256(trace_path),
        "arrays_file": str(arrays_path) if arrays_path else None,
        "arrays_sha256": _sha256(arrays_path) if arrays_path else None,
        "plain_modulus": PLAIN_MODULUS,
        "centered_interval": [-CENTERED_HALF, CENTERED_HALF],
        "coverage": {
            "expected": len(expected),
            "observed": len(rows),
            "missing": [],
            "complete": True,
        },
        "failed_accumulators": failed,
        "first_failed_accumulator": failed[0] if failed else None,
        "note": (
            "FAIL denotes mathematical pre-modulus wrap risk; it is never "
            "downgraded to numerical error."),
        "accumulators": [asdict(row) for row in rows],
    }
    return report, rows


CSV_FIELDS = [field.name for field in RangeRow.__dataclass_fields__.values()]


def _write_csv(path: Path, rows: Sequence[RangeRow], failure: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
        if failure is not None:
            empty = {field: "" for field in CSV_FIELDS}
            empty.update({"name": "__trace__", "kind": "metadata",
                          "status": "FAIL", "reason": failure})
            writer.writerow(empty)


def _write_json(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _default_outputs(trace: Path) -> tuple[Path, Path]:
    stem = trace.with_suffix("")
    return (stem.parent / (stem.name + ".bfv_ranges.csv"),
            stem.parent / (stem.name + ".bfv_ranges.json"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit raw ViT BFV accumulators against the centered plaintext range")
    parser.add_argument("--trace", type=Path, required=True,
                        help="fixed trace JSON or NPZ")
    parser.add_argument("--csv", type=Path, help="CSV output path")
    parser.add_argument("--json", dest="json_output", type=Path,
                        help="JSON output path")
    args = parser.parse_args(argv)
    default_csv, default_json = _default_outputs(args.trace)
    csv_path = args.csv or default_csv
    json_path = args.json_output or default_json
    try:
        report, rows = audit(args.trace)
    except (AuditInputError, OSError, ValueError, json.JSONDecodeError) as error:
        reason = str(error)
        report = {
            "format": AUDIT_FORMAT,
            "status": "FAIL",
            "source_trace": str(args.trace),
            "plain_modulus": PLAIN_MODULUS,
            "centered_interval": [-CENTERED_HALF, CENTERED_HALF],
            "schema_error": reason,
            "failed_accumulators": [],
            "first_failed_accumulator": None,
            "accumulators": [],
            "note": (
                "No range claim is made without exact pre-modulus integer "
                "accumulators and complete coverage metadata."),
        }
        _write_csv(csv_path, [], failure=reason)
        _write_json(json_path, report)
        print(f"BFV_RANGE_AUDIT=FAIL reason={reason}")
        print(f"BFV_RANGE_CSV={csv_path}")
        print(f"BFV_RANGE_JSON={json_path}")
        return 1
    _write_csv(csv_path, rows)
    _write_json(json_path, report)
    print(f"BFV_RANGE_AUDIT={report['status']} accumulators={len(rows)} "
          f"failed={len(report['failed_accumulators'])} "
          f"first_failed={report['first_failed_accumulator'] or 'NONE'}")
    print(f"BFV_RANGE_CSV={csv_path}")
    print(f"BFV_RANGE_JSON={json_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
