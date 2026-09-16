#!/usr/bin/env python3
"""Compare validation-only Packed/Unpacked secure attention traces."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


MAGIC = "OURS_VIT_HETERO_ATTENTION_STAGE_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != MAGIC:
        raise ValueError(f"unsupported heterogeneous trace: {path}")
    fields: dict[str, str] = {}
    cursor = 1
    while cursor < len(lines) and lines[cursor] != "values":
        key, value = lines[cursor].split(" ", 1)
        fields[key] = value
        cursor += 1
    if cursor >= len(lines) or lines[cursor] != "values":
        raise ValueError(f"missing values marker: {path}")
    cursor += 1
    elements = int(fields["elements"])
    shape = tuple(int(value) for value in fields["shape"].split())
    if math.prod(shape) != elements or cursor + elements >= len(lines):
        raise ValueError(f"invalid trace shape: {path}")
    raw = np.asarray(lines[cursor:cursor + elements], dtype=np.int64)
    if lines[cursor + elements:] != ["end"]:
        raise ValueError(f"invalid trace terminator: {path}")
    return raw.reshape(shape).astype(np.float64) / (1 << int(fields["scale"]))


def load_party_checked(run: Path, stage: str) -> np.ndarray:
    first = run / f"hetero_{stage}_party1.txt"
    second = run / f"hetero_{stage}_party2.txt"
    if not first.is_file() or not second.is_file():
        raise ValueError(f"missing {stage} trace in {run}")
    left = load_stage(first)
    right = load_stage(second)
    if not np.array_equal(left, right):
        raise ValueError(f"party trace disagreement for {stage}: {run}")
    return left


def load_logits(run: Path) -> np.ndarray:
    values = []
    for party in (1, 2):
        lines = (run / f"secure_logits_party{party}.txt").read_text(
            encoding="utf-8").splitlines()
        if lines[:1] != ["BLB_VIT_SECURE_LOGITS_V1"]:
            raise ValueError(f"invalid secure logits in {run}")
        start = lines.index("values") + 1
        end = lines.index("end")
        values.append(np.asarray(lines[start:end], dtype=np.float64))
    if not np.array_equal(values[0], values[1]):
        raise ValueError(f"party logits disagreement: {run}")
    return values[0]


def error(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {actual.shape}")
    delta = actual - reference
    denominator = float(np.sum(reference * reference, dtype=np.float64))
    numerator = float(np.sum(delta * delta, dtype=np.float64))
    return {
        "elements": int(delta.size),
        "relative_l2": (math.sqrt(numerator / denominator) if denominator
                        else (0.0 if numerator == 0.0 else math.inf)),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--unpacked", type=Path, required=True)
    parser.add_argument("--head-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-l2-tolerance", type=float, default=5e-2)
    parser.add_argument("--max-abs-tolerance", type=float, default=5e-1)
    args = parser.parse_args()

    rows = []
    for label, stage in (("QK", "block_0_qk"),
                         ("Norm", "block_0_norm"),
                         ("AV", "block_0_av"),
                         ("WO", "block_0_wo"),
                         ("attention_residual", "block_0_attention_residual")):
        metrics = error(load_party_checked(args.packed, stage),
                        load_party_checked(args.unpacked, stage))
        rows.append({
            "stage": label,
            **metrics,
            "status": ("PASS" if
                       metrics["relative_l2"] <= args.relative_l2_tolerance and
                       metrics["max_abs"] <= args.max_abs_tolerance else "FAIL"),
        })
    logits = error(load_logits(args.packed), load_logits(args.unpacked))
    rows.append({
        "stage": "logits",
        **logits,
        "status": ("PASS" if
                   logits["relative_l2"] <= args.relative_l2_tolerance and
                   logits["max_abs"] <= args.max_abs_tolerance else "FAIL"),
    })
    result = {
        "format": "OURS_VIT_PACKING_TRACE_COMPARISON_V1",
        "status": "PASS" if all(row["status"] == "PASS" for row in rows)
                  else "FAIL",
        "assignment": str(args.head_plan.resolve()),
        "assignment_sha256": sha256(args.head_plan),
        "packed_run": str(args.packed.resolve()),
        "unpacked_run": str(args.unpacked.resolve()),
        "trace_is_timing_eligible": False,
        "relative_l2_tolerance": args.relative_l2_tolerance,
        "max_abs_tolerance": args.max_abs_tolerance,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(f"OURS_PACKING_TRACE_COMPARISON status={result['status']}")
    print(f"OURS_PACKING_TRACE_COMPARISON_JSON={args.output}")


if __name__ == "__main__":
    main()
