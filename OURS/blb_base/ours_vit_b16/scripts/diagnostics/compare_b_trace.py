#!/usr/bin/env python3
"""Compare the real B trace with the FP32 and fixed-clear graphs."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import vit_reference as vr


TRACE_MAGIC = "OURS_VIT_B_STAGE_V1"
# The existing attention-trace checker uses these tolerances for the
# fixed-clear/secure boundary.  Keep the raw errors in the report; the
# tolerance is only used for the PASS/FAIL column.
SECURE_REL_L2_TOLERANCE = 5e-2
SECURE_MAX_ABS_TOLERANCE = 5e-1


def load_trace(path: Path) -> tuple[np.ndarray, int, int, tuple[int, ...], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != TRACE_MAGIC:
        raise ValueError(f"unsupported B trace: {path}")
    cursor = 1

    def field(name: str) -> str:
        nonlocal cursor
        prefix = name + " "
        if cursor >= len(lines) or not lines[cursor].startswith(prefix):
            raise ValueError(f"trace field {name!r} is missing: {path}")
        value = lines[cursor][len(prefix):]
        cursor += 1
        return value

    stage = field("stage")
    elements = int(field("elements"))
    ell = int(field("ell"))
    scale = int(field("scale"))
    shape = tuple(int(value) for value in field("shape").split())
    if cursor >= len(lines) or lines[cursor] != "values":
        raise ValueError(f"trace values marker is missing: {path}")
    cursor += 1
    if elements != math.prod(shape) or cursor + elements >= len(lines):
        raise ValueError(f"trace shape/element count mismatch: {path}")
    raw = np.asarray(lines[cursor:cursor + elements], dtype=np.int64)
    cursor += elements
    if cursor != len(lines) - 1 or lines[cursor] != "end":
        raise ValueError(f"trace terminator is invalid: {path}")
    return raw.reshape(shape), ell, scale, shape, stage


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {actual.shape}")
    delta = actual - reference
    squared_error = float(np.sum(delta * delta, dtype=np.float64))
    squared_reference = float(np.sum(reference * reference, dtype=np.float64))
    return {
        "elements": int(delta.size),
        "relative_l2": (math.sqrt(squared_error / squared_reference)
                        if squared_reference else
                        (0.0 if squared_error == 0.0 else math.inf)),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
    }


def valid_values(name: str, value: np.ndarray, logical: int) -> np.ndarray:
    if name in {"q", "k", "v", "attention_residual", "fc1", "gelu",
                "ffn_residual", "final_ln"}:
        return value[:logical]
    if name in {"qk", "softmax"}:
        return value[:, :logical, :logical]
    return value


def secure_logits(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines[:1] != ["BLB_VIT_SECURE_LOGITS_V1"]:
        raise ValueError(f"unsupported secure logits: {path}")
    start = lines.index("values") + 1
    end = lines.index("end")
    return np.asarray(lines[start:end], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--secure-logits", type=Path, required=True)
    parser.add_argument("--secure-prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete-gate", action="store_true")
    parser.add_argument("--block-limit", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()

    model = vr.load_model(args.model)
    pixels = vr.load_input(args.input)
    clear_logits, clear_records = vr.Graph(
        model, fixed=False, physical_tokens=256, block_limit=args.block_limit,
        stop_after_attention=False).run(pixels)
    fixed_logits, fixed_records = vr.Graph(
        model, fixed=True, physical_tokens=256, block_limit=args.block_limit,
        stop_after_attention=False).run(pixels)
    trace_names = {
        "q": ("block_0.q", "block_0_q"),
        "k": ("block_0.k", "block_0_k"),
        "v": ("block_0.v", "block_0_v"),
        "qk": ("block_0.qk", "block_0_qk"),
        "softmax": ("block_0.softmax", "block_0_softmax"),
        "attention_residual": ("block_0.attention_residual",
                               "block_0_attention_residual"),
        "fc1": ("block_0.fc1", "block_0_fc1"),
        "gelu": ("block_0.gelu", "block_0_gelu"),
        "ffn_residual": ("block_0.ffn_residual", "block_0_ffn_residual"),
        "final_ln": ("final_ln", "final_ln"),
    }
    output_names = {
        "AV": ("block_0.attention_context", "block_0_av"),
        "WO": ("block_0.output_projection", "block_0_wo"),
    }
    target_block = f"block_{args.block_limit - 1}"
    trace_names = {label: tuple(name.replace("block_0", target_block)
                               for name in names)
                   for label, names in trace_names.items()}
    output_names = {label: tuple(name.replace("block_0", target_block)
                                for name in names)
                    for label, names in output_names.items()}
    rows: list[dict[str, object]] = []
    logical = int(model["tokens"])
    for label, (record_name, trace_name) in trace_names.items():
        path = args.trace_dir / f"b_{trace_name}_party1.txt"
        raw, _, scale, shape, _ = load_trace(path)
        if not (np.array_equal(raw, load_trace(
                args.trace_dir / f"b_{trace_name}_party2.txt")[0])):
            raise ValueError(f"party trace mismatch: {label}")
        fixed = valid_values(label, fixed_records[record_name]["value"], logical)
        clear = valid_values(label, clear_records[record_name]["value"], logical)
        # The fixed graph folds 1/sqrt(head_dim) into Q before recording it;
        # the FP32 graph applies the same factor when forming QK.  Compare
        # the recorded tensors in the same convention.
        if label == "q":
            clear = clear / math.sqrt(64.0)
        secure = valid_values(label, raw.astype(np.float64) / (1 << scale), logical)
        clear_fixed = metrics(clear, fixed)
        fixed_secure = metrics(fixed, secure)
        rows.append({
            "stage": (label.upper() if label not in
                      {"attention_residual", "ffn_residual", "final_ln"}
                      else label),
            "shape_packing": "x".join(str(value) for value in shape),
            "fp32_fixed": clear_fixed,
            "fixed_secure": fixed_secure,
            "status": ("PASS" if fixed_secure["relative_l2"] <= SECURE_REL_L2_TOLERANCE
                        and fixed_secure["max_abs"] <= SECURE_MAX_ABS_TOLERANCE else "FAIL"),
        })
    for label, (record_name, trace_name) in output_names.items():
        fixed = fixed_records[record_name]["value"][:logical]
        clear = clear_records[record_name]["value"][:logical]
        path = args.trace_dir / f"b_{trace_name}_party1.txt"
        raw, _, scale, shape, _ = load_trace(path)
        if not np.array_equal(raw, load_trace(
                args.trace_dir / f"b_{trace_name}_party2.txt")[0]):
            raise ValueError(f"party trace mismatch: {label}")
        secure_stage = raw.astype(np.float64) / (1 << scale)
        secure_stage = secure_stage[:logical]
        fixed_secure = metrics(fixed, secure_stage)
        rows.append({
            "stage": label,
            "shape_packing": "x".join(str(value) for value in shape),
            "fp32_fixed": metrics(clear, fixed),
            "fixed_secure": fixed_secure,
            "status": ("PASS" if
                       fixed_secure["relative_l2"] <= SECURE_REL_L2_TOLERANCE
                       and fixed_secure["max_abs"] <= SECURE_MAX_ABS_TOLERANCE
                       else "FAIL"),
        })
    secure = secure_logits(args.secure_logits)
    fixed_logit_metrics = metrics(fixed_logits, secure)
    rows.append({
        "stage": "logits",
        "shape_packing": "1x1000 CLS-only",
        "fp32_fixed": metrics(clear_logits, fixed_logits),
        "fixed_secure": fixed_logit_metrics,
        "status": ("PASS" if fixed_logit_metrics["relative_l2"] <= SECURE_REL_L2_TOLERANCE
                    and fixed_logit_metrics["max_abs"] <= SECURE_MAX_ABS_TOLERANCE else "FAIL"),
    })
    paper_order = {
        name: index for index, name in enumerate((
            "Q", "K", "V", "QK", "SOFTMAX", "AV", "WO",
            "attention_residual", "FC1", "GELU", "ffn_residual",
            "final_ln", "logits"))
    }
    rows.sort(key=lambda row: paper_order[str(row["stage"])])
    prediction = args.secure_prediction.read_text(encoding="utf-8").split()
    secure_top1 = int(prediction[1])
    fixed_secure_gate = "PASS" if all(
        row["status"] in {"PASS", "N/A"} for row in rows) else "FAIL"
    fp32_top1 = int(np.argmax(clear_logits))
    fixed_top1 = int(np.argmax(fixed_logits))
    secure_fixed_top1_agreement = secure_top1 == fixed_top1
    fp32_fixed_top1_agreement = fp32_top1 == fixed_top1
    complete_gate = (fixed_secure_gate == "PASS" and
                     secure_fixed_top1_agreement and
                     fp32_fixed_top1_agreement)
    result = {
        "format": "OURS_B_TRACE_COMPARISON_V2",
        "trace_dir": str(args.trace_dir.resolve()),
        "rows": rows,
        "top1": {
            "fp32": fp32_top1,
            "fixed": fixed_top1,
            "secure": secure_top1,
        },
        "fixed_secure_gate": fixed_secure_gate,
        "secure_fixed_top1_agreement": secure_fixed_top1_agreement,
        "fp32_fixed_top1_agreement": fp32_fixed_top1_agreement,
        "complete_b_gate": "PASS" if complete_gate else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        columns = (
            "stage", "shape_packing", "fp32_fixed_relative_l2",
            "fp32_fixed_max_abs", "fixed_secure_relative_l2",
            "fixed_secure_max_abs", "status")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "stage": row["stage"],
                "shape_packing": row["shape_packing"],
                "fp32_fixed_relative_l2":
                    row["fp32_fixed"]["relative_l2"],
                "fp32_fixed_max_abs": row["fp32_fixed"]["max_abs"],
                "fixed_secure_relative_l2":
                    row["fixed_secure"]["relative_l2"],
                "fixed_secure_max_abs": row["fixed_secure"]["max_abs"],
                "status": row["status"],
            })
    print("OURS_B_TRACE_COMPARISON "
          f"fixed_secure={fixed_secure_gate} "
          f"complete={result['complete_b_gate']}")
    print(f"OURS_B_TRACE_COMPARISON_JSON={args.output}")
    print(f"OURS_B_TRACE_COMPARISON_CSV={csv_path}")
    print("OURS_B_TOP1 " + " ".join(
        f"{name}={value}" for name, value in result["top1"].items()))
    if args.require_complete_gate and not complete_gate:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
