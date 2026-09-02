#!/usr/bin/env python3
"""Compare a real A3/A4 secure attention result with fixed-clear ViT.

The secure trace is opened only after QK, normalization, AV, WO, and the
attention residual have completed.  It is a validation sink: its values are
never re-encrypted and never feed the secure execution graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import vit_reference as vr


TRACE_MAGIC = "OURS_VIT_HETERO_ATTENTION_STAGE_V1"
PLAN_MAGIC = "OURS_VIT_HEAD_PLAN_V1"
PLAN_SHAPE = "layers=12,heads=12,logical_tokens=197,physical_tokens=256"


def load_plan(path: Path, layers: int,
              heads: int) -> list[list[tuple[str, float]]]:
    lines = [line.strip() for line in path.read_text(
        encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) < 3 or lines[0] != PLAN_MAGIC or lines[1] != PLAN_SHAPE:
        raise ValueError("unsupported head plan header")
    reader = csv.DictReader(lines[2:])
    required = {"layer", "physical_head", "original_head", "protocol",
                "alpha"}
    if set(reader.fieldnames or ()) != required:
        raise ValueError("head plan columns changed")
    plan = [[("", math.nan) for _ in range(heads)] for _ in range(layers)]
    physical_seen = [set() for _ in range(layers)]
    for row in reader:
        layer = int(row["layer"])
        physical = int(row["physical_head"])
        original = int(row["original_head"])
        if (not 0 <= layer < layers or not 0 <= physical < heads or
                not 0 <= original < heads or
                physical in physical_seen[layer] or plan[layer][original][0]):
            raise ValueError("invalid or duplicate head-plan row")
        physical_seen[layer].add(physical)
        plan[layer][original] = (row["protocol"], float(row["alpha"]))
    if any(len(row) != heads for row in physical_seen) or any(
            not protocol for layer in plan for protocol, _ in layer):
        raise ValueError("head plan does not cover every layer/head")
    return plan


def load_trace(path: Path) -> tuple[np.ndarray, int, int, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != TRACE_MAGIC:
        raise ValueError("unsupported heterogeneous attention trace")
    cursor = 1

    def field(name: str) -> str:
        nonlocal cursor
        prefix = name + " "
        if cursor >= len(lines) or not lines[cursor].startswith(prefix):
            raise ValueError(f"trace field {name!r} is missing")
        value = lines[cursor][len(prefix):]
        cursor += 1
        return value

    stage = field("stage")
    elements = int(field("elements"))
    ell = int(field("ell"))
    scale = int(field("scale"))
    shape = tuple(int(item) for item in field("shape").split())
    if cursor >= len(lines) or lines[cursor] != "values":
        raise ValueError("trace values marker is missing")
    cursor += 1
    if elements != math.prod(shape) or cursor + elements >= len(lines):
        raise ValueError("trace shape/element count mismatch")
    raw = np.asarray(lines[cursor:cursor + elements], dtype=np.int64)
    cursor += elements
    if cursor != len(lines) - 1 or lines[cursor] != "end":
        raise ValueError("trace terminator or trailing data is invalid")
    return raw.reshape(shape), ell, scale, stage


def error_metrics(reference: np.ndarray,
                  actual: np.ndarray) -> dict[str, float | int]:
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64)
    squared_error = float(np.sum(delta * delta, dtype=np.float64))
    squared_reference = float(np.sum(
        np.asarray(reference, dtype=np.float64) ** 2, dtype=np.float64))
    return {
        "elements": int(delta.size),
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": math.sqrt(squared_error / delta.size),
        "relative_l2": (math.sqrt(squared_error / squared_reference)
                        if squared_reference > 0.0 else
                        (0.0 if squared_error == 0.0 else math.inf)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--head-plan", type=Path, required=True)
    parser.add_argument(
        "--secure-input-trace", type=Path,
        help=("optional opened block input; when supplied, validates A3/A4 "
              "in isolation from the upstream A0 prefix"))
    parser.add_argument("--secure-trace", type=Path, required=True)
    parser.add_argument("--peer-input-trace", type=Path)
    parser.add_argument("--peer-trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-relative-l2", type=float, default=0.05)
    parser.add_argument("--max-abs", type=float, default=0.5)
    parser.add_argument("--max-padding-abs", type=float, default=0.0)
    args = parser.parse_args()
    for name in ("max_relative_l2", "max_abs", "max_padding_abs"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")

    model = vr.load_model(args.model)
    pixels = vr.load_input(args.input)
    layers = int(model["layers"])
    heads = int(model["heads"])
    logical_tokens = int(model["tokens"])
    plan = load_plan(args.head_plan, layers, heads)
    graph = vr.Graph(model, fixed=True, physical_tokens=256, block_limit=1,
                     head_plan=plan, stop_after_attention=True)
    isolated_from_secure_input = args.secure_input_trace is not None
    secure_input: np.ndarray | None = None
    input_peer_equal: bool | None = None
    if args.secure_input_trace is None:
        _, records = graph.run(pixels)
        reference = np.asarray(
            records["block_0.attention_residual"]["value"], dtype=np.float64)
    else:
        input_raw, input_ell, input_scale, input_stage = load_trace(
            args.secure_input_trace)
        if input_stage != "block_0_input_residual":
            raise ValueError("secure input trace is not the block-0 input residual")
        if input_raw.shape != (256, int(model["dim"])):
            raise ValueError("secure input trace has an invalid shape")
        secure_input = (input_raw.astype(np.float64) /
                        float(1 << input_scale))
        block = model["blocks"][0]
        normalized = graph.norm(
            secure_input, block["ln1_weight"], block["ln1_bias"],
            "block_0.ln1_affine")
        attention = graph.attention(normalized, block, 0)
        reference = graph.mask_tokens(
            graph.add(secure_input, attention, vr.ACT_SCALE))
        if args.peer_input_trace is not None:
            peer_input_raw, peer_input_ell, peer_input_scale, peer_input_stage = (
                load_trace(args.peer_input_trace))
            input_peer_equal = (
                peer_input_ell == input_ell and
                peer_input_scale == input_scale and
                peer_input_stage == input_stage and
                np.array_equal(peer_input_raw, input_raw))

    raw, ell, scale, stage = load_trace(args.secure_trace)
    if stage != "block_0_attention_residual":
        raise ValueError("secure trace is not the block-0 attention residual")
    if raw.shape != reference.shape:
        raise ValueError(
            f"secure trace shape {raw.shape} != reference {reference.shape}")
    secure = raw.astype(np.float64) / float(1 << scale)
    if secure_input is None:
        logical_reference = reference[:logical_tokens]
        logical_secure = secure[:logical_tokens]
        compared_tensor = "attention_residual"
    else:
        logical_reference = (reference[:logical_tokens] -
                             secure_input[:logical_tokens])
        logical_secure = (secure[:logical_tokens] -
                          secure_input[:logical_tokens])
        compared_tensor = "attention_contribution=residual-output_minus-input"
    logical = error_metrics(logical_reference, logical_secure)
    padding_max_abs = float(np.max(
        np.abs(secure[logical_tokens:]), initial=0.0))
    peer_equal: bool | None = None
    if args.peer_trace is not None:
        peer_raw, peer_ell, peer_scale, peer_stage = load_trace(args.peer_trace)
        peer_equal = (peer_ell == ell and peer_scale == scale and
                      peer_stage == stage and
                      np.array_equal(peer_raw, raw))

    failures = []
    if float(logical["relative_l2"]) > args.max_relative_l2:
        failures.append("logical_relative_l2")
    if float(logical["max_abs"]) > args.max_abs:
        failures.append("logical_max_abs")
    # A secure-input comparison deliberately isolates A3/A4 from the paused
    # A0 prefix.  Preserve and report its padded-row defect, but do not charge
    # that upstream defect to the isolated attention-path gate.
    if not isolated_from_secure_input and padding_max_abs > args.max_padding_abs:
        failures.append("padding_max_abs")
    if input_peer_equal is False:
        failures.append("party_input_trace_mismatch")
    if peer_equal is False:
        failures.append("party_trace_mismatch")
    report = {
        "format": "OURS_VIT_HETERO_ATTENTION_ACCEPTANCE_V1",
        "status": "PASS" if not failures else "FAIL",
        "stage": stage,
        "model": str(args.model.resolve()),
        "input": str(args.input.resolve()),
        "head_plan": str(args.head_plan.resolve()),
        "secure_input_trace": (
            str(args.secure_input_trace.resolve())
            if args.secure_input_trace is not None else None),
        "secure_trace": str(args.secure_trace.resolve()),
        "peer_trace": (str(args.peer_trace.resolve())
                       if args.peer_trace is not None else None),
        "fixed_clear_reference": "vit_reference.Graph(fixed=True)",
        "reference_scope": (
            "A3/A4 attention isolated from opened secure block input"
            if isolated_from_secure_input else
            "canonical input through complete fixed-clear prefix"),
        "compared_tensor": compared_tensor,
        "logical_tokens": logical_tokens,
        "physical_tokens": int(raw.shape[0]),
        "ell": ell,
        "scale": scale,
        "logical_metrics": logical,
        "padding_max_abs": padding_max_abs,
        "padding_gate_applied": not isolated_from_secure_input,
        "party_input_traces_equal": input_peer_equal,
        "party_traces_equal": peer_equal,
        "thresholds": {
            "max_relative_l2": args.max_relative_l2,
            "max_abs": args.max_abs,
            "max_padding_abs": args.max_padding_abs,
        },
        "failures": failures,
        "validation_only": True,
        "feeds_secure_graph": False,
        "client_decrypt_reencrypt": False,
        "mpc_surrogate_for_linear": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print("OURS_HETERO_ATTENTION_ACCEPTANCE "
          f"status={report['status']} relative_l2={logical['relative_l2']:.9g} "
          f"max_abs={logical['max_abs']:.9g} "
          f"padding_max_abs={padding_max_abs:.9g}")
    print(f"OURS_HETERO_ATTENTION_ACCEPTANCE_REPORT={args.output}")
    if failures:
        raise SystemExit("heterogeneous attention acceptance failed: " +
                         ",".join(failures))


if __name__ == "__main__":
    main()
