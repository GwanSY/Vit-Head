#!/usr/bin/env python3
"""Validate A0 traces against FP32 and fixed oracles without NumPy.

The trace run is intentionally separate from performance measurement.  Only
the first 65 logical tokens (and key columns) participate in reported errors.
The standard-library NPY reader keeps this gate runnable in the original BLB
GPU container, whose Python 3.6 image does not include NumPy.
"""

import argparse
import ast
import json
import math
import struct
import zipfile
from pathlib import Path


TOKENS = 65
PHYSICAL_TOKENS = 128
HEADS = 4
HEAD_DIM = 64
DIM = HEADS * HEAD_DIM

# Protocol-level acceptance gates fixed before a run.  Max error catches an
# isolated layout/conversion failure; MSE catches broad numerical drift.  The
# probability tensor is stricter because its semantic range is [0, 1].
SECURE_FIXED_LIMITS = {
    "q": (0.5, 1.0e-3),
    "k": (0.5, 1.0e-3),
    "v": (0.5, 1.0e-3),
    "qk": (0.5, 1.0e-3),
    "softmax": (0.125, 1.0e-4),
    "av": (0.125, 1.0e-3),
    "wo": (0.25, 1.0e-3),
}
HETERO_SECURE_FIXED_LIMITS = {
    "q": (0.5, 1.0e-3),
    "k": (0.5, 1.0e-3),
    "v": (0.5, 1.0e-3),
    "qk": (0.5, 1.0e-3),
    "normalization": (0.125, 1.0e-4),
    "av": (0.25, 1.0e-3),
    "wo": (0.5, 4.0e-3),
}


def read_trace(path, expected):
    values = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            text = line.strip()
            if text:
                values.append(int(text))
    if len(values) != expected:
        raise RuntimeError("{}: expected {}, got {}".format(
            path, expected, len(values)))
    return values


def read_npy(raw, member):
    if raw[:6] != b"\x93NUMPY" or len(raw) < 10:
        raise RuntimeError("{}: invalid NPY payload".format(member))
    major = raw[6]
    if major == 1:
        header_size = struct.unpack_from("<H", raw, 8)[0]
        payload_offset = 10 + header_size
        header_offset = 10
    elif major in (2, 3):
        header_size = struct.unpack_from("<I", raw, 8)[0]
        payload_offset = 12 + header_size
        header_offset = 12
    else:
        raise RuntimeError("{}: unsupported NPY version {}".format(
            member, major))
    header = ast.literal_eval(
        raw[header_offset:payload_offset].decode("latin1").strip())
    if header.get("fortran_order"):
        raise RuntimeError("{}: Fortran-order arrays are unsupported".format(
            member))
    descriptor = header.get("descr")
    formats = {"<f8": "<d", "=f8": "=d", "<i8": "<q", "=i8": "=q"}
    if descriptor not in formats:
        raise RuntimeError("{}: unsupported dtype {}".format(
            member, descriptor))
    shape = tuple(int(item) for item in header.get("shape", ()))
    count = 1
    for extent in shape:
        count *= extent
    payload = raw[payload_offset:]
    expected_bytes = count * 8
    if len(payload) != expected_bytes:
        raise RuntimeError("{}: expected {} data bytes, got {}".format(
            member, expected_bytes, len(payload)))
    values = [item[0] for item in struct.iter_unpack(formats[descriptor],
                                                      payload)]
    return shape, values


def load_oracle(path):
    names = (
        "fp32_q", "fp32_k", "fp32_v", "fp32_qk", "fp32_softmax",
        "fp32_av", "fp32_wo", "fixed_q", "fixed_k", "fixed_v",
        "fixed_qk", "fixed_softmax", "fixed_av", "fixed_wo",
        "fp32_hetero_normalization", "fp32_hetero_av",
        "fp32_hetero_wo", "fixed_hetero_normalization",
        "fixed_hetero_av", "fixed_hetero_wo",
    )
    result = {}
    with zipfile.ZipFile(str(path), "r") as archive:
        for name in names:
            member = name + ".npy"
            result[name] = read_npy(archive.read(member), member)
    return result


def require_shape(oracle, name, shape):
    actual, values = oracle[name]
    if actual != shape:
        raise RuntimeError("{}: expected shape {}, got {}".format(
            name, shape, actual))
    return values


def logical_secure_head(values):
    return [
        values[(head * PHYSICAL_TOKENS + token) * HEAD_DIM + channel]
        for head in range(HEADS)
        for token in range(TOKENS)
        for channel in range(HEAD_DIM)
    ]


def logical_token_major_head(values, physical_tokens):
    return [
        values[token * DIM + head * HEAD_DIM + channel]
        for head in range(HEADS)
        for token in range(TOKENS)
        for channel in range(HEAD_DIM)
    ]


def logical_head_major(values, physical_tokens):
    return [
        values[(head * physical_tokens + token) * HEAD_DIM + channel]
        for head in range(HEADS)
        for token in range(TOKENS)
        for channel in range(HEAD_DIM)
    ]


def logical_matrix(values, physical_tokens):
    return [
        values[(head * physical_tokens + query) * physical_tokens + key]
        for head in range(HEADS)
        for query in range(TOKENS)
        for key in range(TOKENS)
    ]


def logical_token_major(values, width):
    return [values[token * width + channel]
            for token in range(TOKENS) for channel in range(width)]


def quantile(sorted_values, fraction):
    if not sorted_values:
        raise RuntimeError("cannot compute a quantile of an empty tensor")
    rank = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + \
        sorted_values[upper] * weight


def errors(actual, expected):
    if len(actual) != len(expected) or not actual:
        raise RuntimeError("oracle comparison shape mismatch")
    absolute = [abs(float(lhs) - float(rhs))
                for lhs, rhs in zip(actual, expected)]
    ordered = sorted(absolute)
    return {
        "max_abs_error": max(absolute),
        "mse": sum(value * value for value in absolute) / len(absolute),
        "p99_abs_error": quantile(ordered, 0.99),
        "elements_abs_error_gt_1e-2": sum(
            value > 1.0e-2 for value in absolute),
    }


def scaled(values, bits):
    divisor = float(1 << bits)
    return [float(value) / divisor for value in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--protocol", choices=("a0", "hetero"), default="a0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    oracle = load_oracle(args.fixture / "oracle.npz")
    vector_count = HEADS * PHYSICAL_TOKENS * HEAD_DIM
    matrix_count = HEADS * PHYSICAL_TOKENS * PHYSICAL_TOKENS

    secure = {}
    for stage in ("q", "k", "v", "av"):
        secure[stage] = logical_secure_head(read_trace(
            args.trace / ("secure_{}.csv".format(stage)), vector_count))
    matrix_stage = "softmax" if args.protocol == "a0" else "normalization"
    for stage in ("qk", matrix_stage):
        secure[stage] = logical_matrix(read_trace(
            args.trace / ("secure_{}.csv".format(stage)), matrix_count),
            PHYSICAL_TOKENS)
    secure["wo"] = logical_token_major(read_trace(
        args.trace / "secure_wo.csv", PHYSICAL_TOKENS * DIM), DIM)

    fixed = {}
    for stage in ("q", "k", "v"):
        fixed[stage] = logical_token_major_head(require_shape(
            oracle, "fixed_{}".format(stage), (PHYSICAL_TOKENS, DIM)),
            PHYSICAL_TOKENS)
    fixed_av_name = "fixed_av" if args.protocol == "a0" else "fixed_hetero_av"
    fixed_wo_name = "fixed_wo" if args.protocol == "a0" else "fixed_hetero_wo"
    fixed_matrix_name = (
        "fixed_softmax" if args.protocol == "a0"
        else "fixed_hetero_normalization"
    )
    fixed["av"] = logical_head_major(require_shape(
        oracle, fixed_av_name, (HEADS, PHYSICAL_TOKENS, HEAD_DIM)),
        PHYSICAL_TOKENS)
    fixed["qk"] = logical_matrix(require_shape(
        oracle, "fixed_qk", (HEADS, PHYSICAL_TOKENS, PHYSICAL_TOKENS)),
        PHYSICAL_TOKENS)
    fixed[matrix_stage] = logical_matrix(require_shape(
        oracle, fixed_matrix_name,
        (HEADS, PHYSICAL_TOKENS, PHYSICAL_TOKENS)), PHYSICAL_TOKENS)
    fixed["wo"] = logical_token_major(require_shape(
        oracle, fixed_wo_name, (PHYSICAL_TOKENS, DIM)), DIM)

    fp32 = {}
    for stage in ("q", "k", "v"):
        fp32[stage] = logical_token_major_head(require_shape(
            oracle, "fp32_{}".format(stage), (TOKENS, DIM)), TOKENS)
    fp32["q"] = [value / math.sqrt(HEAD_DIM) for value in fp32["q"]]
    fp32_av_name = "fp32_av" if args.protocol == "a0" else "fp32_hetero_av"
    fp32_wo_name = "fp32_wo" if args.protocol == "a0" else "fp32_hetero_wo"
    fp32_matrix_name = (
        "fp32_softmax" if args.protocol == "a0"
        else "fp32_hetero_normalization"
    )
    fp32["av"] = logical_head_major(require_shape(
        oracle, fp32_av_name, (HEADS, TOKENS, HEAD_DIM)), TOKENS)
    fp32["qk"] = logical_matrix(require_shape(
        oracle, "fp32_qk", (HEADS, TOKENS, TOKENS)), TOKENS)
    fp32[matrix_stage] = logical_matrix(require_shape(
        oracle, fp32_matrix_name, (HEADS, TOKENS, TOKENS)), TOKENS)
    fp32["wo"] = logical_token_major(require_shape(
        oracle, fp32_wo_name, (TOKENS, DIM)), DIM)

    # In the heterogeneous protocol only heads 0 and 1 cross into MPC.  The
    # normalization trace consequently contains meaningful values for those
    # two heads only; heads 2 and 3 remain encrypted and are validated at QK
    # and AV instead of being materialized here.
    if args.protocol == "hetero":
        converted_matrix_elements = 2 * TOKENS * TOKENS
        secure[matrix_stage] = secure[matrix_stage][:converted_matrix_elements]
        fixed[matrix_stage] = fixed[matrix_stage][:converted_matrix_elements]
        fp32[matrix_stage] = fp32[matrix_stage][:converted_matrix_elements]

    scales = {"q": 11, "k": 11, "v": 11, "qk": 12,
              matrix_stage: 12, "av": 23, "wo": 12}
    limits = (SECURE_FIXED_LIMITS if args.protocol == "a0"
              else HETERO_SECURE_FIXED_LIMITS)
    report = {
        "status": "PASS",
        "logical_tokens": TOKENS,
        "q_trace_semantics": "Q/sqrt(64), with public scale fused into WQ",
        "normalization_trace_heads": (
            [0, 1] if args.protocol == "hetero" else [0, 1, 2, 3]
        ),
        "stages": {},
        "secure_vs_fixed_limits": {
            stage: {"max_abs_error": limit[0], "mse": limit[1]}
            for stage, limit in limits.items()
        },
    }
    for stage in ("q", "k", "v", "qk", matrix_stage, "av", "wo"):
        scale = scales[stage]
        secure_real = scaled(secure[stage], scale)
        fixed_real = scaled(fixed[stage], scale)
        secure_fixed = errors(secure_real, fixed_real)
        max_limit, mse_limit = limits[stage]
        passed = (
            all(math.isfinite(value) for value in secure_real)
            and secure_fixed["max_abs_error"] <= max_limit
            and secure_fixed["mse"] <= mse_limit
        )
        report["stages"][stage] = {
            "scale_bits": scale,
            "elements": len(secure_real),
            "status": "PASS" if passed else "FAIL",
            "secure_vs_fixed": secure_fixed,
            "fixed_vs_fp32": errors(fixed_real, fp32[stage]),
            "secure_vs_fp32": errors(secure_real, fp32[stage]),
        }
        if not passed:
            report["status"] = "FAIL"

    normalized = scaled(secure[matrix_stage], 12)
    if args.protocol == "a0":
        invariant_values = normalized
        invariant_name = "softmax_invariants"
    else:
        invariant_values = normalized[:TOKENS * TOKENS]
        invariant_name = "relu_norm_head0_invariants"
    row_sums = [sum(invariant_values[offset:offset + TOKENS])
                for offset in range(0, len(invariant_values), TOKENS)]
    report[invariant_name] = {
        "minimum": min(invariant_values),
        "maximum": max(invariant_values),
        "row_sum_minimum": min(row_sums),
        "row_sum_maximum": max(row_sums),
        "max_abs_row_sum_error": max(abs(value - 1.0)
                                     for value in row_sums),
    }
    if min(invariant_values) < 0.0 or max(invariant_values) > 1.0:
        report["status"] = "FAIL"

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
