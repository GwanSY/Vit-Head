#!/usr/bin/env python3
"""Export the 65-token ViT fixture into the original BLB CKKS text layout.

The online layout stays at the donor's 128 physical tokens.  Tokens 65..127
are public zero padding and are removed from every oracle comparison.  The
standard 1/sqrt(64) attention scale is fused into WQ offline; this is an exact
linear reparameterization, not a learned or input-dependent change.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


LOGICAL_TOKENS = 65
PHYSICAL_TOKENS = 128
DIM = 256
HEADS = 4
HEAD_DIM = 64
ELL = 37
SCALE = 12
QKV_WEIGHT_SCALE = 9
QKV_BIAS_SCALE = 11
WO_WEIGHT_SCALE = 6
WO_BIAS_SCALE = 12


def parse_fixture(path):
    words = path.read_text(encoding="utf-8").split()
    if not words:
        raise RuntimeError(f"empty fixture: {path}")
    result = {"magic": [words[0]]}  # type: Dict[str, List[str]]
    scalar = {"tokens", "dim", "heads", "scale_bits"}
    vector = {"alpha", "wq", "wk", "wv", "wo", "bo", "x"}
    index = 1
    while index < len(words):
        key = words[index]
        index += 1
        if key == "end":
            break
        if key in scalar:
            result[key] = [words[index]]
            index += 1
            continue
        if key not in vector:
            raise RuntimeError(f"unknown fixture key {key!r} in {path}")
        values = []  # type: List[str]
        while index < len(words) and words[index] not in scalar | vector | {"end"}:
            values.append(words[index])
            index += 1
        result[key] = values
    return result


def array(items, shape):
    value = np.asarray([float(item) for item in items], dtype=np.float64)
    if value.size != math.prod(shape):
        raise RuntimeError(f"expected {shape}, got {value.size} values")
    return value.reshape(shape)


def fixed(value: np.ndarray, scale: int) -> np.ndarray:
    # Match C++/the donor exporter: Python and NumPy both use nearest-even for
    # exact .5 ties.  Such ties do not occur in the decimal source fixture.
    return np.rint(value * (1 << scale)).astype(np.int64)


def floor_div(value: np.ndarray, divisor: int) -> np.ndarray:
    return np.floor_divide(value, divisor).astype(np.int64, copy=False)


def write_rows(path, value):
    matrix = np.asarray(value, dtype=np.int64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for row in matrix:
            handle.write(",".join(str(int(cell)) for cell in row.reshape(-1)))
            handle.write("\n")
    payload = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
    }


def exp_approx_int(x: np.ndarray) -> np.ndarray:
    inv_log2 = int(math.floor((1.0 / math.log(2.0)) * (1 << SCALE)))
    log2_const = int(math.floor(math.log(2.0) * (1 << SCALE)))
    c1 = int(math.floor(0.3585 * (1 << SCALE)))
    c2 = int(math.floor(1.353 * (1 << SCALE)))
    c3 = int(math.floor(0.344 * (1 << SCALE)))
    temp_l = floor_div(x.astype(np.int64) * inv_log2 * -1, 1 << SCALE)
    exponent = floor_div(temp_l, 1 << SCALE)
    p = x + exponent * log2_const
    p2 = floor_div((p + c2) * (p + c2), 1 << SCALE)
    polynomial = floor_div(c1 * p2, 1 << SCALE) + c3
    return np.floor(
        polynomial.astype(np.float64) * np.power(2.0, -exponent)
    ).astype(np.int64)


def softmax_fixed(qk: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.empty_like(qk, dtype=np.int64)
    masked = qk + mask.reshape(1, 1, PHYSICAL_TOKENS) * (1 << SCALE)
    for head in range(HEADS):
        rows = masked[head] - masked[head].max(axis=1, keepdims=True)
        exponential = exp_approx_int(rows)
        denominator = exponential.sum(axis=1, keepdims=True, dtype=np.int64)
        if np.any(denominator <= 0):
            raise RuntimeError("fixed Softmax denominator is non-positive")
        reciprocal = np.floor(
            (1.0 / (denominator.astype(np.float64) / (1 << SCALE)))
            * (1 << SCALE)
        ).astype(np.int64)
        output[head] = floor_div(exponential * reciprocal, 1 << SCALE)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="trained checkpoint from which the source fixture was exported",
    )
    parser.add_argument("--source-layer", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.fixture.resolve() / "alice_model.txt"
    input_path = args.fixture.resolve() / "bob_input.txt"
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError("missing trained checkpoint: {}".format(checkpoint_path))
    model = parse_fixture(model_path)
    client = parse_fixture(input_path)
    metadata = {
        "tokens": int(model["tokens"][0]),
        "dim": int(model["dim"][0]),
        "heads": int(model["heads"][0]),
        "scale_bits": int(model["scale_bits"][0]),
    }
    expected = {
        "tokens": LOGICAL_TOKENS,
        "dim": DIM,
        "heads": HEADS,
        "scale_bits": SCALE,
    }
    if metadata != expected:
        raise RuntimeError(f"fixture metadata {metadata} != required {expected}")
    if int(client["tokens"][0]) != LOGICAL_TOKENS or int(client["dim"][0]) != DIM:
        raise RuntimeError("client fixture metadata mismatch")

    x = array(client["x"], (LOGICAL_TOKENS, DIM))
    wq, wk, wv, wo = (
        array(model[name], (DIM, DIM)) for name in ("wq", "wk", "wv", "wo")
    )
    bo = array(model["bo"], (DIM,))
    alpha = [int(float(value)) for value in model.get("alpha", [])]
    if len(alpha) != HEADS or any(value not in (0, 1) for value in alpha):
        raise RuntimeError("source alpha must contain four public 0/1 values")
    head_protocols = ["relu" if value else "linear" for value in alpha]

    # Fuse the standard ViT attention divisor into Q.  The QK kernel itself is
    # byte-for-byte the original BLB ct-ct donor path.
    wq_scaled = wq / math.sqrt(HEAD_DIM)
    wq_i = fixed(wq_scaled, QKV_WEIGHT_SCALE)
    wk_i = fixed(wk, QKV_WEIGHT_SCALE)
    wv_i = fixed(wv, QKV_WEIGHT_SCALE)
    wo_i = fixed(wo, WO_WEIGHT_SCALE)
    bo_i = fixed(bo, WO_BIAS_SCALE)

    physical_x = np.zeros((PHYSICAL_TOKENS, DIM), dtype=np.float64)
    physical_x[:LOGICAL_TOKENS] = x
    physical_x_i = fixed(physical_x, SCALE)
    mask = np.full(PHYSICAL_TOKENS, -32, dtype=np.int64)
    mask[:LOGICAL_TOKENS] = 0

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    records.append(write_rows(args.output / "inputs_0_data.txt", physical_x_i))
    records.append(write_rows(args.output / "inputs_0_mask.txt", mask))
    protocol_path = args.output / "head_protocols.txt"
    protocol_path.write_text("\n".join(head_protocols) + "\n", encoding="ascii")
    protocol_payload = protocol_path.read_bytes()
    records.append({
        "path": protocol_path.name,
        "sha256": hashlib.sha256(protocol_payload).hexdigest(),
        "bytes": len(protocol_payload),
        "rows": HEADS,
        "columns": 1,
    })

    def write_qkv(name: str, matrix: np.ndarray) -> None:
        packed = np.empty((HEADS * DIM, HEAD_DIM), dtype=np.int64)
        row = 0
        for head in range(HEADS):
            for input_channel in range(DIM):
                packed[row] = matrix[
                    head * HEAD_DIM : (head + 1) * HEAD_DIM,
                    input_channel,
                ]
                row += 1
        records.append(write_rows(
            args.output / f"bert.encoder.layer.0.attention.self.{name}.weight.txt",
            packed,
        ))
        records.append(write_rows(
            args.output / f"bert.encoder.layer.0.attention.self.{name}.bias.txt",
            np.zeros((HEADS * HEAD_DIM,), dtype=np.int64),
        ))

    write_qkv("query", wq_i)
    write_qkv("key", wk_i)
    write_qkv("value", wv_i)
    # BLB's dense exporter stores [input, output].
    records.append(write_rows(
        args.output / "bert.encoder.layer.0.attention.output.dense.weight.txt",
        wo_i.T,
    ))
    records.append(write_rows(
        args.output / "bert.encoder.layer.0.attention.output.dense.bias.txt",
        bo_i,
    ))

    # FP32 oracle: the unmodified standard ViT Attention definition.
    q_fp = x @ wq.T
    k_fp = x @ wk.T
    v_fp = x @ wv.T
    qh_fp = q_fp.reshape(LOGICAL_TOKENS, HEADS, HEAD_DIM).transpose(1, 0, 2)
    kh_fp = k_fp.reshape(LOGICAL_TOKENS, HEADS, HEAD_DIM).transpose(1, 0, 2)
    vh_fp = v_fp.reshape(LOGICAL_TOKENS, HEADS, HEAD_DIM).transpose(1, 0, 2)
    qk_fp = np.matmul(qh_fp, kh_fp.transpose(0, 2, 1)) / math.sqrt(HEAD_DIM)
    shifted = qk_fp - qk_fp.max(axis=-1, keepdims=True)
    softmax_fp = np.exp(shifted)
    softmax_fp /= softmax_fp.sum(axis=-1, keepdims=True)
    av_fp = np.matmul(softmax_fp, vh_fp)
    joined_fp = av_fp.transpose(1, 0, 2).reshape(LOGICAL_TOKENS, DIM)
    output_fp = joined_fp @ wo.T + bo

    hetero_norm_fp = qk_fp / LOGICAL_TOKENS
    relu_head = np.maximum(qk_fp[0], 0.0)
    relu_denominator = relu_head.sum(axis=-1, keepdims=True)
    relu_denominator = np.where(relu_denominator > 0.0,
                                relu_denominator, 1.0 / (1 << SCALE))
    hetero_norm_fp[0] = relu_head / relu_denominator
    hetero_av_fp = np.matmul(hetero_norm_fp, vh_fp)
    hetero_joined_fp = hetero_av_fp.transpose(1, 0, 2).reshape(
        LOGICAL_TOKENS, DIM)
    hetero_output_fp = hetero_joined_fp @ wo.T + bo

    # Approximate/fixed oracle at every secure boundary.
    x_s5 = floor_div(physical_x_i, 1 << (SCALE - 5))
    q_s11 = floor_div(x_s5 @ wq_i.T, 1 << (5 + QKV_WEIGHT_SCALE - 11))
    k_s11 = floor_div(x_s5 @ wk_i.T, 1 << (5 + QKV_WEIGHT_SCALE - 11))
    v_s11 = floor_div(x_s5 @ wv_i.T, 1 << (5 + QKV_WEIGHT_SCALE - 11))
    qh = q_s11.reshape(PHYSICAL_TOKENS, HEADS, HEAD_DIM).transpose(1, 0, 2)
    kh = k_s11.reshape(PHYSICAL_TOKENS, HEADS, HEAD_DIM).transpose(1, 0, 2)
    vh = v_s11.reshape(PHYSICAL_TOKENS, HEADS, HEAD_DIM).transpose(1, 0, 2)
    qk_s12 = np.stack([
        floor_div(qh[head] @ kh[head].T, 1 << (22 - SCALE))
        for head in range(HEADS)
    ])
    softmax_s12 = softmax_fixed(qk_s12, mask)
    av_s23 = np.stack([
        softmax_s12[head] @ vh[head] for head in range(HEADS)
    ])
    joined_s6 = floor_div(
        av_s23.transpose(1, 0, 2).reshape(PHYSICAL_TOKENS, DIM),
        1 << (SCALE + QKV_BIAS_SCALE - WO_WEIGHT_SCALE),
    )
    output_s12 = joined_s6 @ wo_i.T + bo_i

    hetero_norm_s12 = np.zeros_like(qk_s12)
    valid_qk = qk_s12[:, :LOGICAL_TOKENS, :LOGICAL_TOKENS]
    relu_fixed = np.maximum(valid_qk[0], 0)
    denominator_fixed = relu_fixed.sum(axis=-1, keepdims=True,
                                       dtype=np.int64)
    denominator_fixed = np.maximum(denominator_fixed, 1)
    hetero_norm_s12[0, :LOGICAL_TOKENS, :LOGICAL_TOKENS] = np.floor(
        relu_fixed.astype(np.float64) /
        denominator_fixed.astype(np.float64) * (1 << SCALE)
    ).astype(np.int64)
    reciprocal_65_s20 = int(round((1 << 20) / LOGICAL_TOKENS))
    hetero_norm_s12[1, :LOGICAL_TOKENS, :LOGICAL_TOKENS] = floor_div(
        valid_qk[1] * reciprocal_65_s20, 1 << 20)
    for head in (2, 3):
        hetero_norm_s12[head, :LOGICAL_TOKENS, :LOGICAL_TOKENS] = np.floor(
            valid_qk[head].astype(np.float64) / LOGICAL_TOKENS
        ).astype(np.int64)
    hetero_av_s23 = np.stack([
        hetero_norm_s12[head] @ vh[head] for head in range(HEADS)
    ])
    hetero_joined_s6 = floor_div(
        hetero_av_s23.transpose(1, 0, 2).reshape(PHYSICAL_TOKENS, DIM),
        1 << (SCALE + QKV_BIAS_SCALE - WO_WEIGHT_SCALE),
    )
    hetero_output_s12 = hetero_joined_s6 @ wo_i.T + bo_i

    np.savez_compressed(
        args.output / "oracle.npz",
        fp32_q=q_fp,
        fp32_k=k_fp,
        fp32_v=v_fp,
        fp32_qk=qk_fp,
        fp32_softmax=softmax_fp,
        fp32_av=av_fp,
        fp32_wo=output_fp,
        fixed_q=q_s11,
        fixed_k=k_s11,
        fixed_v=v_s11,
        fixed_qk=qk_s12,
        fixed_softmax=softmax_s12,
        fixed_av=av_s23,
        fixed_wo=output_s12,
        fp32_hetero_normalization=hetero_norm_fp,
        fp32_hetero_av=hetero_av_fp,
        fp32_hetero_wo=hetero_output_fp,
        fixed_hetero_normalization=hetero_norm_s12,
        fixed_hetero_av=hetero_av_s23,
        fixed_hetero_wo=hetero_output_s12,
        public_mask=mask,
    )
    oracle_payload = (args.output / "oracle.npz").read_bytes()
    records.append({
        "path": "oracle.npz",
        "sha256": hashlib.sha256(oracle_payload).hexdigest(),
        "bytes": len(oracle_payload),
    })

    manifest = {
        "schema": "blb.vit65.a0.fixture.v1",
        "source": {
            "fixture": str(args.fixture.resolve()),
            "alice_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "bob_input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest(),
            "checkpoint_layer": args.source_layer,
            "checkpoint_status": "trained_mpcvit_cifar10",
            "source_exporter": "Pencil/lamination_vit/scripts/export_mpcvit_block.py",
        },
        "layout": {
            "logical_tokens": LOGICAL_TOKENS,
            "physical_tokens": PHYSICAL_TOKENS,
            "hidden": DIM,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "ckks_slots": 16384,
            "padding": "public_zero_tokens_65_through_127",
            "qk_packing": "original_BLB_cross_packing",
            "qkv_and_wo": "original_BLB_MHP_BSGS",
        },
        "quantization": {
            "input": SCALE,
            "qkv_input": 5,
            "qkv_weight": QKV_WEIGHT_SCALE,
            "qkv_output": 11,
            "qk_and_softmax": SCALE,
            "wo_weight": WO_WEIGHT_SCALE,
            "wo_bias_and_output": WO_BIAS_SCALE,
        },
        "math": {
            "protocol": "standard_softmax_only",
            "attention_scale": "1/sqrt(64) fused into WQ offline",
            "retraining": False,
            "head_assignment": False,
            "source_alpha": alpha,
            "heterogeneous_protocols": head_protocols,
        },
        "files": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "A0_EXPORT status=PASS logical_tokens=65 physical_tokens=128 "
        "hidden=256 heads=4 head_dim=64 qkv_weight_scale=9"
    )


if __name__ == "__main__":
    main()
