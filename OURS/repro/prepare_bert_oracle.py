#!/usr/bin/env python3
"""Prepare the three-oracle reference shares for the formal BLB layer-0 run.

This file deliberately lives under BLB/repro.  It is an offline, trusted
pre-processing step: it reads the real exported layer-0 checkpoint and input,
evaluates the fixed-clear BLB arithmetic, and writes additive shares for the
secure run.  The party binary never reads a reconstructed reference tensor;
it only reads its own share and reveals a scalar MSE through MPC.

The fixed-clear arithmetic below follows the quantized BLB path in
``bert.py`` (12-bit fixed point, floor truncation, BLB exp/GeLU
approximations).  It is intentionally explicit rather than importing
transformers or another project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
import struct
from pathlib import Path

import numpy as np


MOD_BITS = 37
MODULUS = 1 << MOD_BITS
SCALE_BITS = 12
DIM = 768
SEQ = 128
HEADS = 12
HEAD_DIM = 64
FFN_DIM = 3072


def read_csv(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values: list[int] = []
    for line in path.read_text(encoding="ascii").splitlines():
        for cell in line.split(","):
            if cell.strip():
                values.append(int(cell))
    expected = math.prod(shape)
    if len(values) != expected:
        raise RuntimeError(f"{path}: expected {expected} values, got {len(values)}")
    return np.asarray(values, dtype=np.int64).reshape(shape)


def floor_div(x: np.ndarray, divisor: int) -> np.ndarray:
    return np.floor_divide(x, divisor).astype(np.int64, copy=False)


def layer_norm_fixed(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centered, inverse-sigma and affine output at 12-bit scale."""
    dn = int(math.floor((1.0 / DIM) * (1 << 24)))
    sums = x.sum(axis=1, keepdims=True, dtype=np.int64)
    avg = floor_div(sums * dn, 1 << 24)
    centered = x - avg
    sq = floor_div(centered * centered, 1 << SCALE_BITS)
    variance = floor_div(sq.sum(axis=1, keepdims=True, dtype=np.int64) * dn, 1 << 24)
    if np.any(variance <= 0):
        raise RuntimeError("fixed-clear LayerNorm encountered a non-positive variance")
    inv_sigma = np.floor((1.0 / np.sqrt(variance.astype(np.float64) / (1 << SCALE_BITS))) * (1 << SCALE_BITS)).astype(np.int64)
    normalized = floor_div(centered * inv_sigma, 1 << SCALE_BITS)
    affine = floor_div(normalized * w.reshape(1, -1), 1 << SCALE_BITS) + b.reshape(1, -1)
    return centered, np.broadcast_to(inv_sigma, x.shape).copy(), affine.astype(np.int64)


def exp_approx_int(x: np.ndarray) -> np.ndarray:
    """BOLT/BLB integer exponential approximation, output at 12-bit scale."""
    x = x.astype(np.int64, copy=False)
    inv_log2 = int(math.floor((1.0 / math.log(2.0)) * (1 << SCALE_BITS)))
    log2_const = int(math.floor(math.log(2.0) * (1 << SCALE_BITS)))
    c1 = int(math.floor(0.3585 * (1 << SCALE_BITS)))
    c2 = int(math.floor(1.353 * (1 << SCALE_BITS)))
    c3 = int(math.floor(0.344 * (1 << SCALE_BITS)))
    temp_l = floor_div(x * inv_log2 * -1, 1 << SCALE_BITS)
    l = floor_div(temp_l, 1 << SCALE_BITS)
    p = x + floor_div(l * log2_const * (1 << SCALE_BITS), 1 << SCALE_BITS)
    p2 = floor_div((p + c2) * (p + c2), 1 << SCALE_BITS)
    fp = floor_div(c1 * p2, 1 << SCALE_BITS) + c3
    # For negative x, l is non-negative.  Python's floor division is the
    # same truncation direction used by the reference numpy implementation.
    return np.floor(fp.astype(np.float64) * np.power(2.0, -l)).astype(np.int64)


def softmax_fixed(qk: np.ndarray) -> np.ndarray:
    out = np.empty_like(qk, dtype=np.int64)
    for head in range(HEADS):
        rows = np.floor(qk[head]).astype(np.int64)
        rows = rows - rows.max(axis=1, keepdims=True)
        expv = exp_approx_int(rows)
        recp = floor_div(expv.sum(axis=1, keepdims=True, dtype=np.int64) * 1, 1)
        # recp is represented at 12-bit scale before reciprocal.
        reciprocal = np.floor((1.0 / (recp.astype(np.float64) / (1 << SCALE_BITS))) * (1 << SCALE_BITS)).astype(np.int64)
        out[head] = floor_div(expv * reciprocal, 1 << SCALE_BITS)
    return out


def gelu_fixed(x: np.ndarray) -> np.ndarray:
    c1 = int(math.floor(0.14439048359960427 * (1 << SCALE_BITS)))
    c2 = int(math.floor(0.7077117131613893 * (1 << SCALE_BITS)))
    c3 = int(math.floor(4.5702822654246535 * (1 << SCALE_BITS)))
    c4 = int(math.floor(8.15444702051307 * (1 << SCALE_BITS)))
    c5 = int(math.floor(16.382265425072532 * (1 << SCALE_BITS)))
    ax = np.abs(x)
    temp_y = floor_div(c1 * ax, 1 << SCALE_BITS) - c2
    y = floor_div(temp_y * ax, 1 << SCALE_BITS) + c3
    temp_res = y + floor_div(c1 * ax, 1 << SCALE_BITS) - c4
    res = floor_div(temp_res * y, 1 << SCALE_BITS) + c5 + floor_div(x, 2)
    res = res.astype(np.int64, copy=False)
    res[x > int(math.floor(2.7 * (1 << SCALE_BITS)))] = x[x > int(math.floor(2.7 * (1 << SCALE_BITS)))]
    res[x < int(math.floor(-2.7 * (1 << SCALE_BITS)))] = 0
    return res


def matmul_fixed(x: np.ndarray, w: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # All tensors reaching a linear layer are represented at 12-bit scale in
    # the formal path.  The CKKS plaintext preprocessing encodes the exported
    # integer weights; one fixed-point reduction restores the 12-bit domain.
    product = x.astype(np.int64) @ w.astype(np.int64)
    return floor_div(product, 1 << SCALE_BITS) + bias.reshape(1, -1)


def load_layer0(model: Path) -> dict[str, np.ndarray]:
    def f(name: str, shape: tuple[int, ...]) -> np.ndarray:
        return read_csv(model / name, shape)

    return {
        "input": f("inputs_0_data.txt", (SEQ, DIM)),
        "mask": f("inputs_0_mask.txt", (SEQ,)),
        "q": f("bert.encoder.layer.0.attention.self.query.weight.txt", (HEADS, DIM, HEAD_DIM)),
        "k": f("bert.encoder.layer.0.attention.self.key.weight.txt", (HEADS, DIM, HEAD_DIM)),
        "v": f("bert.encoder.layer.0.attention.self.value.weight.txt", (HEADS, DIM, HEAD_DIM)),
        "bq": f("bert.encoder.layer.0.attention.self.query.bias.txt", (HEADS, HEAD_DIM)),
        "bk": f("bert.encoder.layer.0.attention.self.key.bias.txt", (HEADS, HEAD_DIM)),
        "bv": f("bert.encoder.layer.0.attention.self.value.bias.txt", (HEADS, HEAD_DIM)),
        "wo": f("bert.encoder.layer.0.attention.output.dense.weight.txt", (DIM, DIM)),
        "bo": f("bert.encoder.layer.0.attention.output.dense.bias.txt", (DIM,)),
        "ln1w": f("bert.encoder.layer.0.attention.output.LayerNorm.weight.txt", (DIM,)),
        "ln1b": f("bert.encoder.layer.0.attention.output.LayerNorm.bias.txt", (DIM,)),
        "wi1": f("bert.encoder.layer.0.intermediate.dense.weight.txt", (DIM, FFN_DIM)),
        "bi1": f("bert.encoder.layer.0.intermediate.dense.bias.txt", (FFN_DIM,)),
        "wi2": f("bert.encoder.layer.0.output.dense.weight.txt", (FFN_DIM, DIM)),
        "bi2": f("bert.encoder.layer.0.output.dense.bias.txt", (DIM,)),
        "ln2w": f("bert.encoder.layer.0.output.LayerNorm.weight.txt", (DIM,)),
        "ln2b": f("bert.encoder.layer.0.output.LayerNorm.bias.txt", (DIM,)),
    }


def fixed_clear(layer: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluate the official post-norm BLB integer path from ``bert.py``.

    The exported layer-0 input is the embedding output.  It is *not* fed
    through a made-up first LayerNorm.  The scale transitions below are the
    published ``x_scale/w_scale/b_scale`` values for layer 0 and are kept
    explicit so the secure five-block path has an unambiguous fixed-clear
    target.
    """
    x = layer["input"].astype(np.int64, copy=False)

    # linear1: input scale 12 -> x_scale[0]=5; Q/K/V have scale 11.
    x_qkv = floor_div(x, 1 << (SCALE_BITS - 5))
    q = np.stack([
        x_qkv @ layer["q"][h] + layer["bq"][h]
        for h in range(HEADS)
    ])
    k = np.stack([
        x_qkv @ layer["k"][h] + layer["bk"][h]
        for h in range(HEADS)
    ])
    v = np.stack([
        x_qkv @ layer["v"][h] + layer["bv"][h]
        for h in range(HEADS)
    ])
    # Q/K product is scale 22; layer 0 reduces it to scale 12.
    qk = np.stack([floor_div(q[h] @ k[h].T, 1 << 10)
                   for h in range(HEADS)])
    # The acceptance hook applies BOLT's public postprocess to each local
    # share.  QK is then [head, query, key] row-major and V is
    # [head, token, channel] row-major.
    # Secure conversion keeps V at its true scale 11.  The combined f1 oracle
    # file has one scale field, so only its comparison copy is shifted to
    # scale 12.  The V share used by Softmax*V is never changed.
    f1 = np.concatenate((qk.reshape(-1), (v * 2).reshape(-1)))

    # inputs_0_mask is the public BERT key mask in real score units.  The
    # secure path adds mask[key] * 2^12 to exactly one additive share after
    # QK packing is converted to rows and immediately before MPC Softmax.
    mask = layer["mask"].astype(np.int64, copy=False)
    if np.any(mask > 0) or np.any(mask < -(1 << (MOD_BITS - SCALE_BITS - 1))):
        raise RuntimeError("public attention mask is outside the 37-bit ring")
    masked_qk = qk + mask.reshape(1, 1, SEQ) * (1 << SCALE_BITS)
    sm = softmax_fixed(masked_qk)
    f2 = sm.reshape(-1)

    # Softmax (scale 12) x V (scale 11) is carried at scale 23.
    attention = np.concatenate([
        sm[h] @ v[h] for h in range(HEADS)
    ], axis=1)
    # linear2: scale 23 -> x_scale[1]=6, output/residual scale 12.
    h2 = floor_div(attention, 1 << 17)
    h3 = h2 @ layer["wo"] + layer["bo"] + x

    # Post-attention LayerNorm (the first LayerNorm in BERT).
    c2, s2, a2 = layer_norm_fixed(h3, layer["ln1w"], layer["ln1b"])
    f3 = np.concatenate((c2.reshape(-1), s2[:, 0].reshape(-1)))

    # linear3: LayerNorm output scale 12 -> x_scale[2]=5, then GeLU at scale
    # 12 after the repository's one-bit rescale.
    ffn1_input = floor_div(a2, 1 << 7)
    ffn1 = ffn1_input @ layer["wi1"] + layer["bi1"]
    gelu = gelu_fixed(ffn1 * 2)
    f4 = gelu.reshape(-1)

    # linear4 + the real post-attention residual (a2), followed by the second
    # LayerNorm's pre-rsqrt output.  output.dense is scale 9.
    h6 = floor_div(gelu, 1 << 8)
    ffn2 = h6 @ layer["wi2"] + layer["bi2"]
    residual_at_scale9 = floor_div(a2, 1 << 3)
    h7_scale9 = ffn2 + residual_at_scale9
    h7_scale12 = h7_scale9 * (1 << 3)
    c3, s3, _ = layer_norm_fixed(h7_scale12, layer["ln2w"], layer["ln2b"])
    f5 = np.concatenate((c3.reshape(-1), s3[:, 0].reshape(-1)))
    return {
        "fgf1_ln_attention": f1,
        "fgf2_softmax": f2,
        "fgf3_softmax_v_ln": f3,
        "fgf4_ln_ffn_gelu": f4,
        "fgf5_ffn2_ln": f5,
    }


def fp32_reference(layer: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # A/B is an independent FP32 oracle.  Dequantization uses the published
    # BLB scale tables for layer 0; the fixed-clear tensors above remain the
    # exact 12-bit quantities consumed by the secure comparison.
    x = layer["input"].astype(np.float64) / (1 << SCALE_BITS)
    q = layer["q"].astype(np.float64) / (1 << 6)
    k = layer["k"].astype(np.float64) / (1 << 6)
    v = layer["v"].astype(np.float64) / (1 << 6)
    bq = layer["bq"].astype(np.float64) / (1 << 11)
    bk = layer["bk"].astype(np.float64) / (1 << 11)
    bv = layer["bv"].astype(np.float64) / (1 << 11)
    ln1w = layer["ln1w"].astype(np.float64) / (1 << SCALE_BITS)
    ln1b = layer["ln1b"].astype(np.float64) / (1 << SCALE_BITS)
    # Official BERT is post-norm: Q/K/V consume the embedding tensor.
    qf = np.stack([x @ q[h] + bq[h] for h in range(HEADS)])
    kf = np.stack([x @ k[h] + bk[h] for h in range(HEADS)])
    vf = np.stack([x @ v[h] + bv[h] for h in range(HEADS)])
    qkf = np.stack([qf[h] @ kf[h].T for h in range(HEADS)])
    masked_qkf = qkf + layer["mask"].astype(np.float64).reshape(1, 1, SEQ)
    sm = np.exp(masked_qkf - masked_qkf.max(axis=2, keepdims=True))
    sm /= sm.sum(axis=2, keepdims=True)
    av = np.concatenate([sm[h] @ vf[h] for h in range(HEADS)], axis=1)
    wo = layer["wo"].astype(np.float64) / (1 << 6)
    bo = layer["bo"].astype(np.float64) / (1 << 12)
    h3 = av @ wo + bo + x
    c2 = h3 - h3.mean(axis=1, keepdims=True)
    s2 = 1.0 / np.sqrt((c2 * c2).mean(axis=1, keepdims=True))
    ln2w = layer["ln2w"].astype(np.float64) / (1 << SCALE_BITS)
    ln2b = layer["ln2b"].astype(np.float64) / (1 << SCALE_BITS)
    a2 = c2 * s2 * ln1w + ln1b
    wi1 = layer["wi1"].astype(np.float64) / (1 << 6)
    bi1 = layer["bi1"].astype(np.float64) / (1 << 11)
    gelu = 0.5 * (a2 @ wi1 + bi1) * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * ((a2 @ wi1 + bi1) + 0.044715 * (a2 @ wi1 + bi1) ** 3)))
    wi2 = layer["wi2"].astype(np.float64) / (1 << 5)
    bi2 = layer["bi2"].astype(np.float64) / (1 << 9)
    h7 = gelu @ wi2 + bi2 + a2
    c3 = h7 - h7.mean(axis=1, keepdims=True)
    s3 = 1.0 / np.sqrt((c3 * c3).mean(axis=1, keepdims=True))
    return {
        "fgf1_ln_attention": np.concatenate((qkf.reshape(-1), vf.reshape(-1))),
        "fgf2_softmax": sm.reshape(-1),
        "fgf3_softmax_v_ln": np.concatenate((c2.reshape(-1), s2[:, 0].reshape(-1))),
        "fgf4_ln_ffn_gelu": gelu.reshape(-1),
        "fgf5_ffn2_ln": np.concatenate((c3.reshape(-1), s3[:, 0].reshape(-1))),
    }


def write_share(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.asarray(values, dtype=np.uint64).reshape(-1)
    with path.open("wb") as handle:
        handle.write(b"BLB_ORACLE_SHARE_V1\0")
        handle.write(struct.pack("<Qii", encoded.size, MOD_BITS, SCALE_BITS))
        handle.write(encoded.tobytes(order="C"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = args.model_dir.resolve()
    output = args.output.resolve()
    layer = load_layer0(model)
    fixed = fixed_clear(layer)
    fp32 = fp32_reference(layer)
    manifest: dict[str, object] = {
        "schema": "blb.bert.three-oracle.v3",
        "arithmetic": "bert.py.post_norm.layer0.qk_s12_v_s11.public_key_mask.v3",
        "stage_domain": "12-bit BLB fixed-clear comparison shares; f1 V comparison copy is shifted from s11 to s12; no plaintext reconstruction",
        "layer": 0,
        "mod_bits": MOD_BITS,
        "scale_bits": SCALE_BITS,
        "source_checkpoint_manifest_sha256": hashlib.sha256((model / "manifest.json").read_bytes()).hexdigest(),
        "public_attention_mask": {
            "path": "inputs_0_mask.txt",
            "sha256": hashlib.sha256((model / "inputs_0_mask.txt").read_bytes()).hexdigest(),
            "broadcast": "head,query,key <- mask[key]",
            "secure_add_party": "BOB",
            "scale_bits": SCALE_BITS,
        },
        "stages": {},
    }
    rng = secrets.SystemRandom()
    for name, values in fixed.items():
        flat = np.asarray(values, dtype=np.int64).reshape(-1)
        # CSPRNG shares; no deterministic or all-zero share is accepted.
        alice = np.fromiter((rng.randrange(MODULUS) for _ in range(flat.size)), dtype=np.uint64, count=flat.size)
        target = np.asarray(flat, dtype=np.int64).astype(object)
        bob = np.asarray([(int(v) - int(r)) % MODULUS for v, r in zip(target, alice)], dtype=np.uint64)
        a_path = output / f"{name}.alice.bin"
        b_path = output / f"{name}.bob.bin"
        a_sha = write_share(a_path, alice)
        b_sha = write_share(b_path, bob)
        fp = np.asarray(fp32[name], dtype=np.float64).reshape(-1)
        fixed_real = flat.astype(np.float64) / (1 << SCALE_BITS)
        ab_mse = float(np.mean((fp - fixed_real) ** 2))
        manifest["stages"][name] = {
            "size": int(flat.size),
            "alice": a_path.name,
            "bob": b_path.name,
            "alice_sha256": a_sha,
            "bob_sha256": b_sha,
            "A_B_MSE": ab_mse,
        }
    manifest_path = output / "manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ORACLE_PREPROCESS status=PASS output={output} stages={len(fixed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
