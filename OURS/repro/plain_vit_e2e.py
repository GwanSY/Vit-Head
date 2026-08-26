#!/usr/bin/env python3
"""BLB 端到端 post-norm ViT 的明文参考。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


VECTOR_NAMES = (
    "patch_w", "patch_b", "pos", "wq", "bq", "wk", "bk", "wv", "bv",
    "wo", "bo", "ln1_w", "ln1_b", "ff1_w", "ff1_b", "ff2_w", "ff2_b",
    "ln2_w", "ln2_b", "head_ln_w", "head_ln_b", "classifier_w",
    "classifier_b", "x",
)
SCALAR_NAMES = {
    "patches", "patch_dim", "dim", "heads", "mlp_dim", "classes", "scale_bits"
}


def read_tokens(path: Path) -> dict[str, list[str]]:
    raw = path.read_text(encoding="utf-8").split()
    result: dict[str, list[str]] = {"magic": [raw[0]]}
    keys = SCALAR_NAMES | set(VECTOR_NAMES) | {"end"}
    i = 1
    while i < len(raw):
        key = raw[i]
        i += 1
        if key == "end":
            break
        if key in SCALAR_NAMES:
            result[key] = [raw[i]]
            i += 1
        else:
            result[key] = []
            while i < len(raw) and raw[i] not in keys:
                result[key].append(raw[i])
                i += 1
    return result


def array(items: list[str], shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(items, dtype=np.float64)
    if value.size != int(np.prod(shape)):
        raise ValueError(f"expected {shape}, got {value.size} values")
    return value.reshape(shape)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # SCI/BLB 当前 layer_norm_fix 没有显式 epsilon；fixture 保证方差非零。
    mean = x.mean(axis=-1, keepdims=True)
    variance = np.square(x - mean).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance) * weight + bias


def gelu_blb(x: np.ndarray) -> np.ndarray:
    # SCI FPMath::gelu_approx 中 BLB/BOLT 已有的分段多项式。
    y = np.abs(x)
    z = y * (y - 4.901373660634577)
    p = ((z + y - 24.822721454603112) * (z + 31.65224017185284) + 785.77248299658)
    p *= 0.02084861175412759
    middle = 0.5 * x + p
    return np.where(y > 2.7, np.maximum(x, 0.0), middle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, inp = read_tokens(args.model), read_tokens(args.input)
    patches = int(model["patches"][0])
    patch_dim = int(model["patch_dim"][0])
    dim = int(model["dim"][0])
    heads = int(model["heads"][0])
    mlp_dim = int(model["mlp_dim"][0])
    classes = int(model["classes"][0])
    x = np.zeros((patches, dim), dtype=np.float64)
    x[:, :patch_dim] = array(inp["x"], (patches, patch_dim))
    get = lambda name, shape: array(model[name], shape)

    embedded = x @ get("patch_w", (dim, dim)).T
    embedded += get("patch_b", (dim,)) + get("pos", (patches, dim))
    q = embedded @ get("wq", (dim, dim)).T + get("bq", (dim,))
    k = embedded @ get("wk", (dim, dim)).T + get("bk", (dim,))
    v = embedded @ get("wv", (dim, dim)).T + get("bv", (dim,))
    head_dim = dim // heads
    joined = np.empty_like(embedded)
    for head in range(heads):
        sl = slice(head * head_dim, (head + 1) * head_dim)
        score = q[:, sl] @ k[:, sl].T / np.sqrt(head_dim)
        score -= score.max(axis=-1, keepdims=True)
        probability = np.exp(score)
        probability /= probability.sum(axis=-1, keepdims=True)
        joined[:, sl] = probability @ v[:, sl]
    attention = joined @ get("wo", (dim, dim)).T + get("bo", (dim,))
    hidden = layer_norm(
        embedded + attention, get("ln1_w", (dim,)), get("ln1_b", (dim,))
    )
    ff = hidden @ get("ff1_w", (mlp_dim, dim)).T + get("ff1_b", (mlp_dim,))
    ff = gelu_blb(ff)
    ff = ff @ get("ff2_w", (dim, mlp_dim)).T + get("ff2_b", (dim,))
    encoded = layer_norm(
        hidden + ff, get("ln2_w", (dim,)), get("ln2_b", (dim,))
    )
    pooled = encoded.mean(axis=0, keepdims=True)
    pooled = layer_norm(
        pooled, get("head_ln_w", (dim,)), get("head_ln_b", (dim,))
    )
    logits = pooled @ get("classifier_w", (classes, dim)).T
    logits += get("classifier_b", (classes,))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, logits, fmt="%.12g")
    print(
        f"PLAINTEXT status=PASS predicted_class={int(logits.argmax())} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
