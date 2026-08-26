#!/usr/bin/env python3
"""Generate a deterministic tiny ViT fixture without external model code.

This is a protocol smoke model: it uses the standard ViT equations and the
same operator order as BOLT_VIT, but is not a trained accuracy checkpoint.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


def vector(handle, name: str, data: list[float]) -> None:
    handle.write(name)
    for offset in range(0, len(data), 8):
        handle.write(" " if offset == 0 else "\n")
        handle.write(" ".join(f"{x:.12g}" for x in data[offset : offset + 8]))
    handle.write("\n")


def matvec(weight: list[float], value: list[float], rows: int, cols: int) -> list[float]:
    return [sum(weight[r * cols + c] * value[c] for c in range(cols)) for r in range(rows)]


def layer_norm(value: list[float]) -> list[float]:
    mean = sum(value) / len(value)
    variance = sum((x - mean) ** 2 for x in value) / len(value)
    inverse = 1.0 / math.sqrt(max(variance, 1e-12))
    return [(x - mean) * inverse for x in value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("fixtures/vit_smoke"))
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument(
        "--constant-final-ln", action="store_true",
        help="diagnostic fixture: final LayerNorm gamma=0 with a fixed beta",
    )
    parser.add_argument(
        "--zero-final-ln", action="store_true",
        help="diagnostic fixture: final LayerNorm gamma=beta=0",
    )
    parser.add_argument(
        "--identity-classifier", action="store_true",
        help="diagnostic fixture: expose the first class-count CLS entries",
    )
    parser.add_argument(
        "--bias-only-classifier", action="store_true",
        help="diagnostic fixture: zero classifier weights with public test bias",
    )
    parser.add_argument(
        "--protocol", choices=("BOLT", "IRON"), default="BOLT",
        help="Select the model/input file magic expected by the executable.",
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    image_size, patch_size, channels = 4, 2, 3
    tokens, dim, heads, ff_dim, layers, classes = 5, 8, 2, 16, args.layers, 3
    if layers < 1:
        parser.error("--layers must be positive")
    scale_bits = 12

    def randoms(count: int, std: float = 0.08) -> list[float]:
        return [rng.gauss(0.0, std) for _ in range(count)]

    model: dict[str, object] = {
        "patch_weight": randoms(dim * channels * patch_size * patch_size),
        "patch_bias": randoms(dim, 0.02),
        "class_token": randoms(dim, 0.08),
        "position": randoms(tokens * dim, 0.04),
        "blocks": [],
        "final_ln_weight": [1.0] * dim,
        "final_ln_bias": [0.0] * dim,
        "classifier_weight": randoms(classes * dim, 0.12),
        "classifier_bias": [0.45, 0.0, -0.45],
    }
    if args.constant_final_ln:
        model["final_ln_weight"] = [0.0] * dim
        model["final_ln_bias"] = [0.0625 * (index - 3)
                                  for index in range(dim)]
    if args.zero_final_ln:
        model["final_ln_weight"] = [0.0] * dim
        model["final_ln_bias"] = [0.0] * dim
    if args.identity_classifier:
        model["classifier_weight"] = [
            1.0 if row == col else 0.0
            for row in range(classes) for col in range(dim)
        ]
        model["classifier_bias"] = [0.0] * classes
    if args.bias_only_classifier:
        model["classifier_weight"] = [0.0] * (classes * dim)
        model["classifier_bias"] = [0.25, 0.0, -0.25]
    for _ in range(layers):
        model["blocks"].append(
            {
                "ln1_weight": [1.0] * dim,
                "ln1_bias": [0.0] * dim,
                "wq": randoms(dim * dim),
                "wk": randoms(dim * dim),
                "wv": randoms(dim * dim),
                "wo": randoms(dim * dim),
                "bo": randoms(dim, 0.02),
                "ln2_weight": [1.0] * dim,
                "ln2_bias": [0.0] * dim,
                "ff1_weight": randoms(ff_dim * dim),
                "ff1_bias": randoms(ff_dim, 0.02),
                "ff2_weight": randoms(dim * ff_dim),
                "ff2_bias": randoms(dim, 0.02),
            }
        )
    pixels = randoms(channels * image_size * image_size, 0.5)

    patch_features = channels * patch_size * patch_size
    hidden: list[list[float]] = [list(model["class_token"])]
    for py in range(image_size // patch_size):
        for px in range(image_size // patch_size):
            patch = []
            for channel in range(channels):
                for ky in range(patch_size):
                    for kx in range(patch_size):
                        y, x = py * patch_size + ky, px * patch_size + kx
                        patch.append(pixels[(channel * image_size + y) * image_size + x])
            embedded = matvec(model["patch_weight"], patch, dim, patch_features)
            hidden.append([embedded[i] + model["patch_bias"][i] for i in range(dim)])
    hidden = [
        [row[c] + model["position"][t * dim + c] for c in range(dim)]
        for t, row in enumerate(hidden)
    ]
    head_dim = dim // heads
    for block in model["blocks"]:
        normalized = [layer_norm(row) for row in hidden]
        q = [matvec(block["wq"], row, dim, dim) for row in normalized]
        k = [matvec(block["wk"], row, dim, dim) for row in normalized]
        v = [matvec(block["wv"], row, dim, dim) for row in normalized]
        joined = [[0.0] * dim for _ in range(tokens)]
        for head in range(heads):
            start = head * head_dim
            scores = [
                [
                    sum(q[i][start+c] * k[j][start+c] for c in range(head_dim)) / math.sqrt(head_dim)
                    for j in range(tokens)
                ]
                for i in range(tokens)
            ]
            for i in range(tokens):
                maximum = max(scores[i])
                exps = [math.exp(x - maximum) for x in scores[i]]
                total = sum(exps)
                for c in range(head_dim):
                    joined[i][start+c] = sum(
                        exps[j] / total * v[j][start+c] for j in range(tokens)
                    )
        projected = [matvec(block["wo"], row, dim, dim) for row in joined]
        hidden = [
            [hidden[t][c] + projected[t][c] + block["bo"][c] for c in range(dim)]
            for t in range(tokens)
        ]
        normalized = [layer_norm(row) for row in hidden]
        for t in range(tokens):
            first = matvec(block["ff1_weight"], normalized[t], ff_dim, dim)
            first = [x + block["ff1_bias"][i] for i, x in enumerate(first)]
            first = [0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0))) for x in first]
            second = matvec(block["ff2_weight"], first, dim, ff_dim)
            hidden[t] = [hidden[t][c] + second[c] + block["ff2_bias"][c] for c in range(dim)]
    cls = layer_norm(hidden[0])
    cls = [cls[index] * model["final_ln_weight"][index] +
           model["final_ln_bias"][index] for index in range(dim)]
    logits = matvec(model["classifier_weight"], cls, classes, dim)
    logits = [logits[i] + model["classifier_bias"][i] for i in range(classes)]
    predicted = max(range(classes), key=logits.__getitem__)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "alice_model.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"{args.protocol}_VIT_MODEL_V1\n")
        handle.write(f"image_size {image_size}\npatch_size {patch_size}\nchannels {channels}\n")
        handle.write(f"tokens {tokens}\ndim {dim}\nheads {heads}\nff_dim {ff_dim}\n")
        handle.write(f"layers {layers}\nclasses {classes}\nscale_bits {scale_bits}\n")
        for name in ("patch_weight", "patch_bias", "class_token", "position"):
            vector(handle, name, model[name])
        for index, block in enumerate(model["blocks"]):
            handle.write(f"block {index}\n")
            for name in (
                "ln1_weight", "ln1_bias", "wq", "wk", "wv", "wo", "bo",
                "ln2_weight", "ln2_bias", "ff1_weight", "ff1_bias",
                "ff2_weight", "ff2_bias",
            ):
                vector(handle, name, block[name])
        for name in ("final_ln_weight", "final_ln_bias", "classifier_weight", "classifier_bias"):
            vector(handle, name, model[name])
        handle.write("end\n")
    with (args.out_dir / "bob_image.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            f"{args.protocol}_VIT_INPUT_V1\n"
            f"image_size {image_size}\nchannels {channels}\n"
        )
        vector(handle, "pixels", pixels)
        handle.write("end\n")
    (args.out_dir / "expected_prediction.txt").write_text(
        f"predicted_class {predicted}\n", encoding="utf-8"
    )
    (args.out_dir / "reference_logits.txt").write_text(
        "\n".join(f"{x:.12g}" for x in logits) + "\n", encoding="utf-8"
    )
    print(f"fixture={args.out_dir} expected_class={predicted} logits={logits}")


if __name__ == "__main__":
    main()
