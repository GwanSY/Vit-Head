#!/usr/bin/env python3
"""Evaluate the pinned DeiT checkpoint with an independent NumPy FP32 graph.

The validation manifest is CSV without a header: ``image_path,zero_based_label``.
Relative image paths are resolved against --image-root.  This script deliberately
does not manufacture a score when the licensed ImageNet validation set is absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import erf

from convert_deit_checkpoint import StateDict, EXPECTED_CHECKPOINT_SHA256, sha256


def layer_norm(value: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               epsilon: float) -> np.ndarray:
    mean = value.mean(axis=-1, keepdims=True, dtype=np.float32)
    variance = ((value - mean) ** 2).mean(axis=-1, keepdims=True,
                                           dtype=np.float32)
    normalized = (value - mean) / np.sqrt(variance + np.float32(epsilon))
    return normalized * weight + bias


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - value.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted, dtype=np.float32)
    return exponential / exponential.sum(axis=-1, keepdims=True,
                                         dtype=np.float32)


def tensor(state: StateDict, name: str, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(state.values(name, shape), dtype=np.float32).reshape(shape)


class DeiTNumpy:
    def __init__(self, checkpoint: Path, config: dict[str, object]):
        self.state = StateDict(checkpoint)
        self.image_size = int(config["image_size"])
        self.patch_size = int(config["patch_size"])
        self.channels = int(config["num_channels"])
        self.dim = int(config["hidden_size"])
        self.heads = int(config["num_attention_heads"])
        self.head_dim = self.dim // self.heads
        self.ff_dim = int(config["intermediate_size"])
        self.layers = int(config["num_hidden_layers"])
        self.epsilon = float(config["layer_norm_eps"])
        self.tokens = (self.image_size // self.patch_size) ** 2 + 1
        self.classes = self.state.tensors["classifier.bias"].shape[0]
        self.patch_weight = tensor(self.state,
            "vit.embeddings.patch_embeddings.projection.weight",
            (self.dim, self.channels, self.patch_size, self.patch_size)).reshape(
                self.dim, -1)
        self.patch_bias = tensor(self.state,
            "vit.embeddings.patch_embeddings.projection.bias", (self.dim,))
        self.cls = tensor(self.state, "vit.embeddings.cls_token",
                          (1, 1, self.dim))[0]
        self.position = tensor(self.state, "vit.embeddings.position_embeddings",
                               (1, self.tokens, self.dim))[0]

    def close(self) -> None:
        self.state.close()

    def __call__(self, chw: np.ndarray) -> np.ndarray:
        p = self.patch_size
        patches = chw.reshape(self.channels, self.image_size // p, p,
                              self.image_size // p, p)
        patches = patches.transpose(1, 3, 0, 2, 4).reshape(self.tokens - 1, -1)
        hidden = patches @ self.patch_weight.T + self.patch_bias
        hidden = np.concatenate((self.cls, hidden), axis=0) + self.position
        for layer in range(self.layers):
            prefix = f"vit.encoder.layer.{layer}."
            ln1_w = tensor(self.state, prefix + "layernorm_before.weight", (self.dim,))
            ln1_b = tensor(self.state, prefix + "layernorm_before.bias", (self.dim,))
            normalized = layer_norm(hidden, ln1_w, ln1_b, self.epsilon)
            projections = []
            for name in ("query", "key", "value"):
                weight = tensor(self.state,
                    prefix + f"attention.attention.{name}.weight", (self.dim, self.dim))
                bias = tensor(self.state,
                    prefix + f"attention.attention.{name}.bias", (self.dim,))
                projections.append((normalized @ weight.T + bias).reshape(
                    self.tokens, self.heads, self.head_dim).transpose(1, 0, 2))
            query, key, value = projections
            probability = softmax((query @ key.transpose(0, 2, 1)) /
                                  np.float32(math.sqrt(self.head_dim)))
            context = (probability @ value).transpose(1, 0, 2).reshape(
                self.tokens, self.dim)
            weight = tensor(self.state, prefix + "attention.output.dense.weight",
                            (self.dim, self.dim))
            bias = tensor(self.state, prefix + "attention.output.dense.bias", (self.dim,))
            hidden = hidden + context @ weight.T + bias
            ln2_w = tensor(self.state, prefix + "layernorm_after.weight", (self.dim,))
            ln2_b = tensor(self.state, prefix + "layernorm_after.bias", (self.dim,))
            normalized = layer_norm(hidden, ln2_w, ln2_b, self.epsilon)
            weight1 = tensor(self.state, prefix + "intermediate.dense.weight",
                             (self.ff_dim, self.dim))
            bias1 = tensor(self.state, prefix + "intermediate.dense.bias", (self.ff_dim,))
            first = normalized @ weight1.T + bias1
            first = np.float32(0.5) * first * (
                np.float32(1.0) + erf(first / np.float32(math.sqrt(2.0))))
            weight2 = tensor(self.state, prefix + "output.dense.weight",
                             (self.dim, self.ff_dim))
            bias2 = tensor(self.state, prefix + "output.dense.bias", (self.dim,))
            hidden = hidden + first @ weight2.T + bias2
        final_w = tensor(self.state, "vit.layernorm.weight", (self.dim,))
        final_b = tensor(self.state, "vit.layernorm.bias", (self.dim,))
        cls = layer_norm(hidden, final_w, final_b, self.epsilon)[0]
        classifier = tensor(self.state, "classifier.weight",
                            (self.classes, self.dim))
        classifier_bias = tensor(self.state, "classifier.bias", (self.classes,))
        return cls @ classifier.T + classifier_bias


def preprocess(path: Path, size: int, mean: list[float],
               std: list[float]) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(
        (size, size), Image.Resampling.BILINEAR)
    hwc = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return ((hwc - np.asarray(mean, dtype=np.float32)) /
            np.asarray(std, dtype=np.float32)).transpose(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("checkpoints/deit_tiny_patch16_224"))
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/accuracy/fp32_accuracy.csv"))
    args = parser.parse_args()
    checkpoint = args.checkpoint_dir / "pytorch_model.bin"
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint hash mismatch")
    config = json.loads((args.checkpoint_dir / "config.json").read_text())
    prep = json.loads((args.checkpoint_dir / "preprocessor_config.json").read_text())
    samples: list[tuple[Path, int]] = []
    with args.validation_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 2:
                raise ValueError("each validation row must be image_path,label")
            path = Path(row[0])
            samples.append((path if path.is_absolute() else args.image_root / path,
                            int(row[1])))
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    if not samples:
        raise ValueError("validation manifest contains no samples")
    model = DeiTNumpy(checkpoint, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    try:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("image", "label", "prediction", "correct"))
            for path, label in samples:
                logits = model(preprocess(path, int(prep["size"]),
                                          prep["image_mean"], prep["image_std"]))
                prediction = int(np.argmax(logits))
                is_correct = prediction == label
                correct += is_correct
                writer.writerow((path, label, prediction, int(is_correct)))
            writer.writerow(("SUMMARY", len(samples), correct,
                             format(correct / len(samples), ".12g")))
    finally:
        model.close()
    print(f"FP32_CHECKPOINT_ACCURACY={correct / len(samples):.8f} samples={len(samples)}")


if __name__ == "__main__":
    main()
