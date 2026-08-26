#!/usr/bin/env python3
"""Strictly convert the pinned Hugging Face DeiT-tiny PyTorch archive.

This loader deliberately supports only float32 tensor storage and the exact
ViTForImageClassification state-dict schema used by the pinned checkpoint.  It
does not import torch, execute arbitrary pickle globals, or fuzzy-match names.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import pickle
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


SOURCE_ID = "facebook/deit-tiny-patch16-224"
SOURCE_REVISION = "b3428f18dcc7b543470d07f14b4a4157815d1880"
EXPECTED_CHECKPOINT_SHA256 = (
    "e1a51b0c81ff812e079d2189352747947879fee56429e4480064a6695cb34b2c"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class StorageRef:
    key: str
    count: int


@dataclass(frozen=True)
class TensorRef:
    storage: StorageRef
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


class FloatStorage:
    pass


def rebuild_tensor(storage: StorageRef, offset: int, shape: tuple[int, ...],
                   stride: tuple[int, ...], requires_grad: bool,
                   backward_hooks: object) -> TensorRef:
    del requires_grad, backward_hooks
    return TensorRef(storage, int(offset), tuple(shape), tuple(stride))


class RestrictedTorchUnpickler(pickle.Unpickler):
    """Whitelist only the three globals present in this pinned state dict."""

    def find_class(self, module: str, name: str) -> object:
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch", "FloatStorage"): FloatStorage,
            ("torch._utils", "_rebuild_tensor_v2"): rebuild_tensor,
        }
        try:
            return allowed[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"checkpoint contains forbidden pickle global {module}.{name}"
            ) from error

    def persistent_load(self, identity: object) -> StorageRef:
        if not isinstance(identity, tuple) or len(identity) != 5:
            raise pickle.UnpicklingError("malformed storage persistent id")
        tag, storage_type, key, location, count = identity
        if tag != "storage" or storage_type is not FloatStorage:
            raise pickle.UnpicklingError("only float32 storage is accepted")
        if location != "cpu":
            raise pickle.UnpicklingError(f"non-CPU tensor storage: {location}")
        return StorageRef(str(key), int(count))


class StateDict:
    def __init__(self, archive: Path):
        self.zip = zipfile.ZipFile(archive)
        names = set(self.zip.namelist())
        if "archive/data.pkl" not in names:
            raise ValueError("not the expected PyTorch zip archive")
        raw = self.zip.read("archive/data.pkl")
        parsed = RestrictedTorchUnpickler(io.BytesIO(raw)).load()
        if type(parsed) is not collections.OrderedDict:
            raise ValueError("checkpoint root must be an OrderedDict")
        self.tensors: collections.OrderedDict[str, TensorRef] = parsed

    def close(self) -> None:
        self.zip.close()

    def values(self, name: str, shape: tuple[int, ...]) -> list[float]:
        ref = self.tensors[name]
        if ref.shape != shape:
            raise ValueError(f"shape mismatch for {name}: {ref.shape} != {shape}")
        expected_stride: list[int] = []
        running = 1
        for extent in reversed(shape):
            expected_stride.append(running)
            running *= extent
        if ref.stride != tuple(reversed(expected_stride)):
            raise ValueError(f"non-contiguous tensor is forbidden: {name}")
        raw = self.zip.read(f"archive/data/{ref.storage.key}")
        if len(raw) != ref.storage.count * 4:
            raise ValueError(f"storage size mismatch for {name}")
        count = running
        if ref.offset < 0 or ref.offset + count > ref.storage.count:
            raise ValueError(f"tensor view outside storage for {name}")
        return list(struct.unpack_from(f"<{count}f", raw, ref.offset * 4))


def expected_keys(layers: int) -> set[str]:
    keys = {
        "vit.embeddings.cls_token",
        "vit.embeddings.position_embeddings",
        "vit.embeddings.patch_embeddings.projection.weight",
        "vit.embeddings.patch_embeddings.projection.bias",
        "vit.layernorm.weight",
        "vit.layernorm.bias",
        "classifier.weight",
        "classifier.bias",
    }
    suffixes = (
        "attention.attention.query.weight",
        "attention.attention.query.bias",
        "attention.attention.key.weight",
        "attention.attention.key.bias",
        "attention.attention.value.weight",
        "attention.attention.value.bias",
        "attention.output.dense.weight",
        "attention.output.dense.bias",
        "intermediate.dense.weight",
        "intermediate.dense.bias",
        "output.dense.weight",
        "output.dense.bias",
        "layernorm_before.weight",
        "layernorm_before.bias",
        "layernorm_after.weight",
        "layernorm_after.bias",
    )
    for layer in range(layers):
        keys.update(f"vit.encoder.layer.{layer}.{suffix}" for suffix in suffixes)
    return keys


def emit_vector(handle: BinaryIO, name: str, data: list[float]) -> None:
    handle.write((name + "\n").encode())
    for offset in range(0, len(data), 16):
        line = " ".join(format(value, ".9g") for value in data[offset:offset + 16])
        handle.write((line + "\n").encode())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("checkpoints/deit_tiny_patch16_224"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("checkpoints/deit_tiny_patch16_224/bolt_model.txt"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("checkpoints/deit_tiny_patch16_224/manifest.json"),
    )
    args = parser.parse_args()
    config_path = args.checkpoint_dir / "config.json"
    preprocess_path = args.checkpoint_dir / "preprocessor_config.json"
    checkpoint_path = args.checkpoint_dir / "pytorch_model.bin"
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA256 mismatch: {checkpoint_hash}; expected "
            f"{EXPECTED_CHECKPOINT_SHA256}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
    required_config = {
        "model_type": "vit", "image_size": 224, "patch_size": 16,
        "num_channels": 3, "hidden_size": 192,
        "num_attention_heads": 3, "intermediate_size": 768,
        "num_hidden_layers": 12, "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise ValueError(f"unsupported config {key}={config.get(key)!r}")
    image_size = config["image_size"]
    patch_size = config["patch_size"]
    channels = config["num_channels"]
    dim = config["hidden_size"]
    heads = config["num_attention_heads"]
    ff_dim = config["intermediate_size"]
    layers = config["num_hidden_layers"]
    tokens = (image_size // patch_size) ** 2 + 1

    state = StateDict(checkpoint_path)
    try:
        wanted = expected_keys(layers)
        actual = set(state.tensors)
        missing, unexpected = sorted(wanted - actual), sorted(actual - wanted)
        if missing or unexpected:
            raise ValueError(
                f"strict key check failed; missing={missing}; unexpected={unexpected}"
            )
        classes = state.tensors["classifier.bias"].shape[0]
        if classes != 1000:
            raise ValueError(f"expected 1000 classes, got {classes}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("wb") as out:
            out.write(b"BOLT_VIT_MODEL_V2\n")
            out.write(f"source_id {SOURCE_ID}\n".encode())
            out.write(f"source_revision {SOURCE_REVISION}\n".encode())
            out.write(f"source_sha256 {checkpoint_hash}\n".encode())
            out.write(b"architecture pre_norm_vit\nnorm_order pre\n")
            out.write(f"layer_norm_epsilon {config['layer_norm_eps']:.17g}\n".encode())
            out.write(b"qkv_bias 1\n")
            for key, value in (
                ("image_size", image_size), ("patch_size", patch_size),
                ("channels", channels), ("tokens", tokens), ("dim", dim),
                ("heads", heads), ("ff_dim", ff_dim), ("layers", layers),
                ("classes", classes), ("scale_bits", 12),
            ):
                out.write(f"{key} {value}\n".encode())
            emit_vector(out, "patch_weight", state.values(
                "vit.embeddings.patch_embeddings.projection.weight",
                (dim, channels, patch_size, patch_size)))
            emit_vector(out, "patch_bias", state.values(
                "vit.embeddings.patch_embeddings.projection.bias", (dim,)))
            emit_vector(out, "class_token", state.values(
                "vit.embeddings.cls_token", (1, 1, dim)))
            emit_vector(out, "position", state.values(
                "vit.embeddings.position_embeddings", (1, tokens, dim)))
            for layer in range(layers):
                prefix = f"vit.encoder.layer.{layer}."
                out.write(f"block {layer}\n".encode())
                mappings = (
                    ("ln1_weight", "layernorm_before.weight", (dim,)),
                    ("ln1_bias", "layernorm_before.bias", (dim,)),
                    ("wq", "attention.attention.query.weight", (dim, dim)),
                    ("q_bias", "attention.attention.query.bias", (dim,)),
                    ("wk", "attention.attention.key.weight", (dim, dim)),
                    ("k_bias", "attention.attention.key.bias", (dim,)),
                    ("wv", "attention.attention.value.weight", (dim, dim)),
                    ("v_bias", "attention.attention.value.bias", (dim,)),
                    ("wo", "attention.output.dense.weight", (dim, dim)),
                    ("bo", "attention.output.dense.bias", (dim,)),
                    ("ln2_weight", "layernorm_after.weight", (dim,)),
                    ("ln2_bias", "layernorm_after.bias", (dim,)),
                    ("ff1_weight", "intermediate.dense.weight", (ff_dim, dim)),
                    ("ff1_bias", "intermediate.dense.bias", (ff_dim,)),
                    ("ff2_weight", "output.dense.weight", (dim, ff_dim)),
                    ("ff2_bias", "output.dense.bias", (dim,)),
                )
                for target, source, shape in mappings:
                    emit_vector(out, target, state.values(prefix + source, shape))
            emit_vector(out, "final_ln_weight", state.values(
                "vit.layernorm.weight", (dim,)))
            emit_vector(out, "final_ln_bias", state.values(
                "vit.layernorm.bias", (dim,)))
            emit_vector(out, "classifier_weight", state.values(
                "classifier.weight", (classes, dim)))
            emit_vector(out, "classifier_bias", state.values(
                "classifier.bias", (classes,)))
            out.write(b"end\n")
        os.replace(temporary, args.output)
    finally:
        state.close()

    manifest = {
        "source": {
            "id": SOURCE_ID,
            "revision": SOURCE_REVISION,
            "url": f"https://huggingface.co/{SOURCE_ID}/tree/{SOURCE_REVISION}",
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": sha256(config_path),
            "preprocessor_config_sha256": sha256(preprocess_path),
            "converted_model_sha256": sha256(args.output),
        },
        "architecture": {
            "implementation": "ViTForImageClassification",
            "norm_order": "pre",
            "qkv_bias": True,
            "layer_norm_epsilon": config["layer_norm_eps"],
            "image_size": image_size, "patch_size": patch_size,
            "channels": channels, "tokens": tokens, "dim": dim,
            "heads": heads, "head_dim": dim // heads, "ff_dim": ff_dim,
            "layers": layers, "classes": classes,
            "patch_order": "row-major patches; C,KY,KX within patch",
        },
        "preprocessing": {
            "resize": preprocess.get("size"),
            "resample": "PIL.Image.Resampling.BILINEAR",
            "color": "RGB",
            "layout": "CHW",
            "mean": preprocess.get("image_mean"),
            "std": preprocess.get("image_std"),
        },
        "conversion": {
            "format": "BOLT_VIT_MODEL_V2",
            "strict_missing_keys": [], "strict_unexpected_keys": [],
            "fuzzy_matching": False, "scale_bits": 12,
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"CHECKPOINT_SCHEMA_CONVERSION=PASS tensors={len(wanted)}")
    print(f"checkpoint_sha256={checkpoint_hash}")
    print(f"converted_sha256={manifest['source']['converted_model_sha256']}")


if __name__ == "__main__":
    main()
