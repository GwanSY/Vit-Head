#!/usr/bin/env python3
"""Fail-closed BERT-base safetensors -> BLB integer text preprocessing.

This script intentionally uses only the Python standard library. It exports the
repository's existing fixed-point layouts; it does not create missing tensors,
pooling weights, or a classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
import sys
from pathlib import Path


PINNED_MODEL_SHA256 = "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
MODEL_ID = "google-bert/bert-base-uncased"

MODEL_META = {
    "num_hidden_layers": 12,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
}

TENSORS = {
    "attention.self.query.weight": ((768, 768), "qkv_weight", 6),
    "attention.self.query.bias": ((768,), "qkv_bias", 11),
    "attention.self.key.weight": ((768, 768), "qkv_weight", 6),
    "attention.self.key.bias": ((768,), "qkv_bias", 11),
    "attention.self.value.weight": ((768, 768), "qkv_weight", 6),
    "attention.self.value.bias": ((768,), "qkv_bias", 11),
    "attention.output.dense.weight": ((768, 768), "transpose_2d", 6),
    "attention.output.dense.bias": ((768,), "vector", 12),
    "attention.output.LayerNorm.weight": ((768,), "vector", 12),
    "attention.output.LayerNorm.bias": ((768,), "vector", 12),
    "intermediate.dense.weight": ((3072, 768), "transpose_2d", 6),
    "intermediate.dense.bias": ((3072,), "vector", 11),
    "output.dense.weight": ((768, 3072), "transpose_2d", 5),
    "output.dense.bias": ((768,), "vector", 9),
    "output.LayerNorm.weight": ((768,), "vector", 12),
    "output.LayerNorm.bias": ((768,), "vector", 12),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_layers(value: str) -> list[int]:
    if value == "all":
        return list(range(12))
    layers = sorted({int(item) for item in value.split(",")})
    if not layers or layers[0] < 0 or layers[-1] >= 12:
        raise argparse.ArgumentTypeError("layers must be 0..11 or 'all'")
    return layers


def load_header(mm: mmap.mmap) -> tuple[dict, int]:
    if len(mm) < 8:
        raise RuntimeError("truncated safetensors file")
    header_size = struct.unpack_from("<Q", mm, 0)[0]
    if header_size <= 2 or 8 + header_size > len(mm):
        raise RuntimeError("invalid safetensors header length")
    header = json.loads(mm[8 : 8 + header_size])
    return header, 8 + header_size


def tensor_view(mm: mmap.mmap, data_start: int, spec: dict) -> memoryview:
    if spec.get("dtype") != "F32":
        raise RuntimeError(f"unsupported dtype {spec.get('dtype')}; expected F32")
    begin, end = spec["data_offsets"]
    view = memoryview(mm)[data_start + begin : data_start + end]
    return view.cast("f")


def fixed(value: float, scale: int) -> int:
    return int(round(value * (1 << scale)))


class HashedWriter:
    def __init__(self, final_path: Path):
        self.final_path = final_path
        self.tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        self.handle = self.tmp_path.open("w", encoding="ascii", newline="\n")
        self.digest = hashlib.sha256()
        self.rows = 0

    def row(self, values) -> None:
        line = ",".join(str(value) for value in values) + "\n"
        encoded = line.encode("ascii")
        self.handle.write(line)
        self.digest.update(encoded)
        self.rows += 1

    def close(self) -> dict:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        os.replace(self.tmp_path, self.final_path)
        return {
            "path": self.final_path.name,
            "sha256": self.digest.hexdigest(),
            "rows": self.rows,
            "bytes": self.final_path.stat().st_size,
        }


def export_tensor(view: memoryview, shape: tuple[int, ...], packing: str,
                  scale: int, output: Path) -> dict:
    writer = HashedWriter(output)
    try:
        if packing == "vector":
            for index in range(shape[0]):
                writer.row((fixed(view[index], scale),))
        elif packing == "qkv_bias":
            for head in range(12):
                for offset in range(64):
                    writer.row((fixed(view[head * 64 + offset], scale),))
        elif packing == "transpose_2d":
            out_dim, in_dim = shape
            for input_index in range(in_dim):
                writer.row(
                    fixed(view[output_index * in_dim + input_index], scale)
                    for output_index in range(out_dim)
                )
        elif packing == "qkv_weight":
            out_dim, in_dim = shape
            for head in range(12):
                for input_index in range(in_dim):
                    writer.row(
                        fixed(view[(head * 64 + offset) * in_dim + input_index], scale)
                        for offset in range(64)
                    )
        else:
            raise RuntimeError(f"unknown packing {packing}")
        record = writer.close()
    finally:
        view.release()
    record.update({"shape": list(shape), "packing": packing, "fixed_scale_bits": scale})
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("0"))
    args = parser.parse_args()

    config_path = args.raw_dir.resolve() / "config.json"
    model_path = args.raw_dir.resolve() / "model.safetensors"
    output_dir = args.output_dir.resolve()
    for path in (config_path, model_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"required checkpoint file missing: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, expected in MODEL_META.items():
        if config.get(key) != expected:
            raise RuntimeError(f"metadata mismatch: {key}={config.get(key)!r}, expected {expected}")

    model_sha = sha256_file(model_path)
    if model_sha != PINNED_MODEL_SHA256:
        raise RuntimeError(
            f"checkpoint SHA256 mismatch: {model_sha}; expected {PINNED_MODEL_SHA256}"
        )
    if sys.byteorder != "little":
        raise RuntimeError("this standard-library F32 reader requires a little-endian host")

    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    with model_path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            header, data_start = load_header(mm)
            for layer in args.layers:
                for suffix, (shape, packing, scale) in TENSORS.items():
                    tensor_name = f"bert.encoder.layer.{layer}.{suffix}"
                    spec = header.get(tensor_name)
                    if spec is None:
                        raise RuntimeError(f"checkpoint tensor missing: {tensor_name}")
                    if tuple(spec.get("shape", ())) != shape:
                        raise RuntimeError(
                            f"shape mismatch for {tensor_name}: {spec.get('shape')}, expected {shape}"
                        )
                    filename = tensor_name + ".txt"
                    record = export_tensor(
                        tensor_view(mm, data_start, spec), shape, packing, scale,
                        output_dir / filename,
                    )
                    record["tensor"] = tensor_name
                    record["layer"] = layer
                    files.append(record)

    manifest = {
        "schema": "blb.bert.checkpoint-manifest.v1",
        "model_id": MODEL_ID,
        "source": {
            "config": str(config_path),
            "checkpoint": str(model_path),
            "checkpoint_sha256": model_sha,
        },
        "metadata": MODEL_META,
        "exported_layers": args.layers,
        "pp": {
            "pp_1": "Q/K/V multi-head CKKS BSGS + ct-ct QK^T",
            "pp_2": "CKKS Softmax*V output projection",
            "pp_3": "CKKS FFN1",
            "pp_4": "CKKS FFN2",
        },
        "files": files,
        "pooling_connected": False,
        "classifier_connected": False,
        "single_layer_weights_ready": 0 in args.layers,
        "formal_ready": False,
    }
    manifest_path = output_dir / "manifest.json"
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    print(
        "PREPROCESS status=PASS "
        f"layers={','.join(map(str, args.layers))} files={len(files)} manifest={manifest_path} "
        "formal_ready=no"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PREPROCESS status=BLOCKED reason={exc}", file=sys.stderr)
        raise SystemExit(2)
