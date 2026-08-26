#!/usr/bin/env python3
"""Validate the official BLB text checkpoint and write a SHA256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


IMAGE_DIGEST = "sha256:f696ecc441daeb1e05daaee6b955e12cdc5ea0cbb1f5335dab57f769167d20b4"

TENSORS = {
    "attention.self.query.weight": ((9216, 64), "qkv_head_major_12x768x64"),
    "attention.self.query.bias": ((768, 1), "qkv_bias_12x64"),
    "attention.self.key.weight": ((9216, 64), "qkv_head_major_12x768x64"),
    "attention.self.key.bias": ((768, 1), "qkv_bias_12x64"),
    "attention.self.value.weight": ((9216, 64), "qkv_head_major_12x768x64"),
    "attention.self.value.bias": ((768, 1), "qkv_bias_12x64"),
    "attention.output.dense.weight": ((768, 768), "input_major_768x768"),
    "attention.output.dense.bias": ((768, 1), "vector"),
    "attention.output.LayerNorm.weight": ((768, 1), "vector"),
    "attention.output.LayerNorm.bias": ((768, 1), "vector"),
    "intermediate.dense.weight": ((768, 3072), "input_major_768x3072"),
    "intermediate.dense.bias": ((3072, 1), "vector"),
    "output.dense.weight": ((3072, 768), "input_major_3072x768"),
    "output.dense.bias": ((768, 1), "vector"),
    "output.LayerNorm.weight": ((768, 1), "vector"),
    "output.LayerNorm.bias": ((768, 1), "vector"),
}

EXTRAS = {
    "bert.pooler.dense.weight.txt": ((768, 768), "input_major_768x768"),
    "bert.pooler.dense.bias.txt": ((768, 1), "vector"),
    "classifier.weight.txt": ((768, 2), "input_major_768x2"),
    "classifier.bias.txt": ((2, 1), "vector"),
    "inputs_0_data.txt": ((128, 768), "sequence_major_128x768"),
    "inputs_0_mask.txt": ((128, 1), "attention_mask"),
}


def validate_and_hash(path: Path, shape: tuple[int, int]) -> dict:
    expected_rows, expected_columns = shape
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            line = raw_line.rstrip(b"\r\n")
            columns = 0 if not line else line.count(b",") + 1
            if columns != expected_columns:
                raise RuntimeError(
                    f"shape mismatch {path}: row {rows} has {columns} columns, "
                    f"expected {expected_columns}"
                )
            rows += 1
    if rows != expected_rows:
        raise RuntimeError(f"shape mismatch {path}: {rows} rows, expected {expected_rows}")
    return {
        "path": path.name,
        "shape": list(shape),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--dataset", default="mrpc")
    args = parser.parse_args()
    root = args.model_dir.resolve()
    if not root.is_dir():
        raise RuntimeError(f"missing checkpoint directory: {root}")

    records = []
    for layer in range(12):
        for suffix, (shape, packing) in TENSORS.items():
            name = f"bert.encoder.layer.{layer}.{suffix}.txt"
            path = root / name
            if not path.is_file():
                raise RuntimeError(f"required official checkpoint tensor missing: {path}")
            record = validate_and_hash(path, shape)
            record.update({
                "tensor": name[:-4],
                "layer": layer,
                "packing": packing,
                "scale_source": "w_scale.txt_or_b_scale.txt",
            })
            records.append(record)

    extras = []
    for name, (shape, packing) in EXTRAS.items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"required official checkpoint auxiliary file missing: {path}")
        record = validate_and_hash(path, shape)
        record["packing"] = packing
        extras.append(record)

    scale_files = []
    for name in ("w_scale.txt", "b_scale.txt", "x_scale.txt"):
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing scale metadata: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        scale_files.append({"path": name, "bytes": path.stat().st_size, "sha256": digest})

    manifest = {
        "schema": "blb.bert.checkpoint-manifest.v1",
        "model_id": "official-blb-bert-base-mrpc",
        "source": {
            "zenodo_record": "https://zenodo.org/records/15621724",
            "container_image_digest": IMAGE_DIGEST,
            "container_path": "/home/CKKS-MPC/dataset/quantize/mrpc/weights_txt",
            "dataset": args.dataset,
        },
        "metadata": {
            "num_hidden_layers": 12,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_labels": 2,
            "sequence_length": 128,
        },
        "exported_layers": list(range(12)),
        "pp": {
            "pp_1": "Q/K/V multi-head CKKS BSGS + ct-ct QK^T",
            "pp_2": "CKKS Softmax*V output projection",
            "pp_3": "CKKS FFN1",
            "pp_4": "CKKS FFN2",
        },
        "files": records,
        "auxiliary_files": extras,
        "scale_files": scale_files,
        "pooling_weights_available": True,
        "classifier_weights_available": True,
        "real_layer0_input_available": True,
        "single_layer_weights_ready": True,
        "formal_ready": False,
        "formal_blockers": [
            "three_oracle_runtime_acceptance_pending",
            "pooling_and_classifier_not_wired_into_formal_ckks_path",
        ],
    }
    output = root / "manifest.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(
        "OFFICIAL_CHECKPOINT status=PASS model=bert-base-mrpc layers=12 "
        f"layer_tensors={len(records)} auxiliary={len(extras)} manifest={output} "
        "formal_ready=no"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"OFFICIAL_CHECKPOINT status=BLOCKED reason={exc}")
        raise SystemExit(2)
