#!/usr/bin/env python3
"""Prepare role-separated ViT validation oracle shares.

This is the ViT counterpart of ``repro/prepare_bert_oracle.py``.  It is a
trusted, offline validation step: it evaluates the already existing clear and
fixed references, encodes the selected observations in one common ring, and
writes independent CSPRNG additive shares.  The online protocol reads exactly
one role's share and opens aggregate error scalars, never an intermediate
tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

from vit_reference import Graph, load_input, load_model, round_int


MAGIC = b"BLB_VIT_ORACLE_SHARE_V1\0"
ORACLE_ELL = 43
ORACLE_SCALE = 18
ORACLE_MASK = (1 << ORACLE_ELL) - 1
CLEAR_QUANTIZATION_BOUND = 0.5 / (1 << ORACLE_SCALE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_names(blocks: int) -> list[str]:
    names = ["patch_projection"]
    for layer in range(blocks):
        names.extend((f"block_{layer}.q",
                      f"block_{layer}.k",
                      f"block_{layer}.v",
                      f"block_{layer}.qk",
                      f"block_{layer}.softmax",
                      f"block_{layer}.attention_context",
                      f"block_{layer}.output_projection",
                      f"block_{layer}.attention_residual",
                      f"block_{layer}.ffn_residual"))
    names.extend(("cls", "logits"))
    return names


def valid_value(name: str, value: np.ndarray, logical_tokens: int) -> np.ndarray:
    array = np.asarray(value)
    if name.endswith(".qk") or name.endswith(".softmax"):
        if array.ndim != 3:
            raise ValueError(f"{name}: expected head-by-row-by-column tensor")
        return array[:, :logical_tokens, :logical_tokens].reshape(
            -1, logical_tokens)
    if name.startswith("block_"):
        return array[:logical_tokens]
    return array


def encoded_reference(kind: str, value: np.ndarray,
                      source_scale: int) -> np.ndarray:
    if kind == "fixed":
        if source_scale < 0 or source_scale > ORACLE_SCALE:
            raise ValueError("fixed oracle source scale is out of range")
        raw = round_int(np.asarray(value, dtype=np.float64), source_scale)
        encoded = raw.astype(object) * (1 << (ORACLE_SCALE - source_scale))
    elif kind == "clear":
        encoded = np.rint(np.asarray(value, dtype=np.float64) *
                          (1 << ORACLE_SCALE)).astype(object)
    else:
        raise ValueError(f"unsupported oracle kind: {kind}")
    return np.asarray([int(item) & ORACLE_MASK for item in encoded.reshape(-1)],
                      dtype=np.uint64)


def random_shares(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    byte_count = int(target.size) * np.dtype(np.uint64).itemsize
    alice = np.frombuffer(os.urandom(byte_count), dtype=np.uint64).copy()
    alice &= np.uint64(ORACLE_MASK)
    if target.size and not np.any(alice):
        # This event is cryptographically negligible, but rejecting it keeps
        # the validation artifact's no-zero-random-mask contract explicit.
        return random_shares(target)
    bob = (target - alice) & np.uint64(ORACLE_MASK)
    return alice, bob


def write_share(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<Qii", int(values.size), ORACLE_ELL,
                                 ORACLE_SCALE))
        handle.write(np.ascontiguousarray(values, dtype=np.uint64)
                     .tobytes(order="C"))
    path.chmod(0o600)
    return sha256_file(path)


def safe_token(value: str, field: str) -> str:
    if not value or any(character.isspace() for character in value):
        raise ValueError(f"{field} is not a manifest-safe token")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="preprocessed BOB_VIT_INPUT_V1 file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-limit", type=int, default=0)
    args = parser.parse_args()

    model_path = args.model.resolve(strict=True)
    input_path = args.input.resolve(strict=True)
    output = args.output.resolve()
    reference_path = Path(__file__).with_name("vit_reference.py").resolve(strict=True)
    model = load_model(model_path)
    pixels = load_input(input_path)
    checkpoint_blocks = int(model["layers"])
    blocks = checkpoint_blocks if args.block_limit == 0 else args.block_limit
    if blocks <= 0 or blocks > checkpoint_blocks:
        raise ValueError("block limit is outside the checkpoint")

    _, clear_records = Graph(model, fixed=False, block_limit=blocks).run(pixels)
    _, fixed_records = Graph(model, fixed=True, block_limit=blocks).run(pixels)
    logical_tokens = int(model["tokens"])
    names = selected_names(blocks)
    stages: list[dict[str, Any]] = []
    manifest_lines = [
        "BLB_VIT_ORACLE_MANIFEST_V2",
        f"model_sha256 {sha256_file(model_path)}",
        f"input_sha256 {sha256_file(input_path)}",
        f"reference_sha256 {sha256_file(reference_path)}",
        "source_checkpoint_binding_sha256 " + hashlib.sha256(
            safe_token(str(model["source_sha256"]), "source checkpoint id")
            .encode("utf-8")).hexdigest(),
        f"observer_ell {ORACLE_ELL}",
        f"observer_scale {ORACLE_SCALE}",
        f"logical_tokens {logical_tokens}",
        f"executed_blocks {blocks}",
        f"clear_quantization_bound {CLEAR_QUANTIZATION_BOUND:.17g}",
        f"stages {len(names)}",
    ]

    for name in names:
        if name not in clear_records or name not in fixed_records:
            raise ValueError(f"reference record is missing: {name}")
        clear_row = clear_records[name]
        fixed_row = fixed_records[name]
        clear_value = valid_value(name, clear_row["value"], logical_tokens)
        fixed_value = valid_value(name, fixed_row["value"], logical_tokens)
        source_scale = fixed_row["scale"]
        if not isinstance(source_scale, int):
            raise ValueError(f"{name}: fixed reference scale is not integral")
        if clear_value.shape != fixed_value.shape:
            raise ValueError(
                f"{name}: clear/fixed valid shapes differ: "
                f"{clear_value.shape} != {fixed_value.shape}")
        shape = list(clear_value.shape)
        if len(shape) == 1:
            rows, cols = 1, shape[0]
        elif len(shape) == 2:
            rows, cols = shape
        else:
            raise ValueError(f"{name}: observer only supports rank-1/2 tensors")
        elements = int(clear_value.size)
        packing = safe_token(str(fixed_row["packing"]), f"{name}.packing")
        entry: dict[str, Any] = {
            "name": name, "shape": shape, "rows": rows, "cols": cols,
            "elements": elements, "packing": packing,
            "source_scale": source_scale,
            "observer_ell": ORACLE_ELL, "observer_scale": ORACLE_SCALE,
            "clear_quantization_bound": CLEAR_QUANTIZATION_BOUND,
            "shares": {},
        }
        manifest_lines.append(
            f"stage {name} {rows} {cols} {elements} {packing} "
            f"{source_scale}")
        for kind, value in (("fixed", fixed_value), ("clear", clear_value)):
            target = encoded_reference(kind, value, source_scale)
            alice, bob = random_shares(target)
            kind_entry: dict[str, Any] = {}
            for role, share in (("alice", alice), ("bob", bob)):
                filename = f"{name}.{kind}.{role}.bin"
                digest = write_share(output / filename, share)
                kind_entry[role] = {"file": filename, "sha256": digest}
                manifest_lines.append(
                    f"share {name} {kind} {role} {filename} {digest}")
            entry["shares"][kind] = kind_entry
        stages.append(entry)
    manifest_lines.append("end")

    manifest: dict[str, Any] = {
        "schema": "blb.vit.validation-oracle.v2",
        "security_scope": (
            "trusted offline reference dealer; online parties load one CSPRNG "
            "additive share and reveal aggregate error scalars only"),
        "deployment_requirement": (
            "provision Alice and Bob role shares on physically isolated hosts; "
            "a same-UID shared directory is development-only"),
        "model_sha256": sha256_file(model_path),
        "input_sha256": sha256_file(input_path),
        "reference_sha256": sha256_file(reference_path),
        "source_checkpoint_sha256": str(model["source_sha256"]),
        "source_checkpoint_binding_sha256": hashlib.sha256(
            str(model["source_sha256"]).encode("utf-8")).hexdigest(),
        "observer_ell": ORACLE_ELL,
        "observer_scale": ORACLE_SCALE,
        "clear_quantization": "numpy.rint_nearest_even",
        "clear_quantization_bound": CLEAR_QUANTIZATION_BOUND,
        "logical_tokens": logical_tokens,
        "checkpoint_blocks": checkpoint_blocks,
        "executed_blocks": blocks,
        "intermediate_reconstruction_by_online_parties": False,
        "stages": stages,
    }
    output.mkdir(parents=True, exist_ok=True)
    json_manifest = output / "manifest.json"
    text_manifest = output / "manifest.txt"
    json_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text_manifest.write_text(
        "\n".join(manifest_lines) + "\n", encoding="ascii")
    json_manifest.chmod(0o600)
    text_manifest.chmod(0o600)
    print("VIT_ORACLE_PREPROCESS status=PASS "
          f"output={output} stages={len(stages)} ell={ORACLE_ELL} "
          f"scale={ORACLE_SCALE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
