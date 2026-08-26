#!/usr/bin/env python3
"""Export the trained seven-block MPCViT-CIFAR10 model for Phantom CKKS.

The export is inference-only.  Head permutations are fixed from the public
checkpoint alpha tensors and are applied consistently to Q/K/V and WO.  No
private input is consulted when selecting or ordering heads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch


TOKENS = 65
PHYSICAL_TOKENS = 128
DIM = 256
HEADS = 4
HEAD_DIM = 64
FF_DIM = 512
LAYERS = 7
CLASSES = 10
SCALE = 12
ELL = 37


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed(value: np.ndarray, scale: int) -> np.ndarray:
    return np.rint(np.asarray(value, dtype=np.float64) * (1 << scale)).astype(np.int64)


def write_rows(path: Path, value: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(value, dtype=np.int64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for row in matrix:
            handle.write(",".join(str(int(cell)) for cell in row.reshape(-1)))
            handle.write("\n")
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
    }


def state_dict(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or not isinstance(obj.get("state_dict"), dict):
        raise RuntimeError("checkpoint does not contain state_dict")
    return obj["state_dict"], obj


def cifar_image(path: Path, index: int) -> tuple[np.ndarray, int]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    data = np.asarray(batch["data"], dtype=np.float64)
    labels = batch.get("labels", batch.get("fine_labels"))
    if index < 0 or index >= data.shape[0]:
        raise RuntimeError(f"CIFAR index {index} is outside [0,{data.shape[0]})")
    image = data[index].reshape(3, 32, 32) / 255.0
    mean = np.asarray([0.4914, 0.4822, 0.4465])[:, None, None]
    std = np.asarray([0.2470, 0.2435, 0.2616])[:, None, None]
    return (image - mean) / std, int(labels[index])


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    variance = np.square(x - mean).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + 1.0e-5) * weight + bias


def gelu(x: np.ndarray) -> np.ndarray:
    # PyTorch uses the exact erf form for the checkpoint's default GELU.
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float64))
    return torch.nn.functional.gelu(tensor).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cifar-batch", type=Path, required=True)
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state, checkpoint = state_dict(args.checkpoint)
    image, label = cifar_image(args.cifar_batch, args.image_index)
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    def tensor(name: str) -> np.ndarray:
        if name not in state:
            raise RuntimeError(f"checkpoint tensor is missing: {name}")
        return state[name].detach().cpu().double().numpy()

    conv_w = tensor("tokenizer.conv_layers.0.0.weight")
    conv_b = tensor("tokenizer.conv_layers.0.0.bias")
    class_token = tensor("classifier.class_emb").reshape(DIM)
    position = tensor("classifier.positional_emb").reshape(TOKENS, DIM)
    if conv_w.shape != (DIM, 3, 4, 4):
        raise RuntimeError(f"unexpected tokenizer weight shape {conv_w.shape}")

    # BOB input: token 0 is the public CLS position; tokens 1..64 contain the
    # private 4x4 patches in PyTorch Conv2d channel-major kernel order.
    patch_input = np.zeros((PHYSICAL_TOKENS, DIM), dtype=np.float64)
    patch = 1
    for py in range(0, 32, 4):
        for px in range(0, 32, 4):
            patch_input[patch, :48] = image[:, py:py + 4, px:px + 4].reshape(-1)
            patch += 1
    records.append(write_rows(args.output / "inputs_image_data.txt", fixed(patch_input, SCALE)))
    records.append(write_rows(args.output / "inputs_0_mask.txt", np.r_[np.zeros(TOKENS, dtype=np.int64), np.full(PHYSICAL_TOKENS - TOKENS, -32, dtype=np.int64)]))

    patch_w = np.zeros((DIM, DIM), dtype=np.float64)
    patch_w[:, :48] = conv_w.reshape(DIM, 48)
    records.append(write_rows(args.output / "vit.patch.weight.txt", fixed(patch_w.T, 9)))
    records.append(write_rows(args.output / "vit.patch.bias.txt", fixed(conv_b, SCALE)))
    records.append(write_rows(args.output / "vit.class_token.txt", fixed(class_token, SCALE)))
    records.append(write_rows(args.output / "vit.position.txt", fixed(position, SCALE)))
    records.append(write_rows(args.output / "vit.block.0.pre_norm.weight.txt", fixed(tensor("classifier.blocks.0.pre_norm.weight"), SCALE)))
    records.append(write_rows(args.output / "vit.block.0.pre_norm.bias.txt", fixed(tensor("classifier.blocks.0.pre_norm.bias"), SCALE)))

    # Floating oracle begins with the exact checkpoint patch embedding.
    embedded = np.zeros((TOKENS, DIM), dtype=np.float64)
    embedded[0] = class_token
    embedded[1:] = patch_input[1:TOKENS, :48] @ conv_w.reshape(DIM, 48).T + conv_b
    embedded += position
    oracle: dict[str, np.ndarray] = {"fp32_patch_embedding": embedded.copy()}
    protocols: list[list[str]] = []
    permutations: list[list[int]] = []

    hidden = embedded
    for layer in range(LAYERS):
        base = f"classifier.blocks.{layer}"
        attn = f"{base}.self_attn"
        alpha = tensor(f"{attn}.alpha").reshape(-1)
        if alpha.shape != (HEADS,) or not np.all(np.isin(alpha, [0.0, 1.0])):
            raise RuntimeError(f"layer {layer} alpha is not a fixed public bit vector")
        original_protocols = ["relu" if value >= 0.5 else "linear" for value in alpha]
        permutation = sorted(range(HEADS), key=lambda head: original_protocols[head] != "relu")
        layer_protocols = [original_protocols[head] for head in permutation]
        protocols.append(layer_protocols)
        permutations.append(permutation)

        qkv = tensor(f"{attn}.qkv.weight")
        wo = tensor(f"{attn}.proj.weight")
        bo = tensor(f"{attn}.proj.bias")
        if qkv.shape != (3 * DIM, DIM) or wo.shape != (DIM, DIM):
            raise RuntimeError(f"layer {layer} attention shape mismatch")
        qkv_blocks = qkv.reshape(3, HEADS, HEAD_DIM, DIM)
        qkv_reordered = qkv_blocks[:, permutation].reshape(3 * DIM, DIM)
        wo_reordered = wo.reshape(DIM, HEADS, HEAD_DIM)[:, permutation].reshape(DIM, DIM)

        for part, name in enumerate(("query", "key", "value")):
            weight = qkv_reordered[part * DIM:(part + 1) * DIM]
            if name == "query":
                weight = weight / math.sqrt(HEAD_DIM)
            packed = np.empty((HEADS * DIM, HEAD_DIM), dtype=np.int64)
            weight_i = fixed(weight, 9)
            row = 0
            for head in range(HEADS):
                for input_channel in range(DIM):
                    packed[row] = weight_i[head * HEAD_DIM:(head + 1) * HEAD_DIM, input_channel]
                    row += 1
            stem = args.output / f"bert.encoder.layer.{layer}.attention.self.{name}"
            records.append(write_rows(Path(str(stem) + ".weight.txt"), packed))
            records.append(write_rows(Path(str(stem) + ".bias.txt"), np.zeros(DIM, dtype=np.int64)))
        records.append(write_rows(args.output / f"bert.encoder.layer.{layer}.attention.output.dense.weight.txt", fixed(wo_reordered.T, 6)))
        records.append(write_rows(args.output / f"bert.encoder.layer.{layer}.attention.output.dense.bias.txt", fixed(bo, SCALE)))

        ff1_w = tensor(f"{base}.linear1.weight")
        ff1_b = tensor(f"{base}.linear1.bias")
        ff2_w = tensor(f"{base}.linear2.weight")
        ff2_b = tensor(f"{base}.linear2.bias")
        records.append(write_rows(args.output / f"bert.encoder.layer.{layer}.intermediate.dense.weight.txt", fixed(ff1_w.T, 6)))
        records.append(write_rows(args.output / f"bert.encoder.layer.{layer}.intermediate.dense.bias.txt", fixed(ff1_b, 11)))
        records.append(write_rows(args.output / f"bert.encoder.layer.{layer}.output.dense.weight.txt", fixed(ff2_w.T, 5)))
        records.append(write_rows(args.output / f"bert.encoder.layer.{layer}.output.dense.bias.txt", fixed(ff2_b, 9)))

        norm1_w = tensor(f"{base}.norm1.weight")
        norm1_b = tensor(f"{base}.norm1.bias")
        next_norm_w = tensor(f"classifier.blocks.{layer + 1}.pre_norm.weight") if layer + 1 < LAYERS else tensor("classifier.norm.weight")
        next_norm_b = tensor(f"classifier.blocks.{layer + 1}.pre_norm.bias") if layer + 1 < LAYERS else tensor("classifier.norm.bias")
        records.append(write_rows(args.output / f"bert.encoder.layer.{layer}.attention.output.LayerNorm.weight.txt", fixed(norm1_w, SCALE)))
        records.append(write_rows(args.output / f"bert.encoder.layer.{layer}.attention.output.LayerNorm.bias.txt", fixed(norm1_b, SCALE)))
        records.append(write_rows(args.output / f"bert.encoder.layer.{layer}.output.LayerNorm.weight.txt", fixed(next_norm_w, SCALE)))
        records.append(write_rows(args.output / f"bert.encoder.layer.{layer}.output.LayerNorm.bias.txt", fixed(next_norm_b, SCALE)))

        # Oracle uses original head order/mathematics; the synchronized
        # permutation is checked separately below and cannot change output.
        pre_w = tensor(f"{base}.pre_norm.weight")
        pre_b = tensor(f"{base}.pre_norm.bias")
        normalized = layer_norm(hidden, pre_w, pre_b)
        qkv_value = normalized @ qkv.T
        q, k, v = np.split(qkv_value, 3, axis=-1)
        q = q.reshape(TOKENS, HEADS, HEAD_DIM).transpose(1, 0, 2)
        k = k.reshape(TOKENS, HEADS, HEAD_DIM).transpose(1, 0, 2)
        v = v.reshape(TOKENS, HEADS, HEAD_DIM).transpose(1, 0, 2)
        score = q @ k.transpose(0, 2, 1) / math.sqrt(HEAD_DIM)
        weights = score / TOKENS
        for head, protocol in enumerate(original_protocols):
            if protocol == "relu":
                positive = np.maximum(score[head], 0.0)
                weights[head] = positive / (positive.sum(axis=-1, keepdims=True) + 1.0e-5)
        context = (weights @ v).transpose(1, 0, 2).reshape(TOKENS, DIM)
        attention_output = context @ wo.T + bo
        # Plaintext reorder oracle: permuting Q/K/V head blocks and the
        # matching WO input blocks must be a pure layout transformation.
        q_reordered = q[permutation]
        k_reordered = k[permutation]
        v_reordered = v[permutation]
        score_reordered = q_reordered @ k_reordered.transpose(0, 2, 1) / math.sqrt(HEAD_DIM)
        weights_reordered = score_reordered / TOKENS
        for new_head, protocol in enumerate(layer_protocols):
            if protocol == "relu":
                positive = np.maximum(score_reordered[new_head], 0.0)
                weights_reordered[new_head] = positive / (positive.sum(axis=-1, keepdims=True) + 1.0e-5)
        context_reordered = (weights_reordered @ v_reordered).transpose(1, 0, 2).reshape(TOKENS, DIM)
        reordered_output = context_reordered @ wo_reordered.T + bo
        reorder_error = float(np.max(np.abs(reordered_output - attention_output)))
        if reorder_error > 1.0e-10:
            raise RuntimeError(f"layer {layer} head reorder oracle failed: {reorder_error}")
        attention_residual = hidden + attention_output
        post_attention = layer_norm(attention_residual, norm1_w, norm1_b)
        ff1 = post_attention @ ff1_w.T + ff1_b
        ff2 = gelu(ff1) @ ff2_w.T + ff2_b
        hidden = post_attention + ff2
        oracle[f"fp32_block_{layer}_pre_norm"] = normalized
        oracle[f"fp32_block_{layer}_attention"] = attention_output
        oracle[f"fp32_block_{layer}_reorder_max_abs_error"] = np.asarray([reorder_error])
        oracle[f"fp32_block_{layer}_attention_residual"] = attention_residual
        oracle[f"fp32_block_{layer}_post_attention_norm"] = post_attention
        oracle[f"fp32_block_{layer}_ffn1"] = ff1
        oracle[f"fp32_block_{layer}_output"] = hidden

    final_norm = layer_norm(hidden, tensor("classifier.norm.weight"), tensor("classifier.norm.bias"))
    logits = final_norm[0] @ tensor("classifier.fc.weight").T + tensor("classifier.fc.bias")
    oracle["fp32_final_norm"] = final_norm
    oracle["fp32_logits"] = logits
    oracle["label"] = np.asarray([label], dtype=np.int64)
    oracle["predicted_class"] = np.asarray([int(np.argmax(logits))], dtype=np.int64)
    oracle_path = args.output / "oracle_full.npz"
    np.savez_compressed(oracle_path, **oracle)
    records.append({"path": oracle_path.name, "bytes": oracle_path.stat().st_size, "sha256": sha256(oracle_path)})

    records.append(write_rows(args.output / "classifier.weight.txt", fixed(tensor("classifier.fc.weight").T, 9)))
    records.append(write_rows(args.output / "classifier.bias.txt", fixed(tensor("classifier.fc.bias"), SCALE)))
    protocol_path = args.output / "head_protocols.json"
    protocol_path.write_text(json.dumps({"protocols": protocols, "permutations": permutations}, indent=2) + "\n", encoding="utf-8")
    records.append({"path": protocol_path.name, "bytes": protocol_path.stat().st_size, "sha256": sha256(protocol_path)})
    protocol_text_path = args.output / "head_protocols.txt"
    protocol_text_path.write_text("\n".join(",".join(row) for row in protocols) + "\n", encoding="ascii")
    records.append({"path": protocol_text_path.name, "bytes": protocol_text_path.stat().st_size, "sha256": sha256(protocol_text_path)})

    manifest = {
        "schema": "ours.vit65.full.v1",
        "source": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "arch": checkpoint.get("arch"),
            "reported_metric": checkpoint.get("metric"),
            "epoch": checkpoint.get("epoch"),
            "cifar_batch": str(args.cifar_batch.resolve()),
            "image_index": args.image_index,
            "label": label,
        },
        "layout": {"logical_tokens": TOKENS, "physical_tokens": PHYSICAL_TOKENS, "dim": DIM, "heads": HEADS, "head_dim": HEAD_DIM, "ff_dim": FF_DIM, "layers": LAYERS, "classes": CLASSES},
        "head_protocols": protocols,
        "head_permutations_new_to_old": permutations,
        "retraining": False,
        "private_input_dependent_branching": False,
        "fp32_predicted_class": int(np.argmax(logits)),
        "files": records,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VIT65_FULL_EXPORT status=PASS layers={LAYERS} label={label} fp32_prediction={int(np.argmax(logits))} output={args.output}")


if __name__ == "__main__":
    main()
