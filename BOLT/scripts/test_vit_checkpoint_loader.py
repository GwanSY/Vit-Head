#!/usr/bin/env python3
"""Deterministic loader-format tests; these are not checkpoint integration.

The fixtures exercise exact key mapping and fused/independent QKV handling.
They intentionally end with ``CHECKPOINT_INTEGRATION=NOT_RUN_UNIT_TEST_ONLY``
because this test uses synthetic schema fixtures.  Real checkpoint integration
is checked separately with ``BOLT_VIT_MODEL_INSPECT``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def tensors(fused: bool) -> dict[str, np.ndarray]:
    d, c, p, ff, t, classes = 4, 3, 2, 8, 5, 3
    data: dict[str, np.ndarray] = {
        "patch_embed.proj.weight": np.arange(d * c * p * p, dtype=np.float32).reshape(d, c, p, p),
        "patch_embed.proj.bias": np.arange(d, dtype=np.float32),
        "cls_token": np.arange(d, dtype=np.float32).reshape(1, 1, d),
        "pos_embed": np.arange(t * d, dtype=np.float32).reshape(1, t, d),
        "blocks.0.norm1.weight": np.ones(d, dtype=np.float32),
        "blocks.0.norm1.bias": np.zeros(d, dtype=np.float32),
        "blocks.0.attn.proj.weight": np.eye(d, dtype=np.float32),
        "blocks.0.attn.proj.bias": np.arange(d, dtype=np.float32),
        "blocks.0.norm2.weight": np.ones(d, dtype=np.float32),
        "blocks.0.norm2.bias": np.zeros(d, dtype=np.float32),
        "blocks.0.mlp.fc1.weight": np.arange(ff * d, dtype=np.float32).reshape(ff, d),
        "blocks.0.mlp.fc1.bias": np.arange(ff, dtype=np.float32),
        "blocks.0.mlp.fc2.weight": np.arange(d * ff, dtype=np.float32).reshape(d, ff),
        "blocks.0.mlp.fc2.bias": np.arange(d, dtype=np.float32),
        "norm.weight": np.ones(d, dtype=np.float32),
        "norm.bias": np.zeros(d, dtype=np.float32),
        "head.weight": np.arange(classes * d, dtype=np.float32).reshape(classes, d),
        "head.bias": np.arange(classes, dtype=np.float32),
    }
    if fused:
        data["blocks.0.attn.qkv.weight"] = np.arange(3 * d * d, dtype=np.float32).reshape(3 * d, d)
        data["blocks.0.attn.qkv.bias"] = np.arange(3 * d, dtype=np.float32)
    else:
        for offset, name in enumerate(("q", "k", "v")):
            data[f"blocks.0.attn.{name}_proj.weight"] = np.full((d, d), offset + 1, dtype=np.float32)
            data[f"blocks.0.attn.{name}_proj.bias"] = np.full((d,), offset + 1, dtype=np.float32)
    return data


def run_case(root: Path, fused: bool) -> None:
    checkpoint = root / ("fused.npz" if fused else "independent.npz")
    np.savez(checkpoint, **tensors(fused))
    output = root / ("fused_model.txt" if fused else "independent_model.txt")
    manifest = root / ("fused_manifest.json" if fused else "independent_manifest.json")
    config = root / "config.json"
    config.write_text(json.dumps({
        "model_type": "vit", "image_size": 4, "patch_size": 2,
        "num_channels": 3, "hidden_size": 4, "num_attention_heads": 2,
        "intermediate_size": 8, "num_hidden_layers": 1, "num_labels": 3,
        "layer_norm_eps": 1e-12,
    }) + "\n", encoding="utf-8")
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("convert_vit_checkpoint.py")),
        "--checkpoint", str(checkpoint), "--config", str(config),
        "--output", str(output), "--manifest", str(manifest),
        "--source-id", "loader-test", "--source-revision", "deterministic",
        "--allow-non-b16",
    ], check=True)
    fields = output.read_text(encoding="utf-8").split()
    assert fields[0] == "BOLT_VIT_MODEL_V3"
    assert "q_bias" in fields and "k_bias" in fields and "v_bias" in fields
    assert json.loads(manifest.read_text(encoding="utf-8"))["conversion"]["random_weights"] is False
    inspector = Path(__file__).parents[1] / "SCI/build_vit/bin/BOLT_VIT_MODEL_INSPECT"
    if inspector.is_file():
        subprocess.run([str(inspector), str(output)], check=True,
                       stdout=subprocess.DEVNULL)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="bolt_vit_loader_") as directory:
        root = Path(directory)
        run_case(root, True)
        run_case(root, False)
    print("CHECKPOINT_LOADER_UNIT=PASS fused_qkv=true independent_qkv=true")
    print("CHECKPOINT_INTEGRATION=NOT_RUN_UNIT_TEST_ONLY")


if __name__ == "__main__":
    main()
