#!/usr/bin/env python3
"""Deterministic loader-format tests; never checkpoint integration claims."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from convert_vit_checkpoint import TensorStore, qkv_vectors, sha256


D, C, P, FF, TOKENS, CLASSES = 4, 3, 2, 8, 5, 3
CONVERTER = Path(__file__).with_name("convert_vit_checkpoint.py")


def _common_values() -> dict[str, np.ndarray]:
    return {
        "patch_weight": np.arange(D * C * P * P, dtype=np.float32).reshape(D, C, P, P),
        "patch_bias": np.arange(D, dtype=np.float32),
        "cls": np.arange(D, dtype=np.float32).reshape(1, 1, D),
        "position": np.arange(TOKENS * D, dtype=np.float32).reshape(1, TOKENS, D),
        "ln1_weight": np.ones(D, dtype=np.float32),
        "ln1_bias": np.zeros(D, dtype=np.float32),
        "out_weight": np.eye(D, dtype=np.float32),
        "out_bias": np.arange(D, dtype=np.float32),
        "ln2_weight": np.full(D, 2, dtype=np.float32),
        "ln2_bias": np.full(D, -2, dtype=np.float32),
        "fc1_weight": np.arange(FF * D, dtype=np.float32).reshape(FF, D),
        "fc1_bias": np.arange(FF, dtype=np.float32),
        "fc2_weight": np.arange(D * FF, dtype=np.float32).reshape(D, FF),
        "fc2_bias": np.arange(D, dtype=np.float32),
        "final_weight": np.full(D, 3, dtype=np.float32),
        "final_bias": np.full(D, -3, dtype=np.float32),
        "head_weight": np.arange(CLASSES * D, dtype=np.float32).reshape(CLASSES, D),
        "head_bias": np.arange(CLASSES, dtype=np.float32),
    }


def tensors(schema: str) -> dict[str, np.ndarray]:
    """Build exact timm, Hugging Face, or torchvision state-dict keys."""
    value = _common_values()
    if schema == "timm_fused":
        data = {
            "patch_embed.proj.weight": value["patch_weight"],
            "patch_embed.proj.bias": value["patch_bias"],
            "cls_token": value["cls"],
            "pos_embed": value["position"],
            "blocks.0.norm1.weight": value["ln1_weight"],
            "blocks.0.norm1.bias": value["ln1_bias"],
            "blocks.0.attn.proj.weight": value["out_weight"],
            "blocks.0.attn.proj.bias": value["out_bias"],
            "blocks.0.norm2.weight": value["ln2_weight"],
            "blocks.0.norm2.bias": value["ln2_bias"],
            "blocks.0.mlp.fc1.weight": value["fc1_weight"],
            "blocks.0.mlp.fc1.bias": value["fc1_bias"],
            "blocks.0.mlp.fc2.weight": value["fc2_weight"],
            "blocks.0.mlp.fc2.bias": value["fc2_bias"],
            "norm.weight": value["final_weight"],
            "norm.bias": value["final_bias"],
            "head.weight": value["head_weight"],
            "head.bias": value["head_bias"],
        }
        data["blocks.0.attn.qkv.weight"] = (
            np.arange(3 * D * D, dtype=np.float32).reshape(3 * D, D) + 100)
        data["blocks.0.attn.qkv.bias"] = np.arange(3 * D, dtype=np.float32) + 1000
        return data

    if schema == "hf_independent":
        prefix = "vit.encoder.layer.0"
        data = {
            "vit.embeddings.patch_embeddings.projection.weight": value["patch_weight"],
            "vit.embeddings.patch_embeddings.projection.bias": value["patch_bias"],
            "vit.embeddings.cls_token": value["cls"],
            "vit.embeddings.position_embeddings": value["position"],
            f"{prefix}.layernorm_before.weight": value["ln1_weight"],
            f"{prefix}.layernorm_before.bias": value["ln1_bias"],
            f"{prefix}.attention.output.dense.weight": value["out_weight"],
            f"{prefix}.attention.output.dense.bias": value["out_bias"],
            f"{prefix}.layernorm_after.weight": value["ln2_weight"],
            f"{prefix}.layernorm_after.bias": value["ln2_bias"],
            f"{prefix}.intermediate.dense.weight": value["fc1_weight"],
            f"{prefix}.intermediate.dense.bias": value["fc1_bias"],
            f"{prefix}.output.dense.weight": value["fc2_weight"],
            f"{prefix}.output.dense.bias": value["fc2_bias"],
            "vit.layernorm.weight": value["final_weight"],
            "vit.layernorm.bias": value["final_bias"],
            "classifier.weight": value["head_weight"],
            "classifier.bias": value["head_bias"],
        }
        attention = f"{prefix}.attention.attention"
        for offset, name in enumerate(("query", "key", "value")):
            data[f"{attention}.{name}.weight"] = (
                np.arange(D * D, dtype=np.float32).reshape(D, D) + (offset + 1) * 100)
            data[f"{attention}.{name}.bias"] = (
                np.arange(D, dtype=np.float32) + (offset + 1) * 1000)
        return data

    if schema == "torchvision_fused":
        prefix = "encoder.layers.encoder_layer_0"
        data = {
            "conv_proj.weight": value["patch_weight"],
            "conv_proj.bias": value["patch_bias"],
            "class_token": value["cls"],
            "encoder.pos_embedding": value["position"],
            f"{prefix}.ln_1.weight": value["ln1_weight"],
            f"{prefix}.ln_1.bias": value["ln1_bias"],
            f"{prefix}.self_attention.out_proj.weight": value["out_weight"],
            f"{prefix}.self_attention.out_proj.bias": value["out_bias"],
            f"{prefix}.ln_2.weight": value["ln2_weight"],
            f"{prefix}.ln_2.bias": value["ln2_bias"],
            f"{prefix}.mlp.0.weight": value["fc1_weight"],
            f"{prefix}.mlp.0.bias": value["fc1_bias"],
            f"{prefix}.mlp.3.weight": value["fc2_weight"],
            f"{prefix}.mlp.3.bias": value["fc2_bias"],
            "encoder.ln.weight": value["final_weight"],
            "encoder.ln.bias": value["final_bias"],
            "heads.head.weight": value["head_weight"],
            "heads.head.bias": value["head_bias"],
        }
        attention = f"{prefix}.self_attention"
        data[f"{attention}.in_proj_weight"] = (
            np.arange(3 * D * D, dtype=np.float32).reshape(3 * D, D) + 500)
        data[f"{attention}.in_proj_bias"] = np.arange(3 * D, dtype=np.float32) + 5000
        return data

    raise ValueError(f"unknown fixture schema {schema!r}")


def expected_qkv(data: dict[str, np.ndarray], schema: str) -> tuple[np.ndarray, ...]:
    if schema == "timm_fused":
        weight = data["blocks.0.attn.qkv.weight"]
        bias = data["blocks.0.attn.qkv.bias"]
        return tuple(part.reshape(-1) for part in (*np.split(weight, 3, axis=0),
                                                   *np.split(bias, 3)))
    if schema == "torchvision_fused":
        prefix = "encoder.layers.encoder_layer_0.self_attention"
        weight = data[f"{prefix}.in_proj_weight"]
        bias = data[f"{prefix}.in_proj_bias"]
        return tuple(part.reshape(-1) for part in (*np.split(weight, 3, axis=0),
                                                   *np.split(bias, 3)))
    attention = "vit.encoder.layer.0.attention.attention"
    result: list[np.ndarray] = []
    for suffix in ("weight", "bias"):
        for name in ("query", "key", "value"):
            result.append(data[f"{attention}.{name}.{suffix}"].reshape(-1))
    # qkv_vectors returns Q/K/V weights followed by Q/K/V biases.
    return tuple(result[:3] + result[3:])


def assert_qkv_mapping(data: dict[str, np.ndarray], schema: str) -> tuple[np.ndarray, ...]:
    qkv_keys = {
        key: value for key, value in data.items()
        if ("qkv" in key or "in_proj" in key or
            ".query." in key or ".key." in key or ".value." in key)
    }
    store = TensorStore(qkv_keys)
    actual = qkv_vectors(store, 0, D)
    assert actual[-1] is True
    expected = expected_qkv(data, schema)
    for got, wanted in zip(actual[:-1], expected, strict=True):
        np.testing.assert_array_equal(got, wanted)
    store.assert_all_consumed()
    return expected


def config(path: Path) -> None:
    path.write_text(json.dumps({
        "model_type": "vit", "image_size": 4, "patch_size": 2,
        "num_channels": 3, "hidden_size": 4, "num_attention_heads": 2,
        "intermediate_size": 8, "num_hidden_layers": 1, "num_labels": 3,
        "layer_norm_eps": 1e-12, "qkv_bias": True,
    }) + "\n", encoding="utf-8")


def emitted_vector(tokens: list[str], name: str, count: int) -> np.ndarray:
    start = tokens.index(name) + 1
    return np.asarray([float(item) for item in tokens[start:start + count]],
                      dtype=np.float32)


def run_case(root: Path, schema: str, pin_synthetic_hash: bool = False) -> None:
    data = tensors(schema)
    expected = assert_qkv_mapping(data, schema)
    checkpoint = root / f"{schema}.npz"
    np.savez(checkpoint, **data)
    output = root / f"{schema}_model.txt"
    manifest_path = root / f"{schema}_manifest.json"
    config_path = root / f"{schema}_config.json"
    config(config_path)
    command = [
        sys.executable, str(CONVERTER), "--checkpoint", str(checkpoint),
        "--config", str(config_path), "--output", str(output),
        "--manifest", str(manifest_path), "--source-id", "loader-test",
        "--source-revision", "deterministic", "--allow-non-b16",
        "--synthetic-fixture",
    ]
    if pin_synthetic_hash:
        command.extend(("--expected-sha256", sha256(checkpoint)))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

    tokens = output.read_text(encoding="utf-8").split()
    assert tokens[0] == "BOLT_VIT_MODEL_V3"
    for name, count, wanted in (
            ("wq", D * D, expected[0]), ("q_bias", D, expected[3]),
            ("wk", D * D, expected[1]), ("k_bias", D, expected[4]),
            ("wv", D * D, expected[2]), ("v_bias", D, expected[5])):
        np.testing.assert_array_equal(emitted_vector(tokens, name, count), wanted)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conversion = manifest["conversion"]
    assert conversion["random_weights"] is True
    assert conversion["synthetic_fixture"] is True
    assert conversion["integration_eligible"] is False
    assert conversion["checkpoint_sha256_verified"] is pin_synthetic_hash
    assert conversion["strict_tensor_consumption"] is True
    inspector = Path(__file__).parents[1] / "SCI/build_vit/bin/BOLT_VIT_MODEL_INSPECT"
    if inspector.is_file():
        subprocess.run([str(inspector), str(output)], check=True,
                       stdout=subprocess.DEVNULL)


def assert_failure(command: list[str], message: str) -> None:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    assert result.returncode != 0, result.stdout
    assert message in result.stdout, result.stdout


def test_fail_closed_guards(root: Path) -> None:
    checkpoint = root / "guard.npz"
    np.savez(checkpoint, **tensors("timm_fused"))
    config_path = root / "guard_config.json"
    config(config_path)
    base = [
        sys.executable, str(CONVERTER), "--checkpoint", str(checkpoint),
        "--config", str(config_path), "--output", str(root / "guard_model.txt"),
        "--manifest", str(root / "guard_manifest.json"),
    ]
    assert_failure(base, "requires a pinned --expected-sha256")
    assert_failure(base + ["--allow-non-b16", "--synthetic-fixture",
                           "--expected-sha256", "0" * 64],
                   "checkpoint SHA256 mismatch")

    ambiguous = tensors("timm_fused")
    ambiguous["module.blocks.0.attn.qkv.weight"] = ambiguous[
        "blocks.0.attn.qkv.weight"].copy()
    store = TensorStore({key: value for key, value in ambiguous.items()
                         if "qkv" in key})
    try:
        qkv_vectors(store, 0, D)
    except ValueError as error:
        assert "ambiguous" in str(error)
    else:
        raise AssertionError("duplicate exact aliases were accepted")

    missing_bias = tensors("hf_independent")
    del missing_bias["vit.encoder.layer.0.attention.attention.key.bias"]
    store = TensorStore({key: value for key, value in missing_bias.items()
                         if ".attention.attention." in key})
    try:
        qkv_vectors(store, 0, D)
    except ValueError as error:
        assert "missing block 0 K bias" in str(error)
    else:
        raise AssertionError("missing QKV bias was silently synthesized")

    unexpected = tensors("timm_fused")
    unexpected["unexpected.weight"] = np.ones(1, dtype=np.float32)
    unexpected_checkpoint = root / "unexpected.npz"
    np.savez(unexpected_checkpoint, **unexpected)
    unexpected_command = [
        sys.executable, str(CONVERTER), "--checkpoint", str(unexpected_checkpoint),
        "--config", str(config_path), "--output", str(root / "unexpected_model.txt"),
        "--manifest", str(root / "unexpected_manifest.json"),
        "--allow-non-b16", "--synthetic-fixture",
    ]
    assert_failure(unexpected_command, "unexpected tensors")

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    activation_config = root / "activation_config.json"
    activation_config.write_text(json.dumps({
        **config_data, "hidden_act": "gelu_new",
    }) + "\n", encoding="utf-8")
    activation_command = [
        sys.executable, str(CONVERTER), "--checkpoint", str(checkpoint),
        "--config", str(activation_config),
        "--output", str(root / "activation_model.txt"),
        "--manifest", str(root / "activation_manifest.json"),
        "--allow-non-b16", "--synthetic-fixture",
    ]
    assert_failure(activation_command, "supports only standard GELU")

    epsilon_config = root / "epsilon_config.json"
    epsilon_config.write_text(json.dumps({
        **config_data, "layer_norm_eps": 0.5 / (1 << 12),
    }) + "\n", encoding="utf-8")
    epsilon_command = [
        sys.executable, str(CONVERTER), "--checkpoint", str(checkpoint),
        "--config", str(epsilon_config),
        "--output", str(root / "epsilon_model.txt"),
        "--manifest", str(root / "epsilon_manifest.json"),
        "--allow-non-b16", "--synthetic-fixture",
    ]
    assert_failure(epsilon_command, "must quantize to zero at scale 12")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="blb_vit_loader_") as directory:
        root = Path(directory)
        run_case(root, "timm_fused")
        run_case(root, "hf_independent")
        run_case(root, "torchvision_fused", pin_synthetic_hash=True)
        test_fail_closed_guards(root)
    print("CHECKPOINT_LOADER_UNIT=PASS fused_qkv=true independent_qkv=true torchvision=true")
    print("CHECKPOINT_STRICTNESS_UNIT=PASS sha_pin=true provenance=true aliases=true tensors=true activation=true epsilon=true")
    print("CHECKPOINT_INTEGRATION=NOT_RUN_UNIT_FIXTURES_ONLY")


if __name__ == "__main__":
    main()
