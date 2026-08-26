#!/usr/bin/env python3
"""Convert a real ViT checkpoint to BOLT's strict private model format.

The converter accepts exact Hugging Face ``ViTForImageClassification``, timm,
and torchvision ViT state-dict names.  QKV may be one fused tensor or three
independent projections.  No tensor is synthesized: missing model tensors,
shape mismatches, unexpected architecture metadata, and an unpinned checksum
for an integration-eligible checkpoint are hard errors.

The emitted text is consumed only by ALICE.  BOB receives public dimensions,
never model parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    from safetensors.numpy import load_file as load_safetensors
except ImportError:  # pragma: no cover - exercised only on minimal installs
    load_safetensors = None

from convert_deit_checkpoint import StateDict


B16 = {
    "image_size": 224,
    "patch_size": 16,
    "channels": 3,
    "dim": 768,
    "heads": 12,
    "ff_dim": 3072,
    "layers": 12,
    "classes": 1000,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TensorStore:
    def __init__(self, tensors: dict[str, Any]):
        self.tensors = tensors
        self._consumed: set[str] = set()

    @classmethod
    def open(cls, path: Path) -> "TensorStore":
        if path.suffix == ".safetensors":
            if load_safetensors is None:
                raise RuntimeError("safetensors is required for .safetensors checkpoints")
            loaded = load_safetensors(str(path))
            return cls({key: np.asarray(value) for key, value in loaded.items()})
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as loaded:
                return cls({key: np.asarray(loaded[key]) for key in loaded.files})
        # The restricted loader accepts only the ordinary CPU float32 tensor
        # archive emitted by torch.save(state_dict).  It never executes an
        # arbitrary pickle global.
        archive = StateDict(path)
        tensors: dict[str, Any] = {}
        try:
            for key, ref in archive.tensors.items():
                tensors[key] = np.asarray(
                    archive.values(key, ref.shape), dtype=np.float32).reshape(ref.shape)
        finally:
            archive.close()
        return cls(tensors)

    def keys(self) -> set[str]:
        return set(self.tensors)

    def matches(self, names: list[str]) -> list[str]:
        return [name for name in dict.fromkeys(names) if name in self.tensors]

    def get(self, names: list[str], label: str) -> tuple[str, np.ndarray]:
        matches = self.matches(names)
        if not matches:
            raise ValueError(f"missing {label}; accepted exact keys: {names}")
        if len(matches) != 1:
            raise ValueError(f"ambiguous {label}; multiple exact aliases present: {matches}")
        name = matches[0]
        value = np.asarray(self.tensors[name])
        if value.dtype.kind not in "fiu":
            raise ValueError(f"{label} tensor {name} is not numeric")
        value = value.astype(np.float32, copy=False)
        if not np.isfinite(value).all():
            raise ValueError(f"{label} tensor {name} contains NaN or infinity")
        self._consumed.add(name)
        return name, value

    def assert_all_consumed(self) -> None:
        unexpected = sorted(self.keys() - self._consumed)
        if unexpected:
            raise ValueError(f"strict key check failed; unexpected tensors: {unexpected}")


def vector(store: TensorStore, names: list[str], shape: tuple[int, ...], label: str) -> np.ndarray:
    name, value = store.get(names, label)
    if tuple(value.shape) != shape:
        raise ValueError(f"shape mismatch for {label} ({name}): {tuple(value.shape)} != {shape}")
    return value.reshape(-1)


def aliases(*names: str) -> list[str]:
    # These prefixes cover the state_dict roots emitted by common wrappers;
    # each suffix remains an exact key, so this is not fuzzy matching.
    result: list[str] = []
    for name in names:
        result.extend((name, "module." + name, "model." + name, "module.model." + name))
    return list(dict.fromkeys(result))


def qkv_vectors(store: TensorStore, layer: int, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    hf = f"vit.encoder.layer.{layer}.attention.attention"
    hf_alt = f"encoder.layer.{layer}.attention.attention"
    timm = f"blocks.{layer}.attn"
    torchvision = f"encoder.layers.encoder_layer_{layer}.self_attention"
    fused_weight_names = aliases(
        f"{timm}.qkv.weight", f"{timm}.qkv_w",
        f"{torchvision}.in_proj_weight")
    fused_bias_names = aliases(
        f"{timm}.qkv.bias", f"{timm}.qkv_b",
        f"{torchvision}.in_proj_bias")
    q_weight_names = aliases(f"{hf}.query.weight", f"{hf_alt}.query.weight", f"{timm}.q_proj.weight")
    k_weight_names = aliases(f"{hf}.key.weight", f"{hf_alt}.key.weight", f"{timm}.k_proj.weight")
    v_weight_names = aliases(f"{hf}.value.weight", f"{hf_alt}.value.weight", f"{timm}.v_proj.weight")
    has_fused = bool(store.matches(fused_weight_names))
    has_independent = any(store.matches(names) for names in
                          (q_weight_names, k_weight_names, v_weight_names))
    if has_fused and has_independent:
        raise ValueError(
            f"block {layer} has both fused and independent QKV weights")
    if has_fused:
        fused = vector(store, fused_weight_names, (3 * dim, dim),
                       f"block {layer} fused QKV weight")
        # The V3 target stores biases explicitly.  Missing checkpoint biases
        # must not be silently replaced by synthetic zero parameters.
        bias = vector(store, fused_bias_names, (3 * dim,),
                      f"block {layer} fused QKV bias")
        return (*np.split(fused, 3), *np.split(bias, 3), True)

    q = vector(store, q_weight_names, (dim, dim), f"block {layer} Q weight")
    k = vector(store, k_weight_names, (dim, dim), f"block {layer} K weight")
    v = vector(store, v_weight_names, (dim, dim), f"block {layer} V weight")
    q_bias_names = aliases(f"{hf}.query.bias", f"{hf_alt}.query.bias", f"{timm}.q_proj.bias")
    k_bias_names = aliases(f"{hf}.key.bias", f"{hf_alt}.key.bias", f"{timm}.k_proj.bias")
    v_bias_names = aliases(f"{hf}.value.bias", f"{hf_alt}.value.bias", f"{timm}.v_proj.bias")
    q_bias = vector(store, q_bias_names, (dim,), f"block {layer} Q bias")
    k_bias = vector(store, k_bias_names, (dim,), f"block {layer} K bias")
    v_bias = vector(store, v_bias_names, (dim,), f"block {layer} V bias")
    return q, k, v, q_bias, k_bias, v_bias, True


def block_vectors(store: TensorStore, layer: int, dim: int, ff_dim: int) -> tuple[dict[str, np.ndarray], bool]:
    hf = f"vit.encoder.layer.{layer}"
    hf_alt = f"encoder.layer.{layer}"
    timm = f"blocks.{layer}"
    torchvision = f"encoder.layers.encoder_layer_{layer}"
    q, k, v, q_bias, k_bias, v_bias, has_qkv_bias = qkv_vectors(store, layer, dim)
    return {
        "ln1_weight": vector(store, aliases(f"{hf}.layernorm_before.weight", f"{hf_alt}.layernorm_before.weight", f"{timm}.norm1.weight", f"{torchvision}.ln_1.weight"), (dim,), f"block {layer} LN1 weight"),
        "ln1_bias": vector(store, aliases(f"{hf}.layernorm_before.bias", f"{hf_alt}.layernorm_before.bias", f"{timm}.norm1.bias", f"{torchvision}.ln_1.bias"), (dim,), f"block {layer} LN1 bias"),
        "wq": q, "q_bias": q_bias, "wk": k, "k_bias": k_bias, "wv": v, "v_bias": v_bias,
        "wo": vector(store, aliases(f"{hf}.attention.output.dense.weight", f"{hf_alt}.attention.output.dense.weight", f"{timm}.attn.proj.weight", f"{torchvision}.self_attention.out_proj.weight"), (dim, dim), f"block {layer} output projection weight"),
        "bo": vector(store, aliases(f"{hf}.attention.output.dense.bias", f"{hf_alt}.attention.output.dense.bias", f"{timm}.attn.proj.bias", f"{torchvision}.self_attention.out_proj.bias"), (dim,), f"block {layer} output projection bias"),
        "ln2_weight": vector(store, aliases(f"{hf}.layernorm_after.weight", f"{hf_alt}.layernorm_after.weight", f"{timm}.norm2.weight", f"{torchvision}.ln_2.weight"), (dim,), f"block {layer} LN2 weight"),
        "ln2_bias": vector(store, aliases(f"{hf}.layernorm_after.bias", f"{hf_alt}.layernorm_after.bias", f"{timm}.norm2.bias", f"{torchvision}.ln_2.bias"), (dim,), f"block {layer} LN2 bias"),
        "ff1_weight": vector(store, aliases(f"{hf}.intermediate.dense.weight", f"{hf_alt}.intermediate.dense.weight", f"{timm}.mlp.fc1.weight", f"{torchvision}.mlp.0.weight", f"{torchvision}.mlp.linear_1.weight"), (ff_dim, dim), f"block {layer} MLP FC1 weight"),
        "ff1_bias": vector(store, aliases(f"{hf}.intermediate.dense.bias", f"{hf_alt}.intermediate.dense.bias", f"{timm}.mlp.fc1.bias", f"{torchvision}.mlp.0.bias", f"{torchvision}.mlp.linear_1.bias"), (ff_dim,), f"block {layer} MLP FC1 bias"),
        "ff2_weight": vector(store, aliases(f"{hf}.output.dense.weight", f"{hf_alt}.output.dense.weight", f"{timm}.mlp.fc2.weight", f"{torchvision}.mlp.3.weight", f"{torchvision}.mlp.linear_2.weight"), (dim, ff_dim), f"block {layer} MLP FC2 weight"),
        "ff2_bias": vector(store, aliases(f"{hf}.output.dense.bias", f"{hf_alt}.output.dense.bias", f"{timm}.mlp.fc2.bias", f"{torchvision}.mlp.3.bias", f"{torchvision}.mlp.linear_2.bias"), (dim,), f"block {layer} MLP FC2 bias"),
    }, has_qkv_bias


def emit_vector(handle: Any, name: str, data: np.ndarray) -> None:
    handle.write((name + "\n").encode())
    flat = np.asarray(data, dtype=np.float32).reshape(-1)
    for offset in range(0, flat.size, 16):
        handle.write((" ".join(format(float(value), ".9g") for value in flat[offset:offset + 16]) + "\n").encode())


def read_config(path: Path | None) -> dict[str, Any]:
    return {} if path is None else json.loads(path.read_text(encoding="utf-8"))


def config_bool(config: dict[str, Any], key: str) -> bool | None:
    if key not in config:
        return None
    value = config[key]
    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(f"config {key} must be a JSON boolean or 0/1, got {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preprocessor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-id", default="unidentified-checkpoint")
    parser.add_argument("--source-revision", default="unidentified")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--allow-non-b16", action="store_true",
                        help="only for loader development; not a ViT-B/16 integration claim")
    parser.add_argument(
        "--synthetic-fixture", action="store_true",
        help="mark deterministic/generated loader-test tensors; never integration eligible")
    args = parser.parse_args()

    if any(any(character.isspace() for character in value)
           for value in (args.source_id, args.source_revision)):
        raise ValueError("source-id and source-revision must be single canonical tokens")

    expected_hash = args.expected_sha256.lower() if args.expected_sha256 else None
    if expected_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("--expected-sha256 must be exactly 64 hexadecimal characters")
    if not args.synthetic_fixture and not args.allow_non_b16 and expected_hash is None:
        raise ValueError(
            "real ViT-B/16 integration requires a pinned --expected-sha256")

    checkpoint_hash = sha256(args.checkpoint)
    if expected_hash is not None and checkpoint_hash != expected_hash:
        raise ValueError(f"checkpoint SHA256 mismatch: {checkpoint_hash} != {expected_hash}")
    config = read_config(args.config)
    store = TensorStore.open(args.checkpoint)

    patch_name, patch_raw = store.get(aliases(
        "vit.embeddings.patch_embeddings.projection.weight",
        "embeddings.patch_embeddings.projection.weight",
        "patch_embed.proj.weight", "conv_proj.weight"),
        "patch projection weight")
    if patch_raw.ndim == 4:
        dim, channels, patch_h, patch_w = map(int, patch_raw.shape)
        if patch_h != patch_w:
            raise ValueError("non-square ViT patch projection is unsupported")
        patch_size = patch_h
        patch_weight = patch_raw.reshape(dim, channels * patch_size * patch_size)
    elif patch_raw.ndim == 2:
        dim, patch_dim = map(int, patch_raw.shape)
        patch_size = int(round((patch_dim / 3) ** 0.5))
        channels = 3
        if channels * patch_size * patch_size != patch_dim:
            raise ValueError(f"cannot infer 3-channel patch size from {patch_name} shape {patch_raw.shape}")
        patch_weight = patch_raw
    else:
        raise ValueError(f"unsupported patch projection rank: {patch_raw.ndim}")
    raw_dim, raw_channels, raw_patch_size = dim, channels, patch_size
    patch_bias = vector(store, aliases(
        "vit.embeddings.patch_embeddings.projection.bias",
        "embeddings.patch_embeddings.projection.bias", "patch_embed.proj.bias",
        "conv_proj.bias"), (dim,), "patch projection bias")

    cls = store.get(aliases("vit.embeddings.cls_token", "embeddings.cls_token",
                            "cls_token", "class_token"), "CLS token")[1]
    if tuple(cls.shape) not in ((1, 1, dim), (1, dim)):
        raise ValueError(f"CLS token shape must be [1,1,D] or [1,D], got {cls.shape}")
    cls = cls.reshape(dim)
    position = store.get(aliases("vit.embeddings.position_embeddings",
                                 "embeddings.position_embeddings", "pos_embed",
                                 "encoder.pos_embedding"), "position embedding")[1]
    if position.ndim == 3 and position.shape[0] == 1:
        position = position[0]
    if position.ndim != 2 or position.shape[1] != dim:
        raise ValueError(f"position embedding shape must be [tokens,{dim}], got {position.shape}")
    tokens = int(position.shape[0])
    image_size = int(config.get("image_size", B16["image_size"]))
    patch_size = int(config.get("patch_size", patch_size))
    channels = int(config.get("num_channels", config.get("channels", channels)))
    dim = int(config.get("hidden_size", config.get("dim", dim)))
    heads = int(config.get("num_attention_heads", config.get("heads", B16["heads"])))
    ff_dim = int(config.get("intermediate_size", config.get("ff_dim", B16["ff_dim"])))
    layers = int(config.get("num_hidden_layers", config.get("layers", B16["layers"])))
    classes = int(config.get("num_labels", config.get("classes", 0)))
    if (raw_dim != dim or raw_channels != channels or raw_patch_size != patch_size or
            patch_weight.shape != (dim, channels * patch_size * patch_size)):
        raise ValueError(
            "patch projection shape disagrees with checkpoint/config metadata: "
            f"raw={(raw_dim, raw_channels, raw_patch_size)}, "
            f"config={(dim, channels, patch_size)}")
    if patch_bias.size != dim or cls.size != dim or position.shape[1] != dim:
        raise ValueError("CLS, position, or patch bias dimension disagrees with config")
    classifier_names = aliases("classifier.weight", "head.weight", "heads.head.weight")
    classifier_name, classifier_raw = store.get(classifier_names, "classifier weight")
    if classifier_raw.ndim != 2 or classifier_raw.shape[1] != dim:
        raise ValueError(f"classifier shape must be [classes,{dim}], got {classifier_raw.shape}")
    classes = int(classifier_raw.shape[0]) if classes == 0 else classes
    if classifier_raw.shape[0] != classes:
        raise ValueError(
            "classifier output dimension disagrees with checkpoint/config metadata: "
            f"tensor={classifier_raw.shape[0]}, config={classes}")
    classifier_weight = classifier_raw
    classifier_bias = vector(store, aliases("classifier.bias", "head.bias", "heads.head.bias"), (classes,), "classifier bias")
    model_values = {"image_size": image_size, "patch_size": patch_size, "channels": channels,
                    "dim": dim, "heads": heads, "ff_dim": ff_dim, "layers": layers,
                    "classes": classes, "tokens": tokens}
    if any(model_values[key] <= 0 for key in
           ("image_size", "patch_size", "channels", "dim", "heads",
            "ff_dim", "layers", "classes", "tokens")):
        raise ValueError(f"all architecture dimensions must be positive: {model_values}")
    if image_size % patch_size != 0:
        raise ValueError("image_size must be exactly divisible by patch_size")
    if dim % heads != 0:
        raise ValueError("hidden dimension must be exactly divisible by attention heads")
    for key, expected in B16.items():
        if model_values[key] != expected and not args.allow_non_b16:
            raise ValueError(f"standard ViT-B/16 requires {key}={expected}, got {model_values[key]}")
    is_standard_b16 = all(model_values[key] == expected
                          for key, expected in B16.items())
    # --allow-non-b16 is not a checksum bypass when the loaded tensors turn
    # out to be exactly B/16.  Such a model must either be explicitly marked
    # synthetic or be pinned before any integration artifact is emitted.
    if is_standard_b16 and not args.synthetic_fixture and expected_hash is None:
        raise ValueError(
            "real ViT-B/16 integration requires a pinned --expected-sha256")
    if tokens != (image_size // patch_size) ** 2 + 1:
        raise ValueError("position embedding token count does not match image/patch geometry")
    if config.get("model_type", "vit") not in ("vit", "vision_transformer"):
        raise ValueError(f"unsupported model_type={config.get('model_type')!r}")
    activation = str(config.get("hidden_act", config.get("hidden_activation", "gelu"))).lower()
    if activation != "gelu":
        raise ValueError(
            "BOLT ViT secure graph supports only standard GELU, "
            f"got {activation!r}")
    epsilon = float(config.get("layer_norm_eps", config.get("eps", 1e-12)))
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("LayerNorm epsilon must be finite and non-negative")
    if epsilon >= 0.5 / (1 << 12):
        raise ValueError(
            "LayerNorm epsilon must quantize to zero at scale 12 because "
            "the reused secure LayerNorm core has no explicit epsilon input")

    blocks: list[dict[str, np.ndarray]] = []
    block_qkv_bias: list[bool] = []
    for layer in range(layers):
        block, block_has_qkv_bias = block_vectors(store, layer, dim, ff_dim)
        blocks.append(block)
        block_qkv_bias.append(block_has_qkv_bias)
    if len(set(block_qkv_bias)) != 1:
        raise ValueError("QKV bias presence must be consistent across all blocks")
    qkv_bias = block_qkv_bias[0]
    declared_qkv_bias = config_bool(config, "qkv_bias")
    if declared_qkv_bias is not None and declared_qkv_bias != qkv_bias:
        raise ValueError(
            "checkpoint QKV bias tensors disagree with config qkv_bias="
            f"{config['qkv_bias']!r}")
    final_weight = vector(store, aliases("vit.layernorm.weight", "norm.weight",
                                         "fc_norm.weight", "encoder.ln.weight"),
                          (dim,), "final LayerNorm weight")
    final_bias = vector(store, aliases("vit.layernorm.bias", "norm.bias",
                                       "fc_norm.bias", "encoder.ln.bias"),
                        (dim,), "final LayerNorm bias")
    store.assert_all_consumed()
    norm_order = str(config.get("norm_order", "pre"))
    if norm_order != "pre":
        raise ValueError("BOLT ViT secure graph supports only checkpoint-declared pre-LayerNorm")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as out:
        out.write(b"BOLT_VIT_MODEL_V3\n")
        out.write(f"source_id {args.source_id}\nsource_revision {args.source_revision}\nsource_sha256 {checkpoint_hash}\n".encode())
        out.write(b"architecture pre_norm_vit\nnorm_order pre\n")
        out.write(f"layer_norm_epsilon {epsilon:.17g}\nqkv_bias {int(qkv_bias)}\n".encode())
        for key in ("image_size", "patch_size", "channels", "tokens", "dim", "heads", "ff_dim", "layers", "classes"):
            out.write(f"{key} {model_values[key]}\n".encode())
        out.write(b"scale_bits 12\n")
        emit_vector(out, "patch_weight", patch_weight)
        emit_vector(out, "patch_bias", patch_bias)
        emit_vector(out, "class_token", cls)
        emit_vector(out, "position", position)
        for layer, block in enumerate(blocks):
            out.write(f"block {layer}\n".encode())
            for key in ("ln1_weight", "ln1_bias", "wq", "q_bias", "wk", "k_bias", "wv", "v_bias", "wo", "bo", "ln2_weight", "ln2_bias", "ff1_weight", "ff1_bias", "ff2_weight", "ff2_bias"):
                emit_vector(out, key, block[key])
        emit_vector(out, "final_ln_weight", final_weight)
        emit_vector(out, "final_ln_bias", final_bias)
        emit_vector(out, "classifier_weight", classifier_weight)
        emit_vector(out, "classifier_bias", classifier_bias)
        out.write(b"end\n")
    os.replace(temporary, args.output)

    preprocess = read_config(args.preprocessor) if args.preprocessor else {}
    integration_eligible = (
        is_standard_b16 and not args.allow_non_b16 and
        not args.synthetic_fixture and expected_hash is not None)
    manifest = {
        "source": {"id": args.source_id, "revision": args.source_revision,
                   "checkpoint_sha256": checkpoint_hash,
                   "converted_model_sha256": sha256(args.output)},
        "architecture": {"implementation": "ViTForImageClassification/timm-compatible",
                          "norm_order": "pre", "qkv_bias": qkv_bias,
                          "layer_norm_epsilon": epsilon, **model_values,
                          "head_dim": dim // heads,
                          "patch_order": "row-major patches; C,KY,KX within patch"},
        "preprocessing": preprocess,
        "conversion": {"format": "BOLT_VIT_MODEL_V3", "fuzzy_matching": False,
                        "supports_fused_qkv": True, "supports_independent_qkv": True,
                        "random_weights": args.synthetic_fixture,
                        "synthetic_fixture": args.synthetic_fixture,
                        "checkpoint_sha256_verified": expected_hash is not None,
                        "integration_eligible": integration_eligible,
                        "strict_tensor_consumption": True, "scale_bits": 12,
                        "classifier_source": classifier_name},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CHECKPOINT_LOADER=PASS")
    print(f"CHECKPOINT_ARCHITECTURE={'ViT-B/16' if is_standard_b16 else 'non-B16-development-only'}")
    print(f"CHECKPOINT_INTEGRATION_ELIGIBLE={str(integration_eligible).lower()}")
    print(f"checkpoint_sha256={checkpoint_hash}")
    print(f"converted_sha256={manifest['source']['converted_model_sha256']}")


if __name__ == "__main__":
    main()
