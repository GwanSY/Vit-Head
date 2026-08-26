#!/usr/bin/env python3
"""Run the raw Hugging Face ViT checkpoint in PyTorch for phase-A validation.

This is deliberately dependency-gated.  It refuses to use random weights and
uses ``torch.load(weights_only=True)`` for ordinary PyTorch archives.  The
canonical BOLT text reference remains useful on machines without PyTorch, but
its result must not be called a framework integration result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


B16 = {"image_size": 224, "patch_size": 16, "hidden_size": 768,
       "num_attention_heads": 12, "intermediate_size": 3072,
       "num_hidden_layers": 12, "num_labels": 1000}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    actual_hash = sha256(args.checkpoint)
    if actual_hash != args.expected_sha256:
        raise ValueError(f"checkpoint SHA256 mismatch: {actual_hash} != {args.expected_sha256}")

    try:
        import torch
        from PIL import Image
        from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor
    except ImportError as error:
        raise RuntimeError(
            "phase-A PyTorch reference requires torch, transformers and Pillow; "
            "install them without changing the secure BOLT environment") from error
    try:
        from safetensors.torch import load_file as load_safetensors
    except ImportError:
        load_safetensors = None

    config_data = json.loads(args.config.read_text(encoding="utf-8"))
    for key, expected in B16.items():
        actual = config_data.get(key)
        if actual is None:
            # Hugging Face config uses num_channels but the B16 values above do
            # not need it for this shape gate.
            continue
        if int(actual) != expected:
            raise ValueError(f"phase-A requires {key}={expected}, got {actual}")
    config = ViTConfig.from_json_file(str(args.config))
    model = ViTForImageClassification(config)
    if args.checkpoint.suffix == ".safetensors":
        if load_safetensors is None:
            raise RuntimeError("safetensors.torch is required for a safetensors checkpoint")
        state = load_safetensors(str(args.checkpoint), device="cpu")
    else:
        try:
            state = torch.load(str(args.checkpoint), map_location="cpu",
                               weights_only=True)
        except TypeError as error:
            raise RuntimeError(
                "this PyTorch version lacks safe weights_only loading; refusing an unsafe archive") from error
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint is not a tensor state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()

    if args.preprocessor:
        processor = ViTImageProcessor.from_json_file(str(args.preprocessor))
    else:
        processor = ViTImageProcessor(
            size={"height": config.image_size, "width": config.image_size},
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225])
    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().tolist()
    prediction = max(range(len(logits)), key=logits.__getitem__)
    report = {
        "format": "BOLT_VIT_PYTORCH_REFERENCE_V1",
        "checkpoint_sha256": actual_hash,
        "architecture": "ViTForImageClassification",
        "random_weights": False,
        "top1": prediction,
        "logits": logits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("VIT_PYTORCH_REFERENCE=PASS")
    print(f"VIT_PYTORCH_TOP1={prediction}")
    print(f"VIT_PYTORCH_REFERENCE_REPORT={args.output}")


if __name__ == "__main__":
    main()

