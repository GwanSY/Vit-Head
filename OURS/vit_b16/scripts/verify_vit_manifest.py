#!/usr/bin/env python3
"""Fail-closed verifier for a converted private ViT model and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "image_size": 224,
    "patch_size": 16,
    "channels": 3,
    "tokens": 197,
    "dim": 768,
    "heads": 12,
    "head_dim": 64,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    architecture = document.get("architecture", {})
    for key, expected in EXPECTED.items():
        if architecture.get(key) != expected:
            raise SystemExit(
                f"VIT_MANIFEST status=FAIL reason={key}_mismatch "
                f"actual={architecture.get(key)!r} expected={expected!r}")
    if architecture.get("norm_order") != "pre":
        raise SystemExit("VIT_MANIFEST status=FAIL reason=norm_order_not_pre")
    conversion = document.get("conversion", {})
    required_provenance = {
        "checkpoint_sha256_verified": True,
        "integration_eligible": True,
        "synthetic_fixture": False,
        "strict_tensor_consumption": True,
        "random_weights": False,
    }
    for key, expected in required_provenance.items():
        if conversion.get(key) is not expected:
            raise SystemExit(
                f"VIT_MANIFEST status=FAIL reason={key}_mismatch "
                f"actual={conversion.get(key)!r} expected={expected!r}")
    if conversion.get("format") != "BOLT_VIT_MODEL_V3":
        raise SystemExit("VIT_MANIFEST status=FAIL reason=format_mismatch")
    expected_model_hash = document.get("source", {}).get(
        "converted_model_sha256")
    actual_model_hash = sha256(args.model)
    if not expected_model_hash or actual_model_hash != expected_model_hash:
        raise SystemExit(
            "VIT_MANIFEST status=FAIL reason=converted_model_sha256_mismatch")
    source_hash = document.get("source", {}).get("checkpoint_sha256", "")
    if len(source_hash) != 64 or set(source_hash) - set("0123456789abcdef"):
        raise SystemExit("VIT_MANIFEST status=FAIL reason=source_sha256_invalid")
    print(
        "VIT_MANIFEST status=PASS model=vit_b16 tokens=197 physical_tokens=256 "
        f"source_sha256={source_hash} model_sha256={actual_model_hash}")


if __name__ == "__main__":
    main()
