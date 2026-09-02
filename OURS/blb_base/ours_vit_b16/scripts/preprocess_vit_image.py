#!/usr/bin/env python3
"""Create the BOB-only BOLT ViT input using the checkpoint manifest.

The manifest may contain the repository's canonical ``resize/mean/std``
fields or a Hugging Face ``preprocessor_config.json`` payload with
``size/image_mean/image_std``.  No model parameter is read by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def image_size(prep: dict, architecture: dict) -> int:
    value = prep.get("resize", prep.get("size", architecture.get("image_size", 224)))
    if isinstance(value, dict):
        value = value.get("height", value.get("shortest_edge"))
    if value is None:
        raise ValueError("preprocessing size is missing from manifest")
    return int(value)


def channel_stats(prep: dict, key: str, fallback: list[float]) -> list[float]:
    value = prep.get(key)
    if value is None:
        value = prep.get("image_" + key, fallback)
    if len(value) != 3:
        raise ValueError(f"preprocessing {key} must contain three channel values")
    return [float(item) for item in value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(
        "checkpoints/deit_tiny_patch16_224/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prep = manifest.get("preprocessing", {})
    architecture = manifest.get("architecture", {})
    size = image_size(prep, architecture)
    mean = channel_stats(prep, "mean", DEFAULT_MEAN)
    std = channel_stats(prep, "std", DEFAULT_STD)
    if any(value <= 0.0 for value in std):
        raise ValueError("preprocessing std values must be positive")
    image = Image.open(args.image).convert("RGB").resize(
        (size, size), Image.Resampling.BILINEAR)
    pixels = list(image.getdata())
    values: list[float] = []
    for channel in range(3):
        values.extend(
            ((pixel[channel] / 255.0) - mean[channel]) / std[channel]
            for pixel in pixels
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        out.write(f"BOLT_VIT_INPUT_V1\nimage_size {size}\nchannels 3\npixels\n")
        for offset in range(0, len(values), 16):
            out.write(" ".join(format(value, ".9g")
                               for value in values[offset:offset + 16]) + "\n")
        out.write("end\n")


if __name__ == "__main__":
    main()

