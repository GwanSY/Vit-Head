#!/usr/bin/env python3
"""Evaluate the real local ViT-B/16 checkpoint on an ImageNet manifest."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000,
                        help="print a progress record every N evaluated images; 0 disables")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be nonnegative")

    try:
        import torch
        from PIL import Image
        from transformers import (AutoImageProcessor,
                                  ViTForImageClassification)
    except ImportError as error:
        raise RuntimeError(
            "ImageNet PyTorch evaluation requires the prepared .venv-vit-b16") from error

    samples: list[tuple[Path, int]] = []
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 2:
                raise ValueError("manifest rows must be image_path,zero_based_label")
            image = Path(row[0])
            samples.append((image if image.is_absolute() else args.image_root / image,
                            int(row[1])))
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    if not samples:
        raise ValueError("manifest contains no samples")
    if any(not image.is_file() for image, _ in samples):
        missing = next(image for image, _ in samples if not image.is_file())
        raise FileNotFoundError(missing)

    processor = AutoImageProcessor.from_pretrained(str(args.checkpoint_dir))
    model = ViTForImageClassification.from_pretrained(
        str(args.checkpoint_dir), local_files_only=True, use_safetensors=True)
    model.eval()
    correct1 = 0
    correct5 = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start:start + args.batch_size]
            images = []
            for path, _ in batch:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            logits = model(**inputs).logits
            labels = torch.tensor([label for _, label in batch], dtype=torch.long)
            predictions = logits.argmax(dim=-1)
            correct1 += int((predictions == labels).sum().item())
            top5 = logits.topk(k=5, dim=-1).indices
            correct5 += int((top5 == labels[:, None]).any(dim=1).sum().item())
            completed = min(start + len(batch), len(samples))
            if (args.progress_every > 0 and
                    (completed == len(samples) or
                     completed // args.progress_every !=
                     start // args.progress_every)):
                elapsed = time.perf_counter() - started
                print(
                    "VIT_PYTORCH_IMAGENET_PROGRESS="
                    f"{completed}/{len(samples)} elapsed_seconds={elapsed:.3f}",
                    flush=True)

    elapsed = time.perf_counter() - started

    report = {
        "format": "BOLT_VIT_B16_PYTORCH_IMAGENET_EVAL_V1",
        "checkpoint_dir": str(args.checkpoint_dir),
        "manifest": str(args.manifest),
        "samples": len(samples),
        "top1_correct": correct1,
        "top5_correct": correct5,
        "top1": correct1 / len(samples),
        "top5": correct5 / len(samples),
        "evaluation_seconds": elapsed,
        "images_per_second": len(samples) / elapsed,
        "random_weights": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"VIT_PYTORCH_IMAGENET_SAMPLES={len(samples)}")
    print(f"VIT_PYTORCH_IMAGENET_TOP1={correct1 / len(samples):.8f}")
    print(f"VIT_PYTORCH_IMAGENET_TOP5={correct5 / len(samples):.8f}")
    print(f"VIT_PYTORCH_IMAGENET_REPORT={args.output}")


if __name__ == "__main__":
    main()
