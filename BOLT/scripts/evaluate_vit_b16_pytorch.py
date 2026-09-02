#!/usr/bin/env python3
"""Evaluate the real local ViT-B/16 checkpoint on an ImageNet manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--predictions-output",
        type=Path,
        help="optional per-image CSV with top-1/top-5 predictions and margin",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device; auto selects CUDA when available",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

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
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    model.to(device)
    model.eval()
    correct1 = 0
    correct5 = 0
    prediction_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start:start + args.batch_size]
            images = [Image.open(path).convert("RGB") for path, _ in batch]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {
                name: value.to(device) if torch.is_tensor(value) else value
                for name, value in inputs.items()
            }
            logits = model(**inputs).logits
            labels = torch.tensor(
                [label for _, label in batch], dtype=torch.long, device=device
            )
            predictions = logits.argmax(dim=-1)
            correct1 += int((predictions == labels).sum().item())
            top5 = logits.topk(k=5, dim=-1).indices
            correct5 += int((top5 == labels[:, None]).any(dim=1).sum().item())
            top2_values = logits.topk(k=2, dim=-1).values
            if args.predictions_output:
                for (image, label), prediction, top5_row, top2_row in zip(
                    batch,
                    predictions.cpu().tolist(),
                    top5.cpu().tolist(),
                    top2_values.cpu().tolist(),
                ):
                    prediction_rows.append({
                        "image_path": str(image),
                        "true_label": label,
                        "top1": prediction,
                        "top5": ";".join(str(value) for value in top5_row),
                        "top1_margin": top2_row[0] - top2_row[1],
                        "top1_correct": prediction == label,
                        "top5_correct": label in top5_row,
                    })

    report = {
        "format": "BOLT_VIT_B16_PYTORCH_IMAGENET_EVAL_V1",
        "checkpoint_dir": str(args.checkpoint_dir),
        "manifest": str(args.manifest),
        "samples": len(samples),
        "top1_correct": correct1,
        "top5_correct": correct5,
        "top1": correct1 / len(samples),
        "top5": correct5 / len(samples),
        "random_weights": False,
        "device": str(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "image_path", "true_label", "top1", "top5", "top1_margin",
                "top1_correct", "top5_correct",
            ))
            writer.writeheader()
            writer.writerows(prediction_rows)
    print(f"VIT_PYTORCH_IMAGENET_SAMPLES={len(samples)}")
    print(f"VIT_PYTORCH_IMAGENET_TOP1={correct1 / len(samples):.8f}")
    print(f"VIT_PYTORCH_IMAGENET_TOP5={correct5 / len(samples):.8f}")
    print(f"VIT_PYTORCH_IMAGENET_REPORT={args.output}")


if __name__ == "__main__":
    main()
