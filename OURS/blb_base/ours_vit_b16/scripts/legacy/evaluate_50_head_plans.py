#!/usr/bin/env python3
"""Evaluate fixed-clear head plans on a fixed ImageNet sample subset."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from PIL import Image

import vit_reference as vr


MODEL: dict | None = None
PLANS: dict[str, list[list[tuple[str, float]]]] = {}
PREPROCESS: dict[str, object] = {}


def read_plan(path: Path, layers: int, heads: int) -> list[list[tuple[str, float]]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if lines[:2] != [
            "OURS_VIT_HEAD_PLAN_V1",
            "layers=12,heads=12,logical_tokens=197,physical_tokens=256"]:
        raise ValueError(f"unsupported head plan: {path}")
    rows = csv.DictReader(lines[2:])
    required = {"layer", "physical_head", "original_head", "protocol", "alpha"}
    if set(rows.fieldnames or ()) != required:
        raise ValueError(f"head plan columns changed: {path}")
    plan = [[("", 0.0) for _ in range(heads)] for _ in range(layers)]
    for row in rows:
        layer = int(row["layer"])
        physical = int(row["physical_head"])
        original = int(row["original_head"])
        if not (0 <= layer < layers and 0 <= physical < heads and
                0 <= original < heads):
            raise ValueError(f"head plan index is outside the model: {path}")
        if plan[layer][original][0]:
            raise ValueError(f"duplicate head plan entry: {path}")
        plan[layer][original] = (row["protocol"], float(row["alpha"]))
    if any(not protocol for layer in plan for protocol, _ in layer):
        raise ValueError(f"head plan does not cover every head: {path}")
    return plan


def load_pixels(path: Path) -> np.ndarray:
    size = int(PREPROCESS["size"])
    mean = np.asarray(PREPROCESS["mean"], dtype=np.float64)
    std = np.asarray(PREPROCESS["std"], dtype=np.float64)
    image = Image.open(path).convert("RGB").resize(
        (size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float64) / 255.0
    return ((array - mean) / std).transpose(2, 0, 1)


def worker_init(model: dict, plans: dict[str, list[list[tuple[str, float]]]],
                preprocess: dict[str, object]) -> None:
    global MODEL, PLANS, PREPROCESS
    MODEL = model
    PLANS = plans
    PREPROCESS = preprocess


def evaluate_one(task: tuple[int, str, int, str]) -> dict[str, object]:
    index, image_name, label, plan_name = task
    if MODEL is None:
        raise RuntimeError("worker model is not initialized")
    pixels = load_pixels(Path(image_name))
    logits, _ = vr.Graph(MODEL, fixed=True, head_plan=PLANS[plan_name]).run(pixels)
    top1 = int(np.argmax(logits))
    return {
        "index": index,
        "image_path": image_name,
        "true_label": label,
        "plan": plan_name,
        "top1": top1,
        "top1_correct": top1 == label,
    }


def sha256_rows(rows: list[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for image, label in rows:
        digest.update(f"{image},{label}\n".encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--preprocess-manifest", type=Path, required=True)
    parser.add_argument("--plan", action="append", required=True,
                        help="NAME=HEAD_PLAN_PATH")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit <= 0 or args.workers <= 0:
        raise ValueError("--limit and --workers must be positive")

    model = vr.load_model(args.model)
    prep_payload = json.loads(args.preprocess_manifest.read_text(encoding="utf-8"))
    prep = prep_payload.get("preprocessing", {})
    size = prep.get("resize", prep.get("size", model["image_size"]))
    if isinstance(size, dict):
        size = size.get("height", size.get("shortest_edge"))
    preprocess = {
        "size": int(size),
        "mean": prep.get("mean", prep.get("image_mean", [0.485, 0.456, 0.406])),
        "std": prep.get("std", prep.get("image_std", [0.229, 0.224, 0.225])),
    }

    samples: list[tuple[str, int]] = []
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 2:
                raise ValueError("manifest rows must be image_path,zero_based_label")
            samples.append((row[0], int(row[1])))
    samples = samples[:args.limit]
    if len(samples) != args.limit:
        raise ValueError(f"manifest has only {len(samples)} rows")
    resolved = [(str((args.image_root / image).resolve()
                     if not Path(image).is_absolute() else Path(image).resolve()), label)
                for image, label in samples]
    for image, _ in resolved:
        if not Path(image).is_file():
            raise FileNotFoundError(image)

    plans: dict[str, list[list[tuple[str, float]]]] = {}
    for specification in args.plan:
        name, separator, value = specification.partition("=")
        if not separator or not name or not value:
            raise ValueError("--plan must use NAME=HEAD_PLAN_PATH")
        plans[name] = read_plan(Path(value), int(model["layers"]), int(model["heads"]))
    if not plans:
        raise ValueError("at least one head plan is required")

    tasks = [
        (index, image, label, plan_name)
        for plan_name in plans
        for index, (image, label) in enumerate(resolved)
    ]
    started = time.monotonic()
    context = mp.get_context("fork")
    with context.Pool(args.workers, initializer=worker_init,
                      initargs=(model, plans, preprocess)) as pool:
        results = list(pool.imap_unordered(evaluate_one, tasks, chunksize=1))
    results.sort(key=lambda row: (str(row["plan"]), int(row["index"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output.with_suffix(".csv")
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "index", "image_path", "true_label", "plan", "top1", "top1_correct"))
        writer.writeheader()
        writer.writerows(results)

    summaries = []
    for name in plans:
        subset = [row for row in results if row["plan"] == name]
        correct = sum(bool(row["top1_correct"]) for row in subset)
        summaries.append({
            "plan": name,
            "samples": len(subset),
            "top1_correct": correct,
            "top1": correct / len(subset),
        })
    report = {
        "format": "OURS_VIT_FIXED_CLEAR_50_HEAD_PLAN_EVAL_V1",
        "model": str(args.model.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_rows": args.limit,
        "manifest_sha256": sha256_rows(samples),
        "image_root": str(args.image_root.resolve()),
        "preprocess_manifest": str(args.preprocess_manifest.resolve()),
        "workers": args.workers,
        "elapsed_s": time.monotonic() - started,
        "secure_accuracy": "NOT_RUN_SECURE",
        "summaries": summaries,
        "predictions_csv": str(predictions_path.resolve()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"OURS_50_FIXED_CLEAR status=PASS samples={args.limit} plans={len(plans)}")
    for row in summaries:
        print(f"OURS_50_FIXED_CLEAR plan={row['plan']} top1={row['top1']:.6f} "
              f"correct={row['top1_correct']}/{row['samples']}")
    print(f"OURS_50_FIXED_CLEAR_REPORT={args.output}")


if __name__ == "__main__":
    main()
