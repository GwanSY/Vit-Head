#!/usr/bin/env python3
"""Run the enlarged clear-only sensitivity study.

This command intentionally has no cost-model or secure-runtime dependency. It
is useful when the real HE-MPC composition gate blocks allocator selection:
the 144 isolated risks and the two fixed clear baselines can still be
recomputed, while no secure plan is emitted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import offline_allocator_experiment as experiment
import offline_cost_aware_head_allocator as base


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--preprocess-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-count", type=int, default=512)
    parser.add_argument("--heldout-count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--logit-weight", type=float, default=0.25)
    args = parser.parse_args()
    if args.workers <= 0 or args.batch_size <= 0:
        raise ValueError("workers and batch-size must be positive")
    if not 256 <= args.calibration_count <= 512:
        raise ValueError("calibration-count must be between 256 and 512")
    if args.heldout_count < 1000:
        raise ValueError("heldout-count must be at least 1000")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model = base.vr.load_model(args.model)
    preprocess = base.load_preprocess(args.preprocess_manifest, model)
    calibration_samples = base.read_manifest(
        args.calibration_manifest, args.image_root, args.calibration_count)
    heldout_samples = base.read_manifest(
        args.heldout_manifest, args.image_root, args.heldout_count)
    calibration_pixels = base.load_pixels_for_samples(calibration_samples, preprocess)
    heldout_pixels = base.load_pixels_for_samples(heldout_samples, preprocess)
    calibration_labels = [sample.label for sample in calibration_samples]
    heldout_labels = [sample.label for sample in heldout_samples]

    calibration_pool = experiment.create_pool(
        model, calibration_pixels, calibration_labels,
        calibration_pixels, calibration_labels, args.workers,
        args.logit_weight, args.batch_size)
    try:
        calibration_baseline: list[Any] = []
        calibration_records: list[dict[str, dict[str, Any]]] = []
        for index, pixels in enumerate(calibration_pixels, 1):
            logits, records = base.vr.Graph(
                model, fixed=False, record_values=True).run(pixels)
            calibration_baseline.append(logits)
            calibration_records.append(records)
            if index % 32 == 0 or index == len(calibration_pixels):
                print(f"OURS_CLEAR_BASELINE progress={index}/{len(calibration_pixels)}",
                      flush=True)

        alphas = base.fit_linear_alphas(
            calibration_records, base.MODEL_LAYERS, base.MODEL_HEADS)
        isolated = base.run_isolated_candidates(
            model, calibration_pixels, calibration_labels,
            calibration_baseline, alphas, args.workers, args.logit_weight,
            calibration_pool, args.batch_size, progress_every=12)
        isolated.sort(key=lambda row: (int(row["layer"]), int(row["head"]),
                                       str(row["protocol"])))
        write_csv(output_dir / "isolated_candidate_metrics.csv", [
            "layer", "head", "protocol", "alpha", "validation_loss",
            "baseline_validation_loss", "loss_delta", "mse", "max_abs",
            "relative_l2", "top1_agreement", "baseline_accuracy",
            "candidate_accuracy", "accuracy_drop", "proxy_score", "elements",
        ], isolated)
        isolated_index = base.load_candidate_index(isolated)
        experiment.write_sensitivity_outputs(
            output_dir, isolated, alphas, args.logit_weight)

        heldout_baseline = experiment.clear_baseline_batch(
            model, heldout_pixels, args.batch_size)
        calibration_metrics = base.compare_logits(
            calibration_baseline, calibration_baseline, calibration_labels)
        heldout_metrics = base.compare_logits(
            heldout_baseline, heldout_baseline, heldout_labels)
        write_csv(output_dir / "clear_baseline_accuracy.csv", [
            "dataset", "sample_count", "accuracy_percent", "cross_entropy",
            "status",
        ], [
            {"dataset": "calibration", "sample_count": len(calibration_labels),
             "accuracy_percent": calibration_metrics["baseline_accuracy"] * 100.0,
             "cross_entropy": base.cross_entropy(
                 calibration_baseline, calibration_labels),
             "status": "MEASURED_CLEAR_ONLY"},
            {"dataset": "heldout", "sample_count": len(heldout_labels),
             "accuracy_percent": heldout_metrics["baseline_accuracy"] * 100.0,
             "cross_entropy": base.cross_entropy(heldout_baseline, heldout_labels),
             "status": "MEASURED_CLEAR_ONLY"},
        ])
        (output_dir / "clear_sensitivity_report.json").write_text(
            json.dumps({
                "status": "PASS_CLEAR_ONLY",
                "calibration_count": len(calibration_labels),
                "heldout_count": len(heldout_labels),
                "isolated_entries": len(isolated_index),
                "paths": [base.RELU, base.LINEAR],
                "secure_inference_run": False,
                "cost_model_used": False,
                "allocator_run": False,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        experiment.close_pool(calibration_pool)


if __name__ == "__main__":
    main()
