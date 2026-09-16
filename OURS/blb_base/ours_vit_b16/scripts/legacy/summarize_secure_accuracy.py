#!/usr/bin/env python3
"""Aggregate full-12 secure samples with distribution and provenance checks."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(records: list[dict], key: str) -> dict[str, float]:
    values = [float(record[key]) for record in records]
    return {
        "mean": statistics.fmean(values),
        "stddev_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head-plan", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.run_root.glob("*/sample_result.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if len(records) != args.expected_samples:
        raise ValueError(f"expected {args.expected_samples} secure samples, found {len(records)}")
    if sorted(record["index"] for record in records) != list(
            range(1, args.expected_samples + 1)):
        raise ValueError("secure sample indexes are incomplete or duplicated")
    expected_plan_hash = sha256(args.head_plan)
    expected_manifest_hash = sha256(args.manifest)
    invariant_names = ("binary_sha256", "model_sha256", "image_id", "network",
                       "head_plan_sha256", "manifest_file_sha256",
                       "physical_gpus", "gpu_models", "gpu_uuids",
                       "gpu_driver_versions", "network_shaping_verified")
    invariants = {name: {record[name] for record in records}
                  for name in invariant_names}
    if any(len(values) != 1 for values in invariants.values()):
        raise ValueError(f"secure sample cohort invariants differ: {invariants}")
    if (next(iter(invariants["head_plan_sha256"])) != expected_plan_hash or
            next(iter(invariants["manifest_file_sha256"])) != expected_manifest_hash or
            next(iter(invariants["network"])) != "1Gbps/0.3ms_RTT" or
            next(iter(invariants["network_shaping_verified"])) is not True):
        raise ValueError("secure cohort plan/manifest/network binding failed")

    n = len(records)
    count = lambda name: sum(bool(record[name]) for record in records)
    summary = {
        "format": "OURS_VIT_SECURE_ACCURACY_V1",
        "status": "PASS",
        "samples": n,
        "manifest": str(args.manifest.resolve()),
        "manifest_file_sha256": expected_manifest_hash,
        "head_plan": str(args.head_plan.resolve()),
        "head_plan_sha256": expected_plan_hash,
        "binary_sha256": next(iter(invariants["binary_sha256"])),
        "model_sha256": next(iter(invariants["model_sha256"])),
        "image_id": next(iter(invariants["image_id"])),
        "hardware": {
            "physical_gpus": next(iter(invariants["physical_gpus"])),
            "gpu_models": next(iter(invariants["gpu_models"])),
            "gpu_uuids": next(iter(invariants["gpu_uuids"])),
            "gpu_driver_versions": next(iter(invariants["gpu_driver_versions"])),
        },
        "network": "1Gbps/0.3ms_RTT",
        "network_shaping_verified": True,
        "communication_scope": "Alice_plus_Bob_aggregate",
        "accuracy": {
            "fp32": count("clear_correct") / n,
            "fixed_clear": count("fixed_clear_correct") / n,
            "secure": count("secure_correct") / n,
        },
        "top1_agreement": {
            "fixed_clear_vs_fp32": count("fixed_clear_matches_fp32") / n,
            "secure_vs_fixed_clear": count("secure_matches_fixed_clear") / n,
            "secure_vs_fp32": count("secure_matches_fp32") / n,
        },
        "latency_s": {
            "online": distribution(records, "online_latency_s"),
            "setup_offline": distribution(records, "setup_offline_s"),
            "total": distribution(records, "total_latency_s"),
        },
        "communication_bytes_alice_plus_bob": {
            "online": distribution(records, "online_communication_bytes_alice_plus_bob"),
            "total": distribution(records, "total_communication_bytes_alice_plus_bob"),
        },
        "rounds": distribution(records, "online_rounds_max_party"),
        "conversion_ciphertexts": distribution(records, "conversion_ciphertexts"),
        "he_rotations": distribution(records, "he_rotations"),
        "samples_csv": str(args.output.with_suffix(".csv").resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        columns = (
            "index", "image", "label", "clear_top1", "fixed_clear_top1",
            "secure_top1", "clear_correct", "fixed_clear_correct",
            "secure_correct", "fixed_clear_matches_fp32",
            "secure_matches_fixed_clear", "online_latency_s", "setup_offline_s",
            "total_latency_s", "online_communication_bytes_alice_plus_bob",
            "total_communication_bytes_alice_plus_bob", "online_rounds_max_party",
            "conversion_ciphertexts", "he_rotations")
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: row["index"]))
    print(f"OURS_SECURE_ACCURACY_SUMMARY status=PASS samples={n} "
          f"secure={summary['accuracy']['secure']:.6f} "
          f"agreement={summary['top1_agreement']['secure_vs_fixed_clear']:.6f}")


if __name__ == "__main__":
    main()
