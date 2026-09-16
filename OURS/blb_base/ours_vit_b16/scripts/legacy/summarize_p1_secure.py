#!/usr/bin/env python3
"""Build the strict P1 method table from real full-12 secure cohorts."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
import json
import statistics
from pathlib import Path


def slug(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plans = json.loads(args.plan_manifest.read_text(encoding="utf-8"))
    summaries: dict[str, dict] = {}
    for name, info in plans["plans"].items():
        path = args.run_root / f"{args.tag_prefix}-{slug(name)}" / \
            "secure_accuracy.json"
        if not path.is_file():
            raise ValueError(f"missing secure cohort for {name}: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("samples", 0) < 50 or row.get("status") != "PASS":
            raise ValueError(f"{name} is not a real secure >=50-image cohort")
        if row["head_plan_sha256"] != info["sha256"]:
            raise ValueError(f"{name} secure cohort uses a different plan")
        summaries[name] = row

    invariant_keys = ("manifest_file_sha256", "binary_sha256", "model_sha256",
                      "image_id", "network", "communication_scope")
    for key in invariant_keys:
        values = {json.dumps(row[key], sort_keys=True) for row in summaries.values()}
        if len(values) != 1:
            raise ValueError(f"P1 cohort invariant differs for {key}: {values}")
    hardware = {json.dumps(row["hardware"], sort_keys=True)
                for row in summaries.values()}
    if len(hardware) != 1:
        raise ValueError("P1 methods did not use identical physical hardware")

    ours_counts = plans["plans"]["Ours"]["global_counts"]
    budgeted = [name for name in plans["plans"] if name != "All-Softmax"]
    if any(plans["plans"][name]["global_counts"] != ours_counts
           for name in budgeted):
        raise ValueError("Random/Sensitivity/Cost/Ours path counts differ")

    baseline = summaries["All-Softmax"]["accuracy"]["secure"]
    def record(name: str, row: dict) -> dict:
        return {
            "method": name,
            "head_counts": plans["plans"][name]["global_counts"],
            "samples": row["samples"],
            "accuracy_fp32": row["accuracy"]["fp32"],
            "accuracy_fixed_clear": row["accuracy"]["fixed_clear"],
            "accuracy_secure": row["accuracy"]["secure"],
            "secure_accuracy_drop_vs_all_softmax":
                baseline - row["accuracy"]["secure"],
            "online_latency_s_mean": row["latency_s"]["online"]["mean"],
            "online_latency_s_std": row["latency_s"]["online"]["stddev_sample"],
            "online_latency_s_p95": row["latency_s"]["online"]["p95"],
            "total_latency_s_mean": row["latency_s"]["total"]["mean"],
            "total_latency_s_std":
                row["latency_s"]["total"]["stddev_sample"],
            "total_latency_s_p95": row["latency_s"]["total"]["p95"],
            "communication_online_bytes_mean":
                row["communication_bytes_alice_plus_bob"]["online"]["mean"],
            "communication_total_bytes_mean":
                row["communication_bytes_alice_plus_bob"]["total"]["mean"],
            "rounds_mean": row["rounds"]["mean"],
            "conversion_ciphertexts_mean":
                row["conversion_ciphertexts"]["mean"],
            "he_rotations_mean": row["he_rotations"]["mean"],
            "top1_agreement_secure_fixed":
                row["top1_agreement"]["secure_vs_fixed_clear"],
            "top1_agreement_secure_fp32":
                row["top1_agreement"]["secure_vs_fp32"],
        }

    individual = [record(name, summaries[name]) for name in summaries]
    random_names = sorted(name for name in summaries if name.startswith("Random-"))
    if len(random_names) < 5:
        raise ValueError("formal Random baseline requires at least five seeds")
    random_rows = [next(row for row in individual if row["method"] == name)
                   for name in random_names]
    numeric = [key for key, value in random_rows[0].items()
               if isinstance(value, (int, float)) and key != "samples"]
    random_aggregate = {
        "method": "Random",
        "seeds": plans["random_seeds"],
        "seed_count": len(random_rows),
        "head_counts": ours_counts,
    }
    for key in numeric:
        values = [float(row[key]) for row in random_rows]
        random_aggregate[key] = {
            "mean_across_seeds": statistics.fmean(values),
            "std_across_seeds": sample_std(values),
        }

    result = {
        "format": "OURS_VIT_P1_SECURE_COMPARISON_V1",
        "status": "PASS",
        "evidence": "real_secure_full12",
        "all_softmax_role": "accuracy_and_cost_reference_outside_same_count_budget",
        "budgeted_methods": ["Random", "Sensitivity-only", "Cost-only", "Ours"],
        "same_count_budget": ours_counts,
        "samples_per_plan": summaries["All-Softmax"]["samples"],
        "manifest_sha256": summaries["All-Softmax"]["manifest_file_sha256"],
        "binary_sha256": summaries["All-Softmax"]["binary_sha256"],
        "model_sha256": summaries["All-Softmax"]["model_sha256"],
        "hardware": summaries["All-Softmax"]["hardware"],
        "network": "1Gbps/0.3ms_RTT_verified",
        "communication_scope": "Alice_plus_Bob_aggregate",
        "individual_plans": individual,
        "random_aggregate": random_aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    columns = [key for key in individual[0] if key != "head_counts"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(individual)
    print(f"OURS_P1_SECURE_COMPARISON status=PASS plans={len(individual)}")
    print(f"OURS_P1_SECURE_COMPARISON_JSON={args.output}")


if __name__ == "__main__":
    main()
