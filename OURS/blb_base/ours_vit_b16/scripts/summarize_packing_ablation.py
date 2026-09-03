#!/usr/bin/env python3
"""Summarize timing-eligible paired Packed/Unpacked full-12 runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = (
    "secure_end_to_end_s",
    "setup_offline_s_max_party",
    "setup_plus_secure_s_max_party",
    "online_communication_bytes_alice_plus_bob",
    "total_communication_bytes_alice_plus_bob",
    "conversion_ciphertexts",
    "mpc_group_calls",
    "he_rotations",
)


def read_summary(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: row["value"] for row in csv.DictReader(handle)}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return (ordered[lower] * (upper - index) +
            ordered[upper] * (index - lower))


def stats(rows: list[dict[str, str]], metric: str) -> dict[str, float]:
    values = [float(row[metric]) for row in rows]
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cohorts: dict[str, list[dict[str, str]]] = {}
    for variant in ("packed", "unpacked"):
        paths = sorted((args.run_root / variant).glob("run-*/ours_summary.csv"))
        if not paths:
            raise ValueError(f"no timing-eligible {variant} summaries")
        cohorts[variant] = [read_summary(path) for path in paths]
        if any(row.get("attention_packing") != variant
               for row in cohorts[variant]):
            raise ValueError(f"{variant} cohort contains the wrong runtime mode")
    if len(cohorts["packed"]) != len(cohorts["unpacked"]):
        raise ValueError("Packed and Unpacked repeat counts differ")

    summaries = {
        variant: {metric: stats(rows, metric) for metric in METRICS}
        for variant, rows in cohorts.items()
    }
    reductions = {}
    for metric in METRICS:
        unpacked = summaries["unpacked"][metric]["mean"]
        packed = summaries["packed"][metric]["mean"]
        reductions[metric] = {
            "absolute": unpacked - packed,
            "percent": ((unpacked - packed) / unpacked * 100.0
                        if unpacked else 0.0),
        }
    result = {
        "format": "OURS_VIT_PACKING_ABLATION_SUMMARY_V1",
        "status": "PASS",
        "assignment_policy": "identical_across_variants",
        "network": "1Gbps/0.3ms_RTT_verified_by_cohort_audit",
        "communication_scope": "Alice_plus_Bob_aggregate",
        "latency_scope": {
            "online": "secure_end_to_end_s",
            "total": "setup_plus_secure_s_max_party",
        },
        "repeats_per_variant": len(cohorts["packed"]),
        "variants": summaries,
        "unpacked_minus_packed_reduction": reductions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print("OURS_PACKING_ABLATION_SUMMARY status=PASS "
          f"repeats={len(cohorts['packed'])}")
    print(f"OURS_PACKING_ABLATION_SUMMARY_JSON={args.output}")


if __name__ == "__main__":
    main()
