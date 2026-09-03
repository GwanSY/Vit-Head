#!/usr/bin/env python3
"""Aggregate the quick method runtime rows, including Random mean/std."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


NUMERIC = {
    "online_latency_s": "online_latency",
    "total_latency_s": "total_latency",
    "online_communication_bytes": "online_communication",
    "total_communication_bytes": "total_communication",
    "online_communication_rounds": "rounds",
    "conversion_ciphertexts": "conversion_ciphertexts",
}


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else "NA"


def make_row(method, rows):
    result = {
        "method": method,
        "n": len(rows),
        "head_composition": rows[0]["head_composition"],
        "accuracy_50plus": "NA",
        "accuracy_drop": "NA",
        "top1_agreement": "NA",
    }
    if any(row["head_composition"] != result["head_composition"]
           for row in rows):
        raise ValueError(f"head composition differs within {method}")
    for source, target in NUMERIC.items():
        values = [float(row[source]) for row in rows]
        result[f"{target}_mean"] = statistics.mean(values)
        result[f"{target}_std"] = sample_std(values)
    return result


def pct_reduction(reference, value):
    if reference == 0:
        return "NA"
    return (reference - value) / reference * 100.0


def write_md(path: Path, rows, all_softmax):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Quick P1 method runtime aggregate\n\n")
        handle.write(
            "One fixed input, block 0, packed, same GPU/model/input/image "
            "binary and 1Gbps/0.3ms RTT. Values are mean ± sample std; "
            "n=1 rows have std=NA. Latency uses the slower party and "
            "communication sums both parties. Accuracy is NA because this "
            "is runtime-only.\n\n"
        )
        handle.write("| method | n | heads | online s mean±std | total s mean±std "
                     "| online comm mean±std | total comm mean±std | rounds "
                     "mean±std | conversion cts mean±std | online reduction vs AllS |\n")
        handle.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            def pair(prefix, integer=False):
                mean = row[f"{prefix}_mean"]
                std = row[f"{prefix}_std"]
                if integer:
                    mean_text = f"{mean:,.2f}"
                    std_text = "NA" if std == "NA" else f"{std:,.2f}"
                else:
                    mean_text = f"{mean:.3f}"
                    std_text = "NA" if std == "NA" else f"{std:.3f}"
                return f"{mean_text} ± {std_text}"
            reduction = pct_reduction(
                all_softmax["online_latency_mean"], row["online_latency_mean"])
            reduction_text = ("NA" if reduction == "NA"
                              else f"{reduction:.2f}%")
            handle.write(
                f"| {row['method']} | {row['n']} | {row['head_composition']} "
                f"| {pair('online_latency')} | {pair('total_latency')} "
                f"| {pair('online_communication', True)} "
                f"| {pair('total_communication', True)} "
                f"| {pair('rounds')} | {pair('conversion_ciphertexts')} "
                f"| {reduction_text} |\n"
            )
        handle.write("\nThe All-Softmax row is a reference with 12S; the other rows "
                     "are the same-count 4S+4R+4L cohort. Cost-only is provisional "
                     "because its assignment uses group-level cost ordering plus "
                     "an explicit original-head tie-break.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise ValueError("empty input")
    groups = {}
    for row in rows:
        method = "Random" if row["method"].startswith("Random-") else row["method"]
        groups.setdefault(method, []).append(row)
    if len(groups.get("Random", [])) != 5:
        raise ValueError("Random aggregate requires exactly five seeds")
    aggregate = [make_row(method, groups[method])
                 for method in ("All-Softmax", "Sensitivity-only", "Cost-only",
                                "Random", "Ours")]
    reference = next(row for row in aggregate if row["method"] == "All-Softmax")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps({"format": "OURS_VIT_QUICK_METHOD_AGGREGATE_V1",
                    "random_n": 5, "rows": aggregate},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(args.output_prefix.with_suffix(".md"), aggregate, reference)
    print(f"OURS_QUICK_RUNTIME_AGGREGATE status=PASS csv={args.output_prefix.with_suffix('.csv')}")
    for row in aggregate:
        std = row["online_latency_std"]
        std_text = "NA" if std == "NA" else f"{std:.6f}"
        print(f"OURS_QUICK_RUNTIME_AGG method={row['method']} n={row['n']} "
              f"online_mean_s={row['online_latency_mean']:.6f} "
              f"online_std_s={std_text} "
              f"total_mean_s={row['total_latency_mean']:.6f}")


if __name__ == "__main__":
    main()
