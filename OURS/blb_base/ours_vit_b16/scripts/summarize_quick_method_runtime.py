#!/usr/bin/env python3
"""Summarize same-cohort single-image block-0 runtime experiments.

Latency is the slower party's measured value.  Communication is the sum of
both parties' bytes, while rounds and conversion ciphertexts are graph-level
counts (the maximum of the two party reports).  Accuracy is intentionally
left as NA: this cohort is a one-image/block-0 runtime diagnostic, not the
50+ secure-accuracy evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def read_metrics(path: Path):
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["metric"]] = float(row["value"])
    return result


def read_key_values(path: Path):
    result = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def read_head_counts(path: Path):
    counts = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["layer"]) == 0:
                counts[row["protocol"]] += 1
    expected = ("softmax_mpc", "relu_norm_mpc", "he_linear")
    if sum(counts.values()) != 12:
        raise ValueError(f"layer-0 head plan does not contain 12 heads: {path}")
    for protocol in expected:
        counts.setdefault(protocol, 0)
    return counts


def read_block_profile(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty block profile: {path}")
    return rows[0]


def metric_int(metrics, key):
    return int(round(metrics[key]))


def metric_sum(party1, party2, key):
    return party1[key] + party2[key]


def parse_run(spec: str):
    if "=" not in spec:
        raise ValueError(f"run must be NAME=PATH, got {spec!r}")
    name, raw_path = spec.split("=", 1)
    if not name or not raw_path:
        raise ValueError(f"run must be NAME=PATH, got {spec!r}")
    return name, Path(raw_path)


def build_row(name: str, run_dir: Path):
    status = read_key_values(run_dir / "status.txt")
    if status.get("alice_rc") != "0" or status.get("bob_rc") != "0":
        raise ValueError(f"run is not successful: {run_dir}")
    env = read_key_values(run_dir / "environment.txt")
    selection = read_key_values(run_dir / "gpu_selection.txt")
    p1 = read_metrics(run_dir / "execution_totals_party1.csv")
    p2 = read_metrics(run_dir / "execution_totals_party2.csv")
    counts = read_head_counts(run_dir / "head_plan_party1.csv")
    profile = read_block_profile(run_dir / "block_profile_party1.csv")

    image_id = selection.get("image_id", "")
    if not image_id.startswith("sha256:"):
        raise ValueError(f"missing immutable image id: {run_dir}")
    row = {
        "method": name,
        "run_dir": str(run_dir.resolve()),
        "status": "PASS",
        "backend": env.get("backend", ""),
        "image_id": image_id,
        "binary_sha256": env.get("binary_sha256", ""),
        "model_sha256": env.get("model_sha256", ""),
        "input_sha256": env.get("input_sha256", ""),
        "network": env.get("network", ""),
        "gpu_topology": selection.get("party_topology",
                                       "separate_physical_gpus"),
        "physical_gpus": selection.get("physical_gpus", ""),
        "packing": selection.get("attention_packing", "packed"),
        "ablation": env.get("ablation", ""),
        "scope": "single_image_block0",
        "head_softmax": counts["softmax_mpc"],
        "head_relu": counts["relu_norm_mpc"],
        "head_linear": counts["he_linear"],
        "head_composition": (
            f"{counts['softmax_mpc']}S+{counts['relu_norm_mpc']}R+"
            f"{counts['he_linear']}L"
        ),
        "online_latency_s": max(p1["secure_end_to_end_s"],
                                p2["secure_end_to_end_s"]),
        "total_latency_s": max(p1["setup_plus_secure_s"],
                                p2["setup_plus_secure_s"]),
        "setup_offline_s_party1": p1["setup_offline_s"],
        "setup_offline_s_party2": p2["setup_offline_s"],
        "online_communication_bytes": metric_sum(
            p1, p2, "online_communication_bytes"),
        "total_communication_bytes": metric_sum(
            p1, p2, "total_communication_bytes"),
        "online_communication_rounds": max(
            metric_int(p1, "online_communication_rounds"),
            metric_int(p2, "online_communication_rounds")),
        "ckks_to_mpc_conversions": max(
            metric_int(p1, "ckks_to_mpc"), metric_int(p2, "ckks_to_mpc")),
        "mpc_to_ckks_conversions": max(
            metric_int(p1, "mpc_to_ckks"), metric_int(p2, "mpc_to_ckks")),
        "conversion_ciphertexts": max(
            metric_int(p1, "conversion_ciphertexts"),
            metric_int(p2, "conversion_ciphertexts")),
        "he_rotations": max(metric_int(p1, "he_rotations"),
                            metric_int(p2, "he_rotations")),
        "he_ct_pt_multiply": max(metric_int(p1, "he_ct_pt_multiply"),
                                  metric_int(p2, "he_ct_pt_multiply")),
        "he_ct_ct_multiply": max(metric_int(p1, "he_ct_ct_multiply"),
                                  metric_int(p2, "he_ct_ct_multiply")),
        "block_profile_online_ms": float(profile["block_online_time_ms"]),
        "block_profile_comm_bytes_party1": int(
            round(float(profile["total_communication_bytes"]))),
        "block_profile_rounds_party1": int(
            round(float(profile["communication_rounds"]))),
        "accuracy_50plus": "NA",
        "accuracy_drop": "NA",
        "top1_agreement": "NA",
    }
    return row


def validate_cohort(rows):
    if not rows:
        raise ValueError("no runs supplied")
    keys = ("image_id", "binary_sha256", "model_sha256", "input_sha256",
            "network", "gpu_topology", "packing", "physical_gpus")
    for key in keys:
        values = {row[key] for row in rows}
        if len(values) != 1:
            raise ValueError(f"cohort mismatch in {key}: {sorted(values)}")


def display_value(key, value):
    if key.endswith("_bytes"):
        return f"{int(value):,}"
    if key.endswith("_s") or key.endswith("_ms"):
        return f"{float(value):.3f}"
    return str(value)


def write_csv(path: Path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows):
    columns = [
        ("method", "method"),
        ("head_composition", "heads"),
        ("online_latency_s", "online s"),
        ("total_latency_s", "total s"),
        ("online_communication_bytes", "online comm (both)"),
        ("total_communication_bytes", "total comm (both)"),
        ("online_communication_rounds", "rounds"),
        ("conversion_ciphertexts", "conversion cts"),
        ("he_rotations", "HE rotations"),
        ("accuracy_50plus", "accuracy 50+"),
        ("accuracy_drop", "accuracy drop"),
        ("top1_agreement", "top-1 agreement"),
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Quick P1 method runtime cohort\n\n")
        handle.write(
            "Scope: one fixed image, block 0 only, same input/model/image "
            "binary/GPU cohort, 1Gbps/0.3ms RTT, packed. Latency is the "
            "slower party; communication is the sum of both parties. "
            "Accuracy fields are NA by design for this runtime-only cohort.\n\n"
        )
        handle.write("| " + " | ".join(label for _, label in columns) + " |\n")
        handle.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in rows:
            cells = [display_value(key, row[key]) for key, _ in columns]
            handle.write("| " + " | ".join(cells) + " |\n")
        handle.write("\n## Provenance\n\n")
        handle.write(f"- image_id: `{rows[0]['image_id']}`\n")
        handle.write(f"- binary_sha256: `{rows[0]['binary_sha256']}`\n")
        handle.write(f"- model_sha256: `{rows[0]['model_sha256']}`\n")
        handle.write(f"- input_sha256: `{rows[0]['input_sha256']}`\n")
        handle.write(f"- physical_gpus: `{rows[0]['physical_gpus']}`\n")
        handle.write("- Cost-only is provisional: group-level path-cost ordering "
                     "plus original-head tie-break.\n")
        handle.write("- All-Softmax is a reference row; it is not the same-count "
                     "4S+4R+4L budget.\n")
        handle.write("\nRun directories:\n\n")
        for row in rows:
            handle.write(f"- {row['method']}: `{row['run_dir']}`\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="NAME=RUN_DIR; repeat once per method")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for spec in args.run:
        name, path = parse_run(spec)
        rows.append(build_row(name, path))
    validate_cohort(rows)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps({
        "format": "OURS_VIT_QUICK_METHOD_RUNTIME_V1",
        "cohort_validation": "PASS",
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, rows)
    print(f"OURS_QUICK_RUNTIME_SUMMARY status=PASS csv={csv_path}")
    for row in rows:
        print(
            f"OURS_QUICK_RUNTIME method={row['method']} "
            f"online_s={row['online_latency_s']:.6f} "
            f"total_s={row['total_latency_s']:.6f} "
            f"online_comm_bytes={row['online_communication_bytes']} "
            f"rounds={row['online_communication_rounds']} "
            f"conversion_ct={row['conversion_ciphertexts']}"
        )


if __name__ == "__main__":
    main()
