#!/usr/bin/env python3
"""Record one full-12 secure accuracy sample without promoting clear results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def metrics(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: float(row["value"])
                for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--label", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--head-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = args.run_dir.resolve()
    status = fields(run / "status.txt")
    if status.get("alice_rc") != "0" or status.get("bob_rc") != "0":
        raise ValueError("secure parties did not both exit zero")
    prediction_files = [run / f"prediction_party{party}.txt" for party in (1, 2)]
    logits_files = [run / f"secure_logits_party{party}.txt" for party in (1, 2)]
    if prediction_files[0].read_bytes() != prediction_files[1].read_bytes() or \
            logits_files[0].read_bytes() != logits_files[1].read_bytes():
        raise ValueError("secure parties disagree on prediction or logits")

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    clear_top1 = int(reference["clear"]["top1"])
    fixed_top1 = int(reference["fixed"]["top1"])
    secure_top1 = int(reference["secure"]["top1"])
    totals = [metrics(run / f"execution_totals_party{party}.csv")
              for party in (1, 2)]
    environment = fields(run / "environment.txt")
    selection = fields(run / "gpu_selection.txt")
    qdisc = (run / "qdisc_active.txt").read_text(
        encoding="utf-8", errors="replace")
    if ("netem" not in qdisc or "rate 1Gbit" not in qdisc or
            not re.search(r"delay ((149|150)us|0\.15ms)", qdisc)):
        raise ValueError("sample does not prove active 1 Gbps / 0.3 ms RTT shaping")
    gpu_rows = []
    with (run / "gpu_telemetry_before_run.txt").open(
            newline="", encoding="utf-8") as handle:
        gpu_rows = list(csv.reader(handle))
    selected = [part.strip() for part in selection["physical_gpus"].split(",")]
    selected_rows = [row for row in gpu_rows if row and row[0].strip() in selected]
    if len(selected_rows) != 2:
        raise ValueError("sample GPU telemetry does not cover both parties")
    record = {
        "format": "OURS_VIT_SECURE_ACCURACY_SAMPLE_V1",
        "status": "PASS",
        "index": args.index,
        "image": args.image,
        "label": args.label,
        "manifest": str(args.manifest.resolve()),
        "manifest_file_sha256": sha256(args.manifest),
        "head_plan": str(args.head_plan.resolve()),
        "head_plan_sha256": sha256(args.head_plan),
        "binary_sha256": environment["binary_sha256"],
        "model_sha256": environment["model_sha256"],
        "input_sha256": environment["input_sha256"],
        "image_id": selection.get("image_id", selection.get("outer_image_id")),
        "physical_gpus": selection["physical_gpus"],
        "gpu_models": ";".join(row[1].strip() for row in selected_rows),
        "gpu_uuids": ";".join(row[2].strip() for row in selected_rows),
        "gpu_driver_versions": ";".join(row[3].strip() for row in selected_rows),
        "network": environment["network"],
        "network_shaping_verified": True,
        "communication_scope": "Alice_plus_Bob_aggregate",
        "clear_top1": clear_top1,
        "fixed_clear_top1": fixed_top1,
        "secure_top1": secure_top1,
        "clear_correct": clear_top1 == args.label,
        "fixed_clear_correct": fixed_top1 == args.label,
        "secure_correct": secure_top1 == args.label,
        "fixed_clear_matches_fp32": fixed_top1 == clear_top1,
        "secure_matches_fixed_clear": secure_top1 == fixed_top1,
        "secure_matches_fp32": secure_top1 == clear_top1,
        "online_latency_s": max(row["secure_end_to_end_s"] for row in totals),
        "setup_offline_s": max(row["setup_offline_s"] for row in totals),
        "total_latency_s": max(row["setup_plus_secure_s"] for row in totals),
        "online_communication_bytes_alice_plus_bob": sum(
            int(row["online_communication_bytes"]) for row in totals),
        "total_communication_bytes_alice_plus_bob": sum(
            int(row["total_communication_bytes"]) for row in totals),
        "online_rounds_max_party": int(max(
            row["online_communication_rounds"] for row in totals)),
        "conversion_ciphertexts": int(max(
            row["conversion_ciphertexts"] for row in totals)),
        "he_rotations": int(max(row["he_rotations"] for row in totals)),
        "mpc_call_scope": "reported_separately_by_assignment_markers",
    }
    if record["network"] != "1Gbps/0.3ms_RTT" or not record["image_id"]:
        raise ValueError("sample lacks formal network/image provenance")
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("OURS_SECURE_ACCURACY_SAMPLE "
          f"index={args.index} secure={secure_top1} fixed={fixed_top1} "
          f"label={args.label} agreement={int(secure_top1 == fixed_top1)} status=PASS")


if __name__ == "__main__":
    main()
