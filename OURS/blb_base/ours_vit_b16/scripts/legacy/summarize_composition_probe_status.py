#!/usr/bin/env python3
"""Summarize real composition-probe attempts without inventing measurements."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
import re
from pathlib import Path

from offline_cost_aware_head_allocator import sha256_file


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key_values(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def composition_name(row: dict[str, str]) -> str:
    return f"{int(row['S_heads'])}S{int(row['R_heads'])}{'R'}{int(row['L_heads'])}L"


def status_map(path: Path) -> dict[str, str]:
    return key_values(path)


def inspect_attempt(run_dir: Path) -> dict[str, str] | None:
    status_path = run_dir / "status.txt"
    if not status_path.is_file():
        return None
    status = status_map(status_path)
    selection = key_values(run_dir / "gpu_selection.txt")
    logs = "\n".join(
        (run_dir / party).read_text(encoding="utf-8", errors="replace")
        for party in ("alice.log", "bob.log")
        if (run_dir / party).is_file())
    blocked = re.findall(r"OURS_VIT_RUNTIME status=BLOCKED reason=([^\n]+)", logs)
    if status.get("alice_rc") == "0" and status.get("bob_rc") == "0":
        attempt_status = "COMPLETED_REAL_RUN"
    else:
        attempt_status = "FAILED_REAL_RUN"
    return {
        "run_dir": str(run_dir.resolve()),
        "status": attempt_status,
        "alice_rc": status.get("alice_rc", ""),
        "bob_rc": status.get("bob_rc", ""),
        "failure_reason": blocked[-1] if blocked else "",
        "image_id": selection.get("image_id", ""),
        "binary_sha256": selection.get("binary_sha256", ""),
        "attention_packing": selection.get("attention_packing", ""),
        "heterogeneous_head_grouping": "0" if selection.get(
            "attention_packing") == "unpacked" else "",
        "gpu_direct_plaintext": selection.get("gpu_direct_plaintext", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--measured-costs", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_csv(args.probe_manifest)
    measured = read_csv(args.measured_costs) if args.measured_costs.is_file() else []
    measured_plan_hashes = {row.get("plan_sha256", "") for row in measured
                            if row.get("plan_sha256", "")}
    rows: list[dict[str, str]] = []
    for probe in manifest:
        name = composition_name(probe)
        plan = Path(probe["plan"])
        plan_hash = sha256_file(plan)
        attempts = []
        for root in args.run_root:
            attempt = inspect_attempt(root / f"{probe['split']}-{name.lower()}")
            if attempt is not None:
                attempts.append(attempt)
        completed = [item for item in attempts
                     if item["status"] == "COMPLETED_REAL_RUN"]
        failed = [item for item in attempts
                  if item["status"] == "FAILED_REAL_RUN"]
        if plan_hash in measured_plan_hashes:
            probe_status = "MEASURED_REAL"
        elif failed:
            probe_status = "FAILED_REAL_PROBE"
        elif completed:
            probe_status = "COMPLETED_NOT_COLLECTED"
        else:
            probe_status = ("NOT_MEASURED_NO_REAL_TRAIN"
                            if probe["split"] == "train"
                            else "NOT_MEASURED_NO_REAL_HELDOUT")
        rows.append({
            "composition": name,
            "split": probe["split"],
            "ablation": probe["ablation"],
            "plan": str(plan.resolve()),
            "plan_sha256": plan_hash,
            "status": probe_status,
            "measured": "1" if plan_hash in measured_plan_hashes else "0",
            "measured_layer_rows": str(sum(
                row.get("plan_sha256", "") == plan_hash for row in measured)),
            "attempt_count": str(len(attempts)),
            "completed_attempts": str(len(completed)),
            "failed_attempts": str(len(failed)),
            "failure_reasons": ";".join(sorted({
                item["failure_reason"] for item in failed
                if item["failure_reason"]})),
            "run_dirs": ";".join(item["run_dir"] for item in attempts),
            "image_ids": ";".join(sorted({item["image_id"] for item in attempts
                                             if item["image_id"]})),
            "binary_sha256s": ";".join(sorted({item["binary_sha256"]
                                                for item in attempts
                                                if item["binary_sha256"]})),
            "attention_packing": ";".join(sorted({
                item["attention_packing"] for item in attempts
                if item["attention_packing"]})),
            "heterogeneous_head_grouping": ";".join(sorted({
                item["heterogeneous_head_grouping"] for item in attempts
                if item["heterogeneous_head_grouping"]})),
            "gpu_direct_plaintext": ";".join(sorted({
                item["gpu_direct_plaintext"] for item in attempts
                if item["gpu_direct_plaintext"]})),
        })

    fields = list(rows[0]) if rows else [
        "composition", "split", "ablation", "plan", "plan_sha256", "status",
        "measured", "measured_layer_rows", "attempt_count", "completed_attempts",
        "failed_attempts", "failure_reasons", "run_dirs", "image_ids",
        "binary_sha256s", "attention_packing", "heterogeneous_head_grouping",
        "gpu_direct_plaintext",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
