#!/usr/bin/env python3
"""Fail-closed provenance audit for a set of formal secure runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def fields(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("three-path", "p1", "packing",
                                            "ablation", "full12"), required=True)
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = sorted(path.parent for path in args.run_root.rglob("status.txt"))
    if len(runs) != args.expected_runs:
        raise ValueError(f"expected {args.expected_runs} runs, found {len(runs)}")
    records = []
    for run in runs:
        status = fields(run / "status.txt")
        if status.get("alice_rc") != "0" or status.get("bob_rc") != "0":
            raise ValueError(f"party failure: {run}")
        environment = fields(run / "environment.txt")
        selection = fields(run / "gpu_selection.txt")
        qdisc = (run / "qdisc_active.txt").read_text(
            encoding="utf-8", errors="replace")
        if environment.get("network") != "1Gbps/0.3ms_RTT":
            raise ValueError(f"unshaped run is not formal evidence: {run}")
        if "netem" not in qdisc or "rate 1Gbit" not in qdisc or \
                not re.search(r"delay ((149|150)us|0\.15ms)", qdisc):
            raise ValueError(f"qdisc does not prove 1 Gbps / 0.3 ms RTT: {run}")
        if (selection.get("party_topology") != "separate_physical_gpus" and
                selection.get("outer_selection") !=
                "dynamic_immediately_before_container_start"):
            raise ValueError(f"GPU topology is not two-party/two-GPU: {run}")
        plan = run / "head_plan_party1.csv"
        if not plan.is_file() or not (run / "head_plan_party2.csv").is_file():
            raise ValueError(f"head-plan audit is missing: {run}")
        if plan.read_bytes() != (run / "head_plan_party2.csv").read_bytes():
            raise ValueError(f"party head plans disagree: {run}")
        records.append({
            "run": str(run.resolve()),
            "binary_sha256": environment.get("binary_sha256"),
            "model_sha256": environment.get("model_sha256"),
            "input_sha256": environment.get("input_sha256"),
            "image_id": selection.get("image_id", selection.get("outer_image_id")),
            "plan_audit_sha256": sha256(plan),
            "physical_gpus": selection.get("physical_gpus"),
            "attention_packing": selection.get("attention_packing"),
            "network": environment.get("network"),
            "party_topology": "separate_physical_gpus",
        })
    for invariant in ("binary_sha256", "model_sha256", "input_sha256",
                      "image_id", "network"):
        values = {record[invariant] for record in records}
        if None in values or len(values) != 1:
            raise ValueError(f"cohort invariant {invariant} differs: {values}")
    if args.kind == "packing":
        for invariant in ("plan_audit_sha256", "physical_gpus"):
            values = {record[invariant] for record in records}
            if None in values or len(values) != 1:
                raise ValueError(
                    f"packing pair invariant {invariant} differs: {values}")
        packing = [record["attention_packing"] for record in records]
        if set(packing) != {"packed", "unpacked"} or \
                packing.count("packed") != packing.count("unpacked"):
            raise ValueError(
                "packing cohort must contain balanced Packed/Unpacked runs")
    if args.kind in {"three-path", "p1", "ablation", "full12"}:
        values = {record["physical_gpus"] for record in records}
        if None in values or len(values) != 1:
            raise ValueError(
                f"formal {args.kind} cohort did not use identical GPUs: {values}")
    report = {
        "format": "OURS_VIT_FORMAL_COHORT_AUDIT_V1",
        "status": "PASS",
        "kind": args.kind,
        "runs": records,
        "invariants": {name: records[0][name] for name in
                       ("binary_sha256", "model_sha256", "input_sha256",
                        "image_id", "network")},
        "communication_scope": "Alice_plus_Bob_aggregate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"OURS_PAPER_COHORT status=PASS kind={args.kind} runs={len(runs)}")


if __name__ == "__main__":
    main()
