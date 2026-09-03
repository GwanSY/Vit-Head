#!/usr/bin/env python3
"""Freeze a public Head assignment; FINAL mode refuses incomplete evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fields(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def parse_plan(path: Path) -> list[dict[str, object]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if lines[:2] != ["OURS_VIT_HEAD_PLAN_V1",
                     "layers=12,heads=12,logical_tokens=197,physical_tokens=256"]:
        raise ValueError("head plan does not have the formal ViT-B/16 header")
    rows = []
    for row in csv.DictReader(lines[2:]):
        rows.append({
            "layer": int(row["layer"]),
            "physical_head": int(row["physical_head"]),
            "original_head": int(row["original_head"]),
            "path": row["protocol"],
            "alpha": float(row["alpha"]),
        })
    expected = {(layer, head) for layer in range(12) for head in range(12)}
    actual = {(int(row["layer"]), int(row["original_head"])) for row in rows}
    if len(rows) != 144 or actual != expected:
        raise ValueError("head plan must bind every original Head exactly once")
    for layer in range(12):
        physical = {int(row["physical_head"]) for row in rows
                    if row["layer"] == layer}
        if physical != set(range(12)):
            raise ValueError(f"layer {layer} physical reorder is not a permutation")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--status", choices=("provisional", "final"),
                        required=True)
    parser.add_argument("--a0-report", type=Path)
    parser.add_argument("--a0-run-dir", type=Path)
    parser.add_argument("--secure-accuracy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.plan, args.model, args.binary):
        if not path.is_file():
            raise ValueError(f"required artifact is missing: {path}")
    rows = parse_plan(args.plan)
    plan_hash = sha256(args.plan)
    binary_hash = sha256(args.binary)
    evidence: dict[str, object] = {}

    if args.status == "final":
        if not args.a0_report or not args.a0_run_dir or not args.secure_accuracy:
            raise ValueError(
                "FINAL requires A0 report/run provenance and secure accuracy")
        a0 = json.loads(args.a0_report.read_text(encoding="utf-8"))
        secure = json.loads(args.secure_accuracy.read_text(encoding="utf-8"))
        environment = fields(args.a0_run_dir / "environment.txt")
        if (a0.get("complete_a0_gate") != "PASS" or
                a0.get("fixed_secure_gate") != "PASS" or
                a0.get("fp32_fixed_top1_agreement") is not True or
                a0.get("secure_fixed_top1_agreement") is not True):
            raise ValueError("FINAL refused: complete A0/top-1 gate is not PASS")
        if (secure.get("status") != "PASS" or
                int(secure.get("samples", 0)) < 50):
            raise ValueError("FINAL refused: real secure accuracy has <50 samples")
        if secure.get("head_plan_sha256") != plan_hash:
            raise ValueError("FINAL refused: secure accuracy uses another plan")
        if (secure.get("binary_sha256") != binary_hash or
                environment.get("binary_sha256") != binary_hash):
            raise ValueError("FINAL refused: A0/accuracy binary hashes differ")
        if secure.get("model_sha256") != sha256(args.model):
            raise ValueError("FINAL refused: secure accuracy model hash differs")
        evidence = {
            "a0_report": str(args.a0_report.resolve()),
            "a0_report_sha256": sha256(args.a0_report),
            "a0_fixed_secure_gate": "PASS",
            "fp32_fixed_top1_agreement": True,
            "secure_accuracy_report": str(args.secure_accuracy.resolve()),
            "secure_accuracy_report_sha256": sha256(args.secure_accuracy),
            "secure_samples": secure["samples"],
            "secure_accuracy": secure["accuracy"]["secure"],
            "manifest_sha256": secure["manifest_file_sha256"],
        }
    else:
        evidence = {
            "paper_eligible": False,
            "reason": "A0 and >=50-image real secure gates not yet bound",
        }

    counts = Counter(str(row["path"]) for row in rows)
    layers = []
    for layer in range(12):
        layer_rows = sorted((row for row in rows if row["layer"] == layer),
                            key=lambda row: int(row["physical_head"]))
        layers.append({"layer": layer, "heads_in_physical_execution_order":
                       layer_rows})
    output = {
        "format": "OURS_VIT_HEAD_ASSIGNMENT_V1",
        "status": args.status.upper(),
        "paper_eligible": args.status == "final",
        "model": "ViT-B/16",
        "source_plan": str(args.plan.resolve()),
        "source_plan_sha256": plan_hash,
        "model_artifact": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "binary_artifact": str(args.binary.resolve()),
        "binary_sha256": binary_hash,
        "path_counts": dict(sorted(counts.items())),
        "layers": layers,
        "evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n",
                           encoding="utf-8")
    print(f"OURS_HEAD_ASSIGNMENT status={output['status']} "
          f"plan_sha256={plan_hash} binary_sha256={binary_hash}")
    print(f"OURS_HEAD_ASSIGNMENT_JSON={args.output}")


if __name__ == "__main__":
    main()
