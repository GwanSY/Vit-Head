#!/usr/bin/env python3
"""Build an honest multi-seed ViT validation report.

The formal BOLT executable reveals only the final class.  Consequently this
script records secure top-1 consistency and fixed-clear errors, while keeping
secure intermediate error and first-inconsistent-layer fields explicitly
unavailable.  It never asks the executable to reveal logits or intermediate
shares.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def max_fixed_clear(report: dict[str, Any]) -> float:
    values = [
        row["clear_fixed"]["max_abs"]
        for row in report["operators"]
        if row["clear_fixed"].get("status") == "PASS"
    ]
    return max(values, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, default=Path("fixtures"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        default=Path("scripts/vit_reference.py"))
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with args.summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = row["seed"]
            fixture = args.fixture_root / f"vit_seed_{seed}"
            secure_output = args.output_root / f"vit_seed_{seed}" / "prediction.txt"
            report_path = args.report_dir / f"seed_{seed}.json"
            subprocess.run([
                sys.executable, str(args.reference),
                "--model", str(fixture / "alice_model.txt"),
                "--input", str(fixture / "bob_image.txt"),
                "--secure-prediction", str(secure_output),
                "--output", str(report_path),
            ], check=True, stdout=subprocess.DEVNULL)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            secure_top1 = report["secure"]["top1"]
            fixed_top1 = report["fixed"]["top1"]
            rows.append({
                "seed": int(seed),
                "clear_top1": report["clear"]["top1"],
                "fixed_top1": fixed_top1,
                "secure_top1": secure_top1,
                "secure_fixed_match": (secure_top1 == fixed_top1),
                "fixed_clear_max_abs": max_fixed_clear(report),
                "padding_invariance": report["padding_invariance"]["status"],
                "first_inconsistent_layer":
                    "NOT_AVAILABLE_NO_SECURE_INTERMEDIATE_OBSERVER",
            })

    matched = sum(int(row["secure_fixed_match"]) for row in rows)
    max_error = max((row["fixed_clear_max_abs"] for row in rows), default=0.0)
    diagnostics = {
        "format": "BOLT_VIT_SEED_DIAGNOSTICS_V1",
        "cases": len(rows),
        "secure_fixed_top1_matches": matched,
        "secure_fixed_top1_consistency_rate": (matched / len(rows)
                                                if rows else None),
        "max_abs_error_clear_fixed": max_error,
        "max_abs_error_secure": None,
        "first_inconsistent_layer":
            "NOT_AVAILABLE_NO_SECURE_INTERMEDIATE_OBSERVER",
        "secure_intermediate_reconstruction": False,
        "note": "formal BOLT_VIT reveals only class index; no secure intermediate was decrypted",
        "rows": rows,
    }
    args.report_dir.joinpath("seed_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    with args.report_dir.joinpath("seed_diagnostics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = tuple(rows[0].keys()) if rows else (
            "seed", "clear_top1", "fixed_top1", "secure_top1",
            "secure_fixed_match", "fixed_clear_max_abs",
            "padding_invariance", "first_inconsistent_layer")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"VIT_SEED_CASES={len(rows)}")
    print(f"VIT_SECURE_FIXED_TOP1_MATCHES={matched}/{len(rows)}")
    print(f"VIT_SECURE_FIXED_TOP1_CONSISTENCY_RATE={matched / len(rows):.8f}"
          if rows else "VIT_SECURE_FIXED_TOP1_CONSISTENCY_RATE=NOT_RUN")
    print(f"VIT_MAX_ABS_ERROR_CLEAR_FIXED={max_error:.12g}")
    print("VIT_MAX_ABS_ERROR_SECURE=NOT_AVAILABLE")
    print("VIT_FIRST_INCONSISTENT_LAYER=NOT_AVAILABLE_NO_SECURE_INTERMEDIATE_OBSERVER")
    print(f"VIT_SEED_DIAGNOSTICS={args.report_dir / 'seed_diagnostics.json'}")


if __name__ == "__main__":
    main()
