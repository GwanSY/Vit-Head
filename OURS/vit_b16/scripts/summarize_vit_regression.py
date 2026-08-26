#!/usr/bin/env python3
"""Build an honest multi-seed ViT validation report.

The production BOLT executable reveals only the final class.  A validation
build may additionally reveal final logits, which this script compares with
the fixed reference when present.  Neither mode reveals secure intermediate
tensors, so a first-inconsistent-layer result is always explicitly
unavailable.
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


def secure_fixed_logits_max_abs(report: dict[str, Any]) -> float | None:
    """Return final-logit fixed/secure error, never an intermediate error."""
    comparison = report.get("secure", {}).get("fixed_logits_error")
    if (not isinstance(comparison, dict) or
            comparison.get("status") != "PASS"):
        return None
    value = comparison.get("max_abs")
    return float(value) if value is not None else None


def case_path(root: Path, prefix: str, seed: str) -> Path:
    return root / f"{prefix}{seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, default=Path("fixtures"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).resolve().with_name(
                            "vit_reference.py"))
    parser.add_argument("--fixture-prefix", default="seed_")
    parser.add_argument("--output-prefix", default="seed_")
    logits_mode = parser.add_mutually_exclusive_group()
    logits_mode.add_argument(
        "--require-secure-logits", action="store_true",
        help="fail unless each validation run produced final secure logits")
    logits_mode.add_argument(
        "--ignore-secure-logits", action="store_true",
        help="do not consume validation logits (production/final-class mode)")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with args.summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = row["seed"]
            fixture = case_path(args.fixture_root, args.fixture_prefix, seed)
            run_output = case_path(args.output_root, args.output_prefix, seed)
            secure_output = run_output / "prediction.txt"
            secure_logits = run_output / "secure_logits.txt"
            report_path = args.report_dir / f"seed_{seed}.json"
            reference_args = [
                sys.executable, str(args.reference),
                "--model", str(fixture / "alice_model.txt"),
                "--input", str(fixture / "bob_image.txt"),
                "--secure-prediction", str(secure_output),
                "--output", str(report_path),
            ]
            if (not args.ignore_secure_logits and secure_logits.is_file() and
                    secure_logits.stat().st_size > 0):
                reference_args.extend(["--secure-logits", str(secure_logits)])
            elif args.require_secure_logits:
                raise FileNotFoundError(
                    f"seed {seed} has no validation logits: {secure_logits}")
            subprocess.run(reference_args, check=True,
                           stdout=subprocess.DEVNULL)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            secure_top1 = report["secure"]["top1"]
            fixed_top1 = report["fixed"]["top1"]
            logits_error = secure_fixed_logits_max_abs(report)
            top1_match = secure_top1 == fixed_top1
            padding = report["padding_invariance"]["status"]
            rows.append({
                "seed": int(seed),
                "status": "PASS" if top1_match and padding == "PASS" else "FAIL",
                "clear_top1": report["clear"]["top1"],
                "fixed_top1": fixed_top1,
                "secure_top1": secure_top1,
                "secure_fixed_top1_match": top1_match,
                "fixed_clear_max_abs": max_fixed_clear(report),
                "secure_fixed_logits_max_abs": (
                    logits_error if logits_error is not None
                    else "NOT_AVAILABLE"),
                "secure_error_scope": (
                    "FINAL_LOGITS_ONLY" if logits_error is not None
                    else "NOT_AVAILABLE_FINAL_CLASS_ONLY"),
                "padding_invariance": padding,
                "first_inconsistent_layer":
                    "NOT_AVAILABLE_NO_SECURE_INTERMEDIATE_OBSERVER",
            })

    matched = sum(int(row["secure_fixed_top1_match"]) for row in rows)
    max_fixed_clear_error = max(
        (row["fixed_clear_max_abs"] for row in rows), default=0.0)
    secure_logits_errors = [
        row["secure_fixed_logits_max_abs"] for row in rows
        if row["secure_fixed_logits_max_abs"] != "NOT_AVAILABLE"
    ]
    max_secure_fixed_logits_error = (
        max(secure_logits_errors) if secure_logits_errors else None)
    diagnostics = {
        "format": "BOLT_VIT_SEED_DIAGNOSTICS_V1",
        "cases": len(rows),
        "secure_fixed_top1_matches": matched,
        "secure_fixed_top1_consistency_rate": (matched / len(rows)
                                                if rows else None),
        "max_abs_error_fixed_clear": max_fixed_clear_error,
        "max_abs_error_secure_fixed_logits": max_secure_fixed_logits_error,
        "secure_fixed_logits_cases": len(secure_logits_errors),
        "secure_error_scope": ("FINAL_LOGITS_ONLY"
                               if secure_logits_errors
                               else "NOT_AVAILABLE_FINAL_CLASS_ONLY"),
        "first_inconsistent_layer":
            "NOT_AVAILABLE_NO_SECURE_INTERMEDIATE_OBSERVER",
        "secure_intermediate_reconstruction": False,
        "note": ("validation logits, when present, are final outputs only; "
                 "no secure intermediate was decrypted"),
        "rows": rows,
    }
    args.report_dir.joinpath("seed_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    with args.report_dir.joinpath("seed_diagnostics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = tuple(rows[0].keys()) if rows else (
            "seed", "status", "clear_top1", "fixed_top1", "secure_top1",
            "secure_fixed_top1_match", "fixed_clear_max_abs",
            "secure_fixed_logits_max_abs", "secure_error_scope",
            "padding_invariance", "first_inconsistent_layer")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"VIT_SEED_CASES={len(rows)}")
    print(f"VIT_SECURE_FIXED_TOP1_MATCHES={matched}/{len(rows)}")
    print(f"VIT_SECURE_FIXED_TOP1_CONSISTENCY_RATE={matched / len(rows):.8f}"
          if rows else "VIT_SECURE_FIXED_TOP1_CONSISTENCY_RATE=NOT_RUN")
    print(f"VIT_MAX_ABS_ERROR_FIXED_CLEAR={max_fixed_clear_error:.12g}")
    if max_secure_fixed_logits_error is None:
        print("VIT_MAX_ABS_ERROR_SECURE_FIXED_LOGITS=NOT_AVAILABLE")
    else:
        print("VIT_MAX_ABS_ERROR_SECURE_FIXED_LOGITS="
              f"{max_secure_fixed_logits_error:.12g}")
    print("VIT_FIRST_INCONSISTENT_LAYER=NOT_AVAILABLE_NO_SECURE_INTERMEDIATE_OBSERVER")
    print(f"VIT_SEED_DIAGNOSTICS={args.report_dir / 'seed_diagnostics.json'}")


if __name__ == "__main__":
    main()
