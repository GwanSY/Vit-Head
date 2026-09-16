#!/usr/bin/env python3
"""Summarize a same-binary/same-network full-12 paper suite."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from collect_attention_assignment_costs import measured_rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def totals(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: float(row["value"])
                for row in csv.DictReader(handle)}


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return (values[lower] * (upper - position) +
            values[upper] * (position - lower))


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("+", "_").replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.suite.open(newline="", encoding="utf-8") as handle:
        variants = list(csv.DictReader(handle))
    required = {"name", "ablation", "packing", "head_plan",
                "secure_accuracy_json"}
    if not variants or not required.issubset(variants[0]):
        raise ValueError(f"suite CSV requires columns: {sorted(required)}")

    result_rows = []
    cohort_binary = set()
    cohort_model = set()
    cohort_input = set()
    cohort_image = set()
    for variant in variants:
        run_dirs = sorted((args.run_root / slug(variant["name"])).glob("run-*"))
        if not run_dirs:
            raise ValueError(f"no runs for {variant['name']}")
        run_totals = []
        mpc_calls = []
        for run in run_dirs:
            status = fields(run / "status.txt")
            if status.get("alice_rc") != "0" or status.get("bob_rc") != "0":
                raise ValueError(f"party failure: {run}")
            metadata = [fields(run / f"run_metadata_party{party}.txt")
                        for party in (1, 2)]
            # Setup is performed independently by the two parties and its
            # elapsed time is expected to differ.  Compare every semantic
            # field, including prediction, while aggregating setup below.
            semantic_metadata = [
                {key: value for key, value in row.items()
                 if key != "setup_seconds"}
                for row in metadata
            ]
            if semantic_metadata[0] != semantic_metadata[1]:
                raise ValueError(f"party metadata disagree: {run}")
            if metadata[0].get("partial_run") != "0":
                raise ValueError(f"not a full-12 run: {run}")
            if metadata[0].get("ablation", "").lower() != variant["ablation"].lower():
                raise ValueError(f"ablation mismatch: {run}")
            environment = fields(run / "environment.txt")
            selection = fields(run / "gpu_selection.txt")
            # The restored historical binary omits attention_packing from its
            # party metadata; the launcher selection file is the run-level
            # source of truth for this setting.
            attention_packing = selection.get("attention_packing")
            if attention_packing not in {"packed", "unpacked"}:
                raise ValueError(f"run provenance lacks attention_packing: {run}")
            cohort_binary.add(environment.get("binary_sha256"))
            cohort_model.add(environment.get("model_sha256"))
            cohort_input.add(environment.get("input_sha256"))
            cohort_image.add(selection.get("image_id"))
            parties = [totals(run / f"execution_totals_party{party}.csv")
                       for party in (1, 2)]
            run_totals.append({
                "online_s": max(row["secure_end_to_end_s"] for row in parties),
                "setup_s": max(row["setup_offline_s"] for row in parties),
                "total_s": max(row["setup_plus_secure_s"] for row in parties),
                "online_comm": sum(row["online_communication_bytes"] for row in parties),
                "total_comm": sum(row["total_communication_bytes"] for row in parties),
                "rounds": max(row["online_communication_rounds"] for row in parties),
                "conversion_ciphertexts": max(row["conversion_ciphertexts"] for row in parties),
                "he_rotations": max(row["he_rotations"] for row in parties),
            })
            if variant["ablation"].lower() in {"a3", "a4"}:
                direct = measured_rows(run)
                if len(direct) != 12:
                    raise ValueError(f"missing assignment markers: {run}")
                mpc_calls.append(sum(int(row["mpc_group_calls"])
                                     for row in direct))
        metric_names = run_totals[0].keys()
        row = {
            "name": variant["name"],
            "ablation": variant["ablation"].upper(),
            "packing": variant["packing"],
            "head_plan": str(Path(variant["head_plan"]).resolve()),
            "head_plan_sha256": sha256(Path(variant["head_plan"])),
            "repeats": len(run_dirs),
            "metrics": {name: distribution([item[name] for item in run_totals])
                        for name in metric_names},
            "mpc_group_calls": distribution(mpc_calls) if mpc_calls else None,
            "accuracy": None,
            "accuracy_evidence": "PENDING_REAL_SECURE_COHORT",
        }
        if variant["secure_accuracy_json"]:
            accuracy_path = Path(variant["secure_accuracy_json"])
            accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
            if (accuracy.get("samples", 0) < 50 or
                    accuracy.get("head_plan_sha256") != row["head_plan_sha256"] or
                    accuracy.get("binary_sha256") not in cohort_binary):
                raise ValueError(f"accuracy evidence mismatch: {accuracy_path}")
            row["accuracy"] = accuracy["accuracy"]
            row["top1_agreement"] = accuracy["top1_agreement"]
            row["accuracy_evidence"] = str(accuracy_path.resolve())
        result_rows.append(row)

    for name, values in (("binary", cohort_binary), ("model", cohort_model),
                         ("input", cohort_input), ("image", cohort_image)):
        if None in values or len(values) != 1:
            raise ValueError(f"full-12 cohort {name} invariant differs: {values}")
    complete_accuracy = all(row["accuracy"] is not None for row in result_rows)
    result = {
        "format": "OURS_VIT_FULL12_SUITE_V1",
        "status": ("PASS" if complete_accuracy
                   else "PERFORMANCE_PASS_ACCURACY_PENDING"),
        "binary_sha256": next(iter(cohort_binary)),
        "model_sha256": next(iter(cohort_model)),
        "input_sha256": next(iter(cohort_input)),
        "image_id": next(iter(cohort_image)),
        "network": "1Gbps/0.3ms_RTT_verified_by_cohort_audit",
        "communication_scope": "Alice_plus_Bob_aggregate",
        "latency_repeats_reported_as": "mean_sample_std_p95_min_max",
        "rows": result_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(f"OURS_FULL12_SUITE status={result['status']} rows={len(result_rows)}")
    print(f"OURS_FULL12_SUITE_JSON={args.output}")


if __name__ == "__main__":
    main()
