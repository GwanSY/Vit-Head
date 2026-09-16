#!/usr/bin/env python3
"""Audit directly measured A3/A4 runs; never extrapolate per-Head costs."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
import json
import re
from pathlib import Path


def metric_csv(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as source:
        return {row["metric"]: float(row["value"])
                for row in csv.DictReader(source)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inspect_run(path: Path) -> dict[str, object]:
    require(path.is_dir(), f"missing run directory: {path}")
    status = (path / "status.txt").read_text(encoding="utf-8")
    require("alice_rc=0" in status and "bob_rc=0" in status,
            f"party failure in {path}")
    qdisc = (path / "qdisc_active.txt").read_text(encoding="utf-8").strip()
    require(re.search(r"delay 149us\b", qdisc) is not None,
            f"non-formal RTT shaping in {path}")
    require(re.search(r"rate 1Gbit\b", qdisc) is not None,
            f"non-formal bandwidth shaping in {path}")
    latency = metric_csv(path / "latency_party1.csv")
    totals = metric_csv(path / "execution_totals_party1.csv")
    log = (path / "alice.log").read_text(encoding="utf-8")
    forbidden = [
        "client_decrypt_reencrypt=1",
        "client_reencrypt=1",
        "mpc_reference=1",
        "surrogate=1",
    ]
    require(not any(marker in log for marker in forbidden),
            f"forbidden surrogate/client bridge marker in {path}")
    return {
        "path": str(path.resolve()),
        "latency_s": latency["secure_end_to_end_latency_s"],
        "online_communication_bytes": int(
            latency["online_communication_bytes"]),
        "online_communication_rounds": int(
            latency["online_communication_rounds"]),
        "ckks_to_mpc": int(totals["ckks_to_mpc"]),
        "mpc_to_ckks": int(totals["mpc_to_ckks"]),
        "conversion_ciphertexts": int(totals["conversion_ciphertexts"]),
        "qdisc": qdisc,
        "log": log,
    }


def public_metrics(run: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in run.items() if key != "log"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-softmax", type=Path, required=True)
    parser.add_argument("--a4-pair", type=Path, required=True)
    parser.add_argument("--manual-mixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = inspect_run(args.all_softmax)
    a4 = inspect_run(args.a4_pair)
    mixed = inspect_run(args.manual_mixed)
    a4_log = str(a4["log"])
    mixed_log = str(mixed["log"])

    require("head_count=8 qk_ciphertexts=32 v_ciphertexts=8" in a4_log,
            "A4 pair run lacks real selective QK/V conversion")
    require("execution_domain=he1_grouped_shared_A4" in a4_log,
            "A4 pair run lacks shared-he1 A3 execution")
    require("OURS_A4_NATIVE_CKKS status=PASS" in a4_log,
            "A4 pair run lacks native CKKS Linear")
    require("OURS_A4_CKKS_CONTRIBUTION_MERGE status=PASS" in a4_log,
            "A4 pair run lacks CKKS contribution merge")
    require(a4_log.count("OURS_SELECTIVE_OUTPUT_CONVERSION") == 1,
            "A4 pair run did not perform exactly one output conversion")
    require("stage=a4_all_heads_wo_merged" in a4_log,
            "A4 pair run exported the wrong ciphertext contribution")
    for protocol in ("softmax_mpc", "relu_norm_mpc", "he_linear"):
        require(f"protocol={protocol}" in mixed_log,
                f"manual mixed run lacks {protocol}")

    baseline_latency = float(baseline["latency_s"])
    a4_latency = float(a4["latency_s"])
    latency_change = (a4_latency / baseline_latency - 1.0) * 100.0
    communication_reduction = (
        1.0 - int(a4["online_communication_bytes"]) /
        int(baseline["online_communication_bytes"])) * 100.0
    c2m_reduction = (
        1.0 - int(a4["ckks_to_mpc"]) / int(baseline["ckks_to_mpc"])) * 100.0
    ciphertext_reduction = (
        1.0 - int(a4["conversion_ciphertexts"]) /
        int(baseline["conversion_ciphertexts"])) * 100.0

    report = {
        "format": "OURS_A3_A4_REAL_PATH_AUDIT_V1",
        "measurement_policy": "complete_block_direct_measurement_no_cost_extrapolation",
        "formal_network": "1_Gbps_approximately_0.3_ms_RTT",
        "runs": {
            "all_softmax": public_metrics(baseline),
            "a4_pair_aligned_8_softmax_4_linear": public_metrics(a4),
            "manual_4_softmax_4_relu_4_linear": public_metrics(mixed),
        },
        "verdicts": {
            "a3_selective_conversion": "PASS_STRUCTURAL_CRYPTOGRAPHIC",
            "a4_native_ckks_linear": "PASS_STRUCTURAL_CRYPTOGRAPHIC",
            "no_client_bridge_or_mpc_surrogate": "PASS",
            "a4_latency_vs_all_softmax": (
                "PASS" if a4_latency < baseline_latency else "FAIL"),
            "a3_a4_numeric_acceptance": "FAIL_BLOCKED_BY_PAUSED_A0",
            "full_12_block_run": "NOT_RUN_NUMERICAL_GATE",
        },
        "direct_comparison": {
            "latency_change_percent": latency_change,
            "online_communication_reduction_percent": communication_reduction,
            "ckks_to_mpc_reduction_percent": c2m_reduction,
            "conversion_ciphertext_reduction_percent": ciphertext_reduction,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("OURS_A3_A4_REAL_PATH_AUDIT status=PASS "
          "structural_a3=PASS structural_a4=PASS "
          f"latency_gate={report['verdicts']['a4_latency_vs_all_softmax']} "
          "numeric_gate=FAIL_BLOCKED_BY_PAUSED_A0")
    print(f"OURS_A3_A4_REAL_PATH_AUDIT_FILE={args.output}")


if __name__ == "__main__":
    main()
