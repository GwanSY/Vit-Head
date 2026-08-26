#!/usr/bin/env python3
"""Aggregate the two local-party A0 metrics into system-level numbers."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Union


Scalar = Union[int, float, str]


def scalar(value):
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def parse(path, metric_prefix):
    line_prefixes = tuple(
        f"{metric_prefix}_{suffix} "
        for suffix in ("TIMING", "COMM", "CONVERSIONS", "HE_OPS", "LAYOUT")
    )
    result = {}  # type: Dict[str, Dict[str, Scalar]]
    text = path.read_text(encoding="utf-8", errors="replace")
    if f"{metric_prefix} status=PASS backend=phantom_gpu" not in text:
        raise RuntimeError(
            f"{path}: missing Phantom GPU {metric_prefix} PASS marker"
        )
    for line in text.splitlines():
        for prefix in line_prefixes:
            if line.startswith(prefix):
                fields = {}  # type: Dict[str, Scalar]
                for item in line[len(prefix):].split():
                    if "=" in item:
                        key, value = item.split("=", 1)
                        fields[key] = scalar(value)
                result[prefix.strip()] = fields
    missing = {item.strip() for item in line_prefixes} - result.keys()
    if missing:
        raise RuntimeError(f"{path}: missing metrics {sorted(missing)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alice", type=Path, required=True)
    parser.add_argument("--bob", type=Path, required=True)
    parser.add_argument("--prefix", choices=("A0", "HETERO"), default="A0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    alice = parse(args.alice, args.prefix)
    bob = parse(args.bob, args.prefix)
    timing_name = f"{args.prefix}_TIMING"
    comm_name = f"{args.prefix}_COMM"
    conversion_name = f"{args.prefix}_CONVERSIONS"
    ops_name = f"{args.prefix}_HE_OPS"
    layout_name = f"{args.prefix}_LAYOUT"

    timing_keys = (
        "end_to_end_seconds", "he_seconds", "mpc_and_host_seconds",
        "conversion_seconds", "ciphertext_transfer_seconds",
        "h2d_d2h_seconds", "normalization_wall_seconds",
        "normalization_mpc_seconds",
    )
    report = {
        "status": "PASS",
        "backend": "phantom_gpu",
        "protocol": "BLB_Softmax" if args.prefix == "A0"
                    else "head_heterogeneous_fixed",
        # Parties execute concurrently; the protocol wall clock is the slower
        # local online interval, not the sum of both intervals.
        "timing_seconds": {
            key: max(float(alice[timing_name][key]),
                     float(bob[timing_name][key]))
            for key in timing_keys
        },
        # NetIO counters are local transmitted-byte counters.  Their sum is
        # the bidirectional system communication.
        "communication_bytes": {
            "total": int(alice[comm_name]["total_local_bytes"]) +
                     int(bob[comm_name]["total_local_bytes"]),
            "conversion": int(alice[comm_name]["conversion_local_bytes"]) +
                          int(bob[comm_name]["conversion_local_bytes"]),
            "ciphertext_transfer": int(
                alice[comm_name]["ciphertext_transfer_local_bytes"]
            ) + int(bob[comm_name]["ciphertext_transfer_local_bytes"]),
            "rounds": max(int(alice[comm_name]["rounds"]),
                          int(bob[comm_name]["rounds"])),
        },
        "conversions": {
            key: max(int(alice[conversion_name][key]),
                     int(bob[conversion_name][key]))
            for key in (
                "ckks_to_mpc_ciphertexts", "mpc_to_ckks_ciphertexts",
                "ckks_to_mpc_calls", "mpc_to_ckks_calls",
                "field_to_ring_calls", "masked_rns_coefficients",
            )
        },
        # Only ALICE evaluates HE.  Secure truncation is a two-party protocol,
        # so its logical invocation count is the maximum party-local count;
        # summing it would double-count every invocation.
        "he_operations": {
            key: int(alice[ops_name][key]) +
                 int(bob[ops_name][key])
            for key in (
                "rotations", "ct_ct_multiply", "ct_pt_multiply",
                "relinearization", "gpu_rescale", "host_rescale",
                "conjugation",
            )
        },
        "layout": alice[layout_name],
        "party_local": {"alice": alice, "bob": bob},
    }
    report["he_operations"]["secure_truncation_calls"] = max(
        int(alice[ops_name]["secure_truncation_calls"]),
        int(bob[ops_name]["secure_truncation_calls"]),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
