#!/usr/bin/env python3
"""Collect directly measured Softmax, ReLU, and native CKKS Linear groups.

Rows are copied from successful logs.  Missing profiler fields stay ``N/A``;
this script never divides a packed run, copies a group to another size, or
turns an A2 Linear surrogate into a native Linear measurement.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
from pathlib import Path


GROUP_SIZES = (1, 2, 4, 6, 8, 12)
PATHS = ("Softmax", "ReLU", "Linear")
OUT_COLUMNS = (
    "path", "heads_group", "online_latency_s", "he_time_s", "mpc_time_s",
    "communication_mb", "rounds", "conversion_ct", "he_rotations",
    "he_muls", "measurement_scope", "network", "gpu_topology", "source_run",
)


def values(line: str) -> dict[str, str]:
    return {
        token.split("=", 1)[0]: token.split("=", 1)[1]
        for token in line.split()
        if "=" in token
    }


def read_marker(run: Path, prefix: str) -> list[dict[str, str]]:
    result = []
    for party in ("alice.log", "bob.log"):
        text = (run / party).read_text(encoding="utf-8", errors="replace")
        result.extend(values(line) for line in text.splitlines()
                      if line.startswith(prefix))
    return result


def require_run(run: Path, allow_unshaped: bool) -> str:
    status = (run / "status.txt").read_text(encoding="utf-8")
    if "alice_rc=0" not in status or "bob_rc=0" not in status:
        raise ValueError(f"secure run did not pass: {run}")
    environment = (run / "environment.txt").read_text(encoding="utf-8")
    if "backend=gpu" not in environment:
        raise ValueError(f"run is not a formal GPU/network measurement: {run}")
    if "network=1Gbps/0.3ms_RTT" in environment:
        network = "1Gbps/0.3ms_RTT"
    elif allow_unshaped and "network=unshaped_diagnostic_only" in environment:
        network = "unshaped_diagnostic_only"
    else:
        raise ValueError(f"run is not a formal GPU/network measurement: {run}")
    selection = (run / "gpu_selection.txt").read_text(encoding="utf-8")
    if "party_topology=separate_physical_gpus" not in selection:
        raise ValueError(f"run does not use two physical GPUs: {run}")
    return network


def number(marker: dict[str, str], name: str) -> str:
    return marker.get(name, "N/A")


def aggregate(markers: list[dict[str, str]], path: str, scope: str,
              run: Path, network: str) -> dict[str, object] | None:
    if not markers:
        raise ValueError(f"missing {path} marker in {run}")
    count_key = "active_heads" if path == "Linear" else "head_count"
    counts = {int(marker[count_key]) for marker in markers}
    if len(counts) != 1:
        raise ValueError(f"party group sizes disagree for {path}: {run}")
    count = counts.pop()
    if count not in GROUP_SIZES:
        return None
    domains = {marker.get("execution_domain", "").lower() for marker in markers}
    if path == "Linear":
        if domains != {"he1_native"}:
            raise ValueError(f"Linear row is not native CKKS: {run}")
        latency = max(float(marker["latency_ms"]) for marker in markers)
    else:
        if domains != {"mpc_ckks_hybrid"}:
            raise ValueError(f"{path} row is not the hybrid MPC path: {run}")
        latency = max(float(marker["latency_ms"]) for marker in markers)

    def max_field(name: str) -> str:
        present = [float(marker[name]) for marker in markers if name in marker]
        return str(max(present)) if present else "N/A"

    communications = [int(marker["communication_bytes"]) for marker in markers
                      if "communication_bytes" in marker]
    rounds = [int(marker["rounds"]) for marker in markers if "rounds" in marker]
    return {
        "path": path,
        "heads_group": count,
        "online_latency_s": latency / 1000.0,
        "he_time_s": float(max_field("he_ms")) / 1000.0 if max_field("he_ms") != "N/A" else "N/A",
        "mpc_time_s": float(max_field("mpc_ms")) / 1000.0 if max_field("mpc_ms") != "N/A" else "N/A",
        "communication_mb": sum(communications) / 1_000_000.0,
        "rounds": max(rounds) if rounds else "N/A",
        "conversion_ct": max_field("conversion_ct"),
        "he_rotations": max_field("he_rotations"),
        "he_muls": max_field("he_muls"),
        "measurement_scope": scope,
        "network": network,
        "gpu_topology": "separate_physical_gpus",
        "source_run": str(run.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--include-path", choices=PATHS, action="append")
    parser.add_argument("--allow-unshaped", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for run in args.run:
        network = require_run(run, args.allow_unshaped)
        a2 = read_marker(run, "OURS_HETERO_PATH_PROFILE ")
        for protocol, path in (("softmax_mpc", "Softmax"),
                               ("relu_norm_mpc", "ReLU")):
            if args.include_path and path not in args.include_path:
                continue
            selected = [marker for marker in a2
                        if marker.get("ablation", "").upper() in {"A2", "A4"}
                        and marker.get("protocol") == protocol]
            if selected:
                row = aggregate(selected, path,
                                "A2/A4 direct grouped normalization marker",
                                run, network)
                if row is not None:
                    rows.append(row)

        native = read_marker(run, "OURS_A4_HE_PATH_PROFILE ")
        if native and (not args.include_path or "Linear" in args.include_path):
            row = aggregate(native, "Linear",
                            "A4 direct native CKKS attention path", run, network)
            if row is not None:
                rows.append(row)

    keys = [(row["path"], row["heads_group"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate path/group rows; use one run per group size")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (PATHS.index(str(row["path"])),
                                                        int(row["heads_group"]))))
    print(f"OURS_THREE_PATH_COSTS status=PASS rows={len(rows)}")
    print(f"OURS_THREE_PATH_COSTS_CSV={args.output}")


if __name__ == "__main__":
    main()
