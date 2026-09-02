#!/usr/bin/env python3
"""Collect A2 costs from real grouped secure normalization runs.

Each ``--run`` may contain one or more contiguous protocol groups.  The cost
unit is the complete measured group C(protocol, head_count); packed work is
never divided by the number of heads.  Runs must use two separate dynamically
selected GPUs and formal 1 Gbps / 0.3 ms network shaping.

The A1/A2 Linear implementation is only an MPC semantic surrogate.  Its
diagnostic row is retained with ``selection_eligible=0``.  A4 must provide a
real CKKS row before Linear can participate in automatic selection.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


PROTOCOLS = ("softmax_mpc", "relu_norm_mpc", "he_linear")
PROFILE_PREFIX = "OURS_HETERO_PATH_PROFILE "
OUTPUT_COLUMNS = (
    "layer", "protocol", "latency_ms", "he_ms", "mpc_ms",
    "conversion_ms", "communication_bytes", "rounds", "head_count",
    "packing_utilization", "multiplicative_depth", "ct_ct_multiply",
    "rotations", "relinearization", "rescale", "measured", "backend",
    "network_gbps", "rtt_ms", "execution_domain", "cost_stage",
    "measurement_layer", "measurement_scope", "conversion_scope",
    "cost_scope", "gpu_topology", "selection_eligible", "source_run",
)


@dataclass(frozen=True)
class PartyPathProfile:
    layer: int
    protocol: str
    physical_head_begin: int
    latency_ms: float
    communication_bytes: int
    rounds: int
    head_count: int
    execution_domain: str


def key_values(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def require_success(run: Path) -> str:
    status = run / "status.txt"
    if not status.is_file():
        raise ValueError(f"missing status.txt in {run}")
    values = key_values(status.read_text(encoding="utf-8").replace("\n", " "))
    if values.get("alice_rc") != "0" or values.get("bob_rc") != "0":
        raise ValueError(f"secure run did not pass: {run}")
    environment_path = run / "environment.txt"
    if not environment_path.is_file():
        raise ValueError(f"missing environment.txt in {run}")
    environment = environment_path.read_text(encoding="utf-8")
    if "network=1Gbps/0.3ms_RTT" not in environment:
        raise ValueError(f"run was not measured at 1 Gbps / 0.3 ms RTT: {run}")
    if "cost_scope=a2_normalization_only" not in environment:
        raise ValueError(f"run is not an A2 normalization-only cost run: {run}")
    selection_path = run / "gpu_selection.txt"
    if not selection_path.is_file():
        raise ValueError(f"missing gpu_selection.txt in {run}")
    selection = key_values(
        selection_path.read_text(encoding="utf-8").replace("\n", " "))
    topology = selection.get("party_topology", "")
    if topology != "separate_physical_gpus":
        raise ValueError(
            f"A2 formal costs require two separate physical GPUs: {run}")
    physical_gpus = selection.get("physical_gpus", "").split(",")
    if len(physical_gpus) != 2 or physical_gpus[0] == physical_gpus[1]:
        raise ValueError(f"invalid two-GPU selection in {run}")
    for party_name in ("alice.log", "bob.log"):
        log_path = run / party_name
        if not log_path.is_file():
            raise ValueError(f"missing {party_name} in {run}")
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if ("OURS_NORMALIZATION_COST_STOP status=PASS" not in log or
                "cost_scope=a2_normalization_only" not in log):
            raise ValueError(
                f"{party_name} lacks a successful normalization stop in {run}")
    return topology


def read_path_profiles(run: Path) -> list[PartyPathProfile]:
    parties: list[dict[tuple[int, str, int, int], PartyPathProfile]] = []
    for party_name in ("alice.log", "bob.log"):
        found: dict[tuple[int, str, int, int], PartyPathProfile] = {}
        for line in (run / party_name).read_text(
                encoding="utf-8", errors="replace").splitlines():
            if not line.startswith(PROFILE_PREFIX):
                continue
            values = key_values(line)
            if values.get("status") != "PASS" or values.get("ablation") != "A2":
                continue
            protocol = values["protocol"]
            if protocol not in PROTOCOLS:
                raise ValueError(f"unknown protocol {protocol} in {run}")
            profile = PartyPathProfile(
                layer=int(values["layer"]),
                protocol=protocol,
                physical_head_begin=int(values["physical_head_begin"]),
                latency_ms=float(values["latency_ms"]),
                communication_bytes=int(values["communication_bytes"]),
                rounds=int(values["rounds"]),
                head_count=int(values["head_count"]),
                execution_domain=values["execution_domain"].lower(),
            )
            key = (profile.layer, profile.protocol,
                   profile.physical_head_begin, profile.head_count)
            if key in found:
                raise ValueError(f"duplicate grouped path profile {key} in {run}")
            found[key] = profile
        if not found:
            raise ValueError(f"no A2 path profile in {run}/{party_name}")
        parties.append(found)
    if set(parties[0]) != set(parties[1]):
        raise ValueError(f"party path-profile groups disagree in {run}")

    result: list[PartyPathProfile] = []
    for key in sorted(parties[0]):
        alice, bob = parties[0][key], parties[1][key]
        if alice.execution_domain != bob.execution_domain:
            raise ValueError(f"party execution domains disagree for {key}")
        result.append(PartyPathProfile(
            layer=alice.layer,
            protocol=alice.protocol,
            physical_head_begin=alice.physical_head_begin,
            latency_ms=max(alice.latency_ms, bob.latency_ms),
            communication_bytes=(alice.communication_bytes +
                                 bob.communication_bytes),
            rounds=max(alice.rounds, bob.rounds),
            head_count=alice.head_count,
            execution_domain=alice.execution_domain,
        ))
    return result


def measured_rows(run: Path) -> list[dict[str, object]]:
    gpu_topology = require_success(run)
    rows: list[dict[str, object]] = []
    for path in read_path_profiles(run):
        expected_domain = ("mpc_surrogate" if path.protocol == "he_linear"
                           else "mpc_ckks_hybrid")
        if path.execution_domain != expected_domain:
            raise ValueError(
                f"unexpected execution domain for {path.protocol}: "
                f"{path.execution_domain}, expected {expected_domain}")
        selection_eligible = int(path.protocol != "he_linear")
        rows.append({
            "layer": path.layer,
            "protocol": path.protocol,
            "latency_ms": path.latency_ms,
            # The path marker currently provides total wall time only.  Do not
            # manufacture an HE/MPC split; A3 profiling supplies that detail.
            "he_ms": 0.0,
            "mpc_ms": 0.0,
            "conversion_ms": 0.0,
            "communication_bytes": path.communication_bytes,
            "rounds": path.rounds,
            "head_count": path.head_count,
            "packing_utilization": (
                path.head_count * 197 * 197 /
                (path.head_count * 4 * 16384)),
            "multiplicative_depth": 0,
            "ct_ct_multiply": 0,
            "rotations": 0,
            "relinearization": 0,
            "rescale": 0,
            "measured": 1,
            "backend": "phantom_gpu",
            "network_gbps": 1.0,
            "rtt_ms": 0.3,
            "execution_domain": path.execution_domain,
            "cost_stage": "A2",
            "measurement_layer": path.layer,
            "measurement_scope": "direct_layer_normalization_only",
            "conversion_scope": "excluded_deferred_to_A3",
            "cost_scope": "a2_normalization_only",
            "gpu_topology": gpu_topology,
            "selection_eligible": selection_eligible,
            "source_run": str(run.resolve()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=Path, action="append", required=True,
        help=("formal A2 run; repeat for each required protocol/group size"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replicate-shape-identical-layers", action="store_true",
        help=("reuse directly measured layer-0 grouped kernels for layers "
              "1..11; copied rows are explicitly marked shape_reuse"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for run in args.run:
        run_rows = measured_rows(run)
        if args.replicate_shape_identical_layers:
            if any(int(row["layer"]) != 0 for row in run_rows):
                raise ValueError("shape reuse requires layer-0-only runs")
            expanded: list[dict[str, object]] = []
            for row in run_rows:
                for layer in range(12):
                    copy = dict(row)
                    copy["layer"] = layer
                    if layer != 0:
                        copy["measurement_scope"] = (
                            "shape_reuse_normalization_only")
                    expanded.append(copy)
            run_rows = expanded
        rows.extend(run_rows)

    keys = [(int(row["layer"]), str(row["protocol"]),
             int(row["head_count"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate layer/protocol/group-size cost rows")
    eligible_protocols = {
        str(row["protocol"]) for row in rows
        if int(row["selection_eligible"]) == 1
    }
    if "softmax_mpc" not in eligible_protocols:
        raise ValueError("cost table requires an eligible Softmax group")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(
            rows, key=lambda row: (int(row["layer"]),
                                   str(row["protocol"]),
                                   int(row["head_count"]))))
    print("OURS_A2_COST_COLLECTION status=PASS model=measured_group_cost")
    print("OURS_A2_LINEAR_SURROGATE selection_eligible=0")
    print(f"OURS_A2_COST_CSV={args.output}")


if __name__ == "__main__":
    main()
