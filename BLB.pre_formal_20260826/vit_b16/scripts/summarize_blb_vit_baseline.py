#!/usr/bin/env python3
"""Aggregate the two per-party faithful BLB-ViT profiling files.

The protocol counters are normally identical on both parties, so counts use
the maximum. Communication is explicitly Alice+Bob, as in the paper tables.
Latency uses the slower party's measured secure wall time.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_metric_csv(path: Path) -> dict[str, str]:
    """Read the metric,value latency format emitted by partial/full runs."""
    rows = read_csv(path)
    if rows and set(rows[0]) >= {"metric", "value"}:
        return {row["metric"]: row["value"] for row in rows}
    if len(rows) == 1:
        return rows[0]
    raise ValueError(f"unsupported latency CSV format: {path}")


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir

    latency_rows = [
        read_metric_csv(run_dir / f"latency_party{party}.csv")
        for party in (1, 2)
    ]
    block_rows = [
        read_csv(run_dir / f"block_profile_party{party}.csv") for party in (1, 2)
    ]
    online_stage_rows = [
        read_csv(run_dir / f"block_online_profile_party{party}.csv")
        for party in (1, 2)
    ]
    comm_rows = [
        read_csv(run_dir / f"communication_breakdown_party{party}.csv")
        for party in (1, 2)
    ]

    summary = run_dir / "baseline_summary.csv"
    with summary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerow(
            [
                "secure_end_to_end_latency_s",
                max(as_float(row, "secure_end_to_end_latency_s") for row in latency_rows),
            ]
        )
        writer.writerow(
            [
                "total_communication_bytes_alice_plus_bob",
                sum(as_int(row, "total_communication_bytes") for row in latency_rows),
            ]
        )
        writer.writerow(
            [
                "online_communication_bytes_alice_plus_bob",
                sum(as_int(row, "online_communication_bytes") for row in latency_rows),
            ]
        )
        writer.writerow(
            [
                "initialization_communication_bytes_alice_plus_bob",
                sum(as_int(row, "initialization_communication_bytes") for row in latency_rows),
            ]
        )

    stages: dict[str, list[int]] = {}
    rounds: dict[str, list[int]] = {}
    for rows in comm_rows:
        for row in rows:
            stages.setdefault(row["stage"], []).append(as_int(row, "bytes"))
            rounds.setdefault(row["stage"], []).append(as_int(row, "rounds"))
    with (run_dir / "communication_breakdown_alice_plus_bob.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "bytes", "rounds"])
        for stage in stages:
            writer.writerow([stage, sum(stages[stage]), sum(rounds[stage])])

    field_names = [
        "secure_end_to_end_ms",
        "block_online_time_ms",
        "block_online_comm_bytes",
        "he_time_ms",
        "qkv_he_ms",
        "qk_transpose_he_ms",
        "softmax_v_he_ms",
        "linear2_he_ms",
        "linear1_he_ms",
        "linear2_softmax_v_he_ms",
        "linear3_he_ms",
        "linear4_he_ms",
        "linear_he_linear1_4_ms",
        "linear_he_including_mul_v_ms",
        "nonlinear_softmax_bucket_ms",
        "nonlinear_layernorm_bucket_ms",
        "nonlinear_gelu_core_ms",
        "conversion_time_ms",
        "he_ct_pt_multiply",
        "he_ct_ct_multiply",
        "he_rotations",
        "he_rescales",
        "ckks_to_mpc",
        "mpc_to_ckks",
        "conversion_ciphertexts",
        "conversion_bytes",
        "ct_transfer_bytes",
        "mpc_communication_bytes",
        "total_communication_bytes",
        "communication_rounds",
    ]
    with (run_dir / "block_profile_alice_plus_bob.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["block"] + field_names)
        for index in range(len(block_rows[0])):
            left = block_rows[0][index]
            right = block_rows[1][index]
            values = []
            for field in field_names:
                if field == "block_online_time_ms":
                    # Alice is the CKKS evaluator and is the only party with
                    # independent QKV/QK^T boundaries. Use that one timeline
                    # so all stage wall times remain mutually exclusive.
                    values.append(as_float(left, field))
                elif field.endswith("_ms"):
                    values.append(max(as_float(left, field), as_float(right, field)))
                elif field in {"block_online_comm_bytes", "conversion_bytes", "ct_transfer_bytes",
                               "mpc_communication_bytes",
                               "total_communication_bytes", "communication_rounds"}:
                    values.append(as_int(left, field) + as_int(right, field))
                else:
                    values.append(max(as_int(left, field), as_int(right, field)))
            writer.writerow([left["block"]] + values)

    # The online stage profiler uses the slower party's wall-clock timeline
    # and Alice+Bob sum for send-only communication.  This keeps stage times
    # mutually exclusive and prevents per-party maxima from exceeding the
    # block wall clock.
    stage_names = [
        "QKV", "QK^T", "CKKS_to_MPC", "Softmax", "MPC_to_CKKS",
        "Softmax_x_V", "Output_Projection", "LayerNorm1", "FFN1",
        "GELU", "FFN2", "LayerNorm2",
    ]
    stage_maps = []
    for rows in online_stage_rows:
        stage_maps.append({row["stage"]: row for row in rows
                           if row.get("stage") in stage_names})
    block_time_ms_by_party = [
        as_float(block_rows[party_index][0], "block_online_time_ms")
        for party_index in (0, 1)
    ]
    wall_clock_timeline_party = 0  # Alice owns the exact evaluator boundaries.
    block_online_time_ms = block_time_ms_by_party[wall_clock_timeline_party]
    stage_breakdown = []
    block_online_comm = 0
    for stage in stage_names:
        wall_ms = as_float(stage_maps[wall_clock_timeline_party][stage], "wall_ms")
        send_bytes = sum(as_int(stage_map[stage], "send_bytes")
                         for stage_map in stage_maps)
        block_online_comm += send_bytes
        stage_breakdown.append(
            [
                stage,
                wall_ms / 1000.0,
                wall_ms / block_online_time_ms if block_online_time_ms else 0.0,
                send_bytes,
                0.0,
            ]
        )
    for row in stage_breakdown:
        row[4] = row[3] / block_online_comm if block_online_comm else 0.0

    with (run_dir / "block_online_breakdown_alice_plus_bob.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "wall_s", "wall_share", "send_bytes_alice_plus_bob", "comm_share"])
        writer.writerows(stage_breakdown)

    block_online_summary = run_dir / "block_online_summary.csv"
    with block_online_summary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerow(["BLOCK_ONLINE_TIME", block_online_time_ms / 1000.0])
        writer.writerow(["BLOCK_ONLINE_COMM", block_online_comm])
        writer.writerow(["wall_clock_timeline_party", wall_clock_timeline_party + 1])
        for stage, wall_s, wall_share, send_bytes, comm_share in stage_breakdown:
            writer.writerow([f"{stage}.wall_s", wall_s])
            writer.writerow([f"{stage}.wall_share", wall_share])
            writer.writerow([f"{stage}.send_bytes", send_bytes])
            writer.writerow([f"{stage}.comm_share", comm_share])

    md = run_dir / "baseline_summary.md"
    total_comm = sum(as_int(row, "total_communication_bytes") for row in latency_rows)
    online_comm = sum(as_int(row, "online_communication_bytes") for row in latency_rows)
    latency_seconds = max(
        as_float(row, "secure_end_to_end_latency_s") for row in latency_rows
    )
    conversion_time_ms = max(
        as_float(row[0], "conversion_time_ms") for row in block_rows
    )
    with md.open("w") as stream:
        stream.write("# BLB-ViT Transformer block online profile\n\n")
        stream.write("Protocol: original BLB CKKS + original BLB MPC; profiling-only changes.\n\n")
        stream.write(f"- BLOCK_ONLINE_TIME: **{block_online_time_ms / 1000.0:.6f} s**\n")
        stream.write(f"- BLOCK_ONLINE_COMM (send-only, Alice+Bob): **{block_online_comm} bytes**\n")
        stream.write(f"- Wall-clock stage timeline party: **Alice (CKKS evaluator)**\n")
        stream.write(f"- conversion_time (internal attribution only): **{conversion_time_ms / 1000.0:.6f} s**\n")
        stream.write(f"- Legacy full online communication including patch/embedding: **{online_comm} bytes**\n")
        stream.write(f"- Total communication including initialization: **{total_comm} bytes**\n\n")
        stream.write("| Stage | Wall time (s) | Time share | Send-only bytes | Communication share |\n")
        stream.write("|---|---:|---:|---:|---:|\n")
        for stage, wall_s, wall_share, send_bytes, comm_share in stage_breakdown:
            stream.write(
                f"| {stage} | {wall_s:.6f} | {wall_share * 100:.2f}% | "
                f"{send_bytes} | {comm_share * 100:.2f}% |\n"
            )
        stream.write("\n- Per-block counters: `block_profile_alice_plus_bob.csv`\n")
        stream.write("- Stage profile: `block_online_breakdown_alice_plus_bob.csv`\n")


if __name__ == "__main__":
    main()
