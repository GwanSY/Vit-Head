#!/usr/bin/env python3
"""Aggregate faithful BLB-ViT measurements from the two protocol parties."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


STAGE_NAMES = [
    "QKV", "QK^T", "CKKS_to_MPC", "Softmax", "MPC_to_CKKS",
    "Softmax_x_V", "Output_Projection", "LayerNorm1", "FFN1",
    "GELU", "FFN2", "LayerNorm2",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_metrics(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if not rows or not {"metric", "value"}.issubset(rows[0]):
        raise ValueError(f"unsupported metric CSV: {path}")
    return {row["metric"]: row["value"] for row in rows}


def integer(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def real(row: dict[str, str], key: str) -> float:
    return float(row[key])


def optional_real(row: dict[str, str], key: str) -> float:
    """Read a profiling-only metric while accepting pre-instrumentation runs."""
    return float(row.get(key, "0"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir

    latency = [read_metrics(run_dir / f"latency_party{p}.csv") for p in (1, 2)]
    totals = [read_metrics(run_dir / f"execution_totals_party{p}.csv")
              for p in (1, 2)]
    blocks = [read_csv(run_dir / f"block_profile_party{p}.csv")
              for p in (1, 2)]
    online = [read_csv(run_dir / f"block_online_profile_party{p}.csv")
              for p in (1, 2)]
    communication = [
        read_csv(run_dir / f"communication_breakdown_party{p}.csv")
        for p in (1, 2)
    ]
    if not blocks[0] or len(blocks[0]) != len(blocks[1]):
        raise ValueError("party block profiles are empty or have different lengths")
    block_ids = [row["block"] for row in blocks[0]]
    if block_ids != [row["block"] for row in blocks[1]]:
        raise ValueError("party block profile IDs differ")

    count_metrics = [
        "he_ct_pt_multiply", "he_ct_ct_multiply", "he_rotations",
        "he_rescales", "ckks_to_mpc", "mpc_to_ckks",
        "field_to_ring_calls", "ring_to_field_calls",
        "conversion_ciphertexts",
    ]
    summary_rows: list[tuple[str, object]] = [
        ("model", "ViT-B/16"),
        ("logical_tokens", 197),
        ("physical_tokens", 256),
        ("executed_transformer_blocks", len(block_ids)),
        ("secure_end_to_end_latency_s",
         max(real(row, "secure_end_to_end_latency_s") for row in latency)),
        ("setup_offline_s", max(real(row, "setup_offline_s") for row in totals)),
        ("transformer_blocks_wall_s",
         max(real(row, "transformer_blocks_wall_s") for row in totals)),
        ("he_instrumented_s",
         max(real(row, "he_instrumented_s") for row in totals)),
        ("conversion_instrumented_s",
         max(real(row, "conversion_instrumented_s") for row in totals)),
        ("h2d_instrumented_s",
         max(optional_real(row, "h2d_instrumented_s") for row in totals)),
        ("plaintext_h2d_instrumented_s",
         max(optional_real(row, "plaintext_h2d_instrumented_s") for row in totals)),
        ("plaintext_device_encode_instrumented_s",
         max(optional_real(row, "plaintext_device_encode_instrumented_s")
             for row in totals)),
        ("plaintext_device_release_instrumented_s",
         max(optional_real(row, "plaintext_device_release_instrumented_s")
             for row in totals)),
        ("ciphertext_h2d_instrumented_s",
         max(optional_real(row, "ciphertext_h2d_instrumented_s") for row in totals)),
        ("d2h_instrumented_s",
         max(optional_real(row, "d2h_instrumented_s") for row in totals)),
        ("mpc_nonlinear_bucket_s",
         max(real(row, "mpc_nonlinear_bucket_s") for row in totals)),
        ("total_communication_bytes_alice_plus_bob",
         sum(integer(row, "total_communication_bytes") for row in latency)),
        ("online_communication_bytes_alice_plus_bob",
         sum(integer(row, "online_communication_bytes") for row in latency)),
        ("initialization_communication_bytes_alice_plus_bob",
         sum(integer(row, "initialization_communication_bytes") for row in latency)),
        ("online_communication_rounds_alice_plus_bob",
         sum(integer(row, "online_communication_rounds") for row in latency)),
    ]
    summary_rows.extend(
        (metric, max(integer(row, metric) for row in totals))
        for metric in count_metrics
    )
    with (run_dir / "baseline_summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)
    summary = dict(summary_rows)

    stage_bytes: dict[str, int] = {}
    stage_rounds: dict[str, int] = {}
    for rows in communication:
        for row in rows:
            stage = row["stage"]
            stage_bytes[stage] = stage_bytes.get(stage, 0) + integer(row, "bytes")
            stage_rounds[stage] = stage_rounds.get(stage, 0) + integer(row, "rounds")
    with (run_dir / "communication_breakdown_alice_plus_bob.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", "bytes", "rounds"])
        for stage in stage_bytes:
            writer.writerow([stage, stage_bytes[stage], stage_rounds[stage]])

    float_fields = {
        field for field in blocks[0][0]
        if field.endswith("_ms")
    }
    summed_fields = {
        "block_online_comm_bytes", "conversion_bytes", "ct_transfer_bytes",
        "mpc_communication_bytes", "total_communication_bytes",
        "communication_rounds",
    }
    fields = [field for field in blocks[0][0] if field != "block"]
    with (run_dir / "block_profile_alice_plus_bob.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["block", *fields])
        for left, right in zip(blocks[0], blocks[1]):
            values: list[object] = []
            for field in fields:
                if field == "block_online_time_ms":
                    values.append(real(left, field))
                elif field in float_fields:
                    values.append(max(real(left, field), real(right, field)))
                elif field in summed_fields:
                    values.append(integer(left, field) + integer(right, field))
                else:
                    values.append(max(integer(left, field), integer(right, field)))
            writer.writerow([left["block"], *values])

    stage_maps: list[dict[tuple[str, str], dict[str, str]]] = []
    for rows in online:
        stage_maps.append({(row["block"], row["stage"]): row for row in rows
                           if row["stage"] in STAGE_NAMES})
    breakdown: list[list[object]] = []
    total_wall_ms = 0.0
    total_send_bytes = 0
    per_stage_wall = {name: 0.0 for name in STAGE_NAMES}
    per_stage_bytes = {name: 0 for name in STAGE_NAMES}
    for block in block_ids:
        for stage in STAGE_NAMES:
            key = (block, stage)
            if key not in stage_maps[0] or key not in stage_maps[1]:
                raise ValueError(f"missing online stage {stage} for block {block}")
            wall_ms = real(stage_maps[0][key], "wall_ms")
            send_bytes = sum(integer(mapping[key], "send_bytes")
                             for mapping in stage_maps)
            breakdown.append([block, stage, wall_ms / 1000.0, send_bytes])
            total_wall_ms += wall_ms
            total_send_bytes += send_bytes
            per_stage_wall[stage] += wall_ms
            per_stage_bytes[stage] += send_bytes
    with (run_dir / "block_online_breakdown_alice_plus_bob.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["block", "stage", "wall_s_alice_timeline",
                         "send_bytes_alice_plus_bob"])
        writer.writerows(breakdown)
        for stage in STAGE_NAMES:
            writer.writerow(["TOTAL", stage, per_stage_wall[stage] / 1000.0,
                             per_stage_bytes[stage]])
        writer.writerow(["ALL_BLOCKS", "ALL_STAGES", total_wall_ms / 1000.0,
                         total_send_bytes])

    with (run_dir / "block_online_summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerow(["transformer_blocks", len(block_ids)])
        writer.writerow(["BLOCKS_ONLINE_TIME", total_wall_ms / 1000.0])
        writer.writerow(["BLOCKS_ONLINE_COMM", total_send_bytes])
        writer.writerow(["wall_clock_timeline_party", 1])
        for stage in STAGE_NAMES:
            writer.writerow([f"{stage}.wall_s", per_stage_wall[stage] / 1000.0])
            writer.writerow([f"{stage}.send_bytes", per_stage_bytes[stage]])

    total_comm = int(summary["total_communication_bytes_alice_plus_bob"])
    with (run_dir / "baseline_summary.md").open("w", encoding="utf-8") as stream:
        stream.write("# BLB ViT-B/16 secure inference measurement\n\n")
        stream.write("Protocol: original BLB CKKS + MPC/VOLE; no protocol substitution.\n\n")
        stream.write("- Packing: **197 logical tokens → 256 physical rows**\n")
        stream.write(f"- Executed Transformer blocks: **{len(block_ids)}**\n")
        stream.write(f"- Secure end-to-end: **{float(summary['secure_end_to_end_latency_s']):.6f} s**\n")
        stream.write(f"- Communication (Alice+Bob): **{total_comm} bytes ({total_comm / 1e9:.6f} GB)**\n")
        stream.write(f"- HE internal counter: **{float(summary['he_instrumented_s']):.6f} s**\n")
        stream.write(f"- Conversion internal counter: **{float(summary['conversion_instrumented_s']):.6f} s**\n")
        stream.write(f"- H2D internal counter: **{float(summary['h2d_instrumented_s']):.6f} s**\n")
        stream.write(
            "- Phantom plaintext encode: "
            f"**{float(summary['plaintext_device_encode_instrumented_s']):.6f} s**\n")
        stream.write(
            "- Phantom plaintext release: "
            f"**{float(summary['plaintext_device_release_instrumented_s']):.6f} s**\n")
        stream.write(f"- D2H internal counter: **{float(summary['d2h_instrumented_s']):.6f} s**\n")
        stream.write(f"- MPC nonlinear wall buckets: **{float(summary['mpc_nonlinear_bucket_s']):.6f} s**\n")
        stream.write("\nThe nonlinear bucket is diagnostic and overlaps the prescribed HE/conversion boundaries inside LayerNorm; it is not added to the other counters.\n\n")
        stream.write(
            f"| Stage | {len(block_ids)}-block wall time (s) | "
            "Alice+Bob send bytes |\n")
        stream.write("|---|---:|---:|\n")
        for stage in STAGE_NAMES:
            stream.write(f"| {stage} | {per_stage_wall[stage] / 1000.0:.6f} | {per_stage_bytes[stage]} |\n")
        stream.write("\n- Full counters: `baseline_summary.csv`\n")
        stream.write("- Per-block counters: `block_profile_alice_plus_bob.csv`\n")
        stream.write("- Per-block stage timeline: `block_online_breakdown_alice_plus_bob.csv`\n")


if __name__ == "__main__":
    main()
