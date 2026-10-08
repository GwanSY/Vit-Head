#!/usr/bin/env python3
"""Validate and summarize one real full-network heterogeneous secure run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from collect_attention_assignment_costs import measured_rows, key_values


def metric_table(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: float(row["value"])
                for row in csv.DictReader(handle)}


def integer_metric(rows: list[dict[str, float]], name: str) -> int:
    return int(max(row[name] for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()

    direct_rows = measured_rows(run)
    if len(direct_rows) != 12 or {int(row["layer"]) for row in direct_rows} != set(range(12)):
        raise ValueError("full heterogeneous summary requires one real assignment for every layer")

    metadata = [key_values((run / f"run_metadata_party{party}.txt").read_text(
        encoding="utf-8")) for party in (1, 2)]
    # Older immutable binaries record the packing mode in the launcher
    # provenance, but not in run_metadata_party*.txt.  Use that authoritative
    # per-run selection record as a compatibility fallback; do not infer it
    # from the head assignment.
    selection = key_values((run / "gpu_selection.txt").read_text(
        encoding="utf-8"))
    selected_packing = selection.get("attention_packing")
    if selected_packing not in {"packed", "unpacked"}:
        raise ValueError("run provenance lacks attention_packing")
    for row in metadata:
        row.setdefault("attention_packing", selected_packing)
    invariant_metadata = (
        "protocol", "version", "head_plan", "model", "architecture",
        "logical_tokens", "physical_tokens", "layers",
        "executed_transformer_blocks", "partial_run",
        "selective_conversion", "blb_fast_attention_packing",
        "heterogeneous_head_grouping",
        "attention_packing",
        "native_ckks_linear",
        "client_decrypt_reencrypt", "mpc_linear_surrogate", "backend",
        "prediction",
    )
    if any(metadata[0].get(key) != metadata[1].get(key)
           for key in invariant_metadata):
        raise ValueError("party run metadata disagree")
    meta = metadata[0]
    version = meta.get("version", "").upper()
    if (version not in {"H", "HA", "HP", "HAP", "J", "BP"} or meta.get("partial_run") != "0" or
            meta.get("executed_transformer_blocks") != "12" or
            meta.get("selective_conversion") != "1" or
            meta.get("attention_packing") not in {"packed", "unpacked"} or
            meta.get("blb_fast_attention_packing") not in {"0", "1"} or
            meta.get("heterogeneous_head_grouping") !=
                ("1" if meta.get("attention_packing") == "packed" else "0") or
            meta.get("client_decrypt_reencrypt") != "0" or
            meta.get("mpc_linear_surrogate") != "0"):
        raise ValueError("run metadata do not describe a complete real heterogeneous run")

    totals = [metric_table(run / f"execution_totals_party{party}.csv")
              for party in (1, 2)]
    secure_s = max(row["secure_end_to_end_s"] for row in totals)
    setup_s = max(row["setup_offline_s"] for row in totals)
    setup_plus_secure_s = max(row["setup_plus_secure_s"] for row in totals)
    predictions = [int((run / f"prediction_party{party}.txt").read_text(
        encoding="utf-8").split()[-1]) for party in (1, 2)]
    if predictions[0] != predictions[1]:
        raise ValueError("party predictions disagree")

    head_counts = {
        "softmax": sum(int(row["softmax_heads"]) for row in direct_rows),
        "relu": sum(int(row["relu_heads"]) for row in direct_rows),
        "linear": sum(int(row["linear_heads"]) for row in direct_rows),
    }
    output = {
        "format": "OURS_VIT_HETERO_FULL_RUN_SUMMARY_V2",
        "structural_status": "PASS",
        "performance_status": "MEASURED_REQUIRES_PAIRED_COMPARISON",
        "numerical_accuracy_status": "NOT_EVALUATED",
        "version": version,
        "attention_packing": meta["attention_packing"],
        "network": "1Gbps/0.3ms_RTT",
        "secure_end_to_end_s": secure_s,
        "setup_offline_s_max_party": setup_s,
        "setup_plus_secure_s_max_party": setup_plus_secure_s,
        "transformer_blocks_wall_s": max(
            row["transformer_blocks_wall_s"] for row in totals),
        "attention_assignment_s_sum": sum(
            float(row["latency_ms"]) for row in direct_rows) / 1000.0,
        "he_instrumented_s": max(row["he_instrumented_s"] for row in totals),
        "conversion_instrumented_s": max(
            row["conversion_instrumented_s"] for row in totals),
        "mpc_nonlinear_bucket_s": max(
            row["mpc_nonlinear_bucket_s"] for row in totals),
        "initialization_communication_bytes_alice_plus_bob": sum(
            int(row["initialization_communication_bytes"]) for row in totals),
        "online_communication_bytes_alice_plus_bob": sum(
            int(row["online_communication_bytes"]) for row in totals),
        "total_communication_bytes_alice_plus_bob": sum(
            int(row["total_communication_bytes"]) for row in totals),
        "online_rounds_max_party": integer_metric(totals, "online_communication_rounds"),
        "softmax_heads_total": head_counts["softmax"],
        "relu_heads_total": head_counts["relu"],
        "linear_heads_total": head_counts["linear"],
        "mpc_group_calls": sum(int(row["mpc_group_calls"]) for row in direct_rows),
        "native_linear_groups": sum(
            int(row["native_linear_groups"]) for row in direct_rows),
        "ckks_to_mpc": integer_metric(totals, "ckks_to_mpc"),
        "mpc_to_ckks": integer_metric(totals, "mpc_to_ckks"),
        "conversion_ciphertexts": integer_metric(totals, "conversion_ciphertexts"),
        "he_ct_pt_multiply": integer_metric(totals, "he_ct_pt_multiply"),
        "he_ct_ct_multiply": integer_metric(totals, "he_ct_ct_multiply"),
        "he_rotations": integer_metric(totals, "he_rotations"),
        "he_rescales": integer_metric(totals, "he_rescales"),
        "prediction_party_agreement": 1,
        "prediction": predictions[0],
        "accuracy_claimed": 0,
        "source_run": str(run),
    }

    csv_path = run / "ours_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        writer.writerows(output.items())

    print(f"OURS_HETERO_FULL_SUMMARY status=PASS version={version} "
          f"performance={output['performance_status']} secure_s={secure_s:.6f}")
    print(f"OURS_HETERO_FULL_SUMMARY_CSV={csv_path}")


if __name__ == "__main__":
    main()
