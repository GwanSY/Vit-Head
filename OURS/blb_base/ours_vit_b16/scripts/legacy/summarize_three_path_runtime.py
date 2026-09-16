#!/usr/bin/env python3
"""Summarize the directly measured Softmax/ReLU/Linear group curve.

Each curve point is a real one-block A4 run.  The direct columns select the
requested path marker (rather than the reference path in the same run); the
full-block columns retain the complete secure block runtime for context.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
from pathlib import Path


PATHS = {
    "softmax": ("Softmax", "softmax_mpc", "head_count", "S", "R"),
    "relu": ("ReLU", "relu_norm_mpc", "head_count", "S", "R"),
    "linear": ("Linear", "he_linear", "active_heads", "S", "L"),
}
GROUP_SIZES = (1, 2, 4, 6, 8, 12)


def read_kv(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def read_metrics(path: Path):
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["metric"]] = float(row["value"])
    return result


def marker_values(path: Path, prefix: str):
    result = []
    for log in (path / "alice.log", path / "bob.log"):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith(prefix):
                continue
            values = {}
            for token in line.split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    values[key] = value
            result.append(values)
    return result


def successful(run: Path):
    status = run / "status.txt"
    text = status.read_text(encoding="utf-8")
    if "alice_rc=0" not in text or "bob_rc=0" not in text:
        raise ValueError(f"run did not pass: {run}")
    env = read_kv(run / "environment.txt")
    selection = read_kv(run / "gpu_selection.txt")
    if env.get("backend") != "gpu" or env.get("network") != "1Gbps/0.3ms_RTT":
        raise ValueError(f"run is not the shaped GPU cohort: {run}")
    if selection.get("party_topology") != "separate_physical_gpus":
        raise ValueError(f"run is not a two-physical-GPU run: {run}")
    return env, selection


def make_row(root: Path, short_path: str, group: int):
    label, protocol, count_key, base_symbol, active_symbol = PATHS[short_path]
    run = root / f"{short_path}_g{group}"
    env, selection = successful(run)
    if short_path == "linear":
        markers = marker_values(run, "OURS_A4_HE_PATH_PROFILE ")
        selected = [m for m in markers
                    if int(m.get("active_heads", "-1")) == group]
    else:
        markers = marker_values(run, "OURS_HETERO_PATH_PROFILE ")
        selected = [m for m in markers
                    if m.get("protocol") == protocol and
                    int(m.get("head_count", "-1")) == group]
    if not selected:
        raise ValueError(
            f"missing selected marker for {run}")
    direct_latency = max(float(m["latency_ms"]) for m in selected) / 1000.0
    direct_comm = sum(int(m.get("communication_bytes", "0"))
                      for m in selected)
    direct_rounds = max(int(m.get("rounds", "0")) for m in selected)
    direct_conversion = max(
        int(float(m.get("conversion_ct", "0"))) for m in selected)
    direct_rotations = max(
        int(float(m.get("he_rotations", "0"))) for m in selected)
    direct_muls = max(int(float(m.get("he_muls", "0"))) for m in selected)

    p1 = read_metrics(run / "execution_totals_party1.csv")
    p2 = read_metrics(run / "execution_totals_party2.csv")
    if short_path == "softmax":
        composition = f"{group}S+{12 - group}R"
    elif short_path == "relu":
        composition = f"{12 - group}S+{group}R"
    else:
        composition = f"{12 - group}S+{group}L"
    return {
        "path": label,
        "heads_group": group,
        "head_composition": composition,
        "direct_online_latency_s": direct_latency,
        "direct_communication_bytes_both": direct_comm,
        "direct_rounds": direct_rounds,
        "direct_conversion_ciphertexts": direct_conversion,
        "direct_he_rotations": direct_rotations,
        "direct_he_muls": direct_muls,
        "full_block_online_latency_s": max(
            p1["secure_end_to_end_s"], p2["secure_end_to_end_s"]),
        "full_block_total_latency_s": max(
            p1["setup_plus_secure_s"], p2["setup_plus_secure_s"]),
        "full_block_online_communication_bytes_both": (
            p1["online_communication_bytes"] +
            p2["online_communication_bytes"]),
        "full_block_total_communication_bytes_both": (
            p1["total_communication_bytes"] +
            p2["total_communication_bytes"]),
        "full_block_rounds": max(
            int(p1["online_communication_rounds"]),
            int(p2["online_communication_rounds"])),
        "full_block_conversion_ciphertexts": max(
            int(p1["conversion_ciphertexts"]),
            int(p2["conversion_ciphertexts"])),
        "network": env["network"],
        "gpu_topology": selection["party_topology"],
        "image_id": selection.get("image_id", ""),
        "binary_sha256": env.get("binary_sha256", ""),
        "model_sha256": env.get("model_sha256", ""),
        "input_sha256": env.get("input_sha256", ""),
        "source_run": str(run.resolve()),
    }


def validate(rows):
    if len(rows) != len(PATHS) * len(GROUP_SIZES):
        raise ValueError(f"expected {len(PATHS) * len(GROUP_SIZES)} rows, got {len(rows)}")
    keys = ("network", "gpu_topology", "image_id", "binary_sha256",
            "model_sha256", "input_sha256")
    for key in keys:
        if len({row[key] for row in rows}) != 1:
            raise ValueError(f"cohort mismatch in {key}")


def write_markdown(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Three head-path runtime curve\n\n")
        handle.write(
            "Direct columns are the requested path marker from each real "
            "one-block A4 run; full-block columns are the complete secure "
            "block for context. Same GPU/model/input/network cohort.\n\n"
        )
        handle.write("| path | group | composition | direct online s | direct comm (both) "
                     "| direct rounds | direct conversion cts | full block online s "
                     "| full block total s | full block rounds | full block conversion cts |\n")
        handle.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['path']} | {row['heads_group']} | {row['head_composition']} "
                f"| {row['direct_online_latency_s']:.3f} "
                f"| {row['direct_communication_bytes_both']:,} "
                f"| {row['direct_rounds']} | {row['direct_conversion_ciphertexts']} "
                f"| {row['full_block_online_latency_s']:.3f} "
                f"| {row['full_block_total_latency_s']:.3f} "
                f"| {row['full_block_rounds']} "
                f"| {row['full_block_conversion_ciphertexts']} |\n"
            )
        handle.write("\nProvenance:\n\n")
        handle.write(f"- image_id: `{rows[0]['image_id']}`\n")
        handle.write(f"- binary_sha256: `{rows[0]['binary_sha256']}`\n")
        handle.write(f"- model_sha256: `{rows[0]['model_sha256']}`\n")
        handle.write(f"- input_sha256: `{rows[0]['input_sha256']}`\n")
        handle.write("- communication columns sum both parties; latency uses the slower party.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = [make_row(args.root, short_path, group)
            for short_path in PATHS for group in GROUP_SIZES]
    validate(rows)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output_prefix.with_suffix(".csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.output_prefix.with_suffix(".json").write_text(
        __import__("json").dumps({"format": "OURS_VIT_THREE_PATH_RUNTIME_V1",
                                    "cohort_validation": "PASS",
                                    "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    write_markdown(args.output_prefix.with_suffix(".md"), rows)
    print(f"OURS_THREE_PATH_RUNTIME status=PASS rows={len(rows)} "
          f"csv={args.output_prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
