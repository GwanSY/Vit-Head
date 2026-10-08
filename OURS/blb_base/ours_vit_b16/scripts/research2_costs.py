#!/usr/bin/env python3
"""Resume real cost measurements with the shared Research2 configuration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

from collect_attention_assignment_costs import OUTPUT_COLUMNS, key_values, measured_rows
from make_composition_probe_plan import make_plan, parse_counts, write_plan
from run_research2 import ROOT, load_config, secure_environment

ANCHORS = ("12S0R0L", "10S0R2L", "8S0R4L", "8S4R0L",
           "6S4R2L", "4S8R0L", "4S4R4L")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_gpu_available(config: dict) -> None:
    secure = config["secure"]
    gpu = ",".join(map(str, secure["gpu_pair"] or [secure["gpu_id"]]))
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=utilization.gpu,memory.free",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    devices = [list(map(int, line.split(","))) for line in result.stdout.strip().splitlines()]
    utilization, free_mib = max(x[0] for x in devices), min(x[1] for x in devices)
    if utilization > secure["max_gpu_utilization_percent"] or free_mib < secure["min_free_memory_mib"]:
        print(f"GPU {gpu} busy: utilization={utilization}%, free_memory={free_mib} MiB; "
              f"requires utilization<={secure['max_gpu_utilization_percent']}% and "
              f"free_memory>={secure['min_free_memory_mib']} MiB. "
              "No new secure run started; rerun --stage costs when the configured GPU is available.",
              flush=True)
        raise SystemExit(75)


def prepare(config: dict) -> list[dict]:
    paths = config["paths"]
    allocator = Path(paths["allocator_costs"])
    supplemental = Path(paths["supplemental_costs"])
    if allocator == supplemental:
        raise ValueError("allocator_costs and supplemental_costs must be separate: held-out costs cannot enter training")
    probes = read_rows(Path(paths["probe_manifest"]))
    # Reuse the calibrated Linear coefficients of the frozen public plans.
    alphas = {}
    for probe in probes:
        with Path(probe["plan"]).open() as handle:
            next(handle)
            next(handle)
            for row in csv.DictReader(handle):
                if row["protocol"] == "he_linear":
                    key = int(row["layer"]), int(row["original_head"])
                    value = float(row["alpha"])
                    if key in alphas and alphas[key] != value:
                        raise ValueError(f"conflicting frozen Linear alpha: {key}")
                    alphas[key] = value
    jobs = []
    for probe in probes:
        jobs.append(dict(name=probe["split"] + "_" + probe["composition"],
                         plan=probe["plan"], version="J" if config["secure"]["version"] == "J" else "H",
                         blocks=1, scope="attention", output=str(supplemental)))
    plan_dir = Path(paths["output_dir"]) / "costs/plans"
    for composition in ANCHORS:
        counts = parse_counts(composition)
        plan = plan_dir / (composition + ".txt")
        write_plan(plan, make_plan(counts, alphas))
        jobs.append(dict(name="anchor_" + composition, plan=str(plan),
                         version="J" if config["secure"]["version"] == "J" else "H",
                         blocks=12, scope="full", output=str(allocator)))
    binary_hash = digest(Path(config["secure"]["binary_override"]))
    for job in jobs:
        job["plan_sha256"] = digest(Path(job["plan"]))
        rows = read_rows(Path(job["output"]))
        matching = [row for row in rows if row["plan_sha256"] == job["plan_sha256"]]
        for row in rows:
            selection = key_values((Path(row["source_run"]) / "gpu_selection.txt").read_text())
            pair = config["secure"]["gpu_pair"] or [config["secure"]["gpu_id"]] * 2
            if selection.get("physical_gpus") != ",".join(map(str, pair)):
                raise ValueError("existing costs were measured on a different GPU; use separate cost CSV paths for a new device")
            if (row["version"] != job["version"] or row["binary_sha256"] != binary_hash or
                    row["gpu_topology"] != ("shared_physical_gpu" if pair[0] == pair[1] else "separate_physical_gpus") or
                    str(row["blb_fast_attention_packing"]) != "0"):
                raise ValueError(f"cost provenance differs from current secure configuration: {job['output']}")
        job["complete"] = {int(row["layer"]) for row in matching} == set(range(job["blocks"]))
    report = Path(paths["output_dir"]) / "costs/jobs.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(jobs, indent=2) + "\n")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--version", choices=("B", "H", "HA", "HP", "HAP", "J", "BP"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.version:
        config["secure"]["version"] = args.version
    if config["secure"]["version"] not in ("HA", "J"):
        raise ValueError("cost collection is only for HA (unpacked) or J (packed)")
    if config["secure"]["backend"] != "gpu" or config["secure"]["network_mode"] != "netem":
        raise ValueError("the current cost collector requires backend=gpu and network_mode=netem; CPU is supported by --stage secure")
    jobs = prepare(config)
    pending = [job for job in jobs if not job["complete"]]
    print(f"Cost measurements: {len(jobs) - len(pending)} reusable, {len(pending)} missing", flush=True)
    if args.prepare_only:
        return
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("OURS_VIT_", "BLB_VIT_"))}
    environment.update(secure_environment(config))
    environment.update(CUDA_VISIBLE_DEVICES=str(config["secure"]["gpu_id"]), PYTHONUNBUFFERED="1")
    for job in pending:
        parent = Path(config["paths"]["output_dir"]) / "costs" / job["name"]
        rows = None
        # Recover a completed measurement even if the caller was interrupted
        # before its atomic CSV write. Failed attempts stay available in place.
        for previous in sorted(parent.glob("attempt_*")):
            try:
                candidate = measured_rows(previous)
                selection = key_values((previous / "gpu_selection.txt").read_text())
                pair = config["secure"]["gpu_pair"] or [config["secure"]["gpu_id"]] * 2
                if (candidate and selection.get("physical_gpus") == ",".join(map(str, pair)) and
                        {int(row["layer"]) for row in candidate} == set(range(job["blocks"])) and
                        all(row["plan_sha256"] == job["plan_sha256"] and
                        row["binary_sha256"] == digest(Path(config["secure"]["binary_override"]))
                        and row["version"] == job["version"]
                        and str(row["blb_fast_attention_packing"]) == "0"
                        for row in candidate)):
                    rows = candidate
                    break
            except (OSError, ValueError):
                continue
        if rows is None:
            # Check before the host oracle and output-directory creation.
            # The container launcher checks again immediately before launch.
            require_gpu_available(config)
            index = 1
            while (parent / f"attempt_{index:03d}").exists():
                index += 1
            run = parent / f"attempt_{index:03d}"
            command = ["bash", str(ROOT / "scripts/run_ours_container.sh"),
                       job["version"], job["plan"], os.path.relpath(run, ROOT),
                       str(job["blocks"]), job["scope"]]
            print(f"Cost measurement: {job['name']} -> {run}", flush=True)
            result = subprocess.run(command, cwd=ROOT, env=environment)
            if result.returncode:
                print(f"Cost measurement stopped: {job['name']}, exit={result.returncode}; "
                      f"details={run}", flush=True)
                raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)
            rows = measured_rows(run)
        if {int(row["layer"]) for row in rows} != set(range(job["blocks"])):
            raise ValueError(f"incomplete measured layers: {job['name']}")
        output = Path(job["output"])
        existing = [row for row in read_rows(output) if row["plan_sha256"] != job["plan_sha256"]]
        write_rows(output, existing + rows)
        print(f"Saved {job['name']}: {len(rows)} measured layers", flush=True)
    prepare(config)
    print("Cost measurements collected; run --stage offline to validate the cost model and select heads", flush=True)


if __name__ == "__main__":
    main()
