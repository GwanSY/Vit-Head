#!/usr/bin/env python3
"""Block-0 supplementary B: real GPU two-party runs through the existing runner."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import csv
import datetime
import hashlib
import json
import re
import statistics
import subprocess
import tarfile
import time
from pathlib import Path
import numpy as np
import vit_reference as vr

B = Path(__file__).resolve().parents[2]
OLD = B / "outputs/heterogeneous_block_optimized_20261010"
GROUPS = {"native_S": ("B", "plan_00.txt", False),
          "generic_S": ("H", "plan_00.txt", True),
          "hetero_R": ("H", "plan_01.txt", False)}
ORDERS = [("native_S", "generic_S", "hetero_R"),
          ("hetero_R", "generic_S", "native_S"),
          ("generic_S", "native_S", "hetero_R"),
          ("hetero_R", "native_S", "generic_S"),
          ("native_S", "hetero_R", "generic_S")]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def write(p, data):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(p)


def csvwrite(p, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def kv(line):
    return dict(re.findall(r"(\w+)=([^ ]+)", line))


def table(p):
    return list(csv.DictReader(Path(p).open()))


def metadata(p):
    return dict(l.split("=", 1) for l in Path(p).read_text().splitlines() if "=" in l)


def state(o, stage, **data):
    value = dict(stage=stage, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **data)
    write(o / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def head_plan(p):
    result = [[("softmax_mpc", 1.0) for _ in range(12)] for _ in range(12)]
    for r in csv.DictReader(p.read_text().splitlines()[2:]):
        result[int(r["layer"])][int(r["original_head"])] = (r["protocol"], float(r["alpha"]))
    return result


def prepare(o):
    model = (B / "checkpoints/vit_b16/bolt_model.txt").resolve()
    input_path = o / "secure_input.txt"
    state(o, "PREPARING_NUMERICAL_ORACLES")
    weights = vr.load_model(model)
    x = vr.load_input(input_path)
    # CPU references are exclusively a numerical comparator, never a performance result.
    for name, plan_file in (("S", "plan_00.txt"), ("R", "plan_01.txt")):
        y, _ = vr.Graph(weights, fixed=True, physical_tokens=256,
                        head_plan=head_plan(o / "plans" / plan_file),
                        block_limit=1, record_values=False).run(x)
        assert np.isfinite(y).all()
        target = o / "references" / (name + ".npy")
        if target.exists():
            assert np.array_equal(np.load(target), y), "Changed numerical oracle"
        else:
            np.save(target, y)
    source_files = sorted(p for p in (B / "ours_vit_b16/runtime").iterdir() if p.is_file())
    paths = source_files + [Path(__file__), B / "scripts/run_ours_container.sh",
        B / "scripts/run_ours_vit_b16.sh", B / "ours_vit_b16/scripts/vit_reference.py",
        B / "ours_vit_b16/scripts/verify_head_reorder.py", model,
        (B / "checkpoints/vit_b16/manifest.json").resolve(), input_path, o / "alpha.json",
        o / "build/ours_vit_b16_main", *sorted((o / "plans").glob("*")),
        *sorted((o / "references").glob("*")),
        B / "build/container-gpu/extern/phantom/lib/libPhantom.so",
        B / "build/container-gpu/extern/seal/lib/libseal-4.1.a",
        *sorted((B / "build/container-gpu/lib").glob("libSCI*.a"))]
    hashes = {str(p): sha(p) for p in paths}
    write(o / "frozen_hashes.json", hashes)
    with tarfile.open(o / "audit/source_executed.tar.gz", "w:gz") as t:
        for p in source_files:
            t.add(p, arcname=str(p.relative_to(B)))
    image = subprocess.check_output(["docker", "image", "inspect", "ours:restored-fast-20260903d",
                                    "--format", "{{.Id}}"], text=True).strip()
    git = subprocess.check_output(["git", "-C", str(B.parents[1]), "rev-parse", "HEAD"], text=True).strip()
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
                                   "--format=csv"], text=True)
    (o / "audit/gpus.txt").write_text(gpu)
    write(o / "run_manifest.json", dict(schema="stage23.supplement_b.v1",
        scope="Block 0 including initial LN through FFN2 residual/statistics; embedding, weight preparation, final LN/classifier excluded from Block time",
        numerical_scope="ordinary final logits of the one-block prefix; final LN/classifier outside Block timing",
        origin="new_execution", historical_measurements_reused=0, execution_git=git,
        source_archive_sha256=sha(o / "audit/source_executed.tar.gz"), hashes=hashes,
        docker_image=image, gpu_pair=[3, 4], network="existing isolated loopback netem 1Gbps/0.15ms one-way; effective conditions in each run",
        OMP_threads_per_party=64, MPC_workers_per_party=64, gpu_streams=8,
        security="SEAL tc128 mandatory; unchanged protocol and parameters",
        fixed_sample=json.loads((OLD / "secure_sample.json").read_text()),
        groups=GROUPS, formal_order=ORDERS, repeats_per_group=5,
        validation=dict(relative_L2_max=0.05, max_abs_max=0.5, top1_agreement=True,
                        party_output_agreement=True, nonfinite_max=0),
        timers="steady-clock intervals, separate parties; atomic HE API counters; original nested counters retained separately",
        conversion_bytes="core SCI send bytes and serialized ciphertext send bytes recorded separately, not summed across overlapping scopes",
        stop_after_experiment=True, new_packing=False, new_head_reorder=False,
        alpha_refit=False, training=False))
    state(o, "PREPARED")


def assert_frozen(o):
    for p, h in json.loads((o / "frozen_hashes.json").read_text()).items():
        if sha(p) != h:
            raise RuntimeError("FROZEN_ASSET_CHANGED " + p)


def vector(p):
    lines = p.read_text().splitlines()
    return np.asarray(lines[lines.index("values") + 1:lines.index("end")], dtype=float)


def validate(o, run, group):
    a = vector(run / "secure_logits_party1.txt")
    b = vector(run / "secure_logits_party2.txt")
    y = np.load(o / "references" / ("R.npy" if group == "hetero_R" else "S.npy")).reshape(-1)
    nonfinite = int((~np.isfinite(a)).sum() + (~np.isfinite(b)).sum() + (~np.isfinite(y)).sum())
    relative = float(np.linalg.norm(a - y) / max(float(np.linalg.norm(y)), 1e-30))
    maximum = float(np.max(np.abs(a - y)))
    predictions = [int(np.argmax(v)) for v in (a, b, y)]
    passed = nonfinite == 0 and relative <= 0.05 and maximum <= 0.5 and len(set(predictions)) == 1 and np.array_equal(a, b)
    result = dict(status="PASS" if passed else "FAIL", relative_L2=relative,
                  max_abs=maximum, predictions=predictions, nonfinite=nonfinite,
                  parties_equal=bool(np.array_equal(a, b)), reference_sha256=sha(o / "references" / ("R.npy" if group == "hetero_R" else "S.npy")))
    write(run / "validation.json", result)
    return result


def measured_partition(begin, end, events):
    # Partition the interval union; nested counter spans are never simply added.
    order = {"C2M": 0, "M2C": 1, "Layout": 2, "transport": 3, "CKKS": 4, "MPC": 5}
    spans = []
    for r in events:
        kind = r["kind"]
        category = {"C2M_CORE": "C2M", "M2C_CORE": "M2C", "CT_SEND": "transport",
                    "CT_RECV": "transport", "ENCRYPT_SEND": "transport"}.get(kind, kind)
        if category not in order:
            continue
        a, z = max(begin, int(r["begin_ns"])), min(end, int(r["end_ns"]))
        if a < z:
            spans.append((a, z, category))
    edges = sorted({begin, end, *(v for a, z, _ in spans for v in (a, z))})
    totals = dict.fromkeys(("CKKS", "MPC", "C2M", "M2C", "Layout", "Other"), 0)
    parts = []
    for a, z in zip(edges, edges[1:]):
        active = [k for x, y, k in spans if x <= a and z <= y]
        chosen = min(active, key=order.get) if active else "Other"
        category = "Other" if chosen == "transport" else chosen
        totals[category] += z - a
        parts.append(dict(begin_ns=a, end_ns=z, category=category, reason=chosen))
    assert sum(totals.values()) == end - begin
    assert all(x["end_ns"] == y["begin_ns"] for x, y in zip(parts, parts[1:]))
    return {k + "_s": v / 1e9 for k, v in totals.items()}, parts


def parse_party(run, party):
    lines = (run / ("alice.log" if party == 1 else "bob.log")).read_text().splitlines()
    events, phases = [], {}
    for line_no, line in enumerate(lines, 1):
        if line.startswith("E2E_PHASE "):
            r = kv(line); phases[r["stage"]] = r
        elif line.startswith(("FUSION_OBSERVE ", "EXECUTION_INTERVAL ")):
            r = kv(line); r["log_line"] = line_no; r["party"] = party; events.append(r)
    phase = phases["block_0"]
    a, z = int(phase["begin_ns"]), int(phase["end_ns"])
    block_events = [r for r in events if a <= int(r["begin_ns"]) and int(r["end_ns"]) <= z]
    times, parts = measured_partition(a, z, events)
    totals = {r["metric"]: float(r["value"]) for r in table(run / f"execution_totals_party{party}.csv")}
    profile = table(run / f"block_profile_party{party}.csv")[0]
    meta = metadata(run / f"run_metadata_party{party}.txt")
    nested = next(kv(l) for l in lines if l.startswith("OPTIMIZED_BLOCK_COUNTERS "))
    ops = next(kv(l) for l in lines if l.startswith("HE_API_COUNTS "))
    row = dict(party=party, block_wall_s=(z - a) / 1e9, **times,
        block_send_bytes=int(profile["block_online_comm_bytes"]),
        block_rounds=int(profile["communication_rounds"]),
        initialization_s=totals["setup_offline_s"],
        nested_HE_s=float(nested["he_s"]), nested_MPC_s=float(nested["mpc_nonlinear_s"]),
        nested_Layout_s=float(nested["layout_s"]),
        he_API_rotations=int(ops["rotations"]),
        he_API_ct_ct_multiply_including_square=int(ops["ct_ct_multiply_including_square"]),
        he_API_ct_pt_multiply=int(ops["ct_pt_multiply"]), he_API_conjugations=int(ops["conjugations"]),
        generic_executor_entered=any(l.startswith("OURS_GENERIC_EXECUTOR_ENTER ") for l in lines),
        original_softmax_fallback=any(l.startswith("OURS_HETERO_UNCHANGED_LAYER ") for l in lines))
    for direction in ("C2M", "M2C"):
        e = [r for r in block_events if r["kind"] == direction + "_CORE"]
        row[direction + "_batches"] = len(e)
        row[direction + "_ciphertexts"] = sum(int(r["count"]) for r in e)
        row[direction + "_core_send_bytes"] = sum(int(r["sent_bytes"]) for r in e)
    qkv = [r for r in block_events if r["kind"] == "C2M_CORE" and "layer_norm_part2_attention:" in r.get("call_path", "")]
    row.update(QKV_export_batches=len(qkv), QKV_export_ciphertexts=sum(int(r["count"]) for r in qkv),
               QKV_export_core_s=sum((int(r["end_ns"]) - int(r["begin_ns"])) / 1e9 for r in qkv),
               QKV_export_core_send_bytes=sum(int(r["sent_bytes"]) for r in qkv))
    assert meta["backend"] == "gpu" and meta["client_decrypt_reencrypt"] == "0" and meta["mpc_linear_surrogate"] == "0"
    # Existing profile categories provide a second independent communication closure.
    stage_rows = [r for r in table(run / f"block_online_profile_party{party}.csv") if not r["stage"].startswith("BLOCK_ONLINE_")]
    assert sum(int(r["send_bytes"]) for r in stage_rows) == row["block_send_bytes"]
    write(run / f"exclusive_intervals_party{party}.json", parts)
    csvwrite(run / f"conversion_events_party{party}.csv", [r for r in block_events if r["kind"] in ("C2M_CORE", "M2C_CORE")])
    write(run / f"observed_events_party{party}.json", events)
    return row


def run_all(o):
    done = {r["run_id"]: r for r in (json.loads(l) for l in (o / "measurements.jsonl").read_text().splitlines())} if (o / "measurements.jsonl").exists() else {}
    failed = {r["group"] for r in done.values() if r.get("status") != "PASS"}
    for repeat, order in enumerate(ORDERS, 1):
        for group in order:
            run_id = f"r{repeat:02d}_{group}"
            if run_id in done or group in failed:
                continue
            assert_frozen(o)
            run = o / "runs" / run_id
            assert not run.exists(), "Refuse overwrite " + str(run)
            version, plan, force = GROUPS[group]
            env = os.environ.copy()
            env.update(OURS_VIT_PHYSICAL_GPUS="3,4", OURS_VIT_MODEL=str((B / "checkpoints/vit_b16/bolt_model.txt").resolve()),
                       OURS_VIT_INPUT=str(o / "secure_input.txt"), OURS_VIT_FORCE_GENERIC_ALL_S=str(int(force)),
                       OURS_VIT_MATCH_LINEAR_HE_CONTROL="0", OURS_VIT_MPC_WORKERS="64", OURS_VIT_C2M_HEADROOM_BITS="24",
                       OURS_VIT_BINARY_OVERRIDE=str(o / "build/ours_vit_b16_main"), BLB_VIT_OMP_THREADS="64",
                       OURS_VIT_PORT="28300", BLB_VIT_TIMEOUT_SECONDS="1800")
            command = ["bash", str(B / "scripts/run_ours_container.sh"), version,
                       str(o / "plans" / plan), str(run.relative_to(B)), "1", "logits"]
            state(o, "RUNNING", group=group, repeat=repeat, completed=len(done), run_id=run_id)
            write(o / "raw" / (run_id + ".command.json"), dict(command=command, env={k: v for k, v in env.items() if k.startswith(("OURS_", "BLB_"))}))
            started = time.time()
            with (o / "raw" / (run_id + ".launcher.log")).open("w") as log:
                result = subprocess.run(command, cwd=B, env=env, stdout=log, stderr=subprocess.STDOUT)
            value = dict(run_id=run_id, group=group, repeat=repeat, returncode=result.returncode,
                         launcher_wall_s=time.time() - started, raw_log_dir=str(run), status="UNVERIFIED")
            if result.returncode == 0:
                v = validate(o, run, group)
                value.update(validation=v, status=v["status"])
                value["parties"] = [parse_party(run, p) for p in (1, 2)]
                if group == "generic_S":
                    assert all(r["generic_executor_entered"] and not r["original_softmax_fallback"] for r in value["parties"])
                value["block_communication_GB"] = sum(r["block_send_bytes"] for r in value["parties"]) / 1e9
            if value["status"] != "PASS":
                failed.add(group)
            write(o / "raw" / (run_id + ".result.json"), value)
            with (o / "measurements.jsonl").open("a") as f:
                f.write(json.dumps(value, ensure_ascii=False) + "\n")
            done[run_id] = value
            state(o, "RUN_COMPLETE", run_id=run_id, status=value["status"], completed=len(done), affected_groups_stopped=sorted(failed))
    state(o, "MEASUREMENTS_COMPLETE", runs=len(done), affected_groups_stopped=sorted(failed))


def summarize(o):
    measurements = [json.loads(l) for l in (o / "measurements.jsonl").read_text().splitlines()]
    flat = [dict(run_id=r["run_id"], group=r["group"], repeat=r["repeat"], validation_status=r["status"],
                 relative_L2=r["validation"]["relative_L2"], max_abs=r["validation"]["max_abs"],
                 block_communication_GB=r["block_communication_GB"], **p) for r in measurements if "parties" in r for p in r["parties"]]
    csvwrite(o / "per_run.csv", flat)
    rows = []
    for group in GROUPS:
        for party in (1, 2):
            rr = [r for r in flat if r["group"] == group and r["party"] == party and r["validation_status"] == "PASS"]
            for metric in [k for k, v in (rr[0].items() if rr else []) if isinstance(v, (float, int)) and not isinstance(v, bool) and k not in ("repeat", "party")]:
                values = [r[metric] for r in rr]
                rows.append(dict(group=group, party=party, metric=metric, n=len(values),
                                 mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else "",
                                 status="PASS" if len(values) == 5 else "UNVERIFIED"))
    csvwrite(o / "summary.csv", rows)
    state(o, "SUMMARY_SAVED", completed=len(measurements))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.prepare:
        prepare(args.output)
    if args.run:
        run_all(args.output)
    if args.summarize:
        summarize(args.output)


if __name__ == "__main__":
    main()
