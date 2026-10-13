"""Frozen, resumable two-party follow-up: complete Blocks and complete ViT.

This driver only selects existing head/layout implementations. It never changes
weights, protocol arithmetic, native packing or numerical acceptance thresholds.
"""
import argparse
import json
import random
import time
from pathlib import Path

from experiment import case_config, run_case
from runtime import ROOT, digest, write


SOURCE_NAMES = ("run.py", "runtime.py", "model.py", "heads.py", "experiment.py",
                "block.py", "mixed.py", "config.json", "secure_cost_study.py")


def design(data):
    alpha = json.loads((data / "alpha.json").read_text())["alpha"]
    blocks = [dict(case=name, types=types, layout=layout, organization=organization)
              for name, types, layout, organization in (
        ("S10R2", "S"*10+"R"*2, "joint", "gather"),
        ("S8R4", "S"*8+"R"*4, "joint", "gather"),
        ("S10L2", "S"*10+"L"*2, "joint", "gather"),
        ("S10R1L1", "S"*10+"R"+"L", "joint", "gather"),
        ("S12", "S"*12, "joint", "native"),
        ("SR6", "S"*6+"R"*6, "joint", "gather"),
        ("SR6-slices", "S"*6+"R"*6, "joint", "slices"),
        ("SR6-split", "S"*6+"R"*6, "split", "gather"))]
    random.Random(20261013).shuffle(blocks)
    layers = {"E0": [], "E1": [0, 6], "E2": [0, 3, 6, 9], "E3": list(range(12))}
    per_layer = {"E1": "R"*12, "E2": "S"*6+"R"*6, "E3": "S"*10+"R"*2}
    plans = {}
    for name, ids in layers.items():
        heads = [list(per_layer[name] if layer in ids else "S"*12) for layer in range(12)]
        r = sum(h.count("R") for h in heads)
        assert r == (0 if name == "E0" else 24)
        plans[name] = dict(heads=heads, alpha=alpha, relu_heads=r, changed_layers=ids,
                           extra_type_groups=sum(len(set(h))-1 for h in heads),
                           normalization_groups=sum(int("S" in h)+int("R" in h) for h in heads))
    rng = random.Random(20261013)
    schedule = []
    for repeat in range(1, 7):
        order = list(plans)
        rng.shuffle(order)
        schedule.extend(dict(round=repeat, case=name, folder=f"r{repeat}-{name}") for name in order)
    return dict(blocks=blocks, plans=plans, full_schedule=schedule,
                block_measured_calls=5, full_fresh_repeats=6,
                block_warmup_calls=1, full_warmup_calls=0,
                layer_indexing="zero based; nested evenly spaced layer sets; selected before measurements",
                timing="max(P0 native execution, P1 native execution); complete call; both roles separately profiled",
                purpose="diagnosis, not accuracy constrained allocation or a new packing protocol")


def checked_run(out, name, config, expected):
    folder = out / "log" / name
    cached = folder / "run.json"
    if cached.exists():
        meta = json.loads(cached.read_text())
        if meta["status"] in ("complete", "outside_gate", "matched"):
            if meta["config"] != config or meta["library"]["sha256"] != digest(Path(config["native_library"])):
                raise RuntimeError(f"Cached evidence changed: {folder}")
        else:
            run_case(out, name, config)
    else:
        run_case(out, name, config)
    meta = json.loads(cached.read_text())
    if len(meta["measurements"]) != expected or len(meta["profiles"]) != expected:
        raise RuntimeError(f"Incomplete execution: {folder}")
    if not all(r["numerical_pass"] for r in meta["measurements"]):
        raise RuntimeError(f"Numerical validation failed: {folder}")
    for i, pair in enumerate(meta["profiles"]):
        for rank, part in enumerate(pair):
            execution = i+2 if config["task"] == "block" else 1
            trace = folder / "log" / f"native-p{rank}-e{execution}.tsv"
            if not trace.exists():
                raise RuntimeError(f"Missing trace: {trace}")
    return meta


def run(out, phase):
    out.mkdir(parents=True, exist_ok=True)
    data = ROOT / "out/07"
    native = ROOT / "build/stage2.so"
    frozen = design(data)
    provenance = dict(sources={n: digest(ROOT / n) for n in SOURCE_NAMES},
                      native_sha256=digest(native),
                      checkpoint_sha256=digest(ROOT / "../../BOLT/checkpoints/vit_b16_hf/model.safetensors"),
                      alpha_sha256=digest(data / "alpha.json"),
                      historical_block_summary_sha256=digest(ROOT / "out/11/block_summary.csv"),
                      historical_micro_cost_sha256=digest(ROOT / "out/10/cost.csv"))
    for name, value in (("design.json", frozen), ("provenance.json", provenance)):
        path = out / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise RuntimeError(f"Frozen experiment changed: {path}")
        write(path, value)
    progress = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else dict(started_at=time.time())
    progress.update(status="running", phase=phase, blocks_total=len(frozen["blocks"]), full_total=len(frozen["full_schedule"]))
    write(out / "run.json", progress)
    if phase in ("all", "blocks"):
        for i, c in enumerate(frozen["blocks"], 1):
            cfg = case_config("block", "cpu", study="mixed", types=list(c["types"]), alpha=frozen["plans"]["E0"]["alpha"][0],
                              head_count=12, layout=c["layout"], organization=c["organization"],
                              native_library=str(native), trace_intervals=True, repeats=1, measured_calls=5, plan_name=c["case"])
            checked_run(out / "blocks", c["case"], cfg, 5)
            progress.update(phase="blocks", blocks_completed=i, latest_case=c["case"], updated_at=time.time())
            write(out / "run.json", progress)
            print(f"Block evidence: {i}/{len(frozen['blocks'])} complete ({c['case']})", flush=True)
    if phase in ("all", "full"):
        for i, c in enumerate(frozen["full_schedule"], 1):
            name = c["case"]
            plan = frozen["plans"][name]
            head_plan = None if name == "E0" else dict(heads=plan["heads"], alpha=plan["alpha"])
            cfg = case_config("vit", "cpu", study="mixed", repeats=1, head_plan=head_plan, plan_name=name,
                              layout="joint", native_library=str(native), trace_intervals=True,
                              study_round=c["round"], measured_calls=1)
            checked_run(out / "full", c["folder"], cfg, 1)
            progress.update(phase="full", full_completed=i, latest_case=c["folder"], updated_at=time.time())
            write(out / "run.json", progress)
            print(f"Full ViT evidence: {i}/{len(frozen['full_schedule'])} complete ({c['folder']})", flush=True)
    progress.update(status="execution_complete" if phase == "all" else phase+"_complete", updated_at=time.time())
    write(out / "run.json", progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out/12")
    parser.add_argument("--phase", choices=("all", "blocks", "full"), default="all")
    args = parser.parse_args()
    run(args.out.resolve(), args.phase)
