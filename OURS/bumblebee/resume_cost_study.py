"""Resume a stopped follow-up while preserving an interrupted round as evidence."""
import argparse
import fcntl
import json
import time
from pathlib import Path
from experiment import case_config, save_rows
from runtime import ROOT, digest, write
from secure_cost_study import checked_run


def recover_execution(folder):
    import numpy as np
    from mixed import dependency_depth, load_events
    m=json.loads((folder / "run.json").read_text())
    if len(m.get("measurements",[]))!=1 or not m["measurements"][0]["numerical_pass"]:
        raise RuntimeError(f"Interrupted execution did not save numerical verification: {folder}")
    parties=[load_events(folder / "log" / f"native-p{rank}-e1.tsv") for rank in (0,1)]
    dependency_depth(parties)
    profiles=[]
    for rank,rows in enumerate(parties):
        ops={}
        for r in rows:
            if r["kind"] in ("HLO","HAL"):
                op=ops.setdefault(r["name"],dict(calls=0,seconds=0.,sent_mib=0.,recv_mib=0.))
                op["calls"]+=1;op["seconds"]+=(r["end_ns"]-r["start_ns"])/1e9
                op["sent_mib"]+=r["bytes"]/2**20;op["recv_mib"]+=r["recv_bytes"]/2**20
        for name,count in dict(f_mmul=73,f_batch_mmul=24,f_tensordot=1,f_gelu=12,f_rsqrt=25).items():
            if ops[name]["calls"]!=count: raise RuntimeError("Recovered graph counters differ")
        sends=[r for r in rows if r["kind"]=="send"];recvs=[r for r in rows if r["kind"]=="recv"]
        sent=sum(r["bytes"] for r in sends);recv=sum(r["bytes"] for r in recvs)
        peer=parties[1-rank]
        if sent!=sum(r["bytes"] for r in peer if r["kind"]=="recv"):
            raise RuntimeError("Recovered cross-party payload does not close")
        duration=(rows[0]["end_ns"]-rows[0]["start_ns"])/1e9
        residual=duration-sum(o["seconds"] for n,o in ops.items() if n.startswith("pphlo."))
        if residual<-.02: raise RuntimeError("Recovered HLO intervals overlap native time")
        profiles.append(dict(seconds=duration,sent_mib=sent/2**20,recv_mib=recv/2**20,
                             unclassified_runtime_s=residual,operations=ops))
    vectors=np.load(folder / "logits.npz");ref=vectors["reference"];actual=vectors["b1"]
    d=actual.astype(np.float64)-ref.astype(np.float64)
    relative=float(np.linalg.norm(d)/np.linalg.norm(ref));maximum=float(np.max(np.abs(d)))
    gate=m["config"]["gate"]
    if not np.isfinite(ref).all() or not np.isfinite(actual).all() or relative>gate["relative_l2_max"] or maximum>gate["max_abs_max"] or int(ref.argmax())!=int(actual.argmax()):
        raise RuntimeError("Recovered logits do not pass unchanged numerical thresholds")
    saved=m["measurements"][0]
    if abs(saved["relative_l2"]-relative)>1e-12 or abs(saved["max_abs"]-maximum)>1e-12:
        raise RuntimeError("Saved numerical evidence differs from original logits")
    measurement=dict(m["measurements"][0],online_s=max(p["seconds"] for p in profiles),
                     total_gib=sum(p["sent_mib"] for p in profiles)/1024)
    evidence=dict(status="completed_execution_recovered",measurement=measurement,profiles=profiles,
        proof="Original logits and direct native interval/payload traces; HLO and graph counters close on both parties",
        limitation="Host orchestration/stdout finalization was interrupted; original run.json is preserved; excluded from primary paired rounds")
    path=folder / "recovered_evidence.json"
    if path.exists():
        if json.loads(path.read_text())!=evidence:raise RuntimeError(f"Recovered evidence changed: {folder}")
    else:write(path,evidence)
    return measurement


def recover_supplement(out,recovery):
    rows=[]
    for relative in recovery["supplementary_folders"]:
        p=out / relative;meta=json.loads((p / "run.json").read_text())
        r=(meta["measurements"][0] if meta["status"] in ("complete","matched","outside_gate") else recover_execution(p))
        if not r["numerical_pass"]:raise RuntimeError(f"Supplementary numerical validation failed: {p}")
        config=meta["config"]
        rows.append(dict(case=config["plan_name"],round=config["study_round"],seconds=r["online_s"],payload_mib=r["total_gib"]*1024,
            numerical_pass=r["numerical_pass"],relative_l2=r["relative_l2"],max_abs=r["max_abs"],
            reason="Interrupted round, retained as unpaired supplementary evidence",metadata=str(p / "run.json")))
    save_rows(out / "unpaired_recovered_samples.csv",rows)
    return len(rows)


def prepare_recovery(out,restart_round=None):
    frozen=json.loads((out / "design.json").read_text());path=out / "recovery.json"
    if path.exists():
        recovery=json.loads(path.read_text())
        if recovery["original_design_sha256"]!=digest(out / "design.json"):
            raise RuntimeError("Original frozen design changed")
    else:
        recovery=dict(reason="Session interruption stopped host orchestration; a long gap invalidated continuity of original round 2",
            original_design_sha256=digest(out / "design.json"),
            primary_full_schedule=[dict(c,folder=f"r2b-{c['case']}") if c["round"]==2 else c for c in frozen["full_schedule"]],
            supplementary_folders=[f"full/log/r2-{n}" for n in ("E2","E1","E0")],
            supplemental_verified_calls=3,heads_and_order_unchanged=True,created_at=time.time())
    if restart_round is not None:
        if restart_round not in range(1,7):raise ValueError("Round must be between 1 and 6")
        if any(x["round"]==restart_round for x in recovery.get("interruptions",[])):
            raise RuntimeError("This round already has a recorded restart; resume without --restart-round")
        previous=[c for c in recovery["primary_full_schedule"] if c["round"]==restart_round]
        replacements={c["folder"]:dict(c,folder=f"r{restart_round}b-{c['case']}") for c in previous}
        for c in previous:
            relative=f"full/log/{c['folder']}";p=out / relative / "run.json"
            if p.exists() and json.loads(p.read_text()).get("measurements"):
                if relative not in recovery["supplementary_folders"]:recovery["supplementary_folders"].append(relative)
        if any((out / "full/log" / c["folder"]).exists() for c in replacements.values()):
            raise RuntimeError("Restart targets must be new folders")
        recovery["primary_full_schedule"]=[replacements.get(c["folder"],c) for c in recovery["primary_full_schedule"]]
        recovery.setdefault("interruptions",[]).append(dict(round=restart_round,created_at=time.time(),
            previous_folders=[c["folder"] for c in previous],restarted_folders=[c["folder"] for c in replacements.values()],
            reason="Host orchestration ended during session switch; completed traces/logits retained; whole paired round restarted in its original order"))
    recovery["supplemental_verified_calls"]=recover_supplement(out,recovery)
    write(path,recovery)
    return recovery


def run(restart_round=None,prepare_only=False):
    out=ROOT / "out/12";frozen=json.loads((out / "design.json").read_text())
    recovery=prepare_recovery(out,restart_round)
    if prepare_only:
        print(f"Recovery prepared; {recovery['supplemental_verified_calls']} verified supplementary calls",flush=True)
        return
    schedule=recovery["primary_full_schedule"]
    native=ROOT / "build/stage2.so";progress=json.loads((out / "run.json").read_text())
    progress.update(status="running",phase="full",supplemental_verified_full_calls=recovery["supplemental_verified_calls"],
        recovery="Interrupted paired rounds rerun in full; original calls preserved")
    write(out / "run.json",progress)
    for i,c in enumerate(schedule,1):
        name=c["case"];plan=frozen["plans"][name]
        chosen=None if name=="E0" else dict(heads=plan["heads"],alpha=plan["alpha"])
        config=case_config("vit","cpu",study="mixed",repeats=1,head_plan=chosen,plan_name=name,
            layout="joint",native_library=str(native),trace_intervals=True,study_round=c["round"],measured_calls=1)
        checked_run(out / "full",c["folder"],config,1)
        progress.update(full_completed=i,latest_case=c["folder"],updated_at=time.time())
        write(out / "run.json",progress)
        print(f"Primary full ViT: {i}/{len(schedule)} complete ({c['folder']})",flush=True)
    progress.update(status="execution_complete",updated_at=time.time());write(out / "run.json",progress)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart-round",type=int)
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    with (ROOT / "out/12/.execution.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.restart_round,args.prepare_only)
