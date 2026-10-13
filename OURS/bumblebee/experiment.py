"""Real 2PC attention costs and complete ViT comparisons with compact records."""
import csv
import json
import statistics
import time
from runtime import ROOT, baseline, write, digest


def attention_worker(out):
    import numpy as np
    import jax.numpy as jnp
    from spu import spu_pb2
    import spu.utils.distributed as ppd
    from heads import attention, mapping
    ppd.logger.setLevel(30)
    meta=json.loads((out / "run.json").read_text());config=meta["config"]
    inputs=np.load(config["qkv"])
    h=config.get("head_count",12)
    q,k,v=[inputs[name][:,:h] for name in ("q","k","v")]
    types=config["types"];alpha=config["alpha"][:h]
    if config.get("scope")=="map":
        z=np.asarray((jnp.asarray(q)/8) @ jnp.swapaxes(jnp.asarray(k),-1,-2))
        fn=lambda z:mapping(z,types[0],alpha)
        values=[z]
    else:
        if config.get("organization") == "slices":
            from mixed import sliced_attention
            fn=lambda q,k,v:sliced_attention(q,k,v,types,alpha,config["layout"])
        elif config.get("organization") == "native":
            import jax.nn as nn
            if set(types) != {"S"}:
                raise ValueError("Native attention control requires all Softmax heads")
            fn=lambda q,k,v:nn.softmax((q/8) @ jnp.swapaxes(k,-1,-2),axis=-1) @ v
        else:
            fn=lambda q,k,v:attention(q,k,v,types,alpha,config["layout"])
        values=[q,k,v]
    reference=np.asarray(fn(*[jnp.asarray(x) for x in values]))
    organization_error=0.
    if config.get("organization") in ("slices","native"):
        original=np.asarray(attention(*[jnp.asarray(x) for x in values],types,alpha,config["layout"]))
        organization_error=float(np.max(np.abs(original-reference)))
        if organization_error>1e-6:
            raise RuntimeError(f"Organization intervention changed plaintext arithmetic: {organization_error}")
    ppd.init(meta["spu"]["nodes"],meta["spu"]["devices"])
    values=[ppd.device("P1" if i==0 else "P2")(lambda x:x)(x) for i,x in enumerate(values)]
    device=ppd.device("SPU")
    values=[ppd.Device.move(x,device) for x in values]
    def storage(server,values):
        return [spu_pb2.ValueMetaProto.FromString(x.spu_share.meta).storage_type for x in values]
    storage_types=[client.run_return(storage,[x.refs[rank] for x in values])
                   for rank,client in enumerate(device.node_clients)]
    if any("ashr" not in kind.lower() for row in storage_types for kind in row):
        raise RuntimeError(f"Attention benchmark requires shared intermediates: {storage_types}")
    options=spu_pb2.CompilerOptions();options.enable_optimize_denominator_with_broadcast=True
    secure=ppd.device("SPU")(fn,copts=options)
    start=time.perf_counter();ppd.get(secure(*values));warmup=time.perf_counter()-start
    samples=config.get("measured_calls",1)
    for sample in range(samples):
        start=time.perf_counter();actual=np.asarray(ppd.get(secure(*values)));elapsed=time.perf_counter()-start
        delta=actual.astype(np.float64)-reference.astype(np.float64)
        relative=float(np.linalg.norm(delta)/np.linalg.norm(reference))
        maximum=float(np.max(np.abs(delta)));finite=bool(np.isfinite(actual).all())
        result=dict(repeat=(meta["active_repeat"]-1)*samples+sample+1,request_s=elapsed,warmup_request_s=warmup,
                    relative_l2=relative,max_abs=maximum,finite=finite,
                    numerical_pass=finite and relative<=.05 and maximum<=.5,input_storage_types=storage_types,
                    organization_reference_max_abs=organization_error)
        meta.setdefault("measurements",[]).append(result);write(out / "run.json",meta)


def case_config(task,backend,**values):
    config=json.loads((ROOT / "config.json").read_text())
    for key in ("upstream","model","image","spu_package","gpu_library","phantom"):
        config[key]=str((ROOT / config[key]).resolve())
    config.update(task=task,backend=backend,force_repeats=True,timeout_s=3600,**values)
    return config


def run_case(out,name,config):
    configs=out / "log";configs.mkdir(parents=True,exist_ok=True)
    path=configs / f"{name}.json";write(path,config)
    target=configs / name
    if target.exists():
        meta=json.loads((target / "run.json").read_text())
        if meta["status"] not in ("failed","interrupted","timeout"):
            raise RuntimeError(f"Existing completed evidence: {target}")
        index=1
        while (configs / f"{name}-err{index}").exists():index+=1
        target.rename(configs / f"{name}-err{index}")
    baseline(path,target)
    meta=json.loads((target / "run.json").read_text())
    rows=meta["measurements"]
    return dict(case=name,backend=config["backend"],layout=config.get("layout","joint"),
                repeats=len(rows),online_s=statistics.median(r["online_s"] for r in rows),
                request_s=statistics.median(r["request_s"] for r in rows),
                total_gib=statistics.median(r["total_gib"] for r in rows),
                relative_l2=max(r["relative_l2"] for r in rows),max_abs=max(r["max_abs"] for r in rows),
                numerical_pass=all(r["numerical_pass"] for r in rows),
                native_sha256=meta["library"]["sha256"],metadata=str(target.relative_to(out) / "run.json"))


def save_rows(path,rows):
    temp=path.with_suffix(".tmp")
    with temp.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def costs(out,backend,native=None,data=None):
    accuracy=data or ROOT / "out/07"
    alpha=json.loads((accuracy / "alpha.json").read_text())["alpha"][0]
    qkv=accuracy / "attention.npz"
    if not qkv.exists(): raise RuntimeError("Capture real ViT Q/K/V before cost measurement")
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    path=out / ("costs.csv" if backend=="cpu" else "gpu.csv")
    if path.exists(): rows=list(csv.DictReader(path.open()))
    cases={"S":"S"*12,"R":"R"*12,"L":"L"*12,
           "R1":"S"*11+"R","R4":"S"*8+"R"*4,"R9":"S"*3+"R"*9,
           "M":"S"*6+"R"*3+"L"*3}
    for label,types in cases.items():
        for layout in (("joint","split") if label in ("M","R4") else ("joint",)):
            name=f"{backend}-{label}-{layout}"
            if any(r["case"]==name for r in rows):continue
            config=case_config("attention",backend,types=list(types),alpha=alpha,
                               qkv="/work/"+str(qkv.relative_to(ROOT.parents[1])),
                               qkv_sha256=digest(qkv),layout=layout,repeats=3,plan_name=label)
            if backend=="gpu":config.update(threads=8,cpus=["32-39","80-87"])
            if native:config["native_library"]=str(native)
            row=run_case(out,name,config)
            meta=json.loads((out / row["metadata"]).read_text())
            metrics=[pair[0].get("backend_metrics",{}) for pair in meta["profiles"]]
            fields=["HE_calls","conversion_calls"]
            if backend=="gpu":fields += ["H2D_calls","H2D_bytes","D2H_calls","D2H_bytes"]
            for field in fields:
                row[field]=statistics.median(float(m.get(field,0)) for m in metrics)
            row["batch_mmul_calls"]=statistics.median(p[0]["operations"]["f_batch_mmul"]["calls"] for p in meta["profiles"])
            rows.append(row);save_rows(path,rows)
    metadata=[json.loads((out / r["metadata"]).read_text()) for r in rows]
    if any(m["hashes"]["heads.py"]!=digest(ROOT / "heads.py") or
           m["config"]["qkv_sha256"]!=digest(qkv) or len(m["measurements"])!=3 or
           (native and m["library"]["sha256"]!=digest(native)) for m in metadata):
        raise RuntimeError("Cached costs differ from the requested heads, inputs, library or repeats; use a new output directory")
    shared=all("ashr" in kind.lower() for m in metadata for r in m["measurements"]
               for party in r["input_storage_types"] for kind in party)
    if len(rows)!=9 or not shared or any(str(r["numerical_pass"])!="True" for r in rows):
        raise RuntimeError("Incomplete or invalid shared-input cost evidence")
    info=json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
    for key in ("reason","note","invalidated_rows","costs_valid_for_full_vit"):info.pop(key,None)
    info.update(status="complete",backend=backend,cases=len(rows),repeats_per_case=3,
                cost_sha256=digest(path),qkv_source=digest(qkv),heads_source=digest(ROOT / "heads.py"),
                native_sha256=rows[0]["native_sha256"],numerical_pass=True,
                qkv_visibility="Arithmetic secret sharing; AShr metadata checked on both parties in every repeat",
                cost_scope="Warm attention groups only; candidate ranking, not whole-model latency",
                warmup="One complete call before each measured call; fresh nodes for each repeat")
    write(out / "run.json",info)
    print(f"Secure attention costs complete: {path}",flush=True)


def allocate(out,cost_path,budget=1.0,backend="cpu",data=None):
    import numpy as np
    if (out / "run.json").exists() and json.loads((out / "run.json").read_text())["status"]=="complete":
        raise RuntimeError(f"Existing completed allocation: {out}")
    accuracy=data or ROOT / "out/07"
    plans=json.loads((accuracy / "plans.json").read_text())
    scores=list(csv.DictReader((accuracy / "accuracy.csv").open()))
    cost_meta=json.loads((cost_path.parent / "run.json").read_text())
    if cost_meta["status"]!="complete" or cost_meta["cost_sha256"]!=digest(cost_path):
        raise RuntimeError("Complete the shared-input cost measurements before allocating")
    costs={r["case"].split('-')[1]:float(r["online_s"])
           for r in csv.DictReader(cost_path.open()) if r["backend"]==backend and r["layout"]=="joint"}
    xs=[0,1,4,9,12];ys=[costs[name] for name in ("S","R1","R4","R9","R")]
    rows=[]
    for score in scores:
        if float(score["loss_pp"])>budget:continue
        estimate=0.
        for types in plans[score["plan"]]["heads"]:
            s,r,l=[types.count(kind) for kind in "SRL"]
            if l==0:
                estimate+=float(np.interp(r,xs,ys))
            else:
                estimate+=(s*costs["S"]+r*costs["R"]+l*costs["L"])/12
                if len(set(types))>1:
                    estimate+=costs["M"]-(6*costs["S"]+3*costs["R"]+3*costs["L"])/12
        rows.append(dict(plan=score["plan"],top1=float(score["top1"]),loss_pp=float(score["loss_pp"]),
                         estimated_attention_s=estimate))
    winner=min(rows,key=lambda r:r["estimated_attention_s"])["plan"]
    out.mkdir(exist_ok=True)
    save_rows(out / "allocation.csv",rows)
    write(out / "plan.json",dict(name=winner,**plans[winner]))
    write(out / "run.json",dict(status="proposed",budget_pp=budget,backend=backend,
          cost_sha256=digest(cost_path),plans_sha256=digest(accuracy / "plans.json"),
          accuracy_sha256=digest(accuracy / "accuracy.csv"),
          accuracy_evidence_sha256=digest(accuracy / "predictions.npz"),
          selection_sha256=json.loads((accuracy / "run.json").read_text())["split_hashes"]["selection"],
          proposal=winner,scope="Finite calibrated candidate pool; selection-set empirical accuracy constraint",
          cost_model="Measured whole-head-group costs, interpolated by ReLU group size; three-way residual for Linear mixtures",
          limitation="Predicted attention cost ranks candidates only; complete inference time must be measured directly.",
          accuracy_constraint="Measured full-model plaintext Top-1 after head replacement; calibration proxy only proposes",
          holdout_evaluated=False))
    print(f"Allocated {winner} within {budget:g} pp; awaiting actual full ViT timing",flush=True)
    return winner


def full(out,plan_name,backend,native,repeats=3,layout="joint",data=None):
    data=data or ROOT / "out/07"
    plans=json.loads((data / "plans.json").read_text())
    name=f"{backend}-{plan_name}-{layout}"
    path=out / "results.csv"
    rows=list(csv.DictReader(path.open())) if path.exists() else []
    if any(r["case"]==name for r in rows):return
    config=case_config("vit",backend,repeats=repeats,head_plan=plans[plan_name] if plan_name!="S" else None,
                       plan_name=plan_name,layout=layout,native_library=str(native))
    row=run_case(out,name,config)
    meta=json.loads((out / row["metadata"]).read_text())
    row["batch_mmul_calls"]=statistics.median(p[0]["operations"]["f_batch_mmul"]["calls"] for p in meta["profiles"])
    row["mmul_calls"]=statistics.median(p[0]["operations"]["f_mmul"]["calls"] for p in meta["profiles"])
    for field in ("HE_calls","conversion_calls"):
        row[field]=statistics.median(float(p[0]["backend_metrics"][field]) for p in meta["profiles"])
    accuracy={r["plan"]:r for r in csv.DictReader((data / "accuracy.csv").open())}[plan_name]
    row.update(top1=float(accuracy["top1"]),loss_pp=float(accuracy["loss_pp"]))
    for previous in rows:
        if "mmul_calls" not in previous:
            raw=json.loads((out / previous["metadata"]).read_text())
            previous["mmul_calls"]=statistics.median(p[0]["operations"]["f_mmul"]["calls"] for p in raw["profiles"])
    rows.append(row);save_rows(path,rows)


def select(out,data=None):
    """Keep the fastest measured feasible plan, including the unmodified baseline."""
    data=data or ROOT / "out/07"
    meta=json.loads((out / "run.json").read_text())
    plans=json.loads((data / "plans.json").read_text())
    accuracy=json.loads((data / "run.json").read_text())
    scores={r["plan"]:r for r in csv.DictReader((data / "accuracy.csv").open())}
    if (digest(data / "plans.json")!=meta["plans_sha256"] or
        digest(data / "accuracy.csv")!=meta["accuracy_sha256"] or
        digest(data / "predictions.npz")!=meta["accuracy_evidence_sha256"]):
        raise RuntimeError("Accuracy evidence changed after candidate allocation")
    rows=[r for r in csv.DictReader((out / "results.csv").open()) if r["backend"]==meta["backend"]]
    baseline=next(r for r in rows if r["case"].split('-')[1]=="S")
    base_raw=json.loads((out / baseline["metadata"]).read_text())
    feasible=[]
    for row in rows:
        name=row["case"].split('-')[1]
        raw=json.loads((out / row["metadata"]).read_text())
        expected=plans[name] if name!="S" else None
        if (raw["config"].get("head_plan")!=expected or raw["hashes"]["checkpoint"]!=accuracy["model_sha256"] or
            raw["hashes"]["image"]!=base_raw["hashes"]["image"] or raw["network"]!=base_raw["network"] or
            raw["precision"]!=base_raw["precision"] or raw["hardware"]!=base_raw["hardware"] or
            abs(float(row["online_s"])-statistics.median(r["online_s"] for r in raw["measurements"]))>1e-9 or
            row["native_sha256"]!=baseline["native_sha256"] or int(row["repeats"])<3 or
            row["numerical_pass"]!="True" or not all(r["numerical_pass"] for r in raw["measurements"])):
            raise RuntimeError(f"Incomparable or incomplete full-model evidence: {row['case']}")
        if float(scores[name]["loss_pp"])<=meta["budget_pp"]:feasible.append(row)
    best=min(feasible,key=lambda r:float(r["online_s"]))
    name=best["case"].split('-')[1]
    write(out / "plan.json",dict(name=name,layout=best["layout"],**plans[name]))
    meta.update(status="complete",selected_case=best["case"],winner=name,
                selection_rule="Fastest actual full-model median among measured feasible candidates; baseline included",
                selected_online_s=float(best["online_s"]),observed_loss_pp=float(scores[name]["loss_pp"]),
                selected_gib=float(best["total_gib"]),native_sha256=best["native_sha256"],
                actual_lost=int(scores[name]["lost"]),actual_gained=int(scores[name]["gained"]),
                results_sha256=digest(out / "results.csv"),accuracy_samples=int(scores[name]["samples"]),
                speedup_percent=(1-float(best["online_s"])/float(baseline["online_s"]))*100,
                communication_reduction_percent=(1-float(best["total_gib"])/float(baseline["total_gib"]))*100)
    write(out / "run.json",meta)
    print(f"Selected {best['case']}: {best['online_s']} s; measured loss {float(scores[name]['loss_pp']):g} pp",flush=True)
