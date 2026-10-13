"""Stage two: actual mixed-head costs and additive prediction errors."""
import argparse
import bisect
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from runtime import ROOT, digest, write, profile
from experiment import case_config, run_case, save_rows

MIB = 1 << 20
CATEGORIES = ("link_blocking", "conversion_precision", "organization_packing",
              "matmul_remainder", "nonlinear_remainder", "other_runtime")


def sliced_attention(q, k, v, types, alpha, layout="joint"):
    """Public static slicing preserves the original head arithmetic."""
    import jax.numpy as jnp
    from heads import groups, mapping
    def select(x, ids):
        return jnp.concatenate([x[:, i:i+1] for i in ids], axis=1)
    scores = (q/8) @ jnp.swapaxes(k, -1, -2)
    weights, outputs, order = [], [], []
    for kind, ids in groups(types):
        w = mapping(select(scores, ids), kind, [alpha[i] for i in ids])
        order.extend(ids)
        if layout == "split":outputs.append(w @ select(v, ids))
        else:weights.append(w)
    joined = jnp.concatenate(outputs if layout == "split" else weights, axis=1)
    restored = select(joined, [order.index(i) for i in range(len(types))])
    return restored if layout == "split" else restored @ v


def cases():
    result = [dict(case=f"{k}{h}", types=k*h, head_count=h, layout="joint", organization="gather")
              for k in "SRL" for h in (1,12)]
    for n in (1,3,6,10):
        result.append(dict(case=f"SR{n}",types="S"*(12-n)+"R"*n,
                           head_count=12,layout="joint",organization="gather"))
    for name,types in (("SL6","S"*6+"L"*6),("SRL633","S"*6+"R"*3+"L"*3)):
        result.append(dict(case=name,types=types,head_count=12,layout="joint",organization="gather"))
    for label in ("SR6","SRL633"):
        template=next(c for c in result if c["case"]==label)
        result += [dict(template,case=label+"-split",layout="split"),
                   dict(template,case=label+"-slices",organization="slices")]
    result.append(dict(case="S12-native",types="S"*12,head_count=12,layout="joint",organization="native"))
    random.Random(20261012).shuffle(result)
    return result


def load_events(path):
    rows=list(csv.DictReader(path.open(),delimiter="\t"))
    for row in rows:
        for k in ("start_ns","end_ns","thread","peer","bytes","recv_bytes"):row[k]=int(row[k])
    if not rows or rows[0]["kind"]!="execution":raise RuntimeError(f"Missing trace: {path}")
    if any(r["end_ns"]<r["start_ns"] for r in rows):raise RuntimeError("Clock moved backwards")
    return rows


def event_category(r,include_link=True):
    kind,name=r["kind"],r["name"]
    if include_link and kind in ("send","recv"):return "link_blocking"
    if kind in ("conversion","conversion_region") or (kind=="MPC" and
            name in ("a2b","b2a","msb_a2b","trunc_a","v2a","a2v")):return "conversion_precision"
    if kind=="packing" or (kind=="HLO" and name in (
            "pphlo.gather","pphlo.concatenate","pphlo.reshape","pphlo.transpose","pphlo.slice")):return "organization_packing"
    if kind=="HAL" and name in ("f_mmul","f_batch_mmul","f_tensordot"):return "matmul_remainder"
    if kind=="HLO" and name in ("pphlo.exponential","pphlo.greater","pphlo.maximum","pphlo.reduce",
            "pphlo.reciprocal","pphlo.multiply","pphlo.select","pphlo.custom_call","pphlo.rsqrt"):return "nonlinear_remainder"
    return None


def partition(rows,include_link=True):
    """Disjoint attribution by interval union, not cumulative thread seconds.

    Wait > sharing/precision > organization > arithmetic. A blocked worker can
    overlap another worker's compute, so this is not a critical-path proof.
    """
    start,end=rows[0]["start_ns"],rows[0]["end_ns"]
    edges=[(start,0,5),(end,0,5)]
    for r in rows[1:]:
        category=event_category(r,include_link)
        a,b=max(start,r["start_ns"]),min(end,r["end_ns"])
        if category and b>a:
            i=CATEGORIES.index(category);edges.extend(((a,1,i),(b,-1,i)))
    active=Counter();total=Counter();bounds=[];labels=[];previous=start;current=5
    for stamp,delta,index in sorted(edges):
        if stamp>previous:total[CATEGORIES[current]]+=(stamp-previous)/1e9
        active[index]+=delta
        current=min((i for i,n in active.items() if n>0),default=5)
        if bounds and bounds[-1]==stamp:labels[-1]=CATEGORIES[current]
        else:bounds.append(stamp);labels.append(CATEGORIES[current])
        previous=stamp
    if abs(sum(total.values())-(end-start)/1e9)>1e-6:raise RuntimeError("Attribution does not close")
    return {k:total[k] for k in CATEGORIES},bounds,labels


def union_seconds(rows):
    total,last=0,None
    for a,b in sorted((r["start_ns"],r["end_ns"]) for r in rows):
        total+=b-a if last is None or a>last else max(0,b-last)
        last=b if last is None else max(last,b)
    return total/1e9


def dependency_depth(parties):
    """Observed causal rounds lower bound; thread fork/join edges are absent."""
    points=[]
    for rank,rows in enumerate(parties):
        for r in rows:
            if r["kind"] not in ("send","recv"):continue
            send=r["kind"]=="send"
            sender,receiver=(rank,r["peer"]) if send else (r["peer"],rank)
            points.append((r["start_ns"] if send else r["end_ns"],0 if send else 1,
                           rank,r["thread"],(sender,receiver,r["name"])))
    depth=defaultdict(int);sent={};matched=unmatched=0
    for stamp,kind,rank,thread,key in sorted(points):
        local=(rank,thread)
        if kind==0:
            if key in sent:raise RuntimeError("Duplicate message identifier")
            sent[key]=depth[local]+1
        elif key in sent:
            depth[local]=max(depth[local],sent[key]);matched+=1
        else:unmatched+=1
    if unmatched:raise RuntimeError(f"Unmatched receive events: {unmatched}")
    return max(depth.values(),default=0),matched


def measure(meta,label):
    folder=Path(meta["_folder"]);rows=[];raw_ops=[];raw_spans=[]
    if len(meta["measurements"])!=5 or len(meta["profiles"])!=5:raise RuntimeError(f"Need five calls: {label}")
    for sample,(m,parts) in enumerate(zip(meta["measurements"],meta["profiles"]),1):
        parties=[load_events(folder / "log" / f"native-p{r}-e{sample+1}.tsv") for r in range(2)]
        rank=max(range(2),key=lambda i:parts[i]["seconds"]);wall,_,_=partition(parties[rank])
        wire={k:0 for k in CATEGORIES}
        for party,events in enumerate(parties):
            _,bounds,names=partition(events,False)
            sends=[r for r in events if r["kind"]=="send"]
            if sum(r["bytes"] for r in sends)!=int(parts[party]["backend_metrics"]["send_bytes"]):
                raise RuntimeError(f"Send byte trace does not close: {label}, party {party}")
            if len(sends)!=int(parts[party]["backend_metrics"]["protocol_send_calls"]):raise RuntimeError("Send calls do not close")
            for r in sends:wire[names[max(0,bisect.bisect_right(bounds,r["start_ns"])-1)]]+=r["bytes"]
            for kind in ("conversion","conversion_region","packing"):
                for name in sorted({r["name"] for r in events if r["kind"]==kind}):
                    group=[r for r in events if r["kind"]==kind and r["name"]==name]
                    raw_spans.append(dict(case=label,sample=sample,party=party,kind=kind,name=name,calls=len(group),
                        union_s=union_seconds(group),thread_sum_s=sum((r["end_ns"]-r["start_ns"])/1e9 for r in group)))
            for name,op in parts[party]["operations"].items():raw_ops.append(dict(case=label,sample=sample,party=party,operation=name,**op))
        rounds,matched=dependency_depth(parties)
        row=dict(case=label,sample=sample,seconds=m["online_s"],payload_bytes=sum(wire.values()),slow_party=rank,
            numerical_pass=m["numerical_pass"],relative_l2=m["relative_l2"],max_abs=m["max_abs"],
            interaction_depth_lower_bound=rounds,matched_messages=matched,
            send_calls=sum(sum(r["kind"]=="send" for r in p) for p in parties),
            recv_calls=sum(sum(r["kind"]=="recv" for r in p) for p in parties),
            recv_blocking_union_s=union_seconds([r for r in parties[rank] if r["kind"]=="recv"]),
            send_blocking_union_s=union_seconds([r for r in parties[rank] if r["kind"]=="send"]))
        row["dot_H2A_calls"]=sum(r["name"]=="dot_H2A" for p in parties for r in p)
        row["dot_H2A_scope_s"]=union_seconds([r for r in parties[rank] if r["name"]=="dot_H2A"])
        row["decrypt_moddown_region_s"]=union_seconds([r for r in parties[rank] if r["name"]=="dot_decrypt_moddown"])
        for category in CATEGORIES:row[category+"_s"]=wall[category];row[category+"_bytes"]=wire[category]
        ops=parts[rank]["operations"]
        for key,name in (("exp_s","f_exp"),("comparison_s","pphlo.greater"),("reduce_s","pphlo.reduce"),
                         ("gather_s","pphlo.gather"),("reciprocal_s","f_reciprocal"),("multiply_s","pphlo.multiply")):
            row[key]=ops.get(name,{}).get("seconds",0.)
        for name in ("f_mmul","f_batch_mmul","f_reciprocal"):row[name+"_calls"]=ops.get(name,{}).get("calls",0)
        row["trace_seconds"]=sum(wall.values())
        if abs(row["trace_seconds"]-row["seconds"])>.01:raise RuntimeError("Native and trace wall time differ")
        rows.append(row)
    return rows,raw_ops,raw_spans


def micro(out,data,native):
    alpha=json.loads((data / "alpha.json").read_text())["alpha"][0];qkv=data / "attention.npz"
    suite=cases();path=out / "cases.json"
    if path.exists() and json.loads(path.read_text())!=suite:raise RuntimeError("Cases changed")
    write(path,suite)
    for index,c in enumerate(suite,1):
        label=c["case"];folder=out / "micro/log" / label
        config=case_config("attention","cpu",study="mixed",types=list(c["types"]),alpha=alpha,
            qkv="/work/"+str(qkv.relative_to(ROOT.parents[1])),qkv_sha256=digest(qkv),scope="attention",
            head_count=c["head_count"],layout=c["layout"],organization=c["organization"],native_library=str(native),
            trace_intervals=True,repeats=1,measured_calls=5,plan_name=label)
        if folder.exists() and json.loads((folder / "run.json").read_text())["status"]=="complete":
            saved=json.loads((folder / "run.json").read_text())
            if saved["config"]!=config or saved["library"]["sha256"]!=digest(native):raise RuntimeError(f"Cache changed: {label}")
        else:run_case(out / "micro",label,config)
        # Check each configuration immediately instead of postponing trace validation.
        saved=json.loads((folder / "run.json").read_text());saved["_folder"]=str(folder)
        measure(saved,label)
        print(f"Mixed-head suite: {index}/{len(suite)} configurations complete ({label})",flush=True)
        metadata=json.loads((out / "run.json").read_text());metadata.update(stage="micro",micro_completed=index,micro_total=len(suite))
        write(out / "run.json",metadata)


def full(out,stage1):
    plans=json.loads((stage1 / "plans.json").read_text());native=ROOT / "build/cpu.so"
    for label in ("S","r48"):
        old=json.loads((stage1 / f"log/full/log/cpu-{label}-joint/run.json").read_text())
        if old["library"]["sha256"]!=digest(native):raise RuntimeError("Original binary changed")
        config=case_config("vit","cpu",repeats=2,head_plan=None if label=="S" else plans[label],
                          plan_name=label,layout="joint",native_library=str(native))
        folder=out / "full/log" / label
        if folder.exists() and json.loads((folder / "run.json").read_text())["status"] in ("complete","outside_gate","matched"):continue
        run_case(out / "full",label,config)


def full_rows(out,stage1):
    rows=[]
    for label in ("S","r48"):
        paths=[stage1 / f"log/full/log/cpu-{label}-joint/run.json"]
        extension=out / f"full/log/{label}/run.json"
        if extension.exists():paths.append(extension)
        metas=[json.loads(p.read_text()) for p in paths]
        if len(metas)>1:
            if sum(len(m["measurements"]) for m in metas)!=5:raise RuntimeError("Incomplete full extension")
            for key in ("precision","network","hardware"):
                if metas[0][key]!=metas[1][key]:raise RuntimeError(f"Full extension changed {key}")
            if metas[0]["library"]["sha256"]!=metas[1]["library"]["sha256"]:raise RuntimeError("Full extension changed library")
            for key in ("heads.py","checkpoint","image"):
                if metas[0]["hashes"][key]!=metas[1]["hashes"][key]:raise RuntimeError(f"Full extension changed {key}")
            if metas[0]["config"].get("head_plan")!=metas[1]["config"].get("head_plan"):raise RuntimeError("Full plan changed")
        for path,m in zip(paths,metas):
            for i,measurement in enumerate(m["measurements"],1):
                pair=[profile(path.parent / f"log/b{i}-p{rank}.log") for rank in range(2)]
                rank=max(range(2),key=lambda k:pair[k]["seconds"]);ops=pair[rank]["operations"]
                row=dict(plan=label,source="stage1" if path==paths[0] else "stage2",sample=i,seconds=measurement["online_s"],
                    payload_mib=measurement["total_gib"]*1024,numerical_pass=measurement["numerical_pass"],party=rank,metadata=str(path))
                for name in sorted({k for p in pair for k in p["operations"] if k.startswith('pphlo.')}):
                    row[name+"_s"]=ops.get(name,{}).get("seconds",0.)
                rows.append(row)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    return [{k:r.get(k,0.) for k in fields} for r in rows]


def aggregate(rows,suite,cost_path):
    old={(r["scope"],int(r["heads"]),r["kind"]):r for r in csv.DictReader(cost_path.open())};summaries=[]
    for c in sorted(suite,key=lambda c:(c["head_count"],c["types"],c["case"])):
        group=[r for r in rows if r["case"]==c["case"]]
        if len(group)!=5 or not all(r["numerical_pass"] for r in group):raise RuntimeError(f"Invalid evidence: {c['case']}")
        summary=dict(c,repeats=5,seconds=statistics.median(r["seconds"] for r in group),
            min_s=min(r["seconds"] for r in group),max_s=max(r["seconds"] for r in group),
            iqr_s=float(np.subtract(*np.percentile([r["seconds"] for r in group],[75,25]))),
            payload_mib=statistics.median(r["payload_bytes"]/MIB for r in group))
        for key in group[0]:
            if key not in ("case","sample","numerical_pass","seconds","payload_bytes"):summary[key]=statistics.median(r[key] for r in group)
        for anchor in (1,12):
            for field,source in (("s","seconds"),("mib","mib")):
                prediction=sum(c["types"].count(k)*float(old[("attention",anchor,k)][source])/anchor for k in "SRL")
                summary[f"additive_{anchor}_{field}"]=prediction
                actual=summary['seconds' if field=='s' else 'payload_mib']
                summary[f"excess_{anchor}_{field}"]=actual-prediction
                if field=='s':summary[f"excess_{anchor}_pct"]=100*(actual/prediction-1)
        summaries.append(summary)
    return summaries


def plots(out,summaries,full_data,scope="core"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors=("#e45756","#b279a2","#f2cf5b","#4c78a8","#72b7b2","#bab0ac")
    labels=("Link blocking union","Sharing / precision","Head organization / packing",
            "Matmul remainder","Nonlinear remainder","Other / runtime")
    lookup={r["case"]:r for r in summaries}
    names=["S12","R12","L12","SR1","SR3","SR6","SR10","SL6","SRL633",
           "SR6-split","SRL633-split","SR6-slices","SRL633-slices"]
    names=[n for n in names if n in lookup]
    raw=list(csv.DictReader((out / ("block_samples.csv" if scope=="block" else "samples.csv")).open()))
    x=np.arange(len(names))
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,layout="constrained")
    for ax,suffix,unit in zip(axes,("_s","_bytes"),(1.,MIB)):
        bottom=np.zeros(len(names))
        for cat,color,label in zip(CATEGORIES,colors,labels):
            values=np.array([statistics.mean(float(r[cat+suffix])/unit for r in raw if r["case"]==n) for n in names])
            ax.bar(x,values,bottom=bottom,color=color,label=label);bottom+=values
    axes[0].set_ylabel("Mean native wall time (s), five calls")
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="outside upper center",ncols=3,fontsize=8)
    axes[0].set_title(f"A. Complete ViT Block: disjoint wall time and actual 2PC payload" if scope=="block" else "Attention core: disjoint wall time and actual 2PC payload")
    axes[1].set_ylabel("Mean payload sent by both parties (MiB)")
    axes[1].set_xticks(x,names,rotation=30,ha="right")
    fig.supxlabel("Priority: blocking > sharing/precision > organization > arithmetic. Wait can overlap worker compute.",fontsize=8)
    for suffix in ("pdf","png"):fig.savefig(out / f"interval_attribution_{scope}.{suffix}",dpi=180)
    plt.close(fig)
    operator_plot(out,summaries,scope)
    names=names[:11];selected=[lookup[n] for n in names];x=np.arange(len(names))
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout="constrained")
    for ax,metric in zip(axes,("s","mib")):
        for shift,anchor,color in ((-.26,1,"#bab0ac"),(0,12,"#72b7b2")):
            ax.bar(x+shift,[r[f"additive_{anchor}_{metric}"] for r in selected],width=.25,color=color,
                   label=f"Stage-one {anchor}-head additive prediction")
        actual=[r["seconds" if metric=="s" else "payload_mib"] for r in selected]
        ax.bar(x+.26,actual,width=.25,color="#e45756",label="Measured, five real calls")
        if metric=="s":
            ax.errorbar(x+.26,actual,yerr=[[a-r["min_s"] for a,r in zip(actual,selected)],
                [r["max_s"]-a for a,r in zip(actual,selected)]],fmt="none",ecolor="black",capsize=2)
        ax.set_ylabel("Median wall time (s)" if metric=="s" else "Actual payload (MiB)")
    axes[0].set_title("B. Complete Block: shared baseline + additive head savings" if scope=="block" else "Attention core: naive additive predictions versus actual cost")
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="outside upper center",ncols=3,fontsize=8)
    axes[1].set_xticks(x,names,rotation=25,ha="right")
    for suffix in ("pdf","png"):fig.savefig(out / f"figure_B_{scope}.{suffix}",dpi=180)
    plt.close(fig)
    fields=[k for k in full_data[0] if k.endswith("_s") and k.startswith("pphlo.")]
    delta={k:statistics.mean(r[k] for r in full_data if r["plan"]=="r48")-
             statistics.mean(r[k] for r in full_data if r["plan"]=="S") for k in fields}
    chosen=sorted(delta,key=lambda k:abs(delta[k]),reverse=True)[:9][::-1]
    fig,ax=plt.subplots(figsize=(9,4.5),layout="constrained")
    ax.barh([k[6:-2] for k in chosen],[delta[k] for k in chosen],
            color=["#e45756" if delta[k]>0 else "#72b7b2" for k in chosen])
    count=sum(r["plan"]=="S" for r in full_data)
    ax.axvline(0,color="black",lw=.6);ax.set_xlabel(f"R48 minus Softmax, mean wall seconds across {count} existing executions")
    ax.set_title("Existing full ViT evidence: HLO operation attribution")
    for suffix in ("pdf","png"):fig.savefig(out / f"full_attribution.{suffix}",dpi=180)
    plt.close(fig)


def operator_plot(out,summaries,scope):
    """One-level operator accounting is the main figure; nested regions stay auxiliary."""
    import matplotlib.pyplot as plt
    keys=("matmul","gelu","exponential","comparison","normalization","organization","other")
    labels=("Matrix operations (incl. protocol)","FFN GELU","Exponential","Comparison / ReLU guards",
            "Reduction / normalization","Grouping / rearrangement","Other / runtime")
    colors=("#4c78a8","#9d755d","#72b7b2","#e45756","#b279a2","#f2cf5b","#bab0ac")
    def category(name):
        if name in ("pphlo.dot","pphlo.dot_general","pphlo.convolution","pphlo.tensordot"):return "matmul"
        if name=="pphlo.custom_call":return "gelu"
        if name=="pphlo.exponential":return "exponential"
        if name in ("pphlo.greater","pphlo.less","pphlo.equal","pphlo.maximum","pphlo.minimum"):return "comparison"
        if name in ("pphlo.reduce","pphlo.reciprocal","pphlo.multiply","pphlo.divide","pphlo.select","pphlo.rsqrt"):return "normalization"
        if name in ("pphlo.gather","pphlo.concatenate","pphlo.reshape","pphlo.transpose","pphlo.slice","pphlo.broadcast"):return "organization"
        return "other"
    sample_path=out / ("block_samples.csv" if scope=="block" else "samples.csv")
    op_path=out / ("block_operations.csv" if scope=="block" else "operations.csv")
    samples=list(csv.DictReader(sample_path.open()));operations=list(csv.DictReader(op_path.open()))
    names=[n for n in ("S12","R12","L12","SR1","SR3","SR6","SR10","SL6","SRL633",
        "SR6-split","SRL633-split","SR6-slices","SRL633-slices") if any(r["case"]==n for r in summaries)]
    indexed=defaultdict(list)
    for r in operations:
        if r["operation"].startswith("pphlo."):indexed[(r["case"],int(r["sample"]),int(r["party"]))].append(r)
    parts=[]
    for r in samples:
        counts={k:0. for k in keys};payload={k:0. for k in keys}
        for op in indexed[(r["case"],int(r["sample"]),int(r["slow_party"]))]:
            counts[category(op["operation"])]+=float(op["seconds"])
        residual=float(r["seconds"])-sum(counts.values())
        if residual < -1e-5:raise RuntimeError("HLO time overlaps: cannot form one-level stack")
        counts["other"]+=max(0.,residual)
        for party in range(2):
            for op in indexed[(r["case"],int(r["sample"]),party)]:
                payload[category(op["operation"])]+=float(op["sent_mib"])
        missing=float(r["payload_bytes"])/MIB-sum(payload.values())
        if abs(missing)>.01:raise RuntimeError("HLO payload attribution does not close")
        payload["other"]+=missing
        parts.append(dict(case=r["case"],sample=int(r["sample"]),
            **{k+"_s":counts[k] for k in keys},**{k+"_mib":payload[k] for k in keys}))
    save_rows(out / f"operator_attribution_{scope}.csv",parts)
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,layout="constrained");x=np.arange(len(names))
    for ax,suffix in zip(axes,("_s","_mib")):
        bottom=np.zeros(len(names))
        for key,color,label in zip(keys,colors,labels):
            values=np.array([statistics.mean(r[key+suffix] for r in parts if r["case"]==n) for n in names])
            ax.bar(x,values,bottom=bottom,color=color,label=label);bottom+=values
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="outside upper center",ncols=3,fontsize=8)
    axes[0].set_ylabel("Mean inclusive operator wall time (s)")
    axes[1].set_ylabel("Actual payload sent by both parties (MiB)")
    axes[0].set_title("A. Complete ViT Block: operator time and communication" if scope=="block" else "Attention core: operator time and communication")
    axes[1].set_xticks(x,names,rotation=30,ha="right")
    fig.supxlabel("Five real calls. Each operator includes its protocol work; nested conversion and wait spans are reported separately.",fontsize=8)
    for suffix in ("pdf","png"):fig.savefig(out / f"figure_A_{scope}.{suffix}",dpi=180)
    plt.close(fig)


def block_report(out,stage1,full_data):
    suite=json.loads((out / "block_cases.json").read_text());rows=[];ops=[];spans=[]
    inputs=set()
    for c in suite:
        folder=out / "blocks/log" / c["case"]
        meta=json.loads((folder / "run.json").read_text());meta["_folder"]=str(folder)
        inputs.add(meta["block_input_sha256"])
        a,b,d=measure(meta,c["case"]);rows+=a;ops+=b;spans+=d
    if len(inputs)!=1:raise RuntimeError("Block configurations used different actual inputs")
    save_rows(out / "block_samples.csv",rows);save_rows(out / "block_operations.csv",ops)
    save_rows(out / "block_protocol_spans.csv",spans)
    summaries=aggregate(rows,suite,stage1 / "cost.csv")
    baseline=next(r for r in summaries if r["case"]=="S12")
    pure={k:next(r for r in summaries if r["case"]==k+"12") for k in "SRL"}
    old={(int(r["heads"]),r["kind"]):r for r in csv.DictReader((stage1 / "cost.csv").open()) if r["scope"]=="attention"}
    for r in summaries:
        for anchor in (1,12):
            r[f"core_additive_{anchor}_s"]=r[f"additive_{anchor}_s"]
            r[f"additive_{anchor}_s"]+=baseline["seconds"]-12*float(old[(anchor,"S")]["seconds"])/anchor
            r[f"additive_{anchor}_mib"]+=baseline["payload_mib"]-12*float(old[(anchor,"S")]["mib"])/anchor
            r[f"excess_{anchor}_s"]=r["seconds"]-r[f"additive_{anchor}_s"]
            r[f"excess_{anchor}_pct"]=100*(r["seconds"]/r[f"additive_{anchor}_s"]-1)
            r[f"excess_{anchor}_mib"]=r["payload_mib"]-r[f"additive_{anchor}_mib"]
        r["time_change_pct"]=100*(r["seconds"]/baseline["seconds"]-1)
        r["communication_change_pct"]=100*(r["payload_mib"]/baseline["payload_mib"]-1)
        r["additive_block_pure_s"]=sum(r["types"].count(k)*pure[k]["seconds"]/12 for k in "SRL")
        r["excess_block_pure_s"]=r["seconds"]-r["additive_block_pure_s"]
    save_rows(out / "block_summary.csv",summaries)
    plots(out,summaries,full_data,"block")
    # Main requested figures cover the complete Block; core figures are auxiliary.
    import shutil
    for figure in ("A","B"):
        for suffix in ("pdf","png"):shutil.copy2(out / f"figure_{figure}_block.{suffix}",out / f"figure_{figure}.{suffix}")
    return summaries


def report(out,stage1):
    suite=json.loads((out / "cases.json").read_text());samples=[];ops=[];spans=[]
    for c in suite:
        folder=out / "micro/log" / c["case"]
        meta=json.loads((folder / "run.json").read_text());meta["_folder"]=str(folder)
        a,b,d=measure(meta,c["case"]);samples+=a;ops+=b;spans+=d
    save_rows(out / "samples.csv",samples);save_rows(out / "operations.csv",ops)
    save_rows(out / "protocol_spans.csv",spans)
    summaries=aggregate(samples,suite,stage1 / "cost.csv");save_rows(out / "summary.csv",summaries)
    full_data=full_rows(out,stage1);save_rows(out / "full_samples.csv",full_data)
    attribution=[]
    for key in (k for k in full_data[0] if k.startswith("pphlo.")):
        s=statistics.mean(r[key] for r in full_data if r["plan"]=="S")
        r=statistics.mean(r[key] for r in full_data if r["plan"]=="r48")
        attribution.append(dict(operation=key[:-2],S_mean_s=s,R48_mean_s=r,delta_s=r-s))
    attribution.sort(key=lambda r:abs(r["delta_s"]),reverse=True)
    save_rows(out / "full_attribution.csv",attribution);plots(out,summaries,full_data)
    block_summaries=block_report(out,stage1,full_data)
    lookup={r["case"]:r for r in summaries}
    block_lookup={r["case"]:r for r in block_summaries}
    attr={r["operation"]:r["delta_s"] for r in attribution}
    pure_costs=[r for r in csv.DictReader((stage1 / "cost.csv").open()) if r["scope"]=="attention" and r["heads"]=="12"]
    findings=dict(stage1=str(stage1),protocol="CPU BumbleBee FM64/f18, 25 threads/party, 1 Gbps / 0.25 ms egress",
        micro_cases=len(suite),micro_calls=len(samples),block_cases=len(block_summaries),block_calls=len(block_summaries)*5,
        full_new_calls=sum(r["source"]=="stage2" for r in full_data),
        full_repeats_per_plan=sum(r["plan"]=="S" for r in full_data),
        authorized_scope="Complete single-Block diagnosis and reuse of existing full-model evidence; wait for user before new full-model work",
        interaction_rounds="Observed message dependency-depth lower bound; thread fork/join edges remain unobserved",
        payload="Actual YACL payload sent by both parties; excludes transport headers",
        time_partition="Observed interval unions under a declared priority; blocking can overlap parallel computation",
        repeat_design="One fresh runtime and warmup per core/Block configuration, then five real calls; full evidence reuses existing runs only",
        numerical_pass=all(r["numerical_pass"] for r in samples+full_data),
        block={name:{k:r[k] for k in ("seconds","min_s","max_s","payload_mib","time_change_pct",
            "communication_change_pct","additive_12_s","excess_12_s","additive_block_pure_s",
            "excess_block_pure_s","gather_s","comparison_s","dot_H2A_calls","f_reciprocal_calls")}
            for name,r in block_lookup.items()},
        core_pure_recheck={r["kind"]:dict(stage1_s=float(r["seconds"]),stage2_s=lookup[r["kind"]+"12"]["seconds"],
            difference_pct=100*(lookup[r["kind"]+"12"]["seconds"]/float(r["seconds"])-1)) for r in pure_costs},
        archived_R48_explanation=dict(exp_and_reduce_saved_s=-attr["pphlo.exponential"]-attr["pphlo.reduce"],
            comparison_added_s=attr["pphlo.greater"],gather_added_s=attr["pphlo.gather"],
            multiply_and_reciprocal_added_s=attr["pphlo.multiply"]+attr["pphlo.reciprocal"],
            common_GELU_difference_s=attr["pphlo.custom_call"],
            limitation="Descriptive difference of the three archived runs; common-operator timing drift is not a proven mixed-head effect"),
        block_slice_intervention=dict(original_s=block_lookup["SR6"]["seconds"],
            slices_s=block_lookup["SR6-slices"]["seconds"],original_gather_s=block_lookup["SR6"]["gather_s"],
            slices_gather_s=block_lookup["SR6-slices"]["gather_s"],
            saved_s=block_lookup["SR6"]["seconds"]-block_lookup["SR6-slices"]["seconds"]),
        full={label:dict(median_s=statistics.median(r["seconds"] for r in full_data if r["plan"]==label),
            mean_s=statistics.mean(r["seconds"] for r in full_data if r["plan"]==label),
            min_s=min(r["seconds"] for r in full_data if r["plan"]==label),
            max_s=max(r["seconds"] for r in full_data if r["plan"]==label),
            payload_mib=statistics.median(r["payload_mib"] for r in full_data if r["plan"]==label))
            for label in ("S","r48")},
        gather_interventions={label:dict(original_s=lookup[label]["seconds"],slices_s=lookup[label+"-slices"]["seconds"],
            original_gather_s=lookup[label]["gather_s"],slices_gather_s=lookup[label+"-slices"]["gather_s"],
            saved_s=lookup[label]["seconds"]-lookup[label+"-slices"]["seconds"])
            for label in ("SR6","SRL633")})
    write(out / "findings.json",findings)
    def table(rows,columns):
        from html import escape
        names=dict(case="配置",plan="方案",seconds="耗时(s)",payload_mib="通信(MiB)",additive_12_s="加性预测(s)",
            excess_12_s="偏差(s)",excess_12_pct="偏差(%)",time_change_pct="时间变化(%)",
            communication_change_pct="通信变化(%)",additive_block_pure_s="纯类型预测(s)",
            excess_block_pure_s="纯类型偏差(s)",gather_s="gather(s)",comparison_s="比较(s)",
            dot_H2A_calls="H2A次数",interaction_depth_lower_bound="消息依赖下界",reciprocal_s="倒数(s)",
            operation="算子",S_mean_s="S均值(s)",R48_mean_s="R48均值(s)",delta_s="R48−S(s)")
        return "<table><tr>"+"".join("<th>"+escape(names.get(k,k))+"</th>" for k in columns)+"</tr>"+"".join(
            "<tr>"+"".join("<td>"+escape(f"{r[k]:.3f}" if isinstance(r[k],float) else str(r[k]))+"</td>"
                           for k in columns)+"</tr>" for r in rows)+"</table>"
    body='<html lang="zh"><meta charset="utf-8"><title>BumbleBee 混合头成本实验</title><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:40px auto;padding:20px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:6px}img{max-width:100%}p{max-width:1050px}</style><h1>BumbleBee 第二阶段：混合头实际成本</h1>'
    body+='<p>比较、重排及分组归一化的额外成本会抵消便宜映射的部分收益。原始协议包含 H2A 随机掩码和解密/模数下降，它们属于矩阵/乘法内部；类型分组之间没有另外插入显式 HE/MPC 转换。</p>'
    body+='<p>本轮先完成单 Block 诊断：10 个完整 ViT 第一层 Block 配置，每组一次预热及五次真实两方安全调用。Block 包含两次 LayerNorm、QKV/O 投影、attention、FFN、intrinsic GELU 及残差；输入是同一张真实图片的共享中间激活，模型权重维持私有。另有 17 组 attention core 微基准辅助分解。依据最新指示，没有启动新的完整模型推理；仅复用 out/10 的既有三次 S/R48 结果。诊断配置不按准确率预算筛选。S/R/L 表示 Softmax、归一化 ReLU、线性映射；SR6 表示 12 头中 S6+R6，SRL633 为 S6+R3+L3；core 的 R1 则是单个 R 头。</p>'
    body+='<p>安全输出检查采用相对 L2≤0.05、最大绝对误差≤0.5 及有限值检查，这是数值一致性标准，不是 Top-1 准确率预算。单 Block 的预热计时排除了协议首次初始化，既有完整模型则包含初始化；不能直接将单层时间乘以 12 当作整模型时间。</p>'
    body+='<h2>图 A：算子时间与通信量分解</h2><img src="figure_A.png"><p>主图在同一 HLO 层级分解时间及实际发送字节：矩阵、GELU、指数、比较、归约/归一化、组织及余项，每次均核对总量闭合。算子时间包含其协议内部转换与等待，不能解释为纯 CPU 时间。HE/MPC 转换不是独立 HLO 步骤时，不伪造一个可相加的转换柱。</p>'
    body+='<h2>协议内部转换与等待（辅助区间归属）</h2><img src="interval_attribution_block.png"><p>这张辅助图按区间并集和显式优先级归属，不能用它判断纯转换或纯网络成本。Sharing/precision 包含比较中的算术/布尔共享处理及截断，宽泛的乘法-掩码区域也包含计算。链接调用可能等待对方计算并与本方其他线程并行；不能把累计 HE 秒数、HAL/MPC 子计时或链接等待再次加到主图上。精确 H2A 调用数与实际转换函数见 CSV 和 protocol_audit.json。</p>'
    body+='<h2>图 B：加性预测与实际成本</h2><img src="figure_B.png"><p>完整 Block 预测固定以本轮全 Softmax Block 为公共基准，加上 out/10 单头/12 头成本预测的映射替换收益：CBlock(S) + Σ nₜ[Cₜ(h)−Cₛ(h)]/h。以同样方式预测通信量。原始 attention core 图直接使用 Σ nₜ Cₜ(h)/h。纯类型复测观察漂移，偏差可能包含规模、分组归一化、组织及公共矩阵布局，不能统称为转换开销。</p>'
    body+=table(block_summaries,["case","seconds","payload_mib","additive_12_s","excess_12_s",
                                "time_change_pct","communication_change_pct"])
    body+='<p>同轮完整 Block 的纯类型加性预测为 Σ nₜ CBlock(t)/12。它用于排除只因旧微基准漂移产生的解释；S6+R6 对这项预测仍有正偏差。</p>'
    body+=table(block_summaries,["case","additive_block_pure_s","excess_block_pure_s",
                                "gather_s","comparison_s","dot_H2A_calls","interaction_depth_lower_bound"])
    body+='<h2>辅助 attention core 结果</h2><img src="figure_A_core.png"><img src="figure_B_core.png">'
    body+=table(summaries,["case","seconds","payload_mib","additive_12_s","excess_12_s","excess_12_pct",
                          "gather_s","comparison_s","reciprocal_s","interaction_depth_lower_bound"])
    body+='<h2>R48 完整模型解释</h2><img src="full_attribution.png">'
    body+=table([dict(plan=k,**v) for k,v in findings["full"].items()],["plan","median_s","mean_s","min_s","max_s","payload_mib"])
    body+=table(attribution[:12],["operation","S_mean_s","R48_mean_s","delta_s"])
    body+='<p>上表仅复用既有三次完整模型结果，使用每次较慢一方的 HLO 计时和三次均值。HLO、HAL、MPC 相互嵌套，不能跨层相加。公共投影、FFN、GELU 和 QK/AV 限制全模型收益；机器状态造成的公共算子时差不能全部归于混合头。新的完整模型对照等待下一步指示。</p>'
    e=findings["archived_R48_explanation"]
    body+=f'<p>既有 R48 中，指数和归约共省 {e["exp_and_reduce_saved_s"]:.2f} 秒；比较增加 {e["comparison_added_s"]:.2f} 秒、gather 增加 {e["gather_added_s"]:.2f} 秒、乘法与倒数增加 {e["multiply_and_reciprocal_added_s"]:.2f} 秒。这几项抵消了大部分收益，余下收益只占约 402 秒总时间的一小部分；共同 GELU 的跨配置时差约 {e["common_GELU_difference_s"]:.2f} 秒，量级也足以影响中位数比较。证据支持比较、分组归一化与组织成本为主要线索，未发现另行插入的类型间 HE/MPC 转换步骤。</p>'
    body+='<h2>组织成本干预</h2><p>保持相同头计划、Q/K/V 和映射函数，将公开索引 gather 改为静态 slice/concatenate，每个版本测五次。这验证组织成本可优化，尚未证明完整模型能获得相同加速。</p>'
    body+=table([dict(case=k,**v) for k,v in findings["gather_interventions"].items()],
                ["case","original_s","slices_s","original_gather_s","slices_gather_s","saved_s"])
    body+='<p>完整 Block 的 S6+R6 静态切片结果：</p>'+table([findings["block_slice_intervention"]],
        ["original_s","slices_s","original_gather_s","slices_gather_s","saved_s"])
    body+='<p>gather 消失证明该组织开销可消除；完整 Block 的两组耗时范围仍重叠，不能仅凭这五次结果声称稳定的整体加速。split AV 增加了批量矩阵乘法及 H2A 次数，但未显示明显加速；当前证据不足以把批处理退化列为主要瓶颈。</p>'
    body+='<h2>协议记录与限制</h2><p>interaction_depth_lower_bound 来自实际消息匹配和线程内依赖，没有把密文条数或 send_calls 当作轮次。跨线程 fork/join 未完全观测，因此不能声称获得精确全局轮次。五次连续微基准不是五个独立机器环境；耗时范围仅作描述性统计。</p>'
    body+='<p><a href="block_summary.csv">完整 Block 汇总</a> · <a href="block_samples.csv">Block 逐次记录</a> · <a href="block_operations.csv">Block 算子记录</a> · <a href="block_protocol_spans.csv">Block 协议区间</a> · <a href="summary.csv">core 汇总</a> · <a href="full_samples.csv">既有完整模型记录</a> · <a href="findings.json">核查结论</a></p></html>'
    (out / "report.html").write_text(body)
    meta=json.loads((out / "run.json").read_text());meta.update(status="single_block_complete",stage="report",
        summary_sha256=digest(out / "summary.csv"),sample_sha256=digest(out / "samples.csv"),
        micro_calls=len(samples),block_calls=len(block_summaries)*5,full_new_calls=findings["full_new_calls"],
        numerical_pass=True,next_step="Await user instruction before new full-model comparison")
    meta["block_summary_sha256"]=digest(out / "block_summary.csv")
    meta["block_sample_sha256"]=digest(out / "block_samples.csv")
    write(out / "run.json",meta)
    print(json.dumps(findings,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT / "out/11")
    parser.add_argument("--stage1",type=Path,default=ROOT / "out/10")
    parser.add_argument("--data",type=Path,default=ROOT / "out/07")
    parser.add_argument("--native",type=Path,default=ROOT / "build/stage2.so")
    parser.add_argument("--stage",choices=("all","micro","blocks","full","report"),default="all")
    args=parser.parse_args()
    out=args.out.resolve();data=args.data.resolve();stage1=args.stage1.resolve();native=args.native.resolve()
    out.mkdir(parents=True,exist_ok=True)
    config=dict(stage1=str(stage1),stage1_cost_sha256=digest(stage1 / "cost.csv"),
        qkv_sha256=digest(data / "attention.npz"),trace_native_sha256=digest(native),
        original_native_sha256=digest(ROOT / "build/cpu.so"))
    if not (out / "run.json").exists():write(out / "run.json",dict(status="running",stage=args.stage,config=config))
    elif json.loads((out / "run.json").read_text())["config"]!=config:raise RuntimeError("Evidence inputs changed")
    if args.stage in ("all","micro"):micro(out,data,native)
    if args.stage in ("all","blocks"):
        from block import run_blocks
        run_blocks(out,data,native)
    if args.stage=="full":full(out,stage1)
    if args.stage in ("all","report"):report(out,stage1)


if __name__=="__main__":
    main()
