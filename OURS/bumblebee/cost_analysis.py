"""Role-consistent audit, cost prediction and figures for secure_cost_study.

HLO times are inclusive and form one accounting level. Conversion and blocking
intervals are nested diagnostics and must never be added to those times.
"""
import argparse
import csv
import html
import hashlib
import itertools
import json
import math
import marshal
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment import save_rows
from mixed import dependency_depth, load_events, union_seconds
from runtime import ROOT, digest, profile, write

MIB = 2**20
KEYS = ("matrix", "gelu", "exponential", "comparison", "normalization", "organization", "other")
COLORS = ("#4078a5", "#7353a0", "#e5a33c", "#ce5b5b", "#50a388", "#65727d", "#c5c9ce")
LABELS = ("Matrix products", "GELU", "Exponential", "Explicit comparison", "Reduction / normalization", "Organization", "Other / runtime")


def category(name):
    if name in ("pphlo.dot", "pphlo.dot_general", "pphlo.convolution"): return "matrix"
    if name == "pphlo.custom_call": return "gelu"
    if name == "pphlo.exponential": return "exponential"
    if name in ("pphlo.greater", "pphlo.less", "pphlo.equal", "pphlo.maximum", "pphlo.minimum"): return "comparison"
    if name in ("pphlo.reduce", "pphlo.reciprocal", "pphlo.multiply", "pphlo.divide", "pphlo.select", "pphlo.rsqrt"): return "normalization"
    if name in ("pphlo.gather", "pphlo.concatenate", "pphlo.reshape", "pphlo.transpose", "pphlo.slice", "pphlo.broadcast"): return "organization"
    return "other"


def intersection_seconds(a, b):
    def merge(rows):
        result = []
        for left, right in sorted((r["start_ns"], r["end_ns"]) for r in rows):
            if result and left <= result[-1][1]: result[-1][1] = max(right, result[-1][1])
            else: result.append([left, right])
        return result
    x, y = merge(a), merge(b)
    i = j = total = 0
    while i < len(x) and j < len(y):
        total += max(0, min(x[i][1], y[j][1])-max(x[i][0], y[j][0]))
        if x[i][1] < y[j][1]: i += 1
        else: j += 1
    return total/1e9


def parse_case(folder, label, source, scope, round_number=0):
    m = json.loads((folder / "run.json").read_text())
    if m["status"] not in ("complete", "matched", "outside_gate"):
        raise RuntimeError(f"Not complete: {folder}")
    if not all(x["numerical_pass"] for x in m["measurements"]):
        raise RuntimeError(f"Numerical failure: {folder}")
    config = m["config"]
    samples, roles, operations, spans, audits, layers = [], [], [], [], [], []
    for sample, (measurement, pair) in enumerate(zip(m["measurements"], m["profiles"]), 1):
        execution = sample+1 if scope in ("block", "core") else 1
        parties = [load_events(folder / "log" / f"native-p{rank}-e{execution}.tsv") for rank in range(2)]
        depth, messages = dependency_depth(parties)
        global_span=(max(p[0]["end_ns"] for p in parties)-min(p[0]["start_ns"] for p in parties))/1e9
        meta_fields = dict(source=source, scope=scope, case=label, sample=sample, round=round_number)
        sample_row = dict(**meta_fields, seconds=measurement["online_s"],
            p0_s=pair[0]["seconds"], p1_s=pair[1]["seconds"], slow_party=int(pair[1]["seconds"] > pair[0]["seconds"]),
            payload_mib=sum(p["sent_mib"] for p in pair), request_s=measurement["request_s"],
            compile_s=measurement.get("compile_s", 0.), numerical_pass=measurement["numerical_pass"],
            request_without_compile_s=measurement["request_s"]-measurement.get("compile_s",0.),
            native_global_span_s=global_span,native_global_minus_max_party_s=global_span-measurement["online_s"],
            party_start_skew_s=(parties[1][0]["start_ns"]-parties[0][0]["start_ns"])/1e9,
            party_finish_skew_s=(parties[1][0]["end_ns"]-parties[0][0]["end_ns"])/1e9,
            relative_l2=measurement["relative_l2"], max_abs=measurement["max_abs"],
            reference_top1=measurement.get("reference_top1", ""), secure_top1=measurement.get("secure_top1", ""),
            interaction_depth_lower_bound=depth, matched_messages=messages,
            dot_H2A_calls=sum(r["name"] == "dot_H2A" for p in parties for r in p),
            metadata=str(folder / "run.json"))
        samples.append(sample_row)
        for rank, (part, original_rows) in enumerate(zip(pair, parties)):
            bounds = original_rows[0]
            rows = [dict(r, start_ns=max(bounds["start_ns"], r["start_ns"]), end_ns=min(bounds["end_ns"], r["end_ns"]))
                    for r in original_rows[1:] if r["end_ns"] > bounds["start_ns"] and r["start_ns"] < bounds["end_ns"]]
            send, recv = ([r for r in rows if r["kind"] == kind] for kind in ("send", "recv"))
            payload = sum(r["bytes"] for r in send)
            backend = part["backend_metrics"]
            checks = dict(byte_trace_closed=payload == int(backend["send_bytes"]),
                          send_calls_closed=len(send) == int(backend["protocol_send_calls"]),
                          recv_bytes_closed=sum(r["bytes"] for r in recv) == int(backend["recv_bytes"]),
                          role_bytes_closed=abs(part["sent_mib"]-pair[1-rank]["recv_mib"]) < .01,
                          time_trace_closed=abs((bounds["end_ns"]-bounds["start_ns"])/1e9-part["seconds"]) < .01)
            if not all(checks.values()): raise RuntimeError(f"Trace closure failed: {folder}, {sample}, P{rank}, {checks}")
            seconds = {k: 0. for k in KEYS}; wire = {k: 0. for k in KEYS}
            for name, op in part["operations"].items():
                operations.append(dict(**meta_fields, party=rank, operation=name, **op))
                if name.startswith("pphlo."):
                    seconds[category(name)] += op["seconds"]
                    wire[category(name)] += op["sent_mib"]
            residual = part["seconds"]-sum(seconds.values())
            wire_residual = part["sent_mib"]-sum(wire.values())
            if residual < -.01 or abs(wire_residual) > .01: raise RuntimeError(f"HLO accounting failed: {folder}, {residual}, {wire_residual}")
            seconds["other"] += residual; wire["other"] += wire_residual
            role = dict(**meta_fields, party=rank, seconds=part["seconds"], sent_mib=part["sent_mib"], recv_mib=part["recv_mib"],
                **{k+"_s": v for k, v in seconds.items()}, **{k+"_sent_mib": v for k, v in wire.items()},
                hlo_runtime_residual_s=residual, link_blocking_union_s=union_seconds(send+recv),
                recv_blocking_union_s=union_seconds(recv), send_blocking_union_s=union_seconds(send),
                send_calls=len(send), recv_calls=len(recv),
                dot_H2A_union_s=union_seconds([r for r in rows if r["name"] == "dot_H2A"]),
                dot_decrypt_moddown_inclusive_s=union_seconds([r for r in rows if r["name"] == "dot_decrypt_moddown"]),
                precision_sharing_union_s=union_seconds([r for r in rows if r["kind"] == "MPC" and r["name"] in ("a2b", "b2a", "msb_a2b", "trunc_a", "v2a", "a2v")]),
                protocol_packing_union_s=union_seconds([r for r in rows if r["kind"] == "packing"]))
            for key in KEYS:
                hlo = [r for r in rows if r["kind"] == "HLO" and category(r["name"]) == key]
                role[f"link_overlap_{key}_s"] = intersection_seconds(send+recv, hlo)
            for name in ("dot_H2A", "dot_decrypt_moddown"):
                group = [r for r in rows if r["name"] == name]
                role[name+"_link_overlap_s"] = intersection_seconds(group, send+recv)
                role[name+"_outside_link_s"] = union_seconds(group)-role[name+"_link_overlap_s"]
            for name in ("pphlo.gather", "pphlo.greater", "pphlo.reduce", "pphlo.reciprocal", "pphlo.multiply", "f_reciprocal", "f_mmul", "f_batch_mmul", "f_gelu"):
                op = part["operations"].get(name, {})
                role[name+"_calls"] = op.get("calls", 0)
                role[name+"_seconds"] = op.get("seconds", 0.)
            roles.append(role)
            for kind in ("conversion", "conversion_region", "packing", "MPC"):
                names={r['name'] for r in rows if r['kind']==kind}
                if kind=='MPC':names&={'a2b','b2a','msb_a2b','trunc_a','v2a','a2v'}
                for name in sorted(names):
                    group = [r for r in rows if r["kind"] == kind and r["name"] == name]
                    spans.append(dict(**meta_fields, party=rank, kind=kind, name=name, calls=len(group),
                        union_s=union_seconds(group), thread_sum_s=sum((r["end_ns"]-r["start_ns"])/1e9 for r in group),
                        link_overlap_s=intersection_seconds(group, send+recv)))
            if scope == "vit":
                batch=[r for r in rows if r["kind"]=="HAL" and r["name"]=="f_batch_mmul"]
                matrix=[r for r in rows if r["kind"]=="HLO" and r["name"]=="pphlo.dot_general" and
                        any(r["start_ns"]<=b["start_ns"] and b["end_ns"]<=r["end_ns"] for b in batch)]
                matrix.sort(key=lambda r:r["start_ns"])
                if len(matrix)!=24: raise RuntimeError(f"Cannot identify 12 QK/AV windows: {folder}, P{rank}, {len(matrix)}")
                for layer in range(12):
                    left,right=matrix[2*layer],matrix[2*layer+1]
                    interval=dict(start_ns=left["start_ns"],end_ns=right["end_ns"])
                    inside=[r for r in rows if r["kind"]=="HLO" and left["start_ns"]<=r["start_ns"] and r["end_ns"]<=right["end_ns"]]
                    types="S"*12 if not config.get("head_plan") else "".join(config["head_plan"]["heads"][layer])
                    layers.append(dict(**meta_fields,party=rank,layer=layer,types=types,
                        QK_to_AV_window_s=(right["end_ns"]-left["start_ns"])/1e9,
                        **{k+"_s":sum((r["end_ns"]-r["start_ns"])/1e9 for r in inside if category(r["name"])==k) for k in KEYS},
                        link_blocking_union_s=intersection_seconds([interval],send+recv),
                        gather_calls=sum(r["name"]=="pphlo.gather" for r in inside)))
            audits.append(dict(**meta_fields, party=rank, **checks, hlo_time_closed=abs(sum(seconds.values())-part["seconds"]) < 1e-8,
                               hlo_payload_closed=abs(sum(wire.values())-part["sent_mib"]) < 1e-8))
    common = dict(source=source, scope=scope, case=label, types="".join(config.get("types", [])),
                  layout=config.get("layout", "joint"), organization=config.get("organization", "gather"),
                  library_sha256=m["library"]["sha256"], checkpoint_sha256=m["hashes"]["checkpoint"],
                  input_sha256=m.get("block_input_sha256", m.get("input_sha256", config.get("qkv_sha256", ""))),
                  cpus=config["cpus"], threads=config["threads"], precision=m["precision"], network=m["network"],
                  computation_hashes={k:v for k,v in m["hashes"].items() if k.endswith(".py") and k != "mixed.py"},
                  observer_report_source_sha256=m["hashes"].get("mixed.py"),
                  compiled_sha256=m.get("compilation",{}).get("sha256"),
                  metadata=str(folder / "run.json"))
    return samples, roles, operations, spans, audits, common, layers


def summarize(rows, group_keys, numeric_keys=None):
    groups = defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in group_keys)].append(r)
    result = []
    for key, values in groups.items():
        numeric = numeric_keys or [k for k,v in values[0].items() if isinstance(v, (int,float)) and not isinstance(v,bool) and k not in group_keys+('sample','round')]
        row = dict(zip(group_keys, key)); row["repeats"] = len(values)
        for name in numeric:
            a = [r[name] for r in values]
            row[name+"_median"] = st.median(a); row[name+"_mean"] = st.mean(a)
        if "seconds" in values[0]:
            a = [r["seconds"] for r in values]
            row.update(min_s=min(a), max_s=max(a), std_s=st.stdev(a) if len(a)>1 else 0., iqr_s=float(np.subtract(*np.percentile(a,[75,25]))))
        result.append(row)
    return result


def features(c, model):
    types=c["types"]; s,r,l=[types.count(k) for k in "SRL"]
    groups=len(set(types))
    if model == "group": return [1., r, l, groups-1]
    extra_av=(groups-1) if c["layout"] == "split" else 0
    # Relative indexing elements after existing compiler lowering. A singleton
    # head select becomes a slice; restoration still touches all twelve heads.
    selected=sum(types.count(t) for t in "SRL" if types.count(t)>1)/12
    gather_volume=0. if c["organization"] in ("native", "slices") else (1.+selected if c["layout"] == "joint" else selected*(1.+64/197)+64/197)
    return [1., r, l, int(r>0), int(s>0)+int(r>0)-1, gather_volume, extra_av]


def ridge(train, targets, test, lam):
    x=np.asarray(train,float);y=np.asarray(targets,float);z=np.asarray(test,float)
    center=x.mean(axis=0);scale=x.std(axis=0);center[0]=0.;scale[scale<1e-12]=1.;scale[0]=1.
    x=(x-center)/scale;z=(z-center)/scale
    penalty=np.eye(x.shape[1])*lam;penalty[0,0]=0.
    b=np.linalg.pinv(x.T@x+penalty)@x.T@y
    return z@b, b/scale, float(b[0]-np.sum(b[1:]*center[1:]/scale[1:]))


def choose_lambda(x, y, groups=None):
    grid=(0., .001, .01, .1, 1., 10., 100.)
    groups=list(range(len(y))) if groups is None else groups
    scores=[]
    for lam in grid:
        errors=[]
        for group in dict.fromkeys(groups):
            ids=[j for j in range(len(y)) if groups[j]!=group]
            held=[j for j in range(len(y)) if groups[j]==group]
            p,_,_=ridge([x[j] for j in ids],[y[j] for j in ids],[x[j] for j in held],lam)
            errors.append(st.mean((float(v)-y[j])**2 for j,v in zip(held,p)))
        scores.append((st.mean(errors),lam))
    return min(scores)[1]


def prediction_analysis(out, samples, descriptors):
    index={(r["source"],r["case"]): r for r in summarize([r for r in samples if r["scope"] == "block"], ("source","case"))}
    old=[c for c in descriptors if c["source"] == "out11" and c["scope"] == "block"]
    new=[c for c in descriptors if c["source"] == "out12" and c["scope"] == "block"]
    tests=[c for c in new if c["case"] in ("S10R2","S8R4","S10L2","S10R1L1")]
    if len(tests) != 4: return [], [], {}, []
    costs={(int(r["heads"]),r["kind"]):r for r in csv.DictReader((ROOT / "out/10/cost.csv").open()) if r["scope"] == "attention"}
    models=("single_head", "homogeneous_12", "group", "protocol_features")
    predictions=[];fits={};coefficients=[]
    for target, metric in (("seconds_median","seconds"),("payload_mib_median","mib")):
        base=index[("out11","S12")][target]
        drift=index[("out12","S12")][target]-base if ("out12","S12") in index else 0.
        for model in models:
            x=[features(c,model) for c in old] if model in ("group","protocol_features") else None
            y=[index[("out11",c["case"])][target] for c in old]
            if x is not None:
                lam=choose_lambda(x,y)
                _,b,intercept=ridge(x,y,[x[0]],lam)
                fits[(metric,model)]=(x,y,lam)
                names=("intercept","relu_heads","linear_heads","extra_type_groups") if model == "group" else ("intercept","relu_heads","linear_heads","relu_group_present","extra_normalization_groups","gather_score_volume","extra_AV_calls")
                coefficients.extend(dict(metric=metric,model=model,feature=n,coefficient=float(intercept if i==0 else b[i]),lambda_value=lam,design_rank=int(np.linalg.matrix_rank(x)),training_configs=len(x)) for i,n in enumerate(names))
            for c in tests:
                if x is None:
                    anchor=1 if model == "single_head" else 12
                    p=base+sum(c["types"].count(t)*(float(costs[(anchor,t)][metric])-float(costs[(anchor,"S")][metric]))/anchor for t in "SRL")
                else:
                    p=float(ridge(x,y,[features(c,model)],lam)[0][0])
                actual=index[("out12",c["case"])][target]
                predictions.append(dict(evaluation="new_configuration_holdout",metric=metric,model=model,case=c["case"],
                    prediction=p+drift,uncalibrated_prediction=p,control_intercept_shift=drift,actual=actual,
                    error=p+drift-actual,error_pct=100*(p+drift-actual)/actual))
            # Configuration holdout, not random row split. Pure controls stay in
            # training. Inner CV chooses the penalty without the outer case.
            for i,c in enumerate(old):
                if len(set(c["types"])) == 1: continue
                if x is None:
                    anchor=1 if model == "single_head" else 12
                    p=base+sum(c["types"].count(t)*(float(costs[(anchor,t)][metric])-float(costs[(anchor,"S")][metric]))/anchor for t in "SRL")
                else:
                    ids=[j for j in range(len(old)) if j!=i]
                    tx=[x[j] for j in ids];ty=[y[j] for j in ids]
                    penalty=choose_lambda(tx,ty)
                    p=float(ridge(tx,ty,[x[i]],penalty)[0][0])
                actual=index[("out11",c["case"])][target]
                predictions.append(dict(evaluation="historical_mixture_nested_LOCO",metric=metric,model=model,case=c["case"],
                    prediction=p,uncalibrated_prediction=p,control_intercept_shift=0.,actual=actual,error=p-actual,error_pct=100*(p-actual)/actual))
            # A stricter sensitivity check keeps all layouts of an identical
            # composition together; ordinary LOCO can retain other S6R6 paths.
            compositions=[tuple(c["types"].count(t) for t in "SRL") for c in old]
            for group in dict.fromkeys(compositions):
                held=[i for i,g in enumerate(compositions) if g==group]
                if len(set(old[held[0]]["types"]))==1: continue
                ids=[j for j,g in enumerate(compositions) if g!=group]
                if x is not None:
                    tx=[x[j] for j in ids];ty=[y[j] for j in ids]
                    penalty=choose_lambda(tx,ty,[compositions[j] for j in ids])
                    forecast=ridge(tx,ty,[x[j] for j in held],penalty)[0]
                for pos,i in enumerate(held):
                    c=old[i]
                    if x is None:
                        anchor=1 if model=="single_head" else 12
                        p=base+sum(c["types"].count(t)*(float(costs[(anchor,t)][metric])-float(costs[(anchor,"S")][metric]))/anchor for t in "SRL")
                    else: p=float(forecast[pos])
                    actual=index[("out11",c["case"])][target]
                    predictions.append(dict(evaluation="historical_composition_nested_CV",metric=metric,model=model,case=c["case"],
                        prediction=p,uncalibrated_prediction=p,control_intercept_shift=0.,actual=actual,error=p-actual,error_pct=100*(p-actual)/actual))
    metrics=[]
    groups=defaultdict(list)
    for p in predictions: groups[(p["evaluation"],p["metric"],p["model"])].append(p)
    for (evaluation,metric,model),values in groups.items():
        e=[p["error"] for p in values]
        composition_lookup={c["case"]:tuple(c["types"].count(t) for t in "SRL") for c in old+tests}
        metrics.append(dict(evaluation=evaluation,metric=metric,model=model,configs=len(e),independent_compositions=len({composition_lookup[p["case"]] for p in values}),mae=st.mean(abs(v) for v in e),
                            rmse=math.sqrt(st.mean(v*v for v in e)),mape_pct=st.mean(abs(p["error_pct"]) for p in values),signed_error=st.mean(e)))
    save_rows(out / "model_predictions.csv",predictions);save_rows(out / "model_errors.csv",metrics)
    save_rows(out / "model_coefficients.csv",coefficients)
    return predictions,metrics,fits,coefficients


def additive_components(roles, descriptors):
    """Descriptive, fixed-role mean differences at the same HLO level."""
    result=[]
    for rank in (0,1):
        pure={t:[r for r in roles if r["scope"]=="block" and r["source"]=="out11" and r["case"]==t+"12" and r["party"]==rank] for t in "SRL"}
        if not all(pure.values()): continue
        control=[r for r in roles if r["scope"]=="block" and r["source"]=="out12" and r["case"]=="S12" and r["party"]==rank]
        for c in descriptors:
            if c["scope"]!="block": continue
            values=[r for r in roles if r["scope"]=="block" and r["source"]==c["source"] and r["case"]==c["case"] and r["party"]==rank]
            keys=[k+"_s" for k in KEYS]+[n+"_seconds" for n in ("pphlo.gather","pphlo.greater","pphlo.reduce","pphlo.reciprocal","pphlo.multiply")]
            for key in keys:
                shift=st.mean(r[key] for r in control)-st.mean(r[key] for r in pure["S"]) if control and c["source"]=="out12" else 0.
                prediction=sum(c["types"].count(t)*st.mean(r[key] for r in pure[t])/12 for t in "SRL")+shift
                actual=st.mean(r[key] for r in values)
                result.append(dict(source=c["source"],case=c["case"],party=rank,component=key,actual_mean_s=actual,
                    homogeneous_weighted_mean_s=prediction,control_shift_s=shift,excess_s=actual-prediction,
                    accounting_level="disjoint HLO category" if key in [k+"_s" for k in KEYS] else "individual HLO; nested in category above"))
    return result


def full_operator_differences(operations):
    rows=[]
    full=[r for r in operations if r["scope"]=="vit" and r["operation"].startswith("pphlo.")]
    if not full: return rows
    for rank in (0,1):
        for first,second in (("E0","E1"),("E0","E2"),("E0","E3"),("E1","E3")):
            a=[r for r in full if r["party"]==rank and r["case"]==first]
            b=[r for r in full if r["party"]==rank and r["case"]==second]
            if not a or not b: continue
            na=len({r["round"] for r in a});nb=len({r["round"] for r in b})
            for name in sorted({r["operation"] for r in a+b}):
                aa=[r for r in a if r["operation"]==name];bb=[r for r in b if r["operation"]==name]
                at=sum(r["seconds"] for r in aa)/na;bt=sum(r["seconds"] for r in bb)/nb
                ac=sum(r["calls"] for r in aa)/na;bc=sum(r["calls"] for r in bb)/nb
                rows.append(dict(first=first,second=second,party=rank,operation=name,first_repeats=na,second_repeats=nb,
                    first_mean_s=at,second_mean_s=bt,delta_s=bt-at,first_calls=ac,second_calls=bc,
                    first_sent_mib=sum(r["sent_mib"] for r in aa)/na,second_sent_mib=sum(r["sent_mib"] for r in bb)/nb,
                    same_call_count=ac==bc,interpretation="Observed inclusive HLO delta; common-operator drift is not proved to be caused by grouping"))
    return rows


def full_component_contrasts(roles):
    """One non-overlapping accounting level; no causal attribution of drift."""
    full=[r for r in roles if r['scope']=='vit'];result=[]
    for rank in (0,1):
        for first,second in (("E0","E1"),("E0","E2"),("E0","E3"),("E1","E3")):
            a=[r for r in full if r['party']==rank and r['case']==first]
            b=[r for r in full if r['party']==rank and r['case']==second]
            if not a or not b:continue
            total=st.mean(r['seconds'] for r in b)-st.mean(r['seconds'] for r in a)
            deltas=[]
            for component in KEYS:
                at=st.mean(r[component+'_s'] for r in a);bt=st.mean(r[component+'_s'] for r in b)
                delta=bt-at;deltas.append(delta)
                result.append(dict(first=first,second=second,party=rank,component=component,
                    first_mean_s=at,second_mean_s=bt,delta_s=delta,role_total_delta_s=total,
                    first_repeats=len(a),second_repeats=len(b),
                    interpretation='Disjoint inclusive HLO category; common-operator drift is not proved to be caused by grouping'))
            if abs(sum(deltas)-total)>1e-8:raise RuntimeError('Full component contrasts do not close')
    return result


def archived_full_audit(out):
    """Reparse original logs, including dotted HLO names and fixed roles."""
    samples=[];operations=[];roles=[];plans=[]
    for label in ("S","r48"):
        folder=ROOT / f"out/10/log/full/log/cpu-{label}-joint"
        m=json.loads((folder / "run.json").read_text())
        heads=m['config']['head_plan']['heads'] if m['config'].get('head_plan') else [list('S'*12) for _ in range(12)]
        for layer,types in enumerate(heads):
            plans.append(dict(plan=label,layer=layer,types=''.join(types),softmax_heads=types.count('S'),
                relu_heads=types.count('R'),linear_heads=types.count('L'),type_groups=len(set(types))))
        for repeat,measurement in enumerate(m["measurements"],1):
            pair=[profile(folder / "log" / f"b{repeat}-p{rank}.log") for rank in (0,1)]
            samples.append(dict(plan=label,repeat=repeat,seconds=max(p["seconds"] for p in pair),
                payload_mib=sum(p["sent_mib"] for p in pair),numerical_pass=measurement["numerical_pass"],
                native_sha256=m["library"]["sha256"],metadata=str(folder / "run.json")))
            for rank,p in enumerate(pair):
                seconds={k:0. for k in KEYS}
                for name,op in p["operations"].items():
                    operations.append(dict(plan=label,repeat=repeat,party=rank,operation=name,**op))
                    if name.startswith("pphlo."): seconds[category(name)]+=op["seconds"]
                residual=p["seconds"]-sum(seconds.values())
                if residual < -.01: raise RuntimeError("Archived HLO times do not close")
                seconds["other"]+=residual
                roles.append(dict(plan=label,repeat=repeat,party=rank,seconds=p["seconds"],**{k+"_s":v for k,v in seconds.items()}))
    deltas=[]
    for rank in (0,1):
        for name in sorted({r["operation"] for r in operations if r["operation"].startswith("pphlo.")}):
            a=st.mean(r["seconds"] for r in operations if r["party"]==rank and r["plan"]=="S" and r["operation"]==name) if any(r["party"]==rank and r["plan"]=="S" and r["operation"]==name for r in operations) else 0.
            b=st.mean(r["seconds"] for r in operations if r["party"]==rank and r["plan"]=="r48" and r["operation"]==name) if any(r["party"]==rank and r["plan"]=="r48" and r["operation"]==name for r in operations) else 0.
            deltas.append(dict(party=rank,operation=name,S_mean_s=a,R48_mean_s=b,delta_s=b-a))
    for name,rows in (("archived_full_samples.csv",samples),("archived_full_operations.csv",operations),("archived_full_roles.csv",roles),("archived_R48_deltas.csv",deltas),("archived_head_plans.csv",plans)):
        save_rows(out / name,rows)
    (out / "phase1_time.csv").write_bytes((ROOT / "out/10/time.csv").read_bytes())
    return deltas


def paired_comparisons(rows):
    values={(r["case"],r["round"]):r for r in rows};rng=np.random.default_rng(20261013)
    results=[]
    for a,b in itertools.combinations(("E0","E1","E2","E3"),2):
        ids=sorted(r for c,r in values if c==a and (b,r) in values)
        if not ids: continue
        d=np.array([values[(b,i)]["seconds"]-values[(a,i)]["seconds"] for i in ids])
        bootstrap=rng.choice(d,size=(20000,len(d)),replace=True).mean(axis=1)
        signs=np.array(list(itertools.product((-1,1),repeat=len(d))))
        permutation=np.abs((signs*d).mean(axis=1))
        p=float(np.mean(permutation>=abs(d.mean())-1e-10))
        comm=np.array([values[(b,i)]["payload_mib"]-values[(a,i)]["payload_mib"] for i in ids])
        comm_boot=rng.choice(comm,size=(20000,len(comm)),replace=True).mean(axis=1)
        comm_p=float(np.mean(np.abs((signs*comm).mean(axis=1))>=abs(comm.mean())-1e-10))
        # Fixed Holm correction across six predeclared pairwise comparisons.
        results.append(dict(first=a,second=b,paired_rounds=len(ids),mean_delta_s=float(d.mean()),median_delta_s=float(np.median(d)),
            bootstrap_mean_ci_low_s=float(np.percentile(bootstrap,2.5)),bootstrap_mean_ci_high_s=float(np.percentile(bootstrap,97.5)),
            exact_signflip_two_sided_p=p,
            communication_delta_mib=float(comm.mean()),communication_ci_low_mib=float(np.percentile(comm_boot,2.5)),
            communication_ci_high_mib=float(np.percentile(comm_boot,97.5)),communication_exact_p=comm_p))
    ordered=sorted(range(len(results)),key=lambda i:results[i]["exact_signflip_two_sided_p"]);last=0.
    for rank,i in enumerate(ordered):
        last=max(last,min(1.,results[i]["exact_signflip_two_sided_p"]*(len(results)-rank)))
        results[i]["holm_adjusted_p"]=last
    ordered=sorted(range(len(results)),key=lambda i:results[i]["communication_exact_p"]);last=0.
    for rank,i in enumerate(ordered):
        last=max(last,min(1.,results[i]["communication_exact_p"]*(len(results)-rank)))
        results[i]["communication_holm_p"]=last
    return results


def all_valid_sensitivity(out,rows):
    path=out / 'unpaired_recovered_samples.csv'
    if not path.exists():return
    supplement=list(csv.DictReader(path.open()));result=[]
    for case in ('E0','E1','E2','E3'):
        primary=[r for r in rows if r['case']==case]
        extra=[r for r in supplement if r['case']==case]
        if not primary:continue
        times=[r['seconds'] for r in primary]+[float(r['seconds']) for r in extra]
        communication=[r['payload_mib'] for r in primary]+[float(r['payload_mib']) for r in extra]
        result.append(dict(case=case,primary_calls=len(primary),supplementary_calls=len(extra),all_valid_calls=len(times),
            primary_mean_s=st.mean(r['seconds'] for r in primary),all_valid_mean_s=st.mean(times),
            primary_median_s=st.median(r['seconds'] for r in primary),all_valid_median_s=st.median(times),
            min_s=min(times),max_s=max(times),all_valid_payload_mib_median=st.median(communication),
            interpretation='Descriptive sensitivity only; extra calls are unpaired and reuse the frozen round order; no additional significance test'))
    save_rows(out / 'full_all_valid_sensitivity.csv',result)


def distribution_test(rows,metric="seconds"):
    """Exact Friedman randomization across E1/E2/E3, blocked by round."""
    values={(r["case"],r["round"]):r for r in rows}
    ids=sorted({i for c,i in values if all((name,i) in values for name in ("E1","E2","E3"))})
    if not ids: return []
    observed=[]
    for i in ids:
        a=np.array([values[(name,i)][metric] for name in ("E1","E2","E3")])
        if len(set(a)) != 3: raise RuntimeError("Friedman exact test requires tie correction")
        observed.append(np.argsort(np.argsort(a))+1)
    n=len(ids);ranks=np.sum(observed,axis=0)
    statistic=float(np.sum((ranks-2*n)**2)/n)
    permutations=np.array(list(itertools.permutations((1,2,3))))
    count=extreme=0
    for choice in itertools.product(range(6),repeat=n):
        summed=permutations[list(choice)].sum(axis=0)
        q=float(np.sum((summed-2*n)**2)/n)
        count+=1;extreme+=q>=statistic-1e-10
    return [dict(test="Exact Friedman randomization; equal-24-ReLU plans E1,E2,E3",metric=metric,paired_rounds=n,
                 statistic=statistic,permutations=count,p_value=extreme/count,
                 hypothesis="Any distribution effect within these fixed plans on this host and image")]


def student_t_975(df):
    """Invert the Student t density using deterministic Gaussian quadrature."""
    if df<=0:raise ValueError('Positive residual degrees of freedom required')
    nodes,weights=np.polynomial.legendre.leggauss(96)
    factor=math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(df*math.pi)
    def cdf(x):
        z=x*(nodes+1)/2
        return .5+x/2*float(weights @ (factor*(1+z*z/df)**(-(df+1)/2)))
    low,high=0.,1.
    while cdf(high)<.975:high*=2
    for _ in range(64):
        middle=(low+high)/2
        if cdf(middle)<.975:low=middle
        else:high=middle
    return (low+high)/2


def order_sensitivity(out,rows,schedule):
    lookup={(r['case'],r['round']):r for r in rows}
    rounds=sorted({r for c,r in lookup if all((name,r) in lookup for name in ('E0','E1','E2','E3'))})
    if len(rounds)<3:return
    positions={(c['case'],c['round']):i%4+1 for i,c in enumerate(schedule)}
    names=['intercept']+[f'round_{r}' for r in rounds[1:]]+['E1','E2','E3','position']
    x=[];ys=[]
    for r in rounds:
        for c in ('E0','E1','E2','E3'):
            x.append([1.]+[float(r==j) for j in rounds[1:]]+[float(c==j) for j in ('E1','E2','E3')]+[positions[(c,r)]-2.5])
            ys.append(lookup[(c,r)]['seconds'])
    x=np.asarray(x);y=np.asarray(ys);rank=int(np.linalg.matrix_rank(x));df=len(y)-rank
    b=np.linalg.pinv(x)@y;residual=y-x@b;cov=np.linalg.pinv(x.T@x)*float(residual@residual)/df
    critical=student_t_975(df);result=[]
    for name in ('E1','E2','E3','position'):
        i=names.index(name);se=math.sqrt(max(0.,cov[i,i]))
        result.append(dict(effect=name,estimate_s=float(b[i]),ci_low_s=float(b[i]-critical*se),ci_high_s=float(b[i]+critical*se),
            complete_rounds=len(rounds),residual_df=df,interpretation='Post hoc order sensitivity; statistical association, not measured acceleration'))
    i,j=names.index('E1'),names.index('E3');effect=b[j]-b[i];se=math.sqrt(max(0.,cov[i,i]+cov[j,j]-2*cov[i,j]))
    result.append(dict(effect='E3_minus_E1',estimate_s=float(effect),ci_low_s=float(effect-critical*se),ci_high_s=float(effect+critical*se),
        complete_rounds=len(rounds),residual_df=df,interpretation='Post hoc order sensitivity; statistical association, not measured acceleration'))
    save_rows(out / 'full_order_sensitivity.csv',result)
    save_rows(out / 'full_order_positions.csv',[dict(case=c,mean_position=st.mean(positions[(c,r)] for r in rounds),rounds=len(rounds)) for c in ('E0','E1','E2','E3')])


def full_predictions(out, samples, descriptors, fits):
    full=summarize([r for r in samples if r["scope"] == "vit"], ("case",))
    if len(full)!=4: return []
    v={r["case"]:r for r in full};blocks={(r["source"],r["case"]):r for r in summarize([r for r in samples if r["scope"]=="block"], ("source","case"))}
    if ("out12","S10R2") not in blocks: return []
    plans=json.loads((out / "design.json").read_text())["plans"]
    costs={(int(r["heads"]),r["kind"]):r for r in csv.DictReader((ROOT / "out/10/cost.csv").open()) if r["scope"] == "attention"}
    result=[]
    for metric,target in (("seconds","seconds_median"),("mib","payload_mib_median")):
        for model in ("single_head","homogeneous_12","group","protocol_features","measured_block_transfer"):
            for case in ("E1","E2","E3"):
                delta=0.
                for types in plans[case]["heads"]:
                    if set(types)=={"S"}: continue
                    c=dict(types="".join(types),layout="joint",organization="gather")
                    s=dict(types="S"*12,layout="joint",organization="native")
                    if model in ("single_head","homogeneous_12"):
                        anchor=1 if model=="single_head" else 12
                        d=sum(types.count(t)*(float(costs[(anchor,t)][metric])-float(costs[(anchor,"S")][metric]))/anchor for t in "SRL")
                    elif model=="measured_block_transfer":
                        source,label=("out11","R12") if set(types)=={"R"} else ("out12","SR6") if types.count("R")==6 else ("out12","S10R2")
                        d=blocks[(source,label)][target]-blocks[(source,"S12")][target]
                    else:
                        x,y,lam=fits[(metric,model)]
                        p,_,_=ridge(x,y,[features(c,model),features(s,model)],lam)
                        d=float(p[0]-p[1])
                    delta+=d
                prediction=v["E0"][target]+delta;actual=v[case][target]
                result.append(dict(metric=metric,model=model,case=case,E0_cold_anchor=v["E0"][target],block_delta_sum=delta,
                    prediction=prediction,actual=actual,actual_delta=actual-v["E0"][target],
                    error=prediction-actual,error_pct=100*(prediction-actual)/actual,
                    evaluation="transfer across scope; E0 is measured; training on first Block only"))
    save_rows(out / "full_model_predictions.csv",result)
    contrasts=[]
    for metric in ("seconds","mib"):
        for model in ("single_head","homogeneous_12","group","protocol_features","measured_block_transfer"):
            lookup={r["case"]:r for r in result if r["model"]==model and r["metric"]==metric}
            for a,b in itertools.combinations(("E1","E2","E3"),2):
                predicted=lookup[b]["prediction"]-lookup[a]["prediction"]
                actual=lookup[b]["actual"]-lookup[a]["actual"]
                contrasts.append(dict(metric=metric,model=model,first=a,second=b,predicted_contrast=predicted,
                                      measured_contrast=actual,contrast_error=predicted-actual))
    save_rows(out / "full_prediction_contrasts.csv",contrasts)
    scores=[]
    for metric in ('seconds','mib'):
        for model in ('single_head','homogeneous_12','group','protocol_features','measured_block_transfer'):
            p=[r for r in result if r['metric']==metric and r['model']==model]
            c=[r for r in contrasts if r['metric']==metric and r['model']==model]
            errors=[r['error'] for r in p]
            scores.append(dict(metric=metric,model=model,configs=len(p),mae=st.mean(abs(e) for e in errors),
                rmse=math.sqrt(st.mean(e*e for e in errors)),mape_pct=st.mean(abs(r['error_pct']) for r in p),
                contrast_mae=st.mean(abs(r['contrast_error']) for r in c),
                predicted_pairwise_ties=sum(abs(r['predicted_contrast'])<1e-9 for r in c),
                correct_strict_pairwise_order=sum(r['predicted_contrast']*r['measured_contrast']>0 and abs(r['predicted_contrast'])>=1e-9 for r in c)))
    save_rows(out / 'full_model_errors.csv',scores)
    return result


def write_csv(out, name, rows):
    if rows:
        columns=list(dict.fromkeys(k for r in rows for k in r))
        save_rows(out / name,[{k:r.get(k,"") for k in columns} for r in rows])


def cached_parse(out,folder,label,source,scope,round_number):
    # Completed evidence is immutable. SHA256 guards the metadata and both
    # native traces; cached validation can be reused when only report code changes.
    m=json.loads((folder / "run.json").read_text())
    files=[folder / "run.json"]
    for sample in range(1,len(m["measurements"])+1):
        execution=sample+1 if scope in ("core","block") else 1
        files.extend(folder / "log" / f"native-p{rank}-e{execution}.tsv" for rank in (0,1))
    version=hashlib.sha256(b"".join(marshal.dumps(f.__code__) for f in (parse_case,category,intersection_seconds,load_events,union_seconds,dependency_depth))).hexdigest()
    signature=dict(parser=version,files={str(p.relative_to(ROOT)):digest(p) for p in files},
                   label=label,source=source,scope=scope,round=round_number)
    directory=out / ".parse-cache";directory.mkdir(exist_ok=True)
    cache=directory / ("-".join(folder.relative_to(ROOT).parts)+".json")
    if cache.exists():
        saved=json.loads(cache.read_text())
        if saved["signature"]==signature: return saved["data"]
    data=parse_case(folder,label,source,scope,round_number)
    write(cache,dict(signature=signature,data=data))
    return data


def collect(out, partial=False, include_core=False):
    all_rows=[[],[],[],[],[],[],[]]
    def add(folder,label,source,scope,round_number=0):
        data=cached_parse(out,folder,label,source,scope,round_number)
        for dest,items in zip(all_rows[:5],data[:5]): dest.extend(items)
        all_rows[5].append(data[5])
        all_rows[6].extend(data[6])
        print(f"Audited {source}/{scope}/{label}",flush=True)
    old=ROOT / "out/11"
    for c in json.loads((old / "block_cases.json").read_text()): add(old / "blocks/log" / c["case"],c["case"],"out11","block")
    if include_core:
        for c in json.loads((old / "cases.json").read_text()): add(old / "micro/log" / c["case"],c["case"],"out11","core")
    design=json.loads((out / "design.json").read_text())
    archived_full_audit(out)
    for c in design["blocks"]:
        folder=out / "blocks/log" / c["case"]
        if (folder / "run.json").exists() and json.loads((folder / "run.json").read_text())["status"]=="complete": add(folder,c["case"],"out12","block")
        elif not partial: raise RuntimeError(f"Missing Block {c['case']}")
    schedule=json.loads((out / "recovery.json").read_text())["primary_full_schedule"] if (out / "recovery.json").exists() else design["full_schedule"]
    for c in schedule:
        folder=out / "full/log" / c["folder"]
        if (folder / "run.json").exists() and json.loads((folder / "run.json").read_text())["status"] in ("complete","matched","outside_gate"):
            add(folder,c["case"],"out12","vit",c["round"])
        elif not partial: raise RuntimeError(f"Missing ViT {c['folder']}")
    samples,roles,ops,spans,audits,descriptors,layers=all_rows
    # Directly compare the computation/environment identities, including the
    # shared Block input, private weights and native binary in old/new evidence.
    block_descriptors=[c for c in descriptors if c["scope"]=="block"]
    invariant_keys=("library_sha256","checkpoint_sha256","input_sha256","cpus","threads","precision","network","computation_hashes")
    invariant={key:len({json.dumps(c[key],sort_keys=True) for c in block_descriptors})==1 for key in invariant_keys}
    controls=defaultdict(list)
    for c in block_descriptors: controls[c["case"]].append(c["compiled_sha256"])
    invariant["repeated_control_compiled_graph_identical"]=all(len(set(v))==1 for v in controls.values())
    if not all(invariant.values()): raise RuntimeError(f"Unfair Block environment: {invariant}")
    full_descriptors=[c for c in descriptors if c["scope"]=="vit"]
    full_invariants={key:len({json.dumps(c[key],sort_keys=True) for c in full_descriptors})<=1 for key in invariant_keys}
    full_graphs=defaultdict(list)
    for c in full_descriptors: full_graphs[c["case"]].append(c["compiled_sha256"])
    full_invariants["same_plan_compiled_graph_identical"]=all(len(set(v))==1 for v in full_graphs.values())
    if not all(full_invariants.values()): raise RuntimeError(f"Full model environment changed: {full_invariants}")
    write(out / "audit.json",dict(complete=not partial,closure_checks=len(audits),all_closure_checks_passed=True,
        block_environment_invariants=invariant,configurations=descriptors,
        full_environment_invariants=full_invariants,
        source_note="mixed.py report/plot source hash differs between historical and current records; computational sources and repeated controls' compiled executable SHA256 are identical.",
        timing_correction="Both fixed P0 and P1 roles are reported. Old slow-party-selected waiting comparisons confounded roles.",
        timing="All layouts compare max of the two complete native execution intervals. HLO rows are inclusive, not pure CPU.",
        attribution="Link, HE conversion and precision intervals overlap HLO and each other; they are not additive buckets.",
        interaction="Depth is a message dependency lower bound, not exact communication rounds or send-call count."))
    for name,rows in (("samples.csv",samples),("role_decomposition.csv",roles),("operations.csv",ops),("protocol_spans.csv",spans),("closure_checks.csv",audits)):
        write_csv(out,name,rows)
    write_csv(out,"layer_attention_spans.csv",layers)
    write_csv(out,"component_additive_residuals.csv",additive_components(roles,descriptors))
    write_csv(out,"full_operator_differences.csv",full_operator_differences(ops))
    write_csv(out,"full_component_contrasts.csv",full_component_contrasts(roles))
    summaries=summarize(samples,("source","scope","case"));role_summaries=summarize(roles,("source","scope","case","party"))
    write_csv(out,"cost_summary.csv",summaries);write_csv(out,"role_summary.csv",role_summaries)
    predictions,metrics,fits,coeff=prediction_analysis(out,samples,descriptors)
    full_pred=full_predictions(out,samples,descriptors,fits) if fits else []
    comparisons=paired_comparisons([r for r in samples if r["scope"]=="vit"])
    write_csv(out,"full_paired_comparisons.csv",comparisons)
    full_samples=[r for r in samples if r["scope"]=="vit"]
    all_valid_sensitivity(out,full_samples)
    order_sensitivity(out,full_samples,schedule)
    tests=distribution_test(full_samples)+distribution_test(full_samples,"payload_mib")
    order=sorted(range(len(tests)),key=lambda i:tests[i]["p_value"]);last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,tests[i]["p_value"]*(len(tests)-rank)))
        tests[i]["holm_two_outcomes_p"]=last
    write_csv(out,"full_distribution_test.csv",tests)
    return samples,roles,ops,spans,descriptors,summaries,role_summaries,predictions,metrics,full_pred,comparisons


def figures(out, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    samples,roles,ops,spans,descriptors,summaries,role_summaries,predictions,metrics,full_pred,comparisons=data
    plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"ps.fonttype":42})
    def save(fig,name):
        for suffix in ("pdf","png"): fig.savefig(out / f"{name}.{suffix}",dpi=200,bbox_inches="tight")
        plt.close(fig)
    blocks=[c for c in descriptors if c["scope"]=="block"]
    names=[("out11",c) for c in ("S12","R12","L12","SR1","SR6","SR10","SL6","SRL633","SR6-slices","SR6-split")]
    names += [("out12",c) for c in ("S10R2","S8R4","S10L2","S10R1L1") if any(d["source"]=="out12" and d["case"]==c for d in blocks)]
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,layout="constrained");x=np.arange(len(names))
    for rank,ax in enumerate(axes[:2]):
        bottom=np.zeros(len(names))
        for key,color,label in zip(KEYS,COLORS,LABELS):
            values=np.array([st.mean(r[key+"_s"] for r in roles if r["scope"]=="block" and r["source"]==source and r["case"]==name and r["party"]==rank) for source,name in names])
            ax.bar(x,values,bottom=bottom,color=color,label=label);bottom+=values
        ax.set_ylabel(f"P{rank} inclusive time (s)")
    bottom=np.zeros(len(names))
    for key,color,label in zip(KEYS,COLORS,LABELS):
        values=np.array([sum(st.mean(r[key+"_sent_mib"] for r in roles if r["scope"]=="block" and r["source"]==source and r["case"]==name and r["party"]==rank) for rank in (0,1)) for source,name in names])
        axes[2].bar(x,values,bottom=bottom,color=color);bottom+=values
    axes[2].set_ylabel("Both-party payload (MiB)");axes[2].set_xticks(x,[n+("*" if s=="out12" else "") for s,n in names],rotation=35,ha="right")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncols=4)
    fig.suptitle("A. Complete Block: role-consistent operator and payload accounting")
    fig.supxlabel("Five real calls per configuration. * New held-out configuration. Nested conversion/wait spans are reported separately.",fontsize=8)
    save(fig,"figure_A_block")
    if predictions:
        held=[p for p in predictions if p["evaluation"]=="new_configuration_holdout" and p["metric"]=="seconds"]
        fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
        cases=("S10R2","S8R4","S10L2","S10R1L1");x=np.arange(4);w=.16
        models=("single_head","homogeneous_12","group","protocol_features")
        labels=("1-head additive","12-head additive","Type-group aware","Protocol features")
        actual=[next(p["actual"] for p in held if p["case"]==c) for c in cases]
        axes[0].bar(x-2*w,actual,w,label="Measured",color="#222222")
        quantiles=np.array([np.percentile([r["seconds"] for r in samples if r["scope"]=="block" and r["source"]=="out12" and r["case"]==c],[25,75]) for c in cases])
        axes[0].errorbar(x-2*w,actual,yerr=np.array([np.array(actual)-quantiles[:,0],quantiles[:,1]-np.array(actual)]),fmt="none",ecolor="black",capsize=3,lw=1)
        for i,(m,label,color) in enumerate(zip(models,labels,("#d6a347","#c96262","#448cc3","#469681"))):
            p=[next(p["prediction"] for p in held if p["case"]==c and p["model"]==m) for c in cases]
            axes[0].bar(x+(i-1)*w,p,w,label=label,color=color)
            axes[1].plot(x,[100*(v-a)/a for v,a in zip(p,actual)],"o-",label=label,color=color)
        axes[0].set_ylabel("Complete native Block latency (s)");axes[1].set_ylabel("Prediction error (%)");axes[1].axhline(0,color="black",lw=.6)
        for ax in axes: ax.set_xticks(x,cases,rotation=20)
        fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncols=3)
        fig.supxlabel("Four configurations held out; bars show medians, whiskers IQR. Only S12 adjusts the intercept. Predictions are not speedups.",fontsize=8)
        save(fig,"figure_B_predictions")
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout="constrained")
    for rank,ax in enumerate(axes):
        keys=("recv_blocking_union_s","dot_decrypt_moddown_inclusive_s","dot_H2A_union_s")
        for key,label,color in zip(keys,("Recv blocking union","Decrypt/moddown region (inclusive)","Dot H2A region"),("#4078a5","#ce5b5b","#50a388")):
            values=[st.mean(r[key] for r in roles if r["source"]=="out11" and r["scope"]=="block" and r["case"]==c and r["party"]==rank) for c in ("SR6","SR6-slices","SR6-split")]
            ax.plot(range(3),values,"o-",label=label,color=color)
        ax.set_title(f"Fixed P{rank} role");ax.set_xticks(range(3),("joint/gather","joint/slices","split/gather"),rotation=15);ax.set_ylabel("Observed union (s)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncols=3)
    fig.supxlabel("Historical traces re-audited at fixed party roles. These nested regions overlap and cannot be summed.",fontsize=8)
    save(fig,"figure_C_wait_audit")
    full=[r for r in samples if r["scope"]=="vit"]
    if len({r["case"] for r in full})==4:
        names=("E0","E1","E2","E3");x=np.arange(4)
        fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout="constrained")
        means=[st.mean(r["seconds"] for r in full if r["case"]==n) for n in names]
        deviations=[st.stdev(r['seconds'] for r in full if r['case']==n) if sum(r['case']==n for r in full)>1 else 0. for n in names]
        rounds=sorted({r["round"] for r in full})
        for i in rounds:
            a=[next((r["seconds"] for r in full if r["case"]==n and r["round"]==i),np.nan) for n in names]
            axes[0].plot(x,a,"o-",lw=.8,ms=3,alpha=.6,color="#333333")
        axes[0].errorbar(x,means,yerr=deviations,fmt='D',color='black',ms=5,capsize=3,label='Mean ± SD')
        axes[0].legend(fontsize=8)
        axes[0].set_ylabel("Cold complete ViT execution (s)")
        comm=[st.mean(r["payload_mib"] for r in full if r["case"]==n)/1024 for n in names]
        axes[1].bar(x,comm,color=("#697b8d","#50a388","#4078a5","#ce5b5b"));axes[1].set_ylabel("Actual two-party payload (GiB)")
        for i,v in enumerate(comm):axes[1].text(i,v+.08,f'{v:.4f}',ha='center',fontsize=8)
        for ax in axes[:2]: ax.set_xticks(x,("E0\nall S","E1\n2 × R12","E2\n4 × S6R6","E3\n12 × S10R2"),fontsize=8)
        relative=[(comm[i]-comm[1])*1024 for i in (1,2,3)]
        axes[2].bar(range(3),relative,color=("#50a388","#4078a5","#ce5b5b"))
        axes[2].set_xticks(range(3),('E1','E2','E3'));axes[2].set_ylabel('Payload increase relative to E1 (MiB)')
        for i,v in enumerate(relative):axes[2].text(i,v+.3,f'{v:.2f}',ha='center',fontsize=8)
        fig.suptitle("D. Full ViT: baseline and three allocations of 24 ReLU heads")
        fig.supxlabel("Means of real fresh-runtime calls; lines join the same round. Different head assignments have different plaintext functions.",fontsize=8)
        save(fig,"figure_D_full_vit")
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout="constrained")
        for rank,ax in enumerate(axes):
            bottom=np.zeros(4)
            for key,color,label in zip(KEYS,COLORS,LABELS):
                values=np.array([st.mean(r[key+"_s"] for r in roles if r["scope"]=="vit" and r["case"]==n and r["party"]==rank) for n in names])
                ax.bar(x,values,bottom=bottom,label=label,color=color);bottom+=values
            ax.set_ylabel(f"P{rank} inclusive time (s)")
        axes[1].set_xticks(x,names);fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncols=4)
        fig.suptitle("E. Cross-Block accumulation: complete ViT operator accounting")
        save(fig,"figure_E_full_decomposition")
        if full_pred:
            fig,ax=plt.subplots(figsize=(10,5),layout="constrained");x=np.arange(3)
            cases=("E1","E2","E3")
            anchor=next(r["E0_cold_anchor"] for r in full_pred if r["metric"]=="seconds")
            actual=[next(r["actual_delta"] for r in full_pred if r["metric"]=="seconds" and r["case"]==c) for c in cases]
            ax.plot(x,actual,"ko-",lw=2.5,label="Measured median delta")
            for model,label,color in zip(("single_head","homogeneous_12","group","protocol_features","measured_block_transfer"),
                    ("1-head additive","12-head additive","Type-group aware","Protocol features","Measured Block transfer"),
                    ("#d6a347","#c96262","#448cc3","#469681","#7353a0")):
                values=[next(r["block_delta_sum"] for r in full_pred if r["metric"]=="seconds" and r["model"]==model and r["case"]==c) for c in cases]
                ax.plot(x,values,"o--",label=label,color=color,lw=1.3)
            ax.axhline(0,color="black",lw=.7);ax.set_ylabel("Full-model latency delta relative to E0 (s)")
            ax.set_xticks(x,("E1: 2 × R12","E2: 4 × S6R6","E3: 12 × S10R2"))
            fig.legend(*ax.get_legend_handles_labels(),loc="outside upper center",ncols=3)
            fig.supxlabel("Full-model deltas are real measurements. All model curves are predictions anchored on the measured cold E0.",fontsize=8)
            save(fig,"figure_F_full_prediction")
    group_path=out / 'group_hlo_samples.csv'
    if group_path.exists():
        group_rows=list(csv.DictReader(group_path.open()))
        full_group=[r for r in group_rows if r['scope']=='vit' and r['party']=='0']
        if full_group:
            fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
            comp=[r for r in full_group if r['component']=='relu_score_compare']
            sizes=sorted({int(r['group_heads']) for r in comp})
            def runtime_means(values,per_head=False):
                clusters=defaultdict(list)
                for r in values:
                    clusters[(r['case'],r['round'])].append(float(r['seconds'])/(int(r['group_heads']) if per_head else 1))
                return [st.mean(a) for a in clusters.values()]
            per_size=[runtime_means([r for r in comp if int(r['group_heads'])==n],True) for n in sizes]
            means=[st.mean(a) for a in per_size];sd=[st.stdev(a) if len(a)>1 else 0. for a in per_size]
            axes[0].errorbar(sizes,means,yerr=sd,fmt='o-',capsize=3,color='#ce5b5b');axes[0].set_xticks(sizes)
            axes[0].set_xlabel('ReLU heads per group');axes[0].set_ylabel('Score-comparison time per head (s)')
            for kind,color in (('S','#4078a5'),('R','#50a388')):
                inv=[r for r in full_group if r['component']=='reciprocal' and r['group_kind']==kind]
                ns=sorted({int(r['group_heads'])*197 for r in inv})
                runs=[runtime_means([r for r in inv if int(r['group_heads'])*197==n]) for n in ns]
                ys=[st.mean(a) for a in runs];sd=[st.stdev(a) if len(a)>1 else 0. for a in runs]
                axes[1].errorbar(ns,ys,yerr=sd,fmt='o-',capsize=3,color=color,label=kind)
            axes[1].set_xlabel('Actual denominator elements per call');axes[1].set_ylabel('Inclusive reciprocal call time (s)');axes[1].legend(title='Type')
            cases=('E1','E2','E3');x=np.arange(3);bottom=np.zeros(3)
            for component,label,color in (('head_select','Score selection','#65727d'),('head_restore','Head restoration','#adbac5')):
                vals=[]
                for c in cases:
                    selected=[r for r in full_group if r['case']==c and r['component']==component]
                    count=len({r['round'] for r in selected})
                    vals.append(sum(float(r['seconds']) for r in selected)/count if count else 0.)
                axes[2].bar(x,vals,bottom=bottom,label=label,color=color);bottom+=np.array(vals)
            axes[2].set_xticks(x,('E1\n2 changed layers','E2\n4 changed layers','E3\n12 changed layers'))
            axes[2].set_ylabel('Gather time per complete model (s)');axes[2].legend(fontsize=8)
            fig.suptitle('G. Real group-size costs and accumulation of public indexing')
            fig.supxlabel('P0 fixed role; real native intervals and exact compiler shapes. Whiskers: SD of fresh-runtime means, grouping correlated calls within a run.',fontsize=8)
            save(fig,'figure_G_group_shapes')


def recovery_note(out):
    path=out / 'recovery.json'
    if not path.exists():return ''
    recovered=json.loads(path.read_text());n=recovered['supplemental_verified_calls']
    return f'会话中断涉及的配对轮次按原头计划和顺序完整重跑；原执行的 {n} 次有效调用保留为补充数据，不混入主统计。日志收尾中断的调用按原输出、双方原生时间区间及字节轨迹重新核验，没有填造缺失的后端 stdout 指标。恢复计划和原始记录均保留。'


def report(out, data, partial=False):
    samples,roles,ops,spans,descriptors,summaries,role_summaries,predictions,metrics,full_pred,comparisons=data
    def table(rows, columns):
        def val(v): return f"{v:.3f}" if isinstance(v,float) else str(v)
        return '<table><thead><tr>'+''.join('<th>'+html.escape(n)+'</th>' for n in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(val(r.get(n,'')))+'</td>' for n in columns)+'</tr>' for r in rows)+'</tbody></table>'
    blocks=[r for r in summaries if r["scope"]=="block"]
    full=[r for r in summaries if r["scope"]=="vit"]
    numerical=dict(calls=len(samples),new_block_calls=sum(r["source"]=="out12" and r["scope"]=="block" for r in samples),
                   new_full_calls=sum(r["scope"]=="vit" for r in samples),all_pass=True,
                   worst_relative_l2=max(r["relative_l2"] for r in samples),worst_max_abs=max(r["max_abs"] for r in samples))
    conclusions=dict(complete=not partial,numerical=numerical,block=blocks,full=full,model_errors=metrics,full_comparisons=comparisons,
        protocol="Existing CPU BumbleBee; no new packing or conversion protocol; native binary preserved",
        uncertainty=["Inclusive HLO time includes protocol computation, precision processing and blocking; no independent pure CPU/network partition.",
                     "HE conversion spans are nested; Dot H2A is real, broad decrypt/mask regions are not pure conversion.",
                     "First Block and full ViT have different warm/cold scopes and activations; additive transfer is a prediction, not measured acceleration.",
                     "Diagnostics have no accuracy budget. Numerical agreement with each plan's plaintext reference is not ImageNet accuracy validation.",
                     "Small configuration set at one precision/network/hardware setting; fitted coefficients are predictive associations, not causal protocol costs.",
                     "Six interleaved complete-model runs per plan; exact sign-flip p-values use round exchangeability, Holm corrects six pairs."])
    write(out / "findings.json",conclusions)
    audit=json.loads((out / "audit.json").read_text())
    protocol=json.loads((ROOT / "out/11/protocol_audit.json").read_text())
    protocol["scope"]="10 historical complete Blocks, 8 new complete Block configurations and 4 full ViT plans; newly authorized complete-model diagnosis."
    protocol["wait_correction"]="Fixed P0/P1 role analysis replaces comparisons that switch to whichever party is slower."
    protocol["gather_implementation"]="Public indices; generic Gather iterates output windows and uses scalar slice/update; existing slices avoid this interpreter indexing path."
    protocol["sources"]["libspu/kernel/hlo/indexing.cc"]=digest(ROOT / "build/src/libspu/kernel/hlo/indexing.cc")
    write(out / "protocol_audit.json",protocol)
    title="BumbleBee-ViT 混合头安全成本：复核与完整模型对照"
    body='<html lang="zh"><meta charset="utf-8"><title>'+title+'</title><style>body{font:15px/1.75 sans-serif;max-width:1250px;margin:32px auto;padding:20px}table{border-collapse:collapse;font-size:12px;margin:18px 0}td,th{border:1px solid #ddd;padding:5px 8px}th{background:#eee}img{max-width:100%}p{max-width:1100px}</style><h1>'+title+'</h1>'
    body+='<p>'+('本报告是执行中的中间核查；完整模型结论尚未完成。' if partial else '以下性能来自实际两方执行，预测与区间重分类单独标识。')+'</p>'
    if recovery_note(out):body+='<p>'+recovery_note(out)+'</p>'
    body+='<h2>实验范围与公平性</h2><p>复用 out/10 的单头/12 头微基准及 S/R48 完整模型历史结果、out/11 的 10 组完整第一 Block。新增 S10R2、S8R4、S10L2、S10R1L1，及 S12、S6R6 的 joint/gather、joint/slices、split/gather 对照，每组一次预热后 5 次真实调用。完整 ViT E0–E3 每组 6 次独立冷启动，按轮随机交错。E1 的 R12 位于第 0、6 层；E2 的 S6R6 位于第 0、3、6、9 层；E3 的 S10R2 位于全部 12 层，均为 24 个 R 头。层和头位置预先固定，未进行准确率筛选。</p>'
    body+='<p>保持原始 checkpoint、图片、S/R/L 计算定义、FM64/f18、intrinsic GELU FM32/f12、每方 25 线程与 CPU 绑定、1 Gbps / 0.25 ms egress、同一 stage2.so 观察二进制。Block 输入是相同的真实共享中间激活，权重保持私有。原生 all-S 分支及混合分支沿用原实现，因此比较包含该实现路径的差异。没有调整预训练权重、数值阈值或原生打包协议。</p>'
    body+='<p>L 是已有的 score 线性映射，仍计算 QK 和 AV，不具有新的线性注意力架构所声称的复杂度缩减。历史 core 纯类型复测相对原二进制约在 1.2% 内；新完整推理均使用同一观察二进制，不将跨日期的二进制差值解读成观察器开销或实际加速。</p>'
    body+='<p>数值验收：有限值、相对 L2≤0.05、最大绝对误差≤0.5；完整 ViT 另要求安全输出与该头计划的明文参考 Top-1 一致。这是协议数值验证，不是 ImageNet 准确率预算。'+f'已纳入 {numerical["new_block_calls"]} 次新增 Block、{numerical["new_full_calls"]} 次完整模型调用；最大相对 L2={numerical["worst_relative_l2"]:.5f}，最大绝对误差={numerical["worst_max_abs"]:.5f}。'+'</p>'
    body+='<h2>计时口径与等待复核</h2><p>三个布局统一采用 max(TP0,TP1) 的完整 native execution 计时。模型加载、编译、输入上传、输出 reveal 不计入该区间，request_s 单独保留；完整模型冷执行包含协议首次初始化，单 Block 预热后执行不包含该首次初始化。因此不能把单 Block 时间直接乘以 12 当作端到端实测。实际传输字节为两方 YACL 发送 payload 之和，不包含 TCP/IP 头。</p>'
    body+='<p>旧表每次选取较慢一方，joint/slices 的较慢方常切换到 P1。私有权重导致 P0/P1 的协议角色不同，故旧表中“等待明显增加”的跨布局比较存在角色混杂。现在固定角色重算接收/发送阻塞并集及其 HLO 重叠。等待可能来自对方计算，也可与本方其他线程并行，不能解读为纯网络延迟或再次加到 HLO 时间中。</p><img src="figure_C_wait_audit.png">'
    selected=[r for r in role_summaries if r["scope"]=="block" and r["case"] in ("SR6","SR6-slices","SR6-split")]
    body+=table(selected,["source","case","party","seconds_mean","recv_blocking_union_s_mean","dot_decrypt_moddown_inclusive_s_mean","dot_H2A_union_s_mean","pphlo.gather_seconds_mean"])
    body+='<p>主图在同一 HLO 层级分解时间，每方时间与通信量均核对闭合；转换、截断、布尔比较、密文打包和 link spans 作为嵌套诊断单独提供。HLO normalize 类还包含 LayerNorm、乘法和 reduce 等公共算子，不能全部归于头归一化。'+f'全部 {audit["closure_checks"]} 条双方逐次闭合检查通过。'+'</p>'
    body+='<p>显式 comparison 栏不包含 reduce 内部的 max 比较。归一化组数增加是实际调用事实，但额外倒数时间并非每组都为正；例如旧 SR1 的 P0 倒数均值低于纯类型加性预测，同时 reduce 和组织项有正偏差。真实贡献应使用同层级算子差值及数值范围核算，而不能仅凭更多 reciprocal 调用就指定它为主要瓶颈。</p>'
    body+='<h2>完整 Block：真实成本与加性偏差</h2><img src="figure_A_block.png">'
    body+=table(blocks,["source","case","repeats","seconds_median","min_s","max_s","payload_mib_median","dot_H2A_calls_median"])
    body+='<p>实际 Dot H2A 位于矩阵 OLE 内部：joint 每 Block 两方合计 10 次，split 的 S6R6 为 12 次。它不是为不同头类型之间额外插入的转换边界。dot_decrypt_moddown 包含接收、解码、解密和 modulus-down；MulThenResponse/MultiplyThenMask 区域还包含乘法、掩码和响应提交。因此 conversion_calls 和累计 HE 秒数都不能当作独立转换次数/墙钟成本。实际原生 packing 分支保持不变，精度共享区间包含比较内部的 A/B 共享与截断。</p>'
    body+='<p>组织费用也不完全来自类型共存：现有通用路径在纯 R12/L12 中仍有选择/恢复的 identity gather。辅助纯 S attention core 的通用路径中位数为 15.410 s，native 路径为 14.609 s；前者 gather 约 0.683 s、后者为零，明文等价检查通过。故成本模型应区分原生/通用路径、实际组大小与组数，而不能把全部 gather 成本命名为异构转换。</p>'
    body+='<p>实际编译器把单头选择降为 slice，因此 SR1 和 S10R1L1 的 gather 元素量略少。组织特征按现有降级规则修正；这只改变预测描述，没有改变任何执行代码。R 的实际图包含逐分数 greater、i1×f32 掩码乘法、sum、零分母 greater/select、reciprocal 与归一化乘法。不能把类型名或名为 convert 的算子当作新增 HE/MPC 域转换证据。</p>'
    body+='<p>本轮四个新增组合及完整模型 E1/E2/E3 在每层均按 S/R/L 排列，选择的每个组是连续区间，按组拼回后恢复顺序为 identity。通用路径仍执行公开 gather 并复制本地共享数组。因此这些实测组织费用不表示每种异构头计划在语义上都需要重排；group_hlo_samples.csv 标明 selection_is_contiguous 和 restoration_is_identity。原 R48 的选择位置则来自既有头计划，不能把新连续方案的 identity 属性推广到全部历史配置。</p>'
    if (out / 'group_hlo_summary.csv').exists():
        body+='<h2>实际分组算子成本</h2><img src="figure_G_group_shapes.png"><p>14 个不同 Block 图和 4 个完整模型图的离线编译 SHA 与真实执行完全相同，逐条顶层 HLO 顺序也与原生轨迹一致。形状和类型从这些精确对应的图中读取；表中每个时间与字节均来自真实两方执行，离线编译不作为时间样本。完整模型使用联合 AV；Block 的 split 路径另按原始算子表及协议区间核查，不与联合路径的分组小表混淆。</p>'
        grouped=list(csv.DictReader((out / 'group_hlo_summary.csv').open()))
        selected=[r for r in grouped if r['scope']=='vit' and r['party']=='0' and r['component'] in ('relu_score_compare','relu_mask_multiply','softmax_max','reciprocal','head_select','head_restore')]
        body+=table(selected,['case','component','group_kind','group_heads','real_calls','independent_runtimes','mean_call_s','mean_model_total_s'])
        body+='<p>调用数与时间不成固定比例：E0/E1 的每个分母批次为 2364 元素，E2 的混合层为两个 1182，E3 为 1970 与 394。总分母元素数相同，但分组形状、类型及执行状态不同。判断逆运算额外成本应看实际分组区间，而不是仅看 12/16/24 次调用。逐分数比较、max 归约与公开重排的规模效应也分别核查。</p>'
    body+='<h2>预测模型与配置外验证</h2><p>单头模型以全 S Block 为公共基准，加上 Σnt[Ct(1)−CS(1)]；12 头加性模型改用 Σnt[Ct(12)−CS(12)]/12。组数模型为 β0+βR·nR+βL·nL+γ·(类型组数−1)。协议特征模型使用 nR、nL、R 组是否存在、额外归一化组数、公开 gather 元素量、split 的额外 AV 次数。它们是由计划可预先计算的特征，不使用实测等待或目标耗时作为输入。</p>'
    body+='<p>两个拟合模型只用旧 10 个 Block 配置，按配置交叉验证选择岭惩罚；旧混合方案采用嵌套留一配置验证，四个新组合完全留出。新 S12 对照只校正跨轮次的截距，校正量与未校正预测均保留。报告时间及通信 MAE、RMSE、MAPE，不能将小样本拟合参数解释成协议因果成本。slot=4096 来自现有 elementwise Mul 实现，但不同算子有不同打包路径，本轮不把简化 ceil(size/4096) 估计假装成实测密文填充量。</p>'
    body+='<p>另提供更严格的按头类型数量留出验证：同一 S6R6 的三条组织/矩阵路径同时留出，避免普通留一配置仍保留相同组大小的其他路径。误差表同时标明配置数和独立头数量组合数。主模型的四组新配置留出预测固定不变；分组验证只是敏感性分析。</p>'
    if metrics: body+='<img src="figure_B_predictions.png">'+table(metrics,["evaluation","metric","model","configs","independent_compositions","mae","rmse","mape_pct","signed_error"])
    body+='<h2>完整 ViT：相同 24 R 头的层间分布</h2>'
    if full:
        body+='<img src="figure_D_full_vit.png"><img src="figure_E_full_decomposition.png">'
        body+=table(full,["case","repeats","seconds_median","seconds_mean","min_s","max_s","payload_mib_median","request_s_median","compile_s_median"])
        counters=[r for r in role_summaries if r["scope"]=="vit" and r["party"]==0]
        body+=table(counters,["case","f_mmul_calls_mean","f_batch_mmul_calls_mean","f_gelu_calls_mean","f_reciprocal_calls_mean","pphlo.gather_calls_mean","pphlo.gather_seconds_mean","pphlo.greater_seconds_mean","pphlo.reduce_seconds_mean","pphlo.reciprocal_seconds_mean"])
        body+=table(full,['case','dot_H2A_calls_median','interaction_depth_lower_bound_median','matched_messages_median'])
        body+='<p>E0–E3 的完整模型 Dot H2A 次数相同，都是两方合计 123；类型分组没有为联合 AV 新增独立的 HE/MPC 边界。A/B 共享转换和 trunc 的实际调用与区间另列在 protocol_spans.csv，嵌套于原安全算子；不能把它们、解密区间和等待再次相加。上表的 message depth 只是交互依赖下界，matched_messages 也不是精确轮次。</p>'
        body+=table(comparisons,["first","second","paired_rounds","mean_delta_s","bootstrap_mean_ci_low_s","bootstrap_mean_ci_high_s","exact_signflip_two_sided_p","holm_adjusted_p","communication_delta_mib"])
        body+='<p>配对差使用同一轮中的两方案，置信区间为按轮 bootstrap 的均值差区间；精确双侧 sign-flip 只有 6 对，最小未校正 p=0.03125，多重比较使用 Holm 校正。每个方案代表固定输入及固定层/头分布，不代表所有模型/硬件或严格准确率等价；不得仅凭通信下降或单次计时声称显著加速。</p>'
        sensitivity=out / 'full_all_valid_sensitivity.csv'
        if sensitivity.exists():
            body+=table(list(csv.DictReader(sensitivity.open())),['case','primary_calls','supplementary_calls','primary_median_s','all_valid_median_s','primary_mean_s','all_valid_mean_s','min_s','max_s'])
            body+='<p>中断轮次的全部有效执行另加入上述描述性敏感性分析，避免丢弃已完成的实际数据。补充调用的方案数量不平衡，并复用了原冻结顺序，因此不将它们当成新增独立配对轮次提高检验显著性。主图和主统计仍使用原计划的六个完整重建轮次。</p>'
        test_path=out / "full_distribution_test.csv"
        if test_path.exists(): body+=table(list(csv.DictReader(test_path.open())),["test","metric","paired_rounds","statistic","permutations","p_value","holm_two_outcomes_p"])
        body+=table(comparisons,["first","second","communication_delta_mib","communication_ci_low_mib","communication_ci_high_mib","communication_exact_p","communication_holm_p"])
        body+='<p>层间分布的总体检验对 E1/E2/E3 按轮排列三个方案，穷举 Friedman 秩统计量的全部 6⁶=46656 种排列。它检验这三种预先固定分布在此部署下的差异；不能单独证明组数是唯一因果因素。每层 QK 至 AV 的实际事件区间另存 layer_attention_spans.csv，该窗口不等于整个 Block，亦不包含混合分支的 QK 前缩放。</p>'
        order_path=out / 'full_order_sensitivity.csv'
        if order_path.exists():
            body+=table(list(csv.DictReader(order_path.open())),['effect','estimate_s','ci_low_s','ci_high_s','complete_rounds','residual_df'])
            body+='<p>预先随机的执行顺序没有完全平衡各方案的位置；full_order_positions.csv 保留平均顺序位置。上表用轮次固定效应、方案及轮内位置作事后敏感性分析。系数只是统计关联，区间依赖线性顺序漂移假设，不是新增安全执行、纯分组因果效应或实测加速。</p>'
        if full_pred:
            body+='<img src="figure_F_full_prediction.png">'+table(full_pred,["metric","model","case","block_delta_sum","actual_delta","prediction","actual","error_pct"])
            body+=table(list(csv.DictReader((out / 'full_model_errors.csv').open())),['metric','model','mae','rmse','mape_pct','contrast_mae','predicted_pairwise_ties','correct_strict_pairwise_order'])
            body+='<p>整模型预测以实测冷 E0 为基准，加上各层的 Block 替换成本；measured_block_transfer 使用已测 R12、S6R6、S10R2 的差值，是跨 scope 转移检验。不同层的激活和冷初始化可能造成偏差，该预测不是完整推理实测。contrast_mae 衡量 E1/E2/E3 两两差值的预测误差；绝对总耗时 MAPE 较小不代表能正确区分收益相近的方案。顺序列只比较实测中位数的数值顺序，不代表统计显著排序。</p>'
    else: body+='<p>完整模型正在执行，尚不形成时间或通信分布结论。</p>'
    old_findings=json.loads((ROOT / "out/11/findings.json").read_text())
    e=old_findings["archived_R48_explanation"]
    body+='<h2>历史 R48 通信下降但耗时近似不变</h2><p>out/10 的原二进制及三次冷执行单独保留，未与新观察二进制的时间混算。全 S 中位数 402.046 s，R48 为 402.127 s；总通信 11.173→10.033 GiB（下降 10.21%），时间范围重叠。历史算子均值差：'+f'指数+归约省 {e["exp_and_reduce_saved_s"]:.2f} s，比较增 {e["comparison_added_s"]:.2f} s，gather 增 {e["gather_added_s"]:.2f} s，乘法+倒数增 {e["multiply_and_reciprocal_added_s"]:.2f} s。'+'这些抵消了大部分映射收益，公共投影/FFN/GELU/QK/AV 仍占大比例。公共 GELU 的跨运行时间差只作漂移证据，不能归因于混合头。</p>'
    body+='<p>原 R48 每层 R 头数为 [1,6,3,6,2,3,5,2,10,3,6,1]，十二层均为 S/R 混合组；不是四个 R12 Block。实际头计划另存 archived_head_plans.csv。公共成本、逐层公开重排和小组比较的实测费用共同解释通信下降而总耗时接近不变；不将所有公共算子的残余时间差归因于异构头。</p>'
    body+='<h2>可优化成本与后续分配模型</h2><p>现有证据直接支持公开头选择/恢复的本地组织成本，以及小组比较和归一化的固定启动/交互成本应进入分配模型。组织路径、每类组大小、归一化组数、AV 批次数、参数所有权、精度和网络条件均需记录。split 并未保证提速，观察到的内嵌转换或等待总量也不是可直接消除的额外边界。</p>'
    body+='<p>建议分配器采用可实测校准的 C(layer, nS,nR,nL,layout,organization; precision,ownership,network) 表或分段模型，再以真实全模型执行验证候选。单头线性相加不足以表达组级固定成本；单一组数特征也不能区分 R 与 L、gather 与 slices、joint 与 split。是否需要更复杂回归应由配置外误差决定。本轮没有开发新打包协议，也没有验证任何新打包加速。</p>'
    body+='<p>后续模型结构建议分别预测双方角色：T̂=maxₚ[Bcold,p+Σℓ Cℓ,p(planℓ)]，通信为两方 payload 之和。每个 C 保留类型组大小、原生/通用组织路径、共享/私有状态及矩阵批次，而不是逐头成本之和或逐算子两方最大值之和。此角色与状态分段结构是后续建议，尚未通过多层激活、更多方案或部署条件的配置外验证；本轮已评估的 protocol_features 是其较简化的可预先计算特征代理。</p>'
    body+='<p>尚未验证：大样本准确率预算、更多图片/不同层激活、异机部署和网络变化、真实密文填充效率的因果消融、新打包实现的加速；消息依赖深度只是轮次下界，跨线程 fork/join 尚未完整观测。</p>'
    body+='<p><a href="samples.csv">逐次原始指标</a> · <a href="role_decomposition.csv">双方成本分解</a> · <a href="operations.csv">全部算子</a> · <a href="protocol_spans.csv">协议区间</a> · <a href="model_predictions.csv">预测与误差</a> · <a href="full_paired_comparisons.csv">完整模型配对差</a> · <a href="audit.json">核查清单</a> · <a href="design.json">固定计划</a></p></html>'
    (out / "report.html").write_text(body)
    short_report(out,data,partial)
    files=("samples.csv","cost_summary.csv","role_decomposition.csv","role_summary.csv","operations.csv","protocol_spans.csv","component_additive_residuals.csv","layer_attention_spans.csv","model_predictions.csv","model_errors.csv","full_model_predictions.csv","full_prediction_contrasts.csv","full_distribution_test.csv","full_component_contrasts.csv","full_order_sensitivity.csv","full_all_valid_sensitivity.csv","report.html","summary.txt","findings.json")
    write(out / "artifact_manifest.json",dict(analysis_source_sha256=digest(ROOT / "cost_analysis.py"),
          files={n:digest(out / n) for n in files if (out / n).exists()},historical_inputs={n:digest(ROOT / n) for n in ("out/10/cost.csv","out/11/findings.json","out/11/block_summary.csv")}))


def short_report(out,data,partial):
    samples,roles,ops,spans,descriptors,summaries,role_summaries,predictions,metrics,full_pred,comparisons=data
    block={r["case"]:r for r in summaries if r["source"]=="out12" and r["scope"]=="block"}
    full={r["case"]:r for r in summaries if r["scope"]=="vit"}
    lines=["BumbleBee-ViT 混合头安全成本实测报告", "执行中间结果，完整结论尚未完成。" if partial else "已完成：新增 8 组完整 Block × 5 次、4 组完整 ViT × 6 次；所有新增调用均通过数值验收。", ""]
    lines.append("计时统一采用完整执行的 max(P0,P1)，双方角色分别分解。Block 一次预热后连续 5 次；完整 ViT 为交错随机顺序的独立冷启动。权重、输入、S/R/L 语义、精度、网络与原生协议保持一致。请求总时间及扣除编译时间另列；性能主表采用完整 native execution。");lines.append("")
    if recovery_note(out):lines.extend([recovery_note(out),''])
    if "S12" in block:
        b=block["S12"]
        lines.append(f"本轮全 S Block：{b['seconds_median']:.3f} s，{b['payload_mib_median']:.3f} MiB。四组新增组合的真实中位数：")
        for name in ("S10R2","S8R4","S10L2","S10R1L1"):
            if name not in block: continue
            r=block[name]
            lines.append(f"  {name}: {r['seconds_median']:.3f} s [{r['min_s']:.3f}, {r['max_s']:.3f}]，{r['payload_mib_median']:.3f} MiB；相对 S12 时间 {100*(r['seconds_median']/b['seconds_median']-1):+.2f}%，通信 {100*(r['payload_mib_median']/b['payload_mib_median']-1):+.2f}%。")
        lines.append("")
    lines.append("混合成本的来源：便宜映射的收益会被显式比较、max/sum 归约及分组归一化的规模效应、公开头选择/恢复部分抵消；投影、FFN、GELU、QK/AV 的公共成本又限制全模型收益。不能把这些差异统称为转换成本。joint 的 Dot H2A 为每 Block 双方合计 10 次，S6R6 split 为 12 次，属于矩阵协议内部。宽泛解密/掩码区间和等待存在嵌套，不是独立可相加项。");lines.append("")
    lines.append("关键复核：旧表选取每次较慢一方，slices 经常选到 P1。固定角色后，joint/gather 与 joint/slices 的 P0 接收阻塞均值为 8.80/8.51 s，P1 为 19.10/19.49 s，原先“等待翻倍”主要是角色混杂。gather 是公开索引的本地标量复制，本轮连续组的恢复 gather 还是 identity。现有 slices 确实消除了该算子；两轮完整 Block 总时间范围仍重叠，不能据此声称稳定整体提速。归一化调用变多也不等于倒数计时必然增加。");lines.append("")
    lines.append("四组新配置留出验证的预测误差（仅旧数据训练，新 S12 校准截距）：")
    for metric,label in (("seconds","时间"),("mib","通信")):
        items=[r for r in metrics if r["evaluation"]=="new_configuration_holdout" and r["metric"]==metric]
        lines.append("  "+label+" MAPE: "+"；".join(f"{r['model']} {r['mape_pct']:.2f}%" for r in items))
    lines.append("小样本下更复杂的模型并非普遍更好；历史按组合留出验证另列，回归系数不作因果解释。");lines.append("")
    if full:
        lines.append("相同 24 个 ReLU 头的完整 ViT（E0 为全 S 参照）：")
        for name in ("E0","E1","E2","E3"):
            if name not in full: continue
            r=full[name]
            lines.append(f"  {name}: n={r['repeats']}，{r['seconds_median']:.3f} s [{r['min_s']:.3f}, {r['max_s']:.3f}]，{r['payload_mib_median']/1024:.6f} GiB；请求扣除编译 {r['request_without_compile_s_median']:.3f} s。")
        if len(full)==4:
            a=full["E1"];b=full["E3"]
            lines.append(f"E3−E1 的真实中位数差为 {b['seconds_median']-a['seconds_median']:+.3f} s、{b['payload_mib_median']-a['payload_mib_median']:+.3f} MiB。单头/12 头加性模型对相同 24 R 头预测相同成本，无法表达这项分布差异。")
            q={r['case']:r for r in role_summaries if r['scope']=='vit' and r['party']==0}
            lines.append('  完整模型 E0–E3 均为两方合计 123 次 Dot H2A；没有混合组新增的独立域转换边界。')
            for key,label in (("pphlo.gather_seconds_mean","gather"),("pphlo.greater_seconds_mean","显式比较"),("pphlo.reduce_seconds_mean","reduce"),("pphlo.reciprocal_seconds_mean","reciprocal")):
                lines.append(f"  固定 P0 算子均值 E3−E1：{label} {q['E3'][key]-q['E1'][key]:+.3f} s（inclusive，非纯 CPU）。")
            lines.append("公共算子跨方案时差单独列出，不能全部归因于混合。完整模型配对差、bootstrap 区间、精确双侧 p 与 Holm 校正，以及 E1/E2/E3 的总体分布检验见 CSV。")
            pair=next(r for r in comparisons if r['first']=='E1' and r['second']=='E3')
            lines.append(f"  同轮配对 E3−E1 均值差 {pair['mean_delta_s']:+.3f} s，轮次 bootstrap 95% 区间 [{pair['bootstrap_mean_ci_low_s']:+.3f}, {pair['bootstrap_mean_ci_high_s']:+.3f}] s；精确双侧 p={pair['exact_signflip_two_sided_p']:.5f}，六组比较 Holm p={pair['holm_adjusted_p']:.5f}。")
            tests=list(csv.DictReader((out / 'full_distribution_test.csv').open()))
            for r in tests:
                label='时间' if r['metric']=='seconds' else '通信'
                p=float(r['holm_two_outcomes_p'])
                interpretation='检出固定方案总体差异' if p<.05 else '未检出固定方案总体差异'
                lines.append(f"  E1/E2/E3 总体分布检验（{label}）：{r['paired_rounds']} 轮，精确 p={float(r['p_value']):.5f}，两指标 Holm p={p:.5f}；{interpretation}，不能归因为组数的唯一因果作用。")
            if (out / 'full_order_sensitivity.csv').exists():
                order=list(csv.DictReader((out / 'full_order_sensitivity.csv').open()))
                r=next(r for r in order if r['effect']=='E3_minus_E1')
                lines.append(f"  轮次、轮内位置的事后线性校正：E3−E1 {float(r['estimate_s']):+.3f} s，95% 区间 [{float(r['ci_low_s']):+.3f}, {float(r['ci_high_s']):+.3f}] s。该关联不是实测加速或纯分组因果效应。")
            if (out / 'full_all_valid_sensitivity.csv').exists():
                all_valid={r['case']:r for r in csv.DictReader((out / 'full_all_valid_sensitivity.csv').open())}
                delta=float(all_valid['E3']['all_valid_median_s'])-float(all_valid['E1']['all_valid_median_s'])
                lines.append(f"  纳入全部有效补充调用的描述性敏感性分析：E3−E1 中位数差 {delta:+.3f} s；不追加配对显著性检验。")
            if (out / 'full_model_errors.csv').exists():
                errors=list(csv.DictReader((out / 'full_model_errors.csv').open()))
                lines.append('  完整模型时间预测 MAPE / 方案间差值 MAE：'+'；'.join(f"{r['model']} {float(r['mape_pct']):.2f}% / {float(r['contrast_mae']):.2f} s" for r in errors if r['metric']=='seconds'))
                lines.append('绝对总耗时误差较小不保证方案差值预测准确；数值顺序也不代表显著的性能排序。')
        lines.append("")
    lines.append("历史 R48：原二进制三次冷执行的全 S/R48 中位数为 402.046/402.127 s，通信下降 10.21%。各层 R 数为 [1,6,3,6,2,3,5,2,10,3,6,1]，十二层都有混合组。固定 P0 重算 HLO 均值，指数+reduce 省 34.05 s，显式比较增 17.85 s，gather 增 8.19 s，乘法+倒数增 3.89 s；这些抵消大部分映射收益。公共算子的计时漂移只作不确定性证据。");lines.append("")
    lines.append("后续分配采用按层/类型组大小/归一化组数/组织路径/AV 批次数校准的安全成本表或分段模型，以整模型实测验证候选。建议按固定双方角色及冷/热状态分别建模，再取双方完整时间的最大值；这一完整结构仍待更多配置外验证。不能只按单头收益线性相加，也不能只数类型组。已直接验证可消除本地 gather 算子；归一化/比较联合批处理或新密文打包的收益尚未验证。本轮没有开发新协议。");lines.append("")
    lines.append("未验证范围：准确率预算可行性、多图片和多层激活、异机/网络变化、密文填充损失的因果归属、新打包加速、精确全局交互轮次。数值一致性不是分类准确率等价，预测与计时重分类不是实测加速。")
    (out / "summary.txt").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT / "out/12")
    parser.add_argument("--partial",action="store_true")
    parser.add_argument("--include-core",action="store_true")
    args=parser.parse_args()
    data=collect(args.out.resolve(),args.partial,args.include_core)
    figures(args.out,data);report(args.out,data,args.partial)
    print("Analysis artifacts written",flush=True)
