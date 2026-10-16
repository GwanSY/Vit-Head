"""Audit paired secure evidence and report net benefits without nested timing sums."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import pickle
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from runtime import ROOT, digest, write
from experiment import save_rows
from mixed import load_events, union_seconds
from group_costs import compiler_ops
from cost_analysis import category, KEYS, intersection_seconds, student_t_975

MIB = 1 << 20
T_CRITICAL = {}


def audit_case(folder, partial=False):
    meta = json.loads((folder / "run.json").read_text()); c = meta["config"]
    if meta["status"] != "complete":
        if partial: return None
        raise RuntimeError(f"Incomplete case: {folder}")
    measurements = meta["measurements"]
    if len(measurements) != c["measured_calls"] or len(meta["profiles"]) != len(measurements):
        raise RuntimeError("Measured native profile count differs")
    cache_dir = folder.parents[2] / ".opt-cache"; cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"{folder.name}.pickle"
    raw = [folder / "run.json", folder / "outputs.npz"]
    raw += list((folder / "log").glob("cpu-*.jsonl"))
    raw += list((folder / "log").glob("native-*.tsv"))
    raw += list((folder / "compiler").glob("*.mlir"))
    identity = hashlib.sha256(pickle.dumps([(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(raw)]
        + [(str(Path(__file__)),digest(Path(__file__)))])).hexdigest()
    if cache.exists():
        cached = pickle.loads(cache.read_bytes())
        if cached[0] == identity: return cached[1]
    outputs = np.load(folder / "outputs.npz")
    baseline = outputs["reference_baseline"]
    cpu = [{r["execution"]:r for r in map(json.loads,(folder / "log" / f"cpu-p{rank}.jsonl").read_text().splitlines())}
           for rank in (0,1)]
    rows, roles, operations, detailed, protocols, checks, numeric, paired_outputs = ([] for _ in range(8))
    fields = dict(scope=c["task"], phase=c["phase"], session=c["session"], case=c["plan_name"],
                  metadata=str(folder / "run.json"))
    full = c["task"] == "vit"
    for record in meta.get("warmups",[])+measurements:
        actual = outputs[f"e{record['native_execution']}_{record['variant']}"]
        difference = actual.astype(np.float64)-baseline.astype(np.float64)
        relative = float(np.linalg.norm(difference)/np.linalg.norm(baseline))
        maximum = float(np.max(np.abs(difference)))
        finite = bool(np.isfinite(actual).all() and np.isfinite(baseline).all())
        accepted = finite and relative <= c["gate"]["relative_l2_max"] and maximum <= c["gate"]["max_abs_max"]
        if full: accepted = accepted and actual.argmax() == baseline.argmax()
        if not accepted or abs(relative-record["relative_l2"])>1e-12 or abs(maximum-record["max_abs"])>1e-12:
            raise RuntimeError("Stored output does not pass independently recomputed numerical gates")
        numeric.append(dict(**fields,variant=record["variant"],pair=record["pair"],
            execution=record["native_execution"],warmup=record["warmup"],relative_l2=relative,
            max_abs=maximum,finite=finite,accepted=accepted,
            top1_matches=int(actual.argmax())==int(baseline.argmax()) if full else ""))
    for sample,(measurement,parts) in enumerate(zip(measurements,meta["profiles"]),1):
        e = measurement["native_execution"]; variant = measurement["variant"]
        ir = folder / "compiler" / f"{variant}.mlir"
        if digest(ir)!=measurement["compiled_sha256"]:
            raise RuntimeError("Compiled code identity differs from actual execution")
        ops = compiler_ops(ir)
        pair_fields = dict(**fields,variant=variant,pair=measurement["pair"],execution=e)
        events = [load_events(folder / "log" / f"native-p{rank}-e{e}.tsv") for rank in (0,1)]
        sent = [sum(r["bytes"] for r in p if r["kind"]=="send") for p in events]
        rows.append(dict(**pair_fields,native_s=max(p["seconds"] for p in parts),
            p0_s=parts[0]["seconds"],p1_s=parts[1]["seconds"],payload_bytes=sum(sent),
            payload_mib=sum(sent)/MIB,request_s=measurement["request_s"],compile_s=measurement["compile_s"],
            global_native_span_s=(max(p[0]["end_ns"] for p in events)-min(p[0]["start_ns"] for p in events))/1e9,
            p0_process_cpu_s=cpu[0][e]["process_cpu_s"],p1_process_cpu_s=cpu[1][e]["process_cpu_s"],
            p0_caller_cpu_s=cpu[0][e]["caller_thread_cpu_s"],p1_caller_cpu_s=cpu[1][e]["caller_thread_cpu_s"],
            relative_l2=measurement["relative_l2"],max_abs=measurement["max_abs"],
            numerical_pass=measurement["numerical_pass"],compiled_sha256=measurement["compiled_sha256"]))
        for rank,(part,p) in enumerate(zip(parts,events)):
            bounds = p[0]
            clipped = [dict(r,start_ns=max(bounds["start_ns"],r["start_ns"]),end_ns=min(bounds["end_ns"],r["end_ns"]))
                       for r in p[1:] if r["end_ns"]>bounds["start_ns"] and r["start_ns"]<bounds["end_ns"]]
            send,recv = ([r for r in clipped if r["kind"]==kind] for kind in ("send","recv"))
            link = send+recv
            backend = part["backend_metrics"]
            trace_time = (bounds["end_ns"]-bounds["start_ns"])/1e9
            checkpoint = dict(**pair_fields,party=rank,
                time_closed=abs(trace_time-part["seconds"])<.01,
                send_bytes_closed=sum(r["bytes"] for r in send)==int(backend["send_bytes"]),
                recv_bytes_closed=sum(r["bytes"] for r in recv)==int(backend["recv_bytes"]),
                link_calls_closed=len(send)==int(backend["protocol_send_calls"]),
                cross_party_bytes_closed=sent[rank]==sum(r["bytes"] for r in events[1-rank] if r["kind"]=="recv"),
                native_backend=backend["backend"]=="cpu" and int(backend["HE_calls"])>0)
            if not all(v for k,v in checkpoint.items() if k not in pair_fields and k!="party"):
                raise RuntimeError(f"Native evidence closure failed: {checkpoint}")
            hlo = sorted((r for r in clipped if r["kind"]=="HLO"),key=lambda r:r["start_ns"])
            if [r["name"] for r in hlo] != [o["name"] for o in ops]:
                raise RuntimeError("Chronological HLO sequence differs from saved executable")
            checkpoint["compiled_sequence_matched"] = True
            seconds = {key:0. for key in KEYS}; wire = {key:0. for key in KEYS}
            for name,op in part["operations"].items():
                operations.append(dict(**pair_fields,party=rank,operation=name,**op))
                if name.startswith("pphlo."):
                    seconds[category(name)] += op["seconds"]
                    wire[category(name)] += op["sent_mib"]
            residual = part["seconds"]-sum(seconds.values())
            wire_residual = part["sent_mib"]-sum(wire.values())
            if residual < -.01 or abs(wire_residual)>.01:
                raise RuntimeError("HLO accounting failed")
            seconds["other"] += residual; wire["other"] += wire_residual
            roles.append(dict(**pair_fields,party=rank,native_s=part["seconds"],
                process_cpu_s=cpu[rank][e]["process_cpu_s"],caller_thread_cpu_s=cpu[rank][e]["caller_thread_cpu_s"],
                cpu_measurement_scope_wall_s=cpu[rank][e]["wall_s"],
                link_blocking_union_s=union_seconds(link),recv_blocking_union_s=union_seconds(recv),
                send_blocking_union_s=union_seconds(send),
                wall_outside_observed_blocking_s=trace_time-union_seconds(link),
                **{key+"_s":value for key,value in seconds.items()},
                **{key+"_sent_mib":value for key,value in wire.items()},
                hlo_runtime_residual_s=residual,
                send_calls=len(send),recv_calls=len(recv),dot_H2A_calls=sum(r["name"]=="dot_H2A" for r in clipped)))
            checkpoint["hlo_time_closed"] = abs(sum(seconds.values())-part["seconds"])<1e-8
            checkpoint["hlo_payload_closed"] = abs(sum(wire.values())-part["sent_mib"])<1e-8
            checks.append(checkpoint)
            matrix = [i for i,o in enumerate(ops) if o["name"]=="pphlo.dot_general"]
            if len(matrix) != (24 if full else 2): raise RuntimeError("Joint QK/AV count changed")
            for layer_number in range(12 if full else 1):
                lo,hi = matrix[2*layer_number:2*layer_number+2]
                types = c["head_plan"]["heads"][layer_number] if full else c["types"]
                sr_heads = types.count("S")+types.count("R")
                reciprocal = [i for i in range(lo+1,hi) if ops[i]["name"]=="pphlo.reciprocal"]
                expected = 1 if variant in ("batched","reciprocal_only") else len(set(types)&{"S","R"})
                if len(reciprocal)!=expected: raise RuntimeError("Unexpected actual reciprocal calls")
                norm_multiplies = set()
                for i in reciprocal:
                    if variant=="reciprocal_only":
                        for kind in "SR":
                            if kind not in types: continue
                            target=types.count(kind)*197**2
                            # Take all matching secret/secret multiplies after reciprocal;
                            # equal-sized S/R groups have two, L public scaling is excluded.
                            candidates=[j for j in range(i+1,hi) if ops[j]["name"]=="pphlo.multiply"
                                and ops[j]["numel"]==target and '!pphlo.pub<f32>' not in ops[j]["text"]]
                            if not candidates: raise RuntimeError("Separate normalization multiply missing")
                            norm_multiplies.update(candidates)
                    else:
                        normalized_heads = sr_heads if variant=="batched" else ops[i]["numel"]//197
                        target = normalized_heads*197**2
                        multiply = next((j for j in range(i+1,hi) if ops[j]["name"]=="pphlo.multiply" and ops[j]["numel"]==target),None)
                        if multiply is None: raise RuntimeError("Normalization multiply missing")
                        norm_multiplies.add(multiply)
                for i in range(lo,hi+1):
                    o,r = ops[i],hlo[i]
                    component = "organization" if category(o["name"])=="organization" else "other_attention"
                    if i==lo: component="QK"
                    elif i==hi: component="AV"
                    elif o["name"]=="pphlo.reciprocal": component="reciprocal"
                    elif i in norm_multiplies: component="normalization_multiply"
                    elif o["name"]=="pphlo.exponential": component="softmax_exp"
                    elif o["name"]=="pphlo.reduce":
                        component="softmax_max" if "pphlo.greater" in o["body"] else "row_sum"
                    elif o["name"]=="pphlo.gather": component="public_gather"
                    elif o["name"]=="pphlo.greater":
                        component="relu_score_compare" if o["numel"]==types.count("R")*197**2 else "row_comparison"
                    interval = [r]
                    detailed.append(dict(**pair_fields,party=rank,layer=layer_number,component=component,
                        operation=o["name"],shape="x".join(map(str,o["shape"])),numel=o["numel"],
                        compiler_line=o["line"],wall_s=(r["end_ns"]-r["start_ns"])/1e9,
                        sent_bytes=r["bytes"],recv_bytes=r["recv_bytes"],
                        link_overlap_s=intersection_seconds(interval,link)))
            event_groups = defaultdict(list)
            for r in clipped:
                if r["kind"] in ("MPC","HAL","packing","conversion","conversion_region"):
                    event_groups[(r["kind"],r["name"])].append(r)
            for (kind,name),group in event_groups.items():
                protocols.append(dict(**pair_fields,party=rank,kind=kind,name=name,calls=len(group),
                    inclusive_thread_sum_s=sum((r["end_ns"]-r["start_ns"])/1e9 for r in group),
                    interval_union_s=union_seconds(group),link_overlap_s=intersection_seconds(group,link)))
    paired = defaultdict(dict)
    for record in measurements:
        paired[record["pair"]][record["variant"]] = record
    for pair_number,records in paired.items():
        if "baseline" not in records: continue  # The other cold full arm is in its own runtime.
        for variant,record in records.items():
            if variant=="baseline": continue
            a = outputs[f"e{records['baseline']['native_execution']}_baseline"].astype(np.float64)
            b = outputs[f"e{record['native_execution']}_{variant}"].astype(np.float64)
            paired_outputs.append(dict(**fields,pair=pair_number,variant=variant,
                relative_l2_between_secure_outputs=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                max_abs_between_secure_outputs=float(np.max(np.abs(a-b)))))
    result = (rows,roles,operations,detailed,protocols,checks,numeric,paired_outputs)
    cache.write_bytes(pickle.dumps((identity,result)))
    return result


def uncertainty(values, seed=20261014):
    values = np.asarray(values,dtype=float); n = len(values)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0,n,size=(30000,n))].mean(axis=1)
    lo,hi = np.quantile(means,[.025,.975])
    if n<2: lo=hi=float("nan")  # One runtime supplies no across-runtime uncertainty estimate.
    observed = abs(values.mean())
    extreme = sum(abs(np.dot(signs,values)/n)>=observed-1e-12
                  for signs in itertools.product((-1,1),repeat=n))
    standard_deviation=float(values.std(ddof=1)) if n>1 else 0.
    if n>1:
        if n-1 not in T_CRITICAL: T_CRITICAL[n-1]=student_t_975(n-1)
        margin=T_CRITICAL[n-1]*standard_deviation/math.sqrt(n)
    else: margin=float("nan")
    return dict(mean=float(values.mean()),median=float(np.median(values)),
                ci_low=float(lo),ci_high=float(hi),exact_two_sided_p=extreme/(2**n),
                sd=standard_deviation,n=n,t_ci_low=float(values.mean()-margin),t_ci_high=float(values.mean()+margin))


def holm(rows, field="p_holm"):
    ranked = sorted(enumerate(rows),key=lambda x:x[1]["exact_two_sided_p"])
    previous = 0.
    for i,(_,row) in enumerate(ranked):
        previous = max(previous,min(1.,(len(rows)-i)*row["exact_two_sided_p"]))
        row[field] = previous


def analyze(out, partial=False):
    collections = [[] for _ in range(8)]
    folders = []
    excluded = []
    for phase in ("pilot","confirm","full"):
        for path in sorted((out / phase / "log").glob("*/run.json")):
            if "-err" in path.parent.name:
                m=json.loads(path.read_text())
                excluded.append(dict(metadata=str(path),status=m["status"],reason=m.get("error","Archived aborted attempt"),
                                     recorded_measurements=len(m.get("measurements",[]))))
            else: folders.append(path)
    if excluded: save_rows(out / "excluded_attempts.csv",excluded)
    for path in folders:
        parsed = audit_case(path.parent,partial)
        if parsed:
            for target,rows in zip(collections,parsed): target.extend(rows)
    samples,roles,operations,detailed,protocols,checks,numeric,output_diffs = collections
    if not samples: print("No completed paired executions yet",flush=True); return
    provenance = json.loads((out / "confirmation_provenance.json").read_text())
    if digest(ROOT / "build/stage2.so")!=provenance["native_sha256"] or \
       digest(ROOT / "../../BOLT/checkpoints/vit_b16_hf/model.safetensors")!=provenance["checkpoint_sha256"] or \
       digest(ROOT / "build/spu/libspu.so")!=provenance["original_cpu_sha256"]:
        raise RuntimeError("Pretrained weights or original/native observer libraries changed")
    preflight = json.loads((out / "preflight.json").read_text())
    expected_codes = {(r["case"],r["variant"]):r["sha256"] for r in preflight["compiled"]}
    identity_checks = []
    references = {}; inputs = set()
    for path in folders:
        meta = json.loads(path.read_text())
        if meta["status"]!="complete": continue
        c=meta["config"]
        if meta["library"]["sha256"]!=provenance["native_sha256"] or meta["hashes"]["checkpoint"]!=provenance["checkpoint_sha256"]:
            raise RuntimeError("Actual execution binary/model identity differs")
        if c["phase"] in ("confirm","full"):
            for filename,sha in provenance["sources"].items():
                if filename in meta["hashes"] and meta["hashes"][filename]!=sha:
                    raise RuntimeError(f"Frozen measured source changed: {filename}")
            if c["cpus"]!=["0-24","48-72"] or c["threads"]!=25 or c["layout"]!="joint":
                raise RuntimeError("Measured environment changed")
        if c["task"]=="block":
            inputs.add(meta["input_sha256"])
            if any("ashr" not in s.lower() for party in meta["input_storage_types"] for s in party):
                raise RuntimeError("Block activation is not shared")
            for variant,record in meta["compilations"].items():
                if record["sha256"]!=expected_codes[(c["plan_name"],variant)]:
                    raise RuntimeError("Actual optimized executable differs from frozen offline audit")
        if any("priv" not in s.lower() for party in meta["weight_storage_types"] for s in party):
            raise RuntimeError("Private weights changed to public/shared ownership")
        vectors=np.load(path.parent / "outputs.npz")
        ref_hash=hashlib.sha256(vectors["reference_baseline"].tobytes()).hexdigest()
        key=(c["task"],c["plan_name"])
        if key in references and references[key]!=ref_hash: raise RuntimeError("Plaintext reference changed across paired executions")
        references[key]=ref_hash
        identity_checks.append(dict(metadata=str(path),native_binary=True,pretrained_weights=True,
            shared_block_input=True if c["task"]=="block" else "not_applicable",
            private_weights=True,compiled_identity=True,head_plan_sha256=meta["logical_head_plan_sha256"]))
    if len(inputs)>1: raise RuntimeError("First Block activations changed across cases")
    save_rows(out / "identity_checks.csv",identity_checks)
    for name,rows in zip(("raw_samples","role_costs","operator_costs","attention_operators",
        "protocol_operations","closure_checks","numerical_validation","secure_output_differences"),collections):
        if rows: save_rows(out / f"{name}.csv",rows)
    sample_lookup = {(r["phase"],r["session"],r["case"],r["pair"],r["variant"]):r for r in samples}
    pairs = []
    for key,row in sample_lookup.items():
        if key[-1]=="baseline": continue
        original = sample_lookup.get((*key[:-1],"baseline"))
        if original is None: continue  # A partial cold full pair can have only one finished arm.
        pairs.append(dict(phase=key[0],session=key[1],case=key[2],pair=key[3],variant=key[4],
            baseline_s=original["native_s"],optimized_s=row["native_s"],
            saved_s=original["native_s"]-row["native_s"],
            saved_percent=100*(original["native_s"]-row["native_s"])/original["native_s"],
            baseline_payload_mib=original["payload_mib"],optimized_payload_mib=row["payload_mib"],
            saved_payload_mib=original["payload_mib"]-row["payload_mib"],
            saved_request_s=original["request_s"]-row["request_s"]))
    save_rows(out / "paired_differences.csv",pairs)
    summaries = []
    by_comparison = defaultdict(list)
    for row in pairs: by_comparison[(row["phase"],row["case"],row["variant"])].append(row)
    for (phase,case,variant),rows in sorted(by_comparison.items()):
        runtime_groups = defaultdict(list)
        for row in rows: runtime_groups[row["session"]].append(row)
        time_stats = uncertainty([st.mean(x["saved_s"] for x in group) for group in runtime_groups.values()])
        wire_stats = uncertainty([st.mean(x["saved_payload_mib"] for x in group) for group in runtime_groups.values()])
        percent_stats = uncertainty([st.mean(x["saved_percent"] for x in group) for group in runtime_groups.values()])
        summary = dict(phase=phase,case=case,variant=variant,independent_runtimes=len(runtime_groups),
            paired_calls=len(rows),baseline_mean_s=st.mean(r["baseline_s"] for r in rows),
            optimized_mean_s=st.mean(r["optimized_s"] for r in rows),
            baseline_median_s=st.median(r["baseline_s"] for r in rows),
            optimized_median_s=st.median(r["optimized_s"] for r in rows),
            saved_mean_s=time_stats["mean"],saved_ci_low_s=time_stats["ci_low"],saved_ci_high_s=time_stats["ci_high"],
            saved_t_ci_low_s=time_stats["t_ci_low"],saved_t_ci_high_s=time_stats["t_ci_high"],
            saved_mean_percent=percent_stats["mean"],saved_percent_ci_low=percent_stats["ci_low"],saved_percent_ci_high=percent_stats["ci_high"],
            runtime_difference_sd_s=time_stats["sd"],exact_two_sided_p=time_stats["exact_two_sided_p"],
            baseline_payload_mib=st.mean(r["baseline_payload_mib"] for r in rows),
            optimized_payload_mib=st.mean(r["optimized_payload_mib"] for r in rows),
            saved_payload_mean_mib=wire_stats["mean"],saved_payload_ci_low_mib=wire_stats["ci_low"],
            saved_payload_ci_high_mib=wire_stats["ci_high"],p_holm=1.,
            saved_request_mean_s=st.mean(r["saved_request_s"] for r in rows),
            faster_pairs=sum(r["saved_s"]>0 for r in rows))
        summaries.append(summary)
    for phase in ("pilot","confirm","full"):
        chosen = [r for r in summaries if r["phase"]==phase]
        holm(chosen)
    save_rows(out / "before_after.csv",summaries)
    # Incremental normalization benefit after eliminating the same public gathers.
    ablation=[]
    for key,row in sample_lookup.items():
        if key[-1] not in ("batched","reciprocal_only"): continue
        control=sample_lookup.get((*key[:-1],"organized"))
        if control:
            ablation.append(dict(phase=key[0],session=key[1],case=key[2],pair=key[3],variant=key[4],
                organized_s=control["native_s"],candidate_s=row["native_s"],
                additional_saved_s=control["native_s"]-row["native_s"],
                additional_saved_payload_mib=control["payload_mib"]-row["payload_mib"]))
    if ablation: save_rows(out / "normalization_ablation.csv",ablation)
    def aggregates(data, keys, values, path):
        groups = defaultdict(list)
        for row in data: groups[tuple(row[k] for k in keys)].append(row)
        result = []
        for ids,group in sorted(groups.items()):
            row = dict(zip(keys,ids)); row["real_calls"] = len(group)
            row["independent_runtimes"] = len({r["session"] for r in group})
            for value in values:
                row[value+"_mean"] = st.mean(r[value] for r in group)
                row[value+"_sd"] = st.stdev(r[value] for r in group) if len(group)>1 else 0.
            result.append(row)
        save_rows(out / path,result)
    aggregates(roles,("phase","case","variant","party"),
        ("native_s","process_cpu_s","caller_thread_cpu_s","link_blocking_union_s",
         "recv_blocking_union_s","wall_outside_observed_blocking_s","send_calls","dot_H2A_calls")+tuple(k+"_s" for k in KEYS),"cost_decomposition.csv")
    aggregates(operations,("phase","case","variant","party","operation"),
               ("calls","seconds","sent_mib","recv_mib"),"operator_summary.csv")
    aggregates(protocols,("phase","case","variant","party","kind","name"),
               ("calls","interval_union_s","link_overlap_s"),"protocol_summary.csv")
    # First sum per actual model call, then summarize; do not weight by operator count.
    detail_by_sample = defaultdict(list)
    for r in detailed:
        detail_by_sample[(r["phase"],r["case"],r["variant"],r["party"],r["session"],r["pair"],r["component"])].append(r)
    detail_totals = [dict(zip(("phase","case","variant","party","session","pair","component"),key),
        wall_s=sum(r["wall_s"] for r in group),sent_mib=sum(r["sent_bytes"] for r in group)/MIB,
        link_overlap_s=sum(r["link_overlap_s"] for r in group),calls=len(group))
        for key,group in detail_by_sample.items()]
    aggregates(detail_totals,("phase","case","variant","party","component"),
               ("wall_s","sent_mib","link_overlap_s","calls"),"key_operator_decomposition.csv")
    volume_groups=defaultdict(list)
    for r in detailed:
        if r["component"] in ("reciprocal","normalization_multiply","relu_score_compare","softmax_exp"):
            volume_groups[(r["phase"],r["case"],r["variant"],r["party"],r["session"],r["pair"],r["component"])].append(r)
    volumes=[dict(zip(("phase","case","variant","party","session","pair","component"),key),
                  calls=len(group),total_elements=sum(r["numel"] for r in group)) for key,group in volume_groups.items()]
    save_rows(out / "operation_volumes.csv",volumes)
    volume_lookup={(r["phase"],r["case"],r["party"],r["session"],r["pair"],r["component"],r["variant"]):r for r in volumes}
    for key,r in volume_lookup.items():
        baseline_volume=volume_lookup.get((*key[:-1],"baseline"))
        if baseline_volume and r["total_elements"]!=baseline_volume["total_elements"]:
            raise RuntimeError("Optimization changed arithmetic element volume")
    component_lookup={(r["phase"],r["session"],r["case"],r["pair"],r["variant"],r["party"]):r for r in roles}
    contrasts=[]
    for key,r in component_lookup.items():
        if key[4]=="baseline": continue
        baseline_role=component_lookup.get((*key[:4],"baseline",key[5]))
        if not baseline_role: continue
        for component in KEYS:
            contrasts.append(dict(phase=key[0],session=key[1],case=key[2],pair=key[3],variant=key[4],party=key[5],
                component=component,saved_s=baseline_role[component+"_s"]-r[component+"_s"],
                saved_sent_mib=baseline_role[component+"_sent_mib"]-r[component+"_sent_mib"]))
        if abs(sum(baseline_role[k+"_s"]-r[k+"_s"] for k in KEYS)-(baseline_role["native_s"]-r["native_s"]))>1e-8:
            raise RuntimeError("Fixed-role component contrasts do not close to actual net difference")
    save_rows(out / "fixed_party_component_contrasts.csv",contrasts)
    audit = dict(all_numerical_pass=all(r["accepted"] for r in numeric),
        closure_checks=len(checks),all_closure_pass=all(all(v for k,v in r.items() if k.endswith("closed") or k.endswith("matched") or k=="native_backend") for r in checks),
        measured_calls=len(samples),numerical_checks_including_warmups=len(numeric),
        native_sha256=digest(ROOT / "build/stage2.so"),
        checkpoint_sha256=digest(ROOT / "../../BOLT/checkpoints/vit_b16_hf/model.safetensors"),
        max_relative_l2=max(r["relative_l2"] for r in numeric),max_abs=max(r["max_abs"] for r in numeric),
        partial=partial,summary=summaries)
    write(out / "analysis.json",audit)
    for row in summaries:
        print(f"{row['phase']} {row['case']} {row['variant']}: n={row['independent_runtimes']} "
              f"save {row['saved_mean_s']:+.3f}s [{row['saved_ci_low_s']:+.3f},{row['saved_ci_high_s']:+.3f}] "
              f"({row['saved_mean_percent']:+.2f}%), wire {row['saved_payload_mean_mib']:+.3f}MiB; "
              f"p={row['exact_two_sided_p']:.4f}, Holm={row['p_holm']:.4f}",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT / "out/13")
    parser.add_argument("--partial",action="store_true")
    args=parser.parse_args(); analyze(args.out.resolve(),args.partial)
