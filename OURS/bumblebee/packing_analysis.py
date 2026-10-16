"""Close native packing evidence and summarize independent paired runtimes."""
import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from runtime import ROOT,digest,write
from experiment import save_rows
from mixed import load_events,union_seconds
from optimize_analysis import audit_case,uncertainty,holm
import optimize_analysis as legacy_audit
from packing_intervals import IndexedIntersection

# Preserve the legacy audit's arithmetic, but index its repeated immutable
# per-party link interval list. Validated exactly against its old algorithm.
legacy_audit.intersection_seconds=IndexedIntersection()


def analyze(out,partial=False):
    result=[[] for _ in range(8)]
    packing,identities,schedules=[],[],[]
    provenance=json.loads((out/'packing.json').read_text())
    if digest(out/'packing.so')!=provenance['library_sha256']:raise RuntimeError('Native binary changed')
    for phase in ('pilot','confirm','control','bridge','full'):
        for folder in sorted((out/phase/'log').glob('*/')):
            if not (folder/'run.json').exists():continue
            meta=json.loads((folder/'run.json').read_text())
            if meta['status']!='complete':
                if partial:continue
                raise RuntimeError('Incomplete case: '+str(folder))
            audited=audit_case(folder)
            for dst,rows in zip(result,audited):dst.extend(rows)
            c=meta['config']
            expected=digest(ROOT/'build/stage2.so') if c.get('bridge_arm')=='cached_original' else provenance['library_sha256']
            if meta['library']['sha256']!=expected:raise RuntimeError('Unexpected native library')
            codes={v:(folder/'compiler'/f'{v}.mlir').read_bytes() for v in c['paired_variants']}
            if len(set(codes.values()))!=1:raise RuntimeError('Native variants must execute byte-identical IR')
            logical=meta['logical_head_plan_sha256']
            identities.append(dict(phase=phase,case=c['plan_name'],session=c['session'],
                same_ir_within_runtime=True,logical_plan_sha256=logical,
                library_sha256=meta['library']['sha256'],compiled_sha256=digest(next((folder/'compiler').glob('*.mlir'))),
                checkpoint_sha256=meta['hashes']['checkpoint'],input_sha256=meta['input_sha256'],
                storage_validated=True,bridge_arm=c.get('bridge_arm',''),metadata=str(folder/'run.json')))
            for measurement,profiles in zip(meta['measurements'],meta['profiles']):
                for party in (0,1):
                    e=measurement['native_execution'];events=load_events(folder/'log'/f'native-p{party}-e{e}.tsv')
                    role=next(r for r in result[1] if r['metadata']==str(folder/'run.json') and
                              r['execution']==e and r['party']==party)
                    role['packing_union_s']=union_seconds([r for r in events if r['kind']=='packing'])
                    role['packing_calls']=sum(r['kind']=='packing' for r in events)
                    matrix_kernels=('f_mmul','f_batch_mmul','f_tensordot') if c['task']=='vit' else ('f_mmul','f_batch_mmul')
                    mats=sorted((r for r in events if r['kind']=='HAL' and r['name'] in matrix_kernels),key=lambda x:x['start_ns'])
                    labels=['Q','K','QK','V','AV','output_projection','FFN1','FFN2']*(12 if c['task']=='vit' else 1)
                    # Native all-S Flax einsum computes V,Q,K before QK and
                    # lowers AV as V^T @ W^T. Do not assign mixed-path ordinal
                    # labels to this different (but internally fixed) graph.
                    if c['task']=='block' and set(c['types'])=={'S'}:
                        labels=['V','Q','K','QK','AV','output_projection','FFN1','FFN2']
                        if [r['name'] for r in mats[:5]]!=['f_mmul','f_mmul','f_mmul','f_batch_mmul','f_batch_mmul']:
                            raise RuntimeError('Native S12 compiler schedule differs from audited lowering')
                    if c['task']=='vit':labels=['patch_embedding']+labels+['classifier']
                    if len(mats)!=len(labels):raise RuntimeError('Matrix event graph differs')
                    for index,(label,m) in enumerate(zip(labels,mats)):
                        packs=[r for r in events if r['kind']=='packing' and m['start_ns']<=r['start_ns'] and r['end_ns']<=m['end_ns']]
                        recv=[r for r in events if r['kind']=='recv' and r['end_ns']>m['start_ns'] and r['start_ns']<m['end_ns']]
                        clipped=[dict(r,start_ns=max(r['start_ns'],m['start_ns']),end_ns=min(r['end_ns'],m['end_ns'])) for r in recv]
                        fields=dict(scope=c['task'],phase=phase,case=c['plan_name'],session=c['session'],
                            pair=measurement['pair'],variant=measurement['variant'],party=party,
                            layer=(index-1)//8 if c['task']=='vit' and 1<=index<=96 else (index//8 if c['task']=='block' else -1),component=label,
                            bridge_arm=c.get('bridge_arm',''))
                        packing.append(dict(**fields,matrix_wall_s=(m['end_ns']-m['start_ns'])/1e9,
                            packing_union_s=union_seconds(packs),packing_calls=len(packs),
                            matrix_recv_blocking_union_s=union_seconds(clipped),
                            sent_bytes=m['bytes'],recv_bytes=m['recv_bytes'],
                            evidence=str(folder/'log'/f'native-p{party}-e{e}.tsv')))
                        layout=[r for r in events if r['kind']=='packing_schedule' and m['start_ns']<=r['start_ns']<=m['end_ns']]
                        for row in layout:
                            data=dict(t.split('=') for t in row['name'].split(';'))
                            actual_mode='tree_batch' if measurement['variant']=='tree_batch' else 'original'
                            if data['mode']!=actual_mode:raise RuntimeError('Actual packing mode differs')
                            nums={k:int(v) for k,v in data.items() if k!='mode'}
                            if nums['groups']!=(nums['inputs']+nums['gap']-1)//nums['gap']:raise RuntimeError('Packing layout invalid')
                            launches=(nums['gap'].bit_length())*(1 if actual_mode=='tree_batch' else nums['groups'])
                            if nums['launches']!=launches:raise RuntimeError('Launch count invalid')
                            schedules.append(dict(**fields,**data))
                    if c.get('bridge_arm')!='cached_original':
                        if len([r for r in events if r['kind']=='packing_schedule'])!=sum(p['packing_calls'] for p in packing[-len(mats):]):
                            raise RuntimeError('Packing schedule events do not close')
    names=['raw_samples','role_costs','operator_costs','attention_operators','protocol_operations','closure_checks','numerical_validation','secure_output_differences']
    for name,rows in zip(names,result):
        if rows:save_rows(out/(name+'.csv'),rows)
    for name,rows in [('matrix_packing_raw',packing),('identity_checks',identities),('packing_schedule_raw',schedules)]:
        if rows:save_rows(out/(name+'.csv'),rows)
    raw,roles,operations,detailed,protocols,checks,numeric,differences=result
    groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for row in raw:
        if row['phase']=='bridge':continue
        groups[(row['phase'],row['case'])][row['session']][row['variant']].append(row)
    comparisons=[];paired=[]
    for (phase,case),sessions in sorted(groups.items()):
        values,percents,wire,cpu0,cpu1=[],[],[],[],[]
        baselines,optimized=[],[]
        for session,variants in sorted(sessions.items()):
            if set(variants)!={'baseline','tree_batch'}:
                if partial:continue
                raise RuntimeError('Missing paired arm')
            a={k:st.mean(r[k] for r in variants['baseline']) for k in ('native_s','payload_mib','p0_process_cpu_s','p1_process_cpu_s')}
            b={k:st.mean(r[k] for r in variants['tree_batch']) for k in a}
            values.append(a['native_s']-b['native_s']);percents.append(100*(a['native_s']-b['native_s'])/a['native_s'])
            wire.append(a['payload_mib']-b['payload_mib']);cpu0.append(a['p0_process_cpu_s']-b['p0_process_cpu_s']);cpu1.append(a['p1_process_cpu_s']-b['p1_process_cpu_s'])
            baselines.append(a['native_s']);optimized.append(b['native_s'])
            paired.append(dict(phase=phase,case=case,session=session,baseline_s=a['native_s'],optimized_s=b['native_s'],
                saved_s=values[-1],saved_percent=percents[-1],baseline_mib=a['payload_mib'],optimized_mib=b['payload_mib'],
                saved_mib=wire[-1],saved_p0_cpu_s=cpu0[-1],saved_p1_cpu_s=cpu1[-1]))
        if not values:continue
        u=uncertainty(values);w=uncertainty(wire);p=uncertainty(percents)
        comparisons.append(dict(phase=phase,case=case,independent_runtimes=len(values),
            independent_pairs=len(values),two_party_runtimes=len(values)*(2 if phase=='full' else 1),
            statistical_unit='independent cold paired round' if phase=='full' else 'fresh shared-variant runtime',
            real_calls_per_arm=sum(len(v['baseline']) for v in sessions.values() if 'baseline' in v),
            baseline_mean_s=st.mean(baselines),optimized_mean_s=st.mean(optimized),
            baseline_sd_s=st.stdev(baselines) if len(values)>1 else 0.,optimized_sd_s=st.stdev(optimized) if len(values)>1 else 0.,
            saved_mean_s=u['mean'],saved_ci_low_s=u['ci_low'],saved_ci_high_s=u['ci_high'],saved_sd_s=u['sd'],
            saved_t_ci_low_s=u['t_ci_low'],saved_t_ci_high_s=u['t_ci_high'],
            saved_mean_percent=p['mean'],saved_percent_ci_low=p['ci_low'],saved_percent_ci_high=p['ci_high'],
            p_exact=u['exact_two_sided_p'],p_holm=math.nan,
            saved_payload_mean_mib=w['mean'],saved_payload_ci_low_mib=w['ci_low'],saved_payload_ci_high_mib=w['ci_high'],
            saved_p0_process_cpu_s=st.mean(cpu0),saved_p1_process_cpu_s=st.mean(cpu1)))
    primary=[r for r in comparisons if r['phase']=='confirm']
    if primary:
        for i,row in enumerate(sorted(primary,key=lambda r:r['p_exact'])):
            row['p_holm']=min(1.,max((len(primary)-j)*r['p_exact'] for j,r in enumerate(sorted(primary,key=lambda r:r['p_exact'])[:i+1])))
    if comparisons:save_rows(out/'before_after.csv',comparisons)
    if paired:save_rows(out/'paired_differences.csv',paired)
    # Fixed-role decomposition: inclusive packing is never added to its
    # containing matrix operator. Only HLO categories close total wall time.
    def summarize(rows,keys,metrics,name):
        g=defaultdict(list)
        for r in rows:g[tuple(r[k] for k in keys)].append(r)
        result=[]
        for key,records in sorted(g.items()):
            row=dict(zip(keys,key));row['real_calls']=len(records)
            for metric in metrics:
                values=[float(r[metric]) for r in records];row[metric+'_mean']=st.mean(values);row[metric+'_sd']=st.stdev(values) if len(values)>1 else 0.
            result.append(row)
        if result:save_rows(out/(name+'.csv'),result)
    summarize(roles,['phase','case','variant','party'],
        ['native_s','process_cpu_s','caller_thread_cpu_s','link_blocking_union_s','recv_blocking_union_s','send_calls','dot_H2A_calls','packing_union_s','packing_calls',
         'matrix_s','gelu_s','exponential_s','comparison_s','normalization_s','organization_s','other_s'],'cost_decomposition')
    summarize(packing,['phase','case','variant','party','component'],
        ['matrix_wall_s','packing_union_s','matrix_recv_blocking_union_s','sent_bytes','recv_bytes','packing_calls'],'matrix_packing_summary')
    summarize(detailed,['phase','case','variant','party','component'],['wall_s','sent_bytes','recv_bytes','link_overlap_s'],'key_operator_decomposition')
    # Preserve backend counters separately: cache refill can change total HE
    # counts; packing unit tests establish exact unchanged primitive work.
    counters=[]
    for phase in ('pilot','confirm','control','bridge','full'):
        for folder in sorted((out/phase/'log').glob('*/')):
            if not (folder/'run.json').exists():continue
            m=json.loads((folder/'run.json').read_text())
            if m['status']!='complete':continue
            for r,parts in zip(m['measurements'],m['profiles']):
                for party,part in enumerate(parts):
                    counters.append(dict(phase=phase,case=m['config']['plan_name'],session=m['config']['session'],pair=r['pair'],
                        variant=r['variant'],party=party,bridge_arm=m['config'].get('bridge_arm',''),**part['backend_metrics']))
    if counters:save_rows(out/'backend_counters.csv',counters)
    summarize(counters,['phase','case','variant','party'],
        ['HE_calls','conversion_calls','protocol_send_calls','send_bytes','recv_bytes'],
        'protocol_counter_summary')
    summary=dict(status='partial' if partial else 'analyzed',analyzer_sha256=digest(Path(__file__)),real_measured_calls=len(raw),numerically_checked_outputs=len(numeric),
        fixed_party_closure_checks=len(checks),all_numerical_accepted=all(r['accepted'] for r in numeric),
        max_relative_l2=max((r['relative_l2'] for r in numeric),default=None),max_abs=max((r['max_abs'] for r in numeric),default=None),
        comparisons=comparisons,baseline_library_preserved=digest(ROOT/'build/stage2.so')==json.loads((out/'audit_identity.json').read_text())['observer_sha256'])
    write(out/'analysis.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'out/14');p.add_argument('--partial',action='store_true');a=p.parse_args();analyze(a.out.resolve(),a.partial)
