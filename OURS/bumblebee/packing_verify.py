"""Cross-run model/plan/IR invariants and exact fixed-party cost contrasts."""
import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from runtime import ROOT,digest,write
from experiment import save_rows


def verify(out,partial=False):
    expected={'pilot':3,'confirm':24,'control':10,'bridge':6,'full':12}
    study=json.load((out/'run.json').open())
    if study.get('full_skipped'):expected['full']=0
    metas=[];counts={}
    for phase in expected:
        complete=[]
        for path in sorted((out/phase/'log').glob('*/run.json')):
            m=json.load(path.open())
            if m['status']=='complete':complete.append((path,m))
            elif not partial:raise RuntimeError('Unfinished case: '+str(path))
        counts[phase]=len(complete);metas+=complete
        if not partial and counts[phase]!=expected[phase]:raise RuntimeError('Repeat count differs: '+phase)
    reference=json.load((ROOT/'out/13/confirm/log/confirm-r08-S10R2/run.json').open())
    frozen=json.load((out/'audit_identity.json').open());native=json.load((out/'packing.json').open())
    checks=[]
    def check(name,ok,evidence):
        if not ok:raise RuntimeError('Cross-run invariant failed: '+name+' '+evidence)
        checks.append(dict(check=name,passed=True,evidence=evidence))
    check('original_CPU_library_preserved',digest(ROOT/'build/spu/libspu.so')==frozen['original_cpu_sha256'],'build/spu/libspu.so')
    check('observer_library_preserved',digest(ROOT/'build/stage2.so')==frozen['observer_sha256'],'build/stage2.so')
    check('optimized_library_preserved',digest(out/'packing.so')==native['library_sha256'],'packing.so')
    by_plan=defaultdict(list)
    alpha=json.load((ROOT/'out/07/alpha.json').open())['alpha']
    plans={'S10R2':'S'*10+'R'*2,'S6R6':'S'*6+'R'*6,'S6R3L3':'S'*6+'R'*3+'L'*3,'S12':'S'*12,'R12':'R'*12}
    for path,m in metas:
        c=m['config'];evidence=str(path.relative_to(out));case=c['plan_name'];full=c['task']=='vit'
        check('checkpoint_unchanged',m['hashes']['checkpoint']==reference['hashes']['checkpoint'],evidence)
        check('input_image_unchanged',m['hashes']['image']==reference['hashes']['image'],evidence)
        for name in ('model.py','heads.py'):
            check('original_math_source_'+name,m['hashes'][name]==reference['hashes'][name],evidence)
        check('precision_and_hardware',m['precision']==reference['precision'] and c['cpus']==reference['config']['cpus'] and c['threads']==25,evidence)
        check('numerical_gate_unchanged',c['gate']['relative_l2_max']==.05 and c['gate']['max_abs_max']==.5,evidence)
        head_lists=c['head_plan']['heads'] if full else [c['types']]
        check('fixed_logical_heads',len(head_lists)==(12 if full else 1) and all(''.join(x)==plans[case] for x in head_lists),evidence)
        check('original_alpha',c['head_plan']['alpha']==alpha if full else c['alpha']==alpha[0],evidence)
        check('private_weights',all('priv' in x.lower() for p in m['weight_storage_types'] for x in p),evidence)
        if not full:check('shared_Block_input',all('ashr' in x.lower() for p in m['input_storage_types'] for x in p),evidence)
        expected_lib=frozen['observer_sha256'] if c.get('bridge_arm')=='cached_original' else native['library_sha256']
        check('actual_native_binary',m['library']['sha256']==expected_lib,evidence)
        for r in m.get('warmups',[])+m['measurements']:
            wanted=['1','1'] if r['variant']=='tree_batch' else ['0','0']
            check('both_parties_mode',r['native_packing_mode_ack']==wanted,evidence+f"#e{r['native_execution']}")
            check('recorded_numerical_acceptance',r['numerical_pass'],evidence+f"#e{r['native_execution']}")
        by_plan[(c['task'],case)].append((path,m))
    reuse=[]
    for (scope,case),records in sorted(by_plan.items()):
        for key in ('logical_head_plan_sha256','input_sha256'):
            check('fixed_'+key,len({m[key] for p,m in records})==1,scope+'/'+case)
        codes={r['compiled_sha256'] for p,m in records for r in m['measurements']}
        check('byte_identical_IR_across_runs',len(codes)==1,scope+'/'+case)
        old=ROOT/'out/13/compiler'/f'{case}-baseline.mlir'
        if scope=='block' and old.exists():
            ok=codes=={digest(old)};check('historical_original_graph_reused',ok,case)
            reuse.append(dict(case=case,sha256=next(iter(codes)),historical_sha256=digest(old),identical=ok))
    save_rows(out/'cross_run_invariants.csv',checks)
    if reuse:save_rows(out/'historical_graph_reuse.csv',reuse)
    # P0/P1 HLO components close independently. Packing is a nested matrix
    # observation, CPU-s and link unions are separate observations.
    roles=list(csv.DictReader((out/'role_costs.csv').open()))
    if not partial:
        check('aggregate_role_count_current',len(roles)==2*sum(len(m['measurements']) for p,m in metas),'role_costs.csv')
    fields=['native_s','matrix_s','gelu_s','exponential_s','comparison_s','normalization_s','organization_s','other_s','packing_union_s','process_cpu_s','recv_blocking_union_s']
    groups=defaultdict(dict)
    for r in roles:
        if r['phase']!='full':continue
        groups[(r['session'],r['party'])][r['variant']]=r
    contrasts=[]
    for (session,party),variants in sorted(groups.items()):
        if set(variants)!={'baseline','tree_batch'}:
            if not partial:raise RuntimeError('Missing complete full-model role pair')
            continue
        row=dict(session=int(session),party=int(party))
        for k in fields:row[k+'_saved']=float(variants['baseline'][k])-float(variants['tree_batch'][k])
        check('full_fixed_party_HLO_delta_closed',abs(sum(row[k+'_saved'] for k in fields[1:8])-row['native_s_saved'])<1e-7,str((session,party)))
        contrasts.append(row)
    if contrasts:
        if not partial:check('six_full_pairs_both_parties',len(contrasts)==12,'full_component_contrasts.csv')
        save_rows(out/'full_component_contrasts.csv',contrasts)
        summary=[]
        for party in (0,1):
            data=[r for r in contrasts if r['party']==party]
            if data:summary.append(dict(party=party,paired_rounds=len(data),**{k+'_mean':st.mean(r[k] for r in data) for k in data[0] if k.endswith('_saved')}))
        save_rows(out/'full_component_summary.csv',summary)
    save_rows(out/'cross_run_invariants.csv',checks)
    result=dict(status='partial' if partial else 'verified',completed_runtimes=counts,checks=len(checks),all_passed=True,checkpoint_sha256=reference['hashes']['checkpoint'])
    write(out/'cross_run_verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'out/14');p.add_argument('--partial',action='store_true');a=p.parse_args();verify(a.out.resolve(),a.partial)
