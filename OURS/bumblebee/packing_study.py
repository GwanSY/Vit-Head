"""Real paired 2PC tests of native packing; the attention graph is unchanged."""
import argparse
import json
import random
import time
from pathlib import Path
from runtime import ROOT, digest, write
from experiment import case_config, run_case

CASES={'S10R2':'S'*10+'R'*2,'S6R6':'S'*6+'R'*6,'S6R3L3':'S'*6+'R'*3+'L'*3}


def design(out,phase,sessions,pairs,cases):
    native=out/'packing.so'
    provenance=json.loads(native.with_suffix('.json').read_text())
    if provenance['status']!='built' or not provenance['test_passed']:
        raise RuntimeError('Native exact tests must pass before secure experiments')
    if digest(native)!=provenance['library_sha256']:raise RuntimeError('Native hash differs')
    path=out/f'{phase}_design.json'
    if path.exists():
        d=json.loads(path.read_text())
        if (d['sessions'],d['pairs'],d['cases'])!=(sessions,pairs,cases):raise RuntimeError('Frozen design changed')
        return d
    rng=random.Random(2026101414+sum(map(ord,phase)))
    records=[]
    for session in range(1,sessions+1):
        order=list(cases);rng.shuffle(order)
        for case in order:
            first=['baseline','tree_batch'] if session%2 else ['tree_batch','baseline']
            warm=first[::-1] if (session%4)<2 else first
            schedule=[dict(pair=p+1,order=first if p%2==0 else first[::-1]) for p in range(pairs)]
            records.append(dict(session=session,case=case,warmup_order=warm,pair_schedule=schedule))
    d=dict(phase=phase,sessions=sessions,pairs=pairs,cases=cases,records=records,
        native_sha256=digest(native),primary='mean of independent-runtime paired maximum-party native time differences',
        uncertainty='cluster bootstrap 30000; exact two-sided sign-flip; Holm over three primary mixed configurations',
        benefit_gate='confirm lower bootstrap CI > 0 and Holm p < 0.05; no communication reduction expected',
        original_kernel='same new binary with mode=0 uses original PackingWithModulusDrop sequential loop',
        graph='same original baseline PPHLO; no slices/reciprocal graph optimizations')
    write(path,d);return d


def blocks(out,phase,sessions,pairs,cases):
    d=design(out,phase,sessions,pairs,cases)
    alpha=json.loads((ROOT/'out/07/alpha.json').read_text())['alpha'][0]
    for record in d['records']:
        case=record['case']; types=CASES.get(case,case[0]*12)
        name=f"{phase}-r{record['session']:02}-{case}"
        target=out/phase/'log'/name
        if target.exists() and json.loads((target/'run.json').read_text())['status']=='complete':continue
        c=case_config('block','cpu',types=list(types),alpha=alpha,layout='joint',organization='paired',
            native_library=str(out/'packing.so'),trace_intervals=True,repeats=1,
            measured_calls=2*pairs,paired_variants=['baseline','tree_batch'],
            warmup_order=record['warmup_order'],pair_schedule=record['pair_schedule'],
            plan_name=case,session=record['session'],phase=phase,study='native_packing',native_packing_study=True)
        write(out/'progress.json',dict(stage=phase,active=record,start_epoch=time.time()))
        run_case(out/phase,name,c)


def original_control(out,sessions=3):
    # Immutable stage2.so executes the original kernel, even if an unrelated
    # environment flag is set. This is a bridge control, not a within-runtime
    # speedup comparison between different libraries.
    alpha=json.loads((ROOT/'out/07/alpha.json').read_text())['alpha'][0]
    rng=random.Random(2026101415)
    for session in range(1,sessions+1):
        arms=['cached_original','rebuilt_original'];rng.shuffle(arms)
        for arm in arms:
            native=ROOT/'build/stage2.so' if arm=='cached_original' else out/'packing.so'
            name=f'bridge-r{session:02}-{arm}'
            target=out/'bridge/log'/name
            if target.exists() and json.loads((target/'run.json').read_text())['status']=='complete':continue
            c=case_config('block','cpu',types=list(CASES['S6R6']),alpha=alpha,layout='joint',organization='paired',
                native_library=str(native),trace_intervals=True,repeats=1,measured_calls=1,
                paired_variants=['baseline'],warmup_order=['baseline'],pair_schedule=[dict(pair=1,order=['baseline'])],
                plan_name='S6R6',session=session,phase='bridge',study='native_packing',native_packing_study=True,bridge_arm=arm)
            run_case(out/'bridge',name,c)


def full(out,case,rounds):
    import csv
    rows=list(csv.DictReader((out/'before_after.csv').open()))
    matches=[r for r in rows if r['phase']=='confirm' and r['case']==case]
    if len(matches)!=1 or float(matches[0]['saved_ci_low_s'])<=0 or float(matches[0]['p_holm'])>=.05:
        raise RuntimeError('Clear confirm Block benefit required for full ViT')
    path=out/'full_design.json'
    if path.exists():
        d=json.loads(path.read_text())
        if (d['case'],d['rounds'])!=(case,rounds):raise RuntimeError('Frozen full design differs')
    else:
        orders=[['baseline','tree_batch'] if i%2==0 else ['tree_batch','baseline'] for i in range(rounds)]
        random.Random(2026101416).shuffle(orders)
        d=dict(case=case,rounds=rounds,orders=orders,measurement='cold fresh runtimes, one entire native inference per arm; no warmup')
        write(path,d)
    alpha=json.loads((ROOT/'out/07/alpha.json').read_text())['alpha']
    plan=dict(heads=[list(CASES[case]) for _ in range(12)],alpha=alpha)
    for session,order in enumerate(d['orders'],1):
        for variant in order:
            name=f'full-r{session:02}-{case}-{variant}'
            target=out/'full/log'/name
            if target.exists() and json.loads((target/'run.json').read_text())['status']=='complete':continue
            c=case_config('vit','cpu',head_plan=plan,layout='joint',native_library=str(out/'packing.so'),
                trace_intervals=True,repeats=1,measured_calls=1,paired_variants=[variant],
                warmup_order=[],pair_schedule=[dict(pair=1,order=[variant])],plan_name=case,
                session=session,phase='full',study='native_packing',native_packing_study=True)
            # The worker always uses an independently computed original graph
            # reference even for the arm containing only the optimized variant.
            run_case(out/'full',name,c)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','confirm','control','bridge','full']);p.add_argument('--out',type=Path,default=ROOT/'out/14')
    p.add_argument('--sessions',type=int);p.add_argument('--pairs',type=int,default=1);p.add_argument('--case',default='S10R2');a=p.parse_args();o=a.out.resolve()
    if a.stage=='bridge':original_control(o,a.sessions or 3)
    elif a.stage=='full':full(o,a.case,a.sessions or 6)
    else:blocks(o,a.stage,a.sessions or (8 if a.stage=='confirm' else 1),a.pairs,
               list(CASES) if a.stage!='control' else ['S12','R12'])
