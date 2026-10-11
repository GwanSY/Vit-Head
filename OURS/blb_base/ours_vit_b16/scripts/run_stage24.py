#!/usr/bin/env python3
"""Gated 2/12-block validation and paired timing; use the existing container runner.
No runtime copies, activation observations, fitted thresholds or timing estimates.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv, datetime, hashlib, json, re, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np
import vit_reference as vr

B = Path(__file__).resolve().parents[2]
V = B.parents[1]
O = B / 'outputs/stage24_full_vit_20261010'
OLD = B / 'outputs/heterogeneous_block_optimized_20261010'
MODEL = V / 'BOLT/checkpoints/vit_b16/bolt_model.txt'
MANIFEST = V / 'BOLT/checkpoints/vit_b16/manifest.json'
BIN = B / 'build/container-gpu/bin/ours_vit_b16_main'
GROUPS = ('B', 'H', 'C')  # C = all Softmax, existing native Linear HE context

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(p)
def state(stage, **kw):
    value = dict(stage=stage, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kw)
    write(O/'status.json', value); print(json.dumps(value), flush=True)
def csvwrite(p, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def metadata(p):
    return dict(line.split('=',1) for line in Path(p).read_text().splitlines() if '=' in line)
def table(p): return list(csv.DictReader(Path(p).open()))
def vector(p):
    s=Path(p).read_text().splitlines();return np.asarray(s[s.index('values')+1:s.index('end')],dtype=float)
def kv(line): return dict(re.findall(r'(\w+)=([^ ]+)', line))
def plan(g): return O/'plans'/('plan_03.txt' if g=='H' else 'plan_00.txt')
def hp(g):
    out=[[('softmax_mpc',1.) for _ in range(12)] for _ in range(12)]
    for r in csv.DictReader(plan(g).read_text().splitlines()[2:]):
        out[int(r['layer'])][int(r['original_head'])]=(r['protocol'],float(r['alpha']))
    return out

def prepare():
    O.mkdir(parents=True,exist_ok=True)
    for sub in ('plans','inputs','references','runs'): (O/sub).mkdir(exist_ok=True)
    for name in ('plan_00.txt','plan_03.txt'):
        (O/'plans'/name).write_bytes((OLD/'plans'/name).read_bytes())
    rows=list(csv.reader((OLD/'splits/selection.csv').open()))
    chosen=[dict(image=rows[i][0],label=int(rows[i][1]),selection_index=i) for i in (0,5,10)]
    assert len({r['image'] for r in chosen})==3
    csvwrite(O/'samples.csv',chosen)
    state('PREPARING_INDEPENDENT_REFERENCES')
    model=vr.load_model(MODEL)
    for i,r in enumerate(chosen):
        inp=O/'inputs'/f'image_{i}.txt'
        cmd=['/home/dm/anaconda3/bin/python',str(B/'ours_vit_b16/scripts/preprocess_vit_image.py'),str(V/'datasets/imagenet2012/val'/r['image']),'--manifest',str(MANIFEST),'--output',str(inp)]
        subprocess.run(cmd,check=True);r['input_sha256']=sha(inp)
        if i==0: assert sha(inp)==sha(OLD/'secure_input.txt'), 'preprocessing differs from frozen input'
        x=vr.load_input(inp)
        for blocks in (2,12):
            for g in ('B','H'):
                state('PREPARING_INDEPENDENT_REFERENCES',group=g,image=i,blocks=blocks)
                y,_=vr.Graph(model,fixed=True,physical_tokens=256,head_plan=hp(g),block_limit=blocks,record_values=False).run(x)
                assert np.isfinite(y).all()
                np.save(O/'references'/f'{g}_{blocks}_{i}.npy',y)
    del model
    csvwrite(O/'samples.csv',chosen)
    paths=[BIN,MODEL,MANIFEST,Path(__file__),B/'scripts/run_ours_container.sh',B/'scripts/run_ours_vit_b16.sh',B/'ours_vit_b16/scripts/vit_reference.py',B/'ours_vit_b16/scripts/preprocess_vit_image.py',B/'ours_vit_b16/scripts/verify_head_reorder.py',OLD/'alpha.json',*sorted((B/'ours_vit_b16/runtime').glob('*.cpp')),*sorted((B/'ours_vit_b16/runtime').glob('*.h')),*sorted((O/'plans').glob('*')),*sorted((O/'inputs').glob('*')),*sorted((O/'references').glob('*'))]
    write(O/'frozen_hashes.json',{str(p):sha(p) for p in paths})
    write(O/'contract.json',dict(groups={'B':'Frozen all-Softmax baseline','H':'Only Block 0: 10S/1R/1L, original R head 0 and L head 6','C':'All-Softmax H execution with existing extended native Linear HE context; distinct control, not B'},correctness_images=3,prefix_blocks=2,full_blocks=12,formal_repeats=5,performance_image=chosen[0],sample_rule='Frozen selection rows 0,5,10; no holdout; chosen before outputs',gpu_pair=[3,4],network='1Gbps,0.3ms RTT',relative_L2_limit=.05,max_abs_limit=.5,prediction_agreement_required=True,nonfinite_limit=0,halt_on_any_failure=True,phase_order=['2-block correctness, all groups/images','12-block correctness, all groups/images','interleaved five performance repeats, all groups'],no_new_packing=True,no_alpha_refit=True,no_new_head_selection=True))
    return chosen

def assert_frozen():
    for p,h in json.loads((O/'frozen_hashes.json').read_text()).items():
        if sha(p)!=h: raise RuntimeError('FROZEN_ASSET_CHANGED '+p)

def parse_party(run, party, blocks):
    name='alice' if party==1 else 'bob';lines=(run/f'{name}.log').read_text().splitlines()
    phases={};events=[];counters={};fusion=[]
    for line in lines:
        if line.startswith('E2E_PHASE '):
            r=kv(line);phases[r['stage']]=(int(r['begin_ns']),int(r['end_ns']))
        elif line.startswith('FUSION_OBSERVE ') and re.search(r'kind=(C2M|M2C)_CORE ',line):events.append(kv(line))
        elif line.startswith('OPTIMIZED_BLOCK_COUNTERS '):
            r=kv(line);counters[int(r['layer'])]=r
        elif line.startswith('E2E_FUSION '): fusion.append(kv(line))
    assert set(counters)==set(range(blocks)), 'missing block counters'
    assert [int(r['layer']) for r in fusion]==list(range(blocks)), 'missing ordered fusion records'
    assert all(r['previous_statistics_present']==('0' if i==0 else '1') for i,r in enumerate(fusion)), 'cross-layer statistics absent'
    assert any('E2E_FINAL_LN statistics_source=last_block_ffn2 available=1' in l for l in lines), 'final LN skipped'
    assert any(f'executed_blocks={blocks}' in l and 'OURS_VIT_RUNTIME status=PASS' in l for l in lines)
    expected=['embedding',*[f'block_{i}' for i in range(blocks)],'final_ln','classifier']
    prep=[f'block_{i}_preparation' for i in range(blocks)]+['classifier_preparation']
    envelope=phases['online_envelope_including_layer_preparation']; wall=(envelope[1]-envelope[0])/1e9
    spans=sorted((phases[k][0],phases[k][1],k) for k in expected+prep)
    assert all(a<=z for a,z,_ in spans)
    assert all(spans[i][1]<=spans[i+1][0] for i in range(len(spans)-1)), 'overlapping outer phases'
    assert spans[0][0]>=envelope[0] and spans[-1][1]<=envelope[1]
    duration={k:(phases[k][1]-phases[k][0])/1e9 for k in expected+prep}
    other=wall-sum(duration.values());assert other>=-1e-8
    online=wall-sum(duration[k] for k in prep)
    exclusive=[dict(stage=k,seconds=duration[k],party=party) for k in expected]
    exclusive.append(dict(stage='online_other',seconds=other,party=party))
    assert abs(sum(r['seconds'] for r in exclusive)-online)<1e-7
    totals={r['metric']:float(r['value']) for r in table(run/f'execution_totals_party{party}.csv')}
    meta=metadata(run/f'run_metadata_party{party}.txt')
    assert meta['client_decrypt_reencrypt']=='0' and meta['mpc_linear_surrogate']=='0'
    setup=sum((z-a)/1e9 for k,(a,z) in phases.items() if k in prep+['initial_setup','embedding_preparation'])
    rows=[]
    profile=table(run/f'block_online_profile_party{party}.csv')
    for i in range(blocks):
        a,z=phases[f'block_{i}'];counter=counters[i]
        row=dict(block=i,party=party,block_wall_s=duration[f'block_{i}'],HE_counter_s=float(counter['he_s']),MPC_counter_s=float(counter['mpc_nonlinear_s']),layout_counter_s=float(counter['layout_s']))
        for direction in ('C2M','M2C'):
            e=[r for r in events if r['kind']==direction+'_CORE' and a<=int(r['begin_ns']) and int(r['end_ns'])<=z]
            row[direction+'_core_s']=sum((int(r['end_ns'])-int(r['begin_ns']))/1e9 for r in e)
            row[direction+'_batches']=len(e);row[direction+'_ciphertexts']=sum(int(r['count']) for r in e)
        row['counter_remainder_s']=row['block_wall_s']-sum(row[k] for k in ('HE_counter_s','MPC_counter_s','layout_counter_s','C2M_core_s','M2C_core_s'))
        row['counter_scope']='nested attribution; use stage CSV for mutually exclusive sums'
        # Existing profile push/pop partitions nested conversions out of enclosing stages.
        stages=[r for r in profile if int(r['block'])==i and not r['stage'].startswith('BLOCK_ONLINE_')]
        stage_sum=sum(float(r['wall_ms'])/1000 for r in stages)
        block_record=[r for r in profile if int(r['block'])==i and r['stage']=='BLOCK_ONLINE_TIME'][0]
        assert abs(stage_sum-float(block_record['wall_ms'])/1000)<1e-6
        assert stage_sum<=row['block_wall_s']+1e-6
        rows.append(row)
    online_events=[r for r in events if envelope[0]<=int(r['begin_ns']) and int(r['end_ns'])<=envelope[1] and not any(phases[k][0]<=int(r['begin_ns'])<phases[k][1] for k in prep)]
    summary=dict(online_wall_s=online,online_envelope_s=wall,initialization_s=setup,legacy_graph_s=totals['secure_end_to_end_s'],online_send_bytes=int(totals['online_communication_bytes']),total_send_bytes=int(totals['total_communication_bytes']),online_rounds=int(totals['online_communication_rounds']),model_request_preparation_s=(phases['model_request_preparation'][1]-phases['model_request_preparation'][0])/1e9)
    for direction in ('C2M','M2C'):
        e=[r for r in online_events if r['kind']==direction+'_CORE']
        summary[direction+'_batches']=len(e);summary[direction+'_ciphertexts']=sum(int(r['count']) for r in e)
        summary[direction+'_core_s']=sum((int(r['end_ns'])-int(r['begin_ns']))/1e9 for r in e)
    return summary,rows,exclusive,meta,events

def measure(stage,g,image,blocks,repeat,adopt=False):
    assert_frozen();run=O/'runs'/f'{stage}_{blocks}_{g}_image{image}_r{repeat}'
    adopted=run.exists() and adopt
    if run.exists() and not adopted: raise RuntimeError('Refuse overwrite '+str(run))
    state('RUNNING',experiment_stage=stage,group=g,image=image,blocks=blocks,repeat=repeat,run=str(run))
    env=os.environ.copy();env.update(OURS_VIT_PHYSICAL_GPUS='3,4',OURS_VIT_MODEL=str(MODEL),OURS_VIT_INPUT=str(O/'inputs'/f'image_{image}.txt'),OURS_VIT_MATCH_LINEAR_HE_CONTROL='1' if g=='C' else '0',OURS_VIT_MPC_WORKERS='64',OURS_VIT_C2M_HEADROOM_BITS='24',BLB_VIT_OMP_THREADS='64',OURS_VIT_PORT='28200',BLB_VIT_TIMEOUT_SECONDS='7200')
    # No intermediate diagnostics; logits is the ordinary final classifier on a prefix.
    for key in list(env):
        if key.startswith(('OURS_VIT_B_TRACE','OURS_VIT_HETERO_ATTENTION_TRACE','OURS_VIT_NATIVE_DIAGNOSTIC','BLB_VIT_DIAGNOSTIC','OURS_VIT_CAPTURE_ATTENTION_TRACE')):env.pop(key)
    cmd=['bash',str(B/'scripts/run_ours_container.sh'),'B' if g=='B' else 'H',str(plan(g)),str(run.relative_to(B)),str(blocks),'full' if blocks==12 else 'logits']
    cmdfile=run.with_suffix('.command.json')
    if '--feasibility-resume' in sys.argv:
        deadline=json.loads((O/'feasibility_schedule.json').read_text())['deadline_unix']
        remaining=int(deadline-time.time())-60
        if remaining<=0: raise RuntimeError('FEASIBILITY_TIME_BUDGET_EXHAUSTED')
        env['BLB_VIT_TIMEOUT_SECONDS']=str(min(1200 if blocks==2 else 2700,remaining))
    if adopted:
        binding=json.loads(cmdfile.read_text())
        assert binding['binary_sha256']==sha(BIN), 'adopted binary differs'
        while not (run/'status.txt').exists():
            if time.time()>deadline:
                log=run.with_suffix('.launcher.log').read_text()
                names=re.findall(r'OURS_CONTAINER name=(\S+)',log)
                if len(names)==1:subprocess.run(['docker','stop','-t','30',names[0]],check=False)
                raise RuntimeError('FEASIBILITY_TIME_BUDGET_EXHAUSTED waiting for inherited run')
            time.sleep(5)
        codes=metadata(run/'status.txt')
        returncode=int(codes['alice_rc']) or int(codes['bob_rc'])
        launcher_wall='NOT_RUN'  # former supervisor's monotonic start was not persisted
    else:
        write(cmdfile,dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith(('OURS_','BLB_'))},binary_sha256=sha(BIN),reference_sha256=sha(O/'references'/f'{"B" if g=="C" else g}_{blocks}_{image}.npy')))
        start=time.monotonic()
        with run.with_suffix('.launcher.log').open('w') as f: result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        returncode=result.returncode;launcher_wall=time.monotonic()-start
    row=dict(stage=stage,group=g,image=image,blocks=blocks,repeat=repeat,run=str(run),returncode=returncode,launcher_wall_s=launcher_wall,adopted_inflight_run=adopted)
    if returncode: write(run.with_suffix('.result.json'),row);raise RuntimeError('EXECUTION_FAILED '+str(run))
    a=vector(run/'secure_logits_party1.txt');b=vector(run/'secure_logits_party2.txt');ref=np.load(O/'references'/f'{"B" if g=="C" else g}_{blocks}_{image}.npy')
    row.update(relative_L2=float(np.linalg.norm(a-ref)/np.linalg.norm(ref)),max_abs=float(abs(a-ref).max()),prediction_agreement=bool(a.argmax()==ref.argmax()),party_outputs_equal=bool(np.array_equal(a,b)),nonfinite_count=int((~np.isfinite(a)).sum()+(~np.isfinite(b)).sum()))
    passed=row['relative_L2']<=.05 and row['max_abs']<=.5 and row['prediction_agreement'] and row['party_outputs_equal'] and row['nonfinite_count']==0
    row['numerical_pass']=bool(passed);write(run.with_suffix('.result.json'),row)
    if not passed: raise RuntimeError('NUMERICAL_FAIL '+str(run))
    parties=[parse_party(run,p,blocks) for p in (1,2)]
    hekeys=[k for k in parties[0][3] if k.startswith('he') and ('modulus' in k or 'slots' in k or 'scale_bits' in k)]
    assert all(parties[0][3][k]==parties[1][3][k] for k in hekeys)
    params={k:parties[0][3][k] for k in hekeys};write(run/'he_parameters.json',params)
    existing=O/f'he_parameters_{g}.json'
    if existing.exists(): assert json.loads(existing.read_text())==params
    else:write(existing,params)
    if g=='C' and (O/'he_parameters_H.json').exists():
        assert params==json.loads((O/'he_parameters_H.json').read_text()), 'HE control mismatch'
        baseline=json.loads((O/'he_parameters_B.json').read_text())
        write(O/'he_parameter_audit.json',dict(matched_control_pass=True,baseline_vs_linear_differences={k:dict(B=baseline[k],H=params[k]) for k in params if baseline[k]!=params[k]}))
    critical=max(range(2),key=lambda i:parties[i][0]['online_wall_s'])
    row.update(parties[critical][0]);row['critical_party']=critical+1
    row['initialization_s']=max(p[0]['initialization_s'] for p in parties)
    row['communication_bytes']=sum(p[0]['online_send_bytes'] for p in parties)
    for i,p in enumerate(parties,1):
        for k in ('online_send_bytes','total_send_bytes','online_rounds','online_wall_s'):row[f'party{i}_{k}']=p[0][k]
    def stamp(p):return datetime.datetime.fromisoformat(p.read_text().strip().replace(',','.'))
    row['request_wall_s']=(stamp(run/'end.txt')-stamp(run/'start.txt')).total_seconds()
    csvwrite(run/'block_components.csv',[r for p in parties for r in p[1]])
    csvwrite(run/'exclusive_online_phases.csv',[r for p in parties for r in p[2]])
    write(run/'conversion_events.json',{str(i):p[4] for i,p in enumerate(parties,1)})
    row['timing_closure_pass']=True;row['status']='PASS';write(run.with_suffix('.result.json'),row)
    return row

def publish(rows):
    csvwrite(O/'per_run.csv',rows);csvwrite(O/'correctness.csv',[r for r in rows if r['stage']=='correctness'])
    metrics=['online_wall_s','request_wall_s','initialization_s','legacy_graph_s','communication_bytes','C2M_core_s','M2C_core_s','C2M_batches','M2C_batches','C2M_ciphertexts','M2C_ciphertexts','party1_online_rounds','party2_online_rounds']
    summary=[];decomp=[];components=[]
    for g in GROUPS:
        rs=[r for r in rows if r['stage']=='performance' and r['group']==g]
        r=dict(group=g,repeats=len(rs),status='COMPLETE' if len(rs)==5 else 'NOT_RUN_OR_INCOMPLETE')
        for k in metrics:
            for suffix,fn in [('mean',statistics.mean),('std',statistics.stdev),('median',statistics.median)]:r[k+'_'+suffix]=fn([x[k] for x in rs]) if len(rs)>=(2 if suffix=='std' else 1) else 'NOT_RUN'
        summary.append(r)
        if not rs:continue
        phase_values={};block_values={}
        for x in rs:
            for p in table(Path(x['run'])/'exclusive_online_phases.csv'):
                if int(p['party'])==x['critical_party']:phase_values.setdefault(p['stage'],[]).append(float(p['seconds']))
            for p in table(Path(x['run'])/'block_components.csv'):
                if int(p['party'])==x['critical_party']:
                    for k in ('block_wall_s','HE_counter_s','MPC_counter_s','layout_counter_s','C2M_core_s','M2C_core_s','counter_remainder_s'):
                        block_values.setdefault((int(p['block']),k),[]).append(float(p[k]))
        for stage,values in phase_values.items():decomp.append(dict(group=g,stage=stage,n=len(values),mean_s=statistics.mean(values),std_s=statistics.stdev(values) if len(values)>1 else 'NOT_RUN',median_s=statistics.median(values)))
        for (block,k),values in block_values.items():components.append(dict(group=g,block=block,metric=k,n=len(values),mean_s=statistics.mean(values),std_s=statistics.stdev(values) if len(values)>1 else 'NOT_RUN',median_s=statistics.median(values),scope='nested counters; not additive with exclusive phases'))
    csvwrite(O/'performance_summary.csv',summary);csvwrite(O/'exclusive_breakdown.csv',decomp);csvwrite(O/'block_component_summary.csv',components)
    if all(r['repeats']==5 for r in summary):
        b,h,c=summary
        text=['Stage 2.4: measured full-network secure inference','', '| Method | n | Secure online wall (s) | Request wall (s) | Initialization (s) | Communication (GB) |','|---|---:|---:|---:|---:|---:|']
        for r in summary:
            fmt=lambda k:f"{r[k+'_mean']:.3f} ± {r[k+'_std']:.3f}"
            text.append(f"| {r['group']} | 5 | {fmt('online_wall_s')} | {fmt('request_wall_s')} | {fmt('initialization_s')} | {r['communication_bytes_mean']/1e9:.3f} |")
        text+=['','B: frozen all-Softmax; H: only Block 0 is 10S/1R/1L; C: all-Softmax control with the same existing extended HE context as H.', '',f"All {sum(r['stage']=='correctness' for r in rows)} correctness runs and 15 performance runs passed unchanged logits gates (relative-L2 ≤0.05, max-abs ≤0.5, prediction agreement, finite outputs, identical parties). Each group used three distinct images at both 2 and 12 blocks. No intermediate activation reconstruction.",'',f"Measured mean speedup B/H: {b['online_wall_s_mean']/h['online_wall_s_mean']:.4f}×; matched-context C/H: {c['online_wall_s_mean']/h['online_wall_s_mean']:.4f}×.",f"Measured whole-network conversion core H−B: C2M {h['C2M_core_s_mean']-b['C2M_core_s_mean']:+.3f}s; M2C {h['M2C_core_s_mean']-b['M2C_core_s_mean']:+.3f}s. These are observed differences, not attribution of the entire latency difference.",'','The exclusive table partitions each selected critical party’s online envelope minus measured interleaved offline preparation. Embedding, 12 blocks, final LN, classifier and online_other close to measured online wall. Nested HE/MPC/conversion counters and memory transfer timings remain separate; do not add them to outer phases. Initialization includes streamed per-layer preparation. Launcher wall additionally includes container startup and the independent head-reorder preflight. Communication is the sum of both parties’ sent bytes; protocol-round counters are recorded separately per party, not summed as network RTTs.','', '| Method | Exclusive phase | Mean (s) | Std (s) | Median (s) |','|---|---|---:|---:|---:|']
        text += [f"| {r['group']} | {r['stage']} | {r['mean_s']:.6f} | {r['std_s']:.6f} | {r['median_s']:.6f} |" for r in decomp]
        text+=['','Provenance: frozen_hashes.json, contract.json, samples.csv, per-run command/result JSON and both party logs in runs/. These are three internal ImageNet validation selection images, not an independent official test set. Performance repeats use fixed image 0, interleaved group order. No 6S/3R/3L run is scheduled.']
        (O/'RESULTS.md').write_text('\n'.join(text)+'\n')

def feasibility():
    """One fixed image, two paths, prefix then complete network. Stop for review."""
    assert_frozen()
    rows=[]
    for g,blocks,image in json.loads((O/'feasibility_schedule.json').read_text())['ordered_runs']:
        if time.time()>=json.loads((O/'feasibility_schedule.json').read_text())['deadline_unix']:
            raise RuntimeError('FEASIBILITY_TIME_BUDGET_EXHAUSTED')
        run=O/'runs'/f'correctness_{blocks}_{g}_image{image}_r0'
        result=run.with_suffix('.result.json')
        if result.exists() and json.loads(result.read_text()).get('status')=='PASS':
            row=json.loads(result.read_text())
        else:
            row=measure('correctness',g,image,blocks,0,adopt=run.exists())
        rows.append(row);publish(rows)
        csvwrite(O/'feasibility.csv',rows)
    baseline=next(r for r in rows if r['group']=='B' and r['blocks']==12)
    mixed=next(r for r in rows if r['group']=='H' and r['blocks']==12)
    write(O/'feasibility_result.json',dict(status='PASS_ONE_IMAGE_FEASIBILITY_ONLY',correctness_runs=4,performance_repeats=0,full_network_online_s={r['group']:r['online_wall_s'] for r in (baseline,mixed)},single_observation_ratio_B_over_H=baseline['online_wall_s']/mixed['online_wall_s'],numerical_gates_unchanged=True,formal_speedup_claim_supported=False,matched_HE_control='NOT_RUN',other_images='NOT_RUN',stop_for_review=True))
    state('FEASIBILITY_COMPLETE_AWAITING_REVIEW',correctness_runs=4,performance_runs=0,formal_campaign_started=False)


def main():
    if (O/'status.json').exists():raise RuntimeError('Existing campaign: inspect before resuming; never overwrite')
    prepare();rows=[];publish(rows)
    # Entire prefix gate before any full run; entire correctness gate before timing.
    for blocks in (2,12):
        for image in range(3):
            for g in GROUPS:
                rows.append(measure('correctness',g,image,blocks,0));publish(rows)
    state('CORRECTNESS_PASS',runs=len(rows))
    for repeat in range(1,6):
        order=GROUPS[(repeat-1)%3:]+GROUPS[:(repeat-1)%3]
        for g in order:
            rows.append(measure('performance',g,0,12,repeat));publish(rows)
    assert_frozen();state('COMPLETE',correctness_runs=18,performance_runs=15,no_next_configuration_started=True)

if __name__=='__main__':
    try:
        if '--feasibility-resume' in sys.argv:feasibility()
        else:main()
    except Exception as e:
        import traceback
        traceback.print_exc();state('STOPPED_FAILURE',reason=str(e),performance_continuation_prohibited=True);sys.exit(1)
