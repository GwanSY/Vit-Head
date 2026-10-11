#!/usr/bin/env python3
"""Supplement A scheduler: exactly 24,694 new complete NumPy reference calls.
Model equations are exclusively the existing vit_reference.Graph.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,collections,csv,datetime,hashlib,json,multiprocessing as mp,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
import run_stage22_accuracy as a
from run_stage22_identity import metric_rows,newplan,plansave
ROOT=a.B/'outputs/stage22_supplement_A_minimal_20261011'
PRIOR=a.B/'outputs/stage22_identity_20261010'
UNIFORM=a.B/'outputs/stage22_experiment1_accuracy_20261010'
IDS=['original_M33','random_17_M33','random_29_M33','swapped_M33']
MODEL=None;PLANS={}

def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def save(p,x):a.dump(p,x)
def status(stage,**kw):
    x=dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw);save(ROOT/'status.json',x);print(json.dumps(x),flush=True)
def verify():
    for p,h in read(ROOT/'frozen_hashes.json').items():assert a.sha(p)==h,p

def preflight():
    for p,h in read(PRIOR/'frozen_hashes.json').items():assert a.sha(p)==h,p
    audit=read(PRIOR/'audit.json');assert all(audit['checks'].values())
    assert a.sha(a.OLD/'raw/native_selection_predictions.jsonl')==audit['old_prediction_sha256']
    assert a.sha(UNIFORM/'raw/per_image_predictions.jsonl')==audit['new_prediction_sha256']
    baseline=rows(UNIFORM/'raw/per_image_predictions.jsonl');legacy={r['image']:r for r in rows(a.OLD/'raw/native_selection_predictions.jsonl')};partial=rows(PRIOR/'selection/raw/per_image_predictions.jsonl')
    assert len(baseline)==len(legacy)==5000 and len(partial)==153
    byimage={r['image']:r for r in baseline};assert len(byimage)==5000
    plans=[newplan('all_S','S'*12,'baseline')]
    heads=read(a.OLD/'fixed_head_plans.json')[6]['heads'];assert heads[0]==list('RRRSSSLLLSSS') and all(h==list('S'*12) for h in heads[1:])
    plans.append(newplan('original_M33',''.join(heads[0]),'existing_reference'))
    for pid in IDS[1:3]:
        p=read(PRIOR/f'selection/plans/{pid}.json');plans.append(p)
    plans.append(newplan('swapped_M33','LLLSSSRRRSSS','same_six_type_swap'))
    for p in plans[1:]:assert collections.Counter(p['block0'])=={'S':6,'R':3,'L':3}
    assert len({p['block0'] for p in plans[1:]})==4
    assert {h for h,t in enumerate(plans[1]['block0']) if t!='S'}=={h for h,t in enumerate(plans[-1]['block0']) if t!='S'}
    plansave(ROOT,plans)
    for p in plans:PLANS[p['plan_id']]=p
    for pid in IDS[1:3]:
        prev=read(PRIOR/f'selection/plans/{pid}.json');assert PLANS[pid]['head_plan']==prev['head_plan'] and PLANS[pid]['layers']==prev['layers']
    records={}
    for r in baseline:
        old=legacy[r['image']];assert r['label']==old['label'] and r['predictions']['all_S']['top5']==old['predictions']['0']
        records[r['image']]=dict(index=r['index'],image=r['image'],label=r['label'],input_sha256=r['input_sha256'],predictions={'all_S':dict(r['predictions']['all_S'],source='experiment1_completed'), 'original_M33':dict(top5=old['predictions']['6'],source='historical_optimized_native_reference')})
    seen=set();reuse=[]
    for r in partial:
        assert r['image'] not in seen;seen.add(r['image']);b=byimage[r['image']]
        assert r['status']=='PASS' and r['label']==b['label'] and r['input_sha256']==b['input_sha256'] and r['predictions']['all_S']['top5']==b['predictions']['all_S']['top5']
        for pid in IDS[1:3]:
            pred=r['predictions'][pid];assert len(pred['top5'])==5 and len(set(pred['top5']))==5 and len(pred['logits_sha256'])==64 and np.isfinite(pred['top5_logits']).all()
            records[r['image']]['predictions'][pid]=dict(pred,source='paused_campaign_reused')
            reuse.append(dict(image=r['image'],label=r['label'],plan_id=pid,input_sha256=r['input_sha256'],logits_sha256=pred['logits_sha256'],source_file=str(PRIOR/'selection/raw/per_image_predictions.jsonl')))
    assert len(reuse)==306;a.writecsv(ROOT/'reuse_306.csv',reuse)
    # Keep exact inherited splits and the already precommitted independent subset.
    (ROOT/'splits').mkdir()
    for name in ('selection','calibration','holdout'):(ROOT/f'splits/{name}.csv').write_bytes((a.OLD/f'splits/{name}.csv').read_bytes())
    (ROOT/'splits/validation.csv').write_bytes((PRIOR/'final_validation_manifest.csv').read_bytes())
    sets={name:{r[0] for r in a.readcsv(ROOT/f'splits/{name}.csv')} for name in ('selection','calibration','holdout','validation')}
    assert len(sets['validation'])==5000 and sets['validation']<=sets['holdout']
    assert not(sets['selection']&sets['holdout'] or sets['calibration']&sets['holdout'] or sets['calibration']&sets['selection'])
    assert collections.Counter(int(r[1]) for r in a.readcsv(ROOT/'splits/validation.csv'))=={i:5 for i in range(1000)}
    assert not (PRIOR/'validation/raw/per_image_predictions.jsonl').exists()
    assert set(records)==sets['selection']
    (ROOT/'alpha.json').write_bytes((a.OLD/'alpha.json').read_bytes())
    contract=dict(scope='Stage2.2 supplement A minimal; CPU NumPy complete 12-layer protocol-clear accuracy only',new_inferences={'random17':4847,'random29':4847,'type_swap':5000,'validation_allS':5000,'validation_candidate':5000,'total':24694},reused_random_predictions=306,random_seeds=[17,29],candidate_rule={'eligible':IDS,'criterion':'maximum selection Top1 correct count','tie_break':IDS},validation_rule='Use inherited precommitted 5000-image holdout subset. Lock exactly one candidate before reading any validation image. No reselection.',statistics='Top1 and Top5, paired normal approximate 95% CI, exact two-sided McNemar; Holm3 for the three selection comparisons against original_M33. Final validation compares only candidate vs allS, not candidate vs other same-budget plans.',limits='No extra plans, no full head sensitivity, no heatmap, no training/cost allocator, no secure runs.',negative_results='Report insufficient evidence without appending plans.',historical_reference_metadata_warning=audit['historical_metadata_warning'],model=str(a.MODEL_PATH),physical_tokens=256,logical_tokens=197,block=0,other_blocks='all Softmax',accuracy_backend='existing CPU NumPy Graph(fixed=True)',preprocessing_python=a.PREPROCESS_PYTHON)
    save(ROOT/'contract.json',contract);save(ROOT/'reuse_audit.json',dict(status='PASS',reused_predictions=306,images=153,historical_audit=audit,partial_sha256=a.sha(PRIOR/'selection/raw/per_image_predictions.jsonl'),baseline_sha256=a.sha(UNIFORM/'raw/per_image_predictions.jsonl')))
    files=[Path(__file__),Path(a.__file__),Path(__file__).with_name('run_stage22_identity.py'),a.B/'ours_vit_b16/scripts/vit_reference.py',a.B/'ours_vit_b16/scripts/preprocess_vit_image.py',a.MODEL_PATH,a.MANIFEST,ROOT/'alpha.json',ROOT/'contract.json',ROOT/'head_plans.csv',ROOT/'reuse_audit.json',*sorted((ROOT/'plans').glob('*.json')),*sorted((ROOT/'splits').glob('*.csv'))]
    save(ROOT/'frozen_hashes.json',{str(p):a.sha(p) for p in files})
    with (ROOT/'reused_predictions.jsonl').open('w') as f:
        for r in records.values():f.write(json.dumps(r)+'\n')
    return records

def compute(job):
    sample,ids=job;result=dict(sample,predictions={});start=time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='minimal_attention_') as tmp:
            inp=Path(tmp)/'input.txt'
            subprocess.run([a.PREPROCESS_PYTHON,str(a.B/'ours_vit_b16/scripts/preprocess_vit_image.py'),str(a.V/'datasets/imagenet2012/val'/sample['image']),'--manifest',str(a.MANIFEST),'--output',str(inp)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            result['input_sha256']=a.sha(inp)
            if 'input_sha256' in sample:assert result['input_sha256']==sample['input_sha256']
            x=a.vr.load_input(inp)
        for pid in ids:
            g=a.vr.Graph(MODEL,fixed=True,physical_tokens=256,head_plan=PLANS[pid]['head_plan'],record_values=False)
            assert g.executed_layers==12 and not g.stop_after_attention
            y,_=g.run(x);assert y.size==1000 and np.isfinite(y).all(),pid
            top=np.argsort(-y,kind='stable')[:5]
            result['predictions'][pid]=dict(top5=top.tolist(),top5_logits=y[top].tolist(),logits_sha256=hashlib.sha256(y.tobytes()).hexdigest(),source='new_CPU_NumPy_complete_graph')
        result['status']='PASS'
    except Exception as e:result.update(status='FAILED',error=repr(e))
    result['scheduling_wall_s']=time.monotonic()-start
    return result

def runphase(name,records,jobs,workers,expected,previous=0):
    folder=ROOT/name;folder.mkdir(exist_ok=True);count=0;done=0;start=time.monotonic()
    assert sum(len(ids) for _,ids in jobs)==expected
    with (folder/'new_predictions.jsonl').open('w') as f:
        # Small batch is part of the budget, never an extra inference repeat.
        for subset,nproc in [(jobs[:3],3),(jobs[3:],workers)]:
            with mp.get_context('fork').Pool(nproc) as pool:
                for r in pool.imap(compute,subset,chunksize=1):
                    f.write(json.dumps(r)+'\n');f.flush()
                    if r['status']!='PASS':raise RuntimeError(r.get('error'))
                    target=records[r['image']];assert target['label']==r['label']
                    if 'input_sha256' in target:assert target['input_sha256']==r['input_sha256']
                    target['input_sha256']=r['input_sha256']
                    assert not(set(target['predictions'])&set(r['predictions']))
                    target['predictions'].update(r['predictions']);count+=len(r['predictions']);done+=1
                    assert previous+count<=24694
                    if done<=3 or done%25==0:
                        elapsed=time.monotonic()-start
                        status(name,completed_images=done,total_images=len(jobs),new_inferences=count,phase_inferences=expected,total_new_inferences=previous+count,workers=workers,elapsed_s=elapsed,estimated_phase_remaining_s=(expected-count)*elapsed/count)
    assert count==expected;verify();return count

def output(name,records,pids):
    folder=ROOT/name;rr=list(records.values());assert len(rr)==5000
    changes=[]
    with (folder/'per_image_predictions.jsonl').open('w') as f:
        for r in rr:
            assert set(r['predictions'])==set(pids)
            base=r['predictions']['all_S']['top5'][0]==r['label']
            for pid,pred in r['predictions'].items():
                pred['top1_correct']=pred['top5'][0]==r['label'];pred['top5_correct']=r['label'] in pred['top5'];pred['paired_correctness_change_vs_allS']=int(pred['top1_correct'])-int(base)
                ref=r['predictions'].get('original_M33')
                if ref is not None:pred['paired_correctness_change_vs_original_M33']=int(pred['top1_correct'])-int(ref['top5'][0]==r['label'])
                changes.append(dict(image=r['image'],label=r['label'],plan_id=pid,top1_correct=pred['top1_correct'],top5_correct=pred['top5_correct'],change_vs_allS=pred['paired_correctness_change_vs_allS'],change_vs_original_M33=pred.get('paired_correctness_change_vs_original_M33','NOT_APPLICABLE')))
            f.write(json.dumps(r)+'\n')
    a.writecsv(folder/'per_image_correctness.csv',changes)
    metrics=metric_rows(rr,[PLANS[pid] for pid in pids]);a.writecsv(folder/'accuracy.csv',metrics)
    reference='original_M33' if name=='selection' else 'all_S'
    pairs=[]
    for pid in (IDS[1:] if name=='selection' else [p for p in pids if p!='all_S']):
        for k in (1,5):
            lhs=np.array([r['label'] in r['predictions'][reference]['top5'][:k] for r in rr]);rhs=np.array([r['label'] in r['predictions'][pid]['top5'][:k] for r in rr])
            d=a.paired(lhs,rhs);pairs.append(dict(reference=reference,plan_id=pid,metric=f'Top{k}',loss_vs_reference_pp=-d['gain_pp'],loss_ci95_low_pp=-d['gain_ci95_high_pp'],loss_ci95_high_pp=-d['gain_ci95_low_pp'],**d))
    for k in (1,5):a.holm([r for r in pairs if r['metric']==f'Top{k}'])
    a.writecsv(folder/'paired_comparisons.csv',pairs)
    save(folder/'completion.json',dict(status='COMPLETE',images=5000,plans=pids));return metrics,pairs

def main():
    global MODEL
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=64);args=p.parse_args();assert 1<=args.workers<=96
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'status.json').exists(),'Do not overwrite an existing experiment'
    status('PREFLIGHT');selection=preflight()
    save(ROOT/'execution_environment.json',dict(python=sys.version,numpy=np.__version__,workers=args.workers,cpu_affinity=len(os.sched_getaffinity(0)),gpu_used=False,secure_runs=0,thread_limits={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    jobs=[]
    for r in selection.values():
        missing=[pid for pid in IDS[1:] if pid not in r['predictions']]
        jobs.append(({k:r[k] for k in ('index','image','label','input_sha256')},missing))
    assert collections.Counter(pid for _,ids in jobs for pid in ids)=={'random_17_M33':4847,'random_29_M33':4847,'swapped_M33':5000}
    status('LOADING_MODEL',remaining_inferences=24694,reused_predictions=306);MODEL=a.vr.load_model(a.MODEL_PATH)
    count=runphase('selection',selection,jobs,args.workers,14694)
    sm,sp=output('selection',selection,['all_S']+IDS)
    # Deterministic tie break, no peeking at validation images.
    candidate=max(IDS,key=lambda pid:sum(r['predictions'][pid]['top5'][0]==r['label'] for r in selection.values()))
    lock=dict(candidate=candidate,plan=PLANS[candidate],plan_sha256=a.sha(ROOT/f'plans/{candidate}.json'),selection_predictions_sha256=a.sha(ROOT/'selection/per_image_predictions.jsonl'),validation_manifest_sha256=a.sha(ROOT/'splits/validation.csv'),locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tie_order=IDS)
    save(ROOT/'candidate_lock.json',lock);lockhash=a.sha(ROOT/'candidate_lock.json')
    validation={r[0]:dict(index=5000+i,image=r[0],label=int(r[1]),predictions={}) for i,r in enumerate(a.readcsv(ROOT/'splits/validation.csv'))}
    jobs=[({k:r[k] for k in ('index','image','label')},['all_S',candidate]) for r in validation.values()]
    count+=runphase('validation',validation,jobs,args.workers,10000,count)
    vm,vp=output('validation',validation,['all_S',candidate]);assert count==24694 and a.sha(ROOT/'candidate_lock.json')==lockhash
    assert a.sha(ROOT/f'plans/{candidate}.json')==lock['plan_sha256'];verify()
    lines=['阶段2.2 补充实验A：固定预算的头级配置对照','', '所有结果来自冻结 CPU NumPy 协议明文完整12层 ViT。仅 Block 0 改变，其他11层全 Softmax；没有密文性能或通信测量。','', '| Selection configuration | Block 0 types (head 0–11) | Top-1 (%) | Top-5 (%) | Loss vs all S (pp) | Paired 95% CI (pp) |','|---|---|---:|---:|---:|---|']
    for r in sm:lines.append(f"| {r['plan_id']} | {r['head_types']} | {r['Top1_pct']:.2f} | {r['Top5_pct']:.2f} | {r['loss_vs_allS_pp']:.2f} | [{r['loss_ci95_low_pp']:.2f}, {r['loss_ci95_high_pp']:.2f}] |")
    lines+=['','| Comparison vs original 6S+3R+3L | Top-1 gain (pp) | Paired 95% CI (pp) | Holm p |','|---|---:|---|---:|']
    for r in sp:
        if r['metric']=='Top1':lines.append(f"| {r['plan_id']} | {r['gain_pp']:.2f} | [{r['gain_ci95_low_pp']:.2f}, {r['gain_ci95_high_pp']:.2f}] | {r['holm_p']:.4g} |")
    random_tests=[r for r in sp if r['metric']=='Top1' and r['plan_id'].startswith('random_')];swap=next(r for r in sp if r['metric']=='Top1' and r['plan_id']=='swapped_M33')
    lines+=['','固定 S/R/L 数量后：'+('至少一个预先固定随机分配与原配置存在经 Holm 校正显著的 Top-1 差异，支持当前选择集上的头级配置影响。' if any(r['holm_p']<.05 for r in random_tests) else '两个固定随机对照未提供经校正显著的 Top-1 差异证据；证据不足，不追加计划。'),'','固定六头身份后：'+('交换 R/L 类型与原配置存在经校正显著的 Top-1 差异，支持类型分配影响。' if swap['holm_p']<.05 else '交换 R/L 类型未提供经校正显著的 Top-1 差异证据。'),'','随机配置对照保持数量相同，但并不单独分离头位置与类型指派的贡献。结果不构成头选择算法优于随机的证据。置信区间为配对正态近似95%区间；检验为双侧精确 McNemar，选择集的三项比较按 Top-1/Top-5 分别做 Holm 校正。', '',f'锁定候选：{candidate}。独立验证只验证此候选与全 S 的精度，不能独立复证不同同预算配置之间的差异。', '', '| Independent validation | Top-1 (%) | Top-5 (%) | Loss vs all S (pp) | Paired 95% CI (pp) |','|---|---:|---:|---:|---|']
    for r in vm:lines.append(f"| {r['plan_id']} | {r['Top1_pct']:.2f} | {r['Top5_pct']:.2f} | {r['loss_vs_allS_pp']:.2f} | [{r['loss_ci95_low_pp']:.2f}, {r['loss_ci95_high_pp']:.2f}] |")
    lines+=['','选择集5,000张和独立验证5,000张均来自官方 ImageNet validation 内部划分；验证为此前冻结的43,000张保留集子集，其他38,000张未评估。α来自此前2,000张校准集，未重新拟合。复用306次随机预测，新增完整推理恰为24,694次。原始逐图预测、配对变化和统计见 selection/ 与 validation/，版本哈希见 frozen_hashes.json，头身份见 head_plans.csv，候选锁定见 candidate_lock.json。历史元数据 reference 哈希与实际 native source 不符的问题保留披露于 reuse_audit.json；历史文件未改。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');status('COMPLETE',new_inferences=count,reused_predictions=306,candidate=candidate,secure_runs=0)

if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback;traceback.print_exc()
        if ROOT.exists():status('STOPPED_FAILURE',error=repr(e))
        raise
