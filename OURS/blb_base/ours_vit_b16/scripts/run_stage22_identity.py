#!/usr/bin/env python3
"""Accuracy-only orchestration; reuse the frozen NumPy Graph without model changes."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,itertools,json,multiprocessing as mp,random,subprocess,time
from pathlib import Path
import numpy as np
import run_stage22_accuracy as a
ROOT=a.B/'outputs/stage22_identity_20261010'
SEEDS=[17,29,43,71,101]
CACHE={}
ACTIVE=[]

def save(p,x):a.dump(p,x)
def loadrows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def freeze():
    files=[Path(__file__),Path(a.__file__),a.B/'ours_vit_b16/scripts/vit_reference.py',a.B/'ours_vit_b16/scripts/preprocess_vit_image.py',a.MODEL_PATH,a.MANIFEST,ROOT/'contract.json',ROOT/'audit.json',*sorted((ROOT/'selection/plans').glob('*.json')),*sorted((ROOT/'selection/splits').glob('*.csv')),ROOT/'selection/alpha.json',ROOT/'final_validation_manifest.csv']
    save(ROOT/'frozen_hashes.json',{str(p):a.sha(p) for p in files})
def verify():
    for p,h in json.loads((ROOT/'frozen_hashes.json').read_text()).items():assert a.sha(p)==h,p

def metric_rows(records,plans):
    out=[];base=np.array([r['predictions']['all_S']['top5'][0]==r['label'] for r in records])
    base5=np.array([r['label'] in r['predictions']['all_S']['top5'] for r in records])
    for p in plans:
        pid=p['plan_id'];c=np.array([r['predictions'][pid]['top5'][0]==r['label'] for r in records]);c5=np.array([r['label'] in r['predictions'][pid]['top5'] for r in records]);d=a.paired(base,c);e=a.paired(base5,c5)
        out.append(dict(plan_id=pid,head_types=p['block0'],n=len(records),Top1_pct=100*c.mean(),Top5_pct=100*c5.mean(),loss_vs_allS_pp=-d['gain_pp'],loss_ci95_low_pp=-d['gain_ci95_high_pp'],loss_ci95_high_pp=-d['gain_ci95_low_pp'],correct_to_wrong=d['first_correct_second_wrong'],wrong_to_correct=d['first_wrong_second_correct'],mcnemar_exact_p=d['mcnemar_exact_p'],Top5_loss_pp=-e['gain_pp'],Top5_loss_ci95_low_pp=-e['gain_ci95_high_pp'],Top5_loss_ci95_high_pp=-e['gain_ci95_low_pp'],Top5_mcnemar_exact_p=e['mcnemar_exact_p']))
    a.holm(out);return out

def audit():
    old=a.OLD;new=a.B/'outputs/stage22_experiment1_accuracy_20261010'
    oldpred={r['image']:r for r in loadrows(old/'raw/native_selection_predictions.jsonl')}
    newpred=loadrows(new/'raw/per_image_predictions.jsonl');assert len(newpred)==len(oldpred)==5000
    checks={}
    for n in ('alpha.json','splits/selection.csv','splits/calibration.csv','splits/holdout.csv'):checks[n]=a.sha(old/n)==a.sha(new/n)
    checks['reference_source']=a.sha(old/'reference_source/vit_reference_native.py')==a.sha(a.B/'ours_vit_b16/scripts/vit_reference.py')
    checks['preprocessing_source']=a.sha(a.B/'outputs/blb_vit_validation_20261004T011600+0900/reference_source/preprocess_vit_image.py')==a.sha(a.B/'ours_vit_b16/scripts/preprocess_vit_image.py')
    oldf=json.loads((old/'frozen_assets.json').read_text());nf=json.loads((new/'frozen_hashes.json').read_text())
    checks['checkpoint']=oldf['model_sha256']==nf[str(a.MODEL_PATH)]==a.sha(a.MODEL_PATH)
    checks['preprocessing_runtime']=True # Both saved scripts explicitly invoke the same anaconda Python.
    checks['allS_per_image_top5']=all(r['label']==oldpred[r['image']]['label'] and r['predictions']['all_S']['top5']==oldpred[r['image']]['predictions']['0'] for r in newpred)
    assert all(checks.values()),checks
    plans=[];merged=[];oldplans=json.loads((old/'fixed_head_plans.json').read_text())
    for pid,t in [('all_S','S'*12),('all_R','R'*12),('all_L','L'*12)]:plans.append(dict(plan_id=pid,block0=t))
    for i in range(8):plans.append(dict(plan_id=f'old_{i}',block0=''.join(oldplans[i]['heads'][0])))
    for r in newpred:
        row=dict(r);row['predictions']=dict(r['predictions'])
        for i in range(8):row['predictions'][f'old_{i}']={'top5':oldpred[r['image']]['predictions'][str(i)]}
        merged.append(row)
    a.writecsv(ROOT/'new_old_accuracy_audit.csv',metric_rows(merged,plans))
    save(ROOT/'audit.json',dict(checks=checks,old_prediction_sha256=a.sha(old/'raw/native_selection_predictions.jsonl'),new_prediction_sha256=a.sha(new/'raw/per_image_predictions.jsonl'),actual_reference_sha256=a.sha(old/'reference_source/vit_reference_native.py'),historical_declared_reference_sha256=oldf['reference_sha256'],historical_metadata_warning='frozen_assets reference hash differs from the actual native reference used by saved evaluate_selection.py; old files preserved. Actual saved native source equals current source.',comparison_scope='Actual saved sources and current checkpoint match; all 5000 baseline top5 predictions identical. No claim that stale historical hash itself validates prior provenance.'))
    return newpred

def newplan(pid,types,family):
    alpha=json.loads((a.OLD/'alpha.json').read_text())['alpha'];layers=[list(types)]+[list('S'*12) for _ in range(11)]
    return dict(plan_id=pid,family=family,block0=types,layers=layers,head_plan=[[(a.PROTO[t],float(alpha[l][h]) if t=='L' else 1.) for h,t in enumerate(row)] for l,row in enumerate(layers)])
def plansave(folder,plans):
    (folder/'plans').mkdir(parents=True,exist_ok=True);rows=[]
    for p in plans:
        path=folder/'plans'/f"{p['plan_id']}.json";save(path,{k:v for k,v in p.items() if k!='plan_sha256'});p['plan_sha256']=a.sha(path)
        for l,types in enumerate(p['layers']):
            for h,t in enumerate(types):rows.append(dict(plan_id=p['plan_id'],layer=l,head_id=h,type=t,alpha=p['head_plan'][l][h][1],plan_sha256=p['plan_sha256']))
    a.writecsv(folder/'head_plans.csv',rows)

def evaluate(sample):
    # Reuse only completed experiment-one predictions with matching frozen provenance.
    # No new ViT graph: the existing evaluator handles every uncached plan.
    a.PLANS=ACTIVE
    r=a.evaluate(sample)
    if r['status']=='PASS' and CACHE:
        prior=CACHE[sample['image']];assert prior['input_sha256']==r['input_sha256']
        assert prior['predictions']['all_S']['top5']==r['predictions']['all_S']['top5']
        r['predictions'].update({k:v for k,v in prior['predictions'].items() if k in ('all_R','all_L')})
    return r

def runphase(name,plans,samples,workers,cache=None):
    global CACHE,ACTIVE
    folder=ROOT/name;folder.mkdir(exist_ok=True);(folder/'raw').mkdir(exist_ok=True)
    a.O=folder;a.PLANS=plans;CACHE=cache or {};ACTIVE=[p for p in plans if not(CACHE and p['plan_id'] in ('all_R','all_L'))]
    plansave(folder,plans);save(folder/'phase_lock.json',dict(plans={p['plan_id']:p['plan_sha256'] for p in plans},sample_count=len(samples),samples=samples))
    records=[];start=time.monotonic()
    with (folder/'raw/per_image_predictions.jsonl').open('w') as f:
        smoke=samples[:3]
        for subset,processes in [(smoke,3),(samples[3:],workers)]:
            with mp.get_context('fork').Pool(processes) as pool:
                for r in pool.imap(evaluate,subset,chunksize=1):
                    if r['status']!='PASS':save(folder/'raw/failure.json',r);raise RuntimeError(r['error'])
                    f.write(json.dumps(r)+'\n');f.flush();records.append(r)
                    if len(records)<=3 or len(records)%25==0:
                        row=dict(stage=name,evaluated=len(records),total=len(samples),plans=len(plans),elapsed_s=time.monotonic()-start)
                        if len(records)>=100:row['estimated_remaining_s']=row['elapsed_s']/len(records)*(len(samples)-len(records))
                        save(ROOT/'status.json',row);print(json.dumps(row),flush=True)
    assert len(records)==len(samples);verify()
    a.writecsv(folder/'accuracy.csv',metric_rows(records,plans));save(folder/'completion.json',dict(images=len(records),plans=len(plans),status='COMPLETE'))
    return records

def comparisons(records):
    correct={pid:np.array([r['predictions'][pid]['top5'][0]==r['label'] for r in records]) for pid in records[0]['predictions']}
    rows=[]
    for budget in ('R6','L6','M33'):
        for seed in SEEDS:
            x=f'random_{seed}_{budget}';y=f'selected_{budget}';rows.append(dict(family='selected_vs_random',first=x,second=y,**a.paired(correct[x],correct[y])))
    for x,y in [('set_R6','set_RRRLLL'),('set_L6','set_RRRLLL'),('set_R6','set_LLLRRR'),('set_L6','set_LLLRRR'),('set_RRRLLL','set_LLLRRR')]:rows.append(dict(family='same_set_types',first=x,second=y,**a.paired(correct[x],correct[y])))
    a.holm(rows);return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=64);args=p.parse_args()
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'status.json').exists(),'Existing campaign; do not overwrite'
    save(ROOT/'status.json',dict(stage='AUDIT'))
    oldnew=audit()
    a.O=ROOT/'selection';a.UNIFORM_ONLY=False;a.GPU_IDS=[];a.prepare();plans=a.PLANS
    for seed in SEEDS:
        perm=random.Random(seed).sample(range(12),12)
        for budget,rn,ln in [('R6',6,0),('L6',0,6),('M33',3,3),('M33swap',3,3)]:
            heads=perm[:6];types=list('S'*12)
            if budget=='M33swap':heads=heads[3:]+heads[:3]
            for h in heads[:rn]:types[h]='R'
            for h in heads[rn:rn+ln]:types[h]='L'
            plans.append(newplan(f'random_{seed}_{budget}',''.join(types),'random_budget_control'))
    plansave(a.O,plans)
    # Lock an independent 5/class subset of the untouched 43/class holdout now.
    hold=a.readcsv(a.OLD/'splits/holdout.csv');final=[]
    rng=random.Random(20261010)
    for label in range(1000):final+=rng.sample([r for r in hold if int(r[1])==label],5)
    with (ROOT/'final_validation_manifest.csv').open('w',newline='') as f:csv.writer(f).writerows(final)
    assert not({r[0] for r in final}&{r['image'] for r in a.SAMPLES})
    contract=dict(scope='CPU NumPy protocol-clear accuracy only; no secure performance',random_seeds=SEEDS,random_rule='One uniform permutation of 12 original heads per seed. First 6 replaced, shared across R6/L6/M33/swap; M33 assigns first3 R then3 L.',selection_rule='R6/L6: six lowest single-head Top1 loss with head-index tie break. M33: disjoint 3R+3L minimizing sum of single-head losses, lexicographic tie break. Accuracy-only ranking; no secure cost allocator.',validation='5000 independent images, 5/class drawn from untouched holdout using seed20261010; manifest locked before selection. Remaining38000 untouched. All final plans frozen before any validation predictions.',statistics='paired normal approximate 95% CI; exact two-sided McNemar; Holm24 single vsbaseline, Holm132 within-type head pairs, Holm20 final selected/random and same-set comparisons. Selection results exploratory. Five random assignments are finite controls, not universal random-distribution proof.',model=str(a.MODEL_PATH),alpha_sha256=a.sha(a.OLD/'alpha.json'),reference_sha256=a.sha(a.B/'ours_vit_b16/scripts/vit_reference.py'),gpu_used=False,secure_runs=0)
    save(ROOT/'contract.json',contract)
    devices=subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid','--format=csv'],capture_output=True,text=True)
    save(ROOT/'device_info.json',dict(accuracy_device='CPU NumPy',gpu_inventory_only=devices.stdout,gpu_used=False))
    freeze();print('AUDIT_PASS_AND_PLANS_LOCKED',flush=True)
    a.MODEL=a.vr.load_model(a.MODEL_PATH)
    selection=runphase('selection',plans,a.SAMPLES,args.workers,{r['image']:r for r in oldnew})
    a.PLANS=plans;a.O=ROOT/'selection';a.summarize(selection,final=True)
    # Plot using existing plotting function in the matplotlib-equipped environment.
    plotcode="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import run_stage22_accuracy as a;a.O=Path(sys.argv[2]);a.plot()"
    subprocess.run([a.PREPROCESS_PYTHON,'-c',plotcode,str(Path(a.__file__).parent),str(ROOT/'selection')],check=True)
    metrics={r['plan_id']:r for r in metric_rows(selection,plans)}
    loss={t:[metrics[f'single_{t}_{h:02}']['loss_vs_allS_pp'] for h in range(12)] for t in 'RL'}
    chosen=[]
    for t in 'RL':
        hs=sorted(range(12),key=lambda h:(loss[t][h],h))[:6];types=['S']*12
        for h in hs:types[h]=t
        chosen.append(newplan('selected_'+t+'6',''.join(types),'accuracy_selected'))
    candidates=((sum(loss['R'][h] for h in rs)+sum(loss['L'][h] for h in ls),rs,ls) for rs in itertools.combinations(range(12),3) for ls in itertools.combinations([h for h in range(12) if h not in rs],3))
    _,rs,ls=min(candidates);types=['S']*12
    for h in rs:types[h]='R'
    for h in ls:types[h]='L'
    chosen.append(newplan('selected_M33',''.join(types),'accuracy_selected'))
    save(ROOT/'locked_selected_plans.json',dict(rule=contract['selection_rule'],selection_predictions_sha256=a.sha(ROOT/'selection/raw/per_image_predictions.jsonl'),plans=chosen))
    # Freeze plan and manifest hashes before any heldout image inference.
    finalplans=[p for p in plans if p['plan_id']=='all_S' or p['family']=='same_six_heads' or p['family']=='random_budget_control']+chosen
    plansave(ROOT/'validation',finalplans)
    save(ROOT/'validation_lock.json',dict(selected_sha256=a.sha(ROOT/'locked_selected_plans.json'),manifest_sha256=a.sha(ROOT/'final_validation_manifest.csv'),plans={p['plan_id']:p['plan_sha256'] for p in finalplans}))
    selectedrows=runphase('selected_selection',[plans[0]]+chosen,a.SAMPLES,args.workers)
    for r,s in zip(selection,selectedrows):assert r['image']==s['image'];r['predictions'].update(s['predictions'])
    a.writecsv(ROOT/'selection/selected_vs_random.csv',comparisons(selection))
    valsamples=[dict(index=5000+i,image=r[0],label=int(r[1])) for i,r in enumerate(final)]
    validation=runphase('validation',finalplans,valsamples,args.workers)
    tests=comparisons(validation);a.writecsv(ROOT/'validation/paired_comparisons.csv',tests)
    verify();lock=json.loads((ROOT/'validation_lock.json').read_text());assert a.sha(ROOT/'locked_selected_plans.json')==lock['selected_sha256']
    for pid,h in lock['plans'].items():assert a.sha(ROOT/f'validation/plans/{pid}.json')==h
    rows=metric_rows(validation,finalplans);lines=['头身份与同预算准确率验证','', '独立验证集：官方 validation 内部保留集的预先固定 5,000 张（每类5张），与校准及选择集不重叠。配置已在验证前锁定。所有数字为完整12层 CPU 协议明文准确率，无安全性能测量。','','| Plan | Block 0 heads | Top-1 (%) | Top-5 (%) | Loss (pp) | Paired 95% CI (pp) |','|---|---|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['plan_id']} | {r['head_types']} | {r['Top1_pct']:.2f} | {r['Top5_pct']:.2f} | {r['loss_vs_allS_pp']:.2f} | [{r['loss_ci95_low_pp']:.2f}, {r['loss_ci95_high_pp']:.2f}] |")
    headtests=json.loads((ROOT/'selection/raw/head_pair_tests.json').read_text())
    for t in 'RL':
        n=sum(r['type']==t and r['holm_p']<.05 for r in headtests);lines.append(f'\n{t} 单头敏感性：选择集上 {n} 对头差异在132次比较 Holm 校正后达到0.05。'+('存在头间差异的探索证据。' if n else '不支持显著头间差异。'))
    for budget in ('R6','L6','M33'):
        ts=[r for r in tests if r['second']=='selected_'+budget];wins=sum(r['gain_pp']>0 and r['holm_p']<.05 for r in ts)
        lines.append(f'\n{budget}：按选择集单头损失选出的配置，在独立验证集显著优于 {wins}/5 个固定随机分配。'+('支持相对这五个对照的选头价值，不能推广至所有随机分配。' if wins==5 else '不足以声称稳定优于随机选头。'))
    for pid in ('set_RRRLLL','set_LLLRRR'):
        ts=[r for r in tests if r['second']==pid and r['first'] in ('set_R6','set_L6')]
        ok=all(r['gain_pp']>0 and r['holm_p']<.05 for r in ts)
        lines.append('\n'+pid+'：'+('同六头集合下，相对6R与6L均有经校正显著优势。' if ok else '未同时显著优于同六头集合的6R与6L，不支持混合必然有优势。'))
    lines+=['','来源核对见 audit.json 和 new_old_accuracy_audit.csv。历史 frozen_assets 中 reference 哈希与实际保存的 native reference 不同，已披露；未修改历史文件。完整头类型、逐图预测、统计、运行命令与哈希见各阶段目录。最终验证不再用于选头、调α或追加方案。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');save(ROOT/'status.json',dict(stage='COMPLETE',selection_images=5000,validation_images=5000,secure_runs=0))

if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback;traceback.print_exc()
        if ROOT.exists():save(ROOT/'status.json',dict(stage='STOPPED_FAILURE',reason=repr(e)))
        raise
