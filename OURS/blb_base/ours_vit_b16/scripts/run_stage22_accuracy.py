#!/usr/bin/env python3
"""Block-0 accuracy feasibility using the existing complete protocol-clear ViT.
Only orchestration/statistics: no new attention, training, HE or MPC execution.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse, collections, csv, datetime, hashlib, json, math
import multiprocessing as mp
from pathlib import Path
import subprocess, sys, tempfile, time
import numpy as np
import vit_reference as vr

B=Path(__file__).resolve().parents[2]; V=B.parents[1]
O=B/'outputs/stage22_head_accuracy_20261010'
OLD=B/'outputs/heterogeneous_block_optimized_20261010'
MODEL_PATH=V/'BOLT/checkpoints/vit_b16/bolt_model.txt'
PREPROCESS_PYTHON='/home/dm/anaconda3/bin/python'
MANIFEST=V/'BOLT/checkpoints/vit_b16/manifest.json'
MODEL=None; PLANS=None; SAMPLES=None
GPU_IDS=[]; GPU_CHECK=False; GPU_CALLS=0; UNIFORM_ONLY=False
PROTO={'S':'softmax_mpc','R':'relu_norm_mpc','L':'he_linear'}

class GPUMatmulNumpy:
    """Dispatch only reference np.matmul to FP64 CUDA; preserve the graph.

    Integer products retain the reference's 2**53 proof and rounding.
    Native Linear floating @ products and ring/nonlinear emulation stay on CPU.
    Smoke runs compare every dispatched product bit-for-bit with NumPy.
    """
    def __init__(self, device):
        import torch
        self.torch=torch
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False
        self.device=device
        self.timings=dict(h2d_s=0.,cuda_matmul_s=0.,d2h_s=0.)
        self.transfer_bytes=0

    def __getattr__(self,name):return getattr(np,name)

    def matmul(self,lhs,rhs):
        global GPU_CALLS
        t=self.torch
        a=np.ascontiguousarray(lhs,dtype=np.float64)
        b=np.ascontiguousarray(rhs,dtype=np.float64)
        with t.no_grad():
            start=time.perf_counter()
            ga=t.from_numpy(a).to(self.device);gb=t.from_numpy(b).to(self.device)
            t.cuda.synchronize(self.device)
            self.timings['h2d_s']+=time.perf_counter()-start
            start=time.perf_counter()
            product=t.matmul(ga,gb)
            t.cuda.synchronize(self.device)
            self.timings['cuda_matmul_s']+=time.perf_counter()-start
            start=time.perf_counter()
            result=product.cpu().numpy()
            self.timings['d2h_s']+=time.perf_counter()-start
            self.transfer_bytes+=a.nbytes+b.nbytes+result.nbytes
        if GPU_CHECK:
            expected=np.matmul(lhs,rhs)
            if not np.array_equal(result,expected):
                raise AssertionError(f'CUDA/CPU product mismatch: {a.shape} @ {b.shape}, max_abs={np.max(np.abs(result-expected))}')
        GPU_CALLS+=1
        return result

def init_gpu_worker(gpus,offset):
    # The parent never initializes CUDA; each fork owns its CUDA context.
    slot=(mp.current_process()._identity[0]-1-offset)%len(gpus)
    vr.np=GPUMatmulNumpy(f'cuda:{slot}')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def status(stage,**kw):
    row=dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
    dump(O/'status.json',row);print(json.dumps(row),flush=True)
def writecsv(p,rows):
    if not rows:return
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def readcsv(p):return list(csv.reader(Path(p).open()))

def prepare():
    global PLANS,SAMPLES
    for folder in (O,O/'plans',O/'splits',O/'raw'):folder.mkdir(parents=True,exist_ok=True)
    splitsets={}
    for name,n,per_class in [('calibration',2000,2),('selection',5000,5),('holdout',43000,43)]:
        source=OLD/'splits'/f'{name}.csv';rows=readcsv(source)
        assert len(rows)==n and len({r[0] for r in rows})==n
        counts=collections.Counter(int(r[1]) for r in rows)
        assert set(counts)==set(range(1000)) and set(counts.values())=={per_class}
        splitsets[name]={r[0] for r in rows}
        (O/'splits'/f'{name}.csv').write_bytes(source.read_bytes())
    assert not(splitsets['calibration']&splitsets['selection'] or splitsets['selection']&splitsets['holdout'] or splitsets['calibration']&splitsets['holdout'])
    SAMPLES=[dict(index=i,image=r[0],label=int(r[1])) for i,r in enumerate(readcsv(O/'splits/selection.csv'))]
    alpha=json.loads((OLD/'alpha.json').read_text());assert alpha['manifest_sha256']==sha(O/'splits/calibration.csv')
    (O/'alpha.json').write_bytes((OLD/'alpha.json').read_bytes())
    PLANS=[]
    def add(pid,family,head0):
        if UNIFORM_ONLY and family != "uniform":return
        layers=[list(head0)]+[list('S'*12) for _ in range(11)]
        hp=[[(PROTO[t],float(alpha['alpha'][l][h]) if t=='L' else 1.) for h,t in enumerate(types)] for l,types in enumerate(layers)]
        path=O/'plans'/f'{pid}.json';item=dict(plan_id=pid,family=family,block0=head0,layers=layers,head_plan=hp)
        dump(path,item);item['plan_sha256']=sha(path);PLANS.append(item)
    for t in 'SRL':add('all_'+t,'uniform',t*12)
    for t in 'RL':
        for h in range(12):
            x=list('S'*12);x[h]=t;add(f'single_{t}_{h:02}','single',''.join(x))
    add('set_R6','same_six_heads','R'*6+'S'*6)
    add('set_L6','same_six_heads','L'*6+'S'*6)
    add('set_RRRLLL','same_six_heads','RRRLLL'+'S'*6)
    add('set_LLLRRR','same_six_heads','LLLRRR'+'S'*6)
    assert len(PLANS)==(3 if UNIFORM_ONLY else 31)
    planrows=[]
    for p in PLANS:
        for l,types in enumerate(p['layers']):
            # Record both true checkpoint identity and the public S/R/L order.
            order=sorted(range(12),key=lambda h:('SRL'.index(types[h]),h))
            for h,t in enumerate(types):
                planrows.append(dict(plan_id=p['plan_id'],layer=l,original_head=h,physical_head=order.index(h),type=t,alpha=p['head_plan'][l][h][1],plan_sha256=p['plan_sha256']))
    writecsv(O/'head_plans.csv',planrows)
    dump(O/'contract.json',dict(model='ViT-B/16, frozen checkpoint',reference=str(B/'ours_vit_b16/scripts/vit_reference.py'),inference='Graph(fixed=True, physical_tokens=256), all 12 layers, final LN and classifier',block=0,plans=len(PLANS),selection_samples=5000,calibration_samples=2000,calibration_usage='previous frozen alpha only; no refit',holdout_samples=43000,holdout_evaluated=False,dataset='Internal stratified split of official ImageNet validation; not training data or a new official test set',six_original_heads=[0,1,2,3,4,5],mixed_A={'R':[0,1,2],'L':[3,4,5]},mixed_B={'R':[3,4,5],'L':[0,1,2]},selection_rule='Fixed original index order before any new results',additional_sets=[],loss_reference_bands_pp=[.5,1,2],statistics={'single_head_baseline_comparisons':'two-sided exact McNemar, Holm family of 24','mixed_comparisons':'two-sided exact McNemar, Holm family of 5 including assignment swap','head_heterogeneity':'all 66 within-R plus 66 within-L paired comparisons, Holm family of 132; no additional inference plans','interval':'approximate paired normal 95% CI for correctness differences; exploratory selection-set analysis'},secure_runs=0,new_allocator=False,new_packing=False,preprocess_python=PREPROCESS_PYTHON,smoke_indices=[0,5,10]))
    frozen=[Path(__file__),MODEL_PATH,MANIFEST,B/'ours_vit_b16/scripts/vit_reference.py',B/'ours_vit_b16/scripts/preprocess_vit_image.py',O/'alpha.json',O/'contract.json',O/'head_plans.csv',*sorted((O/'plans').glob('*.json')),*sorted((O/'splits').glob('*.csv'))]
    dump(O/'frozen_hashes.json',{str(p):sha(p) for p in frozen})

def check_frozen():
    for path,h in json.loads((O/'frozen_hashes.json').read_text()).items():assert sha(path)==h,path

def evaluate(sample):
    global GPU_CHECK,GPU_CALLS
    GPU_CHECK=bool(GPU_IDS) and sample['index'] in (0,5,10)
    GPU_CALLS=0
    if GPU_IDS:
        vr.np.timings=dict(h2d_s=0.,cuda_matmul_s=0.,d2h_s=0.)
        vr.np.transfer_bytes=0
    start=time.monotonic();result=dict(sample,predictions={})
    try:
        with tempfile.TemporaryDirectory(prefix='stage22_input_') as tmp:
            inp=Path(tmp)/'input.txt'
            subprocess.run([PREPROCESS_PYTHON,str(B/'ours_vit_b16/scripts/preprocess_vit_image.py'),str(V/'datasets/imagenet2012/val'/sample['image']),'--manifest',str(MANIFEST),'--output',str(inp)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            result['input_sha256']=sha(inp)
            if sample['index']==0:assert sha(inp)==sha(OLD/'secure_input.txt'),'frozen preprocessing mismatch'
            x=vr.load_input(inp)
        for p in PLANS:
            graph=vr.Graph(MODEL,fixed=True,physical_tokens=256,head_plan=p['head_plan'],record_values=False)
            assert graph.executed_layers==12 and not graph.stop_after_attention
            y,_=graph.run(x)
            assert y.size==1000 and np.isfinite(y).all(),p['plan_id']
            if sample['index'] in (0,5,10) and p['plan_id']=='all_S':
                prior=B/'outputs/stage24_full_vit_20261010/references'/f'B_12_{(0,5,10).index(sample["index"])}.npy'
                if prior.exists():assert np.array_equal(y,np.load(prior)), 'full-model baseline reference changed'
            top=np.argsort(-y,kind='stable')[:5]
            result['predictions'][p['plan_id']]=dict(top5=top.tolist(),top5_logits=y[top].tolist(),logits_sha256=hashlib.sha256(y.tobytes()).hexdigest())
        baseline=result['predictions']['all_S']['top5'][0]==sample['label']
        result['baseline_correct']=baseline
        for prediction in result['predictions'].values():
            prediction['top1_correct']=prediction['top5'][0]==sample['label']
            prediction['top5_correct']=sample['label'] in prediction['top5']
            prediction['paired_correctness_change']=int(prediction['top1_correct'])-int(baseline)
        result['status']='PASS'
    except Exception as e:
        result.update(status='FAILED',error=repr(e))
    result['wall_s']=time.monotonic()-start
    if GPU_IDS:
        if result['status']=='PASS':assert GPU_CALLS>0,'No CUDA computation executed'
        result['cuda_matmul_calls']=GPU_CALLS
        result['cuda_cpu_products_exact_checked']=GPU_CHECK
        result['physical_gpu']=GPU_IDS[int(vr.np.device.split(':')[1])]
        result['gpu_dispatch_timings']=dict(vr.np.timings)
        result['host_device_transfer_bytes']=vr.np.transfer_bytes
    return result

def paired(first,second):
    # Positive gain means second is more accurate than first.
    lost=int(np.sum(first&~second));gained=int(np.sum(~first&second));n=len(first)
    gain=100*(gained-lost)/n;discordant=lost+gained
    from scipy.stats import binomtest
    p=float(binomtest(min(lost,gained),discordant,.5).pvalue) if discordant else 1.
    radius=196*math.sqrt(max(0.,(discordant-(gained-lost)**2/n)/(n*(n-1)))) if n>1 else 0.
    return dict(n=n,first_correct_second_wrong=lost,first_wrong_second_correct=gained,gain_pp=gain,gain_ci95_low_pp=gain-radius,gain_ci95_high_pp=gain+radius,mcnemar_exact_p=p)

def holm(rows):
    running=0.;m=len(rows)
    for rank,i in enumerate(sorted(range(m),key=lambda i:rows[i]['mcnemar_exact_p'])):
        running=max(running,min(1.,rows[i]['mcnemar_exact_p']*(m-rank)))
        rows[i]['holm_p']=running

def summarize(records,final=False):
    n=len(records);base=np.array([r['predictions']['all_S']['top5'][0]==r['label'] for r in records]);out=[];correct={}
    for p in PLANS:
        pid=p['plan_id'];c=np.array([r['predictions'][pid]['top5'][0]==r['label'] for r in records]);correct[pid]=c
        counts=collections.Counter(p['block0']);change=paired(base,c) if n>1 else None
        row=dict(plan_id=pid,family=p['family'],block=0,original_head_types=p['block0'],S=counts['S'],R=counts['R'],L=counts['L'],samples=n,status='COMPLETE' if final else 'INCOMPLETE',Top1_pct=100*float(c.mean()) if final else 'NOT_RUN',Top5_pct=100*sum(r['label'] in r['predictions'][pid]['top5'] for r in records)/n if final else 'NOT_RUN',loss_vs_allS_pp=-change['gain_pp'] if final else 'NOT_RUN',baseline_correct_method_wrong=change['first_correct_second_wrong'] if change else 'NOT_RUN',baseline_wrong_method_correct=change['first_wrong_second_correct'] if change else 'NOT_RUN',loss_ci95_low_pp=-change['gain_ci95_high_pp'] if final else 'NOT_RUN',loss_ci95_high_pp=-change['gain_ci95_low_pp'] if final else 'NOT_RUN',mcnemar_exact_p=change['mcnemar_exact_p'] if final else 'NOT_RUN',holm_p='NOT_APPLICABLE',plan_sha256=p['plan_sha256'],selection_manifest_sha256=sha(O/'splits/selection.csv'),raw_predictions='raw/per_image_predictions.jsonl')
        out.append(row)
    if final:
        singles=[r for r in out if r['family']=='single'];holm(singles)
    for family,name in [('uniform','uniform_block0.csv'),('single','head_sensitivity.csv'),('same_six_heads','same_head_set.csv')]:writecsv(O/name,[r for r in out if r['family']==family])
    if not final:return
    if UNIFORM_ONLY:
        lines=['实验一：Block 0 统一替换的完整模型协议明文准确率','',
               '| Block 0 | Top-1 (%) | Top-5 (%) | 相对全 S 损失 (pp) | 正确→错误 | 错误→正确 |',
               '|---|---:|---:|---:|---:|---:|']
        for r in out:
            lines.append(f"| {r['original_head_types']} | {r['Top1_pct']:.2f} | {r['Top5_pct']:.2f} | {r['loss_vs_allS_pp']:.2f} | {r['baseline_correct_method_wrong']} | {r['baseline_wrong_method_correct']} |")
        for r in out[1:]:
            lines.append(f"\n{r['plan_id']} 的 Top-1 损失为 {r['loss_vs_allS_pp']:.2f} pp。0.5、1、2 pp 仅作预声明分析参照；本结果只适用于 Block 0 的当前冻结定义。")
        lines+=['','使用 ImageNet 官方 validation 内部划分的 5,000 张选择集；2,000 张校准集仅用于此前冻结的 α，43,000 张保留集未评估。其余 11 层为 Softmax，包含完整 Embedding、12 层、Final LN 和 Classifier。CPU 仅用于协议明文准确率；没有安全时延或通信结果。', '', 'checkpoint、参考、α、头计划和样本清单哈希见 frozen_hashes.json；实际命令见 command.json；逐图预测与相对全 S 的配对变化见 raw/per_image_predictions.jsonl；完整头身份见 head_plans.csv。实验二、三未执行。']
        (O/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        return
    comparisons=[]
    for a,z in [('set_R6','set_RRRLLL'),('set_L6','set_RRRLLL'),('set_R6','set_LLLRRR'),('set_L6','set_LLLRRR'),('set_RRRLLL','set_LLLRRR')]:comparisons.append(dict(first=a,second=z,**paired(correct[a],correct[z])))
    holm(comparisons);writecsv(O/'mixed_paired_comparisons.csv',comparisons)
    headpairs=[]
    for t in 'RL':
        for a in range(12):
            for z in range(a+1,12):headpairs.append(dict(type=t,first_head=a,second_head=z,**paired(correct[f'single_{t}_{a:02}'],correct[f'single_{t}_{z:02}'])))
    holm(headpairs)
    # Store all prespecified comparison evidence in one compact machine artifact.
    dump(O/'raw/head_pair_tests.json',headpairs)
    lookup={r['plan_id']:r for r in out}
    lines=['Block-0 head-level accuracy feasibility on ViT-B/16','', 'All accuracies are from complete 12-layer protocol-clear inference on the same 5,000-image selection subset. Only Block 0 changes.','', 'Table 1. Uniform replacement within Block 0.','', '| Configuration | Top-1 (%) | Top-5 (%) | Loss vs all S (pp) | Correct→wrong | Wrong→correct |','|---|---:|---:|---:|---:|---:|']
    def tablerow(r):return f"| {r['plan_id']} | {r['Top1_pct']:.2f} | {r['Top5_pct']:.2f} | {r['loss_vs_allS_pp']:.2f} | {r['baseline_correct_method_wrong']} | {r['baseline_wrong_method_correct']} |"
    lines += [tablerow(r) for r in out if r['family']=='uniform']
    lines+=['','Table 2. Single-head replacement sensitivity (original checkpoint head indices).','','| Head | ReLU Top-1 (%) | ReLU loss (pp) | Linear Top-1 (%) | Linear loss (pp) |','|---:|---:|---:|---:|---:|']
    for h in range(12):
        r=lookup[f'single_R_{h:02}'];l=lookup[f'single_L_{h:02}'];lines.append(f"| {h} | {r['Top1_pct']:.2f} | {r['loss_vs_allS_pp']:.2f} | {l['Top1_pct']:.2f} | {l['loss_vs_allS_pp']:.2f} |")
    lines+=['','Table 3. Same six original heads {0,1,2,3,4,5}; other heads stay Softmax.','','| Configuration | Top-1 (%) | Top-5 (%) | Loss vs all S (pp) | Correct→wrong | Wrong→correct |','|---|---:|---:|---:|---:|---:|']
    lines += [tablerow(r) for r in out if r['family']=='same_six_heads']
    lines+=['','Table 4. Paired mixed-versus-pure and assignment-swap comparisons. Positive gain favors the second configuration.','','| First | Second | Gain (pp) | Paired 95% CI (pp) | Holm p |','|---|---|---:|---|---:|']
    lines += [f"| {r['first']} | {r['second']} | {r['gain_pp']:.2f} | [{r['gain_ci95_low_pp']:.2f}, {r['gain_ci95_high_pp']:.2f}] | {r['holm_p']:.4g} |" for r in comparisons]
    lines+=['','实测分析：']
    for t in 'RL':
        r=lookup['all_'+t];loss=r['loss_vs_allS_pp'];lines.append(f"Block 0 全部替换为 {t} 的 Top-1 损失为 {loss:.2f} pp；"+('落在预声明的 2 pp 参照范围内。' if loss<=2 else '超出预声明的 2 pp 参照范围。')+'这只评价当前固定函数及 Block 0，不能推广为所有层的结论。')
        vals=[lookup[f'single_{t}_{h:02}']['loss_vs_allS_pp'] for h in range(12)];sig=[r for r in headpairs if r['type']==t and r['holm_p']<.05]
        lines.append(f"单头 {t} 替换的损失范围为 {min(vals):.2f}–{max(vals):.2f} pp；在预声明的 132 个头间配对比较统一 Holm 校正后，{len(sig)} 对 {t} 头差异达到 0.05 水平。"+('有证据支持头身份的准确率敏感性差异。' if sig else '目前不足以证明头身份间存在统计显著差异。'))
    for pid in ('set_RRRLLL','set_LLLRRR'):
        rs=[r for r in comparisons if r['second']==pid and r['first'] in ('set_R6','set_L6')]
        supported=len(rs)==2 and all(r['gain_pp']>0 and r['holm_p']<.05 for r in rs)
        lines.append(f"{pid}："+('在固定六头集合上，相对 6R 和 6L 均有正向且经校正显著的配对准确率优势。' if supported else '未同时获得相对 6R 和 6L 的经校正显著优势，不能声称混合必然更好。'))
    lines+=['','以上统计用于选择集探索，并非独立最终测试结论。0.5/1/2 pp 为损失分析参照；CI 为配对正确性差异的正态近似，p 值为双侧精确 McNemar 并按预声明比较族做 Holm 校正。未依据本轮结果调整 α、头身份或追加集合。', '', '数据为 ImageNet 官方 validation 内部分层划分：2,000 张只用于先前 α 校准，本轮不拟合；5,000 张用于本轮准确率/选头分析；43,000 张最终保留集本轮未评估，不是新的官方测试集。α 含符号，冻结系数原样复用。模型/参考/预处理/计划/清单哈希见 frozen_hashes.json；逐图 Top-5、对应 logits 和完整 logits 哈希见 raw/per_image_predictions.jsonl；所有 144 个头身份见 head_plans.csv。没有真实密文性能测试。', '', '![R/L head sensitivity](head_sensitivity_heatmap.png)']
    (O/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    rs=list(csv.DictReader((O/'head_sensitivity.csv').open()));lookup={r['plan_id']:r for r in rs}
    values=np.array([[float(lookup[f'single_{t}_{h:02}']['loss_vs_allS_pp']) for h in range(12)] for t in 'RL'])
    bound=max(float(abs(values).max()),.01)
    fig,ax=plt.subplots(figsize=(12,3.1));im=ax.imshow(values,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto')
    ax.set_xticks(range(12));ax.set_yticks([0,1],['Normalized ReLU','Signed Linear']);ax.set_xlabel('Original head index in Block 0')
    ax.set_title('Full-model Top-1 loss relative to all-Softmax (pp), selection n=5,000')
    for i in range(2):
        for h in range(12):ax.text(h,i,f'{values[i,h]:.2f}',ha='center',va='center',color='white' if abs(values[i,h])>.6*bound else 'black',fontsize=9)
    fig.colorbar(im,ax=ax,label='Top-1 loss (pp); negative = gain');fig.tight_layout()
    fig.savefig(O/'head_sensitivity_heatmap.pdf');fig.savefig(O/'head_sensitivity_heatmap.png',dpi=180);plt.close(fig)

def main():
    global MODEL,PLANS,SAMPLES,GPU_IDS,UNIFORM_ONLY,O
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=64);p.add_argument('--gpus',default='');p.add_argument('--resume',action='store_true');p.add_argument('--plot-only',action='store_true');p.add_argument('--uniform-only',action='store_true');a=p.parse_args()
    UNIFORM_ONLY=a.uniform_only
    if UNIFORM_ONLY:O=B/'outputs/stage22_experiment1_accuracy_20261010'
    if a.plot_only:plot();return
    GPU_IDS=[int(x) for x in a.gpus.split(',') if x.strip()]
    if GPU_IDS:
        if len(GPU_IDS)!=len(set(GPU_IDS)) or any(x not in (1,7,0,3,4) for x in GPU_IDS):raise ValueError('Invalid or duplicate GPU IDs')
        os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,GPU_IDS))
    if not 1<=a.workers<=96:raise ValueError('workers must be 1..96')
    if a.resume:
        check_frozen();PLANS=[]
        for path in sorted((O/'plans').glob('*.json')):
            item=json.loads(path.read_text());item['plan_sha256']=sha(path);PLANS.append(item)
        SAMPLES=[dict(index=i,image=r[0],label=int(r[1])) for i,r in enumerate(readcsv(O/'splits/selection.csv'))]
    else:
        if (O/'status.json').exists():raise RuntimeError('Existing run; use --resume after inspecting failure')
        prepare()
    if GPU_IDS:
        dump(O/'gpu_execution.json',dict(physical_gpu_priority=GPU_IDS,workers=a.workers,arithmetic='FP64 CUDA matmul with unchanged original CPU ring/nonlinear/native-Linear emulation',tf32=False,smoke_validation='Every dispatched matrix product must exactly equal CPU NumPy; all-S final logits must equal frozen reference',secure_runs=0))
    status('LOADING_FROZEN_MODEL');MODEL=vr.load_model(MODEL_PATH)
    records=[];predpath=O/'raw/per_image_predictions.jsonl'
    if predpath.exists():
        records=[json.loads(line) for line in predpath.read_text().splitlines()]
        assert all(r['status']=='PASS' for r in records)
    done={r['index'] for r in records};assert len(done)==len(records)
    for r in records:assert SAMPLES[r['index']]['image']==r['image'] and SAMPLES[r['index']]['label']==r['label']
    with predpath.open('a') as f:
        smoke=[SAMPLES[i] for i in (0,5,10) if i not in done]
        status('SMOKE_RUNNING',images=len(smoke),plans=len(PLANS),full_layers=12)
        pool_kw=dict(initializer=init_gpu_worker,initargs=(GPU_IDS,0)) if GPU_IDS else {}
        with mp.get_context('fork').Pool(max(1,len(smoke)),**pool_kw) as pool:
            for r in pool.imap(evaluate,smoke):
                if r['status']!='PASS':dump(O/'raw/failed_image.json',r);raise RuntimeError(r['error'])
                f.write(json.dumps(r)+'\n');f.flush();records.append(r);done.add(r['index']);status('SMOKE_RUNNING',completed_images=len(done),plans_per_image=len(PLANS))
        dump(O/'raw/smoke.json',dict(status='PASS',indices=[0,5,10],plans=len(PLANS),full_model=True,baseline_logits_equal_previous_reference=True,cuda_cpu_products_exact_checked=bool(GPU_IDS)))
        check_frozen();summarize(records)
        remaining=[r for r in SAMPLES if r['index'] not in done];started=time.monotonic();initial=len(records)
        status('FORMAL_RUNNING',evaluated=initial,total=5000,plans=len(PLANS),workers=a.workers)
        pool_kw=dict(initializer=init_gpu_worker,initargs=(GPU_IDS,max(1,len(smoke)))) if GPU_IDS else {}
        with mp.get_context('fork').Pool(a.workers,**pool_kw) as pool:
            for r in pool.imap(evaluate,remaining,chunksize=1):
                if r['status']!='PASS':dump(O/'raw/failed_image.json',r);raise RuntimeError(r['error'])
                f.write(json.dumps(r)+'\n');f.flush();records.append(r)
                if (len(records)-initial)%10==0:
                    elapsed=time.monotonic()-started;completed=len(records)-initial
                    status('FORMAL_RUNNING',evaluated=len(records),total=5000,plans=len(PLANS),workers=a.workers,elapsed_formal_s=elapsed,estimated_remaining_s=elapsed/completed*(5000-len(records)))
                if len(records)%250==0:summarize(records)
    assert len(records)==5000 and len({r['image'] for r in records})==5000
    check_frozen();summarize(sorted(records,key=lambda r:r['index']),final=True)
    if not UNIFORM_ONLY:subprocess.run([PREPROCESS_PYTHON,str(Path(__file__)),'--plot-only'],check=True)
    dump(O/'raw/completion.json',dict(status='PASS',images=5000,plans=len(PLANS),full_model_inferences=5000*len(PLANS),head_plans_unchanged=True,alpha_unchanged=True,heldout_images_evaluated=0,secure_runs=0,prediction_file_sha256=sha(predpath)))
    status('COMPLETE',images=5000,plans=len(PLANS),full_model_inferences=5000*len(PLANS))

if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        if O.exists():status('STOPPED_FAILURE',reason=repr(e))
        raise
