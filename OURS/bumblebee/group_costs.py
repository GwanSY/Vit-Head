"""Associate real HLO intervals with exactly matching compiled tensor shapes."""
import csv
import hashlib
import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path
from experiment import save_rows
from runtime import ROOT, digest, write


def compiler_ops(path):
    records=[]
    for line_number,line in enumerate(path.read_text().splitlines(),1):
        match=re.match(r'^    (?:%[\w#: ,]+ = )?"?(pphlo\.\w+)\b',line)
        if match:
            shape=re.search(r'-> tensor<((?:\d+x)*)!pphlo\.(?:sec|pub)<(\w+)>>',line)
            dims=tuple(int(x) for x in shape.group(1).rstrip('x').split('x')) if shape and shape.group(1) else ()
            records.append(dict(name=match.group(1),line=line_number,text=line,body=line,
                shape=dims,numel=math.prod(dims) if shape else None,dtype=shape.group(2) if shape else ""))
        elif records: records[-1]["body"]+='\n'+line
    return records


def hlo_events(path):
    rows=[]
    with path.open() as stream:
        for r in csv.DictReader(stream,delimiter='\t'):
            if r['kind']=='HLO':
                for k in ('start_ns','end_ns','bytes','recv_bytes'):r[k]=int(r[k])
                rows.append(r)
    return sorted(rows,key=lambda r:r['start_ns'])


def associate(folder,case,round_number,out,scope='vit',source='out12'):
    meta=json.loads((folder / 'run.json').read_text());ir=out / f"compiler/{'block-' if scope=='block' else ''}{case}.mlir"
    if meta['compilation']['sha256']!=digest(ir):raise RuntimeError('IR is not identical to actual secure execution')
    ops=compiler_ops(ir);matrix=[i for i,o in enumerate(ops) if o['name']=='pphlo.dot_general']
    public_indices={}
    for o in ops:
        variable=re.match(r'^    (%[\w#]+) = ',o['text'])
        value=re.search(r'dense<(\[\[.*\]\])> : tensor<\d+x1xi32>',o['text'])
        if o['name']=='pphlo.constant' and variable and value:
            public_indices[variable.group(1)]=[x[0] for x in json.loads(value.group(1))]
    layer_count=1 if scope=='block' else 12
    if len(matrix)!=2*layer_count:raise RuntimeError(f'QK/AV pairs differ: {case}, {len(matrix)}')
    rows=[]
    for sample,rank in [(i,r) for i in range(1,6 if scope=='block' else 2) for r in (0,1)]:
        events=hlo_events(folder / f"log/native-p{rank}-e{sample+1 if scope=='block' else 1}.tsv")
        if [r['name'] for r in events]!=[o['name'] for o in ops]:
            raise RuntimeError(f'Compiled and real operation sequences do not match: {folder}, P{rank}')
        for layer in range(layer_count):
            types=''.join(meta['config']['types']) if scope=='block' else 'S'*12 if not meta['config'].get('head_plan') else ''.join(meta['config']['head_plan']['heads'][layer])
            lo,hi=matrix[2*layer],matrix[2*layer+1]
            groups=[t for t in 'SR' if t in types]
            ids={name:[i for i in range(lo,hi+1) if ops[i]['name']==name] for name in ('pphlo.reciprocal','pphlo.reduce','pphlo.gather','pphlo.exponential','pphlo.greater')}
            if len(ids['pphlo.reciprocal'])!=len(groups):raise RuntimeError('Normalization group count differs')
            descriptors={}
            for i,kind in zip(ids['pphlo.reciprocal'],groups):
                descriptors[i]=('reciprocal',kind,types.count(kind),types.count(kind)*197,types.count(kind)*197)
                if ops[i]['numel']!=types.count(kind)*197:raise RuntimeError('Reciprocal shape differs')
                multiplies=[j for j in range(i+1,hi) if ops[j]['name']=='pphlo.multiply' and ops[j]['numel']==types.count(kind)*197**2]
                if not multiplies:raise RuntimeError('Normalization multiplication missing')
                descriptors[multiplies[0]]=('normalization_multiply',kind,types.count(kind),types.count(kind)*197**2,types.count(kind)*197**2)
            reductions=([('softmax_max','S'),('softmax_sum','S')] if 'S' in types else [])+([('relu_sum','R')] if 'R' in types else [])
            if len(ids['pphlo.reduce'])!=len(reductions):raise RuntimeError('Reduce count differs')
            for i,(component,kind) in zip(ids['pphlo.reduce'],reductions):
                required='pphlo.greater' if component=='softmax_max' else 'pphlo.add'
                if required not in ops[i]['body']:raise RuntimeError('Reduction body differs')
                descriptors[i]=(component,kind,types.count(kind),types.count(kind)*197**2,types.count(kind)*197)
            for i in ids['pphlo.exponential']:
                descriptors[i]=('exponential','S',types.count('S'),types.count('S')*197**2,types.count('S')*197**2)
            exp_index=ids['pphlo.exponential'][0] if ids['pphlo.exponential'] else -1
            for i in ids['pphlo.greater']:
                if 'R' in types and ops[i]['numel']==types.count('R')*197**2:
                    component,kind='relu_score_compare','R'
                elif exp_index>=0 and i<exp_index:component,kind='softmax_max_sanitize','S'
                else:component,kind='relu_zero_guard','R'
                n=types.count(kind)*(197**2 if component=='relu_score_compare' else 197)
                if ops[i]['numel']!=n:raise RuntimeError('Comparison shape differs')
                descriptors[i]=(component,kind,types.count(kind),n,n)
                if component=='relu_score_compare':
                    limit=next(j for j in ids['pphlo.reduce'] if j>i)
                    masks=[j for j in range(i+1,limit) if ops[j]['name']=='pphlo.multiply' and ops[j]['numel']==n]
                    casts=[j for j in range(i+1,limit) if ops[j]['name']=='pphlo.convert' and ops[j]['numel']==n]
                    if len(masks)!=1:raise RuntimeError('ReLU score masking differs')
                    descriptors[masks[0]]=('relu_mask_multiply','R',types.count('R'),n,n)
                    for j in casts:descriptors[j]=('relu_dtype_cast','R',types.count('R'),n,n)
            gathers=ids['pphlo.gather']
            all_groups=[t for t in 'SRL' if t in types]
            if set(types)!={'S'} and meta['config'].get('organization')!='slices':
                # The existing compiler lowers a singleton head selection to a
                # slice. Keep it distinct from generic gather volume/calls.
                gathered_groups=[t for t in all_groups if types.count(t)>1]
                if len(gathers)!=len(gathered_groups)+1:raise RuntimeError(f'Head selects/restore differ: {case}, {len(gathers)}, {gathered_groups}')
                for i,kind in zip(gathers[:-1],gathered_groups):
                    descriptors[i]=('head_select',kind,types.count(kind),12*197**2,types.count(kind)*197**2)
                descriptors[gathers[-1]]=('head_restore','all',12,12*197**2,12*197**2)
                if 'L' in types:
                    linear=[j for j in range(lo+1,gathers[-1]) if ops[j]['name']=='pphlo.multiply' and ops[j]['numel']==types.count('L')*197**2 and '!pphlo.pub<f32>' in ops[j]['text']]
                    if len(linear)!=1:raise RuntimeError('Linear mapping differs')
                    n=types.count('L')*197**2
                    descriptors[linear[0]]=('linear_scale','L',types.count('L'),n,n)
            for i,(component,kind,heads,inputs,outputs) in sorted(descriptors.items()):
                r=events[i]
                select_positions=[j for j,t in enumerate(types) if t==kind]
                restore_order=[j for t in 'SRL' for j,k in enumerate(types) if k==t]
                indices=[]
                if component in ('head_select','head_restore'):
                    operand=re.search(r'"pphlo.gather"\(%[\w#]+, (%[\w#]+)\)',ops[i]['text'])
                    if not operand or operand.group(1) not in public_indices:raise RuntimeError('Public gather index literal not found')
                    indices=public_indices[operand.group(1)]
                    expected=select_positions if component=='head_select' else sorted(range(len(types)),key=lambda j:restore_order[j])
                    if indices!=expected:raise RuntimeError('Actual compiled gather indices differ from the head plan')
                rows.append(dict(source=source,scope=scope,case=case,round=round_number,sample=sample,party=rank,layer=layer,component=component,group_kind=kind,
                    group_heads=heads,input_elements=inputs,output_elements=outputs,seconds=(r['end_ns']-r['start_ns'])/1e9,
                    sent_mib=r['bytes']/2**20,recv_mib=r['recv_bytes']/2**20,compiler_line=ops[i]['line'],
                    selection_is_contiguous=(select_positions==list(range(min(select_positions),max(select_positions)+1))) if component=='head_select' else '',
                    restoration_is_identity=(restore_order==list(range(len(types)))) if component=='head_restore' else '',
                    compiled_public_indices=json.dumps(indices) if indices else '',
                    compiled_sha256=meta['compilation']['sha256'],proof='Exact code SHA256 and entire top-level opcode sequence match'))
    return rows


def run(out):
    frozen=json.loads((out / 'design.json').read_text())
    schedule=json.loads((out / 'recovery.json').read_text())['primary_full_schedule'] if (out / 'recovery.json').exists() else frozen['full_schedule']
    rows=[];cache=out / '.shape-cache';cache.mkdir(exist_ok=True)
    suite=[dict(c,scope='vit',source='out12',folder_path=out / 'full/log' / c['folder']) for c in schedule]
    for source in ('out11','out12'):
        base=ROOT / ('out/11' if source=='out11' else 'out/12')
        for p in (base / 'blocks/log').glob('*/run.json'):
            m=json.loads(p.read_text())
            if m.get('status')!='complete' or m['config']['layout']=='split':continue
            suite.append(dict(case=p.parent.name,round=0,scope='block',source=source,folder_path=p.parent,folder=p.parent.name))
    for c in suite:
        folder=c['folder_path'];p=folder / 'run.json'
        if not p.exists() or json.loads(p.read_text())['status'] not in ('complete','matched','outside_gate'):continue
        count=5 if c['scope']=='block' else 1
        paths=[p,out / f"compiler/{'block-' if c['scope']=='block' else ''}{c['case']}.mlir"]+[folder / f"log/native-p{rank}-e{i+1 if c['scope']=='block' else 1}.tsv" for i in range(1,count+1) for rank in (0,1)]
        identity={str(x.relative_to(ROOT)):digest(x) for x in paths};identity['analysis_source']=digest(Path(__file__))
        target=cache / (c['source']+'-'+c['scope']+'-'+c['folder']+'.json')
        saved=json.loads(target.read_text()) if target.exists() else {}
        if saved.get('identity')!=identity:
            saved=dict(identity=identity,rows=associate(folder,c['case'],c['round'],out,c['scope'],c['source']));write(target,saved)
        rows.extend(saved['rows']);print('Timed tensor shapes audited:',c['folder'],flush=True)
    if not rows:return
    save_rows(out / 'group_hlo_samples.csv',rows)
    by=defaultdict(list)
    for r in rows:by[(r['source'],r['scope'],r['case'],r['party'],r['component'],r['group_kind'],r['group_heads'])].append(r)
    summaries=[]
    for (source,scope,case,party,component,kind,heads),values in by.items():
        per_round=defaultdict(list)
        for r in values:per_round[(r['round'],r['sample'])].append(r)
        totals=[sum(r['seconds'] for r in group) for group in per_round.values()]
        summaries.append(dict(source=source,scope=scope,case=case,party=party,component=component,group_kind=kind,group_heads=heads,
            real_calls=len(values),real_samples=len(per_round),independent_runtimes=len(per_round) if scope=='vit' else 1,median_call_s=st.median(r['seconds'] for r in values),
            mean_call_s=st.mean(r['seconds'] for r in values),mean_model_total_s=st.mean(totals),
            mean_model_sent_mib=st.mean(sum(r['sent_mib'] for r in group) for group in per_round.values()),
            mean_model_recv_mib=st.mean(sum(r['recv_mib'] for r in group) for group in per_round.values()),
            mean_call_sent_mib=st.mean(r['sent_mib'] for r in values),mean_call_recv_mib=st.mean(r['recv_mib'] for r in values)))
    save_rows(out / 'group_hlo_summary.csv',summaries)
    write(out / 'timing_association.json',dict(exact_matches=True,real_timed_operations=len(rows),
        note='Every time/byte row is from a real two-party native trace. Offline compilation supplies only shapes and identities.'))


if __name__=='__main__':run(ROOT / 'out/12')
