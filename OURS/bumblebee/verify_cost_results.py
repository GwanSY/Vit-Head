"""Recheck numerical evidence and immutable execution identities for out/12."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiment import save_rows
from runtime import ROOT, digest, write


def verify(out, partial=False):
    design=json.loads((out / 'design.json').read_text())
    provenance=json.loads((out / 'provenance.json').read_text())
    recovery=json.loads((out / 'recovery.json').read_text())
    actual_library=digest(ROOT / 'build/stage2.so')
    actual_weight=digest(ROOT / '../../BOLT/checkpoints/vit_b16_hf/model.safetensors')
    if actual_library!=provenance['native_sha256'] or actual_weight!=provenance['checkpoint_sha256']:
        raise RuntimeError('Library or pretrained weights changed')
    for name in ('run.py','runtime.py','model.py','heads.py','experiment.py','block.py','config.json'):
        if digest(ROOT / name)!=provenance['sources'][name]:
            raise RuntimeError(f'Computational source changed: {name}')
    cases=[dict(folder=out / 'blocks/log' / c['case'],scope='block',cohort='primary',expected=5)
           for c in design['blocks']]
    cases.extend(dict(folder=out / 'full/log' / c['folder'],scope='vit',cohort='primary',expected=1)
                 for c in recovery['primary_full_schedule'])
    cases.extend(dict(folder=out / p,scope='vit',cohort='supplementary',expected=1)
                 for p in recovery['supplementary_folders'])
    rows=[];references={};counts={}
    for c in cases:
        p=c['folder'] / 'run.json'
        if not p.exists():
            if partial:continue
            raise RuntimeError(f'Missing execution: {p}')
        m=json.loads(p.read_text());recovered=c['folder'] / 'recovered_evidence.json'
        if m['status'] not in ('complete','outside_gate','matched'):
            if c['cohort']=='supplementary' and recovered.exists():pass
            elif partial:continue
            else:raise RuntimeError(f'Unfinished execution: {p}')
        if len(m['measurements'])!=c['expected']:raise RuntimeError(f'Incorrect sample count: {p}')
        if m['library']['sha256']!=actual_library or m['hashes']['checkpoint']!=actual_weight:
            raise RuntimeError(f'Execution identity differs: {p}')
        if digest(out / f"compiler/{'block-' if c['scope']=='block' else ''}{m['config']['plan_name']}.mlir")!=m['compilation']['sha256']:
            raise RuntimeError(f'Compiled executable differs: {p}')
        if c['scope']=='block':
            if any('ashr' not in s.lower() for part in m['measurements'][0]['input_storage_types'] for s in part):
                raise RuntimeError('Block input is not shared')
            if any('priv' not in s.lower() for part in m['weight_storage_types'] for s in part):
                raise RuntimeError('Block weights are not private')
        vectors=np.load(c['folder'] / 'logits.npz') if c['scope']=='vit' else None
        for i,r in enumerate(m['measurements'],1):
            finite=r['finite'];relative=r['relative_l2'];maximum=r['max_abs']
            top1=True
            if vectors is not None:
                ref=vectors['reference'];actual=vectors[f'b{i}']
                difference=actual.astype(np.float64)-ref.astype(np.float64)
                finite=bool(np.isfinite(ref).all() and np.isfinite(actual).all())
                relative=float(np.linalg.norm(difference)/np.linalg.norm(ref))
                maximum=float(np.max(np.abs(difference)))
                top1=int(ref.argmax())==int(actual.argmax())
                if abs(relative-r['relative_l2'])>1e-12 or abs(maximum-r['max_abs'])>1e-12:
                    raise RuntimeError(f'Saved numerical metrics differ from output: {p}')
                reference_hash=hashlib.sha256(ref.tobytes()).hexdigest()
                name=m['config']['plan_name']
                if name in references and references[name]!=reference_hash:
                    raise RuntimeError(f'Plaintext reference changed between runs: {name}')
                references[name]=reference_hash
            gate=m['config']['gate']
            accepted=finite and relative<=gate['relative_l2_max'] and maximum<=gate['max_abs_max'] and top1
            if not accepted or not r['numerical_pass']:raise RuntimeError(f'Numerical acceptance failed: {p}')
            rows.append(dict(cohort=c['cohort'],scope=c['scope'],case=m['config']['plan_name'],sample=i,
                finite=finite,relative_l2=relative,max_abs=maximum,top1_matches=top1 if vectors is not None else '',
                accepted=accepted,basis='Recomputed from saved reference and secure logits' if vectors is not None else 'Error metrics measured by the real execution worker; Block output tensor not archived',
                metadata=str(p)))
            key=c['cohort']+'_'+c['scope'];counts[key]=counts.get(key,0)+1
    if not partial and counts!={'primary_block':40,'primary_vit':24,'supplementary_vit':recovery['supplemental_verified_calls']}:
        raise RuntimeError(f'Incomplete numerical evidence: {counts}')
    save_rows(out / 'numerical_validation.csv',rows)
    write(out / 'numerical_validation.json',dict(complete=not partial,all_pass=True,counts=counts,
        pretrained_weight_sha256=actual_weight,native_sha256=actual_library,plaintext_reference_sha256=references,
        max_relative_l2=max(r['relative_l2'] for r in rows),max_abs=max(r['max_abs'] for r in rows),
        limitation='Full model metrics independently recomputed from saved tensors; Block metrics checked against recorded runtime acceptance, because Block tensors were not archived.'))
    print('Numerical evidence verified:',counts,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true')
    verify(ROOT / 'out/12',parser.parse_args().partial)
