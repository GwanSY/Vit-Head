"""仅在两侧数值验收通过后发布真实 BERT 单层/ViT 第二层统计。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'OURS/blb_base/scripts'))
from report_b_wire_audit import CATEGORIES, parse, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bert', type=Path, required=True, help='BERT run directory')
    p.add_argument('--vit', type=Path, required=True, help='ViT run directory')
    p.add_argument('--vit-validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    bert_gate = json.loads((args.bert / 'validation.json').read_text())
    vit_gate = json.loads(args.vit_validation.read_text())
    if bert_gate.get('status') != 'PASS' or vit_gate.get('acceptance_pass') is not True:
        raise RuntimeError('both real numerical acceptance gates must pass first')
    rotations, wire, totals, sends, conversions = [], [], [], [], []
    for model, folder, block in [('bert', args.bert, 0), ('vit', args.vit, 1)]:
        for party, name in [(1, 'alice'), (2, 'bob')]:
            path = folder / (name + '.log')
            r, w, t, s = parse(path, model, party)
            for target, rows in [(rotations,r), (wire,w), (totals,t), (sends,s)]:
                target.extend(x for x in rows if x['block']==block)
            if party==1:
                section = path.read_text().split(f'WIRE_BEGIN block={block} ',1)[1].split(f'WIRE_TOTAL block={block} ',1)[0]
                for i, m in enumerate(re.finditer(r'in (mpc_to_ckks|ckks_to_mpc), ([^\n]+)\npts.size\(\):(\d+)', section)):
                    conversions.append(dict(model=model, block=block, boundary=i,
                                            direction=m[1], parameters=m[2], batch_size=int(m[3])))
    if Counter((x['model'],x['party']) for x in totals) != Counter({(m,p):1 for m in ('bert','vit') for p in (1,2)}):
        raise ValueError('missing or duplicate online window')
    out=args.output
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in [('rotation_sites',rotations), ('send_only_stages',wire),
                       ('send_only_totals',totals), ('ciphertext_messages',sends),
                       ('conversion_boundaries',conversions)]:
        if not rows: raise ValueError('empty evidence: '+name)
        write_csv(out/(name+'.csv'),rows)
    def table(items, rows, field, key, cloud_only=False):
        result=[]
        for item in items+['TOTAL']:
            row={key:item}
            for model in ('bert','vit'):
                chosen=[x for x in rows if x['model']==model and
                        (not cloud_only or x['party']==1) and
                        (item=='TOTAL' or x[key]==item)]
                row[model]=sum(x[field] for x in chosen)
                if cloud_only: row[model+'_zero_step']=sum(x[field] for x in chosen if x['step']==0)
            row['ratio']=row['vit']/row['bert'] if row['bert'] else None
            result.append(row)
        return result
    write_csv(out/'rotation_by_operator.csv',table(
        ['QKV','QK^T','Softmax_x_V','Output_Projection','FFN','repacking_other'],
        rotations,'count','operator',True))
    write_csv(out/'communication_by_category.csv',table(CATEGORIES,wire,'sent_bytes','category'))
    (out/'scope.json').write_text(json.dumps({
        'status':'PASS_MEASURED_SINGLE_LAYER_COMPARISON',
        'communication':'Alice_sent + Bob_sent; application serialization bytes; no recv double count',
        'online':'complete block window, includes actual online OT refill, excludes setup/keygen/warmup',
        'rotation':'cloud evaluator API calls including zero-step; zero-step separately listed',
        'boundary':'real BERT post-norm layer 0 including final LN affine vs ViT pre-norm layer 1',
        'five_fused_filter':False,
        'note':'Different model boundaries are reported, not removed to force a desired ratio.',
    },indent=2)+'\n')


if __name__=='__main__': main()
