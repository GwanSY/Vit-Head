"""读取独立探针，核对发送字节守恒并汇总旋转步长；不采用旧差值日志桶。"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FUSED = {'layer_norm_part2_attention', 'softmax_he',
         'Softmax_v_linear_layer_norm_part1', 'ln_part2_linear3_gelu_part1',
         'linear4_add_layernorm_part1'}
CATEGORIES = ['ckks_to_mpc', 'mpc_to_ckks', 'nonlinear_MPC', 'ciphertext_transfer', 'other']


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def operator(stage):
    if stage.endswith('/bert_cipher_plain_bsgs'): return 'QKV'
    if stage.endswith('/bert_cipher_cipher_cross_packing'): return 'QK^T'
    if stage.endswith('/softmax_v'): return 'Softmax_x_V'
    if stage.endswith('/bert_cipher_plain_bsgs_2'):
        return 'Output_Projection' if '/Softmax_v_linear_layer_norm_part1/' in stage else 'FFN'
    return 'repacking_other'


def is_fused(stage):
    return any('/'+root+'/' in stage+'/' for root in FUSED)


def parse(path, model, party):
    text = path.read_text()
    rotations = [dict(model=model, party=party, block=int(b), stage=s, site=site,
                      step=int(step), count=int(n), operator=operator(s), five_fused=is_fused(s))
                 for b,s,site,step,n in re.findall(
                     r'ROT_TRACE block=(\d+) stage=(\S+) site=(\S+) step=(-?\d+) count=(\d+)',text)]
    wire = [dict(model=model, party=party, block=int(b), stage=s, category=c, sent_bytes=int(n))
            for b,s,c,n in re.findall(r'WIRE_COUNT block=(\d+) stage=(\S+) category=(\w+) bytes=(\d+)',text)]
    totals = [dict(model=model, party=party, block=int(b), main_sent=int(main), sci_sent=int(sci),
                   sent=int(n), categorized=int(cat), conservation=int(ok))
              for b,main,sci,n,cat,ok in re.findall(
                  r'WIRE_TOTAL block=(\d+) main_sent=(\d+) sci_sent=(\d+) sent=(\d+) categorized=(\d+) conservation=(\d+)',text)]
    sends = [dict(model=model, party=party, block=int(b), stage=s, site=site, ciphertexts=int(cts), sent_bytes=int(n))
             for b,s,site,cts,n in re.findall(
                 r'WIRE_SEND block=(\d+) stage=(\S+) site=(\S+) cts=(\d+) bytes=(\d+)',text)]
    # The first probe wrapped send_encrypted_vector but not HE_CKKS::encrypt_send.
    # Its existing log reports exact serialized bytes (including length headers).
    # Reconcile those independent records before classifying remaining main IO.
    for total in totals:
        block = total['block']
        section = text.split(f'WIRE_BEGIN block={block} ', 1)[1].split(f'WIRE_TOTAL block={block} ', 1)[0]
        seeded = re.findall(r'CKKS_SEEDED_INPUT cts=(\d+) send_bytes=(\d+) secret_key_transferred=0', section)
        residual = [r for r in wire if r['block']==block and r['category']=='other']
        nbytes = sum(int(n) for _,n in seeded)
        if residual and nbytes:
            assert sum(r['sent_bytes'] for r in residual)==nbytes, 'unexplained main-channel traffic'
            for r in residual: r['category']='ciphertext_transfer'
            sends.append(dict(model=model,party=party,block=block,stage='block/seeded_send_reconciliation',
                site='he.h:403:encrypt_send',ciphertexts=sum(int(n) for n,_ in seeded),sent_bytes=nbytes))
    for total in totals:
        parts = [x for x in wire if x['block']==total['block']]
        assert len({(x['stage'],x['category']) for x in parts})==len(parts), 'duplicate log rows'
        assert total['conservation']==1 and total['sent']==total['categorized']==sum(x['sent_bytes'] for x in parts)
        assert sum(x['sent_bytes'] for x in sends if x['block']==total['block']) == sum(x['sent_bytes'] for x in parts if x['category']=='ciphertext_transfer')
    # Warm-up/setup messages must all precede the online snapshot.
    assert totals and 'warm up' not in text[text.index('WIRE_BEGIN'):]
    return rotations, wire, totals, sends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/wire_rotation_audit_20260927')
    out = parser.parse_args().output.resolve()
    rotations, wire, totals, sends = [],[],[],[]
    paths=[]
    conversions=[]
    for model in ['bert','vit']:
        for party,name in [(1,'alice'),(2,'bob')]:
            path=out/model/'run'/(name+'.log'); paths.append(path)
            r,w,t,s=parse(path,model,party)
            rotations+=r;wire+=w;totals+=t;sends+=s
            if party==1:
                log=path.read_text()
                for total in t:
                    block=total['block']
                    section=log.split(f'WIRE_BEGIN block={block} ',1)[1].split(f'WIRE_TOTAL block={block} ',1)[0]
                    for index,match in enumerate(re.finditer(r'in (mpc_to_ckks|ckks_to_mpc), ([^\n]+)\npts.size\(\):(\d+)',section)):
                        conversions.append(dict(model=model,block=block,boundary=index,
                            direction=match[1],parameters=match[2],ciphertexts=int(match[3])))
    assert {(t['model'],t['party'],t['block']) for t in totals} == {
        ('bert',p,0) for p in [1,2]} | {('vit',p,b) for p in [1,2] for b in [0,1]}
    write_csv(out/'rotation_sites.csv',rotations)
    write_csv(out/'send_only_stages.csv',wire)
    write_csv(out/'send_only_totals.csv',totals)
    write_csv(out/'ciphertext_messages.csv',sends)
    write_csv(out/'conversion_boundaries.csv',conversions)
    rot_rows=[]
    for op in ['QKV','QK^T','Softmax_x_V','Output_Projection','FFN','repacking_other','TOTAL']:
        row={'operator':op}
        for model,block,key in [('bert',0,'bert128'),('vit',0,'vit256_first'),('vit',1,'vit256_steady')]:
            selected=[r for r in rotations if r['model']==model and r['party']==1 and r['block']==block and r['five_fused'] and (op=='TOTAL' or r['operator']==op)]
            row[key]=sum(r['count'] for r in selected)
            row[key+'_zero_step']=sum(r['count'] for r in selected if r['step']==0)
        row['ratio_API']=row['vit256_steady']/row['bert128']
        row['ratio_nonzero']=(row['vit256_steady']-row['vit256_steady_zero_step'])/(row['bert128']-row['bert128_zero_step'])
        rot_rows.append(row)
    write_csv(out/'rotation_by_operator.csv',rot_rows)
    comm_rows=[]
    for category in CATEGORIES+['TOTAL']:
        row={'category':category}
        for model,block,key in [('bert',0,'bert128'),('vit',0,'vit256_first'),('vit',1,'vit256_steady')]:
            row[key]=sum(r['sent_bytes'] for r in wire if r['model']==model and r['block']==block and (category=='TOTAL' or r['category']==category))
        row['first_ratio']=row['vit256_first']/row['bert128'] if row['bert128'] else ''
        row['steady_ratio']=row['vit256_steady']/row['bert128'] if row['bert128'] else ''
        comm_rows.append(row)
    write_csv(out/'communication_by_category.csv',comm_rows)
    # Exact loop counts of the current complete diagonal schedule, not a
    # measured or repaired BERT baseline, nor an optimality lower bound.
    formulas=[]
    for length in [128,256]:
        pairs,slots,head_dim=6,16384,64
        blocks=slots//(2*length);feature_cts=head_dim//blocks;depth=blocks.bit_length()-1
        token_rot=2*pairs*length*feature_cts
        tree_rot=2*pairs*length*depth
        formulas.append(dict(tokens=length,feature_blocks=blocks,feature_ciphertexts_per_pair=feature_cts,
            QK_token_rotations=token_rot,QK_feature_reductions=tree_rot,QK_total=token_rot+tree_rot,
            AV_value_rotations=token_rot,AV_diagonal_broadcast=tree_rot,
            scope='analytic complete schedule, NOT measured BERT; includes zero-step API calls'))
    write_csv(out/'complete_schedule_formula.csv',formulas)
    # Prove loop multiplicities and stride histograms for each observed ViT block.
    schedule_checks={}
    for block in [0,1]:
        rr=[r for r in rotations if r['model']=='vit' and r['party']==1 and r['block']==block]
        for suffix,lines in [('bert_cipher_cipher_cross_packing',[3220,3221,3261]),
                             ('softmax_v',[4152,4153]),
                             ('preprocess_softmax_s1_ct_ct',[4460,4464])]:
            for line in lines:
                hist=Counter()
                for r in rr:
                    if r['stage'].endswith('/'+suffix) and r['site']==f'linear_ckks.cpp:{line}:rotate_vector':hist[r['step']]+=r['count']
                if line in [3220,4152]: expected=Counter({i:12 for i in range(256)})
                elif line in [3221,4153]: expected=Counter({i-256:12 for i in range(256)})
                else:
                    each=1536 if line==3261 else 768
                    expected=Counter({sign*256*stride:each for sign in [-1,1] for stride in [1,2,4,8,16]})
                schedule_checks[f'block{block}_line{line}']=hist==expected
    assert all(schedule_checks.values()),schedule_checks
    for block in [0,1]:
        with (out/f'vit/run/block_profile_party1.csv').open() as f:
            row=list(csv.DictReader(f))[block]
        assert next(t['sent'] for t in totals if t['model']=='vit' and t['party']==1 and t['block']==block)==int(row['block_online_comm_bytes'])
    result=dict(status='PASS_MEASUREMENT_AUDIT',algorithm_changes=0,
        runtime_source_changes='read-only sqrt communication logging and donor current_comm accessor only; measured probes precede this logging fix',
        seeded_send_reconciliation='Residual main-channel bytes equal the sum of encrypt_send serialized-byte logs exactly in each block; classified as ciphertext transfer. Raw logs retained.',
        counting='NetIO.counter send-only deltas; unique channel pointers; nested scopes subtract child deltas; sum Alice+Bob',
        units='application serialized payload bytes including serialization length headers; excludes TCP/IP headers',
        online_scope='one transformer block after initialization/preprocessing sync and before layer release; patch/final classifier excluded on both comparisons',
        exclusions=['keygen and context setup','constructor base OT/VOLE and warm-up','layer offline preprocessing/release','recv bytes','old resettable get_comm buckets'],
        ot_note='OT extension messages requested by an online MPC operation remain online; no hypothetical preprocessing cost is subtracted or moved offline',
        primary_comparison='BERT one synthetic block vs ViT steady second block; ViT first block reported separately because initial LN is additional',
        caveat='Same counting scope does not make donor placeholder V/GELU and omitted QK/AV arithmetic equivalent to a real checkpoint block',
        rotation_table=rot_rows,communication_table=comm_rows,schedule_checks=schedule_checks,
        formula=formulas,
        preserved_semantics=['Softmax','GELU','FineGrainFusion','MHP/BSGS','packing principle','security parameters'],
        evidence_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'audit_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'rotations':rot_rows,'communication':comm_rows},indent=2))


if __name__=='__main__': main()
