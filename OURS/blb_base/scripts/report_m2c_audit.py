"""汇总 M2C 内部阶段；父项和子项分表，避免 share conversion 重复累计。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def csv_write(path,rows):
    assert rows, str(path)
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/m2c_audit_20260927')
    p.add_argument('--bert',default='bert_r04');p.add_argument('--vit',default='vit_r03')
    args=p.parse_args();out=args.output.resolve()
    totals=[];details=[];generation=[];transport=[];evidence={};fixtures=[]
    manifests=[]
    for model,dirname in [('bert',args.bert),('vit',args.vit)]:
        manifests.append(json.loads((out/dirname/'m2c_manifest.json').read_text()))
        for party,name in [(1,'alice'),(2,'bob')]:
            path=out/dirname/'run'/(name+'.log');text=path.read_text()
            evidence[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            fixture=re.findall(r'M2C_PUBLIC_FIXTURE bits=(\d+) scale=(\d+) slots=(\d+) max_error=(\S+) pass=(\d+)',text)
            assert len(fixture)==2 and all(r[-1]=='1' for r in fixture),(model,party,'fixture failed')
            fixtures.extend(dict(model=model,party=party,bits=int(b),scale=int(s),slots=int(n),max_error=float(e) if party==2 else None,passed=int(ok)) for b,s,n,e,ok in fixture)
            for b,i,cts,slots,bits,scale,n in re.findall(r'M2C_TOTAL block=(\d+) id=(\d+) cts=(\d+) slots=(\d+) bits=(\d+) scale=(\d+) sent=(\d+)',text):
                totals.append(dict(model=model,party=party,block=int(b),id=int(i),ciphertexts=int(cts),slots=int(slots),bits=int(bits),scale=int(scale),sent=int(n)))
            for b,i,phase,calls,elements,n in re.findall(r'M2C_DETAIL block=(\d+) id=(\d+) phase=(\S+) calls=(\d+) elements=(\d+) sent=(\d+)',text):
                details.append(dict(model=model,party=party,block=int(b),id=int(i),phase=phase,calls=int(calls),elements=int(elements),sent=int(n)))
            for b,site,calls,n in re.findall(r'M2C_GENERATE block=(\d+) site=(\S+) calls=(\d+) rns_coefficients=(\d+) bytes=0',text):
                filename,line=site.split(':')
                source_line=(out/dirname/'source'/filename).read_text().splitlines()[int(line)-1]
                placeholder=model=='bert' and 'encrypt(pt, enc_v_half' in source_line
                generation.append(dict(model=model,party=party,block=int(b),site=site,calls=int(calls),rns_coefficients=int(n),sent=0,m2c_derived=not placeholder))
            for b,stage,site,cts,n in re.findall(r'WIRE_SEND block=(\d+) stage=(\S+) site=(\S+) cts=(\d+) bytes=(\d+)',text):
                transport.append(dict(model=model,party=party,block=int(b),stage=stage,site=site,ciphertexts=int(cts),sent=int(n)))
            for b,n in re.findall(r'WIRE_TOTAL block=(\d+).*?sent=(\d+) categorized=\d+ conservation=1',text):
                assert any(t['model']==model and t['party']==party and t['block']==int(b) for t in totals)
    assert manifests[0]['shared_converter_sha256']==manifests[1]['shared_converter_sha256']
    assert {(t['model'],t['party'],t['block']) for t in totals}=={('bert',p,0) for p in [1,2]}|{('vit',p,b) for p in [1,2] for b in [0,1]}
    previous=ROOT/'outputs/wire_rotation_audit_20260927'
    with (previous/'conversion_boundaries.csv').open() as f:old_calls=list(csv.DictReader(f))
    for model,block in [('bert',0),('vit',0),('vit',1)]:
        before=[]
        for r in old_calls:
            if r['model']==model and int(r['block'])==block and r['direction']=='mpc_to_ckks':
                params=[int(x) for x in r['parameters'].split(':')[1].split(',')]
                before.append((int(r['ciphertexts']),params[0],params[2],params[3]))
        after=[(r['ciphertexts'],r['slots'],r['bits'],r['scale']) for r in totals if r['model']==model and r['party']==1 and r['block']==block]
        assert before==after,(model,block,'conversion workload changed')
    for total in totals:
        rows=[r for r in details if all(r[k]==total[k] for k in ['model','party','block','id'])]
        assert total['sent']==sum(r['sent'] for r in rows if '/' not in r['phase']),total
        for parent in ['slot_ring_extension','coefficient_ring_extension']:
            head=next(r for r in rows if r['phase']==parent)
            children=[r for r in rows if r['phase'].startswith(parent+'/')]
            assert head['sent']==sum(r['sent'] for r in children),(total,parent)
            assert sum(r['elements'] for r in children if r['phase'].endswith('/share_B2A'))==head['elements']
        assert all(r['sent']==0 for r in rows if r['phase'] in ['fft_encode_coefficients','slot_origin_correction','local_field_reduction_rns_ntt'])
    for name,rows in [('m2c_calls.csv',totals),('m2c_phase_details.csv',details),('ciphertext_generation.csv',generation),('ciphertext_transport.csv',transport),('public_fixture.csv',fixtures)]:csv_write(out/name,rows)
    parents=[];children=[]
    for phase in sorted({r['phase'] for r in details}):
        row=dict(phase=phase)
        for model,block,key in [('bert',0,'bert128'),('vit',0,'vit_first'),('vit',1,'vit_second')]:
            rows=[r for r in details if r['model']==model and r['block']==block and r['phase']==phase]
            # Calls and elements describe one logical execution, not two
            # copies of a shared tensor. Bytes sum both parties.
            row[key+'_calls']=sum(r['calls'] for r in rows if r['party']==1)
            row[key+'_calls_unit']='worker primitive calls' if '/' in phase else 'conversion batches'
            if phase in ['fft_encode_coefficients','local_field_reduction_rns_ntt']:
                row[key+'_per_ciphertext_function_calls']=sum(r['ciphertexts'] for r in totals if r['model']==model and r['block']==block and r['party']==1)
            else:
                row[key+'_per_ciphertext_function_calls']=''
            row[key+'_elements']=sum(r['elements'] for r in rows if r['party']==1)
            row[key+'_sent']=sum(r['sent'] for r in rows)
            widths=sorted({r['bits'] for r in totals if r['model']==model and r['block']==block})
            row[key+'_extension_widths']=','.join(f'{n+(16 if phase.startswith("coefficient") else 0)}->{n+40+(16 if phase.startswith("coefficient") else 0)}' for n in widths) if 'extension' in phase else ''
        row['bytes_ratio']=row['vit_second_sent']/row['bert128_sent'] if row['bert128_sent'] else None
        (children if '/' in phase else parents).append(row)
    csv_write(out/'phase_summary.csv',parents);csv_write(out/'share_conversion_summary.csv',children)
    b2a_wire=[]
    for total in totals:
        if total['party']!=1:continue
        for phase in ['slot_ring_extension/share_B2A','coefficient_ring_extension/share_B2A']:
            rows=[r for r in details if all(r[k]==total[k] for k in ['model','block','id']) and r['phase']==phase]
            sender=next(r for r in rows if r['party']==1)
            n=sender['elements'];workers=sender['calls']
            assert n % (workers*128)==0
            # delta=40: packed correction 5N bytes, packed choices N/8,
            # plus a 16-byte hash seed per worker. Remaining communication is
            # in Ferret rcot(), whose pool refill is conditional on inventory.
            fixed=5*n+n//8+16*workers
            measured=sum(r['sent'] for r in rows)
            assert measured>=fixed
            b2a_wire.append(dict(model=total['model'],block=total['block'],id=total['id'],phase=phase,elements=n,worker_calls=workers,fixed_online_payload=fixed,observed_sent=measured,rcot_refill_residual=measured-fixed))
    csv_write(out/'b2a_payload_reconciliation.csv',b2a_wire)
    summary=[]
    for model,block in [('bert',0),('vit',0),('vit',1)]:
        rows=[r for r in totals if r['model']==model and r['block']==block]
        gen=[r for r in generation if r['model']==model and r['block']==block and r['party']==2]
        sends=[r for r in transport if r['model']==model and r['block']==block and r['party']==2]
        summary.append(dict(model=model,block=block,m2c_calls=sum(r['party']==1 for r in rows),ciphertexts=sum(r['ciphertexts'] for r in rows if r['party']==1),slot_elements=sum(r['ciphertexts']*r['slots'] for r in rows if r['party']==1),m2c_sent=sum(r['sent'] for r in rows),client_encrypt_calls=sum(r['calls'] for r in gen),client_encrypt_rns_coefficients=sum(r['rns_coefficients'] for r in gen),client_ciphertext_send_calls=len(sends),client_ciphertexts_sent=sum(r['ciphertexts'] for r in sends),client_ciphertext_bytes=sum(r['sent'] for r in sends)))
        summary[-1]['m2c_derived_encrypt_calls']=sum(r['calls'] for r in gen if r['m2c_derived'])
        summary[-1]['placeholder_encrypt_calls']=sum(r['calls'] for r in gen if not r['m2c_derived'])
    csv_write(out/'comparison.csv',summary)
    result=dict(status='PASS_CONVERSION_AUDIT_NOT_FULL_BERT_VALIDATION',shared_converter_sha256=manifests[0]['shared_converter_sha256'],
        scope='Donor five-block synthetic BERT entry with equivalent M2C vs real ViT second layer; donor QK/AV/GELU/Softmax omissions remain.',
        security='Same signed extension, 40 extension bits per stage, 16 encode guard bits, 40-bit FFT roots, coefficient-domain conversion and per-RNS reduction. Existing HE contexts and activation scales retained.',
        observations=summary,m2c_ratio=summary[2]['m2c_sent']/summary[0]['m2c_sent'],element_ratio=summary[2]['slot_elements']/summary[0]['slot_elements'],ciphertext_ratio=summary[2]['ciphertexts']/summary[0]['ciphertexts'],
        generation_note='Local encrypt calls transmit zero bytes. Downstream client ciphertext transport is separate from mpc_to_ckks; donor placeholder V encryption is explicitly separated from M2C-derived encrypt calls. ViT broadcasts two inverse-sigma plaintexts from 1 to 12 copies each, so 475 M2C plaintexts produce 497 fresh ciphertexts. Donor keeps its existing half packing. These caller behaviors were not modified.',
        nested_note='share_B2A and wrap comparison are children of ring extensions, never added again to parent bytes. Parent calls are batch scopes; children are worker calls. FFT and RNS/NTT also report actual per-ciphertext function calls.',
        ot_note='Actual online OT refill is included and can move between phases depending on pool state; public fixture and startup are outside online window.',
        source='https://www.usenix.org/system/files/usenixsecurity25-xu-tianshi.pdf',paper_mapping={'local_encode':'Algorithm 2 step 1','coefficient_ring_extension_and_local_mod':'Algorithm 2 step 2 / C.3','client_encrypt_and_send':'Algorithm 2 step 3','server_add_plain_share':'Algorithm 2 step 4','pre_fft_extension':'Numerical adapter implementing widened-ring local truncation rationale; not an additional explicit step in Algorithm 2','B2A':'Internal sharing conversion used by extension, not a separate model conversion'},
        raw_log_sha256=evidence)
    with (previous/'communication_by_category.csv').open() as f:
        old=next(r for r in csv.DictReader(f) if r['category']=='mpc_to_ckks')
    result['previous_m2c_sent']={key:int(old[key]) for key in ['bert128','vit256_first','vit256_steady']}
    result['vit_absolute_bytes_change']=summary[2]['m2c_sent']-int(old['vit256_steady'])
    result['unchanged_workload_check']='PASS: ordered calls, ciphertext counts, slots, input bit widths and scales match previous probe for both models.'
    result['b2a_refill_audit']={'formula':'40-bit COT correction 5N + packed choices N/8 + 16 bytes per worker; all observed worker lengths are multiples of 128.',
        'source':['gpu/SCI/src/OT/ferret/silent_ot.h:89','gpu/SCI/deps/emp-ot/emp-ot/ferret/ferret_cot.hpp:152','gpu/SCI/deps/emp-tool/emp-tool/io/io_channel.h:80'],
        'vit_second_coefficient_B2A_refill_bytes':sum(r['rcot_refill_residual'] for r in b2a_wire if r['model']=='vit' and r['block']==1 and r['phase'].startswith('coefficient')),
        'note':'Refill bytes remain online. The residual is inferred by exact wire-length accounting, not a separate rcot counter.'}
    (out/'audit_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
