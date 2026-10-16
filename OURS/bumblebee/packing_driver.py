"""Serial native-packing study with frozen repeats and conditional full ViT."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from runtime import ROOT, digest, write
from packing_study import blocks,original_control,full


def analysis(out):
    import os
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
    with (out/'analysis_driver.log').open('a') as log:
        subprocess.run(['taskset','-c','25-26','/home/dm/anaconda3/envs/ss_ssm/bin/python',
            'packing_analysis.py','--out',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((out/'analysis.json').read_text())


def main(out):
    path=out/'run.json'
    if path.exists():
        meta=json.loads(path.read_text())
        if 'error' in meta:
            meta.setdefault('continuations',[]).append(dict(previous_stage=meta.get('stage'),
                previous_error=meta.pop('error'),resumed_epoch=time.time(),analyzer_sha256=digest(ROOT/'packing_analysis.py')))
    else:
        meta=dict(status='running',study='native RLWE packing-tree batching',started_epoch=time.time(),
            confirmation_sessions=8,confirmation_pairs_per_runtime=1,control_sessions=5,bridge_sessions=3,full_rounds=6,
            code_sha256={n:digest(ROOT/n) for n in ('packing_driver.py','packing_study.py','packing_analysis.py','optimize_study.py','runtime.py')},
            native_sha256=digest(out/'packing.so'),audit_sha256=digest(out/'audit_identity.json'),
            scope='same original attention graph; CPU cryptographic scheduling only')
        write(path,meta)
    def stage(s):meta.update(status='running',stage=s);write(path,meta)
    try:
        stage('pilot');blocks(out,'pilot',1,1,['S10R2','S6R6','S6R3L3'])
        a=analysis(out)
        if not all(r['saved_mean_s']>0 for r in a['comparisons'] if r['phase']=='pilot'):
            meta.update(status='pilot_has_no_consistent_positive_signal',stage='report');write(path,meta);return
        stage('confirm');blocks(out,'confirm',8,1,['S10R2','S6R6','S6R3L3']);a=analysis(out)
        stage('generic_controls');blocks(out,'control',5,1,['S12','R12'])
        stage('original_library_bridge');original_control(out,3);a=analysis(out)
        eligible=[r for r in a['comparisons'] if r['phase']=='confirm' and r['saved_ci_low_s']>0 and r['p_holm']<.05]
        if eligible:
            best=next((r for r in eligible if r['case']=='S10R2'),max(eligible,key=lambda r:r['saved_mean_percent']))
            meta.update(full_trigger=best,full_case=best['case']);stage('full')
            full(out,best['case'],6);analysis(out)
        else:
            meta.update(full_skipped=True,full_skip_reason='No native variant passed confirm net-benefit gate')
        meta.update(status='experiments_complete',stage='report',completed_epoch=time.time());write(path,meta)
    except BaseException as exc:
        meta.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));write(path,meta);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'out/14');a=p.parse_args();main(a.out.resolve())
