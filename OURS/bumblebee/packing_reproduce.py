"""Rebuild and rerun the packing study in a fresh output directory."""
import argparse
import shutil
import subprocess
from pathlib import Path
from runtime import ROOT


def reproduce(evidence,out):
    out.mkdir(parents=True,exist_ok=False)
    for name in ('audit_identity.json','audit_report.html','native.patch','candidates.csv',
                 'actual_matrix_packing.csv','historical_packing_samples.csv','historical_packing_summary.csv'):
        shutil.copy2(evidence/name,out/name)
    shutil.copytree(evidence/'baseline_source',out/'baseline_source')
    for name in ('layout_addendum.html','layout_addendum.json'):
        if (evidence/name).exists():shutil.copy2(evidence/name,out/name)
    subprocess.run(['python','packing_build.py','--out',str(out),'--apply'],cwd=ROOT,check=True)
    subprocess.run(['python','packing_driver.py','--out',str(out)],cwd=ROOT,check=True)
    import os
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(out/'.mplconfig'))
    python='/home/dm/anaconda3/envs/ss_ssm/bin/python'
    for module in ('packing_analysis.py','packing_verify.py','packing_report.py','packing_package.py'):
        subprocess.run(['taskset','-c','25-26',python,module,'--out',str(out)],cwd=ROOT,env=env,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,default=ROOT/'out/14');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    reproduce(a.evidence.resolve(),a.out.resolve())
