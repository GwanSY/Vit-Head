"""Finish, package and validate the authorized out/12 study after execution."""
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import tarfile
import time
import zipfile

from runtime import ROOT, write


def validate_archives(out):
    with zipfile.ZipFile(out / 'results.zip') as archive:
        failed=archive.testzip()
        if failed:raise RuntimeError(f'Corrupt paper package member: {failed}')
    manifest=json.loads((out / 'raw_manifest.json').read_text())
    expected={r['archive_path']:r for r in manifest['files']};verified=set()
    with tarfile.open(out / 'raw_executions.tar.gz','r|gz') as archive:
        for member in archive:
            if member.name not in expected:continue
            row=expected[member.name]
            if not member.isfile() or member.size!=row['bytes']:raise RuntimeError('Raw member size differs')
            h=hashlib.sha256()
            stream=archive.extractfile(member)
            for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
            if h.hexdigest()!=row['sha256']:raise RuntimeError(f'Raw member digest differs: {member.name}')
            verified.add(member.name)
    if verified!=set(expected):raise RuntimeError('Raw evidence missing from archive')
    write(out / 'archive_validation.json',dict(all_pass=True,verified_raw_files=len(verified),
        checks='ZIP CRC for every artifact; SHA256 and size of every original evidence file inside TAR'))


def finish(wait=False):
    out=ROOT / 'out/12'
    while True:
        try:meta=json.loads((out / 'run.json').read_text())
        except json.JSONDecodeError:
            time.sleep(1);continue
        if meta['status'] in ('execution_complete','complete'):break
        if not wait:raise RuntimeError('Secure executions have not finished')
        with (out / '.execution.lock').open('a') as executing:
            try:
                fcntl.flock(executing,fcntl.LOCK_EX|fcntl.LOCK_NB)
                raise RuntimeError('Execution driver ended before completion; inspect its saved logs')
            except BlockingIOError:pass
        time.sleep(25)
    stages=[('tensor_shapes',['group_costs.py']),
            ('analysis',['cost_analysis.py','--out',str(out),'--include-core']),
            ('numerical_recheck',['verify_cost_results.py']),
            ('packaging',['package_cost_results.py'])]
    logs=out / 'analysis_logs';logs.mkdir(exist_ok=True)
    for name,args in stages:
        meta.update(analysis_phase=name,analysis_updated_at=time.time());write(out / 'run.json',meta)
        print('Finalization:',name,flush=True)
        with (logs / (name+'.log')).open('w') as stream:
            subprocess.run([sys.executable,*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    meta.update(analysis_phase='archive_validation',analysis_updated_at=time.time());write(out / 'run.json',meta)
    validate_archives(out)
    meta.update(status='complete',analysis_phase='complete',completed_at=time.time());write(out / 'run.json',meta)
    print('Study complete; actual executions, numerical checks, reports and archives verified',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    with (ROOT / 'out/12/.analysis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finish(args.wait)
