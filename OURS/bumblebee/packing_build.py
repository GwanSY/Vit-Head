"""Build audited CPU packing prototype without replacing baseline libraries."""
import argparse
import difflib
import json
import shutil
import subprocess
from pathlib import Path
from runtime import ROOT, digest, write

FILES = ['libspu/mpc/bumblebee/rlwe/packlwes.h',
         'libspu/mpc/bumblebee/rlwe/packlwes.cc',
         'libspu/mpc/bumblebee/rlwe/packlwes_test.cc',
         'libspu/mpc/bumblebee/arith/bumblebee_dot.cc']


def build(out, apply=False):
    identity = json.loads((out/'audit_identity.json').read_text())
    if digest(ROOT/'build/stage2.so') != identity['observer_sha256']:
        raise RuntimeError('Immutable observer changed')
    if digest(ROOT/'build/spu/libspu.so') != identity['original_cpu_sha256']:
        raise RuntimeError('Immutable original CPU changed')
    if apply:
        # Source snapshots + diff are carried with the experiment. Never
        # rewrite the user's OpenBumbleBee working copy.
        for name in FILES:
            dest=ROOT/'build/src'/name
            dest.write_bytes((out/'baseline_source'/name).read_bytes())
        subprocess.run(['patch','-p1','-i',str(out/'native.patch')],cwd=ROOT/'build/src',check=True)
    patches=[]
    for name in FILES:
        original=out/'baseline_source'/name; changed=ROOT/'build/src'/name
        patches.extend(difflib.unified_diff(original.read_text().splitlines(True),changed.read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        snapshot=out/'optimized_source'/name;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(changed.read_bytes())
    (out/'native.patch').write_text(''.join(patches))
    original=json.loads((ROOT/'build/stage2.json').read_text())
    cmd=list(original['command']); cmd[cmd.index('--name')+1]='ours-bb-packing-build'
    target='//libspu/mpc/bumblebee/rlwe:packlwes_test'
    cmd.extend([target])
    native=out/'packing.so'
    provenance=dict(status='building',backend='cpu',command=cmd,
        original_library_sha256=identity['observer_sha256'],audit_sha256=digest(out/'audit_identity.json'),
        patch_sha256=digest(out/'native.patch'),sources=dict(original['sources']))
    for name in FILES:provenance['sources'][name]=digest(ROOT/'build/src'/name)
    write(native.with_suffix('.json'),provenance)
    if (out/'build.log').exists():
        history=out/'build_attempts';history.mkdir(exist_ok=True)
        shutil.copy2(out/'build.log',history/f"attempt-{len(list(history.glob('*.log')))+1}.log")
    with (out/'build.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    product=ROOT/'build/cache/91451686bbf5ebe79baf6c2b1919d0ed/execroot/spulib/bazel-out/k8-opt/bin/spu/libspu.so'
    # Bazel output mode is read-only. Atomic replacement also avoids partial
    # libraries on a repeated build; do not edit a mounted file in place.
    temporary=native.with_suffix('.next.so')
    shutil.copy2(product,temporary)
    temporary.replace(native)
    testcmd=list(original['command']);testcmd[testcmd.index('--name')+1]='ours-bb-packing-tests'
    testcmd[testcmd.index('build')]='test';testcmd[-1]=target
    testcmd += ['--test_output=all','--test_env=YACL_NUM_THREADS=25',
        '--test_env=OMP_NUM_THREADS=25','--test_env=SPU_BUMBLEBEE_HE_BACKEND=cpu',
        '--test_filter=*BatchedRLWEBitwise*:*RuntimeContextsBitwise*:*PackRLWEs*']
    with (out/'native_tests.log').open('w') as log:
        subprocess.run(testcmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    testroot=ROOT/'build/cache/91451686bbf5ebe79baf6c2b1919d0ed/execroot/spulib/bazel-out/k8-opt/testlogs/libspu/mpc/bumblebee/rlwe/packlwes_test'
    for name in ('test.log','test.xml'):
        if (testroot/name).exists():
            dest=out/('unit_'+name);temporary=dest.with_name(dest.name+'.next')
            shutil.copy2(testroot/name,temporary);temporary.replace(dest)
    provenance.update(status='built',library_sha256=digest(native),test_command=testcmd,
                      test_log_sha256=digest(out/'native_tests.log'),test_passed=True)
    write(native.with_suffix('.json'),provenance)
    print(json.dumps(dict(library=str(native),sha256=provenance['library_sha256'],test_passed=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'out/14');p.add_argument('--apply',action='store_true');a=p.parse_args()
    build(a.out.resolve(),a.apply)
