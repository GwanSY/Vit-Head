"""Snapshot reproducible source and verify complete native-study artifacts."""
import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from runtime import ROOT,digest,write

CODE=('packing_audit.py','packing_build.py','packing_study.py','packing_analysis.py','packing_driver.py',
      'packing_intervals.py','packing_verify.py',
      'packing_report.py','packing_package.py','packing_reproduce.py','runtime.py','run.py',
      'optimize_study.py','optimize_analysis.py','optimized_heads.py','model.py','heads.py',
      'experiment.py','mixed.py','group_costs.py','cost_analysis.py','config.json','profile_build.py','build.py')


def package(out):
    src=out/'repro_source';src.mkdir(exist_ok=True)
    for name in CODE:shutil.copy2(ROOT/name,src/name)
    for p in (ROOT/'out/07').glob('alpha.json'):shutil.copy2(p,src/'alpha.json')
    build_source=out/'build_provenance';build_source.mkdir(exist_ok=True)
    for name in ('stage2.json','stage2-patches.json','cpu.json'):
        shutil.copy2(ROOT/'build'/name,build_source/name)
    (ROOT/'native_packing.patch').write_bytes((out/'native.patch').read_bytes())
    instructions=f'''BumbleBee-ViT CPU 原生密文打包批处理复现

项目目录：{ROOT}
固定硬件/协议：P0 cpus 0-24, P1 cpus 48-72, YACL/OMP25线程，两方独立Docker;
各方egress 1Gbps/0.25ms；FM64/f18，原有GELU FM32/f12，Exp6；原始权重与输入。
默认未启用GPU；新路径仅CPU，SPU_BB_PACK_TREE_BATCH=1，0为原始路径。

查看证据：audit_report.html, report.html, summary.txt, before_after.csv,
raw_samples.csv, role_costs.csv, matrix_packing_raw.csv, packing_schedule_raw.csv,
operator_costs.csv, backend_counters.csv, numerical_validation.csv, closure_checks.csv。
所有native原始记录在各phase/log/<case>/log/native-p*-e*.tsv；编译图compiler/*.mlir；
输出outputs.npz，包括warmup和计时调用，原始stdout与run.json全部保留。

在已有本地依赖、模型、build/cache及Docker镜像环境复现全流程到新目录（不能覆盖证据目录）：
  python packing_reproduce.py --evidence {out} --out out/15
此入口恢复build/src私有副本的4个基线源文件，然后应用native.patch，不修改OpenBumbleBee工作副本。
会编译独立库、执行12个精确测试、8轮三配置确认、5轮单类型对照、3轮原库桥接；
只有Block预先规定的CI与Holm检验通过，才执行6对独立冷启动12层ViT。
全程串行；每arm数值gate保持relativeL2<=.05,maxAbs<=.5，全模型top1与同计划参考一致。

分步骤（仅在新输出目录操作；需要Docker访问权限）：
  python packing_build.py --out <新目录> --apply
  python packing_study.py pilot --out <新目录> --sessions 1 --pairs 1
  python packing_study.py confirm --out <新目录> --sessions 8 --pairs 1
  python packing_study.py control --out <新目录> --sessions 5 --pairs 1
  python packing_study.py bridge --out <新目录> --sessions 3
  OPENBLAS_NUM_THREADS=1 taskset -c 25-26 /home/dm/anaconda3/envs/ss_ssm/bin/python packing_analysis.py --out <新目录>
  python packing_study.py full --out <新目录> --case S10R2 --sessions 6
  OPENBLAS_NUM_THREADS=1 /home/dm/anaconda3/envs/ss_ssm/bin/python packing_analysis.py --out <新目录>
  OPENBLAS_NUM_THREADS=1 /home/dm/anaconda3/envs/ss_ssm/bin/python packing_verify.py --out <新目录>
  OPENBLAS_NUM_THREADS=1 /home/dm/anaconda3/envs/ss_ssm/bin/python packing_report.py --out <新目录>
  python packing_package.py --out <新目录>
新目录先复制audit_identity.json、baseline_source/、native.patch以及历史审计表；推荐自动入口。
同目录已有完成的run不会重跑；固定设计不允许临时改变重复次数。失败原始数据保留。

计时：native两方全部PPHLO执行max；包含安全计算和等待，warm Block初始化不在计时样本；
full每arm新运行时无warmup，首次协议密钥初始化在原生推理内；request另含compile/reveal。
模型加载、参数上传、容器创建不在native/request范围。CPU-s是线程累计，不与wall相加；
Recv等待不能当纯网络；packing属于matrix内区间，不叠加计算。
payload=sum(两方sentbytes)，不重复计算recv；不含TCP头、驱动RPC/参数上传/输出reveal。

原始库不改：build/spu/libspu.so；observer build/stage2.so。
新库packing.so为Docker内Python3.10编译版本；在相同镜像运行，宿主机ABI不作为执行入口。
安全目标沿用原协议；没有改变精度、位宽、近似比较标志、掩码或Beaver使用规则。

results.zip为精简交付；raw_executions.tar.gz包含全部运行日志、编译图、输出与新库。
package_audit.csv与package_identity.json核验文件哈希、tar成员SHA和zip CRC。
repro_source/为最终代码快照；native.patch为核心源码与测试变更。
历史out/12与out/13只复用数据，未将历史不同执行口径的时间作优化对照。
'''
    (out/'REPRODUCE.txt').write_text(instructions)
    raw_files=[]
    for phase in ('pilot','confirm','control','bridge','full'):
        raw_files += [p for p in (out/phase).rglob('*') if p.is_file()]
    for directory in ('baseline_source','optimized_source','repro_source','build_attempts','build_provenance','supplement_source','aborted'):
        raw_files += [p for p in (out/directory).rglob('*') if p.is_file()]
    raw_files += [p for p in out.iterdir() if p.is_file() and p.name not in ('results.zip','raw_executions.tar.gz','package_audit.csv','package_identity.json') and '.next.' not in p.name]
    raw_files=sorted(set(raw_files)); records=[]
    for p in raw_files:records.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=digest(p)))
    temp=out/'raw_executions.next.tar.gz'
    with tarfile.open(temp,'w:gz',compresslevel=6) as tar:
        for p in raw_files:tar.add(p,arcname=str(p.relative_to(out)),recursive=False)
    temp.replace(out/'raw_executions.tar.gz')
    expected={x['path']:x['sha256'] for x in records}
    with tarfile.open(out/'raw_executions.tar.gz','r:gz') as tar:
        members=[m for m in tar if m.isfile()]
        if len(members)!=len(expected):raise RuntimeError('Tar member count differs')
        for m in members:
            h=hashlib.sha256();stream=tar.extractfile(m)
            for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
            if h.hexdigest()!=expected[m.name]:raise RuntimeError('Tar hash mismatch: '+m.name)
    with (out/'package_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    small=[p for p in raw_files if len(p.relative_to(out).parts)==1 and p.suffix in ('.csv','.json','.txt','.html','.pdf','.png','.patch')]
    small += list(src.glob('*'))
    small += [p for p in (out/'optimized_source').rglob('*') if p.is_file()]
    small += [p for p in (out/'baseline_source').rglob('*') if p.is_file()]
    small += list(build_source.glob('*'))
    small += [p for p in (out/'supplement_source').glob('*') if p.is_file()]
    small += [out/'package_audit.csv']
    with zipfile.ZipFile(out/'results.zip','w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in sorted(set(small)):z.write(p,str(p.relative_to(out)))
    with zipfile.ZipFile(out/'results.zip') as z:
        bad=z.testzip()
        if bad:raise RuntimeError('Zip CRC failed: '+bad)
        zip_members=len(z.infolist())
    identity=dict(status='verified',tar_members=len(expected),zip_members=zip_members,
        tar_sha256=digest(out/'raw_executions.tar.gz'),zip_sha256=digest(out/'results.zip'),
        tar_member_hashes_verified=True,zip_crc_verified=True,code_sha256={n:digest(ROOT/n) for n in CODE})
    write(out/'package_identity.json',identity)
    print(json.dumps(identity,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'out/14');a=p.parse_args();package(a.out.resolve())
