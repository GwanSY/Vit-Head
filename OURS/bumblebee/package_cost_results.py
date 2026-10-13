"""Package completed cost evidence, figures and provenance without weights."""
import html
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from runtime import ROOT, digest, write


def run(out):
    meta=json.loads((out / 'run.json').read_text())
    if meta['status'] not in ('execution_complete','complete'):raise RuntimeError('Finish real executions before packaging')
    if not json.loads((out / 'findings.json').read_text())['complete']:raise RuntimeError('Finish analysis before packaging')
    numerical=json.loads((out / 'numerical_validation.json').read_text())
    if not numerical['complete'] or not numerical['all_pass']:raise RuntimeError('Finish numerical rechecks before packaging')
    provenance=out / 'provenance';provenance.mkdir(exist_ok=True)
    for name in ('stage2.json','stage2-patches.json'):
        shutil.copy2(ROOT / 'build' / name,provenance / name)
    sources=provenance / 'sources';sources.mkdir(exist_ok=True)
    names=('secure_cost_study.py','resume_cost_study.py','cost_analysis.py','inspect_cost_ir.py','group_costs.py',
           'package_cost_results.py','verify_cost_results.py','finish_cost_study.py','profile_build.py','runtime.py','experiment.py','block.py','heads.py','model.py','mixed.py','config.json','run.py','README.md')
    for name in names:shutil.copy2(ROOT / name,sources / name)
    audit=json.loads((out / 'audit.json').read_text());folders={Path(c['metadata']).parent for c in audit['configurations']}
    folders.update(ROOT / f'out/10/log/full/log/cpu-{name}-joint' for name in ('S','r48'))
    supplementary_calls=0
    if (out / 'recovery.json').exists():
        recovery=json.loads((out / 'recovery.json').read_text());folders.update(out / p for p in recovery['supplementary_folders'])
        supplementary_calls=recovery['supplemental_verified_calls']
    files=[];header='<html><meta charset="utf-8"><title>原始安全执行索引</title><style>body{font:14px/1.6 sans-serif;max-width:1200px;margin:30px auto}li{margin:4px}</style><h1>原始两方执行与数值证据</h1>'
    index=[header];portable=[header]
    for folder in sorted(folders):
        heading='<h2>'+html.escape(str(folder.relative_to(ROOT)))+'</h2><ul>'
        index.append(heading);portable.append(heading)
        for p in sorted(folder.rglob('*')):
            if not p.is_file() or p.name.endswith('.tmp'):continue
            target=os.path.relpath(p,out)
            index.append('<li><a href="'+html.escape(target,quote=True)+'">'+html.escape(str(p.relative_to(folder)))+'</a></li>')
            archived=p.relative_to(out) if p.is_relative_to(out) else Path('historical') / p.relative_to(ROOT)
            portable.append('<li><a href="'+html.escape(str(archived),quote=True)+'">'+html.escape(str(p.relative_to(folder)))+'</a></li>')
            files.append(dict(path=str(p.relative_to(ROOT)),archive_path=str(archived),bytes=p.stat().st_size,sha256=digest(p)))
        index.append('</ul>');portable.append('</ul>')
    index.append('</html>');portable.append('</html>')
    (out / 'raw_index.html').write_text(''.join(index));(out / 'portable_raw_index.html').write_text(''.join(portable))
    write(out / 'raw_manifest.json',dict(files=files,source_hashes={n:digest(ROOT / n) for n in names},
          scope='New primary and supplementary executions plus reused old core/Block/full evidence; immutable original paths retained'))
    (out / 'PACKAGES.txt').write_text('将 results.zip 与 raw_executions.tar.gz 解压到同一目录。report.html 为图表报告，summary.txt 为简洁报告，portable_raw_index.html 为解压后的原始数据索引。\nraw_index.html 采用本工作区的历史原路径；raw_manifest.json 同时记录原路径、归档路径及 SHA256。\n原始包含本轮全部 Block/ViT 执行与复用的历史 core/Block/S-R48 日志。原预训练权重与原生二进制不重复打包，哈希和构建来源已保存。\n完整模型保留 reference/secure logits；Block 数值误差由真实执行 worker 当场计算并记录，未保存 Block 输出张量。中断后的有效调用与 recovered_evidence.json 保留为补充，不加入主配对统计。\n')
    artifacts=[p for p in out.iterdir() if p.is_file() and p.name!='delivery.json' and p.suffix in ('.csv','.json','.html','.txt','.pdf','.png')]
    artifacts.extend(p for p in provenance.rglob('*') if p.is_file())
    artifacts.extend(p for p in (out / 'compiler').glob('*') if p.is_file())
    with zipfile.ZipFile(out / 'results.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(artifacts):archive.write(p,p.relative_to(out))
    # All raw evidence stays separate from the small paper/data package.
    with tarfile.open(out / 'raw_executions.tar.gz','w:gz',compresslevel=1) as archive:
        for directory in (out / 'blocks/log',out / 'full/log'):
            archive.add(directory,arcname=directory.relative_to(out))
        for folder in sorted(folders):
            if not folder.is_relative_to(out):archive.add(folder,arcname=Path('historical') / folder.relative_to(ROOT))
        for name in ('design.json','recovery.json','provenance.json','raw_manifest.json','portable_raw_index.html','PACKAGES.txt'):
            archive.add(out / name,arcname=name)
    write(out / 'delivery.json',dict(packages={n:dict(bytes=(out / n).stat().st_size,sha256=digest(out / n)) for n in ('results.zip','raw_executions.tar.gz')},
          original_weight_sha256=json.loads((out / 'provenance.json').read_text())['checkpoint_sha256'],
          primary_full_calls=meta['full_total'],supplementary_full_calls=supplementary_calls,new_block_calls=meta['blocks_total']*5,
          archived_reused_raw_evidence=True,all_new_numerical_checks_passed=numerical['all_pass'],protocol_and_pretrained_weights_unchanged=True))
    print('Results and raw executions packaged',flush=True)


if __name__=='__main__':run(ROOT / 'out/12')
