#!/usr/bin/env python3
"""默认验证 Strict BLB-ViT；较弱归属审计和论文性能仅保留历史入口。"""
import argparse
import csv
import difflib
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = 'https://www.usenix.org/system/files/usenixsecurity25-xu-tianshi.pdf'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def table(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vit-run', type=Path, required=True)
    p.add_argument('--bert-run', type=Path)
    p.add_argument('--criterion', choices=['strict_blb_vit', 'blb_based_vit', 'paper_reproduction'],
                   default='strict_blb_vit')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.criterion == 'strict_blb_vit':
        from audit_strict_blb import audit
        report = audit(a.vit_run, a.output)
        print(json.dumps(report, ensure_ascii=False))
        raise SystemExit(0 if report['freeze_allowed'] else 1)
    if a.criterion == 'blb_based_vit':
        from audit_b_adaptations import audit
        report = audit(a.vit_run, a.output)
        print(json.dumps(dict(conclusion=report['conclusion'],
                              active_innovation_count=report['active_innovation_count'],
                              output=str(a.output)), ensure_ascii=False))
        if not report['freeze_allowed']:
            raise SystemExit(1)
        return
    if a.bert_run is None:
        p.error('--criterion paper_reproduction requires --bert-run')
    a.output.mkdir(parents=True, exist_ok=True)
    validation = json.loads((a.vit_run / 'validation.json').read_text())
    if validation.get('version') != 'B' or not validation.get('acceptance_pass'):
        raise ValueError('需要已通过验收的 B 运行')
    metadata = dict(line.split('=', 1) for line in
                    (a.vit_run / 'run_metadata_party1.txt').read_text().splitlines() if '=' in line)
    if metadata.get('executed_transformer_blocks') != '12':
        raise ValueError('需要完整 12 层 ViT')
    selection = dict(line.split('=', 1) for line in
                     (a.vit_run/'gpu_selection.txt').read_text().splitlines() if '=' in line)
    if selection.get('binary_override_sha256') != sha(ROOT/'build/container-gpu/bin/ours_vit_b16_main'):
        raise ValueError('当前 ViT 二进制与被审计运行不一致，不能绑定当前源码说明')
    bert_env = dict(line.split('=', 1) for line in
                    (a.bert_run/'environment.txt').read_text().splitlines() if '=' in line)
    if bert_env.get('binary_sha256') != sha(ROOT/'build/container-gpu/bin/ckks_bert_large_main'):
        raise ValueError('当前 BERT 二进制与被审计运行不一致')
    donor = ROOT / 'gpu/SCI/tests/bert_large_bolt'
    vit = ROOT / 'ours_vit_b16/runtime'
    original = ROOT.parents[1] / 'BLB/gpu/SCI/tests/bert_large_bolt'
    lineage = []
    for name in ['he.cpp', 'he.h', 'linear_ckks.cpp', 'linear_ckks.h', 'nonlinear.cpp',
                 'nonlinear.h', 'ckks_bert.cpp', 'ckks_bert.h', 'ckks_bert_basic.cpp',
                 'bert_utils.cpp', 'bert_utils.h']:
        x, y = (donor/name).read_text().splitlines(), (vit/name).read_text().splitlines()
        m = difflib.SequenceMatcher(None, x, y, autojunk=False)
        lineage.append(dict(file=name, donor_sha256=sha(donor/name), vit_sha256=sha(vit/name),
                            donor_matches_BLB_tree=sha(donor/name) == sha(original/name),
                            donor_lines=len(x), vit_lines=len(y),
                            identical_ordered_lines=sum(b.size for b in m.get_matching_blocks())))
    table(a.output/'source_lineage.csv', lineage)
    shared_differences = []
    shared_original = ROOT.parents[1]/'BLB/gpu/SCI/src'
    for file in sorted(shared_original.rglob('*')):
        if file.is_file() and file.suffix in {'.h', '.hpp', '.cpp'}:
            current = ROOT/'gpu/SCI/src'/file.relative_to(shared_original)
            if current.exists() and sha(file) != sha(current):
                shared_differences.append(dict(file=str(file.relative_to(shared_original)),
                                               donor_sha256=sha(file), current_sha256=sha(current)))
    table(a.output/'shared_sci_changes.csv', shared_differences or [dict(file='none', donor_sha256='', current_sha256='')])
    rows = [
        ('CKKS+MPC', 'SEAL/Phantom + SCI/VOLE', '同一 OURS 构建树、同一 Phantom 动态库', '保留框架，非安全证明'),
        ('安全目标', 'he.cpp: SEAL sec_level_type::tc128', 'he.cpp: 同为 tc128', '安全目标保留；不表示上下文和数值参数逐项相同'),
        ('融合块1', 'linear4_add_layernorm_part1', '同名函数；ViT pre-norm 调整残差/LN 边界', '有适配'),
        ('融合块2', 'layer_norm_part2_attention', '同名函数，真实 QKV/QK、公开 padding', '有适配'),
        ('融合块3', 'softmax_he', 'HE 指数 + MPC 乘法/归一化；非负转换快路径', '有适配'),
        ('融合块4', 'Softmax_v_linear_layer_norm_part1', '同名函数，真实 V、WO、ViT 残差', '有适配'),
        ('融合块5', 'ln_part2_linear3_gelu_part1', '同名函数，真实 FFN1；even_quartic GELU', '有适配，不能声称近似逐项相同'),
        ('BSGS / QKV / WO / FFN', 'bert_cipher_plain_bsgs / bert_cipher_plain_bsgs_2', '保留 BSGS 调用；占位全1权重改为真实 checkpoint 编码', '来源可追溯，权重与编码成本不同'),
        ('原 packing', '头对/密文打包', '头对/BSGS packing 保留；real-imag 输出布局与 padding 紧凑化见 metadata', '部分保留，有新增/修正布局'),
        ('QK / AV', 'cross_packing / softmax_v / preprocess_softmax_s1_ct_ct', 'complete_v1 + direct_target_v1；补全作者占位 AV mask/广播', '不是原快速路径逐项等价的证据'),
        ('转换', 'ckks_to_mpc / mpc_to_ckks', 'wide_centered / wide_shared_rns_extension / fused_prime_wrap', '接口与体系继承，协议实现已修正'),
        ('共享 SCI 转换', '原 z_prime2ring 使用 MSB_to_Wrap、假定非负', '当前共享库使用 wrap_computation_prime + B2A；本次BERT也调用修正版', '超出形状适配，不能称原始转换完全不变；性能影响未单独A/B量化'),
        ('形状', '128 tokens / 768 / 12 heads；合成单层', '197 logical / 256 physical；768 / 12 heads；真实12层', '预期模型适配'),
        ('精度', '论文非线性43/13；作者入口合成输入40/20', '固定37/12；各 HE context 见 metadata', '不能声称与论文数值参数完全一致'),
        ('端云', '两方合成输入，无真实模型/图片所有权验证', 'EdgeCloudRuntime：cloud 模型、edge 图片、公开固定计划', 'ViT 端云契约保留；BERT 性能入口不验证该契约'),
    ]
    table(a.output/'protocol_consistency.csv', [dict(item=x, bert=b, vit=v, conclusion=c) for x,b,v,c in rows])
    kernels = {
        'QKV': 'Linear_ckks::bert_cipher_plain_bsgs [Phantom DEVICE, real weights]',
        'QK^T': 'Linear_ckks::bert_cipher_cipher_cross_packing [complete_v1, direct_target_v1]',
        'Softmax': 'Bert_ckks::softmax_he + Linear_ckks::preprocess_softmax_s1_ct_ct [包含AV预打包]',
        'Softmax_x_V': 'Linear_ckks::softmax_v [Phantom DEVICE]',
        'Output_Projection': 'Linear_ckks::bert_cipher_plain_bsgs_2 [WO]',
        'FFN1': 'Linear_ckks::bert_cipher_plain_bsgs_2 [FFN1]',
        'FFN2': 'Linear_ckks::bert_cipher_plain_bsgs_2 [FFN2]',
        'CKKS_to_MPC': 'Bert_ckks::ckks_to_mpc / centered_field_to_ring_wide',
        'MPC_to_CKKS': 'Bert_ckks::mpc_to_ckks [wide_shared_rns_extension]',
        'LayerNorm1': 'Bert_ckks::layer_norm_nonlinear [SCI MPC]',
        'LayerNorm2': 'Bert_ckks::layer_norm_nonlinear [SCI MPC]',
        'GELU': 'Bert_ckks::gelu_nonlinear [SCI MPC, even_quartic]',
    }
    fingerprint = []
    block_totals = []
    for party in [1, 2]:
        stages = list(csv.DictReader((a.vit_run/f'block_online_profile_party{party}.csv').open()))
        blocks = list(csv.DictReader((a.vit_run/f'block_profile_party{party}.csv').open()))
        for row in stages:
            if row['block'] == '0' and not row['stage'].startswith('BLOCK_'):
                fingerprint.append(dict(party=party, **row, implementation=kernels[row['stage']],
                                        evidence='运行阶段计数+源码分支；不是CUDA profiler内核trace',
                                        counter_caveat=('QK GPU 两处 rescale 循环未更新 rescale_count；6*512*2=6144 次/层为源码推导，CSV 的0不是实际0'
                                                        if row['stage'] == 'QK^T' else '现有instrumentation口径，非全部底层调用证明')))
        for block in blocks:
            for key in ['he_rotations', 'he_ct_pt_multiply', 'he_ct_ct_multiply', 'he_rescales', 'conversion_ciphertexts']:
                measured = sum(int(r[key]) for r in stages if r['block'] == block['block'] and not r['stage'].startswith('BLOCK_'))
                if measured != int(block[key]):
                    raise ValueError(f'计数不闭合: party={party} block={block["block"]} {key}')
        block_totals.append(dict(party=party, **blocks[0]))
    table(a.output/'vit_layer0_fingerprint.csv', fingerprint)
    table(a.output/'vit_layer0_totals.csv', block_totals)
    reference = json.loads((a.vit_run/'reference.json').read_text())
    text = (a.vit_run/'secure_logits_party1.txt').read_text()
    secure = list(map(float, text.split('values\n', 1)[1].split('\nend', 1)[0].split()))
    numeric = []
    for label, x, y in [('clear_fixed', reference['clear']['logits'], reference['fixed']['logits']),
                        ('fixed_secure', reference['fixed']['logits'], secure),
                        ('clear_secure', reference['clear']['logits'], secure)]:
        if len(x) != len(y) or not all(math.isfinite(z) for z in x+y):
            raise ValueError('logits 维度或有限值检查失败')
        numeric.append(dict(pair=label, elements=len(x),
                            relative_l2=math.sqrt(sum((u-v)**2 for u,v in zip(x,y))/max(sum(u*u for u in x),1e-30)),
                            max_abs=max(abs(u-v) for u,v in zip(x,y)),
                            reference_top1=max(range(len(x)), key=x.__getitem__),
                            compared_top1=max(range(len(y)), key=y.__getitem__)))
    table(a.output/'numerical_consistency.csv', numeric)
    if any(row['reference_top1'] != row['compared_top1'] for row in numeric):
        raise ValueError('重新计算的 Top-1 不一致')
    if not math.isclose(numeric[1]['relative_l2'], validation['fixed_relative_l2'], rel_tol=1e-8):
        raise ValueError('重新计算的数值误差与运行验收记录不一致')
    logs = [(a.bert_run/f'{name}.log').read_text() for name in ['alice','bob']]
    def last(log, name):
        values = re.findall(r'^'+re.escape(name)+r':\s*([0-9.eE+-]+)\s*$', log, re.M)
        return float(values[-1]) if values else None
    times = [last(log, 'total time') for log in logs]
    status = (a.bert_run/'status.txt').read_text() if (a.bert_run/'status.txt').exists() else ''
    complete = ('alice_rc=0' in status and 'bob_rc=0' in status
                and all(t is not None for t in times)
                and not any('Error opening file:' in log for log in logs))
    bert = dict(scope='author_synthetic_single_layer', layers_actually_executed=1,
                tokens=128, hidden=768, heads=12, complete=complete, party_online_seconds=times,
                full12_measured_seconds=None, clear_fixed_secure=None,
                operation_counters=None, counter_status='作者入口没有实际操作计数器，不用理论预算冒充',
                paper_table9_gpu_seconds=150, paper_table9_communication_GB=3.0,
                paper_source=PAPER, communication_buckets_each_party=[{
                    k:last(log,k) for k in ['ct_transfer_comm','conversion_comm','other_nonlinear_comm']
                } for log in logs])
    if complete:
        bert['single_layer_max_seconds'] = max(times)
        bert['twelve_layer_linear_extrapolation_seconds_NOT_MEASURED'] = 12*max(times)
        bert['extrapolation_div_paper_latency'] = 12*max(times)/150
        buckets = bert['communication_buckets_each_party']
        if all(value is not None for row in buckets for value in row.values()):
            bert['single_layer_reported_aggregate_bytes_rounded'] = sum(sum(row.values()) for row in buckets)
            bert['twelve_layer_communication_GB_NOT_MEASURED'] = 12*bert['single_layer_reported_aggregate_bytes_rounded']/1e9
        bert['communication_scope'] = '双方最终三个通信桶求和；作者日志有效位有限，非精确字节计数'
        boundaries = []
        for name in ['qdisc_online_start.txt', 'qdisc_online_end.txt']:
            file = a.bert_run/name
            match = re.search(r'Sent (\d+) bytes', file.read_text()) if file.exists() else None
            boundaries.append(int(match.group(1)) if match else None)
        if all(x is not None for x in boundaries):
            bert['qdisc_online_bytes_approx_with_transport_headers'] = boundaries[1]-boundaries[0]
        bert['qdisc_boundary_note'] = '开始边界为轮询begin test标记，结束为进程退出；不是精确应用层通信'
    artifacts = {}
    for folder in [donor, vit, ROOT/'gpu/SCI/src', ROOT/'gpu/SCI/extern/phantom']:
        if folder.exists():
            for file in sorted(folder.rglob('*')):
                if file.is_file() and file.suffix in {'.h', '.hpp', '.cpp', '.cu', '.cuh'}:
                    artifacts[str(file.relative_to(ROOT))] = sha(file)
    for file in [ROOT/'build/container-gpu/bin/ours_vit_b16_main',
                 ROOT/'build/container-gpu/bin/ckks_bert_large_main',
                 ROOT/'build/container-gpu/extern/phantom/lib/libPhantom.so',
                 ROOT/'build/container-gpu/CMakeCache.txt',
                 ROOT/'gpu/SCI/tests/bert_large_bolt/CMakeLists.txt',
                 ROOT/'scripts/validate_b_migration.py',
                 ROOT/'scripts/run_migration_bert.sh']:
        artifacts[str(file.relative_to(ROOT))] = sha(file)
    for folder in [a.vit_run, a.bert_run]:
        for file in sorted(folder.iterdir()):
            if file.is_file():
                artifacts[str(file.resolve())] = sha(file)
    (a.output/'artifact_sha256.json').write_text(json.dumps(artifacts, indent=2)+'\n')
    report = dict(conclusion='NOT_ESTABLISHED', freeze_as_correctness_snapshot=bool(validation['acceptance_pass']),
                  freeze_as_faithful_paper_performance_baseline=False,
                  reason='有明确源码继承和ViT数值证据；真实12层BERT、同口径精度及协议等价性证据不足，且存在超出形状的实现差异。',
                  research_matrix_changed=False, vit_runtime_changed=False,
                  bert=bert, vit_validation=validation, vit_metadata=metadata,
                  environment=dict(bert=bert_env, vit=selection,
                                   bert_qdisc=(a.bert_run/'qdisc_active.txt').read_text(),
                                   vit_qdisc=(a.vit_run/'qdisc_active.txt').read_text()),
                  source_lineage=lineage, numerical=numeric,
                  shared_sci_changes=shared_differences,
                  limitations=['ViT单输入不是ImageNet准确率验证', 'BERT GPU2/5，历史ViT GPU0/1；同机同型号，不是相同GPU UUID',
                               'BERT合成入口不是端云真实模型完整推理', '五组融合函数保留不等于原协议图逐边相同',
                               'CUDA内核launch trace未采集；表中为实际分支对应C++执行函数',
                               'QK GPU rescale实际有两个循环，现有计数漏记6144次/层（源码推导）；5483不能当作完整rescale数',
                               'ViT操作计数使用现有instrumentation口径，阶段闭合不能排除共同漏记'])
    (a.output/'migration_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(conclusion=report['conclusion'], bert_complete=complete, output=str(a.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
