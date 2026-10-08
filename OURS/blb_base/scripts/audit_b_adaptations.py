"""已人工审阅改动的机器复核；源码或证据变化须重新审阅，不能自动继承PASS。"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT/'ours_vit_b16/experiments/b_baseline_adaptations.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def kv(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def audit(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(CATALOG.read_text())
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
    source_changes = [name for name, digest in catalog['reviewed_source_sha256'].items()
                      if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
    check('reviewed_source_unchanged', not source_changes)
    # 枚举新文件，避免新增实现绕过已有文件哈希。
    files = sorted(str(p.relative_to(ROOT)) for folder in catalog['source_roots']
                   for p in (ROOT/folder).rglob('*')
                   if p.is_file() and p.suffix in {'.cpp', '.h', '.hpp', '.cu', '.cuh', '.py'})
    check('reviewed_source_inventory_unchanged', files == catalog['source_inventory'])
    evidence_changes = [name for name, digest in catalog['reviewed_run_sha256'].items()
                        if not (run/name).is_file() or sha(run/name) != digest]
    check('reviewed_run_unchanged', not evidence_changes)
    check('binary_unchanged', sha(ROOT/'build/container-gpu/bin/ours_vit_b16_main') == catalog['binary_sha256'])
    check('phantom_unchanged', sha(ROOT/'build/container-gpu/extern/phantom/lib/libPhantom.so') == catalog['phantom_sha256'])
    selection = kv(run/'gpu_selection.txt')
    check('run_binary_binding', selection.get('binary_override_sha256') == catalog['binary_sha256'])
    result = json.loads((run/'validation.json').read_text())
    check('execution_and_numeric_acceptance', result.get('acceptance_pass') and result.get('execution_pass') and result.get('numerical_pass'))
    check('no_new_P', result.get('version') == 'B' and result.get('new_heterogeneous_packing') is False)
    check('both_exit_zero', kv(run/'status.txt') == {'alice_rc': '0', 'bob_rc': '0'})
    required = dict(version='B', executed_transformer_blocks='12', partial_run='0',
                    head_plan='built_in_all_softmax_original_order',
                    execution_framework='ours_native_edge_cloud_v1', edge_cloud_plan_agreed='1',
                    logical_tokens='197', physical_tokens='256', fixed_point_ring_bits='37',
                    fixed_point_scale_bits='12', attention_diagonal_kernel='complete_v1',
                    gelu_polynomial='even_quartic', conversion_path='wide_shared_rns_v1',
                    client_decrypt_reencrypt='0', mpc_linear_surrogate='0')
    innovation_flags = ['protocol_aware_head_reorder', 'selective_conversion',
                        'heterogeneous_head_grouping', 'native_ckks_linear']
    active = {row['id'] for row in catalog['modifications']
              if row['category'] == 4 and row['active_in_B']}
    metadata = []
    for party in (1, 2):
        meta = kv(run/f'run_metadata_party{party}.txt')
        metadata.append(meta)
        check(f'party{party}_model_execution_contract', all(meta.get(k) == v for k,v in required.items()))
        check(f'party{party}_role', meta.get('edge_cloud_role') == ('cloud' if party == 1 else 'edge'))
        check(f'party{party}_innovation_flags_known', all(k in meta for k in innovation_flags))
        for key in innovation_flags:
            if meta.get(key, '0') != '0':
                active.add(key)
        plan = csv_rows(run/f'head_plan_party{party}.csv')
        check(f'party{party}_144_unique_heads', len(plan) == 144 and
              {(int(r['layer']), int(r['physical_head'])) for r in plan} ==
              {(l,h) for l in range(12) for h in range(12)})
        for row in plan:
            if row['protocol'] != 'softmax_mpc': active.add('heterogeneous_heads')
            if row['original_head'] != row['physical_head']: active.add('head_reorder')
            if float(row['alpha']) != 1.0: active.add('head_alpha')
        check(f'party{party}_head_version', all(r['version'] == 'B' for r in plan))
        blocks = csv_rows(run/f'block_profile_party{party}.csv')
        check(f'party{party}_12_executed_blocks', len(blocks) == 12 and {int(r['block']) for r in blocks} == set(range(12)))
        contract = run/f'edge_cloud_contract_party{party}.txt'
        check(f'party{party}_contract_digest', sha(contract) == meta.get('edge_cloud_contract_sha256'))
        log = (run/('alice.log' if party == 1 else 'bob.log')).read_text()
        for marker in ['OURS_HETERO_NORMALIZATION', 'OURS_HETERO_ASSIGNMENT_PROFILE']:
            if marker in log: active.add(marker)
        check(f'party{party}_public_static_plan', 'public_pre_inference=1 private_dynamic_branch=0' in log)
    check('same_edge_cloud_contract', metadata[0].get('edge_cloud_contract_sha256') == metadata[1].get('edge_cloud_contract_sha256'))
    if result.get('new_heterogeneous_packing') is not False: active.add('new_P')
    check('category4_absent_from_B_execution', not active)
    # 必须读原始 logits，不只信任旧 validation 的 PASS。
    ref = json.loads((run/'reference.json').read_text())
    def logits(party):
        text = (run/f'secure_logits_party{party}.txt').read_text()
        return [float(x) for x in text.split('values\n', 1)[1].split('\nend', 1)[0].split()]
    secure = logits(1)
    check('both_secure_logits_equal', secure == logits(2))
    numerical = []
    for name,x,y in [('clear_fixed', ref['clear']['logits'], ref['fixed']['logits']),
                     ('fixed_secure', ref['fixed']['logits'], secure),
                     ('clear_secure', ref['clear']['logits'], secure)]:
        finite = len(x) == len(y) == 1000 and all(math.isfinite(v) for v in x+y)
        check(name+'_finite_1000', finite)
        if not finite: raise ValueError('invalid logits')
        topx, topy = max(range(1000), key=x.__getitem__), max(range(1000), key=y.__getitem__)
        check(name+'_top1_equal', topx == topy)
        numerical.append(dict(pair=name, relative_l2=math.sqrt(sum((a-b)**2 for a,b in zip(x,y))/max(sum(a*a for a in x),1e-30)),
                              max_abs=max(abs(a-b) for a,b in zip(x,y)), reference_top1=topx, compared_top1=topy))
    check('secure_fixed_original_tolerance', numerical[1]['relative_l2'] <= .05 and numerical[1]['max_abs'] <= .5)
    lineage = []
    donor = ROOT/'gpu/SCI/tests/bert_large_bolt'
    original = ROOT.parents[1]/'BLB/gpu/SCI/tests/bert_large_bolt'
    for name in catalog['donor_files']:
        same = sha(donor/name) == sha(original/name) == catalog['donor_sha256'][name]
        lineage.append(dict(file=name, same_as_reviewed_donor=same, sha256=sha(donor/name)))
    check('source_inheritance', all(r['same_as_reviewed_donor'] for r in lineage))
    rows = catalog['modifications']
    check('classification_complete', all(r['category'] in [1,2,3,4] and r['reason'] and r['source'] for r in rows))
    check('no_reviewed_active_category4', not any(r['category'] == 4 and r['active_in_B'] for r in rows))
    passed = all(checks.values())
    reviewed = checks['reviewed_source_unchanged'] and checks['reviewed_source_inventory_unchanged']
    report = dict(criterion='blb_based_vit_v1', conclusion='PASS_BLB_BASED_VIT_BASELINE' if passed else 'REVIEW_REQUIRED',
                  freeze_allowed=passed, baseline_name='BLB-based ViT baseline', version='B',
                  online_seconds=result['online_s'], active_innovation_count=len(active) if reviewed or active else None, active_innovations=sorted(active),
                  meaning_of_zero='第④类在B执行路径为零；共享仓库包含被B分支排除的后续创新实现',
                  paper_absolute_performance_is_gate=False, full_BERT_is_gate=False,
                  checks=checks, source_changes=source_changes, evidence_changes=evidence_changes,
                  binary_sha256=catalog['binary_sha256'], reviewed_catalog_sha256=sha(CATALOG),
                  numerical=numerical, modifications=rows,
                  limits=['单图完整12层验收，不替代ImageNet数据集级准确率',
                          'GELU近似已改变，不能声称与donor多项式逐项等价；B及后续版本共用相同定义',
                          'QK rescale历史计数漏6144次/层（源码推导）；保留原始日志，不影响本次路径分类',
                          '源码/二进制/计划改变须重新审计；本报告不是密码协议安全证明'],
                  attribution_rule='①②③均属于固定公共基线，各版本沿用；只有④的差分可归因于本文方法，HA到HAP还须固定语义计划。')
    (output/'migration_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    write_csv(output/'adaptation_classification.csv', rows)
    write_csv(output/'numerical_consistency.csv', numerical)
    write_csv(output/'source_lineage.csv', lineage)
    return report
