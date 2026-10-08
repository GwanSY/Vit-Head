"""核验正式 B 与已执行 M2C 探针的源码一致性；不替代网络数值验收。"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from prepare_m2c_audit import ROOT, common_body, function


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime = ROOT / 'ours_vit_b16/runtime'
    audit = ROOT / 'outputs/m2c_audit_20260927'
    generated = common_body(runtime).encode()
    summary = json.loads((audit / 'audit_summary.json').read_text())
    checks = {'recorded_helper_hash': sha(generated) == summary['shared_converter_sha256']}
    for case in ('bert_r04', 'vit_r03'):
        checks[case + '_helper_identical'] = (
            generated == (audit / case / 'source/equivalent_m2c.h').read_bytes())

    # 比较函数本体，避免把无关的通信日志修复误报为 M2C 算法变化。
    signature = 'vector<Plaintext> Bert_ckks::mpc_to_ckks('
    current = (runtime / 'ckks_bert_basic.cpp').read_text()
    a, _, b = function(current, signature)
    artifacts = ROOT / 'outputs/shape_audit_20260927/executed_artifacts'
    with tarfile.open(artifacts / 'runtime_source.tar.gz') as archive:
        previous = archive.extractfile('runtime/ckks_bert_basic.cpp').read().decode()
        x, _, y = function(previous, signature)
        checks['previous_full12_m2c_function_identical'] = current[a:b] == previous[x:y]
        for name in ('he.cpp', 'he.h', 'nonlinear.cpp', 'nonlinear.h'):
            checks['previous_full12_' + name + '_identical'] = (
                archive.extractfile('runtime/' + name).read() == (runtime / name).read_bytes())
    manifest = json.loads((artifacts / 'sha256.json').read_text())
    for name, expected in manifest.items():
        checks['artifact_' + Path(name).name] = sha((ROOT / name).read_bytes()) == expected

    result = {
        'status': 'PASS_SOURCE_IDENTITY_ONLY' if all(checks.values()) else 'FAIL_SOURCE_IDENTITY',
        'checks': checks,
        'helper_sha256': sha(generated),
        'production_m2c_function_sha256': sha(current[a:b].encode()),
        'interpretation': '已测同语义转换取自正式 ViT M2C；B 无需再次替换算法。'
                          '探针将 worker 固定为64并增加统计，正式 B 保留配置与原 profiling。',
        'limits': '源码相同不等于本轮重新执行通过；旧转换 fixture 不等于真实 BERT 单层验收。',
        'bert_entry': {
            'wrapper': 'Vit-Head/BLB/scripts/run_bert_base.sh',
            'target': 'ckks_bert_large_main',
            'input': 'gpu/SCI/tests/bert_large_bolt/ckks_bert_large_main.cpp:402（全1）',
            'placeholder': 'gpu/SCI/tests/bert_large_bolt/ckks_bert.cpp:468（fake V）',
            'alternate': 'bolt_bert_large.cpp 使用 Bert/LIB_BERT_BOLT，非同一 CKKS 五块入口',
        },
        'gates': {
            'real_bert_fixed_clear_secure': 'PENDING_REAL_ENTRY',
            'new_vit_fixed_clear_secure': 'NOT_RUN',
            'new_matched_single_layer_comparison': 'NOT_RUN',
            'new_vit_full12': 'NOT_RUN',
            'final_freeze_eligible': False,
        },
        'runtime_modified_by_this_audit': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(result['status'], args.output)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
