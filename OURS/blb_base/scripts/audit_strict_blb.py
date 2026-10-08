"""严格迁移验收；数值通过不自动授予算法归属或冻结资格。"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_REVIEW_SOURCES = {
    'ours_vit_b16/runtime/' + name for name in (
        'ckks_vit_b16_main.cpp', 'ckks_bert.cpp', 'ckks_bert_basic.cpp',
        'ckks_bert.h', 'linear_ckks.cpp', 'linear_ckks.h', 'he.h', 'he.cpp',
        'nonlinear.cpp', 'nonlinear.h', 'wide_fft_math.h',
        'packed_block_rotation.h', 'secure_row_max.h', 'edge_cloud_runtime.h',
        'hetero_plan.cpp', 'hetero_plan.h')
} | {'ours_vit_b16/scripts/vit_reference.py',
     'gpu/SCI/src/BuildingBlocks/value-extension.cpp',
     'gpu/SCI/src/BuildingBlocks/value-extension.h'}


def audit(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checks = {}
    def read_json(name):
        path = run / name
        return json.loads(path.read_text()) if path.is_file() else {}
    def metadata(party):
        path = run / f'run_metadata_party{party}.txt'
        return dict(line.split('=', 1) for line in path.read_text().splitlines()
                    if '=' in line) if path.is_file() else {}
    result = read_json('validation.json')
    metas = [metadata(p) for p in (1, 2)]
    checks['full_execution_and_numerics'] = result.get('acceptance_pass') is True
    checks['new_packing_disabled'] = result.get('new_heterogeneous_packing') is False
    required = dict(version='B', executed_transformer_blocks='12', partial_run='0',
                    execution_framework='ours_native_edge_cloud_v1',
                    edge_cloud_plan_agreed='1', head_plan='built_in_all_softmax_original_order',
                    softmax_product='ckks', gelu_polynomial='blb_F0_F1',
                    softmax_reciprocal='rsqrt_square_full_ring',
                    secure_exp_clamp='1',
                    softmax_exp_polynomial='clipped_(1+x/64)^64',
                    mpc_to_ckks_protocol='encode_then_coefficient_ring_to_field',
                    vit_residual_addition='ckks_fused')
    for key, value in required.items():
        checks[key] = all(m.get(key) == value for m in metas)
    cloud_log_path = run / 'alice.log'
    cloud_log = cloud_log_path.read_text() if cloud_log_path.is_file() else ''
    for name, marker in {
        'attention_fusion_12_layers': 'STRICT_BLB_FUSION stage=Softmax_v_linear_layer_norm_part1 statistics=HE skip_branch=masked',
        'ffn_fusion_12_layers': 'STRICT_BLB_FUSION stage=linear4_add_layernorm_part1 statistics=HE skip_branch=masked',
        'gelu_graph_12_layers': 'polynomial=blb_F0_F1 ct_ct_per_input_ciphertext=3',
        'softmax_HE_product_12_layers': 'FORMAL_SOFTMAX_PRODUCT status=EXECUTED engine=CKKS',
        'softmax_full_ring_12_layers': 'FORMAL_SOFTMAX_DENOMINATOR_RING input_ell=37 reciprocal_input_ell=37 row_bound=0 guard_factor=0 local_reduce=0',
        'softmax_exponent_12_layers': 'FORMAL_SOFTMAX_EXP_CLIP status=PASS cutoff=-13 comparison=secure_LT divisor=2^6 exponent=2^6 divisor_folded_into_conversion=0',
    }.items():
        checks[name] = cloud_log.count(marker) == 12
    for key in ('selective_conversion', 'protocol_aware_head_reorder',
                'heterogeneous_head_grouping', 'native_ckks_linear',
                'client_decrypt_reencrypt', 'mpc_linear_surrogate'):
        checks['disabled_' + key] = all(m.get(key) == '0' for m in metas)
    for party in (1, 2):
        path = run / f'head_plan_party{party}.csv'
        with path.open() if path.is_file() else open('/dev/null') as stream:
            rows = list(csv.DictReader(stream))
        checks[f'all_144_original_softmax_heads_party{party}'] = (
            len(rows) == 144 and all(
                row.get('version') == 'B' and row.get('protocol') == 'softmax_mpc' and
                row.get('layer') == str(i // 12) and
                row.get('physical_head') == row.get('original_head') == str(i % 12) and
                float(row.get('alpha', 'nan')) == 1.0 for i, row in enumerate(rows)))
    # Reviewed evidence must bind the exact executed binary. It is deliberately
    # separate from a run's numeric acceptance, and is never auto-generated PASS.
    review = read_json('strict_protocol_review.json')
    checks['review_completed'] = review.get('conclusion') == 'PASS_STRICT_BLB_VIT'
    checks['all_C_removed'] = review.get('active_C_count') == 0
    checks['fusion_all_edges_reviewed'] = review.get('fine_grain_fusion_all_edges') is True
    checks['algorithm2_sharing_rounding_moduli_reviewed'] = review.get('algorithm2_verified') is True
    binary = ROOT / 'build/container-gpu/bin/ours_vit_b16_main'
    digest = hashlib.sha256(binary.read_bytes()).hexdigest() if binary.is_file() else None
    selection_path = run / 'gpu_selection.txt'
    selection = dict(line.split('=', 1) for line in selection_path.read_text().splitlines()
                     if '=' in line) if selection_path.is_file() else {}
    checks['review_binary_binding'] = bool(digest and review.get('binary_sha256') == digest and
                                           selection.get('binary_override_sha256') == digest)
    environment_path = run / 'environment.txt'
    environment = dict(line.split('=', 1) for line in environment_path.read_text().splitlines()
                       if '=' in line) if environment_path.is_file() else {}
    checks['startup_binary_binding'] = bool(digest and environment.get('binary_sha256') == digest)
    assets = review.get('asset_sha256', {})
    checks['review_asset_binding'] = all(
        bool(assets.get(key)) and assets[key] == environment.get(key + '_sha256')
        for key in ('model', 'input'))
    phantom = ROOT / 'build/container-gpu/extern/phantom/lib/libPhantom.so'
    checks['review_phantom_binding'] = bool(phantom.is_file() and
        review.get('phantom_sha256') == hashlib.sha256(phantom.read_bytes()).hexdigest())
    # The path may be rebuilt while a process is alive. Bind to the two
    # /proc/<pid>/exe observations made during this reviewed run as well.
    process_identity = run.parent / 'process_binary_sha256.txt'
    process_digests = [line.split()[0] for line in process_identity.read_text().splitlines()
                       if line.strip()] if process_identity.is_file() else []
    if process_identity.is_file():
        checks['executing_process_binary_binding'] = (
            len(process_digests) == 2 and all(value == digest for value in process_digests))
    sources = review.get('reviewed_source_sha256', {})
    checks['review_source_binding'] = REQUIRED_REVIEW_SOURCES <= sources.keys() and all(
        (ROOT / name).is_file() and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value
        for name, value in sources.items())
    allowed = all(checks.values())
    report = dict(criterion='strict_blb_vit', freeze_allowed=allowed,
                  conclusion='PASS_STRICT_BLB_VIT' if allowed else 'NOT_ESTABLISHED',
                  checks=checks, failed_checks=[k for k, v in checks.items() if not v],
                  paper_absolute_latency_required=False)
    (output / 'strict_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
