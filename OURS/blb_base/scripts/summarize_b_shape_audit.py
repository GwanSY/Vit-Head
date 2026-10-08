"""汇总 BERT/ViT 只计数副本；不把供体单层外推当完整网络实测。"""
import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    'rotations': ('rotate_vector', 'rotate_vector_inplace'),
    'ct_ct_multiply_including_square': ('multiply', 'multiply_inplace', 'square', 'square_inplace'),
    'ct_pt_multiply': ('multiply_plain', 'multiply_plain_inplace'),
    'rescale': ('rescale_to_next', 'rescale_to_next_inplace'),
    'relinearize': ('relinearize_inplace',),
    'conjugate': ('complex_conjugate', 'complex_conjugate_inplace'),
}


def counts(log):
    result = {}
    for stage, op, n in re.findall(r'BLB_COUNT stage=(\S+) op=(\S+) count=(\d+)', log):
        result.setdefault(stage, Counter())[op] += int(n)
    if not result:
        raise ValueError('missing completed BLB_COUNT output')
    return result


def aggregate(stages, predicate=lambda s: True):
    c = Counter()
    for stage, values in stages.items():
        if predicate(stage):
            c.update(values)
    return {key: sum(c[op] for op in ops) for key, ops in GROUPS.items()}


def boundaries(log):
    result = []
    pending = None
    for line in log.splitlines():
        match = re.match(r'in (mpc_to_ckks|ckks_to_mpc), (.*)', line)
        if match:
            pending = {'direction': match[1], 'parameters': match[2]}
        match = re.match(r'pts.size\(\):(\d+)', line)
        if pending is not None and match:
            pending['ciphertexts'] = int(match[1])
            result.append(pending)
            pending = None
    return result


def csv_write(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/shape_audit_20260927')
    args = parser.parse_args()
    out = args.output.resolve()
    bert_path = out/'bert_count/run/alice.log'
    vit_path = out/'vit_count_before_r02/run/alice.log'
    bert_log, vit_log = bert_path.read_text(), vit_path.read_text()
    b, v = counts(bert_log), counts(vit_log)
    # ViT prefix additionally includes patch embedding, final LN and classifier.
    # Compare the five fused block roots; retain all API counts separately.
    roots = ('layer_norm_part2_attention', 'softmax_he',
             'Softmax_v_linear_layer_norm_part1',
             'ln_part2_linear3_gelu_part1', 'linear4_add_layernorm_part1')
    fused = lambda s: len(s.split('/')) > 1 and s.split('/')[1] in roots
    specs = [
        ('five_fused_blocks', fused),
        ('QKV_BSGS', lambda s: s.endswith('/bert_cipher_plain_bsgs')),
        ('QK', lambda s: s.endswith('/bert_cipher_cipher_cross_packing')),
        ('Softmax_exp', lambda s: '/exp_he' in s),
        ('Softmax_final_HE_product', lambda s: s.endswith('/element_wise_mul')),
        ('AV_diagonal_layout', lambda s: s.endswith('/preprocess_softmax_s1_ct_ct')),
        ('AV_product', lambda s: s.endswith('/softmax_v')),
        ('WO_BSGS', lambda s: s.startswith('block/Softmax_v_linear_layer_norm_part1/') and s.endswith('/bert_cipher_plain_bsgs_2')),
        ('FFN1_BSGS', lambda s: s.startswith('block/ln_part2_linear3_gelu_part1/') and s.endswith('/bert_cipher_plain_bsgs_2')),
        ('GELU_polynomial', lambda s: s.endswith('/eval_gelu_poly')),
        ('FFN2_BSGS', lambda s: s.startswith('block/linear4_add_layernorm_part1/') and s.endswith('/bert_cipher_plain_bsgs_2')),
    ]
    rows = []
    for label, pred in specs:
        bc, vc = aggregate(b, pred), aggregate(v, pred)
        for metric in GROUPS:
            rows.append(dict(stage=label, metric=metric, bert128=bc[metric], vit256=vc[metric],
                             ratio=vc[metric]/bc[metric] if bc[metric] else '',
                             note='donor omits masks/broadcast; ratio undefined' if label == 'AV_diagonal_layout' else 'source HE API calls; not CUDA launches'))
    csv_write(out/'operator_counts.csv', rows)
    raw = [dict(model=model, stage=s, operation=op, count=n)
           for model, data in [('BERT128', b), ('ViT256_prefix', v)]
           for s, values in data.items() for op, n in values.items()]
    csv_write(out/'all_api_counts.csv', raw)
    bt, vt = boundaries(bert_log), boundaries(vit_log)
    frozen_dir = ROOT/'outputs/strict_blb_20260922/strict_full_r02/native'
    frozen = json.loads((frozen_dir/'validation.json').read_text())
    full_log = (frozen_dir/'alice.log').read_text()
    ft = boundaries(full_log)
    # Independent runtime totals catch duplicate pts.size() lines in the parser.
    for direction in ('mpc_to_ckks', 'ckks_to_mpc'):
        assert sum(x['ciphertexts'] for x in ft if x['direction'] == direction) == frozen['totals'][0][direction]
    trace = [dict(model=model, boundary=i, **row)
             for model, data in [('BERT128_block', bt), ('ViT256_prefix', vt), ('ViT256_frozen12', ft)]
             for i, row in enumerate(data)]
    csv_write(out/'conversion_boundaries.csv', trace)
    conversion_rows = []
    for direction in ('mpc_to_ckks', 'ckks_to_mpc'):
        base = [x for x in bt if x['direction'] == direction]
        full = [x for x in ft if x['direction'] == direction]
        for metric, bf, vf in [('boundary_calls', len(base), len(full)),
                              ('ciphertexts', sum(x['ciphertexts'] for x in base), sum(x['ciphertexts'] for x in full))]:
            conversion_rows.append(dict(direction=direction, metric=metric, bert_one_block=bf,
                bert_times12_extrapolated=bf*12, vit_measured12=vf, ratio=vf/(bf*12)))
    csv_write(out/'conversion_counts.csv', conversion_rows)
    historical_path = ROOT.parent.parent/'BLB/outputs/paper_bert_base_repro_20260825/summary.json'
    historical = json.loads(historical_path.read_text())['gpu']
    timing = {}
    for key in ('total time', 'he_compute_time', 'conversion_time', 'other_nonlinear_time', 'ct_transfer_time'):
        timing[key] = float(re.search(r'^'+re.escape(key)+r':([0-9.e+]+)', bert_log, re.M)[1])
    with (vit_path.parent/'block_profile_party1.csv').open() as f:
        vp = next(csv.DictReader(f))
    paired = []
    for name, bk, vk in [
        ('server_block_wall', 'total time', 'block_online_time_ms'),
        ('HE_bucket', 'he_compute_time', 'he_time_ms'),
        ('conversion_bucket', 'conversion_time', 'conversion_time_ms'),
    ]:
        bs, vs = timing[bk], float(vp[vk])/1000
        paired.append(dict(metric=name, bert128_block_s=bs, vit256_block_s=vs,
                           ratio=vs/bs, note='same GPU0/1 netem; instrumented; donor synthetic vs real checkpoint'))
    nonlinear = sum(float(vp[k]) for k in ('nonlinear_softmax_bucket_ms', 'nonlinear_layernorm_bucket_ms', 'nonlinear_gelu_core_ms'))/1000
    paired.append(dict(metric='nonlinear_wall_bucket', bert128_block_s=timing['other_nonlinear_time'],
                       vit256_block_s=nonlinear, ratio=nonlinear/timing['other_nonlinear_time'],
                       note='bucket scopes differ; ViT overlaps HE/conversion; not exclusive MPC time'))
    csv_write(out/'paired_block_timing.csv', paired)
    safety = {phase: float(ms) for phase, ms in re.findall(r'EXP_SAFETY_PHASE phase=(\w+) ms=([0-9.]+)', vit_log)}
    communication = []
    for party, name in [(1, 'alice'), (2, 'bob')]:
        log = (bert_path.parent/(name+'.log')).read_text()
        buckets = {key: float(re.search(r'^'+key+r':([0-9.e+]+)', log, re.M)[1])
                   for key in ('ct_transfer_comm', 'conversion_comm', 'other_nonlinear_comm')}
        communication.append(dict(model='BERT128_current_probe', party=party,
            scope='one block; rounded printed buckets, not unified wire-byte counter', **buckets))
    result = dict(
        status='NEEDS_AUDIT',
        performance_status='NEEDS_AUDIT',
        final_performance_baseline_frozen=False,
        scope='Strict B algorithm unchanged; cloud/ALICE executed source HE API calls (including zero-step calls), not CUDA kernel launches or backend key-switch count',
        ciphertext_count_scope='conversion payload counts and matrix packing geometry; not unique lifetime allocations or peak live ciphertext count',
        historical_bert=dict(**historical, actual_scope='one synthetic block; server timing times12; communication corrected estimate',
                             exact_extrapolated_seconds=historical['seconds_per_block']*12,
                             full12_checkpoint_validation=False),
        current_bert_counter_probe=dict(timing_seconds=timing, numerical_validation=False,
                                        reason='donor timing fixture contains placeholder masks and omitted reductions'),
        paired_block_timing=paired,
        current_bert_communication_printed_bytes=communication,
        communication_warning='Historical3.072GB is corrected extrapolation; current donor buckets and ViT complete per-party online byte counters differ in scope, no direct7x claim',
        frozen_vit_validation=frozen,
        comparable_five_fused_blocks=dict(bert=aggregate(b, fused), vit=aggregate(v, fused)),
        serial_exp_safety_before_ms=safety,
        findings=[
            dict(id='serial_exp_checks', classification='implementation_regression',
                 evidence='runtime/ckks_bert_basic.cpp exp_he used fpmath[0] over 12*197*197 elements',
                 remedy='same LT/MSB/mux partitioned across existing SCI channels, public fixed chunks'),
            dict(id='duplicate_negative_clamp', classification='implementation_regression',
                 proof='C(z)=max(z,0) on signed ring words; exp_he ends with C, caller immediately applies C; C(C(z))=C(z), no intervening HE operation',
                 remedy='retain exp_he mandatory clamp, remove caller duplicate; no secret is opened'),
            dict(id='QK_rotations', classification='donor_fixture_not_equal_work',
                 evidence='donor linear_ckks.cpp:1187 fake masks; :1349 feature reduction commented out; runtime complete feature reduction',
                 remedy='retain complete QK; restoring omitted math would invalidate checkpoint inference'),
            dict(id='AV_layout_and_rotations', classification='donor_fixture_not_equal_work',
                 evidence='donor linear_ckks.cpp:2233 preprocess_softmax_s1_ct_ct omits masks and diagonal broadcasts; softmax_v caps rotations',
                 remedy='retain complete AV masks/broadcasts and all diagonals'),
            dict(id='donor_V_and_GELU_inputs', classification='donor_fixture_not_equal_work',
                 evidence='donor ckks_bert.cpp Softmax_v_linear_layer_norm_part1 encrypts fake_vec(slot,1.0) as V; ln_part2_linear3_gelu_part1 constructs x_array from constant1; eval_gelu_poly evaluates one polynomial',
                 note='Strict uses real V, real pre-GELU x and paper F0/F1; their conversions cannot be deleted to match fixture counts'),
            dict(id='donor_softmax_row_width', classification='donor_fixture_not_equal_work',
                 evidence='donor ckks_bert_basic.cpp:1574 array_size=dim/nthreads with dim=12*128 and nthreads=48 gives32, not128',
                 note='row max/sum not equivalent; exponent still processes full196608 elements, do not incorrectly divide all Softmax work by4'),
            dict(id='packing_geometry', classification='expected_public_shape_change',
                 evidence='N=32768 unchanged; 16384 slots. QKV input CT 6->12, QK head-pair CT 2->8 (12->48 total); no >4x CT cliff',
                 note='BSGS QKV/WO/FFN1 ct-pt can scale4x because both input CT and diagonal tiling double; FFN2 scales2x'),
            dict(id='context_precision', classification='previous_strict_correctness_adaptation_retained',
                 evidence='same N for corresponding contexts; he1/he2 one extra data prime, he4 prime34->60, fft_scale20->40; tc128 retained',
                 note='not a shape-triggered polynomial degree jump; extra levels/precision affect each operation cost'),
            dict(id='conversion_work', classification='previous_strict_correctness_adaptation_retained',
                 evidence='full signed range, guard16 integer FFT, coefficient extension and per-RNS reduction retained; donor zero-mask/degree shortcuts not restored',
                 note='donor exp_he uses member poly_modulus_degree8192 in C2M despite he_softmax N32768 (probe line331); Strict uses actual32768. CT count alone does not measure converted scalar width, local FFT or MPC cost'),
            dict(id='pre_norm_boundaries', classification='model_graph_adaptation',
                 evidence='first LN standalone; AV->WO->residual->LN2 and FFN2->residual->next LN1 remain HE fused',
                 note='steady block M2C9->12: actual V and attention residual (+2), FFN residual (+1). C2M8->12: QK/V separated by scale (+1), two pre-norm skip exports (+2), real pre-GELU x (+1). Full run also includes initial LN/patch/final LN/classifier'),
            dict(id='patch_and_context_lifetime', classification='no_repeat_found',
                 evidence='ckks_vit_b16_main.cpp patch at1197 precedes layer loop1394; Linear_ckks::set_HE only initialization',
                 note='online H2D/D2H subsecond in frozen full run; weight device encoding still per layer (~99.59s), independent cache experiment not enabled'),
            dict(id='legacy_count_undercoverage', classification='measurement_limitation',
                 evidence='legacy QK rescale reports0, actual API probe nonzero; Softmax/GELU multiply counters incomplete',
                 remedy='use all_api_counts.csv, do not compare legacy zero counters as actual operation absence'),
        ],
        invariants=['all144heads Softmax, original identities', 'fixed checkpoint and 197/256 token geometry',
                    'ring37 scale12; CKKS chains unchanged by this repair', 'GELU F0/F1 and evaluation graph unchanged',
                    'Softmax divisor64 six squares cutoff-13 full-ring reciprocal finalHEproduct unchanged',
                    'Algorithm2 signed conversion unchanged', 'no allocator/selective conversion/new packing'],
        evidence_sha256={os.path.relpath(p, ROOT): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (bert_path, vit_path, historical_path, frozen_dir/'validation.json')},
    )
    candidate = out/'strict_repair_full/native/validation.json'
    if candidate.exists():
        result['repair_full_validation'] = json.loads(candidate.read_text())
        repaired = result['repair_full_validation']
        old_totals, new_totals = frozen['totals'][0], repaired['totals'][0]
        compared = []
        for metric in ('secure_end_to_end_s', 'he_instrumented_s', 'conversion_instrumented_s',
                       'mpc_nonlinear_bucket_s', 'plaintext_device_encode_instrumented_s',
                       'he_ct_pt_multiply', 'he_ct_ct_multiply', 'he_rotations', 'he_rescales',
                       'ckks_to_mpc', 'mpc_to_ckks', 'field_to_ring_calls', 'ring_to_field_calls'):
            compared.append(dict(metric=metric, before=old_totals[metric], after=new_totals[metric],
                note='cloud party1; historical GPUs2/5 vs current0/1; timing buckets overlap; legacy HE counters incomplete'))
        compared.append(dict(metric='aggregate_online_sent_bytes',
            before=frozen['communication']['online_communication_bytes']['aggregate_bytes'],
            after=repaired['communication']['online_communication_bytes']['aggregate_bytes'],
            note='sum of both party sent bytes'))
        csv_write(out/'repair_full_comparison.csv', compared)
        result['historical_comparison_caveat'] = 'single runs on different physical GPU pairs; not repeated paired speedup estimate'
        log = (candidate.parent/'alice.log').read_text()
        phases = {}
        for phase, workers, ms in re.findall(r'EXP_SAFETY_PHASE phase=(\w+) workers=(\d+) ms=([0-9.]+)', log):
            phases.setdefault(phase, []).append(float(ms))
        result['repair_safety_ms_by_layer'] = phases
        result['repair_correctness_status'] = 'PASS' if result['repair_full_validation']['acceptance_pass'] else 'FAIL'
        result['status'] = 'REGRESSION_FIXED_FURTHER_PERFORMANCE_AUDIT_OPEN' if result['repair_correctness_status'] == 'PASS' else 'REPAIR_FAILED'
    (out/'audit_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(out/'audit_summary.json')


if __name__ == '__main__':
    main()
