#!/usr/bin/env python3
"""Read-only B cache sizing from executed packing counts and CKKS levels.

Sizes are uncompressed payload estimates, not allocated memory or measured
cache speedups. No weights, ciphertexts, keys, or baseline files are modified.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def fields(line):
    return dict(word.split('=', 1) for word in line.split() if '=' in word)


def audit(run):
    metadata = dict(line.split('=', 1) for line in
                    (run / 'run_metadata_party1.txt').read_text().splitlines() if '=' in line)
    if metadata.get('version') != 'B' or metadata.get('executed_transformer_blocks') != '12':
        raise ValueError('requires a completed 12-block B run')
    validation = json.loads((run / 'validation.json').read_text())
    if not validation.get('acceptance_pass'):
        raise ValueError('source B must pass numerical and execution validation')
    materialized, records = {}, []
    log = run / 'alice.log'
    for line in log.read_text().splitlines():
        if line.startswith('CKKS_PLAIN_MATERIALIZE '):
            row = fields(line)
            materialized[row['op']] = row
        elif line.startswith('CKKS_DEVICE_PLAINTEXTS '):
            row = fields(line)
            records.append({**materialized[row['op']], **row})
    if len(records) != 50:
        raise ValueError('expected embedding + 12*(QKV,WO,FFN1,FFN2) + classifier')
    stages = [('QKV', 'he1', 'QKV'), ('Output_Projection', 'he2', 'BSGS2'),
              ('FFN1', 'he3', 'BSGS2'), ('FFN2', 'he4', 'BSGS2')]
    encoding = {stage: 0.0 for stage, _, _ in stages}
    with (run / 'block_online_profile_party1.csv').open() as handle:
        for row in csv.DictReader(handle):
            if row['stage'] in encoding:
                encoding[row['stage']] += float(row['plaintext_device_encode_ms']) / 1000
    result = []
    for offset, (stage, context, op) in enumerate(stages):
        degree = int(metadata[context + '_poly_modulus_degree'])
        weights_bytes = biases_bytes = slot_bytes = polynomial_bytes = 0
        for layer in range(12):
            row = records[1 + 4 * layer + offset]
            if row['op'] != op:
                raise ValueError('unexpected operator order')
            # Logged chain is SEAL's index, where chain+1 is active RNS limbs.
            limbs = int(row['input_chain']) + 1
            bias_limbs = int(row.get('output_chain', row['input_chain'])) + 1
            weights, biases = int(row['weight_encodes']), int(row['bias_encodes'])
            weights_bytes += weights * degree * limbs * 8
            biases_bytes += biases * degree * bias_limbs * 8
            slot_bytes += weights * (degree // 2) * 8
            polynomial_bytes += weights * degree * 8
        result.append(dict(
            stage=stage, encoding_instrumented_s=encoding[stage],
            weight_rns_ntt_bytes=weights_bytes, bias_rns_ntt_bytes=biases_bytes,
            per_layer_weight_GiB=weights_bytes / 12 / 2**30,
            all_layers_weight_GiB=weights_bytes / 2**30,
            semantic_double_slot_GiB=slot_bytes / 2**30,
            candidate_i64_coefficient_GiB=polynomial_bytes / 2**30,
            transfer_GBps_to_match_encode_bucket=weights_bytes / 1e9 / encoding[stage],
        ))
    return dict(
        source_run=str(run.resolve()), source_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        baseline_online_s=validation['online_s'], operators=result,
        weight_payload_GiB=sum(row['all_layers_weight_GiB'] for row in result),
        bias_payload_GiB=sum(row['bias_rns_ntt_bytes'] for row in result) / 2**30,
        semantic_slots_GiB=sum(row['semantic_double_slot_GiB'] for row in result),
        candidate_i64_coefficients_GiB=sum(row['candidate_i64_coefficient_GiB'] for row in result),
        scope='12 transformer blocks only; excludes embedding, classifier, keys, activations, allocator overhead',
        i64_status='candidate representation only; requires range and rounding equivalence proof; biases may exceed int64',
        timing_status='encoding buckets are not removable exclusive wall time; transfer rates are arithmetic references, not measured break-even rates',
        implementation_status='feasibility audit only; cache not implemented or benchmarked',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'weight_cache_feasibility.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'weight_cache_sizes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result['operators'][0]))
        writer.writeheader()
        writer.writerows(result['operators'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
