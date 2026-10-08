#!/usr/bin/env python3
"""Audit stage costs without treating overlapping buckets as wall time.

Use --before/--after native output directories from the same public workload.
Raw logs remain untouched. One pair is a diagnostic, not a speedup estimate.
"""
import argparse
import csv
import json
import re
from pathlib import Path


COUNTERS = ('he_rotations', 'he_ct_pt_multiply', 'he_ct_ct_multiply',
            'he_rescales', 'conversion_ciphertexts', 'plaintext_device_encode_ms')


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def public_conditions(root):
    values = {}
    for name in ('environment.txt', 'gpu_selection.txt'):
        for line in (root / name).read_text().splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                values[key] = value
    keys = ('model_sha256', 'input_sha256', 'logical_tokens', 'physical_tokens',
            'version', 'network', 'omp_threads_per_party', 'mpc_workers_per_party',
            'alice_cpus', 'bob_cpus', 'physical_gpu_uuids', 'party_topology',
            'image_id', 'gpu_direct_plaintext', 'gpu_linear_streams')
    if any(not values.get(key) for key in keys):
        raise ValueError('missing comparison provenance')
    return {key: values[key] for key in keys}


def numerical_conditions(root):
    values = dict(line.split('=', 1) for line in
                  (root / 'run_metadata_party1.txt').read_text().splitlines() if '=' in line)
    fields = {'fixed_point_ring_bits', 'fixed_point_scale_bits', 'conversion_path',
              'softmax_product', 'gelu_polynomial', 'gelu_conversion_scale_bits',
              'secure_exp_clamp', 'softmax_exp_polynomial'}
    fields.update(key for key in values if re.match(r'he(?:[1-4]|_softmax|_onetime)_', key))
    if not fields.issubset(values) or len(fields) < 32:
        raise ValueError('missing numerical parameter provenance')
    return {key: values[key] for key in sorted(fields)}


def read_run(root):
    blocks = read_csv(root / 'block_profile_party1.csv')
    stages = [read_csv(root / f'block_online_profile_party{party}.csv')
              for party in (1, 2)]
    stages = [{(r['block'], r['stage']): r for r in rows
               if not r['stage'].startswith('BLOCK_ONLINE_')} for rows in stages]
    if set(stages[0]) != set(stages[1]):
        raise ValueError('party stage coverage differs')
    # Cloud executes the HE operations. Do not add duplicate logical counters
    # across parties. Missing historical counters remain null, never zero.
    rows = {}
    for key, cloud in stages[0].items():
        edge = stages[1][key]
        rows[key] = dict(cloud_wall_ms=float(cloud['wall_ms']),
                         edge_wall_ms=float(edge['wall_ms']),
                         max_party_wall_ms=max(float(cloud['wall_ms']), float(edge['wall_ms'])),
                         sent_bytes=int(cloud['send_bytes']) + int(edge['send_bytes']))
        rows[key].update({c: (float(cloud[c]) if c.endswith('_ms') else int(cloud[c]))
                          if cloud.get(c) not in (None, '') else None
                          for c in COUNTERS})
    totals_path = root / 'execution_totals_party1.csv'
    totals = None
    if totals_path.exists():
        parties = [{r['metric']: float(r['value']) for r in read_csv(
            root / f'execution_totals_party{p}.csv')} for p in (1, 2)]
        totals = {'online_max_party_s': max(p['secure_end_to_end_s'] for p in parties)}
        totals.update({name: sum(p[name] for p in parties) for name in (
            'online_communication_bytes', 'total_communication_bytes')})
    # Every exclusive stage must account for exactly one share of block work.
    if rows and all(r['he_rotations'] is not None for r in rows.values()):
        for block in blocks:
            for name in COUNTERS[:-1]:
                actual = sum(r[name] for (layer, _), r in rows.items() if layer == block['block'])
                if actual != int(block[name]):
                    raise ValueError(f'stage/block counter mismatch: block={block["block"]} {name}')
    return rows, blocks, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old_conditions = public_conditions(args.before)
    new_conditions = public_conditions(args.after)
    if old_conditions != new_conditions:
        changed = [k for k in old_conditions if old_conditions[k] != new_conditions[k]]
        raise ValueError(f'comparison conditions differ: {changed}')
    parameters = numerical_conditions(args.before)
    if parameters != numerical_conditions(args.after):
        raise ValueError('comparison changes precision, approximation, or HE parameters')
    before, old_blocks, old_totals = read_run(args.before)
    after, new_blocks, new_totals = read_run(args.after)
    if set(before) != set(after):
        raise ValueError('before/after must cover identical layers and stages')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in sorted(before, key=lambda key: (int(key[0]), key[1])):
        rows.append(dict(block=key[0], stage=key[1],
                         **{f'before_{k}': v for k, v in before[key].items()},
                         **{f'after_{k}': v for k, v in after[key].items()},
                         delta_max_party_wall_ms=after[key]['max_party_wall_ms'] - before[key]['max_party_wall_ms']))
    with (args.output / 'stages.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(before=str(args.before.resolve()), after=str(args.after.resolve()),
                  comparison_status='single_pair_diagnostic_no_statistical_speedup_claim',
                  matched_public_conditions=new_conditions,
                  matched_numerical_conditions=parameters,
                  before_totals=old_totals, after_totals=new_totals,
                  block_counter_reconciliation='PASS' if after and all(
                      r['he_rotations'] is not None for r in after.values()) else 'UNAVAILABLE',
                  blocks=len(new_blocks),
                  before_block_rotations=sum(int(r['he_rotations']) for r in old_blocks),
                  after_block_rotations=sum(int(r['he_rotations']) for r in new_blocks),
                  timing_caveat='stage maxima and instrumented buckets must not be summed into end-to-end latency')
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
