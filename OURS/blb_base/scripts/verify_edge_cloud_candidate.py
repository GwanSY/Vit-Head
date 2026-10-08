#!/usr/bin/env python3
"""CPU-only CLI checks of a linked OURS edge/cloud candidate; no secure execution."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    all_softmax = root / 'ours_vit_b16/experiments/all_softmax_head_plan.txt'
    plans = {version: None if version == 'B' else all_softmax
             for version in ('B', 'H', 'HA', 'HP', 'HAP', 'J', 'BP')}
    checks = []
    for version, plan in plans.items():
        contracts = []
        for role in ('cloud', 'edge'):
            destination = output / (version + '_' + role)
            command = [str(args.binary.resolve()), '--role', role, '--version', version,
                       '--output', str(destination), '--validate-plan-only']
            if plan is not None:
                command.extend(['--head-plan', str(plan)])
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    universal_newlines=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(result.stdout)
            party = '1' if role == 'cloud' else '2'
            contracts.append((destination / ('edge_cloud_contract_party' + party + '.txt')).read_bytes())
            (output / (version + '_' + role + '.log')).write_text(result.stdout)
        if contracts[0] != contracts[1]:
            raise RuntimeError('role-specific paths changed the public contract')
        checks.append(dict(version=version, status='PASS', roles=2))
    result = subprocess.run([str(args.binary.resolve()), '--role', 'edge', '--party', '1',
                             '--validate-plan-only', '--output', str(output / 'invalid')],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if result.returncode == 0 or (output / 'invalid').exists():
        raise RuntimeError('conflicting role/party was not rejected before execution')
    report = dict(status='PASS', checks=checks, role_conflict_rejected=True,
                  secure_execution=False, gpu_used=False, latency_claimed=False)
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
