"""独立明文参考与两方离线分享验收；不得用于重构生产用户的秘密输入。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'OURS/blb_base/ours_vit_b16/scripts'))
from vit_reference import fixed_softmax, fixed_gelu, fixed_layer_norm, softmax


def reference(model, fixed):
    def read(name):
        return np.loadtxt(model / name, delimiter=',')
    def param(name, scale):
        return read('bert.encoder.layer.0.' + name + '.txt') / (2 ** scale)
    def quant(x, scale=12):
        return np.floor(x * (2 ** scale)) / (2 ** scale) if fixed else x
    x = read('inputs_0_data.txt') / 4096
    z = quant(x, 5)
    qkv = []
    for name in ('query', 'key', 'value'):
        w = param('attention.self.' + name + '.weight', 6).reshape(12, 768, 64)
        b = param('attention.self.' + name + '.bias', 11).reshape(12, 64)
        qkv.append(np.einsum('td,hdf->htf', z, w) + b[:, None, :])
    q, k, v = qkv
    qk = quant(q @ k.transpose(0, 2, 1) / 8)
    v = quant(v, 11)
    scores = qk + read('inputs_0_mask.txt')[None, None, :]
    probability = fixed_softmax(scores) if fixed else softmax(scores)
    av = (probability @ v).transpose(1, 0, 2).reshape(128, 768)
    h = av @ param('attention.output.dense.weight', 6) + param('attention.output.dense.bias', 12) + x
    stages = {'qk': qk, 'softmax': probability}
    def norm(value, prefix, name):
        centered = value - value.mean(-1, keepdims=True)
        variance = (centered ** 2).mean(-1)
        stages[name + '_centered'] = quant(centered)
        stages[name + '_variance'] = quant(variance)
        w = param(prefix + '.weight', 12)
        b = param(prefix + '.bias', 12)
        if fixed:
            return fixed_layer_norm(value, w, b, 1e-12)
        return centered / np.sqrt(variance[:, None] + 1e-12) * w + b
    h = norm(h, 'attention.output.LayerNorm', 'ln1')
    pre = quant(h @ param('intermediate.dense.weight', 6) + param('intermediate.dense.bias', 11))
    gelu = fixed_gelu(pre) if fixed else 0.5 * pre * (1 + np.tanh(np.sqrt(2 / np.pi) * (pre + .044715 * pre ** 3)))
    stages.update(pre_gelu=pre, gelu=gelu)
    ff = gelu @ param('output.dense.weight', 5) + param('output.dense.bias', 9) + h
    stages['output'] = norm(ff, 'output.LayerNorm', 'ln2')
    return stages


def metrics(actual, expected):
    d = actual.reshape(-1) - expected.reshape(-1)
    return {'max_abs': float(np.max(np.abs(d))), 'mse': float(np.mean(d * d)),
            'relative_l2': float(np.linalg.norm(d) / max(np.linalg.norm(expected), 1e-12)),
            'finite': bool(np.isfinite(actual).all() and np.isfinite(expected).all())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT / 'checkpoints/official-blb-mrpc/weights_txt')
    p.add_argument('--run', type=Path)
    p.add_argument('--check-inputs', action='store_true')
    args = p.parse_args()
    manifest = json.loads((args.model / 'manifest.json').read_text())
    checked = 0
    for group in ('files', 'auxiliary_files', 'scale_files'):
        for item in manifest[group]:
            if group == 'files' and item.get('layer') != 0:
                continue
            path = (args.model / item['path']).resolve()
            path.relative_to(args.model.resolve())
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('checkpoint integrity failure: ' + item['path'])
            checked += 1
    if args.check_inputs:
        print(json.dumps({'status': 'PASS_INPUT_FILES_ONLY', 'checked_files': checked,
                          'runtime_numerical_acceptance': False}))
        return 0
    if args.run is None:
        p.error('--run is required unless --check-inputs is used')
    clear, fixed = reference(args.model, False), reference(args.model, True)
    metadata = []
    for party in (1, 2):
        with (args.run / f'stages_party{party}.csv').open() as f:
            metadata.append(list(csv.DictReader(f)))
    if metadata[0] != metadata[1]:
        raise ValueError('party stage metadata differ')
    rows = []
    for row in metadata[0]:
        name, ell, scale, n = row['stage'], int(row['ell']), int(row['scale']), int(row['elements'])
        shares = [np.fromfile(args.run / f'{name}.party{i}.bin', dtype='<u8') for i in (1, 2)]
        if any(a.size != n for a in shares) or n != fixed[name].size:
            raise ValueError('stage shape mismatch: ' + name)
        raw = (shares[0] + shares[1]) & np.uint64((1 << ell) - 1)
        signed = raw.astype(np.int64)
        signed = np.where(signed >= (1 << (ell - 1)), signed - (1 << ell), signed)
        actual = signed.astype(np.float64) / (1 << scale)
        comparison = metrics(actual, fixed[name])
        # Fixed pre-run gate, not fitted to observed results. Same 3% bound
        # at every boundary; absolute allowance is two scale-12 units.
        passed = comparison['finite'] and (comparison['relative_l2'] <= .03 or comparison['max_abs'] <= 2 / 4096)
        rows.append({'stage': name, 'fixed_secure': comparison,
                     'clear_fixed': metrics(fixed[name], clear[name]), 'pass': passed})
    present = {r['stage'] for r in rows}
    passed = present == set(fixed) and all(r['pass'] for r in rows)
    report = {'status': 'PASS' if passed else 'FAIL', 'scope': 'real BERT layer 0',
              'reference_model': 'exported quantized MRPC checkpoint; clear uses floating arithmetic',
              'fixed_secure_relative_l2_limit': .03, 'absolute_fallback': 2 / 4096,
              'checkpoint_manifest_sha256': hashlib.sha256((args.model / 'manifest.json').read_bytes()).hexdigest(),
              'rows': rows, 'production_input_reconstruction': False}
    (args.run / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
