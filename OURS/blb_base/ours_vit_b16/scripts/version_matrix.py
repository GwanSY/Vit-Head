"""The public experiment matrix and immutable semantic plan binding."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path

MATRIX_PATH = Path(__file__).resolve().parents[1] / 'experiments/version_matrix.json'
MATRIX = json.loads(MATRIX_PATH.read_text())
VERSIONS = tuple(MATRIX)

def require_version(version: str) -> str:
    if version not in MATRIX:
        raise ValueError(f'version must be one of {VERSIONS}; obsolete version names are rejected')
    return version

def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def plan_binding(path: str | Path) -> dict:
    lines = [line.strip() for line in Path(path).read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    if lines[:2] != ['OURS_VIT_HEAD_PLAN_V1', 'layers=12,heads=12,logical_tokens=197,physical_tokens=256']:
        raise ValueError('invalid head plan format/dimensions')
    rows = list(csv.DictReader(lines[2:]))
    if len(rows) != 144:
        raise ValueError('head plan must contain 144 heads')
    semantic = []
    for layer in range(12):
        group = [r for r in rows if int(r['layer']) == layer]
        for key in ('physical_head', 'original_head'):
            if sorted(int(r[key]) for r in group) != list(range(12)):
                raise ValueError(f'{key} is not a permutation')
        for row in group:
            alpha = float(row['alpha'])
            if row['protocol'] not in ('softmax_mpc', 'relu_norm_mpc', 'he_linear') or not math.isfinite(alpha):
                raise ValueError('invalid protocol/alpha')
            if row['protocol'] != 'he_linear' and alpha != 1.0:
                raise ValueError('MPC heads require alpha=1')
            semantic.append([layer, int(row['original_head']), row['protocol'], alpha.hex()])
    semantic.sort()
    return {'semantic_plan_sha256': hashlib.sha256(json.dumps(semantic, separators=(',', ':')).encode()).hexdigest(),
            'physical_plan_sha256': file_hash(path)}
