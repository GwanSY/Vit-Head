"""为真实 BERT 单层构建与 Strict ViT 共用内核的独立计量产物。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BLB = Path(__file__).resolve().parents[1]
OURS = BLB.parent / 'OURS/blb_base'
sys.path.insert(0, str(OURS / 'scripts'))
from prepare_b_wire_audit import instrument


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    # Existing build/link metadata use the /workspace/OURS mount.
    out.relative_to(OURS / 'outputs')
    subprocess.run([sys.executable, '-B', str(OURS / 'scripts/prepare_b_wire_audit.py'),
                    '--vit', '--output', str(out)], check=True)
    driver = BLB / 'bert_base/real_bert_layer.cpp'
    compiled_driver = instrument(driver.read_text(), 'real_bert_layer.cpp')
    (out / 'source/ckks_vit_b16_main.cpp').write_text(compiled_driver)
    wire_manifest_path = out / 'wire_source_manifest.json'
    wire_manifest = json.loads(wire_manifest_path.read_text())
    wire_manifest['ckks_vit_b16_main.cpp'] = {
        'original_path': str(driver.relative_to(BLB)),
        'original_sha256': hashlib.sha256(driver.read_bytes()).hexdigest(),
        'instrumented_sha256': hashlib.sha256(compiled_driver.encode()).hexdigest(),
    }
    wire_manifest_path.write_text(json.dumps(wire_manifest, indent=2) + '\n')
    manifest = {
        'entry': str(driver.relative_to(BLB)),
        'entry_sha256': hashlib.sha256(driver.read_bytes()).hexdigest(),
        'runtime': 'OURS/blb_base/ours_vit_b16/runtime',
        'model_graph': 'BERT post-norm layer 0, 128 tokens',
        'algorithm': 'Strict BLB; same M2C and instrumentation as ViT B',
        'heterogeneous_heads': False,
        'allocator': False,
        'new_packing': False,
        'numeric_acceptance': 'PENDING',
        'uses_backup': False,
    }
    (out / 'real_bert_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
