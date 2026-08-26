#!/usr/bin/env python3
"""Verify BLB-local additive oracle shares before a two-party run.

This verifier is deliberately compatible with Python 3.6 because the
official BLB image uses /usr/bin/python3.6.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path


MAGIC = b"BLB_ORACLE_SHARE_V1\0"
STAGES = (
    "fgf1_ln_attention",
    "fgf2_softmax",
    "fgf3_softmax_v_ln",
    "fgf4_ln_ffn_gelu",
    "fgf5_ffn2_ln",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_share(path: Path, expected_size: int, bits: int, scale: int) -> None:
    data = path.read_bytes()
    header_size = len(MAGIC) + struct.calcsize("<Qii")
    if len(data) != header_size + expected_size * 8:
        raise RuntimeError(f"{path}: byte size mismatch")
    if data[: len(MAGIC)] != MAGIC:
        raise RuntimeError(f"{path}: bad share magic")
    size, got_bits, got_scale = struct.unpack_from("<Qii", data, len(MAGIC))
    if size != expected_size or got_bits != bits or got_scale != scale:
        raise RuntimeError(f"{path}: share metadata mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("oracle_dir", type=Path)
    args = parser.parse_args()
    model = args.model_dir.resolve()
    oracle = args.oracle_dir.resolve()
    manifest_path = oracle / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing oracle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "blb.bert.three-oracle.v3":
        raise RuntimeError("unexpected oracle manifest schema")
    if manifest.get("arithmetic") != "bert.py.post_norm.layer0.qk_s12_v_s11.public_key_mask.v3":
        raise RuntimeError("oracle arithmetic is missing the split QK/V scale contract")
    bits = int(manifest.get("mod_bits", 0))
    scale = int(manifest.get("scale_bits", -1))
    if (manifest.get("layer") != 0 or bits != 43 or scale != 13):
        raise RuntimeError(
            "oracle manifest must use the paper BLB nonlinear ell=43/s=13 domain"
        )
    source_manifest = model / "manifest.json"
    expected_source = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    if manifest.get("source_checkpoint_manifest_sha256") != expected_source:
        raise RuntimeError("oracle source checkpoint manifest hash mismatch")
    public_mask = manifest.get("public_attention_mask")
    if not isinstance(public_mask, dict):
        raise RuntimeError("oracle public attention mask metadata is missing")
    mask_path = model / "inputs_0_mask.txt"
    if (public_mask.get("path") != mask_path.name or
            public_mask.get("sha256") != sha256(mask_path) or
            public_mask.get("broadcast") != "head,query,key <- mask[key]" or
            public_mask.get("secure_add_party") != "BOB" or
            public_mask.get("scale_bits") != scale):
        raise RuntimeError("oracle public attention mask metadata mismatch")
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        raise RuntimeError("oracle stages are missing")
    for stage in STAGES:
        entry = stages.get(stage)
        if not isinstance(entry, dict):
            raise RuntimeError(f"oracle stage is missing: {stage}")
        size = int(entry["size"])
        for role in ("alice", "bob"):
            name = entry.get(role)
            if not isinstance(name, str) or Path(name).name != name:
                raise RuntimeError(f"invalid {role} share name for {stage}")
            path = oracle / name
            check_share(path, size, bits, scale)
            if sha256(path) != entry.get(f"{role}_sha256"):
                raise RuntimeError(f"{role} share hash mismatch for {stage}")
        if float(entry.get("A_B_MSE", -1.0)) < 0.0:
            raise RuntimeError(f"invalid A_B_MSE for {stage}")
    print(f"ORACLE_VERIFY status=PASS dir={oracle} stages={len(STAGES)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ORACLE_VERIFY status=BLOCKED reason={exc}")
        raise SystemExit(2)
