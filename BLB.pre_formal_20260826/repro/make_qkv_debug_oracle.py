#!/usr/bin/env python3
"""Create temporary Q/K/V comparison shares in the CKKS cross-packing order."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
import struct
from pathlib import Path

import numpy as np

from prepare_bert_oracle import HEADS, MODULUS, SEQ, load_layer0


MAGIC = b"BLB_ORACLE_SHARE_V1\0"
MOD_BITS = 43
SCALE_BITS = 13


def write_share(path: Path, values: np.ndarray) -> str:
    encoded = np.asarray(values, dtype=np.uint64).reshape(-1)
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<Qii", encoded.size, MOD_BITS, SCALE_BITS))
        handle.write(encoded.tobytes(order="C"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("base_oracle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    shutil.copytree(args.base_oracle, args.output)
    layer = load_layer0(args.model_dir)
    x_qkv = np.floor_divide(layer["input"], 1 << 7).astype(np.int64)
    tensors = {
        "debug_qkv_q": np.stack([x_qkv @ layer["q"][h] + layer["bq"][h]
                                  for h in range(HEADS)]),
        "debug_qkv_k": np.stack([x_qkv @ layer["k"][h] + layer["bk"][h]
                                  for h in range(HEADS)]),
        "debug_qkv_v": np.stack([x_qkv @ layer["v"][h] + layer["bv"][h]
                                  for h in range(HEADS)]),
    }

    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rng = secrets.SystemRandom()
    for name, tensor in tensors.items():
        # The official BERT path uses col_packing=false here and emits
        # [head, token, channel], which is also the fixed-clear order. Move
        # the s=11 QKV value to the diagnostic s=13 domain.
        values = (tensor * 4).reshape(-1)
        alice = np.fromiter(
            (rng.randrange(MODULUS) for _ in range(values.size)),
            dtype=np.uint64,
            count=values.size,
        )
        bob = np.asarray(
            [(int(value) - int(mask)) % MODULUS
             for value, mask in zip(values, alice)],
            dtype=np.uint64,
        )
        alice_path = args.output / f"{name}.alice.bin"
        bob_path = args.output / f"{name}.bob.bin"
        alice_sha = write_share(alice_path, alice)
        bob_sha = write_share(bob_path, bob)
        manifest["stages"][name] = {
            "size": int(values.size),
            "alice": alice_path.name,
            "bob": bob_path.name,
            "alice_sha256": alice_sha,
            "bob_sha256": bob_sha,
            "A_B_MSE": 0.0,
        }
    manifest["qkv_diagnostic"] = {
        "domain": "QKV raw s=11 shifted to s=13",
        "layout": "hidden,token from cross-packing col output",
        "sequence": SEQ,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"QKV_ORACLE status=PASS output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
