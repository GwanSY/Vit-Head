#!/usr/bin/env python3
"""Verify every generated BLB weight file before either party starts."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    root = args.model_dir.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "blb.bert.checkpoint-manifest.v1":
        raise RuntimeError("unsupported manifest schema")
    if manifest.get("metadata", {}).get("num_hidden_layers") != 12:
        raise RuntimeError("manifest is not a 12-layer BERT-base checkpoint")
    if args.layer not in manifest.get("exported_layers", []):
        raise RuntimeError(f"layer {args.layer} was not exported")
    records = [item for item in manifest.get("files", []) if item.get("layer") == args.layer]
    if len(records) != 16:
        raise RuntimeError(f"layer {args.layer} needs 16 tensors, found {len(records)}")
    for item in records:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item.get("bytes"):
            raise RuntimeError(f"missing or size-mismatched weight: {path}")
        actual = sha256(path)
        if actual != item.get("sha256"):
            raise RuntimeError(f"SHA256 mismatch: {path}")
        if not item.get("shape") or not item.get("packing"):
            raise RuntimeError(f"shape/packing absent: {path}")
    print(
        f"MANIFEST status=PASS layer={args.layer} tensors={len(records)} "
        f"model_layers=12 formal_ready={'yes' if manifest.get('formal_ready') else 'no'}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MANIFEST status=BLOCKED reason={exc}")
        raise SystemExit(2)
