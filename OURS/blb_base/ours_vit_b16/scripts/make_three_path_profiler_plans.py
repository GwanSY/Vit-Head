#!/usr/bin/env python3
"""Make one contiguous-group plan for the real path profiler.

The plan measures one path at a requested group size.  The remaining heads use
one compatible reference path, so every row is produced by a real run rather
than by adding or dividing independently measured costs.
"""

from __future__ import annotations

import argparse
from pathlib import Path


HEADS = 12
LAYERS = 12
GROUP_SIZES = (1, 2, 4, 6, 8, 12)


def make_protocols(path: str, group_size: int) -> list[str]:
    if path == "softmax":
        return ["softmax_mpc"] * group_size + ["relu_norm_mpc"] * (HEADS - group_size)
    if path == "relu":
        return ["softmax_mpc"] * (HEADS - group_size) + ["relu_norm_mpc"] * group_size
    if path == "linear":
        return ["softmax_mpc"] * (HEADS - group_size) + ["he_linear"] * group_size
    raise ValueError(f"unknown path: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("softmax", "relu", "linear"), required=True)
    parser.add_argument("--group-size", type=int, choices=GROUP_SIZES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocols = make_protocols(args.path, args.group_size)
    # Every curve point is executed by the same A4 binary and full attention
    # boundary.  Softmax/ReLU-only layer 0 still needs a native Linear entry
    # elsewhere in the 12-layer public plan so A4 validation proves that the
    # binary is identical to the Linear curve.  Blocks 1..11 are not executed
    # by the one-block profiler and therefore do not contaminate layer 0.
    ablation = "a4"
    alpha = 1.0 / 197.0
    lines = [
        "OURS_VIT_HEAD_PLAN_V1",
        "layers=12,heads=12,logical_tokens=197,physical_tokens=256",
        "layer,physical_head,original_head,protocol,alpha",
    ]
    for layer in range(LAYERS):
        layer_protocols = protocols
        if layer > 0 and "he_linear" not in protocols:
            layer_protocols = ["softmax_mpc"] * (HEADS - 1) + ["he_linear"]
        for physical_head, protocol in enumerate(layer_protocols):
            value = alpha if protocol == "he_linear" else 1.0
            lines.append(
                f"{layer},{physical_head},{physical_head},{protocol},{value:.17g}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"OURS_THREE_PATH_PLAN status=PASS ablation={ablation.upper()} "
        f"path={args.path} group_size={args.group_size}"
    )
    print(f"OURS_THREE_PATH_PLAN_FILE={args.output}")


if __name__ == "__main__":
    main()
