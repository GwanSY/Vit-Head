#!/usr/bin/env python3
"""生成确定性的 BLB 端到端小型 ViT 模型和私有 patch 输入。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_vector(handle, name: str, value: np.ndarray) -> None:
    handle.write(f"{name}\n")
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    for start in range(0, flat.size, 8):
        handle.write(" ".join(f"{x:.12g}" for x in flat[start : start + 8]))
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    patches, patch_dim, dim = 4, 4, 8
    heads, mlp_dim, classes, scale_bits = 2, 16, 3, 12

    def weight(rows: int, cols: int, std: float = 0.08) -> np.ndarray:
        return rng.normal(0.0, std, (rows, cols))

    model = {
        "patch_w": weight(dim, dim),
        "patch_b": rng.normal(0.0, 0.02, dim),
        "pos": rng.normal(0.0, 0.03, (patches, dim)),
        "wq": weight(dim, dim),
        "bq": rng.normal(0.0, 0.01, dim),
        "wk": weight(dim, dim),
        "bk": rng.normal(0.0, 0.01, dim),
        "wv": weight(dim, dim),
        "bv": rng.normal(0.0, 0.01, dim),
        "wo": weight(dim, dim),
        "bo": rng.normal(0.0, 0.01, dim),
        "ln1_w": rng.normal(1.0, 0.02, dim),
        "ln1_b": rng.normal(0.0, 0.01, dim),
        "ff1_w": weight(mlp_dim, dim, 0.06),
        "ff1_b": rng.normal(0.0, 0.01, mlp_dim),
        "ff2_w": weight(dim, mlp_dim, 0.06),
        "ff2_b": rng.normal(0.0, 0.01, dim),
        "ln2_w": rng.normal(1.0, 0.02, dim),
        "ln2_b": rng.normal(0.0, 0.01, dim),
        "head_ln_w": rng.normal(1.0, 0.02, dim),
        "head_ln_b": rng.normal(0.0, 0.01, dim),
        "classifier_w": weight(classes, dim, 0.08),
        "classifier_b": np.asarray([0.04, -0.02, 0.01]),
    }
    # patch_dim 之后的列是公开的零填充通道，不应影响 patch embedding。
    model["patch_w"][:, patch_dim:] = 0.0
    private_input = np.clip(rng.normal(0.0, 0.7, (patches, patch_dim)), -1.5, 1.5)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "alice_model.txt").open("w", encoding="utf-8") as out:
        out.write("BLB_VIT_E2E_MODEL_V1\n")
        out.write(f"patches {patches}\npatch_dim {patch_dim}\ndim {dim}\n")
        out.write(f"heads {heads}\nmlp_dim {mlp_dim}\nclasses {classes}\n")
        out.write(f"scale_bits {scale_bits}\n")
        for name, value in model.items():
            write_vector(out, name, value)
        out.write("end\n")
    with (args.out_dir / "bob_input.txt").open("w", encoding="utf-8") as out:
        out.write("BLB_VIT_E2E_INPUT_V1\n")
        out.write(f"patches {patches}\npatch_dim {patch_dim}\n")
        write_vector(out, "x", private_input)
        out.write("end\n")
    print(
        f"FIXTURE status=PASS out={args.out_dir} patches={patches} "
        f"patch_dim={patch_dim} dim={dim} heads={heads} classes={classes}"
    )


if __name__ == "__main__":
    main()
