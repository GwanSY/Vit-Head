#!/usr/bin/env python3
"""Materialize a small train/held-out composition-probe suite.

The resulting plans are for the existing one-block attention benchmark only.
They use contiguous original heads so the measured variable is the
composition.  This script does not launch Docker, GPUs, or any secure code.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from make_composition_probe_plan import make_plan, parse_counts, read_alphas, write_plan


def parse_list(value: str) -> list[tuple[int, int, int]]:
    items = [item.strip() for item in value.split() if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("composition list cannot be empty")
    return [parse_counts(item) for item in items]


def composition_name(counts: tuple[int, int, int]) -> str:
    return f"{counts[0]}S{counts[1]}R{counts[2]}L"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-errors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--train", type=parse_list, default="6,6,0 6,0,6 0,6,6 2,6,4",
        help="space-separated missing compositions used for fitting")
    parser.add_argument(
        "--heldout", type=parse_list, default="10,2,0 4,0,8 0,4,8 4,2,6",
        help="space-separated missing compositions kept strictly held out")
    args = parser.parse_args()
    train = list(args.train)
    heldout = list(args.heldout)
    if set(train) & set(heldout):
        raise ValueError("train and held-out composition lists overlap")
    if len(set(train)) != len(train) or len(set(heldout)) != len(heldout):
        raise ValueError("composition lists contain duplicates")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    alphas = read_alphas(args.candidate_errors)
    rows: list[dict[str, str]] = []
    for split, compositions in (("train", train), ("heldout", heldout)):
        for counts in compositions:
            name = composition_name(counts)
            path = args.output_dir / f"{split}_{name}.txt"
            write_plan(path, make_plan(counts, alphas))
            rows.append({
                "composition": name,
                "S_heads": str(counts[0]), "R_heads": str(counts[1]),
                "L_heads": str(counts[2]), "split": split,
                "ablation": "A4" if counts[2] else "A3",
                "plan": str(path.resolve()),
            })
    manifest = args.output_dir / "composition_probe_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "composition", "S_heads", "R_heads", "L_heads", "split",
            "ablation", "plan"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"OURS_COMPOSITION_PROBE_SUITE status=PASS output={manifest}")
    for row in rows:
        print(f"composition={row['composition']} split={row['split']} "
              f"ablation={row['ablation']} plan={row['plan']}")


if __name__ == "__main__":
    main()
