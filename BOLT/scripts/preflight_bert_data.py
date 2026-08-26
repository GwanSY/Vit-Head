#!/usr/bin/env python3
"""Validate the exact text tensor layout consumed by BOLT_BERT."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


LAYERS = 12
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
INTERMEDIATE = 3072


def csv_shape(path: Path) -> tuple[int, int]:
    rows = 0
    columns: int | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), 1):
            if not row or any(cell.strip() == "" for cell in row):
                raise ValueError(f"{path}: empty cell at row {row_number}")
            try:
                for cell in row:
                    int(cell)
            except ValueError as error:
                raise ValueError(
                    f"{path}: non-integer value at row {row_number}"
                ) from error
            if columns is None:
                columns = len(row)
            elif len(row) != columns:
                raise ValueError(
                    f"{path}: ragged row {row_number}: {len(row)} != {columns}"
                )
            rows += 1
    if rows == 0 or columns is None:
        raise ValueError(f"{path}: empty file")
    return rows, columns


def require_shape(path: Path, expected: tuple[int, int]) -> None:
    if not path.is_file():
        raise ValueError(f"{path}: missing")
    actual = csv_shape(path)
    if actual != expected:
        raise ValueError(f"{path}: shape {actual}, expected {expected}")


def validate_model(weights: Path, num_class: int) -> int:
    checked = 0
    for layer in range(LAYERS):
        prefix = f"bert.encoder.layer.{layer}."
        for name in ("query", "key", "value"):
            require_shape(
                weights / f"{prefix}attention.self.{name}.weight.txt",
                (HEADS * HIDDEN, HEAD_DIM),
            )
            require_shape(
                weights / f"{prefix}attention.self.{name}.bias.txt",
                (HEADS * HEAD_DIM, 1),
            )
            checked += 2
        tensors = {
            "attention.output.dense.weight.txt": (HIDDEN, HIDDEN),
            "attention.output.dense.bias.txt": (HIDDEN, 1),
            "intermediate.dense.weight.txt": (HIDDEN, INTERMEDIATE),
            "intermediate.dense.bias.txt": (INTERMEDIATE, 1),
            "output.dense.weight.txt": (INTERMEDIATE, HIDDEN),
            "output.dense.bias.txt": (HIDDEN, 1),
            "attention.output.LayerNorm.weight.txt": (HIDDEN, 1),
            "attention.output.LayerNorm.bias.txt": (HIDDEN, 1),
            "output.LayerNorm.weight.txt": (HIDDEN, 1),
            "output.LayerNorm.bias.txt": (HIDDEN, 1),
        }
        for suffix, shape in tensors.items():
            require_shape(weights / f"{prefix}{suffix}", shape)
            checked += 1

    globals_and_shapes = {
        "bert.pooler.dense.weight.txt": (HIDDEN, HIDDEN),
        "bert.pooler.dense.bias.txt": (HIDDEN, 1),
        "classifier.weight.txt": (HIDDEN, num_class),
        "classifier.bias.txt": (num_class, 1),
    }
    for name, shape in globals_and_shapes.items():
        require_shape(weights / name, shape)
        checked += 1
    return checked


def validate_inputs(weights: Path, first_id: int, count: int) -> int:
    checked = 0
    for sample_id in range(first_id, first_id + count):
        require_shape(
            weights / f"inputs_{sample_id}_data.txt", (128, HIDDEN)
        )
        require_shape(weights / f"inputs_{sample_id}_mask.txt", (128, 1))
        checked += 2
    return checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", required=True,
        help="task directory whose direct child is weights_txt/",
    )
    parser.add_argument(
        "--role", choices=("alice", "bob", "both"), default="both"
    )
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--num-sample", type=int, default=1)
    parser.add_argument("--num-class", type=int, default=2)
    args = parser.parse_args()

    if args.id < 0 or args.num_sample <= 0:
        parser.error("--id must be nonnegative and --num-sample must be positive")
    if args.num_class != 2:
        parser.error("this BOLT_BERT binary is compiled with NUM_CLASS=2")

    raw_data = args.data.strip()
    if not raw_data:
        print(
            "BOLT_BERT_DATA_PREFLIGHT=FAIL error=--data is empty; "
            "set BERT_DATA to datasets/official_bert/bolt/quantize/mrpc",
            file=sys.stderr,
        )
        return 2

    data_root = Path(raw_data).expanduser().resolve()
    weights = data_root / "weights_txt"
    try:
        if data_root.name == "weights_txt":
            raise ValueError(
                f"--data must be the parent of weights_txt, got {data_root}"
            )
        if not data_root.is_dir():
            raise ValueError(f"--data directory does not exist: {data_root}")
        if not weights.is_dir():
            raise ValueError(
                f"{weights}: missing directory; --data must point to the task "
                "directory whose direct child is weights_txt/"
            )
        checked = 0
        if args.role in ("alice", "both"):
            checked += validate_model(weights, args.num_class)
        if args.role in ("bob", "both"):
            checked += validate_inputs(weights, args.id, args.num_sample)
    except (OSError, ValueError) as error:
        print(f"BOLT_BERT_DATA_PREFLIGHT=FAIL error={error}", file=sys.stderr)
        return 2

    print(
        "BOLT_BERT_DATA_PREFLIGHT=PASS "
        f"role={args.role} checked_files={checked} weights={weights}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
