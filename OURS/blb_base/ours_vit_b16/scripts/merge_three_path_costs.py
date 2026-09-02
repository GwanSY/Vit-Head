#!/usr/bin/env python3
"""Merge disjoint directly measured path rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    fieldnames = None
    for path in args.input:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = fieldnames or reader.fieldnames
            if reader.fieldnames != fieldnames:
                raise ValueError("path-cost CSV headers disagree")
            rows.extend(reader)
    keys = [(row["path"], row["heads_group"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate path/group rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["path"],
                                                        int(row["heads_group"]))))
    print(f"OURS_THREE_PATH_COSTS_MERGE status=PASS rows={len(rows)}")
    print(f"OURS_THREE_PATH_COSTS_CSV={args.output}")


if __name__ == "__main__":
    main()
