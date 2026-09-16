#!/usr/bin/env python3
"""Emit the CSV consumed by the existing full-12 suite runner."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-group", choices=("variable_candidate",
                                                    "final"), required=True)
    parser.add_argument("--names", nargs="*",
                        help="optional exact plan names to include")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    selected_names = set(args.names or [])
    for name, info in payload["plans"].items():
        if selected_names and name not in selected_names:
            continue
        group = info.get("budget_group")
        if args.budget_group == "variable_candidate" and group != "variable_candidate":
            continue
        if args.budget_group == "final" and group not in {
                "reference", "fixed_candidate", "fixed_selected",
                "variable_selected"}:
            continue
        rows.append({
            "name": name,
            "ablation": info["ablation"],
            "packing": "1",
            "head_plan": info["path"],
            "secure_accuracy_json": "",
        })
    if not rows:
        raise ValueError("manifest has no plans for requested group")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "name", "ablation", "packing", "head_plan", "secure_accuracy_json"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"OURS_TWO_BUDGET_SUITE status=PASS group={args.budget_group} "
          f"rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
