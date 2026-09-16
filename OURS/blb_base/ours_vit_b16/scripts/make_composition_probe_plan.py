#!/usr/bin/env python3
"""Create one fixed contiguous-group plan for a composition probe.

This helper only materializes a public, fixed ``OURS_VIT_HEAD_PLAN_V1`` file.
The caller is expected to run the existing one-block A3/A4 attention
benchmark and collect its complete assignment profile.  It never changes the
secure protocol implementation and never selects heads online.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROTOCOLS = ("softmax_mpc", "relu_norm_mpc", "he_linear")
LAYERS = 12
HEADS = 12


def parse_counts(value: str) -> tuple[int, int, int]:
    parts = value.replace("S", ",").replace("R", ",").replace("L", "").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "composition must be S,R,L, for example 6,4,2 or 6S4R2L")
    try:
        counts = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("composition counts must be integers") from error
    if any(count < 0 for count in counts) or sum(counts) != HEADS:
        raise argparse.ArgumentTypeError("composition counts must be non-negative and sum to 12")
    return counts  # type: ignore[return-value]


def read_alphas(path: Path) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["protocol"] != "he_linear":
                continue
            key = int(row["layer"]), int(row["head"])
            alpha = float(row["alpha"])
            previous = result.setdefault(key, alpha)
            if abs(previous - alpha) > 1e-12:
                raise ValueError(f"inconsistent alpha for layer={key[0]} head={key[1]}")
    expected = {(layer, head) for layer in range(LAYERS) for head in range(HEADS)}
    if set(result) != expected:
        missing = sorted(expected - set(result))[:5]
        raise ValueError(f"alpha CSV is incomplete; missing {missing}")
    return result


def make_plan(counts: tuple[int, int, int],
              alphas: dict[tuple[int, int], float]) -> list[list[tuple[str, float]]]:
    softmax, relu, linear = counts
    paths = (["softmax_mpc"] * softmax +
             ["relu_norm_mpc"] * relu +
             ["he_linear"] * linear)
    return [[
        (protocol, alphas[(layer, head)] if protocol == "he_linear" else 1.0)
        for head, protocol in enumerate(paths)
    ] for layer in range(LAYERS)]


def write_plan(path: Path, plan: list[list[tuple[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("OURS_VIT_HEAD_PLAN_V1\n")
        handle.write("layers=12,heads=12,logical_tokens=197,physical_tokens=256\n")
        handle.write("layer,physical_head,original_head,protocol,alpha\n")
        for layer, entries in enumerate(plan):
            for physical, (protocol, alpha) in enumerate(entries):
                handle.write(f"{layer},{physical},{physical},{protocol},{alpha:.17g}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composition", type=parse_counts, required=True)
    parser.add_argument("--candidate-errors", type=Path, required=True,
                        help="isolated_candidate_metrics.csv containing all Linear alphas")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    alphas = read_alphas(args.candidate_errors)
    write_plan(args.output, make_plan(args.composition, alphas))
    softmax, relu, linear = args.composition
    ablation = "A4" if linear else "A3"
    print(f"OURS_COMPOSITION_PROBE_PLAN status=PASS composition={softmax}S{relu}R{linear}L "
          f"ablation={ablation} output={args.output}")


if __name__ == "__main__":
    main()
