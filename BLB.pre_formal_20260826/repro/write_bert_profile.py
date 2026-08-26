#!/usr/bin/env python3
"""Write machine-readable, send-only BLB BERT profiling from party logs."""

import argparse
import csv
import re
from pathlib import Path


STAGE_RE = re.compile(
    r"^BLOCK_ONLINE_STAGE party=(?P<party>\d+) stage=(?P<stage>\S+) "
    r"wall_ms=(?P<wall_ms>[0-9.eE+-]+) "
    r"send_bytes=(?P<send_bytes>\d+) rounds=(?P<rounds>\d+) "
    r"he_ms=(?P<he_ms>[0-9.eE+-]+) "
    r"time_share=(?P<time_share>[0-9.eE+-]+) "
    r"comm_share=(?P<comm_share>[0-9.eE+-]+)$",
    re.MULTILINE,
)
AUDIT_RE = re.compile(
    r"^VOLE_COMM_AUDIT party=(?P<party>\d+) bucket=(?P<bucket>\S+) "
    r"send_bytes=(?P<send_bytes>\d+) calls=(?P<calls>\d+)$",
    re.MULTILINE,
)
AUDIT_EXCLUSIVE_RE = re.compile(
    r"^VOLE_COMM_AUDIT_EXCLUSIVE party=(?P<party>\d+) "
    r"bucket=(?P<bucket>\S+) send_bytes=(?P<send_bytes>\d+)$",
    re.MULTILINE,
)


def single(pattern: str, text: str, name: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing {name}")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()

    stage_rows = []
    audit_rows = []
    summary_rows = []
    for log_name in ("alice.log", "bob.log"):
        path = args.log_dir / log_name
        if not path.is_file():
            raise RuntimeError(f"missing party log: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        party_stages = [match.groupdict() for match in STAGE_RE.finditer(text)]
        if len(party_stages) != 12:
            raise RuntimeError(
                f"{log_name}: expected 12 online stages, got {len(party_stages)}"
            )
        stage_rows.extend(party_stages)
        audit_rows.extend(match.groupdict() for match in AUDIT_RE.finditer(text))
        for match in AUDIT_EXCLUSIVE_RE.finditer(text):
            row = match.groupdict()
            row["calls"] = ""
            audit_rows.append(row)

        party = party_stages[0]["party"]
        online_streams = re.search(
            r"^BLOCK_ONLINE_STREAM_COMM party=\d+ "
            r"primary_send_bytes=(\d+) reverse_send_bytes=(\d+) "
            r"gc_send_bytes=(\d+)$",
            text,
            re.MULTILINE,
        )
        offline_streams = re.search(
            r"^VOLE_OFFLINE_STREAM_COMM party=\d+ "
            r"primary_send_bytes=(\d+) reverse_send_bytes=(\d+) "
            r"gc_send_bytes=(\d+)$",
            text,
            re.MULTILINE,
        )
        if not online_streams or not offline_streams:
            raise RuntimeError(f"{log_name}: missing stream communication audit")
        summary_rows.append(
            {
                "party": party,
                "block_online_seconds": single(
                    r"^BLOCK_ONLINE_TIME seconds=([0-9.eE+-]+)",
                    text,
                    "BLOCK_ONLINE_TIME",
                ),
                "block_online_send_bytes": single(
                    r"^BLOCK_ONLINE_COMM send_bytes=(\d+)",
                    text,
                    "BLOCK_ONLINE_COMM",
                ),
                "conversion_seconds": single(
                    r"^conversion_time:([0-9.eE+-]+)$",
                    text,
                    "conversion_time",
                ),
                "ckks_to_mpc_ciphertexts": single(
                    r"^CKKS_TO_MPC_COUNT value=(\d+)$",
                    text,
                    "CKKS_TO_MPC_COUNT",
                ),
                "mpc_to_ckks_plaintexts": single(
                    r"^MPC_TO_CKKS_COUNT value=(\d+)$",
                    text,
                    "MPC_TO_CKKS_COUNT",
                ),
                "conversion_ciphertexts": single(
                    r"^CONVERSION_CIPHERTEXT_COUNT value=(\d+)$",
                    text,
                    "CONVERSION_CIPHERTEXT_COUNT",
                ),
                "field_to_ring_calls": single(
                    r"^FIELD_TO_RING_COUNT value=(\d+)$",
                    text,
                    "FIELD_TO_RING_COUNT",
                ),
                "ring_to_field_calls": single(
                    r"^RING_TO_FIELD_COUNT value=(\d+)$",
                    text,
                    "RING_TO_FIELD_COUNT",
                ),
                "online_primary_send_bytes": online_streams.group(1),
                "online_reverse_send_bytes": online_streams.group(2),
                "online_gc_send_bytes": online_streams.group(3),
                "offline_primary_send_bytes": offline_streams.group(1),
                "offline_reverse_send_bytes": offline_streams.group(2),
                "offline_gc_send_bytes": offline_streams.group(3),
            }
        )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    with (args.log_dir / "block_online_profile.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)
    with (args.log_dir / "vole_comm_profile.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("party", "bucket", "send_bytes", "calls")
        )
        writer.writeheader()
        writer.writerows(audit_rows)
    with (args.log_dir / "block_online_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(
        "BERT_PROFILE_CSV status=PASS "
        f"stages={args.log_dir / 'block_online_profile.csv'} "
        f"vole={args.log_dir / 'vole_comm_profile.csv'} "
        f"summary={args.log_dir / 'block_online_summary.csv'}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BERT_PROFILE_CSV status=BLOCKED reason={exc}")
        raise SystemExit(2)
