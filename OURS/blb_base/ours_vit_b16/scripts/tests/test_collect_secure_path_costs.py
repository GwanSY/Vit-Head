#!/usr/bin/env python3
"""Fast parser/aggregation test for A2 grouped secure cost runs."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "legacy"))


import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_secure_path_costs as collector


class SecurePathCostCollectorTest(unittest.TestCase):
    def make_run(self, root: Path, protocol: str) -> Path:
        run = root / protocol
        run.mkdir()
        (run / "status.txt").write_text("alice_rc=0\nbob_rc=0\n",
                                        encoding="utf-8")
        (run / "environment.txt").write_text(
            "backend=gpu\nnetwork=1Gbps/0.3ms_RTT\n"
            "cost_scope=a2_normalization_only\n", encoding="utf-8")
        (run / "gpu_selection.txt").write_text(
            "physical_gpus=2,3\n"
            "party_topology=separate_physical_gpus\n", encoding="utf-8")
        for name, latency, communication in (
                ("alice.log", 10.0, 1000), ("bob.log", 11.0, 1200)):
            (run / name).write_text(
                "OURS_HETERO_PATH_PROFILE status=PASS ablation=A2 layer=0 "
                f"protocol={protocol} physical_head_begin=0 head_count=12 "
                f"latency_ms={latency} communication_bytes={communication} "
                "rounds=7 execution_domain=MPC_CKKS_hybrid "
                "conversion_scope=shared_full_boundary "
                "private_dynamic_branch=0\n"
                "OURS_NORMALIZATION_COST_STOP status=PASS ablation=A2 "
                "layer=0 cost_scope=a2_normalization_only "
                "mpc_to_ckks_included=0 av_included=0 wo_included=0\n",
                encoding="utf-8")
        return run

    def test_aggregates_two_parties_without_a3_conversions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.make_run(root, "relu_norm_mpc")
            rows = collector.measured_rows(run)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["latency_ms"], 11.0)
            self.assertEqual(row["communication_bytes"], 2200)
            self.assertEqual(row["rounds"], 7)
            self.assertEqual(row["conversion_ms"], 0.0)
            self.assertEqual(row["conversion_scope"],
                             "excluded_deferred_to_A3")
            self.assertEqual(row["measurement_scope"],
                             "direct_layer_normalization_only")
            self.assertEqual(row["execution_domain"], "mpc_ckks_hybrid")
            self.assertEqual(row["selection_eligible"], 1)


if __name__ == "__main__":
    unittest.main()
