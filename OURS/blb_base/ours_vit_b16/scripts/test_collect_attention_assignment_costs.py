#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_attention_assignment_costs as collector


class AssignmentCostCollectorTest(unittest.TestCase):
    def make_run(self, root: Path, shortcut: bool = False) -> Path:
        run = root / "run"
        run.mkdir()
        (run / "status.txt").write_text("alice_rc=0\nbob_rc=0\n")
        (run / "environment.txt").write_text(
            "backend=gpu\nnetwork=1Gbps/0.3ms_RTT\nablation=A4\n"
            f"binary_sha256={'a' * 64}\n")
        (run / "gpu_selection.txt").write_text(
            "physical_gpus=2,3\nparty_topology=separate_physical_gpus\n"
            f"image_id=sha256:{'b' * 64}\nplan_sha256={'c' * 64}\n")
        header = ("ablation,layer,physical_head,original_head,protocol,alpha,"
                  "contiguous_group,group_size,qk_ciphertexts,"
                  "physical_slot_utilization,logical_slot_utilization,"
                  "execution_status\n")
        plan_rows = []
        for head in range(12):
            protocol = "softmax_mpc" if head < 10 else "he_linear"
            plan_rows.append(
                f"A4,0,{head},{head},{protocol},0.125,0,12,4,1,1,wired\n")
        for party in (1, 2):
            (run / f"head_plan_party{party}.csv").write_text(
                header + "".join(plan_rows))
        for party, latency, communication in (
                ("alice.log", 100.0, 1000), ("bob.log", 101.0, 1200)):
            (run / party).write_text(
                "OURS_A4_HE1_CHAIN status=PASS enabled=1 "
                "data_level_extension=1 modulus_entries=9 "
                "baseline_entries=8\n"
                + ("OURS_BLB_QK_PACKING status=PASS mode=baseline_fast "
                   "rotation_cap=122 mpc_surrogate=0 "
                   "client_decrypt_reencrypt=0 "
                   "a0_numerical_validation=PENDING\n"
                   "OURS_BLB_FAST_AV_LEVEL status=PASS layer=0 context=he2 "
                   "mod_switch_only=50 extra_rescales=0 extra_multiplies=0\n"
                   "OURS_BLB_FAST_AV_LEVEL status=PASS layer=0 context=he1 "
                   "mod_switch_only=10 extra_rescales=0 extra_multiplies=0 "
                   "client_decrypt_reencrypt=0 mpc_reference=0\n"
                   "OURS_BLB_AV_PACKING status=PASS mode=baseline_fast "
                   "active_pairs=5 softmax_cts=48 extra_mask_multiplies=0 "
                   "extra_rescales=0 mpc_surrogate=0 "
                   "client_decrypt_reencrypt=0 "
                   "a0_numerical_validation=PENDING\n"
                   "OURS_BLB_AV_PACKING status=PASS mode=baseline_fast "
                   "active_pairs=1 softmax_cts=48 extra_mask_multiplies=0 "
                   "extra_rescales=0 mpc_surrogate=0 "
                   "client_decrypt_reencrypt=0 "
                   "a0_numerical_validation=PENDING\n"
                   if party == "alice.log" else "")
                + ("OURS_A4_NATIVE_CKKS status=PASS layer=0 linear_heads=2 "
                   "v_alignment_rescales=1 client_decrypt_reencrypt=0 "
                   "mpc_reference=0\n" if party == "alice.log" else "") +
                "OURS_SELECTIVE_OUTPUT_CONVERSION status=PASS layer=0 "
                "stage=a4_mpc_heads_wo ciphertexts=12 "
                "algorithm=BLB_Algorithm_1 client_reencrypt=0\n"
                "OURS_SELECTIVE_OUTPUT_CONVERSION status=PASS layer=0 "
                "stage=a4_linear_heads_wo ciphertexts=12 "
                "algorithm=BLB_Algorithm_1 client_reencrypt=0\n"
                "OURS_A3_A4_ATTENTION_OUTPUT status=PASS layer=0 "
                "ablation=A4 mpc_heads=10 linear_heads=2 "
                "dynamic_selection=0 "
                f"surrogate={'1' if shortcut else '0'}\n"
                "OURS_A3_A4_ASSIGNMENT_PROFILE status=PASS ablation=A4 "
                f"layer=0 latency_ms={latency} "
                f"communication_bytes={communication} rounds=20 "
                "he_instrumented_ms=60 conversion_instrumented_ms=4 "
                "softmax_heads=10 relu_heads=0 linear_heads=2 "
                "protocol_groups=2 mpc_group_calls=1 native_linear_groups=1 "
                "ckks_to_mpc=100 mpc_to_ckks=80 he_ct_pt_multiply=200 "
                "he_ct_ct_multiply=300 he_rotations=400 he_rescales=500 "
                "measurement_scope=complete_attention_assignment "
                "cost_unit=direct_assignment_not_per_head "
                "network=1Gbps_0.3ms_RTT dynamic_selection=0 surrogate=0\n")
        return run

    def test_collects_whole_assignment_without_division(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = collector.measured_rows(
                self.make_run(Path(directory)))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["latency_ms"], 101.0)
            self.assertEqual(rows[0]["communication_bytes"], 2200)
            self.assertEqual(rows[0]["linear_heads"], 2)
            self.assertEqual(rows[0]["cost_unit"],
                             "direct_assignment_not_per_head")
            self.assertEqual(
                rows[0]["protocols_original"].split(";").count("he_linear"),
                2)
            self.assertEqual(rows[0]["a4_data_level_extension"], 1)
            self.assertEqual(rows[0]["blb_fast_attention_packing"], 1)

    def test_rejects_surrogate_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "forbidden A4 shortcut"):
                collector.measured_rows(
                    self.make_run(Path(directory), shortcut=True))

    def test_rejects_removed_full_diagonal_cost_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory))
            alice = run / "alice.log"
            alice.write_text(
                alice.read_text().replace(
                    "OURS_BLB_QK_PACKING status=PASS mode=baseline_fast",
                    "OURS_BLB_QK_PACKING status=PASS mode=full_diagonal",
                    1))
            with self.assertRaisesRegex(ValueError, "BLB-fast QK"):
                collector.measured_rows(run)


if __name__ == "__main__":
    unittest.main()
