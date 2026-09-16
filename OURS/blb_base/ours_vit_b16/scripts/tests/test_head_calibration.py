#!/usr/bin/env python3
"""Fast unit tests for A2 metric, cost-gate, and assignment serialization."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "legacy"))


import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibrate_heterogeneous_heads as calibration


class HeadCalibrationTest(unittest.TestCase):
    def test_relative_l2(self) -> None:
        reference = [np.array([3.0, 4.0])]
        candidate = [np.array([0.0, 0.0])]
        metric = calibration.error_metrics(reference, candidate)
        self.assertAlmostEqual(float(metric["reference_l2"]), 5.0)
        self.assertAlmostEqual(float(metric["l2"]), 5.0)
        self.assertAlmostEqual(float(metric["relative_l2"]), 1.0)

    def write_costs(self, path: Path) -> None:
        columns = sorted(calibration.COST_COLUMNS)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for layer in range(12):
                rows = (
                    ("softmax_mpc", 12, 100.0, "mpc_ckks_hybrid", "A2", 1),
                    ("softmax_mpc", 11, 80.0, "mpc_ckks_hybrid", "A2", 1),
                    ("relu_norm_mpc", 1, 5.0, "mpc_ckks_hybrid", "A2", 1),
                    # Retained as an A1/A2 wiring measurement only.  The
                    # loader must never expose this row to selection.
                    ("he_linear", 1, 1.0, "mpc_surrogate", "A2", 0),
                )
                for (protocol, head_count, latency, execution_domain,
                     cost_stage, selection_eligible) in rows:
                    writer.writerow({
                        "layer": layer,
                        "protocol": protocol,
                        "latency_ms": latency,
                        "he_ms": 0.0,
                        "mpc_ms": 0.0,
                        "conversion_ms": 0.0,
                        "communication_bytes": int(latency * 100),
                        "rounds": int(latency),
                        "head_count": head_count,
                        "packing_utilization": 0.5,
                        "multiplicative_depth": 0,
                        "ct_ct_multiply": 0,
                        "rotations": 0,
                        "relinearization": 0,
                        "rescale": 0,
                        "measured": 1,
                        "backend": "phantom_gpu",
                        "network_gbps": 1.0,
                        "rtt_ms": 0.3,
                        "execution_domain": execution_domain,
                        "cost_stage": cost_stage,
                        "measurement_scope": (
                            "direct_layer_normalization_only" if layer == 0
                            else "shape_reuse_normalization_only"),
                        "conversion_scope": "excluded_deferred_to_A3",
                        "cost_scope": "a2_normalization_only",
                        "gpu_topology": "separate_physical_gpus",
                        "selection_eligible": selection_eligible,
                    })

    def write_assignment_costs(self, path: Path) -> None:
        columns = sorted(calibration.ASSIGNMENT_COST_COLUMNS)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for layer in range(12):
                for candidate, protocols, latency in (
                        ("softmax", ["softmax_mpc"] * 12, 100.0),
                        ("mixed", ["softmax_mpc"] * 10 +
                         ["he_linear"] * 2, 80.0)):
                    linear = protocols.count("he_linear")
                    writer.writerow({
                        "layer": layer,
                        "ablation": "A4" if linear else "A3",
                        "softmax_heads": protocols.count("softmax_mpc"),
                        "relu_heads": 0,
                        "linear_heads": linear,
                        "latency_ms": latency,
                        "he_instrumented_ms": latency / 2,
                        "conversion_instrumented_ms": 4.0,
                        "communication_bytes": int(latency * 100),
                        "rounds": 10,
                        "protocol_groups": 2 if linear else 1,
                        "mpc_group_calls": 1,
                        "native_linear_groups": 1 if linear else 0,
                        "ckks_to_mpc": 100,
                        "mpc_to_ckks": 80,
                        "he_ct_pt_multiply": 200,
                        "he_ct_ct_multiply": 300,
                        "he_rotations": 400,
                        "he_rescales": 500,
                        "measurement_scope":
                            "complete_attention_assignment",
                        "cost_unit": "direct_assignment_not_per_head",
                        "network_gbps": 1.0,
                        "rtt_ms": 0.3,
                        "backend": "phantom_gpu",
                        "gpu_topology": "separate_physical_gpus",
                        "image_id": "sha256:" + "a" * 64,
                        "binary_sha256": "b" * 64,
                        "plan_sha256": (f"{layer:02x}" +
                                        ("1" if candidate == "mixed" else "0") +
                                        "c" * 61),
                        "protocols_original": ";".join(protocols),
                        "alphas_original": ";".join(["0.125"] * 12),
                        "physical_protocols": ";".join(protocols),
                        "original_order": ";".join(
                            str(head) for head in range(12)),
                        "a4_data_level_extension": 1 if linear else 0,
                        "surrogate": 0,
                        "dynamic_selection": 0,
                        "source_run": f"/measured/{candidate}/{layer}",
                    })

    def test_legacy_group_cost_can_be_read_but_not_write_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            costs_path = root / "costs.csv"
            self.write_costs(costs_path)
            costs = calibration.load_costs(costs_path, 12, 2)
            self.assertEqual(len(costs), 36)
            self.assertEqual(costs[(0, "softmax_mpc", 12)].latency_ms, 100.0)
            self.assertNotIn((0, "he_linear", 1), costs)

            model = root / "model.txt"
            model.write_text("public synthetic model binding\n", encoding="utf-8")
            sample_path = root / "input.txt"
            sample_path.write_text("public synthetic input binding\n",
                                   encoding="utf-8")
            samples = [calibration.Sample(sample_path, None)]
            plan = [[("softmax_mpc", 1.0) for _ in range(12)]
                    for _ in range(12)]
            order = [list(range(12)) for _ in range(12)]
            metrics = {}
            for oracle in ("fp32", "fixed"):
                for layer in range(12):
                    for head in range(12):
                        for protocol in calibration.PROTOCOLS:
                            metrics[(oracle, layer, head, protocol)] = {
                                "relative_l2": 0.0,
                                "max_abs": 0.0,
                                "mse": 0.0,
                            }
            runtime_plan = root / "head_plan.txt"
            calibration.write_plan(runtime_plan, plan, order)
            assignment = root / "head_assignment.json"
            with self.assertRaisesRegex(RuntimeError, "retired group-cost"):
                calibration.write_assignment_json(
                    assignment, model, samples, plan, order, costs, metrics,
                    0.1, {"top1_agreement": 1.0}, True, runtime_plan,
                    costs_path)

    def test_selection_uses_measured_group_cost_and_grouping_is_public(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            costs_path = Path(directory) / "costs.csv"
            self.write_costs(costs_path)
            costs = calibration.load_costs(costs_path, 12, 2)
            metrics = {}
            disturbance = {
                "softmax_mpc": 0.0,
                "relu_norm_mpc": 0.20,
                "he_linear": 0.20,
            }
            for oracle in ("fp32", "fixed"):
                for layer in range(12):
                    for head in range(12):
                        for protocol, value in disturbance.items():
                            metrics[(oracle, layer, head, protocol)] = {
                                "relative_l2": value,
                            }
            # Only head 0 admits ReLU.  The selected objective must be the
            # measured Softmax(11)+ReLU(1) groups, not a 12-head per-head
            # extrapolation. Linear has no eligible A4 CKKS cost.
            for oracle in ("fp32", "fixed"):
                for layer in range(12):
                    metrics[(oracle, layer, 0,
                             "relu_norm_mpc")]["relative_l2"] = 0.05
            alphas = [[0.125 for _ in range(12)] for _ in range(12)]
            with self.assertRaisesRegex(RuntimeError,
                                        "retired additive protocol-group"):
                calibration.select_protocols(
                    12, 12, alphas, costs, metrics, 0.15, 2)

    def test_missing_exact_group_size_is_not_extrapolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            costs_path = Path(directory) / "costs.csv"
            self.write_costs(costs_path)
            costs = calibration.load_costs(costs_path, 12, 2)
            # Remove the only cost that makes the 11+1 grouping complete.
            del costs[(0, "softmax_mpc", 11)]
            metrics = {}
            for oracle in ("fp32", "fixed"):
                for layer in range(12):
                    for head in range(12):
                        for protocol in calibration.PROTOCOLS:
                            metrics[(oracle, layer, head, protocol)] = {
                                "relative_l2": (0.0 if protocol ==
                                                "relu_norm_mpc" else 1.0),
                            }
            alphas = [[0.125 for _ in range(12)] for _ in range(12)]
            with self.assertRaisesRegex(RuntimeError,
                                        "retired additive protocol-group"):
                calibration.select_protocols(
                    12, 12, alphas, costs, metrics, 0.15, 2)

    def test_rejects_wrong_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "costs.csv"
            self.write_costs(path)
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["rtt_ms"] = "1.0"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "1 Gbps / 0.3 ms RTT"):
                calibration.load_costs(path, 12, 2)

    def test_direct_assignment_selection_never_adds_group_costs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assignment_costs.csv"
            self.write_assignment_costs(path)
            costs = calibration.load_assignment_costs(path, 12, 12, 2)
            self.assertEqual(len(costs), 24)
            metrics = {}
            for oracle in ("fp32", "fixed"):
                for layer in range(12):
                    for head in range(12):
                        for protocol in calibration.PROTOCOLS:
                            metrics[(oracle, layer, head, protocol)] = {
                                "relative_l2": (0.1 if protocol ==
                                                "he_linear" else 0.0),
                            }
            alphas = [[0.125] * 12 for _ in range(12)]
            plan, _decisions, selected = \
                calibration.select_measured_assignments(
                    12, 12, alphas, costs, metrics, 0.15)
            self.assertEqual(plan[0][-2:],
                             [("he_linear", 0.125),
                              ("he_linear", 0.125)])
            self.assertEqual(selected[0].latency_ms, 80.0)
            summary = calibration.measured_assignment_cost_summary(selected)
            self.assertFalse(summary["is_end_to_end_prediction"])
            self.assertIsNone(summary["under_eight_minutes"])
            self.assertFalse(summary["per_head_extrapolation"])


if __name__ == "__main__":
    unittest.main()
