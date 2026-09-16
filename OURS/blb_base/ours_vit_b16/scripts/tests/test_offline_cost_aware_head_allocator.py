#!/usr/bin/env python3
"""Unit tests for the offline cost-aware head allocator."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import offline_cost_aware_head_allocator as allocator


class OfflineAllocatorTest(unittest.TestCase):
    def write_costs(self, path: Path) -> None:
        fields = sorted(allocator.MeasuredCompositionCostModel.REQUIRED | {
            "a4_data_level_extension"})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            protocols = {
                (12, 0, 0): ["softmax_mpc"] * 12,
                (10, 0, 2): ["softmax_mpc"] * 10 + ["he_linear"] * 2,
            }
            for layer in range(12):
                for counts, values in protocols.items():
                    writer.writerow({
                        "layer": layer,
                        "ablation": "A4" if counts != (12, 0, 0) else "A3",
                        "softmax_heads": counts[0],
                        "relu_heads": counts[1],
                        "linear_heads": counts[2],
                        "latency_ms": 100.0 if counts == (12, 0, 0) else 80.0,
                        "communication_bytes": 1000,
                        "measurement_scope": "complete_attention_assignment",
                        "cost_unit": "direct_assignment_not_per_head",
                        "network_gbps": 1.0,
                        "rtt_ms": 0.3,
                        "backend": "phantom_gpu",
                        "gpu_topology": "separate_physical_gpus",
                        "image_id": "sha256:" + "a" * 64,
                        "binary_sha256": "b" * 64,
                        "plan_sha256": "c" * 64,
                        "protocols_original": ";".join(values),
                        "physical_protocols": ";".join(values),
                        "original_order": ";".join(str(i) for i in range(12)),
                        "surrogate": 0,
                        "dynamic_selection": 0,
                        "source_run": f"/measured/{layer}/{counts}",
                        "a4_data_level_extension": 1 if counts != (12, 0, 0) else 0,
                    })

    def test_sparse_cost_model_does_not_invent_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "costs.csv"
            self.write_costs(path)
            model = allocator.MeasuredCompositionCostModel(path, 12, 12)
            self.assertEqual(model.predict_layer(0, (12, 0, 0)).latency_ms, 100.0)
            self.assertIsNone(model.predict_layer(0, (11, 1, 0)))
            self.assertIn((11, 1, 0), model.missing_compositions())

    def test_head_dp_returns_exact_counts_without_enumeration(self) -> None:
        candidates = {}
        for head in range(12):
            candidates[(0, head, "relu_norm_mpc")] = {
                "proxy_score": float(head + 1),
                "loss_delta": 0.0,
                "relative_l2": 0.0,
            }
            candidates[(0, head, "he_linear")] = {
                "proxy_score": float(100 - head),
                "loss_delta": 0.0,
                "relative_l2": 0.0,
            }
        alphas = [[0.01] * 12]
        paths, score = allocator.min_assignment_for_counts(
            0, (10, 1, 1), candidates, alphas)
        self.assertEqual(sum(path == "softmax_mpc" for path, _ in paths), 10)
        self.assertEqual(sum(path == "relu_norm_mpc" for path, _ in paths), 1)
        self.assertEqual(sum(path == "he_linear" for path, _ in paths), 1)
        self.assertEqual(score, 1.0 + 89.0)
        self.assertEqual(paths[0][1], 1.0)

    def test_runtime_order_is_grouped_and_json_fields_are_unambiguous(self) -> None:
        plan = tuple(tuple(
            [("softmax_mpc", 1.0)] * 4 +
            [("relu_norm_mpc", 1.0)] * 4 +
            [("he_linear", 0.01)] * 4)
            for _ in range(12))
        allocator.validate_runtime_plan(plan)
        order = allocator.physical_order(plan[0])
        self.assertEqual(order, list(range(12)))
        self.assertEqual(allocator.counts_for_plan(plan)[0], (4, 4, 4))


if __name__ == "__main__":
    unittest.main()
