"""Unit tests for the second-phase offline allocator experiment."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import offline_allocator_experiment as experiment
import offline_cost_aware_head_allocator as allocator


class FakeCostModel:
    """A complete, deterministic count cost view for the small oracle."""

    def predict_layer(self, layer: int,
                      counts: tuple[int, int, int]) -> allocator.CostPrediction:
        s, r, l = counts
        return allocator.CostPrediction(
            layer=layer,
            counts=counts,
            latency_ms=float(100 + 3 * r + 5 * l + s),
            communication_bytes=1000 + 10 * r + 20 * l,
            observation_count=1,
            source_runs=("unit-test",),
        )


class OfflineExperimentTest(unittest.TestCase):
    def test_packed_and_unpacked_costs_cannot_mix(self):
        unpacked = [{'attention_packing': 'unpacked', 'heterogeneous_head_grouping': '0', 'plan_sha256': 'x'}]
        packed = [{'attention_packing': 'packed', 'heterogeneous_head_grouping': '1', 'plan_sha256': 'y'}]
        experiment.require_unpacked_probe_provenance(unpacked, Path('HA.csv'))
        experiment.require_unpacked_probe_provenance(packed, Path('J.csv'), 'packed')
        for rows, layout in ((packed, 'unpacked'), (unpacked, 'packed'), (packed+unpacked, 'packed')):
            with self.assertRaises(ValueError):
                experiment.require_unpacked_probe_provenance(rows, Path('mixed.csv'), layout)

    def test_layer_aware_cost_model_uses_only_train_points(self) -> None:
        points = []
        compositions = ((12, 0, 0), (10, 2, 0), (10, 0, 2), (6, 6, 0))
        for layer in range(2):
            for index, counts in enumerate(compositions):
                points.append(experiment.ProbeCostPoint(
                    composition=counts,
                    latency_ms=1000 + 100 * layer + 10 * index,
                    communication_bytes=2000 + 100 * layer + index,
                    source="train",
                    split="train_existing",
                    layer=layer,
                    plan_sha256=f"{index:064x}",
                ))
        latency, communication = experiment.fit_cost_models(points, 1e-3)
        self.assertFalse(any(
            point.split == "strict_heldout" for point in points
        ))
        self.assertEqual(len(latency.FEATURE_NAMES), 23)
        self.assertGreater(latency.predict(1, (6, 0, 6)), 0.0)
        self.assertGreater(communication.predict(0, (6, 0, 6)), 0.0)
        self.assertEqual(
            latency.metadata()["training_compositions"],
            [allocator.composition_name(value) for value in sorted(compositions)],
        )

    def test_holdout_missing_is_explicit_and_not_scored(self) -> None:
        points = [experiment.ProbeCostPoint(
            composition=counts,
            latency_ms=float(100 + index),
            communication_bytes=1000 + index,
            source="existing",
            split="train_existing",
            layer=0,
            plan_sha256=f"{index:064x}",
        ) for index, counts in enumerate(
            ((12, 0, 0), (10, 2, 0), (10, 0, 2), (8, 4, 0),
             (8, 0, 4), (6, 4, 2), (4, 4, 4)))]
        rows, meta = experiment.cost_model_holdout_rows(
            points, [], [], {(6, 6, 0)}, set(), 1e-3)
        self.assertEqual(meta["status"], "INCOMPLETE_MISSING_REAL_MEASUREMENT")
        self.assertFalse(meta["uses_heldout_in_fit"])
        self.assertEqual(rows[0]["status"], "NOT_MEASURED_NO_REAL_HELDOUT")
        self.assertEqual(rows[0]["latency_ape_percent"], "")
        self.assertEqual(rows[-1]["latency_mape_percent"], "NA")

    def test_insufficient_real_train_never_fabricates_holdout_prediction(self) -> None:
        point = experiment.ProbeCostPoint(
            composition=(6, 6, 0), latency_ms=100.0,
            communication_bytes=1000, source="probe", split="train_existing",
            layer=0, plan_sha256="a" * 64)
        rows, meta = experiment.cost_model_holdout_rows(
            [point], [], [], {(10, 2, 0)}, {(6, 0, 6)}, 1e-3)
        self.assertEqual(meta["status"], "INCOMPLETE_INSUFFICIENT_REAL_TRAIN")
        self.assertEqual(rows[0]["status"], "NOT_MEASURED_NO_REAL_HELDOUT")
        self.assertEqual(rows[0]["predicted_latency_ms"], "")
        self.assertEqual(rows[-1]["cost_model_fit_status"],
                         "NOT_FIT_INSUFFICIENT_REAL_TRAIN")

    def test_random_plan_preserves_each_layer_composition(self) -> None:
        plan = tuple(tuple(
            [("softmax_mpc", 1.0)] * 4 +
            [("relu_norm_mpc", 1.0)] * 3 +
            [("he_linear", 0.01)] * 5
        ) for _ in range(allocator.MODEL_LAYERS))
        alphas = [[0.01 + 0.001 * head for head in range(12)]
                  for _ in range(allocator.MODEL_LAYERS)]
        random_plan = experiment.randomize_plan(plan, alphas, 1234)
        self.assertEqual(
            allocator.counts_for_plan(random_plan),
            allocator.counts_for_plan(plan),
        )
        self.assertNotEqual(random_plan[0], plan[0])

    def test_dp_matches_exhaustive_pareto_for_six_heads(self) -> None:
        candidates = {}
        for head in range(12):
            for protocol in ("relu_norm_mpc", "he_linear"):
                candidates[(0, head, protocol)] = {
                    "proxy_score": float((head + 1) *
                                          (1 if protocol == "relu_norm_mpc" else 2)),
                    "loss_delta": 0.0,
                    "relative_l2": 0.0,
                }
        rows = experiment.oracle_rows(
            0, [0, 1, 2, 3, 4, 5], candidates,
            [[0.01] * 12], FakeCostModel())
        summary = rows[0]
        self.assertEqual(summary["enumerated_assignments"], 729)
        self.assertEqual(summary["dp_transition_count"], 168)
        self.assertEqual(summary["pareto_set_equal"], True)
        self.assertEqual(summary["dp_exhaustive_proxy_gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
