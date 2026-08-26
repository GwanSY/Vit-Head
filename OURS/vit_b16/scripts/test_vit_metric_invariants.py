#!/usr/bin/env python3
"""Fail-closed tests for secure aggregate error metric evidence."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from summarize_vit_acceptance import AcceptanceError, _error_metrics
from validate_vit_aggregate_errors import EXPECTED_FIELDS, validate


class VitMetricInvariantTest(unittest.TestCase):
    def test_accepts_mathematically_consistent_metrics(self) -> None:
        result = _error_metrics({
            "elements": 4,
            "max_abs": 4.0,
            "mean_abs": 2.0,
            "rmse": 3.0,
        }, "fixture")
        self.assertEqual(result["rmse"], 3.0)

    def test_rejects_norm_order_violations(self) -> None:
        bad_rows = (
            {"max_abs": 4.0, "mean_abs": 3.0, "rmse": 2.0},
            {"max_abs": 2.0, "mean_abs": 3.0, "rmse": 3.0},
            {"max_abs": 2.0, "mean_abs": 1.0, "rmse": 3.0},
            {"max_abs": 0.0, "mean_abs": 0.0, "rmse": 1.0},
        )
        for row in bad_rows:
            with self.subTest(row=row), self.assertRaises(AcceptanceError):
                _error_metrics({"elements": 4, **row}, "fixture")

    def _write_csv(self, path: Path, **overrides: object) -> None:
        stages = (
            ("patch_projection", 196 * 768),
            ("block_0.attention_residual", 197 * 768),
            ("block_0.ffn_residual", 197 * 768),
            ("cls", 768),
            ("logits", 1000),
        )
        rows: list[dict[str, object]] = []
        for stage, elements in stages:
            rows.append({
                "stage": stage,
                "elements": elements,
                "observer_ell": 43,
                "observer_scale": 18,
                "clear_quantization_bound": 0.5 / (1 << 18),
                "fixed_secure_max_abs": 0.0,
                "fixed_secure_mean_abs": 0.0,
                "fixed_secure_rmse": 0.0,
                "clear_secure_max_abs": 4.0,
                "clear_secure_mean_abs": 2.0,
                "clear_secure_rmse": 3.0,
                "time_ms": 1.0,
                "bytes": 0,
                "logical_rounds": 0,
                "rmse_square_scale": 36,
                "intermediate_reconstruction": "false",
                "rmse_split_bits": (elements - 1).bit_length(),
                "rmse_accumulator": "exact_canonical_two_limb",
                "status": "PASS",
            })
        rows[0].update(overrides)
        fieldnames = list(EXPECTED_FIELDS)
        if "unexpected_payload" in overrides:
            fieldnames.append("unexpected_payload")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_csv_requires_exact_accumulator_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.csv"
            self._write_csv(path)
            self.assertEqual(validate(path, 1), 5)

            for overrides in (
                {"clear_secure_rmse": 1.0},
                {"fixed_secure_rmse": 5e-13},
                {"rmse_square_scale": 18},
                {"rmse_split_bits": 3},
                {"rmse_accumulator": "per_element_truncated"},
            ):
                with self.subTest(overrides=overrides):
                    self._write_csv(path, **overrides)
                    with self.assertRaises(ValueError):
                        validate(path, 1)

            self._write_csv(path, unexpected_payload="forbidden")
            with self.assertRaises(ValueError):
                validate(path, 1)


if __name__ == "__main__":
    unittest.main()
