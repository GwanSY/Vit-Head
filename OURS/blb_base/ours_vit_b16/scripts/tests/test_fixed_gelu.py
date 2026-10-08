#!/usr/bin/env python3
"""Numerical contracts for BLB Algorithm 4 GELU."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import math
import unittest

import numpy as np

from vit_reference import fixed_gelu, quantized


class FixedGeluTest(unittest.TestCase):
    def test_tail_is_exact_relu(self) -> None:
        values = np.asarray([-9.0, -3.0, -2.701, 2.701, 3.0, 9.0])
        np.testing.assert_array_equal(
            fixed_gelu(values), np.maximum(quantized(values, 12), 0.0))

    def test_blb_polynomial_tracks_fp32_gelu(self) -> None:
        values = np.linspace(-2.699, 2.699, 20001)
        expected = 0.5 * values * np.asarray([
            1.0 + math.erf(value / math.sqrt(2.0)) for value in values
        ])
        actual = fixed_gelu(values)
        self.assertLess(float(np.max(np.abs(actual - expected))), 0.04)

    def test_original_polynomial_at_zero(self) -> None:
        # BLB's constant 0.001620808531841547 rounds to 7 at scale 12;
        # the removed even-quartic constant rounds to 113.
        self.assertEqual(float(fixed_gelu(np.asarray([0.0]))[0]), 7.0 / 4096.0)

    def test_central_odd_component_is_x(self) -> None:
        positive = np.linspace(0.0, 2.6, 101)
        difference = fixed_gelu(positive) - fixed_gelu(-positive)
        # Two signed arithmetic shifts may differ by one fixed-point unit.
        np.testing.assert_allclose(difference, positive, atol=2.0 / 4096.0,
                                   rtol=0.0)


if __name__ == "__main__":
    unittest.main()
