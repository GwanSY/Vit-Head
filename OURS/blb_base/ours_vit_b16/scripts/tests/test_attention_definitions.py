"""Boundary cases that distinguish the frozen R/L functions."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vit_reference import fixed_relu_normalization, fixed_linear_scaling


class AttentionDefinitionsTest(unittest.TestCase):
    def test_relu_zero_row_and_positive_normalization(self):
        actual = fixed_relu_normalization(np.array([
            [-2., -1., 0.], [1., 3., -2.],
        ]))
        np.testing.assert_array_equal(actual[0], [0., 0., 0.])
        np.testing.assert_allclose(actual[1], [.25, .75, 0.], atol=2 ** -10, rtol=0)

    def test_linear_preserves_sign_and_has_no_row_normalization(self):
        actual = fixed_linear_scaling(np.array([[-2., 4., 6.]]), .5)
        np.testing.assert_array_equal(actual, [[-1., 2., 3.]])


if __name__ == '__main__':
    unittest.main()
