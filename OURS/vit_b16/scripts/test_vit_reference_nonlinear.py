#!/usr/bin/env python3
"""Known-vector tests for the bit-exact ViT fixed nonlinear reference."""

from __future__ import annotations

import unittest

import numpy as np

import vit_reference as reference


class FixedNonlinearReferenceTest(unittest.TestCase):
    def test_ring_helpers_match_signed_reduce_and_truncate(self) -> None:
        raw = reference._ring_reduce(np.asarray([-1, -3, 7]), 5)
        np.testing.assert_array_equal(raw, [31, 29, 7])
        np.testing.assert_array_equal(reference._ring_signed(raw, 5),
                                      [-1, -3, 7])
        shifted = reference._ring_truncate_reduce(raw, 5, 1)
        np.testing.assert_array_equal(shifted, [15, 14, 3])
        np.testing.assert_array_equal(reference._ring_signed(shifted, 4),
                                      [-1, -2, 3])
        # C++ casts these model constants toward zero, not to nearest.
        self.assertEqual(int(reference._cpp_fixed_constant(
            4.5702822654246535, 11, 20)), 9359)

    def test_gelu_matches_two_party_cpp_oracle(self) -> None:
        input_raw_scale11 = np.asarray([
            -8192, -5529, -5528, -3073, -2048, -1, 0,
            1, 2048, 3073, 5528, 5529, 8192,
        ], dtype=np.int64)
        expected_raw_scale4 = np.asarray([
            0, 0, -1, -2, -3, 0, 0, 0, 13, 22, 42, 43, 64,
        ], dtype=np.int64)
        actual = reference.round_int(
            reference.fixed_gelu(input_raw_scale11 / float(1 << 11)), 4)
        np.testing.assert_array_equal(actual, expected_raw_scale4)

    def test_exp4_matches_two_party_cpp_oracle(self) -> None:
        input_raw = np.asarray([
            -65536, -32768, -16384, -8192, -4096,
            -2839, -2048, -1, 0,
        ], dtype=np.int64)
        expected_raw = np.asarray(
            [0, 1, 74, 552, 1507, 2048, 2477, 4095, 4096],
            dtype=np.int64)
        actual = reference.round_int(
            reference.fixed_exp4(input_raw / float(1 << 12)), 12)
        np.testing.assert_array_equal(actual, expected_raw)

    def test_iron_softmax_matches_two_party_cpp_oracle(self) -> None:
        input_raw = np.asarray(
            [[0, -1, -2048, -4096, -8192, -16384, -32768]],
            dtype=np.int64)
        expected_raw = np.asarray(
            [[1309, 1308, 793, 481, 177, 23, 0]], dtype=np.int64)
        actual = reference.round_int(
            reference.fixed_softmax(input_raw / float(1 << 12)), 12)
        np.testing.assert_array_equal(actual, expected_raw)

    def test_uniform_197_uses_nonwrapping_37_bit_sum(self) -> None:
        # The legacy 19-bit sum would wrap 197*4096.  The selected existing
        # softmax_fix_iron_1 path returns this oracle-verified 37-bit result.
        actual = reference.round_int(
            reference.fixed_softmax(np.zeros((1, 197), dtype=np.float64)), 12)
        self.assertEqual(actual.shape, (1, 197))
        self.assertTrue(np.all(actual == 20))
        self.assertEqual(int(actual.sum()), 3940)


if __name__ == "__main__":
    unittest.main()
