#!/usr/bin/env python3

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[3]
run = root / (sys.argv[1] if len(sys.argv) > 1 else
              "outputs/vit_b16_blb/gpu_softmaxv_context_full_diag_20260829")
probability = np.loadtxt(
    run / "diagnostic_tensor_block_0.softmax_row_party1.txt",
    dtype=np.float64).reshape(12, 256, 256)
v = np.loadtxt(
    run / "diagnostic_tensor_block_0.v_packed_party1.txt",
    dtype=np.float64).reshape(12, 16384)
actual = np.loadtxt(
    run / "softmax_v_context_packed.txt",
    dtype=np.float64).reshape(12, 16384)

packed_probability = np.zeros((48, 16384), dtype=np.float64)
for pair in range(6):
    for diagonal_group in range(8):
        diagonals = range(diagonal_group * 32, (diagonal_group + 1) * 32)
        left = np.concatenate([
            probability[2 * pair, np.arange(256),
                        (np.arange(256) + diagonal) % 256]
            for diagonal in diagonals])
        right = np.concatenate([
            probability[2 * pair + 1, np.arange(256),
                        (np.arange(256) + diagonal) % 256]
            for diagonal in diagonals])
        packed_probability[pair * 8 + diagonal_group] = np.concatenate(
            [left, right])

softmax_input_path = run / "softmax_v_context_packed.txt.softmax_input.txt"
v_input_path = run / "softmax_v_context_packed.txt.v_input.txt"
softmax_for_simulation = packed_probability
v_for_simulation = v
if softmax_input_path.exists() and v_input_path.exists():
    imported_probability = np.loadtxt(
        softmax_input_path, dtype=np.float64).reshape(48, 16384)
    imported_v = np.loadtxt(v_input_path, dtype=np.float64).reshape(12, 16384)
    for name, imported, expected_input in (
            ("softmax_input", imported_probability, packed_probability),
            ("v_input", imported_v, v)):
        input_delta = imported - expected_input
        worst = np.unravel_index(
            int(np.argmax(np.abs(input_delta))), input_delta.shape)
        print(
            name, "actual_min", imported.min(), "actual_max", imported.max(),
            "expected_min", expected_input.min(), "expected_max", expected_input.max(),
            "max", np.abs(input_delta).max(),
            "rmse", np.sqrt(np.mean(input_delta * input_delta)),
            "corr", np.corrcoef(imported.reshape(-1), expected_input.reshape(-1))[0, 1],
            "worst", worst, "actual", imported[worst],
            "expected", expected_input[worst])
    softmax_for_simulation = imported_probability
    v_for_simulation = imported_v

def rotate(values, steps):
    return np.roll(values, -steps)

simulated = np.zeros_like(actual)
for pair in range(6):
    for chunk in range(2):
        total = np.zeros(16384, dtype=np.float64)
        for diagonal in range(256):
            source = softmax_for_simulation[pair * 8 + diagonal // 32]
            start = (diagonal % 32) * 256
            mask_left = np.zeros(16384, dtype=np.float64)
            mask_right = np.zeros(16384, dtype=np.float64)
            for half in (0, 8192):
                mask_left[half + start:half + start + 256 - diagonal] = 1.0
                mask_right[
                    half + start + 256 - diagonal:half + start + 256] = 1.0
            routed_left = source * mask_left
            routed_right = source * mask_right
            alignment = (diagonal % 32 - 31) * 256
            if alignment:
                routed_left = rotate(routed_left, alignment)
                routed_right = rotate(routed_right, alignment)
            for shift in (256, 512, 1024, 2048, 4096):
                routed_left += rotate(routed_left, shift)
                routed_right += rotate(routed_right, shift)
            total += (
                rotate(v_for_simulation[2 * pair + chunk], diagonal) * routed_left
                + rotate(v_for_simulation[2 * pair + chunk], diagonal - 256) * routed_right)
        simulated[2 * pair + chunk] = total

delta = actual - simulated
print("actual", actual.min(), actual.max(), actual.mean(), actual.std())
print("simulated", simulated.min(), simulated.max(), simulated.mean(), simulated.std())
print("error", np.abs(delta).max(), np.sqrt(np.mean(delta * delta)),
      np.corrcoef(actual.reshape(-1), simulated.reshape(-1))[0, 1])
prefinal_path = run / "softmax_v_context_packed.txt.prefinal.txt"
if prefinal_path.exists():
    prefinal = np.loadtxt(prefinal_path, dtype=np.float64).reshape(12, 16384)
    prefinal_delta = prefinal - simulated
    final_delta = actual - prefinal
    print("prefinal", prefinal.min(), prefinal.max(), prefinal.mean(), prefinal.std(),
          "vs_sim_max", np.abs(prefinal_delta).max(),
          "vs_sim_rmse", np.sqrt(np.mean(prefinal_delta * prefinal_delta)),
          "vs_sim_corr", np.corrcoef(prefinal.reshape(-1), simulated.reshape(-1))[0, 1])
    print("final_vs_prefinal", "max", np.abs(final_delta).max(),
          "rmse", np.sqrt(np.mean(final_delta * final_delta)),
          "corr", np.corrcoef(actual.reshape(-1), prefinal.reshape(-1))[0, 1])
for ct in range(12):
    d = delta[ct]
    worst = int(np.argmax(np.abs(d)))
    keep = np.ones(16384, dtype=bool)
    keep[worst] = False
    print("ct", ct, "actual_std", actual[ct].std(), "sim_std", simulated[ct].std(),
          "max", np.abs(d).max(), "rmse", np.sqrt(np.mean(d * d)),
          "corr", np.corrcoef(actual[ct], simulated[ct])[0, 1],
          "worst_slot", worst, "worst_error", d[worst],
          "rmse_without_worst", np.sqrt(np.mean(d[keep] * d[keep])))
