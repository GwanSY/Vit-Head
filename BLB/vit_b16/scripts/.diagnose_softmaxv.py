#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vit_reference as vr

root = Path(__file__).resolve().parents[2]
model = vr.load_model(root / "checkpoints/vit_b16/bolt_model.txt")
pixels = vr.load_input(root / "inputs/vit_b16/bob_image.txt")
graph = vr.Graph(model, fixed=True, block_limit=1)
_, records = graph.run(pixels)
expected = records["block_0.attention_context"]["value"]

run = root / (sys.argv[1] if len(sys.argv) > 1 else
              "outputs/vit_b16_blb/gpu_softmaxv_context_full_diag_20260829")
path = run / "softmax_v_context_packed.txt"
packed = np.loadtxt(path, dtype=np.float64).reshape(12, 16384)

def unpack_head_pairs(values):
    result = np.zeros((256, 768), dtype=np.float64)
    for pair in range(6):
        for chunk in range(2):
            ciphertext = values[2 * pair + chunk]
            for lane in range(2):
                slots = ciphertext[lane * 8192:(lane + 1) * 8192]
                head = 2 * pair + lane
                result[:, head * 64 + chunk * 32:head * 64 + (chunk + 1) * 32] = (
                    slots.reshape(32, 256).T)
    return result

actual = unpack_head_pairs(packed)
softmax = np.loadtxt(
    run / "diagnostic_tensor_block_0.softmax_row_party1.txt",
    dtype=np.float64).reshape(12, 256, 256)
v_packed = np.loadtxt(
    run / "diagnostic_tensor_block_0.v_packed_party1.txt",
    dtype=np.float64).reshape(12, 16384)
v_token_major = unpack_head_pairs(v_packed)
secure_input_expected = np.zeros((256, 768), dtype=np.float64)
for head in range(12):
    secure_input_expected[:, head * 64:(head + 1) * 64] = (
        softmax[head] @ v_token_major[:, head * 64:(head + 1) * 64])

def metrics(name, lhs, rhs):
    delta = lhs - rhs
    print(
        name,
        "actual_min", float(lhs.min()),
        "actual_max", float(lhs.max()),
        "actual_mean", float(lhs.mean()),
        "actual_std", float(lhs.std()),
        "expected_std", float(rhs.std()),
        "max_abs", float(np.abs(delta).max()),
        "rmse", float(np.sqrt(np.mean(delta * delta))),
        "corr", float(np.corrcoef(lhs.reshape(-1), rhs.reshape(-1))[0, 1]),
    )

metrics("all", actual, expected)
metrics("secure_input_all", actual, secure_input_expected)
metrics("secure_input_valid", actual[:197], secure_input_expected[:197])
metrics("valid", actual[:197], expected[:197])
metrics("cls", actual[0], expected[0])
metrics("noncls", actual[1:197], expected[1:197])
for head in range(12):
    metrics(
        f"head_{head}", actual[:197, head * 64:(head + 1) * 64],
        expected[:197, head * 64:(head + 1) * 64])

flat_error = np.abs(actual - expected)
for index in np.argpartition(flat_error.reshape(-1), -30)[-30:][::-1]:
    token, feature = np.unravel_index(index, flat_error.shape)
    print(
        "top_error", "token", token, "feature", feature,
        "head", feature // 64, "head_feature", feature % 64,
        "actual", float(actual[token, feature]),
        "expected", float(expected[token, feature]),
        "abs", float(flat_error[token, feature]))
