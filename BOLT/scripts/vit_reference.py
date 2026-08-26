#!/usr/bin/env python3
"""Independent floating/fixed-clear reference for the canonical BOLT ViT file.

The floating graph is the standard pre-LayerNorm ViT equation used by the
PyTorch/Hugging Face and timm implementations.  The fixed graph is a clear
simulation of BOLT's public fixed-point schedule: 38-bit additive values,
scale 12 activations/weights, scale 4 output from ``gelu_approx_2``, and
physical power-of-two token packing.  It is intentionally not a secure
observer and never reads or reconstructs a secure intermediate.

For a real checkpoint, the output is a validation artifact rather than a
claim that secure numerical equality has been established.  Secure rows are
marked unavailable unless a separately produced, aggregate-only validation
file is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:  # scipy is already used by the repository's FP32 evaluator.
    from scipy.special import erf as _erf
except ImportError:  # pragma: no cover - fallback for minimal environments
    _erf = np.vectorize(math.erf, otypes=[np.float64])


ELL = 38
ACT_SCALE = 12
WEIGHT_SCALE = 12
GELU_INPUT_SCALE = 11
GELU_OUTPUT_SCALE = 4


class TokenReader:
    def __init__(self, path: Path):
        self.tokens = path.read_text(encoding="utf-8").split()
        self.index = 0

    def take(self) -> str:
        if self.index >= len(self.tokens):
            raise ValueError("unexpected end of canonical text file")
        value = self.tokens[self.index]
        self.index += 1
        return value

    def expect(self, value: str) -> None:
        got = self.take()
        if got != value:
            raise ValueError(f"expected {value!r}, got {got!r}")

    def integer(self, name: str) -> int:
        self.expect(name)
        return int(self.take())

    def real(self, name: str) -> float:
        self.expect(name)
        return float(self.take())

    def vector(self, name: str, count: int) -> np.ndarray:
        self.expect(name)
        values = np.asarray([float(self.take()) for _ in range(count)],
                            dtype=np.float64)
        return values


def load_model(path: Path) -> dict[str, Any]:
    reader = TokenReader(path)
    magic = reader.take()
    if magic not in {
        "BOLT_VIT_MODEL_V1", "BOLT_VIT_MODEL_V2", "BOLT_VIT_MODEL_V3",
        "IRON_VIT_MODEL_V1", "IRON_VIT_MODEL_V2", "IRON_VIT_MODEL_V3",
    }:
        raise ValueError(f"unsupported canonical model magic {magic}")
    return _load_model_after_magic(path, magic)


def _load_model_after_magic(path: Path, magic: str) -> dict[str, Any]:
    # Parsing from a fresh reader keeps the metadata path explicit and avoids
    # accepting an accidental token shift in a model produced by a converter.
    reader = TokenReader(path)
    if reader.take() != magic:
        raise ValueError("canonical model changed while being parsed")
    versioned = not magic.endswith("_V1")
    model: dict[str, Any] = {
        "magic": magic,
        "source_id": "unknown",
        "source_revision": "unknown",
        "source_sha256": "not_applicable",
        "architecture": "pre_norm_vit",
        "norm_order": "pre",
        "layer_norm_epsilon": 1e-12,
        "qkv_bias": False,
    }
    if versioned:
        reader.expect("source_id")
        model["source_id"] = reader.take()
        reader.expect("source_revision")
        model["source_revision"] = reader.take()
        reader.expect("source_sha256")
        model["source_sha256"] = reader.take()
        reader.expect("architecture")
        model["architecture"] = reader.take()
        reader.expect("norm_order")
        model["norm_order"] = reader.take()
        model["layer_norm_epsilon"] = reader.real("layer_norm_epsilon")
        model["qkv_bias"] = bool(reader.integer("qkv_bias"))
    fields = ("image_size", "patch_size", "channels", "tokens", "dim",
              "heads", "ff_dim", "layers", "classes", "scale_bits")
    for field in fields:
        model[field] = reader.integer(field)
    if model["scale_bits"] != ACT_SCALE:
        raise ValueError("canonical reference currently requires scale_bits=12")
    dim = model["dim"]
    ff_dim = model["ff_dim"]
    patch_dim = model["channels"] * model["patch_size"] ** 2
    model["patch_weight"] = reader.vector("patch_weight", dim * patch_dim).reshape(dim, patch_dim)
    model["patch_bias"] = reader.vector("patch_bias", dim)
    model["class_token"] = reader.vector("class_token", dim)
    model["position"] = reader.vector("position", model["tokens"] * dim).reshape(model["tokens"], dim)
    blocks: list[dict[str, np.ndarray]] = []
    square = dim * dim
    for layer in range(model["layers"]):
        if reader.take() != "block" or int(reader.take()) != layer:
            raise ValueError(f"unexpected block {layer}")
        block: dict[str, np.ndarray] = {}
        block["ln1_weight"] = reader.vector("ln1_weight", dim)
        block["ln1_bias"] = reader.vector("ln1_bias", dim)
        block["wq"] = reader.vector("wq", square).reshape(dim, dim)
        block["q_bias"] = reader.vector("q_bias", dim) if versioned else np.zeros(dim)
        block["wk"] = reader.vector("wk", square).reshape(dim, dim)
        block["k_bias"] = reader.vector("k_bias", dim) if versioned else np.zeros(dim)
        block["wv"] = reader.vector("wv", square).reshape(dim, dim)
        block["v_bias"] = reader.vector("v_bias", dim) if versioned else np.zeros(dim)
        block["wo"] = reader.vector("wo", square).reshape(dim, dim)
        block["bo"] = reader.vector("bo", dim)
        block["ln2_weight"] = reader.vector("ln2_weight", dim)
        block["ln2_bias"] = reader.vector("ln2_bias", dim)
        block["ff1_weight"] = reader.vector("ff1_weight", ff_dim * dim).reshape(ff_dim, dim)
        block["ff1_bias"] = reader.vector("ff1_bias", ff_dim)
        block["ff2_weight"] = reader.vector("ff2_weight", dim * ff_dim).reshape(dim, ff_dim)
        block["ff2_bias"] = reader.vector("ff2_bias", dim)
        blocks.append(block)
    model["blocks"] = blocks
    model["final_ln_weight"] = reader.vector("final_ln_weight", dim)
    model["final_ln_bias"] = reader.vector("final_ln_bias", dim)
    model["classifier_weight"] = reader.vector("classifier_weight", model["classes"] * dim).reshape(model["classes"], dim)
    model["classifier_bias"] = reader.vector("classifier_bias", model["classes"])
    reader.expect("end")
    if reader.index != len(reader.tokens):
        raise ValueError("unexpected data after canonical model terminator")
    if model["norm_order"] != "pre":
        raise ValueError("only checkpoint-declared pre-LayerNorm is supported")
    return model


def load_input(path: Path) -> np.ndarray:
    reader = TokenReader(path)
    magic = reader.take()
    if magic not in {"BOLT_VIT_INPUT_V1", "IRON_VIT_INPUT_V1"}:
        raise ValueError(f"unsupported canonical input magic {magic}")
    size = reader.integer("image_size")
    channels = reader.integer("channels")
    image = reader.vector("pixels", channels * size * size).reshape(channels, size, size)
    reader.expect("end")
    if reader.index != len(reader.tokens):
        raise ValueError("unexpected data after canonical input terminator")
    return image


def round_int(value: np.ndarray | float, scale: int) -> np.ndarray:
    """Match C++ llround for ordinary finite model values."""
    factor = float(1 << scale)
    array = np.asarray(value, dtype=np.float64)
    return np.where(array >= 0.0, np.floor(array * factor + 0.5),
                    np.ceil(array * factor - 0.5)).astype(np.int64)


def quantized(value: np.ndarray, scale: int) -> np.ndarray:
    return round_int(value, scale).astype(np.float64) / float(1 << scale)


def matmul_integer(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    # The multiplication is evaluated in float64 only to avoid NumPy int64
    # overflow.  The operands are exact integers and the values encountered by
    # this fixed reference are below the 53-bit exact-integer range.
    return np.rint(np.matmul(lhs.astype(np.float64), rhs.astype(np.float64)))


def _floor_shift(value: np.ndarray | int, shift: int) -> np.ndarray:
    """Arithmetic fixed-point truncation used by SCI's signed protocol."""
    array = np.asarray(value, dtype=np.int64)
    if shift <= 0:
        return array << (-shift)
    return np.floor_divide(array, 1 << shift)


def _signed_reduce(value: np.ndarray, bit_length: int) -> np.ndarray:
    mask = (1 << bit_length) - 1
    reduced = np.asarray(value, dtype=np.int64) & mask
    return np.where(reduced >= (1 << (bit_length - 1)),
                    reduced - (1 << bit_length), reduced)


def fixed_linear(value: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                 input_scale: int, output_scale: int) -> np.ndarray:
    x_int = round_int(value, input_scale).astype(np.int64)
    w_int = round_int(weight, WEIGHT_SCALE).astype(np.int64)
    raw = np.matmul(x_int, w_int.T)
    raw_scale = input_scale + WEIGHT_SCALE
    if bias.size:
        raw = raw + round_int(bias, raw_scale).astype(np.int64)
    output_int = _floor_shift(raw, raw_scale - output_scale)
    return output_int.astype(np.float64) / float(1 << output_scale)


def fixed_add(lhs: np.ndarray, rhs: np.ndarray, scale: int) -> np.ndarray:
    value = (round_int(lhs, scale) + round_int(rhs, scale)).astype(np.int64)
    return value.astype(np.float64) / float(1 << scale)


def layer_norm(value: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               epsilon: float) -> np.ndarray:
    mean = value.mean(axis=-1, keepdims=True, dtype=np.float64)
    variance = ((value - mean) ** 2).mean(axis=-1, keepdims=True,
                                          dtype=np.float64)
    safe_epsilon = epsilon if epsilon > 0.0 else 1e-12
    normalized = (value - mean) / np.sqrt(variance + safe_epsilon)
    return normalized * weight + bias


def _bolt_reciprocal_sqrt(values: np.ndarray) -> np.ndarray:
    """Clear arithmetic form of SCI MathFunctions::sqrt(..., inverse=true)."""
    values = np.asarray(values, dtype=np.int64)
    result = np.zeros(values.size, dtype=np.int64)
    m = math.ceil(ACT_SCALE / 2.0)
    for index, raw_value in enumerate(values.reshape(-1)):
        value = int(raw_value)
        if value <= 0:
            # Padding rows are masked after the call.  Their secure value is
            # irrelevant, while zero keeps the clear reference well-defined.
            continue
        msnzb = value.bit_length() - 1
        adjusted = value << (38 - 2 - msnzb)
        adjusted_m = (adjusted >> (38 - m - 2)) & ((1 << m) - 1)
        exponent_parity = (msnzb - ACT_SCALE) & 1
        normalized = (1.0 + adjusted_m / float(1 << m)) * (1 << exponent_parity)
        y = int((1.0 / math.sqrt(normalized)) * (1 << (m + 2)))

        x = adjusted << exponent_parity
        b_previous = x >> (38 - 2 - ACT_SCALE)
        b_current = (y * y * b_previous) >> (2 * m + 4)
        y_current = (3 << ACT_SCALE) - b_current
        x_current = (y * y_current) >> (m + 2 + 1)

        adjust_scale = (38 - 1 - ACT_SCALE) // 2
        sqrt_adjust = 1 << ((ACT_SCALE - msnzb + 1) // 2 + adjust_scale)
        result[index] = (x_current * sqrt_adjust) >> adjust_scale
    return result.reshape(values.shape)


def _fixed_layer_norm_normalized(value: np.ndarray, epsilon: float) -> np.ndarray:
    """Match the statistical output of BOLT's FPMath::layer_norm_fix.

    The BOLT/BERT path deliberately leaves the affine parameters out of this
    MPC helper.  The secure graph applies gamma in the following packed HE
    affine and adds beta as a local share operation.
    """
    x_int = round_int(value, ACT_SCALE).astype(np.int64)
    width = x_int.shape[-1]
    denominator = int((1 << (2 * ACT_SCALE)) / width)
    total = x_int.sum(axis=-1, dtype=np.int64)
    average = _floor_shift(total * denominator, 2 * ACT_SCALE)
    centered = x_int - average[..., None]
    square = _floor_shift(centered * centered, ACT_SCALE)
    variance = _floor_shift(
        square.sum(axis=-1, dtype=np.int64) * denominator, 2 * ACT_SCALE)
    epsilon_lsb = int(round(epsilon * float(1 << ACT_SCALE)))
    if epsilon_lsb:
        variance = variance + epsilon_lsb
    inverse_sigma = _bolt_reciprocal_sqrt(variance)
    normalized = _floor_shift(centered * inverse_sigma[..., None], ACT_SCALE)
    return normalized.astype(np.float64) / float(1 << ACT_SCALE)


def fixed_layer_norm(value: np.ndarray, weight: np.ndarray,
                     bias: np.ndarray, epsilon: float) -> np.ndarray:
    """Match BOLT LayerNorm statistics, HE gamma, and local beta."""
    normalized = round_int(
        _fixed_layer_norm_normalized(value, epsilon), ACT_SCALE)

    gamma = round_int(weight, WEIGHT_SCALE).astype(np.int64)
    beta = round_int(bias, ACT_SCALE).astype(np.int64)
    affine = _floor_shift(normalized * gamma, WEIGHT_SCALE) + beta
    return affine.astype(np.float64) / float(1 << ACT_SCALE)


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - value.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted, dtype=np.float64)
    return exponent / exponent.sum(axis=-1, keepdims=True, dtype=np.float64)


def _fixed_exp4_int(value: np.ndarray) -> np.ndarray:
    """Integer form of FPMath::exp4 (scale 12, with local truncations)."""
    # This low-level helper consumes the already quantized signed integer
    # representation.  Callers operating on real-valued arrays must quantize
    # exactly once before reaching it.
    x = np.asarray(value, dtype=np.int64)
    x_times_inv_ln2 = _floor_shift(x * -5909, ACT_SCALE)
    l_short = np.maximum(_floor_shift(x_times_inv_ln2, ACT_SCALE), 0)
    p = _signed_reduce(x + l_short * 2839, ACT_SCALE + 2)
    p_arg2_square = _floor_shift((p + 5542) * (p + 5542), ACT_SCALE)
    polynomial = _floor_shift(p_arg2_square * 1468, ACT_SCALE) + 1409
    shift = np.minimum(l_short, 13).astype(np.int64)
    return np.floor_divide(polynomial, np.left_shift(1, shift))


def fixed_exp4(value: np.ndarray) -> np.ndarray:
    return _fixed_exp4_int(round_int(value, ACT_SCALE)).astype(np.float64) / float(1 << ACT_SCALE)


def _fixed_div_batch_int(numerator: np.ndarray,
                         denominator: np.ndarray) -> np.ndarray:
    """Integer form of FixOp::div_batch for BOLT's scale-12 softmax."""
    numerator = np.asarray(numerator, dtype=np.int64)
    denominator = np.asarray(denominator, dtype=np.int64)
    result = np.empty_like(numerator)
    batch = numerator.shape[-1]
    groups = numerator.size // batch
    exponent_ell = max(19, ACT_SCALE + int(math.ceil(math.log2(batch))))
    m = math.ceil((ACT_SCALE - 2) / 2.0)
    adjustment_scale = exponent_ell - 1 - ACT_SCALE
    for group in range(groups):
        divisor = int(denominator.reshape(-1)[group])
        if divisor <= 0:
            raise ValueError("BOLT softmax denominator must be positive")
        most_significant = divisor.bit_length() - 1
        adjustment = 1 << (exponent_ell - 1 - most_significant)
        normalized = divisor * adjustment
        # FixOp::div_batch uses the declared normalized scale (ell-1), not
        # the value-dependent bit length, for the LUT index.
        normalized_scale = exponent_ell - 1
        eps = normalized & ((1 << (normalized_scale - m)) - 1)
        index = (normalized >> (normalized_scale - m)) & ((1 << m) - 1)
        k = 1 << m
        point = 1.0 + index / float(k)
        c0 = int((1.0 / (point * (point + 1.0 / k))) * (1 << (m + 3)))
        z = point * (point + 1.0 / k)
        c1 = int(((1.0 / (2.0 * k) + math.sqrt(z)) / z) *
                 (1 << (2 * m + 2)))
        # FPMath builds w at scale ``normalized_scale + m + 3`` and then
        # truncates it to the requested output scale.  For the ViT/BERT
        # scale-12 path this is a 14-bit shift, not a fixed 15-bit shift.
        w_scale = normalized_scale + m + 3
        c1_shift = w_scale - (2 * m + 2)
        reciprocal = _floor_shift(
            (c1 << c1_shift) - c0 * eps, w_scale - ACT_SCALE)
        start = group * batch
        numerators = numerator.reshape(-1)[start:start + batch]
        result.reshape(-1)[start:start + batch] = _floor_shift(
            _floor_shift(numerators * reciprocal, ACT_SCALE) * adjustment,
            adjustment_scale)
    return result


def fixed_softmax(value: np.ndarray) -> np.ndarray:
    x = round_int(value, ACT_SCALE).astype(np.int64)
    shifted = x - x.max(axis=-1, keepdims=True)
    exponent = _fixed_exp4_int(shifted)
    normalized = _fixed_div_batch_int(
        exponent, exponent.sum(axis=-1, dtype=np.int64))
    return normalized.astype(np.float64) / float(1 << ACT_SCALE)


def fixed_gelu(value: np.ndarray) -> np.ndarray:
    # Constants and operation boundaries are copied from FPMath::gelu_approx_2.
    x = _floor_shift(round_int(value, ACT_SCALE), ACT_SCALE - GELU_INPUT_SCALE)
    y = np.abs(x)
    constants = [0.14439048359960427, -0.7077117131613893,
                 4.5702822654246535, -8.15444702051307,
                 16.382265425072532, 0.5]
    c1, c2, c3, c4, c5, c6 = [
        int(constant * (1 << GELU_INPUT_SCALE)) for constant in constants]
    y_c1 = _floor_shift(y * c1, GELU_INPUT_SCALE)
    z = _floor_shift((y_c1 + c2) * y, GELU_INPUT_SCALE) + c3
    p = _floor_shift((z + y_c1 + c4) * z, GELU_INPUT_SCALE) + c5
    low = _floor_shift(x * c6, GELU_INPUT_SCALE) + p
    high = _floor_shift(x + y, 1)
    result = np.where(y < int(2.7 * (1 << GELU_INPUT_SCALE)), low, high)
    return _floor_shift(result, 7).astype(np.float64) / float(1 << GELU_OUTPUT_SCALE)


def patchify(chw: np.ndarray, patch_size: int) -> np.ndarray:
    channels, height, width = chw.shape
    side = height // patch_size
    # Explicit indexing documents the checkpoint's C,KY,KX within each
    # row-major (PY,PX) patch; it avoids depending on a framework flattening
    # convention.
    patches = np.empty((side * side, channels * patch_size * patch_size),
                       dtype=np.float64)
    token = 0
    for py in range(side):
        for px in range(side):
            patches[token] = chw[:, py * patch_size:(py + 1) * patch_size,
                                  px * patch_size:(px + 1) * patch_size].reshape(-1)
            token += 1
    return patches


def next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result <<= 1
    return result


class Graph:
    def __init__(self, model: dict[str, Any], fixed: bool,
                 physical_tokens: int | None = None):
        self.model = model
        self.fixed = fixed
        self.logical = int(model["tokens"])
        self.physical = (physical_tokens if physical_tokens is not None else
                         (next_power_of_two(self.logical) if fixed else self.logical))
        if self.physical < self.logical:
            raise ValueError("physical token count cannot be below logical count")
        self.records: dict[str, dict[str, Any]] = {}

    def record(self, name: str, value: np.ndarray, scale: int | str,
               packing: str) -> np.ndarray:
        self.records[name] = {
            "shape": list(value.shape),
            "scale": scale,
            "bit_length": ELL if self.fixed else 32,
            "packing": packing,
            "value": value.copy(),
        }
        return value

    def mask_tokens(self, value: np.ndarray) -> np.ndarray:
        if self.fixed and value.shape[0] > self.logical:
            value = value.copy()
            value[self.logical:] = 0.0
        return value

    def linear(self, value: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               input_scale: int, output_scale: int) -> np.ndarray:
        if self.fixed:
            return fixed_linear(value, weight, bias, input_scale, output_scale)
        return value @ weight.T + bias

    def add(self, lhs: np.ndarray, rhs: np.ndarray, scale: int) -> np.ndarray:
        return fixed_add(lhs, rhs, scale) if self.fixed else lhs + rhs

    def norm(self, value: np.ndarray, weight: np.ndarray,
             bias: np.ndarray) -> np.ndarray:
        if self.fixed:
            return self.mask_tokens(fixed_layer_norm(
                value, weight, bias, self.model["layer_norm_epsilon"]))
        return layer_norm(value, weight, bias, self.model["layer_norm_epsilon"])

    def attention(self, value: np.ndarray, block: dict[str, np.ndarray],
                  layer: int) -> np.ndarray:
        input_scale = ACT_SCALE if self.fixed else 0
        q = self.linear(value, block["wq"], block["q_bias"], input_scale, ACT_SCALE)
        k = self.linear(value, block["wk"], block["k_bias"], input_scale, ACT_SCALE)
        v = self.linear(value, block["wv"], block["v_bias"], input_scale, ACT_SCALE)
        self.record(f"block_{layer}.q", q, ACT_SCALE if self.fixed else "float64",
                    "physical_token_major_q" if self.fixed else "logical_token_major_q")
        self.record(f"block_{layer}.k", k, ACT_SCALE if self.fixed else "float64",
                    "physical_token_major_k" if self.fixed else "logical_token_major_k")
        self.record(f"block_{layer}.v", v, ACT_SCALE if self.fixed else "float64",
                    "physical_token_major_v" if self.fixed else "logical_token_major_v")
        head_dim = int(self.model["dim"]) // int(self.model["heads"])
        qh = q.reshape(self.physical, self.model["heads"], head_dim).transpose(1, 0, 2)
        kh = k.reshape(self.physical, self.model["heads"], head_dim).transpose(1, 0, 2)
        vh = v.reshape(self.physical, self.model["heads"], head_dim).transpose(1, 0, 2)
        if self.fixed:
            qi, ki = round_int(qh, ACT_SCALE), round_int(kh, ACT_SCALE)
            raw = matmul_integer(qi, np.swapaxes(ki, -1, -2))
            coeff = int(round(1.0 / math.sqrt(head_dim) * (1 << ACT_SCALE)))
            scores = np.floor(raw * coeff / float(1 << (2 * ACT_SCALE))) / float(1 << ACT_SCALE)
        else:
            scores = np.matmul(qh, np.swapaxes(kh, -1, -2)) / math.sqrt(head_dim)
        self.record(f"block_{layer}.qk", scores, ACT_SCALE if self.fixed else "float64",
                    "physical_head_cross_rows" if self.fixed else "logical_head_qk")
        if self.fixed:
            probability = np.zeros((self.model["heads"], self.physical, self.physical),
                                   dtype=np.float64)
            probability[:, :self.logical, :self.logical] = fixed_softmax(
                scores[:, :self.logical, :self.logical])
        else:
            probability = softmax(scores)
        self.record(f"block_{layer}.softmax", probability,
                    ACT_SCALE if self.fixed else "float64",
                    "physical_public_mask_valid_197" if self.fixed else "logical_head_probability")
        if self.fixed:
            context = np.zeros((self.model["heads"], self.physical, head_dim),
                               dtype=np.float64)
            probability_int = round_int(
                probability[:, :self.logical, :self.logical], ACT_SCALE)
            value_int = round_int(vh[:, :self.logical], ACT_SCALE)
            # Softmax and V are both represented at scale 12.  Their product
            # is therefore at scale 24, and BOLT's Softmax×V boundary shifts
            # it once by 12 bits back to the activation scale.  Shifting by
            # 24 here would apply the scale reduction twice and collapse the
            # attention context by another factor of 2**12.
            context_int = _floor_shift(
                np.matmul(probability_int, value_int), ACT_SCALE)
            context[:, :self.logical] = (
                context_int.astype(np.float64) / float(1 << ACT_SCALE))
        else:
            context = np.matmul(probability, vh)
        joined = context.transpose(1, 0, 2).reshape(self.physical, self.model["dim"])
        joined = self.mask_tokens(joined)
        self.record(f"block_{layer}.attention_context", joined,
                    ACT_SCALE if self.fixed else "float64",
                    "physical_merged_heads" if self.fixed else "logical_merged_heads")
        projected = self.linear(joined, block["wo"], block["bo"],
                                ACT_SCALE if self.fixed else 0, ACT_SCALE)
        return self.mask_tokens(projected)

    def run(self, pixels: np.ndarray) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
        patches = patchify(pixels, int(self.model["patch_size"]))
        if self.fixed:
            patches = quantized(patches, ACT_SCALE)
        self.record("patchify", patches, ACT_SCALE if self.fixed else "float64",
                    "chw_to_py_px_c_ky_kx" if self.fixed else "logical_patch_matrix")
        patch_tokens = self.linear(patches, self.model["patch_weight"],
                                   self.model["patch_bias"],
                                   ACT_SCALE if self.fixed else 0, ACT_SCALE)
        self.record("patch_projection", patch_tokens,
                    ACT_SCALE if self.fixed else "float64",
                    "compact_bsgs_valid_patches" if self.fixed else "logical_patch_projection")
        hidden = np.zeros((self.physical, self.model["dim"]), dtype=np.float64)
        hidden[0] = self.model["class_token"]
        hidden[1:self.logical] = patch_tokens
        hidden[:self.logical] += self.model["position"]
        if self.fixed:
            hidden = quantized(hidden, ACT_SCALE)
            hidden = self.mask_tokens(hidden)
        self.record("cls_position", hidden,
                    ACT_SCALE if self.fixed else "float64",
                    "physical_cls_then_raster_padding" if self.fixed else "logical_cls_then_raster")
        for layer, block in enumerate(self.model["blocks"]):
            normalized = self.norm(hidden, block["ln1_weight"], block["ln1_bias"])
            self.record(f"block_{layer}.ln1", normalized,
                        ACT_SCALE if self.fixed else "float64",
                        "physical_rowwise_layernorm" if self.fixed else "logical_rowwise_layernorm")
            attn = self.attention(normalized, block, layer)
            hidden = self.mask_tokens(self.add(hidden, attn, ACT_SCALE))
            self.record(f"block_{layer}.attention_residual", hidden,
                        ACT_SCALE if self.fixed else "float64",
                        "physical_public_token_mask" if self.fixed else "logical_token_major")
            normalized = self.norm(hidden, block["ln2_weight"], block["ln2_bias"])
            self.record(f"block_{layer}.ln2", normalized,
                        ACT_SCALE if self.fixed else "float64",
                        "physical_rowwise_layernorm" if self.fixed else "logical_rowwise_layernorm")
            first = self.linear(normalized, block["ff1_weight"], block["ff1_bias"],
                                ACT_SCALE if self.fixed else 0, ACT_SCALE)
            first = self.mask_tokens(first)
            self.record(f"block_{layer}.fc1", first,
                        ACT_SCALE if self.fixed else "float64",
                        "physical_ffn_bsgs" if self.fixed else "logical_ffn")
            activated = fixed_gelu(first) if self.fixed else (
                0.5 * first * (1.0 + _erf(first / math.sqrt(2.0))))
            activated = self.mask_tokens(activated)
            self.record(f"block_{layer}.gelu", activated,
                        GELU_OUTPUT_SCALE if self.fixed else "float64",
                        "physical_gelu_ell20_to_ell38" if self.fixed else "logical_gelu")
            second = self.linear(activated, block["ff2_weight"], block["ff2_bias"],
                                 GELU_OUTPUT_SCALE if self.fixed else 0, ACT_SCALE)
            hidden = self.mask_tokens(self.add(hidden, second, ACT_SCALE))
            self.record(f"block_{layer}.ffn_residual", hidden,
                        ACT_SCALE if self.fixed else "float64",
                        "physical_public_token_mask" if self.fixed else "logical_token_major")
        hidden = self.norm(hidden, self.model["final_ln_weight"],
                           self.model["final_ln_bias"])
        self.record("final_ln", hidden,
                    ACT_SCALE if self.fixed else "float64",
                    "physical_rowwise_layernorm" if self.fixed else "logical_rowwise_layernorm")
        cls = hidden[:1]
        self.record("cls", cls,
                    ACT_SCALE if self.fixed else "float64",
                    "public_index_cls_share" if self.fixed else "logical_cls")
        logits = self.linear(cls, self.model["classifier_weight"],
                             self.model["classifier_bias"],
                             ACT_SCALE if self.fixed else 0, ACT_SCALE)
        self.record("logits", logits,
                    ACT_SCALE if self.fixed else "float64",
                    "packed_classifier_stride" if self.fixed else "logical_classifier")
        return logits.reshape(-1), self.records


def valid_prefix(value: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if value.ndim == len(target_shape) and all(
            actual >= target for actual, target in zip(value.shape, target_shape)):
        return value[tuple(slice(0, target) for target in target_shape)]
    if value.ndim >= 1 and value.shape[0] >= target_shape[0]:
        return value[:target_shape[0]]
    return value


def error_row(clear: np.ndarray, fixed: np.ndarray, logical: int) -> dict[str, float | int]:
    lhs = np.asarray(clear, dtype=np.float64)
    rhs = np.asarray(valid_prefix(fixed, lhs.shape), dtype=np.float64)
    if lhs.shape != rhs.shape:
        return {"status": "SHAPE_MISMATCH", "clear_shape": list(lhs.shape),
                "fixed_shape": list(rhs.shape)}
    difference = rhs - lhs
    return {
        "status": "PASS",
        "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0,
        "mean_abs": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "elements": int(difference.size),
    }


def secure_placeholder() -> dict[str, Any]:
    return {
        "status": "NOT_AVAILABLE",
        "max_abs": None,
        "rmse": None,
        "reason": "formal secure graph does not reconstruct intermediate tensors; no aggregate validation observer was supplied",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--secure-prediction", type=Path,
                        help="optional BOB prediction file; supplies class only")
    parser.add_argument("--max-layers", type=int,
                        help="validation-only prefix of the checkpoint blocks")
    args = parser.parse_args()

    model = load_model(args.model)
    if args.max_layers is not None:
        if args.max_layers <= 0 or args.max_layers > int(model["layers"]):
            raise ValueError("--max-layers must be in [1, model.layers]")
        model["blocks"] = model["blocks"][:args.max_layers]
        model["layers"] = args.max_layers
    pixels = load_input(args.input)
    expected_size = int(model["image_size"])
    if pixels.shape != (model["channels"], expected_size, expected_size):
        raise ValueError(f"input shape {pixels.shape} != {(model['channels'], expected_size, expected_size)}")
    clear_logits, clear_records = Graph(model, fixed=False).run(pixels)
    fixed_graph = Graph(model, fixed=True)
    fixed_logits, fixed_records = fixed_graph.run(pixels)
    logical_fixed_graph = Graph(model, fixed=True,
                                physical_tokens=int(model["tokens"]))
    logical_fixed_logits, logical_fixed_records = logical_fixed_graph.run(pixels)
    clear_top1 = int(np.argmax(clear_logits))
    fixed_top1 = int(np.argmax(fixed_logits))
    secure_top1: int | None = None
    if args.secure_prediction:
        fields = args.secure_prediction.read_text(encoding="utf-8").split()
        if len(fields) == 2 and fields[0] == "predicted_class":
            secure_top1 = int(fields[1])

    operators: list[dict[str, Any]] = []
    for name, fixed_row in fixed_records.items():
        clear_row = clear_records.get(name)
        if clear_row is None:
            continue
        operators.append({
            "name": name,
            "shape": fixed_row["shape"],
            "scale": fixed_row["scale"],
            "bit_length": fixed_row["bit_length"],
            "packing": fixed_row["packing"],
            "clear": {"shape": clear_row["shape"], "scale": clear_row["scale"],
                      "bit_length": clear_row["bit_length"],
                      "packing": clear_row["packing"]},
            "clear_fixed": error_row(clear_row["value"], fixed_row["value"],
                                      fixed_graph.logical),
            "secure": secure_placeholder(),
        })
    padding_rows: list[dict[str, Any]] = []
    for name, physical_row in fixed_records.items():
        logical_row = logical_fixed_records.get(name)
        if logical_row is None:
            continue
        padding_rows.append({
            "name": name,
            "comparison": error_row(logical_row["value"], physical_row["value"],
                                     fixed_graph.logical),
        })
    padding_max = max(
        (row["comparison"].get("max_abs", float("inf"))
         for row in padding_rows), default=0.0)
    report = {
        "format": "BOLT_VIT_REFERENCE_REPORT_V1",
        "model": {key: model[key] for key in (
            "source_id", "source_revision", "source_sha256", "image_size",
            "patch_size", "channels", "tokens", "dim", "heads", "ff_dim",
            "layers", "classes", "qkv_bias", "layer_norm_epsilon")},
        "packing": {"logical_tokens": model["tokens"],
                    "physical_tokens": fixed_graph.physical,
                    "valid_positions": [0, model["tokens"] - 1],
                    "padding_positions": ([model["tokens"], fixed_graph.physical - 1]
                                           if fixed_graph.physical > model["tokens"] else [])},
        "fixed_point": {"ell": ELL, "activation_scale": ACT_SCALE,
                        "weight_scale": WEIGHT_SCALE,
                        "gelu_input_scale": GELU_INPUT_SCALE,
                        "gelu_output_scale": GELU_OUTPUT_SCALE,
                        "layer_norm_epsilon_note": "BOLT secure LayerNorm adds checkpoint epsilon after variance reduction, quantized at activation scale 12; 1e-12 therefore rounds to zero in the current fixed-point format"},
        "clear": {"top1": clear_top1, "logits": clear_logits.tolist()},
        "fixed": {"top1": fixed_top1, "logits": fixed_logits.tolist(),
                  "top1_matches_clear": fixed_top1 == clear_top1},
        "padding_invariance": {
            "status": "PASS" if padding_max == 0.0 else "FAIL",
            "logical_fixed_top1": int(np.argmax(logical_fixed_logits)),
            "physical_fixed_top1": fixed_top1,
            "max_abs_valid_prefix_difference": float(padding_max),
            "operators": padding_rows,
        },
        "secure": {"top1": secure_top1,
                    "top1_matches_clear": (secure_top1 == clear_top1)
                    if secure_top1 is not None else None,
                    "top1_matches_fixed": (secure_top1 == fixed_top1)
                    if secure_top1 is not None else None,
                    "intermediate_reconstruction": False,
                    "layerwise_error_status": "NOT_AVAILABLE"},
        "operators": operators,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("VIT_FLOAT_REFERENCE=PASS")
    print("VIT_FIXED_CLEAR=PASS")
    print(f"VIT_FLOAT_TOP1={clear_top1}")
    print(f"VIT_FIXED_TOP1={fixed_top1}")
    print(f"VIT_FIXED_CLEAR_TOP1={'PASS' if fixed_top1 == clear_top1 else 'FAIL'}")
    if secure_top1 is None:
        print("VIT_SECURE_TOP1=NOT_PROVIDED")
    else:
        print(f"VIT_SECURE_TOP1={secure_top1}")
        print(f"VIT_SECURE_TOP1_MATCH={'PASS' if secure_top1 == fixed_top1 else 'FAIL'}")
    print(f"VIT_REFERENCE_REPORT={args.output}")


if __name__ == "__main__":
    main()
