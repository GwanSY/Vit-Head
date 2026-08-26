#!/usr/bin/env python3
"""Independent floating/fixed-clear reference for the canonical BOLT ViT file.

The floating graph is the standard pre-LayerNorm ViT equation used by the
PyTorch/Hugging Face and timm implementations.  The fixed graph is a clear
simulation of BOLT's public fixed-point schedule: 37-bit additive values,
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
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:  # scipy is already used by the repository's FP32 evaluator.
    from scipy.special import erf as _erf
except ImportError:  # pragma: no cover - fallback for minimal environments
    _erf = np.vectorize(math.erf, otypes=[np.float64])


ELL = 37
ACT_SCALE = 12
WEIGHT_SCALE = 12
GELU_INPUT_SCALE = 11
GELU_OUTPUT_SCALE = 4
QK_OPERAND_SCALE = 10
PLAIN_MODULUS = 536_903_681
BFV_TRACE_FORMAT = "BOLT_VIT_FIXED_TRACE_V1"
BFV_TRACE_JSON = "bfv_fixed_trace.json"
BFV_TRACE_ARRAYS = "bfv_accumulators.npz"


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
    if magic not in {"BOLT_VIT_MODEL_V1", "BOLT_VIT_MODEL_V2",
                     "BOLT_VIT_MODEL_V3", "IRON_VIT_MODEL_V1",
                     "IRON_VIT_MODEL_V2", "IRON_VIT_MODEL_V3"}:
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
    if reader.take() not in {"BOLT_VIT_INPUT_V1", "IRON_VIT_INPUT_V1"}:
        raise ValueError("unsupported canonical input magic")
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
    lhs_array = np.asarray(lhs)
    rhs_array = np.asarray(rhs)
    if lhs_array.shape[-1] != rhs_array.shape[-2]:
        raise ValueError("integer matmul inner dimensions do not match")
    lhs_peak = int(np.max(np.abs(lhs_array))) if lhs_array.size else 0
    rhs_peak = int(np.max(np.abs(rhs_array))) if rhs_array.size else 0
    absolute_sum_bound = lhs_array.shape[-1] * lhs_peak * rhs_peak
    if absolute_sum_bound > (1 << 53):
        raise ValueError(
            "fixed reference cannot prove an exact float64 integer matmul: "
            f"absolute-sum bound {absolute_sum_bound} exceeds 2^53")
    return np.rint(np.matmul(lhs_array.astype(np.float64),
                            rhs_array.astype(np.float64)))


def signed_raw_accumulator(value: np.ndarray, name: str) -> np.ndarray:
    """Convert an exact, pre-modulus accumulator to signed int64.

    This conversion happens immediately after the mathematical dot product
    and optional bias addition.  No BFV reduction or centered decoding has
    occurred, so an out-of-range value remains visible to the range auditor.
    """
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: raw accumulator contains a non-finite value")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise ValueError(f"{name}: raw accumulator is not exactly integral")
    if array.size and float(np.max(np.abs(array))) > (1 << 53):
        raise ValueError(
            f"{name}: raw accumulator exceeds the exact float64 integer range")
    int64 = np.iinfo(np.int64)
    if (array.size and
            (float(array.min()) < int64.min or float(array.max()) > int64.max)):
        raise ValueError(f"{name}: raw accumulator does not fit signed int64")
    return rounded.astype(np.int64)


def fixed_linear(value: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                 input_scale: int, output_scale: int,
                 weight_scale: int = WEIGHT_SCALE,
                 weight_multiplier: float = 1.0,
                 raw_callback: Callable[[np.ndarray, int], None] | None = None,
                 operation_name: str = "linear") -> np.ndarray:
    x_int = round_int(value, input_scale)
    w_int = round_int(weight * weight_multiplier, weight_scale)
    raw = matmul_integer(x_int, w_int.T)
    if bias.size:
        bias_int = round_int(bias * weight_multiplier,
                             input_scale + weight_scale)
        if bias_int.size and int(np.max(np.abs(bias_int))) > (1 << 53):
            raise ValueError(
                f"{operation_name}: bias exceeds the exact float64 integer range")
        raw += bias_int
    # BOLT uses truncation/reduction after the raw (input+weight) scale.
    raw_scale = input_scale + weight_scale
    if raw_callback is not None:
        raw_callback(signed_raw_accumulator(raw, operation_name), raw_scale)
    if raw_scale >= output_scale:
        divisor = float(1 << (raw_scale - output_scale))
        output_int = np.floor(raw / divisor)
    else:
        output_int = raw * float(1 << (output_scale - raw_scale))
    return output_int / float(1 << output_scale)


def fixed_add(lhs: np.ndarray, rhs: np.ndarray, scale: int) -> np.ndarray:
    return quantized(lhs + rhs, scale)


def layer_norm(value: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               epsilon: float) -> np.ndarray:
    mean = value.mean(axis=-1, keepdims=True, dtype=np.float64)
    variance = ((value - mean) ** 2).mean(axis=-1, keepdims=True,
                                          dtype=np.float64)
    safe_epsilon = epsilon if epsilon > 0.0 else 1e-12
    normalized = (value - mean) / np.sqrt(variance + safe_epsilon)
    return normalized * weight + bias


def _floor_shift(value: np.ndarray, bits: int) -> np.ndarray:
    """Arithmetic right shift used by SCI's clear fixed tests."""
    return np.floor_divide(np.asarray(value, dtype=np.int64), 1 << bits)


def _ring_reduce(value: np.ndarray | int, ell: int) -> np.ndarray:
    """Return the low ``ell`` bits, matching ``FixOp::reduce``."""
    if ell <= 0 or ell >= 63:
        raise ValueError("fixed-clear ring helper requires 0 < ell < 63")
    return np.bitwise_and(np.asarray(value, dtype=np.int64), (1 << ell) - 1)


def _ring_signed(value: np.ndarray | int, ell: int) -> np.ndarray:
    """Decode an ``ell``-bit two's-complement ring element."""
    raw = _ring_reduce(value, ell)
    return np.where(raw >= (1 << (ell - 1)), raw - (1 << ell), raw)


def _ring_add(lhs: np.ndarray | int, rhs: np.ndarray | int,
              ell: int) -> np.ndarray:
    return _ring_reduce(_ring_reduce(lhs, ell) + _ring_reduce(rhs, ell), ell)


def _ring_mul(lhs: np.ndarray | int, lhs_ell: int,
              rhs: np.ndarray | int, rhs_ell: int, output_ell: int,
              rhs_signed: bool = True) -> np.ndarray:
    """Clear reconstruction of the signed-first-operand ``FixOp::mul``."""
    lhs_value = _ring_signed(lhs, lhs_ell)
    rhs_value = (_ring_signed(rhs, rhs_ell) if rhs_signed else
                 _ring_reduce(rhs, rhs_ell))
    return _ring_reduce(lhs_value * rhs_value, output_ell)


def _ring_truncate_reduce(value: np.ndarray | int, ell: int,
                          shift: int) -> np.ndarray:
    """Bit-exact signed ``FixOp::truncate_reduce`` reconstruction."""
    if shift < 0 or shift >= ell:
        raise ValueError("truncate_reduce requires 0 <= shift < ell")
    return _ring_reduce(
        np.floor_divide(_ring_signed(value, ell), 1 << shift), ell - shift)


def _ring_right_shift(value: np.ndarray | int, ell: int,
                      shift: int) -> np.ndarray:
    """Bit-exact signed constant ``FixOp::right_shift`` reconstruction."""
    if shift < 0 or shift > ell:
        raise ValueError("right_shift requires 0 <= shift <= ell")
    return _ring_reduce(
        np.floor_divide(_ring_signed(value, ell), 1 << shift), ell)


def _cpp_fixed_constant(value: float, scale: int, ell: int) -> np.ndarray:
    """Match BOLT's ``uint64_t(double * 2^scale)`` constant truncation."""
    return _ring_reduce(math.trunc(value * (1 << scale)), ell)


def _lookup_inverse_sqrt(value: int, bw: int = ELL,
                         input_scale: int = ACT_SCALE,
                         output_scale: int = ACT_SCALE) -> int:
    """Bit-exact clear form of MathFunctions::sqrt(..., inverse=true).

    This follows SCI/src/cleartext_library_fixed.cpp rather than using a
    floating reciprocal square root.  In particular, its 6-bit lookup and
    one Newton iteration are part of BLB's existing LayerNorm definition.
    """
    if value <= 0:
        return 0
    m = math.ceil(output_scale / 2.0)
    iterations = 1 if output_scale <= 14 else 2
    msnzb = value.bit_length() - 1
    adjust = 1 << (bw - 2 - msnzb)
    sqrt_adjust_scale = math.floor((bw - 1 - input_scale) / 2.0)
    sqrt_adjust = 1 << int(
        math.floor((input_scale - msnzb + 1) / 2.0) +
        sqrt_adjust_scale)
    exponent_parity = (msnzb - input_scale) & 1
    shifted = value * adjust
    index = (shifted >> (bw - 2 - m)) & ((1 << m) - 1)
    normalized = (1.0 + index / float(1 << m)) * (1 << exponent_parity)
    lookup_scale = m + 2
    previous = int((1.0 / math.sqrt(normalized)) * (1 << lookup_scale))
    x_value = 2 * shifted if exponent_parity else shifted
    b_previous = x_value >> (bw - 2 - output_scale)
    y_previous = previous
    current = previous
    for iteration in range(iterations):
        y_squared = y_previous * y_previous
        if iteration == 0:
            b_current = ((y_squared * b_previous) >>
                         (2 * m + 2 * 2))
        else:
            y_squared >>= output_scale + 2
            b_current = (y_squared * b_previous) >> output_scale
        y_current = (3 << output_scale) - b_current
        product = previous * y_current
        current = product >> ((m + 2 + 1) if iteration == 0
                              else (output_scale + 1))
        previous = current
        b_previous = b_current
        y_previous = y_current
    result = (current * sqrt_adjust) >> sqrt_adjust_scale
    # cleartext_library_fixed's historical Saturate helper casts to int32.
    result &= (1 << 32) - 1
    if result >= (1 << 31):
        result -= 1 << 32
    return result


def fixed_layer_norm(
        value: np.ndarray, weight: np.ndarray, bias: np.ndarray,
        epsilon: float,
        raw_callback: Callable[[np.ndarray, int], None] | None = None,
        operation_name: str = "layer_norm_affine") -> np.ndarray:
    del epsilon  # checkpoint 1e-12 quantizes to zero at scale 12
    x = round_int(value, ACT_SCALE)
    width = x.shape[-1]
    reciprocal_width = int((1.0 / width) * (1 << (2 * ACT_SCALE)))
    total = np.sum(x, axis=-1, keepdims=True, dtype=np.int64)
    average = _floor_shift(total * reciprocal_width, 2 * ACT_SCALE)
    centered = x - average
    squares = _floor_shift(centered * centered, ACT_SCALE)
    variance = _floor_shift(
        np.sum(squares, axis=-1, dtype=np.int64) * reciprocal_width,
        2 * ACT_SCALE)
    sigma = np.asarray([_lookup_inverse_sqrt(int(item))
                        for item in variance], dtype=np.int64)[:, None]
    normalized = _floor_shift(centered * sigma, ACT_SCALE)
    # Match the secure adapter exactly: LayerNorm's private affine uses an
    # existing ct-pt elementwise multiply with weight scale 11. Bias is
    # encoded at the raw product scale, then the combined value is truncated.
    affine_weight_scale = 11
    gamma = round_int(weight, affine_weight_scale)
    beta_raw = round_int(bias, ACT_SCALE + affine_weight_scale)
    affine_raw = normalized * gamma + beta_raw
    if raw_callback is not None:
        raw_callback(signed_raw_accumulator(affine_raw, operation_name),
                     ACT_SCALE + affine_weight_scale)
    affine = _floor_shift(affine_raw, affine_weight_scale)
    return affine.astype(np.float64) / float(1 << ACT_SCALE)


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - value.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted, dtype=np.float64)
    return exponent / exponent.sum(axis=-1, keepdims=True, dtype=np.float64)


def _fixed_exp4_raw(value_raw: np.ndarray) -> np.ndarray:
    """Raw-ring emulation of ``FPMath::exp4`` (ell=37, scale=12)."""
    ell = ELL
    scale = ACT_SCALE
    x = _ring_reduce(value_raw, ell)
    ln2 = _ring_reduce(2839, ell)
    inverse_neg_ln2 = _ring_reduce(-5909, ell)

    x_inl = _ring_mul(x, ell, inverse_neg_ln2, ell, ell + scale)
    x_inl = _ring_truncate_reduce(x_inl, ell + scale, scale)
    l_short = _ring_truncate_reduce(x_inl, ell, scale)
    l_short_ell = ell - scale
    l_value = _ring_reduce(l_short << scale, ell)
    l_ln2 = _ring_mul(l_value, ell, ln2, ell, ell + scale)
    l_ln2 = _ring_truncate_reduce(l_ln2, ell + scale, scale)

    # exp4 deliberately narrows p to scale+2 bits before poly1.  Retaining
    # only the decoded real value here would lose this wrap/sign boundary.
    p_ell = scale + 2
    p = _ring_reduce(_ring_add(x, l_ln2, ell), p_ell)
    arg1 = _ring_reduce(1468, p_ell)
    arg2 = _ring_reduce(5542, p_ell)
    arg3 = _ring_reduce(1409, p_ell)
    p_arg2 = _ring_add(p, arg2, p_ell)
    p_arg2_pow2 = _ring_mul(
        p_arg2, p_ell, p_arg2, p_ell, p_ell + scale)
    p_arg2_pow2 = _ring_truncate_reduce(
        p_arg2_pow2, p_ell + scale, scale)
    polynomial = _ring_mul(
        p_arg2_pow2, p_ell, arg1, p_ell, p_ell + scale)
    polynomial = _ring_truncate_reduce(
        polynomial, p_ell + scale, scale)
    polynomial = _ring_add(polynomial, arg3, p_ell)
    polynomial = _ring_reduce(_ring_signed(polynomial, p_ell), ell)

    # The protocol reflags l_short as unsigned and caps only the secret shift,
    # not the polynomial remainder.  In the Softmax call domain l_short is
    # nonnegative; interpreting its raw bits is nevertheless part of exp4.
    shift = np.minimum(_ring_reduce(l_short, l_short_ell), 13)
    divisor = np.left_shift(np.ones_like(shift, dtype=np.int64), shift)
    return _ring_reduce(
        np.floor_divide(_ring_signed(polynomial, ell), divisor), ell)


def fixed_exp4(value: np.ndarray) -> np.ndarray:
    """Bit-exact decoded clear form of BOLT ``FPMath::exp4``."""
    input_raw = _ring_reduce(round_int(value, ACT_SCALE), ELL)
    output_raw = _fixed_exp4_raw(input_raw)
    return (_ring_signed(output_raw, ELL).astype(np.float64) /
            float(1 << ACT_SCALE))


def _lookup_table_exp_iron_raw(value_raw: np.ndarray) -> np.ndarray:
    """Raw form of ``MathFunctions::lookup_table_exp(37, 37, 12, 12)``."""
    ell = ELL
    scale = ACT_SCALE
    digit_size = 8
    positive = _ring_reduce(-_ring_reduce(value_raw, ell), ell)
    num_digits = math.ceil(ell / digit_size)
    digits_exp: list[np.ndarray] = []
    for digit_index in range(num_digits):
        digit_ell = min(digit_size, ell - digit_index * digit_size)
        digit = np.bitwise_and(
            np.right_shift(positive, digit_index * digit_size),
            (1 << digit_ell) - 1)
        # lookup_neg_exp first moves a negative scale into the integer before
        # evaluating exp, which is algebraically digit*2^(8*j-scale).
        exponent = np.ldexp(
            digit.astype(np.float64), digit_index * digit_size - scale)
        lookup = np.floor(np.exp(-exponent) * float(1 << scale)).astype(
            np.int64)
        digits_exp.append(_ring_reduce(lookup, scale + 2))

    # Match the in-place 5-digit multiplication tree and its truncation after
    # every product; evaluating one floating exp for the whole input differs.
    stride = 1
    while stride < num_digits:
        for left in range(0, num_digits, 2 * stride):
            right = left + stride
            if right < num_digits:
                product = _ring_reduce(
                    digits_exp[left] * digits_exp[right], 2 * scale + 2)
                digits_exp[left] = _ring_reduce(
                    np.floor_divide(product, 1 << scale), scale + 2)
        stride *= 2
    return _ring_reduce(digits_exp[0], ell)


def _div_exp37_raw(numerator: np.ndarray,
                   denominator: np.ndarray) -> np.ndarray:
    """Softmax specialization of ``FixOp::div(..., 37, 12)``."""
    ell = ELL
    scale = ACT_SCALE
    numerator_raw = _ring_reduce(numerator, ell)
    denominator_raw = _ring_reduce(denominator, ell)
    if (numerator_raw.ndim == 0 or
            numerator_raw.shape[:-1] != denominator_raw.shape):
        raise ValueError("division numerator rows and denominators do not match")

    rows = numerator_raw.reshape(-1, numerator_raw.shape[-1])
    divisors = denominator_raw.reshape(-1)
    output = np.empty_like(rows)
    m = 5  # ceil((s_out - 2) / 2), with s_out=12
    lookup_size = 1 << m
    c0_table = np.asarray([
        math.trunc((1.0 / ((1.0 + index / lookup_size) *
                           (1.0 + index / lookup_size + 1.0 / lookup_size))) *
                   (1 << (m + 3)))
        for index in range(lookup_size)
    ], dtype=np.int64)
    c1_table = np.asarray([
        math.trunc((((1.0 / (lookup_size * 2.0)) +
                     math.sqrt((1.0 + index / lookup_size) *
                               (1.0 + index / lookup_size +
                                1.0 / lookup_size))) /
                    ((1.0 + index / lookup_size) *
                     (1.0 + index / lookup_size + 1.0 / lookup_size))) *
                   (1 << (2 * m + 2)))
        for index in range(lookup_size)
    ], dtype=np.int64)

    normalized_scale = ell - 1
    adjust_scale = ell - 1 - scale
    for row_index, denominator_value in enumerate(divisors):
        dn = int(denominator_value)
        msnzb = dn.bit_length() - 1 if dn else 0
        adjust = 1 << (ell - 1 - msnzb)
        normalized = int(_ring_reduce(dn * adjust, ell + 1))
        epsilon = normalized & ((1 << (normalized_scale - m)) - 1)
        index = ((normalized & ((1 << normalized_scale) - 1)) >>
                 (normalized_scale - m))

        c0 = int(_ring_reduce(c0_table[index], m + 4))
        c1 = int(_ring_reduce(c1_table[index], 2 * m + 3))
        reciprocal_raw = _ring_add(
            c1 << 32, -c0 * epsilon, normalized_scale + m + 4)
        reciprocal = _ring_truncate_reduce(
            reciprocal_raw, normalized_scale + m + 4, 32)

        quotient = _ring_mul(
            rows[row_index], ell, reciprocal, 13, ell + scale)
        quotient = _ring_truncate_reduce(quotient, ell + scale, scale)
        quotient = _ring_mul(
            quotient, ell, adjust, ell + 1,
            ell + adjust_scale, rhs_signed=False)
        output[row_index] = _ring_truncate_reduce(
            quotient, ell + adjust_scale, adjust_scale)
    return output.reshape(numerator_raw.shape)


def fixed_softmax(value: np.ndarray) -> np.ndarray:
    """Bit-exact clear emulation of BOLT ``softmax_fix_iron_1``.

    The ViT adapter selects this existing 37-bit variant because a 197-item
    row can overflow the legacy ``softmax_fix`` 19-bit exponent sum.
    """
    input_raw = _ring_reduce(round_int(value, ACT_SCALE), ELL)
    if input_raw.ndim == 0 or input_raw.shape[-1] == 0:
        raise ValueError("fixed Softmax requires non-empty rows")
    input_signed = _ring_signed(input_raw, ELL)
    row_max = input_signed.max(axis=-1, keepdims=True)
    shifted = _ring_reduce(input_signed - row_max, ELL)
    exponent = _lookup_table_exp_iron_raw(shifted)
    denominator = _ring_reduce(
        np.sum(exponent, axis=-1, dtype=np.int64), ELL)
    output = _div_exp37_raw(exponent, denominator)
    return (_ring_signed(output, ELL).astype(np.float64) /
            float(1 << ACT_SCALE))


def fixed_gelu(value: np.ndarray) -> np.ndarray:
    """Bit-exact clear emulation of the secure ViT GELU adapter.

    The adapter first shifts ell=37, scale=12 input to scale 11, reduces to
    ell=20, calls ``FPMath::gelu_approx_2``, then returns ell=37, scale=4.
    """
    input_raw = _ring_reduce(round_int(value, ACT_SCALE), ELL)
    scale11 = _ring_right_shift(
        input_raw, ELL, ACT_SCALE - GELU_INPUT_SCALE)
    x = _ring_reduce(scale11, 20)
    ell = 20
    scale = GELU_INPUT_SCALE
    x_signed = _ring_signed(x, ell)
    y = np.where(x_signed < 0, _ring_reduce(-x_signed, ell), x)

    constants = [
        _cpp_fixed_constant(0.14439048359960427, scale, ell),
        _cpp_fixed_constant(-0.7077117131613893, scale, ell),
        _cpp_fixed_constant(4.5702822654246535, scale, ell),
        _cpp_fixed_constant(-8.15444702051307, scale, ell),
        _cpp_fixed_constant(16.382265425072532, scale, ell),
        _cpp_fixed_constant(0.5, scale, ell),
    ]
    c1, c2, c3, c4, c5, half = constants
    product_ell = ell + scale

    y_c1 = _ring_mul(y, ell, c1, ell, product_ell)
    y_c1 = _ring_truncate_reduce(y_c1, product_ell, scale)
    z_left = _ring_add(y_c1, c2, ell)
    z_left = _ring_mul(z_left, ell, y, ell, product_ell)
    z_left = _ring_truncate_reduce(z_left, product_ell, scale)
    z = _ring_add(z_left, c3, ell)

    polynomial = _ring_add(_ring_add(z, y_c1, ell), c4, ell)
    polynomial = _ring_mul(polynomial, ell, z, ell, product_ell)
    polynomial = _ring_truncate_reduce(polynomial, product_ell, scale)
    polynomial = _ring_add(polynomial, c5, ell)
    half_x = _ring_mul(x, ell, half, ell, product_ell)
    half_x = _ring_truncate_reduce(half_x, product_ell, scale)
    gelu_inner = _ring_add(half_x, polynomial, ell)

    within_polynomial = (_ring_signed(y, ell) <
                         math.trunc(2.7 * (1 << scale)))
    nonnegative_half = _ring_right_shift(_ring_add(x, y, ell), ell, 1)
    selected = np.where(within_polynomial, gelu_inner, nonnegative_half)
    extended = _ring_reduce(_ring_signed(selected, ell), ELL)
    output = _ring_right_shift(
        extended, ELL, GELU_INPUT_SCALE - GELU_OUTPUT_SCALE)
    return (_ring_signed(output, ELL).astype(np.float64) /
            float(1 << GELU_OUTPUT_SCALE))


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
                 physical_tokens: int | None = None,
                 capture_bfv_accumulators: bool = False,
                 block_limit: int | None = None):
        self.model = model
        self.fixed = fixed
        if capture_bfv_accumulators and not fixed:
            raise ValueError("BFV accumulator capture requires the fixed graph")
        self.capture_bfv_accumulators = capture_bfv_accumulators
        model_layers = int(model["layers"])
        self.executed_layers = (model_layers if block_limit is None
                                else int(block_limit))
        if self.executed_layers <= 0 or self.executed_layers > model_layers:
            raise ValueError("fixed/clear block limit is outside checkpoint")
        self.logical = int(model["tokens"])
        self.physical = (physical_tokens if physical_tokens is not None else
                         (next_power_of_two(self.logical) if fixed else self.logical))
        if self.physical < self.logical:
            raise ValueError("physical token count cannot be below logical count")
        self.records: dict[str, dict[str, Any]] = {}
        self.bfv_accumulators: dict[str, dict[str, Any]] = {}

    def capture_accumulator(self, name: str, kind: str, raw: np.ndarray,
                            accumulator_scale: int,
                            **metadata: Any) -> None:
        if not self.capture_bfv_accumulators:
            return
        if name in self.bfv_accumulators:
            raise ValueError(f"duplicate BFV accumulator name: {name}")
        if kind not in {"linear", "qk", "sv", "affine"}:
            raise ValueError(f"unsupported BFV accumulator kind: {kind}")
        signed = np.asarray(raw)
        if signed.dtype != np.dtype(np.int64):
            raise ValueError(f"{name}: captured accumulator is not signed int64")
        self.bfv_accumulators[name] = {
            "name": name,
            "kind": kind,
            "accumulator_scale": int(accumulator_scale),
            "raw": signed,
            **metadata,
        }

    def record(self, name: str, value: np.ndarray, scale: int | str,
               packing: str) -> np.ndarray:
        self.records[name] = {
            "shape": list(value.shape),
            "scale": scale,
            "bit_length": ELL if self.fixed else 64,
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
               input_scale: int, output_scale: int,
               weight_scale: int = WEIGHT_SCALE,
               weight_multiplier: float = 1.0,
               operation_name: str | None = None) -> np.ndarray:
        if self.fixed:
            if self.capture_bfv_accumulators and operation_name is None:
                raise ValueError("captured fixed linear requires an operation name")
            callback: Callable[[np.ndarray, int], None] | None = None
            if self.capture_bfv_accumulators:
                callback = lambda raw, scale: self.capture_accumulator(
                    operation_name, "linear", raw, scale,
                    input_scale=int(input_scale),
                    weight_scale=int(weight_scale),
                    output_scale=int(output_scale),
                    bias_included=bool(bias.size),
                    weight_multiplier=float(weight_multiplier))
            return fixed_linear(value, weight, bias, input_scale,
                                output_scale, weight_scale,
                                weight_multiplier, callback,
                                operation_name or "linear")
        return value @ weight.T + bias

    def add(self, lhs: np.ndarray, rhs: np.ndarray, scale: int) -> np.ndarray:
        return fixed_add(lhs, rhs, scale) if self.fixed else lhs + rhs

    def norm(self, value: np.ndarray, weight: np.ndarray,
             bias: np.ndarray, operation_name: str) -> np.ndarray:
        if self.fixed:
            result = np.zeros_like(value)
            callback: Callable[[np.ndarray, int], None] | None = None
            if self.capture_bfv_accumulators:
                def capture_norm(raw: np.ndarray, scale: int) -> None:
                    # The secure ct-pt affine evaluates all physical rows.
                    # Normalized padding is zero, so its raw value is beta;
                    # the following public token mask removes it afterwards.
                    physical_raw = np.empty(
                        (self.physical, int(self.model["dim"])),
                        dtype=np.int64)
                    physical_raw[:self.logical] = raw
                    if self.physical > self.logical:
                        physical_raw[self.logical:] = round_int(bias, scale)
                    self.capture_accumulator(
                        operation_name, "affine", physical_raw, scale,
                        input_scale=ACT_SCALE, weight_scale=11,
                        output_scale=ACT_SCALE, bias_included=True)
                callback = capture_norm
            result[:self.logical] = fixed_layer_norm(
                value[:self.logical], weight, bias,
                self.model["layer_norm_epsilon"], callback, operation_name)
            return self.mask_tokens(result)
        return layer_norm(value, weight, bias, self.model["layer_norm_epsilon"])

    def attention(self, value: np.ndarray, block: dict[str, np.ndarray],
                  layer: int) -> np.ndarray:
        input_scale = ACT_SCALE if self.fixed else 0
        head_dim = int(self.model["dim"]) // int(self.model["heads"])
        if self.fixed:
            q = self.linear(value, block["wq"], block["q_bias"],
                            input_scale, QK_OPERAND_SCALE, 10,
                            1.0 / math.sqrt(head_dim),
                            f"block_{layer}.q_projection")
            k = self.linear(value, block["wk"], block["k_bias"],
                            input_scale, QK_OPERAND_SCALE, 10, 1.0,
                            f"block_{layer}.k_projection")
            v = self.linear(value, block["wv"], block["v_bias"],
                            input_scale, ACT_SCALE, 11, 1.0,
                            f"block_{layer}.v_projection")
        else:
            q = self.linear(value, block["wq"], block["q_bias"], 0, 0)
            k = self.linear(value, block["wk"], block["k_bias"], 0, 0)
            v = self.linear(value, block["wv"], block["v_bias"], 0, 0)
        self.record(f"block_{layer}.q", q, QK_OPERAND_SCALE if self.fixed else "float64",
                    "physical_token_major_q" if self.fixed else "logical_token_major_q")
        self.record(f"block_{layer}.k", k, QK_OPERAND_SCALE if self.fixed else "float64",
                    "physical_token_major_k" if self.fixed else "logical_token_major_k")
        self.record(f"block_{layer}.v", v, ACT_SCALE if self.fixed else "float64",
                    "physical_token_major_v" if self.fixed else "logical_token_major_v")
        qh = q.reshape(self.physical, self.model["heads"], head_dim).transpose(1, 0, 2)
        kh = k.reshape(self.physical, self.model["heads"], head_dim).transpose(1, 0, 2)
        vh = v.reshape(self.physical, self.model["heads"], head_dim).transpose(1, 0, 2)
        if self.fixed:
            qi = round_int(qh, QK_OPERAND_SCALE)
            ki = round_int(kh, QK_OPERAND_SCALE)
            raw = matmul_integer(qi, np.swapaxes(ki, -1, -2))
            self.capture_accumulator(
                f"block_{layer}.qk", "qk",
                signed_raw_accumulator(raw, f"block_{layer}.qk"),
                2 * QK_OPERAND_SCALE,
                lhs_scale=QK_OPERAND_SCALE,
                rhs_scale=QK_OPERAND_SCALE,
                output_scale=ACT_SCALE,
                reduction_dimension=head_dim,
                bias_included=False)
            scores = (np.floor(
                raw / float(1 << (2 * QK_OPERAND_SCALE - ACT_SCALE))) /
                float(1 << ACT_SCALE))
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
            probability_int = round_int(probability, ACT_SCALE)
            value_int = round_int(vh, ACT_SCALE)
            context_raw = matmul_integer(probability_int, value_int)
            context_raw_signed = signed_raw_accumulator(
                context_raw, f"block_{layer}.softmax_v")
            self.capture_accumulator(
                f"block_{layer}.softmax_v", "sv", context_raw_signed,
                2 * ACT_SCALE, lhs_scale=ACT_SCALE, rhs_scale=ACT_SCALE,
                output_scale=ACT_SCALE, reduction_dimension=self.physical,
                bias_included=False)
            context = (_floor_shift(context_raw_signed, ACT_SCALE)
                       .astype(np.float64) / float(1 << ACT_SCALE))
        else:
            context = np.matmul(probability, vh)
        joined = context.transpose(1, 0, 2).reshape(self.physical, self.model["dim"])
        joined = self.mask_tokens(joined)
        self.record(f"block_{layer}.attention_context", joined,
                    ACT_SCALE if self.fixed else "float64",
                    "physical_merged_heads" if self.fixed else "logical_merged_heads")
        projected = self.linear(joined, block["wo"], block["bo"],
                                ACT_SCALE if self.fixed else 0, ACT_SCALE,
                                10 if self.fixed else WEIGHT_SCALE, 1.0,
                                f"block_{layer}.output_projection")
        return self.mask_tokens(projected)

    def run(self, pixels: np.ndarray) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
        patches = patchify(pixels, int(self.model["patch_size"]))
        if self.fixed:
            patches = quantized(patches, ACT_SCALE)
        self.record("patchify", patches, ACT_SCALE if self.fixed else "float64",
                    "chw_to_py_px_c_ky_kx" if self.fixed else "logical_patch_matrix")
        patch_tokens = self.linear(patches, self.model["patch_weight"],
                                   self.model["patch_bias"],
                                   ACT_SCALE if self.fixed else 0, ACT_SCALE,
                                   WEIGHT_SCALE, 1.0, "patch_projection")
        self.record("patch_projection", patch_tokens,
                    ACT_SCALE if self.fixed else "float64",
                    "compact_bsgs_valid_patches" if self.fixed else "logical_patch_projection")
        hidden = np.zeros((self.physical, self.model["dim"]), dtype=np.float64)
        if self.fixed:
            # The secure graph encodes these two server parameters
            # independently before share addition.  Preserve that exact
            # quantization boundary for the CLS row.
            hidden[0] = quantized(self.model["class_token"], ACT_SCALE)
            hidden[1:self.logical] = patch_tokens
            hidden[:self.logical] += quantized(
                self.model["position"], ACT_SCALE)
            hidden = self.mask_tokens(hidden)
        else:
            hidden[0] = self.model["class_token"]
            hidden[1:self.logical] = patch_tokens
            hidden[:self.logical] += self.model["position"]
        self.record("cls_position", hidden,
                    ACT_SCALE if self.fixed else "float64",
                    "physical_cls_then_raster_padding" if self.fixed else "logical_cls_then_raster")
        for layer, block in enumerate(
                self.model["blocks"][:self.executed_layers]):
            normalized = self.norm(
                hidden, block["ln1_weight"], block["ln1_bias"],
                f"block_{layer}.ln1_affine")
            self.record(f"block_{layer}.ln1", normalized,
                        ACT_SCALE if self.fixed else "float64",
                        "physical_rowwise_layernorm" if self.fixed else "logical_rowwise_layernorm")
            attn = self.attention(normalized, block, layer)
            hidden = self.mask_tokens(self.add(hidden, attn, ACT_SCALE))
            self.record(f"block_{layer}.attention_residual", hidden,
                        ACT_SCALE if self.fixed else "float64",
                        "physical_public_token_mask" if self.fixed else "logical_token_major")
            normalized = self.norm(
                hidden, block["ln2_weight"], block["ln2_bias"],
                f"block_{layer}.ln2_affine")
            self.record(f"block_{layer}.ln2", normalized,
                        ACT_SCALE if self.fixed else "float64",
                        "physical_rowwise_layernorm" if self.fixed else "logical_rowwise_layernorm")
            first = self.linear(normalized, block["ff1_weight"], block["ff1_bias"],
                                ACT_SCALE if self.fixed else 0, ACT_SCALE,
                                10 if self.fixed else WEIGHT_SCALE, 1.0,
                                f"block_{layer}.fc1")
            first = self.mask_tokens(first)
            self.record(f"block_{layer}.fc1", first,
                        ACT_SCALE if self.fixed else "float64",
                        "physical_ffn_bsgs" if self.fixed else "logical_ffn")
            activated = fixed_gelu(first) if self.fixed else (
                0.5 * first * (1.0 + _erf(first / math.sqrt(2.0))))
            activated = self.mask_tokens(activated)
            self.record(f"block_{layer}.gelu", activated,
                        GELU_OUTPUT_SCALE if self.fixed else "float64",
                        "physical_gelu_ell20_to_ell37" if self.fixed else "logical_gelu")
            second = self.linear(activated, block["ff2_weight"], block["ff2_bias"],
                                 GELU_OUTPUT_SCALE if self.fixed else 0,
                                 ACT_SCALE, WEIGHT_SCALE, 1.0,
                                 f"block_{layer}.fc2")
            hidden = self.mask_tokens(self.add(hidden, second, ACT_SCALE))
            self.record(f"block_{layer}.ffn_residual", hidden,
                        ACT_SCALE if self.fixed else "float64",
                        "physical_public_token_mask" if self.fixed else "logical_token_major")
        hidden = self.norm(hidden, self.model["final_ln_weight"],
                           self.model["final_ln_bias"], "final_ln_affine")
        self.record("final_ln", hidden,
                    ACT_SCALE if self.fixed else "float64",
                    "physical_rowwise_layernorm" if self.fixed else "logical_rowwise_layernorm")
        cls = hidden[:1]
        self.record("cls", cls,
                    ACT_SCALE if self.fixed else "float64",
                    "public_index_cls_share" if self.fixed else "logical_cls")
        logits = self.linear(cls, self.model["classifier_weight"],
                             self.model["classifier_bias"],
                             ACT_SCALE if self.fixed else 0, ACT_SCALE,
                             11 if self.fixed else WEIGHT_SCALE, 1.0,
                             "classification_head")
        self.record("logits", logits,
                    ACT_SCALE if self.fixed else "float64",
                    "packed_classifier_stride" if self.fixed else "logical_classifier")
        return logits.reshape(-1), self.records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.int64)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def expected_bfv_accumulators(
        model: Mapping[str, Any], physical_tokens: int,
        executed_layers: int | None = None) -> list[dict[str, Any]]:
    """Enumerate all data-dependent BFV accumulators independently."""
    logical = int(model["tokens"])
    dim = int(model["dim"])
    heads = int(model["heads"])
    ff_dim = int(model["ff_dim"])
    classes = int(model["classes"])
    result: list[dict[str, Any]] = [{
        "name": "patch_projection", "kind": "linear",
        "shape": [logical - 1, dim],
        "accumulator_scale": ACT_SCALE + WEIGHT_SCALE,
    }]
    layers = int(model["layers"]) if executed_layers is None else executed_layers
    if layers <= 0 or layers > int(model["layers"]):
        raise ValueError("BFV expected graph block limit is invalid")
    for layer in range(layers):
        prefix = f"block_{layer}"
        result.extend([
            {"name": f"{prefix}.ln1_affine", "kind": "affine",
             "shape": [physical_tokens, dim],
             "accumulator_scale": ACT_SCALE + 11},
            {"name": f"{prefix}.q_projection", "kind": "linear",
             "shape": [physical_tokens, dim],
             "accumulator_scale": ACT_SCALE + 10},
            {"name": f"{prefix}.k_projection", "kind": "linear",
             "shape": [physical_tokens, dim],
             "accumulator_scale": ACT_SCALE + 10},
            {"name": f"{prefix}.v_projection", "kind": "linear",
             "shape": [physical_tokens, dim],
             "accumulator_scale": ACT_SCALE + 11},
            {"name": f"{prefix}.qk", "kind": "qk",
             "shape": [heads, physical_tokens, physical_tokens],
             "accumulator_scale": 2 * QK_OPERAND_SCALE},
            {"name": f"{prefix}.softmax_v", "kind": "sv",
             "shape": [heads, physical_tokens, dim // heads],
             "accumulator_scale": 2 * ACT_SCALE},
            {"name": f"{prefix}.output_projection", "kind": "linear",
             "shape": [physical_tokens, dim],
             "accumulator_scale": ACT_SCALE + 10},
            {"name": f"{prefix}.ln2_affine", "kind": "affine",
             "shape": [physical_tokens, dim],
             "accumulator_scale": ACT_SCALE + 11},
            {"name": f"{prefix}.fc1", "kind": "linear",
             "shape": [physical_tokens, ff_dim],
             "accumulator_scale": ACT_SCALE + 10},
            {"name": f"{prefix}.fc2", "kind": "linear",
             "shape": [physical_tokens, dim],
             "accumulator_scale": GELU_OUTPUT_SCALE + WEIGHT_SCALE},
        ])
    result.append({
        "name": "final_ln_affine", "kind": "affine",
        "shape": [physical_tokens, dim],
        "accumulator_scale": ACT_SCALE + 11,
    })
    result.append({
        "name": "classification_head", "kind": "linear",
        "shape": [1, classes],
        "accumulator_scale": ACT_SCALE + 11,
    })
    return result


def write_bfv_trace(trace_dir: Path, model_path: Path, input_path: Path,
                    model: Mapping[str, Any], fixed_graph: Graph) -> tuple[Path, Path]:
    """Write complete signed pre-modulus arrays plus integrity metadata."""
    if not fixed_graph.fixed or not fixed_graph.capture_bfv_accumulators:
        raise ValueError("BFV trace writer requires a capturing fixed graph")
    expected = expected_bfv_accumulators(
        model, fixed_graph.physical, fixed_graph.executed_layers)
    expected_names = [row["name"] for row in expected]
    observed_names = list(fixed_graph.bfv_accumulators)
    if observed_names != expected_names:
        missing = [name for name in expected_names if name not in observed_names]
        unexpected = [name for name in observed_names if name not in expected_names]
        raise ValueError(
            "BFV accumulator coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}, "
            f"expected_order={expected_names}, observed_order={observed_names}")

    arrays: dict[str, np.ndarray] = {}
    accumulators: list[dict[str, Any]] = []
    for index, specification in enumerate(expected):
        observed = fixed_graph.bfv_accumulators[specification["name"]]
        raw = observed["raw"]
        if (observed["kind"] != specification["kind"] or
                list(raw.shape) != specification["shape"] or
                observed["accumulator_scale"] != specification["accumulator_scale"]):
            raise ValueError(
                f"{specification['name']}: observed BFV accumulator metadata "
                "does not match the independently enumerated graph")
        key = f"raw_{index:03d}"
        arrays[key] = raw
        row = {key_: value for key_, value in observed.items() if key_ != "raw"}
        row.update({
            "domain": "signed_integer_pre_modulus",
            "shape": list(raw.shape),
            "elements": int(raw.size),
            "dtype": "int64",
            "npz_key": key,
            "raw_min": int(raw.min()),
            "raw_max": int(raw.max()),
            "raw_sha256": _sha256_array(raw),
        })
        accumulators.append(row)

    trace_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = trace_dir / BFV_TRACE_ARRAYS
    temporary_arrays_path = trace_dir / (BFV_TRACE_ARRAYS + ".tmp")
    with temporary_arrays_path.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary_arrays_path.replace(arrays_path)
    arrays_sha256 = _sha256_file(arrays_path)

    metadata = {
        "format": BFV_TRACE_FORMAT,
        "plain_modulus": PLAIN_MODULUS,
        "domain": "signed_integer_raw_before_bfv_plaintext_modulus",
        "scope": ["bfv_ct_pt_linear", "bfv_ct_ct_qk",
                  "bfv_ct_ct_softmax_v"],
        "arrays_file": arrays_path.name,
        "arrays_sha256": arrays_sha256,
        "expected_accumulators": expected,
        "coverage": {
            "expected": len(expected),
            "observed": len(accumulators),
            "missing": [],
            "unexpected": [],
            "complete": True,
        },
        "model": {
            "path": str(model_path),
            "file_sha256": _sha256_file(model_path),
            "source_id": model["source_id"],
            "source_revision": model["source_revision"],
            "source_sha256": model["source_sha256"],
            "layers": int(model["layers"]),
            "executed_layers": fixed_graph.executed_layers,
            "tokens": int(model["tokens"]),
            "dim": int(model["dim"]),
            "heads": int(model["heads"]),
            "ff_dim": int(model["ff_dim"]),
            "classes": int(model["classes"]),
        },
        "input": {"path": str(input_path),
                  "file_sha256": _sha256_file(input_path)},
        "packing": {"logical_tokens": int(model["tokens"]),
                    "physical_tokens": fixed_graph.physical},
        "integrity": {
            "array_hash": "sha256(int64 C-order bytes)",
            "archive_hash": "sha256(file bytes)",
            "capture_point": "after exact signed dot-product and bias; before BFV modulus and truncation",
            "float64_integer_exactness_bound": "absolute dot-product sum <= 2^53",
        },
        "accumulators": accumulators,
    }
    metadata_path = trace_dir / BFV_TRACE_JSON
    temporary_metadata_path = trace_dir / (BFV_TRACE_JSON + ".tmp")
    temporary_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary_metadata_path.replace(metadata_path)
    return metadata_path, arrays_path


def valid_prefix(value: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if value.ndim == len(target_shape) and all(
            actual >= target for actual, target in zip(value.shape, target_shape)):
        return value[tuple(slice(0, target) for target in target_shape)]
    if value.ndim >= 1 and value.shape[0] >= target_shape[0]:
        return value[:target_shape[0]]
    return value


def error_row(clear: np.ndarray, fixed: np.ndarray, logical: int) -> dict[str, Any]:
    lhs = np.asarray(clear, dtype=np.float64)
    rhs = np.asarray(valid_prefix(fixed, lhs.shape), dtype=np.float64)
    if lhs.shape != rhs.shape:
        return {"status": "SHAPE_MISMATCH", "clear_shape": list(lhs.shape),
                "fixed_shape": list(rhs.shape)}
    difference = rhs - lhs
    return {
        # PASS here is deliberately a measurement/shape gate, not a numerical
        # accuracy claim.  Different operators have different fixed-point
        # scales and no protocol-wide absolute-error threshold is defensible.
        # Keep the machine-readable distinction explicit so reports cannot
        # silently turn a large measured error into an acceptance result.
        "status": "PASS",
        "status_semantics": "shapes_comparable_metrics_computed",
        "acceptance_status": "NOT_EVALUATED_NO_NUMERIC_THRESHOLD",
        "acceptance_threshold": None,
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


def load_secure_logits(path: Path, expected_elements: int) -> np.ndarray:
    tokens = path.read_text(encoding="utf-8").split()
    if (len(tokens) < 10 or tokens[0] != "BLB_VIT_SECURE_LOGITS_V1" or
            tokens[1] != "elements" or tokens[3] != "ell" or
            tokens[5] != "scale" or tokens[7] != "values" or
            tokens[-1] != "end"):
        raise ValueError("validation logits file has an invalid format")
    elements = int(tokens[2])
    ell = int(tokens[4])
    scale = int(tokens[6])
    values = np.asarray([float(item) for item in tokens[8:-1]],
                        dtype=np.float64)
    if (elements != expected_elements or values.size != elements or
            ell != ELL or scale != ACT_SCALE or
            not np.all(np.isfinite(values))):
        raise ValueError("validation logits metadata/value count mismatch")
    return values


def load_secure_aggregate_errors(path: Path) -> dict[str, dict[str, Any]]:
    required = {
        "stage", "elements", "observer_ell", "observer_scale",
        "clear_quantization_bound", "fixed_secure_max_abs",
        "fixed_secure_mean_abs", "fixed_secure_rmse",
        "clear_secure_max_abs", "clear_secure_mean_abs",
        "clear_secure_rmse", "rmse_square_scale",
        "rmse_split_bits", "rmse_accumulator",
        "intermediate_reconstruction", "status",
    }
    result: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("secure aggregate error CSV schema mismatch")
        for raw in reader:
            name = raw["stage"]
            if (not name or name in result or raw["status"] != "PASS" or
                    raw["intermediate_reconstruction"] != "false"):
                raise ValueError("invalid secure aggregate error row")
            elements = int(raw["elements"])
            ell = int(raw["observer_ell"])
            scale = int(raw["observer_scale"])
            square_scale = int(raw["rmse_square_scale"])
            split_bits = int(raw["rmse_split_bits"])
            quantization_bound = float(raw["clear_quantization_bound"])
            if (elements <= 0 or ell != 43 or scale != 18 or
                    square_scale != 2 * scale or
                    split_bits != (elements - 1).bit_length() or
                    raw["rmse_accumulator"] != "exact_canonical_two_limb" or
                    quantization_bound != 0.5 / (1 << scale)):
                raise ValueError("secure aggregate arithmetic metadata mismatch")

            def metrics(prefix: str) -> dict[str, Any]:
                values = {
                    "max_abs": float(raw[f"{prefix}_max_abs"]),
                    "mean_abs": float(raw[f"{prefix}_mean_abs"]),
                    "rmse": float(raw[f"{prefix}_rmse"]),
                }
                if any(not math.isfinite(value) or value < 0.0
                       for value in values.values()):
                    raise ValueError("invalid secure aggregate metric")
                tolerance = 1e-12 * max(1.0, *values.values())
                if (values["mean_abs"] > values["max_abs"] + tolerance or
                        values["rmse"] < values["mean_abs"] - tolerance or
                        values["rmse"] > values["max_abs"] + tolerance):
                    raise ValueError(
                        "secure aggregate metric invariants are violated")
                return {
                    "status": "PASS",
                    "status_semantics": "aggregate_metrics_revealed_only",
                    "acceptance_status": "NOT_EVALUATED_NO_NUMERIC_THRESHOLD",
                    "acceptance_threshold": None,
                    **values,
                    "elements": elements,
                }

            result[name] = {
                "status": "PASS",
                "fixed_secure": metrics("fixed_secure"),
                "clear_secure": metrics("clear_secure"),
                "clear_quantization_bound": quantization_bound,
                "observer_ell": ell,
                "observer_scale": scale,
                "rmse_square_scale": square_scale,
                "rmse_split_bits": split_bits,
                "rmse_accumulator": raw["rmse_accumulator"],
                "intermediate_reconstruction": False,
                "scope": "validation-only aggregate metrics over valid prefix",
            }
    if not result:
        raise ValueError("secure aggregate error CSV is empty")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trace-dir", type=Path,
        help=("write every fixed BFV linear/QK signed raw accumulator, "
              "before plaintext-modulus reduction, to this directory"))
    parser.add_argument("--secure-prediction", type=Path,
                        help="optional BOB prediction file; supplies class only")
    parser.add_argument(
        "--secure-logits", type=Path,
        help="validation-only final logits; no intermediate tensor")
    parser.add_argument(
        "--secure-aggregate-errors", type=Path,
        help="validation-only aggregate metrics; never an intermediate tensor")
    parser.add_argument(
        "--block-limit", type=int, default=0,
        help="validation-only prefix block count; 0 executes all checkpoint blocks")
    args = parser.parse_args()

    model = load_model(args.model)
    pixels = load_input(args.input)
    expected_size = int(model["image_size"])
    if pixels.shape != (model["channels"], expected_size, expected_size):
        raise ValueError(f"input shape {pixels.shape} != {(model['channels'], expected_size, expected_size)}")
    block_limit = None if args.block_limit == 0 else args.block_limit
    clear_logits, clear_records = Graph(
        model, fixed=False, block_limit=block_limit).run(pixels)
    fixed_graph = Graph(
        model, fixed=True,
        capture_bfv_accumulators=args.trace_dir is not None,
        block_limit=block_limit)
    fixed_logits, fixed_records = fixed_graph.run(pixels)
    logical_fixed_graph = Graph(model, fixed=True,
                                physical_tokens=int(model["tokens"]),
                                block_limit=block_limit)
    logical_fixed_logits, logical_fixed_records = logical_fixed_graph.run(pixels)
    clear_top1 = int(np.argmax(clear_logits))
    fixed_top1 = int(np.argmax(fixed_logits))
    secure_top1: int | None = None
    if args.secure_prediction:
        fields = args.secure_prediction.read_text(encoding="utf-8").split()
        if len(fields) == 2 and fields[0] == "predicted_class":
            secure_top1 = int(fields[1])
    secure_logits: np.ndarray | None = None
    if args.secure_logits:
        secure_logits = load_secure_logits(
            args.secure_logits, int(model["classes"]))
        logits_top1 = int(np.argmax(secure_logits))
        if secure_top1 is not None and secure_top1 != logits_top1:
            raise ValueError(
                "validation logits top-1 differs from secure class output")
        secure_top1 = logits_top1
    secure_aggregates = (
        load_secure_aggregate_errors(args.secure_aggregate_errors)
        if args.secure_aggregate_errors else {})

    operators: list[dict[str, Any]] = []
    for name, fixed_row in fixed_records.items():
        clear_row = clear_records.get(name)
        if clear_row is None:
            continue
        clear_fixed_row = error_row(clear_row["value"], fixed_row["value"],
                                    fixed_graph.logical)
        secure_row: dict[str, Any] = secure_placeholder()
        if name in secure_aggregates:
            secure_row = secure_aggregates[name]
            fixed_secure = secure_row["fixed_secure"]
            clear_secure = secure_row["clear_secure"]
            if (fixed_secure["elements"] != clear_fixed_row["elements"] or
                    clear_secure["elements"] != clear_fixed_row["elements"]):
                raise ValueError(
                    f"secure aggregate element count mismatch at {name}")
            cross_bound = (secure_row["clear_quantization_bound"] +
                           fixed_secure["max_abs"])
            for metric in ("max_abs", "mean_abs", "rmse"):
                tolerance = 1e-12 * max(
                    1.0, clear_secure[metric], clear_fixed_row[metric])
                if abs(clear_secure[metric] - clear_fixed_row[metric]) > \
                        cross_bound + tolerance:
                    raise ValueError(
                        f"secure aggregate {metric} exceeds the clear-oracle "
                        f"quantization cross-check at {name}")
        elif name == "logits" and secure_logits is not None:
            secure_row = {
                "status": "PASS",
                "fixed_secure": error_row(
                    fixed_row["value"].reshape(-1), secure_logits,
                    fixed_graph.logical),
                "clear_secure": error_row(
                    clear_row["value"].reshape(-1), secure_logits,
                    fixed_graph.logical),
                "intermediate_reconstruction": False,
                "scope": "validation-only final model output",
            }
        operators.append({
            "name": name,
            "shape": fixed_row["shape"],
            "scale": fixed_row["scale"],
            "bit_length": fixed_row["bit_length"],
            "packing": fixed_row["packing"],
            "clear": {"shape": clear_row["shape"], "scale": clear_row["scale"],
                      "bit_length": clear_row["bit_length"],
                      "packing": clear_row["packing"]},
            "clear_fixed": clear_fixed_row,
            "secure": secure_row,
        })
    expected_aggregate_names = {
        "patch_projection", "cls", "logits",
        *(f"block_{layer}.{suffix}"
          for layer in range(fixed_graph.executed_layers)
          for suffix in ("attention_residual", "ffn_residual")),
    }
    if secure_aggregates and set(secure_aggregates) != expected_aggregate_names:
        raise ValueError(
            "secure aggregate stage coverage mismatch: "
            f"expected={sorted(expected_aggregate_names)} "
            f"got={sorted(secure_aggregates)}")
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
        "execution": {"checkpoint_blocks": int(model["layers"]),
                      "executed_blocks": fixed_graph.executed_layers,
                      "validation_block_limit":
                          fixed_graph.executed_layers != int(model["layers"])},
        "fixed_point": {"ell": ELL, "activation_scale": ACT_SCALE,
                        "weight_scale": WEIGHT_SCALE,
                        "qk_operand_scale": QK_OPERAND_SCALE,
                        "operator_weight_scales": {
                            "patch_projection": 12, "q": 10, "k": 10,
                            "v": 11, "output_projection": 10,
                            "fc1": 10, "fc2": 12,
                            "classification_head": 11,
                            "layer_norm_affine": 11},
                        "attention_scale_fold": "q_weight_and_bias_times_1/sqrt(head_dim)",
                        "gelu_input_scale": GELU_INPUT_SCALE,
                        "gelu_output_scale": GELU_OUTPUT_SCALE,
                        "layer_norm_epsilon_note": "BOLT FPMath layer_norm_fix has no explicit epsilon; secure path uses checkpoint epsilon metadata only as an audit field"},
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
                    "layerwise_error_status": (
                        "AGGREGATE_ONLY" if secure_aggregates
                        else "NOT_AVAILABLE"),
                    "logits": (secure_logits.tolist()
                               if secure_logits is not None else None),
                    "fixed_logits_error": (
                        error_row(fixed_logits, secure_logits,
                                  fixed_graph.logical)
                        if secure_logits is not None else None),
                    "clear_logits_error": (
                        error_row(clear_logits, secure_logits,
                                  fixed_graph.logical)
                        if secure_logits is not None else None)},
        "operators": operators,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    trace_paths: tuple[Path, Path] | None = None
    if args.trace_dir is not None:
        trace_paths = write_bfv_trace(
            args.trace_dir, args.model, args.input, model, fixed_graph)
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
    if trace_paths is not None:
        print(f"VIT_BFV_FIXED_TRACE={trace_paths[0]}")
        print(f"VIT_BFV_RAW_ARRAYS={trace_paths[1]}")


if __name__ == "__main__":
    main()
