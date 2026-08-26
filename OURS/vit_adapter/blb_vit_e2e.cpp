#include <seal/seal.h>
#include "FloatingPoint/fp-math.h"
#include "GC/integer.h"
#include "GC/semihonest.h"
#include "utils/ArgMapping/ArgMapping.h"

#ifdef I
#undef I
#endif

#include "blb_ckks_linear.h"
#include "blb_vit_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

int party = 0;
int port = 47000;
std::string address = "127.0.0.1";
int num_threads = 1;
int bitlength = blb_vit::kRingBits;

namespace {

using blb_vit::CkksLinear;
using blb_vit::LocalStats;
using blb_vit::encode_ring;
using blb_vit::kRingBits;

sci::SemiHonestParty<sci::NetIO> *gc_party = nullptr;

void setup_gc(sci::IOPack &io, sci::OTPack &ot, int role) {
  gc_party = sci::setup_semi_honest<sci::NetIO>(io.io_GC, role, 1 << 20);
  io.io_GC->flush();
  auto *base = ot.iknp_straight;
  if (role == sci::ALICE) {
    static_cast<sci::SemiHonestGen<sci::NetIO> *>(gc_party)
        ->setup_keys(base->k0, base->s);
  } else {
    static_cast<sci::SemiHonestEva<sci::NetIO> *>(gc_party)
        ->setup_keys(base->k0, base->k1);
  }
}

void teardown_gc() {
  delete circ_exec;
  delete prot_exec;
  circ_exec = nullptr;
  prot_exec = nullptr;
  gc_party = nullptr;
}

struct VitModel {
  int patches = 0;
  int patch_dim = 0;
  int dim = 0;
  int heads = 0;
  int mlp_dim = 0;
  int classes = 0;
  int scale_bits = 12;
  std::vector<double> patch_w, patch_b, pos;
  std::vector<double> wq, bq, wk, bk, wv, bv, wo, bo;
  std::vector<double> ln1_w, ln1_b;
  std::vector<double> ff1_w, ff1_b, ff2_w, ff2_b;
  std::vector<double> ln2_w, ln2_b;
  std::vector<double> head_ln_w, head_ln_b;
  std::vector<double> classifier_w, classifier_b;
};

struct VitInput {
  int patches = 0;
  int patch_dim = 0;
  std::vector<double> x;
};

struct WireConfig {
  int32_t patches;
  int32_t patch_dim;
  int32_t dim;
  int32_t heads;
  int32_t mlp_dim;
  int32_t classes;
  int32_t scale_bits;
};

void expect(std::istream &in, const std::string &wanted) {
  std::string got;
  if (!(in >> got) || got != wanted)
    throw std::runtime_error("expected token '" + wanted + "', got '" +
                             got + "'");
}

std::vector<double> read_values(std::istream &in, size_t count) {
  std::vector<double> values(count);
  for (double &value : values)
    if (!(in >> value)) throw std::runtime_error("truncated numeric vector");
  return values;
}

VitModel load_model(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open ALICE model file: " + path);
  expect(in, "BLB_VIT_E2E_MODEL_V1");
  VitModel model;
  expect(in, "patches"); in >> model.patches;
  expect(in, "patch_dim"); in >> model.patch_dim;
  expect(in, "dim"); in >> model.dim;
  expect(in, "heads"); in >> model.heads;
  expect(in, "mlp_dim"); in >> model.mlp_dim;
  expect(in, "classes"); in >> model.classes;
  expect(in, "scale_bits"); in >> model.scale_bits;
  const size_t dd = static_cast<size_t>(model.dim) * model.dim;
  expect(in, "patch_w"); model.patch_w = read_values(in, dd);
  expect(in, "patch_b"); model.patch_b = read_values(in, model.dim);
  expect(in, "pos");
  model.pos = read_values(in, static_cast<size_t>(model.patches) * model.dim);
  expect(in, "wq"); model.wq = read_values(in, dd);
  expect(in, "bq"); model.bq = read_values(in, model.dim);
  expect(in, "wk"); model.wk = read_values(in, dd);
  expect(in, "bk"); model.bk = read_values(in, model.dim);
  expect(in, "wv"); model.wv = read_values(in, dd);
  expect(in, "bv"); model.bv = read_values(in, model.dim);
  expect(in, "wo"); model.wo = read_values(in, dd);
  expect(in, "bo"); model.bo = read_values(in, model.dim);
  expect(in, "ln1_w"); model.ln1_w = read_values(in, model.dim);
  expect(in, "ln1_b"); model.ln1_b = read_values(in, model.dim);
  expect(in, "ff1_w");
  model.ff1_w = read_values(
      in, static_cast<size_t>(model.mlp_dim) * model.dim);
  expect(in, "ff1_b"); model.ff1_b = read_values(in, model.mlp_dim);
  expect(in, "ff2_w");
  model.ff2_w = read_values(
      in, static_cast<size_t>(model.dim) * model.mlp_dim);
  expect(in, "ff2_b"); model.ff2_b = read_values(in, model.dim);
  expect(in, "ln2_w"); model.ln2_w = read_values(in, model.dim);
  expect(in, "ln2_b"); model.ln2_b = read_values(in, model.dim);
  expect(in, "head_ln_w"); model.head_ln_w = read_values(in, model.dim);
  expect(in, "head_ln_b"); model.head_ln_b = read_values(in, model.dim);
  expect(in, "classifier_w");
  model.classifier_w = read_values(
      in, static_cast<size_t>(model.classes) * model.dim);
  expect(in, "classifier_b");
  model.classifier_b = read_values(in, model.classes);
  expect(in, "end");
  if (model.patches <= 0 || model.patch_dim <= 0 || model.dim <= 0 ||
      model.patch_dim > model.dim || model.heads <= 0 ||
      model.dim % model.heads != 0 || model.mlp_dim <= 0 ||
      model.classes <= 1)
    throw std::runtime_error("invalid ViT dimensions");
  if (model.scale_bits < 8 || model.scale_bits > 14)
    throw std::runtime_error("scale_bits must be in [8,14]");
  return model;
}

VitInput load_input(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open BOB input file: " + path);
  expect(in, "BLB_VIT_E2E_INPUT_V1");
  VitInput input;
  expect(in, "patches"); in >> input.patches;
  expect(in, "patch_dim"); in >> input.patch_dim;
  expect(in, "x");
  input.x = read_values(
      in, static_cast<size_t>(input.patches) * input.patch_dim);
  expect(in, "end");
  return input;
}

std::vector<double> padded_input(const VitInput &input, int dim) {
  std::vector<double> result(static_cast<size_t>(input.patches) * dim, 0.0);
  for (int patch = 0; patch < input.patches; ++patch)
    std::copy_n(input.x.data() + static_cast<size_t>(patch) * input.patch_dim,
                input.patch_dim,
                result.data() + static_cast<size_t>(patch) * dim);
  return result;
}

std::vector<double> compose_weights(const std::vector<double> &outer,
                                    const std::vector<double> &inner,
                                    int dim) {
  std::vector<double> result(static_cast<size_t>(dim) * dim, 0.0);
  for (int out = 0; out < dim; ++out)
    for (int in = 0; in < dim; ++in)
      for (int common = 0; common < dim; ++common)
        result[static_cast<size_t>(out) * dim + in] +=
            outer[static_cast<size_t>(out) * dim + common] *
            inner[static_cast<size_t>(common) * dim + in];
  return result;
}

std::vector<double> composed_bias(const VitModel &model,
                                  const std::vector<double> &weight,
                                  const std::vector<double> &bias) {
  std::vector<double> result(
      static_cast<size_t>(model.patches) * model.dim, 0.0);
  for (int patch = 0; patch < model.patches; ++patch)
    for (int out = 0; out < model.dim; ++out) {
      double value = bias[out];
      for (int common = 0; common < model.dim; ++common)
        value += weight[static_cast<size_t>(out) * model.dim + common] *
                 (model.patch_b[common] +
                  model.pos[static_cast<size_t>(patch) * model.dim + common]);
      result[static_cast<size_t>(patch) * model.dim + out] = value;
    }
  return result;
}

void add_alice_values(FixArray &value, const std::vector<double> &addend,
                      int scale_bits, int role) {
  if (value.size != static_cast<int>(addend.size()))
    throw std::runtime_error("private add shape mismatch");
  if (role != sci::ALICE) return;
  const uint64_t mask = (1ULL << value.ell) - 1;
  for (int i = 0; i < value.size; ++i)
    value.data[i] =
        (value.data[i] + encode_ring(addend[i], scale_bits)) & mask;
}

FixArray secure_matmul(FPMath &fp, const FixArray &a, const FixArray &b,
                       int rows, int common, int cols) {
  if (a.size != rows * common || b.size != common * cols ||
      a.ell != b.ell || a.s != b.s)
    throw std::runtime_error("secure matmul shape/metadata mismatch");
  if (gc_party == nullptr)
    throw std::runtime_error("GC MPC is not initialized");
  FixArray output(fp.party, rows * cols, true, kRingBits, a.s);
  std::vector<uint64_t> output_masks(rows * cols, 0);
  if (fp.party == sci::ALICE) {
    sci::PRG128 prg;
    prg.random_data(output_masks.data(),
                    output_masks.size() * sizeof(uint64_t));
    for (uint64_t &mask : output_masks) mask &= output.ell_mask();
  }
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < cols; ++col) {
      sci::Integer sum(2 * kRingBits, 0, sci::PUBLIC);
      for (int k = 0; k < common; ++k) {
        const uint64_t a_share = a.data[row * common + k];
        const uint64_t b_share = b.data[k * cols + col];
        sci::Integer a_alice(kRingBits,
            fp.party == sci::ALICE ? a_share : 0, sci::ALICE);
        sci::Integer a_bob(kRingBits,
            fp.party == sci::BOB ? a_share : 0, sci::BOB);
        sci::Integer b_alice(kRingBits,
            fp.party == sci::ALICE ? b_share : 0, sci::ALICE);
        sci::Integer b_bob(kRingBits,
            fp.party == sci::BOB ? b_share : 0, sci::BOB);
        sci::Integer a_secret = a_alice + a_bob;
        sci::Integer b_secret = b_alice + b_bob;
        a_secret.resize(2 * kRingBits, true);
        b_secret.resize(2 * kRingBits, true);
        sum = sum + a_secret * b_secret;
      }
      sci::Integer truncated;
      truncated.bits.assign(sum.bits.begin() + a.s,
                            sum.bits.begin() + a.s + kRingBits);
      const int index = row * cols + col;
      sci::Integer mask(kRingBits,
          fp.party == sci::ALICE ? output_masks[index] : 0, sci::ALICE);
      const uint64_t masked = (truncated - mask).reveal<uint64_t>(sci::BOB);
      output.data[index] = fp.party == sci::ALICE
                               ? output_masks[index]
                               : masked & output.ell_mask();
    }
  fp.iopack->io_GC->flush();
  return output;
}

FixArray secure_linear(FPMath &fp, const FixArray &input, int rows,
                       int in_dim, int out_dim,
                       const std::vector<double> &weights,
                       const std::vector<double> &bias, int role) {
  std::vector<uint64_t> transposed(
      static_cast<size_t>(in_dim) * out_dim, 0);
  if (role == sci::ALICE) {
    for (int in = 0; in < in_dim; ++in)
      for (int out = 0; out < out_dim; ++out)
        transposed[static_cast<size_t>(in) * out_dim + out] = encode_ring(
            weights[static_cast<size_t>(out) * in_dim + in], input.s);
  }
  FixArray secret_weight = fp.fix->input(
      sci::ALICE, in_dim * out_dim, transposed.data(), true, kRingBits,
      input.s);
  FixArray output =
      secure_matmul(fp, input, secret_weight, rows, in_dim, out_dim);
  if (!bias.empty() && role == sci::ALICE) {
    const uint64_t mask = (1ULL << output.ell) - 1;
    for (int row = 0; row < rows; ++row)
      for (int out = 0; out < out_dim; ++out)
        output.data[row * out_dim + out] =
            (output.data[row * out_dim + out] +
             encode_ring(bias[out], output.s)) & mask;
  }
  return output;
}

sci::Integer gc_fixed_mul(const sci::Integer &left,
                          const sci::Integer &right, int scale_bits);

FixArray mul_public(FPMath &fp, const FixArray &input, double scalar) {
  const int wide = 2 * kRingBits;
  FixArray output(fp.party, input.size, true, input.ell, input.s);
  std::vector<uint64_t> output_masks(output.size, 0);
  if (fp.party == sci::ALICE) {
    sci::PRG128 prg;
    prg.random_data(output_masks.data(),
                    output_masks.size() * sizeof(uint64_t));
    for (uint64_t &mask : output_masks) mask &= output.ell_mask();
  }
  sci::Integer constant(wide, encode_ring(scalar, input.s, wide),
                        sci::PUBLIC);
  for (int index = 0; index < input.size; ++index) {
    const uint64_t share = input.data[index];
    sci::Integer alice(input.ell,
        fp.party == sci::ALICE ? share : 0, sci::ALICE);
    sci::Integer bob(input.ell,
        fp.party == sci::BOB ? share : 0, sci::BOB);
    sci::Integer value = alice + bob;
    value.resize(wide, true);
    sci::Integer result = gc_fixed_mul(value, constant, input.s);
    result.resize(kRingBits, true);
    sci::Integer mask(kRingBits,
        fp.party == sci::ALICE ? output_masks[index] : 0, sci::ALICE);
    const uint64_t masked =
        (result - mask).reveal<uint64_t>(sci::BOB);
    output.data[index] = fp.party == sci::ALICE
                             ? output_masks[index]
                             : masked & output.ell_mask();
  }
  fp.iopack->io_GC->flush();
  return output;
}

FixArray blb_softmax(FPMath &fp, const FixArray &scores, int rows, int cols) {
  if (scores.size != rows * cols)
    throw std::runtime_error("softmax shape mismatch");
  FixArray output(fp.party, rows * cols, true, scores.ell, scores.s);
  std::vector<uint64_t> output_masks(output.size, 0);
  if (fp.party == sci::ALICE) {
    sci::PRG128 prg;
    prg.random_data(output_masks.data(),
                    output_masks.size() * sizeof(uint64_t));
    for (uint64_t &mask : output_masks) mask &= output.ell_mask();
  }
  const int wide = 2 * kRingBits;
  auto constant = [&](int64_t raw) {
    return sci::Integer(wide, raw, sci::PUBLIC);
  };
  for (int row = 0; row < rows; ++row) {
    std::vector<sci::Integer> values;
    values.reserve(cols);
    for (int col = 0; col < cols; ++col) {
      const uint64_t share = scores.data[row * cols + col];
      sci::Integer alice(scores.ell,
          fp.party == sci::ALICE ? share : 0, sci::ALICE);
      sci::Integer bob(scores.ell,
          fp.party == sci::BOB ? share : 0, sci::BOB);
      sci::Integer value = alice + bob;
      value.resize(wide, true);
      values.push_back(std::move(value));
    }
    sci::Integer maximum = values[0];
    for (int col = 1; col < cols; ++col)
      maximum = maximum.select(values[col].geq(maximum), values[col]);

    std::vector<sci::Integer> exponentials;
    exponentials.reserve(cols);
    sci::Integer exp_sum(wide, 0, sci::PUBLIC);
    for (int col = 0; col < cols; ++col) {
      sci::Integer shifted = values[col] - maximum;
      sci::Integer x_inl = gc_fixed_mul(shifted, constant(-5909), scores.s);
      sci::Integer reduction = x_inl >> scores.s;
      sci::Integer bound = constant(13);
      reduction = reduction.select(reduction.geq(bound), bound);
      sci::Integer reduction_fixed = reduction << scores.s;
      sci::Integer p = shifted +
          gc_fixed_mul(reduction_fixed, constant(2839), scores.s);
      sci::Integer p_arg = p + constant(5542);
      sci::Integer polynomial = gc_fixed_mul(p_arg, p_arg, scores.s);
      polynomial = gc_fixed_mul(polynomial, constant(1468), scores.s) +
                   constant(1409);
      sci::Integer exponential = polynomial >> reduction;
      exp_sum = exp_sum + exponential;
      exponentials.push_back(std::move(exponential));
    }
    for (int col = 0; col < cols; ++col) {
      sci::Integer numerator = exponentials[col] << scores.s;
      sci::Integer probability = numerator / exp_sum;
      probability.resize(kRingBits, true);
      const int index = row * cols + col;
      sci::Integer mask(kRingBits,
          fp.party == sci::ALICE ? output_masks[index] : 0, sci::ALICE);
      const uint64_t masked =
          (probability - mask).reveal<uint64_t>(sci::BOB);
      output.data[index] = fp.party == sci::ALICE
                               ? output_masks[index]
                               : masked & output.ell_mask();
    }
  }
  fp.iopack->io_GC->flush();
  return output;
}

FixArray layer_norm(FPMath &fp, const FixArray &input, int rows, int dim,
                    const std::vector<double> &weight,
                    const std::vector<double> &bias, int role) {
  if (input.size != rows * dim)
    throw std::runtime_error("layer norm shape mismatch");
  FixArray output(fp.party, rows * dim, true, input.ell, input.s);
  std::vector<uint64_t> output_masks(output.size, 0);
  if (role == sci::ALICE) {
    sci::PRG128 prg;
    prg.random_data(output_masks.data(),
                    output_masks.size() * sizeof(uint64_t));
    for (uint64_t &mask : output_masks) mask &= output.ell_mask();
  }
  const int wide = 2 * kRingBits;
  for (int row = 0; row < rows; ++row) {
    std::vector<sci::Integer> values;
    values.reserve(dim);
    sci::Integer sum(wide, 0, sci::PUBLIC);
    for (int channel = 0; channel < dim; ++channel) {
      const uint64_t share = input.data[row * dim + channel];
      sci::Integer alice(input.ell,
          role == sci::ALICE ? share : 0, sci::ALICE);
      sci::Integer bob(input.ell,
          role == sci::BOB ? share : 0, sci::BOB);
      sci::Integer value = alice + bob;
      value.resize(wide, true);
      sum = sum + value;
      values.push_back(std::move(value));
    }
    sci::Integer divisor(wide, dim, sci::PUBLIC);
    sci::Integer mean = sum / divisor;
    std::vector<sci::Integer> centered;
    centered.reserve(dim);
    sci::Integer variance_sum(wide, 0, sci::PUBLIC);
    for (int channel = 0; channel < dim; ++channel) {
      sci::Integer delta = values[channel] - mean;
      variance_sum = variance_sum + delta * delta;
      centered.push_back(std::move(delta));
    }
    sci::Integer variance = variance_sum / divisor;

    sci::Integer sigma(wide, 0, sci::PUBLIC);
    for (int bit = kRingBits - 1; bit >= 0; --bit) {
      sci::Integer candidate = sigma |
          sci::Integer(wide, static_cast<uint64_t>(1) << bit, sci::PUBLIC);
      sci::Integer square = candidate * candidate;
      sigma = sigma.select(variance.geq(square), candidate);
    }

    for (int channel = 0; channel < dim; ++channel) {
      sci::Integer numerator = centered[channel] << input.s;
      sci::Integer normalized = numerator / sigma;
      normalized.resize(kRingBits, true);
      normalized.resize(wide, true);
      const uint64_t gamma_raw =
          role == sci::ALICE ? encode_ring(weight[channel], input.s) : 0;
      const uint64_t beta_raw =
          role == sci::ALICE ? encode_ring(bias[channel], input.s) : 0;
      sci::Integer gamma_small(kRingBits, gamma_raw, sci::ALICE);
      gamma_small.resize(wide, true);
      sci::Integer affine_product = normalized * gamma_small;
      sci::Integer affine;
      affine.bits.assign(affine_product.bits.begin() + input.s,
                         affine_product.bits.begin() + input.s + kRingBits);
      sci::Integer beta(kRingBits, beta_raw, sci::ALICE);
      affine = affine + beta;
      const int index = row * dim + channel;
      sci::Integer mask(kRingBits,
          role == sci::ALICE ? output_masks[index] : 0, sci::ALICE);
      const uint64_t masked =
          (affine - mask).reveal<uint64_t>(sci::BOB);
      output.data[index] = role == sci::ALICE
                               ? output_masks[index]
                               : masked & output.ell_mask();
    }
  }
  fp.iopack->io_GC->flush();
  return output;
}

sci::Integer gc_fixed_mul(const sci::Integer &left,
                          const sci::Integer &right, int scale_bits) {
  sci::Integer product = left * right;
  sci::Integer shifted;
  shifted.bits.assign(product.bits.begin() + scale_bits,
                      product.bits.end());
  shifted.resize(product.size(), true);
  return shifted;
}

FixArray gelu_gc(FPMath &fp, const FixArray &input) {
  const int wide = 2 * kRingBits;
  FixArray output(fp.party, input.size, true, input.ell, input.s);
  std::vector<uint64_t> output_masks(output.size, 0);
  if (fp.party == sci::ALICE) {
    sci::PRG128 prg;
    prg.random_data(output_masks.data(),
                    output_masks.size() * sizeof(uint64_t));
    for (uint64_t &mask : output_masks) mask &= output.ell_mask();
  }
  auto constant = [&](double value) {
    return sci::Integer(wide, encode_ring(value, input.s, wide), sci::PUBLIC);
  };
  for (int index = 0; index < input.size; ++index) {
    const uint64_t share = input.data[index];
    sci::Integer alice(input.ell,
        fp.party == sci::ALICE ? share : 0, sci::ALICE);
    sci::Integer bob(input.ell,
        fp.party == sci::BOB ? share : 0, sci::BOB);
    sci::Integer x = alice + bob;
    x.resize(wide, true);
    sci::Integer y = x.abs();
    sci::Integer z = gc_fixed_mul(
        y, y + constant(-4.901373660634577), input.s);
    sci::Integer lmul = z + y + constant(-24.822721454603112);
    sci::Integer rmul = z + constant(31.65224017185284);
    sci::Integer polynomial =
        gc_fixed_mul(lmul, rmul, input.s) + constant(785.77248299658);
    polynomial = gc_fixed_mul(
        polynomial, constant(0.02084861175412759), input.s);
    sci::Integer middle =
        gc_fixed_mul(x, constant(0.5), input.s) + polynomial;
    const sci::Bit outside =
        y.geq(constant(2.7 + 1.0 / (1ULL << input.s)));
    const sci::Bit nonnegative = x.geq(constant(0.0));
    sci::Integer relu = constant(0.0).select(nonnegative, x);
    sci::Integer result = middle.select(outside, relu);
    result.resize(kRingBits, true);
    sci::Integer mask(kRingBits,
        fp.party == sci::ALICE ? output_masks[index] : 0, sci::ALICE);
    const uint64_t masked =
        (result - mask).reveal<uint64_t>(sci::BOB);
    output.data[index] = fp.party == sci::ALICE
                             ? output_masks[index]
                             : masked & output.ell_mask();
  }
  fp.iopack->io_GC->flush();
  return output;
}

FixArray attention(FPMath &fp, const FixArray &q, const FixArray &k,
                   const FixArray &v, const VitModel &model, int role) {
  const int head_dim = model.dim / model.heads;
  FixArray joined(fp.party, model.patches * model.dim, true, kRingBits,
                  model.scale_bits);
  for (int head = 0; head < model.heads; ++head) {
    FixArray q_head(fp.party, model.patches * head_dim, true, kRingBits,
                    model.scale_bits);
    FixArray kt_head(fp.party, head_dim * model.patches, true, kRingBits,
                     model.scale_bits);
    FixArray v_head(fp.party, model.patches * head_dim, true, kRingBits,
                    model.scale_bits);
    for (int patch = 0; patch < model.patches; ++patch)
      for (int channel = 0; channel < head_dim; ++channel) {
        const int source = patch * model.dim + head * head_dim + channel;
        q_head.data[patch * head_dim + channel] = q.data[source];
        kt_head.data[channel * model.patches + patch] = k.data[source];
        v_head.data[patch * head_dim + channel] = v.data[source];
      }
    FixArray scores = secure_matmul(
        fp, q_head, kt_head, model.patches, head_dim, model.patches);
    scores = mul_public(
        fp, scores, 1.0 / std::sqrt(static_cast<double>(head_dim)));
    FixArray weights =
        blb_softmax(fp, scores, model.patches, model.patches);
    FixArray head_output = secure_matmul(
        fp, weights, v_head, model.patches, model.patches, head_dim);
    for (int patch = 0; patch < model.patches; ++patch)
      for (int channel = 0; channel < head_dim; ++channel)
        joined.data[patch * model.dim + head * head_dim + channel] =
            head_output.data[patch * head_dim + channel];
  }
  return secure_linear(fp, joined, model.patches, model.dim, model.dim,
                       model.wo, model.bo, role);
}

FixArray mean_pool(FPMath &fp, const FixArray &input, int rows, int dim) {
  FixArray sums(fp.party, dim, true, input.ell, input.s);
  std::fill(sums.data, sums.data + dim, 0);
  const uint64_t mask = (1ULL << input.ell) - 1;
  for (int row = 0; row < rows; ++row)
    for (int channel = 0; channel < dim; ++channel)
      sums.data[channel] =
          (sums.data[channel] + input.data[row * dim + channel]) & mask;
  return mul_public(fp, sums, 1.0 / rows);
}

int reconstruct_logits(sci::IOPack &io, const FixArray &logits, int role,
                       const std::string &path) {
  std::vector<uint64_t> peer(logits.size);
  if (role == sci::ALICE) {
    io.io->send_data(logits.data, logits.size * sizeof(uint64_t));
    io.io->flush();
    return -1;
  }
  io.io->recv_data(peer.data(), logits.size * sizeof(uint64_t));
  const uint64_t mask = (1ULL << logits.ell) - 1;
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write logits: " + path);
  out << std::setprecision(12);
  int predicted = 0;
  double best = -INFINITY;
  for (int i = 0; i < logits.size; ++i) {
    const uint64_t raw = (peer[i] + logits.data[i]) & mask;
    const int64_t signed_value =
        raw >= (1ULL << (logits.ell - 1))
            ? static_cast<int64_t>(raw) -
                  static_cast<int64_t>(1ULL << logits.ell)
            : static_cast<int64_t>(raw);
    const double value =
        static_cast<double>(signed_value) / (1ULL << logits.s);
    out << value << (i + 1 == logits.size ? '\n' : ' ');
    if (value > best) {
      best = value;
      predicted = i;
    }
  }
  return predicted;
}

void debug_stage(sci::IOPack &io, const FixArray &value, int role,
                 const std::string &prefix, const std::string &name,
                 bool enabled) {
  if (enabled)
    (void)reconstruct_logits(io, value, role,
                             prefix + "_debug_" + name + ".txt");
}

FixArray run_vit(FPMath &fp, CkksLinear &he, sci::IOPack &io,
                 const VitModel &model,
                 const std::vector<double> &private_patches, int role,
                 const std::string &debug_prefix = "") {
  const bool debug = !debug_prefix.empty();
  const std::vector<double> q_weight =
      role == sci::ALICE ? compose_weights(model.wq, model.patch_w, model.dim)
                         : std::vector<double>(model.dim * model.dim, 0.0);
  const std::vector<double> k_weight =
      role == sci::ALICE ? compose_weights(model.wk, model.patch_w, model.dim)
                         : std::vector<double>(model.dim * model.dim, 0.0);
  const std::vector<double> v_weight =
      role == sci::ALICE ? compose_weights(model.wv, model.patch_w, model.dim)
                         : std::vector<double>(model.dim * model.dim, 0.0);
  FixArray embedded = he.project_private(
      fp, model.patch_w, private_patches, model.patches, model.dim,
      model.scale_bits);
  std::vector<double> embed_bias(
      static_cast<size_t>(model.patches) * model.dim, 0.0);
  for (int patch = 0; patch < model.patches; ++patch)
    for (int channel = 0; channel < model.dim; ++channel)
      embed_bias[patch * model.dim + channel] =
          model.patch_b[channel] + model.pos[patch * model.dim + channel];
  add_alice_values(embedded, embed_bias, model.scale_bits, role);
  debug_stage(io, embedded, role, debug_prefix, "embedded", debug);

  FixArray q = he.project_private(fp, q_weight, private_patches,
                                  model.patches, model.dim, model.scale_bits);
  FixArray k = he.project_private(fp, k_weight, private_patches,
                                  model.patches, model.dim, model.scale_bits);
  FixArray v = he.project_private(fp, v_weight, private_patches,
                                  model.patches, model.dim, model.scale_bits);
  add_alice_values(q, composed_bias(model, model.wq, model.bq),
                   model.scale_bits, role);
  add_alice_values(k, composed_bias(model, model.wk, model.bk),
                   model.scale_bits, role);
  add_alice_values(v, composed_bias(model, model.wv, model.bv),
                   model.scale_bits, role);
  debug_stage(io, q, role, debug_prefix, "q", debug);
  debug_stage(io, k, role, debug_prefix, "k", debug);
  debug_stage(io, v, role, debug_prefix, "v", debug);

  FixArray attention_output = attention(fp, q, k, v, model, role);
  debug_stage(io, attention_output, role, debug_prefix, "attention", debug);
  FixArray first_residual = fp.fix->add(embedded, attention_output);
  FixArray first_norm = layer_norm(fp, first_residual, model.patches,
                                   model.dim, model.ln1_w, model.ln1_b, role);
  debug_stage(io, first_norm, role, debug_prefix, "ln1", debug);
  FixArray hidden = secure_linear(fp, first_norm, model.patches, model.dim,
                                  model.mlp_dim, model.ff1_w, model.ff1_b,
                                  role);
  hidden = gelu_gc(fp, hidden);
  debug_stage(io, hidden, role, debug_prefix, "gelu", debug);
  FixArray feed_forward = secure_linear(
      fp, hidden, model.patches, model.mlp_dim, model.dim, model.ff2_w,
      model.ff2_b, role);
  debug_stage(io, feed_forward, role, debug_prefix, "ff", debug);
  FixArray second_residual = fp.fix->add(first_norm, feed_forward);
  FixArray second_norm = layer_norm(fp, second_residual, model.patches,
                                    model.dim, model.ln2_w, model.ln2_b,
                                    role);
  debug_stage(io, second_norm, role, debug_prefix, "ln2", debug);
  FixArray pooled = mean_pool(fp, second_norm, model.patches, model.dim);
  debug_stage(io, pooled, role, debug_prefix, "pool", debug);
  FixArray head_norm = layer_norm(fp, pooled, 1, model.dim,
                                  model.head_ln_w, model.head_ln_b, role);
  debug_stage(io, head_norm, role, debug_prefix, "head_ln", debug);
  return secure_linear(fp, head_norm, 1, model.dim, model.classes,
                       model.classifier_w, model.classifier_b, role);
}

} // namespace

int main(int argc, char **argv) {
  std::string model_path;
  std::string input_path;
  std::string variant = "blb";
  std::string he_backend = "packed";
  std::string output_prefix = "secure_output";
  int debug = 0;
  int warmup = 0;
  ArgMapping args;
  args.arg("r", party, "Role: ALICE=1, BOB=2");
  args.arg("p", port, "Base TCP port");
  args.arg("ip", address, "ALICE address (BOB only)");
  args.arg("model", model_path, "ALICE-only end-to-end ViT model");
  args.arg("input", input_path, "BOB-only private patch input");
  args.arg("variant", variant, "Must be blb");
  args.arg("he", he_backend, "Compatibility option; must be packed");
  args.arg("output", output_prefix, "BOB output prefix");
  args.arg("debug", debug, "Reconstruct intermediate values to BOB (unsafe)");
  args.arg("warmup", warmup, "Run one unmeasured warmup inference");
  args.parse(argc, argv);

  try {
    if (party != sci::ALICE && party != sci::BOB)
      throw std::runtime_error("r must be 1 or 2");
    if (variant != "blb" || he_backend != "packed")
      throw std::runtime_error("end-to-end adapter requires variant=blb he=packed");
    (void)debug;

    VitModel model;
    if (party == sci::ALICE) model = load_model(model_path);
    sci::IOPack io(party, port, address);
    sci::OTPack ot(&io, party);
    FPMath fp(party, &io, &ot);
    setup_gc(io, ot, party);
    WireConfig config{};
    if (party == sci::ALICE) {
      config = {model.patches, model.patch_dim, model.dim, model.heads,
                model.mlp_dim, model.classes, model.scale_bits};
      io.io->send_data(&config, sizeof(config));
      io.io->flush();
    } else {
      io.io->recv_data(&config, sizeof(config));
      model.patches = config.patches;
      model.patch_dim = config.patch_dim;
      model.dim = config.dim;
      model.heads = config.heads;
      model.mlp_dim = config.mlp_dim;
      model.classes = config.classes;
      model.scale_bits = config.scale_bits;
      model.patch_w.assign(static_cast<size_t>(model.dim) * model.dim, 0.0);
      model.patch_b.assign(model.dim, 0.0);
      model.pos.assign(static_cast<size_t>(model.patches) * model.dim, 0.0);
      model.wq.assign(static_cast<size_t>(model.dim) * model.dim, 0.0);
      model.bq.assign(model.dim, 0.0);
      model.wk = model.wq; model.bk = model.bq;
      model.wv = model.wq; model.bv = model.bq;
      model.wo = model.wq; model.bo = model.bq;
      model.ln1_w.assign(model.dim, 0.0); model.ln1_b = model.ln1_w;
      model.ff1_w.assign(static_cast<size_t>(model.mlp_dim) * model.dim, 0.0);
      model.ff1_b.assign(model.mlp_dim, 0.0);
      model.ff2_w.assign(static_cast<size_t>(model.dim) * model.mlp_dim, 0.0);
      model.ff2_b.assign(model.dim, 0.0);
      model.ln2_w = model.ln1_w; model.ln2_b = model.ln1_w;
      model.head_ln_w = model.ln1_w; model.head_ln_b = model.ln1_w;
      model.classifier_w.assign(
          static_cast<size_t>(model.classes) * model.dim, 0.0);
      model.classifier_b.assign(model.classes, 0.0);
    }

    std::vector<double> private_patches(
        static_cast<size_t>(model.patches) * model.dim, 0.0);
    if (party == sci::BOB) {
      const VitInput input = load_input(input_path);
      if (input.patches != model.patches || input.patch_dim != model.patch_dim)
        throw std::runtime_error("BOB input dimensions do not match model");
      private_patches = padded_input(input, model.dim);
    }

    const auto setup_start = std::chrono::steady_clock::now();
    CkksLinear he(party, io.io, model.dim);
    io.io->sync();
    const auto setup_end = std::chrono::steady_clock::now();
    if (party == sci::BOB)
      std::cout << "SETUP milliseconds="
                << std::chrono::duration<double, std::milli>(setup_end - setup_start).count()
                << " patches=" << model.patches << " patch_dim=" << model.patch_dim
                << " dim=" << model.dim << " heads=" << model.heads
                << " mlp_dim=" << model.mlp_dim << " classes=" << model.classes
                << " he_scheme=ckks he_backend=" << he.backend_name()
                << " he_device=" << he.device_name() << " mpc_device=cpu"
                << " model_scope=tiny_vit_protocol_chain_smoke"
                << " formal_result=no" << std::endl;

    if (warmup) {
      FixArray unused = run_vit(fp, he, io, model, private_patches, party);
      reconstruct_logits(io, unused, party, output_prefix + "_warmup.txt");
      io.io->sync();
    }
    he.synchronize();
    const uint64_t bytes_before = io.get_comm();
    const uint64_t rounds_before = io.get_rounds();
    const auto start = std::chrono::steady_clock::now();
    FixArray logits = run_vit(fp, he, io, model, private_patches, party,
                              debug ? output_prefix : "");
    const int predicted = reconstruct_logits(
        io, logits, party, output_prefix + "_vit_smoke.txt");
    he.synchronize();
    const auto end = std::chrono::steady_clock::now();
    LocalStats local{io.get_comm() - bytes_before,
                     io.get_rounds() - rounds_before,
                     std::chrono::duration<double, std::milli>(end - start).count()};
    LocalStats aggregate{};
    if (party == sci::BOB) {
      io.io->send_data(&local, sizeof(local));
      io.io->flush();
      io.io->recv_data(&aggregate, sizeof(aggregate));
    } else {
      LocalStats peer{};
      io.io->recv_data(&peer, sizeof(peer));
      aggregate = blb_vit::aggregate_parties(local, peer);
      io.io->send_data(&aggregate, sizeof(aggregate));
      io.io->flush();
    }
    if (party == sci::BOB) {
      std::cout << "RESULT variant=vit_smoke formal_result=no he_backend=" << he.backend_name()
                << " online_ms=" << std::fixed << std::setprecision(3)
                << aggregate.milliseconds << " total_bytes=" << aggregate.bytes
                << " total_mib=" << aggregate.bytes / 1048576.0
                << " rounds_sum=" << aggregate.rounds
                << " he_device=" << he.device_name() << " mpc_device=cpu"
                << " scale_bits=" << model.scale_bits
                << " ring_bits=" << kRingBits
                << " protocol_profile=vit_smoke_e2e_" << he.device_name()
                << " he_scheme=ckks"
                << " ckks_poly_degree=8192 ckks_coeff_modulus=60_40_60"
                << " ckks_scale_bits=40 ckks_to_mpc=slot_additive_mask"
                << " he_layers=patch_embedding_fused_qkv"
                << " mpc_backend=sci_halfgate_gc_iknp"
                << " mpc_layers=attention_softmax_wo_residual_layernorm_ffn_gelu_pool_classifier"
                << " softmax_profile=exp4_poly1"
                << " architecture=postnorm_meanpool_vit"
                << " publishable_performance=no" << std::endl;
      std::cout << "VIT_SMOKE_RESULT status=PASS formal_result=no predicted_class=" << predicted
                << " logits=" << model.classes
                << " output=" << output_prefix << "_vit_smoke.txt" << std::endl;
    }
    teardown_gc();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "vit_smoke_e2e error: " << error.what() << std::endl;
    return 2;
  }
}
