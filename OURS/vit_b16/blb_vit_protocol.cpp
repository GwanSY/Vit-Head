#include "blb_vit_protocol.h"

#include "LinearHE/defines-HE.h"
#ifdef BLB_VIT_VALIDATION_OUTPUTS
#include "vit_validation_observer.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace sci;
using namespace sci::bolt;

namespace {

constexpr int kEll = 37;
constexpr int kScale = 12;
constexpr int kQKOperandScale = 10;

std::uint64_t mask(int ell = kEll) {
  return ell == 64 ? ~0ULL : ((1ULL << ell) - 1);
}

std::uint64_t ring_encode(std::int64_t value, int ell = kEll) {
  return static_cast<std::uint64_t>(value) & mask(ell);
}

std::int64_t quantize(double value, int scale) {
  return static_cast<std::int64_t>(
      std::llround(value * std::ldexp(1.0, scale)));
}

std::uint64_t prime_encode(std::int64_t value, std::uint64_t modulus) {
  if (value >= 0) return static_cast<std::uint64_t>(value) % modulus;
  const std::uint64_t magnitude = static_cast<std::uint64_t>(-value) % modulus;
  return magnitude == 0 ? 0 : modulus - magnitude;
}

std::string shape(std::size_t rows, std::size_t cols) {
  return "[" + std::to_string(rows) + "," + std::to_string(cols) + "]";
}

std::uint64_t elapsed_us(
    const std::chrono::steady_clock::time_point &start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start).count());
}

BoltTensorMetadata tensor_metadata(std::size_t rows, std::size_t cols,
                                   std::size_t stride, int ell, int scale,
                                   bool add_half = false) {
  BoltTensorMetadata metadata;
  metadata.rows = rows;
  metadata.cols = cols;
  metadata.packed_stride = stride;
  metadata.share_bit_width = ell;
  metadata.scale = scale;
  metadata.packing = BoltPackingKind::kCompactRepeatedRows;
  metadata.add_plain_modulus_half_on_he_to_ss = add_half;
  return metadata;
}

BoltMatMulMetadata matmul_metadata(std::size_t rows, std::size_t input,
                                   std::size_t output, std::size_t stride,
                                   bool harden) {
  BoltMatMulMetadata metadata;
  metadata.input_rows = rows;
  metadata.input_cols = input;
  metadata.output_cols = output;
  metadata.packed_stride = stride;
  metadata.baby_step = 1;
  while (metadata.baby_step * metadata.baby_step < stride)
    metadata.baby_step <<= 1;
  // Match BLB's ell=37 statistical-hiding budget.  The BERT2 constant is
  // specialized for a 19-bit intermediate and over-floods the wider ViT
  // plaintext field, exhausting the BFV noise budget after ct-ct products.
  // p is the established 29-bit BERT batching prime.  A 71-bit fresh flood
  // gives the same 100-bit statistical circuit-privacy target used by BLB's
  // BERT HE path; the existing modulus-switch primitive is unchanged.
  metadata.noise_flooding_bits = harden ? 71 : 0;
  metadata.modulus_switches = harden ? 2 : 0;
  return metadata;
}

std::vector<FixArray> split_rows(const FixArray &flat, std::size_t rows,
                                 std::size_t cols) {
  if (flat.size != static_cast<int>(rows * cols))
    throw std::invalid_argument("ViT row split shape mismatch");
  std::vector<FixArray> result(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    result[row] = FixArray(flat.party, static_cast<int>(cols), true,
                           flat.ell, flat.s);
    std::memcpy(result[row].data, flat.data + row * cols,
                cols * sizeof(std::uint64_t));
  }
  return result;
}

FixArray flatten(const std::vector<FixArray> &rows) { return concat(rows); }

} // namespace

BoltVitProtocol::BoltVitProtocol(int party, IOPack &io, OTPack &ot, FPMath &fp,
                                 BoltHE &he,
                                 const BoltVitModelMetadata &metadata)
    : party_(party), io_(io), ot_(ot), fp_(fp), he_(he), conversion_(he.view()),
      linear_(he.view()), packing_(he.view()), metadata_(metadata) {
  metadata_.validate(he_.slot_count());
  fail_if_scalar_ciphertext_adapter(false);
  fail_if_generic_ot_matmul(false);
  if (compact_compat_enabled()) {
    compat_packing_ = std::make_unique<BoltVitCompatPacking>(
        he_.slot_count(), metadata_.physical_tokens, metadata_.model_dim,
        metadata_.heads, metadata_.head_dim);
    compat_conversion_ = std::make_unique<BoltVitCompatConversion>(
        he_.view(), fp_);
    compat_linear_ = std::make_unique<BoltVitCompatLinearAdapter>(
        he_.view(), *compat_packing_);
    compat_attention_ = std::make_unique<BoltVitCompatAttentionAdapter>(
        he_.view(), *compat_packing_);
  }
}

bool BoltVitProtocol::compact_compat_enabled() const {
  return he_.slot_count() == 8192 && metadata_.image_size == 224 &&
         metadata_.patch_size == 16 && metadata_.channels == 3 &&
         metadata_.patch_dim == 768 && metadata_.tokens == 197 &&
         metadata_.physical_tokens == 256 && metadata_.model_dim == 768 &&
         metadata_.heads == 12 && metadata_.head_dim == 64 &&
         metadata_.ff_dim == 3072 && metadata_.layers == 12 &&
         metadata_.classes == 1000;
}

void BoltVitProtocol::apply_public_token_mask(
    FixArray &value, const std::string &name, int layer,
    std::vector<VitOperatorTrace> &trace) {
  const std::size_t physical = metadata_.physical_tokens;
  const std::size_t logical = metadata_.tokens;
  if (value.size != static_cast<int>(physical * metadata_.model_dim) ||
      value.ell != static_cast<int>(metadata_.ell)) {
    throw std::invalid_argument("ViT public token mask shape mismatch: " + name);
  }
  const std::uint64_t before_bytes = io_.get_comm();
  const std::uint64_t before_rounds = io_.get_rounds();
  const auto start = std::chrono::steady_clock::now();
  // Multiply by a fixed, public binary mask through the existing FixOp
  // public/secret multiplication path.  No secret predicate or
  // model-dependent branch is used.
  std::vector<std::uint64_t> mask_data(value.size, 1);
  for (std::size_t token = logical; token < physical; ++token)
    for (std::size_t col = 0; col < metadata_.model_dim; ++col)
      mask_data[token * metadata_.model_dim + col] = 0;
  FixArray public_mask = fp_.fix->input(
      PUBLIC, value.size, mask_data.data(), true, value.ell, 0);
  value = fp_.fix->mul(value, public_mask, value.ell);
  VitOperatorTrace row;
  row.phase = "online";
  row.layer = layer;
  row.op = name;
  row.input_shape = row.output_shape =
      shape(physical, metadata_.model_dim);
  row.input_packing = row.output_packing =
      "physical_" + std::to_string(physical) +
      "_public_token_mask_valid_" + std::to_string(logical);
  row.input_scale = row.output_scale = value.s;
  row.ell = value.ell;
  row.backend = "bolt_public_fix_mul_token_mask";
  append_trace(trace, std::move(row), before_bytes, before_rounds,
               elapsed_us(start));
}

void BoltVitProtocol::apply_public_attention_mask(
    FixArray &value, const std::string &name, int layer,
    std::vector<VitOperatorTrace> &trace) {
  const std::size_t physical = metadata_.physical_tokens;
  const std::size_t logical = metadata_.tokens;
  if (value.size != static_cast<int>(physical * physical) ||
      value.ell != static_cast<int>(metadata_.ell)) {
    throw std::invalid_argument("ViT public attention mask shape mismatch: " + name);
  }
  const std::uint64_t before_bytes = io_.get_comm();
  const std::uint64_t before_rounds = io_.get_rounds();
  const auto start = std::chrono::steady_clock::now();
  // The mask removes both padding keys and padding queries through the
  // existing public/secret FixOp multiplication.  The valid logical
  // submatrix is copied to MPC Softmax below, so padding never enters its
  // denominator.
  std::vector<std::uint64_t> mask_data(value.size, 1);
  for (std::size_t row = 0; row < physical; ++row)
    for (std::size_t col = 0; col < physical; ++col)
      if (row >= logical || col >= logical)
        mask_data[row * physical + col] = 0;
  FixArray public_mask = fp_.fix->input(
      PUBLIC, value.size, mask_data.data(), true, value.ell, 0);
  value = fp_.fix->mul(value, public_mask, value.ell);
  VitOperatorTrace row;
  row.phase = "online";
  row.layer = layer;
  row.op = name;
  row.input_shape = row.output_shape =
      shape(physical, physical);
  row.input_packing = "physical_qk_cross_rows";
  row.output_packing = "public_mask_valid_" + std::to_string(logical) +
                       "_by_" + std::to_string(logical) +
                       "_before_mpc_softmax";
  row.input_scale = row.output_scale = value.s;
  row.ell = value.ell;
  row.backend = "bolt_public_fix_mul_attention_padding_mask";
  append_trace(trace, std::move(row), before_bytes, before_rounds,
               elapsed_us(start));
}

void BoltVitProtocol::append_trace(std::vector<VitOperatorTrace> &trace,
                                   VitOperatorTrace row,
                                   std::uint64_t before_bytes,
                                   std::uint64_t before_rounds,
                                   std::uint64_t elapsed_microseconds) {
  row.plain_modulus = he_.parameters().plain_modulus;
  row.bytes = io_.get_comm() - before_bytes;
  row.logical_rounds = io_.get_rounds() - before_rounds;
  row.milliseconds = static_cast<double>(elapsed_microseconds) / 1000.0;
  trace.push_back(std::move(row));
}

void BoltVitProtocol::append_conversion_trace(
    std::vector<VitOperatorTrace> &trace, const std::string &op, int layer,
    std::size_t rows, std::size_t cols, int scale,
    const std::string &input_packing, const std::string &output_packing,
    const std::string &backend, std::uint64_t before_bytes,
    std::uint64_t before_rounds, std::uint64_t elapsed_microseconds) {
  VitOperatorTrace row;
  row.phase = "online";
  row.layer = layer;
  row.op = op;
  row.input_shape = row.output_shape = shape(rows, cols);
  row.input_packing = input_packing;
  row.output_packing = output_packing;
  row.input_scale = row.output_scale = scale;
  row.ell = kEll;
  row.backend = backend;
  append_trace(trace, std::move(row), before_bytes, before_rounds,
               elapsed_microseconds);
}

BoltCipherTensor BoltVitProtocol::shares_to_cipher(const FixArray &input,
                                                   std::size_t rows,
                                                   std::size_t cols,
                                                   std::size_t stride) {
  if (input.size != static_cast<int>(rows * cols) || !input.signed_)
    throw std::invalid_argument("ViT SS-to-HE input metadata mismatch");
  SecretSharedTensor shares;
  shares.metadata = tensor_metadata(rows, cols, stride, input.ell, input.s);
  shares.local_share.assign(input.data, input.data + input.size);
  BoltCipherTensor result = conversion_.ss_to_he_signed_ring(
      shares, shares.metadata, *fp_.fix);
  return result;
}

FixArray BoltVitProtocol::cipher_to_ring(const BoltCipherTensor &cipher,
                                         std::size_t rows, std::size_t cols,
                                         std::size_t stride, int scale) {
  BoltTensorMetadata metadata =
      tensor_metadata(rows, cols, stride, kEll, scale, true);
  BoltCipherTensor view = cipher;
  view.metadata = metadata;
  SecretSharedTensor field = conversion_.he_to_ss(view, metadata);
  FixArray input(party_, static_cast<int>(field.local_share.size()), true,
                 kEll, scale);
  std::memcpy(input.data, field.local_share.data(),
              field.local_share.size() * sizeof(std::uint64_t));
  FixArray p = fp_.fix->input(PUBLIC, input.size, he_.parameters().plain_modulus,
                              true, kEll, scale);
  FixArray p_half = fp_.fix->input(
      PUBLIC, input.size, (he_.parameters().plain_modulus - 1) / 2,
      true, kEll, scale);
  return fp_.fix->sub(fp_.gt_p_sub(input, p), p_half);
}

std::vector<seal::Ciphertext> BoltVitProtocol::compat_slots_to_cipher(
    const std::vector<std::uint64_t> &packed_local_shares,
    int ell, int scale, BoltVitCompatConversionStats *stats) {
  if (!compact_compat_enabled() || compat_conversion_ == nullptr ||
      packed_local_shares.empty() ||
      packed_local_shares.size() % he_.slot_count() != 0 ||
      packed_local_shares.size() >
          static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("ViT compact SS-to-HE slot shape mismatch");
  FixArray packed(party_, static_cast<int>(packed_local_shares.size()), true,
                  ell, scale);
  std::memcpy(packed.data, packed_local_shares.data(),
              packed_local_shares.size() * sizeof(std::uint64_t));
  return compat_conversion_->ss_to_he_arbitrary_slots_signed(packed, stats);
}

FixArray BoltVitProtocol::compat_packed_cipher_to_ring(
    const std::vector<seal::Ciphertext> &ciphertexts,
    std::size_t ciphertext_count, int scale) {
  if (!compact_compat_enabled() || ciphertext_count == 0 ||
      ciphertext_count * he_.slot_count() >
          static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("ViT compact HE-to-SS slot shape mismatch");
  std::vector<std::uint64_t> field = conversion_.he_to_ss_packed_compat(
      ciphertexts, ciphertext_count, true);
  FixArray input(party_, static_cast<int>(field.size()), true, kEll, scale);
  std::memcpy(input.data, field.data(), field.size() * sizeof(std::uint64_t));
  FixArray p = fp_.fix->input(PUBLIC, input.size,
                              he_.parameters().plain_modulus,
                              true, kEll, scale);
  FixArray p_half = fp_.fix->input(
      PUBLIC, input.size, (he_.parameters().plain_modulus - 1) / 2,
      true, kEll, scale);
  return fp_.fix->sub(fp_.gt_p_sub(input, p), p_half);
}

FixArray BoltVitProtocol::compat_packed_linear(
    const FixArray &input, std::size_t rows, std::size_t input_dim,
    std::size_t output_dim, const std::vector<double> &weight,
    const std::vector<double> &bias, int output_scale,
    const std::string &op, int layer,
    std::vector<VitOperatorTrace> &trace, int weight_scale,
    double weight_multiplier) {
  if (!compact_compat_enabled() || compat_packing_ == nullptr ||
      compat_linear_ == nullptr || rows == 0 ||
      rows > metadata_.physical_tokens ||
      input.size != static_cast<int>(rows * input_dim) ||
      input_dim % 32 != 0 || output_dim % 32 != 0 ||
      input.s + weight_scale < output_scale ||
      !std::isfinite(weight_multiplier) || weight_multiplier <= 0.0)
    throw std::invalid_argument("ViT compact linear input metadata mismatch");
  const int raw_scale = input.s + weight_scale;
  FixArray padded(party_, metadata_.physical_tokens * input_dim, true,
                  input.ell, input.s);
  std::fill(padded.data, padded.data + padded.size, 0);
  std::memcpy(padded.data, input.data,
              static_cast<std::size_t>(input.size) * sizeof(std::uint64_t));
  std::vector<std::uint64_t> token_major(
      padded.data, padded.data + padded.size);
  std::vector<std::uint64_t> packed =
      compat_packing_->pack_token_major(token_major, input_dim);

  const std::uint64_t conversion_in_before_bytes = io_.get_comm();
  const std::uint64_t conversion_in_before_rounds = io_.get_rounds();
  const auto conversion_in_start = std::chrono::steady_clock::now();
  BoltVitCompatConversionStats conversion_stats;
  std::vector<seal::Ciphertext> encrypted = compat_slots_to_cipher(
      packed, input.ell, input.s, &conversion_stats);
  VitOperatorTrace conversion_in;
  conversion_in.phase = "online";
  conversion_in.layer = layer;
  conversion_in.op = "ss_to_he_" + op;
  conversion_in.input_shape = shape(rows, input_dim);
  conversion_in.output_shape = shape(metadata_.physical_tokens, input_dim);
  conversion_in.input_packing = "z2ell_token_major_shares";
  conversion_in.output_packing = "bert_compact_feature_major_slots";
  conversion_in.input_scale = conversion_in.output_scale = input.s;
  conversion_in.ell = input.ell;
  conversion_in.counts = conversion_stats.public_merge;
  conversion_in.backend = "bolt_conversion_signed_compact_slots";
  append_trace(trace, std::move(conversion_in), conversion_in_before_bytes,
               conversion_in_before_rounds, elapsed_us(conversion_in_start));

  const std::uint64_t he_before_bytes = io_.get_comm();
  const std::uint64_t he_before_rounds = io_.get_rounds();
  const auto he_start = std::chrono::steady_clock::now();
  BoltOperationCounts counts;
  std::vector<seal::Ciphertext> output;
  if (party_ == ALICE) {
    if (weight.size() != output_dim * input_dim ||
        (!bias.empty() && bias.size() != output_dim))
      throw std::invalid_argument("ALICE compact ViT weight mismatch: " + op);
    PlainWeightTensor plain;
    plain.input_cols = input_dim;
    plain.output_cols = output_dim;
    plain.values.resize(weight.size());
    plain.bias.resize(output_dim, 0);
    for (std::size_t i = 0; i < weight.size(); ++i)
      plain.values[i] = prime_encode(
          quantize(weight[i] * weight_multiplier, weight_scale),
          he_.parameters().plain_modulus);
    for (std::size_t i = 0; i < bias.size(); ++i)
      plain.bias[i] = prime_encode(
          quantize(bias[i] * weight_multiplier, raw_scale),
          he_.parameters().plain_modulus);
    const BoltVitCompatLinearPlan plan = compat_linear_->prepare(plain);
    output = compat_linear_->apply(encrypted, plan, &counts);
  }
  VitOperatorTrace he_row;
  he_row.phase = "online";
  he_row.layer = layer;
  he_row.op = op;
  he_row.input_shape = shape(rows, input_dim);
  he_row.output_shape = shape(rows, output_dim);
  he_row.input_packing = "bert_compact_feature_major_slots";
  he_row.output_packing = "bert_compact_split_feature_halves";
  he_row.input_scale = input.s;
  he_row.output_scale = raw_scale;
  he_row.ell = kEll;
  he_row.counts = counts;
  he_row.backend = "bolt_bert_compat_ct_pt_bsgs";
  append_trace(trace, std::move(he_row), he_before_bytes, he_before_rounds,
               elapsed_us(he_start));

  const std::size_t output_ciphertexts =
      metadata_.physical_tokens * output_dim / he_.slot_count();
  const std::uint64_t conversion_out_before_bytes = io_.get_comm();
  const std::uint64_t conversion_out_before_rounds = io_.get_rounds();
  const auto conversion_out_start = std::chrono::steady_clock::now();
  FixArray packed_ring = compat_packed_cipher_to_ring(
      output, output_ciphertexts, raw_scale);
  append_conversion_trace(
      trace, "he_to_ss_" + op, layer, metadata_.physical_tokens,
      output_dim, raw_scale, "bert_compact_split_feature_halves",
      "z2ell_packed_slot_shares", "bolt_conversion_he_to_ss_packed",
      conversion_out_before_bytes, conversion_out_before_rounds,
      elapsed_us(conversion_out_start));
  std::vector<std::uint64_t> packed_ring_vector(
      packed_ring.data, packed_ring.data + packed_ring.size);
  std::vector<std::uint64_t> unpacked =
      compat_packing_->unpack_linear_output(packed_ring_vector, output_dim);
  FixArray physical(party_, static_cast<int>(unpacked.size()), true, kEll,
                    raw_scale);
  std::memcpy(physical.data, unpacked.data(),
              unpacked.size() * sizeof(std::uint64_t));
  const std::uint64_t truncation_before_bytes = io_.get_comm();
  const std::uint64_t truncation_before_rounds = io_.get_rounds();
  const auto truncation_start = std::chrono::steady_clock::now();
  if (raw_scale > output_scale)
    physical = fp_.fix->right_shift(physical, raw_scale - output_scale);
  if (physical.ell < kEll) physical = fp_.fix->extend(physical, kEll);
  FixArray result;
  if (rows == metadata_.physical_tokens) {
    result = std::move(physical);
  } else {
    result = FixArray(party_, static_cast<int>(rows * output_dim), true,
                      physical.ell, physical.s);
    std::memcpy(result.data, physical.data,
                rows * output_dim * sizeof(std::uint64_t));
  }
  VitOperatorTrace truncation_row;
  truncation_row.phase = "online";
  truncation_row.layer = layer;
  truncation_row.op = op + "_share_truncation";
  truncation_row.input_shape =
      shape(metadata_.physical_tokens, output_dim);
  truncation_row.output_shape = shape(rows, output_dim);
  truncation_row.input_packing = "z2ell_unpacked_physical_token_shares_raw";
  truncation_row.output_packing =
      rows == metadata_.physical_tokens
          ? "z2ell_token_major_shares"
          : "z2ell_valid_token_major_shares_after_public_crop";
  truncation_row.input_scale = raw_scale;
  truncation_row.output_scale = output_scale;
  truncation_row.ell = result.ell;
  truncation_row.backend = "bolt_mpc_fix_right_shift";
  append_trace(trace, std::move(truncation_row), truncation_before_bytes,
               truncation_before_rounds, elapsed_us(truncation_start));
  return result;
}

FixArray BoltVitProtocol::compat_layer_norm_affine(
    const FixArray &normalized, const std::vector<double> &gamma,
    const std::vector<double> &beta, const std::string &name, int layer,
    std::vector<VitOperatorTrace> &trace) {
  if (!compact_compat_enabled() || compat_packing_ == nullptr ||
      compat_linear_ == nullptr ||
      normalized.size != static_cast<int>(metadata_.physical_tokens *
                                          metadata_.model_dim))
    throw std::invalid_argument("ViT compact LayerNorm affine mismatch");
  constexpr int affine_weight_scale = 11;
  const int raw_scale = normalized.s + affine_weight_scale;
  std::vector<std::uint64_t> token_major(
      normalized.data, normalized.data + normalized.size);
  std::vector<std::uint64_t> packed =
      compat_packing_->pack_token_major(token_major, metadata_.model_dim);

  const std::uint64_t conversion_in_before_bytes = io_.get_comm();
  const std::uint64_t conversion_in_before_rounds = io_.get_rounds();
  const auto conversion_in_start = std::chrono::steady_clock::now();
  BoltVitCompatConversionStats conversion_stats;
  std::vector<seal::Ciphertext> encrypted = compat_slots_to_cipher(
      packed, normalized.ell, normalized.s, &conversion_stats);
  VitOperatorTrace conversion_in;
  conversion_in.phase = "online";
  conversion_in.layer = layer;
  conversion_in.op = "ss_to_he_" + name + "_affine";
  conversion_in.input_shape = conversion_in.output_shape =
      shape(metadata_.physical_tokens, metadata_.model_dim);
  conversion_in.input_packing = "z2ell_token_major_shares";
  conversion_in.output_packing = "bert_compact_feature_major_slots";
  conversion_in.input_scale = conversion_in.output_scale = normalized.s;
  conversion_in.ell = normalized.ell;
  conversion_in.counts = conversion_stats.public_merge;
  conversion_in.backend = "bolt_conversion_signed_compact_slots";
  append_trace(trace, std::move(conversion_in), conversion_in_before_bytes,
               conversion_in_before_rounds, elapsed_us(conversion_in_start));

  const std::uint64_t he_before_bytes = io_.get_comm();
  const std::uint64_t he_before_rounds = io_.get_rounds();
  const auto he_start = std::chrono::steady_clock::now();
  BoltOperationCounts counts;
  std::vector<seal::Ciphertext> affine;
  if (party_ == ALICE) {
    if (gamma.size() != metadata_.model_dim ||
        beta.size() != metadata_.model_dim)
      throw std::invalid_argument("ALICE compact LayerNorm affine mismatch");
    std::vector<std::uint64_t> gamma_plain(metadata_.model_dim);
    std::vector<std::uint64_t> beta_plain(metadata_.model_dim);
    for (std::size_t feature = 0; feature < metadata_.model_dim; ++feature) {
      gamma_plain[feature] = prime_encode(
          quantize(gamma[feature], affine_weight_scale),
          he_.parameters().plain_modulus);
      beta_plain[feature] = prime_encode(
          quantize(beta[feature], raw_scale),
          he_.parameters().plain_modulus);
    }
    affine = compat_linear_->elementwise_affine(
        encrypted, gamma_plain, beta_plain, 71, 2, &counts);
  }
  VitOperatorTrace affine_row;
  affine_row.phase = "online";
  affine_row.layer = layer;
  affine_row.op = name + "_affine";
  affine_row.input_shape = affine_row.output_shape =
      shape(metadata_.physical_tokens, metadata_.model_dim);
  affine_row.input_packing = affine_row.output_packing =
      "bert_compact_feature_major_slots";
  affine_row.input_scale = normalized.s;
  affine_row.output_scale = raw_scale;
  affine_row.ell = kEll;
  affine_row.counts = counts;
  affine_row.backend = "bolt_bert_compat_ct_pt_elementwise_affine";
  append_trace(trace, std::move(affine_row), he_before_bytes,
               he_before_rounds, elapsed_us(he_start));

  const std::size_t ciphertext_count =
      metadata_.physical_tokens * metadata_.model_dim / he_.slot_count();
  const std::uint64_t conversion_out_before_bytes = io_.get_comm();
  const std::uint64_t conversion_out_before_rounds = io_.get_rounds();
  const auto conversion_out_start = std::chrono::steady_clock::now();
  FixArray packed_ring = compat_packed_cipher_to_ring(
      affine, ciphertext_count, raw_scale);
  append_conversion_trace(
      trace, "he_to_ss_" + name + "_affine", layer,
      metadata_.physical_tokens, metadata_.model_dim, raw_scale,
      "bert_compact_feature_major_slots", "z2ell_packed_slot_shares",
      "bolt_conversion_he_to_ss_packed", conversion_out_before_bytes,
      conversion_out_before_rounds, elapsed_us(conversion_out_start));
  std::vector<std::uint64_t> packed_vector(
      packed_ring.data, packed_ring.data + packed_ring.size);
  std::vector<std::uint64_t> unpacked = compat_packing_->unpack_token_major(
      packed_vector, metadata_.model_dim);
  FixArray result(party_, static_cast<int>(unpacked.size()), true, kEll,
                  raw_scale);
  std::memcpy(result.data, unpacked.data(),
              unpacked.size() * sizeof(std::uint64_t));
  const std::uint64_t truncation_before_bytes = io_.get_comm();
  const std::uint64_t truncation_before_rounds = io_.get_rounds();
  const auto truncation_start = std::chrono::steady_clock::now();
  result = fp_.fix->right_shift(result, affine_weight_scale);
  if (result.ell < kEll) result = fp_.fix->extend(result, kEll);
  VitOperatorTrace truncation_row;
  truncation_row.phase = "online";
  truncation_row.layer = layer;
  truncation_row.op = name + "_affine_share_truncation";
  truncation_row.input_shape = truncation_row.output_shape =
      shape(metadata_.physical_tokens, metadata_.model_dim);
  truncation_row.input_packing = truncation_row.output_packing =
      "z2ell_token_major_shares";
  truncation_row.input_scale = raw_scale;
  truncation_row.output_scale = normalized.s;
  truncation_row.ell = result.ell;
  truncation_row.backend = "bolt_mpc_fix_right_shift";
  append_trace(trace, std::move(truncation_row), truncation_before_bytes,
               truncation_before_rounds, elapsed_us(truncation_start));
  apply_public_token_mask(result, name + "_output_padding_mask", layer,
                          trace);
  return result;
}

void BoltVitProtocol::add_local_bias(FixArray &value,
                                     const std::vector<double> &bias,
                                     std::size_t rows, std::size_t cols,
                                     int scale) {
  if (party_ != ALICE) return;
  if (bias.size() != cols || value.size != static_cast<int>(rows * cols))
    throw std::invalid_argument("ViT local bias metadata mismatch");
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      value.data[row * cols + col] =
          (value.data[row * cols + col] +
           ring_encode(quantize(bias[col], scale), value.ell)) &
          value.ell_mask();
}

FixArray BoltVitProtocol::packed_linear(
    const FixArray &input, std::size_t rows, std::size_t input_dim,
    std::size_t output_dim, std::size_t stride,
    const std::vector<double> &weight, const std::vector<double> &bias,
    int output_scale, const std::string &op, int layer,
    std::vector<VitOperatorTrace> &trace, const std::string &backend,
    BoltCipherTensor *retained_packed_output, int weight_scale,
    double weight_multiplier) {
  if (compact_compat_enabled() && retained_packed_output == nullptr &&
      rows <= metadata_.physical_tokens && input_dim % 32 == 0 &&
      output_dim % 32 == 0) {
    return compat_packed_linear(
        input, rows, input_dim, output_dim, weight, bias, output_scale, op,
        layer, trace, weight_scale, weight_multiplier);
  }
  if (input.size != static_cast<int>(rows * input_dim) ||
      output_scale < 0 || weight_scale < 0 || weight_scale > kScale ||
      !std::isfinite(weight_multiplier) || weight_multiplier <= 0.0 ||
      input.s + weight_scale < output_scale)
    throw std::invalid_argument("ViT packed linear input metadata mismatch");
  const int raw_scale = input.s + weight_scale;
  const std::uint64_t conversion_in_before_bytes = io_.get_comm();
  const std::uint64_t conversion_in_before_rounds = io_.get_rounds();
  const auto conversion_in_start = std::chrono::steady_clock::now();
  BoltCipherTensor encrypted = shares_to_cipher(input, rows, input_dim, stride);
  append_conversion_trace(
      trace, "ss_to_he_" + op, layer, rows, input_dim, input.s,
      "z2ell_shares", "packed_ciphertext_rows",
      "bolt_conversion_ss_to_he", conversion_in_before_bytes,
      conversion_in_before_rounds, elapsed_us(conversion_in_start));

  const std::uint64_t he_before_bytes = io_.get_comm();
  const std::uint64_t he_before_rounds = io_.get_rounds();
  const auto he_start = std::chrono::steady_clock::now();
  BoltCipherTensor output;
  BoltOperationCounts counts;
  if (party_ == ALICE) {
    if (weight.size() != output_dim * input_dim ||
        (!bias.empty() && bias.size() != output_dim))
      throw std::invalid_argument("ALICE ViT weight metadata mismatch: " + op);
    PlainWeightTensor plain;
    plain.input_cols = input_dim;
    plain.output_cols = output_dim;
    plain.values.resize(weight.size());
    for (std::size_t i = 0; i < weight.size(); ++i)
      plain.values[i] = prime_encode(
          quantize(weight[i] * weight_multiplier, weight_scale),
          he_.parameters().plain_modulus);
    if (!bias.empty()) {
      plain.bias.resize(output_dim);
      for (std::size_t i = 0; i < output_dim; ++i)
        plain.bias[i] = prime_encode(
                                    quantize(bias[i] * weight_multiplier,
                                             raw_scale),
                                    he_.parameters().plain_modulus);
    }
    output = linear_.cipher_plain_matmul_bsgs(
        encrypted, plain,
        matmul_metadata(rows, input_dim, output_dim, stride, true), &counts);
    output.metadata.scale = raw_scale;
  } else {
    output.metadata = tensor_metadata(rows, output_dim, stride, kEll, raw_scale);
  }
  // End the HE-only accounting before the first HE-to-share conversion.  The
  // row uses raw_scale because the following conversion/truncation is a
  // separate adapter operation in the trace.
  VitOperatorTrace row;
  row.phase = "online";
  row.layer = layer;
  row.op = op;
  row.input_shape = shape(rows, input_dim);
  row.output_shape = shape(rows, output_dim);
  row.input_packing = "z2ell_shares";
  row.output_packing = "compact_bsgs_ciphertext_raw";
  row.input_scale = input.s;
  row.output_scale = raw_scale;
  row.ell = kEll;
  row.counts = counts;
  row.backend = backend;
  append_trace(trace, std::move(row), he_before_bytes, he_before_rounds,
               elapsed_us(he_start));

  const std::uint64_t conversion_out_before_bytes = io_.get_comm();
  const std::uint64_t conversion_out_before_rounds = io_.get_rounds();
  const auto conversion_out_start = std::chrono::steady_clock::now();
  FixArray ring = cipher_to_ring(output, rows, output_dim, stride, raw_scale);
  append_conversion_trace(
      trace, "he_to_ss_" + op, layer, rows, output_dim, raw_scale,
      "packed_ciphertext_rows", "z2ell_shares",
      "bolt_conversion_he_to_ss", conversion_out_before_bytes,
      conversion_out_before_rounds, elapsed_us(conversion_out_start));
  const std::uint64_t truncation_before_bytes = io_.get_comm();
  const std::uint64_t truncation_before_rounds = io_.get_rounds();
  const auto truncation_start = std::chrono::steady_clock::now();
  if (raw_scale > output_scale)
    ring = fp_.fix->right_shift(ring, raw_scale - output_scale);
  if (ring.ell < kEll) ring = fp_.fix->extend(ring, kEll);
  VitOperatorTrace truncation_row;
  truncation_row.phase = "online";
  truncation_row.layer = layer;
  truncation_row.op = op + "_share_truncation";
  truncation_row.input_shape = truncation_row.output_shape =
      shape(rows, output_dim);
  truncation_row.input_packing = truncation_row.output_packing =
      "z2ell_token_major_shares";
  truncation_row.input_scale = raw_scale;
  truncation_row.output_scale = output_scale;
  truncation_row.ell = ring.ell;
  truncation_row.backend = "bolt_mpc_fix_right_shift";
  append_trace(trace, std::move(truncation_row), truncation_before_bytes,
               truncation_before_rounds, elapsed_us(truncation_start));
  // The BOLT truncation is necessarily performed in the share domain.  Q/K/V
  // can request an immediate packed SS-to-HE conversion so the truncated
  // value is retained as a row-packed ciphertext for ct-ct attention.  This
  // never creates a ciphertext per scalar element.
  if (retained_packed_output != nullptr) {
    const std::uint64_t retained_before_bytes = io_.get_comm();
    const std::uint64_t retained_before_rounds = io_.get_rounds();
    const auto retained_start = std::chrono::steady_clock::now();
    *retained_packed_output = shares_to_cipher(
        ring, rows, output_dim, stride);
    retained_packed_output->metadata.scale = ring.s;
    append_conversion_trace(
        trace, "ss_to_he_retained_" + op, layer, rows, output_dim, ring.s,
        "z2ell_shares", "packed_ciphertext_rows",
        "bolt_conversion_ss_to_he_retained", retained_before_bytes,
        retained_before_rounds, elapsed_us(retained_start));
  }
  return ring;
}

FixArray BoltVitProtocol::layer_norm(
    const FixArray &input, const std::vector<double> &gamma,
    const std::vector<double> &beta, const std::string &name, int layer,
    std::vector<VitOperatorTrace> &trace) {
  FixArray masked_input = input;
  apply_public_token_mask(masked_input, name + "_input_padding_mask", layer,
                          trace);
  const std::uint64_t before_bytes = io_.get_comm();
  const std::uint64_t before_rounds = io_.get_rounds();
  const auto start = std::chrono::steady_clock::now();
  if (party_ == ALICE &&
      (gamma.size() != metadata_.model_dim ||
       beta.size() != metadata_.model_dim))
    throw std::invalid_argument("ALICE LayerNorm parameter mismatch");

  // Reuse BLB's existing MPC LayerNorm statistics/normalization unchanged,
  // but give its built-in affine a public identity.  Supplying private zero
  // gamma to that routine goes through probabilistic secret truncation and
  // is not an exact zero map.  The checkpoint affine is therefore applied
  // below with the existing encrypted-activation/plain-model operation.  No
  // activation or model parameter is opened.
  const std::size_t logical_elements =
      metadata_.tokens * metadata_.model_dim;
  std::vector<std::uint64_t> unit_fixed(
      logical_elements, 1ULL << metadata_.activation_scale);
  std::vector<std::uint64_t> zero_fixed(logical_elements, 0);
  FixArray unit_share = fp_.fix->input(
      PUBLIC, logical_elements, unit_fixed.data(), true, kEll,
      metadata_.activation_scale);
  FixArray zero_share = fp_.fix->input(
      PUBLIC, logical_elements, zero_fixed.data(), true, kEll,
      metadata_.activation_scale);
  std::vector<FixArray> valid_rows = split_rows(
      masked_input, metadata_.physical_tokens, metadata_.model_dim);
  valid_rows.resize(metadata_.tokens);
  FixArray valid_normalized = flatten(
      fp_.layer_norm_fix(valid_rows, unit_share, zero_share));
  FixArray normalized(party_, input.size, true, input.ell,
                      valid_normalized.s);
  std::fill(normalized.data, normalized.data + normalized.size, 0);
  std::memcpy(normalized.data, valid_normalized.data,
              logical_elements * sizeof(std::uint64_t));
  VitOperatorTrace statistics;
  statistics.phase = "online";
  statistics.layer = layer;
  statistics.op = name + "_statistics";
  statistics.input_shape = statistics.output_shape =
      shape(metadata_.physical_tokens, metadata_.model_dim);
  statistics.input_packing = statistics.output_packing = "z2ell_shares";
  statistics.input_scale = input.s;
  statistics.output_scale = normalized.s;
  statistics.ell = normalized.ell;
  statistics.backend = "bolt_mpc_layernorm";
  append_trace(trace, std::move(statistics), before_bytes, before_rounds,
               elapsed_us(start));

  if (compact_compat_enabled())
    return compat_layer_norm_affine(normalized, gamma, beta, name, layer,
                                    trace);

  constexpr int affine_weight_scale = 11;
  const int raw_scale = normalized.s + affine_weight_scale;
  const std::uint64_t conversion_in_before_bytes = io_.get_comm();
  const std::uint64_t conversion_in_before_rounds = io_.get_rounds();
  const auto conversion_in_start = std::chrono::steady_clock::now();
  BoltCipherTensor encrypted = shares_to_cipher(
      normalized, metadata_.physical_tokens, metadata_.model_dim,
      metadata_.attention_stride);
  append_conversion_trace(
      trace, "ss_to_he_" + name + "_affine", layer,
      metadata_.physical_tokens, metadata_.model_dim, normalized.s,
      "z2ell_shares", "packed_ciphertext_rows",
      "bolt_conversion_ss_to_he", conversion_in_before_bytes,
      conversion_in_before_rounds, elapsed_us(conversion_in_start));

  const std::uint64_t he_before_bytes = io_.get_comm();
  const std::uint64_t he_before_rounds = io_.get_rounds();
  const auto he_start = std::chrono::steady_clock::now();
  BoltCipherTensor affine;
  BoltOperationCounts counts;
  if (party_ == ALICE) {
    std::vector<std::uint64_t> gamma_plain(metadata_.model_dim);
    std::vector<std::uint64_t> beta_plain(metadata_.model_dim);
    for (std::size_t col = 0; col < metadata_.model_dim; ++col) {
      gamma_plain[col] = prime_encode(
          quantize(gamma[col], affine_weight_scale),
          he_.parameters().plain_modulus);
      beta_plain[col] = prime_encode(
          quantize(beta[col], raw_scale), he_.parameters().plain_modulus);
    }
    affine = linear_.cipher_plain_elementwise_affine(
        encrypted, gamma_plain, beta_plain,
        matmul_metadata(metadata_.physical_tokens, metadata_.model_dim,
                        metadata_.model_dim, metadata_.attention_stride, true),
        &counts);
    affine.metadata.scale = raw_scale;
  } else {
    affine.metadata = tensor_metadata(
        metadata_.physical_tokens, metadata_.model_dim,
        metadata_.attention_stride, kEll, raw_scale);
  }
  VitOperatorTrace affine_row;
  affine_row.phase = "online";
  affine_row.layer = layer;
  affine_row.op = name + "_affine";
  affine_row.input_shape = affine_row.output_shape =
      shape(metadata_.physical_tokens, metadata_.model_dim);
  affine_row.input_packing = "z2ell_shares";
  affine_row.output_packing = "compact_elementwise_ciphertext_raw";
  affine_row.input_scale = normalized.s;
  affine_row.output_scale = raw_scale;
  affine_row.ell = kEll;
  affine_row.counts = counts;
  affine_row.backend = "bolt_ct_pt_elementwise_affine";
  append_trace(trace, std::move(affine_row), he_before_bytes,
               he_before_rounds, elapsed_us(he_start));

  const std::uint64_t conversion_out_before_bytes = io_.get_comm();
  const std::uint64_t conversion_out_before_rounds = io_.get_rounds();
  const auto conversion_out_start = std::chrono::steady_clock::now();
  FixArray result = cipher_to_ring(
      affine, metadata_.physical_tokens, metadata_.model_dim,
      metadata_.attention_stride, raw_scale);
  append_conversion_trace(
      trace, "he_to_ss_" + name + "_affine", layer,
      metadata_.physical_tokens, metadata_.model_dim, raw_scale,
      "packed_ciphertext_rows", "z2ell_shares",
      "bolt_conversion_he_to_ss", conversion_out_before_bytes,
      conversion_out_before_rounds, elapsed_us(conversion_out_start));
  const std::uint64_t truncation_before_bytes = io_.get_comm();
  const std::uint64_t truncation_before_rounds = io_.get_rounds();
  const auto truncation_start = std::chrono::steady_clock::now();
  result = fp_.fix->right_shift(result, affine_weight_scale);
  if (result.ell < kEll) result = fp_.fix->extend(result, kEll);
  VitOperatorTrace truncation_row;
  truncation_row.phase = "online";
  truncation_row.layer = layer;
  truncation_row.op = name + "_affine_share_truncation";
  truncation_row.input_shape = truncation_row.output_shape =
      shape(metadata_.physical_tokens, metadata_.model_dim);
  truncation_row.input_packing = truncation_row.output_packing =
      "z2ell_token_major_shares";
  truncation_row.input_scale = raw_scale;
  truncation_row.output_scale = normalized.s;
  truncation_row.ell = result.ell;
  truncation_row.backend = "bolt_mpc_fix_right_shift";
  append_trace(trace, std::move(truncation_row), truncation_before_bytes,
               truncation_before_rounds, elapsed_us(truncation_start));
  apply_public_token_mask(result, name + "_output_padding_mask", layer,
                          trace);
  return result;
}

FixArray BoltVitProtocol::residual_add(
    const FixArray &lhs, const FixArray &rhs, const std::string &name,
    int layer, std::vector<VitOperatorTrace> &trace) {
  if (lhs.size != rhs.size || lhs.ell != rhs.ell || lhs.s != rhs.s)
    throw std::invalid_argument("ViT residual metadata mismatch");
  const std::uint64_t before_bytes = io_.get_comm();
  const std::uint64_t before_rounds = io_.get_rounds();
  const auto start = std::chrono::steady_clock::now();
  FixArray result(party_, lhs.size, true, lhs.ell, lhs.s);
  for (int i = 0; i < lhs.size; ++i)
    result.data[i] = (lhs.data[i] + rhs.data[i]) & lhs.ell_mask();
  VitOperatorTrace row;
  row.phase = "online";
  row.layer = layer;
  row.op = name;
  row.input_shape = "2x" + shape(metadata_.physical_tokens, metadata_.model_dim);
  row.output_shape = shape(metadata_.physical_tokens, metadata_.model_dim);
  row.input_packing = row.output_packing = "z2ell_additive_shares";
  row.input_scale = row.output_scale = lhs.s;
  row.ell = lhs.ell;
  row.backend = "local_share_add";
  append_trace(trace, std::move(row), before_bytes, before_rounds,
               elapsed_us(start));
  if (trace.back().bytes != 0 || trace.back().logical_rounds != 0)
    throw std::logic_error("BOLT fail-closed: residual add communicated");
  apply_public_token_mask(result, name + "_padding_mask", layer, trace);
  return result;
}

FixArray BoltVitProtocol::compat_attention(
    const FixArray &input, const VitBlockModel &block, int layer,
    std::vector<VitOperatorTrace> &trace
#ifdef BLB_VIT_VALIDATION_OUTPUTS
    , VitValidationObserver *validation_observer
#endif
    ) {
  if (!compact_compat_enabled() || compat_packing_ == nullptr ||
      compat_attention_ == nullptr)
    throw std::logic_error("ViT compact attention is unavailable");
  const std::size_t physical = metadata_.physical_tokens;
  const std::size_t logical = metadata_.tokens;
  FixArray q = compat_packed_linear(
      input, physical, metadata_.model_dim, metadata_.model_dim, block.wq,
      block.q_bias, kQKOperandScale, "q_projection", layer, trace, 10,
      1.0 / std::sqrt(static_cast<double>(metadata_.head_dim)));
  FixArray k = compat_packed_linear(
      input, physical, metadata_.model_dim, metadata_.model_dim, block.wk,
      block.k_bias, kQKOperandScale, "k_projection", layer, trace, 10, 1.0);
  FixArray v = compat_packed_linear(
      input, physical, metadata_.model_dim, metadata_.model_dim, block.wv,
      block.v_bias, metadata_.activation_scale, "v_projection", layer,
      trace, 11, 1.0);
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  if (validation_observer != nullptr) {
    const std::string prefix = "block_" + std::to_string(layer) + ".";
    validation_observer->observe(prefix + "q", q, physical,
                                 metadata_.model_dim, logical);
    validation_observer->observe(prefix + "k", k, physical,
                                 metadata_.model_dim, logical);
    validation_observer->observe(prefix + "v", v, physical,
                                 metadata_.model_dim, logical);
  }
#endif

  auto encrypt_head_paired = [&](const FixArray &value,
                                 const std::string &op) {
    const std::uint64_t before_bytes = io_.get_comm();
    const std::uint64_t before_rounds = io_.get_rounds();
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::uint64_t> token_major(value.data,
                                           value.data + value.size);
    std::vector<std::uint64_t> packed =
        compat_packing_->pack_context(token_major);
    BoltVitCompatConversionStats stats;
    std::vector<seal::Ciphertext> ciphertexts = compat_slots_to_cipher(
        packed, value.ell, value.s, &stats);
    VitOperatorTrace row;
    row.phase = "online";
    row.layer = layer;
    row.op = op;
    row.input_shape = row.output_shape =
        shape(physical, metadata_.model_dim);
    row.input_packing = "z2ell_token_major_shares";
    row.output_packing = "bert_paired_head_compact_slots";
    row.input_scale = row.output_scale = value.s;
    row.ell = value.ell;
    row.counts = stats.public_merge;
    row.backend = "bolt_conversion_signed_compact_slots";
    append_trace(trace, std::move(row), before_bytes, before_rounds,
                 elapsed_us(start));
    return ciphertexts;
  };

  std::vector<seal::Ciphertext> q_cipher =
      encrypt_head_paired(q, "ss_to_he_retained_q_projection");
  std::vector<seal::Ciphertext> k_cipher =
      encrypt_head_paired(k, "ss_to_he_retained_k_projection");
  std::vector<seal::Ciphertext> v_cipher =
      encrypt_head_paired(v, "ss_to_he_retained_v_projection");

  const std::uint64_t qk_before_bytes = io_.get_comm();
  const std::uint64_t qk_before_rounds = io_.get_rounds();
  const auto qk_start = std::chrono::steady_clock::now();
  BoltOperationCounts qk_counts;
  std::vector<seal::Ciphertext> qk_cipher;
  if (party_ == ALICE) {
    std::vector<seal::Ciphertext> fused;
    fused.reserve(compat_packing_->fused_qkv_ciphertexts());
    fused.insert(fused.end(), q_cipher.begin(), q_cipher.end());
    fused.insert(fused.end(), k_cipher.begin(), k_cipher.end());
    fused.insert(fused.end(), v_cipher.begin(), v_cipher.end());
    qk_cipher = compat_attention_->qk_cross(fused, &qk_counts);
  }
  VitOperatorTrace qk_row;
  qk_row.phase = "online";
  qk_row.layer = layer;
  qk_row.op = "qk";
  qk_row.input_shape = std::to_string(metadata_.heads) + "x(" +
      shape(physical, metadata_.head_dim) + "x" +
      shape(metadata_.head_dim, physical) + ")";
  qk_row.output_shape = std::to_string(metadata_.heads) + "x" +
      shape(physical, physical);
  qk_row.input_packing = "bert_paired_head_compact_slots";
  qk_row.output_packing = "bert_qk_cross_diagonal_slots";
  qk_row.input_scale = kQKOperandScale;
  qk_row.output_scale = 2 * kQKOperandScale;
  qk_row.ell = kEll;
  qk_row.counts = qk_counts;
  qk_row.backend = "bolt_bert_compat_ct_ct_qk";
  append_trace(trace, std::move(qk_row), qk_before_bytes, qk_before_rounds,
               elapsed_us(qk_start));

  const std::uint64_t qk_conversion_before_bytes = io_.get_comm();
  const std::uint64_t qk_conversion_before_rounds = io_.get_rounds();
  const auto qk_conversion_start = std::chrono::steady_clock::now();
  FixArray qk_packed_ring = compat_packed_cipher_to_ring(
      qk_cipher, compat_packing_->qk_ciphertexts(),
      2 * kQKOperandScale);
  append_conversion_trace(
      trace, "he_to_ss_qk", layer, metadata_.heads * physical, physical,
      2 * kQKOperandScale, "bert_qk_cross_diagonal_slots",
      "z2ell_packed_slot_shares", "bolt_conversion_he_to_ss_packed",
      qk_conversion_before_bytes, qk_conversion_before_rounds,
      elapsed_us(qk_conversion_start));
  std::vector<std::uint64_t> qk_packed(
      qk_packed_ring.data, qk_packed_ring.data + qk_packed_ring.size);
  std::vector<std::uint64_t> qk_token_major =
      compat_packing_->unpack_qk_cross(qk_packed);
  std::vector<FixArray> qk_ring(metadata_.heads);
  for (std::size_t head = 0; head < metadata_.heads; ++head) {
    qk_ring[head] = FixArray(party_, physical * physical, true, kEll,
                            2 * kQKOperandScale);
    std::memcpy(qk_ring[head].data,
                qk_token_major.data() + head * physical * physical,
                physical * physical * sizeof(std::uint64_t));
    apply_public_attention_mask(qk_ring[head], "attention_padding_mask",
                                layer, trace);
  }

  const std::uint64_t qk_truncation_before_bytes = io_.get_comm();
  const std::uint64_t qk_truncation_before_rounds = io_.get_rounds();
  const auto qk_truncation_start = std::chrono::steady_clock::now();
  std::vector<FixArray> scaled_rows;
  scaled_rows.reserve(metadata_.heads * logical);
  for (std::size_t head = 0; head < metadata_.heads; ++head) {
    FixArray scaled = fp_.fix->right_shift(
        qk_ring[head],
        2 * kQKOperandScale - metadata_.activation_scale);
    if (scaled.ell < kEll) scaled = fp_.fix->extend(scaled, kEll);
    auto rows = split_rows(scaled, physical, physical);
    for (std::size_t row = 0; row < logical; ++row) {
      FixArray valid_row(party_, static_cast<int>(logical), true, kEll,
                         metadata_.activation_scale);
      std::memcpy(valid_row.data, rows[row].data,
                  logical * sizeof(std::uint64_t));
      scaled_rows.push_back(std::move(valid_row));
    }
  }
  VitOperatorTrace qk_truncation_row;
  qk_truncation_row.phase = "online";
  qk_truncation_row.layer = layer;
  qk_truncation_row.op = "qk_share_truncation_and_valid_slice";
  qk_truncation_row.input_shape =
      std::to_string(metadata_.heads) + "x" + shape(physical, physical);
  qk_truncation_row.output_shape =
      std::to_string(metadata_.heads) + "x" + shape(logical, logical);
  qk_truncation_row.input_packing =
      "z2ell_physical_qk_shares_public_padding_masked";
  qk_truncation_row.output_packing =
      "z2ell_valid_qk_row_shares";
  qk_truncation_row.input_scale = 2 * kQKOperandScale;
  qk_truncation_row.output_scale = metadata_.activation_scale;
  qk_truncation_row.ell = kEll;
  qk_truncation_row.backend = "bolt_mpc_fix_right_shift_public_valid_slice";
  append_trace(trace, std::move(qk_truncation_row),
               qk_truncation_before_bytes, qk_truncation_before_rounds,
               elapsed_us(qk_truncation_start));
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  if (validation_observer != nullptr) {
    FixArray observed_qk = flatten(scaled_rows);
    validation_observer->observe(
        "block_" + std::to_string(layer) + ".qk", observed_qk,
        metadata_.heads * logical, logical, metadata_.heads * logical);
  }
#endif

  const std::uint64_t soft_before_bytes = io_.get_comm();
  const std::uint64_t soft_before_rounds = io_.get_rounds();
  const auto soft_start = std::chrono::steady_clock::now();
  // Reuse BLB's existing ell=37 lookup/division Softmax.  The BERT exp4
  // variant reduces exponentials to ell=19 before summation; 197 entries can
  // exceed that ring (unlike BERT's <=128 tokens), so it is not a valid
  // runtime-shape generalization for ViT.
  std::vector<FixArray> softmax_rows =
      fp_.softmax_fix_iron_1(scaled_rows);
  VitOperatorTrace soft_row;
  soft_row.phase = "online";
  soft_row.layer = layer;
  soft_row.op = "softmax";
  soft_row.input_shape = soft_row.output_shape =
      std::to_string(metadata_.heads * logical) + "x[" +
      std::to_string(logical) + "]";
  soft_row.input_packing =
      "public_mask_physical_256_to_logical_197";
  soft_row.output_packing = "z2ell_row_shares_valid_submatrix";
  soft_row.input_scale = soft_row.output_scale =
      metadata_.activation_scale;
  soft_row.ell = kEll;
  soft_row.backend = "bolt_mpc_softmax_fix_iron_1_ell37";
  append_trace(trace, std::move(soft_row), soft_before_bytes,
               soft_before_rounds, elapsed_us(soft_start));
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  if (validation_observer != nullptr) {
    FixArray observed_softmax = flatten(softmax_rows);
    validation_observer->observe(
        "block_" + std::to_string(layer) + ".softmax", observed_softmax,
        metadata_.heads * logical, logical, metadata_.heads * logical);
  }
#endif

  std::vector<std::uint64_t> logical_softmax(
      metadata_.heads * logical * logical);
  for (std::size_t row = 0; row < softmax_rows.size(); ++row)
    std::memcpy(logical_softmax.data() + row * logical,
                softmax_rows[row].data,
                logical * sizeof(std::uint64_t));
  std::vector<std::uint64_t> packed_softmax =
      compat_packing_->pack_logical_softmax_diagonal_source(
          logical_softmax, logical);
  const std::uint64_t soft_conversion_before_bytes = io_.get_comm();
  const std::uint64_t soft_conversion_before_rounds = io_.get_rounds();
  const auto soft_conversion_start = std::chrono::steady_clock::now();
  BoltVitCompatConversionStats soft_conversion_stats;
  std::vector<seal::Ciphertext> softmax_cipher = compat_slots_to_cipher(
      packed_softmax, kEll, metadata_.activation_scale,
      &soft_conversion_stats);
  VitOperatorTrace soft_conversion_row;
  soft_conversion_row.phase = "online";
  soft_conversion_row.layer = layer;
  soft_conversion_row.op = "ss_to_he_softmax";
  soft_conversion_row.input_shape =
      std::to_string(metadata_.heads) + "x" + shape(logical, logical);
  soft_conversion_row.output_shape =
      std::to_string(metadata_.heads) + "x" + shape(physical, physical);
  soft_conversion_row.input_packing = "z2ell_valid_softmax_rows";
  soft_conversion_row.output_packing =
      "bert_softmax_diagonal_source_padding_zero";
  soft_conversion_row.input_scale = soft_conversion_row.output_scale =
      metadata_.activation_scale;
  soft_conversion_row.ell = kEll;
  soft_conversion_row.counts = soft_conversion_stats.public_merge;
  soft_conversion_row.backend = "bolt_conversion_signed_compact_slots";
  append_trace(trace, std::move(soft_conversion_row),
               soft_conversion_before_bytes, soft_conversion_before_rounds,
               elapsed_us(soft_conversion_start));

  const std::uint64_t sv_before_bytes = io_.get_comm();
  const std::uint64_t sv_before_rounds = io_.get_rounds();
  const auto sv_start = std::chrono::steady_clock::now();
  BoltOperationCounts sv_counts;
  std::vector<seal::Ciphertext> context;
  if (party_ == ALICE) {
    auto diagonals = compat_attention_->expand_softmax_diagonals(
        softmax_cipher, &sv_counts);
    context = compat_attention_->softmax_v(diagonals, v_cipher, &sv_counts);
  }
  VitOperatorTrace sv_row;
  sv_row.phase = "online";
  sv_row.layer = layer;
  sv_row.op = "softmax_v";
  sv_row.input_shape = std::to_string(metadata_.heads) + "x(" +
      shape(physical, physical) + "x" +
      shape(physical, metadata_.head_dim) + ")";
  sv_row.output_shape = shape(physical, metadata_.model_dim);
  sv_row.input_packing = "bert_softmax_diagonals_plus_paired_v";
  sv_row.output_packing = "bert_paired_head_context_slots";
  sv_row.input_scale = metadata_.activation_scale;
  sv_row.output_scale = 2 * metadata_.activation_scale;
  sv_row.ell = kEll;
  sv_row.counts = sv_counts;
  sv_row.backend = "bolt_bert_compat_ct_ct_softmax_v";
  append_trace(trace, std::move(sv_row), sv_before_bytes, sv_before_rounds,
               elapsed_us(sv_start));

  const std::uint64_t context_conversion_before_bytes = io_.get_comm();
  const std::uint64_t context_conversion_before_rounds = io_.get_rounds();
  const auto context_conversion_start = std::chrono::steady_clock::now();
  FixArray context_packed_ring = compat_packed_cipher_to_ring(
      context, compat_packing_->context_ciphertexts(),
      2 * metadata_.activation_scale);
  append_conversion_trace(
      trace, "he_to_ss_softmax_v", layer, physical, metadata_.model_dim,
      2 * metadata_.activation_scale, "bert_paired_head_context_slots",
      "z2ell_packed_slot_shares", "bolt_conversion_he_to_ss_packed",
      context_conversion_before_bytes, context_conversion_before_rounds,
      elapsed_us(context_conversion_start));
  std::vector<std::uint64_t> context_packed(
      context_packed_ring.data,
      context_packed_ring.data + context_packed_ring.size);
  std::vector<std::uint64_t> context_token_major =
      compat_packing_->unpack_context(context_packed);
  FixArray merged(party_, static_cast<int>(context_token_major.size()), true,
                  kEll, 2 * metadata_.activation_scale);
  std::memcpy(merged.data, context_token_major.data(),
              context_token_major.size() * sizeof(std::uint64_t));
  const std::uint64_t truncation_before_bytes = io_.get_comm();
  const std::uint64_t truncation_before_rounds = io_.get_rounds();
  const auto truncation_start = std::chrono::steady_clock::now();
  merged = fp_.fix->right_shift(merged, metadata_.activation_scale);
  if (merged.ell < kEll) merged = fp_.fix->extend(merged, kEll);
  VitOperatorTrace truncation_row;
  truncation_row.phase = "online";
  truncation_row.layer = layer;
  truncation_row.op = "softmax_v_share_truncation";
  truncation_row.input_shape = truncation_row.output_shape =
      shape(physical, metadata_.model_dim);
  truncation_row.input_packing = truncation_row.output_packing =
      "z2ell_token_major_context_shares";
  truncation_row.input_scale = 2 * metadata_.activation_scale;
  truncation_row.output_scale = metadata_.activation_scale;
  truncation_row.ell = merged.ell;
  truncation_row.backend = "bolt_mpc_fix_right_shift";
  append_trace(trace, std::move(truncation_row), truncation_before_bytes,
               truncation_before_rounds, elapsed_us(truncation_start));
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  if (validation_observer != nullptr)
    validation_observer->observe(
        "block_" + std::to_string(layer) + ".attention_context", merged,
        physical, metadata_.model_dim, logical);
#endif
  FixArray projected = compat_packed_linear(
      merged, physical, metadata_.model_dim, metadata_.model_dim, block.wo,
      block.bo, metadata_.activation_scale, "output_projection", layer,
      trace, 10, 1.0);
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  if (validation_observer != nullptr)
    validation_observer->observe(
        "block_" + std::to_string(layer) + ".output_projection", projected,
        physical, metadata_.model_dim, logical);
#endif
  return projected;
}

FixArray BoltVitProtocol::attention(
    const FixArray &input, const VitBlockModel &block, int layer,
    std::vector<VitOperatorTrace> &trace
#ifdef BLB_VIT_VALIDATION_OUTPUTS
    , VitValidationObserver *validation_observer
#endif
    ) {
  if (compact_compat_enabled())
    return compat_attention(input, block, layer, trace
#ifdef BLB_VIT_VALIDATION_OUTPUTS
                            , validation_observer
#endif
                            );
  BoltCipherTensor q_cipher, k_cipher, v_cipher;
  FixArray q = packed_linear(input, metadata_.physical_tokens,
      metadata_.model_dim,
      metadata_.model_dim, metadata_.attention_stride, block.wq,
      block.q_bias, kQKOperandScale, "q_projection", layer, trace,
      "bolt_ct_pt_bsgs_truncate_ss_to_he", &q_cipher, 10,
      1.0 / std::sqrt(static_cast<double>(metadata_.head_dim)));
  FixArray k = packed_linear(input, metadata_.physical_tokens,
      metadata_.model_dim,
      metadata_.model_dim, metadata_.attention_stride, block.wk,
      block.k_bias, kQKOperandScale, "k_projection", layer, trace,
      "bolt_ct_pt_bsgs_truncate_ss_to_he", &k_cipher, 10);
  FixArray v = packed_linear(input, metadata_.physical_tokens,
      metadata_.model_dim,
      metadata_.model_dim, metadata_.attention_stride, block.wv,
      block.v_bias, metadata_.activation_scale, "v_projection", layer, trace,
      "bolt_ct_pt_bsgs_truncate_ss_to_he_retained_v", &v_cipher, 11);
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  if (validation_observer != nullptr) {
    const std::string prefix = "block_" + std::to_string(layer) + ".";
    validation_observer->observe(prefix + "q", q, metadata_.physical_tokens,
                                 metadata_.model_dim, metadata_.tokens);
    validation_observer->observe(prefix + "k", k, metadata_.physical_tokens,
                                 metadata_.model_dim, metadata_.tokens);
    validation_observer->observe(prefix + "v", v, metadata_.physical_tokens,
                                 metadata_.model_dim, metadata_.tokens);
  }
#endif

  const BoltAttentionPackingMetadata attention_meta{
      metadata_.physical_tokens, metadata_.model_dim, metadata_.heads,
      metadata_.head_dim, metadata_.attention_stride};
  const std::size_t physical = metadata_.physical_tokens;
  const std::size_t logical = metadata_.tokens;
  const std::uint64_t qk_before_bytes = io_.get_comm();
  const std::uint64_t qk_before_rounds = io_.get_rounds();
  const auto qk_start = std::chrono::steady_clock::now();
  BoltOperationCounts qk_counts;
  std::vector<BoltCipherTensor> v_heads(metadata_.heads);
  std::vector<BoltCipherTensor> qk_cipher(metadata_.heads);
  if (party_ == ALICE) {
    for (std::size_t head = 0; head < metadata_.heads; ++head) {
      BoltCipherTensor q_head = packing_.slice_head(
          q_cipher, head, attention_meta, &qk_counts);
      BoltCipherTensor k_head = packing_.slice_head(
          k_cipher, head, attention_meta, &qk_counts);
      v_heads[head] = packing_.slice_head(
          v_cipher, head, attention_meta, &qk_counts);
      BoltCipherTensor kt = packing_.transpose_tokens(
          k_head, attention_meta, &qk_counts);
      qk_cipher[head] = linear_.cipher_cipher_matmul_cross(
          q_head, kt,
          matmul_metadata(metadata_.physical_tokens, metadata_.head_dim,
                          metadata_.physical_tokens,
                          metadata_.attention_stride, true),
          &qk_counts);
      qk_cipher[head].metadata.scale = 2 * kQKOperandScale;
    }
  } else {
    for (std::size_t head = 0; head < metadata_.heads; ++head) {
      v_heads[head].metadata = tensor_metadata(
          metadata_.physical_tokens, metadata_.head_dim,
          metadata_.attention_stride,
          kEll, metadata_.activation_scale);
      qk_cipher[head].metadata = tensor_metadata(
          metadata_.physical_tokens, metadata_.physical_tokens,
          metadata_.attention_stride,
          kEll, 2 * kQKOperandScale);
    }
  }
  VitOperatorTrace qk_row;
  qk_row.phase = "online";
  qk_row.layer = layer;
  qk_row.op = "qk";
  qk_row.input_shape = std::to_string(metadata_.heads) + "x(" +
      shape(physical, metadata_.head_dim) + "x" +
      shape(metadata_.head_dim, physical) + ")";
  qk_row.output_shape = std::to_string(metadata_.heads) + "x" +
      shape(physical, physical);
  qk_row.input_packing = "compact_head_plus_cross_transpose";
  qk_row.output_packing = "cross_packed_rows";
  qk_row.input_scale = kQKOperandScale;
  qk_row.output_scale = 2 * kQKOperandScale;
  qk_row.counts = qk_counts;
  qk_row.backend = "bolt_ct_ct_cross";
  append_trace(trace, std::move(qk_row), qk_before_bytes, qk_before_rounds,
               elapsed_us(qk_start));

  std::vector<FixArray> qk_ring(metadata_.heads);
  for (std::size_t head = 0; head < metadata_.heads; ++head) {
    const std::uint64_t conversion_before_bytes = io_.get_comm();
    const std::uint64_t conversion_before_rounds = io_.get_rounds();
    const auto conversion_start = std::chrono::steady_clock::now();
    qk_ring[head] = cipher_to_ring(
        qk_cipher[head], physical, physical,
        metadata_.attention_stride, 2 * kQKOperandScale);
    append_conversion_trace(
        trace, "he_to_ss_qk_head_" + std::to_string(head), layer,
        physical, physical, 2 * kQKOperandScale,
        "cross_packed_ciphertext_rows", "z2ell_shares",
        "bolt_conversion_he_to_ss", conversion_before_bytes,
        conversion_before_rounds, elapsed_us(conversion_start));
    apply_public_attention_mask(qk_ring[head], "attention_padding_mask", layer,
                                trace);
  }

  const std::uint64_t qk_truncation_before_bytes = io_.get_comm();
  const std::uint64_t qk_truncation_before_rounds = io_.get_rounds();
  const auto qk_truncation_start = std::chrono::steady_clock::now();
  std::vector<FixArray> scaled_rows;
  scaled_rows.reserve(metadata_.heads * metadata_.tokens);
  for (std::size_t head = 0; head < metadata_.heads; ++head) {
    FixArray &qk = qk_ring[head];
    // 1/sqrt(head_dim) is folded into ALICE's private Q weight/bias above;
    // both Q and K are then retained at scale 10.  Their ct-ct product is at
    // scale 20 and is reduced to the existing scale-12 Softmax interface.
    FixArray scaled = fp_.fix->right_shift(
        qk, 2 * kQKOperandScale - metadata_.activation_scale);
    if (scaled.ell < kEll) scaled = fp_.fix->extend(scaled, kEll);
    auto rows = split_rows(scaled, physical, physical);
    // The fixed public mask is applied before entering the existing MPC
    // Softmax by taking only the valid 197x197 submatrix.  Padding therefore
    // cannot contribute an exp term or alter the denominator.
    for (std::size_t row = 0; row < logical; ++row) {
      FixArray valid_row(party_, static_cast<int>(logical), true, kEll,
                         metadata_.activation_scale);
      std::memcpy(valid_row.data, rows[row].data,
                  logical * sizeof(std::uint64_t));
      scaled_rows.push_back(std::move(valid_row));
    }
  }
  VitOperatorTrace qk_truncation_row;
  qk_truncation_row.phase = "online";
  qk_truncation_row.layer = layer;
  qk_truncation_row.op = "qk_share_truncation_and_valid_slice";
  qk_truncation_row.input_shape =
      std::to_string(metadata_.heads) + "x" + shape(physical, physical);
  qk_truncation_row.output_shape =
      std::to_string(metadata_.heads) + "x" + shape(logical, logical);
  qk_truncation_row.input_packing =
      "z2ell_physical_qk_shares_public_padding_masked";
  qk_truncation_row.output_packing = "z2ell_valid_qk_row_shares";
  qk_truncation_row.input_scale = 2 * kQKOperandScale;
  qk_truncation_row.output_scale = metadata_.activation_scale;
  qk_truncation_row.ell = kEll;
  qk_truncation_row.backend = "bolt_mpc_fix_right_shift_public_valid_slice";
  append_trace(trace, std::move(qk_truncation_row),
               qk_truncation_before_bytes, qk_truncation_before_rounds,
               elapsed_us(qk_truncation_start));
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  if (validation_observer != nullptr) {
    FixArray observed_qk = flatten(scaled_rows);
    validation_observer->observe(
        "block_" + std::to_string(layer) + ".qk", observed_qk,
        metadata_.heads * logical, logical, metadata_.heads * logical);
  }
#endif

  const std::uint64_t soft_before_bytes = io_.get_comm();
  const std::uint64_t soft_before_rounds = io_.get_rounds();
  const auto soft_start = std::chrono::steady_clock::now();
  // See the compact path above: retain BLB's existing wide ell=37 Softmax
  // so the 197-term normalization sum cannot wrap in a BERT-sized ell=19
  // accumulator.
  std::vector<FixArray> softmax_rows =
      fp_.softmax_fix_iron_1(scaled_rows);
  VitOperatorTrace soft_row;
  soft_row.phase = "online";
  soft_row.layer = layer;
  soft_row.op = "softmax";
  soft_row.input_shape = soft_row.output_shape =
      std::to_string(metadata_.heads * logical) + "x[" +
      std::to_string(logical) + "]";
  soft_row.input_packing =
      "public_mask_physical_" + std::to_string(physical) +
      "_to_logical_" + std::to_string(logical);
  soft_row.output_packing = "z2ell_row_shares_valid_submatrix";
  soft_row.input_scale = metadata_.activation_scale;
  soft_row.output_scale = metadata_.activation_scale;
  soft_row.backend = "bolt_mpc_softmax_fix_iron_1_ell37";
  append_trace(trace, std::move(soft_row), soft_before_bytes,
               soft_before_rounds, elapsed_us(soft_start));
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  if (validation_observer != nullptr) {
    FixArray observed_softmax = flatten(softmax_rows);
    validation_observer->observe(
        "block_" + std::to_string(layer) + ".softmax", observed_softmax,
        metadata_.heads * logical, logical, metadata_.heads * logical);
  }
#endif

  std::vector<BoltCipherTensor> softmax_cipher(metadata_.heads);
  for (std::size_t head = 0; head < metadata_.heads; ++head) {
    const std::uint64_t conversion_before_bytes = io_.get_comm();
    const std::uint64_t conversion_before_rounds = io_.get_rounds();
    const auto conversion_start = std::chrono::steady_clock::now();
    FixArray flat(party_, physical * physical, true,
                  kEll, metadata_.activation_scale);
    std::fill(flat.data, flat.data + flat.size, 0);
    for (std::size_t row = 0; row < logical; ++row)
      std::memcpy(flat.data + row * physical,
                  softmax_rows[head * logical + row].data,
                  logical * sizeof(std::uint64_t));
    softmax_cipher[head] = shares_to_cipher(
        flat, physical, physical, metadata_.attention_stride);
    append_conversion_trace(
        trace, "ss_to_he_softmax_head_" + std::to_string(head), layer,
        physical, physical, metadata_.activation_scale,
        "z2ell_softmax_valid_rows", "packed_ciphertext_rows",
        "bolt_conversion_ss_to_he", conversion_before_bytes,
        conversion_before_rounds, elapsed_us(conversion_start));
  }
  const std::uint64_t sv_before_bytes = io_.get_comm();
  const std::uint64_t sv_before_rounds = io_.get_rounds();
  const auto sv_start = std::chrono::steady_clock::now();
  BoltOperationCounts sv_counts;
  std::vector<BoltCipherTensor> head_outputs(metadata_.heads);
  if (party_ == ALICE) {
    for (std::size_t head = 0; head < metadata_.heads; ++head) {
      head_outputs[head] = linear_.softmax_v_matmul(
          softmax_cipher[head], v_heads[head],
          matmul_metadata(physical, physical,
                          metadata_.head_dim, metadata_.attention_stride, false),
          &sv_counts);
      head_outputs[head].metadata.scale = 2 * metadata_.activation_scale;
    }
  } else {
    for (auto &head : head_outputs)
      head.metadata = tensor_metadata(
          physical, metadata_.head_dim, metadata_.attention_stride,
          kEll, 2 * metadata_.activation_scale);
  }
  BoltCipherTensor merged;
  if (party_ == ALICE) {
    merged = packing_.merge_heads(head_outputs, attention_meta, &sv_counts);
    merged.metadata.scale = 2 * metadata_.activation_scale;
    // Keep enough noise budget for the public head rotations/masks above;
    // circuit privacy is required at the final fused boundary, before the
    // HE-to-share conversion, rather than on every partial head.
    linear_.harden_after_fusion(
        merged,
        matmul_metadata(physical, physical, metadata_.model_dim,
                        metadata_.attention_stride, true),
        &sv_counts);
  } else {
    merged.metadata = tensor_metadata(
        physical, metadata_.model_dim, metadata_.attention_stride,
        kEll, 2 * metadata_.activation_scale);
  }
  VitOperatorTrace sv_row;
  sv_row.phase = "online";
  sv_row.layer = layer;
  sv_row.op = "softmax_v";
  sv_row.input_shape = std::to_string(metadata_.heads) + "x(" +
      shape(physical, physical) + "x" +
      shape(physical, metadata_.head_dim) + ")";
  sv_row.output_shape = shape(physical, metadata_.model_dim);
  sv_row.input_packing = "packed_softmax_valid_" + std::to_string(logical) +
                         "_plus_retained_v_padding_masked";
  sv_row.output_packing = "packed_multi_head_then_cross_ciphertext_raw";
  sv_row.input_scale = metadata_.activation_scale;
  sv_row.output_scale = 2 * metadata_.activation_scale;
  sv_row.counts = sv_counts;
  sv_row.backend = "bolt_ct_ct_softmax_v_plus_packed_merge";
  append_trace(trace, std::move(sv_row), sv_before_bytes, sv_before_rounds,
               elapsed_us(sv_start));

  const std::uint64_t conversion_out_before_bytes = io_.get_comm();
  const std::uint64_t conversion_out_before_rounds = io_.get_rounds();
  const auto conversion_out_start = std::chrono::steady_clock::now();
  FixArray merged_raw = cipher_to_ring(
      merged, physical, metadata_.model_dim,
      metadata_.attention_stride, 2 * metadata_.activation_scale);
  append_conversion_trace(
      trace, "he_to_ss_softmax_v", layer, physical, metadata_.model_dim,
      2 * metadata_.activation_scale, "cross_packed_ciphertext_rows",
      "z2ell_shares", "bolt_conversion_he_to_ss",
      conversion_out_before_bytes, conversion_out_before_rounds,
      elapsed_us(conversion_out_start));
  const std::uint64_t truncation_before_bytes = io_.get_comm();
  const std::uint64_t truncation_before_rounds = io_.get_rounds();
  const auto truncation_start = std::chrono::steady_clock::now();
  FixArray merged_shares = fp_.fix->right_shift(
      merged_raw, metadata_.activation_scale);
  if (merged_shares.ell < kEll)
    merged_shares = fp_.fix->extend(merged_shares, kEll);
  VitOperatorTrace truncation_row;
  truncation_row.phase = "online";
  truncation_row.layer = layer;
  truncation_row.op = "softmax_v_share_truncation";
  truncation_row.input_shape = truncation_row.output_shape =
      shape(physical, metadata_.model_dim);
  truncation_row.input_packing = truncation_row.output_packing =
      "z2ell_token_major_context_shares";
  truncation_row.input_scale = 2 * metadata_.activation_scale;
  truncation_row.output_scale = metadata_.activation_scale;
  truncation_row.ell = merged_shares.ell;
  truncation_row.backend = "bolt_mpc_fix_right_shift";
  append_trace(trace, std::move(truncation_row), truncation_before_bytes,
               truncation_before_rounds, elapsed_us(truncation_start));
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  if (validation_observer != nullptr)
    validation_observer->observe(
        "block_" + std::to_string(layer) + ".attention_context",
        merged_shares, physical, metadata_.model_dim, logical);
#endif
  FixArray projected = packed_linear(
      merged_shares, physical, metadata_.model_dim,
      metadata_.model_dim, metadata_.attention_stride, block.wo, block.bo,
      metadata_.activation_scale, "output_projection", layer, trace,
      "bolt_ct_pt_bsgs", nullptr, 10);
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  if (validation_observer != nullptr)
    validation_observer->observe(
        "block_" + std::to_string(layer) + ".output_projection", projected,
        physical, metadata_.model_dim, logical);
#endif
  return projected;
}

FixArray BoltVitProtocol::gelu_bolt(
    const FixArray &input, int layer,
    std::vector<VitOperatorTrace> &trace) {
  if (input.s != static_cast<int>(metadata_.activation_scale) ||
      input.ell != static_cast<int>(metadata_.ell))
    throw std::invalid_argument("BOLT GELU input metadata mismatch");
  const std::uint64_t before_bytes = io_.get_comm();
  const std::uint64_t before_rounds = io_.get_rounds();
  const auto start = std::chrono::steady_clock::now();
  FixArray scale11 = fp_.fix->right_shift(
      input, metadata_.activation_scale - metadata_.gelu_input_scale);
  FixArray ell20 = fp_.fix->reduce(scale11, metadata_.gelu_ell);
  FixArray output = fp_.gelu_approx_2(ell20);
  if (output.ell != static_cast<int>(metadata_.ell) ||
      output.s != static_cast<int>(metadata_.gelu_output_scale))
    throw std::logic_error("BOLT gelu_approx_2 output contract changed");
  VitOperatorTrace row;
  row.phase = "online";
  row.layer = layer;
  row.op = "gelu";
  row.input_shape = row.output_shape =
      shape(metadata_.physical_tokens, metadata_.ff_dim);
  row.input_packing = row.output_packing = "z2ell_shares";
  row.input_scale = input.s;
  row.output_scale = output.s;
  row.ell = output.ell;
  row.backend = "bolt_mpc_gelu_approx_2_ell20_scale11_to_ell37_scale4";
  append_trace(trace, std::move(row), before_bytes, before_rounds,
               elapsed_us(start));
  return output;
}

VitProtocolOutput BoltVitProtocol::run(
    const VitModel &model, const std::vector<std::uint64_t> &image_share,
    std::vector<VitOperatorTrace> &trace, std::size_t block_limit
#ifdef BLB_VIT_VALIDATION_OUTPUTS
    , VitValidationObserver *validation_observer
#endif
    ) {
  if (image_share.size() != metadata_.channels * metadata_.image_size *
                                metadata_.image_size)
    throw std::invalid_argument("ViT image share shape mismatch");
  const std::uint64_t patch_before_bytes = io_.get_comm();
  const std::uint64_t patch_before_rounds = io_.get_rounds();
  const auto patch_start = std::chrono::steady_clock::now();
  const std::size_t side = metadata_.image_size / metadata_.patch_size;
  FixArray patches(party_, (metadata_.tokens - 1) * metadata_.patch_dim,
                   true, metadata_.ell, metadata_.activation_scale);
  std::size_t token = 0;
  for (std::size_t py = 0; py < side; ++py)
    for (std::size_t px = 0; px < side; ++px, ++token) {
      std::size_t feature = 0;
      for (std::size_t channel = 0; channel < metadata_.channels; ++channel)
        for (std::size_t ky = 0; ky < metadata_.patch_size; ++ky)
          for (std::size_t kx = 0; kx < metadata_.patch_size; ++kx, ++feature) {
            const std::size_t y = py * metadata_.patch_size + ky;
            const std::size_t x = px * metadata_.patch_size + kx;
            patches.data[token * metadata_.patch_dim + feature] =
                image_share[(channel * metadata_.image_size + y) *
                                 metadata_.image_size + x];
          }
    }
  VitOperatorTrace patch_row;
  patch_row.phase = "online";
  patch_row.op = "patchify";
  patch_row.input_shape = "[C,H,W]";
  patch_row.output_shape = shape(metadata_.tokens - 1, metadata_.patch_dim);
  patch_row.input_packing = "chw_shares";
  patch_row.output_packing = "py_px_c_ky_kx_shares";
  patch_row.input_scale = patch_row.output_scale = metadata_.activation_scale;
  patch_row.backend = "local_public_permutation";
  append_trace(trace, std::move(patch_row), patch_before_bytes,
               patch_before_rounds, elapsed_us(patch_start));

  FixArray patch_tokens = packed_linear(
      patches, metadata_.tokens - 1, metadata_.patch_dim,
      metadata_.model_dim, metadata_.patch_stride, model.patch_weight,
      model.patch_bias, metadata_.activation_scale, "patch_projection", -1,
      trace);
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  if (validation_observer != nullptr)
    validation_observer->observe(
        "patch_projection", patch_tokens, metadata_.tokens - 1,
        metadata_.model_dim, metadata_.tokens - 1);
#endif
  const std::uint64_t embed_before_bytes = io_.get_comm();
  const std::uint64_t embed_before_rounds = io_.get_rounds();
  const auto embed_start = std::chrono::steady_clock::now();
  FixArray hidden(party_, metadata_.physical_tokens * metadata_.model_dim, true,
                  metadata_.ell, metadata_.activation_scale);
  std::fill(hidden.data, hidden.data + hidden.size, 0);
  if (party_ == ALICE) {
    if (model.class_token.size() != metadata_.model_dim ||
        model.position.size() != metadata_.tokens * metadata_.model_dim)
      throw std::invalid_argument("ALICE CLS/position metadata mismatch");
    for (std::size_t col = 0; col < metadata_.model_dim; ++col)
      hidden.data[col] = ring_encode(quantize(
          model.class_token[col], metadata_.activation_scale));
  }
  for (std::size_t row = 0; row < metadata_.tokens - 1; ++row)
    std::memcpy(hidden.data + (row + 1) * metadata_.model_dim,
                patch_tokens.data + row * metadata_.model_dim,
                metadata_.model_dim * sizeof(std::uint64_t));
  if (party_ == ALICE)
    for (std::size_t i = 0;
         i < metadata_.tokens * metadata_.model_dim; ++i)
      hidden.data[i] = (hidden.data[i] + ring_encode(quantize(
          model.position[i], metadata_.activation_scale))) & hidden.ell_mask();
  VitOperatorTrace embed_row;
  embed_row.phase = "online";
  embed_row.op = "cls_position";
  embed_row.input_shape = shape(metadata_.tokens - 1, metadata_.model_dim);
  embed_row.output_shape = shape(metadata_.physical_tokens, metadata_.model_dim);
  embed_row.input_packing = "image_tokens";
  embed_row.output_packing = "cls_then_raster_patch_tokens";
  embed_row.input_scale = embed_row.output_scale = metadata_.activation_scale;
  embed_row.backend = "share_local_model_add";
  append_trace(trace, std::move(embed_row), embed_before_bytes,
               embed_before_rounds, elapsed_us(embed_start));
  apply_public_token_mask(hidden, "embedding_padding_mask", -1, trace);

  const std::size_t executed_layers =
      block_limit == 0 ? metadata_.layers : block_limit;
  if (executed_layers == 0 || executed_layers > metadata_.layers ||
      executed_layers > model.blocks.size())
    throw std::invalid_argument("ViT secure block limit is invalid");
  for (std::size_t layer = 0; layer < executed_layers; ++layer) {
    std::cerr << "BOLT_VIT_PROGRESS role="
              << (party_ == ALICE ? "ALICE" : "BOB")
              << " layer=" << (layer + 1) << '/' << executed_layers
              << " status=start" << std::endl;
    const VitBlockModel &block = model.blocks.at(layer);
    FixArray normalized = layer_norm(hidden, block.ln1_weight,
                                     block.ln1_bias, "ln1", layer, trace);
    FixArray attn = attention(normalized, block, layer, trace
#ifdef BLB_VIT_VALIDATION_OUTPUTS
                              , validation_observer
#endif
                              );
    hidden = residual_add(hidden, attn, "attention_residual", layer, trace);
#ifdef BLB_VIT_VALIDATION_OUTPUTS
    if (validation_observer != nullptr)
      validation_observer->observe(
          "block_" + std::to_string(layer) + ".attention_residual", hidden,
          metadata_.physical_tokens, metadata_.model_dim, metadata_.tokens);
#endif
    normalized = layer_norm(hidden, block.ln2_weight, block.ln2_bias,
                            "ln2", layer, trace);
    FixArray fc1 = packed_linear(
        normalized, metadata_.physical_tokens, metadata_.model_dim,
        metadata_.ff_dim,
        metadata_.ffn_stride, block.ff1_weight, block.ff1_bias,
        metadata_.activation_scale, "fc1", layer, trace,
        "bolt_ct_pt_bsgs", nullptr, 10);
    FixArray activated = gelu_bolt(fc1, layer, trace);
    FixArray fc2 = packed_linear(
        activated, metadata_.physical_tokens, metadata_.ff_dim,
        metadata_.model_dim,
        metadata_.ffn_stride, block.ff2_weight, block.ff2_bias,
        metadata_.activation_scale, "fc2", layer, trace,
        "bolt_ct_pt_bsgs", nullptr, 12);
    hidden = residual_add(hidden, fc2, "ffn_residual", layer, trace);
#ifdef BLB_VIT_VALIDATION_OUTPUTS
    if (validation_observer != nullptr)
      validation_observer->observe(
          "block_" + std::to_string(layer) + ".ffn_residual", hidden,
          metadata_.physical_tokens, metadata_.model_dim, metadata_.tokens);
#endif
    std::cerr << "BOLT_VIT_PROGRESS role="
              << (party_ == ALICE ? "ALICE" : "BOB")
              << " layer=" << (layer + 1) << '/' << executed_layers
              << " status=done" << std::endl;
  }

  hidden = layer_norm(hidden, model.final_ln_weight, model.final_ln_bias,
                      "final_ln", static_cast<int>(executed_layers), trace);
  const std::uint64_t cls_before_bytes = io_.get_comm();
  const std::uint64_t cls_before_rounds = io_.get_rounds();
  const auto cls_start = std::chrono::steady_clock::now();
  FixArray cls(party_, metadata_.model_dim, true, metadata_.ell,
               metadata_.activation_scale);
  std::memcpy(cls.data, hidden.data,
              metadata_.model_dim * sizeof(std::uint64_t));
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  if (validation_observer != nullptr)
    validation_observer->observe("cls", cls, 1, metadata_.model_dim, 1);
#endif
  VitOperatorTrace cls_row;
  cls_row.phase = "online";
  cls_row.op = "cls_extract";
  cls_row.input_shape = shape(metadata_.physical_tokens, metadata_.model_dim);
  cls_row.output_shape = shape(1, metadata_.model_dim);
  cls_row.input_packing = "token_major_shares";
  cls_row.output_packing = "cls_share";
  cls_row.input_scale = cls_row.output_scale = metadata_.activation_scale;
  cls_row.backend = "local_public_index";
  append_trace(trace, std::move(cls_row), cls_before_bytes,
               cls_before_rounds, elapsed_us(cls_start));
  FixArray logits = packed_linear(
      cls, 1, metadata_.model_dim, metadata_.classes,
      metadata_.classifier_stride, model.classifier_weight,
      model.classifier_bias, metadata_.activation_scale,
      "classification_head", static_cast<int>(executed_layers), trace,
      "bolt_ct_pt_bsgs", nullptr, 11);
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  if (validation_observer != nullptr)
    validation_observer->observe("logits", logits, 1, metadata_.classes, 1);
#endif

  const std::uint64_t arg_before_bytes = io_.get_comm();
  const std::uint64_t arg_before_rounds = io_.get_rounds();
  const auto arg_start = std::chrono::steady_clock::now();
  FixArray maximum = logits.subset(0, 1);
  FixArray index(party_, 1, false, metadata_.ell, 0);
  index.data[0] = 0;
  for (std::size_t candidate = 1; candidate < metadata_.classes; ++candidate) {
    FixArray value = logits.subset(candidate, candidate + 1);
    BoolArray greater = fp_.fix->GT(value, maximum);
    maximum = fp_.fix->if_else(greater, value, maximum);
    FixArray candidate_index(party_, 1, false, metadata_.ell, 0);
    candidate_index.data[0] =
        party_ == ALICE ? static_cast<std::uint64_t>(candidate) : 0;
    index = fp_.fix->if_else(greater, candidate_index, index);
  }
  VitOperatorTrace arg_row;
  arg_row.phase = "online";
  arg_row.op = "argmax";
  arg_row.input_shape = shape(1, metadata_.classes);
  arg_row.output_shape = "[1]";
  arg_row.input_packing = "logit_shares";
  arg_row.output_packing = "class_index_share";
  arg_row.input_scale = metadata_.activation_scale;
  arg_row.output_scale = 0;
  arg_row.backend = "bolt_mpc_argmax";
  append_trace(trace, std::move(arg_row), arg_before_bytes,
               arg_before_rounds, elapsed_us(arg_start));
  return {std::move(logits), std::move(index)};
}

void BoltVitProtocol::aggregate_trace(std::vector<VitOperatorTrace> &trace) {
  struct Wire {
    double milliseconds;
    std::uint64_t bytes;
    std::uint64_t rounds;
  };
  if (party_ == BOB) {
    for (const auto &row : trace) {
      Wire wire{row.milliseconds, row.bytes, row.logical_rounds};
      io_.io->send_data(&wire, sizeof(wire));
    }
    io_.io->flush();
    return;
  }
  for (auto &row : trace) {
    Wire peer{};
    io_.io->recv_data(&peer, sizeof(peer));
    // Attribute sequential operator time on ALICE's timeline.  Taking an
    // independent endpoint max for every row double-counts BOB's wait for an
    // ALICE-only HE kernel as part of the following interactive conversion.
    // Communication still combines both send-only endpoint counters below.
    // NetIO counters are send-only; the sum is one copy of each wire byte.
    row.bytes += peer.bytes;
    // Endpoint transition counters describe the same interaction.  max is
    // the critical-path logical-round approximation; summing is forbidden.
    row.logical_rounds = std::max(row.logical_rounds, peer.rounds);
  }
}
