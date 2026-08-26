#include "bolt_vit_protocol.h"

#include "LinearHE/defines-HE.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace sci;
using namespace sci::bolt;

namespace {

// ViT uses p=45,000,081,409 for its wider packed linear products.  The
// signed HE-to-MPC compatibility conversion can hold values below 2p, so
// its ring must satisfy 2p < 2^(ell-1); 38 bits is the smallest safe width.
constexpr int kEll = 38;
constexpr int kScale = 12;

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
  metadata.noise_flooding_bits = harden ? SMUDGING_BITLEN_bert2 : 0;
  metadata.modulus_switches = harden ? 1 : 0;
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

std::size_t bert_baby_step(std::size_t input_dim, std::size_t output_dim,
                           std::size_t num_diag) {
  // Preserve BERT's shape-specific inner BSGS schedules.  The physical token
  // count changes the number of giant groups, but it must not scale the
  // baby-step rotation period: that period is part of the BERT packing
  // layout and controls which token rows are aligned with each plaintext
  // diagonal.  For the BERT-compatible ViT operators these are 2/8/4 for
  // 3072->768, 768->3072 and 768->(768 or a padded classifier width).
  std::size_t baby = 0;
  if (input_dim == 3072 && output_dim == 768) {
    baby = 2;
  } else if (input_dim == 768 && output_dim == 3072) {
    baby = 8;
  } else if (input_dim == 768 && output_dim >= 768) {
    baby = 4;
  } else {
    throw std::invalid_argument(
        "BOLT ViT standard packed schedule has no BERT-compatible shape for " +
        std::to_string(input_dim) + "->" + std::to_string(output_dim));
  }
  if (baby == 0 || num_diag % baby != 0)
    throw std::invalid_argument("BOLT ViT BSGS schedule does not divide diagonals");
  return baby;
}

BoltBertPackingMetadata bert_metadata(const BoltHE &he, std::size_t rows,
                                      std::size_t input_dim,
                                      std::size_t output_dim,
                                      std::size_t baby_step,
                                      std::uint32_t noise_flooding_bits,
                                      std::uint32_t modulus_switches) {
  if (rows == 0 || input_dim == 0 || output_dim == 0 ||
      he.slot_count() % (2 * rows) != 0 ||
      rows * input_dim % he.slot_count() != 0 ||
      rows * output_dim % he.slot_count() != 0) {
    throw std::invalid_argument("BOLT ViT shape is not aligned to BERT compact slots");
  }
  const std::size_t num_diag = he.slot_count() / (2 * rows);
  if (baby_step == 0 || num_diag % baby_step != 0)
    throw std::invalid_argument("BOLT ViT invalid BERT-compatible baby step");
  BoltBertPackingMetadata result;
  result.slot_count = he.slot_count();
  result.image_size = rows;
  result.filter_height = input_dim;
  result.filter_width = output_dim;
  result.baby_step = baby_step;
  result.giant_steps = num_diag / baby_step;
  result.attention_heads = 12;
  result.paired_head_groups = 6;
  result.noise_flooding_bits = noise_flooding_bits;
  result.modulus_switches = modulus_switches;
  return result;
}

std::vector<std::uint64_t> row_to_column(const FixArray &input,
                                         std::size_t rows,
                                         std::size_t cols) {
  if (input.size != static_cast<int>(rows * cols))
    throw std::invalid_argument("BOLT ViT row-to-column shape mismatch");
  std::vector<std::uint64_t> result(rows * cols);
  for (std::size_t col = 0; col < cols; ++col)
    for (std::size_t row = 0; row < rows; ++row)
      result[col * rows + row] = input.data[row * cols + col];
  return result;
}

std::vector<std::uint64_t> unpack_bert_column_output(
    const std::vector<std::uint64_t> &packed, std::size_t rows,
    std::size_t output_dim, std::size_t slot_count);

std::vector<std::uint64_t> unpack_bert_column_output(
    const std::vector<std::uint64_t> &packed, std::size_t rows,
    std::size_t output_dim, std::size_t slot_count) {
  if (rows * output_dim % slot_count != 0 ||
      packed.size() != rows * output_dim)
    throw std::invalid_argument("BOLT ViT ct-pt output unpack shape mismatch");
  const std::size_t half = slot_count / 2;
  const std::size_t columns_per_ct = slot_count / (2 * rows);
  const std::size_t cts = rows * output_dim / slot_count;
  std::vector<std::uint64_t> result(rows * output_dim, 0);
  for (std::size_t ct = 0; ct < cts; ++ct) {
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
      std::size_t column = slot / rows;
      const std::size_t row = slot % rows;
      if (slot >= half) {
        column -= half / rows;
        column += output_dim / 2;
      }
      column += ct * columns_per_ct;
      if (column >= output_dim)
        throw std::logic_error("BOLT ViT ct-pt output column overrun");
      result[row * output_dim + column] = packed[ct * slot_count + slot];
    }
  }
  return result;
}

std::vector<std::uint64_t> unpack_bert_qk_cross(
    const std::vector<std::uint64_t> &packed, std::size_t heads,
    std::size_t rows, std::size_t slot_count) {
  if (heads % 2 != 0 || rows == 0 ||
      packed.size() != heads * rows * rows)
    throw std::invalid_argument("BOLT ViT QK cross output shape mismatch");
  const std::size_t per_group = rows * rows * 2 / slot_count;
  const std::size_t columns_per_ct = slot_count / (2 * rows);
  const std::size_t half = slot_count / 2;
  const std::size_t groups = heads / 2;
  if (per_group == 0 || packed.size() / slot_count != groups * per_group)
    throw std::invalid_argument("BOLT ViT QK cross ciphertext shape mismatch");
  std::vector<std::uint64_t> result(packed.size(), 0);
  for (std::size_t ct = 0; ct < packed.size() / slot_count; ++ct) {
    const std::size_t group = ct / per_group;
    const std::size_t current_col = ct % per_group;
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
      std::size_t column = slot / rows + current_col * columns_per_ct;
      const std::size_t row = slot % rows;
      std::size_t head_offset = 0;
      if (slot >= half) {
        column -= half / rows;
        head_offset = rows * rows;
      }
      if (column >= rows)
        throw std::logic_error("BOLT ViT QK cross column overrun");
      result[group * 2 * rows * rows + head_offset +
             row * rows + (row + column) % rows] =
          packed[ct * slot_count + slot];
    }
  }
  return result;
}

std::vector<std::uint64_t> unpack_bert_softmax_v(
    const std::vector<std::uint64_t> &packed, std::size_t heads,
    std::size_t rows, std::size_t head_dim, std::size_t slot_count) {
  if (heads == 0 || rows == 0 || head_dim == 0 ||
      packed.size() != heads * rows * head_dim || heads % 2 != 0)
    throw std::invalid_argument("BOLT ViT Softmax-V output shape mismatch");
  const std::size_t chunk_width = slot_count / (2 * rows);
  const std::size_t cts_per_head = rows * head_dim / slot_count;
  const std::size_t cts_per_group = 2 * cts_per_head;
  const std::size_t half = slot_count / 2;
  const std::size_t groups = heads / 2;
  if (chunk_width == 0 || cts_per_head == 0 ||
      packed.size() / slot_count != groups * cts_per_group)
    throw std::invalid_argument("BOLT ViT Softmax-V ciphertext shape mismatch");
  std::vector<std::uint64_t> result(packed.size(), 0);
  for (std::size_t ct = 0; ct < packed.size() / slot_count; ++ct) {
    const std::size_t group = ct / cts_per_group;
    const std::size_t within = ct % cts_per_group;
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
      const std::size_t row = slot % rows;
      std::size_t local_column = slot / rows;
      const std::size_t head_local = slot >= half ? 1 : 0;
      if (slot >= half) local_column -= half / rows;
      local_column += within * chunk_width;
      if (local_column >= head_dim)
        throw std::logic_error("BOLT ViT Softmax-V column overrun");
      const std::size_t head = group * 2 + head_local;
      result[head * rows * head_dim + local_column * rows + row] =
          packed[ct * slot_count + slot];
    }
  }
  return result;
}

std::vector<std::uint64_t> preprocess_bert_softmax(
    const std::vector<std::uint64_t> &row_major, std::size_t heads,
    std::size_t rows, std::size_t slot_count) {
  if (row_major.size() != heads * rows * rows || heads % 2 != 0 ||
      slot_count % (2 * rows) != 0)
    throw std::invalid_argument("BOLT ViT Softmax diagonal input shape mismatch");
  // bert_pack_softmax_diagonals() consumes the BERT diagonal-source layout:
  // source ciphertext c, block b contains the circular diagonal d=c*B+b,
  // where B=slot_count/(2*rows).  The source row j stores
  // softmax[j, (j+d) mod rows].  This is the encrypted-share counterpart of
  // BERT's preprocess_softmax_s1 + preprocess_softmax_s2 boundary.  Keeping
  // this as a public permutation of the secret shares lets the existing BERT
  // diagonal packer own all rotations and masks; no per-head conversion is
  // introduced here.
  const std::size_t num_cts_per_group = rows * rows * 2 / slot_count;
  const std::size_t blocks_per_ciphertext = slot_count / (2 * rows);
  const std::size_t half = slot_count / 2;
  const std::size_t groups = heads / 2;
  std::vector<std::uint64_t> result(heads * rows * rows, 0);
  for (std::size_t group = 0; group < groups; ++group) {
    for (std::size_t source_ct = 0; source_ct < num_cts_per_group;
         ++source_ct) {
      const std::size_t ct = group * num_cts_per_group + source_ct;
      for (std::size_t block = 0; block < blocks_per_ciphertext; ++block) {
        const std::size_t diagonal = source_ct * blocks_per_ciphertext + block;
        for (std::size_t row = 0; row < rows; ++row) {
          const std::size_t source_row = (row + diagonal) % rows;
          result[ct * slot_count + block * rows + row] =
              row_major[group * 2 * rows * rows + row * rows + source_row];
          result[ct * slot_count + half + block * rows + row] =
              row_major[group * 2 * rows * rows + rows * rows + row * rows +
                        source_row];
        }
      }
    }
  }
  return result;
}

} // namespace

BoltVitProtocol::BoltVitProtocol(int party, IOPack &io, FPMath &fp,
                                 BoltVitMPC &mpc,
                                 BoltHE &he,
                                 BoltHE &he_tiny,
                                 BoltHE &he_layer_norm,
                                 const BoltVitModelMetadata &metadata)
    : party_(party), io_(io), fp_(fp), mpc_(mpc), he_(he), he_tiny_(he_tiny),
      he_layer_norm_(he_layer_norm), conversion_(he.view()),
      conversion_tiny_(he_tiny.view()),
      conversion_layer_norm_(he_layer_norm.view()), linear_(he.view()),
      linear_tiny_(he_tiny.view()),
      linear_layer_norm_(he_layer_norm.view()),
      packing_(he.view()), metadata_(metadata) {
  metadata_.validate(he_.slot_count());
  if (he_tiny_.slot_count() != he_.slot_count())
    throw std::invalid_argument(
        "BOLT ViT main and tiny HE contexts must have identical slot counts");
  metadata_.validate(he_tiny_.slot_count());
  fail_if_scalar_ciphertext_adapter(false);
  fail_if_generic_ot_matmul(false);
}

std::uint64_t BoltVitProtocol::wire_bytes() const {
  return io_.get_comm() + mpc_.communication();
}

std::uint64_t BoltVitProtocol::wire_rounds() const {
  return io_.get_rounds() + mpc_.rounds();
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
  const std::uint64_t before_bytes = wire_bytes();
  const std::uint64_t before_rounds = wire_rounds();
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
  const std::uint64_t before_bytes = wire_bytes();
  const std::uint64_t before_rounds = wire_rounds();
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
  if (row.plain_modulus == 0)
    row.plain_modulus = he_.parameters().plain_modulus;
  row.bytes = wire_bytes() - before_bytes;
  row.logical_rounds = wire_rounds() - before_rounds;
  row.milliseconds = static_cast<double>(elapsed_microseconds) / 1000.0;
  trace.push_back(std::move(row));
}

void BoltVitProtocol::append_conversion_trace(
    std::vector<VitOperatorTrace> &trace, const std::string &op, int layer,
    std::size_t rows, std::size_t cols, int scale,
    const std::string &input_packing, const std::string &output_packing,
    const std::string &backend, std::uint64_t before_bytes,
    std::uint64_t before_rounds, std::uint64_t elapsed_microseconds,
    std::uint64_t plain_modulus) {
  VitOperatorTrace row;
  row.phase = "online";
  row.layer = layer;
  row.op = op;
  row.input_shape = row.output_shape = shape(rows, cols);
  row.input_packing = input_packing;
  row.output_packing = output_packing;
  row.input_scale = row.output_scale = scale;
  row.ell = kEll;
  row.plain_modulus = plain_modulus;
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
  return conversion_.ss_to_he(shares, shares.metadata);
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

FixArray BoltVitProtocol::packed_field_to_ring(
    const std::vector<std::uint64_t> &field, int scale, BoltHE &field_he) {
  if (field.empty())
    throw std::invalid_argument("BOLT packed HE-to-SS field is empty");
  FixArray input(party_, static_cast<int>(field.size()), true, kEll, scale);
  std::memcpy(input.data, field.data(), field.size() * sizeof(std::uint64_t));
  return mpc_.gt_p_sub(input, field_he.parameters().plain_modulus, scale);
}

std::vector<seal::Ciphertext> BoltVitProtocol::signed_ring_to_he(
    const std::uint64_t *local_share, std::size_t element_count, int ell,
    int scale, BoltConversion &conversion) {
  if (local_share == nullptr || element_count == 0 || ell <= 0 || ell >= 64)
    throw std::invalid_argument("BOLT ViT signed ring-to-HE input is invalid");
  if (element_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("BOLT ViT signed ring-to-HE input is too large");
  FixArray input(party_, static_cast<int>(element_count), true, ell, scale);
  std::memcpy(input.data, local_share,
              element_count * sizeof(std::uint64_t));
  FixArray extended = mpc_.signed_extend_to_64(input);
  return conversion.ss_to_he_packed_signed_ring_compat(
      extended.data, static_cast<std::size_t>(extended.size));
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
    BoltCipherTensor *retained_packed_output, bool input_column_packed) {
  (void)stride;
  // BOLT-BERT uses the main BFV context for fused QKV/attention and the
  // smaller-plaintext-modulus context for its ordinary ct-pt linears.  ViT
  // keeps that operator-role split: the patch stem, FFN, output projection,
  // and classifier are ordinary packed linears and therefore use tiny HE.
  const bool use_tiny_he =
      op == "patch_projection" || op == "fc1" || op == "fc2" ||
      op == "output_projection" || op == "classification_head";
  BoltHE &linear_he = use_tiny_he ? he_tiny_ : he_;
  BoltConversion &linear_conversion =
      use_tiny_he ? conversion_tiny_ : conversion_;
  BoltLinear &linear_backend = use_tiny_he ? linear_tiny_ : linear_;
  if (retained_packed_output != nullptr)
    throw std::logic_error(
        "BOLT ViT packed linear cannot retain a separately re-encrypted tensor");
  if (input.size != static_cast<int>(rows * input_dim) ||
      output_scale < 0 || input.s + static_cast<int>(metadata_.weight_scale) <
                              output_scale)
    throw std::invalid_argument("ViT packed linear input metadata mismatch");
  if (rows != metadata_.physical_tokens)
    throw std::invalid_argument(
        "BOLT ViT formal packed linear requires physical token rows");
  const std::size_t num_diag = linear_he.slot_count() / (2 * rows);
  const std::size_t baby_step =
      bert_baby_step(input_dim, output_dim, num_diag);
  const BoltBertPackingMetadata bert = bert_metadata(
      linear_he, rows, input_dim, output_dim, baby_step,
      SMUDGING_BITLEN_bert2, 1);
  const int raw_scale = input.s + metadata_.weight_scale;
  std::vector<std::uint64_t> column_input;
  if (input_column_packed)
    column_input.assign(input.data, input.data + input.size);
  else
    column_input = row_to_column(input, rows, input_dim);

  const std::uint64_t conversion_in_before_bytes = wire_bytes();
  const std::uint64_t conversion_in_before_rounds = wire_rounds();
  const auto conversion_in_start = std::chrono::steady_clock::now();
  std::vector<seal::Ciphertext> encrypted =
      signed_ring_to_he(column_input.data(), column_input.size(), input.ell,
                        input.s, linear_conversion);
  append_conversion_trace(
      trace, "ss_to_he_" + op, layer, rows, input_dim, input.s,
      input_column_packed ? "bert_column_packed_shares"
                          : "bert_row_to_column_packed_shares",
      "bert_column_packed_ciphertext",
      use_tiny_he ? "bolt_conversion_ss_to_he_tiny_compat"
                  : "bolt_conversion_ss_to_he_compat",
      conversion_in_before_bytes, conversion_in_before_rounds,
      elapsed_us(conversion_in_start), linear_he.parameters().plain_modulus);

  const std::uint64_t he_before_bytes = wire_bytes();
  const std::uint64_t he_before_rounds = wire_rounds();
  const auto he_start = std::chrono::steady_clock::now();
  const std::size_t output_ciphertexts =
      rows * output_dim / linear_he.slot_count();
  std::vector<seal::Ciphertext> output;
  BoltOperationCounts counts;
  if (party_ == ALICE) {
    if (weight.size() != output_dim * input_dim || bias.size() != output_dim)
      throw std::invalid_argument("ALICE ViT weight metadata mismatch: " + op);
    std::vector<std::vector<std::uint64_t>> matrix(
        input_dim, std::vector<std::uint64_t>(output_dim));
    for (std::size_t input_col = 0; input_col < input_dim; ++input_col)
      for (std::size_t output_col = 0; output_col < output_dim; ++output_col)
        matrix[input_col][output_col] = prime_encode(
            quantize(weight[output_col * input_dim + input_col],
                     metadata_.weight_scale),
            linear_he.parameters().plain_modulus);
    std::vector<std::uint64_t> encoded_bias(output_dim);
    for (std::size_t output_col = 0; output_col < output_dim; ++output_col)
      encoded_bias[output_col] = prime_encode(
          quantize(bias[output_col], raw_scale),
          linear_he.parameters().plain_modulus);
    const auto packed_matrix =
        linear_backend.bert_pack_cipher_plain_matrix(matrix, bert);
    const auto packed_bias =
        linear_backend.bert_pack_cipher_plain_bias(encoded_bias, bert);
    output.resize(output_ciphertexts);
    linear_backend.bert_cipher_plain_matmul_bsgs_compat(
        encrypted, packed_matrix.first, packed_matrix.second, packed_bias,
        bert, output, &counts);
  }
  VitOperatorTrace row;
  row.phase = "online";
  row.layer = layer;
  row.op = op;
  row.input_shape = shape(rows, input_dim);
  row.output_shape = shape(rows, output_dim);
  row.input_packing = input_column_packed
                          ? "bert_column_packed_ciphertext"
                          : "bert_column_packed_ciphertext_from_row_major";
  row.output_packing = "bert_compact_bsgs_cross_ciphertext_raw";
  row.input_scale = input.s;
  row.output_scale = raw_scale;
  row.ell = kEll;
  row.counts = counts;
  row.plain_modulus = linear_he.parameters().plain_modulus;
  row.backend = backend.empty() ? "bolt_ct_pt_bsgs_bert_compat" : backend;
  if (use_tiny_he) row.backend += "_tiny_he";
  append_trace(trace, std::move(row), he_before_bytes, he_before_rounds,
               elapsed_us(he_start));

  const std::uint64_t conversion_out_before_bytes = wire_bytes();
  const std::uint64_t conversion_out_before_rounds = wire_rounds();
  const auto conversion_out_start = std::chrono::steady_clock::now();
  std::vector<std::uint64_t> field =
      linear_conversion.he_to_ss_packed_compat(output, output_ciphertexts,
                                               true);
  std::vector<std::uint64_t> row_major = unpack_bert_column_output(
      field, rows, output_dim, linear_he.slot_count());
  FixArray ring = packed_field_to_ring(row_major, raw_scale, linear_he);
  append_conversion_trace(
      trace, "he_to_ss_" + op, layer, rows, output_dim, raw_scale,
      "bert_compact_bsgs_cross_ciphertext_raw", "bert_row_major_shares",
      use_tiny_he ? "bolt_conversion_he_to_ss_tiny_compat"
                  : "bolt_conversion_he_to_ss_compat",
      conversion_out_before_bytes, conversion_out_before_rounds,
      elapsed_us(conversion_out_start), linear_he.parameters().plain_modulus);
  if (raw_scale > output_scale)
    ring = mpc_.right_shift(ring, raw_scale - output_scale);
  if (ring.ell < kEll) ring = fp_.fix->extend(ring, kEll);
  return ring;
}

FixArray BoltVitProtocol::layer_norm(
    const FixArray &input, const std::vector<double> &gamma,
    const std::vector<double> &beta, const std::string &name, int layer,
    std::vector<VitOperatorTrace> &trace) {
  const std::uint64_t before_bytes = wire_bytes();
  const std::uint64_t before_rounds = wire_rounds();
  const auto start = std::chrono::steady_clock::now();
  FixArray masked_input = input;
  apply_public_token_mask(masked_input, name + "_input_padding_mask", layer,
                          trace);
  FixArray normalized = flatten(mpc_.layer_norm(
      split_rows(masked_input, metadata_.physical_tokens, metadata_.model_dim),
      metadata_.layer_norm_epsilon_lsb));
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

  if (party_ == ALICE &&
      (gamma.size() != metadata_.model_dim ||
       beta.size() != metadata_.model_dim))
    throw std::invalid_argument("ALICE LayerNorm parameter mismatch");
  const std::size_t ciphertext_count =
      normalized.size / static_cast<int>(he_layer_norm_.slot_count());
  const int raw_scale = normalized.s + metadata_.weight_scale;
  const std::uint64_t conversion_in_before_bytes = wire_bytes();
  const std::uint64_t conversion_in_before_rounds = wire_rounds();
  const auto conversion_in_start = std::chrono::steady_clock::now();
  std::vector<seal::Ciphertext> encrypted =
      signed_ring_to_he(normalized.data, normalized.size, normalized.ell,
                        normalized.s, conversion_layer_norm_);
  append_conversion_trace(
      trace, "ss_to_he_" + name + "_affine", layer,
      metadata_.physical_tokens, metadata_.model_dim, normalized.s,
      "z2ell_row_major_shares", "bert_ln_row_major_ciphertext",
      "bolt_conversion_ss_to_he_layer_norm_compat",
      conversion_in_before_bytes, conversion_in_before_rounds,
      elapsed_us(conversion_in_start));
  trace.back().plain_modulus = he_layer_norm_.parameters().plain_modulus;

  std::vector<seal::Plaintext> encoded_gamma;
  if (party_ == ALICE) {
    auto ln_view = he_layer_norm_.view();
    encoded_gamma.resize(ciphertext_count);
    for (std::size_t ct = 0; ct < ciphertext_count; ++ct) {
      std::vector<std::uint64_t> slots(he_layer_norm_.slot_count());
      for (std::size_t slot = 0; slot < slots.size(); ++slot)
        slots[slot] = prime_encode(
            quantize(gamma[(ct * slots.size() + slot) % metadata_.model_dim],
                     metadata_.weight_scale),
            he_layer_norm_.parameters().plain_modulus);
      ln_view.encoder->encode(slots, encoded_gamma[ct]);
    }
  }
  const std::uint64_t he_before_bytes = wire_bytes();
  const std::uint64_t he_before_rounds = wire_rounds();
  const auto he_start = std::chrono::steady_clock::now();
  std::vector<seal::Ciphertext> affine_cipher;
  BoltOperationCounts affine_counts;
  if (party_ == ALICE) {
    linear_layer_norm_.bert_layer_norm_affine_compat(
        encrypted, encoded_gamma, SMUDGING_BITLEN_bert3, 2, affine_cipher,
        &affine_counts);
  }
  VitOperatorTrace affine_row;
  affine_row.phase = "online";
  affine_row.layer = layer;
  affine_row.op = name + "_affine";
  affine_row.input_shape = affine_row.output_shape =
      shape(metadata_.physical_tokens, metadata_.model_dim);
  affine_row.input_packing = "bert_ln_row_major_ciphertext";
  affine_row.output_packing = "bert_ln_affine_ciphertext_raw";
  affine_row.input_scale = normalized.s;
  affine_row.output_scale = raw_scale;
  affine_row.ell = kEll;
  affine_row.counts = affine_counts;
  affine_row.backend = "bolt_ct_pt_layernorm_affine_bert_compat";
  append_trace(trace, std::move(affine_row), he_before_bytes,
               he_before_rounds, elapsed_us(he_start));
  trace.back().plain_modulus = he_layer_norm_.parameters().plain_modulus;

  const std::uint64_t conversion_out_before_bytes = wire_bytes();
  const std::uint64_t conversion_out_before_rounds = wire_rounds();
  const auto conversion_out_start = std::chrono::steady_clock::now();
  const std::vector<std::uint64_t> field =
      conversion_layer_norm_.he_to_ss_packed_compat(
          affine_cipher, ciphertext_count, true);
  FixArray affine = packed_field_to_ring(field, raw_scale, he_layer_norm_);
  append_conversion_trace(
      trace, "he_to_ss_" + name + "_affine", layer,
      metadata_.physical_tokens, metadata_.model_dim, raw_scale,
      "bert_ln_affine_ciphertext_raw", "z2ell_row_major_shares",
      "bolt_conversion_he_to_ss_layer_norm_compat",
      conversion_out_before_bytes, conversion_out_before_rounds,
      elapsed_us(conversion_out_start));
  trace.back().plain_modulus = he_layer_norm_.parameters().plain_modulus;
  if (raw_scale > static_cast<int>(metadata_.activation_scale))
    affine = mpc_.right_shift(
        affine, raw_scale - metadata_.activation_scale);
  if (affine.ell < kEll) affine = fp_.fix->extend(affine, kEll);
  add_local_bias(affine, beta, metadata_.physical_tokens, metadata_.model_dim,
                 metadata_.activation_scale);
  apply_public_token_mask(affine, name + "_output_padding_mask", layer,
                          trace);
  return affine;
}

FixArray BoltVitProtocol::residual_add(
    const FixArray &lhs, const FixArray &rhs, const std::string &name,
    int layer, std::vector<VitOperatorTrace> &trace) {
  if (lhs.size != rhs.size || lhs.ell != rhs.ell || lhs.s != rhs.s)
    throw std::invalid_argument("ViT residual metadata mismatch");
  const std::uint64_t before_bytes = wire_bytes();
  const std::uint64_t before_rounds = wire_rounds();
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

FixArray BoltVitProtocol::attention(
    const FixArray &input, const VitBlockModel &block, int layer,
    std::vector<VitOperatorTrace> &trace) {
  const std::size_t physical = metadata_.physical_tokens;
  const std::size_t logical = metadata_.tokens;
  if (metadata_.heads != 12 || metadata_.model_dim != 768 ||
      metadata_.head_dim != 64 || physical != 256)
    throw std::invalid_argument(
        "BOLT ViT packed attention currently requires standard ViT-B/16 shape");

  const BoltBertPackingMetadata qkv_metadata =
      bert_metadata(he_, physical, metadata_.model_dim, metadata_.head_dim,
                    4, 0, 0);
  const int qkv_raw_scale =
      input.s + static_cast<int>(metadata_.weight_scale);
  const int qkv_scale = static_cast<int>(metadata_.activation_scale);
  const std::vector<std::uint64_t> column_input =
      row_to_column(input, physical, metadata_.model_dim);
  const std::uint64_t qkv_conversion_before_bytes = wire_bytes();
  const std::uint64_t qkv_conversion_before_rounds = wire_rounds();
  const auto qkv_conversion_start = std::chrono::steady_clock::now();
  const std::vector<seal::Ciphertext> encrypted =
      signed_ring_to_he(column_input.data(), column_input.size(), input.ell,
                        input.s, conversion_);
  append_conversion_trace(
      trace, "ss_to_he_qkv", layer, physical, metadata_.model_dim, input.s,
      "bert_row_to_column_packed_shares", "bert_column_packed_ciphertext",
      "bolt_conversion_ss_to_he_qkv_fused_compat",
      qkv_conversion_before_bytes, qkv_conversion_before_rounds,
      elapsed_us(qkv_conversion_start));

  std::vector<std::vector<std::vector<std::uint64_t>>> q_weights;
  std::vector<std::vector<std::vector<std::uint64_t>>> k_weights;
  std::vector<std::vector<std::vector<std::uint64_t>>> v_weights;
  std::vector<std::vector<std::uint64_t>> q_bias;
  std::vector<std::vector<std::uint64_t>> k_bias;
  std::vector<std::vector<std::uint64_t>> v_bias;
  if (party_ == ALICE) {
    auto encode_qkv = [&](const std::vector<double> &flat) {
      std::vector<std::vector<std::vector<std::uint64_t>>> result(
          metadata_.heads,
          std::vector<std::vector<std::uint64_t>>(
              metadata_.model_dim,
              std::vector<std::uint64_t>(metadata_.head_dim)));
      if (flat.size() != metadata_.model_dim * metadata_.model_dim)
        throw std::invalid_argument("ALICE ViT QKV weight metadata mismatch");
      for (std::size_t head = 0; head < metadata_.heads; ++head)
        for (std::size_t input_col = 0; input_col < metadata_.model_dim;
             ++input_col)
          for (std::size_t out = 0; out < metadata_.head_dim; ++out)
            result[head][input_col][out] = prime_encode(
                quantize(flat[(head * metadata_.head_dim + out) *
                              metadata_.model_dim + input_col],
                         metadata_.weight_scale),
                he_.parameters().plain_modulus);
      return result;
    };
    auto encode_bias = [&](const std::vector<double> &flat) {
      std::vector<std::vector<std::uint64_t>> result(
          metadata_.heads, std::vector<std::uint64_t>(metadata_.head_dim));
      if (flat.size() != metadata_.model_dim)
        throw std::invalid_argument("ALICE ViT QKV bias metadata mismatch");
      for (std::size_t head = 0; head < metadata_.heads; ++head)
        for (std::size_t out = 0; out < metadata_.head_dim; ++out)
          result[head][out] = prime_encode(
              quantize(flat[head * metadata_.head_dim + out], qkv_raw_scale),
              he_.parameters().plain_modulus);
      return result;
    };
    q_weights = encode_qkv(block.wq);
    k_weights = encode_qkv(block.wk);
    v_weights = encode_qkv(block.wv);
    q_bias = encode_bias(block.q_bias);
    k_bias = encode_bias(block.k_bias);
    v_bias = encode_bias(block.v_bias);
  }

  const std::uint64_t qkv_before_bytes = wire_bytes();
  const std::uint64_t qkv_before_rounds = wire_rounds();
  const auto qkv_start = std::chrono::steady_clock::now();
  std::vector<seal::Ciphertext> qkv_cipher;
  BoltOperationCounts qkv_counts;
  if (party_ == ALICE) {
    const auto q_pack = linear_.bert_pack_qkv_matrix(q_weights, qkv_metadata);
    const auto k_pack = linear_.bert_pack_qkv_matrix(k_weights, qkv_metadata);
    const auto v_pack = linear_.bert_pack_qkv_matrix(v_weights, qkv_metadata);
    const auto q_bias_pack = linear_.bert_pack_qkv_bias(q_bias, qkv_metadata);
    const auto k_bias_pack = linear_.bert_pack_qkv_bias(k_bias, qkv_metadata);
    const auto v_bias_pack = linear_.bert_pack_qkv_bias(v_bias, qkv_metadata);
    qkv_cipher.resize(3 * physical * metadata_.head_dim * metadata_.heads /
                      he_.slot_count());
    linear_.bert_qkv_cipher_plain_matmul_bsgs_compat(
        encrypted, q_pack, k_pack, v_pack, q_bias_pack, k_bias_pack,
        v_bias_pack, qkv_metadata, qkv_cipher, &qkv_counts);
  }
  VitOperatorTrace qkv_row;
  qkv_row.phase = "online";
  qkv_row.layer = layer;
  qkv_row.op = "qkv";
  qkv_row.input_shape = shape(physical, metadata_.model_dim);
  qkv_row.output_shape = std::to_string(metadata_.heads) + "x" +
                         shape(physical, metadata_.head_dim) + " (Q,K,V)";
  qkv_row.input_packing = "bert_column_packed_ciphertext_shared_input";
  qkv_row.output_packing = "bert_fused_qkv_compact_ciphertext_q_k_v_tail";
  qkv_row.input_scale = input.s;
  qkv_row.output_scale = qkv_raw_scale;
  qkv_row.ell = kEll;
  qkv_row.counts = qkv_counts;
  qkv_row.backend = "bolt_ct_pt_qkv_bsgs_bert_compat";
  append_trace(trace, std::move(qkv_row), qkv_before_bytes,
               qkv_before_rounds, elapsed_us(qkv_start));

  // The fused BERT-compatible QKV kernel evaluates the three projections at
  // the raw input+weight scale.  For ViT-B/16 that raw ciphertext value is too
  // wide to enter the Z2^38 QK/Softmax boundary directly: the subsequent
  // QK product would require roughly 58 bits.  BOLT's fixed-point contract
  // truncates every linear result back to activation scale before the next
  // nonlinear operator.  Do that once for the complete fused QKV tensor,
  // preserving the packed Q/K/V order and avoiding any per-head conversion.
  const std::size_t one_projection =
      physical * metadata_.head_dim * metadata_.heads / he_.slot_count();
  const std::size_t qkv_ciphertexts = 3 * one_projection;
  const std::uint64_t qkv_scale_before_bytes = wire_bytes();
  const std::uint64_t qkv_scale_before_rounds = wire_rounds();
  const auto qkv_he_to_ss_start = std::chrono::steady_clock::now();
  const std::vector<std::uint64_t> qkv_field =
      conversion_.he_to_ss_packed_compat(qkv_cipher, qkv_ciphertexts, true);
  append_conversion_trace(
      trace, "he_to_ss_qkv_scale_boundary", layer, 3 * physical,
      metadata_.model_dim, qkv_raw_scale,
      "bert_fused_q_k_v_ciphertext_raw",
      "bert_fused_q_k_v_column_field_shares_raw",
      "bolt_conversion_he_to_ss_qkv_batch_compat",
      qkv_scale_before_bytes, qkv_scale_before_rounds,
      elapsed_us(qkv_he_to_ss_start), he_.parameters().plain_modulus);

  const std::uint64_t qkv_shift_before_bytes = wire_bytes();
  const std::uint64_t qkv_shift_before_rounds = wire_rounds();
  const auto qkv_shift_start = std::chrono::steady_clock::now();
  // The fused QKV kernel does not use ordinary Linear's global output-column
  // layout.  Its compact ciphertexts are grouped by paired heads, and the Q,
  // K and V tails are already in the layouts consumed by the BOLT QK and
  // Softmax-V kernels.  The scale boundary is element-wise, so preserve that
  // layout exactly and truncate the complete packed share vector in place.
  FixArray qkv_ring_packed = packed_field_to_ring(
      qkv_field, qkv_raw_scale, he_);
  FixArray qkv_scaled = mpc_.right_shift(
      qkv_ring_packed,
      qkv_raw_scale - static_cast<int>(metadata_.activation_scale));
  VitOperatorTrace qkv_shift_row;
  qkv_shift_row.phase = "online";
  qkv_shift_row.layer = layer;
  qkv_shift_row.op = "qkv_scale_truncate";
  qkv_shift_row.input_shape = qkv_shift_row.output_shape =
      shape(3 * physical, metadata_.model_dim);
  qkv_shift_row.input_packing = qkv_shift_row.output_packing =
      "bert_fused_q_k_v_compact_shares";
  qkv_shift_row.input_scale = qkv_raw_scale;
  qkv_shift_row.output_scale = metadata_.activation_scale;
  qkv_shift_row.ell = kEll;
  qkv_shift_row.backend = "bolt_mpc_batched_right_shift_qkv_to_activation_scale";
  append_trace(trace, std::move(qkv_shift_row), qkv_shift_before_bytes,
               qkv_shift_before_rounds, elapsed_us(qkv_shift_start));

  const std::uint64_t qkv_ss_to_he_before_bytes = wire_bytes();
  const std::uint64_t qkv_ss_to_he_before_rounds = wire_rounds();
  const auto qkv_ss_to_he_start = std::chrono::steady_clock::now();
  qkv_cipher = signed_ring_to_he(qkv_scaled.data, qkv_scaled.size,
                                 qkv_scaled.ell, qkv_scaled.s, conversion_);
  append_conversion_trace(
      trace, "ss_to_he_qkv_scale_boundary", layer, 3 * physical,
      metadata_.model_dim, metadata_.activation_scale,
      "bert_fused_q_k_v_compact_shares_scale12",
      "bert_fused_q_k_v_compact_ciphertext_scale12",
      "bolt_conversion_ss_to_he_qkv_batch_compat",
      qkv_ss_to_he_before_bytes, qkv_ss_to_he_before_rounds,
      elapsed_us(qkv_ss_to_he_start), he_.parameters().plain_modulus);

  std::vector<seal::Ciphertext> value_cipher;
  if (party_ == ALICE) {
    value_cipher.resize(one_projection);
    for (std::size_t i = 0; i < one_projection; ++i) {
      he_.view().evaluator->copy_ciphertext(
          qkv_cipher[2 * one_projection + i], value_cipher[i]);
      // The QK kernel releases short-lived GPU residency after it finishes.
      // V remains live until Softmax×V, so materialize this retained copy now;
      // otherwise release_gpu_residency() would discard its device-only data.
      he_.view().evaluator->synchronize(value_cipher[i]);
    }
  }

  const std::size_t qk_ciphertexts =
      metadata_.heads * physical * physical / he_.slot_count();
  const std::uint64_t qk_before_bytes = wire_bytes();
  const std::uint64_t qk_before_rounds = wire_rounds();
  const auto qk_start = std::chrono::steady_clock::now();
  BoltOperationCounts qk_counts;
  std::vector<seal::Ciphertext> qk_cipher;
  if (party_ == ALICE) {
    const auto cross_masks =
        linear_.bert_generate_cross_packing_masks(physical);
    qk_cipher.resize(qk_ciphertexts);
    linear_.bert_qk_matmul_cross_compat(
        qkv_cipher, cross_masks, qkv_metadata, qk_cipher, &qk_counts);
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
  qk_row.input_packing = "bert_q_column_packed_x_k_row_packed";
  qk_row.output_packing = "bert_cross_packed_qk_ciphertext";
  qk_row.input_scale = qkv_scale;
  qk_row.output_scale = 2 * qkv_scale;
  qk_row.ell = kEll;
  qk_row.counts = qk_counts;
  qk_row.backend = "bolt_ct_ct_qk_cross_packed_bert_compat";
  append_trace(trace, std::move(qk_row), qk_before_bytes, qk_before_rounds,
               elapsed_us(qk_start));

  const std::uint64_t qk_conversion_before_bytes = wire_bytes();
  const std::uint64_t qk_conversion_before_rounds = wire_rounds();
  const auto qk_conversion_start = std::chrono::steady_clock::now();
  const std::vector<std::uint64_t> qk_field =
      conversion_.he_to_ss_packed_compat(qk_cipher, qk_ciphertexts, true);
  const std::vector<std::uint64_t> qk_cross = unpack_bert_qk_cross(
      qk_field, metadata_.heads, physical, he_.slot_count());
  FixArray qk_all = packed_field_to_ring(
      qk_cross, 2 * qkv_scale, he_);
  append_conversion_trace(
      trace, "he_to_ss_qk", layer, metadata_.heads * physical, physical,
      2 * qkv_scale, "bert_cross_packed_qk_ciphertext",
      "bert_qk_row_major_shares", "bolt_conversion_he_to_ss_qk_compat",
      qk_conversion_before_bytes, qk_conversion_before_rounds,
      elapsed_us(qk_conversion_start));

  const std::uint64_t soft_before_bytes = wire_bytes();
  const std::uint64_t soft_before_rounds = wire_rounds();
  const auto soft_start = std::chrono::steady_clock::now();
  std::vector<FixArray> scaled_rows;
  scaled_rows.reserve(metadata_.heads * logical);
  const std::int64_t coefficient = quantize(
      1.0 / std::sqrt(static_cast<double>(metadata_.head_dim)), kScale);
  for (std::size_t head = 0; head < metadata_.heads; ++head) {
    FixArray qk_head(party_, static_cast<int>(physical * physical), true,
                    kEll, 2 * qkv_scale);
    std::memcpy(qk_head.data,
                qk_all.data + head * physical * physical,
                physical * physical * sizeof(std::uint64_t));
    apply_public_attention_mask(qk_head, "attention_padding_mask", layer,
                                trace);
    FixArray qk_at_24 = mpc_.right_shift(
        qk_head, 2 * qkv_scale - 2 * metadata_.activation_scale);
    FixArray constant = fp_.fix->input(
        PUBLIC, qk_at_24.size, ring_encode(coefficient), true, kEll, kScale);
    FixArray wide = fp_.fix->mul(qk_at_24, constant, 49);
    FixArray scaled = fp_.fix->truncate_reduce(
        wide, 2 * metadata_.activation_scale);
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
  std::vector<FixArray> softmax_rows;
  softmax_rows = mpc_.softmax(scaled_rows);
  VitOperatorTrace soft_row;
  soft_row.phase = "online";
  soft_row.layer = layer;
  soft_row.op = "softmax";
  soft_row.input_shape = soft_row.output_shape =
      std::to_string(metadata_.heads * logical) + "x[" +
      std::to_string(logical) + "]";
  soft_row.input_packing = "bert_qk_cross_to_logical_rows_197";
  soft_row.output_packing = "bert_softmax_row_major_shares_before_diagonal_pack";
  soft_row.input_scale = soft_row.output_scale = metadata_.activation_scale;
  soft_row.ell = kEll;
  soft_row.backend = "bolt_mpc_softmax";
  append_trace(trace, std::move(soft_row), soft_before_bytes,
               soft_before_rounds, elapsed_us(soft_start));

  std::vector<std::uint64_t> softmax_full(
      metadata_.heads * physical * physical, 0);
  for (std::size_t head = 0; head < metadata_.heads; ++head)
    for (std::size_t row = 0; row < logical; ++row)
      std::memcpy(softmax_full.data() + head * physical * physical +
                      row * physical,
                  softmax_rows[head * logical + row].data,
                  logical * sizeof(std::uint64_t));
  const std::vector<std::uint64_t> softmax_packed = preprocess_bert_softmax(
      softmax_full, metadata_.heads, physical, he_.slot_count());
  const std::uint64_t soft_conversion_before_bytes = wire_bytes();
  const std::uint64_t soft_conversion_before_rounds = wire_rounds();
  const auto soft_conversion_start = std::chrono::steady_clock::now();
  const std::vector<seal::Ciphertext> encrypted_softmax =
      signed_ring_to_he(softmax_packed.data(), softmax_packed.size(), kEll,
                        metadata_.activation_scale, conversion_);
  append_conversion_trace(
      trace, "ss_to_he_softmax", layer, metadata_.heads * physical, physical,
      metadata_.activation_scale, "bert_softmax_diagonal_shares",
      "bert_softmax_diagonal_input_ciphertext",
      "bolt_conversion_ss_to_he_softmax_compat",
      soft_conversion_before_bytes, soft_conversion_before_rounds,
      elapsed_us(soft_conversion_start));

  const std::uint64_t diagonal_before_bytes = wire_bytes();
  const std::uint64_t diagonal_before_rounds = wire_rounds();
  const auto diagonal_start = std::chrono::steady_clock::now();
  std::vector<std::vector<std::vector<seal::Ciphertext>>> softmax_diagonal;
  BoltOperationCounts diagonal_counts;
  if (party_ == ALICE)
    softmax_diagonal = linear_.bert_pack_softmax_diagonals(
        encrypted_softmax, qkv_metadata, &diagonal_counts);
  VitOperatorTrace diagonal_row;
  diagonal_row.phase = "online";
  diagonal_row.layer = layer;
  diagonal_row.op = "softmax_diagonal_pack";
  diagonal_row.input_shape = diagonal_row.output_shape =
      std::to_string(metadata_.heads) + "x" + shape(physical, physical);
  diagonal_row.input_packing = "bert_softmax_diagonal_input_ciphertext";
  diagonal_row.output_packing = "bert_diagonal_packed_ct_ct_operands";
  diagonal_row.input_scale = diagonal_row.output_scale =
      metadata_.activation_scale;
  diagonal_row.ell = kEll;
  diagonal_row.counts = diagonal_counts;
  diagonal_row.backend = "bolt_ct_pt_softmax_diagonal_pack_bert_compat";
  append_trace(trace, std::move(diagonal_row), diagonal_before_bytes,
               diagonal_before_rounds, elapsed_us(diagonal_start));

  const std::uint64_t sv_before_bytes = wire_bytes();
  const std::uint64_t sv_before_rounds = wire_rounds();
  const auto sv_start = std::chrono::steady_clock::now();
  std::vector<seal::Ciphertext> sv_cipher;
  BoltOperationCounts sv_counts;
  const BoltBertPackingMetadata sv_metadata =
      bert_metadata(he_, physical, metadata_.model_dim, metadata_.head_dim,
                    1, 0, 2);
  const std::size_t sv_ciphertexts =
      metadata_.heads * physical * metadata_.head_dim / he_.slot_count();
  if (party_ == ALICE) {
    sv_cipher.resize(sv_ciphertexts);
    linear_.bert_softmax_v_matmul_compat(
        softmax_diagonal, value_cipher, sv_metadata, sv_cipher, &sv_counts);
  }
  VitOperatorTrace sv_row;
  sv_row.phase = "online";
  sv_row.layer = layer;
  sv_row.op = "softmax_v";
  sv_row.input_shape = std::to_string(metadata_.heads) + "x(" +
                       shape(physical, physical) + "x" +
                       shape(physical, metadata_.head_dim) + ")";
  sv_row.output_shape = shape(physical, metadata_.model_dim);
  sv_row.input_packing = "bert_diagonal_packed_softmax_x_column_packed_v";
  sv_row.output_packing = "bert_softmax_v_cross_ciphertext_column_heads";
  sv_row.input_scale = metadata_.activation_scale;
  const int softmax_v_raw_scale = metadata_.activation_scale + qkv_scale;
  sv_row.output_scale = softmax_v_raw_scale;
  sv_row.ell = kEll;
  sv_row.counts = sv_counts;
  sv_row.backend = "bolt_ct_ct_softmax_v_diagonal_column_bert_compat";
  append_trace(trace, std::move(sv_row), sv_before_bytes, sv_before_rounds,
               elapsed_us(sv_start));

  const std::uint64_t sv_conversion_before_bytes = wire_bytes();
  const std::uint64_t sv_conversion_before_rounds = wire_rounds();
  const auto sv_conversion_start = std::chrono::steady_clock::now();
  const std::vector<std::uint64_t> sv_field =
      conversion_.he_to_ss_packed_compat(sv_cipher, sv_ciphertexts, true);
  const std::vector<std::uint64_t> sv_column = unpack_bert_softmax_v(
      sv_field, metadata_.heads, physical, metadata_.head_dim,
      he_.slot_count());
  FixArray merged_raw = packed_field_to_ring(
      sv_column, softmax_v_raw_scale, he_);
  append_conversion_trace(
      trace, "he_to_ss_softmax_v", layer, physical, metadata_.model_dim,
      softmax_v_raw_scale,
      "bert_softmax_v_cross_ciphertext", "bert_head_column_shares",
      "bolt_conversion_he_to_ss_softmax_v_compat",
      sv_conversion_before_bytes, sv_conversion_before_rounds,
      elapsed_us(sv_conversion_start));
  FixArray merged_shares = mpc_.right_shift(
      merged_raw, softmax_v_raw_scale -
                         metadata_.activation_scale);

  return packed_linear(
      merged_shares, physical, metadata_.model_dim, metadata_.model_dim,
      metadata_.attention_stride, block.wo, block.bo,
      metadata_.activation_scale, "output_projection", layer, trace,
      "bolt_ct_pt_bsgs_output_projection_bert_compat", nullptr, true);
}

FixArray BoltVitProtocol::gelu_bolt(
    const FixArray &input, int layer,
    std::vector<VitOperatorTrace> &trace) {
  if (input.s != static_cast<int>(metadata_.activation_scale) ||
      input.ell != static_cast<int>(metadata_.ell))
    throw std::invalid_argument("BOLT GELU input metadata mismatch");
  const std::uint64_t before_bytes = wire_bytes();
  const std::uint64_t before_rounds = wire_rounds();
  const auto start = std::chrono::steady_clock::now();
  FixArray scale11 = mpc_.right_shift(
      input, metadata_.activation_scale - metadata_.gelu_input_scale);
  FixArray ell20 = mpc_.reduce(scale11, metadata_.gelu_ell);
  FixArray output = mpc_.gelu(ell20);
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
  row.backend = "bolt_mpc_gelu_approx_2_ell20_scale11_to_ell38_scale4";
  append_trace(trace, std::move(row), before_bytes, before_rounds,
               elapsed_us(start));
  return output;
}

VitProtocolOutput BoltVitProtocol::run(
    const VitModel &model, const std::vector<std::uint64_t> &image_share,
    std::vector<VitOperatorTrace> &trace) {
  if (image_share.size() != metadata_.channels * metadata_.image_size *
                                metadata_.image_size)
    throw std::invalid_argument("ViT image share shape mismatch");
  const std::uint64_t patch_before_bytes = wire_bytes();
  const std::uint64_t patch_before_rounds = wire_rounds();
  const auto patch_start = std::chrono::steady_clock::now();
  const std::size_t side = metadata_.image_size / metadata_.patch_size;
  FixArray patches(party_, metadata_.physical_tokens * metadata_.patch_dim,
                   true, metadata_.ell, metadata_.activation_scale);
  std::fill(patches.data, patches.data + patches.size, 0);
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
  patch_row.output_shape = shape(metadata_.physical_tokens, metadata_.patch_dim);
  patch_row.input_packing = "chw_shares";
  patch_row.output_packing = "py_px_c_ky_kx_shares_padded_to_256";
  patch_row.input_scale = patch_row.output_scale = metadata_.activation_scale;
  patch_row.backend = "local_public_permutation";
  append_trace(trace, std::move(patch_row), patch_before_bytes,
               patch_before_rounds, elapsed_us(patch_start));
  FixArray patch_tokens = packed_linear(
      patches, metadata_.physical_tokens, metadata_.patch_dim,
      metadata_.model_dim, metadata_.patch_stride, model.patch_weight,
      model.patch_bias, metadata_.activation_scale, "patch_projection", -1,
      trace);
  const std::uint64_t embed_before_bytes = wire_bytes();
  const std::uint64_t embed_before_rounds = wire_rounds();
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
  apply_public_token_mask(hidden, "embedding_padding_mask", -1, trace);
  VitOperatorTrace embed_row;
  embed_row.phase = "online";
  embed_row.op = "cls_position";
  embed_row.input_shape = shape(metadata_.physical_tokens, metadata_.model_dim);
  embed_row.output_shape = shape(metadata_.physical_tokens, metadata_.model_dim);
  embed_row.input_packing = "image_tokens";
  embed_row.output_packing = "cls_then_raster_patch_tokens";
  embed_row.input_scale = embed_row.output_scale = metadata_.activation_scale;
  embed_row.backend = "share_local_model_add";
  append_trace(trace, std::move(embed_row), embed_before_bytes,
               embed_before_rounds, elapsed_us(embed_start));

  for (std::size_t layer = 0; layer < metadata_.layers; ++layer) {
    std::cerr << "BOLT_VIT_PROGRESS role="
              << (party_ == ALICE ? "ALICE" : "BOB")
              << " layer=" << (layer + 1) << '/' << metadata_.layers
              << " status=start" << std::endl;
    const VitBlockModel &block = model.blocks.at(layer);
    FixArray normalized = layer_norm(hidden, block.ln1_weight,
                                     block.ln1_bias, "ln1", layer, trace);
    FixArray attn = attention(normalized, block, layer, trace);
    hidden = residual_add(hidden, attn, "attention_residual", layer, trace);
    normalized = layer_norm(hidden, block.ln2_weight, block.ln2_bias,
                            "ln2", layer, trace);
    FixArray fc1 = packed_linear(
        normalized, metadata_.physical_tokens, metadata_.model_dim,
        metadata_.ff_dim,
        metadata_.ffn_stride, block.ff1_weight, block.ff1_bias,
        metadata_.activation_scale, "fc1", layer, trace);
    FixArray activated = gelu_bolt(fc1, layer, trace);
    FixArray fc2 = packed_linear(
        activated, metadata_.physical_tokens, metadata_.ff_dim,
        metadata_.model_dim,
        metadata_.ffn_stride, block.ff2_weight, block.ff2_bias,
        metadata_.activation_scale, "fc2", layer, trace);
    hidden = residual_add(hidden, fc2, "ffn_residual", layer, trace);
    std::cerr << "BOLT_VIT_PROGRESS role="
              << (party_ == ALICE ? "ALICE" : "BOB")
              << " layer=" << (layer + 1) << '/' << metadata_.layers
              << " status=done" << std::endl;
  }

  hidden = layer_norm(hidden, model.final_ln_weight, model.final_ln_bias,
                      "final_ln", metadata_.layers, trace);
  const std::uint64_t cls_before_bytes = wire_bytes();
  const std::uint64_t cls_before_rounds = wire_rounds();
  const auto cls_start = std::chrono::steady_clock::now();
  FixArray cls(party_, metadata_.model_dim, true, metadata_.ell,
               metadata_.activation_scale);
  std::memcpy(cls.data, hidden.data,
              metadata_.model_dim * sizeof(std::uint64_t));
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
  std::size_t classifier_physical = 1;
  while (classifier_physical < metadata_.classes) classifier_physical <<= 1;
  FixArray classifier_input(
      party_, metadata_.physical_tokens * metadata_.model_dim, true,
      metadata_.ell, metadata_.activation_scale);
  for (std::size_t row = 0; row < metadata_.physical_tokens; ++row)
    std::memcpy(classifier_input.data + row * metadata_.model_dim, cls.data,
                metadata_.model_dim * sizeof(std::uint64_t));
  std::vector<double> classifier_weight;
  std::vector<double> classifier_bias;
  if (party_ == ALICE) {
    if (model.classifier_weight.size() !=
            metadata_.classes * metadata_.model_dim ||
        model.classifier_bias.size() != metadata_.classes)
      throw std::invalid_argument("ALICE classifier metadata mismatch");
    classifier_weight.assign(
        classifier_physical * metadata_.model_dim, 0.0);
    classifier_bias.assign(classifier_physical, 0.0);
    for (std::size_t row = 0; row < metadata_.classes; ++row) {
      std::copy(
          model.classifier_weight.begin() + row * metadata_.model_dim,
          model.classifier_weight.begin() + (row + 1) * metadata_.model_dim,
          classifier_weight.begin() + row * metadata_.model_dim);
      classifier_bias[row] = model.classifier_bias[row];
    }
  }
  FixArray classifier_output = packed_linear(
      classifier_input, metadata_.physical_tokens, metadata_.model_dim,
      classifier_physical, metadata_.classifier_stride, classifier_weight,
      classifier_bias, metadata_.activation_scale, "classification_head",
      metadata_.layers, trace,
      "bolt_ct_pt_bsgs_classifier_padded_1024_bert_compat", nullptr, false);
  FixArray logits(party_, static_cast<int>(metadata_.classes), true,
                  metadata_.ell, metadata_.activation_scale);
  std::memcpy(logits.data, classifier_output.data,
              metadata_.classes * sizeof(std::uint64_t));

  const std::uint64_t arg_before_bytes = wire_bytes();
  const std::uint64_t arg_before_rounds = wire_rounds();
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
    row.milliseconds = std::max(row.milliseconds, peer.milliseconds);
    // NetIO counters are send-only; the sum is one copy of each wire byte.
    row.bytes += peer.bytes;
    // Endpoint transition counters describe the same interaction.  max is
    // the critical-path logical-round approximation; summing is forbidden.
    row.logical_rounds = std::max(row.logical_rounds, peer.rounds);
  }
}
