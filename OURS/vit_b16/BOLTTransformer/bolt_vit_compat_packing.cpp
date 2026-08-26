#include "BOLTTransformer/bolt_vit_compat_packing.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace sci {
namespace bolt {
namespace {

std::size_t checked_product(std::initializer_list<std::size_t> factors) {
  std::size_t result = 1;
  for (const std::size_t factor : factors) {
    if (factor != 0 && result > std::numeric_limits<std::size_t>::max() / factor)
      throw std::overflow_error("ViT compatibility packing size overflow");
    result *= factor;
  }
  return result;
}

std::size_t log2_power_of_two(std::size_t value) {
  if (value == 0 || (value & (value - 1)) != 0)
    throw std::invalid_argument("ViT compatibility packing requires a power of two");
  std::size_t result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

} // namespace

BoltVitCompatPacking::BoltVitCompatPacking(std::size_t slot_count,
                                           std::size_t physical_tokens,
                                           std::size_t model_dim,
                                           std::size_t heads,
                                           std::size_t head_dim)
    : slot_count_(slot_count), tokens_(physical_tokens),
      model_dim_(model_dim), heads_(heads), head_dim_(head_dim) {
  validate();
}

void BoltVitCompatPacking::validate() const {
  if (slot_count_ == 0 || tokens_ == 0 || model_dim_ == 0 || heads_ == 0 ||
      head_dim_ == 0 || heads_ % 2 != 0 || heads_ * head_dim_ != model_dim_ ||
      slot_count_ % (2 * tokens_) != 0 ||
      checked_product({tokens_, model_dim_}) % slot_count_ != 0 ||
      checked_product({tokens_, head_dim_, 2}) % slot_count_ != 0 ||
      checked_product({tokens_, tokens_, 2}) % slot_count_ != 0 ||
      head_dim_ % columns_per_ciphertext() != 0) {
    throw std::invalid_argument("invalid ViT BERT-compat packing dimensions");
  }
  // The adapter is deliberately narrow.  Other runtime shapes must first get
  // their own layout round-trip and kernel-count regression.
  if (slot_count_ != 8192 || tokens_ != 256 || model_dim_ != 768 ||
      heads_ != 12 || head_dim_ != 64) {
    throw std::invalid_argument(
        "ViT BERT-compat adapter currently requires T256/D768/H12/Dh64/N8192");
  }
}

BoltBertPackingMetadata BoltVitCompatPacking::bert_metadata(
    std::uint32_t noise_flooding_bits,
    std::uint32_t modulus_switches) const {
  BoltBertPackingMetadata metadata;
  metadata.slot_count = slot_count_;
  metadata.image_size = tokens_;
  metadata.filter_height = model_dim_;
  metadata.filter_width = head_dim_;
  metadata.baby_step = 4;
  metadata.giant_steps = 4;
  metadata.attention_heads = heads_;
  metadata.paired_head_groups = paired_head_groups();
  metadata.noise_flooding_bits = noise_flooding_bits;
  metadata.modulus_switches = modulus_switches;
  metadata.validate(slot_count_);
  if (metadata.attention_heads != 2 * metadata.paired_head_groups ||
      metadata.baby_step * metadata.giant_steps != columns_per_ciphertext())
    throw std::logic_error("ViT compatibility metadata lost its exact layout contract");
  return metadata;
}

std::size_t BoltVitCompatPacking::columns_per_ciphertext() const {
  return slot_count_ / (2 * tokens_);
}

std::size_t BoltVitCompatPacking::qkv_input_ciphertexts() const {
  return tokens_ * model_dim_ / slot_count_;
}

std::size_t BoltVitCompatPacking::projection_ciphertexts() const {
  return tokens_ * head_dim_ * heads_ / slot_count_;
}

std::size_t BoltVitCompatPacking::fused_qkv_ciphertexts() const {
  return 3 * projection_ciphertexts();
}

std::size_t BoltVitCompatPacking::qk_ciphertexts() const {
  return heads_ * tokens_ * tokens_ / slot_count_;
}

std::size_t BoltVitCompatPacking::softmax_ciphertexts() const {
  return qk_ciphertexts();
}

std::size_t BoltVitCompatPacking::context_ciphertexts() const {
  return projection_ciphertexts();
}

std::vector<std::uint64_t> BoltVitCompatPacking::pack_token_major(
    const std::vector<std::uint64_t> &token_major) const {
  return pack_token_major(token_major, model_dim_);
}

std::vector<std::uint64_t> BoltVitCompatPacking::pack_token_major(
    const std::vector<std::uint64_t> &token_major,
    std::size_t feature_dim) const {
  if (feature_dim == 0 ||
      checked_product({tokens_, feature_dim}) % slot_count_ != 0 ||
      token_major.size() != checked_product({tokens_, feature_dim}))
    throw std::invalid_argument("ViT token-major input shape mismatch");
  std::vector<std::uint64_t> packed(token_major.size());
  for (std::size_t token = 0; token < tokens_; ++token)
    for (std::size_t feature = 0; feature < feature_dim; ++feature)
      packed[feature * tokens_ + token] =
          token_major[token * feature_dim + feature];
  return packed;
}

std::vector<std::uint64_t> BoltVitCompatPacking::unpack_token_major(
    const std::vector<std::uint64_t> &packed) const {
  return unpack_token_major(packed, model_dim_);
}

std::vector<std::uint64_t> BoltVitCompatPacking::unpack_token_major(
    const std::vector<std::uint64_t> &packed,
    std::size_t feature_dim) const {
  if (feature_dim == 0 ||
      checked_product({tokens_, feature_dim}) % slot_count_ != 0 ||
      packed.size() != checked_product({tokens_, feature_dim}))
    throw std::invalid_argument("ViT compact input ciphertext shape mismatch");
  std::vector<std::uint64_t> result(tokens_ * feature_dim);
  for (std::size_t token = 0; token < tokens_; ++token)
    for (std::size_t feature = 0; feature < feature_dim; ++feature)
      result[token * feature_dim + feature] =
          packed[feature * tokens_ + token];
  return result;
}

std::vector<std::uint64_t> BoltVitCompatPacking::pack_linear_output(
    const std::vector<std::uint64_t> &token_major,
    std::size_t feature_dim) const {
  if (feature_dim == 0 || feature_dim % (2 * columns_per_ciphertext()) != 0 ||
      token_major.size() != checked_product({tokens_, feature_dim}))
    throw std::invalid_argument("ViT compact linear-output shape mismatch");
  const std::size_t columns = columns_per_ciphertext();
  const std::size_t half = slot_count_ / 2;
  const std::size_t ciphertexts = feature_dim / (2 * columns);
  std::vector<std::uint64_t> packed(ciphertexts * slot_count_);
  for (std::size_t ciphertext = 0; ciphertext < ciphertexts; ++ciphertext)
    for (std::size_t local = 0; local < columns; ++local)
      for (std::size_t token = 0; token < tokens_; ++token) {
        const std::size_t low_feature = ciphertext * columns + local;
        const std::size_t high_feature =
            feature_dim / 2 + ciphertext * columns + local;
        const std::size_t base = ciphertext * slot_count_ + local * tokens_;
        packed[base + token] =
            token_major[token * feature_dim + low_feature];
        packed[base + half + token] =
            token_major[token * feature_dim + high_feature];
      }
  return packed;
}

std::vector<std::uint64_t> BoltVitCompatPacking::unpack_linear_output(
    const std::vector<std::uint64_t> &packed,
    std::size_t feature_dim) const {
  if (feature_dim == 0 || feature_dim % (2 * columns_per_ciphertext()) != 0 ||
      packed.size() != checked_product({tokens_, feature_dim}))
    throw std::invalid_argument("ViT compact linear-output shape mismatch");
  const std::size_t columns = columns_per_ciphertext();
  const std::size_t half = slot_count_ / 2;
  const std::size_t ciphertexts = feature_dim / (2 * columns);
  std::vector<std::uint64_t> token_major(tokens_ * feature_dim);
  for (std::size_t ciphertext = 0; ciphertext < ciphertexts; ++ciphertext)
    for (std::size_t local = 0; local < columns; ++local)
      for (std::size_t token = 0; token < tokens_; ++token) {
        const std::size_t low_feature = ciphertext * columns + local;
        const std::size_t high_feature =
            feature_dim / 2 + ciphertext * columns + local;
        const std::size_t base = ciphertext * slot_count_ + local * tokens_;
        token_major[token * feature_dim + low_feature] =
            packed[base + token];
        token_major[token * feature_dim + high_feature] =
            packed[base + half + token];
      }
  return token_major;
}

std::vector<std::uint64_t> BoltVitCompatPacking::pack_qk_cross(
    const std::vector<std::uint64_t> &head_query_key) const {
  if (head_query_key.size() != checked_product({heads_, tokens_, tokens_}))
    throw std::invalid_argument("ViT QK logical shape mismatch");
  const std::size_t per_group =
      tokens_ * tokens_ * 2 / slot_count_;
  const std::size_t columns = columns_per_ciphertext();
  const std::size_t half = slot_count_ / 2;
  std::vector<std::uint64_t> packed(qk_ciphertexts() * slot_count_);
  for (std::size_t group = 0; group < paired_head_groups(); ++group)
    for (std::size_t ciphertext = 0; ciphertext < per_group; ++ciphertext)
      for (std::size_t slot = 0; slot < slot_count_; ++slot) {
        const bool second = slot >= half;
        const std::size_t row_slot = second ? slot - half : slot;
        const std::size_t diagonal = row_slot / tokens_ + ciphertext * columns;
        const std::size_t query = row_slot % tokens_;
        const std::size_t key = (query + diagonal) % tokens_;
        const std::size_t head = 2 * group + static_cast<std::size_t>(second);
        packed[(group * per_group + ciphertext) * slot_count_ + slot] =
            head_query_key[(head * tokens_ + query) * tokens_ + key];
      }
  return packed;
}

std::vector<std::uint64_t> BoltVitCompatPacking::unpack_qk_cross(
    const std::vector<std::uint64_t> &packed) const {
  if (packed.size() != qk_ciphertexts() * slot_count_)
    throw std::invalid_argument("ViT QK cross ciphertext shape mismatch");
  const std::size_t per_group =
      tokens_ * tokens_ * 2 / slot_count_;
  const std::size_t columns = columns_per_ciphertext();
  const std::size_t half = slot_count_ / 2;
  std::vector<std::uint64_t> result(heads_ * tokens_ * tokens_);
  for (std::size_t group = 0; group < paired_head_groups(); ++group)
    for (std::size_t ciphertext = 0; ciphertext < per_group; ++ciphertext)
      for (std::size_t slot = 0; slot < slot_count_; ++slot) {
        const bool second = slot >= half;
        const std::size_t row_slot = second ? slot - half : slot;
        const std::size_t diagonal = row_slot / tokens_ + ciphertext * columns;
        const std::size_t query = row_slot % tokens_;
        const std::size_t key = (query + diagonal) % tokens_;
        const std::size_t head = 2 * group + static_cast<std::size_t>(second);
        result[(head * tokens_ + query) * tokens_ + key] =
            packed[(group * per_group + ciphertext) * slot_count_ + slot];
      }
  return result;
}

std::vector<std::uint64_t>
BoltVitCompatPacking::pack_softmax_diagonal_source(
    const std::vector<std::uint64_t> &head_query_key) const {
  if (head_query_key.size() != checked_product({heads_, tokens_, tokens_}))
    throw std::invalid_argument("ViT Softmax logical shape mismatch");
  const std::size_t diagonals_per_ciphertext = columns_per_ciphertext();
  const std::size_t ciphertexts_per_group = tokens_ / diagonals_per_ciphertext;
  const std::size_t half = slot_count_ / 2;
  std::vector<std::uint64_t> packed(softmax_ciphertexts() * slot_count_);
  for (std::size_t group = 0; group < paired_head_groups(); ++group)
    for (std::size_t diagonal = 0; diagonal < tokens_; ++diagonal) {
      const std::size_t ciphertext = diagonal / diagonals_per_ciphertext;
      const std::size_t local = diagonal % diagonals_per_ciphertext;
      for (std::size_t query = 0; query < tokens_; ++query) {
        const std::size_t key = (query + diagonal) % tokens_;
        const std::size_t base =
            (group * ciphertexts_per_group + ciphertext) * slot_count_;
        packed[base + local * tokens_ + query] =
            head_query_key[((2 * group) * tokens_ + query) * tokens_ + key];
        packed[base + half + local * tokens_ + query] =
            head_query_key[((2 * group + 1) * tokens_ + query) * tokens_ + key];
      }
    }
  return packed;
}

std::vector<std::uint64_t>
BoltVitCompatPacking::pack_logical_softmax_diagonal_source(
    const std::vector<std::uint64_t> &head_query_key,
    std::size_t logical_tokens) const {
  if (logical_tokens == 0 || logical_tokens > tokens_ ||
      head_query_key.size() !=
          checked_product({heads_, logical_tokens, logical_tokens}))
    throw std::invalid_argument("ViT logical Softmax shape mismatch");
  std::vector<std::uint64_t> physical(heads_ * tokens_ * tokens_, 0);
  for (std::size_t head = 0; head < heads_; ++head)
    for (std::size_t query = 0; query < logical_tokens; ++query)
      for (std::size_t key = 0; key < logical_tokens; ++key)
        physical[(head * tokens_ + query) * tokens_ + key] =
            head_query_key[(head * logical_tokens + query) * logical_tokens +
                           key];
  return pack_softmax_diagonal_source(physical);
}

std::vector<std::uint64_t>
BoltVitCompatPacking::unpack_softmax_diagonal_source(
    const std::vector<std::uint64_t> &packed) const {
  if (packed.size() != softmax_ciphertexts() * slot_count_)
    throw std::invalid_argument("ViT Softmax packed ciphertext shape mismatch");
  const std::size_t diagonals_per_ciphertext = columns_per_ciphertext();
  const std::size_t ciphertexts_per_group = tokens_ / diagonals_per_ciphertext;
  const std::size_t half = slot_count_ / 2;
  std::vector<std::uint64_t> result(heads_ * tokens_ * tokens_);
  for (std::size_t group = 0; group < paired_head_groups(); ++group)
    for (std::size_t diagonal = 0; diagonal < tokens_; ++diagonal) {
      const std::size_t ciphertext = diagonal / diagonals_per_ciphertext;
      const std::size_t local = diagonal % diagonals_per_ciphertext;
      for (std::size_t query = 0; query < tokens_; ++query) {
        const std::size_t key = (query + diagonal) % tokens_;
        const std::size_t base =
            (group * ciphertexts_per_group + ciphertext) * slot_count_;
        result[((2 * group) * tokens_ + query) * tokens_ + key] =
            packed[base + local * tokens_ + query];
        result[((2 * group + 1) * tokens_ + query) * tokens_ + key] =
            packed[base + half + local * tokens_ + query];
      }
    }
  return result;
}

std::vector<std::uint64_t> BoltVitCompatPacking::pack_context(
    const std::vector<std::uint64_t> &token_major) const {
  if (token_major.size() != checked_product({tokens_, model_dim_}))
    throw std::invalid_argument("ViT context logical shape mismatch");
  const std::size_t features_per_ciphertext = columns_per_ciphertext();
  const std::size_t ciphertexts_per_group = head_dim_ / features_per_ciphertext;
  const std::size_t half = slot_count_ / 2;
  std::vector<std::uint64_t> packed(context_ciphertexts() * slot_count_);
  for (std::size_t group = 0; group < paired_head_groups(); ++group)
    for (std::size_t ciphertext = 0; ciphertext < ciphertexts_per_group;
         ++ciphertext)
      for (std::size_t local_feature = 0;
           local_feature < features_per_ciphertext; ++local_feature)
        for (std::size_t token = 0; token < tokens_; ++token) {
          const std::size_t feature =
              ciphertext * features_per_ciphertext + local_feature;
          const std::size_t base =
              (group * ciphertexts_per_group + ciphertext) * slot_count_;
          packed[base + local_feature * tokens_ + token] =
              token_major[token * model_dim_ + 2 * group * head_dim_ + feature];
          packed[base + half + local_feature * tokens_ + token] =
              token_major[token * model_dim_ +
                          (2 * group + 1) * head_dim_ + feature];
        }
  return packed;
}

std::vector<std::uint64_t> BoltVitCompatPacking::unpack_context(
    const std::vector<std::uint64_t> &packed) const {
  if (packed.size() != context_ciphertexts() * slot_count_)
    throw std::invalid_argument("ViT context packed ciphertext shape mismatch");
  const std::size_t features_per_ciphertext = columns_per_ciphertext();
  const std::size_t ciphertexts_per_group = head_dim_ / features_per_ciphertext;
  const std::size_t half = slot_count_ / 2;
  std::vector<std::uint64_t> result(tokens_ * model_dim_);
  for (std::size_t group = 0; group < paired_head_groups(); ++group)
    for (std::size_t ciphertext = 0; ciphertext < ciphertexts_per_group;
         ++ciphertext)
      for (std::size_t local_feature = 0;
           local_feature < features_per_ciphertext; ++local_feature)
        for (std::size_t token = 0; token < tokens_; ++token) {
          const std::size_t feature =
              ciphertext * features_per_ciphertext + local_feature;
          const std::size_t base =
              (group * ciphertexts_per_group + ciphertext) * slot_count_;
          result[token * model_dim_ + 2 * group * head_dim_ + feature] =
              packed[base + local_feature * tokens_ + token];
          result[token * model_dim_ +
                 (2 * group + 1) * head_dim_ + feature] =
              packed[base + half + local_feature * tokens_ + token];
        }
  return result;
}

std::vector<std::uint64_t> BoltVitCompatPacking::qk_cross_mask_slots(
    std::size_t index) const {
  if (index >= 2 * tokens_)
    throw std::out_of_range("ViT QK cross mask index is out of range");
  const bool wrapped = index >= tokens_;
  const std::size_t diagonal = index % tokens_;
  const std::size_t half = slot_count_ / 2;
  const std::size_t start = (diagonal * tokens_) % half;
  std::vector<std::uint64_t> mask(slot_count_, 0);
  const std::size_t begin = wrapped ? tokens_ - diagonal : 0;
  const std::size_t end = wrapped ? tokens_ : tokens_ - diagonal;
  for (std::size_t token = begin; token < end; ++token) {
    mask[start + token] = 1;
    mask[half + start + token] = 1;
  }
  return mask;
}

std::vector<std::uint64_t>
BoltVitCompatPacking::softmax_rotation_mask_slots(
    bool wrapped_part, std::size_t diagonal) const {
  if (diagonal >= tokens_)
    throw std::out_of_range("ViT Softmax diagonal is out of range");
  const std::size_t half = slot_count_ / 2;
  const std::size_t start = (diagonal % columns_per_ciphertext()) * tokens_;
  std::vector<std::uint64_t> mask(slot_count_, 0);
  const std::size_t begin = wrapped_part ? tokens_ - diagonal : 0;
  const std::size_t end = wrapped_part ? tokens_ : tokens_ - diagonal;
  for (std::size_t token = begin; token < end; ++token) {
    mask[start + token] = 1;
    mask[half + start + token] = 1;
  }
  return mask;
}

BoltVitCompatPacking::ExpectedCounts
BoltVitCompatPacking::expected_counts() const {
  ExpectedCounts result;
  const std::size_t groups = paired_head_groups();
  const std::size_t packed_per_group =
      tokens_ * head_dim_ * 2 / slot_count_;
  const std::size_t input_ciphertexts = qkv_input_ciphertexts();
  const std::size_t baby = 4;
  const std::size_t giant = 4;
  result.fused_qkv.rotations =
      input_ciphertexts * (2 * baby - 1) +
      groups * packed_per_group * (giant - 1) * 3;
  result.fused_qkv.multiply_plain =
      groups * giant * input_ciphertexts * packed_per_group * baby * 6;
  result.fused_qkv.ciphertexts = fused_qkv_ciphertexts();

  const std::size_t products_per_group = 2 * tokens_;
  const std::size_t reduction_rotations =
      log2_power_of_two(columns_per_ciphertext());
  result.qk.evaluator_multiply =
      groups * packed_per_group * tokens_ * 2;
  result.qk.relinearizations = groups * products_per_group;
  result.qk.rotations =
      groups * packed_per_group * tokens_ * 2 +
      groups * products_per_group * reduction_rotations;
  result.qk.multiply_plain = groups * products_per_group;
  result.qk.ciphertexts = qk_ciphertexts();

  result.softmax_expand.multiply_plain = groups * tokens_ * 2;
  result.softmax_expand.rotations =
      groups * tokens_ * 2 * reduction_rotations;
  result.softmax_expand.ciphertexts = groups * tokens_ * 2;

  const std::size_t matrices_per_column =
      head_dim_ / columns_per_ciphertext();
  result.softmax_v.evaluator_multiply =
      groups * matrices_per_column * tokens_ * 2;
  result.softmax_v.rotations =
      groups * matrices_per_column * tokens_ * 2;
  result.softmax_v.relinearizations = context_ciphertexts();
  result.softmax_v.ciphertexts = context_ciphertexts();
  return result;
}

} // namespace bolt
} // namespace sci
