#ifndef BLB_VIT_B16_BOLT_VIT_COMPAT_PACKING_H
#define BLB_VIT_B16_BOLT_VIT_COMPAT_PACKING_H

#include "BOLTTransformer/bolt_metadata.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sci {
namespace bolt {

// Runtime description of the deployed BERT compact/cross layout specialized
// only by public ViT dimensions.  It does not implement a new HE kernel: the
// resulting metadata and buffers are consumed by BoltLinear's existing
// bert_*_compat entry points.
class BoltVitCompatPacking {
public:
  BoltVitCompatPacking(std::size_t slot_count = 8192,
                       std::size_t physical_tokens = 256,
                       std::size_t model_dim = 768,
                       std::size_t heads = 12,
                       std::size_t head_dim = 64);

  void validate() const;
  BoltBertPackingMetadata bert_metadata(
      std::uint32_t noise_flooding_bits = 0,
      std::uint32_t modulus_switches = 0) const;

  std::size_t slot_count() const { return slot_count_; }
  std::size_t tokens() const { return tokens_; }
  std::size_t model_dim() const { return model_dim_; }
  std::size_t heads() const { return heads_; }
  std::size_t head_dim() const { return head_dim_; }
  std::size_t paired_head_groups() const { return heads_ / 2; }
  std::size_t columns_per_ciphertext() const;
  std::size_t qkv_input_ciphertexts() const;
  std::size_t projection_ciphertexts() const;
  std::size_t fused_qkv_ciphertexts() const;
  std::size_t qk_ciphertexts() const;
  std::size_t softmax_ciphertexts() const;
  std::size_t context_ciphertexts() const;

  // [token, feature] <-> the BERT feature-major input layout.  Returned
  // vectors concatenate complete BatchEncoder ciphertext slot arrays.
  std::vector<std::uint64_t> pack_token_major(
      const std::vector<std::uint64_t> &token_major) const;
  std::vector<std::uint64_t> unpack_token_major(
      const std::vector<std::uint64_t> &packed) const;
  std::vector<std::uint64_t> pack_token_major(
      const std::vector<std::uint64_t> &token_major,
      std::size_t feature_dim) const;
  std::vector<std::uint64_t> unpack_token_major(
      const std::vector<std::uint64_t> &packed,
      std::size_t feature_dim) const;

  // The deployed BERT ct-pt BSGS kernel pairs the lower and upper halves of
  // the output feature dimension in the two BatchEncoder rows.  These maps
  // describe that kernel output without changing its packing algorithm.
  std::vector<std::uint64_t> pack_linear_output(
      const std::vector<std::uint64_t> &token_major,
      std::size_t feature_dim) const;
  std::vector<std::uint64_t> unpack_linear_output(
      const std::vector<std::uint64_t> &packed,
      std::size_t feature_dim) const;

  // Reference inverse/forward maps for the cross-packed QK ciphertexts.
  // Logical order is [head, query, key].
  std::vector<std::uint64_t> pack_qk_cross(
      const std::vector<std::uint64_t> &head_query_key) const;
  std::vector<std::uint64_t> unpack_qk_cross(
      const std::vector<std::uint64_t> &packed) const;

  // Reorder [head, query, key] Softmax shares into the 96-ciphertext BERT
  // diagonal source layout and invert it.  Expansion to [6][2][256]
  // ciphertext views is local HE work in BoltVitCompatAttentionAdapter.
  std::vector<std::uint64_t> pack_softmax_diagonal_source(
      const std::vector<std::uint64_t> &head_query_key) const;
  std::vector<std::uint64_t> pack_logical_softmax_diagonal_source(
      const std::vector<std::uint64_t> &head_query_key,
      std::size_t logical_tokens) const;
  std::vector<std::uint64_t> unpack_softmax_diagonal_source(
      const std::vector<std::uint64_t> &packed) const;

  // Softmax*V output uses four ciphertexts per paired-head group.  Logical
  // order is [token, model_feature], with contiguous 64-wide heads.
  std::vector<std::uint64_t> pack_context(
      const std::vector<std::uint64_t> &token_major) const;
  std::vector<std::uint64_t> unpack_context(
      const std::vector<std::uint64_t> &packed) const;

  std::vector<std::uint64_t> qk_cross_mask_slots(
      std::size_t index) const;
  std::vector<std::uint64_t> softmax_rotation_mask_slots(
      bool wrapped_part, std::size_t diagonal) const;

  struct ExpectedCounts {
    BoltOperationCounts fused_qkv;
    BoltOperationCounts qk;
    BoltOperationCounts softmax_expand;
    BoltOperationCounts softmax_v;
  };
  ExpectedCounts expected_counts() const;

private:
  std::size_t slot_count_;
  std::size_t tokens_;
  std::size_t model_dim_;
  std::size_t heads_;
  std::size_t head_dim_;
};

} // namespace bolt
} // namespace sci

#endif
