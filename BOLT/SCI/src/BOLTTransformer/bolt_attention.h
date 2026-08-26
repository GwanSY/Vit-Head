#ifndef SCI_BOLT_TRANSFORMER_ATTENTION_H
#define SCI_BOLT_TRANSFORMER_ATTENTION_H

#include "BOLTTransformer/bolt_he.h"

namespace sci {
namespace bolt {

// HE-only layout transforms used between the shared BOLT linear kernels.
// These operations preserve compact packing and never decrypt or create a
// ciphertext per scalar.
class BoltAttentionPacking {
public:
  explicit BoltAttentionPacking(BoltHEView he) : he_(he) {
    he_.require_public_evaluation(false);
  }

  BoltCipherTensor slice_head(
      const BoltCipherTensor &input, std::size_t head_index,
      const BoltAttentionPackingMetadata &metadata,
      BoltOperationCounts *counts = nullptr);

  BoltCipherTensor transpose_tokens(
      const BoltCipherTensor &head,
      const BoltAttentionPackingMetadata &metadata,
      BoltOperationCounts *counts = nullptr);

  BoltCipherTensor merge_heads(
      const std::vector<BoltCipherTensor> &heads,
      const BoltAttentionPackingMetadata &metadata,
      BoltOperationCounts *counts = nullptr);

private:
  seal::Plaintext repeated_range_mask(std::size_t begin,
                                      std::size_t end,
                                      std::size_t stride) const;
  BoltTensorMetadata tensor_metadata(std::size_t rows, std::size_t cols,
                                     const BoltTensorMetadata &source) const;
  void count(std::uint64_t BoltOperationCounts::*member,
             BoltOperationCounts *counts, std::uint64_t amount = 1) const;

  BoltHEView he_;
};

} // namespace bolt
} // namespace sci

#endif
