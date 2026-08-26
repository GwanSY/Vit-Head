#ifndef SCI_BOLT_TRANSFORMER_LINEAR_H
#define SCI_BOLT_TRANSFORMER_LINEAR_H

#include "BOLTTransformer/bolt_he.h"

#include <utility>

namespace sci {
namespace bolt {

class BoltLinear {
public:
  explicit BoltLinear(BoltHEView he) : he_(he) {
    he_.require_public_evaluation(true);
  }

  BoltCipherTensor cipher_plain_matmul_bsgs(
      const BoltCipherTensor &input, const PlainWeightTensor &weight,
      const BoltMatMulMetadata &metadata, BoltOperationCounts *counts = nullptr);

  // Row-packed private affine used after the existing MPC LayerNorm core.
  // `multiplier` and `bias` are ALICE-private plaintext vectors, repeated in
  // every compact row; the operation never exposes either model parameters
  // or the normalized activation to BOB.
  BoltCipherTensor cipher_plain_elementwise_affine(
      const BoltCipherTensor &input,
      const std::vector<std::uint64_t> &multiplier,
      const std::vector<std::uint64_t> &bias,
      const BoltMatMulMetadata &metadata,
      BoltOperationCounts *counts = nullptr);

  BoltCipherTensor cipher_cipher_matmul_cross(
      const BoltCipherTensor &lhs, const BoltCipherTensor &rhs,
      const BoltMatMulMetadata &metadata, BoltOperationCounts *counts = nullptr);

  BoltCipherTensor softmax_v_matmul(
      const BoltCipherTensor &softmax, const BoltCipherTensor &value,
      const BoltMatMulMetadata &metadata, BoltOperationCounts *counts = nullptr);

  // Apply circuit-privacy hardening after all public fusion/packing steps.
  // Callers use this when hardening each partial tensor would leave
  // insufficient noise budget for the subsequent merge.
  void harden_after_fusion(BoltCipherTensor &tensor,
                           const BoltMatMulMetadata &metadata,
                           BoltOperationCounts *counts = nullptr);

  // Exact compatibility kernels used by the existing BOLT BERT layout.
  void bert_cipher_plain_matmul_bsgs_compat(
      const std::vector<seal::Ciphertext> &input,
      const std::vector<std::vector<seal::Plaintext>> &encoded_left,
      const std::vector<std::vector<seal::Plaintext>> &encoded_right,
      const std::vector<seal::Plaintext> &encoded_bias,
      const BoltBertPackingMetadata &metadata,
      std::vector<seal::Ciphertext> &result,
      BoltOperationCounts *counts = nullptr);

  void bert_qkv_cipher_plain_matmul_bsgs_compat(
      const std::vector<seal::Ciphertext> &input,
      const std::vector<std::pair<std::vector<std::vector<seal::Plaintext>>,
                                  std::vector<std::vector<seal::Plaintext>>>> &q,
      const std::vector<std::pair<std::vector<std::vector<seal::Plaintext>>,
                                  std::vector<std::vector<seal::Plaintext>>>> &k,
      const std::vector<std::pair<std::vector<std::vector<seal::Plaintext>>,
                                  std::vector<std::vector<seal::Plaintext>>>> &v,
      const std::vector<std::vector<seal::Plaintext>> &q_bias,
      const std::vector<std::vector<seal::Plaintext>> &k_bias,
      const std::vector<std::vector<seal::Plaintext>> &v_bias,
      const BoltBertPackingMetadata &metadata,
      std::vector<seal::Ciphertext> &result,
      BoltOperationCounts *counts = nullptr);

  void bert_qk_matmul_cross_compat(
      const std::vector<seal::Ciphertext> &qkv,
      const std::vector<seal::Plaintext> &cross_masks,
      const BoltBertPackingMetadata &metadata,
      std::vector<seal::Ciphertext> &result,
      BoltOperationCounts *counts = nullptr);

  void bert_softmax_v_matmul_compat(
      const std::vector<std::vector<std::vector<seal::Ciphertext>>> &softmax,
      const std::vector<seal::Ciphertext> &value,
      const BoltBertPackingMetadata &metadata,
      std::vector<seal::Ciphertext> &result,
      BoltOperationCounts *counts = nullptr);

private:
  BoltCipherTensor cipher_cipher_matmul_cross_impl(
      const BoltCipherTensor &lhs, const BoltCipherTensor &rhs,
      const BoltMatMulMetadata &metadata, BoltOperationCounts *counts);
  std::vector<std::uint64_t> repeated_slots(
      const std::vector<std::uint64_t> &logical,
      std::size_t packed_stride) const;
  void harden_outputs(BoltCipherTensor &tensor,
                      const BoltMatMulMetadata &metadata,
                      BoltOperationCounts *counts);
  std::pair<seal::Ciphertext, seal::Ciphertext>
  bert_rotation_pair(const seal::Ciphertext &input, std::size_t image_size,
                     std::size_t offset, BoltOperationCounts *counts);

  BoltHEView he_;
};

} // namespace bolt
} // namespace sci

#endif
