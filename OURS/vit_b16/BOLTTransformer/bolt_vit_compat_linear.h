#ifndef BLB_VIT_B16_BOLT_VIT_COMPAT_LINEAR_H
#define BLB_VIT_B16_BOLT_VIT_COMPAT_LINEAR_H

#include "BOLTTransformer/bolt_linear.h"
#include "BOLTTransformer/bolt_vit_compat_packing.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sci {
namespace bolt {

struct BoltVitCompatLinearPlan {
  std::size_t input_dim = 0;
  std::size_t output_dim = 0;
  BoltBertPackingMetadata metadata;
  std::vector<std::vector<seal::Plaintext>> encoded_left;
  std::vector<std::vector<seal::Plaintext>> encoded_right;
  std::vector<seal::Plaintext> encoded_bias;
};

// ViT-facing preprocessing and dispatch for BOLT's existing compact BERT
// ct-pt BSGS kernel.  Only public runtime dimensions and plaintext model
// parameters are adapted; the HE arithmetic and hardening remain in
// BoltLinear::bert_cipher_plain_matmul_bsgs_compat.
class BoltVitCompatLinearAdapter {
public:
  BoltVitCompatLinearAdapter(BoltHEView he,
                             const BoltVitCompatPacking &packing);

  BoltVitCompatLinearPlan prepare(
      const PlainWeightTensor &weight,
      std::uint32_t noise_flooding_bits = 71,
      std::uint32_t modulus_switches = 2) const;

  std::vector<seal::Ciphertext> apply(
      const std::vector<seal::Ciphertext> &input,
      const BoltVitCompatLinearPlan &plan,
      BoltOperationCounts *counts = nullptr);

  // Private checkpoint LayerNorm affine over the standard compact
  // feature-major layout.  Gamma/beta remain ALICE-only plaintexts.
  std::vector<seal::Ciphertext> elementwise_affine(
      const std::vector<seal::Ciphertext> &input,
      const std::vector<std::uint64_t> &multiplier,
      const std::vector<std::uint64_t> &bias,
      std::uint32_t noise_flooding_bits = 71,
      std::uint32_t modulus_switches = 2,
      BoltOperationCounts *counts = nullptr);

private:
  BoltBertPackingMetadata linear_metadata(
      std::size_t input_dim, std::size_t output_dim,
      std::uint32_t noise_flooding_bits,
      std::uint32_t modulus_switches) const;
  std::vector<std::vector<seal::Plaintext>> encode_diagonals(
      const PlainWeightTensor &weight, bool swapped) const;
  std::vector<seal::Plaintext> encode_bias(
      const PlainWeightTensor &weight) const;
  void harden(std::vector<seal::Ciphertext> &ciphertexts,
              std::uint32_t noise_flooding_bits,
              std::uint32_t modulus_switches,
              BoltOperationCounts *counts) const;

  BoltHEView he_;
  BoltLinear linear_;
  BoltVitCompatPacking packing_;
};

} // namespace bolt
} // namespace sci

#endif
