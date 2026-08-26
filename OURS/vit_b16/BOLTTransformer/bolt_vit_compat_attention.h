#ifndef BLB_VIT_B16_BOLT_VIT_COMPAT_ATTENTION_H
#define BLB_VIT_B16_BOLT_VIT_COMPAT_ATTENTION_H

#include "BOLTTransformer/bolt_linear.h"
#include "BOLTTransformer/bolt_vit_compat_packing.h"

#include <cstdint>
#include <vector>

namespace sci {
namespace bolt {

// Thin ViT adapter around the already deployed BERT QK and Softmax*V HE
// kernels.  The adapter only creates/consumes their established packing and
// applies the same caller-side flooding/modulus switching used by BERT before
// HE-to-share conversion.  It does not decrypt or reconstruct an intermediate.
class BoltVitCompatAttentionAdapter {
public:
  using SoftmaxDiagonals =
      std::vector<std::vector<std::vector<seal::Ciphertext>>>;

  BoltVitCompatAttentionAdapter(
      BoltHEView he, const BoltVitCompatPacking &packing,
      std::uint32_t noise_flooding_bits = 71,
      std::uint32_t modulus_switches = 2);

  std::vector<seal::Ciphertext> qk_cross(
      const std::vector<seal::Ciphertext> &fused_qkv,
      BoltOperationCounts *counts = nullptr);

  SoftmaxDiagonals expand_softmax_diagonals(
      const std::vector<seal::Ciphertext> &packed_softmax,
      BoltOperationCounts *counts = nullptr);

  std::vector<seal::Ciphertext> softmax_v(
      const SoftmaxDiagonals &softmax,
      const std::vector<seal::Ciphertext> &value,
      BoltOperationCounts *counts = nullptr);

private:
  std::vector<seal::Plaintext> encode_qk_masks() const;
  void harden(std::vector<seal::Ciphertext> &ciphertexts,
              BoltOperationCounts *counts) const;

  BoltHEView he_;
  BoltLinear linear_;
  BoltVitCompatPacking packing_;
  std::uint32_t noise_flooding_bits_;
  std::uint32_t modulus_switches_;
};

} // namespace bolt
} // namespace sci

#endif
