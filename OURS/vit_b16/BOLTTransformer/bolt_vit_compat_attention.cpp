#include "BOLTTransformer/bolt_vit_compat_attention.h"

#include "LinearHE/utils-HE.h"

#include <stdexcept>

namespace sci {
namespace bolt {

BoltVitCompatAttentionAdapter::BoltVitCompatAttentionAdapter(
    BoltHEView he, const BoltVitCompatPacking &packing,
    std::uint32_t noise_flooding_bits, std::uint32_t modulus_switches)
    : he_(he), linear_(he), packing_(packing),
      noise_flooding_bits_(noise_flooding_bits),
      modulus_switches_(modulus_switches) {
  he_.require_public_evaluation(true);
  packing_.validate();
  // A zero setting would silently remove BERT's circuit-privacy hardening.
  // Diagnostic experiments must use a separate non-production harness.
  if (noise_flooding_bits_ == 0 || modulus_switches_ == 0)
    throw std::invalid_argument(
        "ViT compatibility attention requires caller-side HE hardening");
}

std::vector<seal::Plaintext>
BoltVitCompatAttentionAdapter::encode_qk_masks() const {
  std::vector<seal::Plaintext> result(2 * packing_.tokens());
  for (std::size_t index = 0; index < result.size(); ++index)
    he_.encoder->encode(packing_.qk_cross_mask_slots(index), result[index]);
  return result;
}

void BoltVitCompatAttentionAdapter::harden(
    std::vector<seal::Ciphertext> &ciphertexts,
    BoltOperationCounts *counts) const {
  for (auto &ciphertext : ciphertexts) {
    auto context_data = he_.context->get_context_data(ciphertext.parms_id());
    if (context_data == nullptr)
      throw std::logic_error("ViT compatibility hardening lost BFV context");
    flood_ciphertext(ciphertext, context_data, noise_flooding_bits_);
    if (counts != nullptr) ++counts->noise_floodings;
    for (std::uint32_t level = 0; level < modulus_switches_; ++level) {
      context_data = he_.context->get_context_data(ciphertext.parms_id());
      if (context_data == nullptr || context_data->next_context_data() == nullptr)
        throw std::logic_error(
            "ViT compatibility hardening exceeds the BFV modulus chain");
      he_.evaluator->mod_switch_to_next_inplace(ciphertext);
      if (counts != nullptr) ++counts->modulus_switches;
    }
  }
}

std::vector<seal::Ciphertext> BoltVitCompatAttentionAdapter::qk_cross(
    const std::vector<seal::Ciphertext> &fused_qkv,
    BoltOperationCounts *counts) {
  if (fused_qkv.size() != packing_.fused_qkv_ciphertexts())
    throw std::invalid_argument("ViT fused QKV compatibility shape mismatch");
  std::vector<seal::Ciphertext> result(packing_.qk_ciphertexts());
  const auto masks = encode_qk_masks();
  linear_.bert_qk_matmul_cross_compat(
      fused_qkv, masks, packing_.bert_metadata(), result, counts);
  harden(result, counts);
  return result;
}

BoltVitCompatAttentionAdapter::SoftmaxDiagonals
BoltVitCompatAttentionAdapter::expand_softmax_diagonals(
    const std::vector<seal::Ciphertext> &packed_softmax,
    BoltOperationCounts *counts) {
  if (packed_softmax.size() != packing_.softmax_ciphertexts())
    throw std::invalid_argument("ViT packed Softmax compatibility shape mismatch");
  const std::size_t groups = packing_.paired_head_groups();
  const std::size_t tokens = packing_.tokens();
  const std::size_t diagonals_per_ciphertext =
      packing_.columns_per_ciphertext();
  const std::size_t ciphertexts_per_group =
      tokens / diagonals_per_ciphertext;
  SoftmaxDiagonals result(
      groups, std::vector<std::vector<seal::Ciphertext>>(
                  2, std::vector<seal::Ciphertext>(tokens)));
  std::size_t rotation_steps = 0;
  for (std::size_t step = 1; step < diagonals_per_ciphertext; step <<= 1)
    ++rotation_steps;
#pragma omp parallel for collapse(2) schedule(static) num_threads(32)
  for (std::size_t group = 0; group < groups; ++group)
    for (std::size_t diagonal = 0; diagonal < tokens; ++diagonal) {
      const std::size_t source =
          group * ciphertexts_per_group +
          diagonal / diagonals_per_ciphertext;
      seal::Plaintext direct_mask, wrapped_mask;
      he_.encoder->encode(
          packing_.softmax_rotation_mask_slots(false, diagonal), direct_mask);
      he_.encoder->encode(
          packing_.softmax_rotation_mask_slots(true, diagonal), wrapped_mask);
      he_.evaluator->multiply_plain(
          packed_softmax[source], direct_mask, result[group][0][diagonal]);
      he_.evaluator->multiply_plain(
          packed_softmax[source], wrapped_mask, result[group][1][diagonal]);
      for (std::size_t step = 1; step < diagonals_per_ciphertext; step <<= 1) {
        seal::Ciphertext rotated;
        he_.rotate_rows_logical(
            result[group][0][diagonal],
            static_cast<int>(tokens * step), rotated);
        he_.evaluator->add_inplace(result[group][0][diagonal], rotated);
        he_.rotate_rows_logical(
            result[group][1][diagonal],
            static_cast<int>(tokens * step), rotated);
        he_.evaluator->add_inplace(result[group][1][diagonal], rotated);
      }
    }
  if (counts != nullptr) {
    counts->multiply_plain += groups * tokens * 2;
    counts->rotations += groups * tokens * rotation_steps * 2;
    counts->ciphertexts += groups * 2 * tokens;
  }
  return result;
}

std::vector<seal::Ciphertext> BoltVitCompatAttentionAdapter::softmax_v(
    const SoftmaxDiagonals &softmax,
    const std::vector<seal::Ciphertext> &value,
    BoltOperationCounts *counts) {
  if (softmax.size() != packing_.paired_head_groups() ||
      value.size() != packing_.projection_ciphertexts())
    throw std::invalid_argument("ViT Softmax-V compatibility shape mismatch");
  std::vector<seal::Ciphertext> result(packing_.context_ciphertexts());
  linear_.bert_softmax_v_matmul_compat(
      softmax, value, packing_.bert_metadata(), result, counts);
  harden(result, counts);
  return result;
}

} // namespace bolt
} // namespace sci
