#include "BOLTTransformer/bolt_vit_compat_conversion.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace sci {
namespace bolt {

BoltVitCompatConversion::BoltVitCompatConversion(BoltHEView he, FPMath &fp)
    : he_(he), fp_(fp), conversion_(he) {
  he_.require_batching();
}

std::vector<seal::Ciphertext>
BoltVitCompatConversion::ss_to_he_arbitrary_slots_signed(
    const FixArray &packed_local_shares,
    BoltVitCompatConversionStats *stats) {
  he_.require_batching();
  const std::size_t slot_count = he_.encoder->slot_count();
  const std::size_t half_slots = slot_count / 2;
  if (slot_count == 0 || slot_count % 2 != 0 ||
      packed_local_shares.data == nullptr || packed_local_shares.size <= 0 ||
      static_cast<std::size_t>(packed_local_shares.size) % slot_count != 0 ||
      packed_local_shares.party != he_.party ||
      packed_local_shares.party == PUBLIC ||
      !packed_local_shares.signed_ || packed_local_shares.ell <= 0 ||
      packed_local_shares.ell >= 64 || packed_local_shares.s < 0) {
    throw std::invalid_argument(
        "ViT arbitrary-layout signed SS-to-HE input is invalid");
  }

  const std::size_t output_count =
      static_cast<std::size_t>(packed_local_shares.size) / slot_count;
  const std::size_t half_row_count = 2 * output_count;
  if (half_row_count >
      static_cast<std::size_t>(std::numeric_limits<int>::max()) / half_slots) {
    throw std::overflow_error(
        "ViT arbitrary-layout signed SS-to-HE input is too large");
  }

  // Each source half becomes one logical compact row.  The underlying
  // conversion repeats that row in both BatchEncoder halves, preserving its
  // existing protocol and wire format.
  FixArray half_rows(packed_local_shares.party, packed_local_shares.size,
                     true, packed_local_shares.ell,
                     packed_local_shares.s);
  for (std::size_t ct = 0; ct < output_count; ++ct) {
    const std::size_t source = ct * slot_count;
    const std::size_t destination = 2 * ct * half_slots;
    std::memcpy(half_rows.data + destination,
                packed_local_shares.data + source,
                half_slots * sizeof(std::uint64_t));
    std::memcpy(half_rows.data + destination + half_slots,
                packed_local_shares.data + source + half_slots,
                half_slots * sizeof(std::uint64_t));
  }

  BoltTensorMetadata metadata;
  metadata.rows = half_row_count;
  metadata.cols = half_slots;
  metadata.packed_stride = half_slots;
  metadata.share_bit_width =
      static_cast<std::uint32_t>(packed_local_shares.ell);
  metadata.scale = static_cast<std::uint32_t>(packed_local_shares.s);
  metadata.packing = BoltPackingKind::kCompactRepeatedRows;

  SecretSharedTensor shares;
  shares.metadata = metadata;
  shares.local_share.assign(half_rows.data,
                            half_rows.data + half_rows.size);
  BoltCipherTensor repeated = conversion_.ss_to_he_signed_ring(
      shares, metadata, *fp_.fix);

  BoltVitCompatConversionStats local_stats;
  local_stats.slot_count = slot_count;
  local_stats.half_row_slots = half_slots;
  local_stats.input_elements = packed_local_shares.size;
  local_stats.signed_half_row_ciphertexts = half_row_count;
  local_stats.output_ciphertexts = output_count;
  local_stats.public_merge.ciphertexts = output_count;
  local_stats.public_merge.multiply_plain = 2 * output_count;
  if (stats != nullptr) *stats = local_stats;

  if (he_.party == BOB) {
    if (!repeated.ciphertexts.empty()) {
      throw std::logic_error(
          "BOB retained signed SS-to-HE conversion ciphertexts");
    }
    return {};
  }

  he_.require_public_evaluation(false);
  if (repeated.ciphertexts.size() != half_row_count) {
    throw std::runtime_error(
        "signed SS-to-HE conversion returned the wrong half-row count");
  }

  std::vector<std::uint64_t> left_mask(slot_count, 0);
  std::vector<std::uint64_t> right_mask(slot_count, 0);
  std::fill(left_mask.begin(), left_mask.begin() + half_slots, 1);
  std::fill(right_mask.begin() + half_slots, right_mask.end(), 1);
  seal::Plaintext left_plain;
  seal::Plaintext right_plain;
  he_.encoder->encode(left_mask, left_plain);
  he_.encoder->encode(right_mask, right_plain);

  std::vector<seal::Ciphertext> result(output_count);
  for (std::size_t ct = 0; ct < output_count; ++ct) {
    seal::Ciphertext left;
    seal::Ciphertext right;
    he_.evaluator->multiply_plain(repeated.ciphertexts[2 * ct], left_plain,
                                  left);
    he_.evaluator->multiply_plain(repeated.ciphertexts[2 * ct + 1],
                                  right_plain, right);
    he_.evaluator->add(left, right, result[ct]);
  }
  return result;
}

} // namespace bolt
} // namespace sci
