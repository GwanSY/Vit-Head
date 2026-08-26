#ifndef SCI_BOLT_TRANSFORMER_CONVERSION_H
#define SCI_BOLT_TRANSFORMER_CONVERSION_H

#include "BOLTTransformer/bolt_he.h"

class FixOp;

namespace sci {
namespace bolt {

class BoltConversion {
public:
  explicit BoltConversion(BoltHEView he) : he_(he) { he_.require_batching(); }

  BoltCipherTensor ss_to_he(const SecretSharedTensor &input,
                            const BoltTensorMetadata &metadata);

  // BERT/BOLT-compatible statistically correct conversion for arbitrary
  // signed Z2^ell additive shares.  The implementation reuses FixOp's
  // existing signed value-extension protocol to lift the representative into
  // Z2^64 before reducing each share modulo the BFV plaintext prime.  No
  // Boolean-to-prime-field conversion is introduced.
  //
  // For a signed integer x, the failure probability is approximately
  // |x|/2^64 (and at most 2^(ell-1)/2^64), matching the 64-bit-capped
  // ring_to_field path used by BERT.  Neither the value nor an intermediate
  // sign/wrap bit is opened.
  BoltCipherTensor ss_to_he_signed_ring(
      const SecretSharedTensor &input, const BoltTensorMetadata &metadata,
      FixOp &fix);

  SecretSharedTensor he_to_ss(const BoltCipherTensor &input,
                              const BoltTensorMetadata &metadata);

  // Existing BERT tensors occupy every BatchEncoder slot. These compatibility
  // entries preserve that exact wire layout while sharing the protocol code.
  std::vector<seal::Ciphertext>
  ss_to_he_packed_compat(const std::uint64_t *local_share,
                         std::size_t element_count,
                         std::uint32_t share_bit_width);

  std::vector<std::uint64_t>
  he_to_ss_packed_compat(const std::vector<seal::Ciphertext> &input,
                         std::size_t ciphertext_count,
                         bool add_plain_modulus_half);

private:
  std::uint64_t share_to_plain(std::uint64_t value,
                               std::uint32_t bit_width) const;
  std::vector<std::uint64_t>
  pack_repeated_row(const std::uint64_t *row, std::size_t cols,
                    const BoltTensorMetadata &metadata) const;
  std::vector<std::uint64_t>
  pack_repeated_unsigned_row(const std::uint64_t *row, std::size_t cols,
                             const BoltTensorMetadata &metadata) const;
  BoltHEView he_;
};

} // namespace bolt
} // namespace sci

#endif
