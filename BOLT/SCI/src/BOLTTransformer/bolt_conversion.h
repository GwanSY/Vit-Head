#ifndef SCI_BOLT_TRANSFORMER_CONVERSION_H
#define SCI_BOLT_TRANSFORMER_CONVERSION_H

#include "BOLTTransformer/bolt_he.h"

namespace sci {
namespace bolt {

class BoltConversion {
public:
  explicit BoltConversion(BoltHEView he) : he_(he) { he_.require_batching(); }

  BoltCipherTensor ss_to_he(const SecretSharedTensor &input,
                            const BoltTensorMetadata &metadata);

  SecretSharedTensor he_to_ss(const BoltCipherTensor &input,
                              const BoltTensorMetadata &metadata);

  // Existing BERT tensors occupy every BatchEncoder slot. These compatibility
  // entries preserve that exact wire layout while sharing the protocol code.
  std::vector<seal::Ciphertext>
  ss_to_he_packed_compat(const std::uint64_t *local_share,
                         std::size_t element_count,
                         std::uint32_t share_bit_width);

  // Signed Z2^ell shares first lifted by the existing SCI signed-extension
  // protocol to Z2^64.  This is the BOLT ring-to-field boundary used when the
  // MPC side may contain arbitrary (randomized) local shares.  The input is
  // already laid out exactly as the compatibility packed path expects.
  std::vector<seal::Ciphertext>
  ss_to_he_packed_signed_ring_compat(const std::uint64_t *extended_share,
                                     std::size_t element_count);

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

  BoltHEView he_;
};

} // namespace bolt
} // namespace sci

#endif
