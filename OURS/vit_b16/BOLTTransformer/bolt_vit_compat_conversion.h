#ifndef BLB_VIT_B16_BOLT_VIT_COMPAT_CONVERSION_H
#define BLB_VIT_B16_BOLT_VIT_COMPAT_CONVERSION_H

#include "BOLTTransformer/bolt_conversion.h"

#include "FloatingPoint/fp-math.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sci {
namespace bolt {

// Public accounting for the layout-only merge performed after the existing
// signed-ring SS-to-HE protocol.  The signed half-row ciphertexts are the
// ciphertexts sent by the unchanged conversion; two are merged into every
// arbitrary-layout output ciphertext.
struct BoltVitCompatConversionStats {
  std::size_t slot_count = 0;
  std::size_t half_row_slots = 0;
  std::size_t input_elements = 0;
  std::size_t signed_half_row_ciphertexts = 0;
  std::size_t output_ciphertexts = 0;
  BoltOperationCounts public_merge;
};

// Adapts arbitrary complete BatchEncoder slot layouts to the repeated-row
// layout accepted by BoltConversion::ss_to_he_signed_ring.  It does not
// define a new conversion protocol: the underlying conversion reuses the
// existing BERT/BOLT FixOp::extend ring-to-field path, and the resulting
// repeated half rows are joined with fixed public plaintext masks.
//
// Input order is [ciphertext][slot], with all BatchEncoder slots present.
// ALICE receives one ciphertext per slot_count values; BOB receives an empty
// vector, matching the ownership convention of the underlying conversion.
class BoltVitCompatConversion {
public:
  BoltVitCompatConversion(BoltHEView he, FPMath &fp);

  std::vector<seal::Ciphertext> ss_to_he_arbitrary_slots_signed(
      const FixArray &packed_local_shares,
      BoltVitCompatConversionStats *stats = nullptr);

private:
  BoltHEView he_;
  FPMath &fp_;
  BoltConversion conversion_;
};

} // namespace bolt
} // namespace sci

#endif
