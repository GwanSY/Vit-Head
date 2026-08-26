#include "BOLTTransformer/bolt_he.h"
#include "BOLTTransformer/bolt_vit_compat_conversion.h"

#include "FloatingPoint/fp-math.h"
#include "LinearHE/utils-HE.h"
#include "OT/ot_pack.h"
#include "utils/ArgMapping/ArgMapping.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sci;
using namespace sci::bolt;

extern std::uint64_t prime_mod;
std::int32_t bitlength = 37;
std::int32_t num_threads = 1;

namespace {

constexpr std::uint64_t kPlainModulus = 536903681ULL;
constexpr std::uint32_t kEll = 37;
constexpr std::uint64_t kRingModulus = 1ULL << kEll;
constexpr std::uint64_t kRingMask = kRingModulus - 1;
constexpr std::size_t kSlotCount = 8192;
constexpr std::size_t kCiphertextCount = 2;
constexpr int kMinimumNoiseBudget = 80;

std::uint64_t mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

std::vector<std::uint64_t> make_clear_values() {
  std::vector<std::uint64_t> result(kCiphertextCount * kSlotCount);
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = mix(i + 0x564954424c42ULL) & kRingMask;

  // Ring signed boundaries, BFV centered boundaries, and both sides of each
  // BatchEncoder half/target-ciphertext boundary are explicit fixtures.
  const std::array<std::pair<std::size_t, std::uint64_t>, 20> cases{{
      {0, 0},
      {1, 1},
      {2, (kRingModulus >> 1) - 1},
      {3, kRingModulus >> 1},
      {4, (kRingModulus >> 1) + 1},
      {5, kRingMask - 1},
      {6, kRingMask},
      {kSlotCount / 2 - 1, 123456789},
      {kSlotCount / 2, kRingModulus - 123456789},
      {kSlotCount / 2 + 1, (kPlainModulus - 1) / 2},
      {kSlotCount - 1, kRingModulus - (kPlainModulus - 1) / 2},
      {kSlotCount, 17},
      {kSlotCount + 1, kRingModulus - 17},
      {kSlotCount + 2, (kPlainModulus - 1) / 2 + 1},
      {kSlotCount + 3,
       kRingModulus - ((kPlainModulus - 1) / 2 + 1)},
      {kSlotCount + kSlotCount / 2 - 1, 0x123456789ULL & kRingMask},
      {kSlotCount + kSlotCount / 2, 0x7654321ULL},
      {kSlotCount + kSlotCount / 2 + 1,
       kRingModulus - 0x7654321ULL},
      {2 * kSlotCount - 2, kRingModulus >> 1},
      {2 * kSlotCount - 1, (kRingModulus >> 1) - 1},
  }};
  for (const auto &entry : cases) result[entry.first] = entry.second;
  return result;
}

std::uint64_t make_alice_share(std::uint64_t clear, std::size_t index) {
  switch (index % 4) {
  case 0:
    return 0;
  case 1:
    return kRingMask;
  case 2:
    return clear;
  default:
    return clear == kRingMask ? kRingMask - 1 : clear + 1;
  }
}

std::uint64_t signed_ring_to_prime(std::uint64_t value) {
  value &= kRingMask;
  if (value < (kRingModulus >> 1)) return value % kPlainModulus;
  const std::uint64_t magnitude = kRingModulus - value;
  const std::uint64_t residue = magnitude % kPlainModulus;
  return residue == 0 ? 0 : kPlainModulus - residue;
}

bool stats_are_exact(const BoltVitCompatConversionStats &stats) {
  return stats.slot_count == kSlotCount &&
         stats.half_row_slots == kSlotCount / 2 &&
         stats.input_elements == kCiphertextCount * kSlotCount &&
         stats.signed_half_row_ciphertexts == 2 * kCiphertextCount &&
         stats.output_ciphertexts == kCiphertextCount &&
         stats.public_merge.ciphertexts == kCiphertextCount &&
         stats.public_merge.multiply_plain == 2 * kCiphertextCount &&
         stats.public_merge.rotations == 0 &&
         stats.public_merge.evaluator_multiply == 0 &&
         stats.public_merge.relinearizations == 0 &&
         stats.public_merge.modulus_switches == 0 &&
         stats.public_merge.noise_floodings == 0;
}

} // namespace

int main(int argc, char **argv) {
  int party = 0;
  int port = 52610;
  std::string address = "127.0.0.1";
  ArgMapping args;
  args.arg("r", party, "Role: ALICE=1, BOB=2");
  args.arg("p", port, "TCP port");
  args.arg("ip", address, "ALICE address (BOB only)");
  args.parse(argc, argv);

  try {
    if (party != ALICE && party != BOB)
      throw std::invalid_argument("r must be ALICE=1 or BOB=2");
    prime_mod = kPlainModulus;
    IOPack io(party, port, address);
    BoltHE he(party, io.io, BoltHEParameters::vit_b16());
    OTPack ot(&io, party);
    FPMath fp(party, &io, &ot);

    const std::vector<std::uint64_t> clear = make_clear_values();
    FixArray local(party, static_cast<int>(clear.size()), true, kEll, 12);
    std::size_t expected_wraps = 0;
    std::size_t expected_negative = 0;
    for (std::size_t i = 0; i < clear.size(); ++i) {
      const std::uint64_t alice = make_alice_share(clear[i], i);
      if (alice > clear[i]) ++expected_wraps;
      if (clear[i] >= (kRingModulus >> 1)) ++expected_negative;
      local.data[i] = party == ALICE
                          ? alice
                          : (clear[i] - alice) & kRingMask;
    }
    const bool coverage = expected_wraps > 0 &&
                          expected_wraps < clear.size() &&
                          expected_negative > 0 &&
                          expected_negative < clear.size();

    BoltVitCompatConversion conversion(he.view(), fp);
    BoltVitCompatConversionStats stats;
    std::vector<seal::Ciphertext> converted =
        conversion.ss_to_he_arbitrary_slots_signed(local, &stats);
    const bool stats_pass = stats_are_exact(stats);
    const bool ownership_pass = party == ALICE
                                    ? converted.size() == kCiphertextCount
                                    : converted.empty();

    if (party == ALICE) {
      // This is the declared final output of a known-vector conversion unit
      // test.  Production conversion ownership remains ALICE-only.
      send_encrypted_vector(io.io, converted);
    } else {
      std::vector<seal::Ciphertext> received(kCiphertextCount);
      recv_encrypted_vector(he.view().context, io.io, received);
      if (received.size() != kCiphertextCount)
        throw std::runtime_error("arbitrary-layout conversion CT count mismatch");

      int minimum_noise = std::numeric_limits<int>::max();
      std::size_t mismatch_count = 0;
      std::size_t first_mismatch = clear.size();
      for (std::size_t ct = 0; ct < kCiphertextCount; ++ct) {
        minimum_noise = std::min(
            minimum_noise,
            he.view().decryptor->invariant_noise_budget(received[ct]));
        seal::Plaintext plaintext;
        std::vector<std::uint64_t> decoded;
        he.view().decryptor->decrypt(received[ct], plaintext);
        he.view().encoder->decode(plaintext, decoded);
        if (decoded.size() != kSlotCount)
          throw std::runtime_error("BatchEncoder returned the wrong slot count");
        for (std::size_t slot = 0; slot < kSlotCount; ++slot) {
          const std::size_t index = ct * kSlotCount + slot;
          const std::uint64_t expected = signed_ring_to_prime(clear[index]);
          if (decoded[slot] != expected) {
            if (first_mismatch == clear.size()) first_mismatch = index;
            ++mismatch_count;
          }
        }
      }

      const bool noise_pass = minimum_noise >= kMinimumNoiseBudget;
      const bool layout_pass = mismatch_count == 0;
      std::cout << "ARBITRARY_SS_TO_HE_WRAP_COVERAGE="
                << (coverage ? "PASS" : "FAIL")
                << " wraps=" << expected_wraps
                << " non_wraps=" << clear.size() - expected_wraps
                << " negative=" << expected_negative
                << " non_negative=" << clear.size() - expected_negative
                << '\n';
      std::cout << "ARBITRARY_SS_TO_HE_STATS_TEST="
                << (stats_pass ? "PASS" : "FAIL")
                << " signed_half_row_ct="
                << stats.signed_half_row_ciphertexts
                << " output_ct=" << stats.output_ciphertexts
                << " multiply_plain=" << stats.public_merge.multiply_plain
                << " rotations=" << stats.public_merge.rotations << '\n';
      std::cout << "ARBITRARY_SS_TO_HE_OWNERSHIP_TEST="
                << (ownership_pass ? "PASS" : "FAIL") << '\n';
      std::cout << "ARBITRARY_SS_TO_HE_NOISE_BUDGET=" << minimum_noise
                << " minimum_required=" << kMinimumNoiseBudget << '\n';
      std::cout << "ARBITRARY_SS_TO_HE_FULL_SLOT_LAYOUT_TEST="
                << (layout_pass ? "PASS" : "FAIL")
                << " checked_slots=" << clear.size()
                << " mismatch_count=" << mismatch_count;
      if (!layout_pass) std::cout << " first_mismatch=" << first_mismatch;
      std::cout << '\n';

      const bool pass = coverage && stats_pass && ownership_pass &&
                        noise_pass && layout_pass;
      std::cout << "ARBITRARY_SS_TO_HE_TEST="
                << (pass ? "PASS" : "FAIL") << std::endl;
      if (!pass) return 3;
    }

    if (!coverage || !stats_pass || !ownership_pass) return 4;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ARBITRARY_SS_TO_HE_TEST=FAIL role=" << party
              << " error=" << error.what() << std::endl;
    return 2;
  }
}
