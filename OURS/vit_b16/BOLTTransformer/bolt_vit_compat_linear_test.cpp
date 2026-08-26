#include "BOLTTransformer/bolt_he.h"
#include "BOLTTransformer/bolt_vit_compat_conversion.h"
#include "BOLTTransformer/bolt_vit_compat_linear.h"
#include "BOLTTransformer/bolt_vit_compat_packing.h"

#include "FloatingPoint/fp-math.h"
#include "LinearHE/utils-HE.h"
#include "OT/ot_pack.h"
#include "utils/ArgMapping/ArgMapping.h"

#include <algorithm>
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
constexpr std::uint64_t kRing = 1ULL << kEll;
constexpr std::uint64_t kRingMask = kRing - 1;
constexpr std::size_t kTokens = 256;
constexpr std::size_t kDim = 32;

std::uint64_t prime_encode(std::int64_t value) {
  if (value >= 0) return static_cast<std::uint64_t>(value) % kPlainModulus;
  const auto magnitude = static_cast<std::uint64_t>(-value) % kPlainModulus;
  return magnitude == 0 ? 0 : kPlainModulus - magnitude;
}

std::uint64_t mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

} // namespace

int main(int argc, char **argv) {
  int party = 0;
  int port = 52620;
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
    const BoltVitCompatPacking packing;

    std::vector<std::int64_t> clear(kTokens * kDim);
    std::vector<std::uint64_t> packed(kTokens * kDim);
    for (std::size_t token = 0; token < kTokens; ++token)
      for (std::size_t feature = 0; feature < kDim; ++feature) {
        const std::int64_t value =
            static_cast<std::int64_t>((token * 5 + feature * 3) % 19) - 9;
        clear[token * kDim + feature] = value;
        packed[feature * kTokens + token] =
            static_cast<std::uint64_t>(value) & kRingMask;
      }

    FixArray local(party, static_cast<int>(packed.size()), true, kEll, 12);
    for (std::size_t i = 0; i < packed.size(); ++i) {
      const std::uint64_t alice = mix(i + 0x4c494e454152ULL) & kRingMask;
      local.data[i] = party == ALICE
                          ? alice
                          : (packed[i] - alice) & kRingMask;
    }
    BoltVitCompatConversion conversion(he.view(), fp);
    std::vector<seal::Ciphertext> encrypted =
        conversion.ss_to_he_arbitrary_slots_signed(local);

    BoltOperationCounts counts;
    std::vector<seal::Ciphertext> output;
    std::vector<std::int64_t> coefficients(kDim * kDim);
    std::vector<std::int64_t> biases(kDim);
    for (std::size_t row = 0; row < kDim; ++row) {
      biases[row] = static_cast<std::int64_t>(row % 7) - 3;
      for (std::size_t col = 0; col < kDim; ++col)
        coefficients[row * kDim + col] =
            static_cast<std::int64_t>((row * 11 + col * 7) % 9) - 4;
    }
    if (party == ALICE) {
      PlainWeightTensor weight;
      weight.input_cols = kDim;
      weight.output_cols = kDim;
      weight.values.resize(coefficients.size());
      weight.bias.resize(biases.size());
      std::transform(coefficients.begin(), coefficients.end(),
                     weight.values.begin(), prime_encode);
      std::transform(biases.begin(), biases.end(), weight.bias.begin(),
                     prime_encode);
      BoltVitCompatLinearAdapter adapter(he.view(), packing);
      const BoltVitCompatLinearPlan plan = adapter.prepare(weight);
      output = adapter.apply(encrypted, plan, &counts);
      send_encrypted_vector(io.io, output);
    } else {
      output.resize(1);
      recv_encrypted_vector(he.view().context, io.io, output);
      seal::Plaintext plaintext;
      std::vector<std::uint64_t> decoded;
      he.view().decryptor->decrypt(output.front(), plaintext);
      he.view().encoder->decode(plaintext, decoded);
      const auto token_major = packing.unpack_linear_output(decoded, kDim);
      std::size_t mismatches = 0;
      std::size_t first_mismatch = token_major.size();
      for (std::size_t token = 0; token < kTokens; ++token)
        for (std::size_t row = 0; row < kDim; ++row) {
          std::int64_t expected = biases[row];
          for (std::size_t col = 0; col < kDim; ++col)
            expected += coefficients[row * kDim + col] *
                        clear[token * kDim + col];
          const std::size_t index = token * kDim + row;
          if (token_major[index] != prime_encode(expected)) {
            if (first_mismatch == token_major.size()) first_mismatch = index;
            ++mismatches;
          }
        }
      const int noise =
          he.view().decryptor->invariant_noise_budget(output.front());
      const bool pass = mismatches == 0 && noise > 0;
      std::cout << "VIT_COMPACT_LINEAR_FULL_OUTPUT_TEST="
                << (pass ? "PASS" : "FAIL")
                << " checked=" << token_major.size()
                << " mismatches=" << mismatches
                << " noise_budget=" << noise;
      if (mismatches != 0) std::cout << " first_mismatch=" << first_mismatch;
      std::cout << '\n';
      if (!pass) return 3;
    }

    if (party == ALICE) {
      const bool counts_pass =
          counts.ciphertexts == 1 && counts.rotations == 10 &&
          counts.multiply_plain == 32 && counts.noise_floodings == 1 &&
          counts.modulus_switches == 2;
      std::cout << "VIT_COMPACT_LINEAR_COUNTS_TEST="
                << (counts_pass ? "PASS" : "FAIL")
                << " ciphertexts=" << counts.ciphertexts
                << " rotations=" << counts.rotations
                << " multiply_plain=" << counts.multiply_plain
                << " floodings=" << counts.noise_floodings
                << " mod_switches=" << counts.modulus_switches << '\n';
      if (!counts_pass) return 4;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "VIT_COMPACT_LINEAR_TEST=FAIL role=" << party
              << " error=" << error.what() << std::endl;
    return 2;
  }
}
