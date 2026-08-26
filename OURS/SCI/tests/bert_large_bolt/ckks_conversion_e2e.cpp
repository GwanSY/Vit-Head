#include "ckks_bert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// SCI-LinearHE selects its process-global arithmetic prime from this symbol.
// Keep it consistent with the formal BERT fixed-point ring used below.
int bitlength = 37;

namespace {

int party = 0;
int port = 38200;
int seed_count = 3;
int first_seed = 20260810;
// Regression bound for the degree-8192, root-scale-20 FFT at s=12 after the
// secure working-scale lift. 256 raw units equal 0.0625 real; this is not a
// paper accuracy/MSE threshold.
std::uint64_t tolerance = 256;
std::string address = "127.0.0.1";

constexpr std::size_t kDegree = 8192;
constexpr int kEll = 37;
constexpr int kFixedScale = 12;
constexpr int kLiftedScale = 40;
constexpr int kRawInputEll = 30;
constexpr int kRawInputScale = 5;
constexpr int kRawInputCkksScale = 90;
constexpr std::uint64_t kRawInputTolerance = 4;

std::uint64_t ring_mask(int bits) {
  return bits == 64 ? ~std::uint64_t{0}
                    : ((std::uint64_t{1} << bits) - 1);
}

std::int64_t signed_ring(std::uint64_t value, int bits) {
  const std::uint64_t mask = ring_mask(bits);
  value &= mask;
  const std::uint64_t sign = std::uint64_t{1} << (bits - 1);
  if ((value & sign) == 0) {
    return static_cast<std::int64_t>(value);
  }
  return static_cast<std::int64_t>(value) -
         static_cast<std::int64_t>(std::uint64_t{1} << bits);
}

std::vector<std::uint64_t> public_test_vector(
    std::size_t slots, int seed, int ell = kEll,
    int scale = kFixedScale) {
  std::mt19937_64 generator(static_cast<std::uint64_t>(seed));
  std::uniform_int_distribution<std::int64_t> distribution(
      -(std::int64_t{8} << scale), (std::int64_t{8} << scale));
  std::vector<std::uint64_t> values(slots);
  const std::uint64_t mask = ring_mask(ell);
  for (std::size_t i = 0; i < slots; ++i) {
    values[i] = static_cast<std::uint64_t>(distribution(generator)) & mask;
  }
  const std::int64_t edge_values[] = {
      0, 1, -1, std::int64_t{1} << scale,
      -(std::int64_t{1} << scale),
      (std::int64_t{7} << scale) + 3,
      -((std::int64_t{7} << scale) + 3)};
  for (std::size_t i = 0;
       i < std::size(edge_values) && i < values.size(); ++i) {
    values[i] = static_cast<std::uint64_t>(edge_values[i]) & mask;
  }
  return values;
}

std::vector<std::uint64_t> additive_share(
    const std::vector<std::uint64_t> &values, int seed, int ell = kEll) {
  std::mt19937_64 generator(
      static_cast<std::uint64_t>(seed) ^ 0x53484152455f524eULL);
  const std::uint64_t mask = ring_mask(ell);
  std::vector<std::uint64_t> share(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::uint64_t random = generator() & mask;
    share[i] = party == sci::ALICE ? random
                                   : (values[i] - random) & mask;
  }
  return share;
}

struct RingError {
  std::uint64_t max = 0;
  long double sum = 0.0L;
  long double squared_sum = 0.0L;
};

RingError ring_error(const std::vector<std::uint64_t> &actual,
                     const std::vector<std::uint64_t> &expected,
                     int bits) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("diagnostic vector shape mismatch");
  }
  const std::uint64_t mask = ring_mask(bits);
  RingError result;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const std::int64_t delta =
        signed_ring((actual[i] - expected[i]) & mask, bits);
    const std::uint64_t magnitude = static_cast<std::uint64_t>(
        delta < 0 ? -delta : delta);
    result.max = std::max(result.max, magnitude);
    result.sum += static_cast<long double>(magnitude);
    result.squared_sum +=
        static_cast<long double>(magnitude) * magnitude;
  }
  return result;
}

void print_diagnostic(const char *stage, int seed, const RingError &error,
                      std::size_t size) {
  std::cout << "CKKS_CONVERSION_STAGE stage=" << stage << " seed=" << seed
            << " status=INFO acceptance=0"
            << " max_abs_error_raw=" << error.max
            << " mae_raw=" << static_cast<double>(error.sum / size)
            << " mse_raw="
            << static_cast<double>(error.squared_sum / size)
            << " max_abs_error_real="
            << std::ldexp(static_cast<double>(error.max), -kFixedScale)
            << " mae_real="
            << std::ldexp(static_cast<double>(error.sum / size),
                          -kFixedScale)
            << " mse_real="
            << std::ldexp(static_cast<double>(error.squared_sum / size),
                          -2 * kFixedScale)
            << std::endl;
}

// The fixture is public, so this regression may open the two Algorithm-2
// plaintext shares before encryption. This isolates fixed FFT share-linearity
// from ring->field and CKKS decoding without weakening the production path.
void diagnose_algorithm2(Bert_ckks &adapter, HE_CKKS &he,
                         const seal::Plaintext &local_plain,
                         const std::vector<std::uint64_t> &local_slots,
                         const std::vector<std::uint64_t> &expected,
                         int seed) {
  const std::size_t coeff_count = local_plain.coeff_count();
  std::vector<std::uint64_t> local_fft = he.ckks_encode(
      local_slots, static_cast<int>(he.num_slots), false);
  FixArray local_fix(party, static_cast<int>(local_slots.size()), true, kEll,
                     kFixedScale);
  std::copy(local_slots.begin(), local_slots.end(), local_fix.data);
  FixArray extended_fix =
      adapter.nonlinear.fpmath[0]->fix->extend(local_fix, 64);
  std::vector<std::uint64_t> extended_slots(
      extended_fix.data, extended_fix.data + extended_fix.size);
  std::vector<std::uint64_t> extended_fft = he.ckks_encode(
      extended_slots, static_cast<int>(he.num_slots), false);
  bool done = true;
  if (party == sci::ALICE) {
    adapter.io->send_data(local_plain.data(),
                          coeff_count * sizeof(std::uint64_t));
    adapter.io->send_data(local_fft.data(),
                          local_fft.size() * sizeof(std::uint64_t));
    adapter.io->send_data(extended_fft.data(),
                          extended_fft.size() * sizeof(std::uint64_t));
    adapter.io->recv_data(&done, sizeof(done));
    return;
  }

  std::vector<std::uint64_t> alice_plain(coeff_count);
  std::vector<std::uint64_t> alice_fft(local_fft.size());
  std::vector<std::uint64_t> alice_extended_fft(extended_fft.size());
  adapter.io->recv_data(alice_plain.data(),
                        alice_plain.size() * sizeof(std::uint64_t));
  adapter.io->recv_data(alice_fft.data(),
                        alice_fft.size() * sizeof(std::uint64_t));
  adapter.io->recv_data(alice_extended_fft.data(),
                        alice_extended_fft.size() * sizeof(std::uint64_t));

  seal::Plaintext combined = local_plain;
  auto context_data = he.context->get_context_data(combined.parms_id());
  if (!context_data) {
    throw std::runtime_error("diagnostic plaintext level is invalid");
  }
  const auto &moduli = context_data->parms().coeff_modulus();
  for (std::size_t limb = 0; limb < moduli.size(); ++limb) {
    const std::uint64_t modulus = moduli[limb].value();
    for (std::size_t coefficient = 0; coefficient < kDegree;
         ++coefficient) {
      const std::size_t index = limb * kDegree + coefficient;
      combined[index] = static_cast<std::uint64_t>(
          (static_cast<__uint128_t>(combined[index]) +
           alice_plain[index]) % modulus);
    }
  }

  const std::uint64_t mask = ring_mask(kEll);
  std::vector<std::uint64_t> fft_share_sum(local_fft.size());
  for (std::size_t i = 0; i < local_fft.size(); ++i) {
    fft_share_sum[i] = (local_fft[i] + alice_fft[i]) & mask;
  }
  std::vector<std::uint64_t> public_fft = he.ckks_encode(
      expected, static_cast<int>(he.num_slots), false);
  for (std::uint64_t &coefficient : public_fft) {
    coefficient &= mask;
  }
  print_diagnostic("legacy_local_algorithm2_fft_linearity", seed,
                   ring_error(fft_share_sum, public_fft, kEll),
                   public_fft.size());
  std::vector<std::uint64_t> extended_fft_share_sum(extended_fft.size());
  for (std::size_t i = 0; i < extended_fft.size(); ++i) {
    extended_fft_share_sum[i] =
        (extended_fft[i] + alice_extended_fft[i]) & mask;
  }
  print_diagnostic("legacy_signextend64_algorithm2_fft_linearity", seed,
                   ring_error(extended_fft_share_sum, public_fft, kEll),
                   public_fft.size());

  auto combined_coeff_complex = he.ckks_intt(combined);
  std::vector<std::uint64_t> combined_coeff(combined_coeff_complex.size());
  for (std::size_t i = 0; i < combined_coeff.size(); ++i) {
    combined_coeff[i] = combined_coeff_complex[i].real() & mask;
  }
  print_diagnostic("legacy_unscaled_algorithm2_rns_reconstruction", seed,
                   ring_error(combined_coeff, fft_share_sum, kEll),
                   combined_coeff.size());

  std::vector<std::uint64_t> custom_slots =
      he.ckks_decode(combined_coeff_complex, static_cast<int>(kDegree), false);
  for (std::uint64_t &slot : custom_slots) {
    slot &= mask;
  }
  print_diagnostic("legacy_unscaled_algorithm2_custom_decode", seed,
                   ring_error(custom_slots, expected, kEll),
                   custom_slots.size());

  std::vector<double> standard_slots;
  he.encoder->decode(combined, standard_slots);
  std::vector<std::uint64_t> standard_fixed(standard_slots.size());
  for (std::size_t i = 0; i < standard_slots.size(); ++i) {
    const double scaled = std::nearbyint(
        standard_slots[i] * std::ldexp(1.0, kFixedScale));
    if (!std::isfinite(scaled) ||
        std::abs(scaled) > static_cast<double>(
                               std::numeric_limits<std::int64_t>::max())) {
      throw std::runtime_error("standard CKKS decode exceeded int64 range");
    }
    standard_fixed[i] =
        static_cast<std::uint64_t>(static_cast<std::int64_t>(scaled)) & mask;
  }
  print_diagnostic("algorithm2_standard_decode", seed,
                   ring_error(standard_fixed, expected, kEll),
                   standard_fixed.size());
  adapter.io->send_data(&done, sizeof(done));
}

bool verify_open_test_vector(Bert_ckks &adapter,
                             const std::vector<std::uint64_t> &local,
                             const std::vector<std::uint64_t> &expected,
                             int seed, const char *stage, int ell = kEll,
                             int scale = kFixedScale,
                             std::uint64_t stage_tolerance = tolerance) {
  if (local.size() != expected.size()) {
    throw std::runtime_error("conversion output slot count mismatch");
  }
  bool pass = true;
  std::uint64_t max_abs_error = 0;
  long double absolute_error_sum = 0.0L;
  long double squared_error_sum = 0.0L;
  std::size_t first_mismatch = local.size();
  if (party == sci::ALICE) {
    adapter.io->send_data(local.data(),
                          local.size() * sizeof(std::uint64_t));
    adapter.io->recv_data(&pass, sizeof(pass));
  } else {
    std::vector<std::uint64_t> alice(local.size());
    adapter.io->recv_data(alice.data(),
                          alice.size() * sizeof(std::uint64_t));
    const std::uint64_t mask = ring_mask(ell);
    for (std::size_t i = 0; i < local.size(); ++i) {
      const std::uint64_t actual = (alice[i] + local[i]) & mask;
      const std::uint64_t error_word = (actual - expected[i]) & mask;
      const std::int64_t signed_error = signed_ring(error_word, ell);
      const std::uint64_t abs_error = static_cast<std::uint64_t>(
          signed_error < 0 ? -signed_error : signed_error);
      max_abs_error = std::max(max_abs_error, abs_error);
      absolute_error_sum += static_cast<long double>(abs_error);
      squared_error_sum +=
          static_cast<long double>(abs_error) * abs_error;
      if (abs_error > stage_tolerance && first_mismatch == local.size()) {
        first_mismatch = i;
      }
    }
    pass = max_abs_error <= stage_tolerance;
    adapter.io->send_data(&pass, sizeof(pass));
    std::cout << "CKKS_CONVERSION_RESULT stage=" << stage
              << " seed=" << seed
              << " status=" << (pass ? "PASS" : "FAIL")
              << " slots=" << local.size()
              << " max_abs_error_raw=" << max_abs_error
              << " mae_raw="
              << static_cast<double>(absolute_error_sum / local.size())
              << " mse_raw="
              << static_cast<double>(squared_error_sum / local.size())
              << " max_abs_error_real="
              << std::ldexp(static_cast<double>(max_abs_error),
                            -scale)
              << " mae_real="
              << std::ldexp(
                     static_cast<double>(absolute_error_sum / local.size()),
                     -scale)
              << " mse_real="
              << std::ldexp(
                     static_cast<double>(squared_error_sum / local.size()),
                     -2 * scale)
              << " output_ell=" << ell
              << " output_scale=" << scale
              << " tolerance_raw=" << stage_tolerance
              << " first_mismatch="
              << (first_mismatch == local.size()
                      ? -1
                      : static_cast<long long>(first_mismatch))
              << std::endl;
  }
  return pass;
}

void diagnose_algorithm1_decode_linearity(
    Bert_ckks &adapter, HE_CKKS &he,
    const std::vector<std::uint64_t> &expected, int seed) {
  std::vector<double> real(expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    real[i] = std::ldexp(
        static_cast<double>(signed_ring(expected[i], kEll)), -kFixedScale);
  }
  seal::Plaintext encoded;
  he.encoder->encode(real, he.context->last_parms_id(),
                     std::ldexp(1.0, kFixedScale), encoded);
  auto public_coefficients = he.ckks_intt(encoded);
  std::vector<std::uint64_t> public_decoded = he.ckks_decode(
      public_coefficients, static_cast<int>(kDegree), false);

  std::mt19937_64 generator(
      static_cast<std::uint64_t>(seed) ^ 0x4445434f44455f52ULL);
  const std::uint64_t mask = ring_mask(kEll);
  std::vector<HE_CKKS::complex_t> local_coefficients(
      public_coefficients.size());
  for (std::size_t i = 0; i < public_coefficients.size(); ++i) {
    const std::uint64_t secret = public_coefficients[i].real() & mask;
    const std::uint64_t random = generator() & mask;
    const std::uint64_t share =
        party == sci::ALICE ? random : (secret - random) & mask;
    local_coefficients[i] = HE_CKKS::complex_t(share, 0);
  }
  std::vector<std::uint64_t> local_decoded = he.ckks_decode(
      local_coefficients, static_cast<int>(kDegree), false);
  bool done = true;
  if (party == sci::ALICE) {
    adapter.io->send_data(local_decoded.data(),
                          local_decoded.size() * sizeof(std::uint64_t));
    adapter.io->recv_data(&done, sizeof(done));
    return;
  }
  std::vector<std::uint64_t> alice_decoded(local_decoded.size());
  adapter.io->recv_data(alice_decoded.data(),
                        alice_decoded.size() * sizeof(std::uint64_t));
  for (std::size_t i = 0; i < local_decoded.size(); ++i) {
    local_decoded[i] = (local_decoded[i] + alice_decoded[i]) & mask;
    public_decoded[i] &= mask;
  }
  print_diagnostic("legacy_local_algorithm1_fft_linearity", seed,
                   ring_error(local_decoded, public_decoded, kEll),
                   public_decoded.size());
  adapter.io->send_data(&done, sizeof(done));
}

bool isolate_algorithm1(Bert_ckks &adapter, HE_CKKS &he,
                        const std::vector<std::uint64_t> &expected, int seed,
                        std::vector<std::uint64_t> &previous_mask) {
  std::vector<seal::Ciphertext> ciphertexts(1);
  std::vector<seal::Plaintext> shares(1);
  seal::Plaintext baseline_decryption;
  if (party == sci::BOB) {
    std::vector<double> real(expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
      real[i] = std::ldexp(
          static_cast<double>(signed_ring(expected[i], kEll)),
          -kFixedScale);
    }
    seal::Plaintext encoded;
    he.encoder->encode(real, he.context->first_parms_id(),
                       std::ldexp(1.0, kLiftedScale), encoded);
    he.encryptor->encrypt(encoded, ciphertexts[0]);
    send_encrypted_vector(adapter.io, ciphertexts);
    // Public-fixture diagnostic only: receive the evaluator's prepared copy
    // before masking, so mask cancellation is checked at the exact final
    // level/scale consumed by Algorithm 1.
    recv_encrypted_vector(he.context, adapter.io, ciphertexts);
    he.decryptor->decrypt(ciphertexts[0], baseline_decryption);
    recv_encrypted_vector(he.context, adapter.io, ciphertexts);
    he.decryptor->decrypt(ciphertexts[0], shares[0]);
  } else {
    recv_encrypted_vector(he.context, adapter.io, ciphertexts);
    adapter.prepare_ckks_to_mpc_inplace(ciphertexts, kLiftedScale, &he);
    send_encrypted_vector(adapter.io, ciphertexts);
    const std::uint64_t masked_before = adapter.masked_rns_coefficients;
    adapter.mask_ckks_to_mpc_inplace(ciphertexts, shares, &he);
    if (adapter.masked_rns_coefficients - masked_before != kDegree ||
        shares[0].coeff_count() != kDegree) {
      throw std::runtime_error(
          "Algorithm 1 isolation did not mask one complete RNS limb");
    }
    std::vector<std::uint64_t> current_mask(
        shares[0].data(), shares[0].data() + shares[0].coeff_count());
    if (std::all_of(current_mask.begin(), current_mask.end(),
                    [](std::uint64_t value) { return value == 0; }) ||
        (!previous_mask.empty() && current_mask == previous_mask)) {
      throw std::runtime_error(
          "Algorithm 1 isolation mask is zero or was reused");
    }
    previous_mask = std::move(current_mask);
    std::cout << "CKKS_CONVERSION_STAGE stage=algorithm1_isolation_mask_fresh"
              << " seed=" << seed
              << " status=PASS nonzero=1 reused=0" << std::endl;
    send_encrypted_vector(adapter.io, ciphertexts);
  }

  bool mask_cancelled = true;
  if (party == sci::ALICE) {
    adapter.io->send_data(shares[0].data(),
                          shares[0].coeff_count() * sizeof(std::uint64_t));
    adapter.io->recv_data(&mask_cancelled, sizeof(mask_cancelled));
  } else {
    std::vector<std::uint64_t> alice_share(shares[0].coeff_count());
    adapter.io->recv_data(alice_share.data(),
                          alice_share.size() * sizeof(std::uint64_t));
    const auto context_data =
        he.context->get_context_data(shares[0].parms_id());
    if (!context_data ||
        context_data->parms().coeff_modulus().size() != 1) {
      throw std::runtime_error("Algorithm 1 diagnostic level is invalid");
    }
    const std::uint64_t prime =
        context_data->parms().coeff_modulus()[0].value();
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < alice_share.size(); ++i) {
      const std::uint64_t reconstructed = static_cast<std::uint64_t>(
          (static_cast<__uint128_t>(alice_share[i]) + shares[0][i]) %
          prime);
      mismatches += reconstructed != baseline_decryption[i];
    }
    mask_cancelled = mismatches == 0;
    adapter.io->send_data(&mask_cancelled, sizeof(mask_cancelled));
    std::cout << "CKKS_CONVERSION_STAGE stage=algorithm1_rns_mask_cancel"
              << " seed=" << seed
              << " status=" << (mask_cancelled ? "PASS" : "FAIL")
              << " mismatched_ntt_coefficients=" << mismatches << std::endl;
  }

  const std::uint64_t final_prime = he.get_Q(shares[0]);
  auto output = adapter.ckks_to_mpc(
      shares, static_cast<int>(kDegree), final_prime, kEll, kFixedScale,
      &he, false);
  if (output.size() != 1 || output[0].size() != expected.size()) {
    throw std::runtime_error("Algorithm 1 isolation returned invalid packing");
  }
  return mask_cancelled &&
         verify_open_test_vector(adapter, output[0], expected, seed,
                                 "algorithm1_standard_encode");
}

bool run_seed(Bert_ckks &adapter, HE_CKKS &he, int seed,
              std::vector<std::uint64_t> &previous_mask) {
  const std::size_t slots = he.num_slots;
  const auto expected = public_test_vector(slots, seed);
  std::vector<std::vector<std::uint64_t>> input(
      1, additive_share(expected, seed));

  std::vector<seal::Plaintext> input_plain =
      adapter.mpc_to_ckks_rns(input, static_cast<int>(slots), kEll,
                              kFixedScale, kLiftedScale, &he, false);
  if (input_plain.size() != 1 ||
      std::llround(std::log2(input_plain[0].scale())) != kLiftedScale) {
    throw std::runtime_error("per-prime conversion returned invalid metadata");
  }
  diagnose_algorithm2(adapter, he, input_plain[0], input[0], expected, seed);

  std::vector<seal::Ciphertext> ciphertexts(1);
  std::vector<seal::Plaintext> output_plain(1);
  if (party == sci::BOB) {
    he.encryptor->encrypt(input_plain[0], ciphertexts[0]);
    send_encrypted_vector(adapter.io, ciphertexts);
    recv_encrypted_vector(he.context, adapter.io, ciphertexts);
    he.decryptor->decrypt(ciphertexts[0], output_plain[0]);
  } else {
    recv_encrypted_vector(he.context, adapter.io, ciphertexts);
    he.evaluator->add_plain_inplace(ciphertexts[0], input_plain[0]);
    if (std::abs(std::log2(ciphertexts[0].scale()) - kLiftedScale) >
        1e-9) {
      throw std::runtime_error("pre-encryption scale lift produced wrong scale");
    }

    adapter.prepare_ckks_to_mpc_inplace(ciphertexts, kLiftedScale, &he);
    auto final_context = he.context->get_context_data(
        ciphertexts[0].parms_id());
    if (!final_context ||
        final_context->parms().coeff_modulus().size() != 1 ||
        std::abs(std::log2(ciphertexts[0].scale()) - kLiftedScale) >
            1e-9) {
      throw std::runtime_error(
          "high-scale identity mod-switch did not reach final metadata");
    }

    const std::uint64_t masked_before = adapter.masked_rns_coefficients;
    adapter.mask_ckks_to_mpc_inplace(ciphertexts, output_plain, &he);
    if (adapter.masked_rns_coefficients - masked_before != kDegree ||
        output_plain[0].coeff_count() != kDegree) {
      throw std::runtime_error("Algorithm 1 did not mask one complete RNS limb");
    }
    std::vector<std::uint64_t> current_mask(
        output_plain[0].data(),
        output_plain[0].data() + output_plain[0].coeff_count());
    if (std::all_of(current_mask.begin(), current_mask.end(),
                    [](std::uint64_t value) { return value == 0; }) ||
        (!previous_mask.empty() && current_mask == previous_mask)) {
      throw std::runtime_error("Algorithm 1 mask is zero or was reused");
    }
    previous_mask = std::move(current_mask);
    std::cout << "CKKS_CONVERSION_STAGE stage=algorithm1_e2e_mask_fresh"
              << " seed=" << seed
              << " status=PASS nonzero=1 reused=0" << std::endl;
    send_encrypted_vector(adapter.io, ciphertexts);
  }

  const std::uint64_t final_prime = he.get_Q(output_plain[0]);
  auto output = adapter.ckks_to_mpc(
      output_plain, static_cast<int>(kDegree), final_prime, kEll,
      kFixedScale, &he, false);
  if (output.size() != 1 || output[0].size() != slots) {
    throw std::runtime_error("CKKS-to-MPC returned invalid packing");
  }
  const bool full_path_pass = verify_open_test_vector(
      adapter, output[0], expected, seed, "algorithm2_then_algorithm1");
  diagnose_algorithm1_decode_linearity(adapter, he, expected, seed);
  const bool algorithm1_pass =
      isolate_algorithm1(adapter, he, expected, seed, previous_mask);
  return full_path_pass && algorithm1_pass;
}

bool validate_raw_input_public_lift(int seed) {
  constexpr int kLift = 11;
  constexpr int kWorkingEll = kRawInputEll + kLift;
  const std::uint64_t input_mask = ring_mask(kRawInputEll);
  const std::uint64_t working_mask = ring_mask(kWorkingEll);
  std::vector<std::uint64_t> edge_values = {
      0,
      1,
      input_mask,
      (std::uint64_t{1} << (kRawInputEll - 1)) - 1,
      std::uint64_t{1} << (kRawInputEll - 1)};
  std::mt19937_64 generator(
      static_cast<std::uint64_t>(seed) ^ 0x494e4a4543544956ULL);
  bool pass = true;
  for (const std::uint64_t secret : edge_values) {
    const std::uint64_t alice = generator() & input_mask;
    const std::uint64_t bob = (secret - alice) & input_mask;
    const std::uint64_t lifted_secret = (secret << kLift) & working_mask;
    const std::uint64_t lifted_sum =
        ((alice << kLift) + (bob << kLift)) & working_mask;
    const std::int64_t signed_expected =
        signed_ring(secret, kRawInputEll) * (std::int64_t{1} << kLift);
    pass &= lifted_sum == lifted_secret;
    pass &= signed_ring(lifted_secret, kWorkingEll) == signed_expected;
  }
  if (party == sci::BOB) {
    std::cout
        << "CKKS_CONVERSION_STAGE stage=raw_s5_public_injective_share_lift"
        << " seed=" << seed << " status=" << (pass ? "PASS" : "FAIL")
        << " original_ell=" << kRawInputEll
        << " original_scale=" << kRawInputScale
        << " working_ell=" << kWorkingEll
        << " working_scale=" << kRawInputScale + kLift
        << " signed_edge_cases=" << edge_values.size() << std::endl;
  }
  return pass;
}

// Exercise the production layer-0 Algorithm-2 entry metadata: the real input
// has already been securely truncated from ell37/s12 to ell30/s5 and is
// imported at CKKS s90. This small {60,40,60} conversion-test context does not
// contain the production QKV/QK rescale graph, so it deliberately validates
// the combined public-fixture plaintext at the entry level instead of
// pretending that an identity s90->s40 path represents those HE kernels.
bool run_raw_input_algorithm2_seed(
    Bert_ckks &adapter, HE_CKKS &he, int seed,
    std::vector<std::uint64_t> &previous_raw_share) {
  const std::size_t slots = he.num_slots;
  const auto raw_expected = public_test_vector(
      slots, seed ^ 0x5a5a, kRawInputEll, kRawInputScale);
  std::vector<std::vector<std::uint64_t>> input(
      1, additive_share(raw_expected, seed ^ 0x5a5a, kRawInputEll));

  const bool injective_lift_pass = validate_raw_input_public_lift(seed);
  bool local_share_fresh =
      !std::all_of(input[0].begin(), input[0].end(),
                   [](std::uint64_t value) { return value == 0; }) &&
      (previous_raw_share.empty() || input[0] != previous_raw_share);
  previous_raw_share = input[0];
  bool both_shares_fresh = local_share_fresh;
  if (party == sci::ALICE) {
    adapter.io->send_data(&local_share_fresh, sizeof(local_share_fresh));
    adapter.io->recv_data(&both_shares_fresh, sizeof(both_shares_fresh));
  } else {
    bool alice_share_fresh = false;
    adapter.io->recv_data(&alice_share_fresh, sizeof(alice_share_fresh));
    both_shares_fresh &= alice_share_fresh;
    adapter.io->send_data(&both_shares_fresh, sizeof(both_shares_fresh));
    std::cout << "CKKS_CONVERSION_STAGE stage=raw_s5_algorithm2_share_fresh"
              << " seed=" << seed
              << " status=" << (both_shares_fresh ? "PASS" : "FAIL")
              << " both_nonzero=1 reused=0" << std::endl;
  }

  std::vector<seal::Plaintext> input_plain = adapter.mpc_to_ckks_rns(
      input, static_cast<int>(slots), kRawInputEll, kRawInputScale,
      kRawInputCkksScale, &he, false);
  if (input_plain.size() != 1 ||
      std::llround(std::log2(input_plain[0].scale())) !=
          kRawInputCkksScale) {
    throw std::runtime_error(
        "raw s5 Algorithm 2 returned invalid CKKS metadata");
  }

  bool numeric_pass = true;
  if (party == sci::ALICE) {
    adapter.io->send_data(
        input_plain[0].data(),
        input_plain[0].coeff_count() * sizeof(std::uint64_t));
    adapter.io->recv_data(&numeric_pass, sizeof(numeric_pass));
  } else {
    seal::Plaintext combined = input_plain[0];
    std::vector<std::uint64_t> alice_plain(combined.coeff_count());
    adapter.io->recv_data(
        alice_plain.data(),
        alice_plain.size() * sizeof(std::uint64_t));
    const auto context_data =
        he.context->get_context_data(combined.parms_id());
    if (!context_data) {
      throw std::runtime_error("raw s5 Algorithm 2 plaintext level is invalid");
    }
    const auto &moduli = context_data->parms().coeff_modulus();
    for (std::size_t limb = 0; limb < moduli.size(); ++limb) {
      const std::uint64_t modulus = moduli[limb].value();
      for (std::size_t coefficient = 0; coefficient < kDegree;
           ++coefficient) {
        const std::size_t flat = limb * kDegree + coefficient;
        combined[flat] = static_cast<std::uint64_t>(
            (static_cast<__uint128_t>(combined[flat]) +
             alice_plain[flat]) %
            modulus);
      }
    }
    std::vector<double> decoded;
    he.encoder->decode(combined, decoded);
    std::vector<std::uint64_t> decoded_fixed(decoded.size());
    const std::uint64_t input_mask = ring_mask(kRawInputEll);
    for (std::size_t index = 0; index < decoded.size(); ++index) {
      decoded_fixed[index] = static_cast<std::uint64_t>(
          static_cast<std::int64_t>(std::nearbyint(
              std::ldexp(decoded[index], kRawInputScale)))) &
          input_mask;
    }
    const RingError error =
        ring_error(decoded_fixed, raw_expected, kRawInputEll);
    numeric_pass = error.max <= kRawInputTolerance;
    adapter.io->send_data(&numeric_pass, sizeof(numeric_pass));
    std::cout
        << "CKKS_CONVERSION_RESULT stage=raw_s5_algorithm2_s90_entry_decode"
        << " seed=" << seed
        << " status=" << (numeric_pass ? "PASS" : "FAIL")
        << " slots=" << decoded.size()
        << " max_abs_error_raw=" << error.max
        << " mae_raw=" << static_cast<double>(error.sum / decoded.size())
        << " mse_raw="
        << static_cast<double>(error.squared_sum / decoded.size())
        << " max_abs_error_real="
        << std::ldexp(static_cast<double>(error.max), -kRawInputScale)
        << " mae_real="
        << std::ldexp(static_cast<double>(error.sum / decoded.size()),
                      -kRawInputScale)
        << " mse_real="
        << std::ldexp(
               static_cast<double>(error.squared_sum / decoded.size()),
               -2 * kRawInputScale)
        << " tolerance_raw=" << kRawInputTolerance
        << " scope=algorithm2_entry_only_not_paper_table6" << std::endl;
  }
  return injective_lift_pass && both_shares_fresh && numeric_pass;
}

} // namespace

int main(int argc, char **argv) {
  ArgMapping arguments;
  arguments.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  arguments.arg("p", port, "Base port");
  arguments.arg("ip", address, "IP address of ALICE");
  arguments.arg("seeds", seed_count, "Number of deterministic test seeds");
  arguments.arg("seed", first_seed, "First deterministic test seed");
  arguments.arg("tol", tolerance, "Maximum raw fixed-point error");
  arguments.parse(argc, argv);
  if ((party != sci::ALICE && party != sci::BOB) || seed_count <= 0) {
    std::cerr << "invalid conversion regression arguments" << std::endl;
    return 2;
  }

  sci::NetIO main_io(party == sci::ALICE ? nullptr : address.c_str(), port);
  HE_CKKS he(party, &main_io, kDegree, {60, 40, 60}, 20, 40,
             LOCATION::HOST);
  Bert_ckks adapter(party, &main_io, address, port + 1);

  bool pass = true;
  std::vector<std::uint64_t> previous_mask;
  std::vector<std::uint64_t> previous_raw_share;
  for (int index = 0; index < seed_count; ++index) {
    const int current_seed = first_seed + index;
    pass &= run_seed(adapter, he, current_seed, previous_mask);
    pass &= run_raw_input_algorithm2_seed(
        adapter, he, current_seed, previous_raw_share);
  }
  if (party == sci::BOB) {
    std::cout << "CKKS_CONVERSION_E2E_SUMMARY status="
              << (pass ? "PASS" : "FAIL") << " seeds=" << seed_count
              << " degree=" << kDegree << " ell=" << kEll
              << " fixed_scale=" << kFixedScale
              << " lifted_scale=" << kLiftedScale
              << " tolerance_raw=" << tolerance
              << " tolerance_real="
              << std::ldexp(static_cast<double>(tolerance), -kFixedScale)
              << " tolerance_scope=fixed_fft_numeric_bound_not_paper_mse"
              << " raw_input_ell=" << kRawInputEll
              << " raw_input_scale=" << kRawInputScale
              << " raw_input_ckks_scale=" << kRawInputCkksScale
              << " raw_input_tolerance_raw=" << kRawInputTolerance
              << " raw_input_scope=algorithm2_entry_only"
              << " packing=ordinary_full_slot"
              << " opened=test_vector_only" << std::endl;
  }
  return pass ? 0 : 1;
}
