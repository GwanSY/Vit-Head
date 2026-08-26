#include "ckks_bert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace sci;
using namespace std;

int bitlength = 43;

namespace {

constexpr int kDegree = 16384;
constexpr int kScaleIn = 40;
constexpr int kScaleOut = 13;
constexpr int kOutputBits = 43;

uint64_t ring_mask(int bits) {
  return (uint64_t{1} << bits) - 1;
}

int64_t signed_ring(uint64_t value, int bits) {
  value &= ring_mask(bits);
  const uint64_t sign = uint64_t{1} << (bits - 1);
  if ((value & sign) == 0) return static_cast<int64_t>(value);
  return static_cast<int64_t>(value) - static_cast<int64_t>(uint64_t{1} << bits);
}

vector<uint64_t> public_slots(size_t slots) {
  vector<uint64_t> result(slots);
  for (size_t i = 0; i < slots; ++i) {
    const int64_t coefficient =
        (i % 9 == 0) ? -3 : ((static_cast<int>(i) % 13) - 6);
    result[i] = static_cast<uint64_t>(coefficient * (int64_t{1} << 38));
  }
  return result;
}

vector<uint64_t> expected_slots(const vector<uint64_t> &input) {
  vector<uint64_t> result(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    int64_t value = 0;
    memcpy(&value, &input[i], sizeof(value));
    result[i] = static_cast<uint64_t>(value >> (kScaleIn - kScaleOut)) &
                ring_mask(kOutputBits);
  }
  return result;
}

vector<uint64_t> canonical_coefficients(const vector<HE_CKKS::complex_t> &coeff,
                                        uint64_t q) {
  vector<uint64_t> result(coeff.size());
  for (size_t i = 0; i < coeff.size(); ++i) {
    const uint64_t raw = coeff[i].real();
    int64_t signed_value = 0;
    memcpy(&signed_value, &raw, sizeof(raw));
    if (signed_value < 0) {
      const uint64_t magnitude =
          static_cast<uint64_t>(-(signed_value + 1)) + 1;
      const uint64_t reduced = magnitude % q;
      result[i] = reduced == 0 ? 0 : q - reduced;
    } else {
      result[i] = raw % q;
    }
  }
  return result;
}

// Exercise the exact production direction used at every MPC->CKKS boundary:
// both parties convert their ring shares, Bob encrypts only his local
// plaintext, Alice adds her local plaintext and applies Algorithm 1, and Bob
// decrypts.  The diagnostic opens only the aggregate reconstruction error.
bool run_m2c_c2m_e2e(int party, sci::NetIO *io, HE_CKKS &he,
                     Bert_ckks &adapter) {
  const bool production_profile =
      std::getenv("BLB_M2C_PRODUCTION") != nullptr;
  const int input_bits = production_profile ? 35 : 43;
  const int input_scale = production_profile ? 5 : 13;
  const int ckks_scale = production_profile ? 44 : 40;
  const int output_scale = production_profile ? 5 : 13;
  const int degree = production_profile ? 32768 : kDegree;
  const uint64_t mask = ring_mask(input_bits);
  const auto final_context = he.context->last_context_data();
  if (!final_context || final_context->parms().coeff_modulus().size() != 1) {
    throw runtime_error("M2C diagnostic has no final CKKS prime");
  }
  const uint64_t conversion_prime =
      final_context->parms().coeff_modulus().back().value();
  vector<uint64_t> expected(he.num_slots);
  vector<HE_CKKS::scalar_t> local_share(he.num_slots);
  for (size_t i = 0; i < expected.size(); ++i) {
    const int64_t value =
        static_cast<int64_t>((static_cast<int>(i) % 17) - 8) * (1LL << 10);
    expected[i] = static_cast<uint64_t>(value) & mask;
    // Deterministic nonzero shares stand in for the random MPC shares.  The
    // complement is formed in Z_2^43; no share is printed or exported.
    const uint64_t alice_share =
        (uint64_t{0x13579bdf2468} + static_cast<uint64_t>(i) * 7919) & mask;
    if (party == sci::ALICE) {
      local_share[i] = alice_share;
    } else {
      local_share[i] = (expected[i] - alice_share) & mask;
    }
  }

  vector<uint64_t> extension_low(local_share.size());
  vector<uint64_t> extension_high(local_share.size());
  adapter.nonlinear.fpmath[0]->fix->xt->z_extend_wide(
      static_cast<int>(local_share.size()), local_share.data(),
      extension_low.data(), extension_high.data(), input_bits,
      input_bits + 40);
  bool extension_pass = false;
  if (party == sci::ALICE) {
    io->send_data(extension_low.data(), extension_low.size() * sizeof(uint64_t));
    io->send_data(extension_high.data(),
                  extension_high.size() * sizeof(uint64_t));
    io->recv_data(&extension_pass, sizeof(extension_pass));
  } else {
    vector<uint64_t> alice_low(extension_low.size());
    vector<uint64_t> alice_high(extension_high.size());
    io->recv_data(alice_low.data(), alice_low.size() * sizeof(uint64_t));
    io->recv_data(alice_high.data(), alice_high.size() * sizeof(uint64_t));
    const __uint128_t wide_modulus = __uint128_t{1} << (input_bits + 40);
    const __uint128_t wide_mask = wide_modulus - 1;
    uint64_t mismatches = 0;
    uint64_t integer_wraps = 0;
    for (size_t i = 0; i < extension_low.size(); ++i) {
      const __uint128_t lhs =
          (static_cast<__uint128_t>(alice_high[i]) << 64) | alice_low[i];
      const __uint128_t rhs =
          (static_cast<__uint128_t>(extension_high[i]) << 64) |
          extension_low[i];
      const __uint128_t sum = lhs + rhs;
      const uint64_t reconstructed =
          static_cast<uint64_t>(sum & wide_mask);
      mismatches += reconstructed != expected[i];
      integer_wraps += sum >= wide_modulus;
    }
    extension_pass = mismatches == 0;
    cout << "M2C_WIDE_EXTENSION_RECON status="
         << (extension_pass ? "PASS" : "FAIL")
         << " mismatched_values=" << mismatches
         << " integer_wraps=" << integer_wraps << endl;
    io->send_data(&extension_pass, sizeof(extension_pass));
  }

  vector<uint64_t> direct_field(local_share.size());
  adapter.nonlinear.ring_to_field_wide(
      1, local_share.data(), conversion_prime, direct_field.data(),
      static_cast<int>(local_share.size()), input_bits, input_scale,
      ckks_scale, input_bits + 40);
  vector<uint64_t> expected_scaled_slots(expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    const int64_t value = signed_ring(expected[i], input_bits);
    const __int128 scaled = static_cast<__int128>(value)
                            << (ckks_scale - input_scale);
    const __int128 modulus = static_cast<__int128>(conversion_prime);
    const __int128 residue = (scaled % modulus + modulus) % modulus;
    expected_scaled_slots[i] = static_cast<uint64_t>(residue);
  }
  bool direct_field_pass = false;
  if (party == sci::ALICE) {
    io->send_data(direct_field.data(), direct_field.size() * sizeof(uint64_t));
    io->recv_data(&direct_field_pass, sizeof(direct_field_pass));
  } else {
    vector<uint64_t> alice_field(direct_field.size());
    io->recv_data(alice_field.data(), alice_field.size() * sizeof(uint64_t));
    uint64_t mismatches = 0;
    uint64_t aggregate_actual = 0;
    uint64_t aggregate_expected = 0;
    for (size_t i = 0; i < direct_field.size(); ++i) {
      const uint64_t actual = static_cast<uint64_t>(
          (static_cast<__uint128_t>(alice_field[i]) + direct_field[i]) %
          conversion_prime);
      mismatches += actual != expected_scaled_slots[i];
      aggregate_actual = static_cast<uint64_t>(
          (static_cast<__uint128_t>(aggregate_actual) + actual) %
          conversion_prime);
      aggregate_expected = static_cast<uint64_t>(
          (static_cast<__uint128_t>(aggregate_expected) +
           expected_scaled_slots[i]) % conversion_prime);
    }
    direct_field_pass = mismatches == 0;
    cout << "M2C_FIELD_SLOT_RECON status="
         << (direct_field_pass ? "PASS" : "FAIL")
         << " mismatched_slots=" << mismatches
         << " aggregate_actual_mod_q=" << aggregate_actual
         << " aggregate_expected_mod_q=" << aggregate_expected << endl;
    io->send_data(&direct_field_pass, sizeof(direct_field_pass));
  }

  auto encoded = adapter.mpc_to_ckks(
      {local_share}, he.num_slots, conversion_prime, input_bits, input_scale,
      ckks_scale, &he, false);
  if (encoded.size() != 1) {
    throw runtime_error("MPC-to-CKKS diagnostic returned an invalid shape");
  }

  // Check the modulus conversion before encryption, level changes, and
  // masking.  This compares only the reconstructed coefficient aggregate;
  // neither local share is printed.
  // Algorithm 2 encodes the l-bit shares at their MPC scale first and then
  // lifts the encoded coefficients to the CKKS scale during ring-to-field.
  // Compare against that exact paper order; encoding a pre-lifted slot vector
  // would test a different (and non-identical after fixed FFT truncations)
  // operation.
  vector<uint64_t> expected_signed_input = expected;
  const uint64_t expected_input_mask = ring_mask(input_bits);
  const uint64_t expected_input_sign = uint64_t{1} << (input_bits - 1);
  for (uint64_t &value : expected_signed_input) {
    value &= expected_input_mask;
    if ((value & expected_input_sign) != 0) value |= ~expected_input_mask;
  }
  const vector<uint64_t> expected_coeff_words =
      he.ckks_encode(expected_signed_input, he.num_slots, false);
  vector<uint64_t> expected_coeff(expected_coeff_words.size());
  vector<uint64_t> local_coeff = canonical_coefficients(
      he.ckks_intt(encoded[0]), conversion_prime);
  for (size_t i = 0; i < expected_coeff.size(); ++i) {
    int64_t signed_value = 0;
    memcpy(&signed_value, &expected_coeff_words[i], sizeof(signed_value));
    uint64_t canonical = 0;
    if (signed_value < 0) {
      const uint64_t magnitude =
          static_cast<uint64_t>(-(signed_value + 1)) + 1;
      const uint64_t reduced = magnitude % conversion_prime;
      canonical = reduced == 0 ? 0 : conversion_prime - reduced;
    } else {
      canonical = static_cast<uint64_t>(signed_value) % conversion_prime;
    }
    expected_coeff[i] = static_cast<uint64_t>(
        (static_cast<__uint128_t>(canonical) <<
         (ckks_scale - input_scale)) % conversion_prime);
  }
  bool field_pass = false;
  if (party == sci::ALICE) {
    io->send_data(local_coeff.data(), local_coeff.size() * sizeof(uint64_t));
    io->recv_data(&field_pass, sizeof(field_pass));
  } else {
    vector<uint64_t> alice_coeff(local_coeff.size());
    io->recv_data(alice_coeff.data(), alice_coeff.size() * sizeof(uint64_t));
    uint64_t mismatches = 0;
    uint64_t max_error = 0;
    long double squared_error_sum = 0.0L;
    uint64_t aggregate_actual = 0;
    uint64_t aggregate_expected = 0;
    for (size_t i = 0; i < local_coeff.size(); ++i) {
      const uint64_t actual = static_cast<uint64_t>(
          (static_cast<__uint128_t>(alice_coeff[i]) + local_coeff[i]) %
          conversion_prime);
      const int64_t error = signed_ring(actual - expected_coeff[i], 60);
      const uint64_t abs_error =
          static_cast<uint64_t>(error < 0 ? -error : error);
      mismatches += abs_error != 0;
      max_error = max(max_error, abs_error);
      squared_error_sum += static_cast<long double>(error) *
                           static_cast<long double>(error);
      aggregate_actual = static_cast<uint64_t>(
          (static_cast<__uint128_t>(aggregate_actual) + actual) %
          conversion_prime);
      aggregate_expected = static_cast<uint64_t>(
          (static_cast<__uint128_t>(aggregate_expected) + expected_coeff[i]) %
          conversion_prime);
    }
    field_pass = mismatches == 0;
    const long double field_mse_raw =
        squared_error_sum / static_cast<long double>(local_coeff.size());
    const long double field_scale =
        ldexpl(1.0L, ckks_scale);
    const long double field_mse_scaled =
        field_mse_raw / (field_scale * field_scale);
    cout << "M2C_FIELD_RECON status=" << (field_pass ? "PASS" : "FAIL")
         << " mismatched_coefficients=" << mismatches
         << " max_abs_error_raw=" << max_error
         << " mse_raw=" << scientific << setprecision(12)
         << field_mse_raw
         << " mse_scaled=" << field_mse_scaled << defaultfloat
         << " aggregate_actual_mod_q=" << aggregate_actual
         << " aggregate_expected_mod_q=" << aggregate_expected << endl;
    io->send_data(&field_pass, sizeof(field_pass));
  }

  vector<Ciphertext> ciphertexts(1);
  vector<Plaintext> local_plain(1);
  if (party == sci::BOB) {
    he.encryptor->encrypt(encoded[0], ciphertexts[0]);
    send_encrypted_vector(io, ciphertexts);
    recv_encrypted_vector(he.context, io, ciphertexts);
    he.decryptor->decrypt(ciphertexts[0], local_plain[0]);
  } else {
    recv_encrypted_vector(he.context, io, ciphertexts);
    // A production linear block reaches the Algorithm-1 boundary at the
    // final single-prime level.  Reproduce that public level transition
    // before adding Alice's imported share; no ciphertext or share is opened.
    he.evaluator->mod_switch_to_inplace(ciphertexts[0],
                                        he.context->last_parms_id());
    he.evaluator->mod_switch_to_inplace(encoded[0],
                                        he.context->last_parms_id());
    he.evaluator->add_plain_inplace(ciphertexts[0], encoded[0]);
    adapter.mask_ckks_to_mpc_inplace(ciphertexts, local_plain, &he);
    send_encrypted_vector(io, ciphertexts);
  }

  const uint64_t q = he.get_Q(local_plain[0]);
  const auto decoded = adapter.ckks_to_mpc(
      local_plain, degree, q, input_bits, output_scale, &he, false);
  if (decoded.size() != 1 || decoded[0].size() != he.num_slots) {
    throw runtime_error("MPC-to-CKKS diagnostic returned an invalid decode");
  }

  bool pass = false;
  if (party == sci::ALICE) {
    io->send_data(decoded[0].data(), decoded[0].size() * sizeof(uint64_t));
    io->recv_data(&pass, sizeof(pass));
  } else {
    vector<uint64_t> alice_share(decoded[0].size());
    io->recv_data(alice_share.data(), alice_share.size() * sizeof(uint64_t));
    uint64_t max_error = 0;
    uint64_t bad = 0;
    long double squared_error_sum = 0.0L;
    for (size_t i = 0; i < decoded[0].size(); ++i) {
      const uint64_t actual = (alice_share[i] + decoded[0][i]) & mask;
      const int64_t error = signed_ring(actual - expected[i], input_bits);
      const uint64_t abs_error =
          static_cast<uint64_t>(error < 0 ? -error : error);
      max_error = max(max_error, abs_error);
      bad += abs_error != 0;
      squared_error_sum += static_cast<long double>(error) *
                           static_cast<long double>(error);
    }
    pass = max_error <= 8;
    const long double e2e_mse_raw =
        squared_error_sum / static_cast<long double>(decoded[0].size());
    const long double output_scale_value = ldexpl(1.0L, output_scale);
    const long double e2e_mse_scaled =
        e2e_mse_raw / (output_scale_value * output_scale_value);
    cout << "M2C_C2M_E2E_SELFTEST status=" << (pass ? "PASS" : "FAIL")
         << " max_abs_error_raw=" << max_error << " bad_values=" << bad
         << " mse_raw=" << scientific << setprecision(12) << e2e_mse_raw
         << " mse_scaled=" << e2e_mse_scaled << defaultfloat
         << " cts=1 input_scale=" << input_scale
         << " output_scale=" << output_scale
         << " ckks_scale=" << ckks_scale << " output_bits=" << input_bits
         << " profile=" << (production_profile ? "bert_qkv" : "generic")
         << endl;
    io->send_data(&pass, sizeof(pass));
  }
  return pass;
}

}  // namespace

int main(int argc, char **argv) {
  int party = 0;
  int port = 0;
  string address = "127.0.0.1";
  ArgMapping args;
  args.arg("r", party, "party");
  args.arg("p", port, "port");
  args.arg("ip", address, "address");
  args.parse(argc, argv);
  if ((party != sci::ALICE && party != sci::BOB) || port <= 0) {
    throw invalid_argument("usage: r=1|2 p=PORT [ip=ADDRESS]");
  }

  sci::NetIO io(party == sci::ALICE ? nullptr : address.c_str(), port);
  const bool production_profile =
      std::getenv("BLB_M2C_PRODUCTION") != nullptr;
  HE_CKKS he(
      party, &io, production_profile ? 32768 : kDegree,
      production_profile ? vector<int>{60, 35, 60, 60, 60, 60, 60, 60}
                          : vector<int>{60, 60, 60, 60, 60},
      20, production_profile ? 44 : 38, LOCATION::HOST);
  Bert_ckks adapter(party, &io, address, port + 1);

  if (std::getenv("BLB_M2C_C2M_E2E") != nullptr) {
    return run_m2c_c2m_e2e(party, &io, he, adapter) ? 0 : 1;
  }

  const vector<uint64_t> input = public_slots(he.num_slots);
  vector<uint64_t> expected_coefficients;
  vector<uint64_t> expected_ring_coefficients;
  vector<uint64_t> expected;
  vector<Plaintext> local_share(1);
  vector<Ciphertext> ciphertexts(1);

  if (party == sci::BOB) {
    const vector<uint64_t> encoded = he.ckks_encode(
        input, static_cast<int>(he.num_slots), false);
    Plaintext plaintext;
    he.ckks_ntt(encoded, kDegree, ldexp(1.0, kScaleIn), plaintext);
    he.evaluator->mod_switch_to_inplace(plaintext, he.context->last_parms_id());
    const auto baseline_complex = he.ckks_intt(plaintext);
    const uint64_t baseline_q = he.context->get_context_data(
        plaintext.parms_id())->parms().coeff_modulus().back().value();
    expected_coefficients = canonical_coefficients(baseline_complex,
                                                   baseline_q);
    expected_ring_coefficients.resize(baseline_complex.size());
    for (size_t i = 0; i < baseline_complex.size(); ++i) {
      int64_t value = 0;
      const uint64_t raw = baseline_complex[i].real();
      memcpy(&value, &raw, sizeof(value));
      expected_ring_coefficients[i] =
          static_cast<uint64_t>(value) & ring_mask(60);
    }
    const auto baseline_slots = he.ckks_decode(
        baseline_complex, kDegree, false);
    expected.resize(baseline_slots.size());
    for (size_t i = 0; i < baseline_slots.size(); ++i) {
      int64_t value = 0;
      memcpy(&value, &baseline_slots[i], sizeof(value));
      expected[i] = static_cast<uint64_t>(value >> (kScaleIn - kScaleOut)) &
                    ring_mask(kOutputBits);
    }
    he.encryptor->encrypt(plaintext, ciphertexts[0]);
    send_encrypted_vector(&io, ciphertexts);
    recv_encrypted_vector(he.context, &io, ciphertexts);
    he.decryptor->decrypt(ciphertexts[0], local_share[0]);
  } else {
    recv_encrypted_vector(he.context, &io, ciphertexts);
    adapter.mask_ckks_to_mpc_inplace(ciphertexts, local_share, &he);
    send_encrypted_vector(&io, ciphertexts);
  }

  const uint64_t q = he.get_Q(local_share[0]);

  // Public-fixture check of the exact Algorithm-1 mask boundary.  Only the
  // reconstructed coefficient polynomial is opened; the local mask shares
  // are never logged or exported.
  const auto local_coefficients = canonical_coefficients(
      he.ckks_intt(local_share[0]), q);
  bool mask_pass = true;
  uint64_t mask_mismatches = 0;
  vector<uint64_t> field_expected_ring(local_coefficients.size());
  vector<uint64_t> alice_coefficients;
  if (party == sci::ALICE) {
    io.send_data(local_coefficients.data(),
                 local_coefficients.size() * sizeof(uint64_t));
    io.recv_data(&mask_pass, sizeof(mask_pass));
  } else {
    alice_coefficients.resize(local_coefficients.size());
    io.recv_data(alice_coefficients.data(),
                 alice_coefficients.size() * sizeof(uint64_t));
    for (size_t i = 0; i < local_coefficients.size(); ++i) {
      const uint64_t reconstructed = static_cast<uint64_t>(
          (static_cast<__uint128_t>(alice_coefficients[i]) +
           local_coefficients[i]) % q);
      if (reconstructed != expected_coefficients[i]) ++mask_mismatches;
      const int64_t signed_value = reconstructed >= q / 2
                                       ? static_cast<int64_t>(reconstructed - q)
                                       : static_cast<int64_t>(reconstructed);
      field_expected_ring[i] =
          static_cast<uint64_t>(signed_value) & ring_mask(60);
    }
    mask_pass = mask_mismatches == 0;
    cout << "CKKS_MASK_COEFF_SELFTEST status="
         << (mask_pass ? "PASS" : "FAIL")
         << " mismatched_coefficients=" << mask_mismatches << endl;
    io.send_data(&mask_pass, sizeof(mask_pass));
  }

  // Isolate the prime-to-ring operation on the same random Algorithm-1
  // shares. The opened diagnostic is the signed representative of the
  // public fixture only; no local mask coefficient is printed.
  vector<uint64_t> centered_field = local_coefficients;
  const uint64_t field_center = (q - 1) / 2;
  if (party == sci::ALICE) {
    for (uint64_t &value : centered_field) {
      value = static_cast<uint64_t>(
          (static_cast<__uint128_t>(value) + field_center) % q);
    }
  }
  FixArray field_input = adapter.nonlinear.fpmath[0]->fix->input(
      party, static_cast<int>(centered_field.size()), centered_field.data(),
      false, 60, kScaleIn);
  FixArray ring_output = adapter.nonlinear.fpmath[0]->fix->field2ring(
      field_input, q, 60);
  const uint64_t ring_mask_60 = ring_mask(60);
  if (party == sci::ALICE) {
    for (int i = 0; i < ring_output.size; ++i) {
      ring_output.data[i] =
          (ring_output.data[i] - (field_center & ring_mask_60)) & ring_mask_60;
    }
  }
  FixArray opened_ring =
      adapter.nonlinear.fpmath[0]->fix->output(sci::PUBLIC, ring_output);
  if (party == sci::BOB) {
    uint64_t ring_mismatches = 0;
    for (size_t i = 0; i < field_expected_ring.size(); ++i) {
      ring_mismatches += opened_ring.data[i] != field_expected_ring[i];
    }
    cout << "CKKS_FIELD2RING_RANDOM_SHARE_SELFTEST status="
         << (ring_mismatches == 0 ? "PASS" : "FAIL")
         << " mismatched_coefficients=" << ring_mismatches << endl;
  }
  const auto output = adapter.ckks_to_mpc(
      local_share, kDegree, q, kOutputBits, kScaleOut, &he, false);
  if (output.size() != 1 || output[0].size() != he.num_slots) {
    throw runtime_error("mask e2e self-test returned an invalid shape");
  }

  bool pass = mask_pass;
  uint64_t max_error = 0;
  if (party == sci::ALICE) {
    io.send_data(output[0].data(), output[0].size() * sizeof(uint64_t));
  } else {
    vector<uint64_t> alice_share(output[0].size());
    io.recv_data(alice_share.data(), alice_share.size() * sizeof(uint64_t));
    for (size_t i = 0; i < output[0].size(); ++i) {
      const uint64_t actual =
          (alice_share[i] + output[0][i]) & ring_mask(kOutputBits);
      const int64_t error = signed_ring(actual - expected[i], kOutputBits);
      max_error = max<uint64_t>(
          max_error, static_cast<uint64_t>(error < 0 ? -error : error));
    }
    pass = max_error <= 8;
    cout << "CKKS_MASK_E2E_SELFTEST status=" << (pass ? "PASS" : "FAIL")
         << " max_abs_error_raw=" << max_error
         << " cts=1 input_scale=" << kScaleIn
         << " output_scale=" << kScaleOut << " output_bits=" << kOutputBits
         << endl;
    io.send_data(&pass, sizeof(pass));
  }
  if (party == sci::ALICE) io.recv_data(&pass, sizeof(pass));
  return pass ? 0 : 1;
}
