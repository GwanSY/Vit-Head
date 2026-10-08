#include "ckks_bert.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sci;
using namespace std;
using namespace seal;

// SCI's legacy LinearHE utility references this compatibility symbol.  The
// conversion test keeps it explicit instead of relying on an unrelated ViT
// driver translation unit to provide it.
int bitlength = 37;

namespace {

constexpr int kDegree = 32768;
constexpr int kBits = 37;
constexpr int kScale = 12;
constexpr int kCkksScale = 43;
constexpr int kEncodeGuardBits = 16;
constexpr int kWideFftBits = kBits + 40 + kEncodeGuardBits;
constexpr int kC2Count = 128;
constexpr uint64_t kRingMask = (uint64_t{1} << kBits) - 1;

int active_workers(const Bert_ckks &adapter) {
  return adapter.nonlinear.active_workers();
}

int field_bits(uint64_t modulus) {
  int bits = 0;
  uint64_t value = 1;
  while (value < modulus) {
    value <<= 1;
    ++bits;
  }
  return bits;
}

int64_t signed_ring(uint64_t value) {
  value &= kRingMask;
  if ((value & (uint64_t{1} << (kBits - 1))) == 0)
    return static_cast<int64_t>(value);
  return static_cast<int64_t>(value) - static_cast<int64_t>(uint64_t{1} << kBits);
}

uint64_t ring_from_signed(int64_t value) {
  return static_cast<uint64_t>(value) & kRingMask;
}

int64_t centered_prime(uint64_t residue, uint64_t modulus) {
  const uint64_t threshold = (modulus + 1) / 2;
  if (residue >= threshold)
    return -static_cast<int64_t>(modulus - residue);
  return static_cast<int64_t>(residue);
}

uint64_t prime_from_centered(int64_t value, uint64_t modulus) {
  if (value >= 0)
    return static_cast<uint64_t>(value) % modulus;
  const uint64_t magnitude = static_cast<uint64_t>(-(value + 1)) + 1;
  const uint64_t reduced = magnitude % modulus;
  return reduced == 0 ? 0 : modulus - reduced;
}

uint64_t add_mod_prime(uint64_t lhs, uint64_t rhs, uint64_t modulus) {
  const __uint128_t sum = static_cast<__uint128_t>(lhs) + rhs;
  return static_cast<uint64_t>(sum % modulus);
}

uint64_t add_mod_ring(uint64_t lhs, uint64_t rhs) {
  return (lhs + rhs) & kRingMask;
}

uint64_t raw_from_twos_complement(uint64_t word, uint64_t modulus) {
  int64_t signed_word = 0;
  static_assert(sizeof(signed_word) == sizeof(word),
                "unexpected scalar width");
  memcpy(&signed_word, &word, sizeof(word));
  return prime_from_centered(signed_word, modulus);
}

vector<uint64_t> canonical_intt(const vector<complex<HE_CKKS::scalar_t>> &coefficients,
                                uint64_t modulus) {
  vector<uint64_t> result(coefficients.size());
  for (size_t index = 0; index < coefficients.size(); ++index)
    result[index] = raw_from_twos_complement(
        static_cast<uint64_t>(coefficients[index].real()), modulus);
  return result;
}

void exchange_vector(sci::NetIO &io, int party, const vector<uint64_t> &local,
                     vector<uint64_t> &remote) {
  remote.resize(local.size());
  if (party == sci::ALICE) {
    io.send_data(local.data(), local.size() * sizeof(uint64_t));
    io.recv_data(remote.data(), remote.size() * sizeof(uint64_t));
  } else {
    io.recv_data(remote.data(), remote.size() * sizeof(uint64_t));
    io.send_data(local.data(), local.size() * sizeof(uint64_t));
  }
}

bool gate(sci::NetIO &io, int party, const char *name, bool local_status) {
  uint8_t local = local_status ? 1 : 0;
  uint8_t remote = 0;
  if (party == sci::ALICE) {
    io.send_data(&local, sizeof(local));
    io.recv_data(&remote, sizeof(remote));
  } else {
    io.recv_data(&remote, sizeof(remote));
    io.send_data(&local, sizeof(local));
  }
  const bool passed = local != 0 && remote != 0;
  cout << "CONVERSION_TEST " << name << " status="
       << (passed ? "PASS" : "FAIL") << " local=" << static_cast<int>(local)
       << " remote=" << static_cast<int>(remote) << endl;
  return passed;
}

void stop_after(const char *name) {
  cout << "CONVERSION_TEST STOP after " << name << endl;
}

void print_mismatch(size_t index, uint64_t raw_mod_q, int64_t centered_q,
                    uint64_t share0, uint64_t share1,
                    uint64_t reconstructed_ring, int64_t decoded_value,
                    int64_t reference) {
  cout << "CONVERSION_TEST mismatch index=" << index
       << " raw_mod_q=" << raw_mod_q << " centered_q=" << centered_q
       << " share0=" << share0 << " share1=" << share1
       << " reconstructed_ring=" << reconstructed_ring
       << " decoded_value=" << decoded_value << " reference=" << reference
       << endl;
}

vector<double> make_values(size_t slots) {
  vector<double> values(slots);
  mt19937_64 random(0xA0C0FFEE12345678ULL);
  uniform_real_distribution<double> distribution(-1.75, 1.75);
  for (double &value : values)
    value = distribution(random);

  // Keep explicit signed/zero probes in the same vector used by every gate.
  values[0] = 0.0;
  values[1] = 1.0;
  values[2] = -1.0;
  values[3] = 0.5;
  values[4] = -0.5;
  values[5] = 2.0;
  values[6] = -2.0;
  values[slots / 2] = 0.125;
  values[slots - 1] = -0.75;
  return values;
}

vector<uint64_t> fixed_values(const vector<double> &values) {
  vector<uint64_t> result(values.size());
  const double scale = ldexp(1.0, kScale);
  for (size_t index = 0; index < values.size(); ++index)
    result[index] = ring_from_signed(static_cast<int64_t>(llround(values[index] * scale)));
  return result;
}

void make_shares(const vector<uint64_t> &expected, uint64_t modulus,
                 vector<uint64_t> &share0, vector<uint64_t> &share1) {
  share0.resize(expected.size());
  share1.resize(expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    const __uint128_t seed =
        __uint128_t{0x3141592653589793ULL} +
        (__uint128_t{index + 1} * 0x9E3779B97F4A7C15ULL) +
        (__uint128_t{index + 17} * __uint128_t{0xD1B54A32D192ED03ULL});
    share0[index] = static_cast<uint64_t>(seed % modulus);
    share1[index] = expected[index] >= share0[index]
                        ? expected[index] - share0[index]
                        : modulus - (share0[index] - expected[index]);
  }
}

vector<int64_t> make_c2_centered(uint64_t modulus) {
  vector<int64_t> result = {
      0, 1, -1, 1LL << (kScale - 1), -(1LL << (kScale - 1)),
      1LL << kScale, -(1LL << kScale), 2LL << kScale, -(2LL << kScale),
      (1LL << 36) - 1, -(1LL << 36), (1LL << 35) + 12345,
      -((1LL << 35) + 12345)};
  mt19937_64 random(0xC2C2C2C2ULL);
  const int64_t limit = (1LL << (kBits - 2)) - 1;
  uniform_int_distribution<int64_t> distribution(-limit, limit);
  while (static_cast<int>(result.size()) < kC2Count)
    result.push_back(distribution(random));
  for (size_t index = 0; index < result.size(); ++index) {
    if (result[index] >= static_cast<int64_t>(modulus) ||
        result[index] <= -static_cast<int64_t>(modulus))
      throw runtime_error("C2 centered fixture does not fit the prime");
  }
  return result;
}

bool check_native_decode(const vector<complex<double>> &decoded,
                         const vector<double> &reference, double tolerance,
                         double &max_error) {
  max_error = 0.0;
  if (decoded.size() != reference.size())
    return false;
  bool passed = true;
  for (size_t index = 0; index < reference.size(); ++index) {
    const double error = abs(decoded[index].real() - reference[index]);
    max_error = max(max_error, error);
    passed = passed && isfinite(error) && error <= tolerance &&
             abs(decoded[index].imag()) <= tolerance;
  }
  return passed;
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
  if ((party != sci::ALICE && party != sci::BOB) || port <= 0)
    throw invalid_argument("usage: r=1|2 p=PORT [ip=ADDRESS]");

  // The gate is intentionally small enough to run on a laptop by default.
  // A caller may override it to exercise the multi-worker port layout.
  if (getenv("OURS_VIT_MPC_WORKERS") == nullptr)
    setenv("OURS_VIT_MPC_WORKERS", "1", 1);

  sci::NetIO io(party == sci::ALICE ? nullptr : address.c_str(), port);
  HE_CKKS he(party, &io, kDegree, vector<int>{60, 60, 60, 60, 60, 60, 60},
             40, kCkksScale, LOCATION::HOST);
  Bert_ckks adapter(party, &io, address, port + 1);

  const uint64_t conversion_prime = he.get_conversion_prime();
  const int q_bits = field_bits(conversion_prime);
  const vector<double> values = make_values(he.num_slots);
  const vector<uint64_t> expected_fixed = fixed_values(values);
  vector<uint64_t> input_share0;
  vector<uint64_t> input_share1;
  make_shares(expected_fixed, uint64_t{1} << kBits, input_share0, input_share1);

  Plaintext native_plain;
  he.encoder->encode(values, he.context->last_parms_id(),
                     ldexp(1.0, kCkksScale), native_plain);
  vector<complex<double>> native_decoded;
  he.encoder->decode(native_plain, native_decoded);

  // C0: native CKKS encode -> encrypt -> decrypt -> decode.
  vector<Ciphertext> c0_ciphertexts(1);
  if (party == sci::BOB) {
    he.encryptor->encrypt(native_plain, c0_ciphertexts[0]);
    send_encrypted_vector(&io, c0_ciphertexts);
  } else {
    recv_encrypted_vector(he.context, &io, c0_ciphertexts);
  }
  double native_max_error = 0.0;
  const bool native_ok =
      check_native_decode(native_decoded, values, 1e-5, native_max_error);
  bool decrypt_ok = true;
  double decrypt_max_error = 0.0;
  if (party == sci::BOB) {
    Plaintext decrypted;
    he.decryptor->decrypt(c0_ciphertexts[0], decrypted);
    vector<complex<double>> decrypted_decoded;
    he.encoder->decode(decrypted, decrypted_decoded);
    decrypt_ok = check_native_decode(decrypted_decoded, values, 1e-3,
                                     decrypt_max_error);
  }
  if (!gate(io, party, "C0", native_ok && decrypt_ok)) {
    stop_after("C0");
    return 1;
  }
  cout << "CONVERSION_TEST C0 detail=encode_encrypt_decrypt_decode"
       << " native_max_abs=" << native_max_error
       << " decrypt_max_abs=" << decrypt_max_error << endl;

  // C1: add a uniform polynomial mask in the current NTT domain, inverse
  // transform both terms, and reconstruct every coefficient modulo q.
  vector<Ciphertext> masked_ciphertexts(1);
  vector<Plaintext> negative_masks;
  vector<uint64_t> negative_mask_coefficients;
  if (party == sci::ALICE) {
    masked_ciphertexts[0] = c0_ciphertexts[0];
    adapter.mask_ckks_to_mpc_inplace(masked_ciphertexts, negative_masks, &he);
    negative_mask_coefficients.resize(kDegree);
    copy_n(negative_masks[0].data(0), kDegree,
           negative_mask_coefficients.begin());
    send_encrypted_vector(&io, masked_ciphertexts);
    io.send_data(negative_mask_coefficients.data(),
                 negative_mask_coefficients.size() * sizeof(uint64_t));
  } else {
    recv_encrypted_vector(he.context, &io, masked_ciphertexts);
    negative_mask_coefficients.resize(kDegree);
    io.recv_data(negative_mask_coefficients.data(),
                 negative_mask_coefficients.size() * sizeof(uint64_t));
  }

  bool c1_ok = true;
  uint64_t c1_mismatches = 0;
  if (party == sci::BOB) {
    // mask_ckks_to_mpc_inplace changes c0 in the current NTT-domain
    // ciphertext. Compare that exact polynomial, not a decrypted CKKS
    // plaintext (which necessarily contains encryption noise).
    const uint64_t *original = c0_ciphertexts[0].data(0);
    const uint64_t *masked = masked_ciphertexts[0].data(0);
    for (size_t index = 0; index < kDegree; ++index) {
      const uint64_t reconstructed = add_mod_prime(
          masked[index] % conversion_prime, negative_mask_coefficients[index],
          conversion_prime);
      const uint64_t original_residue = original[index] % conversion_prime;
      if (reconstructed != original_residue) {
        if (c1_mismatches < 20) {
          const int64_t reference =
              centered_prime(original_residue, conversion_prime);
          print_mismatch(
              index, original_residue, reference, masked[index],
              negative_mask_coefficients[index], reconstructed,
              centered_prime(reconstructed, conversion_prime), reference);
        }
        ++c1_mismatches;
      }
    }
    c1_ok = c1_mismatches == 0;
  }
  if (!gate(io, party, "C1", c1_ok)) {
    stop_after("C1");
    return 1;
  }
  cout << "CONVERSION_TEST C1 detail=uniform_ntt_polynomial_mask"
       << " coefficients=" << kDegree << " mismatches=" << c1_mismatches
       << endl;

  // C2: standalone Z_q -> Z_(2^l).  The fixtures include field residues
  // q-|x| for negative x and independent random additive shares, so a direct
  // `% 2^l` implementation fails this gate.
  const vector<int64_t> c2_centered = make_c2_centered(conversion_prime);
  vector<uint64_t> c2_raw(c2_centered.size());
  vector<uint64_t> c2_share0(c2_centered.size());
  vector<uint64_t> c2_share1(c2_centered.size());
  for (size_t index = 0; index < c2_centered.size(); ++index)
    c2_raw[index] = prime_from_centered(c2_centered[index], conversion_prime);
  make_shares(c2_raw, conversion_prime, c2_share0, c2_share1);
  vector<uint64_t> c2_local_field =
      party == sci::ALICE ? c2_share0 : c2_share1;
  const uint64_t center_offset = (conversion_prime - 1) / 2;
  if (party == sci::ALICE) {
    for (uint64_t &share : c2_local_field) {
      share = static_cast<uint64_t>(
          (static_cast<__uint128_t>(share) + center_offset) %
          conversion_prime);
    }
  }
  vector<uint64_t> c2_local_ring(c2_local_field.size());
  adapter.nonlinear.field_to_ring(
      active_workers(adapter), c2_local_field.data(),
      conversion_prime, c2_local_ring.data(), static_cast<int>(c2_local_ring.size()),
      kBits, 0, 0);
  if (party == sci::ALICE) {
    for (uint64_t &share : c2_local_ring)
      share = (share - (center_offset & kRingMask)) & kRingMask;
  }
  vector<uint64_t> c2_remote_ring;
  exchange_vector(io, party, c2_local_ring, c2_remote_ring);
  bool c2_ok = true;
  uint64_t c2_mismatches = 0;
  for (size_t index = 0; index < c2_local_ring.size(); ++index) {
    const uint64_t reconstructed =
        add_mod_ring(c2_local_ring[index], c2_remote_ring[index]);
    const uint64_t reference = ring_from_signed(c2_centered[index]);
    if (reconstructed != reference) {
      c2_ok = false;
      if (c2_mismatches < 20) {
        print_mismatch(index, c2_raw[index], c2_centered[index], c2_share0[index],
                       c2_share1[index], reconstructed, signed_ring(reconstructed),
                       c2_centered[index]);
      }
      ++c2_mismatches;
    }
  }
  if (!gate(io, party, "C2", c2_ok)) {
    stop_after("C2");
    return 1;
  }
  cout << "CONVERSION_TEST C2 detail=Zq_to_Z2^l"
       << " field_bits=" << q_bits << " ring_bits=" << kBits
       << " elements=" << c2_centered.size()
       << " mismatches=" << c2_mismatches << endl;

  // C3: local signed-wide CKKS encode/decode on additive shares.  The
  // reconstructed result must preserve slot 0, slot 1, middle, last, and all
  // random lanes relative to the native CKKS decode.
  vector<uint64_t> c3_local_input =
      party == sci::ALICE ? input_share0 : input_share1;
  vector<uint64_t> c3_low(c3_local_input.size());
  vector<uint64_t> c3_high(c3_local_input.size());
  adapter.nonlinear.ring_extend_wide(
      active_workers(adapter), c3_local_input.data(),
      c3_low.data(), c3_high.data(), static_cast<int>(c3_local_input.size()),
      kBits, kWideFftBits);
  const __uint128_t wide_mask = (__uint128_t{1} << kWideFftBits) - 1;
  for (size_t index = 0; index < c3_low.size(); ++index) {
    __uint128_t value = (static_cast<__uint128_t>(c3_high[index]) << 64) |
                        c3_low[index];
    value = (value << kEncodeGuardBits) & wide_mask;
    c3_low[index] = static_cast<uint64_t>(value);
    c3_high[index] = static_cast<uint64_t>(value >> 64);
  }
  vector<HE_CKKS::wide_complex_t> c3_input(kDegree);
  for (size_t index = 0; index < he.num_slots; ++index) {
    const __uint128_t value =
        (static_cast<__uint128_t>(c3_high[index]) << 64) | c3_low[index];
    c3_input[he.matrix_reps_index_map_[index]].real = value;
    c3_input[he.matrix_reps_index_map_[index + he.num_slots]].real = value;
  }
  const auto c3_encoded = he.ckks_encode_wide(c3_input, kDegree, kWideFftBits);
  const vector<uint64_t> c3_local_decoded = he.ckks_decode_wide(
      c3_encoded, kDegree, kBits, kScale + kEncodeGuardBits, kScale, false,
      kWideFftBits);
  vector<uint64_t> c3_remote_decoded;
  exchange_vector(io, party, c3_local_decoded, c3_remote_decoded);
  bool c3_ok = true;
  uint64_t c3_mismatches = 0;
  double c3_max_native_error = 0.0;
  for (size_t index = 0; index < he.num_slots; ++index) {
    const uint64_t reconstructed =
        add_mod_ring(c3_local_decoded[index], c3_remote_decoded[index]);
    const int64_t decoded_raw = signed_ring(reconstructed);
    const double decoded_value = ldexp(static_cast<double>(decoded_raw), -kScale);
    const double reference = native_decoded[index].real();
    const double error = abs(decoded_value - reference);
    c3_max_native_error = max(c3_max_native_error, error);
    const int64_t native_reference_raw =
        static_cast<int64_t>(llround(reference * ldexp(1.0, kScale)));
    const int64_t raw_error = decoded_raw - native_reference_raw;
    const uint64_t absolute_raw_error =
        static_cast<uint64_t>(raw_error < 0 ? -raw_error : raw_error);
    const bool lane_ok = absolute_raw_error <= 8 &&
                         abs(native_decoded[index].imag()) <= 1e-5;
    if (!lane_ok) {
      c3_ok = false;
      if (c3_mismatches < 20) {
        const int64_t centered = signed_ring(expected_fixed[index]);
        print_mismatch(
            index, prime_from_centered(centered, conversion_prime), centered,
            input_share0[index], input_share1[index], reconstructed, decoded_raw,
            native_reference_raw);
      }
      ++c3_mismatches;
    }
  }
  if (!gate(io, party, "C3", c3_ok)) {
    stop_after("C3");
    return 1;
  }
  cout << "CONVERSION_TEST C3 detail=local_wide_CKKS_decode"
       << " direction=IFFT_then_FFT slot_probes=0,1,middle,last"
       << " working_bits=" << kWideFftBits
       << " max_native_abs=" << c3_max_native_error
       << " mismatches=" << c3_mismatches << endl;

  // Optional C4N: BLB's non-negative extension contract.  This gate checks
  // the protocol independently of CKKS so a production Softmax regression
  // cannot be confused with an encoder/NTT regression.  The largest probe is
  // 2^(l-1)-1, which is still valid because the reconstructed l-bit MSB is
  // zero; the local additive shares themselves remain fully random.
  if (getenv("OURS_VIT_CONVERSION_NONNEGATIVE_TEST") != nullptr) {
    const int extension_bits = kBits + 40;
    const __uint128_t extension_mask =
        (__uint128_t{1} << extension_bits) - 1;
    const uint64_t nonnegative_mask = (uint64_t{1} << (kBits - 1)) - 1;
    vector<uint64_t> nonnegative_expected(he.num_slots);
    for (size_t index = 0; index < nonnegative_expected.size(); ++index) {
      if (index == 0)
        nonnegative_expected[index] = 0;
      else if (index == 1)
        nonnegative_expected[index] = 1;
      else if (index == 2)
        nonnegative_expected[index] = nonnegative_mask;
      else
        nonnegative_expected[index] =
            (static_cast<uint64_t>(index) * 104729ULL) & nonnegative_mask;
    }
    vector<uint64_t> nonnegative_share0, nonnegative_share1;
    make_shares(nonnegative_expected, uint64_t{1} << kBits,
                nonnegative_share0, nonnegative_share1);
    vector<uint64_t> nonnegative_local =
        party == sci::ALICE ? nonnegative_share0 : nonnegative_share1;
    vector<uint64_t> nonnegative_low(nonnegative_local.size());
    vector<uint64_t> nonnegative_high(nonnegative_local.size());
    adapter.nonlinear.ring_extend_wide_nonnegative(
        active_workers(adapter), nonnegative_local.data(),
        nonnegative_low.data(), nonnegative_high.data(),
        static_cast<int>(nonnegative_local.size()), kBits, extension_bits);
    vector<uint64_t> nonnegative_remote_low;
    vector<uint64_t> nonnegative_remote_high;
    exchange_vector(io, party, nonnegative_low, nonnegative_remote_low);
    exchange_vector(io, party, nonnegative_high, nonnegative_remote_high);
    bool nonnegative_ok = true;
    uint64_t nonnegative_mismatches = 0;
    for (size_t index = 0; index < nonnegative_expected.size(); ++index) {
      const __uint128_t local =
          (static_cast<__uint128_t>(nonnegative_high[index]) << 64) |
          nonnegative_low[index];
      const __uint128_t remote =
          (static_cast<__uint128_t>(nonnegative_remote_high[index]) << 64) |
          nonnegative_remote_low[index];
      const __uint128_t reconstructed = (local + remote) & extension_mask;
      if (reconstructed != nonnegative_expected[index]) {
        nonnegative_ok = false;
        ++nonnegative_mismatches;
      }
    }
    if (!gate(io, party, "C4N", nonnegative_ok)) {
      stop_after("C4N");
      return 1;
    }
    cout << "CONVERSION_TEST C4N detail=known_nonnegative_wide_extension"
         << " extension_bits=" << extension_bits
         << " max_valid_value=" << nonnegative_mask
         << " mismatches=" << nonnegative_mismatches << endl;
  }

  // C4: the complete MPC -> CKKS -> encrypted addition -> CKKS -> MPC
  // identity.  This is the production boundary, including per-RNS local
  // NTT construction and the uniform mask cancellation.
  vector<vector<HE_CKKS::scalar_t>> local_import(1);
  local_import[0] = c3_local_input;
  vector<Plaintext> imported = adapter.mpc_to_ckks(
      local_import, he.num_slots, conversion_prime, kBits, kScale,
      kCkksScale, &he, false);
  if (imported.size() != 1)
    throw runtime_error("C4 imported plaintext count mismatch");

  vector<Ciphertext> c4_ciphertexts(1);
  vector<Plaintext> c4_local_plaintext(1);
  if (party == sci::BOB) {
    he.encryptor->encrypt(imported[0], c4_ciphertexts[0]);
    send_encrypted_vector(&io, c4_ciphertexts);
  } else {
    recv_encrypted_vector(he.context, &io, c4_ciphertexts);
    he.evaluator->add_plain_inplace(c4_ciphertexts[0], imported[0]);
    he.evaluator->mod_switch_to_inplace(c4_ciphertexts[0],
                                        he.context->last_parms_id());
    adapter.mask_ckks_to_mpc_inplace(c4_ciphertexts, c4_local_plaintext, &he);
    send_encrypted_vector(&io, c4_ciphertexts);
  }
  if (party == sci::BOB) {
    recv_encrypted_vector(he.context, &io, c4_ciphertexts);
    he.decryptor->decrypt(c4_ciphertexts[0], c4_local_plaintext[0]);
  }

  const auto c4_local_output = adapter.ckks_to_mpc(
      c4_local_plaintext, kDegree, conversion_prime, kBits, kScale, &he, false);
  if (c4_local_output.size() != 1 ||
      c4_local_output[0].size() != he.num_slots)
    throw runtime_error("C4 output shape mismatch");
  vector<uint64_t> c4_remote_output;
  exchange_vector(io, party, c4_local_output[0], c4_remote_output);
  bool c4_ok = true;
  uint64_t c4_mismatches = 0;
  uint64_t c4_max_error = 0;
  for (size_t index = 0; index < he.num_slots; ++index) {
    const uint64_t reconstructed =
        add_mod_ring(c4_local_output[0][index], c4_remote_output[index]);
    const int64_t decoded_value = signed_ring(reconstructed);
    const int64_t reference = signed_ring(expected_fixed[index]);
    const int64_t error = decoded_value - reference;
    const uint64_t absolute = static_cast<uint64_t>(error < 0 ? -error : error);
    c4_max_error = max(c4_max_error, absolute);
    if (absolute > 8) {
      c4_ok = false;
      if (c4_mismatches < 20) {
        print_mismatch(
            index, prime_from_centered(reference, conversion_prime), reference,
            input_share0[index], input_share1[index], reconstructed,
            decoded_value, reference);
      }
      ++c4_mismatches;
    }
  }
  if (!gate(io, party, "C4", c4_ok)) {
    stop_after("C4");
    return 1;
  }
  cout << "CONVERSION_TEST C4 detail=full_CKKS_MPC_CKKS_identity"
       << " max_abs_error_raw=" << c4_max_error
       << " bad_values=" << c4_mismatches
       << " input_scale=" << kScale << " ckks_scale=" << kCkksScale
       << " fft_guard_bits=" << kEncodeGuardBits
       << " rns_limbs="
       << he.context->first_context_data()->parms().coeff_modulus().size()
       << endl;

  // The fixture above starts from mpc_to_ckks(), whose encoder and decoder
  // intentionally share the same convention.  A production observer starts
  // from Phantom/SEAL's native CKKSEncoder instead, so exercise that boundary
  // explicitly as well.  Keep it opt-in while isolating the representation
  // issue; once this gate is green it is folded into the mandatory C4 gate.
  if (getenv("OURS_VIT_CONVERSION_NATIVE_TEST") != nullptr) {
    // First inspect the native plaintext without any MPC operation.  This is
    // the exact representation handed to the observer conversion.  If this
    // direct native-plaintext FFT already fails, the bug is necessarily in
    // the root direction, normalization, or slot map rather than in shares.
    const auto native_coefficients = he.ckks_intt(native_plain);
    const int native_working_bits = 100;
    const __uint128_t native_working_mask =
        (__uint128_t{1} << native_working_bits) - 1;
    vector<HE_CKKS::wide_complex_t> native_wide(kDegree);
    uint64_t native_max_abs_coefficient = 0;
    for (size_t index = 0; index < native_coefficients.size(); ++index) {
      int64_t centered = 0;
      const uint64_t word =
          static_cast<uint64_t>(native_coefficients[index].real());
      memcpy(&centered, &word, sizeof(centered));
      const uint64_t magnitude = static_cast<uint64_t>(
          centered < 0 ? -(centered + 1) : centered);
      native_max_abs_coefficient =
          max(native_max_abs_coefficient, magnitude + (centered < 0 ? 1 : 0));
      __int128_t signed_value = static_cast<__int128_t>(centered);
      native_wide[index].real = static_cast<__uint128_t>(signed_value) &
                                native_working_mask;
    }
    const auto native_direct_output = he.ckks_decode_wide(
        native_wide, kDegree, kBits, kCkksScale, kScale, false,
        native_working_bits);
    double native_direct_max_error = 0.0;
    uint64_t native_direct_bad_values = 0;
    for (size_t index = 0; index < he.num_slots; ++index) {
      const int64_t decoded = signed_ring(native_direct_output[index]);
      const int64_t reference = signed_ring(expected_fixed[index]);
      const double error = abs(ldexp(static_cast<double>(decoded), -kScale) -
                              values[index]);
      native_direct_max_error = max(native_direct_max_error, error);
      if (abs(decoded - reference) > 8)
        ++native_direct_bad_values;
    }
    cout << "CONVERSION_TEST C4N_DIRECT_NATIVE_FFT"
         << " max_abs_value_error=" << native_direct_max_error
         << " bad_values=" << native_direct_bad_values << " slots="
         << native_direct_output.size() << " max_abs_coeff="
         << native_max_abs_coefficient << " probes=";
    for (size_t index : {size_t{0}, size_t{1}, he.num_slots / 2,
                         he.num_slots - 1}) {
      cout << (index == 0 ? "" : ",")
           << ldexp(static_cast<double>(signed_ring(native_direct_output[index])),
                    -kScale);
    }
    cout << endl;

    vector<complex<HE_CKKS::scalar_t>> native_decrypted_original_coefficients;
    vector<uint64_t> native_decrypted_original_ntt;
    if (party == sci::BOB) {
      Plaintext native_decrypted_original;
      he.decryptor->decrypt(c0_ciphertexts[0], native_decrypted_original);
      native_decrypted_original_ntt.assign(
          native_decrypted_original.data(),
          native_decrypted_original.data() + kDegree);
      native_decrypted_original_coefficients =
          he.ckks_intt(native_decrypted_original);
      cout << "CONVERSION_TEST C4N_FORM"
           << " native_plain_ntt=" << native_plain.is_ntt_form()
           << " ciphertext_ntt=" << c0_ciphertexts[0].is_ntt_form()
           << " decrypted_ntt=" << native_decrypted_original.is_ntt_form()
           << " decrypted_size=" << native_decrypted_original.coeff_count()
           << " parms_equal="
           << (native_plain.parms_id() == native_decrypted_original.parms_id())
           << " first=";
      for (size_t coefficient = 0; coefficient < 4; ++coefficient) {
        if (coefficient != 0) cout << ',';
        cout << native_plain.data()[coefficient] << ':'
             << native_decrypted_original.data()[coefficient];
      }
      cout << endl;
      uint64_t max_coeff_delta = 0;
      uint64_t max_signed_coeff_delta = 0;
      for (size_t index = 0; index < native_coefficients.size(); ++index) {
        int64_t lhs_signed = 0;
        int64_t rhs_signed = 0;
        const uint64_t lhs =
            static_cast<uint64_t>(native_coefficients[index].real());
        const uint64_t rhs =
            static_cast<uint64_t>(
                native_decrypted_original_coefficients[index].real());
        memcpy(&lhs_signed, &lhs, sizeof(lhs_signed));
        memcpy(&rhs_signed, &rhs, sizeof(rhs_signed));
        const uint64_t delta = lhs >= rhs ? lhs - rhs : rhs - lhs;
        max_coeff_delta = max(max_coeff_delta, delta);
        const int64_t signed_delta = rhs_signed - lhs_signed;
        const uint64_t signed_magnitude = static_cast<uint64_t>(
            signed_delta < 0 ? -(signed_delta + 1) : signed_delta);
        max_signed_coeff_delta = max(
            max_signed_coeff_delta,
            signed_magnitude + (signed_delta < 0 ? uint64_t{1} : uint64_t{0}));
      }
      cout << "CONVERSION_TEST C4N_NATIVE_DECRYPT_COEFF"
           << " max_raw_mod_2^64_delta=" << max_coeff_delta
           << " max_signed_delta=" << max_signed_coeff_delta << endl;
    }

    // Remove encryption noise from the diagnosis: use an exact additive
    // plaintext split (Alice=0, Bob=native_plain) and run only the secure
    // field-to-ring/FFT conversion.  This distinguishes a bad native
    // coefficient convention from a ciphertext-level noise/level issue.
    vector<double> zero_slots(he.num_slots, 0.0);
    Plaintext native_zero;
    he.encoder->encode(zero_slots, native_plain.parms_id(), native_plain.scale(),
                       native_zero);
    vector<Plaintext> native_plain_split(1);
    native_plain_split[0] = party == sci::ALICE ? native_zero : native_plain;
    const auto native_plain_local_output = adapter.ckks_to_mpc(
        native_plain_split, kDegree, he.get_Q(native_plain_split[0]), kBits,
        kScale, &he, false);
    vector<uint64_t> native_plain_remote_output;
    exchange_vector(io, party, native_plain_local_output[0],
                    native_plain_remote_output);
    bool native_plain_ok = true;
    uint64_t native_plain_mismatches = 0;
    uint64_t native_plain_max_error = 0;
    for (size_t index = 0; index < he.num_slots; ++index) {
      const uint64_t reconstructed = add_mod_ring(
          native_plain_local_output[0][index],
          native_plain_remote_output[index]);
      const int64_t decoded_value = signed_ring(reconstructed);
      const int64_t reference = signed_ring(expected_fixed[index]);
      const int64_t error = decoded_value - reference;
      const uint64_t absolute =
          static_cast<uint64_t>(error < 0 ? -error : error);
      native_plain_max_error = max(native_plain_max_error, absolute);
      if (absolute > 8) {
        native_plain_ok = false;
        if (native_plain_mismatches < 20) {
          print_mismatch(index,
                         prime_from_centered(reference, conversion_prime),
                         reference, input_share0[index], input_share1[index],
                         reconstructed, decoded_value, reference);
        }
        ++native_plain_mismatches;
      }
    }
    if (!gate(io, party, "C4P", native_plain_ok)) {
      stop_after("C4P");
      return 1;
    }
    cout << "CONVERSION_TEST C4P detail=native_plaintext_split"
         << " max_abs_error_raw=" << native_plain_max_error
         << " bad_values=" << native_plain_mismatches << endl;

    // Also use random additive shares in the native NTT representation.  The
    // zero/native split above does not exercise the prime-wrap branch with a
    // large local share; this one does, while remaining free of encryption
    // noise.
    const uint64_t native_prime = conversion_prime;
    vector<uint64_t> native_ntt_share0(kDegree), native_ntt_share1(kDegree);
    mt19937_64 native_share_random(0x5EEDC4AULL);
    for (size_t coefficient = 0; coefficient < kDegree; ++coefficient) {
      native_ntt_share0[coefficient] =
          static_cast<uint64_t>(native_share_random() % native_prime);
      const uint64_t expected = native_plain.data()[coefficient] % native_prime;
      native_ntt_share1[coefficient] =
          expected >= native_ntt_share0[coefficient]
              ? expected - native_ntt_share0[coefficient]
              : native_prime -
                    (native_ntt_share0[coefficient] - expected);
    }
    Plaintext native_ntt_plain0;
    Plaintext native_ntt_plain1;
    native_ntt_plain0.parms_id() = parms_id_zero;
    native_ntt_plain1.parms_id() = parms_id_zero;
    native_ntt_plain0.resize(kDegree);
    native_ntt_plain1.resize(kDegree);
    native_ntt_plain0.parms_id() = native_plain.parms_id();
    native_ntt_plain1.parms_id() = native_plain.parms_id();
    native_ntt_plain0.scale() = native_plain.scale();
    native_ntt_plain1.scale() = native_plain.scale();
    copy(native_ntt_share0.begin(), native_ntt_share0.end(),
         native_ntt_plain0.data());
    copy(native_ntt_share1.begin(), native_ntt_share1.end(),
         native_ntt_plain1.data());
    vector<Plaintext> native_random_split(1);
    native_random_split[0] =
        party == sci::ALICE ? native_ntt_plain0 : native_ntt_plain1;
    const auto native_random_local_output = adapter.ckks_to_mpc(
        native_random_split, kDegree, he.get_Q(native_random_split[0]), kBits,
        kScale, &he, false);
    vector<uint64_t> native_random_remote_output;
    exchange_vector(io, party, native_random_local_output[0],
                    native_random_remote_output);
    bool native_random_ok = true;
    uint64_t native_random_mismatches = 0;
    uint64_t native_random_max_error = 0;
    for (size_t index = 0; index < he.num_slots; ++index) {
      const uint64_t reconstructed = add_mod_ring(
          native_random_local_output[0][index],
          native_random_remote_output[index]);
      const int64_t decoded_value = signed_ring(reconstructed);
      const int64_t reference = signed_ring(expected_fixed[index]);
      const int64_t error = decoded_value - reference;
      const uint64_t absolute =
          static_cast<uint64_t>(error < 0 ? -error : error);
      native_random_max_error = max(native_random_max_error, absolute);
      if (absolute > 8) {
        native_random_ok = false;
        if (native_random_mismatches < 20) {
          print_mismatch(index,
                         prime_from_centered(reference, conversion_prime),
                         reference, native_ntt_share0[index],
                         native_ntt_share1[index], reconstructed, decoded_value,
                         reference);
        }
        ++native_random_mismatches;
      }
    }
    if (!gate(io, party, "C4R", native_random_ok)) {
      stop_after("C4R");
      return 1;
    }
    cout << "CONVERSION_TEST C4R detail=native_random_ntt_split"
         << " max_abs_error_raw=" << native_random_max_error
         << " bad_values=" << native_random_mismatches << endl;

    vector<Ciphertext> native_conversion_ciphertexts = c0_ciphertexts;
    vector<Plaintext> native_server_share(1);
    vector<uint64_t> native_negative_mask_coefficients;
    if (party == sci::ALICE) {
      adapter.mask_ckks_to_mpc_inplace(native_conversion_ciphertexts,
                                       native_server_share, &he);
      native_negative_mask_coefficients.assign(
          native_server_share[0].data(),
          native_server_share[0].data() + kDegree);
      send_encrypted_vector(&io, native_conversion_ciphertexts);
      io.send_data(native_negative_mask_coefficients.data(),
                   native_negative_mask_coefficients.size() * sizeof(uint64_t));
    } else {
      // Alice owns the ciphertext evaluator role and sends the masked
      // ciphertext.  Do not send Bob's original copy here: an unpaired
      // ciphertext would remain in Alice's receive stream and would be
      // consumed as the first SCI message below, making the conversion look
      // numerically non-additive.
    }
    if (party == sci::BOB) {
      recv_encrypted_vector(he.context, &io, native_conversion_ciphertexts);
      native_negative_mask_coefficients.resize(kDegree);
      io.recv_data(native_negative_mask_coefficients.data(),
                   native_negative_mask_coefficients.size() * sizeof(uint64_t));
      he.decryptor->decrypt(native_conversion_ciphertexts[0],
                            native_server_share[0]);

      const uint64_t q = conversion_prime;
      uint64_t mask_cancel_mismatches = 0;
      uint64_t max_mask_cancel_delta = 0;
      for (size_t coefficient = 0; coefficient < kDegree; ++coefficient) {
        const uint64_t reconstructed = add_mod_prime(
            native_server_share[0].data()[coefficient],
            native_negative_mask_coefficients[coefficient], q);
        // Decryption contains CKKS noise.  Compare mask cancellation with
        // the decrypted original ciphertext, not the noise-free plaintext.
        const uint64_t reference =
            native_decrypted_original_ntt[coefficient] % q;
        const uint64_t delta = reconstructed >= reference
                                   ? reconstructed - reference
                                   : reference - reconstructed;
        max_mask_cancel_delta = max(max_mask_cancel_delta, delta);
        if (delta != 0) ++mask_cancel_mismatches;
      }
      cout << "CONVERSION_TEST C4N_MASK_CANCEL"
           << " max_ntt_residue_delta=" << max_mask_cancel_delta
           << " mismatches=" << mask_cancel_mismatches << " first=";
      for (size_t coefficient = 0; coefficient < 4; ++coefficient) {
        const uint64_t reconstructed = add_mod_prime(
            native_server_share[0].data()[coefficient],
            native_negative_mask_coefficients[coefficient], q);
        if (coefficient != 0) cout << ',';
        cout << native_decrypted_original_ntt[coefficient] << ':'
             << reconstructed;
      }
      cout << endl;

      Plaintext negative_mask_plain;
      negative_mask_plain.parms_id() = parms_id_zero;
      negative_mask_plain.resize(kDegree);
      negative_mask_plain.parms_id() = native_plain.parms_id();
      negative_mask_plain.scale() = native_plain.scale();
      copy(native_negative_mask_coefficients.begin(),
           native_negative_mask_coefficients.end(), negative_mask_plain.data());
      const auto masked_decrypted_coefficients =
          he.ckks_intt(native_server_share[0]);
      const auto decrypted_mask_coefficients =
          he.ckks_intt(negative_mask_plain);
      uint64_t max_centered_mask_cancel_delta = 0;
      uint64_t centered_mask_cancel_mismatches = 0;
      uint64_t max_mask_delta_from_original = 0;
      uint64_t mask_delta_from_original_mismatches = 0;
      const uint64_t q_mask = conversion_prime;
      for (size_t coefficient = 0; coefficient < kDegree; ++coefficient) {
        const uint64_t decrypted_raw = static_cast<uint64_t>(
            masked_decrypted_coefficients[coefficient].real());
        const uint64_t mask_raw = static_cast<uint64_t>(
            decrypted_mask_coefficients[coefficient].real());
        const uint64_t native_raw = static_cast<uint64_t>(
            native_decrypted_original_coefficients[coefficient].real());
        const uint64_t reconstructed = add_mod_prime(
            decrypted_raw, raw_from_twos_complement(mask_raw, q_mask), q_mask);
        const uint64_t reference = raw_from_twos_complement(native_raw, q_mask);
        const uint64_t forward_delta = reconstructed >= reference
                                           ? reconstructed - reference
                                           : reference - reconstructed;
        const uint64_t circular_delta = min(forward_delta,
                                             q_mask - forward_delta);
        max_centered_mask_cancel_delta = max(
            max_centered_mask_cancel_delta, circular_delta);
        if (circular_delta > 1000) ++centered_mask_cancel_mismatches;

        const uint64_t masked_value = raw_from_twos_complement(
            static_cast<uint64_t>(masked_decrypted_coefficients[coefficient].real()),
            q_mask);
        const uint64_t original_value = raw_from_twos_complement(
            static_cast<uint64_t>(
                native_decrypted_original_coefficients[coefficient].real()),
            q_mask);
        const uint64_t mask_value = raw_from_twos_complement(mask_raw, q_mask);
        const uint64_t expected_masked_delta =
            mask_value == 0 ? 0 : q_mask - mask_value;
        const uint64_t actual_masked_delta =
            masked_value >= original_value
                ? masked_value - original_value
                : q_mask - (original_value - masked_value);
        const uint64_t mask_delta = actual_masked_delta >= expected_masked_delta
                                        ? actual_masked_delta - expected_masked_delta
                                        : expected_masked_delta - actual_masked_delta;
        const uint64_t circular_mask_delta =
            min(mask_delta, q_mask - mask_delta);
        max_mask_delta_from_original =
            max(max_mask_delta_from_original, circular_mask_delta);
        if (circular_mask_delta > 1000)
          ++mask_delta_from_original_mismatches;
      }
      cout << "CONVERSION_TEST C4N_MASK_CANCEL_COEFF"
           << " max_centered_delta=" << max_centered_mask_cancel_delta
           << " mismatches=" << centered_mask_cancel_mismatches
           << " max_mask_delta=" << max_mask_delta_from_original
           << " mask_delta_mismatches=" << mask_delta_from_original_mismatches
           << endl;
    }

    const auto native_local_output = adapter.ckks_to_mpc(
        native_server_share, kDegree, he.get_Q(native_server_share[0]),
        kBits, kScale, &he, false);
    vector<uint64_t> native_remote_output;
    exchange_vector(io, party, native_local_output[0], native_remote_output);
    bool native_conversion_ok = true;
    uint64_t native_conversion_mismatches = 0;
    uint64_t native_conversion_max_error = 0;
    for (size_t index = 0; index < he.num_slots; ++index) {
      const uint64_t reconstructed = add_mod_ring(
          native_local_output[0][index], native_remote_output[index]);
      const int64_t decoded_value = signed_ring(reconstructed);
      const int64_t reference = signed_ring(expected_fixed[index]);
      const int64_t error = decoded_value - reference;
      const uint64_t absolute =
          static_cast<uint64_t>(error < 0 ? -error : error);
      native_conversion_max_error =
          max(native_conversion_max_error, absolute);
      if (absolute > 8) {
        native_conversion_ok = false;
        if (native_conversion_mismatches < 20) {
          print_mismatch(
              index, prime_from_centered(reference, conversion_prime),
              reference, input_share0[index], input_share1[index],
              reconstructed, decoded_value, reference);
        }
        ++native_conversion_mismatches;
      }
    }
    cout << "CONVERSION_TEST C4N_RECON probes=";
    for (size_t index : {size_t{0}, size_t{1}, he.num_slots / 2,
                         he.num_slots - 1}) {
      const uint64_t reconstructed = add_mod_ring(
          native_local_output[0][index], native_remote_output[index]);
      cout << (index == 0 ? "" : ",") << signed_ring(reconstructed);
    }
    cout << endl;
    if (!gate(io, party, "C4N", native_conversion_ok)) {
      stop_after("C4N");
      return 1;
    }
    cout << "CONVERSION_TEST C4N detail=native_CKKS_to_MPC_identity"
         << " max_abs_error_raw=" << native_conversion_max_error
         << " bad_values=" << native_conversion_mismatches << endl;
  }

  // Optional post-gate arithmetic smoke.  C0-C4 deliberately stop at the
  // requested conversion identity, but Softmax also depends on the stronger
  // property that two independently converted additive shares can be
  // multiplied in CKKS and converted back.  Keep this diagnostic opt-in so
  // the required C0-C4 gate remains unchanged while making that boundary
  // failure reproducible without running ViT.
  if (getenv("OURS_VIT_CONVERSION_MUL_TEST") != nullptr) {
    HE_CKKS mul_he(party, &io, 8192, vector<int>{60, 40, 60}, 40, 40,
                   LOCATION::HOST);
    const vector<uint64_t> mul_expected = fixed_values(
        make_values(mul_he.num_slots));
    vector<uint64_t> mul_x0, mul_x1;
    make_shares(mul_expected, uint64_t{1} << kBits, mul_x0, mul_x1);
    vector<uint64_t> mul_y0, mul_y1;
    make_shares(mul_expected, uint64_t{1} << kBits, mul_y0, mul_y1);
    const vector<uint64_t> &local_x =
        party == sci::ALICE ? mul_x0 : mul_x1;
    const vector<uint64_t> &local_y =
        party == sci::ALICE ? mul_y0 : mul_y1;
    vector<vector<HE_CKKS::scalar_t>> x_share(1, local_x);
    vector<vector<HE_CKKS::scalar_t>> y_share(1, local_y);
    const uint64_t mul_prime = mul_he.get_conversion_prime();
    vector<Plaintext> x_plain = adapter.mpc_to_ckks(
        x_share, mul_he.num_slots, mul_prime, kBits, kScale, 40,
        &mul_he, false);
    vector<Plaintext> y_plain = adapter.mpc_to_ckks(
        y_share, mul_he.num_slots, mul_prime, kBits, kScale, 40,
        &mul_he, false);
    vector<Ciphertext> product_ciphertexts(1);
    vector<Plaintext> product_server_share(1);
    vector<Ciphertext> x_ciphertexts(1), y_ciphertexts(1);
    if (party == sci::BOB) {
      mul_he.encryptor->encrypt(x_plain[0], x_ciphertexts[0]);
      mul_he.encryptor->encrypt(y_plain[0], y_ciphertexts[0]);
      send_encrypted_vector(&io, x_ciphertexts);
      send_encrypted_vector(&io, y_ciphertexts);
    } else {
      recv_encrypted_vector(mul_he.context, &io, x_ciphertexts);
      recv_encrypted_vector(mul_he.context, &io, y_ciphertexts);
      mul_he.evaluator->add_plain_inplace(x_ciphertexts[0], x_plain[0]);
      mul_he.evaluator->add_plain_inplace(y_ciphertexts[0], y_plain[0]);
      mul_he.evaluator->multiply_inplace(x_ciphertexts[0], y_ciphertexts[0]);
      mul_he.evaluator->relinearize_inplace(x_ciphertexts[0],
                                            *mul_he.relin_keys);
      mul_he.evaluator->rescale_to_next_inplace(x_ciphertexts[0]);
      product_ciphertexts[0] = std::move(x_ciphertexts[0]);
      adapter.mask_ckks_to_mpc_inplace(product_ciphertexts,
                                       product_server_share, &mul_he);
      send_encrypted_vector(&io, product_ciphertexts);
    }
    if (party == sci::BOB) {
      recv_encrypted_vector(mul_he.context, &io, product_ciphertexts);
      mul_he.decryptor->decrypt(product_ciphertexts[0], product_server_share[0]);
    }
    const auto product_local = adapter.ckks_to_mpc(
        product_server_share, 8192, mul_he.get_Q(product_server_share[0]),
        kBits, kScale, &mul_he, false);
    vector<uint64_t> product_remote;
    exchange_vector(io, party, product_local[0], product_remote);
    bool mul_ok = true;
    uint64_t mul_mismatches = 0;
    uint64_t mul_max_error = 0;
    for (size_t index = 0; index < mul_he.num_slots; ++index) {
      const int64_t x_value = signed_ring(mul_expected[index]);
      const int64_t y_value = signed_ring(mul_expected[index]);
      const int64_t expected_product = (x_value * y_value) >> kScale;
      const uint64_t reconstructed = add_mod_ring(
          product_local[0][index], product_remote[index]);
      const int64_t decoded_product = signed_ring(reconstructed);
      const int64_t error = decoded_product - expected_product;
      const uint64_t absolute =
          static_cast<uint64_t>(error < 0 ? -error : error);
      mul_max_error = max(mul_max_error, absolute);
      if (absolute > 32) {
        mul_ok = false;
        if (mul_mismatches < 20) {
          print_mismatch(
              index, prime_from_centered(expected_product, mul_prime),
              expected_product, mul_x0[index], mul_x1[index], reconstructed,
              decoded_product, expected_product);
        }
        ++mul_mismatches;
      }
    }
    if (!gate(io, party, "C4M", mul_ok)) {
      stop_after("C4M");
      return 1;
    }
    cout << "CONVERSION_TEST C4M detail=CKKS_multiply_boundary"
         << " max_abs_error_raw=" << mul_max_error
         << " bad_values=" << mul_mismatches << endl;
  }

  // Optional post-gate exponentiation smoke.  This is the exact BLB HE
  // polynomial used by Softmax, but on a one-ciphertext fixture so a failure
  // cannot be hidden by the later row reduction or reciprocal.  The required
  // C0-C4 conversion gates remain unchanged and still stop before this block.
  if (getenv("OURS_VIT_CONVERSION_EXP_TEST") != nullptr) {
    HE_CKKS exp_he_context(
        party, &io, 32768,
        vector<int>{60, 40, 40, 40, 40, 40, 40, 40, 60}, 40, 40,
        LOCATION::HOST);
    adapter.linear.he_softmax = &exp_he_context;
    vector<uint64_t> exp_expected(exp_he_context.num_slots);
    const array<double, 15> exp_fixture = {
        -32.0, -13.25, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0,
        -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0};
    for (size_t index = 0; index < exp_expected.size(); ++index) {
      const double value = exp_fixture[index % exp_fixture.size()];
      exp_expected[index] = ring_from_signed(static_cast<int64_t>(llround(
          value * ldexp(1.0, kScale))));
    }
    vector<uint64_t> exp_share0, exp_share1;
    make_shares(exp_expected, uint64_t{1} << kBits, exp_share0, exp_share1);
    FixArray exp_local(party, static_cast<int>(exp_expected.size()), true,
                       kBits, kScale);
    const vector<uint64_t> &exp_share =
        party == sci::ALICE ? exp_share0 : exp_share1;
    copy(exp_share.begin(), exp_share.end(), exp_local.data);
    const FixArray exp_output = adapter.exp_he(exp_local);
    vector<uint64_t> exp_remote;
    exchange_vector(io, party,
                    vector<uint64_t>(exp_output.data,
                                     exp_output.data + exp_output.size),
                    exp_remote);
    bool exp_ok = true;
    uint64_t exp_mismatches = 0;
    uint64_t exp_max_error = 0;
    for (size_t index = 0; index < exp_expected.size(); ++index) {
      const uint64_t reconstructed = add_mod_ring(
          exp_output.data[index], exp_remote[index]);
      const int64_t decoded_value = signed_ring(reconstructed);
      const double input_value =
          ldexp(static_cast<double>(signed_ring(exp_expected[index])),
                -kScale);
      const int64_t reference = input_value < -13.0 ? 0 :
          static_cast<int64_t>(llround(pow(1.0 + input_value/64.0, 64) * ldexp(1.0, kScale)));
      const int64_t error = decoded_value - reference;
      const uint64_t absolute =
          static_cast<uint64_t>(error < 0 ? -error : error);
      exp_max_error = max(exp_max_error, absolute);
      if (absolute > 32) {
        exp_ok = false;
        if (exp_mismatches < 20) {
          print_mismatch(
              index, prime_from_centered(reference, exp_he_context.get_conversion_prime()),
              reference, exp_share0[index], exp_share1[index], reconstructed,
              decoded_value, reference);
        }
        ++exp_mismatches;
      }
    }
    if (!gate(io, party, "C4E", exp_ok)) {
      stop_after("C4E");
      return 1;
    }
    cout << "CONVERSION_TEST C4E detail=CKKS_exp_polynomial_boundary"
         << " max_abs_error_raw=" << exp_max_error
         << " bad_values=" << exp_mismatches << endl;
  }

  // Optional Softmax denominator smoke.  This isolates the MPC reciprocal
  // square-root and the one intentional fixed-point multiplication used by
  // div_batch_he_V2 from CKKS packing, slot broadcast, and HE ciphertext
  // multiplication.  The fixture spans the denominator range of a 256-way
  // exp sum, including values that exercise both the normalization and the
  // fixed-point truncation boundaries.
  if (getenv("OURS_VIT_CONVERSION_RECIP_TEST") != nullptr) {
    const vector<double> denominator_values = {
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,
        256.0, 512.0, 1024.0, 2048.0, 4096.0,
        // These are representative 256-way exp sums observed in B.  Keep
        // non-powers here so the reciprocal gate exercises the same
        // normalization interval as production Softmax, not only exact
        // binary boundaries.
        4393.0, 4827.0, 5456.0, 5921.0,
        1.25, 3.75, 127.5, 4827.375};
    const size_t denominator_count = denominator_values.size();
    vector<uint64_t> denominator_expected(denominator_count);
    for (size_t index = 0; index < denominator_count; ++index) {
      denominator_expected[index] = ring_from_signed(static_cast<int64_t>(
          llround(denominator_values[index] * ldexp(1.0, kScale))));
    }
    vector<uint64_t> denominator_share0, denominator_share1;
    make_shares(denominator_expected, uint64_t{1} << kBits,
                denominator_share0, denominator_share1);
    FixArray denominator_local(party, static_cast<int>(denominator_count),
                               false, kBits, kScale);
    const vector<uint64_t> &denominator_share =
        party == sci::ALICE ? denominator_share0 : denominator_share1;
    copy(denominator_share.begin(), denominator_share.end(),
         denominator_local.data);

    FPMath *reciprocal_math = adapter.nonlinear.fpmath[0];
    FixArray reciprocal_sqrt =
        reciprocal_math->sqrt(denominator_local, true);
    FixArray reciprocal_square = reciprocal_math->fix->mul(
        reciprocal_sqrt, reciprocal_sqrt,
        reciprocal_sqrt.ell + reciprocal_sqrt.s);
    reciprocal_square = reciprocal_math->fix->truncate_reduce(
        reciprocal_square, reciprocal_sqrt.s);
    FixArray opened_sqrt =
        reciprocal_math->fix->output(sci::PUBLIC, reciprocal_sqrt);
    FixArray opened_square =
        reciprocal_math->fix->output(sci::PUBLIC, reciprocal_square);

    bool reciprocal_ok = true;
    uint64_t reciprocal_mismatches = 0;
    uint64_t reciprocal_max_error = 0;
    for (size_t index = 0; index < denominator_count; ++index) {
      const uint64_t sqrt_raw = opened_sqrt.data[index] &
                                 opened_sqrt.ell_mask();
      const uint64_t square_raw = opened_square.data[index] &
                                  opened_square.ell_mask();
      const int64_t expected_sqrt = static_cast<int64_t>(llround(
          ldexp(1.0, kScale) / sqrt(denominator_values[index])));
      const int64_t expected_square = static_cast<int64_t>(llround(
          ldexp(1.0, kScale) / denominator_values[index]));
      const int64_t actual_sqrt = static_cast<int64_t>(sqrt_raw);
      const int64_t actual_square = static_cast<int64_t>(square_raw);
      const int64_t sqrt_error = actual_sqrt - expected_sqrt;
      const int64_t square_error = actual_square - expected_square;
      const uint64_t absolute_sqrt = static_cast<uint64_t>(
          sqrt_error < 0 ? -sqrt_error : sqrt_error);
      const uint64_t absolute_square = static_cast<uint64_t>(
          square_error < 0 ? -square_error : square_error);
      reciprocal_max_error = max(
          reciprocal_max_error, max(absolute_sqrt, absolute_square));
      if (absolute_sqrt > 128 || absolute_square > 128) {
        reciprocal_ok = false;
        if (reciprocal_mismatches < 20) {
          print_mismatch(
              index, denominator_expected[index],
              static_cast<int64_t>(denominator_expected[index]),
              denominator_share0[index], denominator_share1[index],
              denominator_expected[index], actual_square, expected_square);
        }
        ++reciprocal_mismatches;
      }
    }
    if (!gate(io, party, "C4S", reciprocal_ok)) {
      stop_after("C4S");
      return 1;
    }
    cout << "CONVERSION_TEST C4S detail=secure_reciprocal_sqrt_square"
         << " max_abs_error_raw=" << reciprocal_max_error
         << " bad_values=" << reciprocal_mismatches << endl;
  }

  // Optional end-to-end reciprocal/broadcast smoke.  Unlike C4S, this calls
  // the exact div_batch_he_V2 implementation, including conversion of both
  // operands to CKKS, one full-slot HE product, mask cancellation, and the
  // row-major denominator broadcast.  Three ciphertexts exercise the same
  // multi-ciphertext indexing contract used by the 12-head Softmax tensor.
  if (getenv("OURS_VIT_CONVERSION_DIV_TEST") != nullptr) {
    const int kDivisionRows =
        getenv("OURS_VIT_CONVERSION_DIV_LARGE_TEST") != nullptr ? 3072 : 48;
    constexpr int kDivisionBatch = 256;
    const array<double, 24> division_denominators = {
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,
        256.0, 512.0, 1024.0, 2048.0, 4096.0,
        4393.0, 4827.0, 5456.0, 5921.0, 1.25, 3.75, 127.5,
        4827.375, 5179.625, 6011.125, 7003.875};
    vector<uint64_t> division_denominator_expected(kDivisionRows);
    for (int row = 0; row < kDivisionRows; ++row) {
      division_denominator_expected[row] = ring_from_signed(
          static_cast<int64_t>(llround(
              division_denominators[row % division_denominators.size()] *
              ldexp(1.0, kScale))));
    }
    vector<uint64_t> division_numerator_expected(
        static_cast<size_t>(kDivisionRows) * kDivisionBatch);
    for (int row = 0; row < kDivisionRows; ++row) {
      for (int key = 0; key < kDivisionBatch; ++key) {
        // Softmax tails reach raw fixed-point values 1--16 at scale 12.
        // Keep those values in this gate: a normal-sized numerator test can
        // pass while the CKKS product still erases the low-probability tail.
        const int pattern = key % 13;
        const double numerator =
            pattern == 0 ? 0.0 :
            pattern == 1 ? 1.0 / 4096.0 :
            pattern == 2 ? 2.0 / 4096.0 :
            pattern == 3 ? 4.0 / 4096.0 :
            pattern == 4 ? 8.0 / 4096.0 :
            pattern == 5 ? 16.0 / 4096.0 :
            pattern == 6 ? 0.125 :
            pattern == 7 ? 0.5 : 1.0;
        division_numerator_expected[
            static_cast<size_t>(row) * kDivisionBatch + key] =
            ring_from_signed(static_cast<int64_t>(llround(
                numerator * ldexp(1.0, kScale))));
      }
    }
    vector<uint64_t> division_dn0, division_dn1;
    vector<uint64_t> division_nm0, division_nm1;
    make_shares(division_denominator_expected, uint64_t{1} << kBits,
                division_dn0, division_dn1);
    make_shares(division_numerator_expected, uint64_t{1} << kBits,
                division_nm0, division_nm1);
    FixArray division_dn_local(party, kDivisionRows, false, kBits, kScale);
    FixArray division_nm_local(
        party, kDivisionRows * kDivisionBatch, false, kBits, kScale);
    copy((party == sci::ALICE ? division_dn0 : division_dn1).begin(),
         (party == sci::ALICE ? division_dn0 : division_dn1).end(),
         division_dn_local.data);
    copy((party == sci::ALICE ? division_nm0 : division_nm1).begin(),
         (party == sci::ALICE ? division_nm0 : division_nm1).end(),
         division_nm_local.data);

    HE_CKKS division_he(party, &io, 8192, vector<int>{60, 40, 60}, 40, 40,
                        LOCATION::HOST);
    adapter.linear.he_onetime = &division_he;
    const FixArray division_output = adapter.div_batch_he_V2(
        division_nm_local, division_dn_local, kDivisionBatch, kBits, kScale,
        false, false, false);
    vector<uint64_t> division_remote;
    exchange_vector(io, party,
                    vector<uint64_t>(
                        division_output.data,
                        division_output.data + division_output.size),
                    division_remote);
    bool division_ok = true;
    uint64_t division_mismatches = 0;
    uint64_t division_max_error = 0;
    for (int row = 0; row < kDivisionRows; ++row) {
      const double denominator =
          division_denominators[row % division_denominators.size()];
      const int64_t reciprocal = static_cast<int64_t>(llround(
          ldexp(1.0, kScale) / denominator));
      for (int key = 0; key < kDivisionBatch; ++key) {
        const size_t index = static_cast<size_t>(row) * kDivisionBatch + key;
        const int64_t numerator = signed_ring(division_numerator_expected[index]);
        const int64_t reference = (numerator * reciprocal) >> kScale;
        const uint64_t reconstructed = add_mod_ring(
            division_output.data[index], division_remote[index]);
        const int64_t decoded = signed_ring(reconstructed);
        const int64_t error = decoded - reference;
        const uint64_t absolute = static_cast<uint64_t>(
            error < 0 ? -error : error);
        division_max_error = max(division_max_error, absolute);
        if (absolute > 128) {
          division_ok = false;
          if (division_mismatches < 20) {
            print_mismatch(
                index, division_denominator_expected[row],
                static_cast<int64_t>(division_denominator_expected[row]),
                division_nm0[index], division_nm1[index], reconstructed,
                decoded, reference);
          }
          ++division_mismatches;
        }
      }
    }
    if (!gate(io, party, "C4D", division_ok)) {
      stop_after("C4D");
      return 1;
    }
    cout << "CONVERSION_TEST C4D detail=secure_division_broadcast"
         << " rows=" << kDivisionRows << " batch=" << kDivisionBatch
         << " max_abs_error_raw=" << division_max_error
         << " bad_values=" << division_mismatches << endl;
  }

  // Optional signed-comparison smoke.  B's row maximum and GELU selector
  // depend on the fork's actual signed comparator orientation; keep this
  // small gate separate from C0-C4 so the conversion contract remains the
  // first mandatory stop point while comparator regressions are observable.
  if (getenv("OURS_VIT_CONVERSION_COMPARISON_TEST") != nullptr) {
    const vector<int64_t> lhs_values = {
        -32, -13, -1, 0, 1, 2, 13, 32};
    const vector<int64_t> rhs_values = {
        -31, -13, 0, 0, -1, 3, 12, 33};
    const size_t comparison_count = lhs_values.size();
    vector<uint64_t> lhs_expected(comparison_count);
    vector<uint64_t> rhs_expected(comparison_count);
    for (size_t index = 0; index < comparison_count; ++index) {
      lhs_expected[index] = ring_from_signed(
          lhs_values[index] * (int64_t{1} << kScale));
      rhs_expected[index] = ring_from_signed(
          rhs_values[index] * (int64_t{1} << kScale));
    }
    vector<uint64_t> lhs0, lhs1, rhs0, rhs1;
    make_shares(lhs_expected, uint64_t{1} << kBits, lhs0, lhs1);
    make_shares(rhs_expected, uint64_t{1} << kBits, rhs0, rhs1);
    FixArray lhs_local(party, static_cast<int>(comparison_count), true,
                       kBits, kScale);
    FixArray rhs_local(party, static_cast<int>(comparison_count), true,
                       kBits, kScale);
    copy((party == sci::ALICE ? lhs0 : lhs1).begin(),
         (party == sci::ALICE ? lhs0 : lhs1).end(), lhs_local.data);
    copy((party == sci::ALICE ? rhs0 : rhs1).begin(),
         (party == sci::ALICE ? rhs0 : rhs1).end(), rhs_local.data);
    FPMath *comparison_math = adapter.nonlinear.fpmath[0];
    BoolArray lhs_is_ge = comparison_math->fix->LE(lhs_local, rhs_local);
    FixArray selected = comparison_math->fix->if_else(
        lhs_is_ge, lhs_local, rhs_local);
    FixArray opened = comparison_math->fix->output(sci::PUBLIC, selected);
    bool comparison_ok = true;
    uint64_t comparison_mismatches = 0;
    uint64_t comparison_max_error = 0;
    for (size_t index = 0; index < comparison_count; ++index) {
      const int64_t expected = max(lhs_values[index], rhs_values[index]) *
                               (int64_t{1} << kScale);
      const int64_t actual = signed_ring(opened.data[index]);
      const int64_t error = actual - expected;
      const uint64_t absolute = static_cast<uint64_t>(
          error < 0 ? -error : error);
      comparison_max_error = max(comparison_max_error, absolute);
      if (absolute != 0) {
        comparison_ok = false;
        if (comparison_mismatches < 20) {
          print_mismatch(
              index, ring_from_signed(expected),
              expected, lhs0[index], lhs1[index],
              static_cast<uint64_t>(actual) & kRingMask, actual, expected);
        }
        ++comparison_mismatches;
      }
    }
    if (!gate(io, party, "C4T", comparison_ok)) {
      stop_after("C4T");
      return 1;
    }
    cout << "CONVERSION_TEST C4T detail=signed_comparison_max"
         << " max_abs_error_raw=" << comparison_max_error
         << " bad_values=" << comparison_mismatches << endl;
  }
  cout << "CONVERSION_TEST status=PASS gates=C0,C1,C2,C3,C4" << endl;
  return 0;
}
