#include "he.h"
#include "FloatingPoint/fp-math.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sci;
using namespace std;

namespace {

HE_CKKS::wide_scalar_t sign_extend_64(uint64_t value, int width) {
  const auto mask = (HE_CKKS::wide_scalar_t{1} << width) - 1;
  HE_CKKS::wide_scalar_t result = value;
  if (value >> 63) {
    result |= mask ^ ((HE_CKKS::wide_scalar_t{1} << 64) - 1);
  }
  return result & mask;
}

HE_CKKS::wide_scalar_t sign_extend_bits(uint64_t value, int from_bits,
                                        int to_bits) {
  const auto from_mask = (uint64_t{1} << from_bits) - 1;
  const auto to_mask = (HE_CKKS::wide_scalar_t{1} << to_bits) - 1;
  HE_CKKS::wide_scalar_t result = value & from_mask;
  if ((value >> (from_bits - 1)) & 1) {
    result |= to_mask ^ from_mask;
  }
  return result & to_mask;
}

} // namespace

int main(int argc, char **argv) {
  int party = 0;
  int port = 0;
  string address = "127.0.0.1";
  ArgMapping amap;
  amap.arg("r", party, "party");
  amap.arg("p", port, "port");
  amap.arg("ip", address, "address");
  amap.parse(argc, argv);
  if (party != sci::ALICE && party != sci::BOB || port <= 0) {
    throw invalid_argument("usage: r=1|2 p=PORT [ip=ADDRESS]");
  }

  sci::NetIO io(party == sci::ALICE ? nullptr : address.c_str(), port);
  HE_CKKS he(party, &io, 32768,
            vector<int>{60, 35, 60, 60, 60, 60, 60, 60}, 20, 44,
            LOCATION::HOST);
  IOPack iopack(party, port + 1, address);
  OTPack otpack(&iopack, party);
  FPMath fpmath(party, &iopack, &otpack);

  const int degree = 32768;
  const int slots = degree / 2;
  constexpr int ring_bits = 60;
  constexpr int wide_bits = ring_bits + 40;
  const uint64_t input_mask = (uint64_t{1} << 43) - 1;
  vector<uint64_t> input(static_cast<size_t>(slots), 0);
  for (int i = 0; i < slots; ++i) {
    const int64_t value = (i % 7 == 0) ? -3 : ((i % 11) - 5);
    input[i] = static_cast<uint64_t>(value * (int64_t{1} << 20));
  }

  const auto encoded = he.ckks_encode(input, slots, false);

  vector<HE_CKKS::complex_t> old_input(static_cast<size_t>(degree));
  vector<HE_CKKS::wide_complex_t> wide_input(static_cast<size_t>(degree));
  for (int i = 0; i < degree; ++i) {
    old_input[i] = encoded[i];
    wide_input[i].real = sign_extend_64(encoded[i], wide_bits);
    wide_input[i].imag = 0;
  }
  const auto old_decoded = he.ckks_decode(old_input, degree, false);
  const auto wide_decoded =
      he.ckks_decode_wide(wide_input, degree, 43, 20, 20, false,
                          wide_bits);

  // Reproduce the production field-to-ring -> signed-extension boundary on
  // a public synthetic share.  The value is exchanged only after the local
  // diagnostic transform; no production/private tensor is used.
  vector<uint64_t> wide_low(static_cast<size_t>(degree));
  vector<uint64_t> wide_high(static_cast<size_t>(degree));
  const HE_CKKS::wide_scalar_t wide_mask =
      (HE_CKKS::wide_scalar_t{1} << wide_bits) - 1;
  constexpr uint64_t diagnostic_prime = 1152921504592822273ULL;
  constexpr uint64_t field_prime = diagnostic_prime;
  const uint64_t field_center = (diagnostic_prime - 1) / 2;
  const uint64_t ring_mask = (uint64_t{1} << ring_bits) - 1;
  vector<uint64_t> field_share(static_cast<size_t>(degree));
  vector<uint64_t> ring_share(static_cast<size_t>(degree));
  vector<uint64_t> ring_low(static_cast<size_t>(degree));
  vector<uint64_t> ring_high(static_cast<size_t>(degree));
  for (int i = 0; i < degree; ++i) {
    int64_t signed_encoded = 0;
    memcpy(&signed_encoded, &encoded[i], sizeof(signed_encoded));
    const uint64_t magnitude = signed_encoded < 0
                                   ? static_cast<uint64_t>(-(signed_encoded + 1)) + 1
                                   : static_cast<uint64_t>(signed_encoded);
    const uint64_t reduced = magnitude % diagnostic_prime;
    const uint64_t secret = signed_encoded < 0
                                ? (reduced == 0 ? 0 : diagnostic_prime - reduced)
                                : reduced;
    const uint64_t alice_mask =
        (uint64_t{0x9e3779b97f4a7c15ULL} * (i + 17)) % diagnostic_prime;
    field_share[i] = party == sci::ALICE
                         ? (diagnostic_prime - alice_mask) % diagnostic_prime
                         : (secret + alice_mask) % diagnostic_prime;
    if (party == sci::ALICE) {
      field_share[i] = (field_share[i] + field_center) % diagnostic_prime;
    }
  }
  FixArray field_input = fpmath.fix->input(
      party, degree, field_share.data(), false, 60, 20);
  FixArray ring_output = fpmath.fix->field2ring(field_input, diagnostic_prime,
                                                ring_bits);
  copy(ring_output.data, ring_output.data + degree, ring_share.begin());
  if (party == sci::ALICE) {
    for (uint64_t &value : ring_share) value = (value - field_center) & ring_mask;
  }
  fpmath.fix->xt->s_extend_wide(
      degree, ring_share.data(), ring_low.data(), ring_high.data(), ring_bits,
      wide_bits);
  for (int i = 0; i < degree; ++i) {
    const HE_CKKS::wide_scalar_t wide_share =
        (static_cast<HE_CKKS::wide_scalar_t>(ring_high[i]) << 64) |
        ring_low[i];
    wide_low[i] = static_cast<uint64_t>(wide_share);
    wide_high[i] = static_cast<uint64_t>(wide_share >> 64);
  }
  vector<HE_CKKS::wide_complex_t> shared_wide(static_cast<size_t>(degree));
  vector<HE_CKKS::wide_complex_t> public_wide(static_cast<size_t>(degree));
  for (int i = 0; i < degree; ++i) {
    shared_wide[i].real =
        (static_cast<HE_CKKS::wide_scalar_t>(wide_high[i]) << 64) |
        wide_low[i];
    public_wide[i].real = sign_extend_bits(encoded[i], 43, wide_bits);
  }
  const auto shared_decoded =
      he.ckks_decode_wide(shared_wide, degree, 43, 20, 20, false,
                          wide_bits);
  const auto public_decoded =
      he.ckks_decode_wide(public_wide, degree, 43, 20, 20, false,
                          wide_bits);
  vector<uint64_t> opened(shared_decoded.size());
  if (party == sci::ALICE) {
    io.send_data(shared_decoded.data(), shared_decoded.size() * sizeof(uint64_t));
    copy(shared_decoded.begin(), shared_decoded.end(), opened.begin());
  } else {
    vector<uint64_t> other(shared_decoded.size());
    io.recv_data(other.data(), other.size() * sizeof(uint64_t));
    for (size_t i = 0; i < opened.size(); ++i) {
      opened[i] = (shared_decoded[i] + other[i]) & input_mask;
    }
  }
  if (party == sci::ALICE) {
    vector<uint64_t> other(shared_decoded.size());
    io.recv_data(other.data(), other.size() * sizeof(uint64_t));
    for (size_t i = 0; i < opened.size(); ++i) {
      opened[i] = (opened[i] + other[i]) & input_mask;
    }
  } else {
    // Bob's share was sent by the other branch above.
    io.send_data(shared_decoded.data(), shared_decoded.size() * sizeof(uint64_t));
  }
  uint64_t shared_ring_mismatch = 0;
  int shared_ring_nonzero = 0;
  uint64_t shared_ring_max_abs = 0;
  const uint64_t signed_output_modulus = uint64_t{1} << 43;
  const uint64_t signed_output_half = uint64_t{1} << 42;
  for (int i = 0; i < slots; ++i) {
    shared_ring_mismatch =
        max(shared_ring_mismatch, opened[i] ^ public_decoded[i]);
    if (opened[i] != public_decoded[i]) ++shared_ring_nonzero;
    uint64_t delta = (opened[i] - public_decoded[i]) & input_mask;
    if (delta >= signed_output_half) delta = signed_output_modulus - delta;
    shared_ring_max_abs = max(shared_ring_max_abs, delta);
  }

  uint64_t old_wide_mismatch = 0;
  uint64_t roundtrip_mismatch = 0;
  for (int i = 0; i < slots; ++i) {
    old_wide_mismatch = max(
        old_wide_mismatch, (old_decoded[i] & input_mask) ^ wide_decoded[i]);
    roundtrip_mismatch = max(roundtrip_mismatch,
                             (input[i] & input_mask) ^ wide_decoded[i]);
  }

  // Exercise the actual production scale transition.  The old diagnostic
  // used s=20, where the 43-bit ring values are tiny; the BERT path enters
  // this transform at s=40 and exits at s=13/11.
  vector<uint64_t> high_input(static_cast<size_t>(slots), 0);
  for (int i = 0; i < slots; ++i) {
    const int64_t coefficient = (i % 7 == 0) ? -3 : ((i % 11) - 5);
    high_input[i] = static_cast<uint64_t>(coefficient * (int64_t{1} << 38));
  }
  const auto high_encoded = he.ckks_encode(high_input, slots, false);
  vector<HE_CKKS::wide_complex_t> high_public(static_cast<size_t>(degree));
  for (int i = 0; i < degree; ++i) {
    high_public[i].real = sign_extend_bits(high_encoded[i], 43, wide_bits);
  }
  const auto high_decoded =
      he.ckks_decode_wide(high_public, degree, 43, 40, 13, false,
                          wide_bits);
  vector<HE_CKKS::wide_complex_t> pre_shift_public(
      static_cast<size_t>(degree));
  for (int i = 0; i < degree; ++i) {
    int64_t signed_high = 0;
    memcpy(&signed_high, &high_encoded[i], sizeof(signed_high));
    const int64_t shifted = signed_high >> 27;
    pre_shift_public[i].real = sign_extend_bits(
        static_cast<uint64_t>(shifted), 43, wide_bits);
  }
  const auto pre_shift_decoded =
      he.ckks_decode_wide(pre_shift_public, degree, 43, 13, 13, false,
                          wide_bits);

  // Exercise the complete production C2M boundary on a public synthetic
  // CKKS polynomial: canonical field shares -> centered field-to-ring ->
  // signed secure scale truncation -> l=43 -> Appendix-C.4 wide local FFT.
  // This deliberately uses a fresh mask and never touches a model tensor.
  vector<uint64_t> c2m_field_share(static_cast<size_t>(degree));
  vector<uint64_t> c2m_ring_share(static_cast<size_t>(degree));
  vector<uint64_t> c2m_shifted_share(static_cast<size_t>(degree));
  vector<uint64_t> c2m_low(static_cast<size_t>(degree));
  vector<uint64_t> c2m_high(static_cast<size_t>(degree));
  for (int i = 0; i < degree; ++i) {
    int64_t signed_encoded = 0;
    memcpy(&signed_encoded, &high_encoded[i], sizeof(signed_encoded));
    const uint64_t magnitude =
        signed_encoded < 0
            ? static_cast<uint64_t>(-(signed_encoded + 1)) + 1
            : static_cast<uint64_t>(signed_encoded);
    const uint64_t reduced = magnitude % field_prime;
    const uint64_t secret =
        signed_encoded < 0 ? (reduced == 0 ? 0 : field_prime - reduced)
                           : reduced;
    const uint64_t alice_mask =
        (uint64_t{0xd1b54a32d192ed03ULL} * (i + 23)) % field_prime;
    c2m_field_share[i] =
        party == sci::ALICE
            ? alice_mask
            : (secret + field_prime - alice_mask) % field_prime;
    if (party == sci::ALICE) {
      c2m_field_share[i] = static_cast<uint64_t>(
          (static_cast<__uint128_t>(c2m_field_share[i]) + field_center) %
          field_prime);
    }
  }
  FixArray c2m_field = fpmath.fix->input(
      party, degree, c2m_field_share.data(), false, 60, 40);
  FixArray c2m_centered =
      fpmath.fix->field2ring(c2m_field, field_prime, 60);
  const uint64_t ring60_mask = (uint64_t{1} << 60) - 1;
  copy(c2m_centered.data, c2m_centered.data + degree, c2m_ring_share.begin());
  if (party == sci::ALICE) {
    for (auto &value : c2m_ring_share)
      value = (value - (field_center & ring60_mask)) & ring60_mask;
  }
  FixArray c2m_ring = fpmath.fix->input(
      party, degree, c2m_ring_share.data(), true, 60, 40);
  c2m_ring = fpmath.fix->right_shift(c2m_ring, 27);
  c2m_ring = fpmath.fix->reduce(c2m_ring, 43);
  copy(c2m_ring.data, c2m_ring.data + degree, c2m_shifted_share.begin());
  fpmath.fix->xt->s_extend_wide(
      degree, c2m_shifted_share.data(), c2m_low.data(), c2m_high.data(),
      43, 83);
  vector<HE_CKKS::wide_complex_t> c2m_wide(static_cast<size_t>(degree));
  for (int i = 0; i < degree; ++i) {
    c2m_wide[i].real =
        (static_cast<HE_CKKS::wide_scalar_t>(c2m_high[i]) << 64) |
        c2m_low[i];
    c2m_wide[i].imag = 0;
  }
  const auto c2m_decoded =
      he.ckks_decode_wide(c2m_wide, degree, 43, 13, 13, false, 83);
  vector<uint64_t> c2m_opened(c2m_decoded.size());
  if (party == sci::ALICE) {
    io.send_data(c2m_decoded.data(), c2m_decoded.size() * sizeof(uint64_t));
    copy(c2m_decoded.begin(), c2m_decoded.end(), c2m_opened.begin());
    io.recv_data(c2m_low.data(), c2m_decoded.size() * sizeof(uint64_t));
    for (size_t i = 0; i < c2m_opened.size(); ++i)
      c2m_opened[i] = (c2m_opened[i] + c2m_low[i]) & input_mask;
  } else {
    io.recv_data(c2m_low.data(), c2m_decoded.size() * sizeof(uint64_t));
    for (size_t i = 0; i < c2m_opened.size(); ++i)
      c2m_opened[i] = (c2m_decoded[i] + c2m_low[i]) & input_mask;
    io.send_data(c2m_decoded.data(), c2m_decoded.size() * sizeof(uint64_t));
  }
  uint64_t c2m_max_abs = 0;
  for (int i = 0; i < slots; ++i) {
    uint64_t delta = (c2m_opened[i] - pre_shift_decoded[i]) & input_mask;
    if (delta >= (uint64_t{1} << 42))
      delta = (uint64_t{1} << 43) - delta;
    c2m_max_abs = max(c2m_max_abs, delta);
  }

  // Public diagnostic for the current Appendix-C.4 ordering: retain the
  // source-scale field/ring value in Z_2^60, extend to Z_2^100, run the wide
  // local FFT, and only then shift 40 -> 13.  This uses the same synthetic
  // CKKS polynomial and fresh additive mask as the preceding check.
  vector<uint64_t> post_low(static_cast<size_t>(degree));
  vector<uint64_t> post_high(static_cast<size_t>(degree));
  fpmath.fix->xt->s_extend_wide(
      degree, c2m_ring_share.data(), post_low.data(), post_high.data(),
      ring_bits, wide_bits);
  vector<HE_CKKS::wide_complex_t> post_wide(static_cast<size_t>(degree));
  for (int i = 0; i < degree; ++i) {
    post_wide[i].real =
        (static_cast<HE_CKKS::wide_scalar_t>(post_high[i]) << 64) |
        post_low[i];
    post_wide[i].imag = 0;
  }
  const auto post_decoded =
      he.ckks_decode_wide(post_wide, degree, 43, 40, 13, false, wide_bits);
  vector<uint64_t> post_opened(post_decoded.size());
  if (party == sci::ALICE) {
    io.send_data(post_decoded.data(), post_decoded.size() * sizeof(uint64_t));
    copy(post_decoded.begin(), post_decoded.end(), post_opened.begin());
    io.recv_data(post_low.data(), post_decoded.size() * sizeof(uint64_t));
    for (size_t i = 0; i < post_opened.size(); ++i)
      post_opened[i] = (post_opened[i] + post_low[i]) & input_mask;
  } else {
    io.recv_data(post_low.data(), post_decoded.size() * sizeof(uint64_t));
    for (size_t i = 0; i < post_opened.size(); ++i)
      post_opened[i] = (post_decoded[i] + post_low[i]) & input_mask;
    io.send_data(post_decoded.data(), post_decoded.size() * sizeof(uint64_t));
  }
  uint64_t post_c2m_max_abs = 0;
  for (int i = 0; i < slots; ++i) {
    uint64_t delta = (post_opened[i] - high_decoded[i]) & input_mask;
    if (delta >= (uint64_t{1} << 42))
      delta = (uint64_t{1} << 43) - delta;
    post_c2m_max_abs = max(post_c2m_max_abs, delta);
  }
  uint64_t high_scale_max_abs = 0;
  uint64_t pre_shift_max_abs = 0;
  for (int i = 0; i < slots; ++i) {
    const int64_t expected_signed =
        static_cast<int64_t>(high_input[i]) >> 27;
    const uint64_t expected = static_cast<uint64_t>(expected_signed) & input_mask;
    uint64_t delta = (high_decoded[i] - expected) & input_mask;
    if (delta >= (uint64_t{1} << 42)) delta = (uint64_t{1} << 43) - delta;
    high_scale_max_abs = max(high_scale_max_abs, delta);
    uint64_t pre_delta = (pre_shift_decoded[i] - expected) & input_mask;
    if (pre_delta >= (uint64_t{1} << 42))
      pre_delta = (uint64_t{1} << 43) - pre_delta;
    pre_shift_max_abs = max(pre_shift_max_abs, pre_delta);
  }
  cout << "CKKS_FFT_SELFTEST party=" << party
       << " old_vs_wide_max_xor=" << old_wide_mismatch
       << " roundtrip_max_xor=" << roundtrip_mismatch
         << " random_share_max_xor=" << shared_ring_mismatch
         << " random_share_nonzero=" << shared_ring_nonzero
         << " random_share_max_abs=" << shared_ring_max_abs
       << " high_scale_max_abs=" << high_scale_max_abs
       << " pre_shift_max_abs=" << pre_shift_max_abs
       << " c2m_pipeline_max_abs=" << c2m_max_abs
       << " c2m_postfft_pipeline_max_abs=" << post_c2m_max_abs << endl;
  for (int i = 0; i < 8 && i < slots; ++i) {
    cout << "CKKS_FFT_SELFTEST_SAMPLE party=" << party << " i=" << i
         << " opened=" << opened[i]
         << " expected=" << public_decoded[i]
         << " xor=" << (opened[i] ^ public_decoded[i]) << endl;
  }
  if (old_wide_mismatch != 0 || shared_ring_mismatch != 0) return 1;
  return 0;
}
