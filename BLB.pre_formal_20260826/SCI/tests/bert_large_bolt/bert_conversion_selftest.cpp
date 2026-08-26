#include "nonlinear.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sci;
using namespace std;

namespace {

constexpr uint64_t kFieldPrime = 1152921504592822273ULL;
constexpr int kRingBits = 60;
constexpr int kScale = 40;
constexpr int kWideBits = kRingBits + 40;

static const vector<int64_t> kKnownValues = {
    -(35LL << 40), -(3LL << 40), -1LL, 0LL,
    1LL, (3LL << 40), (35LL << 40)};

uint64_t mod_field(int64_t value) {
  const __int128 q = static_cast<__int128>(kFieldPrime);
  __int128 residue = static_cast<__int128>(value) % q;
  if (residue < 0) residue += q;
  return static_cast<uint64_t>(residue);
}

uint64_t mod_ring(int64_t value) {
  const uint64_t mask = (uint64_t{1} << kRingBits) - 1;
  return static_cast<uint64_t>(value) & mask;
}

vector<uint64_t> make_local_share(int party) {
  vector<uint64_t> share(kKnownValues.size());
  for (size_t index = 0; index < kKnownValues.size(); ++index) {
    // The deterministic nonzero Alice share makes the test exercise the
    // actual additive-share conversion, including signed negative residues.
    const uint64_t alice_share =
        (kFieldPrime / 3 + static_cast<uint64_t>(index) * 17) %
        kFieldPrime;
    const uint64_t field_value = mod_field(kKnownValues[index]);
    if (party == sci::ALICE) {
      share[index] = alice_share;
    } else {
      share[index] = (field_value + kFieldPrime - alice_share) %
                     kFieldPrime;
    }
  }
  return share;
}

vector<uint64_t> make_ring_local_share(int party) {
  vector<uint64_t> share(kKnownValues.size());
  const uint64_t mask = (uint64_t{1} << kRingBits) - 1;
  for (size_t index = 0; index < kKnownValues.size(); ++index) {
    const uint64_t alice_share =
        (uint64_t{1} << (kRingBits - 1)) + static_cast<uint64_t>(index) * 19;
    if (party == sci::ALICE) {
      share[index] = alice_share & mask;
    } else {
      share[index] =
          (mod_ring(kKnownValues[index]) + (uint64_t{1} << kRingBits) -
           (alice_share & mask)) & mask;
    }
  }
  return share;
}

vector<uint64_t> make_centered_field_share(int party) {
  vector<uint64_t> share = make_local_share(party);
  if (party == sci::ALICE) {
    const uint64_t center = (kFieldPrime - 1) / 2;
    for (auto &value : share) {
      value = static_cast<uint64_t>((static_cast<__uint128_t>(value) +
                                     center) % kFieldPrime);
    }
  }
  return share;
}

void assert_opened(const vector<uint64_t> &opened,
                   const vector<uint64_t> &expected,
                   const char *stage) {
  if (opened.size() != expected.size()) {
    throw runtime_error(string(stage) + " output size mismatch");
  }
  for (size_t index = 0; index < opened.size(); ++index) {
    if (opened[index] != expected[index]) {
      throw runtime_error(string(stage) + " mismatch at index " +
                          to_string(index) + " got=" +
                          to_string(opened[index]) + " expected=" +
                          to_string(expected[index]));
    }
  }
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
  if (party != sci::ALICE && party != sci::BOB) {
    throw invalid_argument("r must be 1 or 2");
  }

  // Reuse the same worker-0 FPMath/XTProtocol implementation as production,
  // without constructing 63 unrelated sockets for this seven-value test.
  IOPack iopack(party, port, address);
  OTPack otpack(&iopack, party);
  FPMath fpmath(party, &iopack, &otpack);
  const int size = static_cast<int>(kKnownValues.size());
  const vector<uint64_t> field_share = make_local_share(party);
  vector<uint64_t> ring_share(size, 0);
  auto field_to_ring = [&](const vector<uint64_t> &input_share,
                           vector<uint64_t> &output_share) {
    const int field_bits = static_cast<int>(ceil(log2(kFieldPrime)));
    FixArray input = fpmath.fix->input(
        party, size, const_cast<uint64_t *>(input_share.data()), false,
        field_bits, kScale);
    FixArray output = fpmath.fix->field2ring(input, kFieldPrime, field_bits);
    if (output.ell > kRingBits) {
      output = fpmath.fix->reduce(output, kRingBits);
    }
    for (int i = 0; i < size; ++i) {
      output_share[i] = output.data[i] & ((uint64_t{1} << kRingBits) - 1);
    }
  };
  field_to_ring(field_share, ring_share);
  FixArray ring_local(party, size, true, kRingBits, kScale);
  copy(ring_share.begin(), ring_share.end(), ring_local.data);
  FixArray opened_ring =
      fpmath.fix->output(sci::PUBLIC, ring_local);
  vector<uint64_t> expected_ring;
  expected_ring.reserve(kKnownValues.size());
  for (const int64_t value : kKnownValues) {
    expected_ring.push_back(mod_ring(static_cast<int64_t>(mod_field(value))));
  }
  assert_opened(vector<uint64_t>(opened_ring.data,
                                 opened_ring.data + opened_ring.size),
                expected_ring, "field_to_ring");

  vector<uint64_t> centered_field = make_centered_field_share(party);
  vector<uint64_t> centered_ring(size, 0);
  field_to_ring(centered_field, centered_ring);
  if (party == sci::ALICE) {
    const uint64_t center_ring = ((kFieldPrime - 1) / 2) &
                                 ((uint64_t{1} << kRingBits) - 1);
    const uint64_t mask = (uint64_t{1} << kRingBits) - 1;
    for (auto &value : centered_ring) value = (value - center_ring) & mask;
  }
  FixArray centered_ring_local(party, size, true, kRingBits, kScale);
  copy(centered_ring.begin(), centered_ring.end(), centered_ring_local.data);
  FixArray opened_centered_ring =
      fpmath.fix->output(sci::PUBLIC, centered_ring_local);
  vector<uint64_t> expected_signed_ring;
  expected_signed_ring.reserve(kKnownValues.size());
  for (const int64_t value : kKnownValues) {
    expected_signed_ring.push_back(mod_ring(value));
  }
  assert_opened(vector<uint64_t>(opened_centered_ring.data,
                                 opened_centered_ring.data +
                                     opened_centered_ring.size),
                expected_signed_ring, "centered_field_to_ring");

  FixArray shifted = fpmath.fix->input(
      party, size, centered_ring.data(), true, kRingBits, kScale);
  shifted = fpmath.fix->right_shift(shifted, 27);
  shifted = fpmath.fix->reduce(shifted, 43);
  FixArray opened_shifted = fpmath.fix->output(sci::PUBLIC, shifted);
  vector<uint64_t> expected_shifted;
  for (const int64_t value : kKnownValues) {
    expected_shifted.push_back(static_cast<uint64_t>(value >> 27) &
                               ((uint64_t{1} << 43) - 1));
  }
  assert_opened(vector<uint64_t>(opened_shifted.data,
                                 opened_shifted.data + opened_shifted.size),
                expected_shifted, "centered_signed_right_shift");

  const vector<uint64_t> ring_input = make_ring_local_share(party);
  vector<uint64_t> field_output(size, 0);
  vector<uint64_t> low(size), high(size);
  fpmath.fix->xt->s_extend_wide(size, ring_input.data(), low.data(),
                                high.data(), kRingBits, kWideBits);
  const __uint128_t modulus_q = static_cast<__uint128_t>(kFieldPrime);
  const __uint128_t wide_mod_q =
      (static_cast<__uint128_t>(1) << kWideBits) % modulus_q;
  for (int i = 0; i < size; ++i) {
    const __uint128_t wide =
        (static_cast<__uint128_t>(high[i]) << 64) | low[i];
    const __uint128_t scaled = wide % modulus_q;
    field_output[i] = static_cast<uint64_t>(
        party == sci::ALICE ? (scaled + modulus_q - wide_mod_q) % modulus_q
                            : scaled);
  }
  // ring_to_field_wide returns canonical shares modulo Q.  Open without a
  // 2^60 reduction; the two local Q-residues sum to less than 2Q, so a
  // 64-bit public opening preserves the integer sum before reducing by Q.
  FixArray field_local(party, size, false, 64, kScale);
  copy(field_output.begin(), field_output.end(), field_local.data);
  FixArray opened_field =
      fpmath.fix->output(sci::PUBLIC, field_local);
  vector<uint64_t> expected_field;
  expected_field.reserve(kKnownValues.size());
  for (const int64_t value : kKnownValues) {
    expected_field.push_back(mod_field(value));
  }
  vector<uint64_t> opened_field_mod_q(opened_field.size);
  for (int i = 0; i < opened_field.size; ++i) {
    opened_field_mod_q[i] = opened_field.data[i] % kFieldPrime;
  }
  assert_opened(opened_field_mod_q, expected_field, "ring_to_field_wide");

  cout << "CONVERSION_SELFTEST status=PASS values=" << size
       << " field_prime=" << kFieldPrime << " ring_bits=" << kRingBits
       << " wide_bits=" << kWideBits << endl;
  return 0;
}
