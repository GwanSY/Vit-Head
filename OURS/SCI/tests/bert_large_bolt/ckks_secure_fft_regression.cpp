#include "ckks_bert.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int bitlength = 37;

namespace {

int party = 0;
int port = 64400;
int seed = 20260810;
std::string address = "127.0.0.1";

constexpr int kDegree = 8192;
constexpr int kEll = 37;
constexpr int kScale = 12;

std::uint64_t mask_for(int ell) {
  return (std::uint64_t{1} << ell) - 1;
}

std::int64_t signed_ring(std::uint64_t value) {
  value &= mask_for(kEll);
  const std::uint64_t sign = std::uint64_t{1} << (kEll - 1);
  return (value & sign)
             ? static_cast<std::int64_t>(value) -
                   static_cast<std::int64_t>(std::uint64_t{1} << kEll)
             : static_cast<std::int64_t>(value);
}

std::vector<std::uint64_t> fixture(std::size_t size) {
  std::mt19937_64 generator(seed);
  std::uniform_int_distribution<std::int64_t> distribution(-32768, 32768);
  std::vector<std::uint64_t> result(size);
  for (auto &value : result) {
    value = static_cast<std::uint64_t>(distribution(generator)) &
            mask_for(kEll);
  }
  return result;
}

std::vector<std::uint64_t> shares(
    const std::vector<std::uint64_t> &secret, std::uint64_t domain) {
  std::mt19937_64 generator(static_cast<std::uint64_t>(seed) ^ domain);
  const std::uint64_t mask = mask_for(kEll);
  std::vector<std::uint64_t> result(secret.size());
  for (std::size_t i = 0; i < secret.size(); ++i) {
    const std::uint64_t random = generator() & mask;
    result[i] = party == sci::ALICE ? random
                                    : (secret[i] - random) & mask;
  }
  return result;
}

struct Error {
  std::uint64_t max = 0;
  long double sum = 0;
  std::size_t first = static_cast<std::size_t>(-1);
};

std::uint64_t slot_map_hash(const HE_CKKS &he, bool &is_permutation) {
  std::uint64_t hash = 1469598103934665603ULL;
  std::vector<bool> seen(he.poly_modulus_degree, false);
  is_permutation =
      he.matrix_reps_index_map_.size() == he.poly_modulus_degree;
  for (const std::size_t value : he.matrix_reps_index_map_) {
    if (value >= he.poly_modulus_degree || seen[value]) {
      is_permutation = false;
    } else {
      seen[value] = true;
    }
    hash ^= static_cast<std::uint64_t>(value);
    hash *= 1099511628211ULL;
  }
  return hash;
}

bool report_slot_map(Bert_ckks &adapter, const HE_CKKS &he) {
  bool is_permutation = false;
  const std::uint64_t local_hash = slot_map_hash(he, is_permutation);
  bool pass = is_permutation;
  if (party == sci::ALICE) {
    adapter.io->send_data(&local_hash, sizeof(local_hash));
    adapter.io->send_data(&is_permutation, sizeof(is_permutation));
    adapter.io->recv_data(&pass, sizeof(pass));
    return pass;
  }
  std::uint64_t alice_hash = 0;
  bool alice_is_permutation = false;
  adapter.io->recv_data(&alice_hash, sizeof(alice_hash));
  adapter.io->recv_data(&alice_is_permutation, sizeof(alice_is_permutation));
  pass = is_permutation && alice_is_permutation && alice_hash == local_hash;
  adapter.io->send_data(&pass, sizeof(pass));
  std::cout << "SECURE_FFT_REGRESSION stage=slot_map_determinism status="
            << (pass ? "PASS" : "FAIL") << " permutation="
            << (is_permutation && alice_is_permutation ? 1 : 0)
            << " two_party_hash_equal=" << (alice_hash == local_hash ? 1 : 0)
            << " hash=" << local_hash << std::endl;
  return pass;
}

Error compare(const std::vector<std::uint64_t> &actual,
              const std::vector<std::uint64_t> &expected) {
  Error error;
  const std::uint64_t mask = mask_for(kEll);
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const std::int64_t delta = signed_ring((actual[i] - expected[i]) & mask);
    const std::uint64_t magnitude = static_cast<std::uint64_t>(
        delta < 0 ? -delta : delta);
    error.max = std::max(error.max, magnitude);
    error.sum += magnitude;
    if (magnitude != 0 && error.first == static_cast<std::size_t>(-1)) {
      error.first = i;
    }
  }
  return error;
}

bool reconstruct_and_report(Bert_ckks &adapter, const char *stage,
                            const std::vector<std::uint64_t> &local,
                            const std::vector<std::uint64_t> &expected,
                            std::uint64_t tolerance) {
  bool pass = true;
  if (party == sci::ALICE) {
    adapter.io->send_data(local.data(), local.size() * sizeof(std::uint64_t));
    adapter.io->recv_data(&pass, sizeof(pass));
    return pass;
  }
  std::vector<std::uint64_t> alice(local.size());
  adapter.io->recv_data(alice.data(), alice.size() * sizeof(std::uint64_t));
  const std::uint64_t mask = mask_for(kEll);
  std::vector<std::uint64_t> reconstructed(local.size());
  for (std::size_t i = 0; i < local.size(); ++i) {
    reconstructed[i] = (alice[i] + local[i]) & mask;
  }
  const Error error = compare(reconstructed, expected);
  pass = error.max <= tolerance;
  adapter.io->send_data(&pass, sizeof(pass));
  std::cout << "SECURE_FFT_REGRESSION stage=" << stage
            << " status=" << (pass ? "PASS" : "FAIL")
            << " max_abs_error_raw=" << error.max
            << " mae_raw=" << static_cast<double>(error.sum / local.size())
            << " first_mismatch="
            << (error.first == static_cast<std::size_t>(-1)
                    ? -1
                    : static_cast<long long>(error.first))
            << std::endl;
  return pass;
}

std::vector<std::uint64_t> public_encode_reference(
    HE_CKKS &he, const std::vector<std::uint64_t> &slots) {
  std::vector<std::uint64_t> signed_slots(slots.size());
  for (std::size_t i = 0; i < slots.size(); ++i) {
    signed_slots[i] = static_cast<std::uint64_t>(signed_ring(slots[i]));
  }
  auto result = he.ckks_encode(signed_slots, static_cast<int>(he.num_slots),
                               false);
  for (auto &value : result) {
    value &= mask_for(kEll);
  }
  return result;
}

std::vector<std::uint64_t> public_decode_reference(
    HE_CKKS &he, const std::vector<std::uint64_t> &coefficients) {
  std::vector<HE_CKKS::complex_t> signed_coefficients(coefficients.size());
  for (std::size_t i = 0; i < coefficients.size(); ++i) {
    signed_coefficients[i] = HE_CKKS::complex_t(
        static_cast<std::uint64_t>(signed_ring(coefficients[i])), 0);
  }
  auto result = he.ckks_decode(signed_coefficients, kDegree, false);
  for (auto &value : result) {
    value &= mask_for(kEll);
  }
  return result;
}

} // namespace

int main(int argc, char **argv) {
  ArgMapping arguments;
  arguments.arg("r", party, "Role");
  arguments.arg("p", port, "Base port");
  arguments.arg("ip", address, "ALICE address");
  arguments.arg("seed", seed, "Fixture seed");
  arguments.parse(argc, argv);
  if (party != sci::ALICE && party != sci::BOB) {
    return 2;
  }

  sci::NetIO main_io(party == sci::ALICE ? nullptr : address.c_str(), port);
  HE_CKKS he(party, &main_io, kDegree, {60, 40, 60}, 20, 40,
             LOCATION::HOST);
  Bert_ckks adapter(party, &main_io, address, port + 1);

  bool pass = report_slot_map(adapter, he);

  const auto slots = fixture(he.num_slots);
  const auto slot_shares = shares(slots, 0x454e434f44455f52ULL);
  const auto public_coefficients = public_encode_reference(he, slots);
  const auto secure_coefficients = he.ckks_encode_secure(
      slot_shares, static_cast<int>(he.num_slots), kEll, kScale,
      adapter.nonlinear.fpmath[0]->fix, false);
  pass &= reconstruct_and_report(adapter, "encode_share_linearity",
                                 secure_coefficients, public_coefficients, 0);

  const auto coefficient_shares =
      shares(public_coefficients, 0x4445434f44455f52ULL);
  std::vector<HE_CKKS::complex_t> coefficient_complex(kDegree);
  for (int i = 0; i < kDegree; ++i) {
    coefficient_complex[i] = HE_CKKS::complex_t(coefficient_shares[i], 0);
  }
  const auto public_decoded =
      public_decode_reference(he, public_coefficients);
  const auto secure_decoded = he.ckks_decode_secure(
      coefficient_complex, kDegree, kEll, kScale,
      adapter.nonlinear.fpmath[0]->fix, false);
  pass &= reconstruct_and_report(adapter, "decode_share_linearity",
                                 secure_decoded, public_decoded, 0);

  std::vector<HE_CKKS::complex_t> roundtrip_complex(kDegree);
  for (int i = 0; i < kDegree; ++i) {
    roundtrip_complex[i] = HE_CKKS::complex_t(secure_coefficients[i], 0);
  }
  const auto roundtrip = he.ckks_decode_secure(
      roundtrip_complex, kDegree, kEll, kScale,
      adapter.nonlinear.fpmath[0]->fix, false);
  pass &= reconstruct_and_report(adapter, "secure_roundtrip_equivalence",
                                 roundtrip, public_decoded, 0);
  if (party == sci::BOB) {
    const Error approximation = compare(public_decoded, slots);
    std::cout << "SECURE_FFT_REGRESSION stage=fixed_roundtrip_approximation"
              << " status=INFO max_abs_error_raw=" << approximation.max
              << " mae_raw="
              << static_cast<double>(approximation.sum / slots.size())
              << " first_mismatch="
              << (approximation.first == static_cast<std::size_t>(-1)
                      ? -1
                      : static_cast<long long>(approximation.first))
              << std::endl;
    std::cout << "SECURE_FFT_REGRESSION_SUMMARY status="
              << (pass ? "PASS" : "FAIL") << " degree=" << kDegree
              << " seed=" << seed << std::endl;
  }
  return pass ? 0 : 1;
}
