#include "nonlinear.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

int party;
int port = 37100;
int num_threads = 2;
std::string address = "127.0.0.1";

constexpr std::uint64_t kPrime = 922114682275233793ULL;
constexpr int kFieldBits = 60;
constexpr int kRingBits = 37;
constexpr int kSize = 128;

std::uint64_t ring_mask(int bits) {
  return bits == 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << bits) - 1);
}

std::vector<std::uint64_t> make_values(int scale_delta, bool divisible) {
  std::mt19937_64 generator(0x434b4b53434f4e56ULL + scale_delta);
  std::vector<std::uint64_t> values(kSize);
  const int shift = divisible ? std::max(0, -scale_delta) : 0;
  for (int i = 0; i < kSize; ++i) {
    const std::uint64_t base = 1 + (generator() & ((std::uint64_t{1} << 17) - 1));
    values[i] = base << shift;
  }
  return values;
}

std::vector<std::uint64_t> field_share(
    const std::vector<std::uint64_t> &value, bool uniform_mask) {
  std::mt19937_64 generator(0x4649454c44534852ULL);
  std::vector<std::uint64_t> share(value.size());
  for (std::size_t i = 0; i < value.size(); ++i) {
    const std::uint64_t random =
        uniform_mask ? generator() % kPrime : generator() % (value[i] + 1);
    share[i] = party == sci::ALICE
                   ? random
                   : (value[i] + kPrime - random) % kPrime;
  }
  return share;
}

std::vector<std::uint64_t> ring_share(
    const std::vector<std::uint64_t> &value, int bits) {
  std::mt19937_64 generator(0x52494e4753484152ULL);
  const std::uint64_t mask = ring_mask(bits);
  std::vector<std::uint64_t> share(value.size());
  for (std::size_t i = 0; i < value.size(); ++i) {
    const std::uint64_t random = generator() & mask;
    share[i] = party == sci::ALICE ? random : (value[i] - random) & mask;
  }
  return share;
}

bool verify_ring(NonLinear &nonlinear, const std::string &name,
                 const std::vector<std::uint64_t> &local,
                 const std::vector<std::uint64_t> &expected,
                 std::uint64_t tolerance) {
  bool pass = true;
  std::uint64_t max_error = 0;
  if (party == sci::ALICE) {
    nonlinear.iopackArr[0]->io->send_data(
        local.data(), local.size() * sizeof(std::uint64_t));
    nonlinear.iopackArr[0]->io->recv_data(&pass, sizeof(pass));
  } else {
    std::vector<std::uint64_t> alice(local.size());
    nonlinear.iopackArr[0]->io->recv_data(
        alice.data(), alice.size() * sizeof(std::uint64_t));
    const std::uint64_t mask = ring_mask(kRingBits);
    for (std::size_t i = 0; i < local.size(); ++i) {
      const std::uint64_t actual = (alice[i] + local[i]) & mask;
      const std::uint64_t want = expected[i] & mask;
      const std::uint64_t direct = actual > want ? actual - want : want - actual;
      const std::uint64_t wrapped = (mask + 1) - direct;
      const std::uint64_t error = std::min(direct, wrapped);
      max_error = std::max(max_error, error);
      pass = pass && error <= tolerance;
    }
    nonlinear.iopackArr[0]->io->send_data(&pass, sizeof(pass));
    std::cout << "CONVERSION_REGRESSION name=" << name
              << " status=" << (pass ? "PASS" : "FAIL")
              << " max_ring_error=" << max_error << std::endl;
  }
  return pass;
}

bool verify_field(NonLinear &nonlinear, const std::string &name,
                  const std::vector<std::uint64_t> &local,
                  const std::vector<std::uint64_t> &expected) {
  bool pass = true;
  std::uint64_t max_error = 0;
  if (party == sci::ALICE) {
    nonlinear.iopackArr[0]->io->send_data(
        local.data(), local.size() * sizeof(std::uint64_t));
    nonlinear.iopackArr[0]->io->recv_data(&pass, sizeof(pass));
  } else {
    std::vector<std::uint64_t> alice(local.size());
    nonlinear.iopackArr[0]->io->recv_data(
        alice.data(), alice.size() * sizeof(std::uint64_t));
    for (std::size_t i = 0; i < local.size(); ++i) {
      const std::uint64_t actual =
          static_cast<std::uint64_t>((static_cast<__uint128_t>(alice[i]) +
                                      local[i]) %
                                     kPrime);
      const std::uint64_t want = expected[i] % kPrime;
      const std::uint64_t direct = actual > want ? actual - want : want - actual;
      const std::uint64_t error = std::min(direct, kPrime - direct);
      max_error = std::max(max_error, error);
      pass = pass && error == 0;
    }
    nonlinear.iopackArr[0]->io->send_data(&pass, sizeof(pass));
    std::cout << "CONVERSION_REGRESSION name=" << name
              << " status=" << (pass ? "PASS" : "FAIL")
              << " max_field_error=" << max_error << std::endl;
  }
  return pass;
}

bool run_field_case(NonLinear &nonlinear, const std::string &name, int threads,
                    int s_in, int s_out, bool uniform_mask) {
  const int delta = s_out - s_in;
  const std::vector<std::uint64_t> values = make_values(delta, delta < 0);
  std::vector<std::uint64_t> input = field_share(values, uniform_mask);
  std::vector<std::uint64_t> output(values.size());
  nonlinear.field_to_ring(threads, input.data(), kPrime, output.data(),
                          static_cast<int>(output.size()), kRingBits, s_in,
                          s_out);
  std::vector<std::uint64_t> expected(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    expected[i] = delta >= 0 ? values[i] << delta : values[i] >> -delta;
  }
  // truncate_reduce_lsb is an approximate local-share truncation and can be
  // one unit away even when the secret itself is divisible by 2^delta.
  return verify_ring(nonlinear, name, output, expected, delta < 0 ? 1 : 0);
}

bool run_ring_case(NonLinear &nonlinear, const std::string &name, int threads,
                   int s_in, int s_out) {
  const int delta = s_out - s_in;
  constexpr int input_bits = 30;
  const std::vector<std::uint64_t> values = make_values(delta, delta < 0);
  std::vector<std::uint64_t> input = ring_share(values, input_bits);
  std::vector<std::uint64_t> output(values.size());
  nonlinear.ring_to_field(threads, input.data(), kPrime, output.data(),
                          static_cast<int>(output.size()), input_bits, s_in,
                          s_out);
  std::vector<std::uint64_t> expected(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    expected[i] = delta >= 0 ? values[i] << delta : values[i] >> -delta;
  }
  return verify_field(nonlinear, name, output, expected);
}

} // namespace

int main(int argc, char **argv) {
  ArgMapping arguments;
  arguments.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  arguments.arg("p", port, "Port Number");
  arguments.arg("nt", num_threads, "Number of threads");
  arguments.arg("ip", address, "IP Address of server (ALICE)");
  arguments.parse(argc, argv);
  if (num_threads < 1 || num_threads > 2) {
    std::cerr << "This focused regression accepts --nt 1 or --nt 2" << std::endl;
    return 2;
  }

  NonLinear nonlinear(party, address, port);
  bool pass = true;
  // A no-wrap fixture isolates the odd-thread FPMath role reversal itself.
  pass &= run_field_case(nonlinear, "field_same_scale_no_prime_wrap", 2, 12,
                         12, false);
  // Uniform masks are the actual additive-share distribution at a conversion
  // boundary and exercise wrap modulo the prime.
  pass &= run_field_case(nonlinear, "field_same_scale_uniform", num_threads,
                         12, 12, true);
  pass &= run_field_case(nonlinear, "field_scale_up_uniform", num_threads, 4,
                         12, true);
  pass &= run_field_case(nonlinear, "field_scale_down_uniform", num_threads,
                         20, 12, true);
  pass &= run_ring_case(nonlinear, "ring_same_scale_uniform", num_threads, 12,
                        12);
  pass &= run_ring_case(nonlinear, "ring_scale_up_uniform", num_threads, 12,
                        20);
  pass &= run_ring_case(nonlinear, "ring_scale_down_uniform", num_threads, 20,
                        12);

  if (party == sci::BOB) {
    std::cout << "CONVERSION_REGRESSION_SUMMARY status="
              << (pass ? "PASS" : "FAIL")
              << " threads=" << num_threads
              << " formal_ring_scale_delta=67"
              << " formal_ring_scale_supported="
              << ((30 + 67 < 64) ? 1 : 0) << std::endl;
  }
  return pass ? 0 : 1;
}
