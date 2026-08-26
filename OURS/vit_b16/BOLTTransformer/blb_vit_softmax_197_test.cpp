#include "FloatingPoint/fp-math.h"
#include "OT/ot_pack.h"
#include "utils/ArgMapping/ArgMapping.h"
#include "utils/io_pack.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sci;

extern std::uint64_t prime_mod;
std::int32_t bitlength = 37;
std::int32_t num_threads = 1;

namespace {

constexpr std::size_t kTokens = 197;
constexpr std::size_t kRows = 2;
constexpr int kEll = 37;
constexpr int kScale = 12;
constexpr std::uint64_t kRingModulus = 1ULL << kEll;
constexpr std::uint64_t kRingMask = kRingModulus - 1;
constexpr std::int64_t kScaleFactor = 1LL << kScale;

std::uint64_t mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

std::uint64_t ring_encode(std::int64_t value) {
  return static_cast<std::uint64_t>(value) & kRingMask;
}

std::int64_t ring_decode(std::uint64_t value) {
  value &= kRingMask;
  if (value < (kRingModulus >> 1))
    return static_cast<std::int64_t>(value);
  return static_cast<std::int64_t>(value - kRingModulus);
}

std::vector<std::vector<std::uint64_t>> make_clear_rows() {
  std::vector<std::vector<std::uint64_t>> rows(
      kRows, std::vector<std::uint64_t>(kTokens));

  // A uniform attention row is the regression for the old BERT-sized ell=19
  // exponential accumulator: 197 * 4096 = 806912 exceeds 2^19.
  std::fill(rows[0].begin(), rows[0].end(), ring_encode(0));

  // A smooth, bounded score row with a local peak and deterministic jitter.
  // Its roughly [-2.5, 1.5] range is representative of post-scaled attention
  // scores while avoiding any dependency on model parameters in this unit
  // test.
  for (std::size_t key = 0; key < kTokens; ++key) {
    const double distance = static_cast<double>(
        std::abs(static_cast<int>(key) - 73));
    double score = 0.72 * std::sin(0.173 * static_cast<double>(key)) +
                   0.31 * std::cos(0.071 * static_cast<double>(key)) -
                   0.012 * distance;
    // Keep the largest probability separated by much more than one output
    // ULP so the argmax assertion measures Softmax, not a quantized tie.
    if (key == 73) score += 1.50;
    rows[1][key] = ring_encode(
        static_cast<std::int64_t>(std::llround(score * kScaleFactor)));
  }
  return rows;
}

std::vector<std::vector<double>> clear_softmax(
    const std::vector<std::vector<std::uint64_t>> &rows) {
  std::vector<std::vector<double>> result(
      rows.size(), std::vector<double>(kTokens));
  for (std::size_t row = 0; row < rows.size(); ++row) {
    std::int64_t maximum = std::numeric_limits<std::int64_t>::min();
    for (std::uint64_t value : rows[row])
      maximum = std::max(maximum, ring_decode(value));
    double denominator = 0.0;
    for (std::size_t key = 0; key < kTokens; ++key) {
      const double shifted = static_cast<double>(
          ring_decode(rows[row][key]) - maximum) / kScaleFactor;
      result[row][key] = std::exp(shifted);
      denominator += result[row][key];
    }
    for (double &value : result[row]) value /= denominator;
  }
  return result;
}

struct RowMetrics {
  std::int64_t raw_sum = 0;
  std::uint64_t max_ulp_error = 0;
  std::uint64_t total_ulp_error = 0;
  std::size_t expected_argmax = 0;
  std::size_t actual_argmax = 0;
  bool nonnegative = true;
};

RowMetrics validate_row(const FixArray &revealed,
                        const std::vector<double> &reference) {
  if (revealed.party != PUBLIC || revealed.size != static_cast<int>(kTokens) ||
      revealed.ell != kEll || revealed.s != kScale)
    throw std::runtime_error("unexpected revealed Softmax metadata");

  RowMetrics metrics;
  std::int64_t best_actual = std::numeric_limits<std::int64_t>::min();
  double best_expected = -1.0;
  for (std::size_t key = 0; key < kTokens; ++key) {
    const std::int64_t actual = ring_decode(revealed.data[key]);
    const std::int64_t expected = static_cast<std::int64_t>(
        std::llround(reference[key] * kScaleFactor));
    metrics.nonnegative = metrics.nonnegative && actual >= 0;
    metrics.raw_sum += actual;
    const std::uint64_t error = static_cast<std::uint64_t>(
        actual >= expected ? actual - expected : expected - actual);
    metrics.max_ulp_error = std::max(metrics.max_ulp_error, error);
    metrics.total_ulp_error += error;
    if (actual > best_actual) {
      best_actual = actual;
      metrics.actual_argmax = key;
    }
    if (reference[key] > best_expected) {
      best_expected = reference[key];
      metrics.expected_argmax = key;
    }
  }
  return metrics;
}

} // namespace

int main(int argc, char **argv) {
  int party = 0;
  int port = 52640;
  std::string address = "127.0.0.1";
  ArgMapping args;
  args.arg("r", party, "Role: ALICE=1, BOB=2");
  args.arg("p", port, "TCP port");
  args.arg("ip", address, "ALICE address (BOB only)");
  args.parse(argc, argv);

  try {
    if (party != ALICE && party != BOB)
      throw std::invalid_argument("r must be ALICE=1 or BOB=2");

    IOPack io(party, port, address);
    OTPack ot(&io, party);
    FPMath fp(party, &io, &ot);

    const auto clear_rows = make_clear_rows();
    const auto reference = clear_softmax(clear_rows);
    std::vector<FixArray> shared_rows;
    shared_rows.reserve(kRows);
    std::size_t nonzero_alice_shares = 0;
    std::size_t nonzero_bob_shares = 0;
    for (std::size_t row = 0; row < kRows; ++row) {
      FixArray local(party, static_cast<int>(kTokens), true, kEll, kScale);
      for (std::size_t key = 0; key < kTokens; ++key) {
        const std::size_t index = row * kTokens + key;
        const std::uint64_t alice_share =
            mix(0x5649545f534f4654ULL + index) & kRingMask;
        const std::uint64_t bob_share =
            (clear_rows[row][key] - alice_share) & kRingMask;
        nonzero_alice_shares += alice_share != 0;
        nonzero_bob_shares += bob_share != 0;
        local.data[key] = party == ALICE ? alice_share : bob_share;
      }
      shared_rows.push_back(std::move(local));
    }
    const bool share_coverage =
        nonzero_alice_shares == kRows * kTokens &&
        nonzero_bob_shares == kRows * kTokens;

    // This calls the existing MPC implementation directly.  Only the two
    // declared unit-test output rows are opened; neither input shares nor any
    // max/exponential/sum/division intermediate is reconstructed.
    std::vector<FixArray> output = fp.softmax_fix_iron_1(shared_rows);
    if (output.size() != kRows)
      throw std::runtime_error("Softmax returned the wrong number of rows");

    std::vector<RowMetrics> metrics;
    metrics.reserve(kRows);
    for (std::size_t row = 0; row < kRows; ++row) {
      FixArray revealed = fp.fix->output(PUBLIC, output[row]);
      metrics.push_back(validate_row(revealed, reference[row]));
    }

    // The IRON lookup/division approximation is deliberately allowed a few
    // scale-12 ULP.  These bounds are still far below the old ell=19 uniform
    // wrap result (roughly 59 per item and a row sum around 11600).
    const bool uniform_pass =
        metrics[0].nonnegative && metrics[0].max_ulp_error <= 4 &&
        std::llabs(metrics[0].raw_sum - kScaleFactor) <= 256;
    const bool nonuniform_pass =
        metrics[1].nonnegative && metrics[1].max_ulp_error <= 10 &&
        metrics[1].total_ulp_error <= 512 &&
        std::llabs(metrics[1].raw_sum - kScaleFactor) <= 256 &&
        metrics[1].actual_argmax == metrics[1].expected_argmax;
    const bool no_ell19_wrap = metrics[0].raw_sum < 2 * kScaleFactor;
    const bool pass = share_coverage && uniform_pass && nonuniform_pass &&
                      no_ell19_wrap;

    if (party == ALICE) {
      std::cout << "VIT_SOFTMAX_197_RANDOM_SHARE_COVERAGE="
                << (share_coverage ? "PASS" : "FAIL")
                << " alice_nonzero=" << nonzero_alice_shares
                << " bob_nonzero=" << nonzero_bob_shares << '\n';
      std::cout << "VIT_SOFTMAX_197_UNIFORM="
                << (uniform_pass ? "PASS" : "FAIL")
                << " raw_sum=" << metrics[0].raw_sum
                << " max_ulp_error=" << metrics[0].max_ulp_error
                << " total_ulp_error=" << metrics[0].total_ulp_error << '\n';
      std::cout << "VIT_SOFTMAX_197_NONUNIFORM="
                << (nonuniform_pass ? "PASS" : "FAIL")
                << " raw_sum=" << metrics[1].raw_sum
                << " max_ulp_error=" << metrics[1].max_ulp_error
                << " total_ulp_error=" << metrics[1].total_ulp_error
                << " expected_argmax=" << metrics[1].expected_argmax
                << " actual_argmax=" << metrics[1].actual_argmax << '\n';
      std::cout << "VIT_SOFTMAX_197_NO_ELL19_WRAP="
                << (no_ell19_wrap ? "PASS" : "FAIL") << '\n';
      std::cout << "VIT_SOFTMAX_197_TEST=" << (pass ? "PASS" : "FAIL")
                << std::endl;
    }
    return pass ? 0 : 3;
  } catch (const std::exception &error) {
    std::cerr << "VIT_SOFTMAX_197_TEST=FAIL role=" << party
              << " error=" << error.what() << std::endl;
    return 2;
  }
}
