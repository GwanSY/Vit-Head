#include "bolt_vit_mpc.h"

#include "FloatingPoint/fp-math.h"
#include "OT/ot_pack.h"
#include "utils/ArgMapping/ArgMapping.h"
#include "utils/io_pack.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace sci;

std::uint64_t prime_mod = 536903681ULL;
std::int32_t bitlength = 38;
std::int32_t num_threads = BoltVitMPC::kThreads;

namespace {

int party = 0;
int port = 47480;
std::string address = "127.0.0.1";

constexpr int kEll = 38;
constexpr int kScale = 12;
constexpr std::size_t kLayerNormRows = 256;
constexpr std::size_t kLayerNormWidth = 768;
constexpr std::size_t kSoftmaxRows = 12 * 197;
constexpr std::size_t kSoftmaxWidth = 197;

std::uint64_t ring(std::int64_t value) {
  return static_cast<std::uint64_t>(value) & ((1ULL << kEll) - 1);
}

std::int64_t layer_norm_clear(std::size_t row, std::size_t col) {
  return static_cast<std::int64_t>((row * 1315423911ULL +
                                    col * 2654435761ULL + 17) %
                                   160001) -
         80000;
}

std::int64_t softmax_clear(std::size_t row, std::size_t col) {
  return static_cast<std::int64_t>((row * 31337ULL + col * 1009ULL + 23) %
                                   24577) -
         24576;
}

std::uint64_t alice_share(std::size_t index) {
  return (0x9E3779B97F4A7C15ULL * (index + 1) +
          0xD1B54A32D192ED03ULL) & ((1ULL << kEll) - 1);
}

std::uint64_t hash_value(std::uint64_t hash, std::int64_t value) {
  // FNV-1a over the signed fixed-point representatives.  This keeps the
  // result compact while checking every revealed synthetic element.
  std::uint64_t word = static_cast<std::uint64_t>(value);
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (word >> (byte * 8)) & 0xffU;
    hash *= 1099511628211ULL;
  }
  return hash;
}

template <typename ClearFn>
std::vector<FixArray> make_rows(ClearFn clear, std::size_t rows,
                                std::size_t width) {
  std::vector<FixArray> result;
  result.reserve(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    FixArray local(party, static_cast<int>(width), true, kEll, kScale);
    for (std::size_t col = 0; col < width; ++col) {
      const std::size_t index = row * width + col;
      const std::uint64_t a = alice_share(index);
      const std::uint64_t b = (ring(clear(row, col)) - a) & ((1ULL << kEll) - 1);
      local.data[col] = party == ALICE ? a : b;
    }
    result.push_back(std::move(local));
  }
  return result;
}

template <typename ClearFn>
std::uint64_t reveal_hash(FPMath &fp, const std::vector<FixArray> &rows,
                         ClearFn clear, std::size_t width) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::size_t row = 0; row < rows.size(); ++row) {
    FixArray opened = fp.fix->output(BOB, rows[row]);
    if (party != BOB) continue;
    for (std::size_t col = 0; col < width; ++col) {
      const std::int64_t decoded = signed_val(opened.data[col], opened.ell);
      hash = hash_value(hash, decoded);
    }
  }
  return hash;
}

} // namespace

int main(int argc, char **argv) {
  ArgMapping args;
  args.arg("r", party, "Role: ALICE=1, BOB=2");
  args.arg("p", port, "TCP port");
  args.arg("ip", address, "ALICE address (BOB only)");
  args.parse(argc, argv);

  try {
    IOPack io(party, port, address);
    OTPack ot(&io, party);
    FPMath fp(party, &io, &ot);
    BoltVitMPC mpc(party, address, port + 1);

    auto ln_input = make_rows(layer_norm_clear, kLayerNormRows,
                              kLayerNormWidth);
    auto ln_output = mpc.layer_norm(ln_input, 0);
    const std::uint64_t ln_hash =
        reveal_hash(fp, ln_output, layer_norm_clear, kLayerNormWidth);

    auto soft_input = make_rows(softmax_clear, kSoftmaxRows, kSoftmaxWidth);
    auto soft_output = mpc.softmax(soft_input);
    const std::uint64_t soft_hash =
        reveal_hash(fp, soft_output, softmax_clear, kSoftmaxWidth);

    if (party == BOB) {
      std::cout << "BOLT_VIT_MPC_SHAPE_TEST"
                << " layer_norm_hash=" << ln_hash
                << " softmax_hash=" << soft_hash
                << " layer_norm_elements=" << kLayerNormRows * kLayerNormWidth
                << " softmax_elements=" << kSoftmaxRows * kSoftmaxWidth
                << " status=PASS" << std::endl;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "BOLT_VIT_MPC_SHAPE_TEST error: " << error.what()
              << std::endl;
    return 2;
  }
}
