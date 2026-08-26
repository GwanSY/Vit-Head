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
int port = 47440;
std::string address = "127.0.0.1";

constexpr int kEll = 38;
constexpr int kScale = 12;

std::uint64_t ring(std::int64_t value) {
  return static_cast<std::uint64_t>(value) & ((1ULL << kEll) - 1);
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

    // Every chunk is non-empty, and the values span negative and positive
    // representatives.  This validates the same alternating-party 32-thread
    // right-shift path used at all ViT scale boundaries.
    constexpr std::size_t count = 4096;
    std::vector<std::int64_t> clear(count);
    std::vector<std::uint64_t> local(count);
    for (std::size_t i = 0; i < count; ++i) {
      clear[i] = static_cast<std::int64_t>((i * 7919 + 123) % 200001) -
                 100000;
      const std::uint64_t alice_share =
          (0x9E3779B97F4A7C15ULL * (i + 1) +
           0xD1B54A32D192ED03ULL) & ((1ULL << kEll) - 1);
      const std::uint64_t bob_share =
          (ring(clear[i]) - alice_share) & ((1ULL << kEll) - 1);
      local[i] = party == ALICE ? alice_share : bob_share;
    }

    FixArray input(party, static_cast<int>(count), true, kEll, kScale);
    std::memcpy(input.data, local.data(), count * sizeof(std::uint64_t));
    bool pass = true;
    for (const int amount : {1, 12, 20}) {
      const FixArray shifted = mpc.right_shift(input, amount);
      const FixArray opened = fp.fix->output(BOB, shifted);
      if (party == BOB) {
        for (std::size_t i = 0; i < count; ++i) {
          const std::int64_t divisor = 1LL << amount;
          const std::int64_t expected =
              clear[i] >= 0
                  ? clear[i] / divisor
                  : -((-clear[i] + divisor - 1) / divisor);
          const std::int64_t decoded = signed_val(opened.data[i], opened.ell);
          if (decoded != expected) {
            std::cerr << "BOLT_VIT_MPC_THREAD_FIRST_MISMATCH amount="
                      << amount << " index=" << i
                      << " decoded=" << decoded
                      << " expected=" << expected << std::endl;
            pass = false;
            break;
          }
        }
      }
      if (!pass) break;
    }
    if (party == BOB)
      std::cout << "BOLT_VIT_MPC_THREAD_TEST status="
                << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "BOLT_VIT_MPC_THREAD_TEST error: " << error.what()
              << std::endl;
    return 2;
  }
}
