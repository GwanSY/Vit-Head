#include "BOLTTransformer/bolt_conversion.h"
#include "BOLTTransformer/bolt_he.h"
#include "FloatingPoint/fixed-point.h"
#include "utils/ArgMapping/ArgMapping.h"
#include "utils/io_pack.h"
#include "bolt_vit_mpc.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sci;
using namespace sci::bolt;

std::uint64_t prime_mod = 536903681ULL;
std::int32_t bitlength = 37;
std::int32_t num_threads = 1;

namespace {

int party = 0;
int port = 47430;
std::string address = "127.0.0.1";

constexpr int kEll = 38;
constexpr int kScale = 12;

std::uint64_t mask() { return (1ULL << kEll) - 1; }

std::uint64_t ring(std::int64_t value) {
  return static_cast<std::uint64_t>(value) & mask();
}

void exchange_alice_share(IOPack &io, const std::vector<std::uint64_t> &local,
                          std::vector<std::uint64_t> &alice_share) {
  const std::uint64_t size = local.size();
  if (party == ALICE) {
    io.io->send_data(&size, sizeof(size));
    io.io->send_data(local.data(), local.size() * sizeof(std::uint64_t));
    return;
  }
  std::uint64_t peer_size = 0;
  io.io->recv_data(&peer_size, sizeof(peer_size));
  if (peer_size != size)
    throw std::runtime_error("signed conversion share size mismatch");
  alice_share.resize(size);
  io.io->recv_data(alice_share.data(), alice_share.size() * sizeof(std::uint64_t));
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
    BoltHE he(party, io.io, BoltHEParameters::vit_main(), true);
    BoltConversion conversion(he.view());
    BoltVitMPC mpc(party, address, port + 1);

    const std::size_t count = he.slot_count();
    std::vector<std::int64_t> clear(count);
    std::vector<std::uint64_t> local(count);
    for (std::size_t i = 0; i < count; ++i) {
      clear[i] = static_cast<std::int64_t>((i * 97 + 19) % 2001) - 1000;
      // MPC outputs are not generally split into small representatives.  Use
      // an arbitrary Z2^38 Alice share and the complementary ring share to
      // exercise the same signed-extension boundary as production.
      const std::uint64_t alice_share =
          (0x9E3779B97F4A7C15ULL * (i + 1) + 0xD1B54A32D192ED03ULL) & mask();
      const std::uint64_t clear_ring = ring(clear[i]);
      const std::uint64_t bob_share = (clear_ring - alice_share) & mask();
      local[i] = party == ALICE ? alice_share : bob_share;
    }
    FixArray input(party, static_cast<int>(count), true, kEll, kScale);
    std::memcpy(input.data, local.data(), count * sizeof(std::uint64_t));
    FixArray extended = mpc.signed_extend_to_64(input);
    const auto encrypted = conversion.ss_to_he_packed_signed_ring_compat(
        extended.data, count);
    const auto field = conversion.he_to_ss_packed_compat(encrypted, 1, true);
    std::vector<std::uint64_t> alice_share;
    exchange_alice_share(io, field, alice_share);

    bool pass = true;
    if (party == BOB) {
      const std::uint64_t p = he.parameters().plain_modulus;
      const std::uint64_t half = (p - 1) / 2;
      for (std::size_t i = 0; i < count; ++i) {
        const std::uint64_t reconstructed = (field[i] + alice_share[i]) % p;
        const std::int64_t decoded = static_cast<std::int64_t>(
            (reconstructed + p - half) % p);
        const std::int64_t signed_decoded =
            decoded > static_cast<std::int64_t>(p / 2)
                ? decoded - static_cast<std::int64_t>(p)
                : decoded;
        if (signed_decoded != clear[i]) {
          std::cerr << "BOLT_VIT_SIGNED_CONVERSION_FIRST_MISMATCH index=" << i
                    << " decoded=" << signed_decoded
                    << " expected=" << clear[i] << std::endl;
          pass = false;
          break;
        }
      }
      std::cout << "BOLT_VIT_SIGNED_CONVERSION_TEST status="
                << (pass ? "PASS" : "FAIL") << std::endl;
    }
    return pass ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "BOLT_VIT_SIGNED_CONVERSION_TEST error: " << error.what()
              << std::endl;
    return 2;
  }
}
