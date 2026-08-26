#include "BOLTTransformer/bolt_conversion.h"
#include "BOLTTransformer/bolt_he.h"
#include "BOLTTransformer/bolt_linear.h"
#include "LinearHE/defines-HE.h"
#include "utils/ArgMapping/ArgMapping.h"
#include "utils/io_pack.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sci;
using namespace sci::bolt;

// The legacy LinearHE object file references these public build parameters;
// the production BOLT tests define them in their executable entry point.
std::uint64_t prime_mod = 536903681ULL;
std::int32_t bitlength = 37;
std::int32_t num_threads = 1;

namespace {

int party = 0;
int port = 47410;
std::string address = "127.0.0.1";

BoltBertPackingMetadata metadata_for(const BoltHE &he, std::size_t input_dim,
                                     std::size_t output_dim) {
  const std::size_t rows = 256;
  const std::size_t num_diag = he.slot_count() / (2 * rows);
  std::size_t baby = 0;
  if (input_dim == 3072 && output_dim == 768)
    baby = 2;
  else if (input_dim == 768 && output_dim == 3072)
    baby = 8;
  else if (input_dim == 768 && output_dim >= 768)
    baby = 4;
  else
    throw std::invalid_argument("unsupported BOLT ViT linear test shape");

  BoltBertPackingMetadata metadata;
  metadata.slot_count = he.slot_count();
  metadata.image_size = rows;
  metadata.filter_height = input_dim;
  metadata.filter_width = output_dim;
  metadata.baby_step = baby;
  metadata.giant_steps = num_diag / baby;
  metadata.attention_heads = 12;
  metadata.paired_head_groups = 6;
  // Exercise the same ciphertext hardening used by the formal ViT graph.
  // The previous regression only checked the raw BSGS arithmetic and could
  // not detect a plaintext/noise-budget problem at the production boundary.
  metadata.noise_flooding_bits = SMUDGING_BITLEN_bert2;
  metadata.modulus_switches = 1;
  metadata.validate(he.slot_count());
  return metadata;
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
    throw std::runtime_error("BOLT ViT linear share size mismatch");
  alice_share.resize(size);
  io.io->recv_data(alice_share.data(), alice_share.size() * sizeof(std::uint64_t));
}

bool run_case(BoltHE &he, BoltConversion &conversion, BoltLinear &linear,
              IOPack &io, std::size_t input_dim, std::size_t output_dim) {
  const auto metadata = metadata_for(he, input_dim, output_dim);
  const std::size_t rows = metadata.image_size;
  const std::size_t input_count = rows * input_dim;
  const std::size_t output_count = rows * output_dim;
  const std::size_t input_ciphertexts = input_count / he.slot_count();
  const std::size_t output_ciphertexts = output_count / he.slot_count();
  const std::uint64_t p = he.parameters().plain_modulus;
  const std::uint64_t ring_mask = (1ULL << 38) - 1;

  // The BOB share varies by physical row.  The matrix has both input and
  // output dependent values, so a wrong diagonal period or output layout
  // cannot be hidden by a row-constant input.
  std::vector<std::uint64_t> row_major(input_count, 0);
  if (party == BOB)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t col = 0; col < input_dim; ++col)
        row_major[row * input_dim + col] =
            static_cast<std::uint64_t>(static_cast<std::int64_t>(row % 7) - 3) &
            ring_mask;
  std::vector<std::uint64_t> column_packed(input_count, 0);
  for (std::size_t col = 0; col < input_dim; ++col)
    for (std::size_t row = 0; row < rows; ++row)
      column_packed[col * rows + row] = row_major[row * input_dim + col];
  const auto encrypted = conversion.ss_to_he_packed_compat(
      column_packed.data(), column_packed.size(), 38);

  std::vector<std::vector<std::uint64_t>> matrix(
      input_dim, std::vector<std::uint64_t>(output_dim));
  for (std::size_t input = 0; input < input_dim; ++input)
    for (std::size_t output = 0; output < output_dim; ++output) {
      const std::int64_t value =
          static_cast<std::int64_t>((input + 2 * output) % 9) - 4;
      matrix[input][output] = value < 0 ? p - static_cast<std::uint64_t>(-value)
                                        : static_cast<std::uint64_t>(value);
    }
  std::vector<std::uint64_t> bias(output_dim);
  for (std::size_t output = 0; output < output_dim; ++output) {
    const std::int64_t value = static_cast<std::int64_t>(output % 5) - 2;
    bias[output] = value < 0 ? p - static_cast<std::uint64_t>(-value)
                             : static_cast<std::uint64_t>(value);
  }

  std::vector<seal::Ciphertext> result;
  if (party == ALICE) {
    const auto packed = linear.bert_pack_cipher_plain_matrix(matrix, metadata);
    const auto packed_bias = linear.bert_pack_cipher_plain_bias(bias, metadata);
    result.resize(output_ciphertexts);
    BoltOperationCounts counts;
    linear.bert_cipher_plain_matmul_bsgs_compat(
        encrypted, packed.first, packed.second, packed_bias, metadata, result,
        &counts);
  }

  const auto field = conversion.he_to_ss_packed_compat(
      result, output_ciphertexts, true);
  std::vector<std::uint64_t> alice_share;
  exchange_alice_share(io, field, alice_share);

  bool pass = true;
  if (party == BOB) {
    const std::uint64_t half = (p - 1) / 2;
    for (std::size_t ct = 0; ct < output_ciphertexts && pass; ++ct) {
      for (std::size_t slot = 0; slot < he.slot_count(); ++slot) {
        std::size_t column = slot / rows;
        const std::size_t row = slot % rows;
        if (slot >= he.slot_count() / 2) {
          column -= he.slot_count() / (2 * rows);
          column += output_dim / 2;
        }
        column += ct * he.slot_count() / (2 * rows);
        if (column >= output_dim) {
          pass = false;
          break;
        }
        const std::size_t index = ct * he.slot_count() + slot;
        const std::uint64_t reconstructed =
            (field[index] + alice_share[index]) % p;
        const std::uint64_t decoded = (reconstructed + p - half) % p;
        __int128 expected_signed = 0;
        for (std::size_t input = 0; input < input_dim; ++input) {
          const std::int64_t x = static_cast<std::int64_t>(row % 7) - 3;
          const std::int64_t w =
              static_cast<std::int64_t>((input + 2 * column) % 9) - 4;
          expected_signed += static_cast<__int128>(x) * w;
        }
        expected_signed += static_cast<std::int64_t>(column % 5) - 2;
        const std::int64_t expected_mod = static_cast<std::int64_t>(
            (expected_signed % static_cast<__int128>(p) + p) % p);
        const std::uint64_t expected = static_cast<std::uint64_t>(expected_mod);
        if (decoded != expected) {
          std::cerr << "BOLT_VIT_LINEAR_FIRST_MISMATCH input=" << input_dim
                    << " output=" << output_dim << " ct=" << ct
                    << " slot=" << slot << " row=" << row
                    << " column=" << column << " decoded=" << decoded
                    << " expected=" << expected << std::endl;
          pass = false;
          break;
        }
      }
    }
    std::cout << "BOLT_VIT_LINEAR_TEST input=" << input_dim
              << " output=" << output_dim << " status="
              << (pass ? "PASS" : "FAIL") << std::endl;
  }
  return pass;
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
    BoltHE he(party, io.io, BoltHEParameters::vit_linear(), true);
    BoltConversion conversion(he.view());
    BoltLinear linear(he.view());
    const bool first = run_case(he, conversion, linear, io, 768, 768);
    const bool second = run_case(he, conversion, linear, io, 768, 1024);
    const bool third = run_case(he, conversion, linear, io, 768, 3072);
    const bool fourth = run_case(he, conversion, linear, io, 3072, 768);
    const bool pass = first && second && third && fourth;
    if (party == ALICE)
      std::cout << "BOLT_VIT_LINEAR_TESTS=" << (pass ? "PASS" : "FAIL")
                << std::endl;
    return pass ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "BOLT_VIT_LINEAR_TEST error: " << error.what() << std::endl;
    return 2;
  }
}
