#include "BOLTTransformer/bolt_conversion.h"
#include "BOLTTransformer/bolt_he.h"
#include "BOLTTransformer/bolt_linear.h"
#include "utils/ArgMapping/ArgMapping.h"
#include "utils/io_pack.h"

#include <cstdint>
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
int port = 47420;
std::string address = "127.0.0.1";

BoltBertPackingMetadata metadata_for(const BoltHE &he) {
  BoltBertPackingMetadata metadata;
  metadata.slot_count = he.slot_count();
  metadata.image_size = 256;
  metadata.filter_height = 768;
  metadata.filter_width = 64;
  metadata.baby_step = 1;
  metadata.giant_steps = 16;
  metadata.attention_heads = 12;
  metadata.paired_head_groups = 6;
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
    throw std::runtime_error("BOLT ViT Softmax-V share size mismatch");
  alice_share.resize(size);
  io.io->recv_data(alice_share.data(), alice_share.size() * sizeof(std::uint64_t));
}

std::vector<std::uint64_t> pack_test_softmax(const BoltHE &he) {
  constexpr std::size_t rows = 256;
  constexpr std::size_t heads = 12;
  const std::size_t num_cts_per_group = rows * rows * 2 / he.slot_count();
  const std::size_t blocks_per_ct = he.slot_count() / (2 * rows);
  const std::size_t half = he.slot_count() / 2;
  std::vector<std::uint64_t> packed(
      heads * rows * rows, 0);
  for (std::size_t group = 0; group < heads / 2; ++group)
    for (std::size_t source_ct = 0; source_ct < num_cts_per_group;
         ++source_ct)
      for (std::size_t block = 0; block < blocks_per_ct; ++block) {
        const std::size_t diagonal = source_ct * blocks_per_ct + block;
        for (std::size_t row = 0; row < rows; ++row) {
          const std::size_t source_col = (row + diagonal) % rows;
          // A source-column-dependent diagonal value catches a reversed or
          // otherwise permuted diagonal schedule; identity would not.
          const std::uint64_t value = source_col + 1;
          const std::size_t ct = group * num_cts_per_group + source_ct;
          packed[ct * he.slot_count() + block * rows + row] = value;
          packed[ct * he.slot_count() + half + block * rows + row] = value;
        }
      }
  if (party == ALICE)
    std::fill(packed.begin(), packed.end(), 0ULL);
  return packed;
}

std::vector<std::uint64_t> pack_values(const BoltHE &he) {
  constexpr std::size_t rows = 256;
  constexpr std::size_t heads = 12;
  constexpr std::size_t head_dim = 64;
  const std::size_t cts_per_group =
      2 * rows * head_dim / he.slot_count();
  const std::size_t columns_per_ct = he.slot_count() / (2 * rows);
  const std::size_t half = he.slot_count() / 2;
  std::vector<std::uint64_t> packed(
      heads * rows * head_dim, 0);
  for (std::size_t group = 0; group < heads / 2; ++group)
    for (std::size_t within = 0; within < cts_per_group; ++within)
      for (std::size_t local_col = 0; local_col < columns_per_ct; ++local_col)
        for (std::size_t row = 0; row < rows; ++row) {
          const std::size_t ct = group * cts_per_group + within;
          const std::size_t even_head = group * 2;
          const std::size_t odd_head = even_head + 1;
          const std::size_t col = within * columns_per_ct + local_col;
          const std::int64_t even_value =
              static_cast<std::int64_t>((even_head + row + 2 * col) % 11) - 5;
          const std::int64_t odd_value =
              static_cast<std::int64_t>((odd_head + row + 2 * col) % 11) - 5;
          packed[ct * he.slot_count() + local_col * rows + row] =
              static_cast<std::uint64_t>(even_value) & ((1ULL << 38) - 1);
          packed[ct * he.slot_count() + half + local_col * rows + row] =
              static_cast<std::uint64_t>(odd_value) & ((1ULL << 38) - 1);
        }
  if (party == ALICE)
    std::fill(packed.begin(), packed.end(), 0ULL);
  return packed;
}

bool run_test(BoltHE &he, BoltConversion &conversion, BoltLinear &linear,
              IOPack &io) {
  const auto metadata = metadata_for(he);
  const auto softmax_local = pack_test_softmax(he);
  const auto value_local = pack_values(he);
  const auto encrypted_softmax = conversion.ss_to_he_packed_compat(
      softmax_local.data(), softmax_local.size(), 38);
  const auto encrypted_value = conversion.ss_to_he_packed_compat(
      value_local.data(), value_local.size(), 38);

  std::vector<std::vector<std::vector<seal::Ciphertext>>> packed_softmax;
  std::vector<seal::Ciphertext> result;
  const std::size_t softmax_cts =
      metadata.paired_head_groups * metadata.image_size *
      metadata.image_size * 2 / he.slot_count();
  const std::size_t value_cts = metadata.paired_head_groups *
      metadata.filter_width / (he.slot_count() / metadata.image_size / 2);
  BoltOperationCounts counts;
  if (party == ALICE) {
    packed_softmax = linear.bert_pack_softmax_diagonals(
        encrypted_softmax, metadata, &counts);
    result.resize(value_cts);
    linear.bert_softmax_v_matmul_compat(
        packed_softmax, encrypted_value, metadata, result, &counts);
  }

  const auto field = conversion.he_to_ss_packed_compat(result, value_cts, true);
  std::vector<std::uint64_t> alice_share;
  exchange_alice_share(io, field, alice_share);

  bool pass = true;
  if (party == BOB) {
    const std::uint64_t p = he.parameters().plain_modulus;
    const std::uint64_t half = (p - 1) / 2;
    const std::size_t slot_half = he.slot_count() / 2;
    const std::size_t rows = metadata.image_size;
    const std::size_t chunk_width = he.slot_count() / (2 * rows);
    const std::size_t cts_per_group = value_cts / metadata.paired_head_groups;
    for (std::size_t ct = 0; ct < value_cts && pass; ++ct)
      for (std::size_t slot = 0; slot < he.slot_count(); ++slot) {
        const std::size_t group = ct / cts_per_group;
        const std::size_t within = ct % cts_per_group;
        const std::size_t row = slot % rows;
        std::size_t col = slot / rows;
        const std::size_t head_local = slot >= slot_half ? 1 : 0;
        if (slot >= slot_half) col -= chunk_width;
        col += within * chunk_width;
        const std::size_t head = group * 2 + head_local;
        __int128 expected_signed = 0;
        for (std::size_t source = 0; source < rows; ++source) {
          const std::int64_t softmax_value =
              static_cast<std::int64_t>(source + 1);
          const std::int64_t value =
              static_cast<std::int64_t>((head + source + 2 * col) % 11) - 5;
          expected_signed += static_cast<__int128>(softmax_value) * value;
        }
        const std::uint64_t expected = static_cast<std::uint64_t>(
            (expected_signed % static_cast<__int128>(p) + p) % p);
        const std::size_t index = ct * he.slot_count() + slot;
        const std::uint64_t reconstructed =
            (field[index] + alice_share[index]) % p;
        const std::uint64_t decoded = (reconstructed + p - half) % p;
        if (decoded != expected) {
          std::cerr << "BOLT_VIT_SOFTMAX_V_FIRST_MISMATCH ct=" << ct
                    << " slot=" << slot << " head=" << head
                    << " row=" << row << " col=" << col
                    << " decoded=" << decoded
                    << " expected=" << expected << std::endl;
          pass = false;
          break;
        }
      }
    std::cout << "BOLT_VIT_SOFTMAX_V_TEST status="
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
    BoltHE he(party, io.io, BoltHEParameters::vit_main(), true);
    BoltConversion conversion(he.view());
    BoltLinear linear(he.view());
    return run_test(he, conversion, linear, io) ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "BOLT_VIT_SOFTMAX_V_TEST error: " << error.what()
              << std::endl;
    return 2;
  }
}
