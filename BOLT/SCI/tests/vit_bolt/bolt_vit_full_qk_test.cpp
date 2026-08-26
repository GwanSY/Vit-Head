#include "BOLTTransformer/bolt_conversion.h"
#include "BOLTTransformer/bolt_he.h"
#include "BOLTTransformer/bolt_linear.h"
#include "LinearHE/defines-HE.h"
#include "utils/ArgMapping/ArgMapping.h"
#include "utils/io_pack.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace sci;
using namespace sci::bolt;

std::uint64_t prime_mod = 536903681ULL;
std::int32_t bitlength = 37;
std::int32_t num_threads = 1;

namespace {

int party = 0;
int port = 47400;
std::string address = "127.0.0.1";

BoltBertPackingMetadata qk_metadata(const BoltHE &he) {
  BoltBertPackingMetadata metadata;
  metadata.slot_count = he.slot_count();
  metadata.image_size = 256;
  metadata.filter_height = 768;
  metadata.filter_width = 64;
  metadata.baby_step = 4;
  metadata.giant_steps = 4;
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
    throw std::runtime_error("full QK share size mismatch");
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
    BoltLinear linear(he.view());
    const auto metadata = qk_metadata(he);
    const std::uint64_t plaintext_modulus = he.parameters().plain_modulus;
    const std::size_t qk_block = metadata.image_size * metadata.filter_width *
                                 metadata.attention_heads / he.slot_count();
    const std::size_t input_ciphertexts =
        metadata.image_size * metadata.filter_height / he.slot_count();
    std::vector<std::uint64_t> local(input_ciphertexts * he.slot_count(), 0);
    if (party == BOB) {
      for (std::size_t ct = 0; ct < input_ciphertexts; ++ct)
        for (std::size_t slot = 0; slot < he.slot_count(); ++slot) {
          const std::size_t input_col =
              ct * (he.slot_count() / metadata.image_size) +
              slot / metadata.image_size;
          const std::size_t row = slot % metadata.image_size;
          const std::int64_t value =
              static_cast<std::int64_t>((input_col + 3 * row) % 9) - 4;
          local[ct * he.slot_count() + slot] =
              static_cast<std::uint64_t>(value) & ((1ULL << 38) - 1);
        }
    }
    const auto encrypted_input = conversion.ss_to_he_packed_compat(
        local.data(), local.size(), 38);

    // Signed head-dependent Q/K/V weights and biases, together with an
    // input-dimension-dependent encrypted vector, exercise every diagonal of
    // the fused BSGS pack.  A constant input would not detect a wrong
    // half-row rotation period.
    std::vector<std::vector<std::vector<std::uint64_t>>> weights(
        metadata.attention_heads,
        std::vector<std::vector<std::uint64_t>>(
            metadata.filter_height,
            std::vector<std::uint64_t>(metadata.filter_width, 1)));
    std::vector<std::vector<std::uint64_t>> biases(
        metadata.attention_heads,
        std::vector<std::uint64_t>(metadata.filter_width, 0));
    for (std::size_t head = 0; head < metadata.attention_heads; ++head) {
      for (std::size_t input = 0; input < metadata.filter_height; ++input)
        for (std::size_t output = 0; output < metadata.filter_width; ++output)
          {
            const std::int64_t weight = static_cast<std::int64_t>(
                (input * 5 + output * 2 + head * 3) % 13) - 6;
            weights[head][input][output] =
                weight >= 0
                    ? static_cast<std::uint64_t>(weight)
                    : plaintext_modulus - static_cast<std::uint64_t>(-weight);
          }
      for (std::size_t output = 0; output < metadata.filter_width; ++output) {
        const std::int64_t bias = static_cast<std::int64_t>(
            (head * 3 + output * 5) % 11) - 5;
        biases[head][output] =
            bias >= 0
                ? static_cast<std::uint64_t>(bias)
                : plaintext_modulus - static_cast<std::uint64_t>(-bias);
      }
    }
    std::vector<seal::Ciphertext> qkv;
    BoltOperationCounts qkv_counts;
    if (party == ALICE) {
      const auto packed = linear.bert_pack_qkv_matrix(weights, metadata);
      const auto packed_bias = linear.bert_pack_qkv_bias(biases, metadata);
      const std::size_t one_projection = qk_block;
      qkv.resize(3 * one_projection);
      linear.bert_qkv_cipher_plain_matmul_bsgs_compat(
          encrypted_input, packed, packed, packed, packed_bias, packed_bias,
          packed_bias, metadata, qkv, &qkv_counts);
      // The production ViT graph crosses HE->SS and re-encrypts Q/K before
      // QK.  Two level drops reproduce the original BERT raw-QKV hardening
      // for this kernel-only diagnostic and prevent an artificial BFV noise
      // exhaustion from being mistaken for a packing error.
      for (auto &ciphertext : qkv) {
        he.view().evaluator->mod_switch_to_next_inplace(ciphertext);
        he.view().evaluator->mod_switch_to_next_inplace(ciphertext);
      }
    }

    // Check the QKV HE-to-SS boundary before it is consumed by QK.  The
    // expected value is the weighted input sum plus the head bias.
    const std::size_t qkv_output_ciphertexts = 3 * qk_block;
    const auto qkv_field = conversion.he_to_ss_packed_compat(
        qkv, qkv_output_ciphertexts, true);
    std::vector<std::uint64_t> qkv_alice_share;
    exchange_alice_share(io, qkv_field, qkv_alice_share);
    // Validate every packed QKV slot.  A packed ciphertext group contains
    // the even head in the upper half and the odd head in the lower half.
    bool qkv_pass = true;
    if (party == BOB) {
      const std::uint64_t p = he.parameters().plain_modulus;
      const std::uint64_t half = (p - 1) / 2;
      std::cerr << "FULL_QKV_SAMPLES";
      for (std::size_t ct : {std::size_t(0), std::size_t(4),
                             std::size_t(24), std::size_t(28)}) {
        const std::size_t upper = ct * he.slot_count();
        const std::size_t lower = upper + he.slot_count() / 2;
        const std::uint64_t upper_value =
            ((qkv_field[upper] + qkv_alice_share[upper]) % p + p - half) % p;
        const std::uint64_t lower_value =
            ((qkv_field[lower] + qkv_alice_share[lower]) % p + p - half) % p;
        std::cerr << " ct" << ct << "=" << upper_value << "/" << lower_value;
      }
      std::cerr << std::endl;
      for (std::size_t i = 0; i < qkv_field.size(); ++i) {
        const std::size_t ct = i / he.slot_count();
        const std::size_t slot = i % he.slot_count();
        const std::size_t local_ct = ct % qk_block;
        const std::size_t head = (local_ct / 4) * 2 +
                                 (slot >= he.slot_count() / 2 ? 1 : 0);
        const std::uint64_t reconstructed =
            (qkv_field[i] + qkv_alice_share[i]) % p;
        const std::uint64_t decoded =
            (reconstructed + p - half) % p;
        const std::size_t row = slot % metadata.image_size;
        const std::size_t half_columns =
            he.slot_count() / (2 * metadata.image_size);
        const std::size_t output =
            (local_ct % 4) * half_columns +
            (slot >= he.slot_count() / 2
                 ? (slot - he.slot_count() / 2) / metadata.image_size
                 : slot / metadata.image_size);
        __int128 expected_signed = static_cast<std::int64_t>(
            (head * 3 + output * 5) % 11) - 5;
        for (std::size_t input = 0; input < metadata.filter_height; ++input) {
          const std::int64_t input_value =
              static_cast<std::int64_t>((input + 3 * row) % 9) - 4;
          const std::int64_t weight = static_cast<std::int64_t>(
              (input * 5 + output * 2 + head * 3) % 13) - 6;
          expected_signed += input_value * weight;
        }
        const std::uint64_t expected = static_cast<std::uint64_t>(
            (expected_signed % static_cast<__int128>(p) + p) % p);
        if (decoded != expected) {
          qkv_pass = false;
          std::cerr << "FULL_QKV_FIRST_MISMATCH index=" << i
                    << " head=" << head << " decoded=" << decoded
                    << " expected=" << expected << std::endl;
          break;
        }
      }
    }

    // Re-encrypt known packed Q/K/V values for the QK-only assertion.  The
    // head-dimension term is intentional: a head-dimension-constant fixture
    // would validate only token/head placement and could miss a wrong inner
    // product mapping in the packed ct-ct kernel.
    std::vector<std::uint64_t> clean_local(
        qkv_output_ciphertexts * he.slot_count(), 0);
    for (std::size_t ct = 0; ct < qkv_output_ciphertexts; ++ct) {
      const std::size_t local_ct = ct % qk_block;
      const std::size_t group = local_ct / 4;
      const std::size_t head_base = group * 2;
      for (std::size_t slot = 0; slot < he.slot_count(); ++slot) {
        const std::size_t head = head_base +
                                 (slot >= he.slot_count() / 2 ? 1 : 0);
        const std::size_t row = slot % metadata.image_size;
        const std::size_t half_columns =
            he.slot_count() / (2 * metadata.image_size);
        const std::size_t dimension =
            (local_ct % 4) * half_columns +
            (slot >= he.slot_count() / 2
                 ? (slot - he.slot_count() / 2) / metadata.image_size
                 : slot / metadata.image_size);
        const std::int64_t value = static_cast<std::int64_t>(
            (row + 3 * head + 5 * dimension) % 9) - 4;
        clean_local[ct * he.slot_count() + slot] =
            party == BOB ? (static_cast<std::uint64_t>(value) &
                            ((1ULL << 38) - 1))
                         : 0;
      }
    }
    const auto clean_qkv = conversion.ss_to_he_packed_compat(
        clean_local.data(), clean_local.size(), 38);

    std::vector<seal::Ciphertext> qk;
    BoltOperationCounts counts;
    if (party == ALICE) {
      const std::size_t result_ciphertexts =
          metadata.paired_head_groups * metadata.image_size *
          metadata.image_size * 2 / he.slot_count();
      qk.resize(result_ciphertexts);
      linear.bert_qk_matmul_cross_compat(
          clean_qkv,
          linear.bert_generate_cross_packing_masks(metadata.image_size),
          metadata, qk, &counts);
    }

    const std::size_t result_ciphertexts =
        metadata.paired_head_groups * metadata.image_size *
        metadata.image_size * 2 / he.slot_count();
    const auto field = conversion.he_to_ss_packed_compat(
        qk, result_ciphertexts, true);
    std::vector<std::uint64_t> alice_share;
    exchange_alice_share(io, field, alice_share);

    bool pass = qkv_pass;
    if (party == BOB) {
      const std::uint64_t p = he.parameters().plain_modulus;
      const std::uint64_t half = (p - 1) / 2;
      const std::size_t qk_per_group = metadata.image_size *
                                       metadata.image_size * 2 /
                                       he.slot_count();
      const std::size_t qk_columns_per_ct = he.slot_count() /
                                            (2 * metadata.image_size);
      std::cerr << "FULL_QK_SAMPLES";
      for (std::size_t head = 0; head < metadata.attention_heads; ++head) {
        const std::size_t group = head / 2;
        const std::size_t ct = group * qk_per_group;
        const std::size_t slot = (head % 2) * he.slot_count() / 2;
        const std::size_t index = ct * he.slot_count() + slot;
        const std::uint64_t value =
            ((field[index] + alice_share[index]) % p + p - half) % p;
        std::cerr << " h" << head << "=" << value;
      }
      std::cerr << std::endl;
      const std::size_t per_group = metadata.image_size * metadata.image_size *
                                    2 / he.slot_count();
      const std::size_t columns_per_ct = he.slot_count() /
                                         (2 * metadata.image_size);
      for (std::size_t i = 0; i < field.size(); ++i) {
        const std::uint64_t reconstructed =
            (field[i] + alice_share[i]) % p;
        const std::uint64_t decoded =
            (reconstructed + p - half) % p;
        const std::size_t ct = i / he.slot_count();
        const std::size_t slot = i % he.slot_count();
        const std::size_t group = ct / per_group;
        const std::size_t current_col = ct % per_group;
        std::size_t column = slot / metadata.image_size +
                             current_col * columns_per_ct;
        const std::size_t row = slot % metadata.image_size;
        const std::size_t head = group * 2 +
                                 (slot >= he.slot_count() / 2 ? 1 : 0);
        if (slot >= he.slot_count() / 2)
          column -= he.slot_count() / (2 * metadata.image_size);
        const std::size_t source_row = (row + column) % metadata.image_size;
        __int128 expected_qk_signed = 0;
        for (std::size_t dimension = 0;
             dimension < metadata.filter_width; ++dimension) {
          const std::int64_t q_value = static_cast<std::int64_t>(
              (row + 3 * head + 5 * dimension) % 9) - 4;
          const std::int64_t k_value = static_cast<std::int64_t>(
              (source_row + 3 * head + 5 * dimension) % 9) - 4;
          expected_qk_signed += static_cast<__int128>(q_value) * k_value;
        }
        const std::uint64_t expected_qk = static_cast<std::uint64_t>(
            (expected_qk_signed % static_cast<__int128>(p) + p) % p);
        const std::size_t output_index =
            head * metadata.image_size * metadata.image_size +
            row * metadata.image_size +
            (row + column) % metadata.image_size;
        (void)output_index;
        if (decoded != expected_qk) {
          pass = false;
          std::cerr << "FULL_QK_FIRST_MISMATCH index=" << i
                    << " head=" << head
                    << " decoded=" << decoded
                    << " reconstructed=" << reconstructed << std::endl;
          break;
        }
      }
    }

    std::uint8_t status = pass ? 1 : 0;
    if (party == BOB)
      io.io->send_data(&status, sizeof(status));
    else
      io.io->recv_data(&status, sizeof(status));
    if (party == ALICE) {
      std::cout << "FULL_QK_CONSTANT_TEST rotations=" << counts.rotations
                << " qkv_multiply_plain=" << qkv_counts.multiply_plain
                << " evaluator_multiply=" << counts.evaluator_multiply
                << " relinearizations=" << counts.relinearizations
                << " multiply_plain=" << counts.multiply_plain
                << " status=" << (status ? "PASS" : "FAIL") << std::endl;
    }
    return status ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "BOLT_VIT_FULL_QK_TEST error: " << error.what() << std::endl;
    return 1;
  }
}
