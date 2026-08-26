#include "BOLTTransformer/bolt_he.h"
#include "BOLTTransformer/bolt_vit_compat_attention.h"
#include "BOLTTransformer/bolt_vit_compat_packing.h"

#include "LinearHE/utils-HE.h"
#include "utils/ArgMapping/ArgMapping.h"
#include "utils/io_pack.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace sci;
using namespace sci::bolt;

extern std::uint64_t prime_mod;
std::int32_t bitlength = 37;
std::int32_t num_threads = 1;

namespace {

constexpr std::uint64_t kPlainModulus = 536903681ULL;
constexpr std::size_t kSlots = 8192;
constexpr std::size_t kTokens = 256;
constexpr std::size_t kLogicalTokens = 197;
constexpr std::size_t kDim = 768;
constexpr std::size_t kHeads = 12;
constexpr std::size_t kHeadDim = 64;
constexpr std::uint32_t kFloodBits = 71;
constexpr std::uint32_t kModulusSwitches = 2;

std::uint64_t prime_encode(std::int64_t value) {
  if (value >= 0) return static_cast<std::uint64_t>(value) % kPlainModulus;
  const std::uint64_t magnitude = static_cast<std::uint64_t>(-value) %
                                  kPlainModulus;
  return magnitude == 0 ? 0 : kPlainModulus - magnitude;
}

std::int64_t centered(std::uint64_t value, std::uint64_t period) {
  return static_cast<std::int64_t>(value % period) -
         static_cast<std::int64_t>(period / 2);
}

std::int64_t q_value(std::size_t head, std::size_t token,
                     std::size_t feature) {
  return centered(head * 131 + token * 17 + feature * 29 + 7, 1021);
}

std::int64_t k_value(std::size_t head, std::size_t token,
                     std::size_t feature) {
  return centered(head * 313 + token * 43 + feature * 71 + 19, 4093);
}

std::int64_t v_value(std::size_t head, std::size_t token,
                     std::size_t feature) {
  return centered(head * 2017 + token * 811 + feature * 1213 + 101,
                  32749);
}

// Every valid row sums to exactly 4096, the production Softmax scale.  The
// 20/21 pattern is query/head dependent so every physical diagonal and both
// BatchEncoder halves carry distinguishable nonzero values.
std::uint64_t softmax_value(std::size_t head, std::size_t query,
                            std::size_t key) {
  const std::size_t rank =
      (key + head * 31 + query * 67) % kLogicalTokens;
  return rank < 156 ? 21 : 20;
}

std::vector<std::uint64_t> make_projection_token_major(char kind) {
  std::vector<std::uint64_t> result(kTokens * kDim);
  for (std::size_t token = 0; token < kTokens; ++token)
    for (std::size_t head = 0; head < kHeads; ++head)
      for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
        std::int64_t value = 0;
        if (kind == 'q') value = q_value(head, token, feature);
        else if (kind == 'k') value = k_value(head, token, feature);
        else if (kind == 'v') value = v_value(head, token, feature);
        else throw std::invalid_argument("unknown compact attention fixture");
        result[token * kDim + head * kHeadDim + feature] =
            prime_encode(value);
      }
  return result;
}

std::vector<std::uint64_t> make_logical_softmax() {
  std::vector<std::uint64_t> result(
      kHeads * kLogicalTokens * kLogicalTokens);
  for (std::size_t head = 0; head < kHeads; ++head)
    for (std::size_t query = 0; query < kLogicalTokens; ++query)
      for (std::size_t key = 0; key < kLogicalTokens; ++key)
        result[(head * kLogicalTokens + query) * kLogicalTokens + key] =
            softmax_value(head, query, key);
  return result;
}

std::vector<seal::Ciphertext> client_encrypt_packed(
    BoltHE &he, NetIO *io, const std::vector<std::uint64_t> &packed,
    std::size_t expected_ciphertexts) {
  if (packed.size() != expected_ciphertexts * kSlots)
    throw std::invalid_argument("compact attention encrypted input shape mismatch");
  std::vector<seal::Ciphertext> result(expected_ciphertexts);
  if (he.view().party == BOB) {
    for (std::size_t ct = 0; ct < expected_ciphertexts; ++ct) {
      seal::Plaintext plaintext;
      he.view().encoder->encode(
          std::vector<std::uint64_t>(packed.begin() + ct * kSlots,
                                     packed.begin() + (ct + 1) * kSlots),
          plaintext);
      he.view().encryptor->encrypt(plaintext, result[ct]);
    }
    send_encrypted_vector(io, result);
    result.clear();
  } else {
    recv_encrypted_vector(he.view().context, io, result);
    if (result.size() != expected_ciphertexts)
      throw std::runtime_error("compact attention encrypted input count mismatch");
  }
  return result;
}

struct DecodedCiphertexts {
  std::vector<std::uint64_t> slots;
  int minimum_noise = std::numeric_limits<int>::max();
};

DecodedCiphertexts receive_and_decode(BoltHE &he, NetIO *io,
                                      std::size_t ciphertexts) {
  std::vector<seal::Ciphertext> received(ciphertexts);
  recv_encrypted_vector(he.view().context, io, received);
  if (received.size() != ciphertexts)
    throw std::runtime_error("compact attention output count mismatch");
  DecodedCiphertexts result;
  result.slots.resize(ciphertexts * kSlots);
  for (std::size_t ct = 0; ct < ciphertexts; ++ct) {
    result.minimum_noise = std::min(
        result.minimum_noise,
        he.view().decryptor->invariant_noise_budget(received[ct]));
    seal::Plaintext plaintext;
    std::vector<std::uint64_t> decoded;
    he.view().decryptor->decrypt(received[ct], plaintext);
    he.view().encoder->decode(plaintext, decoded);
    if (decoded.size() != kSlots)
      throw std::runtime_error("BatchEncoder output slot count mismatch");
    std::copy(decoded.begin(), decoded.end(),
              result.slots.begin() + ct * kSlots);
  }
  return result;
}

BoltOperationCounts sum_counts(const BoltOperationCounts &lhs,
                               const BoltOperationCounts &rhs) {
  BoltOperationCounts result;
  result.ciphertexts = lhs.ciphertexts + rhs.ciphertexts;
  result.rotations = lhs.rotations + rhs.rotations;
  result.multiply_plain = lhs.multiply_plain + rhs.multiply_plain;
  result.evaluator_multiply = lhs.evaluator_multiply + rhs.evaluator_multiply;
  result.relinearizations = lhs.relinearizations + rhs.relinearizations;
  result.modulus_switches = lhs.modulus_switches + rhs.modulus_switches;
  result.noise_floodings = lhs.noise_floodings + rhs.noise_floodings;
  return result;
}

bool same_counts(const BoltOperationCounts &lhs,
                 const BoltOperationCounts &rhs) {
  return lhs.ciphertexts == rhs.ciphertexts &&
         lhs.rotations == rhs.rotations &&
         lhs.multiply_plain == rhs.multiply_plain &&
         lhs.evaluator_multiply == rhs.evaluator_multiply &&
         lhs.relinearizations == rhs.relinearizations &&
         lhs.modulus_switches == rhs.modulus_switches &&
         lhs.noise_floodings == rhs.noise_floodings;
}

void print_counts(const std::string &name, const BoltOperationCounts &counts,
                  bool pass) {
  std::cout << name << "=" << (pass ? "PASS" : "FAIL")
            << " ciphertexts=" << counts.ciphertexts
            << " rotations=" << counts.rotations
            << " multiply_plain=" << counts.multiply_plain
            << " ct_ct_multiply=" << counts.evaluator_multiply
            << " relinearizations=" << counts.relinearizations
            << " floodings=" << counts.noise_floodings
            << " mod_switches=" << counts.modulus_switches << '\n';
}

std::pair<std::size_t, std::size_t> validate_qk(
    const BoltVitCompatPacking &packing,
    const std::vector<std::uint64_t> &packed_slots) {
  const std::vector<std::uint64_t> actual =
      packing.unpack_qk_cross(packed_slots);
  std::size_t mismatches = 0;
  std::size_t first = actual.size();
  const std::int64_t half = static_cast<std::int64_t>((kPlainModulus - 1) / 2);
  for (std::size_t head = 0; head < kHeads; ++head)
    for (std::size_t query = 0; query < kTokens; ++query)
      for (std::size_t key = 0; key < kTokens; ++key) {
        std::int64_t expected = 0;
        for (std::size_t feature = 0; feature < kHeadDim; ++feature)
          expected += q_value(head, query, feature) *
                      k_value(head, key, feature);
        if (expected < -half || expected > half)
          throw std::overflow_error("QK fixture exceeds the BFV centered field");
        const std::size_t index =
            (head * kTokens + query) * kTokens + key;
        if (actual[index] != prime_encode(expected)) {
          if (first == actual.size()) first = index;
          ++mismatches;
        }
      }
  return {mismatches, first};
}

std::pair<std::size_t, std::size_t> validate_context(
    const BoltVitCompatPacking &packing,
    const std::vector<std::uint64_t> &packed_slots) {
  const std::vector<std::uint64_t> actual = packing.unpack_context(packed_slots);
  std::size_t mismatches = 0;
  std::size_t first = actual.size();
  const std::int64_t half = static_cast<std::int64_t>((kPlainModulus - 1) / 2);
  for (std::size_t query = 0; query < kTokens; ++query)
    for (std::size_t head = 0; head < kHeads; ++head)
      for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
        std::int64_t expected = 0;
        if (query < kLogicalTokens)
          for (std::size_t key = 0; key < kLogicalTokens; ++key)
            expected += static_cast<std::int64_t>(
                            softmax_value(head, query, key)) *
                        v_value(head, key, feature);
        if (expected < -half || expected > half)
          throw std::overflow_error(
              "Softmax-V fixture exceeds the BFV centered field");
        const std::size_t index =
            query * kDim + head * kHeadDim + feature;
        if (actual[index] != prime_encode(expected)) {
          if (first == actual.size()) first = index;
          ++mismatches;
        }
      }
  return {mismatches, first};
}

double elapsed_seconds(const std::chrono::steady_clock::time_point &start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

int main(int argc, char **argv) {
  int party = 0;
  int port = 52630;
  std::string address = "127.0.0.1";
  ArgMapping args;
  args.arg("r", party, "Role: ALICE=1, BOB=2");
  args.arg("p", port, "TCP port");
  args.arg("ip", address, "ALICE address (BOB only)");
  args.parse(argc, argv);

  try {
    if (party != ALICE && party != BOB)
      throw std::invalid_argument("r must be ALICE=1 or BOB=2");
    prime_mod = kPlainModulus;
    IOPack io(party, port, address);
    BoltHE he(party, io.io, BoltHEParameters::vit_b16());
    const BoltVitCompatPacking packing;
    BoltVitCompatAttentionAdapter attention(
        he.view(), packing, kFloodBits, kModulusSwitches);
    std::cout << "VIT_COMPACT_ATTENTION_STATUS role=" << party
              << " phase=inputs status=start" << std::endl;

    const auto q_packed =
        packing.pack_context(make_projection_token_major('q'));
    const auto k_packed =
        packing.pack_context(make_projection_token_major('k'));
    const auto v_packed =
        packing.pack_context(make_projection_token_major('v'));
    std::vector<seal::Ciphertext> q = client_encrypt_packed(
        he, io.io, q_packed, packing.projection_ciphertexts());
    std::vector<seal::Ciphertext> k = client_encrypt_packed(
        he, io.io, k_packed, packing.projection_ciphertexts());
    std::vector<seal::Ciphertext> v = client_encrypt_packed(
        he, io.io, v_packed, packing.projection_ciphertexts());

    bool local_pass = true;
    BoltOperationCounts qk_counts;
    std::vector<seal::Ciphertext> qk_output;
    const auto qk_start = std::chrono::steady_clock::now();
    if (party == ALICE) {
      std::cout << "VIT_COMPACT_ATTENTION_STATUS role=1 phase=qk status=start"
                << std::endl;
      std::vector<seal::Ciphertext> fused;
      fused.reserve(packing.fused_qkv_ciphertexts());
      fused.insert(fused.end(), q.begin(), q.end());
      fused.insert(fused.end(), k.begin(), k.end());
      fused.insert(fused.end(), v.begin(), v.end());
      qk_output = attention.qk_cross(fused, &qk_counts);
      send_encrypted_vector(io.io, qk_output);

      BoltOperationCounts expected = packing.expected_counts().qk;
      expected.noise_floodings += packing.qk_ciphertexts();
      expected.modulus_switches +=
          kModulusSwitches * packing.qk_ciphertexts();
      const bool counts_pass = same_counts(qk_counts, expected);
      print_counts("VIT_COMPACT_QK_COUNTS_TEST", qk_counts, counts_pass);
      local_pass = local_pass && counts_pass;
      std::cout << "VIT_COMPACT_ATTENTION_STATUS role=1 phase=qk status=done"
                << " seconds=" << elapsed_seconds(qk_start) << std::endl;
      q.clear();
      k.clear();
      qk_output.clear();
    } else {
      DecodedCiphertexts decoded = receive_and_decode(
          he, io.io, packing.qk_ciphertexts());
      const auto validation = validate_qk(packing, decoded.slots);
      const bool numeric_pass = validation.first == 0;
      const bool noise_pass = decoded.minimum_noise > 0;
      std::cout << "VIT_COMPACT_QK_FULL_NUMERIC_TEST="
                << (numeric_pass ? "PASS" : "FAIL")
                << " checked=" << kHeads * kTokens * kTokens
                << " mismatches=" << validation.first
                << " minimum_noise_budget=" << decoded.minimum_noise;
      if (!numeric_pass) std::cout << " first_mismatch=" << validation.second;
      std::cout << '\n';
      std::cout << "VIT_COMPACT_QK_HARDENING_TEST="
                << (noise_pass ? "PASS" : "FAIL")
                << " flood_bits=" << kFloodBits
                << " mod_switches=" << kModulusSwitches
                << " minimum_noise_budget=" << decoded.minimum_noise << '\n';
      local_pass = local_pass && numeric_pass && noise_pass;
    }

    const auto logical_softmax = make_logical_softmax();
    const auto packed_softmax =
        packing.pack_logical_softmax_diagonal_source(
            logical_softmax, kLogicalTokens);
    std::vector<seal::Ciphertext> softmax = client_encrypt_packed(
        he, io.io, packed_softmax, packing.softmax_ciphertexts());

    BoltOperationCounts sv_counts;
    const auto sv_start = std::chrono::steady_clock::now();
    if (party == ALICE) {
      std::cout << "VIT_COMPACT_ATTENTION_STATUS role=1 phase=softmax_v status=start"
                << std::endl;
      std::vector<seal::Ciphertext> context;
      {
        auto diagonals = attention.expand_softmax_diagonals(
            softmax, &sv_counts);
        softmax.clear();
        context = attention.softmax_v(diagonals, v, &sv_counts);
      }
      send_encrypted_vector(io.io, context);

      const auto expected_parts = packing.expected_counts();
      BoltOperationCounts expected = sum_counts(
          expected_parts.softmax_expand, expected_parts.softmax_v);
      expected.noise_floodings += packing.context_ciphertexts();
      expected.modulus_switches +=
          kModulusSwitches * packing.context_ciphertexts();
      const bool counts_pass = same_counts(sv_counts, expected);
      print_counts("VIT_COMPACT_SOFTMAX_V_COUNTS_TEST", sv_counts,
                   counts_pass);
      local_pass = local_pass && counts_pass;
      std::cout
          << "VIT_COMPACT_ATTENTION_STATUS role=1 phase=softmax_v status=done"
          << " seconds=" << elapsed_seconds(sv_start) << std::endl;
    } else {
      DecodedCiphertexts decoded = receive_and_decode(
          he, io.io, packing.context_ciphertexts());
      const auto validation = validate_context(packing, decoded.slots);
      const bool numeric_pass = validation.first == 0;
      const bool noise_pass = decoded.minimum_noise > 0;
      std::cout << "VIT_COMPACT_SOFTMAX_V_FULL_NUMERIC_TEST="
                << (numeric_pass ? "PASS" : "FAIL")
                << " checked=" << kTokens * kDim
                << " mismatches=" << validation.first
                << " logical_tokens=" << kLogicalTokens
                << " padding_tokens=" << kTokens - kLogicalTokens
                << " minimum_noise_budget=" << decoded.minimum_noise;
      if (!numeric_pass) std::cout << " first_mismatch=" << validation.second;
      std::cout << '\n';
      std::cout << "VIT_COMPACT_SOFTMAX_V_HARDENING_TEST="
                << (noise_pass ? "PASS" : "FAIL")
                << " flood_bits=" << kFloodBits
                << " mod_switches=" << kModulusSwitches
                << " minimum_noise_budget=" << decoded.minimum_noise << '\n';
      local_pass = local_pass && numeric_pass && noise_pass;
    }

    std::cout << "VIT_COMPACT_ATTENTION_INTEGRATION_TEST="
              << (local_pass ? "PASS" : "FAIL")
              << " role=" << party
              << " q_scale=10 k_scale=10 qk_scale=20"
              << " softmax_scale=12 v_scale=12 context_scale=24"
              << std::endl;
    return local_pass ? 0 : 3;
  } catch (const std::exception &error) {
    std::cerr << "VIT_COMPACT_ATTENTION_INTEGRATION_TEST=FAIL role=" << party
              << " error=" << error.what() << std::endl;
    return 2;
  }
}
