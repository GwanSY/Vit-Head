#include "BOLTTransformer/bolt_conversion.h"
#include "BOLTTransformer/bolt_he.h"
#include "BOLTTransformer/bolt_linear.h"

#include "FloatingPoint/fp-math.h"
#include "LinearHE/utils-HE.h"
#include "OT/ot_pack.h"
#include "utils/ArgMapping/ArgMapping.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sci;
using namespace sci::bolt;

extern std::uint64_t prime_mod;
std::int32_t bitlength = 37;
std::int32_t num_threads = 1;

namespace {

constexpr std::uint64_t kPlainModulus = 536903681ULL;
constexpr std::uint32_t kEll = 37;

std::int64_t centered(std::uint64_t value) {
  value %= kPlainModulus;
  return value <= (kPlainModulus - 1) / 2
             ? static_cast<std::int64_t>(value)
             : static_cast<std::int64_t>(value) -
                   static_cast<std::int64_t>(kPlainModulus);
}

} // namespace

int main(int argc, char **argv) {
  int party = 0;
  int port = 52300;
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
    OTPack ot(&io, party);
    FPMath fp(party, &io, &ot);

    const std::uint64_t variance_value = 103424;
    const std::uint64_t variance_share = (1ULL << kEll) - 1234567;
    FixArray variance(party, 1, true, kEll, 12);
    for (int i = 0; i < variance.size; ++i)
      variance.data[i] = party == ALICE
                             ? variance_share
                             : (variance_value - variance_share) &
                                 ((1ULL << kEll) - 1);
    FixArray inverse_sigma = fp.sqrt(variance, true);
    FixArray inverse_sigma_open = fp.fix->output(PUBLIC, inverse_sigma);
    const bool rsqrt_pass = inverse_sigma_open.data[0] == 815;
    if (party == BOB)
      std::cout << "MPC_RSQRT_BIT_EXACT_TEST="
                << (rsqrt_pass ? "PASS" : "FAIL")
                << " expected=815 actual=" << inverse_sigma_open.data[0]
                << std::endl;

    const std::uint64_t ring_modulus = 1ULL << kEll;
    const std::vector<std::uint64_t> clear{
        1, 1, ring_modulus - 1, ring_modulus - 12345,
        0, 12345, 200000000, ring_modulus - 200000000};
    const std::vector<std::uint64_t> alice_shares{
        0, ring_modulus - 17, 17, ring_modulus - 1000,
        0, 12345, ring_modulus - 17, 17};
    const std::vector<std::uint8_t> expected_wrap{0, 1, 0, 1, 0, 0, 1, 0};
    const std::vector<std::uint8_t> expected_sign{0, 0, 1, 1, 0, 0, 0, 1};
    const std::uint64_t ring_mask = ring_modulus - 1;
    FixArray input(party, static_cast<int>(clear.size()), true, kEll, 12);
    for (std::size_t i = 0; i < clear.size(); ++i) {
      input.data[i] = party == ALICE
                          ? alice_shares[i]
                          : (clear[i] - alice_shares[i]) & ring_mask;
    }
    bool share_coverage_pass = true;
    for (std::size_t i = 0; i < clear.size(); ++i) {
      const std::uint64_t bob_share =
          (clear[i] - alice_shares[i]) & ring_mask;
      const std::uint8_t wrap = alice_shares[i] > clear[i];
      const std::uint8_t sign = clear[i] >= (ring_modulus >> 1);
      share_coverage_pass = share_coverage_pass &&
                            ((alice_shares[i] + bob_share) & ring_mask) ==
                                clear[i] &&
                            wrap == expected_wrap[i] &&
                            sign == expected_sign[i];
    }
    if (party == BOB)
      std::cout << "SIGNED_CONVERSION_SHARE_COVERAGE_TEST="
                << (share_coverage_pass ? "PASS" : "FAIL") << std::endl;

    BoltTensorMetadata metadata;
    metadata.rows = 1;
    metadata.cols = clear.size();
    metadata.packed_stride = 8;
    metadata.share_bit_width = kEll;
    metadata.scale = 12;
    SecretSharedTensor shares;
    shares.metadata = metadata;
    shares.local_share.assign(input.data, input.data + input.size);
    BoltConversion conversion(he.view());
    BoltCipherTensor converted = conversion.ss_to_he_signed_ring(
        shares, metadata, *fp.fix);

    if (party == ALICE) {
      send_encrypted_vector(io.io, converted.ciphertexts);
    } else {
      std::vector<seal::Ciphertext> ciphertexts(1);
      recv_encrypted_vector(he.view().context, io.io, ciphertexts);
      if (ciphertexts.size() != 1)
        throw std::runtime_error("signed conversion returned wrong CT count");
      const int noise = he.view().decryptor->invariant_noise_budget(
          ciphertexts.front());
      seal::Plaintext plaintext;
      std::vector<std::uint64_t> slots;
      he.view().decryptor->decrypt(ciphertexts.front(), plaintext);
      he.view().encoder->decode(plaintext, slots);
      bool pass = true;
      std::cout << "SIGNED_CONVERSION_NOISE_BUDGET=" << noise << '\n';
      for (std::size_t i = 0; i < clear.size(); ++i) {
        const std::int64_t expected =
            clear[i] < (ring_modulus >> 1)
                ? static_cast<std::int64_t>(clear[i])
                : static_cast<std::int64_t>(clear[i] - ring_modulus);
        const std::int64_t actual = centered(slots[i]);
        std::cout << "SIGNED_CONVERSION_CASE index=" << i
                  << " expected=" << expected << " actual=" << actual
                  << '\n';
        pass = pass && expected == actual;
      }
      bool repeated_pass = true;
      for (std::size_t slot = 0; slot < slots.size(); ++slot) {
        const std::size_t col = slot % metadata.packed_stride;
        const std::int64_t expected =
            clear[col] < (ring_modulus >> 1)
                ? static_cast<std::int64_t>(clear[col])
                : static_cast<std::int64_t>(clear[col] - ring_modulus);
        repeated_pass = repeated_pass && centered(slots[slot]) == expected;
      }
      pass = pass && repeated_pass;
      std::cout << "SIGNED_CONVERSION_REPEATED_LAYOUT_TEST="
                << (repeated_pass ? "PASS" : "FAIL") << '\n';
      std::cout << "SIGNED_CONVERSION_TEST=" << (pass ? "PASS" : "FAIL")
                << std::endl;
      if (!pass) return 3;
    }

    // Continue through the same identity ct-pt BSGS and HE->SS boundary used
    // by the classifier.  This catches packing or field-to-ring errors that a
    // direct conversion decrypt cannot expose.
    BoltCipherTensor linear_output;
    if (party == ALICE) {
      PlainWeightTensor weight;
      weight.input_cols = clear.size();
      weight.output_cols = 3;
      weight.values.assign(weight.input_cols * weight.output_cols, 0);
      for (std::size_t i = 0; i < weight.output_cols; ++i)
        weight.values[i * weight.input_cols + i] = 1ULL << 11;
      BoltMatMulMetadata linear_metadata;
      linear_metadata.input_rows = 1;
      linear_metadata.input_cols = clear.size();
      linear_metadata.output_cols = 3;
      linear_metadata.packed_stride = 8;
      linear_metadata.baby_step = 4;
      BoltLinear linear(he.view());
      linear_output = linear.cipher_plain_matmul_bsgs(
          converted, weight, linear_metadata);
      linear_output.metadata.scale = 23;
    } else {
      linear_output.metadata = metadata;
      linear_output.metadata.cols = 3;
      linear_output.metadata.scale = 23;
    }
    BoltTensorMetadata output_metadata = linear_output.metadata;
    output_metadata.add_plain_modulus_half_on_he_to_ss = true;
    SecretSharedTensor field = conversion.he_to_ss(linear_output,
                                                    output_metadata);
    FixArray field_input(party, 3, true, kEll, 23);
    for (std::size_t i = 0; i < 3; ++i)
      field_input.data[i] = field.local_share[i];
    FixArray p = fp.fix->input(PUBLIC, 3, kPlainModulus, true, kEll, 23);
    FixArray p_half = fp.fix->input(PUBLIC, 3, (kPlainModulus - 1) / 2,
                                    true, kEll, 23);
    FixArray ring = fp.fix->sub(fp.gt_p_sub(field_input, p), p_half);
    ring = fp.fix->right_shift(ring, 11);
    if (ring.ell < static_cast<int>(kEll))
      ring = fp.fix->extend(ring, kEll);
    FixArray opened = fp.fix->output(PUBLIC, ring);
    bool roundtrip_pass = true;
    for (std::size_t i = 0; i < 3; ++i) {
      const std::int64_t actual =
          opened.data[i] < (ring_modulus >> 1)
              ? static_cast<std::int64_t>(opened.data[i])
              : static_cast<std::int64_t>(opened.data[i] - ring_modulus);
      const std::int64_t expected =
          clear[i] < (ring_modulus >> 1)
              ? static_cast<std::int64_t>(clear[i])
              : static_cast<std::int64_t>(clear[i] - ring_modulus);
      if (party == BOB)
        std::cout << "SIGNED_LINEAR_ROUNDTRIP_CASE index=" << i
                  << " expected=" << expected << " actual=" << actual
                  << '\n';
      roundtrip_pass = roundtrip_pass && expected == actual;
    }
    if (party == BOB)
      std::cout << "SIGNED_LINEAR_ROUNDTRIP_TEST="
                << (roundtrip_pass ? "PASS" : "FAIL") << std::endl;
    if (party == BOB && !roundtrip_pass) return 4;

    bool dense_pass_all = true;
    SecretSharedTensor trivial_shares;
    trivial_shares.metadata = metadata;
    trivial_shares.local_share.assign(clear.size(), 0);
    if (party == BOB) trivial_shares.local_share = clear;
    BoltCipherTensor trivial_cipher = conversion.ss_to_he(
        trivial_shares, metadata);
    if (party == ALICE) {
      std::vector<seal::Ciphertext> probes(3);
      probes[0] = trivial_cipher.ciphertexts.front();
      he.view().rotate_rows_logical(trivial_cipher.ciphertexts.front(), 1,
                                    probes[1]);
      he.view().rotate_rows_logical(trivial_cipher.ciphertexts.front(), -1,
                                    probes[2]);
      send_encrypted_vector(io.io, probes);
    } else {
      std::vector<seal::Ciphertext> probes(3);
      recv_encrypted_vector(he.view().context, io.io, probes);
      for (std::size_t probe = 0; probe < probes.size(); ++probe) {
        seal::Plaintext plaintext;
        std::vector<std::uint64_t> slots;
        he.view().decryptor->decrypt(probes[probe], plaintext);
        he.view().encoder->decode(plaintext, slots);
        const int step = probe == 0 ? 0 : (probe == 1 ? 1 : -1);
        std::cout << "ROTATION_PROBE step=" << step;
        for (std::size_t slot = 0; slot < 8; ++slot)
          std::cout << ' ' << centered(slots[slot]);
        std::cout << '\n';
      }
    }
    const std::vector<std::int64_t> dense_coeff{
        237, -311, 19, 401, -127, 83, -509, 61,
        -73, 211, -389, 17, 293, -157, 43, 521,
        109, -47, 313, -229, 7, 419, -181, 97};
    BoltCipherTensor dense_output;
    if (party == ALICE) {
      PlainWeightTensor dense_weight;
      dense_weight.input_cols = clear.size();
      dense_weight.output_cols = 3;
      dense_weight.values.resize(dense_coeff.size());
      for (std::size_t i = 0; i < dense_coeff.size(); ++i)
        dense_weight.values[i] = dense_coeff[i] >= 0
            ? static_cast<std::uint64_t>(dense_coeff[i])
            : kPlainModulus - static_cast<std::uint64_t>(-dense_coeff[i]);
      BoltMatMulMetadata dense_metadata;
      dense_metadata.input_rows = 1;
      dense_metadata.input_cols = clear.size();
      dense_metadata.output_cols = 3;
      dense_metadata.packed_stride = 8;
      dense_metadata.baby_step = 4;
      BoltLinear linear(he.view());
      dense_output = linear.cipher_plain_matmul_bsgs(
          converted, dense_weight, dense_metadata);
      send_encrypted_vector(io.io, dense_output.ciphertexts);
    } else {
      std::vector<seal::Ciphertext> ciphertexts(1);
      recv_encrypted_vector(he.view().context, io.io, ciphertexts);
      const int noise = he.view().decryptor->invariant_noise_budget(
          ciphertexts.front());
      seal::Plaintext plaintext;
      std::vector<std::uint64_t> slots;
      he.view().decryptor->decrypt(ciphertexts.front(), plaintext);
      he.view().encoder->decode(plaintext, slots);
      bool dense_pass = true;
      std::cout << "SIGNED_DENSE_LINEAR_NOISE_BUDGET=" << noise << '\n';
      for (std::size_t slot = 0; slot < 8; ++slot)
        std::cout << "SIGNED_DENSE_LINEAR_SLOT index=" << slot
                  << " value=" << centered(slots[slot]) << '\n';
      for (std::size_t row = 0; row < 3; ++row) {
        __int128 accumulator = 0;
        for (std::size_t col = 0; col < clear.size(); ++col) {
          const std::int64_t value =
              clear[col] < (ring_modulus >> 1)
                  ? static_cast<std::int64_t>(clear[col])
                  : static_cast<std::int64_t>(clear[col] - ring_modulus);
          accumulator += static_cast<__int128>(value) *
                         dense_coeff[row * clear.size() + col];
        }
        std::int64_t residue = static_cast<std::int64_t>(
            accumulator % static_cast<__int128>(kPlainModulus));
        if (residue < 0) residue += kPlainModulus;
        const std::int64_t expected = centered(residue);
        const std::int64_t actual = centered(slots[row]);
        dense_pass = dense_pass && expected == actual;
        std::cout << "SIGNED_DENSE_LINEAR_CASE index=" << row
                  << " expected=" << expected << " actual=" << actual
                  << '\n';
      }
      std::cout << "SIGNED_DENSE_LINEAR_TEST="
                << (dense_pass ? "PASS" : "FAIL") << std::endl;
      dense_pass_all = dense_pass;
    }

    // Isolate the adapter's final LayerNorm strategy: unchanged MPC
    // normalization with a public identity affine, followed by the private
    // checkpoint affine in BFV.  Only the final zero-affine result is opened.
    const std::vector<std::uint64_t> ln_clear{
        4096, ring_modulus - 8192, 12288, ring_modulus - 16384,
        20480, ring_modulus - 24576, 28672, ring_modulus - 32768};
    FixArray ln_input = fp.fix->input(
        ALICE, static_cast<int>(ln_clear.size()),
        const_cast<std::uint64_t *>(ln_clear.data()), true, kEll, 12);
    std::vector<FixArray> ln_rows{ln_input};
    std::vector<std::uint64_t> zeros(ln_clear.size(), 0);
    std::vector<std::uint64_t> ones(ln_clear.size(), 1ULL << 12);
    FixArray ln_weight = fp.fix->input(
        PUBLIC, static_cast<int>(ones.size()), ones.data(), true, kEll, 12);
    FixArray ln_bias = fp.fix->input(
        PUBLIC, static_cast<int>(zeros.size()), zeros.data(), true, kEll, 12);
    FixArray ln_normalized = concat(
        fp.layer_norm_fix(ln_rows, ln_weight, ln_bias));
    // This is a dedicated known-input unit test, not the production graph:
    // compare the unchanged MPC LayerNorm output with BLB's bit-exact clear
    // implementation before exercising the private BFV affine adapter.
    FixArray ln_opened = fp.fix->output(PUBLIC, ln_normalized);
    const std::vector<std::int64_t> ln_expected{
        1222, -1223, 2852, -2853, 4482, -4483, 6112, -6113};
    bool ln_pass = true;
    for (std::size_t i = 0; i < ln_expected.size(); ++i) {
      const std::int64_t actual =
          ln_opened.data[i] < (ring_modulus >> 1)
              ? static_cast<std::int64_t>(ln_opened.data[i])
              : static_cast<std::int64_t>(ln_opened.data[i] - ring_modulus);
      ln_pass = ln_pass && actual == ln_expected[i];
      if (party == BOB)
        std::cout << "MPC_LAYERNORM_CASE index=" << i
                  << " expected=" << ln_expected[i]
                  << " actual=" << actual << '\n';
    }
    if (party == BOB)
      std::cout << "MPC_LAYERNORM_BIT_EXACT_TEST="
                << (ln_pass ? "PASS" : "FAIL") << std::endl;
    SecretSharedTensor ln_shares;
    ln_shares.metadata = metadata;
    ln_shares.local_share.assign(ln_normalized.data,
                                 ln_normalized.data + ln_normalized.size);
    BoltCipherTensor ln_cipher = conversion.ss_to_he_signed_ring(
        ln_shares, metadata, *fp.fix);
    if (party == ALICE) {
      BoltMatMulMetadata affine_metadata;
      affine_metadata.input_rows = 1;
      affine_metadata.input_cols = zeros.size();
      affine_metadata.output_cols = zeros.size();
      affine_metadata.packed_stride = 8;
      affine_metadata.baby_step = 4;
      BoltLinear linear(he.view());
      BoltCipherTensor affine = linear.cipher_plain_elementwise_affine(
          ln_cipher, zeros, zeros, affine_metadata);
      send_encrypted_vector(io.io, affine.ciphertexts);
    } else {
      std::vector<seal::Ciphertext> ciphertexts(1);
      recv_encrypted_vector(he.view().context, io.io, ciphertexts);
      seal::Plaintext plaintext;
      std::vector<std::uint64_t> slots;
      he.view().decryptor->decrypt(ciphertexts.front(), plaintext);
      he.view().encoder->decode(plaintext, slots);
      bool converted_zero = true;
      for (std::size_t i = 0; i < zeros.size(); ++i)
        converted_zero = converted_zero && slots[i] == 0;
      std::cout << "ZERO_AFFINE_LAYERNORM_ADAPTER_TEST="
                << (converted_zero ? "PASS" : "FAIL") << std::endl;
      ln_pass = ln_pass && converted_zero;
    }
    // Generic packed ct-ct MatMul is the QK/SV primitive used by the current
    // graph. Treat this small matrix product as the unit protocol's output.
    const std::vector<std::uint64_t> q_clear{1, 3, 2, 1, 2, 0, 1, 4};
    const std::vector<std::uint64_t> kt_clear{
        2, 1, 0, 1, 3, 2, 0, 2, 1, 4, 1, 3};
    BoltTensorMetadata q_metadata = metadata;
    q_metadata.rows = 2;
    q_metadata.cols = 4;
    q_metadata.packed_stride = 4;
    BoltTensorMetadata kt_metadata = q_metadata;
    kt_metadata.rows = 4;
    kt_metadata.cols = 3;
    SecretSharedTensor q_shares, kt_shares;
    q_shares.metadata = q_metadata;
    kt_shares.metadata = kt_metadata;
    q_shares.local_share.assign(q_clear.size(), 0);
    kt_shares.local_share.assign(kt_clear.size(), 0);
    if (party == BOB) {
      q_shares.local_share = q_clear;
      kt_shares.local_share = kt_clear;
    }
    BoltCipherTensor q_cipher = conversion.ss_to_he(q_shares, q_metadata);
    BoltCipherTensor kt_cipher = conversion.ss_to_he(kt_shares, kt_metadata);
    bool ctct_pass = true;
    if (party == ALICE) {
      BoltMatMulMetadata ctct_metadata;
      ctct_metadata.input_rows = 2;
      ctct_metadata.input_cols = 4;
      ctct_metadata.output_cols = 3;
      ctct_metadata.packed_stride = 4;
      ctct_metadata.baby_step = 2;
      BoltLinear linear(he.view());
      BoltCipherTensor product = linear.cipher_cipher_matmul_cross(
          q_cipher, kt_cipher, ctct_metadata);
      send_encrypted_vector(io.io, product.ciphertexts);
    } else {
      std::vector<seal::Ciphertext> products(2);
      recv_encrypted_vector(he.view().context, io.io, products);
      const std::vector<std::uint64_t> expected{9, 15, 11, 20, 8, 13};
      for (std::size_t row = 0; row < products.size(); ++row) {
        seal::Plaintext plaintext;
        std::vector<std::uint64_t> slots;
        he.view().decryptor->decrypt(products[row], plaintext);
        he.view().encoder->decode(plaintext, slots);
        for (std::size_t col = 0; col < 3; ++col) {
          const auto actual = slots[col] % kPlainModulus;
          ctct_pass = ctct_pass && actual == expected[row * 3 + col];
          std::cout << "PACKED_CT_CT_CASE row=" << row << " col=" << col
                    << " expected=" << expected[row * 3 + col]
                    << " actual=" << actual << '\n';
        }
      }
      std::cout << "PACKED_CT_CT_TEST=" << (ctct_pass ? "PASS" : "FAIL")
                << std::endl;
    }
    if (party == BOB &&
        !(rsqrt_pass && share_coverage_pass && ln_pass &&
          dense_pass_all && ctct_pass))
      return 5;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "SIGNED_CONVERSION_TEST error: " << error.what() << std::endl;
    return 2;
  }
}
