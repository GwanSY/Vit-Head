#include "BOLTTransformer/bolt_conversion.h"
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

// Legacy LinearHE utilities expose these process-wide configuration symbols.
// The shared backend itself uses BoltHEParameters; the definitions only keep
// the unchanged BOLT flooding helper link-compatible.
std::uint64_t prime_mod = 536903681ULL;
std::int32_t bitlength = 37;
std::int32_t num_threads = 1;

namespace {

int party = 0;
int port = 47200;
std::string address = "127.0.0.1";

BoltTensorMetadata tensor_metadata(std::size_t rows, std::size_t cols,
                                   std::size_t stride) {
  BoltTensorMetadata metadata;
  metadata.rows = rows;
  metadata.cols = cols;
  metadata.packed_stride = stride;
  metadata.share_bit_width = 20;
  metadata.scale = 0;
  metadata.packing = BoltPackingKind::kCompactRepeatedRows;
  return metadata;
}

SecretSharedTensor shared_input(const BoltTensorMetadata &metadata,
                                const std::vector<std::uint64_t> &clear) {
  SecretSharedTensor input;
  input.metadata = metadata;
  input.local_share.assign(clear.size(), 0);
  if (party == BOB) input.local_share = clear;
  return input;
}

BoltMatMulMetadata matmul_metadata(std::size_t rows, std::size_t inner,
                                   std::size_t cols, std::size_t stride,
                                   std::uint32_t flood_bits,
                                   std::uint32_t modulus_switches) {
  BoltMatMulMetadata metadata;
  metadata.input_rows = rows;
  metadata.input_cols = inner;
  metadata.output_cols = cols;
  metadata.packed_stride = stride;
  metadata.baby_step = 2;
  metadata.noise_flooding_bits = flood_bits;
  metadata.modulus_switches = modulus_switches;
  return metadata;
}

std::vector<std::uint64_t>
plain_matmul(const std::vector<std::uint64_t> &lhs, std::size_t rows,
             std::size_t inner, const std::vector<std::uint64_t> &rhs,
             std::size_t cols, std::uint64_t modulus) {
  std::vector<std::uint64_t> output(rows * cols, 0);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      unsigned __int128 sum = 0;
      for (std::size_t k = 0; k < inner; ++k) {
        sum += static_cast<unsigned __int128>(lhs[row * inner + k]) *
               rhs[k * cols + col];
      }
      output[row * cols + col] = static_cast<std::uint64_t>(sum % modulus);
    }
  }
  return output;
}

std::vector<std::uint64_t>
reconstruct_final(IOPack &io, const SecretSharedTensor &shares,
                  std::uint64_t modulus) {
  const std::uint64_t size = shares.local_share.size();
  if (party == ALICE) {
    io.io->send_data(&size, sizeof(size));
    io.io->send_data(shares.local_share.data(), size * sizeof(std::uint64_t));
    return {};
  }
  std::uint64_t peer_size = 0;
  io.io->recv_data(&peer_size, sizeof(peer_size));
  if (peer_size != size)
    throw std::runtime_error("final reconstruction size mismatch");
  std::vector<std::uint64_t> alice_share(size);
  io.io->recv_data(alice_share.data(), size * sizeof(std::uint64_t));
  std::vector<std::uint64_t> clear(size);
  for (std::size_t i = 0; i < size; ++i)
    clear[i] = (alice_share[i] + shares.local_share[i]) % modulus;
  return clear;
}

bool synchronize_status(IOPack &io, bool local_status) {
  std::uint8_t wire_status = local_status ? 1 : 0;
  if (party == BOB) {
    io.io->send_data(&wire_status, sizeof(wire_status));
    return local_status;
  }
  io.io->recv_data(&wire_status, sizeof(wire_status));
  return wire_status == 1;
}

void print_ct_pt(const BoltOperationCounts &counts, bool status) {
  std::cout << "BOLT_BACKEND_TEST op=ct_pt_matmul\n"
            << "batch_encoder=true\n"
            << "scalar_ciphertext=false\n"
            << "rotations=" << counts.rotations << "\n"
            << "multiply_plain=" << counts.multiply_plain << "\n"
            << "ciphertexts=" << counts.ciphertexts << "\n"
            << "noise_floodings=" << counts.noise_floodings << "\n"
            << "modulus_switches=" << counts.modulus_switches << "\n"
            << "status=" << (status ? "PASS" : "FAIL") << std::endl;
}

void print_ct_ct(const char *op, const BoltOperationCounts &counts,
                 bool status) {
  std::cout << "BOLT_BACKEND_TEST op=" << op << "\n"
            << "batch_encoder=true\n"
            << "scalar_ciphertext=false\n"
            << "rotations=" << counts.rotations << "\n"
            << "evaluator_multiply=" << counts.evaluator_multiply << "\n"
            << "relinearizations=" << counts.relinearizations << "\n"
            << "ciphertexts=" << counts.ciphertexts << "\n"
            << "noise_floodings=" << counts.noise_floodings << "\n"
            << "modulus_switches=" << counts.modulus_switches << "\n"
            << "generic_ot_matmul=false\n"
            << "status=" << (status ? "PASS" : "FAIL") << std::endl;
}

bool verify_fail_closed(BoltHE &he) {
  bool scalar = false, generic = false, batching = false, galois = false,
       relinearization = false;
  try { fail_if_scalar_ciphertext_adapter(true); }
  catch (const std::logic_error &) { scalar = true; }
  try { fail_if_generic_ot_matmul(true); }
  catch (const std::logic_error &) { generic = true; }
  try {
    BoltHEView invalid = he.view();
    invalid.encoder = nullptr;
    invalid.require_batching();
  } catch (const std::logic_error &) { batching = true; }
  try {
    BoltHEView invalid = he.view();
    invalid.galois_keys = nullptr;
    invalid.require_public_evaluation(false);
  } catch (const std::logic_error &) { galois = true; }
  try {
    BoltHEView invalid = he.view();
    invalid.relinearization_keys = nullptr;
    invalid.require_public_evaluation(true);
  } catch (const std::logic_error &) { relinearization = true; }
  const bool pass = scalar && generic && batching && galois && relinearization;
  std::cout << "BOLT_FAIL_CLOSED scalar_adapter=" << (scalar ? "true" : "false")
            << " generic_ot_matmul=" << (generic ? "true" : "false")
            << " missing_batch_encoder=" << (batching ? "true" : "false")
            << " missing_galois_key=" << (galois ? "true" : "false")
            << " missing_relin_key=" << (relinearization ? "true" : "false")
            << " status=" << (pass ? "PASS" : "FAIL") << std::endl;
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
    BoltHE he(party, io.io, BoltHEParameters::bert_main());
    BoltConversion conversion(he.view());
    BoltLinear linear(he.view());
    const std::uint64_t modulus = he.parameters().plain_modulus;
    bool all_passed = true;
    if (party == ALICE) all_passed = verify_fail_closed(he);

    const std::vector<std::uint64_t> x{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<std::uint64_t> w{
        1, 0, 2, 0, 1, 1, 2, 1, 0, 1, 2, 1};
    PlainWeightTensor weight;
    weight.output_cols = 3;
    weight.input_cols = 4;
    weight.values.resize(12);
    for (std::size_t out = 0; out < 3; ++out)
      for (std::size_t in = 0; in < 4; ++in)
        weight.values[out * 4 + in] = w[in * 3 + out];
    weight.bias = {1, 2, 3};
    const auto x_metadata = tensor_metadata(2, 4, 4);
    BoltCipherTensor encrypted_x =
        conversion.ss_to_he(shared_input(x_metadata, x), x_metadata);
    BoltOperationCounts ct_pt_counts;
    BoltCipherTensor encrypted_ct_pt;
    if (party == ALICE) {
      encrypted_ct_pt = linear.cipher_plain_matmul_bsgs(
          encrypted_x, weight,
          matmul_metadata(2, 4, 3, 4, SMUDGING_BITLEN_bert2, 1),
          &ct_pt_counts);
    } else {
      encrypted_ct_pt.metadata = tensor_metadata(2, 3, 4);
    }
    const auto ct_pt_metadata = tensor_metadata(2, 3, 4);
    const SecretSharedTensor ct_pt_shares =
        conversion.he_to_ss(encrypted_ct_pt, ct_pt_metadata);
    std::vector<std::uint64_t> ct_pt_reference =
        plain_matmul(x, 2, 4, w, 3, modulus);
    for (std::size_t row = 0; row < 2; ++row)
      for (std::size_t col = 0; col < 3; ++col)
        ct_pt_reference[row * 3 + col] =
            (ct_pt_reference[row * 3 + col] + weight.bias[col]) % modulus;
    const auto ct_pt_clear = reconstruct_final(io, ct_pt_shares, modulus);
    bool ct_pt_pass = party == BOB ? ct_pt_clear == ct_pt_reference : true;
    ct_pt_pass = synchronize_status(io, ct_pt_pass);
    if (party == ALICE) print_ct_pt(ct_pt_counts, ct_pt_pass);
    all_passed = all_passed && ct_pt_pass;

    const std::vector<std::uint64_t> q{1, 3, 2, 1, 2, 0, 1, 4};
    const std::vector<std::uint64_t> kt{2, 1, 0, 1, 3, 2,
                                        0, 2, 1, 4, 1, 3};
    const auto q_metadata = tensor_metadata(2, 4, 4);
    const auto kt_metadata = tensor_metadata(4, 3, 4);
    BoltCipherTensor encrypted_q =
        conversion.ss_to_he(shared_input(q_metadata, q), q_metadata);
    BoltCipherTensor encrypted_kt =
        conversion.ss_to_he(shared_input(kt_metadata, kt), kt_metadata);
    BoltOperationCounts qk_counts;
    BoltCipherTensor encrypted_qk;
    if (party == ALICE) {
      encrypted_qk = linear.cipher_cipher_matmul_cross(
          encrypted_q, encrypted_kt,
          matmul_metadata(2, 4, 3, 4, SMUDGING_BITLEN_bert1, 2),
          &qk_counts);
    } else {
      encrypted_qk.metadata = tensor_metadata(2, 3, 4);
    }
    const auto qk_metadata = tensor_metadata(2, 3, 4);
    const SecretSharedTensor qk_shares =
        conversion.he_to_ss(encrypted_qk, qk_metadata);
    const auto qk_reference = plain_matmul(q, 2, 4, kt, 3, modulus);
    const auto qk_clear = reconstruct_final(io, qk_shares, modulus);
    bool qk_pass = party == BOB ? qk_clear == qk_reference : true;
    qk_pass = synchronize_status(io, qk_pass);
    if (party == ALICE) print_ct_ct("qk_ct_ct", qk_counts, qk_pass);
    all_passed = all_passed && qk_pass;

    const std::vector<std::uint64_t> softmax{1, 2, 1, 0, 0, 1, 2, 1};
    const std::vector<std::uint64_t> value{1, 2, 0, 0, 1, 3,
                                           2, 0, 1, 1, 2, 2};
    const auto softmax_metadata = tensor_metadata(2, 4, 4);
    const auto value_metadata = tensor_metadata(4, 3, 4);
    BoltCipherTensor encrypted_softmax = conversion.ss_to_he(
        shared_input(softmax_metadata, softmax), softmax_metadata);
    BoltCipherTensor encrypted_value = conversion.ss_to_he(
        shared_input(value_metadata, value), value_metadata);
    BoltOperationCounts softmax_v_counts;
    BoltCipherTensor encrypted_softmax_v;
    if (party == ALICE) {
      encrypted_softmax_v = linear.softmax_v_matmul(
          encrypted_softmax, encrypted_value,
          matmul_metadata(2, 4, 3, 4, 0, 2), &softmax_v_counts);
    } else {
      encrypted_softmax_v.metadata = tensor_metadata(2, 3, 4);
    }
    const auto softmax_v_metadata = tensor_metadata(2, 3, 4);
    const SecretSharedTensor softmax_v_shares =
        conversion.he_to_ss(encrypted_softmax_v, softmax_v_metadata);
    const auto softmax_v_reference =
        plain_matmul(softmax, 2, 4, value, 3, modulus);
    const auto softmax_v_clear =
        reconstruct_final(io, softmax_v_shares, modulus);
    bool softmax_v_pass =
        party == BOB ? softmax_v_clear == softmax_v_reference : true;
    softmax_v_pass = synchronize_status(io, softmax_v_pass);
    if (party == ALICE)
      print_ct_ct("softmax_v_ct_ct", softmax_v_counts, softmax_v_pass);
    all_passed = all_passed && softmax_v_pass;

    if (party == ALICE)
      std::cout << "VIT_BOLT_KERNELS=" << (all_passed ? "PASS" : "FAIL")
                << std::endl;
    return all_passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "BOLT_VIT_KERNEL_TEST error: " << error.what() << std::endl;
    return 1;
  }
}
