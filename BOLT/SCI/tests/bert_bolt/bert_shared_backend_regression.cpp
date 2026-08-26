#include "BOLTTransformer/bolt_he.h"
#include "BOLTTransformer/bolt_linear.h"
#include "linear.h"
#include "utils/ArgMapping/ArgMapping.h"
#include "utils/io_pack.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sci;
using namespace sci::bolt;
using namespace seal;
using namespace std;

uint64_t prime_mod = 65537ULL;
int32_t bitlength = 37;
int32_t num_threads = 1;

namespace {
int party = 0;
int port = 47300;
string address = "127.0.0.1";
string output_path = "outputs/regression/bert_shared_backend.csv";

HE legacy_view(BoltHE &backend) {
  BoltHEView source = backend.view();
  HE view;
  view.party = source.party;
  view.io = source.io;
  view.poly_modulus_degree = source.poly_modulus_degree;
  view.plain_mod = source.plain_modulus;
  view.plain_mod_2 = (source.plain_modulus - 1) / 2;
  view.context = source.context;
  view.encryptor = source.encryptor;
  view.decryptor = source.decryptor;
  view.evaluator = source.evaluator;
  view.encoder = source.encoder;
  view.gal_keys = source.galois_keys;
  view.relin_keys = source.relinearization_keys;
  view.zero = nullptr;
  return view;
}

BoltBertPackingMetadata regression_metadata() {
  BoltBertPackingMetadata metadata;
  metadata.slot_count = 8192;
  metadata.image_size = 1024;
  metadata.filter_height = 8;
  metadata.filter_width = 8;
  metadata.baby_step = 2;
  metadata.giant_steps = 2;
  metadata.attention_heads = 12;
  metadata.paired_head_groups = 6;
  metadata.noise_flooding_bits = SMUDGING_BITLEN_bert2;
  metadata.modulus_switches = 1;
  return metadata;
}

uint64_t hash_slots(const vector<uint64_t> &values) {
  uint64_t hash = 1469598103934665603ULL;
  for (uint64_t value : values) {
    for (int byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (8 * byte)) & 0xff;
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

string hex_hash(uint64_t hash) {
  ostringstream stream;
  stream << hex << setw(16) << setfill('0') << hash;
  return stream.str();
}

vector<uint64_t> receive_and_decode(BoltHE &backend, size_t count) {
  vector<Ciphertext> ciphertexts(count);
  BoltHEView view = backend.view();
  recv_encrypted_vector(view.context, view.io, ciphertexts);
  vector<uint64_t> all_slots;
  all_slots.reserve(count * view.encoder->slot_count());
  for (const Ciphertext &ciphertext : ciphertexts) {
    Plaintext plaintext;
    vector<uint64_t> slots;
    view.decryptor->decrypt(ciphertext, plaintext);
    view.encoder->decode(plaintext, slots);
    all_slots.insert(all_slots.end(), slots.begin(), slots.end());
  }
  return all_slots;
}
} // namespace

int main(int argc, char **argv) {
  ArgMapping args;
  args.arg("r", party, "Role: ALICE=1, BOB=2");
  args.arg("p", port, "TCP port");
  args.arg("ip", address, "ALICE address (BOB only)");
  args.arg("output", output_path, "Regression CSV path");
  args.parse(argc, argv);

  try {
    IOPack io(party, port, address);
    BoltHEParameters parameters;
    parameters.poly_modulus_degree = 8192;
    parameters.coeff_modulus_bits = {54, 54, 55, 55};
    parameters.plain_modulus = prime_mod;
    BoltHE backend(party, io.io, parameters);
    HE he = legacy_view(backend);

    BoltOperationCounts shared_counts[3];
    if (party == ALICE) {
      Linear linear;
      FCMetadata data{};
      data.slot_count = 8192;
      data.image_size = 1024;
      data.filter_h = 8;
      data.filter_w = 8;
      data.filter_size = 64;
      data.inp_ct = 1;
      data.pack_num = 1;

      vector<vector<uint64_t>> weights(8, vector<uint64_t>(8));
      vector<uint64_t> bias(8);
      for (size_t row = 0; row < 8; ++row) {
        bias[row] = row + 1;
        for (size_t col = 0; col < 8; ++col)
          weights[row][col] = (row * 3 + col * 5 + 1) % 17;
      }
      PreprocessParams_2 packed = linear.params_preprocessing_ct_pt(
          &he, 1024, 8, 8, weights, bias, data);
      vector<uint64_t> packed_input(8192);
      for (size_t i = 0; i < packed_input.size(); ++i)
        packed_input[i] = (i * 7 + 3) % 31;
      vector<Ciphertext> encrypted_input =
          linear.bert_efficient_preprocess_vec(&he, packed_input, data);

      vector<Ciphertext> original(1);
      set_bolt_shared_backend_for_regression(false);
      linear.bert_cipher_plain_bsgs_2(
          &he, encrypted_input, packed.cross_mat_single.first,
          packed.cross_mat_single.second, packed.cross_bias_single, data,
          original);
      set_bolt_shared_backend_for_regression(true);

      vector<Ciphertext> shared(1);
      BoltLinear shared_backend(backend.view());
      shared_backend.bert_cipher_plain_matmul_bsgs_compat(
          encrypted_input, packed.cross_mat_single.first,
          packed.cross_mat_single.second, packed.cross_bias_single,
          regression_metadata(), shared, &shared_counts[0]);

      FCMetadata attention_data{};
      attention_data.slot_count = 8192;
      attention_data.image_size = 64;
      attention_data.filter_h = 768;
      attention_data.filter_w = 64;
      attention_data.filter_size = 768 * 64;
      vector<uint64_t> attention_slots(8192);
      for (size_t i = 0; i < attention_slots.size(); ++i)
        attention_slots[i] = (i * 11 + 5) % 37;
      Plaintext attention_plaintext;
      backend.view().encoder->encode(attention_slots, attention_plaintext);
      Ciphertext attention_ciphertext;
      backend.view().encryptor->encrypt(attention_plaintext,
                                        attention_ciphertext);
      vector<Ciphertext> qk_inputs(12, attention_ciphertext);
      vector<Plaintext> cross_masks =
          linear.generate_cross_packing_masks(&he, attention_data);
      vector<Ciphertext> original_qk(1), shared_qk(2);
      set_bolt_shared_backend_for_regression(false);
      linear.bert_cipher_cipher_cross_packing(
          &he, attention_data, qk_inputs, cross_masks, original_qk);
      set_bolt_shared_backend_for_regression(true);
      BoltBertPackingMetadata attention_metadata;
      attention_metadata.slot_count = 8192;
      attention_metadata.image_size = 64;
      attention_metadata.filter_height = 768;
      attention_metadata.filter_width = 64;
      attention_metadata.baby_step = 1;
      attention_metadata.giant_steps = 1;
      attention_metadata.attention_heads = 12;
      attention_metadata.paired_head_groups = 1;
      shared_backend.bert_qk_matmul_cross_compat(
          qk_inputs, cross_masks, attention_metadata, shared_qk,
          &shared_counts[1]);
      // The production fused QKV path supplies a larger combined QK+V output
      // buffer.  The QK compatibility kernel must fill its prefix and accept
      // the reserved V tail.
      shared_qk.resize(1);

      vector<vector<vector<Ciphertext>>> softmax(
          1, vector<vector<Ciphertext>>(
                 2, vector<Ciphertext>(64, attention_ciphertext)));
      vector<Ciphertext> value(1, attention_ciphertext);
      vector<Ciphertext> original_softmax_v(1), shared_softmax_v(1);
      set_bolt_shared_backend_for_regression(false);
      linear.softmax_v(&he, softmax, value, attention_data,
                       original_softmax_v);
      set_bolt_shared_backend_for_regression(true);
      attention_metadata.modulus_switches = 2;
      shared_backend.bert_softmax_v_matmul_compat(
          softmax, value, attention_metadata, shared_softmax_v,
          &shared_counts[2]);

      send_encrypted_vector(io.io, original);
      send_encrypted_vector(io.io, shared);
      send_encrypted_vector(io.io, original_qk);
      send_encrypted_vector(io.io, shared_qk);
      send_encrypted_vector(io.io, original_softmax_v);
      send_encrypted_vector(io.io, shared_softmax_v);
      io.io->send_data(shared_counts, sizeof(shared_counts));
      uint8_t status = 0;
      io.io->recv_data(&status, sizeof(status));
      cout << "BERT_SHARED_BACKEND=" << (status ? "PASS" : "FAIL")
           << endl;
      return status ? 0 : 1;
    }

    const vector<uint64_t> original_slots = receive_and_decode(backend, 1);
    const vector<uint64_t> shared_slots = receive_and_decode(backend, 1);
    const vector<uint64_t> original_qk_slots = receive_and_decode(backend, 1);
    const vector<uint64_t> shared_qk_slots = receive_and_decode(backend, 1);
    const vector<uint64_t> original_softmax_v_slots =
        receive_and_decode(backend, 1);
    const vector<uint64_t> shared_softmax_v_slots =
        receive_and_decode(backend, 1);
    io.io->recv_data(shared_counts, sizeof(shared_counts));
    const uint64_t original_hash = hash_slots(original_slots);
    const uint64_t shared_hash = hash_slots(shared_slots);
    const uint64_t original_qk_hash = hash_slots(original_qk_slots);
    const uint64_t shared_qk_hash = hash_slots(shared_qk_slots);
    const uint64_t original_softmax_v_hash =
        hash_slots(original_softmax_v_slots);
    const uint64_t shared_softmax_v_hash = hash_slots(shared_softmax_v_slots);
    const bool ct_pt_pass = original_slots == shared_slots &&
                            shared_counts[0].ciphertexts == 1 &&
                            shared_counts[0].rotations == 4;
    const bool qk_pass = original_qk_slots == shared_qk_slots &&
                         shared_counts[1].ciphertexts == 1 &&
                         shared_counts[1].evaluator_multiply > 0 &&
                         shared_counts[1].relinearizations > 0;
    const bool softmax_v_pass =
        original_softmax_v_slots == shared_softmax_v_slots &&
        shared_counts[2].ciphertexts == 1 &&
        shared_counts[2].evaluator_multiply > 0 &&
        shared_counts[2].relinearizations > 0;
    const bool pass = ct_pt_pass && qk_pass && softmax_v_pass;

    ofstream csv(output_path);
    if (!csv) throw runtime_error("cannot create BERT regression CSV");
    csv << "op,original_output_hash,shared_output_hash,original_ciphertexts,"
           "shared_ciphertexts,original_rotations,shared_rotations,status\n";
    csv << "bert_ct_pt_bsgs," << hex_hash(original_hash) << ','
        << hex_hash(shared_hash) << ",1," << shared_counts[0].ciphertexts << ",4,"
        << shared_counts[0].rotations << ','
        << (ct_pt_pass ? "PASS" : "FAIL") << '\n';
    csv << "bert_qk_cross," << hex_hash(original_qk_hash) << ','
        << hex_hash(shared_qk_hash) << ",1," << shared_counts[1].ciphertexts
        << ',' << shared_counts[1].rotations << ','
        << shared_counts[1].rotations << ','
        << (qk_pass ? "PASS" : "FAIL") << '\n';
    csv << "bert_softmax_v," << hex_hash(original_softmax_v_hash) << ','
        << hex_hash(shared_softmax_v_hash) << ",1,"
        << shared_counts[2].ciphertexts << ',' << shared_counts[2].rotations
        << ',' << shared_counts[2].rotations << ','
        << (softmax_v_pass ? "PASS" : "FAIL")
        << '\n';
    csv.close();
    const uint8_t status = pass ? 1 : 0;
    io.io->send_data(&status, sizeof(status));
    return pass ? 0 : 1;
  } catch (const exception &error) {
    cerr << "BERT_SHARED_BACKEND_REGRESSION error: " << error.what() << endl;
    return 1;
  }
}
