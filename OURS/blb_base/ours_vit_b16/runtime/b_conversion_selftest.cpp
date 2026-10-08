#include "ckks_bert.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sci;
using namespace std;

int bitlength = 37;

namespace {

constexpr int kDegree = 32768;
constexpr int kBits = 37;
constexpr int kScale = 12;
constexpr int kCkksScale = 43;
constexpr int kLayerNormScale = 38;
constexpr int kConversionScale = 49;
const int kCiphertexts = [] {
  const char *value = getenv("OURS_VIT_B_SELFTEST_CIPHERTEXTS");
  return value == nullptr ? 12 : atoi(value);
}();
constexpr int kQkvInputScale = 93;

uint64_t ring_mask() { return (uint64_t{1} << kBits) - 1; }

int64_t signed_ring(uint64_t value) {
  value &= ring_mask();
  if ((value & (uint64_t{1} << (kBits - 1))) == 0)
    return static_cast<int64_t>(value);
  return static_cast<int64_t>(value) -
         static_cast<int64_t>(uint64_t{1} << kBits);
}

shared_ptr<const SEALContext::ContextData> context_at_chain(
    HE_CKKS &he, size_t chain_index) {
  for (auto context = he.context->first_context_data(); context;
       context = context->next_context_data()) {
    if (context->chain_index() == chain_index)
      return context;
  }
  throw runtime_error("B public QKV fixture chain is absent");
}

vector<double> unpack_qkv_tensor(const vector<vector<complex<double>>> &packed,
                                 const FCMetadata &data) {
  const int ciphertexts =
      PACKING_NUM * data.image_size * data.filter_w / data.slot_count;
  if (static_cast<int>(packed.size()) != ciphertexts)
    throw runtime_error("B public QKV fixture ciphertext count mismatch");
  vector<double> output(static_cast<size_t>(PACKING_NUM) * data.image_size *
                        data.filter_w);
  const int tensors_per_ciphertext =
      data.slot_count / (data.image_size * data.filter_w);
  if (tensors_per_ciphertext != 1)
    throw runtime_error("B public QKV fixture expects ViT full-slot packing");
  for (int ciphertext = 0; ciphertext < ciphertexts; ++ciphertext) {
    if (static_cast<int>(packed[ciphertext].size()) != data.slot_count)
      throw runtime_error("B public QKV fixture slot count mismatch");
    for (int slot = 0; slot < data.slot_count; ++slot) {
      int feature = slot / data.image_size;
      const int row = slot % data.image_size;
      int paired_head_offset = 0;
      if (slot >= data.slot_count / 2) {
        feature -= data.slot_count / data.image_size / 2;
        paired_head_offset = data.filter_w * data.image_size;
      }
      const size_t destination =
          static_cast<size_t>(row) * data.filter_w + feature +
          paired_head_offset +
          static_cast<size_t>(ciphertext / 2) * data.image_size *
              data.filter_w * 2 +
          static_cast<size_t>(ciphertext % 2) * data.filter_w / 2;
      if (destination >= output.size())
        throw runtime_error("B public QKV fixture unpack overflow");
      output[destination] = packed[ciphertext][slot].real();
    }
  }
  return output;
}

vector<double> unpack_qk_tensor(const vector<vector<complex<double>>> &packed,
                                const FCMetadata &data) {
  const int ciphertexts =
      PACKING_NUM * data.image_size * data.image_size / data.slot_count;
  if (static_cast<int>(packed.size()) != ciphertexts)
    throw runtime_error("B public QK fixture ciphertext count mismatch");
  vector<double> output(static_cast<size_t>(PACKING_NUM) * data.image_size *
                        data.image_size);
  const int ciphertexts_per_pair =
      data.image_size * data.image_size * 2 / data.slot_count;
  const int columns_per_half =
      data.slot_count / 2 / data.image_size;
  for (int ciphertext = 0; ciphertext < ciphertexts; ++ciphertext) {
    const int pair = ciphertext / ciphertexts_per_pair;
    const int column_block = ciphertext % ciphertexts_per_pair;
    for (int slot = 0; slot < data.slot_count; ++slot) {
      int diagonal = slot / data.image_size +
                     column_block * columns_per_half;
      const int row = slot % data.image_size;
      const bool upper_half = slot >= data.slot_count / 2;
      if (upper_half) diagonal -= columns_per_half;
      const int head = pair * 2 + (upper_half ? 1 : 0);
      const int column = (row + diagonal) % data.image_size;
      const size_t destination =
          (static_cast<size_t>(head) * data.image_size + row) *
              data.image_size +
          column;
      output[destination] = packed[ciphertext][slot].real();
    }
  }
  return output;
}

vector<double> unpack_bsgs_output_tensor(
    const vector<vector<complex<double>>> &packed,
    const FCMetadata &data) {
  const int ciphertexts =
      data.image_size * data.filter_w / data.slot_count;
  if (static_cast<int>(packed.size()) != ciphertexts)
    throw runtime_error("B public WO fixture ciphertext count mismatch");
  const int features_per_half =
      data.slot_count / (2 * data.image_size);
  vector<double> output(
      static_cast<size_t>(data.image_size) * data.filter_w);
  for (int ciphertext = 0; ciphertext < ciphertexts; ++ciphertext) {
    for (int slot = 0; slot < data.slot_count; ++slot) {
      const int token = slot % data.image_size;
      const bool upper_half = slot >= data.slot_count / 2;
      const int local_feature =
          (slot % (data.slot_count / 2)) / data.image_size;
      const int feature = ciphertext * features_per_half + local_feature +
                          (upper_half ? data.filter_w / 2 : 0);
      output[static_cast<size_t>(token) * data.filter_w + feature] =
          packed[ciphertext][slot].real();
    }
  }
  return output;
}

}  // namespace

int main(int argc, char **argv) {
  int party = 0;
  int port = 0;
  int device_backend = 0;
  int linear_scale_cycle = 0;
  string address = "127.0.0.1";
  ArgMapping args;
  args.arg("r", party, "party");
  args.arg("p", port, "port");
  args.arg("ip", address, "address");
  args.arg("d", device_backend, "use Phantom device transfer");
  args.arg("m", linear_scale_cycle,
           "0=conversion, 1=ct-pt scale cycle, 2=public imported LayerNorm "
           "affine, 3=public native-CKKS LayerNorm affine, "
           "4=public packed QKV/QK, 5=public packed Softmax*V, "
           "6=public GELU range regression, 7=public centered wide conversion, "
           "8=public compact masked Softmax, 9=public real-pair conversion, "
           "10=public full-range integer-lift import");
  args.parse(argc, argv);
  if ((party != sci::ALICE && party != sci::BOB) || port <= 0 ||
      (device_backend != 0 && device_backend != 1))
    throw invalid_argument(
        "usage: r=1|2 p=PORT [ip=ADDRESS] [d=0|1] [m=0|1]");
  if (linear_scale_cycle < 0 || linear_scale_cycle > 10)
    throw invalid_argument("m must be in [0, 10]");

  sci::NetIO io(party == sci::ALICE ? nullptr : address.c_str(), port);
  const vector<int> coefficient_bits =
      linear_scale_cycle == 8
          ? vector<int>{60, 40, 40, 40, 40, 40, 40, 40, 60}
          : linear_scale_cycle == 4
          ? vector<int>{60, 35, 60, 60, 60, 60, 60, 60}
          : (linear_scale_cycle == 6
                 ? vector<int>{60, 32, 60, 60, 60, 60, 60, 60}
                 : vector<int>{60, 60, 60, 60, 60, 60, 60});
  const int he_scale = linear_scale_cycle == 8 ? 40 : linear_scale_cycle == 4
                           ? 44
                           : (linear_scale_cycle == 5 ? 43 : kCkksScale);
  HE_CKKS he(party, &io, kDegree,
             coefficient_bits, 40, he_scale,
             device_backend ? LOCATION::DEVICE : LOCATION::HOST);
  if (linear_scale_cycle == 10) {
    Bert_ckks adapter(party, &io, address, port + 1);
    const int count = 3;
    vector<vector<uint64_t>> local(count, vector<uint64_t>(he.num_slots));
    vector<vector<int64_t>> expected(count, vector<int64_t>(he.num_slots));
    const array<int64_t, 9> endpoints = {
        0, 1, -1, (int64_t{1} << 36) - 1, -(int64_t{1} << 36),
        int64_t{1} << 30, -(int64_t{1} << 30), 4096, -4096};
    for (int ct = 0; ct < count; ++ct) {
      for (size_t i = 0; i < he.num_slots; ++i) {
        const uint64_t index = ct * he.num_slots + i;
        expected[ct][i] = i % 2 == 0 ? endpoints[(i / 2) % endpoints.size()]
            : signed_ring((index + 7) * 0xbf58476d1ce4e5b9ULL);
        const uint64_t share = (index + 1) * 0x9e3779b97f4a7c15ULL & ring_mask();
        local[ct][i] = party == sci::ALICE ? share
            : (uint64_t(expected[ct][i]) - share) & ring_mask();
      }
    }
    const auto before = adapter.current_comm();
    auto plains = adapter.mpc_to_ckks(local, he.num_slots,
        he.get_conversion_prime(), kBits, kScale, kCkksScale, &he, false);
    const auto conversion_bytes = adapter.current_comm() - before;
    vector<Ciphertext> cts(count);
    double max_error_raw = 0;
    int failed = 0;
    if (party == sci::BOB) {
      he.encrypt_send(plains, cts);
      recv_encrypted_vector(he.context, &io, cts);
      for (int ct = 0; ct < count; ++ct) {
        Plaintext plain;
        vector<complex<double>> decoded;
        he.decryptor->decrypt(cts[ct], plain);
        he.encoder->decode(plain, decoded);
        for (size_t i = 0; i < he.num_slots; ++i) {
          const double error = abs(ldexp(decoded[i].real(), kScale) - expected[ct][i]);
          if (!isfinite(error) || error > 2) ++failed;
          max_error_raw = max(max_error_raw, error);
        }
      }
      io.send_data(&failed, sizeof(failed));
      io.send_data(&max_error_raw, sizeof(max_error_raw));
    } else {
      recv_encrypted_vector(he.context, &io, cts);
      for (int ct = 0; ct < count; ++ct)
        he.evaluator->add_plain_inplace(cts[ct], plains[ct]);
      send_encrypted_vector(&io, cts);
      io.recv_data(&failed, sizeof(failed));
      io.recv_data(&max_error_raw, sizeof(max_error_raw));
    }
    cout << "B_PUBLIC_INTEGER_LIFT status=" << (failed ? "FAIL" : "PASS")
         << " max_abs_error_raw=" << max_error_raw
         << " tested=" << count * he.num_slots
         << " full_signed_input_bits=" << kBits
         << " conversion_send_bytes=" << conversion_bytes << endl;
    return failed ? 1 : 0;
  }
  if (linear_scale_cycle == 9) {
    Bert_ckks adapter(party, &io, address, port + 1);
    const auto id = he.context->last_context_data()->parms_id();
    vector<Ciphertext> ciphertexts(3);
    vector<Plaintext> shares;
    for (int ct = 0; ct < 3; ++ct) {
      vector<complex<double>> values(he.num_slots);
      for (size_t i = 0; i < values.size(); ++i)
        values[i] = {double(int(i % 257) - 128) / 4.0 + 3 * ct,
                     double(int(i % 13) - 6)};
      if (party == sci::BOB) {
        Plaintext plain;
        he.encoder->encode(values, id, ldexp(1.0, 49), plain);
        he.encryptor->encrypt(plain, ciphertexts[ct]);
      }
    }
    if (party == sci::BOB) {
      send_encrypted_vector(&io, ciphertexts);
      ciphertexts.resize(2);
      recv_encrypted_vector(he.context, &io, ciphertexts);
      shares.resize(ciphertexts.size());
      for (size_t i = 0; i < ciphertexts.size(); ++i)
        he.decryptor->decrypt(ciphertexts[i], shares[i]);
    } else {
      recv_encrypted_vector(he.context, &io, ciphertexts);
      adapter.pack_real_ckks_pairs(ciphertexts, &he);
      adapter.mask_ckks_to_mpc_inplace(ciphertexts, shares, &he);
      send_encrypted_vector(&io, ciphertexts);
    }
    const auto result = adapter.ckks_pairs_to_mpc(shares, 3, kBits, kScale, &he);
    FixArray local(party, 3 * he.num_slots, true, kBits, kScale);
    for (int ct = 0; ct < 3; ++ct)
      copy(result[ct].begin(), result[ct].end(), local.data + ct * he.num_slots);
    auto opened = adapter.nonlinear.fpmath[0]->fix->output(sci::PUBLIC, local);
    int64_t max_error = 0;
    for (int ct = 0; ct < 3; ++ct)
      for (size_t i = 0; i < he.num_slots; ++i) {
        const int64_t expected = (int64_t(i % 257) - 128) * 1024 + 3 * ct * 4096;
        max_error = max(max_error, int64_t(llabs(signed_ring(opened.data[ct * he.num_slots + i]) - expected)));
      }
    cout << "B_PUBLIC_REAL_PAIRS status=" << (max_error <= 4 ? "PASS" : "FAIL")
         << " max_abs_error_raw=" << max_error
         << " source_cts=3 packed_cts=2 nonzero_imaginary_input=1" << endl;
    return max_error <= 4 ? 0 : 1;
  }
  if (linear_scale_cycle == 8) {
    Bert_ckks adapter(party, &io, address, port + 1);
    adapter.linear.he_softmax = &he;
    adapter.pre_norm_attention_output_only = true;
    adapter.linear.data_lin1_0.image_size = 256;
    adapter.public_attention_mask.assign(256, -10000);
    fill_n(adapter.public_attention_mask.begin(), 197, 0);
    adapter.public_attention_mask_ready = true;
    FixArray input(party, 256 * 256, true, kBits, kScale);
    for (int row = 0; row < 256; ++row) {
      for (int key = 0; key < 256; ++key) {
        const int64_t value = key >= 197 ? -10000LL * 4096
            : (row % 2 == 0 ? 0 : -int64_t(key % 17) * 4096);
        const int index = row * 256 + key;
        const uint64_t share = uint64_t(index + 1) * 0x9e3779b97f4a7c15ULL & ring_mask();
        input.data[index] = party == sci::ALICE ? share
            : (uint64_t(value) - share) & ring_mask();
      }
    }
    const auto rows = adapter.softmax_fix_he(party, input, adapter.nonlinear.fpmath, 1);
    FixArray flattened(party, input.size, true, kBits, kScale);
    for (int row = 0; row < 256; ++row)
      copy_n(rows[row].data, 256, flattened.data + row * 256);
    auto opened = adapter.nonlinear.fpmath[0]->fix->output(sci::PUBLIC, flattened);
    int errors = 0;
    double max_error = 0.0;
    for (int row = 0; row < 256; ++row) {
      double denom = 0;
      for (int key = 0; key < 197; ++key)
        denom += row % 2 == 0 ? 1.0 : pow(1.0 - double(key % 17) / 128.0, 128);
      for (int key = 0; key < 256; ++key) {
        const double value = ldexp(double(signed_ring(opened.data[row * 256 + key])), -kScale);
        const double expected = (key >= 197 || row >= 197) ? 0.0 :
            (row % 2 == 0 ? 1.0 : pow(1.0 - double(key % 17) / 128.0, 128)) / denom;
        max_error = max(max_error, abs(value - expected));
        if (((key >= 197 || row >= 197) && value != 0.0) || abs(value - expected) > 0.002) ++errors;
      }
    }
    cout << "B_PUBLIC_COMPACT_SOFTMAX status=" << (errors ? "FAIL" : "PASS")
         << " errors=" << errors << " max_abs_error=" << max_error
         << " queries=197 keys=197 slot_tail_padding=1" << endl;
    adapter.linear.he_softmax = nullptr;
    return errors ? 1 : 0;
  }
  if (linear_scale_cycle == 7) {
    Bert_ckks adapter(party, &io, address, port + 1);
    const uint64_t q = he.context->last_context_data()->parms().coeff_modulus()[0].value();
    const uint64_t center = (q - 1) / 2;
    const int count = 131079;
    vector<uint64_t> input(count), low(count), high(count), peer_low(count), peer_high(count);
    vector<__uint128_t> expected(count);
    const uint64_t cases[] = {0, 1, 2, center - 1, center, center + 1, q - 2, q - 1};
    for (int i = 0; i < count; ++i) {
      const uint64_t a = i % 3 == 0 ? cases[(i / 3) % 8]
          : uint64_t(__uint128_t(0x9e3779b97f4a7c15ULL) * (i + 1) % q);
      const uint64_t b = i % 3 == 0 ? cases[(i / 24) % 8]
          : uint64_t(__uint128_t(0xd1b54a32d192ed03ULL) * (i + 1) % q);
      input[i] = party == sci::ALICE ? a : b;
      const uint64_t residue = (a + b) % q;
      expected[i] = residue > center ? __uint128_t(residue) - q : residue;
    }
    bool passed = true;
    for (int width : {100, 120}) {
      const auto start = chrono::steady_clock::now();
      const uint64_t comm_before = adapter.current_comm();
      adapter.nonlinear.centered_field_to_ring_wide(
          adapter.nonlinear.active_workers(), input.data(), q, low.data(),
          high.data(), count, width);
      const double seconds = chrono::duration<double>(chrono::steady_clock::now() - start).count();
      const uint64_t conversion_bytes = adapter.current_comm() - comm_before;
      if (party == sci::ALICE) {
        io.send_data(low.data(), count * sizeof(uint64_t));
        io.send_data(high.data(), count * sizeof(uint64_t));
        io.recv_data(peer_low.data(), count * sizeof(uint64_t));
        io.recv_data(peer_high.data(), count * sizeof(uint64_t));
      } else {
        io.recv_data(peer_low.data(), count * sizeof(uint64_t));
        io.recv_data(peer_high.data(), count * sizeof(uint64_t));
        io.send_data(low.data(), count * sizeof(uint64_t));
        io.send_data(high.data(), count * sizeof(uint64_t));
      }
      const __uint128_t mask = (__uint128_t{1} << width) - 1;
      int errors = 0;
      for (int i = 0; i < count; ++i) {
        const __uint128_t value = (__uint128_t(high[i]) << 64) | low[i];
        const __uint128_t peer = (__uint128_t(peer_high[i]) << 64) | peer_low[i];
        if (((value + peer) & mask) != (expected[i] & mask)) ++errors;
      }
      passed &= errors == 0;
      cout << "B_PUBLIC_CENTERED_WIDE width=" << width << " samples=" << count
           << " errors=" << errors << " seconds=" << seconds
           << " comm_bytes=" << conversion_bytes << endl;
    }
    cout << "B_PUBLIC_CENTERED_WIDE status=" << (passed ? "PASS" : "FAIL") << endl;
    return passed ? 0 : 1;
  }
  if (linear_scale_cycle == 6) {
    Bert_ckks adapter(party, &io, address, port + 1);
    const auto source_context = context_at_chain(he, 4);
    vector<double> values(he.num_slots);
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = i % 4 == 0 ? -48.0 :
                  (i % 4 == 1 ? 44.0 : double(int(i % 25) - 12) / 4.0);
    vector<Ciphertext> source(1);
    if (party == sci::BOB) {
      Plaintext plain;
      he.encoder->encode(values, source_context->parms_id(),
                         ldexp(1.0, 64), plain);
      he.encryptor->encrypt(plain, source[0]);
      send_encrypted_vector(&io, source);
    } else {
      recv_encrypted_vector(he.context, &io, source);
    }
    bool passed = true;
    // 49 is a regression witness, not a selectable production path.
    for (int scale : {49, Bert_ckks::kGeluConversionScaleBits}) {
      vector<Ciphertext> result(2);
      if (party == sci::ALICE) {
        vector<UnifiedCiphertext> input;
        input.emplace_back(source[0]);
        if (device_backend) input[0].to_device(*he.context, *he.dcontext);
        vector<UnifiedCiphertext> output;
        adapter.eval_gelu_poly(output, input, &he, scale);
        for (int branch=0; branch<2; ++branch) {
          if (device_backend) output[branch].to_host(*he.context, *he.dcontext);
          result[branch] = output[branch].hcipher();
        }
        send_encrypted_vector(&io, result);
      } else {
        recv_encrypted_vector(he.context, &io, result);
        for (int branch=0; branch<2; ++branch) {
        Plaintext plain;
        he.decryptor->decrypt(result[branch], plain);
        vector<complex<double>> decoded;
        he.encoder->decode(plain, decoded);
        double max_error = 0.0, interior_error = 0.0;
        for (size_t i = 0; i < values.size(); ++i) {
          const double x2 = values[i] * values[i];
          const double sign = branch == 0 ? -1.0 : 1.0;
          const double expected = 0.020848611754127593*x2*x2
              + sign*(-0.18352506127082727)*x2*values[i]
              + 0.5410550166368381*x2
              + (0.5+sign*(-0.03798164612714154))*values[i]
              + 0.001620808531841547;
          const double error = abs(decoded[i].real() - expected);
          if (!isfinite(error)) { passed = false; break; }
          max_error = max(max_error, error);
          if (abs(values[i]) <= 3.0) interior_error = max(interior_error, error);
        }
        const bool accurate = max_error <= 0.5 && interior_error <= 0.01;
        passed = passed && (scale == 49 ? !accurate : accurate);
        cout << "B_PUBLIC_GELU_RANGE scale=" << scale
             << " max_abs_error=" << max_error
             << " interior_max_abs_error=" << interior_error
             << " expected=" << (scale == 49 ? "OVERFLOW" : "ACCURATE")
             << " accurate=" << accurate << " branch=" << branch << endl;
        }
      }
    }
    uint8_t success = passed ? 1 : 0;
    if (party == sci::BOB) io.send_data(&success, 1);
    else io.recv_data(&success, 1);
    cout << "B_PUBLIC_GELU_RANGE_SELFTEST status="
         << (success ? "PASS" : "FAIL") << endl;
    return success ? 0 : 1;
  }
  if (linear_scale_cycle == 5) {
    setenv("OURS_VIT_B_PUBLIC_FIXTURE", "1", 1);
    FCMetadata data{};
    data.slot_count = he.num_slots;
    data.filter_h = COMMON_DIM;
    data.filter_w = OUTPUT_DIM;
    data.image_size = 256;
    const int feature_blocks = data.slot_count / 2 / data.image_size;
    const int softmax_ciphertexts_per_pair =
        data.image_size * data.image_size * 2 / data.slot_count;
    const int softmax_ciphertexts =
        PACKING_NUM / 2 * softmax_ciphertexts_per_pair;
    const int v_ciphertexts_per_pair = data.filter_w / feature_blocks;
    const int v_ciphertexts = PACKING_NUM / 2 * v_ciphertexts_per_pair;

    auto attention_value = [&](int head, int row, int column) {
      const int diagonal =
          (column - row + data.image_size) % data.image_size;
      if (diagonal == 0) return 0.40 + 0.001 * (head % 3);
      if (diagonal == 1) return 0.30 - 0.001 * (head % 3);
      if (diagonal == 7) return 0.20;
      if (diagonal == data.image_size - 3) return 0.10;
      return 0.0;
    };
    auto value_value = [&](int head, int row, int feature) {
      return 0.35 * sin((head + 1) * 0.11 + (row + 1) * 0.037 +
                        (feature + 1) * 0.021) +
             0.15 * cos((head + 1) * 0.07 - (row + 1) * 0.019 +
                        (feature + 1) * 0.043);
    };

    if (party == sci::BOB) {
      vector<Ciphertext> result(2 * v_ciphertexts);
      recv_encrypted_vector(he.context, &io, result);
      vector<vector<complex<double>>> decoded(
          result.size(), vector<complex<double>>(data.slot_count));
      for (int ciphertext = 0; ciphertext < static_cast<int>(result.size());
           ++ciphertext) {
        Plaintext plaintext;
        he.decryptor->decrypt(result[ciphertext], plaintext);
        he.encoder->decode(plaintext, decoded[ciphertext]);
      }
      vector<vector<complex<double>>> av_decoded(
          decoded.begin(), decoded.begin() + v_ciphertexts);
      vector<vector<complex<double>>> wo_decoded(
          decoded.begin() + v_ciphertexts, decoded.end());
      const auto unpacked = unpack_qkv_tensor(av_decoded, data);
      FCMetadata wo_data = data;
      wo_data.filter_h = COMMON_DIM;
      wo_data.filter_w = COMMON_DIM;
      const auto wo_unpacked =
          unpack_bsgs_output_tensor(wo_decoded, wo_data);
      vector<double> unpacked_imag(unpacked.size(), 0.0);
      for (int ciphertext = 0; ciphertext < v_ciphertexts; ++ciphertext) {
        for (int slot = 0; slot < data.slot_count; ++slot) {
          int feature = slot / data.image_size;
          const int row = slot % data.image_size;
          int paired_head_offset = 0;
          if (slot >= data.slot_count / 2) {
            feature -= data.slot_count / data.image_size / 2;
            paired_head_offset = data.filter_w * data.image_size;
          }
          const size_t destination =
              static_cast<size_t>(row) * data.filter_w + feature +
              paired_head_offset +
              static_cast<size_t>(ciphertext / 2) * data.image_size *
                  data.filter_w * 2 +
              static_cast<size_t>(ciphertext % 2) * data.filter_w / 2;
          unpacked_imag[destination] = decoded[ciphertext][slot].imag();
        }
      }
      double max_abs = 0.0;
      double max_abs_imag = 0.0;
      long double squared_error = 0.0L;
      long double inverse_squared_error = 0.0L;
      double wo_max_abs = 0.0;
      double wo_max_abs_imag = 0.0;
      long double wo_squared_error = 0.0L;
      uint64_t elements = 0;
      vector<double> head_max_abs(PACKING_NUM, 0.0);
      vector<double> head_max_imag(PACKING_NUM, 0.0);
      vector<long double> head_squared_error(PACKING_NUM, 0.0L);
      vector<array<long double, 2>> half_squared_error(PACKING_NUM);
      vector<array<uint64_t, 2>> half_elements(PACKING_NUM);
      size_t worst_index = 0;
      for (int head = 0; head < PACKING_NUM; ++head) {
        for (int row = 0; row < data.image_size; ++row) {
          for (int feature = 0; feature < data.filter_w; ++feature) {
            double reference = 0.0;
            double inverse_reference = 0.0;
            for (int column = 0; column < data.image_size; ++column) {
              const double attention =
                  attention_value(head, row, column);
              reference += attention * value_value(head, column, feature);
              const int diagonal =
                  (column - row + data.image_size) % data.image_size;
              const int inverse_column =
                  (row - diagonal + data.image_size) % data.image_size;
              inverse_reference +=
                  attention * value_value(head, inverse_column, feature);
            }
            const size_t index =
                (static_cast<size_t>(head) * data.image_size + row) *
                    data.filter_w +
                feature;
            const double error = unpacked[index] - reference;
            if (abs(error) > max_abs) {
              max_abs = abs(error);
              worst_index = index;
            }
            squared_error += static_cast<long double>(error) * error;
            const double inverse_error = unpacked[index] - inverse_reference;
            inverse_squared_error +=
                static_cast<long double>(inverse_error) * inverse_error;
            head_max_abs[head] = max(head_max_abs[head], abs(error));
            head_squared_error[head] +=
                static_cast<long double>(error) * error;
            const int half = feature / feature_blocks;
            half_squared_error[head][half] +=
                static_cast<long double>(error) * error;
            ++half_elements[head][half];
            max_abs_imag = max(max_abs_imag, abs(unpacked_imag[index]));
            head_max_imag[head] =
                max(head_max_imag[head], abs(unpacked_imag[index]));
            const size_t wo_index =
                static_cast<size_t>(row) * COMMON_DIM +
                head * data.filter_w + feature;
            const double wo_error = wo_unpacked[wo_index] - reference;
            wo_max_abs = max(wo_max_abs, abs(wo_error));
            wo_squared_error +=
                static_cast<long double>(wo_error) * wo_error;
            ++elements;
          }
        }
      }
      const double rmse =
          sqrt(static_cast<double>(squared_error / elements));
      for (int head = 0; head < PACKING_NUM; ++head) {
        cout << "B_PUBLIC_AV_HEAD status=DIAGNOSTIC head=" << head
             << " max_abs=" << head_max_abs[head]
             << " rmse="
             << sqrt(static_cast<double>(
                    head_squared_error[head] /
                    (data.image_size * data.filter_w)))
             << " max_abs_imag=" << head_max_imag[head]
             << " half0_rmse="
             << sqrt(static_cast<double>(head_squared_error[head] == 0.0L
                    ? 0.0L
                    : half_squared_error[head][0] /
                          half_elements[head][0]))
             << " half1_rmse="
             << sqrt(static_cast<double>(head_squared_error[head] == 0.0L
                    ? 0.0L
                    : half_squared_error[head][1] /
                          half_elements[head][1]))
             << endl;
      }
      const bool passed = max_abs <= 1e-3 && max_abs_imag <= 1e-3;
      for (const auto &ciphertext : wo_decoded)
        for (const auto &slot : ciphertext)
          wo_max_abs_imag = max(wo_max_abs_imag, abs(slot.imag()));
      const double wo_rmse =
          sqrt(static_cast<double>(wo_squared_error / elements));
      const bool wo_passed =
          wo_max_abs <= 1e-3 && wo_max_abs_imag <= 1e-3;
      cout << "B_PUBLIC_AV_SELFTEST status="
           << (passed ? "PASS" : "FAIL")
           << " backend=" << (device_backend ? "phantom_device" : "seal_host")
           << " max_abs=" << max_abs << " rmse=" << rmse
           << " inverse_diagonal_rmse="
           << sqrt(static_cast<double>(inverse_squared_error / elements))
           << " max_abs_imag=" << max_abs_imag
           << " worst_head="
           << worst_index / (data.image_size * data.filter_w)
           << " worst_row="
           << (worst_index / data.filter_w) % data.image_size
           << " worst_feature=" << worst_index % data.filter_w
           << " elements=" << elements
           << " fixture=public_deterministic" << endl;
      cout << "B_PUBLIC_WO_SELFTEST status="
           << (wo_passed ? "PASS" : "FAIL")
           << " backend=" << (device_backend ? "phantom_device" : "seal_host")
           << " input_chain=3 max_abs=" << wo_max_abs
           << " rmse=" << wo_rmse
           << " max_abs_imag=" << wo_max_abs_imag
           << " elements=" << elements
           << " weights=public_identity fixture=public_deterministic" << endl;
      return passed && wo_passed ? 0 : 1;
    }

    vector<vector<double>> packed_softmax(
        softmax_ciphertexts, vector<double>(data.slot_count, 0.0));
    for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
      for (int diagonal = 0; diagonal < data.image_size; ++diagonal) {
        const int ciphertext =
            pair * softmax_ciphertexts_per_pair + diagonal / feature_blocks;
        const int block = diagonal % feature_blocks;
        for (int row = 0; row < data.image_size; ++row) {
          const int column = (row + diagonal) % data.image_size;
          packed_softmax[ciphertext][block * data.image_size + row] =
              attention_value(2 * pair, row, column);
          packed_softmax[ciphertext][data.slot_count / 2 +
                                      block * data.image_size + row] =
              attention_value(2 * pair + 1, row, column);
        }
      }
    }
    vector<vector<double>> packed_v(
        v_ciphertexts, vector<double>(data.slot_count, 0.0));
    for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
      for (int feature_ciphertext = 0;
           feature_ciphertext < v_ciphertexts_per_pair;
           ++feature_ciphertext) {
        const int ciphertext =
            pair * v_ciphertexts_per_pair + feature_ciphertext;
        for (int feature_block = 0; feature_block < feature_blocks;
             ++feature_block) {
          const int feature =
              feature_ciphertext * feature_blocks + feature_block;
          for (int row = 0; row < data.image_size; ++row) {
            packed_v[ciphertext][feature_block * data.image_size + row] =
                value_value(2 * pair, row, feature);
            packed_v[ciphertext][data.slot_count / 2 +
                                 feature_block * data.image_size + row] =
                value_value(2 * pair + 1, row, feature);
          }
        }
      }
    }

    const auto first_context = he.context->first_context_data();
    if (!first_context)
      throw runtime_error("B public AV fixture has no CKKS data level");
    const double scale = ldexp(1.0, he.he_scale);
    Plaintext zero_plaintext;
    he.encoder->encode(vector<double>(data.slot_count, 0.0),
                       first_context->parms_id(), scale, zero_plaintext);
    Ciphertext encrypted_zero;
    he.encryptor->encrypt(zero_plaintext, encrypted_zero);
    auto encrypt_public = [&](const vector<vector<double>> &packed) {
      vector<UnifiedCiphertext> result;
      result.reserve(packed.size());
      for (const auto &slots : packed) {
        Plaintext plaintext;
        he.encoder->encode(slots, first_context->parms_id(), scale, plaintext);
        Ciphertext ciphertext = encrypted_zero;
        he.evaluator->add_plain_inplace(ciphertext, plaintext);
        UnifiedCiphertext unified(std::move(ciphertext));
        if (device_backend)
          unified.to_device(*he.context, *he.dcontext);
        result.push_back(std::move(unified));
      }
      return result;
    };
    vector<UnifiedCiphertext> encrypted_softmax =
        encrypt_public(packed_softmax);
    vector<UnifiedCiphertext> encrypted_v = encrypt_public(packed_v);
    cout << "B_PUBLIC_AV_FIXTURE stage=device_inputs_ready"
         << " softmax_cts=" << encrypted_softmax.size()
         << " v_cts=" << encrypted_v.size() << endl;

    Linear_ckks linear;
    linear.party = party;
    linear.io = &io;
    vector<vector<vector<UnifiedCiphertext>>> diagonal_softmax;
    linear.preprocess_softmax_s1_ct_ct(
        &he, encrypted_softmax, data, diagonal_softmax);
    const size_t target_limbs =
        diagonal_softmax.front().front().front().coeff_modulus_size();
    for (auto &ciphertext : encrypted_v) {
      while (ciphertext.coeff_modulus_size() > target_limbs) {
        if (device_backend)
          he.devaluator->mod_switch_to_next_inplace(ciphertext);
        else
          he.evaluator->mod_switch_to_next_inplace(ciphertext);
      }
      if (ciphertext.coeff_modulus_size() != target_limbs)
        throw runtime_error("B public AV V level alignment failed");
    }
    vector<UnifiedCiphertext> av_result(
        v_ciphertexts,
        UnifiedCiphertext(device_backend ? LOCATION::DEVICE : LOCATION::HOST));
    linear.softmax_v(
        &he, diagonal_softmax, encrypted_v, data, av_result, 1.0);
    if (av_result.front().coeff_modulus_size() != 4 ||
        abs(log2(av_result.front().scale()) - 69.0) > 1e-6) {
      throw runtime_error("B public AV production metadata mismatch");
    }
    FCMetadata output_data{};
    output_data.slot_count = he.num_slots;
    output_data.filter_h = COMMON_DIM;
    output_data.filter_w = COMMON_DIM;
    output_data.image_size = data.image_size;
    vector<vector<uint64_t>> identity(
        COMMON_DIM, vector<uint64_t>(COMMON_DIM, 0));
    vector<uint64_t> zero_bias(COMMON_DIM, 0);
    for (int feature = 0; feature < COMMON_DIM; ++feature)
      identity[feature][feature] = uint64_t{1} << 6;
    const auto packed_identity =
        linear.attention_output_projection_weights_for_bsgs(identity);
    auto output_params = linear.params_preprocessing_ct_pt(
        &he, data.image_size, COMMON_DIM, COMMON_DIM, packed_identity,
        zero_bias,
        output_data, 6, 12, 3, av_result.front().scale());
    vector<UnifiedCiphertext> wo_result;
    linear.linear_2(
        &he, av_result, output_params, output_data, wo_result);
    cout << "B_PUBLIC_WO_FIXTURE stage=output_projection_ready"
         << " input_chain=3 input_scale_bits="
         << log2(av_result.front().scale())
         << " output_cts=" << wo_result.size() << endl;
    vector<Ciphertext> host_result(av_result.size() + wo_result.size());
    for (size_t index = 0; index < av_result.size(); ++index) {
      if (device_backend)
        av_result[index].to_host(*he.context, *he.dcontext);
      host_result[index] = std::move(av_result[index].hcipher());
    }
    for (size_t index = 0; index < wo_result.size(); ++index) {
      if (device_backend)
        wo_result[index].to_host(*he.context, *he.dcontext);
      host_result[av_result.size() + index] =
          std::move(wo_result[index].hcipher());
    }
    send_encrypted_vector(&io, host_result);
    return 0;
  }
  if (linear_scale_cycle == 4) {
    setenv("OURS_VIT_B_PUBLIC_FIXTURE", "1", 1);
    const bool dense_qkv_fixture =
        getenv("OURS_VIT_B_PUBLIC_DENSE_QKV") != nullptr;
    const bool impulse_qkv_fixture =
        getenv("OURS_VIT_B_PUBLIC_IMPULSE_QKV") != nullptr;
    FCMetadata data{};
    data.slot_count = he.num_slots;
    data.filter_h = COMMON_DIM;
    data.filter_w = OUTPUT_DIM;
    data.image_size = 256;
    const auto input_context = context_at_chain(he, 5);
    const double input_scale = ldexp(1.0, kQkvInputScale);
    vector<double> input(static_cast<size_t>(data.image_size) * COMMON_DIM);
    for (int row = 0; row < data.image_size; ++row) {
      for (int feature = 0; feature < COMMON_DIM; ++feature) {
        input[static_cast<size_t>(row) * COMMON_DIM + feature] =
            0.25 * sin((row + 1) * 0.17 + (feature + 1) * 0.013) +
            0.10 * cos((row + 1) * 0.031 - (feature + 1) * 0.019);
      }
    }
    vector<double> packed_input(input.size());
    for (int feature = 0; feature < COMMON_DIM; ++feature)
      for (int row = 0; row < data.image_size; ++row)
        packed_input[static_cast<size_t>(feature) * data.image_size + row] =
            input[static_cast<size_t>(row) * COMMON_DIM + feature];
    vector<vector<uint64_t>> bias(PACKING_NUM,
                                  vector<uint64_t>(OUTPUT_DIM));
    vector<vector<vector<uint64_t>>> identity(
        PACKING_NUM,
        vector<vector<uint64_t>>(COMMON_DIM,
                                 vector<uint64_t>(OUTPUT_DIM)));
    auto fixture_weight = [&](int head, int input_feature, int output_feature) {
      if (impulse_qkv_fixture) {
        return head == 0 && input_feature == 123 && output_feature == 7
                   ? (int64_t{1} << 6)
                   : int64_t{0};
      }
      if (!dense_qkv_fixture) {
        return input_feature == head * OUTPUT_DIM + output_feature
                   ? (int64_t{1} << 6)
                   : int64_t{0};
      }
      const double value =
          0.060 * sin((head + 1) * 0.173 + (input_feature + 1) * 0.0071 +
                      (output_feature + 1) * 0.0137) +
          0.040 * cos((head + 1) * 0.097 - (input_feature + 1) * 0.0113 +
                      (output_feature + 1) * 0.0191);
      return static_cast<int64_t>(llround(value * (1 << 6)));
    };
    for (int head = 0; head < PACKING_NUM; ++head) {
      for (int input_feature = 0; input_feature < COMMON_DIM; ++input_feature)
        for (int output_feature = 0; output_feature < OUTPUT_DIM;
             ++output_feature)
          identity[head][input_feature][output_feature] =
              static_cast<uint64_t>(fixture_weight(
                  head, input_feature, output_feature)) & ring_mask();
    }

    const int input_ciphertexts =
        data.image_size * COMMON_DIM / data.slot_count;
    vector<Ciphertext> encrypted_input(input_ciphertexts);
    if (party == sci::BOB) {
      const int tensor_ciphertexts =
          data.image_size * data.filter_w * PACKING_NUM / data.slot_count;
      vector<Ciphertext> result(3 * tensor_ciphertexts);
      recv_encrypted_vector(he.context, &io, result);
      if (static_cast<int>(result.size()) != 3 * tensor_ciphertexts)
        throw runtime_error("B public QKV result partition mismatch");
      vector<vector<complex<double>>> decoded(
          result.size(), vector<complex<double>>(data.slot_count));
      for (size_t ciphertext = 0; ciphertext < result.size(); ++ciphertext) {
        Plaintext plaintext;
        he.decryptor->decrypt(result[ciphertext], plaintext);
        he.encoder->decode(plaintext, decoded[ciphertext]);
      }
      double global_max_abs = 0.0;
      long double global_squared_error = 0.0L;
      uint64_t compared = 0;
      const size_t tensor_elements =
          static_cast<size_t>(PACKING_NUM) * data.image_size * OUTPUT_DIM;
      vector<vector<double>> expected_tensor(
          3, vector<double>(tensor_elements, 0.0));
      for (int tensor = 0; tensor < 3; ++tensor) {
        vector<vector<complex<double>>> tensor_packed(
            decoded.begin() + tensor * tensor_ciphertexts,
            decoded.begin() + (tensor + 1) * tensor_ciphertexts);
        const auto unpacked = unpack_qkv_tensor(tensor_packed, data);
        double max_abs = 0.0;
        long double squared_error = 0.0L;
        double max_abs_imag = 0.0;
        for (int head = 0; head < PACKING_NUM; ++head) {
          for (int row = 0; row < data.image_size; ++row) {
            for (int feature = 0; feature < OUTPUT_DIM; ++feature) {
              const size_t output_index =
                  (static_cast<size_t>(head) * data.image_size + row) *
                      OUTPUT_DIM +
                  feature;
              const size_t input_index =
                  static_cast<size_t>(row) * COMMON_DIM +
                  head * OUTPUT_DIM + feature;
              double expected_value = 0.0;
              for (int input_feature = 0; input_feature < COMMON_DIM;
                   ++input_feature) {
                const int64_t weight_raw = fixture_weight(
                    head, input_feature, feature);
                expected_value +=
                    input[static_cast<size_t>(row) * COMMON_DIM +
                          input_feature] *
                    static_cast<double>(weight_raw) / (1 << 6);
              }
              if (tensor == 0) expected_value *= 0.125;
              expected_tensor[tensor][output_index] = expected_value;
              const double error = unpacked[output_index] - expected_value;
              max_abs = max(max_abs, abs(error));
              squared_error += static_cast<long double>(error) * error;
              ++compared;
            }
          }
        }
        for (const auto &ciphertext : tensor_packed)
          for (const auto &slot : ciphertext)
            max_abs_imag = max(max_abs_imag, abs(slot.imag()));
        if (dense_qkv_fixture || impulse_qkv_fixture) {
          cout << "B_PUBLIC_DENSE_QKV_SAMPLE tensor="
               << (tensor == 0 ? "Q" : tensor == 1 ? "K" : "V")
               << " got=";
          for (int feature = 0; feature < 8; ++feature) {
            if (feature != 0) cout << ',';
            cout << unpacked[static_cast<size_t>(feature)];
          }
          cout << " expected=";
          for (int feature = 0; feature < 8; ++feature) {
            if (feature != 0) cout << ',';
            double sample_expected = 0.0;
            for (int input_feature = 0; input_feature < COMMON_DIM;
                 ++input_feature) {
              sample_expected +=
                  input[static_cast<size_t>(input_feature)] *
                  static_cast<double>(fixture_weight(
                      0, input_feature, feature)) / (1 << 6);
            }
            if (tensor == 0) sample_expected *= 0.125;
            cout << sample_expected;
          }
          cout << endl;
          if (impulse_qkv_fixture) {
            size_t worst = 0;
            double worst_error = -1.0;
            for (int row = 0; row < data.image_size; ++row) {
              for (int feature = 0; feature < OUTPUT_DIM; ++feature) {
                const double expected_at =
                    input[static_cast<size_t>(row) * COMMON_DIM + 123] *
                    (feature == 7 ? 1.0 : 0.0) *
                    (tensor == 0 ? 0.125 : 1.0);
                const size_t index =
                    (static_cast<size_t>(row) * OUTPUT_DIM) + feature;
                const double error = abs(unpacked[index] - expected_at);
                if (error > worst_error) {
                  worst_error = error;
                  worst = index;
                }
              }
            }
            const int worst_row = static_cast<int>(worst / OUTPUT_DIM);
            const int worst_feature = static_cast<int>(worst % OUTPUT_DIM);
            const size_t worst_flat =
                static_cast<size_t>(worst_row) * OUTPUT_DIM + worst_feature;
            cout << "B_PUBLIC_IMPULSE_SAMPLE tensor="
                 << (tensor == 0 ? "Q" : tensor == 1 ? "K" : "V")
                 << " row=" << worst_row << " feature=" << worst_feature
                 << " got="
                 << unpacked[worst_flat]
                 << " expected="
                 << input[static_cast<size_t>(worst_row) * COMMON_DIM + 123] *
                        (worst_feature == 7 ? 1.0 : 0.0) *
                        (tensor == 0 ? 0.125 : 1.0)
                 << endl;

            // An impulse is also a packing probe.  The expected output is
            // intentionally a single (head,row,feature) lane; report the
            // largest observed lanes so a deterministic BSGS permutation is
            // visible even when the target lane itself is zero.
            vector<size_t> peak_indices(unpacked.size());
            iota(peak_indices.begin(), peak_indices.end(), size_t{0});
            sort(peak_indices.begin(), peak_indices.end(),
                 [&unpacked](size_t lhs, size_t rhs) {
                   return abs(unpacked[lhs]) > abs(unpacked[rhs]);
                 });
            cout << "B_PUBLIC_IMPULSE_PEAKS tensor="
                 << (tensor == 0 ? "Q" : tensor == 1 ? "K" : "V");
            const size_t peak_count = min<size_t>(12, peak_indices.size());
            for (size_t peak = 0; peak < peak_count; ++peak) {
              const size_t flat = peak_indices[peak];
              const int head = static_cast<int>(
                  flat / (static_cast<size_t>(data.image_size) * OUTPUT_DIM));
              const size_t head_offset =
                  flat % (static_cast<size_t>(data.image_size) * OUTPUT_DIM);
              const int row = static_cast<int>(head_offset / OUTPUT_DIM);
              const int feature = static_cast<int>(head_offset % OUTPUT_DIM);
              cout << " [head=" << head << " row=" << row
                   << " feature=" << feature
                   << " value=" << unpacked[flat] << "]";
            }
            cout << endl;
          }
        }
        global_max_abs = max(global_max_abs, max_abs);
        global_squared_error += squared_error;
        cout << "B_PUBLIC_QKV_TENSOR status="
             << (max_abs <= 1e-3 && max_abs_imag <= 1e-3 ? "PASS" : "FAIL")
             << " tensor=" << (tensor == 0 ? "Q" : tensor == 1 ? "K" : "V")
             << " max_abs=" << max_abs
             << " rmse="
             << sqrt(static_cast<double>(squared_error /
                                         (PACKING_NUM * data.image_size *
                                          OUTPUT_DIM)))
             << " max_abs_imag=" << max_abs_imag << endl;
      }
      const bool qkv_passed = global_max_abs <= 1e-3;
      cout << "B_PUBLIC_QKV_SELFTEST status="
           << (qkv_passed ? "PASS" : "FAIL")
           << " backend=" << (device_backend ? "phantom_device" : "seal_host")
           << " max_abs=" << global_max_abs
           << " rmse="
           << sqrt(static_cast<double>(global_squared_error / compared))
           << " fixture=public_deterministic" << endl;

      const int qk_ciphertexts =
          data.image_size * data.image_size * PACKING_NUM / data.slot_count;
      vector<Ciphertext> qk_result(qk_ciphertexts);
      recv_encrypted_vector(he.context, &io, qk_result);
      vector<vector<complex<double>>> qk_decoded(
          qk_ciphertexts, vector<complex<double>>(data.slot_count));
      for (int ciphertext = 0; ciphertext < qk_ciphertexts; ++ciphertext) {
        Plaintext plaintext;
        he.decryptor->decrypt(qk_result[ciphertext], plaintext);
        he.encoder->decode(plaintext, qk_decoded[ciphertext]);
      }
      const auto qk = unpack_qk_tensor(qk_decoded, data);
      double qk_max_abs = 0.0;
      long double qk_squared_error = 0.0L;
      for (int head = 0; head < PACKING_NUM; ++head) {
        for (int row = 0; row < data.image_size; ++row) {
          for (int column = 0; column < data.image_size; ++column) {
            double reference = 0.0;
            for (int feature = 0; feature < OUTPUT_DIM; ++feature) {
              const size_t lhs =
                  (static_cast<size_t>(head) * data.image_size + row) *
                      OUTPUT_DIM + feature;
              const size_t rhs =
                  (static_cast<size_t>(head) * data.image_size + column) *
                      OUTPUT_DIM + feature;
              reference += expected_tensor[0][lhs] * expected_tensor[1][rhs];
            }
            const size_t index =
                (static_cast<size_t>(head) * data.image_size + row) *
                    data.image_size +
                column;
            const double error = qk[index] - reference;
            qk_max_abs = max(qk_max_abs, abs(error));
            qk_squared_error += static_cast<long double>(error) * error;
          }
        }
      }
      const uint64_t qk_elements = static_cast<uint64_t>(PACKING_NUM) *
                                   data.image_size * data.image_size;
      const double qk_rmse =
          sqrt(static_cast<double>(qk_squared_error / qk_elements));
      const bool qk_passed = qk_max_abs <= 1e-2;
      cout << "B_PUBLIC_QK_SELFTEST status="
           << (qk_passed ? "PASS" : "FAIL")
           << " backend=" << (device_backend ? "phantom_device" : "seal_host")
           << " max_abs=" << qk_max_abs << " rmse=" << qk_rmse
           << " elements=" << qk_elements
           << " fixture=public_deterministic" << endl;
      return qkv_passed && qk_passed ? 0 : 1;
    }

    Linear_ckks linear;
    linear.party = party;
    linear.io = &io;
    auto params = linear.params_preprocessing_ct_ct(
        &he, identity, identity, identity, bias, bias, bias, data,
        input_context->chain_index(), input_scale);
    cout << "B_PUBLIC_QKV_FIXTURE stage=packed_weights_ready" << endl;
    // The fixture is deliberately public, so encrypt it locally with the
    // protocol public key. This isolates QKV from unrelated client-side
    // encryption and input-transfer time while exercising the same evaluator.
    vector<double> first_slots(packed_input.begin(),
                               packed_input.begin() + data.slot_count);
    Plaintext first_plaintext;
    he.encoder->encode(first_slots, input_context->parms_id(), input_scale,
                       first_plaintext);
    cout << "B_PUBLIC_QKV_FIXTURE stage=first_plaintext_ready" << endl;
    he.encryptor->encrypt(first_plaintext, encrypted_input[0]);
    cout << "B_PUBLIC_QKV_FIXTURE stage=first_ciphertext_ready" << endl;
    // Reuse the one public-fixture encryption and add public plaintext
    // deltas. This keeps the self-test focused on QKV and avoids generating
    // twelve independent 32768-degree encryptions. It is never used by the
    // model runtime or any private input path.
    for (int ciphertext = 1; ciphertext < input_ciphertexts; ++ciphertext) {
      vector<double> delta(data.slot_count);
      const size_t offset = static_cast<size_t>(ciphertext) * data.slot_count;
      for (int slot = 0; slot < data.slot_count; ++slot)
        delta[slot] = packed_input[offset + slot] - first_slots[slot];
      Plaintext delta_plaintext;
      he.encoder->encode(delta, input_context->parms_id(), input_scale,
                         delta_plaintext);
      encrypted_input[ciphertext] = encrypted_input[0];
      he.evaluator->add_plain_inplace(encrypted_input[ciphertext],
                                      delta_plaintext);
    }
    vector<UnifiedCiphertext> unified_input;
    unified_input.reserve(encrypted_input.size());
    for (auto &ciphertext : encrypted_input) {
      UnifiedCiphertext unified(std::move(ciphertext));
      if (device_backend)
        unified.to_device(*he.context, *he.dcontext);
      unified_input.push_back(std::move(unified));
    }
    cout << "B_PUBLIC_QKV_FIXTURE stage=device_input_ready" << endl;
    const int tensor_ciphertexts =
        data.image_size * data.filter_w * PACKING_NUM / data.slot_count;
    vector<UnifiedCiphertext> result;
    vector<UnifiedCiphertext> qkv_observer;
    linear.linear_1(&he, unified_input, params, data, result, &qkv_observer);
    vector<Ciphertext> host_result(qkv_observer.size());
    for (size_t index = 0; index < qkv_observer.size(); ++index) {
      if (device_backend)
        qkv_observer[index].to_host(*he.context, *he.dcontext);
      host_result[index] = std::move(qkv_observer[index].hcipher());
    }
    send_encrypted_vector(&io, host_result);
    const int qk_ciphertexts =
        data.image_size * data.image_size * PACKING_NUM / data.slot_count;
    vector<Ciphertext> host_qk(qk_ciphertexts);
    for (int index = 0; index < qk_ciphertexts; ++index) {
      if (device_backend)
        result[index].to_host(*he.context, *he.dcontext);
      host_qk[index] = std::move(result[index].hcipher());
    }
    send_encrypted_vector(&io, host_qk);
    return 0;
  }
  if (linear_scale_cycle == 3) {
    vector<double> x(he.num_slots);
    vector<double> sigma(he.num_slots);
    vector<double> weights(he.num_slots);
    vector<double> biases(he.num_slots);
    vector<double> reference(he.num_slots);
    for (size_t index = 0; index < he.num_slots; ++index) {
      x[index] = static_cast<double>(static_cast<int>(index % 31) - 15) /
                 16.0;
      sigma[index] = 0.5 * (1 + static_cast<int>(index % 3));
      weights[index] = index % 3 == 0 ? 1.0 : (index % 3 == 1 ? 0.5 : -0.5);
      biases[index] = index & 1 ? 0.25 : -0.25;
      reference[index] = x[index] * sigma[index] * weights[index] +
                         biases[index];
    }
    if (party == sci::BOB) {
      Plaintext x_plaintext;
      Plaintext sigma_plaintext;
      he.encoder->encode(x, ldexp(1.0, kLayerNormScale), x_plaintext);
      he.encoder->encode(sigma, ldexp(1.0, kLayerNormScale),
                         sigma_plaintext);
      vector<Ciphertext> inputs(2);
      he.encryptor->encrypt(x_plaintext, inputs[0]);
      he.encryptor->encrypt(sigma_plaintext, inputs[1]);
      send_encrypted_vector(&io, inputs);
      vector<Ciphertext> result(1);
      recv_encrypted_vector(he.context, &io, result);
      Plaintext result_plaintext;
      he.decryptor->decrypt(result[0], result_plaintext);
      vector<double> decoded;
      he.encoder->decode(result_plaintext, decoded);
      double max_abs = 0.0;
      long double squared_error = 0.0L;
      for (size_t index = 0; index < he.num_slots; ++index) {
        const double error = decoded[index] - reference[index];
        max_abs = max(max_abs, abs(error));
        squared_error += static_cast<long double>(error) * error;
      }
      cout << "B_NATIVE_CKKS_LAYERNORM_AFFINE_SELFTEST status="
           << (max_abs <= 1e-3 ? "PASS" : "FAIL")
           << " backend=" << (device_backend ? "phantom_device" : "seal_host")
           << " max_abs=" << max_abs
           << " rmse="
           << sqrt(static_cast<double>(squared_error / he.num_slots))
           << endl;
      return max_abs <= 1e-3 ? 0 : 1;
    }
    vector<Ciphertext> inputs(2);
    recv_encrypted_vector(he.context, &io, inputs);
    if (inputs.size() != 2)
      throw runtime_error("native CKKS LayerNorm input count mismatch");
    vector<Ciphertext> result(1);
    if (!device_backend) {
      Plaintext weight;
      Plaintext bias;
      he.encoder->encode(weights, inputs[0].parms_id(),
                         ldexp(1.0, kLayerNormScale), weight);
      he.encoder->encode(biases, inputs[0].parms_id(),
                         ldexp(1.0, 3 * kLayerNormScale), bias);
      he.evaluator->multiply_inplace(inputs[0], inputs[1]);
      he.evaluator->relinearize_inplace(inputs[0], *he.relin_keys);
      he.evaluator->multiply_plain_inplace(inputs[0], weight);
      he.evaluator->add_plain_inplace(inputs[0], bias);
      he.evaluator->rescale_to_next_inplace(inputs[0]);
      result[0] = std::move(inputs[0]);
    } else {
      UnifiedCiphertext x_ciphertext(std::move(inputs[0]));
      UnifiedCiphertext sigma_ciphertext(std::move(inputs[1]));
      x_ciphertext.to_device(*he.context, *he.dcontext);
      sigma_ciphertext.to_device(*he.context, *he.dcontext);
      UnifiedPlaintext weight(LOCATION::DEVICE);
      UnifiedPlaintext bias(LOCATION::DEVICE);
      const auto chain_index = x_ciphertext.dcipher().chain_index();
      he.dencoder->encode(*he.dcontext, weights,
                          ldexp(1.0, kLayerNormScale), weight, chain_index);
      he.dencoder->encode(*he.dcontext, biases,
                          ldexp(1.0, 3 * kLayerNormScale), bias, chain_index);
      he.devaluator->multiply_inplace(x_ciphertext, sigma_ciphertext);
      he.devaluator->relinearize_inplace(x_ciphertext, *he.drelin_keys);
      he.devaluator->multiply_plain_inplace(x_ciphertext, weight);
      he.devaluator->add_plain_inplace(x_ciphertext, bias);
      he.devaluator->rescale_to_next_inplace(x_ciphertext);
      x_ciphertext.to_host(*he.context, *he.dcontext);
      result[0] = std::move(x_ciphertext.hcipher());
    }
    send_encrypted_vector(&io, result);
    return 0;
  }
  const uint64_t mask = ring_mask();
  vector<vector<uint64_t>> expected(
      kCiphertexts, vector<uint64_t>(he.num_slots));
  vector<vector<HE_CKKS::scalar_t>> local_shares(
      kCiphertexts, vector<HE_CKKS::scalar_t>(he.num_slots));
  vector<vector<uint64_t>> sigma_expected(
      kCiphertexts, vector<uint64_t>(he.num_slots));
  vector<vector<HE_CKKS::scalar_t>> sigma_local_shares(
      kCiphertexts, vector<HE_CKKS::scalar_t>(he.num_slots));
  for (int ciphertext = 0; ciphertext < kCiphertexts; ++ciphertext) {
    for (size_t index = 0; index < expected[ciphertext].size(); ++index) {
      // Cover the small signed activations used by ViT as well as progressively
      // larger magnitudes.  The fixture remains deterministic and public.
      const int shift = 4 + (ciphertext % 8);
      const int64_t raw =
          static_cast<int64_t>((static_cast<int>(index + ciphertext) % 31) -
                               15) *
          (int64_t{1} << shift);
      expected[ciphertext][index] = static_cast<uint64_t>(raw) & mask;
      const uint64_t flat_index =
          static_cast<uint64_t>(ciphertext) * he.num_slots + index;
      const uint64_t alice_share =
          (uint64_t{0x13579bdf} + flat_index * 7919 +
           (flat_index * flat_index + 17) * 104729) &
          mask;
      local_shares[ciphertext][index] =
          party == sci::ALICE
              ? alice_share
              : (expected[ciphertext][index] - alice_share) & mask;
      const int64_t sigma_raw =
          (int64_t{1} << (kScale - 1)) *
          (1 + static_cast<int>((index + 2 * ciphertext) % 3));
      sigma_expected[ciphertext][index] =
          static_cast<uint64_t>(sigma_raw) & mask;
      const uint64_t sigma_alice_share =
          (uint64_t{0x2468ace1} + flat_index * 65537 +
           (flat_index * flat_index + 31) * 8191) &
          mask;
      sigma_local_shares[ciphertext][index] =
          party == sci::ALICE
              ? sigma_alice_share
              : (sigma_expected[ciphertext][index] - sigma_alice_share) & mask;
    }
  }

  Bert_ckks adapter(party, &io, address, port + 1);

  const uint64_t conversion_prime = he.get_conversion_prime();
  // The observer opens only this deterministic public synthetic vector. It
  // is intentionally unavailable in the model runtime invocation.
  setenv("OURS_VIT_B_PUBLIC_FIXTURE", "1", 1);
  setenv("OURS_VIT_B_M2C_OPEN", "1", 1);
  int import_scale = linear_scale_cycle == 2 ? kLayerNormScale : kCkksScale;
  if (const char *override_scale =
          getenv("OURS_VIT_B_SELFTEST_IMPORT_SCALE")) {
    import_scale = atoi(override_scale);
    if (import_scale <= 0 || import_scale >= 60)
      throw invalid_argument("OURS_VIT_B_SELFTEST_IMPORT_SCALE is invalid");
  }
  // CKKS multiplication scales are semantic metadata. If the imported
  // operand scale changes, the affine weight and bias scales must move with
  // it so x*sigma*weight and bias are add-compatible before the rescale.
  const double affine_weight_scale = ldexp(1.0, import_scale);
  const double affine_bias_scale = ldexp(1.0, 3 * import_scale);
  vector<Plaintext> imported = adapter.mpc_to_ckks(
      local_shares, he.num_slots, conversion_prime, kBits, kScale,
      import_scale, &he, false);
  if (imported.size() != kCiphertexts)
    throw runtime_error("B conversion self-test import shape mismatch");
  if (getenv("OURS_VIT_B_MPC_TO_CKKS_ONLY") != nullptr)
    return 0;
  vector<Plaintext> sigma_imported;
  if (linear_scale_cycle == 2) {
    sigma_imported = adapter.mpc_to_ckks(
        sigma_local_shares, he.num_slots, conversion_prime, kBits, kScale,
        import_scale, &he, false);
    if (sigma_imported.size() != kCiphertexts)
      throw runtime_error("B LayerNorm self-test sigma import mismatch");
    for (int ciphertext = 0; ciphertext < kCiphertexts; ++ciphertext) {
      for (size_t index = 0; index < expected[ciphertext].size(); ++index) {
        const int64_t x_raw = signed_ring(expected[ciphertext][index]);
        const int64_t sigma_raw =
            signed_ring(sigma_expected[ciphertext][index]);
        const int64_t weight_raw =
            ((index + ciphertext) % 3 == 0)
                ? (int64_t{1} << kScale)
                : (((index + ciphertext) % 3 == 1)
                       ? (int64_t{1} << (kScale - 1))
                       : -(int64_t{1} << (kScale - 1)));
        const int64_t bias_raw =
            ((index + ciphertext) & 1) ? (int64_t{1} << (kScale - 2))
                                       : -(int64_t{1} << (kScale - 2));
        const int64_t product =
            ((x_raw * sigma_raw) >> kScale) * weight_raw;
        const int64_t result_raw = (product >> kScale) + bias_raw;
        expected[ciphertext][index] =
            static_cast<uint64_t>(result_raw) & mask;
      }
    }
  }

  // Continue through a real encrypted round trip. The ciphertext contains
  // only the same public deterministic fixture; the test reveals error
  // metrics, never a model or input intermediate.
  vector<Ciphertext> ciphertexts(kCiphertexts);
  vector<Ciphertext> sigma_ciphertexts;
  if (party == sci::BOB) {
    for (int ciphertext = 0; ciphertext < kCiphertexts; ++ciphertext)
      he.encryptor->encrypt(imported[ciphertext], ciphertexts[ciphertext]);
    send_encrypted_vector(&io, ciphertexts);
    if (linear_scale_cycle == 2) {
      sigma_ciphertexts.resize(kCiphertexts);
      for (int ciphertext = 0; ciphertext < kCiphertexts; ++ciphertext)
        he.encryptor->encrypt(sigma_imported[ciphertext],
                              sigma_ciphertexts[ciphertext]);
      send_encrypted_vector(&io, sigma_ciphertexts);
      if (!device_backend) {
        vector<Ciphertext> public_inputs(2 * kCiphertexts);
        recv_encrypted_vector(he.context, &io, public_inputs);
        double max_abs_x = 0.0;
        double max_abs_sigma = 0.0;
        double max_abs_imag_x = 0.0;
        double max_abs_imag_sigma = 0.0;
        for (int ciphertext = 0; ciphertext < kCiphertexts; ++ciphertext) {
          Plaintext x_plaintext;
          Plaintext sigma_plaintext;
          he.decryptor->decrypt(public_inputs[ciphertext], x_plaintext);
          he.decryptor->decrypt(public_inputs[kCiphertexts + ciphertext],
                                sigma_plaintext);
          vector<complex<double>> decoded_x;
          vector<complex<double>> decoded_sigma;
          he.encoder->decode(x_plaintext, decoded_x);
          he.encoder->decode(sigma_plaintext, decoded_sigma);
          for (size_t index = 0; index < he.num_slots; ++index) {
            const int shift = 4 + (ciphertext % 8);
            const int64_t x_raw =
                static_cast<int64_t>(
                    (static_cast<int>(index + ciphertext) % 31) - 15) *
                (int64_t{1} << shift);
            max_abs_x = max(
                max_abs_x,
                abs(decoded_x[index].real() -
                    ldexp(static_cast<double>(x_raw), -kScale)));
            max_abs_sigma = max(
                max_abs_sigma,
                abs(decoded_sigma[index].real() -
                    ldexp(static_cast<double>(signed_ring(
                              sigma_expected[ciphertext][index])),
                          -kScale)));
            max_abs_imag_x =
                max(max_abs_imag_x, abs(decoded_x[index].imag()));
            max_abs_imag_sigma =
                max(max_abs_imag_sigma, abs(decoded_sigma[index].imag()));
          }
        }
        cout << "B_IMPORTED_CIPHERTEXT_INPUT_DECODE status="
             << (max(max_abs_x, max_abs_sigma) <= 1e-3 ? "PASS" : "FAIL")
             << " max_abs_x=" << max_abs_x
             << " max_abs_sigma=" << max_abs_sigma
             << " max_abs_imag_x=" << max_abs_imag_x
             << " max_abs_imag_sigma=" << max_abs_imag_sigma << endl;
      }
      vector<Ciphertext> public_result(kCiphertexts);
      recv_encrypted_vector(he.context, &io, public_result);
      double max_abs = 0.0;
      long double squared_error = 0.0L;
      double max_value = -numeric_limits<double>::infinity();
      double min_value = numeric_limits<double>::infinity();
      uint64_t compared = 0;
      for (int ciphertext = 0; ciphertext < kCiphertexts; ++ciphertext) {
        Plaintext plaintext;
        he.decryptor->decrypt(public_result[ciphertext], plaintext);
        vector<double> decoded;
        he.encoder->decode(plaintext, decoded);
        for (size_t index = 0; index < he.num_slots; ++index) {
          const double reference =
              ldexp(static_cast<double>(signed_ring(
                        expected[ciphertext][index])),
                    -kScale);
          const double error = decoded[index] - reference;
          max_abs = max(max_abs, abs(error));
          squared_error += static_cast<long double>(error) * error;
          min_value = min(min_value, decoded[index]);
          max_value = max(max_value, decoded[index]);
          ++compared;
        }
      }
      cout << "B_LAYERNORM_AFFINE_DIRECT_DECODE status="
           << (max_abs <= 1e-3 ? "PASS" : "FAIL")
           << " max_abs=" << max_abs
           << " rmse=" << sqrt(static_cast<double>(squared_error / compared))
           << " range=[" << min_value << ',' << max_value << ']'
           << " elements=" << compared << endl;
    }
    vector<Ciphertext> masked(kCiphertexts);
    recv_encrypted_vector(he.context, &io, masked);
    vector<Plaintext> local_plaintext(kCiphertexts);
    for (int ciphertext = 0; ciphertext < kCiphertexts; ++ciphertext)
      he.decryptor->decrypt(masked[ciphertext], local_plaintext[ciphertext]);
    const auto output = adapter.ckks_to_mpc(
        local_plaintext, kDegree, he.get_Q(local_plaintext[0]), kBits,
        kScale, &he, false);
    vector<vector<uint64_t>> alice_output(kCiphertexts);
    for (int ciphertext = 0; ciphertext < kCiphertexts; ++ciphertext) {
      alice_output[ciphertext].resize(output[ciphertext].size());
      io.recv_data(alice_output[ciphertext].data(),
                   alice_output[ciphertext].size() * sizeof(uint64_t));
    }
    uint64_t max_raw = 0;
    uint64_t bad = 0;
    for (int ciphertext = 0; ciphertext < kCiphertexts; ++ciphertext) {
      for (size_t index = 0; index < output[ciphertext].size(); ++index) {
        const uint64_t actual =
            (alice_output[ciphertext][index] + output[ciphertext][index]) &
            mask;
        const int64_t error =
            signed_ring(actual - expected[ciphertext][index]);
        const uint64_t absolute =
            static_cast<uint64_t>(error < 0 ? -error : error);
        max_raw = max(max_raw, absolute);
        bad += absolute > 8;
      }
    }
    cout << "B_MPC_CKKS_MPC_SELFTEST status="
         << (bad == 0 ? "PASS" : "FAIL")
         << " max_abs_error_raw=" << max_raw << " bad_values=" << bad
         << endl;
    cout << "B_CONVERSION_SELFTEST status="
         << (bad == 0 ? "PASS" : "FAIL")
         << " fixture=public_deterministic encrypted_round_trip=1" << endl;
    return bad == 0 ? 0 : 1;
  }

  // Complete the exact production loop: BOB encrypts its imported share;
  // ALICE adds its share, moves both to the final level, then applies the
  // current-RNS polynomial mask before both parties decode to MPC.
  recv_encrypted_vector(he.context, &io, ciphertexts);
  if (linear_scale_cycle == 2) {
    sigma_ciphertexts.resize(kCiphertexts);
    recv_encrypted_vector(he.context, &io, sigma_ciphertexts);
  }
  if (device_backend) {
    for (int ciphertext = 0; ciphertext < kCiphertexts; ++ciphertext) {
      UnifiedCiphertext device_ciphertext(std::move(ciphertexts[ciphertext]));
      UnifiedPlaintext device_plaintext(std::move(imported[ciphertext]));
      device_ciphertext.to_device(*he.context, *he.dcontext);
      device_plaintext.to_device(*he.context, *he.dcontext);
      he.devaluator->add_plain_inplace(device_ciphertext, device_plaintext);
      if (linear_scale_cycle == 2) {
        UnifiedCiphertext device_sigma(
            std::move(sigma_ciphertexts[ciphertext]));
        UnifiedPlaintext device_sigma_plaintext(
            std::move(sigma_imported[ciphertext]));
        device_sigma.to_device(*he.context, *he.dcontext);
        device_sigma_plaintext.to_device(*he.context, *he.dcontext);
        he.devaluator->add_plain_inplace(device_sigma,
                                         device_sigma_plaintext);
        UnifiedCiphertext device_conjugate(LOCATION::DEVICE);
        he.devaluator->complex_conjugate(device_ciphertext,
                                         *he.dgal_keys,
                                         device_conjugate);
        he.devaluator->add_inplace(device_ciphertext, device_conjugate);
        UnifiedCiphertext device_sigma_conjugate(LOCATION::DEVICE);
        he.devaluator->complex_conjugate(device_sigma,
                                         *he.dgal_keys,
                                         device_sigma_conjugate);
        he.devaluator->add_inplace(device_sigma, device_sigma_conjugate);
        const auto chain_index = device_ciphertext.dcipher().chain_index();
        vector<double> weights(he.num_slots);
        vector<double> biases(he.num_slots);
        for (size_t index = 0; index < he.num_slots; ++index) {
          const double weight =
              (index + ciphertext) % 3 == 0
                  ? 1.0
                  : ((index + ciphertext) % 3 == 1 ? 0.5 : -0.5);
          weights[index] = 0.25 * weight;
          biases[index] = (index + ciphertext) & 1 ? 0.25 : -0.25;
        }
        UnifiedPlaintext weight(LOCATION::DEVICE);
        UnifiedPlaintext bias(LOCATION::DEVICE);
        he.dencoder->encode(*he.dcontext, weights,
                            affine_weight_scale, weight,
                            chain_index);
        he.dencoder->encode(*he.dcontext, biases,
                            affine_bias_scale, bias,
                            chain_index);
        he.devaluator->multiply_inplace(device_ciphertext, device_sigma);
        he.devaluator->relinearize_inplace(device_ciphertext,
                                            *he.drelin_keys);
        he.devaluator->multiply_plain_inplace(device_ciphertext, weight);
        he.devaluator->add_plain_inplace(device_ciphertext, bias);
        he.devaluator->rescale_to_next_inplace(device_ciphertext);
      }
      device_ciphertext.to_host(*he.context, *he.dcontext);
      ciphertexts[ciphertext] = std::move(device_ciphertext.hcipher());
    }
  } else {
    vector<Ciphertext> public_imported_inputs;
    if (linear_scale_cycle == 2)
      public_imported_inputs.resize(2 * kCiphertexts);
    for (int ciphertext = 0; ciphertext < kCiphertexts; ++ciphertext) {
      he.evaluator->add_plain_inplace(ciphertexts[ciphertext],
                                      imported[ciphertext]);
      if (linear_scale_cycle == 2) {
        he.evaluator->add_plain_inplace(sigma_ciphertexts[ciphertext],
                                        sigma_imported[ciphertext]);
        public_imported_inputs[ciphertext] = ciphertexts[ciphertext];
        public_imported_inputs[kCiphertexts + ciphertext] =
            sigma_ciphertexts[ciphertext];
        Ciphertext conjugate;
        he.evaluator->complex_conjugate(ciphertexts[ciphertext],
                                        *he.gal_keys, conjugate);
        he.evaluator->add_inplace(ciphertexts[ciphertext], conjugate);
        Ciphertext sigma_conjugate;
        he.evaluator->complex_conjugate(sigma_ciphertexts[ciphertext],
                                        *he.gal_keys, sigma_conjugate);
        he.evaluator->add_inplace(sigma_ciphertexts[ciphertext],
                                  sigma_conjugate);
        vector<double> weights(he.num_slots);
        vector<double> biases(he.num_slots);
        for (size_t index = 0; index < he.num_slots; ++index) {
          const double weight =
              (index + ciphertext) % 3 == 0
                  ? 1.0
                  : ((index + ciphertext) % 3 == 1 ? 0.5 : -0.5);
          weights[index] = 0.25 * weight;
          biases[index] = (index + ciphertext) & 1 ? 0.25 : -0.25;
        }
        Plaintext weight;
        Plaintext bias;
        he.encoder->encode(weights, ciphertexts[ciphertext].parms_id(),
                           affine_weight_scale, weight);
        he.encoder->encode(biases, ciphertexts[ciphertext].parms_id(),
                           affine_bias_scale, bias);
        he.evaluator->multiply_inplace(ciphertexts[ciphertext],
                                       sigma_ciphertexts[ciphertext]);
        he.evaluator->relinearize_inplace(ciphertexts[ciphertext],
                                          *he.relin_keys);
        he.evaluator->multiply_plain_inplace(ciphertexts[ciphertext], weight);
        he.evaluator->add_plain_inplace(ciphertexts[ciphertext], bias);
        he.evaluator->rescale_to_next_inplace(ciphertexts[ciphertext]);
      }
    }
    if (linear_scale_cycle == 2)
      send_encrypted_vector(&io, public_imported_inputs);
  }
  if (linear_scale_cycle == 1) {
    const vector<double> public_one(he.num_slots, 1.0);
    for (int ciphertext = 0; ciphertext < kCiphertexts; ++ciphertext) {
      Plaintext multiplier;
      he.encoder->encode(public_one, ciphertexts[ciphertext].parms_id(),
                         ldexp(1.0, kCkksScale), multiplier);
      he.evaluator->multiply_plain_inplace(ciphertexts[ciphertext], multiplier);
    }
    adapter.prepare_ckks_to_mpc_inplace(ciphertexts, kConversionScale, &he);
  } else if (linear_scale_cycle == 0) {
    for (int ciphertext = 0; ciphertext < kCiphertexts; ++ciphertext) {
      he.evaluator->mod_switch_to_inplace(ciphertexts[ciphertext],
                                          he.context->last_parms_id());
    }
  }
  if (linear_scale_cycle == 2)
    adapter.prepare_ckks_to_mpc_inplace(ciphertexts, kConversionScale, &he);
  if (linear_scale_cycle == 2)
    send_encrypted_vector(&io, ciphertexts);
  vector<Plaintext> local_plaintext;
  adapter.mask_ckks_to_mpc_inplace(ciphertexts, local_plaintext, &he);
  send_encrypted_vector(&io, ciphertexts);
  const auto output = adapter.ckks_to_mpc(
      local_plaintext, kDegree, he.get_Q(local_plaintext[0]), kBits, kScale,
      &he, false);
  for (const auto &part : output)
    io.send_data(part.data(), part.size() * sizeof(uint64_t));
  cout << "B_CONVERSION_BACKEND backend="
       << (device_backend ? "phantom_device" : "seal_host")
       << " linear_scale_cycle=" << linear_scale_cycle << endl;
  if (linear_scale_cycle == 2)
    cout << "B_LAYERNORM_AFFINE_SELFTEST fixture=public_deterministic"
         << " backend=" << (device_backend ? "phantom_device" : "seal_host")
         << endl;
  return 0;
}
