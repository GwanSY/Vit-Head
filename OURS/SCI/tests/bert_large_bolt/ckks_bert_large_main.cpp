/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <fstream>
#include <iostream>
#include <thread>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <regex>
#include "ckks_bert.h"

using namespace sci;
using namespace std;
using namespace seal;

#define MAX_THREADS 32
// #define SCI_HE

int party = 0;
int port = 8000;
int num_threads = 1;
string address = "127.0.0.1";
string model_path = "checkpoints/official-blb-mrpc/weights_txt";
string backend_mode = "cpu";
int bitlength = 37;
int32_t dim = 768;
int32_t array_size = 128;
int32_t bw_x = 37;
int32_t bw_y = 37;
int32_t s_x = 12;
int32_t s_y = 12;
int32_t input_size = dim*array_size;
int32_t poly_modulus_degree=8192;
int manifest_verified = 0;
int requested_layers = 1;

// The implementation is present, but the runtime still fails closed unless
// an authenticated oracle-share manifest is present next to the BLB-local
// checkpoint.  This prevents the old synthetic harness from being promoted
// by a command-line flag or by fabricated log lines.
constexpr bool kFormalLayer0InputReady = true;
constexpr bool kFormalThreeOracleReady = true;

// A diagnostic run may continue after a scalar-only oracle gate fails so one
// expensive two-party execution reports every downstream block.  It never
// changes a tensor, skips an operator, or turns the final result into PASS.
// The default remains fail-closed.
bool oracle_diagnostic_continue = false;
bool oracle_gate_failed = false;
string first_oracle_failure;

bool signed_ = true;

template<typename T> void write_to_file2(std::string filename, T data)
{
    // std::ofstream file(filename, std::ios::app);  // 使用std::ios::app模式打开文件，代表续写
    std::ofstream file(filename);
    // printf("data size:%ld\n",data.size());
    // fflush(stdout);
    for (size_t i = 0; i < data.size(); ++i)
    {
        file << data[i];
        if (i != data.size() - 1)  // 不在最后一个元素后添加逗号
        {
            file << ",";
        }
    }
    file.close();
}

template<typename T> std::vector<T> read_from_file(std::string filename)
{
    std::vector<T> data;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line, ','))
    {
        std::istringstream iss(line);
        T value;
        iss >> value;
        data.push_back(value);
    }
    file.close();
    return data;
}

static vector<uint64_t> read_signed_tensor(const string &filename,
                                           size_t expected_size) {
  ifstream file(filename);
  if (!file.good()) {
    throw runtime_error("required real layer-0 input is missing: " + filename);
  }
  vector<uint64_t> values;
  string line;
  while (getline(file, line)) {
    istringstream row(line);
    string cell;
    while (getline(row, cell, ',')) {
      if (cell.empty()) continue;
      size_t consumed = 0;
      long long signed_value = stoll(cell, &consumed);
      if (consumed != cell.size()) {
        throw runtime_error("invalid integer in real layer-0 input: " + filename);
      }
      values.push_back(static_cast<uint64_t>(signed_value));
    }
  }
  if (values.size() != expected_size) {
    throw runtime_error("real layer-0 input shape mismatch: " + filename);
  }
  return values;
}

static vector<int64_t> read_public_layer0_mask(const string &filename,
                                               size_t expected_rows) {
  ifstream file(filename);
  if (!file.good()) {
    throw runtime_error("required attention mask is missing: " + filename);
  }
  vector<int64_t> mask;
  string line;
  while (getline(file, line)) {
    if (line.empty()) continue;
    istringstream row(line);
    long long mask_value = 0;
    char trailing = 0;
    if (!(row >> mask_value) || (row >> trailing)) {
      throw runtime_error("invalid attention mask integer: " + filename);
    }
    // BERT's public additive key mask is zero for valid keys and negative for
    // padding.  Keep its scaled value inside the signed 37-bit MPC ring.
    const int value_bits = bitlength - s_x - 1;
    const int64_t min_value = -(int64_t(1) << value_bits);
    if (mask_value > 0 || mask_value < min_value) {
      throw runtime_error("attention mask is outside the supported public range: " +
                          filename);
    }
    mask.push_back(static_cast<int64_t>(mask_value));
  }
  if (mask.size() != expected_rows) {
    throw runtime_error("attention mask shape mismatch: " + filename);
  }
  if (none_of(mask.begin(), mask.end(), [](int64_t value) {
        return value == 0;
      })) {
    throw runtime_error("attention mask has no valid key: " + filename);
  }
  return mask;
}

struct OracleStageReference {
  vector<uint64_t> local_share;
  size_t size = 0;
  int ell = 0;
  int scale = 0;
  double a_b_mse = 0.0;
};

static constexpr const char *kOracleStages[] = {
    "fgf1_ln_attention", "fgf2_softmax", "fgf3_softmax_v_ln",
    "fgf4_ln_ffn_gelu", "fgf5_ffn2_ln"};

static bool parse_json_number_after(const string &text, size_t start,
                                    const string &key, double &value) {
  size_t key_pos = text.find("\"" + key + "\"", start);
  if (key_pos == string::npos) return false;
  size_t colon = text.find(':', key_pos);
  if (colon == string::npos) return false;
  size_t end = colon + 1;
  while (end < text.size() && isspace(static_cast<unsigned char>(text[end]))) ++end;
  size_t consumed = 0;
  try {
    value = stod(text.substr(end), &consumed);
  } catch (...) {
    return false;
  }
  return consumed != 0;
}

static OracleStageReference read_oracle_share(const string &filename,
                                              size_t expected_size) {
  ifstream file(filename, ios::binary);
  if (!file.good()) {
    throw runtime_error("required oracle share is missing: " + filename);
  }
  // The Python preprocessor writes the identifier plus one terminating NUL.
  // Do not put an explicit \0 in this literal: C++ would then append a second
  // implicit NUL and shift the binary header by one byte.
  const char magic[] = "BLB_ORACLE_SHARE_V1";
  char got[sizeof(magic)];
  file.read(got, sizeof(got));
  if (file.gcount() != static_cast<streamsize>(sizeof(got)) ||
      memcmp(got, magic, sizeof(magic)) != 0) {
    throw runtime_error("oracle share header mismatch: " + filename);
  }
  uint64_t size = 0;
  int32_t ell = 0;
  int32_t scale = 0;
  file.read(reinterpret_cast<char *>(&size), sizeof(size));
  file.read(reinterpret_cast<char *>(&ell), sizeof(ell));
  file.read(reinterpret_cast<char *>(&scale), sizeof(scale));
  if (!file.good() || size != expected_size || ell != bitlength ||
      scale != s_x) {
    throw runtime_error("oracle share shape/scale mismatch: " + filename +
                        " expected_size=" + to_string(expected_size) +
                        " got_size=" + to_string(size) +
                        " expected_ell=" + to_string(bitlength) +
                        " got_ell=" + to_string(ell) +
                        " expected_scale=" + to_string(s_x) +
                        " got_scale=" + to_string(scale));
  }
  OracleStageReference result;
  result.size = static_cast<size_t>(size);
  result.ell = ell;
  result.scale = scale;
  result.local_share.resize(result.size);
  file.read(reinterpret_cast<char *>(result.local_share.data()),
            static_cast<streamsize>(result.size * sizeof(uint64_t)));
  if (!file.good()) {
    throw runtime_error("oracle share payload is truncated: " + filename);
  }
  char trailing = 0;
  if (file.read(&trailing, 1)) {
    throw runtime_error("oracle share has trailing bytes: " + filename);
  }
  return result;
}

static map<string, OracleStageReference> load_oracle_references(
    const string &model_dir, int owner_party) {
  const char *verified = getenv("BLB_ORACLE_MANIFEST_VERIFIED");
  if (verified == nullptr || string(verified) != "1") {
    throw runtime_error("three-oracle manifest_not_verified");
  }
  const string oracle_dir =
      getenv("BLB_ORACLE_DIR") ? getenv("BLB_ORACLE_DIR")
                               : model_dir + "/oracle_shares";
  ifstream manifest_file(oracle_dir + "/manifest.json");
  if (!manifest_file.good()) {
    throw runtime_error("three-oracle manifest is missing: " + oracle_dir);
  }
  string manifest((istreambuf_iterator<char>(manifest_file)),
                  istreambuf_iterator<char>());
  if (manifest.find("\"blb.bert.three-oracle.v3\"") == string::npos ||
      manifest.find(
          "\"bert.py.post_norm.layer0.qk_s12_v_s11.public_key_mask.v3\"") ==
          string::npos) {
    throw runtime_error(
        "three-oracle manifest predates the split QK/V scale contract");
  }
  map<string, OracleStageReference> references;
  for (const char *stage_name : kOracleStages) {
    const string stage(stage_name);
    size_t stage_pos = manifest.find("\"" + stage + "\"");
    if (stage_pos == string::npos) {
      throw runtime_error("three-oracle stage is missing from manifest: " + stage);
    }
    double size_number = 0.0;
    double a_b_mse = 0.0;
    if (!parse_json_number_after(manifest, stage_pos, "size", size_number) ||
        !parse_json_number_after(manifest, stage_pos, "A_B_MSE", a_b_mse) ||
        size_number <= 0.0 || !isfinite(a_b_mse) || a_b_mse < 0.0) {
      throw runtime_error("three-oracle manifest metadata is invalid: " + stage);
    }
    const size_t expected_size = static_cast<size_t>(size_number);
    const string role = owner_party == sci::ALICE ? "alice" : "bob";
    OracleStageReference reference = read_oracle_share(
        oracle_dir + "/" + stage + "." + role + ".bin", expected_size);
    reference.a_b_mse = a_b_mse;
    references.emplace(stage, std::move(reference));
  }
  return references;
}

struct SecureStageErrorMetrics {
  double max_abs = 0.0;
  double mae = 0.0;
  double mse = numeric_limits<double>::quiet_NaN();
  uint64_t max_abs_raw = 0;
  uint64_t max_square_safe_raw = 0;
  bool square_safe = false;
  size_t opened_scalars = 0;
};

static SecureStageErrorMetrics secure_stage_error_metrics(
    Bert_ckks &bert_ckks, const FixArray &secure,
    const OracleStageReference &reference) {
  if (secure.size != static_cast<int>(reference.size) ||
      secure.ell != reference.ell || secure.s != reference.scale) {
    throw runtime_error("secure/oracle stage shape or scale mismatch");
  }
  if (!secure.signed_ || secure.ell <= 1 || secure.ell >= 63 ||
      secure.s < 0) {
    throw runtime_error("secure/oracle stage has unsupported fixed-point metadata");
  }
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  FixArray expected(bert_ckks.party, secure.size, secure.signed_, secure.ell,
                   secure.s);
  copy(reference.local_share.begin(), reference.local_share.end(), expected.data);
  FixArray difference = fpmath->fix->sub(secure, expected);

  // Extend by one bit before negation so abs(-2^(ell-1)) is representable.
  // All operations below remain secret-shared; no difference element is ever
  // reconstructed.
  BoolArray difference_negative = fpmath->fix->MSB(difference);
  FixArray difference_extended = fpmath->fix->extend(
      difference, difference.ell + 1, difference_negative.data);
  FixArray difference_negated = fpmath->fix->mul(
      difference_extended, uint64_t(-1), difference_extended.ell,
      difference_negative.data);
  FixArray absolute_difference = fpmath->fix->if_else(
      difference_negative, difference_negated, difference_extended);
  // The mux result is mathematically non-negative. Marking it unsigned changes
  // only the public type metadata and lets the 63-bit accumulator use its full
  // [0, 2^63-1] range.
  absolute_difference.signed_ = false;

  vector<FixArray> max_input;
  max_input.emplace_back(absolute_difference);
  FixArray secret_max = fpmath->fix->max(max_input);
  if (secret_max.size != 1) {
    throw runtime_error("secure max-absolute reduction did not produce a scalar");
  }
  FixArray opened_max = fpmath->fix->output(sci::PUBLIC, secret_max);
  const uint64_t max_abs_raw = opened_max.data[0];

  // The aggregate ring is unsigned 63-bit because FixOp::mul requires ell <
  // 64. Prove up front that the complete absolute-error tensor fits one
  // scalar. We never fall back to small blocks: doing so could degenerate into
  // opening one scalar per tensor element.
  constexpr int kAccumulatorEll = 63;
  constexpr uint64_t kAccumulatorMax =
      (uint64_t(1) << kAccumulatorEll) - 1;
  const uint64_t element_count = static_cast<uint64_t>(secure.size);
  const uint64_t theoretical_max_abs = uint64_t(1) << (secure.ell - 1);
  if (theoretical_max_abs > kAccumulatorMax / element_count) {
    throw runtime_error(
        "secure oracle MAE tensor cannot fit one 63-bit aggregate scalar");
  }

  // Zero-extension changes the share modulus from 2^(ell+1) to 2^63 without
  // revealing the absolute-error vector. Its full-tensor sum is opened as one
  // aggregate scalar for MAE.
  FixArray absolute_accumulator =
      fpmath->fix->extend(absolute_difference, kAccumulatorEll);
  uint64_t absolute_sum_share = 0;
  for (int i = 0; i < absolute_accumulator.size; ++i) {
    absolute_sum_share =
        (absolute_sum_share + absolute_accumulator.data[i]) & kAccumulatorMax;
  }
  FixArray secret_absolute_sum(bert_ckks.party, 1, false, kAccumulatorEll,
                               secure.s);
  secret_absolute_sum.data[0] = absolute_sum_share;
  FixArray opened_absolute_sum =
      fpmath->fix->output(sci::PUBLIC, secret_absolute_sum);
  const uint64_t actual_absolute_bound = max_abs_raw * element_count;
  if (opened_absolute_sum.data[0] > actual_absolute_bound) {
    throw runtime_error(
        "secure oracle MAE scalar exceeds its proved full-tensor bound");
  }

  // Derive floor(sqrt((2^63-1)/N)) without a floating-point security
  // assumption. MSE is computed only when the public max scalar proves that
  // the entire square sum fits one 63-bit scalar. Otherwise return MAX/MAE and
  // let the caller print square_safe=0 before failing closed.
  const uint64_t per_square_budget = kAccumulatorMax / element_count;
  uint64_t max_square_safe_raw =
      static_cast<uint64_t>(sqrt(static_cast<long double>(per_square_budget)));
  while (max_square_safe_raw + 1 <=
         per_square_budget / (max_square_safe_raw + 1)) {
    ++max_square_safe_raw;
  }
  while (max_square_safe_raw != 0 &&
         max_square_safe_raw > per_square_budget / max_square_safe_raw) {
    --max_square_safe_raw;
  }
  const bool square_safe = max_abs_raw <= max_square_safe_raw;

  const long double count = static_cast<long double>(secure.size);
  const long double value_scale = std::ldexp(1.0L, secure.s);
  SecureStageErrorMetrics metrics;
  metrics.max_abs = static_cast<double>(
      static_cast<long double>(max_abs_raw) / value_scale);
  metrics.mae = static_cast<double>(
      static_cast<long double>(opened_absolute_sum.data[0]) /
      (value_scale * count));
  metrics.max_abs_raw = max_abs_raw;
  metrics.max_square_safe_raw = max_square_safe_raw;
  metrics.square_safe = square_safe;
  metrics.opened_scalars = 2;

  if (square_safe) {
    FixArray square = fpmath->fix->mul(
        absolute_difference, absolute_difference, kAccumulatorEll);
    uint64_t square_sum_share = 0;
    for (int i = 0; i < square.size; ++i) {
      square_sum_share =
          (square_sum_share + square.data[i]) & kAccumulatorMax;
    }
    FixArray secret_square_sum(bert_ckks.party, 1, false, kAccumulatorEll,
                               2 * secure.s);
    secret_square_sum.data[0] = square_sum_share;
    FixArray opened_square_sum =
        fpmath->fix->output(sci::PUBLIC, secret_square_sum);
    const uint64_t max_square_raw = max_abs_raw * max_abs_raw;
    const uint64_t actual_square_bound = max_square_raw * element_count;
    if (opened_square_sum.data[0] > actual_square_bound) {
      throw runtime_error(
          "secure oracle MSE scalar exceeds its proved full-tensor bound");
    }
    const long double square_scale = std::ldexp(1.0L, 2 * secure.s);
    metrics.mse = static_cast<double>(
        static_cast<long double>(opened_square_sum.data[0]) /
        (square_scale * count));
    metrics.opened_scalars = 3;
  }

  if (!isfinite(metrics.max_abs) || metrics.max_abs < 0.0 ||
      !isfinite(metrics.mae) || metrics.mae < 0.0 ||
      (metrics.square_safe &&
       (!isfinite(metrics.mse) || metrics.mse < 0.0))) {
    throw runtime_error("secure oracle aggregate scalar is invalid");
  }
  return metrics;
}

static FixArray make_layernorm_parameter_share(
    Bert_ckks &bert_ckks, FPMath *fpmath, const vector<vector<uint64_t>> &params,
    size_t rows, size_t cols, int ell, int scale) {
  if (params.size() != 1 || params[0].size() != cols) {
    // On BOB the model vector is intentionally empty; the owner input below
    // still creates a zero local share. Alice must have the complete tensor.
    if (bert_ckks.party == sci::ALICE) {
      throw runtime_error("real LayerNorm parameter is missing or malformed");
    }
  }
  vector<uint64_t> expanded(rows * cols, 0);
  if (bert_ckks.party == sci::ALICE) {
    if (params.size() != 1 || params[0].size() != cols) {
      throw runtime_error("real LayerNorm parameter is missing or malformed");
    }
    for (size_t row = 0; row < rows; ++row) {
      copy(params[0].begin(), params[0].end(), expanded.begin() + row * cols);
    }
  }
  return fpmath->fix->input(sci::ALICE, static_cast<int>(expanded.size()),
                            expanded.data(), true, ell, scale);
}

void ifft_weight_test(int party, Bert_ckks bert_ckks){
  vector<HE_CKKS::complex_t> x(8192),y(8192);
  vector<HE_CKKS::scalar_t> y_real(8192),y_imag(8192);
  // bert_ckks.linear.set_HE(8192,vector<int>{31,20,31},20);
  for(int i=0;i<8192;i++){
    if(i%100==0){
      cout << "i:" << i << endl;
    }
    for(int j=0;j<8192;j++){
      if(j==i){
        x[j] = HE_CKKS::complex_t(1*(1ULL<<20),0);
      }
      else{
        x[j] = HE_CKKS::complex_t(0,0);
      }
    }
    y = bert_ckks.linear.he1->ifft(x, 8192, true);
    for(int j=0;j<8192;j++){  
      y_real[j] = y[j].real();
      y_imag[j] = y[j].imag();
    }
    write_to_file2("outputs/legacy_debug/y_real.txt", y_real);
    write_to_file2("outputs/legacy_debug/y_imag.txt", y_imag);
  }

  
  // write_to_file2("outputs/legacy_debug/y_int.txt", y_int);
}

void ifft_test(int party, Bert_ckks bert_ckks){
  vector<HE_CKKS::complex_t> x(8192),y(8192);
  for(int j=0;j<8192;j++){
    if(j==1){
      x[j] = HE_CKKS::complex_t(1*(1ULL<<20),0);
    }
    else{
      x[j] = HE_CKKS::complex_t(0,0);
    }
    // x[j] = HE_CKKS::complex_t(rand()%(1ULL<<32),rand()%(1ULL<<32));
  }
  y = bert_ckks.linear.he1->ifft(x, 8192, true);
  write_to_file2("outputs/legacy_debug/y.txt", y);
  y = bert_ckks.linear.he1->ifft(x, 8192, true);
  write_to_file2("outputs/legacy_debug/y_matrix.txt", y);
}

void field_to_ring_test(int party, Bert_ckks bert_ckks){
  vector<HE_CKKS::scalar_t> x_field(1),x_ring(1);
  if(party == sci::ALICE){
    x_field[0]=3;
  }
  else{
    x_field[0]=11;
  }
  cout << "init ok" << endl;
  uint64_t Q = 13;
  int size = 1;
  int bitlength = 4;
  int s_in = 2;
  int s_out = 2;
  bert_ckks.nonlinear.field_to_ring(
      1,
      x_field.data(),
      Q,
      x_ring.data(),
      size,
      bitlength,
      s_in,
      s_out
  );
  cout << "x_ring[0]:" << x_ring[0] << endl;
}

void ring_to_field_test(int party, Bert_ckks bert_ckks){
  vector<HE_CKKS::scalar_t> x_field(1),x_ring(1);
  if(party == sci::ALICE){
    x_ring[0]=4;
  }
  else{
    x_ring[0]=6;
  }
  cout << "init ok" << endl;
  uint64_t Q = 1000000000339;
  int size = 1;
  int bitlength = 3;
  int s_in = 1;
  int s_out = 2;
  bert_ckks.nonlinear.ring_to_field(
      1,
      x_ring.data(),
      Q,
      x_field.data(),
      size,
      bitlength,
      s_in,
      s_out
  );
  cout << "x_field[0]:" << x_field[0] << endl;
}

void mpc_to_ckks_test(int party, Bert_ckks bert_ckks){
  vector<vector<HE_CKKS::scalar_t>> x_share(1);
  if(party == sci::ALICE){
    for(int i=0;i<bert_ckks.linear.he1->num_slots;i++){
      x_share[0].push_back(4*(1<<bert_ckks.linear.he1->fft_scale));
    }
    x_share[0][0]=2*(1<<bert_ckks.linear.he1->fft_scale);
  }
  else{
    for(int i=0;i<bert_ckks.linear.he1->num_slots;i++){
      x_share[0].push_back(0);
    }
  }
  vector<Plaintext> y_share;
  uint64_t Q = bert_ckks.linear.he1->get_Q();
  for(int i=0;i<x_share.size();i++){
    y_share = bert_ckks.mpc_to_ckks(x_share, bert_ckks.linear.he1->num_slots, Q, 40, bert_ckks.linear.he1->fft_scale, bert_ckks.linear.he1->he_scale, bert_ckks.linear.he1);
  }
  x_share = bert_ckks.ckks_to_mpc(y_share, bert_ckks.linear.he1->poly_modulus_degree, Q, 40, bert_ckks.linear.he1->he_scale, bert_ckks.linear.he1);
  cout << "x_share[0][0]:" << x_share[0][0] << endl;
  cout << "comm:" << bert_ckks.get_comm() << endl;
  cout << "round:" << bert_ckks.get_round() << endl;
}

// void square_test(int party, Bert_ckks bert_ckks){
//   FixArray x(party, 4096, false, 41, 20);
//   FixArray y;
//   y = bert_ckks.element_wise_square(4096, x);
//   cout << "y.data[0]:" << y.data[0] << endl;
//   cout << "comm:" << bert_ckks.get_comm() << endl;
//   cout << "round:" << bert_ckks.get_round() << endl;
// }

void layernorm_nonlinear_test(int party, Bert_ckks bert_ckks){
  int bitwidth=40, s=20;
  map<string, FixArray> result,input;
  FixArray x_flat_avg, sigma_flat;
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  auto start = high_resolution_clock::now();
  auto start_init = high_resolution_clock::now();
  // 1. layernorm+attention
  x_flat_avg = fpmath->fix->input(party, bert_ckks.linear.data_lin1_0.image_size*bert_ckks.linear.data_lin1_0.filter_h,1, true, bitwidth, s);
  sigma_flat = fpmath->fix->input(party, bert_ckks.linear.data_lin1_0.image_size, 1, true, bitwidth, s);
  input["x_flat_avg"] = x_flat_avg;
  input["sigma_flat"] = sigma_flat;
  result = bert_ckks.layer_norm_nonlinear(input);
  bert_ckks.other_nonlinear_la += ((high_resolution_clock::now() - start)).count()/1e+9;
  bert_ckks.other_nonlinear_comm += bert_ckks.get_comm();
  bert_ckks.other_nonlinear_round += bert_ckks.get_round();
  cout << "--------layer_norm_nonlinear done--------" << endl;
  cout << "--------linear part 4 done--------" << endl;
  cout << "-----------comm------------" << endl;
  cout << "ct_transfer_comm:" << bert_ckks.ct_transfer_comm << endl;
  cout << "conversion_comm:" << bert_ckks.conversion_comm << endl;
  cout << "other_nonlinear_comm:" << bert_ckks.other_nonlinear_comm << endl;
  cout << "-----------round------------" << endl;
  cout << "ct_transfer_round:" << bert_ckks.ct_transfer_round << endl;
  cout << "conversion_round:" << bert_ckks.conversion_round << endl;
  cout << "other_nonlinear_round:" << bert_ckks.other_nonlinear_round << endl;
  cout << "-----------time------------" << endl;
  auto end = high_resolution_clock::now();
  bert_ckks.total_la = ((end - start_init)).count()/1e+9;
  cout << "total time:" << bert_ckks.total_la << endl;
  cout << "ct_transfer_time:" << bert_ckks.ct_transfer_la << endl;
  cout << "conversion_time:" << bert_ckks.conversion_la << endl;
  cout << "other_nonlinear_time:" << bert_ckks.other_nonlinear_la << endl;
  cout << "he_compute_time:" << bert_ckks.he_compute_la << endl;
  cout << "other time:" << bert_ckks.total_la-bert_ckks.ct_transfer_la-bert_ckks.conversion_la-bert_ckks.other_nonlinear_la-bert_ckks.he_compute_la << endl;

}

void layernorm_test(int party, Bert_ckks bert_ckks){
  int bitwidth=40, s=20;
  vector<uint64_t> x(dim*array_size);
  map<string, FixArray> result,input;
  FixArray x_flat_avg, sigma_flat;
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  x_flat_avg = fpmath->fix->input(party, bert_ckks.linear.data_lin1_0.image_size*bert_ckks.linear.data_lin1_0.filter_h,1, true, bitwidth, s);
  sigma_flat = fpmath->fix->input(party, bert_ckks.linear.data_lin1_0.image_size,1, true, bitwidth, s);
  x_flat_avg.ell = bitwidth;
  x_flat_avg.s = s;
  sigma_flat.ell = bitwidth;
  sigma_flat.s = s;
  input["x_flat_avg"] = x_flat_avg;
  input["sigma_flat"] = sigma_flat;
  result = bert_ckks.layer_norm_nonlinear(input);
  vector<vector<HE_CKKS::scalar_t>> result2;
  cout << "result[x_flat_avg]:" << result["x_flat_avg"].size << endl;
  cout << "result[sigma_flat]:" << result["sigma_flat"].size << endl;
  cout << "result[w]:" << result["w"].size << endl;
  cout << "result[b]:" << result["b"].size << endl;
  cout << "comm:" << bert_ckks.get_comm() << endl;
  cout << "round:" << bert_ckks.get_round() << endl;
  double tmp_comm = bert_ckks.n_comm;
  double tmp_round = bert_ckks.n_rounds;
  result2 = bert_ckks.layer_norm_part2_attention(result);
  cout << "fusion comm:" << bert_ckks.n_comm-tmp_comm << endl;
  cout << "fusion round:" << bert_ckks.n_rounds-tmp_round << endl;
  cout << "-----------comm------------" << endl;
  cout << "ct_transfer_comm:" << bert_ckks.ct_transfer_comm << endl;
  cout << "conversion_comm:" << bert_ckks.conversion_comm << endl;
  cout << "other_nonlinear_comm:" << bert_ckks.other_nonlinear_comm << endl;
  cout << "-----------round------------" << endl;
  cout << "ct_transfer_round:" << bert_ckks.ct_transfer_round << endl;
  cout << "conversion_round:" << bert_ckks.conversion_round << endl;
  cout << "other_nonlinear_round:" << bert_ckks.other_nonlinear_round << endl;
}


void gelu_test(int party, Bert_ckks bert_ckks){
  int bitwidth=40, s=20;
  map<string, FixArray> result,input;
  FixArray x_flat_avg, sigma_flat;
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  x_flat_avg = fpmath->fix->input(party, bert_ckks.linear.data_lin1_0.image_size*bert_ckks.linear.data_lin1_0.filter_h,1, true, bitwidth, s);
  sigma_flat = fpmath->fix->input(party, bert_ckks.linear.data_lin1_0.image_size,1, true, bitwidth, s);
  x_flat_avg.ell = bitwidth;
  x_flat_avg.s = s;
  sigma_flat.ell = bitwidth;
  sigma_flat.s = s;
  input["x_flat_avg"] = x_flat_avg;
  input["sigma_flat"] = sigma_flat;
  result = bert_ckks.layer_norm_nonlinear(input);
  cout << "layer_norm_part1 comm:" << bert_ckks.get_comm() << endl;
  cout << "layer_norm_part1 round:" << bert_ckks.get_round() << endl;

  double tmp_comm = bert_ckks.n_comm;
  double tmp_round = bert_ckks.n_rounds;
  map<string, FixArray> result2;
  result2 = bert_ckks.ln_part2_linear3_gelu_part1(result);
  FixArray result3;
  result3 = bert_ckks.gelu_nonlinear(result2);
  cout << "fusion comm:" << bert_ckks.n_comm-tmp_comm << endl;
  cout << "fusion round:" << bert_ckks.n_rounds-tmp_round << endl;
  cout << "-----------comm------------" << endl;
  cout << "ct_transfer_comm:" << bert_ckks.ct_transfer_comm << endl;
  cout << "conversion_comm:" << bert_ckks.conversion_comm << endl;
  cout << "other_nonlinear_comm:" << bert_ckks.other_nonlinear_comm << endl;
  cout << "-----------round------------" << endl;
  cout << "ct_transfer_round:" << bert_ckks.ct_transfer_round << endl;
  cout << "conversion_round:" << bert_ckks.conversion_round << endl;
  cout << "other_nonlinear_round:" << bert_ckks.other_nonlinear_round << endl;
}

void part4_test(int party, Bert_ckks bert_ckks){
  int bitwidth=40, s=20;
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  FixArray input = fpmath->fix->input(party, bert_ckks.linear.data_lin4.image_size*3072, 1, true, bitwidth, s);
  // cout << "bert_ckks.linear.data_lin4.image_size*3072:" << bert_ckks.linear.data_lin4.image_size*3072 << endl;
  input.ell = bitwidth;
  input.s = s;
  double tmp_comm = bert_ckks.n_comm;
  double tmp_round = bert_ckks.n_rounds;
  map<string, FixArray> result2;
  result2 = bert_ckks.linear4_add_layernorm_part1(input, 0);
  cout << "linear4_add_layernorm_part1 comm:" << bert_ckks.n_comm-tmp_comm << endl;
  cout << "linear4_add_layernorm_part1 round:" << bert_ckks.n_rounds-tmp_round << endl;
  cout << "-----------comm------------" << endl;
  cout << "ct_transfer_comm:" << bert_ckks.ct_transfer_comm << endl;
  cout << "conversion_comm:" << bert_ckks.conversion_comm << endl;
  cout << "other_nonlinear_comm:" << bert_ckks.other_nonlinear_comm << endl;
  cout << "-----------round------------" << endl;
  cout << "ct_transfer_round:" << bert_ckks.ct_transfer_round << endl;
  cout << "conversion_round:" << bert_ckks.conversion_round << endl;
  cout << "other_nonlinear_round:" << bert_ckks.other_nonlinear_round << endl;
}

void part2_test(int party, Bert_ckks bert_ckks){
  int bitwidth=40, s=20;
  auto data = bert_ckks.linear.data_lin2;
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  FixArray input = fpmath->fix->input(party, PACKING_NUM*data.image_size*data.image_size, 1, true, bitwidth, s);
  // cout << "bert_ckks.linear.data_lin4.image_size*3072:" << bert_ckks.linear.data_lin4.image_size*3072 << endl;
  double tmp_comm = bert_ckks.n_comm;
  double tmp_round = bert_ckks.n_rounds;
  map<string, FixArray> result2;
  result2 = bert_ckks.Softmax_v_linear_layer_norm_part1(input, 0);
  cout << "linear4_add_layernorm_part1 comm:" << bert_ckks.n_comm-tmp_comm << endl;
  cout << "linear4_add_layernorm_part1 round:" << bert_ckks.n_rounds-tmp_round << endl;
  cout << "-----------comm------------" << endl;
  cout << "ct_transfer_comm:" << bert_ckks.ct_transfer_comm << endl;
  cout << "conversion_comm:" << bert_ckks.conversion_comm << endl;
  cout << "other_nonlinear_comm:" << bert_ckks.other_nonlinear_comm << endl;
  cout << "-----------round------------" << endl;
  cout << "ct_transfer_round:" << bert_ckks.ct_transfer_round << endl;
  cout << "conversion_round:" << bert_ckks.conversion_round << endl;
  cout << "other_nonlinear_round:" << bert_ckks.other_nonlinear_round << endl;
}

void softmax_test(int party, Bert_ckks bert_ckks){
  int bitwidth=40, s=18;
  FixArray input,result;
  FCMetadata data = bert_ckks.linear.data_lin1_0;
  int softmax_dim = data.image_size;
  int softmax_output_size = PACKING_NUM*softmax_dim*softmax_dim;
  input = bert_ckks.nonlinear.fpmath[0]->fix->input(party, softmax_output_size, 1, true, bitwidth, s);
  result = bert_ckks.softmax_he(input);
  bert_ckks.other_nonlinear_comm += bert_ckks.get_comm();
  bert_ckks.other_nonlinear_round += bert_ckks.get_round();
  cout << "-----------comm------------" << endl;
  cout << "ct_transfer_comm:" << bert_ckks.ct_transfer_comm << endl;
  cout << "conversion_comm:" << bert_ckks.conversion_comm << endl;
  cout << "other_nonlinear_comm:" << bert_ckks.other_nonlinear_comm << endl;
  cout << "-----------round------------" << endl;
  cout << "ct_transfer_round:" << bert_ckks.ct_transfer_round << endl;
  cout << "conversion_round:" << bert_ckks.conversion_round << endl;
  cout << "other_nonlinear_round:" << bert_ckks.other_nonlinear_round << endl;
}

static void emit_oracle_stage(Bert_ckks &bert_ckks, const string &name,
                              const FixArray &secure,
                              const map<string, OracleStageReference> &refs) {
  auto it = refs.find(name);
  if (it == refs.end()) {
    throw runtime_error("three-oracle reference is missing: " + name);
  }
  const SecureStageErrorMetrics metrics =
      secure_stage_error_metrics(bert_ckks, secure, it->second);
  const streamsize old_precision = cout.precision();
  cout << setprecision(17) << "ORACLE_AGGREGATE name=" << name
       << " accumulator_ell=63 aggregation_blocks=1"
       << " max_abs_raw=" << metrics.max_abs_raw
       << " max_square_safe_raw=" << metrics.max_square_safe_raw
       << " B_C_MAX_ABS=" << metrics.max_abs
       << " B_C_MAE=" << metrics.mae
       << " square_safe=" << (metrics.square_safe ? 1 : 0)
       << " opened_scalars=" << metrics.opened_scalars
       << " intermediate_tensor_opened=0" << endl;
  if (metrics.square_safe) {
    cout << "ORACLE_STAGE name=" << name
         << " A_B_MSE=" << it->second.a_b_mse
         << " B_C_MSE=" << metrics.mse
         << " B_C_MAX_ABS=" << metrics.max_abs
         << " B_C_MAE=" << metrics.mae << endl;
  }
  cout.precision(old_precision);
  const bool stage_failed =
      !metrics.square_safe || metrics.mse >= 1e-11;
  cout << "BERT_STAGE_GATE name=" << name
       << " status=" << (stage_failed ? "FAIL" : "PASS")
       << " threshold=1e-11"
       << " metric=fixed_secure_s12_whole_stage_mse"
       << " paper_table6_equivalent=0"
       << " diagnostic_continue=" << (oracle_diagnostic_continue ? 1 : 0)
       << endl;
  if (!stage_failed) {
    return;
  }

  oracle_gate_failed = true;
  if (first_oracle_failure.empty()) {
    first_oracle_failure = name;
  }
  string reason;
  if (!metrics.square_safe) {
    reason =
        "secure oracle full-tensor MSE exceeds 63-bit aggregate bound for " +
        name + " max_abs=" + to_string(metrics.max_abs) +
        " mae=" + to_string(metrics.mae) + " square_safe=0";
  } else {
    reason = "fixed-clear/secure MSE gate failed for " + name +
             " value=" + to_string(metrics.mse) +
             " max_abs=" + to_string(metrics.max_abs) +
             " mae=" + to_string(metrics.mae);
  }
  if (!oracle_diagnostic_continue) {
    throw runtime_error(reason);
  }
  cout << "BERT_STAGE_DIAGNOSTIC status=CONTINUE name=" << name
       << " final_result_will_fail=1" << endl;
}

static void emit_oracle_slice_diagnostic(
    Bert_ckks &bert_ckks, const string &stage_name, const string &slice_name,
    const FixArray &secure, size_t offset, size_t size,
    const map<string, OracleStageReference> &refs) {
  auto it = refs.find(stage_name);
  if (it == refs.end()) {
    throw runtime_error("three-oracle reference is missing: " + stage_name);
  }
  if (size == 0 || offset > static_cast<size_t>(secure.size) ||
      size > static_cast<size_t>(secure.size) - offset ||
      offset > it->second.local_share.size() ||
      size > it->second.local_share.size() - offset) {
    throw runtime_error("secure/oracle diagnostic slice is out of range: " +
                        stage_name + "." + slice_name);
  }

  // A public contiguous slice of additive shares is still an additive share;
  // it does not reconstruct any QK/V element.  The existing aggregate helper
  // opens only MAX/MAE and, when its public bound proves it safe, one MSE
  // scalar.  Keeping QK and V separate makes scale/layout failures
  // distinguishable without adding a debug plaintext recovery path.
  FixArray secure_slice(bert_ckks.party, static_cast<int>(size),
                        secure.signed_, secure.ell, secure.s);
  copy(secure.data + offset, secure.data + offset + size,
       secure_slice.data);
  OracleStageReference reference_slice;
  reference_slice.size = size;
  reference_slice.ell = it->second.ell;
  reference_slice.scale = it->second.scale;
  reference_slice.local_share.assign(
      it->second.local_share.begin() + offset,
      it->second.local_share.begin() + offset + size);

  const SecureStageErrorMetrics metrics = secure_stage_error_metrics(
      bert_ckks, secure_slice, reference_slice);
  const streamsize old_precision = cout.precision();
  cout << setprecision(17)
       << "ORACLE_SLICE_AGGREGATE stage=" << stage_name
       << " slice=" << slice_name << " offset=" << offset
       << " elements=" << size << " accumulator_ell=63"
       << " aggregation_blocks=1 max_abs_raw=" << metrics.max_abs_raw
       << " max_square_safe_raw=" << metrics.max_square_safe_raw
       << " B_C_MAX_ABS=" << metrics.max_abs
       << " B_C_MAE=" << metrics.mae
       << " square_safe=" << (metrics.square_safe ? 1 : 0)
       << " opened_scalars=" << metrics.opened_scalars
       << " intermediate_tensor_opened=0";
  if (metrics.square_safe) {
    cout << " B_C_MSE=" << metrics.mse;
  }
  cout << endl;
  cout.precision(old_precision);
}

void transformer_block_test(
    int party, Bert_ckks &bert_ckks,
    const map<string, OracleStageReference> &oracle_references){
  cout << "begin formal layer-0 test" << endl;
  const int bitwidth = bitlength;
  const int s = s_x;
  map<string, FixArray> result,input;
  FixArray layer0_input;
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  auto start = high_resolution_clock::now();
  auto start_init = high_resolution_clock::now();

  // The Zenodo checkpoint contains the already-exported layer-0 hidden
  // tensor. It is private input owned by BOB; neither party uses a synthetic
  // zero/one tensor.  The exported public key mask is validated and applied
  // before MPC Softmax (sample 0 has 44 valid and 84 padded positions).
  const string input_file = model_path + "/inputs_0_data.txt";
  const string mask_file = model_path + "/inputs_0_mask.txt";
  // The embedding tensor is BOB's private input.  Do not even open the data
  // file on Alice; FixOp::input ignores the non-owner pointer and creates the
  // local zero share internally.  The pointer is therefore null on Alice,
  // rather than a synthetic/public copy of the checkpoint.
  vector<uint64_t> input_values;
  if (party == sci::BOB) {
    input_values =
        read_signed_tensor(input_file, static_cast<size_t>(input_size));
  }
  const vector<int64_t> public_attention_mask =
      read_public_layer0_mask(mask_file, static_cast<size_t>(array_size));
  bert_ckks.set_public_attention_mask(public_attention_mask);
  const size_t valid_keys = static_cast<size_t>(count(
      public_attention_mask.begin(), public_attention_mask.end(), int64_t(0)));
  cout << "PUBLIC_ATTENTION_MASK status=PASS semantics=public_key_mask "
       << "add_party=BOB valid_keys=" << valid_keys
       << " masked_keys=" << public_attention_mask.size() - valid_keys
       << endl;
  layer0_input = fpmath->fix->input(
      sci::BOB, input_size,
      party == sci::BOB ? input_values.data() : nullptr,
      true, bitwidth, s);

  // BERT-base is post-norm.  Layer 0 therefore starts with the real exported
  // embedding tensor and sends it directly through the existing pp_1/BSGS
  // Q/K/V + ct-ct QK^T path.  There is no preceding synthetic/identity
  // LayerNorm.
  bert_ckks.attention_input_residual_share = layer0_input;
  bert_ckks.attention_input_residual_ready = true;
  // The official layer-0 x_scale table consumes Q/K/V input at scale 5.
  // Preserve the original s12 share for the attention residual, and perform
  // the seven-bit transition with the existing secure truncation protocol.
  constexpr int kLayer0QkvInputScale = 5;
  const int qkv_shift = layer0_input.s - kLayer0QkvInputScale;
  if (qkv_shift != 7) {
    throw runtime_error("formal layer-0 checkpoint input is not at scale 12");
  }
  FixArray layer0_qkv_input =
      fpmath->fix->truncate_reduce(layer0_input, qkv_shift);
  if (layer0_qkv_input.s != kLayer0QkvInputScale) {
    throw runtime_error("secure layer-0 QKV truncation produced wrong scale");
  }
  if (layer0_qkv_input.size != array_size * dim) {
    throw runtime_error("formal layer-0 QKV input shape mismatch");
  }
  // The checkpoint file and the residual path use sequence-major
  // [token, hidden] order.  BOLT's pp_1/BSGS kernel, however, consumes the
  // same tensor in the original `bert.cpp` column packing:
  // [hidden, token].  This is a fixed public permutation of each party's
  // additive share, so it neither reconstructs the input nor changes the
  // protocol.  Keep layer0_input untouched for the post-attention residual.
  FixArray layer0_qkv_column(
      party, layer0_qkv_input.size, layer0_qkv_input.signed_,
      layer0_qkv_input.ell, layer0_qkv_input.s);
  for (int token = 0; token < array_size; ++token) {
    for (int hidden = 0; hidden < dim; ++hidden) {
      layer0_qkv_column.data[hidden * array_size + token] =
          layer0_qkv_input.data[token * dim + hidden];
    }
  }
  cout << "REAL_LAYER0_INPUT status=PASS elements=" << input_size
       << " residual_scale=" << layer0_input.s
       << " qkv_scale=" << layer0_qkv_input.s
       << " qkv_shift=" << qkv_shift
       << " source_layout=token_hidden_row_major"
       << " qkv_layout=hidden_token_column_major"
       << " public_share_permutation=1" << endl;
  vector<vector<HE_CKKS::scalar_t>> result2;
  result2 = bert_ckks.attention_from_real_input(layer0_qkv_column);
  cout << "--------linear part 1 done--------" << endl;
  // 2. softmax
  FixArray qk,result_fix;
  const size_t qk_values = INPUT_DIM * INPUT_DIM * PACKING_NUM;
  const size_t v_values = INPUT_DIM * COMMON_DIM;
  const size_t qk_cts =
      qk_values / bert_ckks.linear.data_lin1_0.slot_count;
  const size_t v_cts =
      v_values / bert_ckks.linear.data_lin1_0.slot_count;
  if (qk_cts != 12 || v_cts != 6 ||
      result2.size() != qk_cts + v_cts ||
      any_of(result2.begin(), result2.end(), [&](const auto &packed) {
        return packed.size() !=
               static_cast<size_t>(bert_ckks.linear.data_lin1_0.slot_count);
      })) {
    throw runtime_error("formal attention QK/V 12+6 full-slot layout mismatch");
  }
  const size_t packed_values = result2.size() * result2[0].size();
  if (packed_values != qk_values + v_values) {
    throw runtime_error("formal attention output does not contain exact QK^T and V shares");
  }
  cout << "FORMAL_QKV_LAYOUT status=PASS qk_cts=" << qk_cts
       << " qk_scale=" << s << " v_cts=" << v_cts
       << " v_scale=" << ATTENTION_V_SCALE
       << " slots=" << bert_ckks.linear.data_lin1_0.slot_count << endl;
  qk = FixArray(party, static_cast<int>(qk_values), true, bitwidth, s);
  bert_ckks.attention_v_share = FixArray(party, static_cast<int>(v_values),
                                         true, bitwidth,
                                         ATTENTION_V_SCALE);
  size_t packed_offset = 0;
  for (const auto &packed : result2) {
    for (auto value : packed) {
      if (packed_offset < qk_values) {
        qk.data[packed_offset] = value;
      } else {
        bert_ckks.attention_v_share.data[packed_offset - qk_values] = value;
      }
      ++packed_offset;
    }
  }
  if (packed_offset != packed_values) {
    throw runtime_error("formal attention output flattening failed");
  }
  // The HE kernel returns QK^T and V in BOLT's diagonal/cross packing.  That
  // packing must remain intact for the following secure Softmax*V path, but
  // an oracle comparison is meaningful only after the same public local
  // permutation used by the clear implementation.  Reorder copies here;
  // never reconstruct either party's share and never mutate qk/V.
  vector<uint64_t> qk_row(qk_values);
  vector<uint64_t> v_head_major_compare(v_values);
  bert_ckks.linear.plain_cross_packing_postprocess(
      qk.data, qk_row.data(), false, bert_ckks.linear.data_lin1_0);
  // V is correctly returned at scale 11.  Only the local additive-share copy
  // used by the all-s12 oracle stage is shifted to scale 12; the packed V
  // consumed by secure Softmax*V remains untouched at scale 11.
  FixArray attention_v_compare = fpmath->fix->scale_up(
      bert_ckks.attention_v_share, bitwidth, s);
  bert_ckks.linear.plain_cross_packing_postprocess_v(
      attention_v_compare.data, v_head_major_compare.data(), false,
      bert_ckks.linear.data_lin1_0);
  FixArray attention_stage(party, static_cast<int>(packed_values), true,
                           bitwidth, s);
  copy(qk_row.begin(), qk_row.end(), attention_stage.data);
  copy(v_head_major_compare.begin(), v_head_major_compare.end(),
       attention_stage.data + qk_values);
  emit_oracle_slice_diagnostic(
      bert_ckks, "fgf1_ln_attention", "qk_head_query_key_s12",
      attention_stage, 0, qk_values, oracle_references);
  emit_oracle_slice_diagnostic(
      bert_ckks, "fgf1_ln_attention", "v_head_token_channel_s12_compare",
      attention_stage, qk_values, v_values, oracle_references);
  emit_oracle_stage(bert_ckks, "fgf1_ln_attention", attention_stage,
                    oracle_references);
  bert_ckks.attention_v_ready = true;
  cout << "qk.size(): " << qk.size << endl;
  qk = bert_ckks.softmax_he(qk);
  emit_oracle_stage(bert_ckks, "fgf2_softmax", qk, oracle_references);
  bert_ckks.other_nonlinear_comm += bert_ckks.get_comm();
  bert_ckks.other_nonlinear_round += bert_ckks.get_round();
  cout << "--------softmax done--------" << endl;
  // 3. linear part2
  result = bert_ckks.Softmax_v_linear_layer_norm_part1(qk, 0);
  FixArray softmax_v_ln_stage(
      party, result["x_flat_avg"].size + result["sigma_flat"].size,
      true, bitwidth, s);
  copy(result["x_flat_avg"].data,
       result["x_flat_avg"].data + result["x_flat_avg"].size,
       softmax_v_ln_stage.data);
  copy(result["sigma_flat"].data,
       result["sigma_flat"].data + result["sigma_flat"].size,
       softmax_v_ln_stage.data + result["x_flat_avg"].size);
  emit_oracle_stage(bert_ckks, "fgf3_softmax_v_ln", softmax_v_ln_stage,
                    oracle_references);
  cout << "--------linear part 2 done--------" << endl;
  // 4. layernorm nonlinear
  // This is the post-attention LayerNorm of the official BERT block.
  FixArray ln1_w = make_layernorm_parameter_share(
      bert_ckks, fpmath, bert_ckks.linear.w_ln_1, array_size, dim, bitwidth, s);
  FixArray ln1_b = make_layernorm_parameter_share(
      bert_ckks, fpmath, bert_ckks.linear.b_ln_1, array_size, dim, bitwidth, s);
  result["w"] = ln1_w;
  result["b"] = ln1_b;
  start = high_resolution_clock::now();
  result = bert_ckks.layer_norm_nonlinear(result);
  bert_ckks.other_nonlinear_la += ((high_resolution_clock::now() - start)).count()/1e+9;
  cout << "--------layer_norm_nonlinear done--------" << endl;
  bert_ckks.other_nonlinear_comm += bert_ckks.get_comm();
  bert_ckks.other_nonlinear_round += bert_ckks.get_round();
  // 5. linear part3
  result = bert_ckks.ln_part2_linear3_gelu_part1(result);
  cout << "--------linear part 3 done--------" << endl;
  // 6. gelu nonlinear
  start = high_resolution_clock::now();
  result_fix = bert_ckks.gelu_nonlinear(result);
  emit_oracle_stage(bert_ckks, "fgf4_ln_ffn_gelu", result_fix,
                    oracle_references);
  bert_ckks.other_nonlinear_la += ((high_resolution_clock::now() - start)).count()/1e+9;
  cout << "--------gelu nonlinear done--------" << endl;
  bert_ckks.other_nonlinear_comm += bert_ckks.get_comm();
  bert_ckks.other_nonlinear_round += bert_ckks.get_round();
  // 7. linear part4
  result = bert_ckks.linear4_add_layernorm_part1(result_fix, 0);
  FixArray ffn2_ln_stage(
      party, result["x_flat_avg"].size + result["sigma_flat"].size,
      true, bitwidth, s);
  copy(result["x_flat_avg"].data,
       result["x_flat_avg"].data + result["x_flat_avg"].size,
       ffn2_ln_stage.data);
  copy(result["sigma_flat"].data,
       result["sigma_flat"].data + result["sigma_flat"].size,
       ffn2_ln_stage.data + result["x_flat_avg"].size);
  emit_oracle_stage(bert_ckks, "fgf5_ffn2_ln", ffn2_ln_stage,
                    oracle_references);
  // Complete the second (output) LayerNorm's MPC rsqrt/affine hand-off with
  // the real checkpoint parameters.  The pre-rsqrt shares above are the
  // Figure-10 acceptance boundary; this call ensures the formal path does
  // not silently omit ln2 weights/biases.
  result["w"] = make_layernorm_parameter_share(
      bert_ckks, fpmath, bert_ckks.linear.w_ln_2, array_size, dim, bitwidth, s);
  result["b"] = make_layernorm_parameter_share(
      bert_ckks, fpmath, bert_ckks.linear.b_ln_2, array_size, dim, bitwidth, s);
  result = bert_ckks.layer_norm_nonlinear(result);
  FixArray final_ln2 = bert_ckks.layer_norm_affine_ckks(result);
  if (final_ln2.size != input_size) {
    throw runtime_error("formal final LayerNorm output shape mismatch");
  }
  cout << "FINAL_LAYERNORM status=PASS real_ln2_parameters=1 ckks_affine=1"
       << endl;
  auto end = high_resolution_clock::now();
  cout << "--------linear part 4 done--------" << endl;
  cout << "-----------comm------------" << endl;
  cout << "ct_transfer_comm:" << bert_ckks.ct_transfer_comm << endl;
  cout << "conversion_comm:" << bert_ckks.conversion_comm << endl;
  cout << "other_nonlinear_comm:" << bert_ckks.other_nonlinear_comm << endl;
  cout << "-----------round------------" << endl;
  cout << "ct_transfer_round:" << bert_ckks.ct_transfer_round << endl;
  cout << "conversion_round:" << bert_ckks.conversion_round << endl;
  cout << "other_nonlinear_round:" << bert_ckks.other_nonlinear_round << endl;
  cout << "-----------time------------" << endl;
  bert_ckks.total_la = ((end - start_init)).count()/1e+9;
  cout << "total time:" << bert_ckks.total_la << endl;
  cout << "ct_transfer_time:" << bert_ckks.ct_transfer_la << endl;
  cout << "ct_transfer_time(H<->D):" << bert_ckks.ct_transfer_mem_la << endl;
  cout << "conversion_time:" << bert_ckks.conversion_la << endl;
  cout << "other_nonlinear_time:" << bert_ckks.other_nonlinear_la << endl;
  cout << "he_compute_time:" << bert_ckks.he_compute_la << endl;
  cout << "other time:" << bert_ckks.total_la-bert_ckks.ct_transfer_la-bert_ckks.conversion_la-bert_ckks.other_nonlinear_la-bert_ckks.he_compute_la << endl;
}

int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("N", dim, "Number of operation operations");
  amap.arg("nt", num_threads, "Number of threads");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("model", model_path, "BLB-local BERT weights directory");
  amap.arg("backend", backend_mode,
           "HE backend: cpu/host or gpu/device (GPU build required)");
  amap.arg("manifest_verified", manifest_verified,
           "Set to 1 only after repro/verify_bert_manifest.py passes");
  amap.arg("layers", requested_layers, "Formal BERT layer count");

  amap.parse(argc, argv);
  const char *oracle_continue_env = getenv("BLB_ORACLE_CONTINUE");
  oracle_diagnostic_continue =
      oracle_continue_env != nullptr && string(oracle_continue_env) == "1";
  cout << "ORACLE_POLICY fixed_secure_threshold=1e-11"
       << " paper_table6_equivalent=0"
       << " diagnostic_continue=" << (oracle_diagnostic_continue ? 1 : 0)
       << endl;
  LOCATION backend = LOCATION::HOST;
  if (backend_mode == "cpu" || backend_mode == "host") {
    backend = LOCATION::HOST;
  } else if (backend_mode == "gpu" || backend_mode == "device") {
    backend = LOCATION::DEVICE;
  } else {
    cerr << "FORMAL_GATE status=BLOCKED reason=invalid_backend value="
         << backend_mode << endl;
    return 2;
  }
  if (manifest_verified != 1) {
    cerr << "FORMAL_GATE status=BLOCKED reason=manifest_not_verified" << endl;
    return 2;
  }
  if (requested_layers != 1) {
    cerr << "FORMAL_GATE status=BLOCKED reason=single_layer_not_yet_accepted requested_layers="
         << requested_layers << endl;
    return 2;
  }
  if (!kFormalLayer0InputReady) {
    cerr << "FORMAL_GATE status=BLOCKED reason=real_layer0_input_not_implemented" << endl;
    return 2;
  }
  if (!kFormalThreeOracleReady) {
    cerr << "FORMAL_GATE status=BLOCKED reason=three_oracle_runtime_not_implemented" << endl;
    return 2;
  }
  ifstream manifest(model_path + (model_path.back() == '/' ? "" : "/") + "manifest.json");
  if (!manifest.good()) {
    cerr << "FORMAL_GATE status=BLOCKED reason=missing_manifest" << endl;
    return 2;
  }
  printf("number of points: %d\n", input_size);
  /********** Setup HE_CKKS ***********/
  /********************************************/
  /********** Setup IO and Base OTs ***********/
  /********************************************/
  std::cout << "All Base OTs Done" << std::endl;

  /************ Generate Test Data ************/
  /********************************************/

  // char fix_key[] = "\x61\x7e\xcd\xa2\xa0\x51\x1e\x96"
  //                      "\x5e\x41\xc2\x9b\x15\x3f\xc7\x7a";

  PRG128 prg(sci::fix_key);
  using scalar_ty = uint64_t;
  try {
  map<string, OracleStageReference> oracle_references =
      load_oracle_references(model_path, party);
  cout << "HE_BACKEND mode="
       << (backend == LOCATION::DEVICE ? "gpu" : "cpu") << endl;
  Bert_ckks bert_ckks(party, port, address, backend, model_path);
  // Bert bt(party, port, address, "checkpoints/bert-base/weights_txt", false);
  std::random_device rdv;
  std::vector<double> x(4096);
  vector<scalar_ty> x_mpc(4096);
  vector<double> ifft_r_complex(8192);
  vector<uint64_t> des(8192);
  vector<uint64_t> des_seal(8192);
  vector<scalar_ty> y(8192);
  Plaintext y_ntt;
  vector<uint64_t> y_ntt2(8192);
  vector<double> y_double(8192);
  std::uniform_real_distribution<double> uniform(1.0, 4.0);
  // std::generate_n(x.data(), 4096, [&]() { return uniform(rdv);});
  // y = bert_ckks.linear.ckks_decode(x, 8);
  // y = bert_ckks.linear.ckks_encode(x, 4);
  // des = read_from_file<uint64_t>("outputs/legacy_debug/destination.txt");
  // des_seal = read_from_file<uint64_t>("outputs/legacy_debug/destination_seal.txt");
  // for(int i=0;i<des.size();i++){
  //   if(des[i] != des_seal[i]){
  //     cout << "error" << endl;
  //     cout << i << endl;
  //     cout << des[i] << endl;
  //     cout << des_seal[i] << endl;
  //   }
  // }

  // x = read_from_file<double>("outputs/legacy_debug/r.txt");
  // ifft_r_complex = read_from_file<double>("outputs/legacy_debug/ifft_r_complex.txt");
  // write_to_file2("outputs/legacy_debug/my_x.txt", x);
  // field_to_ring_test(party,bert_ckks);
  // ring_to_field_test(party,bert_ckks);
  // mpc_to_ckks_test(party,bert_ckks);
  // square_test(party,bert_ckks);
  // layernorm_nonlinear_test(party,bert_ckks);
  // layernorm_test(party,bert_ckks);
  // part2_test(party,bert_ckks);
  // gelu_test(party,bert_ckks);
  // part4_test(party,bert_ckks);
  // softmax_test(party,bert_ckks);
  transformer_block_test(party, bert_ckks, oracle_references);
  bert_ckks.get_comm();
  cout << "ROTATION_COUNT value=" << bert_ckks.linear.rotation_count << endl;
  cout << "CT_CT_MULTIPLY_COUNT value=" << bert_ckks.linear.ct_ct_multiply_count << endl;
  cout << "CT_PT_MULTIPLY_COUNT value=" << bert_ckks.linear.ct_pt_multiply_count << endl;
  cout << "CKKS_TO_MPC_COUNT value=" << bert_ckks.ckks_to_mpc_conversions << endl;
  cout << "MPC_TO_CKKS_COUNT value=" << bert_ckks.mpc_to_ckks_conversions << endl;
  cout << "FIELD_TO_RING_COUNT value=" << bert_ckks.field_to_ring_calls << endl;
  cout << "MASKED_RNS_COEFFICIENT_COUNT value=" << bert_ckks.masked_rns_coefficients << endl;
  cout << "TRUNCATION_COUNT value=" << truncation_call_count.load() << endl;
  cout << "COMMUNICATION_BYTES value=" << static_cast<uint64_t>(bert_ckks.n_comm) << endl;
  cout << "END_TO_END_LATENCY_SECONDS value=" << bert_ckks.total_la << endl;
  if (oracle_gate_failed) {
    cout << "STAGE_MSE status=FAIL figure10_blocks=5 first_failure="
         << first_oracle_failure << endl;
    cout << "SINGLE_LAYER_SECURITY status=FAIL" << endl;
    throw runtime_error(
        "diagnostic run completed with fixed-clear/secure stage failures; "
        "first_failure=" + first_oracle_failure);
  }
  cout << "STAGE_MSE status=PASS figure10_blocks=5" << endl;
  cout << "SINGLE_LAYER_SECURITY status=PASS" << endl;
  // ifft_test(party,bert_ckks);
  return 0;
  } catch (const exception &error) {
    cerr << "SINGLE_LAYER_FORMAL status=BLOCKED reason=" << error.what() << endl;
    return 2;
  }
}
