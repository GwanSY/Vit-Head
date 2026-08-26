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
int bitlength = 43;
int32_t dim = 768;
int32_t array_size = 128;
int32_t bw_x = 43;
int32_t bw_y = 43;
int32_t s_x = 13;
int32_t s_y = 13;
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
// Benchmark runs must not include the secure/oracle acceptance reductions in
// the measured protocol path.  Correctness runs keep the default fail-closed
// oracle behavior.
bool benchmark_only = false;

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
    // padding. Keep its public value inside the signed 43-bit MPC ring.
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
  if (getenv("BLB_QKV_AGGREGATE") != nullptr &&
      string(getenv("BLB_QKV_AGGREGATE")) == "1") {
    // Temporary diagnostic-only references for the real layer-0 Q/K/V edge.
    // They are loaded only when the diagnostic is explicitly requested and
    // never participate in the normal five-stage acceptance run.
    const char *debug_stages[] = {"debug_qkv_q", "debug_qkv_k",
                                  "debug_qkv_v"};
    for (const char *stage_name : debug_stages) {
      const string stage(stage_name);
      size_t stage_pos = manifest.find("\"" + stage + "\"");
      if (stage_pos == string::npos) {
        throw runtime_error("QKV diagnostic stage is missing from manifest: " +
                            stage);
      }
      double size_number = 0.0;
      double a_b_mse = 0.0;
      if (!parse_json_number_after(manifest, stage_pos, "size", size_number) ||
          !parse_json_number_after(manifest, stage_pos, "A_B_MSE", a_b_mse) ||
          size_number <= 0.0 || !isfinite(a_b_mse) || a_b_mse < 0.0) {
        throw runtime_error("QKV diagnostic manifest metadata is invalid: " +
                            stage);
      }
      const size_t expected_size = static_cast<size_t>(size_number);
      const string role = owner_party == sci::ALICE ? "alice" : "bob";
      OracleStageReference reference = read_oracle_share(
          oracle_dir + "/" + stage + "." + role + ".bin", expected_size);
      reference.a_b_mse = a_b_mse;
      references.emplace(stage, std::move(reference));
    }
  }
  return references;
}

struct SecureStageErrorMetrics {
  double max_abs = 0.0;
  double mae = 0.0;
  double signed_mean = 0.0;
  double mse = numeric_limits<double>::quiet_NaN();
  uint64_t max_abs_raw = 0;
  int64_t signed_sum_raw = 0;
  uint64_t max_square_safe_raw = 0;
  bool square_safe = false;
  bool signed_sum_available = false;
  size_t opened_scalars = 0;
};

static SecureStageErrorMetrics secure_stage_error_metrics(
    Bert_ckks &bert_ckks, const FixArray &secure,
    const OracleStageReference &reference, bool skip_mse = false) {
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
  cout << "ORACLE_AGGREGATE_DEBUG stage_size=" << element_count
       << " secure_ell=" << secure.ell << " secure_scale=" << secure.s
       << " max_abs_raw=" << max_abs_raw
       << " absolute_sum_raw=" << opened_absolute_sum.data[0]
       << " max_times_count=" << actual_absolute_bound << endl;
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

  // Optional aggregate-only orientation diagnostic.  This opens one signed
  // scalar for the whole tensor, never an element or a block.  It is useful
  // for distinguishing a uniform offset/layout error from symmetric CKKS
  // approximation noise while keeping the acceptance path unchanged unless
  // explicitly requested.
  const bool signed_sum_requested =
      getenv("BLB_ERROR_SIGNED_SUM") != nullptr &&
      string(getenv("BLB_ERROR_SIGNED_SUM")) == "1";
  if (signed_sum_requested) {
    FixArray signed_difference =
        fpmath->fix->extend(difference, kAccumulatorEll,
                            difference_negative.data);
    uint64_t signed_sum_share = 0;
    for (int i = 0; i < signed_difference.size; ++i) {
      signed_sum_share =
          (signed_sum_share + signed_difference.data[i]) & kAccumulatorMax;
    }
    FixArray secret_signed_sum(bert_ckks.party, 1, true, kAccumulatorEll,
                               secure.s);
    secret_signed_sum.data[0] = signed_sum_share;
    FixArray opened_signed_sum =
        fpmath->fix->output(sci::PUBLIC, secret_signed_sum);
    const uint64_t residue = opened_signed_sum.data[0] & kAccumulatorMax;
    const uint64_t signed_limit = uint64_t(1) << (kAccumulatorEll - 1);
    const int64_t signed_sum_raw =
        (residue & signed_limit)
            ? static_cast<int64_t>(residue - (uint64_t(1) << kAccumulatorEll))
            : static_cast<int64_t>(residue);
    metrics.signed_sum_raw = signed_sum_raw;
    metrics.signed_mean = static_cast<double>(
        static_cast<long double>(signed_sum_raw) /
        (ldexp(1.0L, secure.s) * count));
    metrics.signed_sum_available = true;
    ++metrics.opened_scalars;
  }

  // Some boundary audits intentionally use MAX/MAE only.  This keeps the
  // existing full-tensor square-safety fail-closed rule for normal oracle
  // stages while allowing a diagnostic to report its first two aggregates.
  if (skip_mse) {
    return metrics;
  }

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
    cout << "ORACLE_MSE_AGGREGATE_DEBUG stage_size=" << element_count
         << " max_abs_raw=" << max_abs_raw
         << " square_sum_raw=" << opened_square_sum.data[0]
         << " max_square_raw=" << max_square_raw
         << " max_square_times_count=" << actual_square_bound << endl;
    if (opened_square_sum.data[0] > actual_square_bound) {
      if (!oracle_diagnostic_continue) {
        throw runtime_error(
            "secure oracle MSE scalar exceeds its proved full-tensor bound");
      }
      // Diagnostic continuation is fail-closed: retain the public aggregate
      // evidence, mark MSE unavailable, and let later Figure-10 boundaries
      // run.  The final driver still reports BLOCKED and never treats this as
      // a correctness pass.
      metrics.square_safe = false;
      ++metrics.opened_scalars;
    } else {
      const long double square_scale = std::ldexp(1.0L, 2 * secure.s);
      metrics.mse = static_cast<double>(
          static_cast<long double>(opened_square_sum.data[0]) /
          (square_scale * count));
      metrics.opened_scalars = 3;
    }
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
    size_t rows, size_t cols, int ell, int scale, int layer_id = 0) {
  const bool layer_present =
      layer_id >= 0 && static_cast<size_t>(layer_id) < params.size() &&
      params[layer_id].size() == cols;
  if (!layer_present) {
    // On BOB the model vector is intentionally empty; the owner input below
    // still creates a zero local share. Alice must have the complete tensor.
    if (bert_ckks.party == sci::ALICE) {
      throw runtime_error("real LayerNorm parameter is missing or malformed");
    }
  }
  vector<uint64_t> expanded(rows * cols, 0);
  if (bert_ckks.party == sci::ALICE) {
    if (!layer_present) {
      throw runtime_error("real LayerNorm parameter is missing or malformed");
    }
    // The exported BERT LayerNorm parameters are fixed at s=12.  The paper
    // evaluates the nonlinear domain at s=13; multiplying the signed ring
    // representation by two preserves the parameter's real value without
    // changing the parameter owner or introducing an MPC conversion.
    constexpr int kCheckpointParameterScale = 12;
    if (scale < kCheckpointParameterScale ||
        scale - kCheckpointParameterScale > 1) {
      throw runtime_error("unsupported LayerNorm parameter scale transition");
    }
    const int shift = scale - kCheckpointParameterScale;
    for (size_t row = 0; row < rows; ++row) {
      for (size_t column = 0; column < cols; ++column) {
        expanded[row * cols + column] =
            params[layer_id][column] << shift;
      }
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
  uint64_t Q = bert_ckks.linear.he1->get_conversion_prime();
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
  map<char*,FixArray> result,input;
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
  map<char*,FixArray> result,input;
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
  map<char*,FixArray> result,input;
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
  map<char*,FixArray> result2;
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
  map<char*,FixArray> result2;
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
  map<char*,FixArray> result2;
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
  if (benchmark_only) {
    return;
  }
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
  if (metrics.signed_sum_available) {
    cout << "ORACLE_SIGNED_ERROR name=" << name
         << " signed_sum_raw=" << metrics.signed_sum_raw
         << " B_C_SIGNED_MEAN=" << metrics.signed_mean << endl;
  }
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
       << " metric=fixed_secure_paper_ell_s_whole_stage_mse"
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
    const map<string, OracleStageReference> &refs, bool skip_mse = false) {
  if (benchmark_only) {
    return;
  }
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
      bert_ckks, secure_slice, reference_slice, skip_mse);
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

  // Optional scalar-only scale diagnosis.  This is deliberately outside the
  // production path: it tests a public candidate multiplier against additive
  // oracle shares and opens only the same MAX/MAE scalars as the normal
  // acceptance diagnostic.  No QKV element, ciphertext, or share is printed.
  // It is useful while auditing a migrated CKKS scale contract, but it must
  // never be enabled for benchmark numbers.
  const char *scale_probe = getenv("BLB_ORACLE_SCALE_PROBE");
  if (scale_probe != nullptr && string(scale_probe) == "1") {
    const uint64_t mask = (uint64_t{1} << secure_slice.ell) - 1;
    const int64_t candidates[] = {2, 4, 8, 64, 128, 4096, 8192};
    for (const int64_t candidate : candidates) {
      OracleStageReference scaled = reference_slice;
      for (uint64_t &value : scaled.local_share) {
        if (candidate >= 0) {
          value = static_cast<uint64_t>(
              (static_cast<__uint128_t>(value) *
               static_cast<uint64_t>(candidate)) & mask);
        }
      }
      const SecureStageErrorMetrics probe = secure_stage_error_metrics(
          bert_ckks, secure_slice, scaled);
      cout << setprecision(17) << "ORACLE_SCALE_PROBE stage=" << stage_name
           << " slice=" << slice_name << " multiplier=" << candidate
           << " max_abs_raw=" << probe.max_abs_raw
           << " B_C_MAX_ABS=" << probe.max_abs
           << " B_C_MAE=" << probe.mae << endl;
    }
  }
}

void transformer_block_test(
    int party, Bert_ckks &bert_ckks,
    const map<string, OracleStageReference> &oracle_references){
  cout << "begin formal layer-0 test" << endl;
  const int bitwidth = bitlength;
  const int s = s_x;
  map<char*,FixArray> result,input;
  FixArray layer0_input;
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  auto start = high_resolution_clock::now();

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
    // The checkpoint export is s=12.  Re-express the same private tensor in
    // the paper's s=13 nonlinear ring before it enters any protocol.
    for (auto &value : input_values) value <<= (s_x - 12);
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

  // The exported embedding tensor is the block input, not part of the block.
  // Synchronize after sharing it, then align the legacy delta counters before
  // starting both the wall-clock and send-only online measurements.
  bert_ckks.io->sync();
  bert_ckks.get_comm();
  bert_ckks.get_round();
  const uint64_t block_online_comm_start = bert_ckks.current_comm();
  const auto block_online_stream_start = bert_ckks.current_stream_comm();
  const auto start_init = high_resolution_clock::now();
  bert_ckks.profile_start(BlbViTProfileStage::QKV);

  // BERT-base is post-norm.  Layer 0 therefore starts with the real exported
  // embedding tensor and sends it directly through the existing pp_1/BSGS
  // Q/K/V + ct-ct QK^T path.  There is no preceding synthetic/identity
  // LayerNorm.
  bert_ckks.attention_input_residual_share = layer0_input;
  bert_ckks.attention_input_residual_ready = true;
  // The official layer-0 x_scale table consumes Q/K/V input at scale 5.
  // Preserve the paper s13 share for the attention residual, and perform
  // the seven-bit transition with the existing secure truncation protocol.
  constexpr int kLayer0QkvInputScale = 5;
  const int qkv_shift = layer0_input.s - kLayer0QkvInputScale;
  if (qkv_shift != s_x - kLayer0QkvInputScale) {
    throw runtime_error("formal layer-0 checkpoint input scale transition is invalid");
  }
  FixArray layer0_qkv_input =
      fpmath->fix->truncate_reduce(layer0_input, qkv_shift);
  if (layer0_qkv_input.s != kLayer0QkvInputScale) {
    throw runtime_error("secure layer-0 QKV truncation produced wrong scale");
  }
  if (layer0_qkv_input.size != array_size * dim) {
    throw runtime_error("formal layer-0 QKV input shape mismatch");
  }
  if (getenv("BLB_QKV_AGGREGATE") != nullptr &&
      string(getenv("BLB_QKV_AGGREGATE")) == "1") {
    // Diagnostic-only input-boundary audit.  Bob's reference is the signed
    // floor divide specified by bert.py (s=13 -> s=5); Alice contributes the
    // zero reference share.  secure_stage_error_metrics opens only aggregate
    // MAX/MAE/MSE scalars and never opens an input element.
    OracleStageReference input_reference;
    input_reference.size = static_cast<size_t>(layer0_qkv_input.size);
    input_reference.ell = layer0_qkv_input.ell;
    input_reference.scale = layer0_qkv_input.s;
    input_reference.local_share.assign(input_reference.size, 0);
    if (party == sci::BOB) {
      const uint64_t output_mask = layer0_qkv_input.ell == 64
                                        ? uint64_t(-1)
                                        : ((uint64_t(1) << layer0_qkv_input.ell) - 1);
      for (size_t i = 0; i < input_reference.size; ++i) {
        int64_t signed_value = 0;
        memcpy(&signed_value, &input_values[i], sizeof(signed_value));
        // input_values is the checkpoint s=12 tensor lifted to s=13.
        // C++ division truncates toward zero, so spell out floor division for
        // negative values to match numpy.floor_divide in the oracle.
        const int64_t quotient = signed_value >= 0
                                     ? (signed_value >> qkv_shift)
                                     : -(((-signed_value) +
                                          ((int64_t(1) << qkv_shift) - 1)) >>
                                         qkv_shift);
        input_reference.local_share[i] =
            static_cast<uint64_t>(quotient) & output_mask;
      }
    }
    const SecureStageErrorMetrics input_metrics = secure_stage_error_metrics(
        bert_ckks, layer0_qkv_input, input_reference, true);
    cout << setprecision(17)
         << "ORACLE_AGGREGATE name=layer0_qkv_input"
         << " max_abs_raw=" << input_metrics.max_abs_raw
         << " B_C_MAX_ABS=" << input_metrics.max_abs
         << " B_C_MAE=" << input_metrics.mae
         << " square_safe=" << (input_metrics.square_safe ? 1 : 0)
         << " intermediate_tensor_opened=0" << endl;
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
  if (getenv("BLB_QKV_AGGREGATE") != nullptr &&
      string(getenv("BLB_QKV_AGGREGATE")) == "1") {
    const size_t qkv_cts_per_tensor =
        static_cast<size_t>(COMMON_DIM / OUTPUT_DIM * array_size *
                            OUTPUT_DIM / bert_ckks.linear.data_lin1_0.slot_count);
    const size_t expected_qkv_cts = qkv_cts_per_tensor * 3;
    if (qkv_cts_per_tensor == 0 || result2.size() != expected_qkv_cts) {
      throw runtime_error("QKV diagnostic ciphertext layout mismatch");
    }
    const size_t slots = bert_ckks.linear.data_lin1_0.slot_count;
    const size_t tensor_size = static_cast<size_t>(array_size) * COMMON_DIM;
    vector<uint64_t> packed(qkv_cts_per_tensor * slots);
    vector<uint64_t> row_major(tensor_size);
    FPMath *qkv_fpmath = bert_ckks.nonlinear.fpmath[0];
    const char *names[] = {"debug_qkv_q", "debug_qkv_k", "debug_qkv_v"};
    for (size_t tensor = 0; tensor < 3; ++tensor) {
      size_t offset = 0;
      for (size_t ct = 0; ct < qkv_cts_per_tensor; ++ct) {
        if (result2[tensor * qkv_cts_per_tensor + ct].size() != slots) {
          throw runtime_error("QKV diagnostic slot count mismatch");
        }
        copy(result2[tensor * qkv_cts_per_tensor + ct].begin(),
             result2[tensor * qkv_cts_per_tensor + ct].end(),
             packed.begin() + offset);
        offset += slots;
      }
      fill(row_major.begin(), row_major.end(), 0);
      bert_ckks.linear.plain_cross_packing_postprocess_v(
          packed.data(), row_major.data(), false,
          bert_ckks.linear.data_lin1_0);
      FixArray at_qkv_scale(
          party, static_cast<int>(tensor_size), true, bitwidth,
          ATTENTION_V_SCALE);
      copy(row_major.begin(), row_major.end(), at_qkv_scale.data);
      FixArray at_paper_scale = qkv_fpmath->fix->scale_up(
          at_qkv_scale, bitwidth, s_x);
      const auto debug_ref = oracle_references.find(names[tensor]);
      cout << "QKV_DEBUG_META tensor=" << names[tensor]
           << " secure_size=" << at_paper_scale.size
           << " secure_ell=" << at_paper_scale.ell
           << " secure_scale=" << at_paper_scale.s
           << " ref_size="
           << (debug_ref == oracle_references.end()
                   ? 0
                   : static_cast<unsigned long long>(debug_ref->second.size))
           << " ref_ell="
           << (debug_ref == oracle_references.end() ? 0 : debug_ref->second.ell)
           << " ref_scale="
           << (debug_ref == oracle_references.end() ? 0
                                                     : debug_ref->second.scale)
           << endl;
      emit_oracle_stage(bert_ckks, names[tensor], at_paper_scale,
                        oracle_references);
      const char *per_ct = getenv("BLB_QKV_PER_CIPHERTEXT");
      if (per_ct != nullptr && string(per_ct) == "1") {
        for (size_t ct = 0; ct < qkv_cts_per_tensor; ++ct) {
          emit_oracle_slice_diagnostic(
              bert_ckks, names[tensor], "ct" + to_string(ct), at_paper_scale,
              ct * slots, slots, oracle_references, true);
        }
      }
    }
    cout << "QKV_DEBUG status=COMPLETE qkv_cts_per_tensor="
         << qkv_cts_per_tensor << " output_scale=" << ATTENTION_V_SCALE
         << " comparison_scale=" << s_x << endl;
    return;
  }
  // 2. softmax
  FixArray qk,result_fix;
  const size_t qk_values = INPUT_DIM * INPUT_DIM * PACKING_NUM;
  const size_t v_values = INPUT_DIM * COMMON_DIM;
  const size_t qk_cts =
      qk_values / bert_ckks.linear.data_lin1_0.slot_count;
  const size_t v_cts =
      v_values / bert_ckks.linear.data_lin1_0.slot_count;
  const bool he_v_path = bert_ckks.attention_v_he_ready;
  if (qk_cts != 12 || v_cts != 6 ||
      result2.size() != (he_v_path ? qk_cts : qk_cts + v_cts) ||
      any_of(result2.begin(), result2.end(), [&](const auto &packed) {
        return packed.size() !=
               static_cast<size_t>(bert_ckks.linear.data_lin1_0.slot_count);
      })) {
    throw runtime_error("formal attention QK/V full-slot layout mismatch");
  }
  const size_t packed_values = result2.size() * result2[0].size();
  const size_t expected_values = he_v_path ? qk_values : qk_values + v_values;
  if (packed_values != expected_values) {
    throw runtime_error("formal attention output does not contain the expected QK/V shares");
  }
  cout << "FORMAL_QKV_LAYOUT status=PASS qk_cts=" << qk_cts
       << " qk_scale=" << s << " v_cts=" << (he_v_path ? 0 : v_cts)
       << " v_scale=" << ATTENTION_V_SCALE
       << " v_transport=" << (he_v_path ? "HE_at_Alice" : "MPC_to_CKKS")
       << " slots=" << bert_ckks.linear.data_lin1_0.slot_count << endl;
  qk = FixArray(party, static_cast<int>(qk_values), true, bitwidth, s);
  if (!he_v_path) {
    bert_ckks.attention_v_share = FixArray(
        party, static_cast<int>(v_values), true, bitwidth,
        ATTENTION_V_SCALE);
  }
  size_t packed_offset = 0;
  for (const auto &packed : result2) {
    for (auto value : packed) {
      if (packed_offset < qk_values) {
        qk.data[packed_offset] = value;
      } else if (!he_v_path) {
        bert_ckks.attention_v_share.data[packed_offset - qk_values] = value;
      }
      ++packed_offset;
    }
  }
  if (packed_offset != packed_values) {
    throw runtime_error("formal attention output flattening failed");
  }
  if (he_v_path) {
    // The original BLB BERT boundary exports only QK to MPC.  V is already
    // retained as Alice's HE ciphertext by layer_norm_part2_attention and is
    // consumed by the unchanged Softmax*V kernel below.
    bert_ckks.attention_v_ready = false;
  } else {
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
  // used by the legacy oracle stage is shifted to the paper scale; the packed V
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
      bert_ckks, "fgf1_ln_attention", "qk_head_query_key_paper_scale",
      attention_stage, 0, qk_values, oracle_references);
  emit_oracle_slice_diagnostic(
      bert_ckks, "fgf1_ln_attention", "v_head_token_channel_paper_scale_compare",
      attention_stage, qk_values, v_values, oracle_references);
  emit_oracle_stage(bert_ckks, "fgf1_ln_attention", attention_stage,
                    oracle_references);
  bert_ckks.attention_v_ready = true;
  }
  cout << "qk.size(): " << qk.size << endl;
  bert_ckks.profile_switch(BlbViTProfileStage::SOFTMAX);
  qk = bert_ckks.softmax_he(qk);
  emit_oracle_stage(bert_ckks, "fgf2_softmax", qk, oracle_references);
  bert_ckks.other_nonlinear_comm += bert_ckks.get_comm();
  bert_ckks.other_nonlinear_round += bert_ckks.get_round();
  cout << "--------softmax done--------" << endl;
  // 3. Figure-10 Block 4: Softmax*V -> output projection -> residual ->
  // LayerNorm linear first half. This is the fused call used by the official
  // BLB BERT driver; do not exit at the ViT pre-norm residual boundary.
  result = bert_ckks.Softmax_v_linear_layer_norm_part1(qk, 0);
  cout << "--------linear part 2 done--------" << endl;
  // 4. layernorm nonlinear
  // This is the post-attention LayerNorm of the official BERT block.
  bert_ckks.profile_switch(BlbViTProfileStage::LAYERNORM1);
  FixArray ln1_w = make_layernorm_parameter_share(
      bert_ckks, fpmath, bert_ckks.linear.w_ln_1, array_size, dim, bitwidth, s);
  FixArray ln1_b = make_layernorm_parameter_share(
      bert_ckks, fpmath, bert_ckks.linear.b_ln_1, array_size, dim, bitwidth, s);
  result["w"] = ln1_w;
  result["b"] = ln1_b;
  FixArray softmax_v_ln_stage(
      party, result["x_flat_avg"].size + array_size,
      true, bitwidth, s);
  copy(result["x_flat_avg"].data,
       result["x_flat_avg"].data + result["x_flat_avg"].size,
       softmax_v_ln_stage.data);
  for (int token = 0; token < array_size; ++token) {
    softmax_v_ln_stage.data[result["x_flat_avg"].size + token] =
        result["sigma_flat"].data[token * dim];
  }
  emit_oracle_stage(bert_ckks, "fgf3_softmax_v_ln", softmax_v_ln_stage,
                    oracle_references);
  start = high_resolution_clock::now();
  result = bert_ckks.layer_norm_nonlinear(result);
  bert_ckks.other_nonlinear_la +=
      ((high_resolution_clock::now() - start)).count() / 1e9;
  cout << "--------layer_norm_nonlinear done--------" << endl;
  bert_ckks.other_nonlinear_comm += bert_ckks.get_comm();
  bert_ckks.other_nonlinear_round += bert_ckks.get_round();
  // 5. linear part3
  bert_ckks.profile_switch(BlbViTProfileStage::FFN1);
  result = bert_ckks.ln_part2_linear3_gelu_part1(result);
  cout << "--------linear part 3 done--------" << endl;
  // 6. gelu nonlinear
  bert_ckks.profile_switch(BlbViTProfileStage::GELU);
  start = high_resolution_clock::now();
  result_fix = bert_ckks.gelu_nonlinear(result);
  emit_oracle_stage(bert_ckks, "fgf4_ln_ffn_gelu", result_fix,
                    oracle_references);
  bert_ckks.other_nonlinear_la += ((high_resolution_clock::now() - start)).count()/1e+9;
  cout << "--------gelu nonlinear done--------" << endl;
  bert_ckks.other_nonlinear_comm += bert_ckks.get_comm();
  bert_ckks.other_nonlinear_round += bert_ckks.get_round();
  // 7. linear part4
  bert_ckks.profile_switch(BlbViTProfileStage::FFN2);
  result = bert_ckks.linear4_add_layernorm_part1(result_fix, 0);
  // Figure-10 Block 1 returns the centered tensor and variance. Attach the
  // real checkpoint parameters in place of the official artifact's synthetic
  // all-one test shares, then finish only the final model-output LayerNorm.
  FixArray ln2_w = make_layernorm_parameter_share(
      bert_ckks, fpmath, bert_ckks.linear.w_ln_2, array_size, dim, bitwidth, s);
  FixArray ln2_b = make_layernorm_parameter_share(
      bert_ckks, fpmath, bert_ckks.linear.b_ln_2, array_size, dim, bitwidth, s);
  result["w"] = ln2_w;
  result["b"] = ln2_b;
  bert_ckks.profile_switch(BlbViTProfileStage::LAYERNORM2);
  FixArray ffn2_ln_stage(
      party, result["x_flat_avg"].size + array_size,
      true, bitwidth, s);
  copy(result["x_flat_avg"].data,
       result["x_flat_avg"].data + result["x_flat_avg"].size,
       ffn2_ln_stage.data);
  for (int token = 0; token < array_size; ++token) {
    ffn2_ln_stage.data[result["x_flat_avg"].size + token] =
        result["sigma_flat"].data[token * dim];
  }
  emit_oracle_stage(bert_ckks, "fgf5_ffn2_ln", ffn2_ln_stage,
                    oracle_references);
  start = high_resolution_clock::now();
  result = bert_ckks.layer_norm_nonlinear(result);
  bert_ckks.other_nonlinear_la +=
      ((high_resolution_clock::now() - start)).count() / 1e9;
  FixArray final_ln2 = bert_ckks.layer_norm_affine_ckks(result);
  if (final_ln2.size != input_size) {
    throw runtime_error("formal final LayerNorm output shape mismatch");
  }
  cout << "FINAL_LAYERNORM status=PASS real_ln2_parameters=1 ckks_affine=1"
       << endl;
  bert_ckks.profile_stop();
  auto end = high_resolution_clock::now();
  bert_ckks.block_online_comm_bytes =
      bert_ckks.current_comm() - block_online_comm_start;
  bert_ckks.block_online_profile_seconds =
      chrono::duration<double>(end - start_init).count();
  // Classify any final LayerNorm MPC bytes left after its internal conversion
  // accounting. This changes only reporting; get_comm() sends no data.
  bert_ckks.other_nonlinear_comm += bert_ckks.get_comm();
  bert_ckks.other_nonlinear_round += bert_ckks.get_round();
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

  static const array<const char *, kBlbViTProfileStageCount> stage_names = {
      "QKV", "QK^T", "CKKS_to_MPC", "Softmax", "MPC_to_CKKS",
      "Softmax_x_V", "Output_Projection", "LayerNorm1", "FFN1",
      "GELU", "FFN2", "LayerNorm2"};
  uint64_t profiled_bytes = 0;
  double profiled_ms = 0.0;
  for (size_t index = 0; index < kBlbViTProfileStageCount; ++index) {
    const auto stage = static_cast<BlbViTProfileStage>(index);
    const auto &metric = bert_ckks.profile_metric(stage);
    profiled_bytes += metric.send_bytes;
    profiled_ms += metric.wall_ms;
    cout << "BLOCK_ONLINE_STAGE party=" << party
         << " stage=" << stage_names[index]
         << " wall_ms=" << metric.wall_ms
         << " send_bytes=" << metric.send_bytes
         << " rounds=" << metric.rounds
         << " he_ms=" << bert_ckks.profile_he_ms(stage)
         << " time_share="
         << (bert_ckks.block_online_profile_seconds > 0.0
                 ? metric.wall_ms /
                       (bert_ckks.block_online_profile_seconds * 10.0)
                 : 0.0)
         << " comm_share="
         << (bert_ckks.block_online_comm_bytes > 0
                 ? 100.0 * static_cast<double>(metric.send_bytes) /
                       static_cast<double>(bert_ckks.block_online_comm_bytes)
                 : 0.0)
         << endl;
  }
  if (profiled_bytes != bert_ckks.block_online_comm_bytes) {
    throw runtime_error("BERT online stage communication does not cover block");
  }
  cout << "BLOCK_ONLINE_TIME seconds="
       << bert_ckks.block_online_profile_seconds
       << " profiled_ms=" << profiled_ms << endl;
  cout << "BLOCK_ONLINE_COMM send_bytes="
       << bert_ckks.block_online_comm_bytes
       << " profiled_bytes=" << profiled_bytes << endl;
  const auto block_online_stream_end = bert_ckks.current_stream_comm();
  cout << "BLOCK_ONLINE_STREAM_COMM party=" << party
       << " primary_send_bytes="
       << block_online_stream_end[0] - block_online_stream_start[0]
       << " reverse_send_bytes="
       << block_online_stream_end[1] - block_online_stream_start[1]
       << " gc_send_bytes="
       << block_online_stream_end[2] - block_online_stream_start[2]
       << endl;

  static const array<const char *, kBlbViTCommAuditBucketCount>
      audit_names = {"QK_to_Softmax_conversion",
                     "Softmax_to_CKKS_conversion",
                     "Softmax_rec_conversion", "other_conversion",
                     "Softmax_rec_scope", "Softmax_cmp", "Softmax_mux"};
  for (size_t index = 0; index < kBlbViTCommAuditBucketCount; ++index) {
    const auto bucket = static_cast<BlbViTCommAuditBucket>(index);
    const auto &metric = bert_ckks.profile_comm_audit(bucket);
    cout << "VOLE_COMM_AUDIT party=" << party
         << " bucket=" << audit_names[index]
         << " send_bytes=" << metric.send_bytes
         << " calls=" << metric.calls << endl;
  }
  const uint64_t softmax_rec_scope_bytes = bert_ckks.profile_comm_audit(
      BlbViTCommAuditBucket::SOFTMAX_REC).send_bytes;
  const uint64_t softmax_rec_conversion_bytes = bert_ckks.profile_comm_audit(
      BlbViTCommAuditBucket::SOFTMAX_REC_CONVERSION).send_bytes;
  if (softmax_rec_scope_bytes < softmax_rec_conversion_bytes) {
    throw runtime_error("Softmax reciprocal audit scope is inconsistent");
  }
  cout << "VOLE_COMM_AUDIT_EXCLUSIVE party=" << party
       << " bucket=Softmax_rec_nonconversion"
       << " send_bytes="
       << softmax_rec_scope_bytes - softmax_rec_conversion_bytes << endl;
}

// Decode the public packing emitted by the original Block-2 QKV/ QK path.
// QK stays at the paper nonlinear scale.  The checkpoint's V edge is the
// original s=11 edge; for the regular (post-norm) Block-2 path it is decoded
// at s=13 and then reduced once with the existing fixed-point truncation.
static FixArray decode_full_bert_qkv(
    Bert_ckks &bert_ckks,
    const vector<vector<HE_CKKS::scalar_t>> &qkv,
    bool v_at_nonlinear_scale) {
  const int bitwidth = NL_ELL;
  const int scale = NL_SCALE;
  const size_t qk_values = INPUT_DIM * INPUT_DIM * PACKING_NUM;
  const size_t v_values = INPUT_DIM * COMMON_DIM;
  const size_t slot_count = bert_ckks.linear.data_lin1_0.slot_count;
  const bool he_v_path = bert_ckks.attention_v_he_ready;
  const size_t qk_cts = qk_values / slot_count;
  const size_t v_cts = v_values / slot_count;
  const size_t expected_cts = qk_cts + (he_v_path ? 0 : v_cts);
  const size_t degree = slot_count * 2;
  const bool full_slot_layout =
      qkv.size() == expected_cts &&
      all_of(qkv.begin(), qkv.end(), [&](const auto &packed) {
        return packed.size() == slot_count;
      });
  // The original Figure-10 inter-block edge sends only the first half of the
  // CKKS ciphertext vector.  ckks_to_mpc(..., half=true) returns both halves
  // in each decoded part, so normalize that local representation back to the
  // original full-slot ciphertext order before feeding Softmax.  This is a
  // layout adapter only: it adds no conversion, communication, or arithmetic.
  const bool half_slot_layout =
      expected_cts % 2 == 0 && qkv.size() == expected_cts / 2 &&
      all_of(qkv.begin(), qkv.end(), [&](const auto &packed) {
        return packed.size() == degree;
      });
  if (!full_slot_layout && !half_slot_layout) {
    throw runtime_error("Figure-10 QK/V packing layout mismatch");
  }

  FixArray qk(bert_ckks.party, static_cast<int>(qk_values), true,
              bitwidth, scale);
  FixArray v_at_scale(bert_ckks.party, static_cast<int>(v_values), true,
                      bitwidth, v_at_nonlinear_scale ? scale
                                                      : ATTENTION_V_SCALE);
  vector<vector<HE_CKKS::scalar_t>> full_slot_qkv;
  if (full_slot_layout) {
    full_slot_qkv = qkv;
  } else {
    full_slot_qkv.resize(expected_cts, vector<HE_CKKS::scalar_t>(slot_count));
    const size_t half_cts = expected_cts / 2;
    for (size_t index = 0; index < half_cts; ++index) {
      copy(qkv[index].begin(), qkv[index].begin() + slot_count,
           full_slot_qkv[index].begin());
      copy(qkv[index].begin() + slot_count, qkv[index].end(),
           full_slot_qkv[index + half_cts].begin());
    }
    cout << "FORMAL_LAYOUT_ADAPTER stage=ln2_attention_qkv"
         << " status=PASS source=xyj_half_slot target=ordinary_full_slot"
         << " source_cts=" << qkv.size()
         << " target_cts=" << full_slot_qkv.size()
         << " communication=0" << endl;
  }
  size_t packed_offset = 0;
  for (const auto &packed : full_slot_qkv) {
    for (const auto value : packed) {
      if (packed_offset < qk_values) {
        qk.data[packed_offset] = value;
      } else if (!he_v_path && packed_offset < qk_values + v_values) {
        v_at_scale.data[packed_offset - qk_values] = value;
      } else {
        throw runtime_error("Figure-10 QK/V output is oversized");
      }
      ++packed_offset;
    }
  }
  if (packed_offset != qk_values + (he_v_path ? 0 : v_values)) {
    throw runtime_error("Figure-10 QK/V output is incomplete");
  }
  if (!he_v_path) {
    bert_ckks.attention_v_share = v_at_nonlinear_scale
        ? bert_ckks.nonlinear.fpmath[0]->fix->right_shift(
              v_at_scale, NL_SCALE - ATTENTION_V_SCALE)
        : v_at_scale;
    if (bert_ckks.attention_v_share.s != ATTENTION_V_SCALE) {
      throw runtime_error("Figure-10 V scale transition produced wrong metadata");
    }
    bert_ckks.attention_v_ready = true;
  } else {
    bert_ckks.attention_v_ready = false;
  }
  return qk;
}

static map<char *, FixArray> attach_bert_layernorm_parameters(
    Bert_ckks &bert_ckks, map<char *, FixArray> value,
    const vector<vector<uint64_t>> &weights,
    const vector<vector<uint64_t>> &bias, int layer_id) {
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  value["w"] = make_layernorm_parameter_share(
      bert_ckks, fpmath, weights, array_size, dim, NL_ELL, NL_SCALE,
      layer_id);
  value["b"] = make_layernorm_parameter_share(
      bert_ckks, fpmath, bias, array_size, dim, NL_ELL, NL_SCALE,
      layer_id);
  return value;
}

// One complete post-norm BERT block after its Block-2 QKV boundary.  The
// calls deliberately follow Figure 10: Block 3 Softmax, Block 4
// Softmax*V/output/residual/LN-part1, MPC rsqrt, Block 5 LN-affine/FFN1/
// GeLU-linear, MPC GeLU, Block 1 FFN2/residual/LN-part1, and MPC rsqrt.
// No pre-norm-only branch is enabled here.
static map<char *, FixArray> run_full_bert_fused_block(
    Bert_ckks &bert_ckks, FixArray qk, int layer_id,
    const FixArray &attention_residual) {
  if (layer_id < 0 || layer_id >= ATTENTION_LAYERS) {
    throw runtime_error("full BERT layer id is out of range");
  }
  if (attention_residual.size != input_size ||
      attention_residual.ell != NL_ELL ||
      attention_residual.s != NL_SCALE) {
    throw runtime_error("Figure-10 attention residual metadata mismatch");
  }
  bert_ckks.attention_input_residual_share = attention_residual;
  bert_ckks.attention_input_residual_ready = true;
  bert_ckks.ffn_residual_ready = false;

  qk = bert_ckks.softmax_he(qk);
  map<char *, FixArray> result =
      bert_ckks.Softmax_v_linear_layer_norm_part1(qk, layer_id);
  result = attach_bert_layernorm_parameters(
      bert_ckks, std::move(result), bert_ckks.linear.w_ln_1,
      bert_ckks.linear.b_ln_1, layer_id);
  result = bert_ckks.layer_norm_nonlinear(std::move(result));

  result = bert_ckks.ln_part2_linear3_gelu_part1(result, layer_id);
  FixArray gelu = bert_ckks.gelu_nonlinear(result);
  result = bert_ckks.linear4_add_layernorm_part1(gelu, layer_id);
  result = attach_bert_layernorm_parameters(
      bert_ckks, std::move(result), bert_ckks.linear.w_ln_2,
      bert_ckks.linear.b_ln_2, layer_id);
  result = bert_ckks.layer_norm_nonlinear(std::move(result));

  for (const auto &entry : result) {
    if (entry.first != nullptr && string(entry.first) == "x_flat_avg") {
      if (entry.second.size != input_size || entry.second.ell != NL_ELL ||
          entry.second.s != NL_SCALE) {
        throw runtime_error("Figure-10 LN2 boundary metadata mismatch");
      }
    }
  }
  cout << "FULL_BERT_LAYER layer=" << layer_id << " status=PASS"
       << " figure10_blocks=5 boundary=LN2_pre_rsqrt" << endl;
  return result;
}

static uint64_t full_model_ring_mask() {
  return (uint64_t(1) << bitlength) - 1;
}

// The pooler and classifier are MPC-only in the BLB reference.  Alice secret
// shares the real checkpoint parameters once; the matrix products and tanh
// then use the existing SCI primitives without revealing the [CLS] vector.
static vector<uint64_t> run_full_pool_classifier(
    Bert_ckks &bert_ckks, const FixArray &hidden) {
  const size_t pool_weight_count = static_cast<size_t>(COMMON_DIM) * COMMON_DIM;
  const size_t pool_bias_count = COMMON_DIM;
  const size_t classifier_weight_count =
      static_cast<size_t>(COMMON_DIM) * NUM_CLASS;
  const size_t classifier_bias_count = NUM_CLASS;
  const size_t total = pool_weight_count + pool_bias_count +
                       classifier_weight_count + classifier_bias_count;
  const uint64_t mask = full_model_ring_mask();
  vector<uint64_t> wire(total, 0);
  vector<uint64_t> clear;
  if (bert_ckks.party == sci::ALICE) {
    if (bert_ckks.linear.w_p.size() != COMMON_DIM ||
        bert_ckks.linear.w_c.size() != COMMON_DIM ||
        bert_ckks.linear.b_p.size() != COMMON_DIM ||
        bert_ckks.linear.b_c.size() != NUM_CLASS) {
      throw runtime_error("full BERT pool/classifier weights are missing");
    }
    clear.reserve(total);
    for (const auto &row : bert_ckks.linear.w_p) {
      if (row.size() != COMMON_DIM) {
        throw runtime_error("full BERT pooler weight shape mismatch");
      }
      clear.insert(clear.end(), row.begin(), row.end());
    }
    clear.insert(clear.end(), bert_ckks.linear.b_p.begin(),
                 bert_ckks.linear.b_p.end());
    for (const auto &row : bert_ckks.linear.w_c) {
      if (row.size() != NUM_CLASS) {
        throw runtime_error("full BERT classifier weight shape mismatch");
      }
      clear.insert(clear.end(), row.begin(), row.end());
    }
    clear.insert(clear.end(), bert_ckks.linear.b_c.begin(),
                 bert_ckks.linear.b_c.end());
    if (clear.size() != total) {
      throw runtime_error("full BERT pool/classifier parameter size mismatch");
    }
    PRG128 prg;
    prg.random_data(wire.data(), wire.size() * sizeof(uint64_t));
    for (auto &value : wire) value &= mask;
    bert_ckks.io->send_data(wire.data(), wire.size() * sizeof(uint64_t));
  } else {
    bert_ckks.io->recv_data(wire.data(), wire.size() * sizeof(uint64_t));
  }

  vector<uint64_t> pool_w(pool_weight_count), pool_b(pool_bias_count);
  vector<uint64_t> classifier_w(classifier_weight_count), classifier_b(classifier_bias_count);
  size_t offset = 0;
  auto receive_share = [&](vector<uint64_t> &dst) {
    for (size_t i = 0; i < dst.size(); ++i, ++offset) {
      if (bert_ckks.party == sci::ALICE) {
        dst[i] = (clear[offset] - wire[offset]) & mask;
      } else {
        dst[i] = wire[offset] & mask;
      }
    }
  };
  receive_share(pool_w);
  receive_share(pool_b);
  receive_share(classifier_w);
  receive_share(classifier_b);

  vector<uint64_t> cls_share(COMMON_DIM);
  copy(hidden.data, hidden.data + COMMON_DIM, cls_share.data());
  vector<uint64_t> pool_linear(COMMON_DIM), pool_input(COMMON_DIM);
  vector<uint64_t> tanh_output(COMMON_DIM);
  vector<uint64_t> classifier_linear(NUM_CLASS), logits(NUM_CLASS);
  bert_ckks.nonlinear.p_matrix_mul_iron(
      1, cls_share.data(), pool_w.data(), pool_linear.data(),
      1, COMMON_DIM, COMMON_DIM,
      NL_ELL, NL_ELL, NL_ELL, NL_SCALE, NL_SCALE, 2 * NL_SCALE);
  for (int i = 0; i < COMMON_DIM; ++i) {
    pool_linear[i] = (pool_linear[i] + pool_b[i]) & mask;
  }
  bert_ckks.nonlinear.right_shift(
      1, pool_linear.data(), NL_SCALE, pool_input.data(), COMMON_DIM,
      NL_ELL, 2 * NL_SCALE);
  bert_ckks.nonlinear.tanh(
      1, pool_input.data(), tanh_output.data(), COMMON_DIM, NL_ELL, NL_SCALE);
  bert_ckks.nonlinear.n_matrix_mul_iron(
      1, tanh_output.data(), classifier_w.data(), classifier_linear.data(),
      1, 1, COMMON_DIM, NUM_CLASS,
      NL_ELL, NL_ELL, NL_ELL, NL_SCALE, NL_SCALE, 2 * NL_SCALE);
  for (int i = 0; i < NUM_CLASS; ++i) {
    classifier_linear[i] = (classifier_linear[i] + classifier_b[i]) & mask;
  }
  bert_ckks.nonlinear.right_shift(
      1, classifier_linear.data(), NL_SCALE, logits.data(), NUM_CLASS,
      NL_ELL, 2 * NL_SCALE);

  FixArray logits_share(bert_ckks.party, NUM_CLASS, true, NL_ELL, NL_SCALE);
  copy(logits.begin(), logits.end(), logits_share.data);
  FixArray opened = bert_ckks.nonlinear.fpmath[0]->fix->output(
      sci::PUBLIC, logits_share);
  vector<uint64_t> result(opened.data, opened.data + NUM_CLASS);
  auto signed_value = [mask](uint64_t value) -> int64_t {
    value &= mask;
    const uint64_t sign = uint64_t(1) << (NL_ELL - 1);
    if (value & sign) {
      return static_cast<int64_t>(value - (uint64_t(1) << NL_ELL));
    }
    return static_cast<int64_t>(value);
  };
  cout << "FULL_CLASSIFIER status=PASS logits_paper_scale=";
  for (int i = 0; i < NUM_CLASS; ++i) {
    if (i) cout << ",";
    cout << static_cast<double>(signed_value(result[i])) /
                 static_cast<double>(uint64_t(1) << NL_SCALE);
  }
  const int prediction = signed_value(result[0]) >= signed_value(result[1]) ? 0 : 1;
  cout << " prediction=" << prediction << endl;
  return result;
}

static void full_bert_gpu_inference(
    Bert_ckks &bert_ckks,
    const high_resolution_clock::time_point &started) {
  const string input_file = model_path + "/inputs_0_data.txt";
  const string mask_file = model_path + "/inputs_0_mask.txt";
  vector<uint64_t> input_values;
  if (bert_ckks.party == sci::BOB) {
    input_values = read_signed_tensor(input_file, static_cast<size_t>(input_size));
    for (auto &value : input_values) value <<= (s_x - 12);
  }
  const vector<int64_t> mask_values = read_public_layer0_mask(
      mask_file, static_cast<size_t>(array_size));
  bert_ckks.set_public_attention_mask(mask_values);
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  FixArray hidden = fpmath->fix->input(
      sci::BOB, input_size,
      bert_ckks.party == sci::BOB ? input_values.data() : nullptr,
      true, bitlength, s_x);
  if (hidden.size != input_size || hidden.ell != bitlength || hidden.s != s_x) {
    throw runtime_error("full BERT private input metadata mismatch");
  }

  // Layer 0 is the only post-norm entry without a preceding fused Block 1.
  // Preserve the checkpoint's x_scale[0]=5 QKV edge with the existing secure
  // truncation, while keeping the residual itself in the paper's (43,13)
  // nonlinear ring.
  bert_ckks.attention_v_ready = false;
  bert_ckks.attention_v_he_ready = false;
  bert_ckks.attention_v_he_cts.clear();
  bert_ckks.attention_input_residual_share = hidden;
  bert_ckks.attention_input_residual_ready = true;
  constexpr int kLayer0QkvInputScale = 5;
  FixArray layer0_qkv_input = fpmath->fix->truncate_reduce(
      hidden, hidden.s - kLayer0QkvInputScale);
  if (layer0_qkv_input.s != kLayer0QkvInputScale) {
    throw runtime_error("full BERT layer-0 QKV truncation produced wrong scale");
  }
  FixArray layer0_qkv_column(
      bert_ckks.party, layer0_qkv_input.size, layer0_qkv_input.signed_,
      layer0_qkv_input.ell, layer0_qkv_input.s);
  for (int token = 0; token < array_size; ++token) {
    for (int feature = 0; feature < dim; ++feature) {
      layer0_qkv_column.data[feature * array_size + token] =
          layer0_qkv_input.data[token * dim + feature];
    }
  }
  vector<vector<HE_CKKS::scalar_t>> qkv =
      bert_ckks.attention_from_real_input(layer0_qkv_column, 0);
  FixArray qk = decode_full_bert_qkv(bert_ckks, qkv, false);
  map<char *, FixArray> layer_boundary = run_full_bert_fused_block(
      bert_ckks, std::move(qk), 0, hidden);

  // At every non-final boundary Figure 10 feeds Block 1's (centered, sigma)
  // pair through MPC rsqrt into the next Block 2.  The existing
  // layer_norm_part2_attention call performs that next Block 2 affine+QKV
  // fusion and records its affine output as the residual consumed by Block 4.
  for (int layer_id = 1; layer_id < 12; ++layer_id) {
    bert_ckks.attention_v_ready = false;
    bert_ckks.attention_v_he_ready = false;
    bert_ckks.attention_v_he_cts.clear();
    qkv = bert_ckks.layer_norm_part2_attention(layer_boundary, layer_id);
    qk = decode_full_bert_qkv(bert_ckks, qkv, true);
    if (!bert_ckks.attention_input_residual_ready) {
      throw runtime_error(
          "Figure-10 Block-2 did not produce the next attention residual");
    }
    const FixArray next_attention_residual =
        bert_ckks.attention_input_residual_share;
    layer_boundary = run_full_bert_fused_block(
        bert_ckks, std::move(qk), layer_id, next_attention_residual);
  }

  // The final Block 1 is followed by the model's single output affine before
  // the MPC-only pooler/classifier.  This is intentionally outside the 12
  // repeated blocks and is the only use of the existing final affine helper.
  hidden = bert_ckks.layer_norm_affine_ckks(layer_boundary);
  if (hidden.size != input_size || hidden.ell != NL_ELL ||
      hidden.s != NL_SCALE) {
    throw runtime_error("full BERT final LayerNorm output metadata mismatch");
  }
  run_full_pool_classifier(bert_ckks, hidden);
  bert_ckks.total_la =
      chrono::duration<double>(high_resolution_clock::now() - started).count();
  cout << "FULL_BERT_INFERENCE status=PASS layers=12 pooler=1 classifier=1"
       << " latency_seconds=" << bert_ckks.total_la << endl;
}

static uint64_t rounded_metric(double value) {
  if (!isfinite(value) || value <= 0.0) {
    return 0;
  }
  return static_cast<uint64_t>(llround(value));
}

static void print_secure_benchmark_metrics(
    Bert_ckks &bert_ckks, double preprocessing_seconds,
    double process_wall_seconds) {
  const uint64_t total_bytes = bert_ckks.block_online_comm_bytes > 0
                                   ? bert_ckks.block_online_comm_bytes
                                   : rounded_metric(bert_ckks.n_comm);
  const uint64_t ct_transfer_bytes =
      rounded_metric(bert_ckks.ct_transfer_comm);
  const uint64_t conversion_bytes = rounded_metric(bert_ckks.conversion_comm);
  const uint64_t other_nonlinear_bytes =
      rounded_metric(bert_ckks.other_nonlinear_comm);
  const uint64_t accounted_bytes = ct_transfer_bytes + conversion_bytes +
                                   other_nonlinear_bytes;
  const uint64_t unclassified_bytes =
      total_bytes > accounted_bytes ? total_bytes - accounted_bytes : 0;

  cout << "SECURE_BENCHMARK_METRICS party=" << bert_ckks.party
       << " online_end_to_end_seconds=" << bert_ckks.total_la
       << " preprocessing_seconds=" << preprocessing_seconds
       << " process_wall_seconds=" << process_wall_seconds
       << " linear_he_seconds=" << bert_ckks.he_compute_la
       << " nonlinear_mpc_seconds=" << bert_ckks.other_nonlinear_la
       << " mpc_conversion_seconds=" << bert_ckks.conversion_la
       << " ct_transfer_seconds=" << bert_ckks.ct_transfer_la << endl;
  cout << "COMMUNICATION_BREAKDOWN party=" << bert_ckks.party
       << " total_sent_bytes=" << total_bytes
       << " ct_transfer_bytes=" << ct_transfer_bytes
       << " he_mpc_conversion_bytes=" << conversion_bytes
       << " nonlinear_mpc_bytes=" << other_nonlinear_bytes
       << " unclassified_bytes=" << unclassified_bytes << endl;
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
  amap.arg("benchmark_only", benchmark_only,
           "Skip secure oracle reductions during timing (0/1)");

  amap.parse(argc, argv);
  const auto program_started = high_resolution_clock::now();
  const char *oracle_continue_env = getenv("BLB_ORACLE_CONTINUE");
  oracle_diagnostic_continue =
      oracle_continue_env != nullptr && string(oracle_continue_env) == "1";
  cout << "ORACLE_POLICY fixed_secure_threshold=1e-11"
       << " paper_table6_equivalent=0"
       << " diagnostic_continue=" << (oracle_diagnostic_continue ? 1 : 0)
       << " benchmark_only=" << (benchmark_only ? 1 : 0)
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
  if (requested_layers != 1 && requested_layers != 12) {
    cerr << "FORMAL_GATE status=BLOCKED reason=unsupported_layer_count requested_layers="
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
  map<string, OracleStageReference> oracle_references;
  if (requested_layers == 1 && !benchmark_only) {
    oracle_references = load_oracle_references(model_path, party);
  }
  // CKKS conversion applies BLB Appendix C.4 locally at its extended-ring
  // boundary. Keep SCI's normal truncation everywhere else so the original
  // Softmax, LayerNorm, and GELU protocols remain unchanged.
  set_local_truncation_mode(false);
  cout << "TRUNCATION_MODE mode=exact_mpc scope=nonlinear"
       << " conversion=BLB_Appendix_C4_local" << endl;
#if USE_VOLE
  cout << "OT_EXTENSION mode=VOLE_PCG implementation=SilentOT" << endl;
#else
  cout << "OT_EXTENSION mode=IKNP implementation=SplitIKNP" << endl;
#endif
  cout << "HE_BACKEND mode="
       << (backend == LOCATION::DEVICE ? "gpu" : "cpu") << endl;
  const auto setup_started = high_resolution_clock::now();
  Bert_ckks bert_ckks(party, port, address, backend, model_path,
                      requested_layers);
  const double setup_seconds =
      chrono::duration<double>(high_resolution_clock::now() - setup_started).count();
  cout << "PREPROCESSING_SECONDS value=" << setup_seconds << endl;
  // Do not let Bob start the online timer while Alice is still preparing the
  // fused HE caches.  The paper's reported end-to-end number is the online
  // secure execution after preprocessing; both parties must enter it together.
  bert_ckks.io->sync();
  if (requested_layers == 12) {
    // n_comm is a cumulative cursor and already includes the constructor's
    // VOLE/SilentOT preprocessing.  Use direct counter snapshots around the
    // synchronized inference boundary so the full-model result has the same
    // online-only, send-only definition as BLOCK_ONLINE_COMM.
    const uint64_t full_online_comm_start = bert_ckks.current_comm();
    const auto full_inference_started = high_resolution_clock::now();
    full_bert_gpu_inference(bert_ckks, full_inference_started);
    const uint64_t full_online_comm_end = bert_ckks.current_comm();
    if (full_online_comm_end < full_online_comm_start) {
      throw runtime_error("full BERT communication counter moved backwards");
    }
    bert_ckks.block_online_comm_bytes =
        full_online_comm_end - full_online_comm_start;
    const double process_wall_seconds =
        chrono::duration<double>(high_resolution_clock::now() - program_started).count();
    cout << "COMMUNICATION_BYTES value="
         << bert_ckks.block_online_comm_bytes << endl;
    cout << "END_TO_END_LATENCY_SECONDS value=" << bert_ckks.total_la
         << endl;
    print_secure_benchmark_metrics(bert_ckks, setup_seconds,
                                   process_wall_seconds);
    return 0;
  }
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
  const double process_wall_seconds =
      chrono::duration<double>(high_resolution_clock::now() - program_started).count();
  cout << "ROTATION_COUNT value=" << bert_ckks.linear.rotation_count << endl;
  cout << "QK_ROTATION_COUNT value=" << bert_ckks.linear.qk_rotation_count << endl;
  cout << "SOFTMAX_V_ROTATION_COUNT value="
       << bert_ckks.linear.softmax_v_rotation_count << endl;
  cout << "CT_CT_MULTIPLY_COUNT value=" << bert_ckks.linear.ct_ct_multiply_count << endl;
  cout << "CT_PT_MULTIPLY_COUNT value=" << bert_ckks.linear.ct_pt_multiply_count << endl;
  cout << "CKKS_TO_MPC_COUNT value=" << bert_ckks.ckks_to_mpc_conversions << endl;
  cout << "MPC_TO_CKKS_COUNT value=" << bert_ckks.mpc_to_ckks_conversions << endl;
  cout << "FIELD_TO_RING_COUNT value=" << bert_ckks.field_to_ring_calls << endl;
  cout << "RING_TO_FIELD_COUNT value=" << bert_ckks.ring_to_field_calls << endl;
  cout << "CONVERSION_CIPHERTEXT_COUNT value="
       << bert_ckks.conversion_ciphertexts << endl;
  cout << "HE_RESCALE_COUNT value=" << bert_ckks.he_rescale_count << endl;
  cout << "MASKED_RNS_COEFFICIENT_COUNT value=" << bert_ckks.masked_rns_coefficients << endl;
  cout << "TRUNCATION_COUNT value=" << truncation_call_count.load() << endl;
  cout << "COMMUNICATION_BYTES value="
       << (bert_ckks.block_online_comm_bytes > 0
               ? bert_ckks.block_online_comm_bytes
               : static_cast<uint64_t>(bert_ckks.n_comm))
       << endl;
  cout << "END_TO_END_LATENCY_SECONDS value=" << bert_ckks.total_la << endl;
  print_secure_benchmark_metrics(bert_ckks, setup_seconds,
                                 process_wall_seconds);
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
