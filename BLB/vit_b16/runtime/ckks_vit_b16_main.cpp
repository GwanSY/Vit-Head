// Faithful BLB CKKS/MPC baseline for ViT-B/16.
//
// This driver deliberately keeps the original BLB evaluator graph:
//   CKKS BSGS linear layers -> BLB MPC nonlinear layers -> CKKS BSGS linear
// layers.  The only ViT-specific code here is model/input adaptation and the
// two pre-norm residual boundaries required by the ViT graph.  No BFV/BOLT
// operator, heterogeneous attention, alternate Softmax, alternate GELU, or
// alternate LayerNorm is selected by this executable.

#include "ckks_bert.h"
#include "vit_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace sci;
using namespace std;

// SCI's LinearHE utility library expects the benchmark's fixed-point width
// as a process-global symbol.  The original BERT executable provided it from
// its main translation unit; keep the same ABI for this standalone driver.
int bitlength = NL_ELL;

namespace {

constexpr int kBitwidth = NL_ELL;
constexpr int kScale = NL_SCALE;
constexpr int kAttentionVScale = ATTENTION_V_SCALE;
constexpr int kDim = COMMON_DIM;
constexpr int kHeads = PACKING_NUM;
constexpr int kHeadDim = OUTPUT_DIM;
constexpr int kFfDim = INTER_DIM;
constexpr int kQkvWeightScale = 6;
constexpr int kQkvBiasScale = 11;
constexpr int kLogicalTokens = 197;
constexpr int kPhysicalTokens = 256;
constexpr int kLayers = ATTENTION_LAYERS;
constexpr int kClasses = 1000;
constexpr int kPhysicalClasses = 1024;
constexpr int kPatchSize = 16;
constexpr int kImageSize = 224;
constexpr int kChannels = 3;

// The secure BLB QK operator is intentionally unchanged.  Standard ViT
// semantics apply 1/sqrt(head_dim) to QK^T. ViT-B/16 has head_dim=64, so the
// runtime applies the exact public factor 1/8 to Q through CKKS scale metadata
// after the original six-bit Q projection. This avoids losing three weight
// precision bits and changes no packing, modulus chain, or MPC primitive.
static_assert(kHeadDim == 64, "ViT-B/16 attention head dimension changed");

struct Args {
  int party = sci::ALICE;
  int port = 8000;
  string address = "127.0.0.1";
  string model = "checkpoints/vit_b16/bolt_model.txt";
  string input = "outputs/vit_b16_gpu/bob_image.txt";
  string output = "outputs/vit_b16_blb_ckks_gpu2_lan";
  string backend = "gpu";
  int logical_tokens = kLogicalTokens;
  int physical_tokens = kPhysicalTokens;
  // Profiling-only early stop.  The default remains the complete 12-block
  // ViT graph; stopping after a block never changes the operator path.
  int stop_after_block = kLayers;
};

struct BlockProfile {
  int block = 0;
  double secure_end_to_end_ms = 0.0;
  double block_online_time_ms = 0.0;
  uint64_t block_online_comm_bytes = 0;
  double he_time_ms = 0.0;
  double qkv_he_ms = 0.0;
  double qk_transpose_he_ms = 0.0;
  double softmax_v_he_ms = 0.0;
  double linear2_he_ms = 0.0;
  double linear1_he_ms = 0.0;
  double linear2_softmax_v_he_ms = 0.0;
  double linear3_he_ms = 0.0;
  double linear4_he_ms = 0.0;
  double linear_he_linear1_4_ms = 0.0;
  double linear_he_including_mul_v_ms = 0.0;
  // These names describe the measured scope precisely.  LayerNorm includes
  // BLB's prescribed CKKS square/affine boundaries, while GELU is the strict
  // MPC tail after BLB's CKKS polynomial preparation.
  double nonlinear_softmax_bucket_ms = 0.0;
  double nonlinear_layernorm_bucket_ms = 0.0;
  double nonlinear_gelu_core_ms = 0.0;
  double conversion_time_ms = 0.0;
  double h2d_time_ms = 0.0;
  double plaintext_h2d_time_ms = 0.0;
  double plaintext_device_encode_time_ms = 0.0;
  double plaintext_device_release_time_ms = 0.0;
  double ciphertext_h2d_time_ms = 0.0;
  double d2h_time_ms = 0.0;
  uint64_t he_ct_pt_multiply = 0;
  uint64_t he_ct_ct_multiply = 0;
  uint64_t he_rotations = 0;
  uint64_t he_rescales = 0;
  uint64_t ckks_to_mpc = 0;
  uint64_t mpc_to_ckks = 0;
  uint64_t field_to_ring_calls = 0;
  uint64_t ring_to_field_calls = 0;
  uint64_t conversion_ciphertexts = 0;
  uint64_t conversion_bytes = 0;
  uint64_t ct_transfer_bytes = 0;
  uint64_t mpc_communication_bytes = 0;
  uint64_t total_communication_bytes = 0;
  uint64_t communication_rounds = 0;
  array<BlbViTProfileMetric, kBlbViTProfileStageCount> online_stages{};
  array<BlbViTCommAuditMetric, kBlbViTCommAuditBucketCount> comm_audit{};
};

static const char *profile_stage_name(BlbViTProfileStage stage) {
  switch (stage) {
    case BlbViTProfileStage::QKV: return "QKV";
    case BlbViTProfileStage::QK_TRANSPOSE: return "QK^T";
    case BlbViTProfileStage::CKKS_TO_MPC: return "CKKS_to_MPC";
    case BlbViTProfileStage::SOFTMAX: return "Softmax";
    case BlbViTProfileStage::MPC_TO_CKKS: return "MPC_to_CKKS";
    case BlbViTProfileStage::SOFTMAX_V: return "Softmax_x_V";
    case BlbViTProfileStage::OUTPUT_PROJECTION: return "Output_Projection";
    case BlbViTProfileStage::LAYERNORM1: return "LayerNorm1";
    case BlbViTProfileStage::FFN1: return "FFN1";
    case BlbViTProfileStage::GELU: return "GELU";
    case BlbViTProfileStage::FFN2: return "FFN2";
    case BlbViTProfileStage::LAYERNORM2: return "LayerNorm2";
  }
  return "unknown";
}

static const char *comm_audit_bucket_name(BlbViTCommAuditBucket bucket) {
  switch (bucket) {
    case BlbViTCommAuditBucket::QK_TO_SOFTMAX_CONVERSION:
      return "QK_to_Softmax_conversion";
    case BlbViTCommAuditBucket::SOFTMAX_TO_CKKS_CONVERSION:
      return "Softmax_to_CKKS_conversion";
    case BlbViTCommAuditBucket::SOFTMAX_REC_CONVERSION:
      return "Softmax_rec_conversion";
    case BlbViTCommAuditBucket::OTHER_CONVERSION:
      return "other_conversion";
    case BlbViTCommAuditBucket::SOFTMAX_REC:
      return "Softmax_rec_scope";
    case BlbViTCommAuditBucket::SOFTMAX_CMP:
      return "Softmax_cmp";
    case BlbViTCommAuditBucket::SOFTMAX_MUX:
      return "Softmax_mux";
  }
  return "unknown";
}

struct StageCommunication {
  string stage;
  uint64_t bytes = 0;
  uint64_t rounds = 0;
};

class CommunicationTracker {
 public:
  explicit CommunicationTracker(Bert_ckks &bert)
      : bert_(bert), last_bytes_(bert.current_comm()),
        last_rounds_(bert.current_round()) {}

  void mark(const string &stage) {
    const uint64_t now_bytes = bert_.current_comm();
    const uint64_t now_rounds = bert_.current_round();
    stages_.push_back({stage, now_bytes - last_bytes_,
                       now_rounds - last_rounds_});
    last_bytes_ = now_bytes;
    last_rounds_ = now_rounds;
  }

  const vector<StageCommunication> &stages() const { return stages_; }

 private:
  Bert_ckks &bert_;
  uint64_t last_bytes_;
  uint64_t last_rounds_;
  vector<StageCommunication> stages_;
};

static void usage(const char *name) {
  cerr << "usage: " << name
       << " --party 1|2 --port P --address HOST --model FILE"
          " --input FILE --output DIR --backend gpu|cpu"
          " --stop-after-block N\n";
}

static Args parse_args(int argc, char **argv) {
  Args args;
  auto next = [&](int &i, const string &key) -> string {
    if (i + 1 >= argc) throw invalid_argument("missing value for " + key);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const string key = argv[i];
    if (key == "--party") args.party = stoi(next(i, key));
    else if (key == "--port") args.port = stoi(next(i, key));
    else if (key == "--address") args.address = next(i, key);
    else if (key == "--model") args.model = next(i, key);
    else if (key == "--input") args.input = next(i, key);
    else if (key == "--output") args.output = next(i, key);
    else if (key == "--backend") args.backend = next(i, key);
    else if (key == "--logical-tokens") args.logical_tokens = stoi(next(i, key));
    else if (key == "--physical-tokens") args.physical_tokens = stoi(next(i, key));
    else if (key == "--stop-after-block")
      args.stop_after_block = stoi(next(i, key));
    else if (key == "--help" || key == "-h") {
      usage(argv[0]);
      exit(0);
    } else {
      throw invalid_argument("unknown option " + key);
    }
  }
  if (args.party != sci::ALICE && args.party != sci::BOB)
    throw invalid_argument("--party must be 1 (Alice) or 2 (Bob)");
  if (args.logical_tokens != kLogicalTokens ||
      args.physical_tokens != kPhysicalTokens)
    throw invalid_argument(
        "faithful ViT-B/16 baseline requires logical_tokens=197 and "
        "physical_tokens=256");
  if (args.backend != "gpu" && args.backend != "device" &&
      args.backend != "cpu" && args.backend != "host")
    throw invalid_argument("--backend must be gpu or cpu");
  if (args.stop_after_block < 1 || args.stop_after_block > kLayers)
    throw invalid_argument("--stop-after-block must be in [1, 12]");
  return args;
}

static uint64_t ring_mask(int ell) {
  if (ell <= 0 || ell >= 64) throw invalid_argument("invalid ring width");
  return (uint64_t{1} << ell) - 1;
}

static uint64_t quantize(double value, int scale, int ell = kBitwidth) {
  if (!isfinite(value) || scale < 0 || ell <= 0 || ell >= 64)
    throw invalid_argument("invalid fixed-point value");
  const long double scaled = ldexp(static_cast<long double>(value), scale);
  if (scaled > static_cast<long double>(numeric_limits<int64_t>::max()) ||
      scaled < static_cast<long double>(numeric_limits<int64_t>::min()))
    throw out_of_range("fixed-point checkpoint value exceeds int64");
  const int64_t rounded = static_cast<int64_t>(llround(scaled));
  return static_cast<uint64_t>(rounded) & ring_mask(ell);
}

static vector<uint64_t> quantized(const vector<double> &values, int scale,
                                  int ell = kBitwidth) {
  vector<uint64_t> result(values.size());
  transform(values.begin(), values.end(), result.begin(),
            [scale, ell](double value) { return quantize(value, scale, ell); });
  return result;
}

static vector<vector<uint64_t>> matrix_input_major(
    const vector<double> &output_major, int output_rows, int input_dim,
    int output_dim, int scale) {
  if (output_rows != output_dim ||
      output_major.size() != static_cast<size_t>(output_rows) * input_dim)
    throw invalid_argument("ViT matrix shape mismatch");
  vector<vector<uint64_t>> result(input_dim,
                                  vector<uint64_t>(output_dim, 0));
  for (int out = 0; out < output_dim; ++out)
    for (int in = 0; in < input_dim; ++in)
      result[in][out] = quantize(output_major[out * input_dim + in], scale);
  return result;
}

static vector<vector<vector<uint64_t>>> qkv_input_major(
    const vector<double> &output_major, int dim, int heads, int head_dim,
    int scale) {
  if (output_major.size() != static_cast<size_t>(dim) * dim ||
      heads * head_dim != dim)
    throw invalid_argument("ViT QKV matrix shape mismatch");
  vector<vector<vector<uint64_t>>> result(
      heads, vector<vector<uint64_t>>(dim, vector<uint64_t>(head_dim, 0)));
  for (int head = 0; head < heads; ++head)
      for (int out = 0; out < head_dim; ++out)
      for (int in = 0; in < dim; ++in)
        result[head][in][out] =
            quantize(output_major[(head * head_dim + out) * dim + in], scale);
  return result;
}

static BertModel convert_vit_to_blb_model(const VitModel &vit) {
  if (vit.architecture != "pre_norm_vit" || vit.norm_order != "pre" ||
      vit.image_size != kImageSize || vit.patch_size != kPatchSize ||
      vit.channels != kChannels || vit.tokens != kLogicalTokens ||
      vit.dim != kDim || vit.heads != kHeads || vit.ff_dim != kFfDim ||
      vit.layers != kLayers || vit.classes != 1000 || !vit.qkv_bias) {
    throw runtime_error(
        "checkpoint is not the declared ViT-B/16 pre-norm full model");
  }
  if (vit.scale_bits != kScale) {
    throw runtime_error(
        "canonical BLB-ViT baseline requires scale_bits=" +
        to_string(kScale) + "; refusing a precision change");
  }
  const int patch_dim = vit.channels * vit.patch_size * vit.patch_size;
  BertModel model;
  model.w_q.resize(kLayers);
  model.w_k.resize(kLayers);
  model.w_v.resize(kLayers);
  model.w_o.resize(kLayers);
  model.w_i_1.resize(kLayers);
  model.w_i_2.resize(kLayers);
  model.b_q.resize(kLayers);
  model.b_k.resize(kLayers);
  model.b_v.resize(kLayers);
  model.b_o.resize(kLayers);
  model.b_i_1.resize(kLayers);
  model.b_i_2.resize(kLayers);
  model.w_ln_1.resize(kLayers);
  model.b_ln_1.resize(kLayers);
  model.w_ln_2.resize(kLayers);
  model.b_ln_2.resize(kLayers);

  for (int layer = 0; layer < kLayers; ++layer) {
    const VitBlockModel &src = vit.blocks.at(layer);
    // ViT graph adaptation only: preserve BLB's QKV weight/bias fractional
    // bits. The exact public 1/sqrt(head_dim) factor is applied after Q in
    // linear_1 so quantization retains all six original Q fractional bits.
    model.w_q[layer] = qkv_input_major(
        src.wq, kDim, kHeads, kHeadDim, kQkvWeightScale);
    model.w_k[layer] = qkv_input_major(
        src.wk, kDim, kHeads, kHeadDim, kQkvWeightScale);
    model.w_v[layer] = qkv_input_major(
        src.wv, kDim, kHeads, kHeadDim, kQkvWeightScale);
    model.b_q[layer].resize(kHeads);
    model.b_k[layer].resize(kHeads);
    model.b_v[layer].resize(kHeads);
    for (int head = 0; head < kHeads; ++head) {
      model.b_q[layer][head].resize(kHeadDim);
      model.b_k[layer][head].resize(kHeadDim);
      model.b_v[layer][head].resize(kHeadDim);
      for (int out = 0; out < kHeadDim; ++out) {
        model.b_q[layer][head][out] =
            quantize(src.q_bias[head * kHeadDim + out], kQkvBiasScale);
        model.b_k[layer][head][out] =
            quantize(src.k_bias[head * kHeadDim + out], kQkvBiasScale);
        model.b_v[layer][head][out] =
            quantize(src.v_bias[head * kHeadDim + out], kQkvBiasScale);
      }
    }
    model.w_o[layer] = matrix_input_major(src.wo, kDim, kDim, kDim, 6);
    model.b_o[layer] = quantized(src.bo, 12);
    model.w_i_1[layer] = matrix_input_major(src.ff1_weight, kFfDim, kDim,
                                             kFfDim, 6);
    model.b_i_1[layer] = quantized(src.ff1_bias, 11);
    model.w_i_2[layer] = matrix_input_major(src.ff2_weight, kDim, kFfDim,
                                             kDim, 5);
    model.b_i_2[layer] = quantized(src.ff2_bias, 9);
    model.w_ln_1[layer] = quantized(src.ln1_weight, 12);
    model.b_ln_1[layer] = quantized(src.ln1_bias, 12);
    model.w_ln_2[layer] = quantized(src.ln2_weight, 12);
    model.b_ln_2[layer] = quantized(src.ln2_bias, 12);
  }

  // Patch projection is a ViT-specific first linear layer, but its evaluator
  // path is the original BLB BSGS2/CKKS path.  BertModel does not own this
  // layer, so it is prepared by the driver after Bert_ckks initialization.
  model.w_p = matrix_input_major(vit.patch_weight, kDim, patch_dim, kDim, 6);
  model.b_p = quantized(vit.patch_bias, 12);
  model.w_c = matrix_input_major(vit.classifier_weight, vit.classes, kDim,
                                  vit.classes, 12);
  model.b_c = quantized(vit.classifier_bias, 12);
  if (model.w_c.size() != static_cast<size_t>(kDim))
    throw runtime_error("ViT classifier conversion failed");
  return model;
}

static FixArray input_parameter_share(Bert_ckks &bert,
                                      const vector<uint64_t> &parameter,
                                      int rows, int columns, int ell,
                                      int scale) {
  if (parameter.size() != static_cast<size_t>(columns))
    throw invalid_argument("parameter input shape mismatch");
  vector<uint64_t> expanded(static_cast<size_t>(rows) * columns, 0);
  if (bert.party == sci::ALICE) {
    for (int row = 0; row < rows; ++row)
      copy(parameter.begin(), parameter.end(), expanded.begin() +
                                                   static_cast<size_t>(row) * columns);
  }
  return bert.nonlinear.fpmath[0]->fix->input(
      sci::ALICE, static_cast<int>(expanded.size()), expanded.data(), true, ell,
      scale);
}

static FixArray input_layernorm_share(Bert_ckks &bert,
                                      const vector<vector<uint64_t>> &params,
                                      int layer, int rows, int columns) {
  if (bert.party == sci::ALICE &&
      (layer < 0 || static_cast<size_t>(layer) >= params.size() ||
       params[layer].size() != static_cast<size_t>(columns)))
    throw runtime_error("ViT LayerNorm parameter is missing");
  vector<uint64_t> row_parameter(columns, 0);
  if (bert.party == sci::ALICE) row_parameter = params[layer];
  return input_parameter_share(bert, row_parameter, rows, columns,
                               kBitwidth, kScale);
}

static FixArray input_vector_share(Bert_ckks &bert,
                                   const vector<double> &values, int rows,
                                   int scale) {
  vector<uint64_t> fixed(values.size(), 0);
  if (bert.party == sci::ALICE) fixed = quantized(values, scale);
  return input_parameter_share(bert, fixed, rows, static_cast<int>(values.size()),
                               kBitwidth, scale);
}

static vector<uint64_t> image_to_patch_input(const VitInput &input,
                                             int physical_tokens) {
  if (input.image_size != kImageSize || input.channels != kChannels ||
      input.pixels.size() != static_cast<size_t>(kImageSize) * kImageSize *
                                  kChannels)
    throw runtime_error("ViT input image shape mismatch");
  const int patches_per_side = kImageSize / kPatchSize;
  const int patch_dim = kChannels * kPatchSize * kPatchSize;
  vector<uint64_t> result(static_cast<size_t>(physical_tokens) * kDim, 0);
  for (int patch = 0; patch < patches_per_side * patches_per_side; ++patch) {
    const int patch_y = (patch / patches_per_side) * kPatchSize;
    const int patch_x = (patch % patches_per_side) * kPatchSize;
    const int token = patch + 1;  // token 0 is CLS
    int offset = 0;
    for (int channel = 0; channel < kChannels; ++channel)
      for (int y = 0; y < kPatchSize; ++y)
        for (int x = 0; x < kPatchSize; ++x) {
          const size_t source =
              (static_cast<size_t>(channel) * kImageSize + patch_y + y) *
                  kImageSize + patch_x + x;
          result[static_cast<size_t>(token) * patch_dim + offset++] =
              quantize(input.pixels[source], 12);
        }
  }
  return result;
}

static void decode_attention_output(Bert_ckks &bert,
                                     const vector<vector<uint64_t>> &packed,
                                     FixArray &qk, FixArray &v) {
  const int tokens = bert.linear.data_lin1_0.image_size;
  const size_t slot = bert.linear.data_lin1_0.slot_count;
  const size_t qk_values = static_cast<size_t>(tokens) * tokens * kHeads;
  const size_t v_values = static_cast<size_t>(tokens) * kDim;
  // This is the existing BLB half-decode boundary: each returned share holds
  // both CKKS halves (degree values), not one encoder slot vector.  Retain
  // that compatibility packing and only derive its count from the ViT shape.
  const size_t decoded_values = 2 * slot;
  const size_t expected = (qk_values + v_values) / decoded_values;
  if (packed.size() != expected)
    throw runtime_error("BLB QK/V ciphertext count does not match physical ViT layout");
  qk = FixArray(bert.party, static_cast<int>(qk_values), true, kBitwidth,
                kScale);
  v = FixArray(bert.party, static_cast<int>(v_values), true, kBitwidth,
               kAttentionVScale);
  size_t offset = 0;
  for (const auto &ct_share : packed) {
    if (ct_share.size() != decoded_values)
      throw runtime_error("BLB QK/V half-decode size mismatch");
    for (uint64_t value : ct_share) {
      if (offset >= qk_values + v_values)
        throw runtime_error("BLB QK/V output exceeds physical ViT layout");
      if (offset < qk_values) qk.data[offset] = value;
      else v.data[offset - qk_values] = value;
      ++offset;
    }
  }
  if (offset != qk_values + v_values)
    throw runtime_error("BLB QK/V output has incomplete physical padding");
}

static uint64_t counter_delta(uint64_t now, uint64_t before) {
  if (now < before) throw runtime_error("profiling counter moved backwards");
  return now - before;
}

static void write_block_profile(const string &dir, int party,
                                const vector<BlockProfile> &rows) {
  ofstream out(dir + "/block_profile_party" + to_string(party) + ".csv");
  if (!out) throw runtime_error("cannot write block profile");
  out << "block,secure_end_to_end_ms,block_online_time_ms,"
         "block_online_comm_bytes,he_time_ms,qkv_he_ms,qk_transpose_he_ms,"
         "softmax_v_he_ms,linear2_he_ms,linear1_he_ms,"
         "linear2_softmax_v_he_ms,linear3_he_ms,linear4_he_ms,"
         "linear_he_linear1_4_ms,linear_he_including_mul_v_ms,"
         "nonlinear_softmax_bucket_ms,nonlinear_layernorm_bucket_ms,"
         "nonlinear_gelu_core_ms,conversion_time_ms,h2d_time_ms,"
         "plaintext_h2d_time_ms,plaintext_device_encode_time_ms,"
         "plaintext_device_release_time_ms,"
         "ciphertext_h2d_time_ms,d2h_time_ms,"
         "he_ct_pt_multiply,he_ct_ct_multiply,he_rotations,he_rescales,"
         "ckks_to_mpc,mpc_to_ckks,field_to_ring_calls,ring_to_field_calls,"
         "conversion_ciphertexts,conversion_bytes,"
         "ct_transfer_bytes,mpc_communication_bytes,total_communication_bytes,"
         "communication_rounds\n";
  out << fixed << setprecision(6);
  for (const BlockProfile &r : rows) {
    out << r.block << ',' << r.secure_end_to_end_ms << ','
        << r.block_online_time_ms << ',' << r.block_online_comm_bytes << ','
        << r.he_time_ms << ',' << r.qkv_he_ms << ','
        << r.qk_transpose_he_ms << ',' << r.softmax_v_he_ms << ','
        << r.linear2_he_ms << ',' << r.linear1_he_ms << ','
        << r.linear2_softmax_v_he_ms << ',' << r.linear3_he_ms << ','
        << r.linear4_he_ms << ','
        << r.linear_he_linear1_4_ms << ',' << r.linear_he_including_mul_v_ms
        << ',' << r.nonlinear_softmax_bucket_ms << ','
        << r.nonlinear_layernorm_bucket_ms << ','
        << r.nonlinear_gelu_core_ms << ',' << r.conversion_time_ms << ','
        << r.h2d_time_ms << ',' << r.plaintext_h2d_time_ms << ','
        << r.plaintext_device_encode_time_ms << ','
        << r.plaintext_device_release_time_ms << ','
        << r.ciphertext_h2d_time_ms << ',' << r.d2h_time_ms << ','
        << r.he_ct_pt_multiply << ',' << r.he_ct_ct_multiply << ','
        << r.he_rotations << ',' << r.he_rescales << ',' << r.ckks_to_mpc
        << ',' << r.mpc_to_ckks << ',' << r.field_to_ring_calls << ','
        << r.ring_to_field_calls << ',' << r.conversion_ciphertexts << ','
        << r.conversion_bytes << ',' << r.ct_transfer_bytes << ','
        << r.mpc_communication_bytes << ',' << r.total_communication_bytes
        << ',' << r.communication_rounds << '\n';
  }
}

static void write_online_stage_profiles(
    const string &dir, int party, const vector<BlockProfile> &profiles) {
  ofstream out(dir + "/block_online_profile_party" + to_string(party) +
               ".csv");
  if (!out) throw runtime_error("cannot write block online profile");
  out << "block,stage,wall_ms,send_bytes,rounds\n";
  out << fixed << setprecision(6);
  for (const BlockProfile &profile : profiles) {
    for (size_t index = 0; index < kBlbViTProfileStageCount; ++index) {
      const auto stage = static_cast<BlbViTProfileStage>(index);
      const auto &metric = profile.online_stages[index];
      out << profile.block << ',' << profile_stage_name(stage) << ','
          << metric.wall_ms << ',' << metric.send_bytes << ','
          << metric.rounds << '\n';
    }
    out << profile.block << ",BLOCK_ONLINE_TIME,"
        << profile.block_online_time_ms << ",0,0\n";
    out << profile.block << ",BLOCK_ONLINE_COMM,0,"
        << profile.block_online_comm_bytes << ",0\n";
  }
}

static void write_comm_audit_profile(const string &dir, int party,
                                     const vector<BlockProfile> &rows) {
  ofstream out(dir + "/block_comm_audit_party" + to_string(party) +
               ".csv");
  if (!out) throw runtime_error("cannot write communication audit profile");
  out << "block,bucket,send_bytes,calls\n";
  for (const BlockProfile &profile : rows) {
    for (size_t index = 0; index < kBlbViTCommAuditBucketCount; ++index) {
      const auto bucket = static_cast<BlbViTCommAuditBucket>(index);
      const auto &metric = profile.comm_audit[index];
      out << profile.block << ',' << comm_audit_bucket_name(bucket) << ','
          << metric.send_bytes << ',' << metric.calls << '\n';
    }
  }
}

static void write_communication_breakdown(
    const string &dir, int party, const vector<StageCommunication> &stages,
    uint64_t initialization_bytes, uint64_t initialization_rounds,
    uint64_t online_bytes, uint64_t online_rounds) {
  ofstream out(dir + "/communication_breakdown_party" + to_string(party) +
               ".csv");
  if (!out) throw runtime_error("cannot write communication breakdown");
  out << "stage,bytes,rounds\n";
  out << "initialization," << initialization_bytes << ','
      << initialization_rounds << '\n';
  for (const auto &stage : stages)
    out << stage.stage << ',' << stage.bytes << ',' << stage.rounds << '\n';
  out << "online_total," << online_bytes << ',' << online_rounds << '\n';
  out << "total," << initialization_bytes + online_bytes << ','
      << initialization_rounds + online_rounds << '\n';
}

static void write_latency_summary(const string &dir, int party,
                                  double secure_seconds,
                                  uint64_t initialization_bytes,
                                  uint64_t online_bytes,
                                  uint64_t online_rounds,
                                  uint64_t total_conversions,
                                  uint64_t total_ciphertexts) {
  ofstream out(dir + "/latency_party" + to_string(party) + ".csv");
  if (!out) throw runtime_error("cannot write latency summary");
  out << "metric,value\n";
  out << fixed << setprecision(6);
  out << "secure_end_to_end_latency_s," << secure_seconds << '\n';
  out << "secure_end_to_end_latency_ms," << secure_seconds * 1000.0 << '\n';
  out << "initialization_communication_bytes," << initialization_bytes << '\n';
  out << "online_communication_bytes," << online_bytes << '\n';
  out << "total_communication_bytes," << initialization_bytes + online_bytes
      << '\n';
  out << "online_communication_rounds," << online_rounds << '\n';
  out << "ckks_to_mpc_conversions," << total_conversions << '\n';
  out << "conversion_ciphertexts," << total_ciphertexts << '\n';
}

static void write_execution_totals(const string &dir, int party,
                                   double setup_seconds,
                                   double secure_seconds,
                                   const vector<BlockProfile> &profiles,
                                   const Bert_ckks &bert,
                                   uint64_t initialization_bytes,
                                   uint64_t online_bytes,
                                   uint64_t online_rounds) {
  double block_wall_ms = 0.0;
  double nonlinear_mpc_bucket_ms = 0.0;
  for (const BlockProfile &profile : profiles) {
    block_wall_ms += profile.block_online_time_ms;
    nonlinear_mpc_bucket_ms += profile.nonlinear_softmax_bucket_ms +
                               profile.nonlinear_layernorm_bucket_ms +
                               profile.nonlinear_gelu_core_ms;
  }
  ofstream out(dir + "/execution_totals_party" + to_string(party) + ".csv");
  if (!out) throw runtime_error("cannot write execution totals");
  out << "metric,value,scope\n" << fixed << setprecision(6);
  out << "setup_offline_s," << setup_seconds << ",model_setup\n";
  out << "setup_plus_secure_s," << setup_seconds + secure_seconds
      << ",measured_setup_plus_secure_graph\n";
  out << "secure_end_to_end_s," << secure_seconds << ",full_secure_graph\n";
  out << "transformer_blocks_wall_s," << block_wall_ms / 1000.0
      << ",mutually_exclusive_stage_timeline\n";
  out << "he_instrumented_s," << bert.he_compute_la
      << ",original_blb_internal_counter\n";
  out << "conversion_instrumented_s," << bert.conversion_la
      << ",original_blb_internal_counter\n";
  out << "h2d_instrumented_s,"
      << bert.h2d_time + bert.linear.plaintext_h2d_time +
             bert.linear.ciphertext_h2d_time
      << ",ciphertext_plus_plaintext_existing_phantom_boundaries\n";
  out << "plaintext_h2d_instrumented_s," << bert.linear.plaintext_h2d_time
      << ",real_checkpoint_packed_plaintexts_to_phantom\n";
  out << "plaintext_device_encode_instrumented_s,"
      << bert.linear.plaintext_device_encode_time
      << ",official_phantom_encoder_real_checkpoint_packed_slots\n";
  out << "plaintext_device_release_instrumented_s,"
      << bert.linear.plaintext_device_release_time
      << ",operator_scoped_phantom_plaintext_lifetime_cleanup\n";
  out << "ciphertext_h2d_instrumented_s,"
      << bert.h2d_time + bert.linear.ciphertext_h2d_time
      << ",ciphertexts_to_existing_phantom_boundaries\n";
  out << "d2h_instrumented_s,"
      << bert.d2h_time + bert.linear.ciphertext_d2h_time
      << ",ciphertext_existing_phantom_boundaries\n";
  out << "mpc_nonlinear_bucket_s," << nonlinear_mpc_bucket_ms / 1000.0
      << ",softmax_layernorm_gelu_wall_buckets_not_additive_with_he_conversion\n";
  out << "he_ct_pt_multiply," << bert.linear.ct_pt_multiply_count
      << ",full_secure_graph\n";
  out << "he_ct_ct_multiply," << bert.linear.ct_ct_multiply_count
      << ",full_secure_graph\n";
  out << "he_rotations," << bert.linear.rotation_count
      << ",full_secure_graph\n";
  out << "he_rescales,"
      << bert.linear.rescale_count + bert.he_rescale_count
      << ",full_secure_graph\n";
  out << "ckks_to_mpc," << bert.ckks_to_mpc_conversions
      << ",full_secure_graph\n";
  out << "mpc_to_ckks," << bert.mpc_to_ckks_conversions
      << ",full_secure_graph\n";
  out << "field_to_ring_calls," << bert.field_to_ring_calls
      << ",full_secure_graph\n";
  out << "ring_to_field_calls," << bert.ring_to_field_calls
      << ",full_secure_graph\n";
  out << "conversion_ciphertexts," << bert.conversion_ciphertexts
      << ",full_secure_graph\n";
  out << "initialization_communication_bytes," << initialization_bytes
      << ",this_party\n";
  out << "online_communication_bytes," << online_bytes << ",this_party\n";
  out << "total_communication_bytes," << initialization_bytes + online_bytes
      << ",this_party\n";
  out << "online_communication_rounds," << online_rounds << ",this_party\n";
}

static LOCATION parse_backend(const string &backend) {
  if (backend == "gpu" || backend == "device") return LOCATION::DEVICE;
  return LOCATION::HOST;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Args args = parse_args(argc, argv);
    cout << "TRUNCATION_MODE mode=original_blb_interactive" << endl;
    filesystem::create_directories(args.output);
    const LOCATION backend = parse_backend(args.backend);

    VitModel vit;
    BertModel blb_model;
    if (args.party == sci::ALICE) {
      vit = load_vit_model(args.model);
      blb_model = convert_vit_to_blb_model(vit);
    }

    const auto setup_start = chrono::steady_clock::now();
    Bert_ckks bert(args.party, args.port, args.address, backend,
                   args.party == sci::ALICE ? &blb_model : nullptr,
                   args.physical_tokens, false, args.stop_after_block);
    double setup_seconds =
        chrono::duration<double>(chrono::steady_clock::now() - setup_start)
            .count();
    bert.io->sync();

    if (args.party == sci::ALICE) {
      if (vit.tokens != kLogicalTokens || vit.dim != kDim ||
          vit.classes != 1000)
        throw runtime_error("ViT model metadata changed during baseline setup");
    }

    // Patch embedding is the one ViT-specific linear layer outside a
    // Transformer block.  It deliberately goes through the original BSGS2
    // CKKS path, using the same scale schedule as BLB's BERT linear2.
    vector<vector<uint64_t>> patch_weight;
    vector<uint64_t> patch_bias(kDim, 0);
    PreprocessParams_2 patch_params;
    FCMetadata classifier_metadata;
    classifier_metadata.filter_h = kDim;
    classifier_metadata.filter_w = kPhysicalClasses;
    classifier_metadata.image_size = args.physical_tokens;
    classifier_metadata.slot_count = bert.linear.he2->num_slots;
    PreprocessParams_2 classifier_params;
    const auto patch_setup_start = chrono::steady_clock::now();
    if (args.party == sci::ALICE) {
      patch_weight = blb_model.w_p;
      patch_bias = blb_model.b_p;
      patch_params = bert.linear.params_preprocessing_ct_pt(
          bert.linear.he2, kDim, kDim, kDim, patch_weight, patch_bias,
          bert.linear.data_lin2, 6, 12, 5,
          std::ldexp(1.0, bert.linear.he2->he_scale));
    }

    // Alice's patch packing is offline.  Synchronize before establishing the
    // online communication baseline so this boundary is not charged to the
    // secure inference latency.  The patch parameters themselves remain the
    // original BLB BSGS2 plaintexts.
    bert.io->sync();
    setup_seconds += chrono::duration<double>(chrono::steady_clock::now() -
                                               patch_setup_start).count();
    uint64_t initialization_bytes = bert.current_comm();
    uint64_t initialization_rounds = bert.current_round();

    FPMath *fpmath = bert.nonlinear.fpmath[0];
    CommunicationTracker comm(bert);
    comm.mark("initialization_boundary");
    const auto online_start = chrono::steady_clock::now();
    double secure_protocol_seconds = 0.0;
    const bool diagnostic_partial_classifier =
        getenv("BLB_VIT_DIAGNOSTIC_PARTIAL_CLASSIFIER") != nullptr;
    auto write_diagnostic_cls = [&](const FixArray &tensor,
                                    const string &stage) {
      if (!diagnostic_partial_classifier) return;
      if (tensor.size < kDim)
        throw runtime_error("diagnostic CLS tensor is too small");
      FixArray cls_share(bert.party, kDim, true, tensor.ell, tensor.s);
      copy(tensor.data, tensor.data + kDim, cls_share.data);
      FixArray opened = fpmath->fix->output(sci::PUBLIC, cls_share);
      ofstream out(args.output + "/diagnostic_cls_" + stage + "_party" +
                   to_string(args.party) + ".txt");
      if (!out) throw runtime_error("cannot write diagnostic CLS");
      const uint64_t mask = ring_mask(opened.ell);
      const uint64_t sign = uint64_t{1} << (opened.ell - 1);
      out << fixed << setprecision(10);
      for (int i = 0; i < kDim; ++i) {
        const uint64_t raw = opened.data[i] & mask;
        const int64_t value = (raw & sign)
                                  ? static_cast<int64_t>(
                                        raw - (uint64_t{1} << opened.ell))
                                  : static_cast<int64_t>(raw);
        out << static_cast<double>(value) /
                   static_cast<double>(uint64_t{1} << opened.s)
            << '\n';
      }
    };

    vector<uint64_t> image_patch_values(
        static_cast<size_t>(args.physical_tokens) * kDim, 0);
    if (args.party == sci::BOB) {
      image_patch_values = image_to_patch_input(load_vit_input(args.input),
                                                args.physical_tokens);
    }
    FixArray patch_input = fpmath->fix->input(
        sci::BOB, static_cast<int>(image_patch_values.size()),
        image_patch_values.data(), true, kBitwidth, kScale);
    FixArray hidden = bert.linear2_from_mpc(
        patch_input, patch_params, bert.linear.data_lin2, 0);
    patch_params = PreprocessParams_2{};
    comm.mark("patch_embedding");

    // Class token and position embeddings are model-owned Alice inputs.  The
    // public zero rows at [197, 256) are intentional physical padding needed
    // by the original full-slot BSGS packing; they are masked as keys below.
    vector<uint64_t> embedding_fixed(
        static_cast<size_t>(args.physical_tokens) * kDim, 0);
    if (args.party == sci::ALICE) {
      for (int feature = 0; feature < kDim; ++feature) {
        embedding_fixed[feature] =
            (quantize(vit.class_token[feature], 12) +
             quantize(vit.position[feature], 12)) & ring_mask(kBitwidth);
      }
      for (int token = 1; token < kLogicalTokens; ++token)
        for (int feature = 0; feature < kDim; ++feature)
          embedding_fixed[static_cast<size_t>(token) * kDim + feature] =
              quantize(vit.position[static_cast<size_t>(token) * kDim +
                                    feature],
                       12);
    }
    FixArray embedding_share = fpmath->fix->input(
        sci::ALICE, static_cast<int>(embedding_fixed.size()), embedding_fixed.data(),
        true, kBitwidth, kScale);
    hidden = fpmath->fix->add(hidden, embedding_share);
    write_diagnostic_cls(hidden, "embedding");
    comm.mark("position_and_cls_embedding");
    secure_protocol_seconds +=
        chrono::duration<double>(chrono::steady_clock::now() - online_start)
            .count();

    vector<int64_t> attention_mask(args.physical_tokens, -10000);
    fill(attention_mask.begin(), attention_mask.begin() + kLogicalTokens, 0);
    bert.set_public_attention_mask(attention_mask);
    cout << "BLB_VIT_BASELINE protocol=original_blb_ckks_mpc"
         << " architecture=vit_b16_full pre_norm=1 logical_tokens="
         << kLogicalTokens << " physical_tokens=" << kPhysicalTokens
         << " backend=" << (backend == LOCATION::DEVICE ? "gpu" : "cpu")
         << " layers=" << kLayers << endl;
    cout << "BLB_VIT_BASELINE boundary=padding logical_rows=197 physical_rows=256"
         << " padding_is_public_key_mask=1 alternate_protocol=0" << endl;

    vector<BlockProfile> profiles;
    profiles.reserve(kLayers);
    auto write_run_metadata = [&](int prediction, bool prediction_ready) {
      ofstream meta(args.output + "/run_metadata_party" +
                    to_string(args.party) + ".txt");
      meta << "protocol=original_blb_ckks_mpc\n"
           << "model=ViT-B/16\n"
           << "architecture=pre_norm_vit\n"
           << "logical_tokens=197\nphysical_tokens=256\n"
           << "layers=12\n"
           << "executed_transformer_blocks=" << args.stop_after_block << '\n'
           << "partial_run=" << (args.stop_after_block < kLayers ? 1 : 0)
           << '\n'
           << "patch_embedding=original_bsgs2_ckks\n"
           << "attention_score_scale=1/sqrt(head_dim)=1/8\n"
           << "attention_scale=public_ckks_q_scale_metadata_div8\n"
           << "fixed_point_ring_bits=37\nfixed_point_scale_bits=12\n"
           << "softmax=original_blb_softmax_he_mpc\n"
           << "layernorm=original_blb_layer_norm_mpc_ckks_affine\n"
           << "gelu=original_blb_gelu_ckks_mpc\n"
           << "classifier=cls_only_original_bsgs2_ckks\n"
           << "classifier_logical_outputs=1000\n"
           << "classifier_physical_outputs=1024\n"
           << "offline_materialization=streamed_per_layer_same_blb_plaintexts\n"
           << "offline_cache_lifetime=current_layer_only\n"
           << "communication_scope=this_party; aggregate=party1_plus_party2\n"
           << "backend=" << (backend == LOCATION::DEVICE ? "gpu" : "cpu")
           << "\nsetup_seconds=" << fixed << setprecision(6) << setup_seconds
           << '\n';
      if (prediction_ready) meta << "prediction=" << prediction << '\n';
    };
    for (int layer = 0; layer < args.stop_after_block; ++layer) {
      // Materialize exactly the original BLB plaintext packing for the layer
      // that is about to run, then release it after use.  Keeping all twelve
      // ViT/256 caches resident exceeds host memory; this scheduling changes
      // neither the online graph nor any cryptographic parameter.
      const auto layer_setup_start = chrono::steady_clock::now();
      const uint64_t layer_setup_comm0 = bert.current_comm();
      const uint64_t layer_setup_round0 = bert.current_round();
      if (args.party == sci::ALICE)
        bert.linear.weights_preprocess_layer(blb_model, layer);
      bert.io->sync();
      setup_seconds += chrono::duration<double>(chrono::steady_clock::now() -
                                                 layer_setup_start).count();
      initialization_bytes += bert.current_comm() - layer_setup_comm0;
      initialization_rounds += bert.current_round() - layer_setup_round0;
      comm.mark("block_" + to_string(layer) + "_offline_preprocessing");

      const uint64_t comm0 = bert.current_comm();
      const uint64_t rounds0 = bert.current_round();
      const uint64_t ctpt0 = bert.linear.ct_pt_multiply_count;
      const uint64_t ctct0 = bert.linear.ct_ct_multiply_count;
      const uint64_t rot0 = bert.linear.rotation_count;
      const uint64_t resc0 = bert.linear.rescale_count;
      const uint64_t he_resc0 = bert.he_rescale_count;
      const uint64_t c2m0 = bert.ckks_to_mpc_conversions;
      const uint64_t m2c0 = bert.mpc_to_ckks_conversions;
      const uint64_t f2r0 = bert.field_to_ring_calls;
      const uint64_t r2f0 = bert.ring_to_field_calls;
      const uint64_t convct0 = bert.conversion_ciphertexts;
      const uint64_t convcomm0 = static_cast<uint64_t>(bert.conversion_comm);
      const uint64_t ctcomm0 = static_cast<uint64_t>(bert.ct_transfer_comm);
      const double convtime0 = bert.conversion_la;
      const double h2d0 = bert.h2d_time;
      const double plain_h2d0 = bert.linear.plaintext_h2d_time;
      const double plain_device_encode0 =
          bert.linear.plaintext_device_encode_time;
      const double plain_device_release0 =
          bert.linear.plaintext_device_release_time;
      const double linear_ct_h2d0 = bert.linear.ciphertext_h2d_time;
      const double d2h0 = bert.d2h_time;
      const double linear_ct_d2h0 = bert.linear.ciphertext_d2h_time;

      BlockProfile profile;
      profile.block = layer;
      bert.profile_start(BlbViTProfileStage::LAYERNORM1);

      // ViT pre-norm LayerNorm uses BLB's existing square/mean/rsqrt MPC
      // path followed by the existing CKKS affine path.
      const auto ln1_start = chrono::steady_clock::now();
      FixArray ln1_w = input_layernorm_share(
          bert, bert.linear.w_ln_1, layer, args.physical_tokens, kDim);
      FixArray ln1_b = input_layernorm_share(
          bert, bert.linear.b_ln_1, layer, args.physical_tokens, kDim);
      map<char *, FixArray> ln1 =
          bert.layer_norm_part1_real(hidden, ln1_w, ln1_b);
      profile.nonlinear_layernorm_bucket_ms += chrono::duration<double, milli>(
          chrono::steady_clock::now() - ln1_start).count();
      comm.mark("block_" + to_string(layer) + "_layer_norm_1");
      bert.profile_switch(BlbViTProfileStage::QKV);

      // Q/K/V: preserve BLB FineGrainFusion by feeding the centered,
      // inverse-sigma, gamma, and beta shares directly into the original
      // fused LayerNorm-affine + pp_1/BSGS QKV kernel.  The pre-norm residual
      // stays on the external hidden branch and is not re-exported from CKKS.
      bert.attention_input_residual_share = hidden;
      bert.attention_input_residual_ready = true;
      bert.pre_norm_attention_output_only = true;
      const double he_before_qkv = bert.he_compute_la;
      const auto qkv_packed = bert.layer_norm_part2_attention(ln1, layer);
      profile.linear1_he_ms = (bert.he_compute_la - he_before_qkv) * 1000.0;
      comm.mark("block_" + to_string(layer) + "_qkv_projection_qkt");

      FixArray qk, v;
      decode_attention_output(bert, qkv_packed, qk, v);
      bert.attention_v_share = std::move(v);
      bert.attention_v_ready = true;
      bert.profile_switch(BlbViTProfileStage::SOFTMAX);

      const auto softmax_start = chrono::steady_clock::now();
      qk = bert.softmax_he(qk);
      profile.nonlinear_softmax_bucket_ms = chrono::duration<double, milli>(
          chrono::steady_clock::now() - softmax_start).count();
      comm.mark("block_" + to_string(layer) + "_softmax");

      // Original BLB Softmax*V and output-projection path.  The pre-norm
      // branch exits at the same CKKS residual boundary that ViT requires;
      // it does not replace any internal operator.
      const double he_before_linear2 = bert.he_compute_la;
      auto attention_result =
          bert.Softmax_v_linear_layer_norm_part1(qk, layer);
      profile.linear2_softmax_v_he_ms =
          (bert.he_compute_la - he_before_linear2) * 1000.0;
      bert.pre_norm_attention_output_only = false;
      hidden = attention_result.at("residual");
      comm.mark("block_" + to_string(layer) + "_softmax_v_output_projection");
      bert.profile_switch(BlbViTProfileStage::LAYERNORM2);

      // Second pre-norm LayerNorm, again with the original BLB protocol.
      const auto ln2_start = chrono::steady_clock::now();
      FixArray ln2_w = input_layernorm_share(
          bert, bert.linear.w_ln_2, layer, args.physical_tokens, kDim);
      FixArray ln2_b = input_layernorm_share(
          bert, bert.linear.b_ln_2, layer, args.physical_tokens, kDim);
      map<char *, FixArray> ln2 =
          bert.layer_norm_part1_real(hidden, ln2_w, ln2_b);
      profile.nonlinear_layernorm_bucket_ms += chrono::duration<double, milli>(
          chrono::steady_clock::now() - ln2_start).count();
      comm.mark("block_" + to_string(layer) + "_layer_norm_2");
      bert.profile_switch(BlbViTProfileStage::FFN1);

      // FFN1 + the original CKKS polynomial preparation for GELU.
      bert.ffn_residual_share = hidden;
      bert.ffn_residual_ready = true;
      bert.pre_norm_ffn_residual_from_external = true;
      const double he_before_linear3 = bert.he_compute_la;
      // ln_part2_linear3_gelu_part1 consumes the original BLB LayerNorm
      // MPC map (centered share + inverse-sigma share + parameters).  The
      // CKKS affine result above is the ViT FFN input, but passing that flat
      // tensor would drop the original BLB LayerNorm hand-off.
      auto ffn1 = bert.ln_part2_linear3_gelu_part1(ln2, layer);
      profile.linear3_he_ms = (bert.he_compute_la - he_before_linear3) * 1000.0;
      bert.pre_norm_ffn_residual_from_external = false;
      comm.mark("block_" + to_string(layer) + "_ffn1");
      bert.profile_switch(BlbViTProfileStage::GELU);

      const auto gelu_start = chrono::steady_clock::now();
      FixArray gelu = bert.gelu_nonlinear(ffn1);
      profile.nonlinear_gelu_core_ms = chrono::duration<double, milli>(
          chrono::steady_clock::now() - gelu_start).count();
      comm.mark("block_" + to_string(layer) + "_gelu");
      bert.profile_switch(BlbViTProfileStage::FFN2);

      // FFN2 + residual at ViT's second pre-norm boundary.
      bert.ffn_residual_share = hidden;
      bert.ffn_residual_ready = true;
      bert.pre_norm_ffn_output_only = true;
      const double he_before_linear4 = bert.he_compute_la;
      auto ffn2_result = bert.linear4_add_layernorm_part1(gelu, layer);
      profile.linear4_he_ms = (bert.he_compute_la - he_before_linear4) * 1000.0;
      bert.pre_norm_ffn_output_only = false;
      hidden = ffn2_result.at("residual");
      write_diagnostic_cls(hidden, "block_" + to_string(layer));
      comm.mark("block_" + to_string(layer) + "_ffn2");

      bert.profile_stop();
      profile.block_online_time_ms = 0.0;
      profile.block_online_comm_bytes = 0;
      for (size_t stage_index = 0;
           stage_index < kBlbViTProfileStageCount; ++stage_index) {
        const auto stage = static_cast<BlbViTProfileStage>(stage_index);
        profile.online_stages[stage_index] = bert.profile_metric(stage);
        profile.block_online_time_ms +=
            profile.online_stages[stage_index].wall_ms;
        profile.block_online_comm_bytes +=
            profile.online_stages[stage_index].send_bytes;
      }
      profile.secure_end_to_end_ms = profile.block_online_time_ms;
      const uint64_t measured_block_comm = bert.current_comm() - comm0;
      if (profile.block_online_comm_bytes != measured_block_comm) {
        throw runtime_error("online stage communication does not cover block");
      }
      
      secure_protocol_seconds += profile.block_online_time_ms / 1000.0;
      profile.qkv_he_ms = bert.profile_he_ms(BlbViTProfileStage::QKV);
      profile.qk_transpose_he_ms =
          bert.profile_he_ms(BlbViTProfileStage::QK_TRANSPOSE);
      profile.softmax_v_he_ms =
          bert.profile_he_ms(BlbViTProfileStage::SOFTMAX_V);
      profile.linear2_he_ms =
          bert.profile_he_ms(BlbViTProfileStage::OUTPUT_PROJECTION);
      profile.linear1_he_ms = profile.qkv_he_ms + profile.qk_transpose_he_ms;
      profile.linear2_softmax_v_he_ms = profile.softmax_v_he_ms;
      profile.linear3_he_ms = bert.profile_he_ms(BlbViTProfileStage::FFN1);
      profile.linear4_he_ms = bert.profile_he_ms(BlbViTProfileStage::FFN2);
      profile.he_time_ms = profile.qkv_he_ms + profile.qk_transpose_he_ms +
                           profile.softmax_v_he_ms + profile.linear2_he_ms +
                           profile.linear3_he_ms + profile.linear4_he_ms;
      profile.linear_he_linear1_4_ms = profile.he_time_ms;
      // This field is intentionally explicit: original BLB linear2 includes
      // Softmax*V and output projection, so it is the same CKKS total with the
      // mul-v stage accounted for rather than hidden in MPC time.
      profile.linear_he_including_mul_v_ms = profile.linear_he_linear1_4_ms;
      profile.conversion_time_ms = (bert.conversion_la - convtime0) * 1000.0;
      profile.plaintext_h2d_time_ms =
          (bert.linear.plaintext_h2d_time - plain_h2d0) * 1000.0;
      profile.plaintext_device_encode_time_ms =
          (bert.linear.plaintext_device_encode_time -
           plain_device_encode0) * 1000.0;
      profile.plaintext_device_release_time_ms =
          (bert.linear.plaintext_device_release_time -
           plain_device_release0) * 1000.0;
      profile.ciphertext_h2d_time_ms =
          ((bert.h2d_time - h2d0) +
           (bert.linear.ciphertext_h2d_time - linear_ct_h2d0)) * 1000.0;
      profile.h2d_time_ms =
          ((bert.h2d_time - h2d0) +
           (bert.linear.plaintext_h2d_time - plain_h2d0) +
           (bert.linear.ciphertext_h2d_time - linear_ct_h2d0)) * 1000.0;
      profile.d2h_time_ms =
          ((bert.d2h_time - d2h0) +
           (bert.linear.ciphertext_d2h_time - linear_ct_d2h0)) * 1000.0;
      profile.he_ct_pt_multiply = counter_delta(
          bert.linear.ct_pt_multiply_count, ctpt0);
      profile.he_ct_ct_multiply = counter_delta(
          bert.linear.ct_ct_multiply_count, ctct0);
      profile.he_rotations = counter_delta(bert.linear.rotation_count, rot0);
      profile.he_rescales =
          counter_delta(bert.linear.rescale_count, resc0) +
          counter_delta(bert.he_rescale_count, he_resc0);
      profile.ckks_to_mpc = counter_delta(bert.ckks_to_mpc_conversions, c2m0);
      profile.mpc_to_ckks = counter_delta(bert.mpc_to_ckks_conversions, m2c0);
      profile.field_to_ring_calls =
          counter_delta(bert.field_to_ring_calls, f2r0);
      profile.ring_to_field_calls =
          counter_delta(bert.ring_to_field_calls, r2f0);
      profile.conversion_ciphertexts = counter_delta(
          bert.conversion_ciphertexts, convct0);
      for (size_t audit_index = 0;
           audit_index < kBlbViTCommAuditBucketCount; ++audit_index) {
        profile.comm_audit[audit_index] = bert.profile_comm_audit(
            static_cast<BlbViTCommAuditBucket>(audit_index));
      }
      const uint64_t comm1 = bert.current_comm();
      const uint64_t rounds1 = bert.current_round();
      profile.total_communication_bytes = comm1 - comm0;
      profile.communication_rounds = rounds1 - rounds0;
      profile.conversion_bytes = counter_delta(
          static_cast<uint64_t>(bert.conversion_comm), convcomm0);
      profile.ct_transfer_bytes = counter_delta(
          static_cast<uint64_t>(bert.ct_transfer_comm), ctcomm0);
      profile.mpc_communication_bytes =
          profile.total_communication_bytes >= profile.conversion_bytes +
                  profile.ct_transfer_bytes
              ? profile.total_communication_bytes - profile.conversion_bytes -
                    profile.ct_transfer_bytes
              : 0;
      profiles.push_back(profile);
      cout << "BLB_VIT_BLOCK block=" << layer
           << " secure_end_to_end_ms=" << profile.secure_end_to_end_ms
           << " he_ms=" << profile.he_time_ms
           << " softmax_bucket_ms=" << profile.nonlinear_softmax_bucket_ms
           << " layernorm_bucket_ms="
           << profile.nonlinear_layernorm_bucket_ms
           << " gelu_core_ms=" << profile.nonlinear_gelu_core_ms
           << " total_comm_bytes=" << profile.total_communication_bytes
           << " rotations=" << profile.he_rotations
           << " rescales=" << profile.he_rescales << endl;

      const auto release_start = chrono::steady_clock::now();
      const uint64_t release_comm0 = bert.current_comm();
      const uint64_t release_round0 = bert.current_round();
      if (args.party == sci::ALICE)
        bert.linear.release_preprocessed_layer(layer);
      bert.io->sync();
      setup_seconds += chrono::duration<double>(chrono::steady_clock::now() -
                                                 release_start).count();
      initialization_bytes += bert.current_comm() - release_comm0;
      initialization_rounds += bert.current_round() - release_round0;
      comm.mark("block_" + to_string(layer) + "_offline_cache_release");

    }

    if (args.stop_after_block < kLayers && !diagnostic_partial_classifier) {
      // Stop only after a completed secure block.  The final LayerNorm and
      // classifier are intentionally skipped so this is a Transformer-block
      // measurement, not a partial full-model latency.
      comm.mark("profiling_stop_after_block");
      const double secure_seconds =
          profiles.front().block_online_time_ms / 1000.0;
      const uint64_t online_bytes = bert.current_comm() - initialization_bytes;
      const uint64_t online_rounds =
          bert.current_round() - initialization_rounds;
      write_block_profile(args.output, args.party, profiles);
      write_online_stage_profiles(args.output, args.party, profiles);
      write_comm_audit_profile(args.output, args.party, profiles);
      write_communication_breakdown(
          args.output, args.party, comm.stages(), initialization_bytes,
          initialization_rounds, online_bytes, online_rounds);
      write_latency_summary(
          args.output, args.party, secure_seconds, initialization_bytes,
          online_bytes, online_rounds, bert.ckks_to_mpc_conversions,
          bert.conversion_ciphertexts);
      write_execution_totals(
          args.output, args.party, setup_seconds, secure_seconds, profiles,
          bert, initialization_bytes, online_bytes, online_rounds);
      write_run_metadata(-1, false);
      cout << fixed << setprecision(6)
           << "One-block profiling stop after block "
           << args.stop_after_block << '\n'
           << "Block-0 secure latency: " << profiles.front().secure_end_to_end_ms
           << " ms\n"
           << "BLOCK_ONLINE_TIME: " << secure_seconds << " s\n"
           << "BLOCK_ONLINE_COMM: "
           << profiles.front().block_online_comm_bytes << " bytes\n"
           << "Total communication (this party, including initialization): "
           << initialization_bytes + online_bytes << " bytes\n"
           << "BLB_VIT_BASELINE status=PARTIAL executed_blocks="
           << args.stop_after_block
           << " full_structure=0 protocol_substitution=0" << endl;
      return 0;
    }

    // Final ViT LayerNorm is the same original BLB LayerNorm/CKKS affine
    // sequence, applied to all physical rows.  Padding rows do not enter CLS.
    const auto final_ln_start = chrono::steady_clock::now();
    vector<uint64_t> final_ln_w(kDim, 0), final_ln_b(kDim, 0);
    if (args.party == sci::ALICE) {
      final_ln_w = quantized(vit.final_ln_weight, 12);
      final_ln_b = quantized(vit.final_ln_bias, 12);
    }
    FixArray final_w = input_parameter_share(
        bert, final_ln_w, args.physical_tokens, kDim, kBitwidth, kScale);
    FixArray final_b = input_parameter_share(
        bert, final_ln_b, args.physical_tokens, kDim, kBitwidth, kScale);
    map<char *, FixArray> final_ln =
        bert.layer_norm_part1_real(hidden, final_w, final_b);
    hidden = bert.layer_norm_affine_ckks(final_ln);
    write_diagnostic_cls(hidden, "final_ln");
    comm.mark("final_layer_norm");
    secure_protocol_seconds += chrono::duration<double>(
        chrono::steady_clock::now() - final_ln_start).count();
    cout << "BLB_VIT_FINAL_LAYER_NORM elapsed_ms="
         << chrono::duration<double, milli>(chrono::steady_clock::now() -
                                             final_ln_start)
                    .count()
         << endl;

    // ViT classification is CLS-only and has no BERT pooler.  Run the original
    // BLB BSGS2 CKKS linear evaluator on all physical rows, then select CLS.
    // The 1000-class weight is zero-padded to 1024 solely to satisfy that
    // existing full-slot layout; no alternate packing or protocol is used.
    const auto classifier_setup_start = chrono::steady_clock::now();
    const uint64_t classifier_setup_comm0 = bert.current_comm();
    const uint64_t classifier_setup_round0 = bert.current_round();
    if (args.party == sci::ALICE) {
      vector<vector<uint64_t>> classifier_weight(
          kDim, vector<uint64_t>(kPhysicalClasses, 0));
      vector<uint64_t> classifier_bias(kPhysicalClasses, 0);
      if (static_cast<int>(blb_model.w_c.size()) != kDim ||
          static_cast<int>(blb_model.b_c.size()) != kClasses)
        throw runtime_error("ViT classifier conversion shape mismatch");
      for (int in = 0; in < kDim; ++in) {
        if (static_cast<int>(blb_model.w_c[in].size()) != kClasses)
          throw runtime_error("ViT classifier row shape mismatch");
        copy(blb_model.w_c[in].begin(), blb_model.w_c[in].end(),
             classifier_weight[in].begin());
      }
      copy(blb_model.b_c.begin(), blb_model.b_c.end(),
           classifier_bias.begin());
      classifier_params = bert.linear.params_preprocessing_ct_pt(
          bert.linear.he2, kDim, kDim, kPhysicalClasses,
          classifier_weight, classifier_bias, classifier_metadata, 12, 12,
          5, std::ldexp(1.0, bert.linear.he2->he_scale));
    }
    bert.io->sync();
    setup_seconds += chrono::duration<double>(chrono::steady_clock::now() -
                                               classifier_setup_start).count();
    initialization_bytes += bert.current_comm() - classifier_setup_comm0;
    initialization_rounds += bert.current_round() - classifier_setup_round0;
    comm.mark("classifier_offline_preprocessing");
    const auto classifier_start = chrono::steady_clock::now();
    FixArray all_logits = bert.linear2_from_mpc(
        hidden, classifier_params, classifier_metadata, kLayers);
    FixArray cls_logits(bert.party, kClasses, true, all_logits.ell,
                        all_logits.s);
    copy(all_logits.data, all_logits.data + kClasses, cls_logits.data);
    FixArray opened_logits = fpmath->fix->output(sci::PUBLIC, cls_logits);
    int prediction = 0;
    auto signed_fixed = [](uint64_t value, int ell) -> int64_t {
      const uint64_t mask = ring_mask(ell);
      value &= mask;
      const uint64_t sign = uint64_t{1} << (ell - 1);
      return (value & sign) ? static_cast<int64_t>(value - (uint64_t{1} << ell))
                            : static_cast<int64_t>(value);
    };
    int64_t best = numeric_limits<int64_t>::min();
    ofstream logits_out(args.output + "/secure_logits_party" +
                        to_string(args.party) + ".txt");
    if (!logits_out) throw runtime_error("cannot write secure logits");
    logits_out << "BLB_VIT_SECURE_LOGITS_V1\n"
               << "elements " << kClasses << "\n"
               << "ell " << cls_logits.ell << "\n"
               << "scale " << cls_logits.s << "\n"
               << "values\n";
    logits_out << fixed << setprecision(10);
    for (int out = 0; out < kClasses; ++out) {
      const int64_t value = signed_fixed(opened_logits.data[out],
                                         cls_logits.ell);
      logits_out << static_cast<double>(value) /
                        static_cast<double>(uint64_t{1} << cls_logits.s)
                 << '\n';
      if (value > best) {
        best = value;
        prediction = out;
      }
    }
    logits_out << "end\n";
    ofstream prediction_out(args.output + "/prediction_party" +
                            to_string(args.party) + ".txt");
    if (!prediction_out) throw runtime_error("cannot write prediction");
    prediction_out << "predicted_class " << prediction << '\n';
    comm.mark("pooling_classification");
    secure_protocol_seconds += chrono::duration<double>(
        chrono::steady_clock::now() - classifier_start).count();
    const double secure_seconds = secure_protocol_seconds;
    const uint64_t online_bytes = bert.current_comm() - initialization_bytes;
    const uint64_t online_rounds = bert.current_round() - initialization_rounds;

    write_block_profile(args.output, args.party, profiles);
    write_online_stage_profiles(args.output, args.party, profiles);
    write_comm_audit_profile(args.output, args.party, profiles);
    write_communication_breakdown(args.output, args.party, comm.stages(),
                                   initialization_bytes, initialization_rounds,
                                   online_bytes, online_rounds);
    write_latency_summary(
        args.output, args.party, secure_seconds, initialization_bytes,
        online_bytes, online_rounds, bert.ckks_to_mpc_conversions,
        bert.conversion_ciphertexts);
    write_execution_totals(
        args.output, args.party, setup_seconds, secure_seconds, profiles, bert,
        initialization_bytes, online_bytes, online_rounds);

    write_run_metadata(prediction, true);

    cout << fixed << setprecision(6)
         << "Secure end-to-end latency: " << secure_seconds << " s\n"
         << "Total communication (this party): " << online_bytes << " bytes\n"
         << "Total communication (including initialization, this party): "
         << initialization_bytes + online_bytes << " bytes\n"
         << "CKKS->MPC conversions: " << bert.ckks_to_mpc_conversions << '\n'
         << "MPC->CKKS conversions: " << bert.mpc_to_ckks_conversions << '\n'
         << "Conversion ciphertexts: " << bert.conversion_ciphertexts << '\n'
         << "Prediction: " << prediction << '\n'
         << "BLB_VIT_BASELINE status=PASS full_structure=1 protocol_substitution=0"
         << " local_truncation=0" << endl;
    return 0;
  } catch (const exception &error) {
    cerr << "BLB_VIT_BASELINE status=BLOCKED reason=" << error.what() << endl;
    return 2;
  }
}
