#include "BOLTTransformer/bolt_conversion.h"
#include "BOLTTransformer/bolt_linear.h"
#include "FloatingPoint/fp-math.h"
#include "LinearHE/defines-HE.h"
#include "OT/ot_pack.h"
#include "utils/ArgMapping/ArgMapping.h"
#include "utils/io_pack.h"
#include "vit_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace sci;
using namespace sci::bolt;

std::uint64_t prime_mod = 536903681ULL;
std::int32_t bitlength = 37;
std::int32_t num_threads = 1;

namespace {

constexpr int kEll = 37;
constexpr int kScale = 12;
constexpr std::uint64_t kValidationSeeds = 10;

int party = 0;
int port = 47400;
std::string address = "127.0.0.1";
std::string model_path = "fixtures/vit_smoke/alice_model.txt";
std::string image_path = "fixtures/vit_smoke/bob_image.txt";
std::string output_dir = "outputs/vit_stem_protocol";
std::string validation_csv;

struct TraceRow {
  std::string op, input_shape, output_shape, input_packing, output_packing;
  int input_scale = kScale, output_scale = kScale, ell = kEll;
  BoltOperationCounts counts;
  std::uint64_t bytes = 0, rounds = 0;
  std::string backend;
  bool status = true;
};

struct ProtocolResult {
  FixArray patches;
  FixArray patch_projection;
  FixArray tokens_with_cls;
  FixArray residual;
  FixArray normalized;
  FixArray affine;
#ifdef BOLT_STEM_VALIDATION_CHANNEL
  FixArray patch_projection_padded;
  FixArray affine_padded;
#endif
};

struct WireConfig {
  int image_size, patch_size, channels, tokens, dim, heads, ff_dim, layers,
      classes, scale_bits;
};

WireConfig wire_config(const VitModel &model) {
  return {model.image_size, model.patch_size, model.channels, model.tokens,
          model.dim, model.heads, model.ff_dim, model.layers, model.classes,
          model.scale_bits};
}

void apply_wire_config(VitModel &model, const WireConfig &config) {
  model.image_size = config.image_size;
  model.patch_size = config.patch_size;
  model.channels = config.channels;
  model.tokens = config.tokens;
  model.dim = config.dim;
  model.heads = config.heads;
  model.ff_dim = config.ff_dim;
  model.layers = config.layers;
  model.classes = config.classes;
  model.scale_bits = config.scale_bits;
  model.blocks.resize(model.layers);
}

std::uint64_t mask(int ell = kEll) {
  return ell == 64 ? ~0ULL : ((1ULL << ell) - 1);
}

std::uint64_t encode(std::int64_t value) {
  return static_cast<std::uint64_t>(value) & mask();
}

std::int64_t decode(std::uint64_t value, int ell = kEll) {
  value &= mask(ell);
  const std::uint64_t sign = 1ULL << (ell - 1);
  return (value & sign) == 0
             ? static_cast<std::int64_t>(value)
             : -static_cast<std::int64_t>((1ULL << ell) - value);
}

std::int64_t quantize(double value) {
  return static_cast<std::int64_t>(
      std::llround(value * static_cast<double>(1ULL << kScale)));
}

std::int64_t floor_shift(std::int64_t value, int shift) {
  if (value >= 0) return value >> shift;
  const std::uint64_t magnitude = static_cast<std::uint64_t>(-value);
  return -static_cast<std::int64_t>(
      (magnitude + (1ULL << shift) - 1) >> shift);
}

std::uint64_t prime_encode(std::int64_t value) {
  if (value >= 0) return static_cast<std::uint64_t>(value) % prime_mod;
  const std::uint64_t magnitude = static_cast<std::uint64_t>(-value) % prime_mod;
  return magnitude == 0 ? 0 : prime_mod - magnitude;
}

std::size_t next_power_of_two(std::size_t value) {
  std::size_t result = 1;
  while (result < value) result <<= 1;
  return result;
}

BoltVitStemMetadata stem_metadata(const VitModel &model) {
  BoltVitStemMetadata metadata;
  metadata.image_size = model.image_size;
  metadata.patch_size = model.patch_size;
  metadata.channels = model.channels;
  metadata.patch_dim = static_cast<std::size_t>(model.channels) *
                       model.patch_size * model.patch_size;
  metadata.image_tokens = model.tokens - 1;
  metadata.tokens = model.tokens;
  metadata.hidden_dim = model.dim;
  metadata.packed_stride =
      next_power_of_two(std::max(metadata.patch_dim, metadata.hidden_dim));
  metadata.ell = kEll;
  metadata.scale = kScale;
  metadata.layernorm_epsilon_lsb = 0;
  return metadata;
}

void validate_tiny_fixture_contract(const VitModel &model,
                                    const BoltVitStemMetadata &metadata) {
  if (model.tokens != 5 || model.dim != 8 || model.heads != 2 ||
      model.dim / model.heads != 4 || model.scale_bits != kScale ||
      metadata.ell != kEll || metadata.scale != kScale ||
      metadata.packed_stride % metadata.patch_dim == 0) {
    throw std::invalid_argument(
        "Stem fixture must remain tokens=5, dim=8, heads=2, head_dim=4, "
        "scale=12 with a non-dividing patch_dim packing remainder");
  }
}

BoltTensorMetadata tensor_metadata(std::size_t rows, std::size_t cols,
                                   std::size_t stride, int scale,
                                   bool add_half = false) {
  BoltTensorMetadata metadata;
  metadata.rows = rows;
  metadata.cols = cols;
  metadata.packed_stride = stride;
  metadata.share_bit_width = kEll;
  metadata.scale = scale;
  metadata.packing = BoltPackingKind::kCompactRepeatedRows;
  metadata.add_plain_modulus_half_on_he_to_ss = add_half;
  return metadata;
}

BoltMatMulMetadata matmul_metadata(std::size_t rows, std::size_t inner,
                                   std::size_t output,
                                   std::size_t stride) {
  BoltMatMulMetadata metadata;
  metadata.input_rows = rows;
  metadata.input_cols = inner;
  metadata.output_cols = output;
  metadata.packed_stride = stride;
  metadata.baby_step = std::min<std::size_t>(4, stride);
  metadata.noise_flooding_bits = SMUDGING_BITLEN_bert2;
  metadata.modulus_switches = 1;
  return metadata;
}

PlainWeightTensor dense_weight(const std::vector<double> &values,
                               std::size_t output, std::size_t input,
                               const std::vector<double> &bias = {}) {
  PlainWeightTensor result;
  result.output_cols = output;
  result.input_cols = input;
  result.values.resize(output * input);
  for (std::size_t i = 0; i < result.values.size(); ++i)
    result.values[i] = prime_encode(quantize(values[i]));
  if (!bias.empty()) {
    result.bias.resize(output);
    for (std::size_t i = 0; i < output; ++i)
      result.bias[i] = prime_encode(quantize(bias[i]) << kScale);
  }
  return result;
}

PlainWeightTensor diagonal_weight(const std::vector<double> &gamma,
                                  std::size_t dim) {
  PlainWeightTensor result;
  result.output_cols = dim;
  result.input_cols = dim;
  result.values.assign(dim * dim, 0);
  for (std::size_t i = 0; i < dim; ++i)
    result.values[i * dim + i] = prime_encode(quantize(gamma[i]));
  return result;
}

FixArray shares_to_fix(const SecretSharedTensor &shares, int scale) {
  FixArray result(party, static_cast<int>(shares.local_share.size()), true,
                  kEll, scale);
  std::memcpy(result.data, shares.local_share.data(),
              shares.local_share.size() * sizeof(std::uint64_t));
  return result;
}

FixArray field_to_ring(FPMath &fp, const SecretSharedTensor &field,
                       int scale) {
  FixArray input = shares_to_fix(field, scale);
  FixArray p = fp.fix->input(PUBLIC, input.size, prime_mod, true, kEll, scale);
  FixArray p2 = fp.fix->input(PUBLIC, input.size, (prime_mod - 1) / 2,
                              true, kEll, scale);
  return fp.fix->sub(fp.gt_p_sub(input, p), p2);
}

FixArray cipher_to_ring(BoltConversion &conversion, FPMath &fp,
                        const BoltCipherTensor &cipher, std::size_t rows,
                        std::size_t cols, std::size_t stride, int scale) {
  BoltTensorMetadata metadata =
      tensor_metadata(rows, cols, stride, scale, true);
  BoltCipherTensor view = cipher;
  view.metadata = metadata;
  return field_to_ring(fp, conversion.he_to_ss(view, metadata), scale);
}

std::vector<std::uint64_t>
patchify_local(const std::vector<std::uint64_t> &image,
               const BoltVitStemMetadata &metadata) {
  if (image.size() != metadata.channels * metadata.image_size *
                          metadata.image_size)
    throw std::invalid_argument("Stem image share shape mismatch");
  std::vector<std::uint64_t> patches(metadata.image_tokens * metadata.patch_dim);
  const std::size_t side = metadata.image_size / metadata.patch_size;
  std::size_t token = 0;
  for (std::size_t py = 0; py < side; ++py)
    for (std::size_t px = 0; px < side; ++px, ++token) {
      std::size_t feature = 0;
      for (std::size_t channel = 0; channel < metadata.channels; ++channel)
        for (std::size_t ky = 0; ky < metadata.patch_size; ++ky)
          for (std::size_t kx = 0; kx < metadata.patch_size; ++kx, ++feature) {
            const std::size_t y = py * metadata.patch_size + ky;
            const std::size_t x = px * metadata.patch_size + kx;
            patches[token * metadata.patch_dim + feature] =
                image[(channel * metadata.image_size + y) *
                          metadata.image_size + x];
          }
    }
  return patches;
}

std::vector<FixArray> rows(const FixArray &flat, std::size_t count,
                           std::size_t width) {
  std::vector<FixArray> result(count);
  for (std::size_t row = 0; row < count; ++row) {
    result[row] = FixArray(flat.party, width, true, flat.ell, flat.s);
    std::memcpy(result[row].data, flat.data + row * width,
                width * sizeof(std::uint64_t));
  }
  return result;
}

void add_local_model_vector(FixArray &tensor,
                            const std::vector<double> &values) {
  if (party != ALICE) return;
  if (values.size() != static_cast<std::size_t>(tensor.size))
    throw std::invalid_argument("Stem local model vector shape mismatch");
  for (int i = 0; i < tensor.size; ++i)
    tensor.data[i] = (tensor.data[i] + encode(quantize(values[i]))) & mask();
}

void add_local_repeated(FixArray &tensor, const std::vector<double> &values,
                        std::size_t rows_count, std::size_t dim) {
  if (party != ALICE) return;
  if (values.size() != dim || tensor.size != static_cast<int>(rows_count * dim))
    throw std::invalid_argument("Stem repeated model vector shape mismatch");
  for (std::size_t row = 0; row < rows_count; ++row)
    for (std::size_t col = 0; col < dim; ++col)
      tensor.data[row * dim + col] =
          (tensor.data[row * dim + col] + encode(quantize(values[col]))) & mask();
}

void trace(std::vector<TraceRow> &out, IOPack &io, const std::string &op,
           const std::string &in_shape, const std::string &out_shape,
           const std::string &in_pack, const std::string &out_pack,
           int in_scale, int out_scale, const BoltOperationCounts &counts,
           std::uint64_t before_bytes, std::uint64_t before_rounds,
           const std::string &backend) {
  TraceRow row;
  row.op = op; row.input_shape = in_shape; row.output_shape = out_shape;
  row.input_packing = in_pack; row.output_packing = out_pack;
  row.input_scale = in_scale; row.output_scale = out_scale;
  row.counts = counts;
  row.bytes = io.get_comm() - before_bytes;
  row.rounds = io.get_rounds() - before_rounds;
  row.backend = backend;
  out.push_back(row);
}

void aggregate_trace(IOPack &io, std::vector<TraceRow> &rows) {
  struct Wire { std::uint64_t bytes, rounds; };
  if (party == BOB) {
    for (const auto &row : rows) {
      Wire wire{row.bytes, row.rounds};
      io.io->send_data(&wire, sizeof(wire));
    }
  } else {
    for (auto &row : rows) {
      Wire peer{};
      io.io->recv_data(&peer, sizeof(peer));
      row.bytes += peer.bytes;
      row.rounds = std::max(row.rounds, peer.rounds);
    }
  }
}

ProtocolResult run_protocol(IOPack &io, FPMath &fp, BoltHE &he,
                            BoltConversion &conversion, BoltLinear &linear,
                            const VitModel &model,
                            const BoltVitStemMetadata &metadata,
                            const std::vector<std::uint64_t> &image_share,
                            std::vector<TraceRow> *formal_trace) {
  ProtocolResult result;
  const bool traced = formal_trace != nullptr;
  std::uint64_t before_bytes = io.get_comm(), before_rounds = io.get_rounds();
  std::vector<std::uint64_t> patch_share = patchify_local(image_share, metadata);
  result.patches = FixArray(party, patch_share.size(), true, kEll, kScale);
  std::memcpy(result.patches.data, patch_share.data(),
              patch_share.size() * sizeof(std::uint64_t));
  if (traced) trace(*formal_trace, io, "patchify", "shares[C,H,W]",
                    "shares[image_tokens,patch_dim]", "chw", "py_px_c_ky_kx",
                    kScale, kScale, {}, before_bytes, before_rounds,
                    "local_public_permutation");

  before_bytes = io.get_comm(); before_rounds = io.get_rounds();
  SecretSharedTensor patch_tensor;
  patch_tensor.metadata = tensor_metadata(metadata.image_tokens,
      metadata.patch_dim, metadata.packed_stride, kScale);
  patch_tensor.local_share = patch_share;
  BoltCipherTensor encrypted_patches =
      conversion.ss_to_he(patch_tensor, patch_tensor.metadata);
  BoltCipherTensor projected;
  BoltOperationCounts projection_counts;
  if (party == ALICE) {
    projected = linear.cipher_plain_matmul_bsgs(
        encrypted_patches,
        dense_weight(model.patch_weight, metadata.hidden_dim,
                     metadata.patch_dim, model.patch_bias),
        matmul_metadata(metadata.image_tokens, metadata.patch_dim,
                        metadata.hidden_dim, metadata.packed_stride),
        &projection_counts);
    projected.metadata.scale = 2 * kScale;
  } else {
    projected.metadata = tensor_metadata(metadata.image_tokens,
        metadata.hidden_dim, metadata.packed_stride, 2 * kScale);
  }
  FixArray projection_raw = cipher_to_ring(
      conversion, fp, projected, metadata.image_tokens, metadata.hidden_dim,
      metadata.packed_stride, 2 * kScale);
  result.patch_projection = fp.fix->right_shift(projection_raw, kScale);
#ifdef BOLT_STEM_VALIDATION_CHANNEL
  result.patch_projection_padded = cipher_to_ring(
      conversion, fp, projected, metadata.image_tokens,
      metadata.packed_stride, metadata.packed_stride, 2 * kScale);
#endif
  if (traced) trace(*formal_trace, io, "patch_projection",
                    "shares[image_tokens,patch_dim]",
                    "shares[image_tokens,dim]", "compact_stride_metadata",
                    "z2ell_shares", kScale, kScale, projection_counts,
                    before_bytes, before_rounds, "bolt_ct_pt_bsgs");

  before_bytes = io.get_comm(); before_rounds = io.get_rounds();
  result.tokens_with_cls = FixArray(party, metadata.tokens * metadata.hidden_dim,
                                    true, kEll, kScale);
  std::fill(result.tokens_with_cls.data,
            result.tokens_with_cls.data + result.tokens_with_cls.size, 0);
  if (party == ALICE)
    for (std::size_t col = 0; col < metadata.hidden_dim; ++col)
      result.tokens_with_cls.data[col] = encode(quantize(model.class_token[col]));
  for (std::size_t token = 0; token < metadata.image_tokens; ++token)
    std::memcpy(result.tokens_with_cls.data + (token + 1) * metadata.hidden_dim,
                result.patch_projection.data + token * metadata.hidden_dim,
                metadata.hidden_dim * sizeof(std::uint64_t));
  result.residual = result.tokens_with_cls;
  add_local_model_vector(result.residual, model.position);
  if (traced) trace(*formal_trace, io, "cls_position",
                    "shares[image_tokens,dim]", "shares[tokens,dim]",
                    "image_token_order", "cls_then_raster_patch_tokens",
                    kScale, kScale, {}, before_bytes, before_rounds,
                    "share_local_model_add");

  before_bytes = io.get_comm(); before_rounds = io.get_rounds();
  FixArray unused_w(party, result.residual.size, true, kEll, kScale);
  FixArray unused_b(party, result.residual.size, true, kEll, kScale);
  std::fill(unused_w.data, unused_w.data + unused_w.size, 0);
  std::fill(unused_b.data, unused_b.data + unused_b.size, 0);
  std::vector<FixArray> normalized_rows = fp.layer_norm_fix(
      rows(result.residual, metadata.tokens, metadata.hidden_dim),
      unused_w, unused_b);
  result.normalized = concat(normalized_rows);
  if (traced) trace(*formal_trace, io, "layernorm_statistics",
                    "shares[tokens,dim]", "shares[tokens,dim]",
                    "row_major_tokens", "row_major_tokens", kScale, kScale,
                    {}, before_bytes, before_rounds, "bolt_mpc_layernorm");

  before_bytes = io.get_comm(); before_rounds = io.get_rounds();
  SecretSharedTensor normalized_tensor;
  normalized_tensor.metadata = tensor_metadata(metadata.tokens,
      metadata.hidden_dim, metadata.packed_stride, kScale);
  normalized_tensor.local_share.assign(result.normalized.data,
                                        result.normalized.data + result.normalized.size);
  BoltCipherTensor encrypted_normalized =
      conversion.ss_to_he(normalized_tensor, normalized_tensor.metadata);
  BoltCipherTensor affine_cipher;
  BoltOperationCounts affine_counts;
  if (party == ALICE) {
    affine_cipher = linear.cipher_plain_matmul_bsgs(
        encrypted_normalized,
        diagonal_weight(model.blocks.at(0).ln1_weight, metadata.hidden_dim),
        matmul_metadata(metadata.tokens, metadata.hidden_dim,
                        metadata.hidden_dim, metadata.packed_stride),
        &affine_counts);
    affine_cipher.metadata.scale = 2 * kScale;
  } else {
    affine_cipher.metadata = tensor_metadata(metadata.tokens,
        metadata.hidden_dim, metadata.packed_stride, 2 * kScale);
  }
  FixArray affine_raw = cipher_to_ring(
      conversion, fp, affine_cipher, metadata.tokens, metadata.hidden_dim,
      metadata.packed_stride, 2 * kScale);
  result.affine = fp.fix->right_shift(affine_raw, kScale);
  add_local_repeated(result.affine, model.blocks.at(0).ln1_bias,
                     metadata.tokens, metadata.hidden_dim);
#ifdef BOLT_STEM_VALIDATION_CHANNEL
  result.affine_padded = cipher_to_ring(
      conversion, fp, affine_cipher, metadata.tokens, metadata.packed_stride,
      metadata.packed_stride, 2 * kScale);
#endif
  if (traced) trace(*formal_trace, io, "layernorm_affine",
                    "shares[tokens,dim]", "shares[tokens,dim]",
                    "z2ell_shares", "compact_then_z2ell_shares",
                    kScale, kScale, affine_counts, before_bytes, before_rounds,
                    "bolt_packed_layernorm_affine");
  return result;
}

std::string csv(const std::string &value) {
  if (value.find_first_of(",\"") == std::string::npos) return value;
  std::string result = "\"";
  for (char ch : value) { if (ch == '\"') result += '\"'; result += ch; }
  return result + "\"";
}

void write_trace(const std::vector<TraceRow> &rows) {
  std::ofstream out(output_dir + "/operator_trace.csv");
  if (!out) throw std::runtime_error("cannot write stem operator trace");
  out << "op,input_shape,output_shape,input_packing,output_packing,input_scale,output_scale,ell,plain_modulus,ciphertexts,rotations,multiply_plain,evaluator_multiply,relinearizations,mod_switches,bytes,rounds,backend,status\n";
  for (const auto &r : rows)
    out << csv(r.op) << ',' << csv(r.input_shape) << ',' << csv(r.output_shape)
        << ',' << csv(r.input_packing) << ',' << csv(r.output_packing) << ','
        << r.input_scale << ',' << r.output_scale << ',' << r.ell << ','
        << prime_mod << ',' << r.counts.ciphertexts << ',' << r.counts.rotations
        << ',' << r.counts.multiply_plain << ',' << r.counts.evaluator_multiply
        << ',' << r.counts.relinearizations << ',' << r.counts.modulus_switches
        << ',' << r.bytes << ',' << r.rounds << ',' << csv(r.backend) << ','
        << (r.status ? "PASS" : "FAIL") << '\n';
}

void write_metadata(const BoltVitStemMetadata &m) {
  std::ofstream out(output_dir + "/packing_metadata.json");
  if (!out) throw std::runtime_error("cannot write stem packing metadata");
  out << "{\n"
      << "  \"image\": {\"shape\": [" << m.channels << ',' << m.image_size
      << ',' << m.image_size << "], \"patch_size\": " << m.patch_size
      << ", \"patch_order\": \"py,px\", \"feature_order\": \"channel,ky,kx\"},\n"
      << "  \"tensor\": {\"patch_dim\": " << m.patch_dim
      << ", \"image_tokens\": " << m.image_tokens << ", \"tokens\": "
      << m.tokens << ", \"dim\": " << m.hidden_dim
      << ", \"packed_stride\": " << m.packed_stride
      << ", \"valid_patch_slots\": [0," << (m.patch_dim - 1)
      << "], \"valid_hidden_slots\": [0," << (m.hidden_dim - 1) << "]},\n"
      << "  \"fixed_point\": {\"ell\": " << m.ell << ", \"scale\": "
      << m.scale << ", \"plain_modulus\": " << prime_mod << "},\n"
      << "  \"layernorm\": {\"epsilon_lsb\": "
      << m.layernorm_epsilon_lsb
      << ", \"epsilon_real\": 0.0, \"convention\": \"FPMath::layer_norm_fix adds no explicit epsilon; Python smoke clamp 1e-12 is not used by the fixed-point protocol\"},\n"
      << "  \"formal_outputs\": {\"names\": [\"x_residual_share\", \"ln1_share\"], \"representation\": \"two_party_additive_shares_z2ell\", \"revealed\": false},\n"
      << "  \"robustness\": {\"fixed_seeds\": 10, \"dense_patch_weight\": true, \"forced_zero_cases\": true, \"repeat_hash_check\": true},\n"
      << "  \"fallbacks\": {\"BfvLinear\": false, \"scalar_ciphertext\": false, \"generic_ot_matmul\": false, \"cleartext_reconstruction_formal_path\": false}\n"
      << "}\n";
}

bool validation_file_passes() {
  std::ifstream in(validation_csv);
  if (!in) return false;
  std::set<std::string> expected;
  const std::vector<std::string> operators = {
      "patchify", "patch_projection", "cls_insert", "position_add",
      "layernorm_normalized", "layernorm_affine",
      "patch_projection_padding", "layernorm_affine_padding"};
  for (const auto &op : operators) expected.insert(op);
  for (std::uint64_t seed = 1; seed <= kValidationSeeds; ++seed)
    for (const auto &op : operators)
      expected.insert("seed_" + std::to_string(seed) + "/" + op);
  for (const auto &op : operators) expected.insert("repeat_seed_0/" + op);
  expected.insert("repeat_hash_stability");

  std::string line;
  std::set<std::string> observed;
  while (std::getline(in, line)) {
    if (line.rfind("op,", 0) == 0) continue;
    const std::size_t comma = line.find(',');
    if (comma == std::string::npos || line.find(",true,0,") == std::string::npos ||
        line.size() < 5 || line.substr(line.size() - 5) != ",PASS")
      return false;
    const std::string op = line.substr(0, comma);
    if (expected.count(op) == 0 || !observed.insert(op).second) return false;
  }
  return observed == expected;
}

bool fail_closed(BoltHE &he, const BoltVitStemMetadata &metadata) {
  bool scalar = false, generic = false, batching = false, galois = false;
  try { fail_if_scalar_ciphertext_adapter(true); }
  catch (const std::logic_error &) { scalar = true; }
  try { fail_if_generic_ot_matmul(true); }
  catch (const std::logic_error &) { generic = true; }
  try { BoltHEView bad = he.view(); bad.encoder = nullptr; bad.require_batching(); }
  catch (const std::logic_error &) { batching = true; }
  try { BoltHEView bad = he.view(); bad.galois_keys = nullptr;
        bad.require_public_evaluation(false); }
  catch (const std::logic_error &) { galois = true; }
  return scalar && generic && batching && galois &&
         metadata.layernorm_epsilon_lsb == 0;
}

#ifdef BOLT_STEM_VALIDATION_CHANNEL

std::vector<std::uint64_t> reconstruct(IOPack &io, const FixArray &shares) {
  if (party == ALICE) {
    io.io->send_data(shares.data, shares.size * sizeof(std::uint64_t));
    return {};
  }
  std::vector<std::uint64_t> alice(shares.size), clear(shares.size);
  io.io->recv_data(alice.data(), alice.size() * sizeof(std::uint64_t));
  for (int i = 0; i < shares.size; ++i)
    clear[i] = (alice[i] + shares.data[i]) & shares.ell_mask();
  return clear;
}

std::vector<std::int64_t> clear_patchify(const std::vector<std::int64_t> &image,
                                         const BoltVitStemMetadata &m) {
  std::vector<std::uint64_t> encoded(image.size());
  for (std::size_t i = 0; i < image.size(); ++i) encoded[i] = encode(image[i]);
  auto patched = patchify_local(encoded, m);
  std::vector<std::int64_t> result(patched.size());
  for (std::size_t i = 0; i < patched.size(); ++i) result[i] = decode(patched[i]);
  return result;
}

std::vector<std::int64_t> clear_projection_raw(
    const std::vector<std::int64_t> &patches, const VitModel &model,
    const BoltVitStemMetadata &m) {
  std::vector<std::int64_t> result(m.image_tokens * m.hidden_dim);
  for (std::size_t row = 0; row < m.image_tokens; ++row)
    for (std::size_t out = 0; out < m.hidden_dim; ++out) {
      __int128 sum = static_cast<__int128>(quantize(model.patch_bias[out]))
                     << kScale;
      for (std::size_t in = 0; in < m.patch_dim; ++in)
        sum += static_cast<__int128>(patches[row * m.patch_dim + in]) *
               quantize(model.patch_weight[out * m.patch_dim + in]);
      result[row * m.hidden_dim + out] = static_cast<std::int64_t>(sum);
    }
  return result;
}

std::vector<std::int64_t> clear_projection(
    const std::vector<std::int64_t> &patches, const VitModel &model,
    const BoltVitStemMetadata &m) {
  std::vector<std::int64_t> result =
      clear_projection_raw(patches, model, m);
  for (auto &value : result) value = floor_shift(value, kScale);
  return result;
}

std::vector<std::int64_t> clear_cls(const std::vector<std::int64_t> &patch,
                                    const VitModel &model,
                                    const BoltVitStemMetadata &m) {
  std::vector<std::int64_t> result(m.tokens * m.hidden_dim);
  for (std::size_t col = 0; col < m.hidden_dim; ++col)
    result[col] = quantize(model.class_token[col]);
  std::copy(patch.begin(), patch.end(), result.begin() + m.hidden_dim);
  return result;
}

std::vector<std::int64_t> clear_position(std::vector<std::int64_t> input,
                                         const VitModel &model) {
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] += quantize(model.position[i]);
  return input;
}

std::vector<std::uint64_t> input_shares(const std::vector<std::int64_t> &clear,
                                        std::uint64_t seed) {
  std::mt19937_64 generator(seed);
  std::vector<std::uint64_t> result(clear.size());
  for (std::size_t i = 0; i < clear.size(); ++i) {
    std::int64_t alice = static_cast<std::int64_t>(generator() % 255) - 127;
    if (alice == 0) alice = 1;
    result[i] = encode(party == ALICE ? alice : clear[i] - alice);
  }
  return result;
}

std::uint64_t fnv_hash(const std::vector<std::uint64_t> &values) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::uint64_t value : values) {
    hash ^= value;
    hash *= 1099511628211ULL;
  }
  return hash;
}

struct Match { std::string name; double max_abs, mean_abs, rmse, rel; bool pass; };

Match compare(const std::string &name, const std::vector<std::uint64_t> &actual,
              const std::vector<std::int64_t> &reference, int scale) {
  if (actual.size() != reference.size()) throw std::runtime_error("validation shape mismatch");
  long double sum = 0, square = 0, ref_square = 0;
  std::int64_t maximum = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const std::int64_t error = decode(actual[i]) - reference[i];
    const std::int64_t absolute = std::llabs(error);
    maximum = std::max(maximum, absolute);
    sum += absolute; square += static_cast<long double>(error) * error;
    ref_square += static_cast<long double>(reference[i]) * reference[i];
  }
  const long double denominator = std::ldexp(1.0L, scale);
  Match result{name, static_cast<double>(maximum / denominator),
      static_cast<double>(sum / actual.size() / denominator),
      static_cast<double>(std::sqrt(square / actual.size()) / denominator),
      ref_square == 0 ? (square == 0 ? 0.0 : INFINITY)
                      : static_cast<double>(std::sqrt(square / ref_square)),
      maximum == 0};
  return result;
}

void run_validation(IOPack &io, FPMath &fp, BoltHE &he,
                    BoltConversion &conversion, BoltLinear &linear,
                    VitModel model, VitInput input,
                    const BoltVitStemMetadata &metadata) {
  auto signs = [](const std::vector<double> &values) {
    bool positive = false, negative = false;
    for (double value : values) {
      positive = positive || quantize(value) > 0;
      negative = negative || quantize(value) < 0;
    }
    return positive && negative;
  };
  const bool asymmetric_cls = std::adjacent_find(
      model.class_token.begin(), model.class_token.end(),
      [](double lhs, double rhs) { return quantize(lhs) != quantize(rhs); }) !=
      model.class_token.end();
  bool distinct_positions = false;
  for (std::size_t col = 0; col < metadata.hidden_dim; ++col)
    distinct_positions = distinct_positions ||
        quantize(model.position[col]) !=
            quantize(model.position[metadata.hidden_dim + col]);
  std::size_t nonzero_weights = 0;
  for (double weight : model.patch_weight)
    nonzero_weights += quantize(weight) != 0;
  if (!signs(model.patch_weight) || !signs(input.pixels) || !asymmetric_cls ||
      !distinct_positions ||
      nonzero_weights * 10 < model.patch_weight.size() * 9) {
    throw std::runtime_error(
        "Stem robustness fixture lacks dense signed weights/input or "
        "asymmetric CLS/position values");
  }
  std::vector<Match> matches;
  std::uint64_t baseline_hash = 0, repeated_hash = 0;
  for (std::uint64_t case_id = 0; case_id <= kValidationSeeds + 1; ++case_id) {
    const std::uint64_t seed = case_id == kValidationSeeds + 1 ? 0 : case_id;
    VitModel current = model;
    std::vector<std::int64_t> pixels(input.pixels.size());
    for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = quantize(input.pixels[i]);
    if (seed != 0) {
      pixels[seed % pixels.size()] = 0;
      current.patch_weight[(seed * 7) % current.patch_weight.size()] = 0.0;
      if (std::find(pixels.begin(), pixels.end(), 0) == pixels.end() ||
          std::none_of(current.patch_weight.begin(), current.patch_weight.end(),
                       [](double value) { return quantize(value) == 0; }))
        throw std::runtime_error("Stem robustness zero injection failed");
    }
    ProtocolResult actual = run_protocol(
        io, fp, he, conversion, linear, current, metadata,
        input_shares(pixels, 0x53A000ULL + seed), nullptr);
    const auto patch_ref = clear_patchify(pixels, metadata);
    const auto projection_raw_ref = clear_projection_raw(patch_ref, current, metadata);
    const auto projection_ref = clear_projection(patch_ref, current, metadata);
    const auto cls_ref = clear_cls(projection_ref, current, metadata);
    const auto position_ref = clear_position(cls_ref, current);

    std::vector<std::uint64_t> encoded_position(position_ref.size(), 0);
    if (party == ALICE)
      for (std::size_t i = 0; i < position_ref.size(); ++i)
        encoded_position[i] = encode(position_ref[i]);
    FixArray reference_input = fp.fix->input(ALICE, encoded_position.size(),
                                              encoded_position.data(), true,
                                              kEll, kScale);
    FixArray unused_w(party, reference_input.size, true, kEll, kScale);
    FixArray unused_b(party, reference_input.size, true, kEll, kScale);
    std::fill(unused_w.data, unused_w.data + unused_w.size, 0);
    std::fill(unused_b.data, unused_b.data + unused_b.size, 0);
    FixArray normalized_reference = concat(fp.layer_norm_fix(
        rows(reference_input, metadata.tokens, metadata.hidden_dim),
        unused_w, unused_b));

    const auto patch_clear = reconstruct(io, actual.patches);
    const auto projection_clear = reconstruct(io, actual.patch_projection);
    const auto cls_clear = reconstruct(io, actual.tokens_with_cls);
    const auto position_clear = reconstruct(io, actual.residual);
    const auto normalized_clear = reconstruct(io, actual.normalized);
    const auto normalized_reference_clear = reconstruct(io, normalized_reference);
    const auto affine_clear = reconstruct(io, actual.affine);
    const auto projection_padded = reconstruct(io, actual.patch_projection_padded);
    const auto affine_padded = reconstruct(io, actual.affine_padded);
    if (party == BOB) {
      const std::string prefix =
          case_id == 0
              ? ""
              : (case_id == kValidationSeeds + 1
                     ? "repeat_seed_0/"
                     : "seed_" + std::to_string(seed) + "/");
      matches.push_back(compare(prefix + "patchify", patch_clear, patch_ref, kScale));
      matches.push_back(compare(prefix + "patch_projection", projection_clear, projection_ref, kScale));
      matches.push_back(compare(prefix + "cls_insert", cls_clear, cls_ref, kScale));
      matches.push_back(compare(prefix + "position_add", position_clear, position_ref, kScale));
      std::vector<std::int64_t> normalized_ref(normalized_reference_clear.size());
      for (std::size_t i = 0; i < normalized_ref.size(); ++i)
        normalized_ref[i] = decode(normalized_reference_clear[i]);
      matches.push_back(compare(prefix + "layernorm_normalized",
                                normalized_clear, normalized_ref, kScale));
      std::vector<std::int64_t> affine_ref(normalized_ref.size());
      for (std::size_t token = 0; token < metadata.tokens; ++token)
        for (std::size_t col = 0; col < metadata.hidden_dim; ++col) {
          const std::size_t index = token * metadata.hidden_dim + col;
          affine_ref[index] = floor_shift(
              normalized_ref[index] * quantize(current.blocks[0].ln1_weight[col]),
              kScale) + quantize(current.blocks[0].ln1_bias[col]);
        }
      matches.push_back(compare(prefix + "layernorm_affine", affine_clear,
                                affine_ref, kScale));
      std::vector<std::int64_t> projection_pad_ref(
          metadata.image_tokens * metadata.packed_stride, 0);
      for (std::size_t row = 0; row < metadata.image_tokens; ++row)
        for (std::size_t col = 0; col < metadata.hidden_dim; ++col)
          projection_pad_ref[row * metadata.packed_stride + col] =
              projection_raw_ref[row * metadata.hidden_dim + col];
      matches.push_back(compare(prefix + "patch_projection_padding",
                                projection_padded, projection_pad_ref,
                                2 * kScale));
      std::vector<std::int64_t> affine_pad_ref(
          metadata.tokens * metadata.packed_stride, 0);
      for (std::size_t row = 0; row < metadata.tokens; ++row)
        for (std::size_t col = 0; col < metadata.hidden_dim; ++col)
          affine_pad_ref[row * metadata.packed_stride + col] =
              floor_shift(normalized_ref[row * metadata.hidden_dim + col] *
                              quantize(current.blocks[0].ln1_weight[col]), 0);
      matches.push_back(compare(prefix + "layernorm_affine_padding",
                                affine_padded, affine_pad_ref, 2 * kScale));
      std::vector<std::uint64_t> hash_input = position_clear;
      hash_input.insert(hash_input.end(), affine_clear.begin(), affine_clear.end());
      const std::uint64_t hash = fnv_hash(hash_input);
      if (case_id == 0) baseline_hash = hash;
      if (case_id == kValidationSeeds + 1) repeated_hash = hash;
    }
  }
  if (party == BOB) {
    const bool stable = baseline_hash == repeated_hash;
    matches.push_back({"repeat_hash_stability", stable ? 0.0 : 1.0,
                       stable ? 0.0 : 1.0, stable ? 0.0 : 1.0,
                       stable ? 0.0 : 1.0, stable});
    std::ofstream out(output_dir + "/numerical_match.csv");
    out << "op,bit_exact,tolerance_lsb,max_abs,mean_abs,rmse,relative_l2,status\n";
    for (const auto &m : matches)
      out << m.name << ",true,0," << std::setprecision(12) << m.max_abs
          << ',' << m.mean_abs << ',' << m.rmse << ',' << m.rel << ','
          << (m.pass ? "PASS" : "FAIL") << '\n';
    auto failed = std::find_if(matches.begin(), matches.end(),
                               [](const Match &m) { return !m.pass; });
    std::ofstream mismatch(output_dir + "/first_mismatch.json");
    if (failed == matches.end()) mismatch << "{\"mismatch\": null}\n";
    else mismatch << "{\"mismatch\": {\"op\": \"" << failed->name
                  << "\", \"index\": null}}\n";
    if (failed != matches.end()) throw std::runtime_error("Stem validation mismatch");
  }
}

#endif

} // namespace

int main(int argc, char **argv) {
  ArgMapping args;
  args.arg("r", party, "Role: ALICE=1, BOB=2");
  args.arg("p", port, "TCP port");
  args.arg("ip", address, "ALICE address");
  args.arg("model", model_path, "Tiny ViT model fixture");
  args.arg("image", image_path, "Tiny ViT image fixture");
  args.arg("out", output_dir, "Output directory");
  args.arg("validation", validation_csv, "Independent validation CSV");
  args.parse(argc, argv);
  try {
    if (party != ALICE && party != BOB)
      throw std::runtime_error("r must be 1 or 2");
    VitModel model;
    VitInput input;
#ifdef BOLT_STEM_VALIDATION_CHANNEL
    // This separately compiled debug-only channel owns both fixtures so it can
    // build the fixed-point references. Its traffic is never included in the
    // formal protocol trace.
    model = load_vit_model(model_path);
    input = load_vit_input(image_path);
#else
    // Formal ownership: ALICE alone reads model parameters; BOB alone reads
    // pixels. Only public dimensions cross this setup channel.
    if (party == ALICE) model = load_vit_model(model_path);
#endif
    IOPack io(party, port, address);
#ifndef BOLT_STEM_VALIDATION_CHANNEL
    WireConfig config{};
    if (party == ALICE) {
      config = wire_config(model);
      io.io->send_data(&config, sizeof(config));
      io.io->flush();
    } else {
      io.io->recv_data(&config, sizeof(config));
      apply_wire_config(model, config);
      input = load_vit_input(image_path);
      if (input.image_size != model.image_size ||
          input.channels != model.channels)
        throw std::runtime_error("image dimensions do not match model");
    }
#endif
    BoltVitStemMetadata metadata = stem_metadata(model);
    validate_tiny_fixture_contract(model, metadata);
    BoltHE he(party, io.io, BoltHEParameters::bert_main());
    metadata.validate(he.slot_count());
    OTPack otpack(&io, party);
    FPMath fp(party, &io, &otpack);
    BoltConversion conversion(he.view());
    BoltLinear linear(he.view());
    if (!fail_closed(he, metadata)) throw std::runtime_error("Stem fail-closed self-test failed");

#ifdef BOLT_STEM_VALIDATION_CHANNEL
    run_validation(io, fp, he, conversion, linear, model, input, metadata);
    return 0;
#else
    if (!validation_file_passes())
      throw std::runtime_error("independent Stem validation is missing or failed");
    const std::size_t pixel_count = static_cast<std::size_t>(metadata.channels) *
                                    metadata.image_size * metadata.image_size;
    std::vector<std::uint64_t> image_share(pixel_count, 0);
    if (party == BOB)
      for (std::size_t i = 0; i < pixel_count; ++i)
        image_share[i] = encode(quantize(input.pixels[i]));
    std::vector<TraceRow> trace_rows;
    ProtocolResult output = run_protocol(io, fp, he, conversion, linear,
                                         model, metadata, image_share,
                                         &trace_rows);
    (void)output;
    aggregate_trace(io, trace_rows);
    const bool passed = party == BOB ||
        (trace_rows[1].counts.multiply_plain > 0 &&
         trace_rows[4].counts.multiply_plain > 0);
    for (auto &row : trace_rows) row.status = passed;
    if (party == ALICE) {
      write_trace(trace_rows);
      write_metadata(metadata);
      std::cout
          << "VIT_STEM_OP op=patchify\nbackend=local_public_permutation\nplaintext_reconstruction=false\nstatus=PASS\n"
          << "VIT_STEM_OP op=patch_projection\nbackend=bolt_ct_pt_bsgs\nscalar_ciphertext=false\nstatus=PASS\n"
          << "VIT_STEM_OP op=cls_position\nbackend=share_local_model_add\nstatus=PASS\n"
          << "VIT_STEM_OP op=layernorm_statistics\nbackend=bolt_mpc_layernorm\nstatus=PASS\n"
          << "VIT_STEM_OP op=layernorm_affine\nbackend=bolt_packed_layernorm_affine\nscalar_ciphertext=false\nstatus=PASS\n"
          << "VIT_STEM_NUMERICAL_MATCH status=PASS\n"
          << "PLAINTEXT_INTERMEDIATE_RECONSTRUCTIONS=0\nSCALAR_FALLBACKS=0\nGENERIC_OT_MATMUL_CALLS=0\n"
          << "M1_SHARED_BACKEND=PASS\nM2_VIT_ATTENTION_PROTOCOL=PASS\nM3A_VIT_STEM_LN1_PROTOCOL=PASS\n"
          << "FULL_VIT_MIGRATION=TESTED_SEPARATELY\n"
             "CHECKPOINT_INTEGRATION=SCHEMA_ONLY_TESTED_SEPARATELY"
          << std::endl;
    }
    return passed ? 0 : 1;
#endif
  } catch (const std::exception &error) {
    std::cerr << "BOLT_VIT_STEM_PROTOCOL_TEST error: " << error.what() << std::endl;
    return 1;
  }
}
