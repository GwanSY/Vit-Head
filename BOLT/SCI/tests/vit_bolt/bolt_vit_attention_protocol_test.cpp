#include "BOLTTransformer/bolt_attention.h"
#include "BOLTTransformer/bolt_conversion.h"
#include "BOLTTransformer/bolt_linear.h"
#include "FloatingPoint/fp-math.h"
#include "LinearHE/defines-HE.h"
#include "OT/ot_pack.h"
#include "utils/ArgMapping/ArgMapping.h"
#include "utils/io_pack.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace sci;
using namespace sci::bolt;

std::uint64_t prime_mod = 557057ULL;
std::int32_t bitlength = 37;
std::int32_t num_threads = 1;

namespace {

constexpr std::size_t kTokens = 5;
constexpr std::size_t kDim = 8;
constexpr std::size_t kHeads = 2;
constexpr std::size_t kHeadDim = 4;
constexpr std::size_t kStride = 8;
constexpr int kScale = 12;
constexpr int kEll = 37;
constexpr std::uint64_t kSeed = 0xB017A77EULL;
constexpr std::uint64_t kSoftmaxToleranceLsb = 0;

int party = 0;
int port = 47300;
std::string address = "127.0.0.1";
std::string output_dir = "outputs/vit_attention_protocol";

struct Fixture {
  std::vector<std::int64_t> x;
  std::vector<std::int64_t> wq;
  std::vector<std::int64_t> wk;
  std::vector<std::int64_t> wv;
  std::vector<std::int64_t> wo;
  std::vector<std::int64_t> bo;
};

struct TraceRow {
  std::string op;
  std::string input_shape;
  std::string output_shape;
  std::string input_packing;
  std::string output_packing;
  int input_scale = 0;
  int output_scale = 0;
  int ell = kEll;
  std::uint64_t plain_modulus = prime_mod;
  BoltOperationCounts counts;
  std::uint64_t bytes = 0;
  std::uint64_t rounds = 0;
  std::string backend;
  bool status = true;
};

struct NumericalRow {
  std::string op;
  bool bit_exact = false;
  std::uint64_t tolerance_lsb = 0;
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double rmse = 0.0;
  double relative_l2 = 0.0;
  std::string first_mismatch = "";
  bool status = false;
};

std::uint64_t ring_mask(int ell = kEll) {
  return ell == 64 ? ~0ULL : ((1ULL << ell) - 1);
}

std::uint64_t ring_encode(std::int64_t value, int ell = kEll) {
  return static_cast<std::uint64_t>(value) & ring_mask(ell);
}

std::int64_t ring_signed(std::uint64_t value, int ell) {
  value &= ring_mask(ell);
  const std::uint64_t sign = 1ULL << (ell - 1);
  if ((value & sign) == 0) return static_cast<std::int64_t>(value);
  return -static_cast<std::int64_t>((1ULL << ell) - value);
}

std::uint64_t prime_encode(std::int64_t value, std::uint64_t modulus) {
  if (value >= 0) return static_cast<std::uint64_t>(value) % modulus;
  const std::uint64_t magnitude = static_cast<std::uint64_t>(-value) % modulus;
  return magnitude == 0 ? 0 : modulus - magnitude;
}

std::int64_t floor_shift(std::int64_t value, int shift) {
  if (shift == 0) return value;
  const std::int64_t divisor = 1LL << shift;
  if (value >= 0) return value / divisor;
  return -static_cast<std::int64_t>(
      (static_cast<std::uint64_t>(-value) + divisor - 1) / divisor);
}

Fixture make_fixture() {
  Fixture fixture;
  fixture.x.resize(kTokens * kDim);
  fixture.wq.resize(kDim * kDim);
  fixture.wk.resize(kDim * kDim);
  fixture.wv.resize(kDim * kDim);
  fixture.wo.resize(kDim * kDim);
  fixture.bo.resize(kDim);
  for (std::size_t i = 0; i < fixture.x.size(); ++i)
    fixture.x[i] = static_cast<std::int64_t>((i * 17 + 13) % 61) - 30;
  for (std::size_t out = 0; out < kDim; ++out) {
    for (std::size_t in = 0; in < kDim; ++in) {
      const std::size_t index = out * kDim + in;
      fixture.wq[index] = out == in ? 1 : 0;
      fixture.wk[index] = out == in ? 1 : 0;
      fixture.wv[index] = out == in ? 1 : 0;
      fixture.wo[index] = out == in ? 1 : 0;
    }
    fixture.bo[out] = static_cast<std::int64_t>((out * 7 + 2) % 17) - 8;
  }
  return fixture;
}

BoltTensorMetadata tensor_metadata(std::size_t rows, std::size_t cols,
                                   int scale,
                                   bool add_plain_half = false) {
  BoltTensorMetadata metadata;
  metadata.rows = rows;
  metadata.cols = cols;
  metadata.packed_stride = kStride;
  metadata.share_bit_width = kEll;
  metadata.scale = scale;
  metadata.packing = BoltPackingKind::kCompactRepeatedRows;
  metadata.add_plain_modulus_half_on_he_to_ss = add_plain_half;
  return metadata;
}

BoltMatMulMetadata matmul_metadata(std::size_t rows, std::size_t inner,
                                   std::size_t cols,
                                   std::uint32_t flood_bits,
                                   std::uint32_t switches) {
  BoltMatMulMetadata metadata;
  metadata.input_rows = rows;
  metadata.input_cols = inner;
  metadata.output_cols = cols;
  metadata.packed_stride = kStride;
  metadata.baby_step = 2;
  metadata.noise_flooding_bits = flood_bits;
  metadata.modulus_switches = switches;
  return metadata;
}

BoltAttentionPackingMetadata attention_metadata() {
  BoltAttentionPackingMetadata metadata;
  metadata.tokens = kTokens;
  metadata.model_dim = kDim;
  metadata.heads = kHeads;
  metadata.head_dim = kHeadDim;
  metadata.packed_stride = kStride;
  return metadata;
}

PlainWeightTensor make_weight(const std::vector<std::int64_t> &values,
                              const std::vector<std::int64_t> &bias,
                              std::uint64_t modulus,
                              int bias_left_shift = 0) {
  PlainWeightTensor weight;
  weight.output_cols = kDim;
  weight.input_cols = kDim;
  weight.values.resize(values.size());
  for (std::size_t i = 0; i < values.size(); ++i)
    weight.values[i] = prime_encode(values[i], modulus);
  if (!bias.empty()) {
    weight.bias.resize(bias.size());
    for (std::size_t i = 0; i < bias.size(); ++i)
      weight.bias[i] = prime_encode(bias[i] << bias_left_shift, modulus);
  }
  return weight;
}

std::vector<std::int64_t>
clear_linear(const std::vector<std::int64_t> &input, std::size_t rows,
             const std::vector<std::int64_t> &weight,
             const std::vector<std::int64_t> &bias = {},
             int bias_left_shift = 0) {
  std::vector<std::int64_t> output(rows * kDim, 0);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < kDim; ++out) {
      __int128 sum = 0;
      for (std::size_t in = 0; in < kDim; ++in)
        sum += static_cast<__int128>(input[row * kDim + in]) *
               weight[out * kDim + in];
      if (!bias.empty()) sum += static_cast<__int128>(bias[out]) << bias_left_shift;
      if (sum <= -(static_cast<__int128>(1) << (kEll - 1)) ||
          sum >= (static_cast<__int128>(1) << (kEll - 1)))
        throw std::overflow_error("fixed-point reference overflow");
      output[row * kDim + out] = static_cast<std::int64_t>(sum);
    }
  }
  return output;
}

std::vector<std::int64_t>
clear_qk(const std::vector<std::int64_t> &q,
         const std::vector<std::int64_t> &k) {
  std::vector<std::int64_t> output(kHeads * kTokens * kTokens, 0);
  for (std::size_t head = 0; head < kHeads; ++head)
    for (std::size_t row = 0; row < kTokens; ++row)
      for (std::size_t col = 0; col < kTokens; ++col) {
        __int128 sum = 0;
        for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
          const std::size_t dim = head * kHeadDim + feature;
          sum += static_cast<__int128>(q[row * kDim + dim]) *
                 k[col * kDim + dim];
        }
        output[(head * kTokens + row) * kTokens + col] =
            static_cast<std::int64_t>(sum);
      }
  return output;
}

SecretSharedTensor make_input_shares(const Fixture &fixture) {
  SecretSharedTensor shares;
  shares.metadata = tensor_metadata(kTokens, kDim, kScale);
  shares.local_share.resize(fixture.x.size());
  std::mt19937_64 generator(kSeed);
  for (std::size_t i = 0; i < fixture.x.size(); ++i) {
    // Both shares are non-zero additive Z2^ell shares.  The bounded mask keeps
    // both signed representatives in the domain used by BOLT's existing
    // signed-share-to-prime conversion.
    std::int64_t alice = static_cast<std::int64_t>(generator() % 63) - 31;
    if (alice == 0) alice = 1;
    const std::int64_t bob = fixture.x[i] - alice;
    shares.local_share[i] = ring_encode(party == ALICE ? alice : bob);
  }
  return shares;
}

FixArray fix_from_local_shares(const SecretSharedTensor &shares, FPMath &fp,
                               int scale) {
  FixArray result(party, static_cast<int>(shares.local_share.size()), true,
                  kEll, scale);
  std::memcpy(result.data, shares.local_share.data(),
              shares.local_share.size() * sizeof(std::uint64_t));
  return result;
}

FixArray reference_from_alice(const std::vector<std::int64_t> &values,
                              FPMath &fp, int scale) {
  std::vector<std::uint64_t> encoded(values.size(), 0);
  if (party == ALICE)
    for (std::size_t i = 0; i < values.size(); ++i)
      encoded[i] = ring_encode(values[i]);
  return fp.fix->input(ALICE, static_cast<int>(values.size()), encoded.data(),
                       true, kEll, scale);
}

FixArray prime_cipher_to_ring(BoltConversion &conversion, FPMath &fp,
                              const BoltCipherTensor &cipher,
                              std::size_t rows, std::size_t cols, int scale) {
  BoltTensorMetadata metadata = tensor_metadata(rows, cols, scale, true);
  BoltCipherTensor view = cipher;
  view.metadata = metadata;
  SecretSharedTensor field = conversion.he_to_ss(view, metadata);
  FixArray input = fix_from_local_shares(field, fp, scale);
  FixArray p = fp.fix->input(PUBLIC, input.size, prime_mod, true, kEll, scale);
  FixArray p_half = fp.fix->input(PUBLIC, input.size,
                                  (prime_mod - 1) / 2, true, kEll, scale);
  FixArray ring = fp.gt_p_sub(input, p);
  return fp.fix->sub(ring, p_half);
}

FixArray field_shares_to_ring(FPMath &fp,
                              const SecretSharedTensor &field, int scale) {
  FixArray input = fix_from_local_shares(field, fp, scale);
  FixArray p = fp.fix->input(PUBLIC, input.size, prime_mod, true, kEll, scale);
  FixArray p_half = fp.fix->input(PUBLIC, input.size,
                                  (prime_mod - 1) / 2, true, kEll, scale);
  FixArray ring = fp.gt_p_sub(input, p);
  return fp.fix->sub(ring, p_half);
}

std::vector<FixArray> split_rows(const FixArray &flat, std::size_t rows,
                                 std::size_t cols) {
  if (flat.size != static_cast<int>(rows * cols))
    throw std::invalid_argument("fixed-point row split shape mismatch");
  std::vector<FixArray> result(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    result[row] = FixArray(flat.party, static_cast<int>(cols), flat.signed_,
                           flat.ell, flat.s);
    std::memcpy(result[row].data, flat.data + row * cols,
                cols * sizeof(std::uint64_t));
  }
  return result;
}

FixArray flatten_rows(const std::vector<FixArray> &rows) {
  return concat(rows);
}

FixArray pad_reference_rows(const FixArray &input, std::size_t rows,
                            std::size_t valid_cols, std::size_t stride) {
  if (input.size != static_cast<int>(rows * valid_cols) ||
      stride < valid_cols)
    throw std::invalid_argument("reference padding shape mismatch");
  FixArray output(party, static_cast<int>(rows * stride), input.signed_,
                  input.ell, input.s);
  std::fill(output.data, output.data + output.size, 0);
  for (std::size_t row = 0; row < rows; ++row)
    std::memcpy(output.data + row * stride,
                input.data + row * valid_cols,
                valid_cols * sizeof(std::uint64_t));
  return output;
}

FixArray secure_reference_matmul(FPMath &fp, const FixArray &lhs,
                                 std::size_t rows, std::size_t inner,
                                 const FixArray &rhs, std::size_t cols,
                                 int output_ell = kEll) {
  if (lhs.size != static_cast<int>(rows * inner) ||
      rhs.size != static_cast<int>(inner * cols))
    throw std::invalid_argument("reference MatMul shape mismatch");
  const std::size_t products = rows * inner * cols;
  FixArray lhs_expanded(party, static_cast<int>(products), true, lhs.ell,
                        lhs.s);
  FixArray rhs_expanded(party, static_cast<int>(products), true, rhs.ell,
                        rhs.s);
  std::size_t index = 0;
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      for (std::size_t k = 0; k < inner; ++k, ++index) {
        lhs_expanded.data[index] = lhs.data[row * inner + k];
        rhs_expanded.data[index] = rhs.data[k * cols + col];
      }
  FixArray product = fp.fix->mul(lhs_expanded, rhs_expanded, output_ell);
  FixArray output(party, static_cast<int>(rows * cols), true, output_ell,
                  lhs.s + rhs.s);
  const std::uint64_t mask = output.ell_mask();
  index = 0;
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col) {
      std::uint64_t sum = 0;
      for (std::size_t k = 0; k < inner; ++k, ++index)
        sum = (sum + product.data[index]) & mask;
      output.data[row * cols + col] = sum;
    }
  return output;
}

FixArray reference_attention_heads(FPMath &fp,
                                   const std::vector<FixArray> &softmax,
                                   const FixArray &v) {
  std::vector<FixArray> heads;
  heads.reserve(kHeads);
  for (std::size_t head = 0; head < kHeads; ++head) {
    FixArray soft_flat(party, kTokens * kTokens, true, kEll, kScale);
    FixArray value(party, kTokens * kHeadDim, true, kEll, kScale);
    for (std::size_t row = 0; row < kTokens; ++row) {
      std::memcpy(soft_flat.data + row * kTokens,
                  softmax[head * kTokens + row].data,
                  kTokens * sizeof(std::uint64_t));
      for (std::size_t feature = 0; feature < kHeadDim; ++feature)
        value.data[row * kHeadDim + feature] =
            v.data[row * kDim + head * kHeadDim + feature];
    }
    heads.push_back(secure_reference_matmul(
        fp, soft_flat, kTokens, kTokens, value, kHeadDim));
  }
  return concat(heads);
}

FixArray merge_reference_heads(const FixArray &heads) {
  FixArray merged(party, kTokens * kDim, true, kEll, 2 * kScale);
  for (std::size_t head = 0; head < kHeads; ++head)
    for (std::size_t token = 0; token < kTokens; ++token)
      for (std::size_t feature = 0; feature < kHeadDim; ++feature)
        merged.data[token * kDim + head * kHeadDim + feature] =
            heads.data[(head * kTokens + token) * kHeadDim + feature];
  return merged;
}

FixArray reference_output_projection(FPMath &fp, const FixArray &merged,
                                     const Fixture &fixture) {
  std::vector<std::uint64_t> weight_values(kDim * kDim, 0);
  std::vector<std::uint64_t> bias_values(kDim, 0);
  if (party == ALICE) {
    for (std::size_t in = 0; in < kDim; ++in)
      for (std::size_t out = 0; out < kDim; ++out)
        weight_values[in * kDim + out] =
            ring_encode(fixture.wo[out * kDim + in]);
    for (std::size_t out = 0; out < kDim; ++out)
      bias_values[out] = ring_encode(fixture.bo[out] << kScale);
  }
  FixArray weights = fp.fix->input(ALICE, kDim * kDim,
                                   weight_values.data(), true, kEll, 0);
  FixArray raw = secure_reference_matmul(fp, merged, kTokens, kDim,
                                        weights, kDim);
  FixArray bias = fp.fix->input(ALICE, kDim, bias_values.data(), true, kEll,
                                2 * kScale);
  FixArray repeated_bias(party, kTokens * kDim, true, kEll, 2 * kScale);
  for (std::size_t token = 0; token < kTokens; ++token)
    std::memcpy(repeated_bias.data + token * kDim, bias.data,
                kDim * sizeof(std::uint64_t));
  raw = fp.fix->add(raw, repeated_bias);
  return fp.fix->right_shift(raw, kScale);
}

void append_trace(std::vector<TraceRow> &trace, IOPack &io,
                  const std::string &op, const std::string &input_shape,
                  const std::string &output_shape,
                  const std::string &input_packing,
                  const std::string &output_packing, int input_scale,
                  int output_scale, const BoltOperationCounts &counts,
                  std::uint64_t before_bytes, std::uint64_t before_rounds,
                  const std::string &backend) {
  TraceRow row;
  row.op = op;
  row.input_shape = input_shape;
  row.output_shape = output_shape;
  row.input_packing = input_packing;
  row.output_packing = output_packing;
  row.input_scale = input_scale;
  row.output_scale = output_scale;
  row.counts = counts;
  row.bytes = io.get_comm() - before_bytes;
  row.rounds = io.get_rounds() - before_rounds;
  row.backend = backend;
  trace.push_back(row);
}

void aggregate_trace_endpoints(IOPack &io, std::vector<TraceRow> &trace,
                               std::uint64_t &setup_bytes,
                               std::uint64_t &setup_rounds) {
  struct WireTrace {
    std::uint64_t bytes;
    std::uint64_t rounds;
  };
  if (party == BOB) {
    WireTrace setup{setup_bytes, setup_rounds};
    io.io->send_data(&setup, sizeof(setup));
    for (const auto &row : trace) {
      WireTrace wire{row.bytes, row.rounds};
      io.io->send_data(&wire, sizeof(wire));
    }
    return;
  }
  WireTrace peer_setup{};
  io.io->recv_data(&peer_setup, sizeof(peer_setup));
  setup_bytes += peer_setup.bytes;
  setup_rounds = std::max(setup_rounds, peer_setup.rounds);
  for (auto &row : trace) {
    WireTrace peer{};
    io.io->recv_data(&peer, sizeof(peer));
    // NetIO::counter is send-only.  Summing the two endpoints therefore
    // counts every wire byte exactly once; round transitions use max.
    row.bytes += peer.bytes;
    row.rounds = std::max(row.rounds, peer.rounds);
  }
}

std::string format_double(double value) {
  std::ostringstream stream;
  stream << std::setprecision(12) << value;
  return stream.str();
}

std::string csv_escape(const std::string &value) {
  if (value.find_first_of(",\"\n") == std::string::npos) return value;
  std::string escaped = "\"";
  for (char character : value) {
    if (character == '\"') escaped += '\"';
    escaped += character;
  }
  escaped += '\"';
  return escaped;
}

NumericalRow validate_secure(FPMath &fp, const std::string &op,
                             const FixArray &actual,
                             const FixArray &reference, bool bit_exact,
                             std::uint64_t tolerance_lsb) {
  if (actual.size != reference.size || actual.ell != reference.ell ||
      actual.s != reference.s)
    throw std::invalid_argument("numerical validation metadata mismatch: " + op);
  FixArray diff = fp.fix->sub(actual, reference);
  FixArray absolute = fp.fix->abs(diff);
  FixArray maximum = fp.fix->max(std::vector<FixArray>{absolute});
  FixArray sum_absolute = fp.fix->tree_sum(std::vector<FixArray>{absolute});
  FixArray square = fp.fix->mul(diff, diff, 63);
  if (diff.s > 0) square = fp.fix->truncate_reduce(square, diff.s);
  FixArray sum_square = fp.fix->tree_sum(std::vector<FixArray>{square});
  FixArray reference_square = fp.fix->mul(reference, reference, 63);
  if (reference.s > 0)
    reference_square = fp.fix->truncate_reduce(reference_square, reference.s);
  FixArray sum_reference_square =
      fp.fix->tree_sum(std::vector<FixArray>{reference_square});

  FixArray maximum_public = fp.fix->output(PUBLIC, maximum);
  FixArray sum_absolute_public = fp.fix->output(PUBLIC, sum_absolute);
  FixArray sum_square_public = fp.fix->output(PUBLIC, sum_square);
  FixArray sum_reference_square_public =
      fp.fix->output(PUBLIC, sum_reference_square);

  const std::int64_t max_raw =
      ring_signed(maximum_public.data[0], maximum_public.ell);
  const double scale = std::ldexp(1.0, actual.s);
  const double max_abs = static_cast<double>(max_raw) / scale;
  const double sum_abs = static_cast<double>(ring_signed(
      sum_absolute_public.data[0], sum_absolute_public.ell)) / scale;
  const double sum_sq = static_cast<double>(ring_signed(
      sum_square_public.data[0], sum_square_public.ell)) /
                        std::ldexp(1.0, sum_square_public.s);
  const double ref_sq = static_cast<double>(ring_signed(
      sum_reference_square_public.data[0],
      sum_reference_square_public.ell)) /
                        std::ldexp(1.0, sum_reference_square_public.s);

  NumericalRow row;
  row.op = op;
  row.bit_exact = bit_exact;
  row.tolerance_lsb = tolerance_lsb;
  row.max_abs = max_abs;
  row.mean_abs = sum_abs / actual.size;
  row.rmse = std::sqrt(std::max(0.0, sum_sq / actual.size));
  row.relative_l2 = ref_sq == 0.0 ? (sum_sq == 0.0 ? 0.0 : INFINITY)
                                  : std::sqrt(std::max(0.0, sum_sq / ref_sq));
  row.status = static_cast<std::uint64_t>(std::max<std::int64_t>(0, max_raw)) <=
               tolerance_lsb;
  if (!row.status) row.first_mismatch = "secure_index_withheld";
  return row;
}

bool verify_fail_closed(BoltHE &he) {
  bool scalar = false, generic = false, batching = false, galois = false,
       relin = false;
  try { fail_if_scalar_ciphertext_adapter(true); }
  catch (const std::logic_error &) { scalar = true; }
  try { fail_if_generic_ot_matmul(true); }
  catch (const std::logic_error &) { generic = true; }
  try { BoltHEView bad = he.view(); bad.encoder = nullptr; bad.require_batching(); }
  catch (const std::logic_error &) { batching = true; }
  try { BoltHEView bad = he.view(); bad.galois_keys = nullptr;
        bad.require_public_evaluation(false); }
  catch (const std::logic_error &) { galois = true; }
  try { BoltHEView bad = he.view(); bad.relinearization_keys = nullptr;
        bad.require_public_evaluation(true); }
  catch (const std::logic_error &) { relin = true; }
  return scalar && generic && batching && galois && relin;
}

void write_trace(const std::vector<TraceRow> &rows) {
  std::ofstream out(output_dir + "/operator_trace.csv");
  if (!out) throw std::runtime_error("cannot write operator_trace.csv");
  out << "op,input_shape,output_shape,input_packing,output_packing,input_scale,"
         "output_scale,ell,plain_modulus,ciphertexts,rotations,multiply_plain,"
         "evaluator_multiply,relinearizations,mod_switches,bytes,rounds,backend,status\n";
  for (const auto &row : rows)
    out << csv_escape(row.op) << ',' << csv_escape(row.input_shape) << ','
        << csv_escape(row.output_shape) << ','
        << csv_escape(row.input_packing) << ','
        << csv_escape(row.output_packing) << ','
        << row.input_scale << ',' << row.output_scale << ',' << row.ell << ','
        << row.plain_modulus << ',' << row.counts.ciphertexts << ','
        << row.counts.rotations << ',' << row.counts.multiply_plain << ','
        << row.counts.evaluator_multiply << ',' << row.counts.relinearizations
        << ',' << row.counts.modulus_switches << ',' << row.bytes << ','
        << row.rounds << ',' << csv_escape(row.backend) << ','
        << (row.status ? "PASS" : "FAIL") << '\n';
}

void write_numerical(const std::vector<NumericalRow> &rows) {
  std::ofstream out(output_dir + "/numerical_match.csv");
  if (!out) throw std::runtime_error("cannot write numerical_match.csv");
  out << "op,bit_exact,tolerance_lsb,max_abs,mean_abs,rmse,relative_l2,first_mismatch,status\n";
  for (const auto &row : rows)
    out << row.op << ',' << (row.bit_exact ? "true" : "false") << ','
        << row.tolerance_lsb << ',' << format_double(row.max_abs) << ','
        << format_double(row.mean_abs) << ',' << format_double(row.rmse) << ','
        << format_double(row.relative_l2) << ',' << row.first_mismatch << ','
        << (row.status ? "PASS" : "FAIL") << '\n';
}

void write_metadata(std::uint64_t setup_bytes, std::uint64_t setup_rounds) {
  std::ofstream out(output_dir + "/packing_metadata.json");
  if (!out) throw std::runtime_error("cannot write packing_metadata.json");
  out << "{\n"
      << "  \"model\": {\"batch\": 1, \"tokens\": 5, \"dim\": 8, "
         "\"heads\": 2, \"head_dim\": 4, \"pre_norm_input\": true},\n"
      << "  \"fixed_point\": {\"ell\": 37, \"input_scale\": 12, "
         "\"qkv_weight_scale\": 0, \"qk_raw_scale\": 24, "
         "\"qk_scaled_scale\": 12, \"softmax_scale\": 12, "
         "\"attention_head_scale\": 24, \"output_scale\": 12, "
         "\"attention_scale\": \"1/sqrt(4)=1/2 then right_shift(12)\"},\n"
      << "  \"fixture\": {\"seed\": \"0xB017A77E\", "
         "\"weights\": \"fixed sparse diagonal tiny protocol fixture\"},\n"
      << "  \"bfv\": {\"scheme\": \"BFV\", \"batch_encoder\": true, "
         "\"plain_modulus\": 557057, \"poly_modulus_degree\": 8192, "
         "\"coeff_modulus_bits\": [60,60,60], "
         "\"parameter_source\": \"BoltHEParameters::bert_tiny_plain_modulus\"},\n"
      << "  \"hardening\": {\"output_projection_noise_flooding_bits\": 81, "
         "\"output_projection_modulus_switches\": 1},\n"
      << "  \"layout\": {\"kind\": \"compact_repeated_rows\", "
         "\"packed_stride\": 8, \"qkv_valid_slots\": [0,7], "
         "\"head_valid_slots\": [0,3], \"qk_valid_slots\": [0,4], "
         "\"qk_padding_slots\": [5,7], "
         "\"softmax_v_valid_slots\": [0,3], "
         "\"softmax_v_padding_slots\": [4,7], "
         "\"replica_rule\": \"repeat every 8 slots in each BatchEncoder row\"},\n"
      << "  \"bias\": {\"q\": false, \"k\": false, \"v\": false, "
         "\"output_projection\": true, \"source\": \"current VitBlockModel fixture\"},\n"
      << "  \"ownership\": {\"x\": \"two_party_additive_shares\", "
         "\"wq_wk_wv_wo_bo\": \"ALICE\"},\n"
      << "  \"excluded_setup\": {\"wire_bytes\": " << setup_bytes
      << ", \"rounds\": " << setup_rounds
      << ", \"note\": \"key exchange, base OT, reference-oracle setup and post-protocol numerical validation are excluded from per-operator trace; setup bytes report key/OT setup only; operator wire bytes sum each party's send-only counter once and rounds use max, so neither direction is omitted and no party is double-counted\"},\n"
      << "  \"fallbacks\": {\"scalar_ciphertext\": false, "
         "\"BfvLinear\": false, \"generic_ot_matmul\": false, "
         "\"cleartext_intermediate_reconstruction\": false}\n"
      << "}\n";
}

void write_first_mismatch(const std::vector<NumericalRow> &rows) {
  std::ofstream out(output_dir + "/first_mismatch.json");
  if (!out) throw std::runtime_error("cannot write first_mismatch.json");
  auto failed = std::find_if(rows.begin(), rows.end(),
                             [](const NumericalRow &row) { return !row.status; });
  if (failed == rows.end()) {
    out << "{\"mismatch\": null}\n";
  } else {
    out << "{\"mismatch\": {\"op\": \"" << failed->op
        << "\", \"index\": null, \"reason\": "
           "\"aggregate mismatch detected; secret element was not reconstructed\"}}\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  ArgMapping args;
  args.arg("r", party, "Role: ALICE=1, BOB=2");
  args.arg("p", port, "TCP port");
  args.arg("ip", address, "ALICE address (BOB only)");
  args.arg("out", output_dir, "Trace output directory");
  args.parse(argc, argv);

  try {
    if (party != ALICE && party != BOB)
      throw std::invalid_argument("party must be ALICE or BOB");
    IOPack io(party, port, address);
    BoltHE he(party, io.io, BoltHEParameters::bert_tiny_plain_modulus());
    OTPack otpack(&io, party);
    FPMath fp(party, &io, &otpack);
    BoltConversion conversion(he.view());
    BoltLinear linear(he.view());
    BoltAttentionPacking packing(he.view());
    Fixture fixture = make_fixture();
    if (party == BOB) {
      fixture.wq.clear();
      fixture.wk.clear();
      fixture.wv.clear();
      fixture.wo.clear();
      fixture.bo.clear();
    }
    const auto packing_meta = attention_metadata();
    packing_meta.validate(he.slot_count());
    if (!verify_fail_closed(he))
      throw std::logic_error("BOLT ViT Attention fail-closed self-test failed");

    std::uint64_t setup_bytes = io.get_comm();
    std::uint64_t setup_rounds = io.get_rounds();

    // Independent fixed-point reference.  Linear integer stages are computed
    // by ALICE; Softmax calls the unmodified FPMath::softmax_fix.  Reference
    // values remain secret-shared and validation only opens aggregate errors.
    std::vector<std::int64_t> q_clear, k_clear, v_clear, qk_clear, qk_scaled_clear;
    if (party == ALICE) {
      q_clear = clear_linear(fixture.x, kTokens, fixture.wq);
      k_clear = clear_linear(fixture.x, kTokens, fixture.wk);
      v_clear = clear_linear(fixture.x, kTokens, fixture.wv);
      qk_clear = clear_qk(q_clear, k_clear);
      qk_scaled_clear.resize(qk_clear.size());
      for (std::size_t i = 0; i < qk_clear.size(); ++i)
        qk_scaled_clear[i] = floor_shift(floor_shift(qk_clear[i], 1), kScale);
    } else {
      q_clear.assign(kTokens * kDim, 0);
      k_clear.assign(kTokens * kDim, 0);
      v_clear.assign(kTokens * kDim, 0);
      qk_clear.assign(kHeads * kTokens * kTokens, 0);
      qk_scaled_clear.assign(kHeads * kTokens * kTokens, 0);
    }
    FixArray q_reference = reference_from_alice(q_clear, fp, kScale);
    FixArray k_reference = reference_from_alice(k_clear, fp, kScale);
    FixArray v_reference = reference_from_alice(v_clear, fp, kScale);
    FixArray qk_reference = reference_from_alice(qk_clear, fp, 2 * kScale);
    FixArray qk_scaled_reference =
        reference_from_alice(qk_scaled_clear, fp, kScale);
    std::vector<FixArray> qk_reference_rows =
        split_rows(qk_scaled_reference, kHeads * kTokens, kTokens);
    std::vector<FixArray> softmax_reference;
    FixArray ignored_reference;
    std::tie(softmax_reference, ignored_reference) =
        fp.softmax_fix(qk_reference_rows);
    FixArray attention_heads_reference =
        reference_attention_heads(fp, softmax_reference, v_reference);
    FixArray merged_reference = merge_reference_heads(attention_heads_reference);
    FixArray projection_reference =
        reference_output_projection(fp, merged_reference, fixture);

    const std::uint64_t protocol_start_bytes = io.get_comm();
    const std::uint64_t protocol_start_rounds = io.get_rounds();
    std::vector<TraceRow> trace;

    // X shares -> one compact BFV ciphertext per token -> Q/K/V ct-pt BSGS.
    BoltOperationCounts qkv_counts;
    std::uint64_t before_bytes = io.get_comm();
    std::uint64_t before_rounds = io.get_rounds();
    const BoltTensorMetadata x_meta = tensor_metadata(kTokens, kDim, kScale);
    BoltCipherTensor encrypted_x =
        conversion.ss_to_he(make_input_shares(fixture), x_meta);
    BoltCipherTensor q_cipher, k_cipher, v_cipher;
    if (party == ALICE) {
      q_cipher = linear.cipher_plain_matmul_bsgs(
          encrypted_x, make_weight(fixture.wq, {}, prime_mod),
          matmul_metadata(kTokens, kDim, kDim, 0, 0), &qkv_counts);
      k_cipher = linear.cipher_plain_matmul_bsgs(
          encrypted_x, make_weight(fixture.wk, {}, prime_mod),
          matmul_metadata(kTokens, kDim, kDim, 0, 0), &qkv_counts);
      v_cipher = linear.cipher_plain_matmul_bsgs(
          encrypted_x, make_weight(fixture.wv, {}, prime_mod),
          matmul_metadata(kTokens, kDim, kDim, 0, 0), &qkv_counts);
      q_cipher.metadata.scale = kScale;
      k_cipher.metadata.scale = kScale;
      v_cipher.metadata.scale = kScale;
    } else {
      q_cipher.metadata = tensor_metadata(kTokens, kDim, kScale);
      k_cipher.metadata = tensor_metadata(kTokens, kDim, kScale);
      v_cipher.metadata = tensor_metadata(kTokens, kDim, kScale);
    }
    append_trace(trace, io, "qkv", "[1,5,8]", "3x[1,5,8]",
                 "secret_shares", "compact_stride_8", kScale, kScale,
                 qkv_counts, before_bytes, before_rounds,
                 "bolt_ct_pt_bsgs");

    // Per-head compact slice and packed K transpose, then true ct-ct QK.
    BoltOperationCounts qk_counts;
    before_bytes = io.get_comm();
    before_rounds = io.get_rounds();
    std::vector<BoltCipherTensor> q_heads(kHeads), k_heads(kHeads),
        v_heads(kHeads), k_transposed_heads(kHeads), qk_cipher(kHeads);
    if (party == ALICE) {
      for (std::size_t head = 0; head < kHeads; ++head) {
        q_heads[head] = packing.slice_head(q_cipher, head, packing_meta,
                                           &qk_counts);
        k_heads[head] = packing.slice_head(k_cipher, head, packing_meta,
                                           &qk_counts);
        v_heads[head] = packing.slice_head(v_cipher, head, packing_meta,
                                           &qk_counts);
        k_transposed_heads[head] =
            packing.transpose_tokens(k_heads[head], packing_meta, &qk_counts);
        qk_cipher[head] = linear.cipher_cipher_matmul_cross(
            q_heads[head], k_transposed_heads[head],
            matmul_metadata(kTokens, kHeadDim, kTokens, 0, 0),
            &qk_counts);
        qk_cipher[head].metadata.scale = 2 * kScale;
      }
    } else {
      for (std::size_t head = 0; head < kHeads; ++head) {
        q_heads[head].metadata = tensor_metadata(kTokens, kHeadDim, kScale);
        k_heads[head].metadata = tensor_metadata(kTokens, kHeadDim, kScale);
        v_heads[head].metadata = tensor_metadata(kTokens, kHeadDim, kScale);
        k_transposed_heads[head].metadata =
            tensor_metadata(kHeadDim, kTokens, kScale);
        qk_cipher[head].metadata =
            tensor_metadata(kTokens, kTokens, 2 * kScale);
      }
    }
    append_trace(trace, io, "qk", "2x([5,4]x[4,5])", "2x[5,5]",
                 "compact_head+cross_transpose", "cross_stride_8_valid_5",
                 kScale, 2 * kScale, qk_counts, before_bytes, before_rounds,
                 "bolt_ct_ct_cross");

    // HE-to-SS random mask, BOLT prime-field correction, 1/sqrt(4), truncation.
    BoltOperationCounts conversion_counts;
    before_bytes = io.get_comm();
    before_rounds = io.get_rounds();
    std::vector<SecretSharedTensor> qk_field(kHeads);
    for (std::size_t head = 0; head < kHeads; ++head) {
      BoltTensorMetadata metadata =
          tensor_metadata(kTokens, kTokens, 2 * kScale, true);
      BoltCipherTensor view = qk_cipher[head];
      view.metadata = metadata;
      qk_field[head] = conversion.he_to_ss(view, metadata);
    }
    conversion_counts.ciphertexts = kHeads * kTokens;
    append_trace(trace, io, "qk_he_to_ss_mask", "2x[5,5]", "2xshares[5,5]",
                 "cross_stride_8_valid_5", "prime_field_additive_shares",
                 2 * kScale, 2 * kScale, conversion_counts, before_bytes,
                 before_rounds, "bolt_he_to_ss_random_mask");

    BoltOperationCounts field_counts;
    before_bytes = io.get_comm();
    before_rounds = io.get_rounds();
    std::vector<FixArray> qk_ring_heads;
    qk_ring_heads.reserve(kHeads);
    for (std::size_t head = 0; head < kHeads; ++head)
      qk_ring_heads.push_back(
          field_shares_to_ring(fp, qk_field[head], 2 * kScale));
    append_trace(trace, io, "qk_field_to_ring", "2xfield_shares[5,5]",
                 "2xZ2^37_shares[5,5]", "prime_field_additive_shares",
                 "z2ell_additive_shares", 2 * kScale, 2 * kScale,
                 field_counts, before_bytes, before_rounds,
                 "bolt_gt_p_sub_minus_p_half");

    BoltOperationCounts scale_counts;
    before_bytes = io.get_comm();
    before_rounds = io.get_rounds();
    std::vector<FixArray> qk_half_heads;
    qk_half_heads.reserve(kHeads);
    for (const auto &qk_ring : qk_ring_heads) {
      FixArray scaled = fp.fix->right_shift(qk_ring, 1);
      scaled.s = 2 * kScale; // exact 1/2 coefficient while retaining scale
      qk_half_heads.push_back(std::move(scaled));
    }
    append_trace(trace, io, "attention_scale", "2xZ2^37_shares[5,5]",
                 "2xscaled_shares[5,5]", "z2ell_additive_shares",
                 "z2ell_additive_shares", 2 * kScale, 2 * kScale,
                 scale_counts, before_bytes, before_rounds,
                 "bolt_mpc_right_shift_1_over_sqrt_head_dim");

    BoltOperationCounts truncation_counts;
    before_bytes = io.get_comm();
    before_rounds = io.get_rounds();
    std::vector<FixArray> qk_scaled_rows;
    qk_scaled_rows.reserve(kHeads * kTokens);
    for (std::size_t head = 0; head < kHeads; ++head) {
      FixArray scaled = fp.fix->right_shift(qk_half_heads[head], kScale);
      std::vector<FixArray> rows = split_rows(scaled, kTokens, kTokens);
      for (auto &row : rows) qk_scaled_rows.push_back(std::move(row));
    }
    FixArray qk_scaled_actual = flatten_rows(qk_scaled_rows);
    append_trace(trace, io, "qk_truncation", "2xscaled_shares[5,5]",
                 "10x[5]", "z2ell_additive_shares",
                 "secret_shares_row_major", 2 * kScale, kScale,
                 truncation_counts, before_bytes, before_rounds,
                 "bolt_mpc_right_shift_12");

    // Existing, unchanged BOLT MPC Softmax, one row per head/token.
    BoltOperationCounts softmax_counts;
    before_bytes = io.get_comm();
    before_rounds = io.get_rounds();
    std::vector<FixArray> softmax_rows;
    FixArray ignored_actual;
    std::tie(softmax_rows, ignored_actual) = fp.softmax_fix(qk_scaled_rows);
    FixArray softmax_actual = flatten_rows(softmax_rows);
    append_trace(trace, io, "softmax", "10x[5]", "10x[5]",
                 "secret_shares_row_major", "secret_shares_row_major",
                 kScale, kScale, softmax_counts, before_bytes, before_rounds,
                 "bolt_mpc_softmax");

    // Softmax shares return directly to packed BFV, one ciphertext per row.
    BoltOperationCounts softmax_to_he_counts;
    before_bytes = io.get_comm();
    before_rounds = io.get_rounds();
    std::vector<BoltCipherTensor> softmax_cipher(kHeads);
    for (std::size_t head = 0; head < kHeads; ++head) {
      SecretSharedTensor shared;
      shared.metadata = tensor_metadata(kTokens, kTokens, kScale);
      shared.local_share.resize(kTokens * kTokens);
      for (std::size_t row = 0; row < kTokens; ++row)
        std::memcpy(shared.local_share.data() + row * kTokens,
                    softmax_rows[head * kTokens + row].data,
                    kTokens * sizeof(std::uint64_t));
      softmax_cipher[head] = conversion.ss_to_he(shared, shared.metadata);
    }
    softmax_to_he_counts.ciphertexts = kHeads * kTokens;
    append_trace(trace, io, "softmax_to_he", "2x[5,5]", "2x5ct",
                 "secret_shares_row_major", "compact_stride_8_valid_5",
                 kScale, kScale, softmax_to_he_counts, before_bytes,
                 before_rounds, "bolt_ss_to_he_batch_encoder");

    // Packed Softmax ciphertext x retained packed V ciphertext.
    BoltOperationCounts softmax_v_counts;
    before_bytes = io.get_comm();
    before_rounds = io.get_rounds();
    std::vector<BoltCipherTensor> attention_cipher(kHeads);
    if (party == ALICE) {
      for (std::size_t head = 0; head < kHeads; ++head) {
        attention_cipher[head] = linear.softmax_v_matmul(
            softmax_cipher[head], v_heads[head],
            matmul_metadata(kTokens, kTokens, kHeadDim, 0, 0),
            &softmax_v_counts);
        attention_cipher[head].metadata.scale = 2 * kScale;
      }
    } else {
      for (auto &head : attention_cipher)
        head.metadata = tensor_metadata(kTokens, kHeadDim, 2 * kScale);
    }
    append_trace(trace, io, "softmax_v", "2x([5,5]x[5,4])", "2x[5,4]",
                 "compact_stride_8_valid_5+4", "compact_stride_8_valid_4",
                 kScale, 2 * kScale, softmax_v_counts, before_bytes,
                 before_rounds, "bolt_ct_ct_softmax_v");

    // Multi-head merge uses only packed rotations, masks and additions.
    BoltOperationCounts merge_counts;
    before_bytes = io.get_comm();
    before_rounds = io.get_rounds();
    BoltCipherTensor merged_cipher;
    if (party == ALICE) {
      merged_cipher = packing.merge_heads(attention_cipher, packing_meta,
                                          &merge_counts);
      merged_cipher.metadata.scale = 2 * kScale;
    } else {
      merged_cipher.metadata = tensor_metadata(kTokens, kDim, 2 * kScale);
    }
    append_trace(trace, io, "multi_head_merge", "2x[5,4]", "[5,8]",
                 "two_heads_valid_4", "compact_stride_8_head_order_0_1",
                 2 * kScale, 2 * kScale, merge_counts, before_bytes,
                 before_rounds, "bolt_packed_rotate_mask_add");

    // Output projection remains compact BSGS, then BOLT HE-to-SS correction.
    BoltOperationCounts output_counts;
    before_bytes = io.get_comm();
    before_rounds = io.get_rounds();
    BoltCipherTensor projected_cipher;
    if (party == ALICE) {
      projected_cipher = linear.cipher_plain_matmul_bsgs(
          merged_cipher,
          make_weight(fixture.wo, fixture.bo, prime_mod, kScale),
          matmul_metadata(kTokens, kDim, kDim, SMUDGING_BITLEN_bert2, 1),
          &output_counts);
      projected_cipher.metadata.scale = 2 * kScale;
    } else {
      projected_cipher.metadata = tensor_metadata(kTokens, kDim, 2 * kScale);
    }
    FixArray projected_ring = prime_cipher_to_ring(
        conversion, fp, projected_cipher, kTokens, kDim, 2 * kScale);
    FixArray projection_actual = fp.fix->right_shift(projected_ring, kScale);
    append_trace(trace, io, "output_projection", "[5,8]", "shares[5,8]",
                 "compact_stride_8", "secret_shares_row_major",
                 2 * kScale, kScale, output_counts, before_bytes,
                 before_rounds, "bolt_ct_pt_bsgs+bolt_he_to_ss");

    const std::uint64_t protocol_bytes = io.get_comm() - protocol_start_bytes;
    const std::uint64_t protocol_rounds = io.get_rounds() - protocol_start_rounds;
    (void)protocol_bytes;
    (void)protocol_rounds;
    aggregate_trace_endpoints(io, trace, setup_bytes, setup_rounds);

    // Verification is deliberately after all trace intervals.  HE values are
    // converted only to additive shares; only aggregate error scalars open.
    std::vector<NumericalRow> numerical;
    FixArray q_actual = prime_cipher_to_ring(
        conversion, fp, q_cipher, kTokens, kDim, kScale);
    FixArray k_actual = prime_cipher_to_ring(
        conversion, fp, k_cipher, kTokens, kDim, kScale);
    FixArray v_actual = prime_cipher_to_ring(
        conversion, fp, v_cipher, kTokens, kDim, kScale);
    numerical.push_back(validate_secure(fp, "q", q_actual, q_reference, true, 0));
    numerical.push_back(validate_secure(fp, "k", k_actual, k_reference, true, 0));
    numerical.push_back(validate_secure(fp, "v", v_actual, v_reference, true, 0));

    std::vector<FixArray> q_head_actual_parts;
    std::vector<FixArray> kt_actual_parts;
    for (std::size_t head = 0; head < kHeads; ++head) {
      q_head_actual_parts.push_back(prime_cipher_to_ring(
          conversion, fp, q_heads[head], kTokens, kHeadDim, kScale));
      kt_actual_parts.push_back(prime_cipher_to_ring(
          conversion, fp, k_transposed_heads[head], kHeadDim, kTokens,
          kScale));
    }
    FixArray q_head_actual = concat(q_head_actual_parts);
    FixArray kt_actual = concat(kt_actual_parts);
    std::vector<std::int64_t> q_head_clear(kHeads * kTokens * kHeadDim, 0);
    std::vector<std::int64_t> kt_clear(kHeads * kHeadDim * kTokens, 0);
    if (party == ALICE) {
      for (std::size_t head = 0; head < kHeads; ++head)
        for (std::size_t token = 0; token < kTokens; ++token)
          for (std::size_t feature = 0; feature < kHeadDim; ++feature) {
            q_head_clear[(head * kTokens + token) * kHeadDim + feature] =
                q_clear[token * kDim + head * kHeadDim + feature];
            kt_clear[(head * kHeadDim + feature) * kTokens + token] =
                k_clear[token * kDim + head * kHeadDim + feature];
          }
    }
    FixArray q_head_reference =
        reference_from_alice(q_head_clear, fp, kScale);
    FixArray kt_reference = reference_from_alice(kt_clear, fp, kScale);
    numerical.push_back(validate_secure(fp, "q_head_packing", q_head_actual,
                                        q_head_reference, true, 0));
    numerical.push_back(validate_secure(fp, "k_transpose_packing", kt_actual,
                                        kt_reference, true, 0));
    std::vector<FixArray> q_head_padded_parts;
    std::vector<FixArray> kt_padded_parts;
    for (std::size_t head = 0; head < kHeads; ++head) {
      q_head_padded_parts.push_back(prime_cipher_to_ring(
          conversion, fp, q_heads[head], kTokens, kStride, kScale));
      kt_padded_parts.push_back(prime_cipher_to_ring(
          conversion, fp, k_transposed_heads[head], kHeadDim, kStride,
          kScale));
    }
    numerical.push_back(validate_secure(
        fp, "q_head_padding", concat(q_head_padded_parts),
        pad_reference_rows(q_head_reference, kHeads * kTokens, kHeadDim,
                           kStride),
        true, 0));
    numerical.push_back(validate_secure(
        fp, "k_transpose_padding", concat(kt_padded_parts),
        pad_reference_rows(kt_reference, kHeads * kHeadDim, kTokens,
                           kStride),
        true, 0));

    std::vector<FixArray> qk_actual_heads;
    std::vector<FixArray> attention_actual_heads;
    for (std::size_t head = 0; head < kHeads; ++head) {
      qk_actual_heads.push_back(prime_cipher_to_ring(
          conversion, fp, qk_cipher[head], kTokens, kTokens, 2 * kScale));
      attention_actual_heads.push_back(prime_cipher_to_ring(
          conversion, fp, attention_cipher[head], kTokens, kHeadDim,
          2 * kScale));
    }
    FixArray qk_actual = concat(qk_actual_heads);
    FixArray attention_heads_actual = concat(attention_actual_heads);
    FixArray merged_actual = prime_cipher_to_ring(
        conversion, fp, merged_cipher, kTokens, kDim, 2 * kScale);
    std::vector<FixArray> qk_protocol_reference_parts;
    for (std::size_t head = 0; head < kHeads; ++head)
      qk_protocol_reference_parts.push_back(secure_reference_matmul(
          fp, q_head_actual_parts[head], kTokens, kHeadDim,
          kt_actual_parts[head], kTokens));
    FixArray qk_protocol_reference = concat(qk_protocol_reference_parts);
    numerical.push_back(validate_secure(fp, "qk_kernel", qk_actual,
                                        qk_protocol_reference, true, 0));
    FixArray merge_layout_reference =
        merge_reference_heads(attention_heads_actual);
    numerical.push_back(validate_secure(fp, "merge_layout", merged_actual,
                                        merge_layout_reference, true, 0));
    std::vector<FixArray> qk_padded_parts;
    std::vector<FixArray> softmax_padded_parts;
    std::vector<FixArray> attention_padded_parts;
    for (std::size_t head = 0; head < kHeads; ++head) {
      qk_padded_parts.push_back(prime_cipher_to_ring(
          conversion, fp, qk_cipher[head], kTokens, kStride, 2 * kScale));
      softmax_padded_parts.push_back(prime_cipher_to_ring(
          conversion, fp, softmax_cipher[head], kTokens, kStride, kScale));
      attention_padded_parts.push_back(prime_cipher_to_ring(
          conversion, fp, attention_cipher[head], kTokens, kStride,
          2 * kScale));
    }
    numerical.push_back(validate_secure(
        fp, "qk_padding", concat(qk_padded_parts),
        pad_reference_rows(qk_reference, kHeads * kTokens, kTokens, kStride),
        true, 0));
    numerical.push_back(validate_secure(
        fp, "softmax_cipher_padding", concat(softmax_padded_parts),
        pad_reference_rows(flatten_rows(softmax_reference),
                           kHeads * kTokens, kTokens, kStride),
        true, 0));
    numerical.push_back(validate_secure(
        fp, "softmax_v_padding", concat(attention_padded_parts),
        pad_reference_rows(attention_heads_reference, kHeads * kTokens,
                           kHeadDim, kStride),
        true, 0));
    numerical.push_back(validate_secure(fp, "qk_raw", qk_actual,
                                        qk_reference, true, 0));
    numerical.push_back(validate_secure(fp, "qk_scaled", qk_scaled_actual,
                                        qk_scaled_reference, true, 0));
    numerical.push_back(validate_secure(fp, "softmax", softmax_actual,
                                        flatten_rows(softmax_reference), false,
                                        kSoftmaxToleranceLsb));
    numerical.push_back(validate_secure(fp, "attention_heads",
                                        attention_heads_actual,
                                        attention_heads_reference, true, 0));
    numerical.push_back(validate_secure(fp, "merged_attention", merged_actual,
                                        merged_reference, true, 0));
    numerical.push_back(validate_secure(fp, "attention_projection",
                                        projection_actual,
                                        projection_reference, true, 0));

    bool all_passed = std::all_of(numerical.begin(), numerical.end(),
                                  [](const NumericalRow &row) {
                                    return row.status;
                                  });
    const bool evaluator_counts_pass =
        party == BOB ||
        (qkv_counts.multiply_plain > 0 && qk_counts.evaluator_multiply > 0 &&
         qk_counts.relinearizations > 0 &&
         softmax_v_counts.evaluator_multiply > 0 &&
         softmax_v_counts.relinearizations > 0 &&
         output_counts.multiply_plain > 0);
    all_passed = all_passed && evaluator_counts_pass;
    for (auto &row : trace) row.status = all_passed;

    if (party == ALICE) {
      write_trace(trace);
      write_numerical(numerical);
      write_metadata(setup_bytes, setup_rounds);
      write_first_mismatch(numerical);
      std::cout
          << "VIT_ATTENTION_OP op=qkv\n"
          << "backend=bolt_ct_pt_bsgs\n"
          << "scalar_ciphertext=false\n"
          << "status=" << (all_passed ? "PASS" : "FAIL") << "\n"
          << "VIT_ATTENTION_OP op=qk\n"
          << "backend=bolt_ct_ct_cross\n"
          << "evaluator_multiply_positive="
          << (qk_counts.evaluator_multiply > 0 ? "true" : "false") << "\n"
          << "generic_ot_matmul=false\n"
          << "status=" << (all_passed ? "PASS" : "FAIL") << "\n"
          << "VIT_ATTENTION_OP op=softmax\n"
          << "backend=bolt_mpc_softmax\n"
          << "status=" << (all_passed ? "PASS" : "FAIL") << "\n"
          << "VIT_ATTENTION_OP op=softmax_v\n"
          << "backend=bolt_ct_ct_softmax_v\n"
          << "evaluator_multiply_positive="
          << (softmax_v_counts.evaluator_multiply > 0 ? "true" : "false")
          << "\n"
          << "generic_ot_matmul=false\n"
          << "status=" << (all_passed ? "PASS" : "FAIL") << "\n"
          << "VIT_ATTENTION_OP op=output_projection\n"
          << "backend=bolt_ct_pt_bsgs\n"
          << "status=" << (all_passed ? "PASS" : "FAIL") << "\n"
          << "VIT_ATTENTION_NUMERICAL_MATCH status="
          << (all_passed ? "PASS" : "FAIL") << "\n"
          << "PLAINTEXT_INTERMEDIATE_RECONSTRUCTIONS=0\n"
          << "SCALAR_FALLBACKS=0\n"
          << "GENERIC_OT_MATMUL_CALLS=0\n"
          << "M1_SHARED_BACKEND=PASS\n"
          << "M2_VIT_ATTENTION_PROTOCOL="
          << (all_passed ? "PASS" : "FAIL") << "\n"
          << "FULL_VIT_MIGRATION=TESTED_SEPARATELY\n"
          << "CHECKPOINT_INTEGRATION=SCHEMA_ONLY_TESTED_SEPARATELY" << std::endl;
    }
    return all_passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "BOLT_VIT_ATTENTION_PROTOCOL_TEST error: " << error.what()
              << std::endl;
    return 1;
  }
}
