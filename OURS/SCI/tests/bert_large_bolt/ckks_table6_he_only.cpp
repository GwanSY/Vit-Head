#include "linear_ckks.h"

#include "LinearHE/utils-HE.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// SCI-LinearHE selects its process-global arithmetic prime through this
// symbol.  This executable does not call either conversion protocol, but the
// linked BLB library still requires the formal BERT value.
int bitlength = 37;

namespace {

constexpr int kSequence = 128;
constexpr int kHidden = 768;
constexpr int kIntermediate = 3072;
constexpr int kFixedScale = 12;
constexpr double kPaperGate = 1e-11;

int party = 0;
int port = 65400;
std::string address = "127.0.0.1";
std::string mode = "audit";
std::string model_path =
    "checkpoints/official-blb-mrpc/weights_txt";
std::string oracle_path = "outputs/bert_oracle_public_mask_v3";

struct Table6Profile {
  int block;
  const char *subgraph;
  std::size_t degree;
  std::vector<int> current_constructor_bits;
  std::vector<int> paper_reported_bits;
  int scale_bits;
  int multiplicative_depth;
  double paper_mse;
};

const std::array<Table6Profile, 5> &profiles() {
  // Keep this snapshot adjacent to the executable rather than changing the
  // production HE constructors merely for a diagnostic.  audit_profiles()
  // prints a source locator, and strict source-token checks below make drift
  // fail closed.
  static const std::array<Table6Profile, 5> value{{
      {1, "FFN2+residual+LayerNorm(part1)", 16384,
       {60, 34, 60, 60, 60, 60}, {60, 60, 60, 60, 60}, 38, 4, 7.2e-13},
      {2, "LayerNorm(part2)+QKV+QK^T", 32768,
       {60, 34, 60, 60, 60, 60, 60},
       {60, 35, 60, 60, 60, 60, 60, 60}, 44, 7, 3.4e-12},
      {3, "Softmax linear operators", 32768,
       {60, 40, 40, 40, 40, 40, 40, 40, 60},
       {60, 40, 40, 40, 40, 40, 40, 40, 60}, 40, 7, 5.2e-13},
      {4, "Softmax*V+output projection+LayerNorm(part1)", 32768,
       {60, 60, 60, 60, 60, 60, 60},
       {60, 60, 60, 60, 60, 60, 60}, 43, 7, 2.2e-13},
      {5, "LayerNorm(part2)+FFN1+GeLU linear polynomial", 32768,
       {60, 32, 60, 60, 60, 60, 60, 60},
       {60, 32, 60, 60, 60, 60, 60, 60}, 46, 6, 6.9e-14},
  }};
  return value;
}

std::string join_bits(const std::vector<int> &bits) {
  std::string result = "{";
  for (std::size_t i = 0; i < bits.size(); ++i) {
    if (i != 0) {
      result += ',';
    }
    result += std::to_string(bits[i]);
  }
  result += '}';
  return result;
}

int significant_bits(std::uint64_t value) {
  int bits = 0;
  while (value != 0) {
    ++bits;
    value >>= 1;
  }
  return bits;
}

std::vector<int> context_modulus_bits(
    const std::shared_ptr<const seal::SEALContext::ContextData> &data) {
  if (!data) {
    throw std::runtime_error("SEAL context data is unavailable");
  }
  std::vector<int> result;
  for (const auto &prime : data->parms().coeff_modulus()) {
    result.push_back(significant_bits(prime.value()));
  }
  return result;
}

void require_source_profile_tokens() {
  // The production profiles are currently literals in set_HE().  This check
  // prevents this independent audit snapshot from silently going stale.
  std::string source_path = __FILE__;
  const auto slash = source_path.find_last_of("/\\");
  source_path = slash == std::string::npos
                    ? "linear_ckks.cpp"
                    : source_path.substr(0, slash + 1) + "linear_ckks.cpp";
  std::ifstream source(source_path);
  if (!source.good()) {
    source.clear();
    source.open("SCI/tests/bert_large_bolt/linear_ckks.cpp");
  }
  if (!source.good()) {
    throw std::runtime_error(
        "cannot open production linear_ckks.cpp for profile drift check");
  }
  const std::string text((std::istreambuf_iterator<char>(source)),
                         std::istreambuf_iterator<char>());
  const std::array<const char *, 5> tokens{{
      "std::vector<int>{60, 34, 60, 60, 60, 60}",
      "std::vector<int>{60, 34, 60, 60, 60, 60, 60}",
      "std::vector<int>{60, 40, 40, 40, 40, 40, 40, 40, 60}",
      "std::vector<int>{60, 60, 60, 60, 60, 60, 60}",
      "std::vector<int>{60, 32, 60, 60, 60, 60, 60, 60}",
  }};
  for (const char *token : tokens) {
    if (text.find(token) == std::string::npos) {
      throw std::runtime_error(
          "Table 6 profile snapshot drifted from Linear_ckks::set_HE");
    }
  }
}

void audit_profiles() {
  require_source_profile_tokens();
  bool all_declared_match = true;
  for (const auto &profile : profiles()) {
    seal::EncryptionParameters parameters(seal::scheme_type::ckks);
    parameters.set_poly_modulus_degree(profile.degree);
    parameters.set_coeff_modulus(seal::CoeffModulus::Create(
        profile.degree, profile.current_constructor_bits));
    seal::SEALContext context(parameters, true,
                              seal::sec_level_type::tc128);
    if (!context.parameters_set()) {
      throw std::runtime_error("current Table 6 CKKS parameters are invalid");
    }
    const auto key_bits = context_modulus_bits(context.key_context_data());
    const auto data_bits = context_modulus_bits(context.first_context_data());
    const bool declared_match =
        profile.current_constructor_bits == profile.paper_reported_bits;
    all_declared_match &= declared_match;
    std::cout << "TABLE6_PARAMETER_AUDIT block=" << profile.block
              << " subgraph=\"" << profile.subgraph << "\""
              << " degree=" << profile.degree
              << " mul_depth=" << profile.multiplicative_depth
              << " scale_bits=" << profile.scale_bits
              << " current_constructor_chain="
              << join_bits(profile.current_constructor_bits)
              << " seal_key_level_chain=" << join_bits(key_bits)
              << " seal_first_data_chain=" << join_bits(data_bits)
              << " paper_reported_chain="
              << join_bits(profile.paper_reported_bits)
              << " paper_mse=" << std::scientific << profile.paper_mse
              << " declared_match=" << (declared_match ? 1 : 0)
              << std::defaultfloat << std::endl;
  }
  std::cout << "TABLE6_PARAMETER_AUDIT_SUMMARY status=PASS"
            << " audited_blocks=5 paper_declared_match="
            << (all_declared_match ? 1 : 0)
            << " source=Linear_ckks::set_HE"
            << " note=paper_chain_semantics_are_reported_verbatim_while_SEAL_key_and_first_data_levels_are_separate"
            << std::endl;
  if (!all_declared_match) {
    std::cout << "TABLE6_PAPER_PARAMETER_MATCH status=FAIL"
              << " mismatched_blocks=1,2"
              << " implication=precision_must_be_measured_and_cannot_be_claimed_from_the_paper_table"
              << std::endl;
  }
}

std::uint64_t ring_mask(int bits) {
  return bits == 64 ? ~std::uint64_t{0}
                    : ((std::uint64_t{1} << bits) - 1);
}

std::int64_t centered_ring(std::uint64_t value, int bits) {
  value &= ring_mask(bits);
  const std::uint64_t sign = std::uint64_t{1} << (bits - 1);
  if ((value & sign) == 0) {
    return static_cast<std::int64_t>(value);
  }
  return static_cast<std::int64_t>(value) -
         static_cast<std::int64_t>(std::uint64_t{1} << bits);
}

std::vector<std::uint64_t> read_oracle_share(const std::string &path,
                                             std::uint64_t expected_size) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    throw std::runtime_error("missing oracle share: " + path);
  }
  constexpr char magic[] = "BLB_ORACLE_SHARE_V1\0";
  std::array<char, sizeof(magic) - 1> actual_magic{};
  input.read(actual_magic.data(), actual_magic.size());
  if (!input || std::memcmp(actual_magic.data(), magic,
                            actual_magic.size()) != 0) {
    throw std::runtime_error("invalid oracle share magic: " + path);
  }
  std::uint64_t size = 0;
  std::int32_t ell = 0;
  std::int32_t scale = 0;
  input.read(reinterpret_cast<char *>(&size), sizeof(size));
  input.read(reinterpret_cast<char *>(&ell), sizeof(ell));
  input.read(reinterpret_cast<char *>(&scale), sizeof(scale));
  if (!input || size != expected_size || ell != bitlength ||
      scale != kFixedScale) {
    throw std::runtime_error("oracle share metadata mismatch: " + path);
  }
  std::vector<std::uint64_t> values(size);
  input.read(reinterpret_cast<char *>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(values[0])));
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("oracle share payload mismatch: " + path);
  }
  return values;
}

std::vector<std::int64_t> reconstruct_public_fixed_stage(
    const std::string &directory, const std::string &stage,
    std::uint64_t expected_size) {
  const auto alice =
      read_oracle_share(directory + '/' + stage + ".alice.bin", expected_size);
  const auto bob =
      read_oracle_share(directory + '/' + stage + ".bob.bin", expected_size);
  std::vector<std::int64_t> result(expected_size);
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i] = centered_ring(alice[i] + bob[i], bitlength);
  }
  return result;
}

std::string with_slash(std::string path) {
  if (!path.empty() && path.back() != '/') {
    path.push_back('/');
  }
  return path;
}

std::vector<double> row_to_column(const std::vector<std::int64_t> &row,
                                  int rows, int columns, int scale_bits) {
  if (row.size() != static_cast<std::size_t>(rows * columns)) {
    throw std::runtime_error("row-to-column input shape mismatch");
  }
  std::vector<double> result(row.size());
  const double denominator = std::ldexp(1.0, scale_bits);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      result[j * rows + i] = static_cast<double>(row[i * columns + j]) /
                             denominator;
    }
  }
  return result;
}

std::vector<double> packed_to_row(const std::vector<double> &packed,
                                  const FCMetadata &data) {
  const int ciphertexts =
      data.image_size * data.filter_w / data.slot_count;
  if (packed.size() !=
      static_cast<std::size_t>(ciphertexts * data.slot_count)) {
    throw std::runtime_error("packed output shape mismatch");
  }
  std::vector<double> output(data.image_size * data.filter_w);
  for (int ct = 0; ct < ciphertexts; ++ct) {
    for (int slot = 0; slot < data.slot_count; ++slot) {
      int j = slot / data.image_size;
      const int token = slot % data.image_size;
      if (slot >= data.slot_count / 2) {
        j -= data.slot_count / data.image_size / 2;
        j += data.filter_w / 2;
      }
      j += ct * data.slot_count / data.image_size / 2;
      output[token * data.filter_w + j] =
          packed[ct * data.slot_count + slot];
    }
  }
  return output;
}

std::vector<double> clear_ffn2(
    const std::vector<std::int64_t> &gelu,
    const std::vector<std::vector<std::uint64_t>> &weights,
    const std::vector<std::uint64_t> &bias) {
  if (gelu.size() != static_cast<std::size_t>(kSequence * kIntermediate) ||
      weights.size() != kIntermediate || bias.size() != kHidden) {
    throw std::runtime_error("clear FFN2 input/checkpoint shape mismatch");
  }
  for (const auto &row : weights) {
    if (row.size() != kHidden) {
      throw std::runtime_error("clear FFN2 weight row mismatch");
    }
  }
  std::vector<double> output(kSequence * kHidden);
  const double input_denominator = std::ldexp(1.0, kFixedScale);
  const double weight_denominator = std::ldexp(1.0, 5);
  const double bias_denominator = std::ldexp(1.0, 9);
#pragma omp parallel for
  for (int token = 0; token < kSequence; ++token) {
    for (int out = 0; out < kHidden; ++out) {
      long double sum =
          static_cast<long double>(
              static_cast<std::int64_t>(bias[out])) /
          bias_denominator;
      for (int in = 0; in < kIntermediate; ++in) {
        const long double x =
            static_cast<long double>(gelu[token * kIntermediate + in]) /
            input_denominator;
        const long double w = static_cast<long double>(
                                  static_cast<std::int64_t>(weights[in][out])) /
                              weight_denominator;
        sum += x * w;
      }
      output[token * kHidden + out] = static_cast<double>(sum);
    }
  }
  return output;
}

struct ErrorMetrics {
  double mse = 0.0;
  double max_abs = 0.0;
  double mean_abs = 0.0;
};

ErrorMetrics error_metrics(const std::vector<double> &actual,
                           const std::vector<double> &expected) {
  if (actual.size() != expected.size() || actual.empty()) {
    throw std::runtime_error("precision comparison shape mismatch");
  }
  long double square_sum = 0.0L;
  long double abs_sum = 0.0L;
  double max_abs = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = actual[i] - expected[i];
    if (!std::isfinite(error)) {
      throw std::runtime_error("HE-only comparison produced non-finite data");
    }
    const double absolute = std::abs(error);
    max_abs = std::max(max_abs, absolute);
    abs_sum += absolute;
    square_sum += static_cast<long double>(error) * error;
  }
  return {static_cast<double>(square_sum / actual.size()), max_abs,
          static_cast<double>(abs_sum / actual.size())};
}

void run_block1_ffn2() {
  if (party != sci::ALICE && party != sci::BOB) {
    throw std::runtime_error("block1 mode requires r=1 or r=2");
  }
  sci::NetIO io(party == sci::ALICE ? nullptr : address.c_str(), port);
  const auto &profile = profiles().front();
  HE_CKKS he(party, &io, profile.degree,
             profile.current_constructor_bits, 20, profile.scale_bits,
             LOCATION::HOST);

  FCMetadata data{};
  data.filter_h = kIntermediate;
  data.filter_w = kHidden;
  data.image_size = kSequence;
  data.slot_count = static_cast<int>(he.num_slots);
  const int input_ciphertexts =
      kSequence * kIntermediate / data.slot_count;
  const int output_ciphertexts =
      kSequence * kHidden / data.slot_count;
  if (input_ciphertexts * data.slot_count != kSequence * kIntermediate ||
      output_ciphertexts * data.slot_count != kSequence * kHidden) {
    throw std::runtime_error("Block 1 fixed tensor is not slot aligned");
  }

  std::vector<UnifiedCiphertext> encrypted_input(
      input_ciphertexts, UnifiedCiphertext(LOCATION::HOST));
  if (party == sci::BOB) {
    const auto gelu = reconstruct_public_fixed_stage(
        oracle_path, "fgf4_ln_ffn_gelu",
        static_cast<std::uint64_t>(kSequence) * kIntermediate);
    const auto input_column =
        row_to_column(gelu, kSequence, kIntermediate, kFixedScale);
#pragma omp parallel for
    for (int i = 0; i < input_ciphertexts; ++i) {
      std::vector<double> slots(
          input_column.begin() + i * data.slot_count,
          input_column.begin() + (i + 1) * data.slot_count);
      seal::Plaintext plain;
      he.encoder->encode(slots, std::ldexp(1.0, profile.scale_bits), plain);
      he.encryptor->encrypt(plain, encrypted_input[i]);
    }
    send_encrypted_vector(&io, encrypted_input);

    std::vector<UnifiedCiphertext> encrypted_output(
        output_ciphertexts, UnifiedCiphertext(LOCATION::HOST));
    recv_encrypted_vector(he.context, &io, encrypted_output);
    std::vector<double> packed_output;
    packed_output.reserve(static_cast<std::size_t>(output_ciphertexts) *
                          data.slot_count);
    for (auto &ciphertext : encrypted_output) {
      seal::Plaintext plain;
      he.decryptor->decrypt(ciphertext, plain);
      std::vector<double> slots;
      he.encoder->decode(plain, slots);
      packed_output.insert(packed_output.end(), slots.begin(),
                           slots.begin() + data.slot_count);
    }
    const auto actual = packed_to_row(packed_output, data);
    const std::string model = with_slash(model_path);
    const auto weights = read_data(
        model + "bert.encoder.layer.0.output.dense.weight.txt");
    const auto bias = read_bias(
        model + "bert.encoder.layer.0.output.dense.bias.txt", kHidden);
    const auto expected = clear_ffn2(gelu, weights, bias);
    const auto metrics = error_metrics(actual, expected);
    const auto context_data = he.context->get_context_data(
        encrypted_output.front().hcipher().parms_id());
    if (!context_data) {
      throw std::runtime_error("Block 1 output level is unavailable");
    }
    const bool pass = metrics.mse < kPaperGate;
    std::cout << std::scientific << std::setprecision(12)
              << "TABLE6_HE_ONLY block=1 stage=ffn2_matmul status="
              << (pass ? "PASS" : "FAIL")
              << " mse=" << metrics.mse
              << " max_abs=" << metrics.max_abs
              << " mean_abs=" << metrics.mean_abs
              << " paper_gate=" << kPaperGate
              << " paper_block1_reported_mse=" << profile.paper_mse
              << " input_scale_bits=" << profile.scale_bits
              << " output_scale_bits="
              << std::log2(encrypted_output.front().scale())
              << " output_level_limbs="
              << context_data->parms().coeff_modulus().size()
              << " layout_in=feature_token_column_major"
              << " layout_out=bolt_cross_packing_then_public_row_postprocess"
              << " values=" << actual.size()
              << " conversion_count=0"
              << " input_source=real_layer0_fixed_oracle_public_test_vector"
              << std::defaultfloat << std::endl;
    std::cout << "TABLE6_BLOCK1 status=BLOCKED"
              << " completed=real_FFN2_ct_pt_MatMul"
              << " missing=residual_add_and_LayerNorm_HE_subgraph"
              << " reason=partial_subgraph_is_not_the_paper_full_Block1_MSE"
              << std::endl;
    std::cout << "TABLE6_BLOCKS_2_TO_5 status=BLOCKED"
              << " reason=independent_HE_only_full_subgraphs_not_yet_implemented"
              << " conversion_metrics_included=0" << std::endl;
    if (!pass) {
      throw std::runtime_error("Block 1 FFN2 HE-only MSE gate failed");
    }
  } else {
    const std::string model = with_slash(model_path);
    const auto weights = read_data(
        model + "bert.encoder.layer.0.output.dense.weight.txt");
    const auto bias = read_bias(
        model + "bert.encoder.layer.0.output.dense.bias.txt", kHidden);
    Linear_ckks linear;
    linear.party = party;
    linear.io = &io;
    auto preprocessed = linear.params_preprocessing_ct_pt(
        &he, kSequence, kIntermediate, kHidden, weights, bias, data, 5, 9);
    recv_encrypted_vector(he.context, &io, encrypted_input);
    std::vector<UnifiedCiphertext> encrypted_output;
    linear.linear_2(&he, encrypted_input, preprocessed, data,
                    encrypted_output);
    send_encrypted_vector(&io, encrypted_output);
    std::cout << "TABLE6_HE_ONLY_SERVER block=1 stage=ffn2_matmul"
              << " status=PASS input_cts=" << encrypted_input.size()
              << " output_cts=" << encrypted_output.size()
              << " rotations=" << linear.rotation_count
              << " ct_pt_multiply=" << linear.ct_pt_multiply_count
              << " conversion_count=0" << std::endl;
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    ArgMapping arguments;
    arguments.arg("mode", mode, "audit or block1");
    arguments.arg("r", party, "Role for block1 mode: ALICE=1, BOB=2");
    arguments.arg("p", port, "Block1 two-party port");
    arguments.arg("ip", address, "ALICE address");
    arguments.arg("model", model_path, "Real BLB checkpoint directory");
    arguments.arg("oracle", oracle_path,
                  "Public fixed-clear layer0 oracle directory");
    arguments.parse(argc, argv);

    audit_profiles();
    if (mode == "audit") {
      std::cout << "TABLE6_HE_ONLY_SUMMARY status=BLOCKED"
                << " reason=precision_execution_not_requested"
                << " run=mode=block1_for_real_FFN2_partial_subgraph"
                << " conversion_metrics_included=0" << std::endl;
      return 0;
    }
    if (mode != "block1") {
      throw std::runtime_error("mode must be audit or block1");
    }
    run_block1_ffn2();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "TABLE6_HE_ONLY_FATAL status=FAIL reason=" << error.what()
              << std::endl;
    return 1;
  }
}
