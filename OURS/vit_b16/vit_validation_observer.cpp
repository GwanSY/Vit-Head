#include "vit_validation_observer.h"

#ifdef BLB_VIT_VALIDATION_OUTPUTS

#include "FloatingPoint/fixed-point.h"
#include "defines.h"

#include <openssl/sha.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace sci;

namespace {

constexpr int kObserverEll = 43;
constexpr int kObserverScale = 18;
constexpr std::uint64_t kSafeMaxRaw = 1ULL << 31; // 8192 at scale 18.
const char kShareMagic[] = "BLB_VIT_ORACLE_SHARE_V1";

std::string sha256_file(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot hash validation file: " + path);
  SHA256_CTX context;
  SHA256_Init(&context);
  char buffer[1 << 20];
  while (input) {
    input.read(buffer, sizeof(buffer));
    const std::streamsize count = input.gcount();
    if (count > 0)
      SHA256_Update(&context, buffer, static_cast<std::size_t>(count));
  }
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256_Final(digest, &context);
  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0');
  for (unsigned char byte : digest)
    encoded << std::setw(2) << static_cast<unsigned>(byte);
  return encoded.str();
}

std::string sha256_text(const std::string &value) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char *>(value.data()), value.size(),
         digest);
  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0');
  for (unsigned char byte : digest)
    encoded << std::setw(2) << static_cast<unsigned>(byte);
  return encoded.str();
}

bool valid_sha256(const std::string &value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void require_sha256(const std::string &actual, const std::string &expected,
                    const std::string &what) {
  if (!valid_sha256(expected) || actual != expected)
    throw std::runtime_error("ViT oracle " + what + " SHA-256 mismatch");
}

template <typename T>
void read_named(std::istream &input, const std::string &wanted, T &value) {
  std::string name;
  if (!(input >> name >> value) || name != wanted)
    throw std::runtime_error("ViT oracle manifest expected field: " + wanted);
}

std::vector<std::string> expected_stages(std::size_t blocks) {
  std::vector<std::string> result{"patch_projection"};
  for (std::size_t layer = 0; layer < blocks; ++layer) {
    result.push_back("block_" + std::to_string(layer) + ".q");
    result.push_back("block_" + std::to_string(layer) + ".k");
    result.push_back("block_" + std::to_string(layer) + ".v");
    result.push_back("block_" + std::to_string(layer) + ".qk");
    result.push_back("block_" + std::to_string(layer) + ".softmax");
    result.push_back("block_" + std::to_string(layer) +
                     ".attention_context");
    result.push_back("block_" + std::to_string(layer) +
                     ".output_projection");
    result.push_back("block_" + std::to_string(layer) +
                     ".attention_residual");
    result.push_back("block_" + std::to_string(layer) + ".ffn_residual");
  }
  result.push_back("cls");
  result.push_back("logits");
  return result;
}

std::string csv(const std::string &value) {
  std::string escaped = value;
  std::size_t position = 0;
  while ((position = escaped.find('"', position)) != std::string::npos) {
    escaped.insert(position, 1, '"');
    position += 2;
  }
  return '"' + escaped + '"';
}

FixArray sum_halves(const FixArray &wide, std::size_t elements) {
  if (wide.size != static_cast<int>(2 * elements) || wide.ell != 64)
    throw std::invalid_argument("ViT observer aggregate shape mismatch");
  FixArray result(wide.party, 2, false, 64, wide.s);
  result.data[0] = result.data[1] = 0;
  for (std::size_t reference = 0; reference < 2; ++reference)
    for (std::size_t index = 0; index < elements; ++index)
      result.data[reference] += wide.data[reference * elements + index];
  return result;
}

double decoded(std::uint64_t raw, int scale) {
  return std::ldexp(static_cast<double>(raw), -scale);
}

int ceil_log2(std::size_t value) {
  int result = 0;
  std::size_t power = 1;
  while (power < value) {
    power <<= 1;
    ++result;
  }
  return result;
}

__uint128_t square_sum_raw(std::uint64_t high_sum,
                           std::uint64_t low_sum, int split_bits) {
  return (static_cast<__uint128_t>(high_sum) << split_bits) + low_sum;
}

long double decoded_square_sum(std::uint64_t high_sum,
                               std::uint64_t low_sum, int split_bits,
                               std::size_t elements) {
  const __uint128_t raw_sum = square_sum_raw(high_sum, low_sum, split_bits);
  return std::ldexp(static_cast<long double>(raw_sum),
                    -2 * kObserverScale) /
         static_cast<long double>(elements);
}

void validate_raw_aggregates(std::uint64_t maximum,
                             std::uint64_t sum_absolute,
                             std::uint64_t square_high,
                             std::uint64_t square_low, int split_bits,
                             std::size_t elements,
                             const std::string &stage,
                             const std::string &reference) {
  const __uint128_t count = elements;
  const __uint128_t max_raw = maximum;
  const __uint128_t abs_raw = sum_absolute;
  const __uint128_t square_raw =
      square_sum_raw(square_high, square_low, split_bits);
  const __uint128_t max_square = max_raw * max_raw;
  const __uint128_t split_base = __uint128_t{1} << split_bits;
  const bool all_zero = maximum == 0 && sum_absolute == 0 && square_raw == 0;
  if (split_bits <= 0 || split_bits >= 63 || square_low >= split_base ||
      abs_raw < max_raw || abs_raw > count * max_raw ||
      square_raw < max_square || square_raw > count * max_square ||
      abs_raw * abs_raw > count * square_raw ||
      ((maximum == 0) != all_zero))
    throw std::runtime_error("ViT observer exact aggregate invariant failed at " +
                             stage + " for " + reference);
}

void validate_metrics(double maximum, double mean, double rmse,
                      const std::string &stage,
                      const std::string &reference) {
  const double tolerance =
      1e-12 * std::max({1.0, maximum, mean, rmse});
  if (!std::isfinite(maximum) || !std::isfinite(mean) ||
      !std::isfinite(rmse) || maximum < 0.0 || mean < 0.0 || rmse < 0.0 ||
      mean > maximum + tolerance || rmse < mean - tolerance ||
      rmse > maximum + tolerance)
    throw std::runtime_error("ViT observer metric invariant failed at " +
                             stage + " for " + reference);
}

} // namespace

VitValidationObserver::VitValidationObserver(
    int party, IOPack &io, FPMath &fp, const std::string &manifest_path,
    const std::string &model_path, const std::string &input_path,
    const std::string &reference_path,
    const std::string &runtime_source_sha256, std::size_t logical_tokens,
    std::size_t executed_blocks, const std::string &output_dir)
    : party_(party), io_(io), fp_(fp), manifest_path_(manifest_path),
      manifest_dir_(std::filesystem::path(manifest_path).parent_path().string()),
      output_dir_(output_dir) {
  load_manifest(manifest_path, model_path, input_path, reference_path,
                runtime_source_sha256, logical_tokens, executed_blocks);
}

void VitValidationObserver::load_manifest(
    const std::string &manifest_path, const std::string &model_path,
    const std::string &input_path, const std::string &reference_path,
    const std::string &runtime_source_sha256, std::size_t logical_tokens,
    std::size_t executed_blocks) {
  std::ifstream input(manifest_path);
  if (!input) throw std::runtime_error("ViT oracle manifest is missing");
  std::string magic;
  if (!(input >> magic) || magic != "BLB_VIT_ORACLE_MANIFEST_V2")
    throw std::runtime_error("ViT oracle manifest magic mismatch");
  std::string model_sha, input_sha, reference_sha, source_sha;
  std::size_t manifest_logical = 0, manifest_blocks = 0, stage_count = 0;
  read_named(input, "model_sha256", model_sha);
  read_named(input, "input_sha256", input_sha);
  read_named(input, "reference_sha256", reference_sha);
  read_named(input, "source_checkpoint_binding_sha256", source_sha);
  read_named(input, "observer_ell", ell_);
  read_named(input, "observer_scale", scale_);
  read_named(input, "logical_tokens", manifest_logical);
  read_named(input, "executed_blocks", manifest_blocks);
  read_named(input, "clear_quantization_bound", clear_quantization_bound_);
  read_named(input, "stages", stage_count);
  if (ell_ != kObserverEll || scale_ != kObserverScale ||
      manifest_logical != logical_tokens || manifest_blocks != executed_blocks ||
      !std::isfinite(clear_quantization_bound_) ||
      clear_quantization_bound_ != std::ldexp(0.5, -kObserverScale))
    throw std::runtime_error("ViT oracle manifest arithmetic/config mismatch");
  if (!valid_sha256(model_sha) || !valid_sha256(input_sha) ||
      !valid_sha256(reference_sha) || !valid_sha256(source_sha))
    throw std::runtime_error("ViT oracle manifest has an invalid SHA-256");

  const std::vector<std::string> expected = expected_stages(executed_blocks);
  if (stage_count != expected.size())
    throw std::runtime_error("ViT oracle stage count mismatch");
  stages_.reserve(stage_count);
  const std::string role = party_ == ALICE ? "alice" : "bob";
  for (std::size_t index = 0; index < stage_count; ++index) {
    std::string tag;
    Stage stage;
    if (!(input >> tag >> stage.name >> stage.rows >> stage.cols >>
          stage.elements >> stage.packing >> stage.source_scale) ||
        tag != "stage" ||
        stage.name != expected[index] || stage.rows == 0 || stage.cols == 0 ||
        stage.elements != stage.rows * stage.cols || stage.source_scale < 0 ||
        stage.source_scale > scale_)
      throw std::runtime_error("ViT oracle stage metadata mismatch");
    for (const std::string &kind : {std::string("fixed"),
                                    std::string("clear")}) {
      for (const std::string &listed_role : {std::string("alice"),
                                             std::string("bob")}) {
        std::string share_tag, share_stage, share_kind, share_role;
        ShareFile share;
        if (!(input >> share_tag >> share_stage >> share_kind >> share_role >>
              share.filename >> share.sha256) || share_tag != "share" ||
            share_stage != stage.name || share_kind != kind ||
            share_role != listed_role || !valid_sha256(share.sha256) ||
            std::filesystem::path(share.filename).filename() != share.filename)
          throw std::runtime_error("ViT oracle share manifest mismatch");
        if (listed_role == role) {
          if (kind == "fixed") stage.fixed = share;
          else stage.clear = share;
        }
      }
    }
    stages_.push_back(std::move(stage));
  }
  std::string end, trailing;
  if (!(input >> end) || end != "end" || (input >> trailing))
    throw std::runtime_error("ViT oracle manifest terminator mismatch");

  const std::string manifest_digest = sha256_file(manifest_path);
  char peer_digest[65] = {};
  if (party_ == ALICE) {
    io_.io->send_data(manifest_digest.data(), 64);
    io_.io->flush();
    io_.io->recv_data(peer_digest, 64);
  } else {
    io_.io->recv_data(peer_digest, 64);
    io_.io->send_data(manifest_digest.data(), 64);
    io_.io->flush();
  }
  if (manifest_digest != std::string(peer_digest, 64))
    throw std::runtime_error("ViT parties supplied different oracle manifests");

  if (party_ == ALICE) {
    require_sha256(sha256_file(model_path), model_sha, "model");
    if (sha256_text(runtime_source_sha256) != source_sha)
      throw std::runtime_error("ViT oracle source checkpoint hash mismatch");
  } else {
    require_sha256(sha256_file(input_path), input_sha, "input");
  }
  require_sha256(sha256_file(reference_path), reference_sha,
                 "reference implementation");
}

std::vector<std::uint64_t> VitValidationObserver::load_share(
    const Stage &stage, const ShareFile &share) const {
  const std::string path =
      (std::filesystem::path(manifest_dir_) / share.filename).string();
  require_sha256(sha256_file(path), share.sha256, "role share");
  std::ifstream input(path, std::ios::binary);
  char magic[sizeof(kShareMagic)];
  input.read(magic, sizeof(magic));
  std::uint64_t elements = 0;
  std::int32_t ell = 0, scale = 0;
  input.read(reinterpret_cast<char *>(&elements), sizeof(elements));
  input.read(reinterpret_cast<char *>(&ell), sizeof(ell));
  input.read(reinterpret_cast<char *>(&scale), sizeof(scale));
  if (!input || std::memcmp(magic, kShareMagic, sizeof(magic)) != 0 ||
      elements != stage.elements || ell != ell_ || scale != scale_)
    throw std::runtime_error("ViT oracle role share header mismatch");
  std::vector<std::uint64_t> result(stage.elements);
  input.read(reinterpret_cast<char *>(result.data()),
             static_cast<std::streamsize>(result.size() * sizeof(result[0])));
  char trailing = 0;
  if (!input || input.read(&trailing, 1))
    throw std::runtime_error("ViT oracle role share payload mismatch");
  return result;
}

void VitValidationObserver::observe(const std::string &name,
                                    const FixArray &secure,
                                    std::size_t physical_rows,
                                    std::size_t cols,
                                    std::size_t valid_rows) {
  if (next_stage_ >= stages_.size())
    throw std::runtime_error("unexpected extra ViT validation observation");
  const Stage &stage = stages_[next_stage_];
  if (stage.name != name || stage.rows != valid_rows || stage.cols != cols ||
      stage.elements != valid_rows * cols ||
      secure.size != static_cast<int>(physical_rows * cols) ||
      secure.ell != 37 || secure.s != stage.source_scale || !secure.signed_ ||
      valid_rows > physical_rows)
    throw std::runtime_error("ViT secure/oracle observation contract mismatch: " +
                             name);
  const std::uint64_t before_bytes = io_.get_comm();
  const std::uint64_t before_rounds = io_.get_rounds();
  const auto start = std::chrono::steady_clock::now();

  FixArray compact(party_, static_cast<int>(stage.elements), true, 37,
                   stage.source_scale);
  for (std::size_t row = 0; row < valid_rows; ++row)
    std::memcpy(compact.data + row * cols, secure.data + row * cols,
                cols * sizeof(std::uint64_t));
  FixArray lifted = fp_.fix->scale_up(compact, ell_, scale_);
  const std::vector<std::uint64_t> fixed_share = load_share(stage, stage.fixed);
  const std::vector<std::uint64_t> clear_share = load_share(stage, stage.clear);
  FixArray fixed(party_, static_cast<int>(stage.elements), true, ell_, scale_);
  FixArray clear(party_, static_cast<int>(stage.elements), true, ell_, scale_);
  std::copy(fixed_share.begin(), fixed_share.end(), fixed.data);
  std::copy(clear_share.begin(), clear_share.end(), clear.data);

  FixArray difference = concat(
      {fp_.fix->sub(lifted, fixed), fp_.fix->sub(lifted, clear)});
  BoolArray positive = fp_.fix->GT(difference, 0);
  FixArray negative = fp_.fix->mul(difference, std::uint64_t(-1), ell_);
  FixArray absolute = fp_.fix->if_else(positive, difference, negative);
  std::vector<FixArray> absolute_by_reference{
      absolute.subset(0, static_cast<int>(stage.elements)),
      absolute.subset(static_cast<int>(stage.elements), absolute.size)};
  FixArray maximum = fp_.fix->max(absolute_by_reference);
  FixArray opened_maximum = fp_.fix->output(PUBLIC, maximum);
  if (opened_maximum.data[0] >= kSafeMaxRaw ||
      opened_maximum.data[1] >= kSafeMaxRaw)
    throw std::runtime_error(
        "ViT observer arithmetic safety bound exceeded at " + name);

  FixArray absolute_unsigned = absolute;
  absolute_unsigned.signed_ = false;
  FixArray absolute_wide = fp_.fix->extend(absolute_unsigned, 64);
  FixArray sum_absolute = sum_halves(absolute_wide, stage.elements);
  FixArray opened_sum_absolute = fp_.fix->output(PUBLIC, sum_absolute);

  FixArray square = fp_.fix->mul(difference, difference, 63);
  // With max(|d|) < 2^31 raw, each exact square is below 2^62. Split each
  // square at ceil(log2(N)): the high-limb sum stays below 2^62 and the
  // low-limb sum below 2^36. Canonicalize the latter in secret so that the
  // opened high/low pair is a one-to-one representation of the total sum of
  // squares, not an extra statistic about the per-element remainders.
  square.signed_ = false;
  const int square_split = ceil_log2(stage.elements);
  FixArray square_high = fp_.fix->truncate_reduce(square, square_split);
  FixArray square_low = fp_.fix->reduce(square, square_split);
  FixArray sum_square_high = sum_halves(
      fp_.fix->extend(square_high, 64), stage.elements);
  FixArray sum_square_low = sum_halves(
      fp_.fix->extend(square_low, 64), stage.elements);
  // Normalize sum(low_i) into a carry plus a canonical low limb while both
  // values remain secret-shared. Metadata is set to raw-integer scale for the
  // existing truncation/addition primitives and the single PUBLIC output.
  sum_square_high.s = 0;
  sum_square_low.s = square_split;
  FixArray square_carry =
      fp_.fix->truncate_reduce(sum_square_low, square_split);
  square_carry = fp_.fix->extend(square_carry, 64);
  FixArray canonical_high = fp_.fix->add(sum_square_high, square_carry);
  FixArray canonical_low = fp_.fix->reduce(sum_square_low, square_split);
  canonical_low.s = 0;
  canonical_low = fp_.fix->extend(canonical_low, 64);
  FixArray opened_sum_square = fp_.fix->output(
      PUBLIC, concat({canonical_high, canonical_low}));

  validate_raw_aggregates(
      opened_maximum.data[0], opened_sum_absolute.data[0],
      opened_sum_square.data[0], opened_sum_square.data[2], square_split,
      stage.elements, name, "fixed");
  validate_raw_aggregates(
      opened_maximum.data[1], opened_sum_absolute.data[1],
      opened_sum_square.data[1], opened_sum_square.data[3], square_split,
      stage.elements, name, "clear");

  Metrics result;
  result.name = name;
  result.elements = stage.elements;
  const double count = static_cast<double>(stage.elements);
  result.fixed_max_abs = decoded(opened_maximum.data[0], scale_);
  result.clear_max_abs = decoded(opened_maximum.data[1], scale_);
  result.fixed_mean_abs = decoded(opened_sum_absolute.data[0], scale_) / count;
  result.clear_mean_abs = decoded(opened_sum_absolute.data[1], scale_) / count;
  result.rmse_square_scale = 2 * scale_;
  result.rmse_split_bits = square_split;
  result.fixed_rmse = static_cast<double>(std::sqrt(decoded_square_sum(
      opened_sum_square.data[0], opened_sum_square.data[2],
      square_split, stage.elements)));
  result.clear_rmse = static_cast<double>(std::sqrt(decoded_square_sum(
      opened_sum_square.data[1], opened_sum_square.data[3],
      square_split, stage.elements)));
  validate_metrics(result.fixed_max_abs, result.fixed_mean_abs,
                   result.fixed_rmse, name, "fixed");
  validate_metrics(result.clear_max_abs, result.clear_mean_abs,
                   result.clear_rmse, name, "clear");
  result.milliseconds = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  result.bytes = io_.get_comm() - before_bytes;
  result.logical_rounds = io_.get_rounds() - before_rounds;
  metrics_.push_back(result);
  ++next_stage_;
}

void VitValidationObserver::finalize() {
  if (next_stage_ != stages_.size())
    throw std::runtime_error("ViT validation observations are incomplete");
  if (party_ != ALICE) return;
  std::ofstream output(output_dir_ + "/secure_aggregate_errors.csv");
  if (!output)
    throw std::runtime_error("cannot write ViT aggregate error report");
  output << "stage,elements,observer_ell,observer_scale,"
            "clear_quantization_bound,fixed_secure_max_abs,"
            "fixed_secure_mean_abs,fixed_secure_rmse,clear_secure_max_abs,"
            "clear_secure_mean_abs,clear_secure_rmse,time_ms,bytes,"
            "logical_rounds,rmse_square_scale,intermediate_reconstruction,"
            "rmse_split_bits,rmse_accumulator,status\n";
  for (const Metrics &row : metrics_)
    output << csv(row.name) << ',' << row.elements << ',' << ell_ << ','
           << scale_ << ',' << std::setprecision(17)
           << clear_quantization_bound_ << ',' << row.fixed_max_abs << ','
           << row.fixed_mean_abs << ',' << row.fixed_rmse << ','
           << row.clear_max_abs << ',' << row.clear_mean_abs << ','
           << row.clear_rmse << ',' << row.milliseconds << ',' << row.bytes
           << ',' << row.logical_rounds << ',' << row.rmse_square_scale
           << ",false," << row.rmse_split_bits
           << ",exact_canonical_two_limb,PASS\n";
}

#endif
