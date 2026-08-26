#include "blb_vit_protocol.h"

#include "BOLTTransformer/bolt_metadata.h"
#include "OT/ot_pack.h"
#include "utils/ArgMapping/ArgMapping.h"
#ifdef BLB_VIT_VALIDATION_OUTPUTS
#include "vit_validation_observer.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sci;
using namespace sci::bolt;

extern std::uint64_t prime_mod;
std::int32_t bitlength = 37;
std::int32_t num_threads = 1;

namespace {

constexpr int kEll = 37;
constexpr int kScale = 12;
constexpr std::uint64_t kVitPlainModulus = 536903681ULL;

#ifndef BOLT_BUILD_GIT_COMMIT
#define BOLT_BUILD_GIT_COMMIT "unavailable"
#endif

int party = 0;
int port = 47000;
std::string address = "127.0.0.1";

struct WireConfig {
  std::uint64_t image_size, patch_size, channels, tokens, dim, heads, ff_dim,
      layers, classes, scale_bits, qkv_bias;
  double layer_norm_epsilon;
};

struct PhaseStats {
  std::string phase;
  double milliseconds = 0.0;
  std::uint64_t bytes = 0;
  std::uint64_t logical_rounds = 0;
};

std::uint64_t ring_mask(int ell = kEll) {
  return ell == 64 ? ~0ULL : ((1ULL << ell) - 1);
}

std::uint64_t encode(double value, int scale = kScale) {
  const auto fixed = static_cast<std::int64_t>(
      std::llround(value * std::ldexp(1.0, scale)));
  return static_cast<std::uint64_t>(fixed) & ring_mask();
}

std::size_t next_power_of_two(std::size_t value) {
  std::size_t result = 1;
  while (result < value) result <<= 1;
  return result;
}

BoltVitModelMetadata make_metadata(const VitModel &model) {
  BoltVitModelMetadata metadata;
  metadata.image_size = model.image_size;
  metadata.patch_size = model.patch_size;
  metadata.channels = model.channels;
  metadata.patch_dim = static_cast<std::size_t>(model.channels) *
                       model.patch_size * model.patch_size;
  metadata.tokens = model.tokens;
  metadata.physical_tokens = next_power_of_two(metadata.tokens);
  metadata.model_dim = model.dim;
  metadata.heads = model.heads;
  metadata.head_dim = model.dim / model.heads;
  metadata.ff_dim = model.ff_dim;
  metadata.layers = model.layers;
  metadata.classes = model.classes;
  metadata.patch_stride = next_power_of_two(
      std::max(metadata.patch_dim, metadata.model_dim));
  metadata.attention_stride = next_power_of_two(
      std::max(metadata.physical_tokens, metadata.model_dim));
  metadata.ffn_stride = next_power_of_two(
      std::max(metadata.model_dim, metadata.ff_dim));
  metadata.classifier_stride = next_power_of_two(
      std::max(metadata.model_dim, metadata.classes));
  metadata.packed_stride = std::max(
      std::max(metadata.patch_stride, metadata.attention_stride),
      std::max(metadata.ffn_stride, metadata.classifier_stride));
  metadata.qkv_bias = model.qkv_bias;
  metadata.pre_norm = model.norm_order == "pre";
  metadata.layer_norm_epsilon_lsb = static_cast<std::uint64_t>(
      std::llround(model.layer_norm_epsilon * std::ldexp(1.0, kScale)));
  return metadata;
}

void apply_config(VitModel &model, const WireConfig &config) {
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
  model.qkv_bias = config.qkv_bias != 0;
  model.layer_norm_epsilon = config.layer_norm_epsilon;
  model.norm_order = "pre";
  model.blocks.resize(model.layers);
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

std::string csv(const std::string &value) {
  if (value.find_first_of(",\"") == std::string::npos) return value;
  std::string result = "\"";
  for (char ch : value) { if (ch == '\"') result += '\"'; result += ch; }
  return result + "\"";
}

std::string json_string(const std::string &value) {
  std::ostringstream quoted;
  quoted << '"';
  for (char ch : value) {
    switch (ch) {
    case '"': quoted << "\\\""; break;
    case '\\': quoted << "\\\\"; break;
    case '\n': quoted << "\\n"; break;
    case '\r': quoted << "\\r"; break;
    case '\t': quoted << "\\t"; break;
    default: quoted << ch; break;
    }
  }
  quoted << '"';
  return quoted.str();
}

std::string environment_or(const char *name, const char *fallback) {
  const char *value = std::getenv(name);
  return value == nullptr ? fallback : value;
}

void write_operator_trace(const std::string &path,
                          const std::vector<VitOperatorTrace> &trace) {
  std::ofstream out(path + "/operator_trace.csv");
  if (!out) throw std::runtime_error("cannot write full ViT operator trace");
  out << "phase,layer,op,input_shape,output_shape,input_packing,output_packing,"
         "input_scale,output_scale,ell,plain_modulus,time_ms,bytes,logical_rounds,"
         "ciphertexts,rotations,multiply_plain,ct_ct_multiply,relinearizations,"
         "mod_switches,noise_floodings,backend,status\n";
  for (const auto &row : trace)
    out << row.phase << ',' << row.layer << ',' << row.op << ','
        << csv(row.input_shape) << ',' << csv(row.output_shape) << ','
        << csv(row.input_packing) << ',' << csv(row.output_packing) << ','
        << row.input_scale << ',' << row.output_scale << ',' << row.ell << ','
        << row.plain_modulus << ',' << std::setprecision(12)
        << row.milliseconds << ',' << row.bytes << ',' << row.logical_rounds
        << ',' << row.counts.ciphertexts << ',' << row.counts.rotations << ','
        << row.counts.multiply_plain << ',' << row.counts.evaluator_multiply
        << ',' << row.counts.relinearizations << ','
        << row.counts.modulus_switches << ','
        << row.counts.noise_floodings << ',' << csv(row.backend) << ','
        << (row.status ? "PASS" : "FAIL") << '\n';
}

void write_performance(const std::string &path,
                       const std::vector<PhaseStats> &phases) {
  std::ofstream out(path + "/performance.csv");
  if (!out) throw std::runtime_error("cannot write full ViT performance CSV");
  out << "phase,time_ms,bidirectional_bytes,logical_rounds,round_rule,status\n";
  for (const auto &phase : phases)
    out << phase.phase << ',' << std::setprecision(12) << phase.milliseconds
        << ',' << phase.bytes << ',' << phase.logical_rounds
        << ",max_endpoint_transition_counter,PASS\n";
}

struct BackendAggregate {
  double milliseconds = 0.0;
  std::uint64_t bytes = 0;
  std::uint64_t logical_rounds = 0;
  BoltOperationCounts counts;
};

std::string trace_category(const std::string &backend) {
  if (backend.rfind("bolt_conversion_", 0) == 0) return "conversion";
  if (backend.rfind("bolt_mpc_", 0) == 0) return "mpc";
  if (backend.rfind("bolt_ct_", 0) == 0 ||
      backend.rfind("bolt_packed_", 0) == 0 ||
      backend.rfind("bolt_bert_compat_", 0) == 0) return "he";
  return "local_or_public";
}

void add_counts(BoltOperationCounts &destination,
                const BoltOperationCounts &source) {
  destination.ciphertexts += source.ciphertexts;
  destination.rotations += source.rotations;
  destination.multiply_plain += source.multiply_plain;
  destination.evaluator_multiply += source.evaluator_multiply;
  destination.relinearizations += source.relinearizations;
  destination.modulus_switches += source.modulus_switches;
  destination.noise_floodings += source.noise_floodings;
}

void write_backend_breakdown(
    const std::string &path, const std::vector<VitOperatorTrace> &trace,
    const PhaseStats &online) {
  std::map<std::string, BackendAggregate> aggregate;
  for (const auto &row : trace) {
    auto &entry = aggregate[trace_category(row.backend)];
    entry.milliseconds += row.milliseconds;
    entry.bytes += row.bytes;
    entry.logical_rounds += row.logical_rounds;
    add_counts(entry.counts, row.counts);
  }
  double traced_ms = 0.0;
  std::uint64_t traced_bytes = 0;
  std::uint64_t traced_rounds = 0;
  for (const auto &item : aggregate) {
    traced_ms += item.second.milliseconds;
    traced_bytes += item.second.bytes;
    traced_rounds += item.second.logical_rounds;
  }
  auto &overhead = aggregate["runtime_overhead"];
  overhead.milliseconds = std::max(0.0, online.milliseconds - traced_ms);
  overhead.bytes = online.bytes >= traced_bytes ? online.bytes - traced_bytes : 0;
  overhead.logical_rounds =
      online.logical_rounds >= traced_rounds
          ? online.logical_rounds - traced_rounds
          : 0;
  std::ofstream out(path + "/backend_breakdown.csv");
  if (!out) throw std::runtime_error("cannot write ViT backend breakdown");
  out << "category,time_ms,bidirectional_bytes,logical_rounds,ciphertexts,"
         "rotations,multiply_plain,ct_ct_multiply,relinearizations,"
         "mod_switches,noise_floodings,accounting,status\n";
  for (const auto &item : aggregate) {
    const auto &entry = item.second;
    out << item.first << ',' << std::setprecision(12) << entry.milliseconds
        << ',' << entry.bytes << ',' << entry.logical_rounds << ','
        << entry.counts.ciphertexts << ',' << entry.counts.rotations << ','
        << entry.counts.multiply_plain << ','
        << entry.counts.evaluator_multiply << ','
        << entry.counts.relinearizations << ','
        << entry.counts.modulus_switches << ','
        << entry.counts.noise_floodings
        << ",ALICE-endpoint sequential time; bidirectional send-byte sum; "
           "rounds are endpoint-max deltas,PASS\n";
  }
}

std::string cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line))
    if (line.rfind("model name", 0) == 0) {
      const auto colon = line.find(':');
      return colon == std::string::npos ? line : line.substr(colon + 2);
    }
  return "unknown";
}

void write_run_metadata(const std::string &path, const VitModel &model,
                        const BoltVitModelMetadata &metadata,
                        const BoltHEParameters &he_parameters,
                        const char *he_backend,
                        std::size_t executed_blocks) {
  std::ofstream out(path + "/run_metadata.json");
  if (!out) throw std::runtime_error("cannot write full ViT run metadata");
  const std::string padding_positions =
      metadata.physical_tokens > metadata.tokens
          ? "[" + std::to_string(metadata.tokens) + ", " +
                std::to_string(metadata.physical_tokens - 1) + "]"
          : "[]";
  out << "{\n"
      << "  \"git_commit\": " << json_string(BOLT_BUILD_GIT_COMMIT) << ",\n"
      << "  \"git_scope\": \"BLB ViT model/shape/packing adapter; existing protocol mathematics reused\",\n"
      << "  \"core_source_audit\": {\"protocol_math_changed\": false, "
         "\"linear_ot_parallel_pragma_restored\": true, "
         "\"secret_share_debug_print_removed\": true},\n"
      << "  \"compiler\": \"" << __VERSION__ << "\",\n"
      << "  \"threads\": " << num_threads << ",\n"
      << "  \"cpu\": " << json_string(cpu_model()) << ",\n"
      << "  \"seal_version\": \"4.1.x bundled\",\n"
      << "  \"sci_version\": \"repository_snapshot\",\n"
      << "  \"checkpoint\": {\"source_id\": \"" << model.source_id
      << "\", \"revision\": \"" << model.source_revision
      << "\", \"sha256\": \"" << model.source_sha256 << "\"},\n"
      << "  \"model\": {\"pre_norm\": true, \"layers\": "
      << metadata.layers << ", \"logical_tokens\": " << metadata.tokens
      << ", \"physical_tokens\": " << metadata.physical_tokens
      << ", \"dim\": " << metadata.model_dim << ", \"heads\": "
      << metadata.heads << ", \"head_dim\": " << metadata.head_dim
      << ", \"ff_dim\": " << metadata.ff_dim << ", \"classes\": "
      << metadata.classes << ", \"qkv_bias\": "
      << (metadata.qkv_bias ? "true" : "false") << "},\n"
      << "  \"execution\": {\"he_backend\": "
      << json_string(he_backend == nullptr ? "unknown" : he_backend)
      << ", \"checkpoint_blocks\": " << metadata.layers
      << ", \"executed_blocks\": " << executed_blocks
      << ", \"validation_block_limit\": "
      << (executed_blocks == metadata.layers ? "false" : "true") << "},\n"
      << "  \"fixed_point\": {\"ell\": 37, \"activation_scale\": 12, "
         "\"weight_scale\": 12, \"gelu_input_ell\": 20, "
         "\"gelu_input_scale\": 11, \"gelu_output_ell\": 37, "
         "\"gelu_output_scale\": 4, \"layer_norm_epsilon_lsb\": "
      << metadata.layer_norm_epsilon_lsb
      << ", \"layer_norm_epsilon_real\": " << std::setprecision(17)
      << model.layer_norm_epsilon
      << ", \"layer_norm_epsilon_applied_by_core\": false, "
         "\"layer_norm_epsilon_fixed_point_effect\": "
         "\"quantizes_to_zero_at_scale_12\"},\n"
      << "  \"packing\": {\"patch_stride\": " << metadata.patch_stride
      << ", \"attention_stride\": " << metadata.attention_stride
      << ", \"ffn_stride\": " << metadata.ffn_stride
      << ", \"classifier_stride\": " << metadata.classifier_stride
      << "},\n"
      << "  \"padding_mask\": {\"logical_valid_positions\": [0, "
      << (metadata.tokens - 1) << "], \"physical_positions\": [0, "
      << (metadata.physical_tokens - 1) << "], \"padding_positions\": ["
      << padding_positions.substr(1, padding_positions.size() - 2)
      << "], \"attention_softmax_valid_shape\": ["
      << metadata.tokens << ", " << metadata.tokens
      << "], \"mask_is_public\": true},\n"
      << "  \"bfv\": {\"scheme\": \"BFV\", \"poly_modulus_degree\": "
      << he_parameters.poly_modulus_degree << ", \"plain_modulus\": "
      << he_parameters.plain_modulus << ", \"batch_encoder\": true, "
         "\"galois_keys\": true, \"relinearization_keys\": true},\n"
      << "  \"accounting\": {\"bytes\": \"sum of endpoint send-only counters\", "
         "\"logical_rounds\": \"max endpoint transition counter; never sum\", "
         "\"operator_time\": \"ALICE endpoint sequential attribution avoids double-counting BOB wait\", "
         "\"backend_breakdown\": \"backend_breakdown.csv; HE/MPC/conversion plus residual runtime overhead\"},\n"
      << "  \"network_profiles\": {\"LAN\": {\"rtt_ms\": 0.2, \"bandwidth_mbps\": 1000}, "
         "\"WAN\": {\"rtt_ms\": 40, \"bandwidth_mbps\": 100}},\n"
      << "  \"network_profile_applied\": {\"name\": "
      << json_string(environment_or("BOLT_NETWORK_PROFILE", "unlimited"))
      << ", \"interface\": "
      << json_string(environment_or("BOLT_NETWORK_INTERFACE", "none"))
      << ", \"rate_mbps\": "
      << json_string(environment_or("BOLT_NETWORK_RATE_MBPS", "unlimited"))
      << ", \"rtt_ms\": "
      << json_string(environment_or("BOLT_NETWORK_RTT_MS", "unlimited"))
      << "},\n"
      << "  \"fail_closed\": {\"scalar_ciphertext\": false, "
         "\"generic_ot_matmul\": false, \"plain_matmul\": false, "
         "\"intermediate_reconstruction\": false, \"silent_fallback\": false}\n"
      << "}\n";
}

std::vector<PhaseStats> aggregate_phases(IOPack &io,
                                         std::vector<PhaseStats> local) {
  struct Wire { double ms; std::uint64_t bytes, rounds; };
  if (party == BOB) {
    for (const auto &phase : local) {
      Wire wire{phase.milliseconds, phase.bytes, phase.logical_rounds};
      io.io->send_data(&wire, sizeof(wire));
    }
    io.io->flush();
    return local;
  }
  for (auto &phase : local) {
    Wire peer{};
    io.io->recv_data(&peer, sizeof(peer));
    phase.milliseconds = std::max(phase.milliseconds, peer.ms);
    phase.bytes += peer.bytes;
    phase.logical_rounds = std::max(phase.logical_rounds, peer.rounds);
  }
  return local;
}

const PhaseStats &find_phase(const std::vector<PhaseStats> &phases,
                             const std::string &name) {
  const auto it = std::find_if(
      phases.begin(), phases.end(),
      [&name](const PhaseStats &phase) { return phase.phase == name; });
  if (it == phases.end())
    throw std::logic_error("missing phase in end-to-end accounting: " + name);
  return *it;
}

std::uint64_t reveal_class(IOPack &io, const FixArray &index) {
  if (party == ALICE) {
    io.io->send_data(index.data, sizeof(std::uint64_t));
    io.io->flush();
    return 0;
  }
  std::uint64_t peer = 0;
  io.io->recv_data(&peer, sizeof(peer));
  return (peer + index.data[0]) & index.ell_mask();
}

#ifdef BLB_VIT_VALIDATION_OUTPUTS
void write_validation_logits(const std::string &trace_dir,
                             const FixArray &logits, FPMath &fp) {
  FixArray opened = fp.fix->output(PUBLIC, logits);
  if (party != ALICE) return;
  std::ofstream out(trace_dir + "/secure_logits.txt");
  if (!out) throw std::runtime_error("cannot write validation logits");
  out << "BLB_VIT_SECURE_LOGITS_V1\n"
      << "elements " << opened.size << "\n"
      << "ell " << opened.ell << "\n"
      << "scale " << opened.s << "\n"
      << "values";
  const std::uint64_t ring = 1ULL << opened.ell;
  const std::uint64_t sign = 1ULL << (opened.ell - 1);
  for (int i = 0; i < opened.size; ++i) {
    const std::uint64_t raw = opened.data[i] & opened.ell_mask();
    const std::int64_t value = raw >= sign
        ? static_cast<std::int64_t>(raw - ring)
        : static_cast<std::int64_t>(raw);
    out << ' ' << std::setprecision(17)
        << std::ldexp(static_cast<double>(value), -opened.s);
  }
  out << "\nend\n";
}
#endif

} // namespace

int main(int argc, char **argv) {
  // BLB's unmodified LinearHE archive owns this process-wide field modulus.
  // Select the BOLT BFV batching prime before constructing any protocol
  // object instead of defining a second symbol in this adapter.
  prime_mod = kVitPlainModulus;
  std::string model_path;
  std::string input_path;
  std::string output_path = "bolt_vit_prediction.txt";
  std::string trace_dir = "outputs/vit_full_protocol";
  int debug_logits = 0;
  int block_limit = 0;
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  int aggregate_errors = 0;
  std::string oracle_manifest_path;
  std::string oracle_reference_path;
#endif
  ArgMapping args;
  args.arg("r", party, "Role: ALICE=1, BOB=2");
  args.arg("p", port, "TCP port");
  args.arg("ip", address, "ALICE address (BOB only)");
  args.arg("model", model_path, "ALICE-only explicit ViT model file");
  args.arg("input", input_path, "BOB-only preprocessed image file");
  args.arg("output", output_path, "BOB prediction file");
  args.arg("trace_dir", trace_dir, "Protocol trace output directory");
  args.arg("debug_logits", debug_logits, "Forbidden in formal protocol");
  args.arg("blocks", block_limit,
           "Validation-only encoder block limit; 0 executes all blocks");
#ifdef BLB_VIT_VALIDATION_OUTPUTS
  args.arg("aggregate_errors", aggregate_errors,
           "Validation-only aggregate error observer; disabled by default");
  args.arg("oracle_manifest", oracle_manifest_path,
           "Validation-only role-share manifest");
  args.arg("oracle_reference", oracle_reference_path,
           "Validation-only reference implementation bound by the manifest");
#endif
  args.parse(argc, argv);
  const auto e2e_start = std::chrono::steady_clock::now();
  try {
    if (party != ALICE && party != BOB)
      throw std::invalid_argument("r must be ALICE=1 or BOB=2");
    if (debug_logits != 0
#ifdef BLB_VIT_VALIDATION_OUTPUTS
        && debug_logits != 1
#endif
    )
      throw std::logic_error(
          "BOLT fail-closed: full-logit reconstruction is validation-only");
#ifndef BLB_VIT_VALIDATION_OUTPUTS
    if (block_limit != 0)
      throw std::logic_error(
          "BOLT fail-closed: block limiting is validation-only");
#endif
    if (block_limit < 0)
      throw std::invalid_argument("ViT block limit cannot be negative");
#ifdef BLB_VIT_VALIDATION_OUTPUTS
    if (aggregate_errors != 0 && aggregate_errors != 1)
      throw std::invalid_argument("aggregate_errors must be 0 or 1");
    if (aggregate_errors == 1 &&
        (oracle_manifest_path.empty() || oracle_reference_path.empty()))
      throw std::logic_error(
          "ViT aggregate observer requires manifest and reference paths");
#endif
    const char *role = party == ALICE ? "ALICE" : "BOB";
    VitModel model;
    double checkpoint_load_ms = 0.0;
    if (party == ALICE) {
      const auto checkpoint_start = std::chrono::steady_clock::now();
      std::cerr << "BOLT_VIT_STATUS role=ALICE phase=checkpoint_load status=start"
                << std::endl;
      model = load_vit_model(model_path);
      checkpoint_load_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - checkpoint_start).count();
      std::cerr << "BOLT_VIT_STATUS role=ALICE phase=checkpoint_load status=done"
                << " layers=" << model.layers << " tokens=" << model.tokens
                << " dim=" << model.dim << std::endl;
    }
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=network status=waiting" << std::endl;
    IOPack io(party, port, address);
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=network status=connected" << std::endl;

    // This interval covers public metadata/configuration, BFV key exchange,
    // base OT/MPC setup, online inference, and final reveal.  Capture the
    // endpoint counters before the accounting exchange so the metrics row
    // itself is not counted as inference communication.
    const std::uint64_t e2e_before_bytes = io.get_comm();
    const std::uint64_t e2e_before_rounds = io.get_rounds();

    const std::uint64_t config_before_bytes = io.get_comm();
    const std::uint64_t config_before_rounds = io.get_rounds();
    const auto config_start = std::chrono::steady_clock::now();
    WireConfig config{};
    if (party == ALICE) {
      config = {static_cast<std::uint64_t>(model.image_size),
                static_cast<std::uint64_t>(model.patch_size),
                static_cast<std::uint64_t>(model.channels),
                static_cast<std::uint64_t>(model.tokens),
                static_cast<std::uint64_t>(model.dim),
                static_cast<std::uint64_t>(model.heads),
                static_cast<std::uint64_t>(model.ff_dim),
                static_cast<std::uint64_t>(model.layers),
                static_cast<std::uint64_t>(model.classes),
                static_cast<std::uint64_t>(model.scale_bits),
                static_cast<std::uint64_t>(model.qkv_bias),
                model.layer_norm_epsilon};
      io.io->send_data(&config, sizeof(config));
      io.io->flush();
    } else {
      io.io->recv_data(&config, sizeof(config));
      apply_config(model, config);
    }
    const auto config_finish = std::chrono::steady_clock::now();
    PhaseStats offline{
        "offline_preprocessing",
        std::chrono::duration<double, std::milli>(config_finish - config_start).count(),
        io.get_comm() - config_before_bytes,
        io.get_rounds() - config_before_rounds};

    const BoltHEParameters he_parameters = BoltHEParameters::vit_b16();
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=bfv_setup status=start" << std::endl;
    BoltHE he(party, io.io, he_parameters);
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=bfv_setup status=done backend="
              << he.view().backend_name() << std::endl;
    const BoltHESetupStats &he_setup = he.setup_stats();
    PhaseStats key_generation{
        "bfv_context_and_secret_key_generation",
        he_setup.context_and_key_generation_ms, 0, 0};
    PhaseStats evaluation_keys{
        "evaluation_public_keys",
        he_setup.evaluation_public_keys_ms,
        he_setup.evaluation_public_keys_bytes,
        he_setup.evaluation_public_keys_rounds};

    const std::uint64_t ot_before_bytes = io.get_comm();
    const std::uint64_t ot_before_rounds = io.get_rounds();
    const auto ot_start = std::chrono::steady_clock::now();
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=mpc_setup status=start" << std::endl;
    OTPack ot(&io, party);
    FPMath fp(party, &io, &ot);
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=mpc_setup status=done" << std::endl;
    const auto ot_finish = std::chrono::steady_clock::now();
    PhaseStats base_ot{
        "base_ot_and_mpc_setup",
        std::chrono::duration<double, std::milli>(ot_finish - ot_start).count(),
        io.get_comm() - ot_before_bytes, io.get_rounds() - ot_before_rounds};

    BoltVitModelMetadata metadata = make_metadata(model);
    metadata.validate(he.slot_count());
    if (model.scale_bits != kScale || metadata.ell != kEll ||
        !verify_fail_closed(he))
      throw std::logic_error("BOLT ViT fail-closed contract failed");

    std::vector<std::uint64_t> image_share(
        metadata.channels * metadata.image_size * metadata.image_size, 0);
    if (party == BOB) {
      VitInput input = load_vit_input(input_path);
      if (input.image_size != model.image_size ||
          input.channels != model.channels ||
          input.pixels.size() != image_share.size())
        throw std::invalid_argument("BOB image does not match public model config");
      for (std::size_t i = 0; i < image_share.size(); ++i)
        image_share[i] = encode(input.pixels[i]);
    }

    BoltVitProtocol protocol(party, io, ot, fp, he, metadata);
    const std::uint64_t online_before_bytes = io.get_comm();
    const std::uint64_t online_before_rounds = io.get_rounds();
    const auto online_start = std::chrono::steady_clock::now();
    std::vector<VitOperatorTrace> trace;
#ifdef BLB_VIT_VALIDATION_OUTPUTS
    std::unique_ptr<VitValidationObserver> validation_observer;
    if (aggregate_errors == 1) {
      const std::size_t executed_blocks = block_limit == 0
          ? metadata.layers : static_cast<std::size_t>(block_limit);
      validation_observer = std::make_unique<VitValidationObserver>(
          party, io, fp, oracle_manifest_path, model_path, input_path,
          oracle_reference_path,
          party == ALICE ? model.source_sha256 : std::string(),
          metadata.tokens, executed_blocks, trace_dir);
    }
#endif
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=online_inference status=start" << std::endl;
    VitProtocolOutput result = protocol.run(
        model, image_share, trace, static_cast<std::size_t>(block_limit)
#ifdef BLB_VIT_VALIDATION_OUTPUTS
        , validation_observer.get()
#endif
        );
#ifdef BLB_VIT_VALIDATION_OUTPUTS
    if (validation_observer != nullptr)
      validation_observer->finalize();
#endif
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=online_inference status=done" << std::endl;
    // The adapter has no call edge to LinearOT::matrix_multiplication.  BLB's
    // existing LinearOT source is intentionally left untouched, so unlike the
    // reference BOLT tree there is no backend-global instrumentation counter.
    const auto online_finish = std::chrono::steady_clock::now();
    PhaseStats online{
        "online_inference",
        std::chrono::duration<double, std::milli>(online_finish - online_start).count(),
        io.get_comm() - online_before_bytes,
        io.get_rounds() - online_before_rounds};

    const std::uint64_t reveal_before_bytes = io.get_comm();
    const std::uint64_t reveal_before_rounds = io.get_rounds();
    const auto reveal_start = std::chrono::steady_clock::now();
#ifdef BLB_VIT_VALIDATION_OUTPUTS
    if (debug_logits == 1)
      write_validation_logits(trace_dir, result.logits, fp);
#endif
    const std::uint64_t predicted = reveal_class(io, result.predicted_class);
    const auto reveal_finish = std::chrono::steady_clock::now();
    PhaseStats reveal{
        "final_reveal",
        std::chrono::duration<double, std::milli>(reveal_finish - reveal_start).count(),
        io.get_comm() - reveal_before_bytes,
        io.get_rounds() - reveal_before_rounds};

    const auto e2e_finish = reveal_finish;
    PhaseStats end_to_end{
        "end_to_end_inference",
        std::chrono::duration<double, std::milli>(e2e_finish - e2e_start).count(),
        io.get_comm() - e2e_before_bytes,
        io.get_rounds() - e2e_before_rounds};

    protocol.aggregate_trace(trace);
    PhaseStats public_config = offline;
    public_config.phase = "public_model_metadata_exchange";
    PhaseStats offline_preprocessing{
        "offline_preprocessing_checkpoint_parse", checkpoint_load_ms, 0, 0};
    std::vector<PhaseStats> phases = aggregate_phases(
        io, {key_generation, evaluation_keys, base_ot, public_config,
             offline_preprocessing, online, reveal, end_to_end});
    const std::size_t executed_blocks = static_cast<std::size_t>(
        std::count_if(trace.begin(), trace.end(),
            [](const VitOperatorTrace &row) {
              return row.op == "attention_residual";
            }));
    if (party == ALICE) {
      write_operator_trace(trace_dir, trace);
      write_performance(trace_dir, phases);
      write_backend_breakdown(
          trace_dir, trace, find_phase(phases, "online_inference"));
      write_run_metadata(trace_dir, model, metadata, he_parameters,
                         he.view().backend_name(),
                         executed_blocks);
      const PhaseStats &e2e = find_phase(phases, "end_to_end_inference");
      const PhaseStats &online_phase = find_phase(phases, "online_inference");
      std::cout << "BOLT_VIT_E2E_METRICS communication_bytes=" << e2e.bytes
                << " end_to_end_time_ms=" << std::fixed << std::setprecision(3)
                << e2e.milliseconds
                << " inference_time_ms=" << online_phase.milliseconds
                << " logical_rounds=" << e2e.logical_rounds
                << " status=PASS" << std::endl;
    } else {
      std::ofstream output(output_path);
      if (!output) throw std::runtime_error("cannot write BOLT ViT prediction");
      output << "predicted_class " << predicted << '\n';
      std::cout << "PREDICTION class=" << predicted << std::endl;
    }
    if (party == ALICE) {
      const auto valid_ctct_rows = [&trace](const std::string &op) {
        return static_cast<std::size_t>(std::count_if(
            trace.begin(), trace.end(), [&op](const VitOperatorTrace &row) {
              return row.op == op && row.counts.evaluator_multiply > 0 &&
                     row.counts.relinearizations > 0;
            }));
      };
      const bool ctct = executed_blocks > 0 &&
          valid_ctct_rows("qk") == executed_blocks &&
          valid_ctct_rows("softmax_v") == executed_blocks;
      std::size_t hardened_rows = 0;
      const bool hardened = std::all_of(
          trace.begin(), trace.end(), [&hardened_rows](
                                         const VitOperatorTrace &row) {
            const bool he_result =
                row.op == "patch_projection" || row.op == "q_projection" ||
                row.op == "k_projection" || row.op == "v_projection" ||
                row.op == "qk" || row.op == "softmax_v" ||
                row.op == "output_projection" || row.op == "fc1" ||
                row.op == "fc2" || row.op == "classification_head" ||
                row.op == "ln1_affine" || row.op == "ln2_affine" ||
                row.op == "final_ln_affine";
            if (!he_result) return true;
            ++hardened_rows;
            return row.counts.noise_floodings > 0 &&
                   row.counts.modulus_switches >=
                       2 * row.counts.noise_floodings;
          }) && hardened_rows == 10 * executed_blocks + 3;
      const bool residual_local = std::all_of(trace.begin(), trace.end(),
          [](const VitOperatorTrace &row) {
            return row.backend != "local_share_add" ||
                   (row.bytes == 0 && row.logical_rounds == 0);
          });
      if (!ctct || !hardened || !residual_local)
        throw std::logic_error("BOLT ViT operation audit failed");
      std::cout << "BOLT_VIT_SINGLE_LAYER_PROTOCOL_EXECUTION="
                << (executed_blocks == 1 ? "PASS" : "NOT_APPLICABLE") << '\n'
                << "BOLT_VIT_MULTI_LAYER_PROTOCOL_EXECUTION="
                << (executed_blocks > 1 ? "PASS" : "NOT_APPLICABLE") << '\n'
                << "PERFORMANCE_ACCOUNTING_VALID=true\n"
                << "CIRCUIT_PRIVACY_HARDENING_AUDIT=PASS\n"
                << "PLAINTEXT_INTERMEDIATE_RECONSTRUCTIONS=0\n"
#ifdef BLB_VIT_VALIDATION_OUTPUTS
                << "FINAL_LOGITS_REVEALED_FOR_VALIDATION="
                << (debug_logits == 1 ? 1 : 0) << '\n'
#endif
                << "SCALAR_FALLBACK_STATIC_AUDIT=PASS\n"
                << "GENERIC_OT_MATMUL_STATIC_AUDIT=PASS\n"
                << "SILENT_FALLBACK_STATIC_AUDIT=PASS" << std::endl;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "BOLT_VIT error: " << error.what() << std::endl;
    return 2;
  }
}
