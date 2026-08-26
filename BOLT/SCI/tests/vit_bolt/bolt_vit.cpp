#include "bolt_vit_protocol.h"

#include "BOLTTransformer/bolt_metadata.h"
#include "LinearOT/linear-ot.h"
#include "OT/ot_pack.h"
#include "utils/ArgMapping/ArgMapping.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sci;
using namespace sci::bolt;

std::uint64_t prime_mod = 536903681ULL;
std::int32_t bitlength = 38;
std::int32_t num_threads = BoltVitMPC::kThreads;

namespace {

constexpr int kEll = 38;
constexpr int kScale = 12;

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

int requested_layer_limit() {
  const char *value = std::getenv("BOLT_VIT_MAX_LAYERS");
  if (value == nullptr || *value == '\0') return 0;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 || parsed > 12)
    throw std::invalid_argument(
        "BOLT_VIT_MAX_LAYERS must be an integer in [1,12]");
  return static_cast<int>(parsed);
}

void write_operator_trace(const std::string &path,
                          const std::vector<VitOperatorTrace> &trace) {
  std::ofstream out(path + "/operator_trace.csv");
  if (!out) throw std::runtime_error("cannot write full ViT operator trace");
  out << "phase,layer,op,input_shape,output_shape,input_packing,output_packing,"
         "input_scale,output_scale,ell,plain_modulus,time_ms,bytes,logical_rounds,"
         "ciphertexts,rotations,multiply_plain,ct_ct_multiply,relinearizations,"
         "mod_switches,backend,status\n";
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
        << row.counts.modulus_switches << ',' << csv(row.backend) << ','
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
      backend.rfind("bolt_packed_", 0) == 0) return "he";
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
    const std::string &path, const std::vector<VitOperatorTrace> &trace) {
  std::map<std::string, BackendAggregate> aggregate;
  for (const auto &row : trace) {
    auto &entry = aggregate[trace_category(row.backend)];
    entry.milliseconds += row.milliseconds;
    entry.bytes += row.bytes;
    entry.logical_rounds += row.logical_rounds;
    add_counts(entry.counts, row.counts);
  }
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
        << ",sequential operator-row sum; rounds are per-endpoint deltas,PASS\n";
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
                        const BoltHEParameters &he_tiny_parameters,
                        const BoltHEParameters &he_layer_norm_parameters) {
  std::ofstream out(path + "/run_metadata.json");
  if (!out) throw std::runtime_error("cannot write full ViT run metadata");
  const std::string padding_positions =
      metadata.physical_tokens > metadata.tokens
          ? "[" + std::to_string(metadata.tokens) + ", " +
                std::to_string(metadata.physical_tokens - 1) + "]"
          : "[]";
  out << "{\n"
      << "  \"git_commit\": " << json_string(BOLT_BUILD_GIT_COMMIT) << ",\n"
      << "  \"git_scope\": \"nearest parent worktree; BOLT is an untracked directory\",\n"
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
      << "  \"fixed_point\": {\"ell\": 38, \"activation_scale\": 12, "
         "\"weight_scale\": 12, \"gelu_input_ell\": 20, "
         "\"gelu_input_scale\": 11, \"gelu_output_ell\": 38, "
         "\"gelu_output_scale\": 4, \"layer_norm_epsilon_lsb\": "
      << metadata.layer_norm_epsilon_lsb
      << ", \"layer_norm_epsilon_real\": " << std::setprecision(17)
      << model.layer_norm_epsilon
      << ", \"layer_norm_epsilon_applied_by_core\": true,"
         " \"layer_norm_epsilon_quantization_scale\": 12},\n"
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
      << "  \"bfv\": {\"scheme\": \"BFV\", \"contexts\": {"
         "\"main\": {\"poly_modulus_degree\": "
      << he_parameters.poly_modulus_degree << ", \"plain_modulus\": "
      << he_parameters.plain_modulus << ", \"coeff_modulus_bits\": [54,54,55,55]}, "
         "\"tiny_linear\": {\"poly_modulus_degree\": "
      << he_tiny_parameters.poly_modulus_degree
      << ", \"plain_modulus\": " << he_tiny_parameters.plain_modulus
      << ", \"coeff_modulus_bits\": [60,60,60]}, "
         "\"layer_norm\": {\"poly_modulus_degree\": "
      << he_layer_norm_parameters.poly_modulus_degree
      << ", \"plain_modulus\": "
      << he_layer_norm_parameters.plain_modulus
      << ", \"coeff_modulus_bits\": [54,54,55,55]}}, "
         "\"security_level\": \"tc128\", \"batch_encoder\": true, "
         "\"galois_keys\": true, \"relinearization_keys\": true},\n"
      << "  \"accounting\": {\"bytes\": \"sum of endpoint send-only counters\", "
         "\"logical_rounds\": \"max endpoint transition counter; never sum\", "
         "\"backend_breakdown\": \"backend_breakdown.csv; HE/MPC/conversion rows are separated by adapter trace\"},\n"
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

} // namespace

int main(int argc, char **argv) {
  std::string model_path;
  std::string input_path;
  std::string output_path = "bolt_vit_prediction.txt";
  std::string trace_dir = "outputs/vit_full_protocol";
  int debug_logits = 0;
  ArgMapping args;
  args.arg("r", party, "Role: ALICE=1, BOB=2");
  args.arg("p", port, "TCP port");
  args.arg("ip", address, "ALICE address (BOB only)");
  args.arg("model", model_path, "ALICE-only explicit ViT model file");
  args.arg("input", input_path, "BOB-only preprocessed image file");
  args.arg("output", output_path, "BOB prediction file");
  args.arg("trace_dir", trace_dir, "Protocol trace output directory");
  args.arg("debug_logits", debug_logits, "Forbidden in formal protocol");
  args.parse(argc, argv);
  try {
    if (party != ALICE && party != BOB)
      throw std::invalid_argument("r must be ALICE=1 or BOB=2");
    if (debug_logits != 0)
      throw std::logic_error(
          "BOLT fail-closed: full-logit reconstruction is validation-only");
    const char *role = party == ALICE ? "ALICE" : "BOB";
    VitModel model;
    if (party == ALICE) {
      std::cerr << "BOLT_VIT_STATUS role=ALICE phase=checkpoint_load status=start"
                << std::endl;
      model = load_vit_model(model_path);
      const int layer_limit = requested_layer_limit();
      if (layer_limit != 0 && layer_limit < model.layers) {
        model.layers = layer_limit;
        model.blocks.resize(static_cast<std::size_t>(layer_limit));
        std::cerr << "BOLT_VIT_STATUS role=ALICE phase=checkpoint_load"
                  << " layer_limit=" << layer_limit << std::endl;
      }
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
    const auto e2e_start = std::chrono::steady_clock::now();

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

    const BoltHEParameters he_parameters = BoltHEParameters::vit_main();
    const BoltHEParameters he_tiny_parameters =
        BoltHEParameters::vit_linear();
    const BoltHEParameters he_layer_norm_parameters =
        BoltHEParameters::bert_layer_norm();
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=bfv_setup status=start" << std::endl;
#ifdef BOLT_VIT_GPU_ENTRY
    BoltHE he(party, io.io, he_parameters, true);
#else
    BoltHE he(party, io.io, he_parameters, false);
#endif
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=bfv_setup status=done" << std::endl;
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=bfv_tiny_linear_setup status=start" << std::endl;
#ifdef BOLT_VIT_GPU_ENTRY
    BoltHE he_tiny(party, io.io, he_tiny_parameters, true);
#else
    BoltHE he_tiny(party, io.io, he_tiny_parameters, false);
#endif
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=bfv_tiny_linear_setup status=done" << std::endl;
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=bfv_layer_norm_setup status=start" << std::endl;
#ifdef BOLT_VIT_GPU_ENTRY
    BoltHE he_layer_norm(party, io.io, he_layer_norm_parameters, true);
#else
    BoltHE he_layer_norm(party, io.io, he_layer_norm_parameters, false);
#endif
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=bfv_layer_norm_setup status=done" << std::endl;
    const BoltHESetupStats &he_setup = he.setup_stats();
    const BoltHESetupStats &he_tiny_setup = he_tiny.setup_stats();
    const BoltHESetupStats &he_layer_norm_setup =
        he_layer_norm.setup_stats();
    PhaseStats key_generation{
        "bfv_context_and_secret_key_generation",
        he_setup.context_and_key_generation_ms +
            he_tiny_setup.context_and_key_generation_ms +
            he_layer_norm_setup.context_and_key_generation_ms,
        0, 0};
    PhaseStats evaluation_keys{
        "evaluation_public_keys",
        he_setup.evaluation_public_keys_ms +
            he_tiny_setup.evaluation_public_keys_ms +
            he_layer_norm_setup.evaluation_public_keys_ms,
        he_setup.evaluation_public_keys_bytes +
            he_tiny_setup.evaluation_public_keys_bytes +
            he_layer_norm_setup.evaluation_public_keys_bytes,
        he_setup.evaluation_public_keys_rounds +
            he_tiny_setup.evaluation_public_keys_rounds +
            he_layer_norm_setup.evaluation_public_keys_rounds};

    const std::uint64_t ot_before_bytes = io.get_comm();
    const std::uint64_t ot_before_rounds = io.get_rounds();
    const auto ot_start = std::chrono::steady_clock::now();
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=mpc_setup status=start" << std::endl;
    OTPack ot(&io, party);
    FPMath fp(party, &io, &ot);
    BoltVitMPC mpc(party, address, port + 1);
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=mpc_setup status=done" << std::endl;
    const auto ot_finish = std::chrono::steady_clock::now();
    PhaseStats base_ot{
        "base_ot_and_mpc_setup",
        std::chrono::duration<double, std::milli>(ot_finish - ot_start).count(),
        io.get_comm() - ot_before_bytes + mpc.communication(),
        io.get_rounds() - ot_before_rounds + mpc.rounds()};

    BoltVitModelMetadata metadata = make_metadata(model);
    metadata.validate(he.slot_count());
    metadata.validate(he_tiny.slot_count());
    if (model.scale_bits != kScale || metadata.ell != kEll ||
        !verify_fail_closed(he) || !verify_fail_closed(he_tiny) ||
        !verify_fail_closed(he_layer_norm))
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

    BoltVitProtocol protocol(party, io, fp, mpc, he, he_tiny, he_layer_norm,
                             metadata);
    reset_linear_ot_generic_matmul_calls();
    const std::uint64_t online_before_bytes =
        io.get_comm() + mpc.communication();
    const std::uint64_t online_before_rounds =
        io.get_rounds() + mpc.rounds();
    const auto online_start = std::chrono::steady_clock::now();
    std::vector<VitOperatorTrace> trace;
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=online_inference status=start" << std::endl;
    VitProtocolOutput result = protocol.run(model, image_share, trace);
    std::cerr << "BOLT_VIT_STATUS role=" << role
              << " phase=online_inference status=done" << std::endl;
    const std::uint64_t generic_ot_calls = linear_ot_generic_matmul_calls();
    if (generic_ot_calls != 0)
      throw std::logic_error(
          "BOLT fail-closed: generic OT matrix multiplication executed in formal ViT graph");
    const auto online_finish = std::chrono::steady_clock::now();
    PhaseStats online{
        "online_inference",
        std::chrono::duration<double, std::milli>(online_finish - online_start).count(),
        io.get_comm() + mpc.communication() - online_before_bytes,
        io.get_rounds() + mpc.rounds() - online_before_rounds};

    const std::uint64_t reveal_before_bytes =
        io.get_comm() + mpc.communication();
    const std::uint64_t reveal_before_rounds =
        io.get_rounds() + mpc.rounds();
    const auto reveal_start = std::chrono::steady_clock::now();
    const std::uint64_t predicted = reveal_class(io, result.predicted_class);
    const auto reveal_finish = std::chrono::steady_clock::now();
    PhaseStats reveal{
        "final_reveal",
        std::chrono::duration<double, std::milli>(reveal_finish - reveal_start).count(),
        io.get_comm() + mpc.communication() - reveal_before_bytes,
        io.get_rounds() + mpc.rounds() - reveal_before_rounds};

    const auto e2e_finish = reveal_finish;
    PhaseStats end_to_end{
        "end_to_end_inference",
        std::chrono::duration<double, std::milli>(e2e_finish - e2e_start).count(),
        io.get_comm() + mpc.communication() - e2e_before_bytes,
        io.get_rounds() + mpc.rounds() - e2e_before_rounds};

    protocol.aggregate_trace(trace);
    PhaseStats public_config = offline;
    public_config.phase = "public_model_metadata_exchange";
    PhaseStats offline_preprocessing{"offline_preprocessing", 0.0, 0, 0};
    std::vector<PhaseStats> phases = aggregate_phases(
        io, {key_generation, evaluation_keys, base_ot, public_config,
             offline_preprocessing, online, reveal, end_to_end});
    if (party == ALICE) {
      write_operator_trace(trace_dir, trace);
      write_performance(trace_dir, phases);
      write_backend_breakdown(trace_dir, trace);
      write_run_metadata(trace_dir, model, metadata, he_parameters,
                         he_tiny_parameters, he_layer_norm_parameters);
      const PhaseStats &e2e = find_phase(phases, "end_to_end_inference");
      std::cout << "BOLT_VIT_E2E_METRICS communication_bytes=" << e2e.bytes
                << " end_to_end_time_ms=" << std::fixed << std::setprecision(3)
                << e2e.milliseconds << " inference_time_ms=" << e2e.milliseconds
                << " logical_rounds=" << e2e.logical_rounds
                << " status=PASS" << std::endl;
    } else {
      std::ofstream output(output_path);
      if (!output) throw std::runtime_error("cannot write BOLT ViT prediction");
      output << "predicted_class " << predicted << '\n';
      std::cout << "PREDICTION class=" << predicted << std::endl;
    }
    if (party == ALICE) {
      const bool ctct = std::any_of(trace.begin(), trace.end(),
          [](const VitOperatorTrace &row) {
            return (row.op == "qk" || row.op == "softmax_v") &&
                   row.counts.evaluator_multiply > 0 &&
                   row.counts.relinearizations > 0;
          });
      const bool residual_local = std::all_of(trace.begin(), trace.end(),
          [](const VitOperatorTrace &row) {
            return row.backend != "local_share_add" ||
                   (row.bytes == 0 && row.logical_rounds == 0);
          });
      if (!ctct || !residual_local)
        throw std::logic_error("BOLT ViT operation audit failed");
      std::cout << "BOLT_VIT_SINGLE_LAYER_PROTOCOL_EXECUTION="
                << (metadata.layers == 1 ? "PASS" : "NOT_APPLICABLE") << '\n'
                << "BOLT_VIT_MULTI_LAYER_PROTOCOL_EXECUTION="
                << (metadata.layers > 1 ? "PASS" : "NOT_APPLICABLE") << '\n'
                << "PERFORMANCE_ACCOUNTING_VALID=true\n"
                << "PLAINTEXT_INTERMEDIATE_RECONSTRUCTIONS=0\n"
                << "SCALAR_FALLBACKS=0\nGENERIC_OT_MATMUL_CALLS="
                << generic_ot_calls << "\n"
                << "SILENT_FALLBACKS=0" << std::endl;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "BOLT_VIT error: " << error.what() << std::endl;
    return 2;
  }
}
