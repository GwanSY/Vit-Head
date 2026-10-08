#include "hetero_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ours_vit {
namespace {

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(trim(field));
  return fields;
}

HeadProtocol parse_protocol(const std::string &value) {
  if (value == "softmax_mpc") return HeadProtocol::SoftmaxMpc;
  if (value == "relu_norm_mpc") return HeadProtocol::ReluNormMpc;
  if (value == "he_linear") return HeadProtocol::HeLinear;
  throw std::invalid_argument("unknown head protocol: " + value);
}

int parse_int(const std::string &value, const std::string &field,
              const std::string &source, int line_number) {
  size_t used = 0;
  int result = 0;
  try {
    result = std::stoi(value, &used);
  } catch (const std::exception &) {
    throw std::invalid_argument(source + ":" + std::to_string(line_number) +
                                ": invalid " + field);
  }
  if (used != value.size())
    throw std::invalid_argument(source + ":" + std::to_string(line_number) +
                                ": invalid " + field);
  return result;
}

double parse_double(const std::string &value, const std::string &field,
                    const std::string &source, int line_number) {
  size_t used = 0;
  double result = 0.0;
  try {
    result = std::stod(value, &used);
  } catch (const std::exception &) {
    throw std::invalid_argument(source + ":" + std::to_string(line_number) +
                                ": invalid " + field);
  }
  if (used != value.size() || !std::isfinite(result))
    throw std::invalid_argument(source + ":" + std::to_string(line_number) +
                                ": non-finite or invalid " + field);
  return result;
}

}  // namespace

Version parse_version(const std::string &value) {
  if (value == "B") return Version::B;
  if (value == "H") return Version::H;
  if (value == "HA") return Version::HA;
  if (value == "HP") return Version::HP;
  if (value == "HAP") return Version::HAP;
  if (value == "J") return Version::J;
  if (value == "BP") return Version::BP;
  throw std::invalid_argument("--version must be B,H,HA,HP,HAP,J,BP");
}
const char *to_string(Version value) {
  switch (value) {
    case Version::B: return "B";
    case Version::H: return "H";
    case Version::HA: return "HA";
    case Version::HP: return "HP";
    case Version::HAP: return "HAP";
    case Version::J: return "J";
    case Version::BP: return "BP";
  }
  throw std::invalid_argument("invalid version");
}
bool HeteroPlan::has_linear() const {
  for (const auto &layer : layers_)
    for (const auto &head : layer.heads)
      if (head.protocol == HeadProtocol::HeLinear) return true;
  return false;
}

const char *to_string(HeadProtocol value) {
  switch (value) {
    case HeadProtocol::SoftmaxMpc: return "softmax_mpc";
    case HeadProtocol::ReluNormMpc: return "relu_norm_mpc";
    case HeadProtocol::HeLinear: return "he_linear";
  }
  return "invalid";
}

HeteroPlan HeteroPlan::all_softmax() {
  HeteroPlan plan;
  plan.layers_.resize(kPlanLayers);
  for (auto &layer : plan.layers_) {
    layer.heads.reserve(kPlanHeads);
    for (int head = 0; head < kPlanHeads; ++head)
      layer.heads.push_back({head, HeadProtocol::SoftmaxMpc, 1.0});
  }
  return plan;
}

HeteroPlan HeteroPlan::load_file(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open head plan: " + path);
  return parse(input, path);
}

HeteroPlan HeteroPlan::parse(std::istream &input, const std::string &source) {
  HeteroPlan plan;
  plan.layers_.resize(kPlanLayers);
  std::array<std::array<bool, kPlanHeads>, kPlanLayers> seen{};
  bool magic = false;
  bool dimensions = false;
  bool columns = false;
  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    line = trim(line);
    if (line.empty() || line.front() == '#') continue;
    if (!magic) {
      if (line != "OURS_VIT_HEAD_PLAN_V1")
        throw std::invalid_argument(source + ":1: missing OURS_VIT_HEAD_PLAN_V1");
      magic = true;
      continue;
    }
    if (line == "layers=12,heads=12,logical_tokens=197,physical_tokens=256") {
      dimensions = true;
      continue;
    }
    if (line == "layer,physical_head,original_head,protocol,alpha") {
      columns = true;
      continue;
    }
    if (!dimensions || !columns)
      throw std::invalid_argument(source + ":" + std::to_string(line_number) +
                                  ": metadata/header must precede rows");
    const auto fields = split_csv(line);
    if (fields.size() != 5)
      throw std::invalid_argument(source + ":" + std::to_string(line_number) +
                                  ": expected five CSV fields");
    const int layer = parse_int(fields[0], "layer", source, line_number);
    const int physical =
        parse_int(fields[1], "physical_head", source, line_number);
    const int original =
        parse_int(fields[2], "original_head", source, line_number);
    if (layer < 0 || layer >= kPlanLayers || physical < 0 ||
        physical >= kPlanHeads || original < 0 || original >= kPlanHeads)
      throw std::out_of_range(source + ":" + std::to_string(line_number) +
                              ": layer/head index out of range");
    if (seen[layer][physical])
      throw std::invalid_argument(source + ":" + std::to_string(line_number) +
                                  ": duplicate layer/physical_head");
    seen[layer][physical] = true;
    if (plan.layers_[layer].heads.empty())
      plan.layers_[layer].heads.resize(kPlanHeads);
    plan.layers_[layer].heads[physical] = {
        original, parse_protocol(fields[3]),
        parse_double(fields[4], "alpha", source, line_number)};
  }
  if (!magic || !dimensions || !columns)
    throw std::invalid_argument(source + ": incomplete plan header");
  for (int layer = 0; layer < kPlanLayers; ++layer)
    for (int head = 0; head < kPlanHeads; ++head)
      if (!seen[layer][head])
        throw std::invalid_argument(source + ": missing layer=" +
                                    std::to_string(layer) + " physical_head=" +
                                    std::to_string(head));
  return plan;
}

void HeteroPlan::validate(Version version) const {
  if (layers_.size() != kPlanLayers)
    throw std::invalid_argument("head plan must contain 12 layers");
  for (int layer = 0; layer < kPlanLayers; ++layer) {
    if (layers_[layer].heads.size() != kPlanHeads)
      throw std::invalid_argument("head plan layer must contain 12 heads");
    std::array<bool, kPlanHeads> original_seen{};
    std::array<bool, 3> protocol_group_seen{};
    for (int physical = 0; physical < kPlanHeads; ++physical) {
      const auto &head = layers_[layer].heads[physical];
      if (head.original_head < 0 || head.original_head >= kPlanHeads ||
          original_seen[head.original_head])
        throw std::invalid_argument("original_head must be a permutation in layer " +
                                    std::to_string(layer));
      original_seen[head.original_head] = true;
      if (!std::isfinite(head.alpha))
        throw std::invalid_argument("head alpha must be finite");
      if (head.protocol != HeadProtocol::HeLinear &&
          std::abs(head.alpha - 1.0) > 1e-12)
        throw std::invalid_argument(
            "alpha is only legal for he_linear heads; use 1 for MPC heads");
      if ((version == Version::B || version == Version::BP) &&
          (head.protocol != HeadProtocol::SoftmaxMpc ||
           head.original_head != physical || std::abs(head.alpha - 1.0) > 1e-12))
        throw std::invalid_argument(
            "B must be all-softmax with the original head order and alpha=1");
      if (version != Version::B &&
          (physical == 0 ||
           head.protocol != layers_[layer].heads[physical - 1].protocol)) {
        const int protocol = static_cast<int>(head.protocol);
        if (protocol_group_seen[protocol])
          throw std::invalid_argument(
              "H/HA/HP/HAP/J/BP require one contiguous group per protocol in layer " +
              std::to_string(layer));
        protocol_group_seen[protocol] = true;
      }
    }
  }

}

void HeteroPlan::write_audit_csv(const std::string &path,
                                 Version version) const {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write head-plan audit: " + path);
  constexpr int slots_per_ciphertext = 16384;
  constexpr int qk_ciphertexts_per_head = 4;
  const double logical_utilization =
      static_cast<double>(kPlanLogicalTokens) * kPlanLogicalTokens /
      (qk_ciphertexts_per_head * slots_per_ciphertext);
  output << "version,layer,physical_head,original_head,protocol,alpha,"
            "contiguous_group,group_size,qk_ciphertexts,"
            "physical_slot_utilization,logical_slot_utilization,execution_status\n";
  output << std::fixed << std::setprecision(9);
  for (int layer = 0; layer < kPlanLayers; ++layer) {
    int group = -1;
    for (int head = 0; head < kPlanHeads;) {
      ++group;
      const int group_start = head;
      const auto protocol = layers_[layer].heads[head].protocol;
      while (head < kPlanHeads && layers_[layer].heads[head].protocol == protocol)
        ++head;
      const int group_size = head - group_start;
      for (int physical = group_start; physical < head; ++physical) {
        const auto &spec = layers_[layer].heads[physical];
        output << to_string(version) << ',' << layer << ',' << physical << ','
               << spec.original_head << ',' << to_string(spec.protocol) << ','
               << spec.alpha << ',' << group << ',' << group_size << ','
               << qk_ciphertexts_per_head << ",1.000000000,"
               << logical_utilization << ','
               << (version == Version::B ? "original_blb" : "selective_mpc_native_linear")
               << '\n';
      }
    }
  }
}

}  // namespace ours_vit
