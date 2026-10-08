#pragma once

#include "hetero_plan.h"
#include <openssl/sha.h>

#include <array>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ours_vit {

enum class EdgeCloudRole { Cloud = 1, Edge = 2 };

inline EdgeCloudRole edge_cloud_role(int party) {
  if (party == 1) return EdgeCloudRole::Cloud;
  if (party == 2) return EdgeCloudRole::Edge;
  throw std::invalid_argument("edge/cloud party must be 1 or 2");
}

struct EdgeCloudModelSpec {
  std::string revision;
  int logical_tokens;
  int physical_tokens;
  int width;
  int head_dim;
  int classes;
  int executed_blocks;
  // Model authors register fused operator boundaries here, not crypto kernels.
  std::vector<std::string> block_nodes = {
      "ln1", "qkv", "attention_norm", "attention_output",
      "ln2", "ffn1", "gelu", "ffn2"};
};

struct CompiledHeadLayer {
  std::vector<HeadProtocol> protocols;
  std::vector<double> alphas;
};

// Native counterpart of lamination's prepare/static-forward lifecycle.
// Only a public contract digest crosses the existing channel during setup;
// online node dispatch is inline and never serializes a tensor or share.
class EdgeCloudRuntime {
 public:
  EdgeCloudRuntime(int party, EdgeCloudModelSpec model, const HeteroPlan &plan,
                   Version version)
      : role_(edge_cloud_role(party)), model_(std::move(model)),
        version_(version) {
    plan.validate(version);
    if (model_.revision.empty() || model_.logical_tokens <= 0 ||
        model_.physical_tokens < model_.logical_tokens || model_.width <= 0 ||
        model_.head_dim <= 0 || model_.classes <= 0 || model_.executed_blocks < 0 ||
        model_.executed_blocks > static_cast<int>(plan.layers().size()) ||
        model_.block_nodes.empty())
      throw std::invalid_argument("invalid edge/cloud model descriptor");
    std::ostringstream contract;
    contract.imbue(std::locale::classic());
    contract << "OURS_EDGE_CLOUD_V1\n" << model_.revision << '\n'
             << model_.logical_tokens << ' ' << model_.physical_tokens << ' '
             << model_.width << ' ' << model_.head_dim << ' ' << model_.classes
             << ' ' << model_.executed_blocks << ' ' << to_string(version_) << '\n'
             << "output=public_to_both;crypto=ckks_mpc_wide_shared_rns_v1;"
             << "attention=complete_diagonal_he_product_v1;"
             << "conversion=algorithm2_coefficient_extension_v1;softmax=blb_t6_threshold_minus13_v1;input_ciphertext=seeded_v1\n";
    for (const auto &node : model_.block_nodes) {
      if (node.empty() || node.find_first_of("\n\r ") != std::string::npos)
        throw std::invalid_argument("invalid public node identifier");
      contract << node << '\n';
    }
    for (const auto &layer : plan.layers()) {
      CompiledHeadLayer compiled;
      if (layer.heads.size() * model_.head_dim != static_cast<size_t>(model_.width))
        throw std::invalid_argument("head plan/model width mismatch");
      for (const auto &head : layer.heads) {
        compiled.protocols.push_back(head.protocol);
        compiled.alphas.push_back(head.alpha);
        contract << head.original_head << ' ' << to_string(head.protocol) << ' '
                 << std::hexfloat << head.alpha << '\n';
      }
      layers_.push_back(std::move(compiled));
    }
    // Bind public behavior, not model/input/output paths or CUDA device IDs.
    for (const char *key : {
        "OURS_VIT_MPC_WORKERS",
        "OURS_VIT_SECURE_EXP_CLAMP",
        "OURS_VIT_B_TRACE", "OURS_VIT_HETERO_ATTENTION_TRACE",
        "OURS_VIT_PARTIAL_FINAL_LOGITS", "OURS_VIT_B_PARTIAL_FINAL_LOGITS",
        "OURS_VIT_STOP_AFTER_ATTENTION", "OURS_VIT_B_STOP_AFTER_ATTENTION",
        "OURS_VIT_B_TRACE_STOP_AFTER",
        "OURS_VIT_B_SKIP_FINAL_LN", "OURS_VIT_B_SKIP_BSGS2_FLOOD",
        "OURS_VIT_B_PUBLIC_CLASSIFIER_FIXTURE", "OURS_VIT_B_AGGREGATE_MAX",
        "OURS_VIT_B_AGGREGATE_FILTER", "OURS_VIT_B_NATIVE_OBSERVER_CHECK",
        "OURS_VIT_CONVERSION_EXP_DIRECT_CHECK", "OURS_VIT_SOFTMAX_INTERNAL_DIAGNOSTIC",
        "BLB_VIT_DIAGNOSTIC_PARTIAL_CLASSIFIER", "BLB_QKV_AGGREGATE",
        "BLB_LN_COMPONENTS_AGGREGATE"}) {
      const char *value = std::getenv(key);
      contract << key << '=' << (value == nullptr ? "<unset>" : value) << '\n';
    }
    contract_ = contract.str();
    SHA256(reinterpret_cast<const unsigned char *>(contract_.data()),
           contract_.size(), digest_.data());
  }

  bool is_cloud() const { return role_ == EdgeCloudRole::Cloud; }
  bool is_edge() const { return !is_cloud(); }
  const char *role_name() const { return is_cloud() ? "cloud" : "edge"; }
  const std::string &public_contract() const { return contract_; }
  std::string contract_digest() const {
    std::ostringstream out;
    for (auto byte : digest_)
      out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    return out.str();
  }

  template <class Channel> void agree(Channel &io) {
    if (agreed_) throw std::logic_error("edge/cloud contract already agreed");
    std::array<unsigned char, SHA256_DIGEST_LENGTH + 1> local{}, peer{};
    std::copy(digest_.begin(), digest_.end(), local.begin());
    local.back() = static_cast<unsigned char>(role_);
    if (is_cloud()) {
      io.send_data(local.data(), local.size());
      io.flush();
      io.recv_data(peer.data(), peer.size());
    } else {
      io.recv_data(peer.data(), peer.size());
      io.send_data(local.data(), local.size());
      io.flush();
    }
    if (!std::equal(digest_.begin(), digest_.end(), peer.begin()) ||
        peer.back() != (is_cloud() ? 2 : 1))
      throw std::runtime_error("edge/cloud public model/plan/options mismatch");
    agreed_ = true;
  }

  template <class Backend, class Model>
  void prepare_layer(Backend &backend, Model &weights, int layer) {
    if (!agreed_ || active_ != -1 || layer != next_layer_ ||
        layer >= model_.executed_blocks)
      throw std::logic_error("invalid edge/cloud layer preparation order");
    const auto &compiled = layers_.at(layer);
    if (version_ != Version::B)
      backend.configure_heterogeneous_attention_layer(
          layer, compiled.protocols, compiled.alphas, version_);
    if (is_cloud()) backend.linear.weights_preprocess_layer(weights, layer);
    active_ = layer;
    next_node_ = 0;
  }

  template <class Function>
  decltype(auto) execute_node(int layer, const char *node, Function &&function) {
    if (!agreed_ || active_ != layer || next_node_ >= model_.block_nodes.size() ||
        model_.block_nodes[next_node_] != node)
      throw std::logic_error("edge/cloud static operator order mismatch");
    ++next_node_;
    return std::forward<Function>(function)();
  }

  template <class Backend, class Tensor>
  auto attention_norm(Backend &backend, Tensor &qk, int layer) {
    const auto &compiled = layers_.at(layer);
    return execute_node(layer, "attention_norm", [&]() {
      if (version_ == Version::B) return backend.softmax_he(qk);
      return backend.heterogeneous_attention_norm(
          qk, compiled.protocols, compiled.alphas, model_.logical_tokens, layer,
          uses_packing(version_));
    });
  }

  template <class Backend, class Tensor>
  auto attention_output(Backend &backend, Tensor &qk, int layer) {
    return execute_node(layer, "attention_output", [&]() {
      return version_ != Version::B
          ? backend.heterogeneous_attention_output(qk, layer)
          : backend.Softmax_v_linear_layer_norm_part1(qk, layer);
    });
  }

  template <class Backend>
  void release_layer(Backend &backend, int layer, bool diagnostic_prefix = false) {
    if (active_ != layer || (!diagnostic_prefix && next_node_ != model_.block_nodes.size()))
      throw std::logic_error("edge/cloud layer released before static forward completed");
    if (is_cloud()) backend.linear.release_preprocessed_layer(layer);
    active_ = -1;
    ++next_layer_;
  }

  const CompiledHeadLayer &layer_plan(int layer) const { return layers_.at(layer); }

 private:
  EdgeCloudRole role_;
  EdgeCloudModelSpec model_;
  Version version_;
  std::vector<CompiledHeadLayer> layers_;
  std::string contract_;
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest_{};
  bool agreed_ = false;
  int active_ = -1;
  int next_layer_ = 0;
  size_t next_node_ = 0;
};

}  // namespace ours_vit
