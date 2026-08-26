#pragma once

#include <string>
#include <vector>

struct VitBlockModel {
  std::vector<double> ln1_weight, ln1_bias;
  std::vector<double> wq, wk, wv, q_bias, k_bias, v_bias, wo, bo;
  std::vector<double> ln2_weight, ln2_bias;
  std::vector<double> ff1_weight, ff1_bias, ff2_weight, ff2_bias;
};

struct VitModel {
  std::string source_id = "tiny_fixture";
  std::string source_revision = "local";
  std::string source_sha256 = "not_applicable";
  std::string architecture = "pre_norm_vit";
  std::string norm_order = "pre";
  // Standard ViT checkpoints use epsilon=1e-12 when legacy V1 fixtures do
  // not carry explicit architecture metadata.  The secure fixed-point path
  // quantizes this value at activation scale 12, so the wire format remains
  // unchanged for the legacy fixture.
  double layer_norm_epsilon = 1e-12;
  bool qkv_bias = false;
  int image_size = 0;
  int patch_size = 0;
  int channels = 0;
  int tokens = 0;
  int dim = 0;
  int heads = 0;
  int ff_dim = 0;
  int layers = 0;
  int classes = 0;
  int scale_bits = 12;
  std::vector<double> patch_weight, patch_bias;
  std::vector<double> class_token, position;
  std::vector<VitBlockModel> blocks;
  std::vector<double> final_ln_weight, final_ln_bias;
  std::vector<double> classifier_weight, classifier_bias;
};

struct VitInput {
  int image_size = 0;
  int channels = 0;
  std::vector<double> pixels;
};

VitModel load_vit_model(const std::string &path);
VitInput load_vit_input(const std::string &path);
