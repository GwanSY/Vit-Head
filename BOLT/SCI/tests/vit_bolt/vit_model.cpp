#include "vit_model.h"

#include <fstream>
#include <stdexcept>

namespace {

void expect(std::istream &in, const std::string &wanted) {
  std::string got;
  if (!(in >> got) || got != wanted)
    throw std::runtime_error("expected token '" + wanted + "', got '" + got + "'");
}

std::vector<double> values(std::istream &in, size_t count) {
  std::vector<double> out(count);
  for (double &value : out)
    if (!(in >> value)) throw std::runtime_error("truncated numeric vector");
  return out;
}

} // namespace

VitModel load_vit_model(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open ALICE model: " + path);
  std::string magic;
  if (!(in >> magic)) throw std::runtime_error("empty ViT model file");
  const bool bolt_format = magic == "BOLT_VIT_MODEL_V1" ||
                           magic == "BOLT_VIT_MODEL_V2" ||
                           magic == "BOLT_VIT_MODEL_V3";
  const bool iron_format = magic == "IRON_VIT_MODEL_V1" ||
                           magic == "IRON_VIT_MODEL_V2" ||
                           magic == "IRON_VIT_MODEL_V3";
  if (!bolt_format && !iron_format)
    throw std::runtime_error("unsupported ViT model format: " + magic);
  const bool version_two = magic == "BOLT_VIT_MODEL_V2" ||
                           magic == "BOLT_VIT_MODEL_V3" ||
                           magic == "IRON_VIT_MODEL_V2" ||
                           magic == "IRON_VIT_MODEL_V3";
  VitModel model;
  if (version_two) {
    expect(in, "source_id"); in >> model.source_id;
    expect(in, "source_revision"); in >> model.source_revision;
    expect(in, "source_sha256"); in >> model.source_sha256;
    expect(in, "architecture"); in >> model.architecture;
    expect(in, "norm_order"); in >> model.norm_order;
    expect(in, "layer_norm_epsilon"); in >> model.layer_norm_epsilon;
    int qkv_bias = 0;
    expect(in, "qkv_bias"); in >> qkv_bias;
    if (model.architecture != "pre_norm_vit" || model.norm_order != "pre" ||
        (qkv_bias != 0 && qkv_bias != 1) ||
        model.layer_norm_epsilon < 0.0)
      throw std::runtime_error("unsupported checkpoint architecture metadata");
    model.qkv_bias = qkv_bias == 1;
  }
  expect(in, "image_size"); in >> model.image_size;
  expect(in, "patch_size"); in >> model.patch_size;
  expect(in, "channels"); in >> model.channels;
  expect(in, "tokens"); in >> model.tokens;
  expect(in, "dim"); in >> model.dim;
  expect(in, "heads"); in >> model.heads;
  expect(in, "ff_dim"); in >> model.ff_dim;
  expect(in, "layers"); in >> model.layers;
  expect(in, "classes"); in >> model.classes;
  expect(in, "scale_bits"); in >> model.scale_bits;
  if (model.image_size <= 0 || model.patch_size <= 0 ||
      model.image_size % model.patch_size || model.channels <= 0 ||
      model.dim <= 0 || model.heads <= 0 || model.dim % model.heads ||
      model.ff_dim <= 0 || model.layers <= 0 || model.classes <= 1 ||
      model.scale_bits < 8 || model.scale_bits > 14)
    throw std::runtime_error("invalid ViT dimensions");
  const int side = model.image_size / model.patch_size;
  if (model.tokens != side * side + 1)
    throw std::runtime_error("tokens must equal patch count plus CLS");

  const size_t patch_features = static_cast<size_t>(model.channels) *
                                model.patch_size * model.patch_size;
  expect(in, "patch_weight");
  model.patch_weight = values(in, static_cast<size_t>(model.dim) * patch_features);
  expect(in, "patch_bias"); model.patch_bias = values(in, model.dim);
  expect(in, "class_token"); model.class_token = values(in, model.dim);
  expect(in, "position");
  model.position = values(in, static_cast<size_t>(model.tokens) * model.dim);
  const size_t square = static_cast<size_t>(model.dim) * model.dim;
  model.blocks.resize(model.layers);
  for (int layer = 0; layer < model.layers; ++layer) {
    int id = -1;
    expect(in, "block"); in >> id;
    if (id != layer) throw std::runtime_error("unexpected block id");
    auto &block = model.blocks[layer];
    expect(in, "ln1_weight"); block.ln1_weight = values(in, model.dim);
    expect(in, "ln1_bias"); block.ln1_bias = values(in, model.dim);
    expect(in, "wq"); block.wq = values(in, square);
    if (version_two) { expect(in, "q_bias"); block.q_bias = values(in, model.dim); }
    expect(in, "wk"); block.wk = values(in, square);
    if (version_two) { expect(in, "k_bias"); block.k_bias = values(in, model.dim); }
    expect(in, "wv"); block.wv = values(in, square);
    if (version_two) { expect(in, "v_bias"); block.v_bias = values(in, model.dim); }
    if (!version_two) {
      block.q_bias.assign(model.dim, 0.0);
      block.k_bias.assign(model.dim, 0.0);
      block.v_bias.assign(model.dim, 0.0);
    }
    expect(in, "wo"); block.wo = values(in, square);
    expect(in, "bo"); block.bo = values(in, model.dim);
    expect(in, "ln2_weight"); block.ln2_weight = values(in, model.dim);
    expect(in, "ln2_bias"); block.ln2_bias = values(in, model.dim);
    expect(in, "ff1_weight");
    block.ff1_weight = values(in, static_cast<size_t>(model.ff_dim) * model.dim);
    expect(in, "ff1_bias"); block.ff1_bias = values(in, model.ff_dim);
    expect(in, "ff2_weight");
    block.ff2_weight = values(in, static_cast<size_t>(model.dim) * model.ff_dim);
    expect(in, "ff2_bias"); block.ff2_bias = values(in, model.dim);
  }
  expect(in, "final_ln_weight"); model.final_ln_weight = values(in, model.dim);
  expect(in, "final_ln_bias"); model.final_ln_bias = values(in, model.dim);
  expect(in, "classifier_weight");
  model.classifier_weight = values(in, static_cast<size_t>(model.classes) * model.dim);
  expect(in, "classifier_bias"); model.classifier_bias = values(in, model.classes);
  expect(in, "end");
  std::string trailing;
  if (in >> trailing)
    throw std::runtime_error("unexpected data after ViT model terminator");
  return model;
}

VitInput load_vit_input(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open BOB image: " + path);
  std::string magic;
  if (!(in >> magic) ||
      (magic != "BOLT_VIT_INPUT_V1" && magic != "IRON_VIT_INPUT_V1"))
    throw std::runtime_error("unsupported ViT input format: " + magic);
  VitInput input;
  expect(in, "image_size"); in >> input.image_size;
  expect(in, "channels"); in >> input.channels;
  if (input.image_size <= 0 || input.channels <= 0)
    throw std::runtime_error("invalid image dimensions");
  expect(in, "pixels");
  input.pixels = values(in, static_cast<size_t>(input.channels) *
                                input.image_size * input.image_size);
  expect(in, "end");
  return input;
}
