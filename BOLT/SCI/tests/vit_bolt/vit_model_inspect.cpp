#include "vit_model.h"

#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: BOLT_VIT_MODEL_INSPECT MODEL");
    const VitModel model = load_vit_model(argv[1]);
    if (model.source_id.empty() || model.source_revision.empty() ||
        model.source_sha256.size() != 64 || model.norm_order != "pre" ||
        !model.qkv_bias || model.blocks.size() != static_cast<std::size_t>(model.layers))
      throw std::runtime_error("checkpoint model metadata contract failed");
    std::cout << "CHECKPOINT_MODEL_LOADER=PASS\n"
              << "source_id=" << model.source_id << '\n'
              << "source_revision=" << model.source_revision << '\n'
              << "source_sha256=" << model.source_sha256 << '\n'
              << "shape=image:" << model.image_size << " patch:"
              << model.patch_size << " tokens:" << model.tokens << " dim:"
              << model.dim << " heads:" << model.heads << " layers:"
              << model.layers << " classes:" << model.classes << '\n'
              << "strict_loader=true\nqkv_bias=true\npre_norm=true\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "BOLT_VIT_MODEL_INSPECT error: " << error.what() << '\n';
    return 2;
  }
}
