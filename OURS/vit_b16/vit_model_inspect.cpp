#include "vit_model.h"

#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: blb_vit_model_inspect MODEL\n";
    return 2;
  }
  try {
    const VitModel model = load_vit_model(argv[1]);
    std::cout << "VIT_MODEL_LOAD=PASS"
              << " image_size=" << model.image_size
              << " patch_size=" << model.patch_size
              << " channels=" << model.channels
              << " tokens=" << model.tokens
              << " dim=" << model.dim
              << " heads=" << model.heads
              << " ff_dim=" << model.ff_dim
              << " layers=" << model.layers
              << " classes=" << model.classes
              << " norm_order=" << model.norm_order
              << " qkv_bias=" << model.qkv_bias << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "VIT_MODEL_LOAD=FAIL reason=" << error.what() << '\n';
    return 1;
  }
}
