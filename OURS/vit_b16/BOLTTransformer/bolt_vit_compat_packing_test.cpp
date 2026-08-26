#include "BOLTTransformer/bolt_vit_compat_packing.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using sci::bolt::BoltVitCompatPacking;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<std::uint64_t> sequence(std::size_t size,
                                    std::uint64_t salt) {
  std::vector<std::uint64_t> result(size);
  for (std::size_t i = 0; i < size; ++i)
    result[i] = (i * 0x9e3779b97f4a7c15ULL) ^ salt;
  return result;
}

} // namespace

int main() {
  try {
    const BoltVitCompatPacking packing;
    const auto metadata = packing.bert_metadata();
    require(metadata.slot_count == 8192 && metadata.image_size == 256 &&
                metadata.filter_height == 768 &&
                metadata.filter_width == 64 && metadata.baby_step == 4 &&
                metadata.giant_steps == 4 &&
                metadata.attention_heads == 12 &&
                metadata.paired_head_groups == 6,
            "ViT BERT compatibility metadata mismatch");
    require(packing.columns_per_ciphertext() == 16,
            "columns-per-ciphertext mismatch");
    require(packing.qkv_input_ciphertexts() == 24 &&
                packing.projection_ciphertexts() == 24 &&
                packing.fused_qkv_ciphertexts() == 72 &&
                packing.qk_ciphertexts() == 96 &&
                packing.softmax_ciphertexts() == 96 &&
                packing.context_ciphertexts() == 24,
            "ViT BERT compatibility ciphertext count mismatch");

    const auto token_major = sequence(256 * 768, 0x1111);
    require(packing.unpack_token_major(
                packing.pack_token_major(token_major)) == token_major,
            "token-major compact round trip failed");
    require(packing.unpack_linear_output(
                packing.pack_linear_output(token_major, 768), 768) ==
                token_major,
            "linear-output compact round trip failed");
    const auto wide_token_major = sequence(256 * 3072, 0x1212);
    require(packing.unpack_token_major(
                packing.pack_token_major(wide_token_major, 3072), 3072) ==
                wide_token_major,
            "wide token-major compact round trip failed");
    require(packing.unpack_linear_output(
                packing.pack_linear_output(wide_token_major, 3072), 3072) ==
                wide_token_major,
            "wide linear-output compact round trip failed");

    const auto qk = sequence(12 * 256 * 256, 0x2222);
    require(packing.unpack_qk_cross(packing.pack_qk_cross(qk)) == qk,
            "QK cross round trip failed");
    require(packing.unpack_softmax_diagonal_source(
                packing.pack_softmax_diagonal_source(qk)) == qk,
            "Softmax diagonal-source round trip failed");
    const auto logical_softmax = sequence(12 * 197 * 197, 0x2525);
    const auto padded_softmax = packing.unpack_softmax_diagonal_source(
        packing.pack_logical_softmax_diagonal_source(logical_softmax, 197));
    for (std::size_t head = 0; head < 12; ++head)
      for (std::size_t query = 0; query < 256; ++query)
        for (std::size_t key = 0; key < 256; ++key) {
          const auto actual =
              padded_softmax[(head * 256 + query) * 256 + key];
          if (query < 197 && key < 197) {
            const auto expected =
                logical_softmax[(head * 197 + query) * 197 + key];
            require(actual == expected,
                    "valid Softmax entry changed during public padding");
          } else {
            require(actual == 0,
                    "public Softmax padding entry is not exactly zero");
          }
        }

    const auto context = sequence(256 * 768, 0x3333);
    require(packing.unpack_context(packing.pack_context(context)) == context,
            "Softmax-V context round trip failed");

    for (std::size_t diagonal = 0; diagonal < 256; ++diagonal) {
      const auto direct = packing.qk_cross_mask_slots(diagonal);
      const auto wrapped = packing.qk_cross_mask_slots(256 + diagonal);
      std::size_t selected = 0;
      for (std::size_t slot = 0; slot < direct.size(); ++slot) {
        require(direct[slot] + wrapped[slot] <= 1,
                "QK masks overlap");
        selected += direct[slot] + wrapped[slot];
      }
      require(selected == 2 * 256,
              "QK masks do not select one full row per BatchEncoder half");

      const auto soft_direct =
          packing.softmax_rotation_mask_slots(false, diagonal);
      const auto soft_wrapped =
          packing.softmax_rotation_mask_slots(true, diagonal);
      selected = 0;
      for (std::size_t slot = 0; slot < soft_direct.size(); ++slot) {
        require(soft_direct[slot] + soft_wrapped[slot] <= 1,
                "Softmax masks overlap");
        selected += soft_direct[slot] + soft_wrapped[slot];
      }
      require(selected == 2 * 256,
              "Softmax masks do not select one full diagonal per half");
    }

    const auto counts = packing.expected_counts();
    require(counts.fused_qkv.ciphertexts == 72 &&
                counts.fused_qkv.rotations == 384 &&
                counts.fused_qkv.multiply_plain == 55296,
            "fused-QKV operation count mismatch");
    require(counts.qk.ciphertexts == 96 &&
                counts.qk.evaluator_multiply == 12288 &&
                counts.qk.relinearizations == 3072 &&
                counts.qk.rotations == 24576 &&
                counts.qk.multiply_plain == 3072,
            "QK operation count mismatch");
    require(counts.softmax_expand.ciphertexts == 3072 &&
                counts.softmax_expand.rotations == 12288 &&
                counts.softmax_expand.multiply_plain == 3072,
            "Softmax expansion operation count mismatch");
    require(counts.softmax_v.ciphertexts == 24 &&
                counts.softmax_v.evaluator_multiply == 12288 &&
                counts.softmax_v.relinearizations == 24 &&
                counts.softmax_v.rotations == 12288,
            "Softmax-V operation count mismatch");

    bool rejected = false;
    try {
      (void)BoltVitCompatPacking(8192, 197, 768, 12, 64);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "logical T197 must not enter the physical T256 kernel");

    std::cout << "VIT_BERT_COMPAT_PACKING=PASS\n"
              << "VIT_BERT_COMPAT_LAYOUT=T256_D768_H12_DH64_N8192\n"
              << "VIT_BERT_COMPAT_CIPHERTEXTS=qkv_in:24,qkv_out:72,qk:96,softmax:96,context:24\n"
              << "VIT_BERT_COMPAT_PER_BLOCK=ctct:24576,rotations:49152,multiply_plain:6144\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "VIT_BERT_COMPAT_PACKING=FAIL reason=" << error.what()
              << '\n';
    return 1;
  }
}
