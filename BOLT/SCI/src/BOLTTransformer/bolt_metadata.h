#ifndef SCI_BOLT_TRANSFORMER_METADATA_H
#define SCI_BOLT_TRANSFORMER_METADATA_H

#include "seal/seal.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sci {
namespace bolt {

enum class BoltPackingKind {
  kCompactRepeatedRows,
  kCrossPackedRows,
};

struct BoltHEParameters {
  std::size_t poly_modulus_degree = 8192;
  std::vector<int> coeff_modulus_bits{54, 54, 55, 55};
  std::uint64_t plain_modulus = 536903681ULL;
  seal::sec_level_type security_level = seal::sec_level_type::tc128;

  static BoltHEParameters bert_main() { return BoltHEParameters{}; }

  static BoltHEParameters bert_tiny_plain_modulus() {
    BoltHEParameters result;
    result.coeff_modulus_bits = {60, 60, 60};
    result.plain_modulus = 557057ULL;
    return result;
  }

  // ViT-B/16 uses the same polynomial degree, coefficient modulus chain and
  // tc128 security target as BOLT-BERT.  Its fixed-point dot products are
  // larger, however: the BERT tiny plaintext modulus would wrap ordinary ViT
  // linears and the fused attention QK path need a wider plaintext modulus
  // after the QKV scale boundary.  ViT uses a 38-bit ring: the HE-to-MPC
  // boundary can temporarily contain a value below 2p, and 2p must remain
  // below the signed half-range used by SCI comparisons.  These are
  // shape/range-derived ViT parameters; the BERT presets above remain
  // unchanged.
  static BoltHEParameters vit_main() {
    BoltHEParameters result = bert_main();
    result.plain_modulus = 45000081409ULL;
    return result;
  }

  static BoltHEParameters vit_linear() {
    // ViT widens the plaintext modulus to keep its larger fixed-point dot
    // products in range.  The BERT tiny chain has only three 60-bit primes;
    // after the BOLT noise-flooding step and one modulus switch that chain
    // does not leave enough decryption margin for the wider ViT modulus.
    // Keep the BERT hardening/packing schedule, but use the tc128 main chain
    // for the shape-derived ViT modulus.
    BoltHEParameters result = vit_main();
    result.plain_modulus = 45000081409ULL;
    return result;
  }

  static BoltHEParameters bert_layer_norm() {
    BoltHEParameters result;
    result.plain_modulus = 4295049217ULL;
    return result;
  }
};

struct BoltTensorMetadata {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t packed_stride = 0;
  std::uint32_t share_bit_width = 37;
  std::uint32_t scale = 12;
  BoltPackingKind packing = BoltPackingKind::kCompactRepeatedRows;
  bool add_plain_modulus_half_on_he_to_ss = false;

  void validate(std::size_t slot_count) const {
    if (rows == 0 || cols == 0 || packed_stride == 0) {
      throw std::invalid_argument("BOLT tensor metadata has a zero dimension");
    }
    if (share_bit_width == 0 || share_bit_width >= 64) {
      throw std::invalid_argument("BOLT supports Z2^ell shares with 1 <= ell < 64");
    }
    if (packing != BoltPackingKind::kCompactRepeatedRows &&
        packing != BoltPackingKind::kCrossPackedRows) {
      throw std::invalid_argument("BOLT scalar ciphertext layout is forbidden");
    }
    if (packed_stride < cols || (packed_stride & (packed_stride - 1)) != 0) {
      throw std::invalid_argument("BOLT packed stride must be a power of two covering cols");
    }
    if (slot_count % 2 != 0 || (slot_count / 2) % packed_stride != 0) {
      throw std::invalid_argument("BOLT packed stride must divide each BatchEncoder row");
    }
  }
};

struct BoltMatMulMetadata {
  std::size_t input_rows = 0;
  std::size_t input_cols = 0;
  std::size_t output_cols = 0;
  std::size_t packed_stride = 0;
  std::size_t baby_step = 0;
  BoltPackingKind input_packing = BoltPackingKind::kCompactRepeatedRows;
  BoltPackingKind output_packing = BoltPackingKind::kCompactRepeatedRows;
  std::uint32_t noise_flooding_bits = 0;
  std::uint32_t modulus_switches = 0;

  void validate(std::size_t slot_count) const {
    if (input_rows == 0 || input_cols == 0 || output_cols == 0 ||
        packed_stride == 0) {
      throw std::invalid_argument("BOLT MatMul metadata has a zero dimension");
    }
    if (packed_stride < input_cols || packed_stride < output_cols ||
        (packed_stride & (packed_stride - 1)) != 0) {
      throw std::invalid_argument("BOLT MatMul packed stride is invalid");
    }
    if ((slot_count / 2) % packed_stride != 0) {
      throw std::invalid_argument("BOLT MatMul packed stride does not fit BatchEncoder");
    }
    if (baby_step == 0 || baby_step > packed_stride) {
      throw std::invalid_argument("BOLT BSGS baby step is invalid");
    }
    if (input_packing != BoltPackingKind::kCompactRepeatedRows ||
        output_packing != BoltPackingKind::kCompactRepeatedRows) {
      throw std::invalid_argument("BOLT MatMul requires compact BatchEncoder packing");
    }
  }
};

// Explicit ViT Attention layout.  A logical row occupies the first
// `valid_cols` slots of every compact stride-sized replica.  This metadata is
// deliberately separate from model code so non-full token/head dimensions do
// not trigger scalar remainder handling.
struct BoltAttentionPackingMetadata {
  std::size_t tokens = 0;
  std::size_t model_dim = 0;
  std::size_t heads = 0;
  std::size_t head_dim = 0;
  std::size_t packed_stride = 0;

  void validate(std::size_t slot_count) const {
    if (tokens == 0 || model_dim == 0 || heads == 0 || head_dim == 0 ||
        packed_stride == 0 || heads * head_dim != model_dim ||
        packed_stride < model_dim ||
        (packed_stride & (packed_stride - 1)) != 0 ||
        slot_count % 2 != 0 || (slot_count / 2) % packed_stride != 0) {
      throw std::invalid_argument(
          "BOLT ViT Attention packing metadata is invalid");
    }
  }
};

struct BoltVitStemMetadata {
  std::size_t image_size = 0;
  std::size_t patch_size = 0;
  std::size_t channels = 0;
  std::size_t patch_dim = 0;
  std::size_t image_tokens = 0;
  std::size_t tokens = 0;
  std::size_t hidden_dim = 0;
  std::size_t packed_stride = 0;
  std::uint32_t ell = 37;
  std::uint32_t scale = 12;
  std::uint64_t layernorm_epsilon_lsb = 0;

  void validate(std::size_t slot_count) const {
    if (image_size == 0 || patch_size == 0 || channels == 0 ||
        image_size % patch_size != 0 || patch_dim == 0 || image_tokens == 0 ||
        tokens != image_tokens + 1 || hidden_dim == 0 ||
        patch_dim != channels * patch_size * patch_size ||
        image_tokens != (image_size / patch_size) *
                            (image_size / patch_size) ||
        packed_stride < patch_dim || packed_stride < hidden_dim ||
        (packed_stride & (packed_stride - 1)) != 0 ||
        slot_count % 2 != 0 || (slot_count / 2) % packed_stride != 0 ||
        ell == 0 || ell >= 64 || scale == 0) {
      throw std::invalid_argument("BOLT ViT Stem metadata is invalid");
    }
  }
};

// End-to-end ViT schedule.  This describes the public model topology and the
// fixed-point/packing contract only; model weights remain ALICE-private.
struct BoltVitModelMetadata {
  std::size_t batch = 1;
  std::size_t image_size = 0;
  std::size_t patch_size = 0;
  std::size_t channels = 0;
  std::size_t patch_dim = 0;
  std::size_t tokens = 0;
  // `tokens` is the logical checkpoint token count.  The secure graph may
  // use a larger public physical row count when the compact layout cannot
  // represent the logical count directly (ViT-B/16 uses 197 -> 256).
  std::size_t physical_tokens = 0;
  std::size_t model_dim = 0;
  std::size_t heads = 0;
  std::size_t head_dim = 0;
  std::size_t ff_dim = 0;
  std::size_t layers = 0;
  std::size_t classes = 0;
  std::size_t packed_stride = 0;
  std::size_t patch_stride = 0;
  std::size_t attention_stride = 0;
  std::size_t ffn_stride = 0;
  std::size_t classifier_stride = 0;
  std::uint32_t ell = 38;
  std::uint32_t activation_scale = 12;
  std::uint32_t weight_scale = 12;
  std::uint32_t gelu_ell = 20;
  std::uint32_t gelu_input_scale = 11;
  std::uint32_t gelu_output_scale = 4;
  std::uint64_t layer_norm_epsilon_lsb = 0;
  bool pre_norm = true;
  bool qkv_bias = false;

  void validate(std::size_t slot_count) const {
    const std::size_t patch_side = image_size / patch_size;
    if (batch != 1 || image_size == 0 || patch_size == 0 ||
        image_size % patch_size != 0 || channels == 0 || patch_dim == 0 ||
        patch_dim != channels * patch_size * patch_size ||
        tokens != patch_side * patch_side + 1 || physical_tokens < tokens ||
        (physical_tokens & (physical_tokens - 1)) != 0 || model_dim == 0 ||
        heads == 0 || model_dim % heads != 0 || head_dim != model_dim / heads ||
        ff_dim == 0 || layers == 0 || classes < 2 || packed_stride == 0 ||
        patch_stride < patch_dim || patch_stride < model_dim ||
        attention_stride < physical_tokens || attention_stride < model_dim ||
        ffn_stride < model_dim || ffn_stride < ff_dim ||
        classifier_stride < model_dim || classifier_stride < classes ||
        (packed_stride & (packed_stride - 1)) != 0 ||
        (patch_stride & (patch_stride - 1)) != 0 ||
        (attention_stride & (attention_stride - 1)) != 0 ||
        (ffn_stride & (ffn_stride - 1)) != 0 ||
        (classifier_stride & (classifier_stride - 1)) != 0 ||
        slot_count % 2 != 0 || (slot_count / 2) % packed_stride != 0 ||
        (slot_count / 2) % patch_stride != 0 ||
        (slot_count / 2) % attention_stride != 0 ||
        (slot_count / 2) % ffn_stride != 0 ||
        (slot_count / 2) % classifier_stride != 0 ||
        ell != 38 || activation_scale != 12 || weight_scale != 12 ||
        gelu_ell != 20 || gelu_input_scale != 11 ||
        gelu_output_scale != 4 || !pre_norm) {
      throw std::invalid_argument("BOLT end-to-end ViT metadata is invalid");
    }
  }
};

// Metadata for the already-deployed BOLT BERT compact/cross layout. Keeping
// these values explicit removes the former 768/3072/12-head branches from the
// shared SEAL kernels without changing their packing.
struct BoltBertPackingMetadata {
  std::size_t slot_count = 0;
  std::size_t image_size = 0;
  std::size_t filter_height = 0;
  std::size_t filter_width = 0;
  std::size_t baby_step = 0;
  std::size_t giant_steps = 0;
  std::size_t attention_heads = 0;
  std::size_t paired_head_groups = 0;
  std::uint32_t noise_flooding_bits = 0;
  std::uint32_t modulus_switches = 0;

  void validate(std::size_t encoder_slots) const {
    if (slot_count != encoder_slots || slot_count == 0 || image_size == 0 ||
        filter_height == 0 || filter_width == 0 || baby_step == 0 ||
        giant_steps == 0 || attention_heads == 0 ||
        paired_head_groups == 0 || attention_heads < 2 * paired_head_groups ||
        slot_count % (2 * image_size) != 0) {
      throw std::invalid_argument("BOLT BERT compact/cross packing metadata is invalid");
    }
  }
};

struct BoltOperationCounts {
  std::uint64_t ciphertexts = 0;
  std::uint64_t rotations = 0;
  std::uint64_t multiply_plain = 0;
  std::uint64_t evaluator_multiply = 0;
  std::uint64_t relinearizations = 0;
  std::uint64_t modulus_switches = 0;
  std::uint64_t noise_floodings = 0;
};

struct SecretSharedTensor {
  BoltTensorMetadata metadata;
  std::vector<std::uint64_t> local_share;

  void validate(std::size_t slot_count) const {
    metadata.validate(slot_count);
    if (local_share.size() != metadata.rows * metadata.cols) {
      throw std::invalid_argument("BOLT secret-share tensor size mismatch");
    }
  }
};

struct PlainWeightTensor {
  // Matrix is output_cols x input_cols, row-major.
  std::size_t output_cols = 0;
  std::size_t input_cols = 0;
  std::vector<std::uint64_t> values;
  std::vector<std::uint64_t> bias;

  void validate() const {
    if (output_cols == 0 || input_cols == 0 ||
        values.size() != output_cols * input_cols) {
      throw std::invalid_argument("BOLT plaintext weight tensor size mismatch");
    }
    if (!bias.empty() && bias.size() != output_cols) {
      throw std::invalid_argument("BOLT plaintext bias size mismatch");
    }
  }
};

struct BoltCipherTensor {
  BoltTensorMetadata metadata;
  std::vector<seal::Ciphertext> ciphertexts;

  void validate(std::size_t slot_count, bool require_ciphertexts = true) const {
    metadata.validate(slot_count);
    if (require_ciphertexts && ciphertexts.size() != metadata.rows) {
      throw std::invalid_argument("BOLT cipher tensor must contain one compact ciphertext per row");
    }
  }
};

inline void fail_if_scalar_ciphertext_adapter(bool scalar_adapter_used) {
  if (scalar_adapter_used) {
    throw std::logic_error(
        "BOLT fail-closed: scalar ciphertext adapter is forbidden");
  }
}

inline void fail_if_generic_ot_matmul(bool generic_ot_used) {
  if (generic_ot_used) {
    throw std::logic_error(
        "BOLT fail-closed: LinearOT::matrix_multiplication is forbidden for ct-ct MatMul");
  }
}

} // namespace bolt
} // namespace sci

#endif
