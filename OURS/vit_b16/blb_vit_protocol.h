#pragma once

#include "BOLTTransformer/bolt_attention.h"
#include "BOLTTransformer/bolt_conversion.h"
#include "BOLTTransformer/bolt_linear.h"
#include "BOLTTransformer/bolt_vit_compat_attention.h"
#include "BOLTTransformer/bolt_vit_compat_conversion.h"
#include "BOLTTransformer/bolt_vit_compat_linear.h"
#include "BOLTTransformer/bolt_vit_compat_packing.h"
#include "FloatingPoint/fp-math.h"
#include "OT/ot_pack.h"
#include "utils/io_pack.h"
#include "vit_model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef BLB_VIT_VALIDATION_OUTPUTS
class VitValidationObserver;
#endif

struct VitOperatorTrace {
  std::string phase;
  int layer = -1;
  std::string op;
  std::string input_shape;
  std::string output_shape;
  std::string input_packing;
  std::string output_packing;
  int input_scale = 0;
  int output_scale = 0;
  int ell = 37;
  std::uint64_t plain_modulus = 0;
  sci::bolt::BoltOperationCounts counts;
  double milliseconds = 0.0;
  std::uint64_t bytes = 0;
  std::uint64_t logical_rounds = 0;
  std::string backend;
  bool status = true;
};

struct VitProtocolOutput {
  FixArray logits;
  FixArray predicted_class;
};

class BoltVitProtocol {
public:
  BoltVitProtocol(int party, sci::IOPack &io, sci::OTPack &ot, FPMath &fp,
                  sci::bolt::BoltHE &he,
                  const sci::bolt::BoltVitModelMetadata &metadata);

  VitProtocolOutput run(const VitModel &model,
                        const std::vector<std::uint64_t> &image_share,
                        std::vector<VitOperatorTrace> &trace,
                        std::size_t block_limit = 0
#ifdef BLB_VIT_VALIDATION_OUTPUTS
                        , VitValidationObserver *validation_observer = nullptr
#endif
                        );

  void aggregate_trace(std::vector<VitOperatorTrace> &trace);

private:
  FixArray packed_linear(const FixArray &input,
                              std::size_t rows, std::size_t input_dim,
                              std::size_t output_dim, std::size_t stride,
                              const std::vector<double> &weight,
                              const std::vector<double> &bias,
                              int output_scale, const std::string &op,
                              int layer, std::vector<VitOperatorTrace> &trace,
                              const std::string &backend = "bolt_ct_pt_bsgs",
                              sci::bolt::BoltCipherTensor *retained_packed_output = nullptr,
                              int weight_scale = 12,
                              double weight_multiplier = 1.0);
  FixArray compat_packed_linear(
      const FixArray &input, std::size_t rows, std::size_t input_dim,
      std::size_t output_dim, const std::vector<double> &weight,
      const std::vector<double> &bias, int output_scale,
      const std::string &op, int layer,
      std::vector<VitOperatorTrace> &trace, int weight_scale,
      double weight_multiplier);
  FixArray compat_layer_norm_affine(
      const FixArray &normalized, const std::vector<double> &gamma,
      const std::vector<double> &beta, const std::string &name, int layer,
      std::vector<VitOperatorTrace> &trace);
  FixArray compat_attention(
      const FixArray &input, const VitBlockModel &block, int layer,
      std::vector<VitOperatorTrace> &trace
#ifdef BLB_VIT_VALIDATION_OUTPUTS
      , VitValidationObserver *validation_observer
#endif
      );
  FixArray compat_packed_cipher_to_ring(
      const std::vector<seal::Ciphertext> &ciphertexts,
      std::size_t ciphertext_count, int scale);
  std::vector<seal::Ciphertext> compat_slots_to_cipher(
      const std::vector<std::uint64_t> &packed_local_shares,
      int ell, int scale,
      sci::bolt::BoltVitCompatConversionStats *stats = nullptr);
  bool compact_compat_enabled() const;
  FixArray layer_norm(const FixArray &input,
                           const std::vector<double> &gamma,
                           const std::vector<double> &beta,
                           const std::string &name, int layer,
                           std::vector<VitOperatorTrace> &trace);
  FixArray attention(const FixArray &input,
                          const VitBlockModel &block, int layer,
                          std::vector<VitOperatorTrace> &trace
#ifdef BLB_VIT_VALIDATION_OUTPUTS
                          , VitValidationObserver *validation_observer
#endif
                          );
  FixArray gelu_bolt(const FixArray &input, int layer,
                          std::vector<VitOperatorTrace> &trace);
  FixArray residual_add(const FixArray &lhs,
                             const FixArray &rhs,
                             const std::string &name, int layer,
                             std::vector<VitOperatorTrace> &trace);
  void apply_public_token_mask(
      FixArray &value, const std::string &name, int layer,
      std::vector<VitOperatorTrace> &trace);
  void apply_public_attention_mask(
      FixArray &value, const std::string &name, int layer,
      std::vector<VitOperatorTrace> &trace);
  FixArray cipher_to_ring(const sci::bolt::BoltCipherTensor &cipher,
                               std::size_t rows, std::size_t cols,
                               std::size_t stride, int scale);
  sci::bolt::BoltCipherTensor shares_to_cipher(const FixArray &input,
                                               std::size_t rows,
                                               std::size_t cols,
                                               std::size_t stride);
  void add_local_bias(FixArray &value,
                      const std::vector<double> &bias,
                      std::size_t rows, std::size_t cols, int scale);
  void append_trace(std::vector<VitOperatorTrace> &trace,
                    VitOperatorTrace row, std::uint64_t before_bytes,
                    std::uint64_t before_rounds,
                    std::uint64_t elapsed_microseconds);
  void append_conversion_trace(std::vector<VitOperatorTrace> &trace,
                               const std::string &op, int layer,
                               std::size_t rows, std::size_t cols, int scale,
                               const std::string &input_packing,
                               const std::string &output_packing,
                               const std::string &backend,
                               std::uint64_t before_bytes,
                               std::uint64_t before_rounds,
                               std::uint64_t elapsed_microseconds);

  int party_;
  sci::IOPack &io_;
  sci::OTPack &ot_;
  FPMath &fp_;
  sci::bolt::BoltHE &he_;
  sci::bolt::BoltConversion conversion_;
  sci::bolt::BoltLinear linear_;
  sci::bolt::BoltAttentionPacking packing_;
  sci::bolt::BoltVitModelMetadata metadata_;
  std::unique_ptr<sci::bolt::BoltVitCompatPacking> compat_packing_;
  std::unique_ptr<sci::bolt::BoltVitCompatConversion> compat_conversion_;
  std::unique_ptr<sci::bolt::BoltVitCompatLinearAdapter> compat_linear_;
  std::unique_ptr<sci::bolt::BoltVitCompatAttentionAdapter> compat_attention_;
};
