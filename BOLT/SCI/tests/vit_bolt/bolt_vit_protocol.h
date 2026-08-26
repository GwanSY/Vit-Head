#pragma once

#include "BOLTTransformer/bolt_attention.h"
#include "BOLTTransformer/bolt_conversion.h"
#include "BOLTTransformer/bolt_linear.h"
#include "FloatingPoint/fp-math.h"
#include "bolt_vit_mpc.h"
#include "utils/io_pack.h"
#include "vit_model.h"

#include <cstdint>
#include <string>
#include <vector>

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
  int ell = 38;
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
  BoltVitProtocol(int party, sci::IOPack &io, FPMath &fp,
                  BoltVitMPC &mpc,
                  sci::bolt::BoltHE &he,
                  sci::bolt::BoltHE &he_tiny,
                  sci::bolt::BoltHE &he_layer_norm,
                  const sci::bolt::BoltVitModelMetadata &metadata);

  VitProtocolOutput run(const VitModel &model,
                        const std::vector<std::uint64_t> &image_share,
                        std::vector<VitOperatorTrace> &trace);

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
                              bool input_column_packed = false);
  FixArray layer_norm(const FixArray &input,
                           const std::vector<double> &gamma,
                           const std::vector<double> &beta,
                           const std::string &name, int layer,
                           std::vector<VitOperatorTrace> &trace);
  FixArray attention(const FixArray &input,
                          const VitBlockModel &block, int layer,
                          std::vector<VitOperatorTrace> &trace);
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
  FixArray packed_field_to_ring(const std::vector<std::uint64_t> &field,
                                int scale, sci::bolt::BoltHE &field_he);
  std::vector<seal::Ciphertext> signed_ring_to_he(
      const std::uint64_t *local_share, std::size_t element_count, int ell,
      int scale, sci::bolt::BoltConversion &conversion);
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
                               std::uint64_t elapsed_microseconds,
                               std::uint64_t plain_modulus = 0);
  std::uint64_t wire_bytes() const;
  std::uint64_t wire_rounds() const;

  int party_;
  sci::IOPack &io_;
  FPMath &fp_;
  BoltVitMPC &mpc_;
  sci::bolt::BoltHE &he_;
  sci::bolt::BoltHE &he_tiny_;
  sci::bolt::BoltHE &he_layer_norm_;
  sci::bolt::BoltConversion conversion_;
  sci::bolt::BoltConversion conversion_tiny_;
  sci::bolt::BoltConversion conversion_layer_norm_;
  sci::bolt::BoltLinear linear_;
  sci::bolt::BoltLinear linear_tiny_;
  sci::bolt::BoltLinear linear_layer_norm_;
  sci::bolt::BoltAttentionPacking packing_;
  sci::bolt::BoltVitModelMetadata metadata_;
};
