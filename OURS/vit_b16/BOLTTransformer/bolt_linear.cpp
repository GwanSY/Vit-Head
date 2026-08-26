#include "BOLTTransformer/bolt_linear.h"

#include "LinearHE/utils-HE.h"

#include <algorithm>
#include <stdexcept>

namespace sci {
namespace bolt {
namespace {

void increment(std::uint64_t BoltOperationCounts::*member,
               BoltOperationCounts *counts, std::uint64_t amount = 1) {
  if (counts != nullptr) {
    counts->*member += amount;
  }
}

BoltTensorMetadata output_tensor_metadata(const BoltMatMulMetadata &metadata) {
  BoltTensorMetadata result;
  result.rows = metadata.input_rows;
  result.cols = metadata.output_cols;
  result.packed_stride = metadata.packed_stride;
  result.packing = metadata.output_packing;
  return result;
}

} // namespace

std::vector<std::uint64_t> BoltLinear::repeated_slots(
    const std::vector<std::uint64_t> &logical,
    std::size_t packed_stride) const {
  if (logical.size() > packed_stride) {
    throw std::invalid_argument("BOLT logical packed vector exceeds stride");
  }
  std::vector<std::uint64_t> slots(he_.encoder->slot_count(), 0);
  for (std::size_t block = 0; block < slots.size(); block += packed_stride) {
    std::copy(logical.begin(), logical.end(), slots.begin() + block);
  }
  return slots;
}

void BoltLinear::harden_outputs(BoltCipherTensor &tensor,
                                const BoltMatMulMetadata &metadata,
                                BoltOperationCounts *counts) {
  for (auto &ciphertext : tensor.ciphertexts) {
    if (metadata.noise_flooding_bits != 0) {
      auto context_data = he_.context->get_context_data(ciphertext.parms_id());
      if (context_data == nullptr) {
        throw std::logic_error("BOLT cannot locate ciphertext context for flooding");
      }
      flood_ciphertext(ciphertext, context_data,
                       metadata.noise_flooding_bits);
      increment(&BoltOperationCounts::noise_floodings, counts);
    }
    for (std::uint32_t level = 0; level < metadata.modulus_switches; ++level) {
      const auto context_data = he_.context->get_context_data(ciphertext.parms_id());
      if (context_data == nullptr || context_data->next_context_data() == nullptr) {
        throw std::logic_error(
            "BOLT fail-closed: requested modulus switch exceeds the BFV chain");
      }
      he_.evaluator->mod_switch_to_next_inplace(ciphertext);
      increment(&BoltOperationCounts::modulus_switches, counts);
    }
  }
}

BoltCipherTensor BoltLinear::cipher_plain_matmul_bsgs(
    const BoltCipherTensor &input, const PlainWeightTensor &weight,
    const BoltMatMulMetadata &metadata, BoltOperationCounts *counts) {
  he_.require_public_evaluation(false);
  metadata.validate(he_.encoder->slot_count());
  input.validate(he_.encoder->slot_count(), true);
  weight.validate();
  if (he_.party != ALICE) {
    throw std::logic_error("BOLT HE evaluation must execute on ALICE");
  }
  if (input.metadata.rows != metadata.input_rows ||
      input.metadata.cols != metadata.input_cols ||
      input.metadata.packed_stride != metadata.packed_stride ||
      weight.input_cols != metadata.input_cols ||
      weight.output_cols != metadata.output_cols) {
    throw std::invalid_argument("BOLT ct-pt MatMul metadata mismatch");
  }

  const std::size_t stride = metadata.packed_stride;
  const std::size_t baby_step = metadata.baby_step;
  const std::size_t giant_steps = (stride + baby_step - 1) / baby_step;
  BoltCipherTensor result;
  result.metadata = output_tensor_metadata(metadata);
  result.metadata.share_bit_width = input.metadata.share_bit_width;
  result.metadata.scale = input.metadata.scale;
  result.ciphertexts.resize(metadata.input_rows);

  for (std::size_t row = 0; row < metadata.input_rows; ++row) {
    std::vector<seal::Ciphertext> baby_rotations(baby_step);
    baby_rotations[0] = input.ciphertexts[row];
    for (std::size_t baby = 1; baby < baby_step; ++baby) {
      he_.rotate_rows_logical(input.ciphertexts[row],
                              static_cast<int>(baby),
                              baby_rotations[baby]);
      increment(&BoltOperationCounts::rotations, counts);
    }

    bool output_initialized = false;
    seal::Ciphertext output;
    for (std::size_t giant = 0; giant < giant_steps; ++giant) {
      const std::size_t giant_offset = giant * baby_step;
      bool group_initialized = false;
      seal::Ciphertext group;
      for (std::size_t baby = 0; baby < baby_step; ++baby) {
        const std::size_t diagonal = giant_offset + baby;
        if (diagonal >= stride) {
          break;
        }
        std::vector<std::uint64_t> logical(stride, 0);
        bool nonzero = false;
        for (std::size_t packed_col = 0; packed_col < stride; ++packed_col) {
          const std::size_t output_col =
              (packed_col + stride - (giant_offset % stride)) % stride;
          const std::size_t input_col = (output_col + diagonal) % stride;
          if (output_col < metadata.output_cols &&
              input_col < metadata.input_cols) {
            logical[packed_col] =
                weight.values[output_col * metadata.input_cols + input_col] %
                he_.plain_modulus;
            nonzero = nonzero || logical[packed_col] != 0;
          }
        }
        if (!nonzero) {
          continue;
        }
        seal::Plaintext diagonal_plaintext;
        he_.encoder->encode(repeated_slots(logical, stride),
                            diagonal_plaintext);
        seal::Ciphertext product;
        he_.evaluator->multiply_plain(baby_rotations[baby],
                                      diagonal_plaintext, product);
        increment(&BoltOperationCounts::multiply_plain, counts);
        if (!group_initialized) {
          group = std::move(product);
          group_initialized = true;
        } else {
          he_.evaluator->add_inplace(group, product);
        }
      }
      if (!group_initialized) {
        continue;
      }
      if (giant_offset != 0) {
        seal::Ciphertext rotated;
        he_.rotate_rows_logical(group, static_cast<int>(giant_offset),
                                rotated);
        group = std::move(rotated);
        increment(&BoltOperationCounts::rotations, counts);
      }
      if (!output_initialized) {
        output = std::move(group);
        output_initialized = true;
      } else {
        he_.evaluator->add_inplace(output, group);
      }
    }

    if (!output_initialized) {
      he_.encryptor->encrypt_zero(output);
      increment(&BoltOperationCounts::ciphertexts, counts);
    }
    if (!weight.bias.empty()) {
      std::vector<std::uint64_t> logical(stride, 0);
      for (std::size_t col = 0; col < metadata.output_cols; ++col) {
        logical[col] = weight.bias[col] % he_.plain_modulus;
      }
      seal::Plaintext bias_plaintext;
      he_.encoder->encode(repeated_slots(logical, stride), bias_plaintext);
      he_.evaluator->add_plain_inplace(output, bias_plaintext);
    }
    result.ciphertexts[row] = std::move(output);
  }
  increment(&BoltOperationCounts::ciphertexts, counts,
            result.ciphertexts.size());
  harden_outputs(result, metadata, counts);
  return result;
}

BoltCipherTensor BoltLinear::cipher_plain_elementwise_affine(
    const BoltCipherTensor &input,
    const std::vector<std::uint64_t> &multiplier,
    const std::vector<std::uint64_t> &bias,
    const BoltMatMulMetadata &metadata, BoltOperationCounts *counts) {
  he_.require_public_evaluation(false);
  metadata.validate(he_.encoder->slot_count());
  input.validate(he_.encoder->slot_count(), true);
  if (he_.party != ALICE)
    throw std::logic_error("BOLT HE evaluation must execute on ALICE");
  if (metadata.input_cols != metadata.output_cols ||
      input.metadata.rows != metadata.input_rows ||
      input.metadata.cols != metadata.input_cols ||
      input.metadata.packed_stride != metadata.packed_stride ||
      multiplier.size() != metadata.input_cols ||
      bias.size() != metadata.output_cols)
    throw std::invalid_argument("BOLT ct-pt elementwise affine metadata mismatch");

  std::vector<std::uint64_t> multiplier_slots(metadata.packed_stride, 0);
  std::vector<std::uint64_t> bias_slots(metadata.packed_stride, 0);
  bool nonzero_multiplier = false;
  for (std::size_t col = 0; col < metadata.input_cols; ++col) {
    multiplier_slots[col] = multiplier[col] % he_.plain_modulus;
    bias_slots[col] = bias[col] % he_.plain_modulus;
    nonzero_multiplier = nonzero_multiplier || multiplier_slots[col] != 0;
  }
  seal::Plaintext multiplier_plaintext, bias_plaintext;
  he_.encoder->encode(repeated_slots(multiplier_slots, metadata.packed_stride),
                      multiplier_plaintext);
  he_.encoder->encode(repeated_slots(bias_slots, metadata.packed_stride),
                      bias_plaintext);

  BoltCipherTensor result;
  result.metadata = input.metadata;
  result.metadata.cols = metadata.output_cols;
  result.ciphertexts.resize(metadata.input_rows);
  for (std::size_t row = 0; row < metadata.input_rows; ++row) {
    if (nonzero_multiplier) {
      he_.evaluator->multiply_plain(input.ciphertexts[row],
                                    multiplier_plaintext,
                                    result.ciphertexts[row]);
      increment(&BoltOperationCounts::multiply_plain, counts);
    } else {
      // SEAL deliberately rejects transparent evaluator outputs.  A fresh
      // encryption of zero is the semantically secure equivalent and keeps
      // the wire shape identical for a zero LayerNorm gamma fixture.
      he_.encryptor->encrypt_zero(result.ciphertexts[row]);
      increment(&BoltOperationCounts::ciphertexts, counts);
    }
    he_.evaluator->add_plain_inplace(result.ciphertexts[row], bias_plaintext);
  }
  increment(&BoltOperationCounts::ciphertexts, counts,
            result.ciphertexts.size());
  harden_outputs(result, metadata, counts);
  return result;
}

BoltCipherTensor BoltLinear::cipher_cipher_matmul_cross_impl(
    const BoltCipherTensor &lhs, const BoltCipherTensor &rhs,
    const BoltMatMulMetadata &metadata, BoltOperationCounts *counts) {
  he_.require_public_evaluation(true);
  metadata.validate(he_.encoder->slot_count());
  lhs.validate(he_.encoder->slot_count(), true);
  rhs.validate(he_.encoder->slot_count(), true);
  if (he_.party != ALICE) {
    throw std::logic_error("BOLT HE evaluation must execute on ALICE");
  }
  if (lhs.metadata.rows != metadata.input_rows ||
      lhs.metadata.cols != metadata.input_cols ||
      rhs.metadata.rows != metadata.input_cols ||
      rhs.metadata.cols != metadata.output_cols ||
      lhs.metadata.packed_stride != metadata.packed_stride ||
      rhs.metadata.packed_stride != metadata.packed_stride) {
    throw std::invalid_argument("BOLT ct-ct cross MatMul metadata mismatch");
  }

  const std::size_t stride = metadata.packed_stride;
  std::vector<std::uint64_t> logical_mask(stride, 0);
  logical_mask[0] = 1;
  seal::Plaintext first_slot_mask;
  he_.encoder->encode(repeated_slots(logical_mask, stride), first_slot_mask);

  BoltCipherTensor result;
  result.metadata = output_tensor_metadata(metadata);
  result.metadata.share_bit_width = lhs.metadata.share_bit_width;
  result.metadata.scale = lhs.metadata.scale;
  result.ciphertexts.resize(metadata.input_rows);

  for (std::size_t row = 0; row < metadata.input_rows; ++row) {
    bool output_initialized = false;
    seal::Ciphertext output;
    for (std::size_t inner = 0; inner < metadata.input_cols; ++inner) {
      if (lhs.ciphertexts[row].parms_id() != rhs.ciphertexts[inner].parms_id()) {
        throw std::logic_error(
            "BOLT fail-closed: ct-ct operands must be at the same modulus level");
      }
      seal::Ciphertext scalar;
      if (inner == 0) {
        scalar = lhs.ciphertexts[row];
      } else {
        he_.rotate_rows_logical(lhs.ciphertexts[row],
                                static_cast<int>(inner), scalar);
        increment(&BoltOperationCounts::rotations, counts);
      }
      he_.evaluator->multiply_plain_inplace(scalar, first_slot_mask);
      increment(&BoltOperationCounts::multiply_plain, counts);

      for (std::size_t shift = 1; shift < stride; shift <<= 1) {
        seal::Ciphertext rotated;
        he_.rotate_rows_logical(scalar, -static_cast<int>(shift), rotated);
        he_.evaluator->add_inplace(scalar, rotated);
        increment(&BoltOperationCounts::rotations, counts);
      }

      seal::Ciphertext product;
      he_.evaluator->multiply(scalar, rhs.ciphertexts[inner], product);
      increment(&BoltOperationCounts::evaluator_multiply, counts);
      he_.evaluator->relinearize_inplace(product,
                                         *he_.relinearization_keys);
      increment(&BoltOperationCounts::relinearizations, counts);
      if (!output_initialized) {
        output = std::move(product);
        output_initialized = true;
      } else {
        he_.evaluator->add_inplace(output, product);
      }
    }
    result.ciphertexts[row] = std::move(output);
  }
  increment(&BoltOperationCounts::ciphertexts, counts,
            result.ciphertexts.size());
  harden_outputs(result, metadata, counts);
  return result;
}

BoltCipherTensor BoltLinear::cipher_cipher_matmul_cross(
    const BoltCipherTensor &lhs, const BoltCipherTensor &rhs,
    const BoltMatMulMetadata &metadata, BoltOperationCounts *counts) {
  fail_if_generic_ot_matmul(false);
  return cipher_cipher_matmul_cross_impl(lhs, rhs, metadata, counts);
}

BoltCipherTensor BoltLinear::softmax_v_matmul(
    const BoltCipherTensor &softmax, const BoltCipherTensor &value,
    const BoltMatMulMetadata &metadata, BoltOperationCounts *counts) {
  fail_if_generic_ot_matmul(false);
  return cipher_cipher_matmul_cross_impl(softmax, value, metadata, counts);
}

void BoltLinear::harden_after_fusion(
    BoltCipherTensor &tensor, const BoltMatMulMetadata &metadata,
    BoltOperationCounts *counts) {
  he_.require_public_evaluation(false);
  tensor.validate(he_.encoder->slot_count(), true);
  metadata.validate(he_.encoder->slot_count());
  harden_outputs(tensor, metadata, counts);
}

std::pair<seal::Ciphertext, seal::Ciphertext>
BoltLinear::bert_rotation_pair(const seal::Ciphertext &input,
                               std::size_t image_size, std::size_t offset,
                               BoltOperationCounts *counts) {
  seal::Ciphertext left;
  seal::Ciphertext right;
  he_.rotate_rows_logical(input, static_cast<int>(offset), left);
  he_.rotate_rows_logical(input,
                          -static_cast<int>(image_size - offset), right);
  increment(&BoltOperationCounts::rotations, counts, 2);
  return {std::move(left), std::move(right)};
}

void BoltLinear::bert_cipher_plain_matmul_bsgs_compat(
    const std::vector<seal::Ciphertext> &input,
    const std::vector<std::vector<seal::Plaintext>> &encoded_left,
    const std::vector<std::vector<seal::Plaintext>> &encoded_right,
    const std::vector<seal::Plaintext> &encoded_bias,
    const BoltBertPackingMetadata &metadata,
    std::vector<seal::Ciphertext> &result, BoltOperationCounts *counts) {
  he_.require_public_evaluation(false);
  metadata.validate(he_.encoder->slot_count());
  if (he_.party != ALICE)
    throw std::logic_error("BOLT BERT HE evaluation must execute on ALICE");

  const std::size_t output_ciphertexts =
      metadata.image_size * metadata.filter_width / metadata.slot_count;
  const std::size_t num_diag =
      metadata.slot_count / metadata.image_size / 2;
  if (input.empty() || result.size() != output_ciphertexts ||
      encoded_bias.size() != output_ciphertexts ||
      encoded_left.size() < metadata.baby_step * metadata.giant_steps ||
      encoded_right.size() < metadata.baby_step * metadata.giant_steps) {
    throw std::invalid_argument("BOLT BERT ct-pt packed tensor shape mismatch");
  }

  std::vector<std::vector<seal::Ciphertext>> rotated(
      input.size(), std::vector<seal::Ciphertext>(2 * metadata.baby_step));
#pragma omp parallel for collapse(2) num_threads(32)
  for (std::size_t j = 0; j < metadata.baby_step; ++j) {
    for (std::size_t ct = 0; ct < input.size(); ++ct) {
      if (j == 0) {
        rotated[ct][j] = input[ct];
      } else {
        he_.rotate_rows_logical(
            input[ct], static_cast<int>((num_diag - j) * metadata.image_size),
            rotated[ct][j]);
      }
      he_.evaluator->rotate_columns(rotated[ct][j], *he_.galois_keys,
                                    rotated[ct][j + metadata.baby_step]);
    }
  }
  increment(&BoltOperationCounts::rotations, counts,
            input.size() * (2 * metadata.baby_step - 1));

  std::vector<std::vector<seal::Ciphertext>> giant_results(
      output_ciphertexts,
      std::vector<seal::Ciphertext>(metadata.giant_steps));
#pragma omp parallel for collapse(2) schedule(dynamic) num_threads(32)
  for (std::size_t giant = 0; giant < metadata.giant_steps; ++giant) {
    for (std::size_t out_ct = 0; out_ct < output_ciphertexts; ++out_ct) {
      for (std::size_t ct = 0; ct < input.size(); ++ct) {
        seal::Ciphertext partial;
        bool initialized = false;
        for (std::size_t baby = 0; baby < metadata.baby_step; ++baby) {
          const std::size_t diagonal = metadata.baby_step * giant + baby +
                                       out_ct * num_diag;
          if (diagonal >= encoded_left.size() ||
              ct >= encoded_left[diagonal].size() ||
              ct >= encoded_right[diagonal].size()) {
            throw std::invalid_argument("BOLT BERT encoded diagonal shape mismatch");
          }
          seal::Ciphertext left_product;
          seal::Ciphertext right_product;
          he_.evaluator->multiply_plain(rotated[ct][baby],
                                        encoded_left[diagonal][ct],
                                        left_product);
          he_.evaluator->multiply_plain(
                rotated[ct][baby + metadata.baby_step],
                encoded_right[diagonal][ct], right_product);
          he_.evaluator->add_inplace(left_product, right_product);
          if (!initialized) {
            partial = std::move(left_product);
            initialized = true;
          } else {
            he_.evaluator->add_inplace(partial, left_product);
          }
        }
        if (ct == 0) {
          giant_results[out_ct][giant] = std::move(partial);
        } else {
          he_.evaluator->add_inplace(giant_results[out_ct][giant], partial);
        }
      }
    }
  }
  increment(&BoltOperationCounts::multiply_plain, counts,
            output_ciphertexts * metadata.giant_steps * input.size() *
                metadata.baby_step * 2);

#pragma omp parallel for num_threads(32)
  for (std::size_t out_ct = 0; out_ct < output_ciphertexts; ++out_ct) {
    result[out_ct] = giant_results[out_ct][0];
    for (std::size_t giant = 1; giant < metadata.giant_steps; ++giant) {
      seal::Ciphertext rotated_giant;
      he_.rotate_rows_logical(
          giant_results[out_ct][giant],
          -static_cast<int>(metadata.baby_step * giant * metadata.image_size),
          rotated_giant);
      he_.evaluator->add_inplace(result[out_ct], rotated_giant);
    }
    he_.evaluator->add_plain_inplace(result[out_ct], encoded_bias[out_ct]);
  }
  increment(&BoltOperationCounts::rotations, counts,
            output_ciphertexts * (metadata.giant_steps - 1));

  increment(&BoltOperationCounts::ciphertexts, counts, result.size());
  for (auto &ciphertext : result) {
    if (metadata.noise_flooding_bits != 0) {
      auto context_data = he_.context->get_context_data(ciphertext.parms_id());
      flood_ciphertext(ciphertext, context_data,
                       metadata.noise_flooding_bits);
      increment(&BoltOperationCounts::noise_floodings, counts);
    }
    for (std::uint32_t level = 0; level < metadata.modulus_switches; ++level) {
      he_.evaluator->mod_switch_to_next_inplace(ciphertext);
      increment(&BoltOperationCounts::modulus_switches, counts);
    }
  }
}

void BoltLinear::bert_qkv_cipher_plain_matmul_bsgs_compat(
    const std::vector<seal::Ciphertext> &input,
    const std::vector<std::pair<std::vector<std::vector<seal::Plaintext>>,
                                std::vector<std::vector<seal::Plaintext>>>> &q,
    const std::vector<std::pair<std::vector<std::vector<seal::Plaintext>>,
                                std::vector<std::vector<seal::Plaintext>>>> &k,
    const std::vector<std::pair<std::vector<std::vector<seal::Plaintext>>,
                                std::vector<std::vector<seal::Plaintext>>>> &v,
    const std::vector<std::vector<seal::Plaintext>> &q_bias,
    const std::vector<std::vector<seal::Plaintext>> &k_bias,
    const std::vector<std::vector<seal::Plaintext>> &v_bias,
    const BoltBertPackingMetadata &metadata,
    std::vector<seal::Ciphertext> &result, BoltOperationCounts *counts) {
  he_.require_public_evaluation(false);
  metadata.validate(he_.encoder->slot_count());
  if (he_.party != ALICE)
    throw std::logic_error("BOLT BERT HE evaluation must execute on ALICE");
  const std::size_t num_diag =
      metadata.slot_count / metadata.image_size / 2;
  const std::size_t packed_per_group =
      metadata.image_size * metadata.filter_width * 2 / metadata.slot_count;
  const std::size_t one_projection = metadata.image_size *
      metadata.filter_width * metadata.attention_heads / metadata.slot_count;
  const std::size_t expected_output = 3 * one_projection;
  if (input.empty() || q.size() != metadata.paired_head_groups ||
      k.size() != metadata.paired_head_groups ||
      v.size() != metadata.paired_head_groups ||
      q_bias.size() != metadata.paired_head_groups ||
      k_bias.size() != metadata.paired_head_groups ||
      v_bias.size() != metadata.paired_head_groups ||
      result.size() != expected_output) {
    throw std::invalid_argument("BOLT BERT fused QKV packing shape mismatch");
  }

  std::vector<std::vector<seal::Ciphertext>> rotated(
      input.size(), std::vector<seal::Ciphertext>(2 * metadata.baby_step));
  for (std::size_t baby = 0; baby < metadata.baby_step; ++baby) {
    for (std::size_t ct = 0; ct < input.size(); ++ct) {
      if (baby == 0) {
        rotated[ct][baby] = input[ct];
      } else {
        he_.rotate_rows_logical(
            input[ct],
            static_cast<int>((num_diag - baby) * metadata.image_size),
            rotated[ct][baby]);
        increment(&BoltOperationCounts::rotations, counts);
      }
      he_.evaluator->rotate_columns(rotated[ct][baby], *he_.galois_keys,
                                    rotated[ct][baby + metadata.baby_step]);
      increment(&BoltOperationCounts::rotations, counts);
    }
  }

  for (std::size_t group = 0; group < metadata.paired_head_groups; ++group) {
    const auto &q_left = q[group].first;
    const auto &q_right = q[group].second;
    const auto &k_left = k[group].first;
    const auto &k_right = k[group].second;
    const auto &v_left = v[group].first;
    const auto &v_right = v[group].second;
    std::vector<std::vector<seal::Ciphertext>> q_giants(
        packed_per_group,
        std::vector<seal::Ciphertext>(metadata.giant_steps));
    std::vector<std::vector<seal::Ciphertext>> k_giants = q_giants;
    std::vector<std::vector<seal::Ciphertext>> v_giants = q_giants;

    for (std::size_t giant = 0; giant < metadata.giant_steps; ++giant) {
      for (std::size_t ct = 0; ct < input.size(); ++ct) {
        for (std::size_t packed = 0; packed < packed_per_group; ++packed) {
          seal::Ciphertext q_sum, k_sum, v_sum;
          bool initialized = false;
          for (std::size_t baby = 0; baby < metadata.baby_step; ++baby) {
            const std::size_t diagonal = metadata.baby_step * giant + baby +
                                         packed * num_diag;
            if (diagonal >= q_left.size() || ct >= q_left[diagonal].size() ||
                diagonal >= q_right.size() || ct >= q_right[diagonal].size() ||
                diagonal >= k_left.size() || ct >= k_left[diagonal].size() ||
                diagonal >= k_right.size() || ct >= k_right[diagonal].size() ||
                diagonal >= v_left.size() || ct >= v_left[diagonal].size() ||
                diagonal >= v_right.size() || ct >= v_right[diagonal].size()) {
              throw std::invalid_argument("BOLT BERT fused QKV diagonal mismatch");
            }
            seal::Ciphertext ql, qr, kl, kr, vl, vr;
            he_.evaluator->multiply_plain(rotated[ct][baby],
                                          q_left[diagonal][ct], ql);
            he_.evaluator->multiply_plain(
                rotated[ct][baby + metadata.baby_step],
                q_right[diagonal][ct], qr);
            he_.evaluator->multiply_plain(rotated[ct][baby],
                                          k_left[diagonal][ct], kl);
            he_.evaluator->multiply_plain(
                rotated[ct][baby + metadata.baby_step],
                k_right[diagonal][ct], kr);
            he_.evaluator->multiply_plain(rotated[ct][baby],
                                          v_left[diagonal][ct], vl);
            he_.evaluator->multiply_plain(
                rotated[ct][baby + metadata.baby_step],
                v_right[diagonal][ct], vr);
            increment(&BoltOperationCounts::multiply_plain, counts, 6);
            he_.evaluator->add_inplace(ql, qr);
            he_.evaluator->add_inplace(kl, kr);
            he_.evaluator->add_inplace(vl, vr);
            if (!initialized) {
              q_sum = std::move(ql);
              k_sum = std::move(kl);
              v_sum = std::move(vl);
              initialized = true;
            } else {
              he_.evaluator->add_inplace(q_sum, ql);
              he_.evaluator->add_inplace(k_sum, kl);
              he_.evaluator->add_inplace(v_sum, vl);
            }
          }
          if (ct == 0) {
            q_giants[packed][giant] = std::move(q_sum);
            k_giants[packed][giant] = std::move(k_sum);
            v_giants[packed][giant] = std::move(v_sum);
          } else {
            he_.evaluator->add_inplace(q_giants[packed][giant], q_sum);
            he_.evaluator->add_inplace(k_giants[packed][giant], k_sum);
            he_.evaluator->add_inplace(v_giants[packed][giant], v_sum);
          }
        }
      }
    }

    for (std::size_t packed = 0; packed < packed_per_group; ++packed) {
      seal::Ciphertext q_out = q_giants[packed][0];
      seal::Ciphertext k_out = k_giants[packed][0];
      seal::Ciphertext v_out = v_giants[packed][0];
      for (std::size_t giant = 1; giant < metadata.giant_steps; ++giant) {
        const int rotation = -static_cast<int>(
            metadata.baby_step * giant * metadata.image_size);
        seal::Ciphertext rotated_giant;
        he_.rotate_rows_logical(q_giants[packed][giant], rotation,
                                rotated_giant);
        he_.evaluator->add_inplace(q_out, rotated_giant);
        he_.rotate_rows_logical(k_giants[packed][giant], rotation,
                                rotated_giant);
        he_.evaluator->add_inplace(k_out, rotated_giant);
        he_.rotate_rows_logical(v_giants[packed][giant], rotation,
                                rotated_giant);
        he_.evaluator->add_inplace(v_out, rotated_giant);
        increment(&BoltOperationCounts::rotations, counts, 3);
      }
      he_.evaluator->add_plain_inplace(q_out, q_bias[group][packed]);
      he_.evaluator->add_plain_inplace(k_out, k_bias[group][packed]);
      he_.evaluator->add_plain_inplace(v_out, v_bias[group][packed]);
      result[packed + group * packed_per_group] = std::move(q_out);
      result[packed + group * packed_per_group + one_projection] =
          std::move(k_out);
      result[packed + group * packed_per_group + 2 * one_projection] =
          std::move(v_out);
    }
  }
  increment(&BoltOperationCounts::ciphertexts, counts, result.size());
}

void BoltLinear::bert_qk_matmul_cross_compat(
    const std::vector<seal::Ciphertext> &qkv,
    const std::vector<seal::Plaintext> &cross_masks,
    const BoltBertPackingMetadata &metadata,
    std::vector<seal::Ciphertext> &result, BoltOperationCounts *counts) {
  he_.require_public_evaluation(true);
  metadata.validate(he_.encoder->slot_count());
  if (he_.party != ALICE)
    throw std::logic_error("BOLT BERT HE evaluation must execute on ALICE");
  const std::size_t temp_result_size =
      metadata.image_size * metadata.filter_width * 2 / metadata.slot_count;
  const std::size_t num_cts_per_result =
      metadata.image_size * metadata.image_size * 2 / metadata.slot_count;
  const std::size_t columns_per_ciphertext =
      metadata.slot_count / 2 / metadata.image_size;
  const std::size_t qk_block = metadata.image_size * metadata.filter_width *
                               metadata.attention_heads / metadata.slot_count;
  // Linear::linear_1 passes its combined QK-and-V output buffer here.  The
  // QK kernel owns only the prefix; the caller appends the retained V
  // ciphertexts afterwards.  Require enough prefix storage without rejecting
  // that protocol-preserving V tail.
  if (cross_masks.size() < 2 * metadata.image_size ||
      result.size() < metadata.paired_head_groups * num_cts_per_result ||
      qkv.size() < 2 * qk_block) {
    throw std::invalid_argument("BOLT BERT QK cross-packing shape mismatch");
  }

#pragma omp parallel for schedule(static) num_threads(6)
  for (std::size_t group = 0; group < metadata.paired_head_groups; ++group) {
    std::vector<seal::Ciphertext> products(2 * metadata.image_size);
    for (std::size_t packed = 0; packed < temp_result_size; ++packed) {
      const seal::Ciphertext &q =
          qkv[packed + group * temp_result_size];
      const seal::Ciphertext &k =
          qkv[packed + group * temp_result_size + qk_block];
      for (std::size_t offset = 0; offset < metadata.image_size; ++offset) {
        auto rotated = bert_rotation_pair(k, metadata.image_size, offset,
                                          nullptr);
        seal::Ciphertext first_product;
        seal::Ciphertext second_product;
        he_.evaluator->multiply(q, rotated.first, first_product);
        he_.evaluator->multiply(q, rotated.second, second_product);
        if (packed == 0) {
          products[offset] = std::move(first_product);
          products[offset + metadata.image_size] =
              std::move(second_product);
        } else {
          he_.evaluator->add_inplace(products[offset], first_product);
          he_.evaluator->add_inplace(
              products[offset + metadata.image_size], second_product);
        }
      }
    }
    for (std::size_t index = 0; index < products.size(); ++index) {
      he_.evaluator->relinearize_inplace(products[index],
                                         *he_.relinearization_keys);
      for (std::size_t step = 1; step < columns_per_ciphertext; step <<= 1) {
        seal::Ciphertext rotated;
        he_.rotate_rows_logical(
            products[index], static_cast<int>(step * metadata.image_size),
            rotated);
        he_.evaluator->add_inplace(products[index], rotated);
      }
      he_.evaluator->multiply_plain_inplace(products[index],
                                            cross_masks[index]);
    }
    for (std::size_t out_ct = 0; out_ct < num_cts_per_result; ++out_ct) {
      const std::size_t base = columns_per_ciphertext * out_ct;
      he_.evaluator->add(products[base],
                         products[base + metadata.image_size],
                         result[group * num_cts_per_result + out_ct]);
      for (std::size_t col = 1; col < columns_per_ciphertext; ++col) {
        he_.evaluator->add_inplace(
            result[group * num_cts_per_result + out_ct],
            products[base + col]);
        he_.evaluator->add_inplace(
            result[group * num_cts_per_result + out_ct],
            products[base + col + metadata.image_size]);
      }
    }
  }
  std::size_t reduction_rotation_steps = 0;
  for (std::size_t step = 1; step < columns_per_ciphertext; step <<= 1)
    ++reduction_rotation_steps;
  increment(&BoltOperationCounts::rotations, counts,
            metadata.paired_head_groups *
                (temp_result_size * metadata.image_size * 2 +
                 2 * metadata.image_size *
                     reduction_rotation_steps));
  increment(&BoltOperationCounts::evaluator_multiply, counts,
            metadata.paired_head_groups * temp_result_size *
                metadata.image_size * 2);
  increment(&BoltOperationCounts::relinearizations, counts,
            metadata.paired_head_groups * 2 * metadata.image_size);
  increment(&BoltOperationCounts::multiply_plain, counts,
            metadata.paired_head_groups * 2 * metadata.image_size);
  increment(&BoltOperationCounts::ciphertexts, counts,
            metadata.paired_head_groups * num_cts_per_result);
}

void BoltLinear::bert_softmax_v_matmul_compat(
    const std::vector<std::vector<std::vector<seal::Ciphertext>>> &softmax,
    const std::vector<seal::Ciphertext> &value,
    const BoltBertPackingMetadata &metadata,
    std::vector<seal::Ciphertext> &result, BoltOperationCounts *counts) {
  he_.require_public_evaluation(true);
  metadata.validate(he_.encoder->slot_count());
  if (he_.party != ALICE)
    throw std::logic_error("BOLT BERT HE evaluation must execute on ALICE");
  const std::size_t matrices_per_column =
      metadata.filter_width /
      (metadata.slot_count / metadata.image_size / 2);
  const std::size_t output_ciphertexts = metadata.paired_head_groups *
      metadata.image_size * metadata.filter_width * 2 / metadata.slot_count;
  if (softmax.size() != metadata.paired_head_groups ||
      value.size() < metadata.paired_head_groups * matrices_per_column ||
      result.size() != output_ciphertexts) {
    throw std::invalid_argument("BOLT BERT Softmax-V packing shape mismatch");
  }

  for (std::size_t group = 0; group < metadata.paired_head_groups; ++group) {
    if (softmax[group].size() != 2 ||
        softmax[group][0].size() < metadata.image_size ||
        softmax[group][1].size() < metadata.image_size) {
      throw std::invalid_argument("BOLT BERT Softmax diagonal packing mismatch");
    }
  }
#pragma omp parallel for collapse(2) num_threads(24)
  for (std::size_t group = 0; group < metadata.paired_head_groups; ++group) {
    for (std::size_t ct = 0; ct < matrices_per_column; ++ct) {
      seal::Ciphertext output;
      bool initialized = false;
      for (std::size_t diagonal = 0; diagonal < metadata.image_size;
           ++diagonal) {
        auto rotated = bert_rotation_pair(
            value[group * matrices_per_column + ct], metadata.image_size,
            diagonal, nullptr);
        seal::Ciphertext left;
        seal::Ciphertext right;
        he_.evaluator->multiply(rotated.first,
                                softmax[group][0][diagonal], left);
        he_.evaluator->multiply(rotated.second,
                                softmax[group][1][diagonal], right);
        he_.evaluator->add_inplace(left, right);
        if (!initialized) {
          output = std::move(left);
          initialized = true;
        } else {
          he_.evaluator->add_inplace(output, left);
        }
      }
      result[group * metadata.image_size * metadata.filter_width * 2 /
                 metadata.slot_count +
             ct] = std::move(output);
    }
  }
  increment(&BoltOperationCounts::rotations, counts,
            output_ciphertexts * metadata.image_size * 2);
  increment(&BoltOperationCounts::evaluator_multiply, counts,
            output_ciphertexts * metadata.image_size * 2);
#pragma omp parallel for num_threads(24)
  for (std::size_t index = 0; index < result.size(); ++index) {
    auto &ciphertext = result[index];
    he_.evaluator->relinearize_inplace(ciphertext,
                                       *he_.relinearization_keys);
    for (std::uint32_t level = 0; level < metadata.modulus_switches; ++level) {
      he_.evaluator->mod_switch_to_next_inplace(ciphertext);
    }
  }
  increment(&BoltOperationCounts::relinearizations, counts, result.size());
  increment(&BoltOperationCounts::modulus_switches, counts,
            result.size() * metadata.modulus_switches);
  increment(&BoltOperationCounts::ciphertexts, counts, result.size());
}

} // namespace bolt
} // namespace sci
