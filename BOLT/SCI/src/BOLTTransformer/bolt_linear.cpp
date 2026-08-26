#include "BOLTTransformer/bolt_linear.h"

#include "LinearHE/utils-HE.h"

#include <algorithm>
#include <cmath>
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

void finish_gpu_outputs(BoltEvaluator *evaluator,
                        std::vector<seal::Ciphertext> &outputs) {
  if (evaluator == nullptr || !evaluator->gpu_enabled()) {
    return;
  }
  // Intermediate ciphertext objects are intentionally short-lived.  Sync
  // only the public outputs, then release all remaining device temporaries so
  // their host-data-pointer cache cannot outlive the SEAL object.
  for (auto &ciphertext : outputs) {
    evaluator->synchronize(ciphertext);
  }
  evaluator->release_gpu_residency();
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
      he_.evaluator->synchronize(ciphertext);
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
    he_.evaluator->copy_ciphertext(input.ciphertexts[row], baby_rotations[0]);
    for (std::size_t baby = 1; baby < baby_step; ++baby) {
      he_.evaluator->rotate_rows(input.ciphertexts[row],
                                 static_cast<int>(baby), *he_.galois_keys,
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
        he_.evaluator->rotate_rows(group, static_cast<int>(giant_offset),
                                   *he_.galois_keys, rotated);
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
  finish_gpu_outputs(he_.evaluator, result.ciphertexts);
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
        he_.evaluator->copy_ciphertext(lhs.ciphertexts[row], scalar);
      } else {
        he_.evaluator->rotate_rows(lhs.ciphertexts[row],
                                   static_cast<int>(inner), *he_.galois_keys,
                                   scalar);
        increment(&BoltOperationCounts::rotations, counts);
      }
      he_.evaluator->multiply_plain_inplace(scalar, first_slot_mask);
      increment(&BoltOperationCounts::multiply_plain, counts);

      for (std::size_t shift = 1; shift < stride; shift <<= 1) {
        seal::Ciphertext rotated;
        he_.evaluator->rotate_rows(scalar, -static_cast<int>(shift),
                                   *he_.galois_keys, rotated);
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
  finish_gpu_outputs(he_.evaluator, result.ciphertexts);
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

std::pair<seal::Ciphertext, seal::Ciphertext>
BoltLinear::bert_rotation_pair(const seal::Ciphertext &input,
                               std::size_t image_size, std::size_t offset,
                               BoltOperationCounts *counts) {
  seal::Ciphertext left;
  seal::Ciphertext right;
  he_.evaluator->rotate_rows(input, static_cast<int>(offset),
                             *he_.galois_keys, left);
  he_.evaluator->rotate_rows(
      input, -static_cast<int>(image_size - offset), *he_.galois_keys, right);
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
  for (std::size_t j = 0; j < metadata.baby_step; ++j) {
    for (std::size_t ct = 0; ct < input.size(); ++ct) {
      if (j == 0) {
        he_.evaluator->copy_ciphertext(input[ct], rotated[ct][j]);
      } else {
        he_.evaluator->rotate_rows(
            input[ct], static_cast<int>((num_diag - j) * metadata.image_size),
            *he_.galois_keys, rotated[ct][j]);
        increment(&BoltOperationCounts::rotations, counts);
      }
      he_.evaluator->rotate_columns(rotated[ct][j], *he_.galois_keys,
                                    rotated[ct][j + metadata.baby_step]);
      increment(&BoltOperationCounts::rotations, counts);
    }
  }

  std::vector<std::vector<seal::Ciphertext>> giant_results(
      output_ciphertexts,
      std::vector<seal::Ciphertext>(metadata.giant_steps));
  for (std::size_t giant = 0; giant < metadata.giant_steps; ++giant) {
    for (std::size_t ct = 0; ct < input.size(); ++ct) {
      for (std::size_t out_ct = 0; out_ct < output_ciphertexts; ++out_ct) {
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
          increment(&BoltOperationCounts::multiply_plain, counts, 2);
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

  for (std::size_t out_ct = 0; out_ct < output_ciphertexts; ++out_ct) {
    he_.evaluator->copy_ciphertext(giant_results[out_ct][0], result[out_ct]);
    for (std::size_t giant = 1; giant < metadata.giant_steps; ++giant) {
      seal::Ciphertext rotated_giant;
      he_.evaluator->rotate_rows(
          giant_results[out_ct][giant],
          -static_cast<int>(metadata.baby_step * giant * metadata.image_size),
          *he_.galois_keys, rotated_giant);
      increment(&BoltOperationCounts::rotations, counts);
      he_.evaluator->add_inplace(result[out_ct], rotated_giant);
    }
    he_.evaluator->add_plain_inplace(result[out_ct], encoded_bias[out_ct]);
  }

  increment(&BoltOperationCounts::ciphertexts, counts, result.size());
  for (auto &ciphertext : result) {
    if (metadata.noise_flooding_bits != 0) {
      he_.evaluator->synchronize(ciphertext);
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
  finish_gpu_outputs(he_.evaluator, result);
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
        he_.evaluator->copy_ciphertext(input[ct], rotated[ct][baby]);
      } else {
        he_.evaluator->rotate_rows(
            input[ct],
            static_cast<int>((num_diag - baby) * metadata.image_size),
            *he_.galois_keys, rotated[ct][baby]);
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
      seal::Ciphertext q_out;
      seal::Ciphertext k_out;
      seal::Ciphertext v_out;
      he_.evaluator->copy_ciphertext(q_giants[packed][0], q_out);
      he_.evaluator->copy_ciphertext(k_giants[packed][0], k_out);
      he_.evaluator->copy_ciphertext(v_giants[packed][0], v_out);
      for (std::size_t giant = 1; giant < metadata.giant_steps; ++giant) {
        const int rotation = -static_cast<int>(
            metadata.baby_step * giant * metadata.image_size);
        seal::Ciphertext rotated_giant;
        he_.evaluator->rotate_rows(q_giants[packed][giant], rotation,
                                   *he_.galois_keys, rotated_giant);
        he_.evaluator->add_inplace(q_out, rotated_giant);
        he_.evaluator->rotate_rows(k_giants[packed][giant], rotation,
                                   *he_.galois_keys, rotated_giant);
        he_.evaluator->add_inplace(k_out, rotated_giant);
        he_.evaluator->rotate_rows(v_giants[packed][giant], rotation,
                                   *he_.galois_keys, rotated_giant);
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
  finish_gpu_outputs(he_.evaluator, result);
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

  for (std::size_t group = 0; group < metadata.paired_head_groups; ++group) {
    std::vector<seal::Ciphertext> products(2 * metadata.image_size);
    for (std::size_t packed = 0; packed < temp_result_size; ++packed) {
      const seal::Ciphertext &q =
          qkv[packed + group * temp_result_size];
      const seal::Ciphertext &k =
          qkv[packed + group * temp_result_size + qk_block];
      for (std::size_t offset = 0; offset < metadata.image_size; ++offset) {
        auto rotated = bert_rotation_pair(k, metadata.image_size, offset,
                                          counts);
        seal::Ciphertext first_product;
        seal::Ciphertext second_product;
        he_.evaluator->multiply(q, rotated.first, first_product);
        he_.evaluator->multiply(q, rotated.second, second_product);
        increment(&BoltOperationCounts::evaluator_multiply, counts, 2);
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
      increment(&BoltOperationCounts::relinearizations, counts);
      for (std::size_t step = 1; step < columns_per_ciphertext; step <<= 1) {
        seal::Ciphertext rotated;
        he_.evaluator->rotate_rows(
            products[index], static_cast<int>(step * metadata.image_size),
            *he_.galois_keys, rotated);
        he_.evaluator->add_inplace(products[index], rotated);
        increment(&BoltOperationCounts::rotations, counts);
      }
      he_.evaluator->multiply_plain_inplace(products[index],
                                            cross_masks[index]);
      increment(&BoltOperationCounts::multiply_plain, counts);
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
  increment(&BoltOperationCounts::ciphertexts, counts,
            metadata.paired_head_groups * num_cts_per_result);
  finish_gpu_outputs(he_.evaluator, result);
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
    for (std::size_t ct = 0; ct < matrices_per_column; ++ct) {
      seal::Ciphertext output;
      bool initialized = false;
      for (std::size_t diagonal = 0; diagonal < metadata.image_size;
           ++diagonal) {
        auto rotated = bert_rotation_pair(
            value[group * matrices_per_column + ct], metadata.image_size,
            diagonal, counts);
        seal::Ciphertext left;
        seal::Ciphertext right;
        he_.evaluator->multiply(rotated.first,
                                softmax[group][0][diagonal], left);
        he_.evaluator->multiply(rotated.second,
                                softmax[group][1][diagonal], right);
        increment(&BoltOperationCounts::evaluator_multiply, counts, 2);
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
  for (auto &ciphertext : result) {
    he_.evaluator->relinearize_inplace(ciphertext,
                                       *he_.relinearization_keys);
    increment(&BoltOperationCounts::relinearizations, counts);
    for (std::uint32_t level = 0; level < metadata.modulus_switches; ++level) {
      he_.evaluator->mod_switch_to_next_inplace(ciphertext);
      increment(&BoltOperationCounts::modulus_switches, counts);
    }
  }
  increment(&BoltOperationCounts::ciphertexts, counts, result.size());
  finish_gpu_outputs(he_.evaluator, result);
}

std::pair<std::vector<std::vector<seal::Plaintext>>,
          std::vector<std::vector<seal::Plaintext>>>
BoltLinear::bert_pack_cipher_plain_matrix(
    const std::vector<std::vector<std::uint64_t>> &matrix,
    const BoltBertPackingMetadata &metadata) {
  he_.require_batching();
  metadata.validate(he_.encoder->slot_count());
  if (matrix.size() != metadata.filter_height ||
      metadata.filter_width == 0 || metadata.filter_width % 2 != 0 ||
      metadata.filter_height %
              (metadata.slot_count / metadata.image_size / 2) != 0 ||
      metadata.filter_width %
              (2 * (metadata.slot_count / metadata.image_size / 2)) != 0) {
    throw std::invalid_argument("BOLT ViT ct-pt matrix packing shape mismatch");
  }
  for (const auto &row : matrix) {
    if (row.size() != metadata.filter_width)
      throw std::invalid_argument("BOLT ViT ct-pt matrix row shape mismatch");
  }

  const std::size_t num_diag =
      metadata.slot_count / metadata.image_size / 2;
  const std::size_t half_width = metadata.filter_width / 2;
  const std::size_t half_slots = metadata.slot_count / 2;
  const std::size_t matrices_per_col = metadata.filter_width /
                                        (2 * num_diag);
  const std::size_t plaintexts_per_diagonal =
      metadata.filter_height * metadata.image_size / metadata.slot_count;

  std::vector<std::vector<seal::Plaintext>> left;
  std::vector<std::vector<seal::Plaintext>> right;

  auto pack_family = [&](bool second_family) {
    auto &destination = second_family ? right : left;
    for (std::size_t col = 0; col < matrices_per_col; ++col) {
      std::size_t matrix_flag = 0;
      for (std::size_t diagonal = 0; diagonal < num_diag; ++diagonal) {
        std::vector<std::uint64_t> slots;
        std::vector<seal::Plaintext> packed(plaintexts_per_diagonal);
        std::size_t packed_index = 0;
        for (std::size_t block = 0;
             block < metadata.filter_height / num_diag; ++block) {
          for (std::size_t lane = 0; lane < num_diag; ++lane) {
            for (std::size_t row = 0; row < metadata.image_size; ++row) {
              const std::size_t local_column =
                  (lane + diagonal) % num_diag + col * num_diag;
              const bool use_second_half = second_family
                  ? (matrix_flag == 0)
                  : (matrix_flag != 0);
              const std::size_t actual_column =
                  local_column + (use_second_half ? half_width : 0);
              slots.push_back(matrix[block * num_diag + lane][actual_column]);
            }
            if (slots.size() % half_slots == 0) {
              matrix_flag ^= 1U;
              const std::size_t offset = slots.size() - half_slots;
              // BOLT's original BERT cross-packing rotates with the same
              // shape-specific n1 used by its BSGS baby loop.  ViT recomputes
              // baby_step from the physical-token diagonal count, so the
              // period remains baby_step after changing the token shape.
              const std::size_t rotation =
                  (diagonal % metadata.baby_step) * metadata.image_size;
              std::rotate(slots.begin() + offset,
                          slots.end() - rotation, slots.end());
              if (slots.size() == metadata.slot_count) {
                if (second_family)
                  std::rotate(slots.begin(), slots.begin() + half_slots,
                              slots.end());
                seal::Plaintext plaintext;
                he_.encoder->encode(slots, plaintext);
                packed[packed_index++] = std::move(plaintext);
                slots.clear();
              }
            }
          }
        }
        if (!slots.empty() || packed_index != plaintexts_per_diagonal)
          throw std::logic_error("BOLT ViT ct-pt matrix packing did not fill slots");
        destination.push_back(std::move(packed));
      }
    }
  };

  // Keep this explicit duplicate-free family builder as the single source of
  // truth for both diagonal families.  It is the BERT packing loop with only
  // the dimensions and BSGS baby step supplied by metadata.
  pack_family(false);
  pack_family(true);
  return {std::move(left), std::move(right)};
}

std::vector<seal::Plaintext> BoltLinear::bert_pack_cipher_plain_bias(
    const std::vector<std::uint64_t> &bias,
    const BoltBertPackingMetadata &metadata) {
  he_.require_batching();
  metadata.validate(he_.encoder->slot_count());
  if (bias.size() != metadata.filter_width ||
      metadata.image_size == 0 ||
      metadata.image_size * metadata.filter_width % metadata.slot_count != 0)
    throw std::invalid_argument("BOLT ViT ct-pt bias packing shape mismatch");
  const std::size_t count = metadata.image_size * metadata.filter_width /
                            metadata.slot_count;
  const std::size_t half = metadata.slot_count / 2;
  std::vector<seal::Plaintext> result;
  result.reserve(count);
  std::size_t first = 0;
  std::size_t second = metadata.filter_width / 2;
  for (std::size_t packed = 0; packed < count; ++packed) {
    std::vector<std::uint64_t> slots(metadata.slot_count, 0);
    for (std::size_t row = 0; row < metadata.slot_count;
         row += metadata.image_size) {
      const bool upper = row < half;
      const std::size_t index = upper ? first++ : second++;
      if (index >= bias.size())
        throw std::logic_error("BOLT ViT bias packing overran output dimension");
      for (std::size_t offset = 0; offset < metadata.image_size; ++offset)
        slots[row + offset] = bias[index];
    }
    seal::Plaintext plaintext;
    he_.encoder->encode(slots, plaintext);
    result.push_back(std::move(plaintext));
  }
  if (first != metadata.filter_width / 2 || second != metadata.filter_width)
    throw std::logic_error("BOLT ViT bias packing did not consume bias");
  return result;
}

std::vector<std::pair<std::vector<std::vector<seal::Plaintext>>,
                      std::vector<std::vector<seal::Plaintext>>>>
BoltLinear::bert_pack_qkv_matrix(
    const std::vector<std::vector<std::vector<std::uint64_t>>> &weights,
    const BoltBertPackingMetadata &metadata) {
  he_.require_batching();
  metadata.validate(he_.encoder->slot_count());
  if (weights.size() != metadata.attention_heads ||
      metadata.attention_heads != 2 * metadata.paired_head_groups ||
      metadata.filter_width == 0 ||
      metadata.filter_height %
              (metadata.slot_count / metadata.image_size / 2) != 0 ||
      metadata.filter_width %
              (metadata.slot_count / metadata.image_size / 2) != 0)
    throw std::invalid_argument("BOLT ViT QKV packing shape mismatch");
  for (const auto &head : weights) {
    if (head.size() != metadata.filter_height)
      throw std::invalid_argument("BOLT ViT QKV input shape mismatch");
    for (const auto &row : head)
      if (row.size() != metadata.filter_width)
        throw std::invalid_argument("BOLT ViT QKV output shape mismatch");
  }

  const std::size_t num_diag =
      metadata.slot_count / metadata.image_size / 2;
  const std::size_t half_slots = metadata.slot_count / 2;
  const std::size_t matrices_per_col = metadata.filter_width / num_diag;
  const std::size_t plaintexts_per_diagonal =
      metadata.filter_height * metadata.image_size / metadata.slot_count;
  std::vector<std::pair<std::vector<std::vector<seal::Plaintext>>,
                        std::vector<std::vector<seal::Plaintext>>>> result(
      metadata.paired_head_groups);

  for (std::size_t group = 0; group < metadata.paired_head_groups; ++group) {
    for (int family = 0; family < 2; ++family) {
      auto &destination = family == 0 ? result[group].first : result[group].second;
      for (std::size_t col = 0; col < matrices_per_col; ++col) {
        std::size_t matrix_flag = 0;
        for (std::size_t diagonal = 0; diagonal < num_diag; ++diagonal) {
          std::vector<std::uint64_t> slots;
          std::vector<seal::Plaintext> packed(plaintexts_per_diagonal);
          std::size_t packed_index = 0;
          for (std::size_t block = 0;
               block < metadata.filter_height / num_diag; ++block) {
            for (std::size_t lane = 0; lane < num_diag; ++lane) {
              for (std::size_t row = 0; row < metadata.image_size; ++row) {
                const std::size_t local_column =
                    (lane + diagonal) % num_diag + col * num_diag;
                const std::size_t head =
                    group * 2 + ((family == 0) ? matrix_flag : (matrix_flag ^ 1U));
                slots.push_back(weights[head][block * num_diag + lane]
                                      [local_column]);
              }
              if (slots.size() % half_slots == 0) {
                matrix_flag ^= 1U;
                const std::size_t offset = slots.size() - half_slots;
                // This is the BERT fused-QKV packing schedule generalized to
                // ViT dimensions.  The rotation period is the same
                // shape-specific baby step as the BSGS inner loop.
                const std::size_t rotation =
                    (diagonal % metadata.baby_step) * metadata.image_size;
                std::rotate(slots.begin() + offset,
                            slots.end() - rotation, slots.end());
                if (slots.size() == metadata.slot_count) {
                  if (family == 1)
                    std::rotate(slots.begin(), slots.begin() + half_slots,
                                slots.end());
                  seal::Plaintext plaintext;
                  he_.encoder->encode(slots, plaintext);
                  packed[packed_index++] = std::move(plaintext);
                  slots.clear();
                }
              }
            }
          }
          if (!slots.empty() || packed_index != plaintexts_per_diagonal)
            throw std::logic_error("BOLT ViT QKV packing did not fill slots");
          destination.push_back(std::move(packed));
        }
      }
    }
  }
  return result;
}

std::vector<std::vector<seal::Plaintext>> BoltLinear::bert_pack_qkv_bias(
    const std::vector<std::vector<std::uint64_t>> &bias,
    const BoltBertPackingMetadata &metadata) {
  he_.require_batching();
  metadata.validate(he_.encoder->slot_count());
  if (bias.size() != metadata.attention_heads ||
      metadata.attention_heads != 2 * metadata.paired_head_groups)
    throw std::invalid_argument("BOLT ViT QKV bias shape mismatch");
  for (const auto &head : bias)
    if (head.size() != metadata.filter_width)
      throw std::invalid_argument("BOLT ViT QKV bias width mismatch");

  const std::size_t packed_per_group = metadata.image_size *
      metadata.filter_width * 2 / metadata.slot_count;
  std::vector<std::vector<seal::Plaintext>> result(
      metadata.paired_head_groups,
      std::vector<seal::Plaintext>(packed_per_group));
  for (std::size_t group = 0; group < metadata.paired_head_groups; ++group) {
    std::size_t first = 0;
    std::size_t second = 0;
    for (std::size_t packed = 0; packed < packed_per_group; ++packed) {
      std::vector<std::uint64_t> slots(metadata.slot_count, 0);
      for (std::size_t row = 0; row < metadata.slot_count;
           row += metadata.image_size) {
        if (row < metadata.slot_count / 2) {
          if (first >= metadata.filter_width)
            throw std::logic_error("BOLT ViT QKV bias first half overrun");
          for (std::size_t offset = 0; offset < metadata.image_size; ++offset)
            slots[row + offset] = bias[group * 2][first];
          ++first;
        } else {
          if (second >= metadata.filter_width)
            throw std::logic_error("BOLT ViT QKV bias second half overrun");
          for (std::size_t offset = 0; offset < metadata.image_size; ++offset)
            slots[row + offset] = bias[group * 2 + 1][second];
          ++second;
        }
      }
      seal::Plaintext plaintext;
      he_.encoder->encode(slots, plaintext);
      result[group][packed] = std::move(plaintext);
    }
    if (first != metadata.filter_width || second != metadata.filter_width)
      throw std::logic_error("BOLT ViT QKV bias packing did not consume bias");
  }
  return result;
}

std::vector<seal::Plaintext> BoltLinear::bert_generate_cross_packing_masks(
    std::size_t image_size) {
  he_.require_batching();
  if (image_size == 0 || he_.encoder->slot_count() % (2 * image_size) != 0)
    throw std::invalid_argument("BOLT ViT cross-mask image size is invalid");
  const std::size_t slot_count = he_.encoder->slot_count();
  const std::size_t half = slot_count / 2;
  std::vector<seal::Plaintext> result(2 * image_size);
  for (std::size_t diagonal = 0; diagonal < image_size; ++diagonal) {
    std::vector<std::uint64_t> left(slot_count, 0);
    std::vector<std::uint64_t> right(slot_count, 0);
    for (std::size_t k = 0; k < image_size - diagonal; ++k) {
      left[k + (diagonal * image_size) % half] = 1;
      left[k + half + (diagonal * image_size) % half] = 1;
    }
    for (std::size_t k = image_size - diagonal; k < image_size; ++k) {
      right[k + (diagonal * image_size) % half] = 1;
      right[k + half + (diagonal * image_size) % half] = 1;
    }
    he_.encoder->encode(left, result[diagonal]);
    he_.encoder->encode(right, result[diagonal + image_size]);
  }
  return result;
}

std::vector<std::vector<std::vector<seal::Ciphertext>>>
BoltLinear::bert_pack_softmax_diagonals(
    const std::vector<seal::Ciphertext> &matrix,
    const BoltBertPackingMetadata &metadata, BoltOperationCounts *counts) {
  he_.require_public_evaluation(false);
  metadata.validate(he_.encoder->slot_count());
  if (he_.party != ALICE)
    throw std::logic_error("BOLT softmax diagonal packing executes on ALICE");
  const std::size_t num_diag = metadata.image_size;
  const std::size_t num_diag_per_ct =
      metadata.slot_count / metadata.image_size / 2;
  const std::size_t per_group = metadata.image_size * metadata.image_size * 2 /
                                metadata.slot_count;
  if (matrix.size() != metadata.paired_head_groups * per_group ||
      num_diag_per_ct == 0 || (num_diag_per_ct & (num_diag_per_ct - 1)) != 0)
    throw std::invalid_argument("BOLT ViT softmax ciphertext packing shape mismatch");

  std::vector<std::vector<seal::Plaintext>> masks(
      2, std::vector<seal::Plaintext>(num_diag));
  for (std::size_t diagonal = 0; diagonal < num_diag; ++diagonal) {
    std::vector<std::uint64_t> left(metadata.slot_count, 0);
    std::vector<std::uint64_t> right(metadata.slot_count, 0);
    const std::size_t start = (diagonal % num_diag_per_ct) * num_diag;
    for (std::size_t j = 0; j < num_diag; ++j) {
      const std::uint64_t left_value = j < num_diag - diagonal ? 1 : 0;
      const std::uint64_t right_value = j < num_diag - diagonal ? 0 : 1;
      left[start + j] = left_value;
      left[start + j + metadata.slot_count / 2] = left_value;
      right[start + j] = right_value;
      right[start + j + metadata.slot_count / 2] = right_value;
    }
    he_.encoder->encode(left, masks[0][diagonal]);
    he_.encoder->encode(right, masks[1][diagonal]);
  }

  std::vector<std::vector<std::vector<seal::Ciphertext>>> result(
      metadata.paired_head_groups,
      std::vector<std::vector<seal::Ciphertext>>(
          2, std::vector<seal::Ciphertext>(num_diag)));
  for (std::size_t group = 0; group < metadata.paired_head_groups; ++group) {
    for (std::size_t diagonal = 0; diagonal < num_diag; ++diagonal) {
      const std::size_t index = group * per_group + diagonal / num_diag_per_ct;
      seal::Ciphertext current_left, current_right;
      he_.evaluator->multiply_plain(matrix[index], masks[0][diagonal],
                                    current_left);
      he_.evaluator->multiply_plain(matrix[index], masks[1][diagonal],
                                    current_right);
      increment(&BoltOperationCounts::multiply_plain, counts, 2);
      for (std::size_t step = 1; step < num_diag_per_ct; step <<= 1) {
        seal::Ciphertext rotated;
        he_.evaluator->rotate_rows(
            current_left, static_cast<int>(num_diag * step),
            *he_.galois_keys, rotated);
        he_.evaluator->add_inplace(current_left, rotated);
        he_.evaluator->rotate_rows(
            current_right, static_cast<int>(num_diag * step),
            *he_.galois_keys, rotated);
        he_.evaluator->add_inplace(current_right, rotated);
        increment(&BoltOperationCounts::rotations, counts, 2);
      }
      result[group][0][diagonal] = std::move(current_left);
      result[group][1][diagonal] = std::move(current_right);
    }
  }
  return result;
}

void BoltLinear::bert_layer_norm_affine_compat(
    const std::vector<seal::Ciphertext> &input,
    const std::vector<seal::Plaintext> &encoded_weights,
    std::uint32_t noise_flooding_bits, std::uint32_t modulus_switches,
    std::vector<seal::Ciphertext> &result, BoltOperationCounts *counts) {
  he_.require_public_evaluation(false);
  if (he_.party != ALICE || input.empty() || input.size() != encoded_weights.size())
    throw std::invalid_argument("BOLT LayerNorm affine shape mismatch");
  result.resize(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    he_.evaluator->multiply_plain(input[i], encoded_weights[i], result[i]);
    increment(&BoltOperationCounts::multiply_plain, counts);
    if (noise_flooding_bits != 0) {
      he_.evaluator->synchronize(result[i]);
      auto context_data = he_.context->get_context_data(result[i].parms_id());
      if (context_data == nullptr)
        throw std::logic_error("BOLT LayerNorm affine context is invalid");
      flood_ciphertext(result[i], context_data, noise_flooding_bits);
      increment(&BoltOperationCounts::noise_floodings, counts);
    }
    for (std::uint32_t level = 0; level < modulus_switches; ++level) {
      he_.evaluator->mod_switch_to_next_inplace(result[i]);
      increment(&BoltOperationCounts::modulus_switches, counts);
    }
  }
  increment(&BoltOperationCounts::ciphertexts, counts, result.size());
  finish_gpu_outputs(he_.evaluator, result);
}

} // namespace bolt
} // namespace sci
