#include "BOLTTransformer/bolt_vit_compat_linear.h"

#include "LinearHE/utils-HE.h"

#include <algorithm>
#include <stdexcept>

namespace sci {
namespace bolt {

BoltVitCompatLinearAdapter::BoltVitCompatLinearAdapter(
    BoltHEView he, const BoltVitCompatPacking &packing)
    : he_(he), linear_(he), packing_(packing) {
  he_.require_public_evaluation(true);
  packing_.validate();
}

BoltBertPackingMetadata BoltVitCompatLinearAdapter::linear_metadata(
    std::size_t input_dim, std::size_t output_dim,
    std::uint32_t noise_flooding_bits,
    std::uint32_t modulus_switches) const {
  const std::size_t columns = packing_.columns_per_ciphertext();
  if (input_dim == 0 || output_dim == 0 ||
      input_dim % (2 * columns) != 0 ||
      output_dim % (2 * columns) != 0 ||
      noise_flooding_bits == 0 || modulus_switches == 0)
    throw std::invalid_argument(
        "ViT compact linear dimensions or hardening are invalid");
  BoltBertPackingMetadata metadata;
  metadata.slot_count = packing_.slot_count();
  metadata.image_size = packing_.tokens();
  metadata.filter_height = input_dim;
  metadata.filter_width = output_dim;
  metadata.baby_step = 4;
  metadata.giant_steps = 4;
  metadata.attention_heads = packing_.heads();
  metadata.paired_head_groups = packing_.paired_head_groups();
  metadata.noise_flooding_bits = noise_flooding_bits;
  metadata.modulus_switches = modulus_switches;
  metadata.validate(packing_.slot_count());
  const std::size_t diagonals =
      packing_.slot_count() / (2 * packing_.tokens());
  if (metadata.baby_step * metadata.giant_steps != diagonals)
    throw std::logic_error(
        "ViT compact linear BSGS factors do not cover every diagonal");
  return metadata;
}

std::vector<std::vector<seal::Plaintext>>
BoltVitCompatLinearAdapter::encode_diagonals(
    const PlainWeightTensor &weight, bool swapped) const {
  const std::size_t slots = packing_.slot_count();
  const std::size_t half = slots / 2;
  const std::size_t tokens = packing_.tokens();
  const std::size_t diagonals = slots / (2 * tokens);
  const std::size_t input_ciphertexts =
      tokens * weight.input_cols / slots;
  const std::size_t output_ciphertexts =
      tokens * weight.output_cols / slots;
  const std::size_t baby_step = 4;
  std::vector<std::vector<seal::Plaintext>> encoded(
      output_ciphertexts * diagonals,
      std::vector<seal::Plaintext>(input_ciphertexts));
  // OpenMP workers cannot propagate exceptions across the parallel region.
  // Record the (normally unreachable) packing invariant failures per task and
  // report them deterministically once every independent encoding completes.
  std::vector<unsigned char> packing_status(encoded.size(), 0);

#pragma omp parallel for collapse(2) schedule(static) num_threads(32)
  for (std::size_t output_ciphertext = 0;
       output_ciphertext < output_ciphertexts; ++output_ciphertext) {
    for (std::size_t diagonal = 0; diagonal < diagonals; ++diagonal) {
      const std::size_t task = output_ciphertext * diagonals + diagonal;
      std::vector<std::uint64_t> buffer;
      buffer.reserve(slots);
      bool upper_output_half = swapped;
      std::size_t encoded_input = 0;
      for (std::size_t input = 0; input < weight.input_cols; ++input) {
        const std::size_t local_input = input % diagonals;
        std::size_t output = output_ciphertext * diagonals +
                             (local_input + diagonal) % diagonals;
        if (upper_output_half) output += weight.output_cols / 2;
        const std::uint64_t coefficient =
            weight.values[output * weight.input_cols + input];
        buffer.insert(buffer.end(), tokens, coefficient);
        if (buffer.size() % half != 0) continue;

        const std::size_t shift = (diagonal % baby_step) * tokens;
        if (shift != 0) {
          auto half_begin = buffer.end() - static_cast<std::ptrdiff_t>(half);
          std::rotate(half_begin,
                      buffer.end() - static_cast<std::ptrdiff_t>(shift),
                      buffer.end());
        }
        upper_output_half = !upper_output_half;
        if (buffer.size() == slots) {
          if (swapped)
            std::rotate(buffer.begin(), buffer.begin() + half, buffer.end());
          if (encoded_input >= input_ciphertexts) {
            packing_status[task] = 1;
            break;
          }
          he_.encoder->encode(
              buffer,
              encoded[task][encoded_input++]);
          buffer.clear();
        }
      }
      if (packing_status[task] == 0 &&
          (!buffer.empty() || encoded_input != input_ciphertexts))
        packing_status[task] = 2;
    }
  }
  for (const unsigned char status : packing_status) {
    if (status == 1)
      throw std::logic_error(
          "ViT compact linear encoded too many input ciphertexts");
    if (status == 2)
      throw std::logic_error(
          "ViT compact linear diagonal packing was incomplete");
  }
  return encoded;
}

std::vector<seal::Plaintext> BoltVitCompatLinearAdapter::encode_bias(
    const PlainWeightTensor &weight) const {
  const std::size_t slots = packing_.slot_count();
  const std::size_t half = slots / 2;
  const std::size_t tokens = packing_.tokens();
  const std::size_t columns = packing_.columns_per_ciphertext();
  const std::size_t output_ciphertexts =
      tokens * weight.output_cols / slots;
  std::vector<seal::Plaintext> encoded(output_ciphertexts);
#pragma omp parallel for schedule(static) num_threads(32)
  for (std::size_t ciphertext = 0; ciphertext < output_ciphertexts;
       ++ciphertext) {
    std::vector<std::uint64_t> packed(slots, 0);
    for (std::size_t local = 0; local < columns; ++local) {
      const std::size_t low = ciphertext * columns + local;
      const std::size_t high = weight.output_cols / 2 + low;
      for (std::size_t token = 0; token < tokens; ++token) {
        packed[local * tokens + token] = weight.bias[low];
        packed[half + local * tokens + token] = weight.bias[high];
      }
    }
    he_.encoder->encode(packed, encoded[ciphertext]);
  }
  return encoded;
}

BoltVitCompatLinearPlan BoltVitCompatLinearAdapter::prepare(
    const PlainWeightTensor &weight,
    std::uint32_t noise_flooding_bits,
    std::uint32_t modulus_switches) const {
  if (he_.party != ALICE)
    throw std::logic_error(
        "ViT private compact linear preprocessing is ALICE-only");
  weight.validate();
  if (weight.bias.empty())
    throw std::invalid_argument(
        "ViT compact linear requires an explicit private bias");
  BoltVitCompatLinearPlan plan;
  plan.input_dim = weight.input_cols;
  plan.output_dim = weight.output_cols;
  plan.metadata = linear_metadata(weight.input_cols, weight.output_cols,
                                  noise_flooding_bits, modulus_switches);
  plan.encoded_left = encode_diagonals(weight, false);
  plan.encoded_right = encode_diagonals(weight, true);
  plan.encoded_bias = encode_bias(weight);
  return plan;
}

std::vector<seal::Ciphertext> BoltVitCompatLinearAdapter::apply(
    const std::vector<seal::Ciphertext> &input,
    const BoltVitCompatLinearPlan &plan,
    BoltOperationCounts *counts) {
  if (he_.party != ALICE)
    throw std::logic_error("ViT compact linear evaluation is ALICE-only");
  const std::size_t expected_input =
      packing_.tokens() * plan.input_dim / packing_.slot_count();
  const std::size_t expected_output =
      packing_.tokens() * plan.output_dim / packing_.slot_count();
  if (input.size() != expected_input ||
      plan.encoded_bias.size() != expected_output)
    throw std::invalid_argument("ViT compact linear ciphertext shape mismatch");
  std::vector<seal::Ciphertext> result(expected_output);
  linear_.bert_cipher_plain_matmul_bsgs_compat(
      input, plan.encoded_left, plan.encoded_right, plan.encoded_bias,
      plan.metadata, result, counts);
  return result;
}

void BoltVitCompatLinearAdapter::harden(
    std::vector<seal::Ciphertext> &ciphertexts,
    std::uint32_t noise_flooding_bits,
    std::uint32_t modulus_switches,
    BoltOperationCounts *counts) const {
  if (noise_flooding_bits == 0 || modulus_switches == 0)
    throw std::invalid_argument(
        "ViT compact affine requires circuit-privacy hardening");
  for (auto &ciphertext : ciphertexts) {
    auto context_data = he_.context->get_context_data(ciphertext.parms_id());
    if (context_data == nullptr)
      throw std::logic_error("ViT compact affine lost BFV context");
    flood_ciphertext(ciphertext, context_data, noise_flooding_bits);
    if (counts != nullptr) ++counts->noise_floodings;
    for (std::uint32_t level = 0; level < modulus_switches; ++level) {
      context_data = he_.context->get_context_data(ciphertext.parms_id());
      if (context_data == nullptr ||
          context_data->next_context_data() == nullptr)
        throw std::logic_error(
            "ViT compact affine hardening exceeds the BFV chain");
      he_.evaluator->mod_switch_to_next_inplace(ciphertext);
      if (counts != nullptr) ++counts->modulus_switches;
    }
  }
}

std::vector<seal::Ciphertext>
BoltVitCompatLinearAdapter::elementwise_affine(
    const std::vector<seal::Ciphertext> &input,
    const std::vector<std::uint64_t> &multiplier,
    const std::vector<std::uint64_t> &bias,
    std::uint32_t noise_flooding_bits,
    std::uint32_t modulus_switches,
    BoltOperationCounts *counts) {
  if (he_.party != ALICE)
    throw std::logic_error("ViT compact affine evaluation is ALICE-only");
  if (multiplier.empty() || multiplier.size() != bias.size() ||
      multiplier.size() % (2 * packing_.columns_per_ciphertext()) != 0 ||
      input.size() * 2 * packing_.columns_per_ciphertext() !=
          multiplier.size())
    throw std::invalid_argument("ViT compact affine shape mismatch");
  const std::size_t slots = packing_.slot_count();
  const std::size_t tokens = packing_.tokens();
  std::vector<seal::Ciphertext> result(input.size());
#pragma omp parallel for schedule(static) num_threads(32)
  for (std::size_t ciphertext = 0; ciphertext < input.size(); ++ciphertext) {
    std::vector<std::uint64_t> multiplier_slots(slots);
    std::vector<std::uint64_t> bias_slots(slots);
    for (std::size_t slot = 0; slot < slots; ++slot) {
      const std::size_t feature =
          ciphertext * (slots / tokens) + slot / tokens;
      multiplier_slots[slot] = multiplier[feature];
      bias_slots[slot] = bias[feature];
    }
    seal::Plaintext multiplier_plain, bias_plain;
    he_.encoder->encode(multiplier_slots, multiplier_plain);
    he_.encoder->encode(bias_slots, bias_plain);
    he_.evaluator->multiply_plain(input[ciphertext], multiplier_plain,
                                  result[ciphertext]);
    he_.evaluator->add_plain_inplace(result[ciphertext], bias_plain);
  }
  if (counts != nullptr) {
    counts->multiply_plain += input.size();
    counts->ciphertexts += input.size();
  }
  harden(result, noise_flooding_bits, modulus_switches, counts);
  return result;
}

} // namespace bolt
} // namespace sci
