#include "BOLTTransformer/bolt_attention.h"

#include <stdexcept>
#include <utility>

namespace sci {
namespace bolt {

void BoltAttentionPacking::count(
    std::uint64_t BoltOperationCounts::*member, BoltOperationCounts *counts,
    std::uint64_t amount) const {
  if (counts != nullptr) counts->*member += amount;
}

seal::Plaintext BoltAttentionPacking::repeated_range_mask(
    std::size_t begin, std::size_t end, std::size_t stride) const {
  if (begin >= end || end > stride)
    throw std::invalid_argument("BOLT Attention mask range is invalid");
  std::vector<std::uint64_t> slots(he_.encoder->slot_count(), 0);
  for (std::size_t block = 0; block < slots.size(); block += stride)
    for (std::size_t slot = begin; slot < end; ++slot)
      slots[block + slot] = 1;
  seal::Plaintext mask;
  he_.encoder->encode(slots, mask);
  return mask;
}

BoltTensorMetadata BoltAttentionPacking::tensor_metadata(
    std::size_t rows, std::size_t cols,
    const BoltTensorMetadata &source) const {
  BoltTensorMetadata result = source;
  result.rows = rows;
  result.cols = cols;
  result.packing = BoltPackingKind::kCompactRepeatedRows;
  return result;
}

BoltCipherTensor BoltAttentionPacking::slice_head(
    const BoltCipherTensor &input, std::size_t head_index,
    const BoltAttentionPackingMetadata &metadata,
    BoltOperationCounts *counts) {
  he_.require_public_evaluation(false);
  metadata.validate(he_.encoder->slot_count());
  input.validate(he_.encoder->slot_count(), true);
  if (he_.party != ALICE)
    throw std::logic_error("BOLT Attention HE packing executes on ALICE");
  if (head_index >= metadata.heads || input.metadata.rows != metadata.tokens ||
      input.metadata.cols != metadata.model_dim ||
      input.metadata.packed_stride != metadata.packed_stride) {
    throw std::invalid_argument("BOLT Attention head slice metadata mismatch");
  }

  BoltCipherTensor result;
  result.metadata = tensor_metadata(metadata.tokens, metadata.head_dim,
                                    input.metadata);
  result.ciphertexts.resize(metadata.tokens);
  const std::size_t offset = head_index * metadata.head_dim;
  const seal::Plaintext valid =
      repeated_range_mask(0, metadata.head_dim, metadata.packed_stride);
  for (std::size_t token = 0; token < metadata.tokens; ++token) {
    if (offset == 0) {
      result.ciphertexts[token] = input.ciphertexts[token];
    } else {
      he_.rotate_rows_logical(input.ciphertexts[token],
                              static_cast<int>(offset),
                              result.ciphertexts[token]);
      count(&BoltOperationCounts::rotations, counts);
    }
    he_.evaluator->multiply_plain_inplace(result.ciphertexts[token], valid);
    count(&BoltOperationCounts::multiply_plain, counts);
  }
  count(&BoltOperationCounts::ciphertexts, counts, result.ciphertexts.size());
  return result;
}

BoltCipherTensor BoltAttentionPacking::transpose_tokens(
    const BoltCipherTensor &head,
    const BoltAttentionPackingMetadata &metadata,
    BoltOperationCounts *counts) {
  he_.require_public_evaluation(false);
  metadata.validate(he_.encoder->slot_count());
  head.validate(he_.encoder->slot_count(), true);
  if (he_.party != ALICE)
    throw std::logic_error("BOLT Attention HE packing executes on ALICE");
  if (head.metadata.rows != metadata.tokens ||
      head.metadata.cols != metadata.head_dim ||
      head.metadata.packed_stride != metadata.packed_stride) {
    throw std::invalid_argument("BOLT Attention transpose metadata mismatch");
  }

  BoltCipherTensor result;
  result.metadata = tensor_metadata(metadata.head_dim, metadata.tokens,
                                    head.metadata);
  result.ciphertexts.resize(metadata.head_dim);
  const seal::Plaintext first_slot =
      repeated_range_mask(0, 1, metadata.packed_stride);

  for (std::size_t feature = 0; feature < metadata.head_dim; ++feature) {
    bool initialized = false;
    seal::Ciphertext output;
    for (std::size_t token = 0; token < metadata.tokens; ++token) {
      seal::Ciphertext scalar;
      if (feature == 0) {
        scalar = head.ciphertexts[token];
      } else {
        he_.rotate_rows_logical(head.ciphertexts[token],
                                static_cast<int>(feature), scalar);
        count(&BoltOperationCounts::rotations, counts);
      }
      he_.evaluator->multiply_plain_inplace(scalar, first_slot);
      count(&BoltOperationCounts::multiply_plain, counts);
      if (token != 0) {
        seal::Ciphertext placed;
        he_.rotate_rows_logical(scalar, -static_cast<int>(token), placed);
        scalar = std::move(placed);
        count(&BoltOperationCounts::rotations, counts);
      }
      if (!initialized) {
        output = std::move(scalar);
        initialized = true;
      } else {
        he_.evaluator->add_inplace(output, scalar);
      }
    }
    result.ciphertexts[feature] = std::move(output);
  }
  count(&BoltOperationCounts::ciphertexts, counts, result.ciphertexts.size());
  return result;
}

BoltCipherTensor BoltAttentionPacking::merge_heads(
    const std::vector<BoltCipherTensor> &heads,
    const BoltAttentionPackingMetadata &metadata,
    BoltOperationCounts *counts) {
  he_.require_public_evaluation(false);
  metadata.validate(he_.encoder->slot_count());
  if (he_.party != ALICE)
    throw std::logic_error("BOLT Attention HE packing executes on ALICE");
  if (heads.size() != metadata.heads)
    throw std::invalid_argument("BOLT Attention head count mismatch");
  for (const auto &head : heads) {
    head.validate(he_.encoder->slot_count(), true);
    if (head.metadata.rows != metadata.tokens ||
        head.metadata.cols != metadata.head_dim ||
        head.metadata.packed_stride != metadata.packed_stride)
      throw std::invalid_argument("BOLT Attention merge metadata mismatch");
  }

  BoltCipherTensor result;
  result.metadata = tensor_metadata(metadata.tokens, metadata.model_dim,
                                    heads.front().metadata);
  result.ciphertexts.resize(metadata.tokens);
  for (std::size_t token = 0; token < metadata.tokens; ++token) {
    bool initialized = false;
    seal::Ciphertext output;
    for (std::size_t head_index = 0; head_index < metadata.heads;
         ++head_index) {
      seal::Ciphertext placed;
      const std::size_t offset = head_index * metadata.head_dim;
      if (offset == 0) {
        placed = heads[head_index].ciphertexts[token];
      } else {
        he_.rotate_rows_logical(heads[head_index].ciphertexts[token],
                                -static_cast<int>(offset), placed);
        count(&BoltOperationCounts::rotations, counts);
      }
      const seal::Plaintext mask = repeated_range_mask(
          offset, offset + metadata.head_dim, metadata.packed_stride);
      he_.evaluator->multiply_plain_inplace(placed, mask);
      count(&BoltOperationCounts::multiply_plain, counts);
      if (!initialized) {
        output = std::move(placed);
        initialized = true;
      } else {
        he_.evaluator->add_inplace(output, placed);
      }
    }
    result.ciphertexts[token] = std::move(output);
  }
  count(&BoltOperationCounts::ciphertexts, counts, result.ciphertexts.size());
  return result;
}

} // namespace bolt
} // namespace sci
