#include "BOLTTransformer/bolt_conversion.h"

#include "FloatingPoint/fixed-point.h"
#include "LinearHE/utils-HE.h"
#include "utils/prg.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace sci {
namespace bolt {

std::uint64_t BoltConversion::share_to_plain(std::uint64_t value,
                                             std::uint32_t bit_width) const {
  const std::uint64_t ring_modulus = 1ULL << bit_width;
  const std::uint64_t ring_mask = ring_modulus - 1;
  value &= ring_mask;
  const std::uint64_t sign = 1ULL << (bit_width - 1);
  if (value < sign) {
    return value % he_.plain_modulus;
  }
  const std::uint64_t magnitude = ring_modulus - value;
  const std::uint64_t residue = magnitude % he_.plain_modulus;
  return residue == 0 ? 0 : he_.plain_modulus - residue;
}

std::vector<std::uint64_t> BoltConversion::pack_repeated_row(
    const std::uint64_t *row, std::size_t cols,
    const BoltTensorMetadata &metadata) const {
  std::vector<std::uint64_t> slots(he_.encoder->slot_count(), 0);
  for (std::size_t block = 0; block < slots.size();
       block += metadata.packed_stride) {
    for (std::size_t col = 0; col < cols; ++col) {
      slots[block + col] = share_to_plain(row[col], metadata.share_bit_width);
    }
  }
  return slots;
}

std::vector<std::uint64_t> BoltConversion::pack_repeated_unsigned_row(
    const std::uint64_t *row, std::size_t cols,
    const BoltTensorMetadata &metadata) const {
  const std::uint64_t ring_mask =
      (1ULL << metadata.share_bit_width) - 1;
  std::vector<std::uint64_t> slots(he_.encoder->slot_count(), 0);
  for (std::size_t block = 0; block < slots.size();
       block += metadata.packed_stride)
    for (std::size_t col = 0; col < cols; ++col)
      slots[block + col] = (row[col] & ring_mask) % he_.plain_modulus;
  return slots;
}

BoltCipherTensor BoltConversion::ss_to_he_signed_ring(
    const SecretSharedTensor &input, const BoltTensorMetadata &metadata,
    FixOp &fix) {
  he_.require_batching();
  metadata.validate(he_.encoder->slot_count());
  input.validate(he_.encoder->slot_count());
  if (input.metadata.rows != metadata.rows ||
      input.metadata.cols != metadata.cols ||
      input.metadata.packed_stride != metadata.packed_stride ||
      input.metadata.packing != metadata.packing)
    throw std::invalid_argument("BOLT signed ring SS-to-HE metadata mismatch");

  const std::size_t count = metadata.rows * metadata.cols;
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("BOLT signed ring SS-to-HE input is too large");

  // FixOp::extend dispatches to the existing BERT/BOLT signed-extension
  // protocol because this array is marked signed.  Thus a negative x is
  // lifted to x mod 2^64 rather than zero-extended from its ell-bit encoding.
  const std::uint64_t ring_mask =
      (1ULL << metadata.share_bit_width) - 1;
  FixArray ring_shares(he_.party, static_cast<int>(count), true,
                  static_cast<int>(metadata.share_bit_width),
                  static_cast<int>(metadata.scale));
  std::memcpy(ring_shares.data, input.local_share.data(),
              count * sizeof(std::uint64_t));
  for (std::size_t i = 0; i < count; ++i)
    ring_shares.data[i] &= ring_mask;

  // Let S be the ordinary-integer sum of the two extended uint64 shares.
  // For x >= 0, S is x or x+2^64 and the latter is the overwhelmingly likely
  // event.  For x < 0, S is x+2^64 or x+2*2^64 and the former is the
  // overwhelmingly likely event.  In both good cases S-2^64=x, so ALICE
  // subtracts 2^64 mod p once.  The bad-event probability is approximately
  // |x|/2^64, exactly the statistical correction used by BERT ring_to_field.
  FixArray extended = fix.extend(ring_shares, 64);
  const std::uint64_t two_to_64_mod_p = static_cast<std::uint64_t>(
      (static_cast<__uint128_t>(1) << 64) % he_.plain_modulus);
  std::vector<std::uint64_t> corrected_shares(count);
  for (std::size_t i = 0; i < count; ++i) {
    corrected_shares[i] = extended.data[i] % he_.plain_modulus;
    if (he_.party == ALICE)
      corrected_shares[i] =
          (corrected_shares[i] + he_.plain_modulus - two_to_64_mod_p) %
          he_.plain_modulus;
  }

  BoltCipherTensor result;
  result.metadata = metadata;
  if (he_.party == BOB) {
    result.ciphertexts.resize(metadata.rows);
    for (std::size_t row = 0; row < metadata.rows; ++row) {
      seal::Plaintext plaintext;
      he_.encoder->encode(pack_repeated_unsigned_row(
          corrected_shares.data() + row * metadata.cols, metadata.cols,
          metadata), plaintext);
      he_.encryptor->encrypt(plaintext, result.ciphertexts[row]);
    }
    send_encrypted_vector(he_.io, result.ciphertexts);
    result.ciphertexts.clear();
    return result;
  }

  result.ciphertexts.resize(metadata.rows);
  recv_encrypted_vector(he_.context, he_.io, result.ciphertexts);
  for (std::size_t row = 0; row < metadata.rows; ++row) {
    seal::Plaintext data_plain;
    he_.encoder->encode(pack_repeated_unsigned_row(
        corrected_shares.data() + row * metadata.cols, metadata.cols,
        metadata),
        data_plain);
    he_.evaluator->add_plain_inplace(result.ciphertexts[row], data_plain);
  }
  return result;
}

BoltCipherTensor
BoltConversion::ss_to_he(const SecretSharedTensor &input,
                         const BoltTensorMetadata &metadata) {
  he_.require_batching();
  metadata.validate(he_.encoder->slot_count());
  input.validate(he_.encoder->slot_count());
  if (input.metadata.rows != metadata.rows ||
      input.metadata.cols != metadata.cols ||
      input.metadata.packed_stride != metadata.packed_stride ||
      input.metadata.packing != metadata.packing) {
    throw std::invalid_argument("BOLT SS-to-HE metadata mismatch");
  }

  BoltCipherTensor result;
  result.metadata = metadata;
  if (he_.party == BOB) {
    result.ciphertexts.resize(metadata.rows);
    for (std::size_t row = 0; row < metadata.rows; ++row) {
      const auto slots = pack_repeated_row(
          &input.local_share[row * metadata.cols], metadata.cols, metadata);
      seal::Plaintext plaintext;
      he_.encoder->encode(slots, plaintext);
      he_.encryptor->encrypt(plaintext, result.ciphertexts[row]);
    }
    send_encrypted_vector(he_.io, result.ciphertexts);
    // Only ALICE evaluates the ciphertext. Keeping a second local copy on BOB
    // would invite accidental clear reconstruction and doubles memory.
    result.ciphertexts.clear();
  } else {
    result.ciphertexts.resize(metadata.rows);
    recv_encrypted_vector(he_.context, he_.io, result.ciphertexts);
    for (std::size_t row = 0; row < metadata.rows; ++row) {
      const auto slots = pack_repeated_row(
          &input.local_share[row * metadata.cols], metadata.cols, metadata);
      seal::Plaintext plaintext;
      he_.encoder->encode(slots, plaintext);
      he_.evaluator->add_plain_inplace(result.ciphertexts[row], plaintext);
    }
  }
  return result;
}

SecretSharedTensor
BoltConversion::he_to_ss(const BoltCipherTensor &input,
                         const BoltTensorMetadata &metadata) {
  he_.require_batching();
  metadata.validate(he_.encoder->slot_count());
  if (input.metadata.rows != metadata.rows ||
      input.metadata.cols != metadata.cols ||
      input.metadata.packed_stride != metadata.packed_stride) {
    throw std::invalid_argument("BOLT HE-to-SS metadata mismatch");
  }

  SecretSharedTensor result;
  result.metadata = metadata;
  result.local_share.resize(metadata.rows * metadata.cols);

  if (he_.party == ALICE) {
    input.validate(he_.encoder->slot_count(), true);
    std::vector<seal::Ciphertext> masked = input.ciphertexts;
    PRG128 prg;
    std::vector<std::uint64_t> logical_mask(metadata.cols);
    for (std::size_t row = 0; row < metadata.rows; ++row) {
      prg.random_mod_p<std::uint64_t>(logical_mask.data(), metadata.cols,
                                     he_.plain_modulus);
      for (std::size_t col = 0; col < metadata.cols; ++col) {
        result.local_share[row * metadata.cols + col] = logical_mask[col];
      }

      std::vector<std::uint64_t> slots(he_.encoder->slot_count(), 0);
      for (std::size_t block = 0; block < slots.size();
           block += metadata.packed_stride) {
        for (std::size_t col = 0; col < metadata.cols; ++col) {
          slots[block + col] = logical_mask[col];
        }
      }
      seal::Plaintext mask_plaintext;
      he_.encoder->encode(slots, mask_plaintext);
      he_.evaluator->sub_plain_inplace(masked[row], mask_plaintext);

      if (metadata.add_plain_modulus_half_on_he_to_ss) {
        const std::uint64_t half = (he_.plain_modulus - 1) / 2;
        for (std::size_t block = 0; block < slots.size();
             block += metadata.packed_stride) {
          for (std::size_t col = 0; col < metadata.cols; ++col) {
            slots[block + col] = half;
          }
        }
        seal::Plaintext half_plaintext;
        he_.encoder->encode(slots, half_plaintext);
        he_.evaluator->add_plain_inplace(masked[row], half_plaintext);
      }
    }
    send_encrypted_vector(he_.io, masked);
  } else {
    he_.require_decryption();
    std::vector<seal::Ciphertext> masked(metadata.rows);
    recv_encrypted_vector(he_.context, he_.io, masked);
    for (std::size_t row = 0; row < metadata.rows; ++row) {
      seal::Plaintext plaintext;
      std::vector<std::uint64_t> slots;
      he_.decryptor->decrypt(masked[row], plaintext);
      he_.encoder->decode(plaintext, slots);
      for (std::size_t col = 0; col < metadata.cols; ++col) {
        result.local_share[row * metadata.cols + col] = slots[col];
      }
    }
  }
  return result;
}

std::vector<seal::Ciphertext> BoltConversion::ss_to_he_packed_compat(
    const std::uint64_t *local_share, std::size_t element_count,
    std::uint32_t share_bit_width) {
  he_.require_batching();
  const std::size_t slot_count = he_.encoder->slot_count();
  if (local_share == nullptr || element_count == 0 ||
      element_count % slot_count != 0 || share_bit_width == 0 ||
      share_bit_width >= 64) {
    throw std::invalid_argument("BOLT packed SS-to-HE compatibility shape is invalid");
  }
  const std::size_t ciphertext_count = element_count / slot_count;
  std::vector<seal::Ciphertext> result(ciphertext_count);
  if (he_.party == BOB) {
    for (std::size_t ct = 0; ct < ciphertext_count; ++ct) {
      std::vector<std::uint64_t> slots(slot_count);
      for (std::size_t slot = 0; slot < slot_count; ++slot)
        slots[slot] = share_to_plain(local_share[ct * slot_count + slot],
                                     share_bit_width);
      seal::Plaintext plaintext;
      he_.encoder->encode(slots, plaintext);
      he_.encryptor->encrypt(plaintext, result[ct]);
    }
    send_encrypted_vector(he_.io, result);
    result.clear();
  } else {
    recv_encrypted_vector(he_.context, he_.io, result);
    for (std::size_t ct = 0; ct < ciphertext_count; ++ct) {
      std::vector<std::uint64_t> slots(slot_count);
      for (std::size_t slot = 0; slot < slot_count; ++slot)
        slots[slot] = share_to_plain(local_share[ct * slot_count + slot],
                                     share_bit_width);
      seal::Plaintext plaintext;
      he_.encoder->encode(slots, plaintext);
      he_.evaluator->add_plain_inplace(result[ct], plaintext);
    }
  }
  return result;
}

std::vector<std::uint64_t> BoltConversion::he_to_ss_packed_compat(
    const std::vector<seal::Ciphertext> &input,
    std::size_t ciphertext_count, bool add_plain_modulus_half) {
  he_.require_batching();
  const std::size_t slot_count = he_.encoder->slot_count();
  if (ciphertext_count == 0 ||
      (he_.party == ALICE && input.size() != ciphertext_count)) {
    throw std::invalid_argument("BOLT packed HE-to-SS compatibility shape is invalid");
  }
  std::vector<std::uint64_t> result(ciphertext_count * slot_count);
  if (he_.party == ALICE) {
    std::vector<seal::Ciphertext> masked(input);
    PRG128 prg;
    prg.random_mod_p<std::uint64_t>(result.data(), result.size(),
                                   he_.plain_modulus);
    std::vector<std::uint64_t> half_slots(
        slot_count, (he_.plain_modulus - 1) / 2);
    seal::Plaintext half_plaintext;
    if (add_plain_modulus_half)
      he_.encoder->encode(half_slots, half_plaintext);
    for (std::size_t ct = 0; ct < ciphertext_count; ++ct) {
      std::vector<std::uint64_t> mask(
          result.begin() + ct * slot_count,
          result.begin() + (ct + 1) * slot_count);
      seal::Plaintext mask_plaintext;
      he_.encoder->encode(mask, mask_plaintext);
      he_.evaluator->sub_plain_inplace(masked[ct], mask_plaintext);
      if (add_plain_modulus_half)
        he_.evaluator->add_plain_inplace(masked[ct], half_plaintext);
    }
    send_encrypted_vector(he_.io, masked);
  } else {
    he_.require_decryption();
    std::vector<seal::Ciphertext> masked(ciphertext_count);
    recv_encrypted_vector(he_.context, he_.io, masked);
    for (std::size_t ct = 0; ct < ciphertext_count; ++ct) {
      seal::Plaintext plaintext;
      std::vector<std::uint64_t> decoded;
      he_.decryptor->decrypt(masked[ct], plaintext);
      he_.encoder->decode(plaintext, decoded);
      std::copy(decoded.begin(), decoded.end(),
                result.begin() + ct * slot_count);
    }
  }
  return result;
}

} // namespace bolt
} // namespace sci
