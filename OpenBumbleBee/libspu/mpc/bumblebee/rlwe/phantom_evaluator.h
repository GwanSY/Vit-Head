// Copyright 2026
//
// An exact coefficient bridge from OpenBumbleBee's existing SEAL objects to
// PhantomFHE. This is backend plumbing only: it does not change parameters,
// packing, the arithmetic circuit, or the HE/MPC conversion protocol.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "seal/ciphertext.h"
#include "seal/context.h"
#include "seal/evaluator.h"
#include "seal/galoiskeys.h"
#include "seal/plaintext.h"

namespace spu::mpc::bumblebee {

enum class BumblebeeHeBackend { kCpu, kGpu };

// The default is GPU. Accepted environment values are gpu/phantom and
// cpu/seal. Invalid values and unavailable GPUs are hard failures.
BumblebeeHeBackend GetBumblebeeHeBackend();
bool UsePhantomHeBackend();
const char* BumblebeeHeBackendName();
void ValidateBumblebeeHeBackend();

class PhantomEvaluator final {
 public:
  explicit PhantomEvaluator(const seal::SEALContext& context,
                            const seal::GaloisKeys* galois_keys = nullptr);
  ~PhantomEvaluator();

  PhantomEvaluator(const PhantomEvaluator&) = delete;
  PhantomEvaluator& operator=(const PhantomEvaluator&) = delete;

  bool using_gpu() const;

  void add_inplace(seal::Ciphertext& lhs,
                   const seal::Ciphertext& rhs) const;
  void sub_inplace(seal::Ciphertext& lhs,
                   const seal::Ciphertext& rhs) const;
  void negate(const seal::Ciphertext& input, seal::Ciphertext& output) const;
  void add_plain_inplace(seal::Ciphertext& cipher,
                         const seal::Plaintext& plain) const;
  void sub_plain_inplace(seal::Ciphertext& cipher,
                         const seal::Plaintext& plain) const;
  void multiply_plain(const seal::Ciphertext& cipher,
                      const seal::Plaintext& plain,
                      seal::Ciphertext& output) const;
  void multiply_plain_inplace(seal::Ciphertext& cipher,
                              const seal::Plaintext& plain) const;
  void multiply_sub_plain_inplace(seal::Ciphertext& cipher,
                                  const seal::Plaintext& multiplier,
                                  const seal::Plaintext& subtractor) const;
  void apply_galois_inplace(seal::Ciphertext& cipher,
                            std::uint32_t galois_elt,
                            const seal::GaloisKeys& galois_keys) const;
  void mod_switch_to_inplace(seal::Ciphertext& cipher,
                             seal::parms_id_type target) const;
  void ntt_inplace(seal::Ciphertext& cipher) const;
  void inverse_ntt_inplace(seal::Ciphertext& cipher) const;
  void ntt_inplace(seal::Plaintext& plain) const;
  void inverse_ntt_inplace(seal::Plaintext& plain) const;

  // Execute one existing OpenBumbleBee packing butterfly while its temporary
  // ciphertexts remain on the device. The shift, add/subtract, Galois, and add
  // sequence is unchanged and no matrix operation is fused.
  void packing_butterfly_inplace(seal::Ciphertext* even,
                                 const seal::Ciphertext* odd,
                                 std::size_t shift,
                                 std::uint32_t galois_elt,
                                 const seal::GaloisKeys& galois_keys) const;

  // Execute the existing level-by-level packing tree with intermediate RLWEs
  // resident on CUDA. `shifts` and `galois_elts` are supplied by the original
  // caller, so the packing algorithm and key-switch schedule are unchanged.
  void packing_butterfly_tree_inplace(
      seal::Ciphertext* ciphers, std::size_t count, std::size_t tree_width,
      const std::vector<std::size_t>& shifts,
      const std::vector<std::uint32_t>& galois_elts,
      const seal::GaloisKeys& galois_keys) const;

  // Execute the existing ciphertext/plaintext multiply-add sequence. No new
  // packing, blocking, matrix algorithm, or fused CUDA MatMul kernel is used.
  void multiply_add_sequence(const seal::Ciphertext* ciphers,
                             const seal::Plaintext* plains,
                             std::size_t count,
                             seal::Ciphertext& output,
                             bool inverse_ntt_output = false,
                             seal::parms_id_type output_target =
                                 seal::parms_id_zero) const;

  // Cache one matrix operation's immutable inputs on the device. Existing
  // multiply_add_sequence calls and their arithmetic order stay unchanged;
  // only repeated host uploads are replaced by reads from this resident set.
  void prepare_multiply_add_inputs(const seal::Ciphertext* ciphers,
                                   std::size_t cipher_count,
                                   const seal::Plaintext* plains,
                                   std::size_t plain_count) const;
  void clear_multiply_add_inputs() const;

 private:
  struct GpuState;
  const seal::SEALContext& context_;
  std::unique_ptr<seal::Evaluator> cpu_;
  std::unique_ptr<GpuState> gpu_;
};

}  // namespace spu::mpc::bumblebee
