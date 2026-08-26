#ifndef SCI_BOLT_TRANSFORMER_EVALUATOR_H
#define SCI_BOLT_TRANSFORMER_EVALUATOR_H

#include "seal/evaluator.h"
#include "seal/galoiskeys.h"
#include "seal/relinkeys.h"

#include <memory>

namespace sci {
namespace bolt {

// BOLT keeps SEAL ciphertexts as its tensor and wire representation.  This
// class is the execution seam: the default path delegates to SEAL, while the
// GPU entry points run the same BFV operation in BLB's Phantom backend and
// copy the result back to the unchanged BOLT representation.
class BoltEvaluator {
public:
  explicit BoltEvaluator(const seal::SEALContext &context);
  ~BoltEvaluator();

  BoltEvaluator(const BoltEvaluator &) = delete;
  BoltEvaluator &operator=(const BoltEvaluator &) = delete;

  // GPU is intentionally opt-in and is enabled only by BOLT_*_GPU targets.
  // The public entry point uses device 0 for reproducible comparisons.
  void enable_gpu(const seal::GaloisKeys &galois_keys,
                  const seal::RelinKeys &relinearization_keys,
                  int device = 0);

  bool gpu_enabled() const noexcept;

  void add(const seal::Ciphertext &lhs, const seal::Ciphertext &rhs,
           seal::Ciphertext &destination) const;
  void add_inplace(seal::Ciphertext &lhs,
                   const seal::Ciphertext &rhs) const;
  void add_plain_inplace(seal::Ciphertext &ciphertext,
                         const seal::Plaintext &plaintext) const;
  void sub_plain_inplace(seal::Ciphertext &ciphertext,
                         const seal::Plaintext &plaintext) const;
  void multiply(const seal::Ciphertext &lhs, const seal::Ciphertext &rhs,
                seal::Ciphertext &destination) const;
  void multiply_plain(const seal::Ciphertext &ciphertext,
                      const seal::Plaintext &plaintext,
                      seal::Ciphertext &destination) const;
  void multiply_plain_inplace(seal::Ciphertext &ciphertext,
                              const seal::Plaintext &plaintext) const;
  void relinearize_inplace(seal::Ciphertext &ciphertext,
                           const seal::RelinKeys &relinearization_keys) const;
  void mod_switch_to_next_inplace(seal::Ciphertext &ciphertext) const;
  void rotate_rows(const seal::Ciphertext &ciphertext, int steps,
                   const seal::GaloisKeys &galois_keys,
                   seal::Ciphertext &destination) const;
  void rotate_columns(const seal::Ciphertext &ciphertext,
                      const seal::GaloisKeys &galois_keys,
                      seal::Ciphertext &destination) const;

  // Keep BOLT's host representation as the public API, but allow the GPU
  // backend to keep the actual ciphertext resident between evaluator calls.
  // These two helpers are used at the few places where BOLT itself copies or
  // directly serializes a ciphertext outside the evaluator.
  void copy_ciphertext(const seal::Ciphertext &source,
                       seal::Ciphertext &destination) const;
  void synchronize(const seal::Ciphertext &ciphertext) const;
  void release_gpu_residency() const;

private:
  class GpuState;

  const seal::SEALContext *context_ = nullptr;
  std::unique_ptr<seal::Evaluator> host_evaluator_;
  std::unique_ptr<GpuState> gpu_;
};

} // namespace bolt
} // namespace sci

#endif
