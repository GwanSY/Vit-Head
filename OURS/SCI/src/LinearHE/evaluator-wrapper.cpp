#include "evaluator-wrapper.h"
#include <evaluate.cuh>

void PhantomEvaluator::negate_inplace(PhantomCiphertext &encrypted) const {
  phantom::negate_inplace(context_, encrypted);
}

void PhantomEvaluator::add_inplace(PhantomCiphertext &encrypted1,
                                   const PhantomCiphertext &encrypted2) const {
  phantom::add_inplace(context_, encrypted1, encrypted2);
}

void PhantomEvaluator::sub_inplace(PhantomCiphertext &encrypted1,
                                   const PhantomCiphertext &encrypted2,
                                   const bool &negate) const {
  phantom::sub_inplace(context_, encrypted1, encrypted2, negate);
}

void PhantomEvaluator::add_plain_inplace(PhantomCiphertext &encrypted,
                                         const PhantomPlaintext &plain) const {
  phantom::add_plain_inplace(context_, encrypted, plain);
}

void PhantomEvaluator::sub_plain_inplace(PhantomCiphertext &encrypted,
                                         const PhantomPlaintext &plain) const {
  phantom::sub_plain_inplace(context_, encrypted, plain);
}

void PhantomEvaluator::multiply_plain_inplace(
    PhantomCiphertext &encrypted, const PhantomPlaintext &plain) const {
  ct_pt_multiply_count_.fetch_add(1, std::memory_order_relaxed);
  phantom::multiply_plain_inplace(context_, encrypted, plain);
}

void PhantomEvaluator::multiply_inplace(
    PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) const {
  ct_ct_multiply_count_.fetch_add(1, std::memory_order_relaxed);
  phantom::multiply_inplace(context_, encrypted1, encrypted2);
}

void PhantomEvaluator::relinearize_inplace(
    PhantomCiphertext &encrypted, const PhantomRelinKey &relin_keys) const {
  relinearization_count_.fetch_add(1, std::memory_order_relaxed);
  phantom::relinearize_inplace(context_, encrypted, relin_keys);
}

void PhantomEvaluator::rescale_to_next(const PhantomCiphertext &encrypted,
                                       PhantomCiphertext &destination) const {
  rescale_count_.fetch_add(1, std::memory_order_relaxed);
  destination = std::move(phantom::rescale_to_next(context_, encrypted));
}

void PhantomEvaluator::mod_switch_to_next(
    const PhantomCiphertext &encrypted, PhantomCiphertext &destination) const {
  destination = std::move(phantom::mod_switch_to_next(context_, encrypted));
}

void PhantomEvaluator::rotate_vector_inplace(
    PhantomCiphertext &encrypted, int step,
    const PhantomGaloisKey &galois_key) const {
  const auto n_full_slots =
      context_.key_context_data().parms().poly_modulus_degree() >> 1;
  step %= n_full_slots;
  if (((step + n_full_slots) % n_full_slots) == 0) {
    return;
  }
  rotation_count_.fetch_add(1, std::memory_order_relaxed);
  phantom::rotate_inplace(context_, encrypted, step, galois_key);
}

void PhantomEvaluator::apply_galois_inplace(
    PhantomCiphertext &encrypted, std::size_t galois_elt,
    const PhantomGaloisKey &galois_key) const {
  rotation_count_.fetch_add(1, std::memory_order_relaxed);
  phantom::apply_galois_inplace(context_, encrypted, galois_elt, galois_key);
}

void PhantomEvaluator::complex_conjugate_inplace(
    PhantomCiphertext &encrypted, const PhantomGaloisKey &galois_key) const {
  conjugation_count_.fetch_add(1, std::memory_order_relaxed);
  phantom::rotate_inplace(context_, encrypted, 0, galois_key);
}

void PhantomEvaluator::reset_operation_counters() const {
  rotation_count_.store(0, std::memory_order_relaxed);
  ct_ct_multiply_count_.store(0, std::memory_order_relaxed);
  ct_pt_multiply_count_.store(0, std::memory_order_relaxed);
  relinearization_count_.store(0, std::memory_order_relaxed);
  rescale_count_.store(0, std::memory_order_relaxed);
  conjugation_count_.store(0, std::memory_order_relaxed);
}

std::uint64_t PhantomEvaluator::rotation_count() const {
  return rotation_count_.load(std::memory_order_relaxed);
}

std::uint64_t PhantomEvaluator::ct_ct_multiply_count() const {
  return ct_ct_multiply_count_.load(std::memory_order_relaxed);
}

std::uint64_t PhantomEvaluator::ct_pt_multiply_count() const {
  return ct_pt_multiply_count_.load(std::memory_order_relaxed);
}

std::uint64_t PhantomEvaluator::relinearization_count() const {
  return relinearization_count_.load(std::memory_order_relaxed);
}

std::uint64_t PhantomEvaluator::rescale_count() const {
  return rescale_count_.load(std::memory_order_relaxed);
}

std::uint64_t PhantomEvaluator::conjugation_count() const {
  return conjugation_count_.load(std::memory_order_relaxed);
}
