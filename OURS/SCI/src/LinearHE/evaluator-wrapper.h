#ifndef LINEARHE_EVALUATOR_WRAPPER_H_
#define LINEARHE_EVALUATOR_WRAPPER_H_

#include "phantom-host-compat.h"
#include <ciphertext.h>
#include <context.cuh>
#include <plaintext.h>
#include <secretkey.h>
#include <atomic>
#include <cstdint>

class PhantomEvaluator {

public:
  PhantomEvaluator() = delete;

  PhantomEvaluator(const PhantomContext &context) : context_(context) {}

  ~PhantomEvaluator() = default;

  // encrypted = -encrypted
  void negate_inplace(PhantomCiphertext &encrypted) const;

  inline void negate(const PhantomCiphertext &encrypted,
                     PhantomCiphertext &destination) const {
    destination = encrypted;
    negate_inplace(destination);
  }

  // encrypted1 += encrypted2
  void add_inplace(PhantomCiphertext &encrypted1,
                   const PhantomCiphertext &encrypted2) const;

  inline void add(const PhantomCiphertext &encrypted1,
                  const PhantomCiphertext &encrypted2,
                  PhantomCiphertext &destination) const {
    destination = encrypted1;
    add_inplace(destination, encrypted2);
  }

  // if negate = false (default): encrypted1 -= encrypted2
  // if negate = true: encrypted1 = encrypted2 - encrypted1
  void sub_inplace(PhantomCiphertext &encrypted1,
                   const PhantomCiphertext &encrypted2,
                   const bool &negate = false) const;

  inline void sub(const PhantomCiphertext &encrypted1,
                  const PhantomCiphertext &encrypted2,
                  PhantomCiphertext &destination,
                  const bool &negate = false) const {
    destination = encrypted1;
    sub_inplace(destination, encrypted2, negate);
  }

  // encrypted += plain
  void add_plain_inplace(PhantomCiphertext &encrypted,
                         const PhantomPlaintext &plain) const;

  inline void add_plain(const PhantomCiphertext &encrypted,
                        const PhantomPlaintext &plain,
                        PhantomCiphertext &destination) const {
    destination = encrypted;
    add_plain_inplace(destination, plain);
  }

  // encrypted -= plain
  void sub_plain_inplace(PhantomCiphertext &encrypted,
                         const PhantomPlaintext &plain) const;

  inline void sub_plain(const PhantomCiphertext &encrypted,
                        const PhantomPlaintext &plain,
                        PhantomCiphertext &destination) const {
    destination = encrypted;
    sub_plain_inplace(destination, plain);
  }

  // encrypted *= plain
  void multiply_plain_inplace(PhantomCiphertext &encrypted,
                              const PhantomPlaintext &plain) const;

  inline void multiply_plain(const PhantomCiphertext &encrypted,
                             const PhantomPlaintext &plain,
                             PhantomCiphertext &destination) const {
    destination = encrypted;
    multiply_plain_inplace(destination, plain);
  }

  // encrypted1 *= encrypted2
  void multiply_inplace(PhantomCiphertext &encrypted1,
                        const PhantomCiphertext &encrypted2) const;

  inline void multiply(const PhantomCiphertext &encrypted1,
                       const PhantomCiphertext &encrypted2,
                       PhantomCiphertext &destination) const {
    destination = encrypted1;
    multiply_inplace(destination, encrypted2);
  }

  inline void square_inplace(PhantomCiphertext &encrypted) const {
    multiply_inplace(encrypted, encrypted);
  }

  inline void square(const PhantomCiphertext &encrypted,
                     PhantomCiphertext &destination) {
    destination = encrypted;
    multiply_inplace(destination, encrypted);
  }

  void relinearize_inplace(PhantomCiphertext &encrypted,
                           const PhantomRelinKey &relin_keys) const;

  inline void relinearize(const PhantomCiphertext &encrypted,
                          const PhantomRelinKey &relin_keys,
                          PhantomCiphertext &destination) const {
    destination = encrypted;
    relinearize_inplace(destination, relin_keys);
  }

  void rescale_to_next(const PhantomCiphertext &encrypted,
                       PhantomCiphertext &destination) const;

  inline void rescale_to_next_inplace(PhantomCiphertext &encrypted) const {
    PhantomCiphertext destination;
    rescale_to_next(encrypted, destination);
    encrypted = std::move(destination);
  }

  void mod_switch_to_next(const PhantomCiphertext &encrypted,
                          PhantomCiphertext &destination) const;

  inline void mod_switch_to_next_inplace(PhantomCiphertext &encrypted) const {
    PhantomCiphertext destination;
    mod_switch_to_next(encrypted, destination);
    encrypted = std::move(destination);
  }

  void rotate_vector_inplace(PhantomCiphertext &encrypted, int step,
                             const PhantomGaloisKey &galois_key) const;

  // Apply an explicit automorphism.  This is needed when a host adapter has
  // a logical slot layout whose generator differs from Phantom's convenience
  // rotate_vector() helper.
  void apply_galois_inplace(PhantomCiphertext &encrypted,
                            std::size_t galois_elt,
                            const PhantomGaloisKey &galois_key) const;

  inline void rotate_vector(const PhantomCiphertext &encrypted, int step,
                            const PhantomGaloisKey &galois_key,
                            PhantomCiphertext &destination) const {
    destination = encrypted;
    rotate_vector_inplace(destination, step, galois_key);
  }

  void complex_conjugate_inplace(PhantomCiphertext &encrypted,
                                 const PhantomGaloisKey &galois_key) const;

  inline void complex_conjugate(const PhantomCiphertext &encrypted,
                                const PhantomGaloisKey &galois_key,
                                PhantomCiphertext &destination) const {
    destination = encrypted;
    complex_conjugate_inplace(destination, galois_key);
  }

  // Runtime counters are incremented at the Phantom API boundary, so they
  // describe operations that were actually dispatched to the GPU rather
  // than predictions made from a packing formula.
  void reset_operation_counters() const;
  std::uint64_t rotation_count() const;
  std::uint64_t ct_ct_multiply_count() const;
  std::uint64_t ct_pt_multiply_count() const;
  std::uint64_t relinearization_count() const;
  std::uint64_t rescale_count() const;
  std::uint64_t conjugation_count() const;

private:
  const PhantomContext &context_;
  mutable std::atomic<std::uint64_t> rotation_count_{0};
  mutable std::atomic<std::uint64_t> ct_ct_multiply_count_{0};
  mutable std::atomic<std::uint64_t> ct_pt_multiply_count_{0};
  mutable std::atomic<std::uint64_t> relinearization_count_{0};
  mutable std::atomic<std::uint64_t> rescale_count_{0};
  mutable std::atomic<std::uint64_t> conjugation_count_{0};
};

#endif // LINEARHE_EVALUATOR_WRAPPER_H_
