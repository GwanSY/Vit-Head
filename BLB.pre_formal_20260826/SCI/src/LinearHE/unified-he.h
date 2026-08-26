#ifndef LINEARHE_UNIFIED_HE_H_
#define LINEARHE_UNIFIED_HE_H_

#include "evaluator-wrapper.h"
#include <seal/ciphertext.h>
#include <seal/evaluator.h>
#include <seal/galoiskeys.h>
#include <seal/plaintext.h>
#include <seal/publickey.h>

enum LOCATION { HOST = 0, DEVICE, HOST_AND_DEVICE, UNDEF };

// ******************** UnifiedPlaintext ********************

class UnifiedPlaintext {
public:
  UnifiedPlaintext(LOCATION loc = UNDEF) : loc_(loc) {}

  UnifiedPlaintext(const seal::Plaintext &hplain);

  UnifiedPlaintext(seal::Plaintext &&hplain);

  UnifiedPlaintext(const PhantomPlaintext &dplain);

  UnifiedPlaintext(PhantomPlaintext &&dplain);

  ~UnifiedPlaintext() = default;

  UnifiedPlaintext &operator=(seal::Plaintext &&);

  const seal::Plaintext &hplain() const {
    if (loc_ == HOST || loc_ == HOST_AND_DEVICE) {
      return host_plain_;
    }
    throw std::runtime_error("UnifiedPlaintext: NOT in HOST");
  }
  seal::Plaintext &hplain() {
    if (loc_ == HOST || loc_ == HOST_AND_DEVICE) {
      return host_plain_;
    }
    throw std::runtime_error("UnifiedPlaintext: NOT in HOST");
  }
  const PhantomPlaintext &dplain() const {
    if (loc_ == DEVICE || loc_ == HOST_AND_DEVICE) {
      return device_plain_;
    }
    throw std::runtime_error("UnifiedPlaintext: NOT in DEVICE");
  }
  PhantomPlaintext &dplain() {
    if (loc_ == DEVICE || loc_ == HOST_AND_DEVICE) {
      return device_plain_;
    }
    throw std::runtime_error("UnifiedPlaintext: NOT in DEVICE");
  }

  operator const seal::Plaintext &() const { return hplain(); }
  operator seal::Plaintext &() { return hplain(); }
  operator const PhantomPlaintext &() const { return dplain(); }
  operator PhantomPlaintext &() { return dplain(); }

  static void to_device(const seal::SEALContext &hcontext,
                        const seal::Plaintext &hplain,
                        const PhantomContext &dcontext,
                        PhantomPlaintext &dplain);
  void to_device(const seal::SEALContext &hcontext,
                 const PhantomContext &dcontext);

  const double &scale() const;
  double &scale();

private:
  LOCATION loc_ = UNDEF;
  seal::Plaintext host_plain_;
  PhantomPlaintext device_plain_;
};

// ******************** UnifiedCiphertext ********************

class UnifiedCiphertext {
public:
  UnifiedCiphertext(LOCATION loc = UNDEF);

  UnifiedCiphertext(const seal::Ciphertext &cipher);

  UnifiedCiphertext(seal::Ciphertext &&cipher);

  UnifiedCiphertext(const PhantomCiphertext &cipher);

  UnifiedCiphertext(PhantomCiphertext &&cipher);

  ~UnifiedCiphertext() = default;

  LOCATION location() const { return loc_; }

  bool is_host() const { return loc_ == HOST; }
  bool is_device() const { return loc_ == DEVICE; }

  const seal::Ciphertext &hcipher() const {
    if (loc_ != HOST) {
      throw std::runtime_error("UnifiedCiphertext: NOT in HOST");
    }
    return host_cipher_;
  }
  seal::Ciphertext &hcipher() {
    if (loc_ != HOST) {
      throw std::runtime_error("UnifiedCiphertext: NOT in HOST");
    }
    return host_cipher_;
  }
  const PhantomCiphertext &dcipher() const {
    if (loc_ != DEVICE) {
      throw std::runtime_error("UnifiedCiphertext: NOT in DEVICE");
    }
    return device_cipher_;
  }
  PhantomCiphertext &dcipher() {
    if (loc_ != DEVICE) {
      throw std::runtime_error("UnifiedCiphertext: NOT in DEVICE");
    }
    return device_cipher_;
  }

  static void to_device(const seal::SEALContext &hcontext,
                        const seal::Ciphertext &hcipher,
                        const PhantomContext &dcontext,
                        PhantomCiphertext &dcipher);
  void to_device(const seal::SEALContext &hcontext,
                 const PhantomContext &dcontext);

  operator const seal::Ciphertext &() const { return hcipher(); }
  operator seal::Ciphertext &() { return hcipher(); }

  static void to_host(const PhantomContext &dcontext,
                      const PhantomCiphertext &dcipher,
                      const seal::SEALContext &hcontext,
                      seal::Ciphertext &hcipher);
  void to_host(const seal::SEALContext &hcontext,
               const PhantomContext &dcontext);

  operator const PhantomCiphertext &() const { return dcipher(); }
  operator PhantomCiphertext &() { return dcipher(); }

  // Unified API for SEAL & Phantom
  void save(std::ostream &stream) const;
  void load(const seal::SEALContext &hcontext, const PhantomContext &dcontext,
            std::istream &stream);

  std::size_t coeff_modulus_size() const;
  const double &scale() const;
  double &scale();

private:
  LOCATION loc_ = UNDEF;

  seal::Ciphertext host_cipher_;
  PhantomCiphertext device_cipher_;
};

// TODO: ******************** UnifiedPublicKey ********************

void kswitchkey_to_device(const seal::SEALContext &hcontext,
                          const std::vector<seal::PublicKey> &hksk,
                          const PhantomContext &dcontext,
                          std::vector<PhantomPublicKey> &dksk);

void kswitchkey_to_device(const seal::SEALContext &hcontext,
                          const std::vector<seal::PublicKey> &hksk,
                          const PhantomContext &dcontext,
                          PhantomRelinKey &dksk);

void galoiskeys_to_device(const seal::SEALContext &hcontext,
                          const seal::GaloisKeys &hks,
                          const PhantomContext &dcontext,
                          PhantomGaloisKey &dks);

// ******************** UnifiedEvaluator ********************

class UnifiedEvaluator {
public:
  UnifiedEvaluator(const seal::SEALContext &hcontext,
                   const PhantomContext &dcontext);

  ~UnifiedEvaluator();

private:
  seal::Evaluator *hevaluator_ = nullptr;
  PhantomEvaluator *devaluator_ = nullptr;
};

#endif // LINEARHE_UNIFIED_HE_H_
