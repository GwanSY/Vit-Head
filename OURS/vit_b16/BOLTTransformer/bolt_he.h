#ifndef SCI_BOLT_TRANSFORMER_HE_H
#define SCI_BOLT_TRANSFORMER_HE_H

#include "BOLTTransformer/bolt_metadata.h"
#include "LinearHE/evaluator-wrapper.h"
#include "utils/emp-tool.h"

#include <cstdint>
#include <memory>

namespace sci {
namespace bolt {

// Compatibility evaluator used by the ViT adapter.  The public ViT adapter
// still owns ordinary SEAL ciphertexts so that its CPU path, conversion wire
// format, and validation fixtures remain unchanged.  In a GPU build the
// evaluator uploads an operand pair to Phantom, executes the operation on the
// device, and downloads only the result.  This is intentionally a narrow
// backend boundary: it lets us validate Phantom/BFV semantics before changing
// the higher-level packed tensor representation to device-resident buffers.
class BoltEvaluator {
public:
  BoltEvaluator(seal::SEALContext *context, seal::Evaluator *host_evaluator,
                std::size_t poly_modulus_degree,
                PhantomContext *device_context = nullptr,
                PhantomEvaluator *device_evaluator = nullptr,
                PhantomRelinKey *device_relinearization_keys = nullptr,
                PhantomGaloisKey *device_galois_keys = nullptr,
                bool use_gpu = false);

  bool uses_gpu() const { return use_gpu_; }

  void add_inplace(seal::Ciphertext &encrypted1,
                   const seal::Ciphertext &encrypted2) const;
  void add(const seal::Ciphertext &encrypted1,
           const seal::Ciphertext &encrypted2,
           seal::Ciphertext &destination) const;
  void add_plain_inplace(seal::Ciphertext &encrypted,
                         const seal::Plaintext &plain) const;
  void sub_plain_inplace(seal::Ciphertext &encrypted,
                         const seal::Plaintext &plain) const;
  void multiply_plain_inplace(seal::Ciphertext &encrypted,
                              const seal::Plaintext &plain) const;
  void multiply_plain(const seal::Ciphertext &encrypted,
                      const seal::Plaintext &plain,
                      seal::Ciphertext &destination) const;
  void multiply(const seal::Ciphertext &encrypted1,
                const seal::Ciphertext &encrypted2,
                seal::Ciphertext &destination) const;
  void relinearize_inplace(seal::Ciphertext &encrypted,
                           const seal::RelinKeys &relin_keys) const;
  void mod_switch_to_next_inplace(seal::Ciphertext &encrypted) const;
  void rotate_columns(const seal::Ciphertext &encrypted,
                      const seal::GaloisKeys &galois_keys,
                      seal::Ciphertext &destination) const;

  // Apply a logical BatchEncoder row rotation.  The GPU path uses the same
  // explicit Galois element as the BLB host path, rather than Phantom's
  // generator-dependent rotate_vector convenience API.
  void rotate_rows_logical(seal::Ciphertext &encrypted, int steps,
                           const seal::GaloisKeys &galois_keys) const;

private:
  void require_gpu() const;

#ifdef BLB_VIT_USE_GPU
  PhantomCiphertext upload(const seal::Ciphertext &ciphertext) const;
  void download(PhantomCiphertext &&ciphertext,
                seal::Ciphertext &destination) const;
  PhantomPlaintext upload_plain(const seal::Plaintext &plaintext) const;
#endif

  seal::SEALContext *context_ = nullptr;
  seal::Evaluator *host_evaluator_ = nullptr;
  std::size_t poly_modulus_degree_ = 0;
  PhantomContext *device_context_ = nullptr;
  PhantomEvaluator *device_evaluator_ = nullptr;
  PhantomRelinKey *device_relinearization_keys_ = nullptr;
  PhantomGaloisKey *device_galois_keys_ = nullptr;
  bool use_gpu_ = false;
};

struct BoltHEView {
  int party = 0;
  NetIO *io = nullptr;
  std::size_t poly_modulus_degree = 0;
  std::uint64_t plain_modulus = 0;
  seal::SEALContext *context = nullptr;
  seal::Encryptor *encryptor = nullptr;
  seal::Decryptor *decryptor = nullptr;
  BoltEvaluator *evaluator = nullptr;
  seal::BatchEncoder *encoder = nullptr;
  seal::GaloisKeys *galois_keys = nullptr;
  seal::RelinKeys *relinearization_keys = nullptr;

  void require_batching() const;
  void require_public_evaluation(bool need_relinearization) const;
  void require_decryption() const;

  bool uses_gpu() const {
    return evaluator != nullptr && evaluator->uses_gpu();
  }

  const char *backend_name() const {
    return uses_gpu() ? "phantom_gpu" : "seal_cpu";
  }

  // Logical BatchEncoder row rotation.  The checked-in BLB SEAL fork uses a
  // different GaloisTool generator from BatchEncoder; calling SEAL's
  // rotate_rows directly therefore moves by the wrong logical slot count.
  // This adapter applies logical row rotations through the selected backend.
  void rotate_rows_logical(const seal::Ciphertext &input, int steps,
                           seal::Ciphertext &output) const;
};

struct BoltHESetupStats {
  double context_and_key_generation_ms = 0.0;
  double evaluation_public_keys_ms = 0.0;
  std::uint64_t evaluation_public_keys_bytes = 0;
  std::uint64_t evaluation_public_keys_rounds = 0;
};

class BoltHE {
public:
  BoltHE(int party, NetIO *io, const BoltHEParameters &parameters);
  ~BoltHE() = default;

  BoltHE(const BoltHE &) = delete;
  BoltHE &operator=(const BoltHE &) = delete;

  BoltHEView view();
  const BoltHEParameters &parameters() const { return parameters_; }
  const BoltHESetupStats &setup_stats() const { return setup_stats_; }
  std::size_t slot_count() const;
  bool uses_gpu() const { return gpu_enabled_; }

private:
  int party_;
  NetIO *io_;
  BoltHEParameters parameters_;
  std::unique_ptr<seal::SEALContext> context_;
  std::unique_ptr<seal::Encryptor> encryptor_;
  std::unique_ptr<seal::Decryptor> decryptor_;
  std::unique_ptr<seal::Evaluator> host_evaluator_;
  std::unique_ptr<BoltEvaluator> evaluator_;
  std::unique_ptr<seal::BatchEncoder> encoder_;
  std::unique_ptr<seal::GaloisKeys> galois_keys_;
  std::unique_ptr<seal::RelinKeys> relinearization_keys_;
#ifdef BLB_VIT_USE_GPU
  std::unique_ptr<PhantomContext> device_context_;
  std::unique_ptr<PhantomEvaluator> device_evaluator_;
  std::unique_ptr<PhantomRelinKey> device_relinearization_keys_;
  std::unique_ptr<PhantomGaloisKey> device_galois_keys_;
#endif
  bool gpu_enabled_ = false;
  BoltHESetupStats setup_stats_;
};

} // namespace bolt
} // namespace sci

#endif
