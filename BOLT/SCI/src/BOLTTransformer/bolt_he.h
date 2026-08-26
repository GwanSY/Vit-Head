#ifndef SCI_BOLT_TRANSFORMER_HE_H
#define SCI_BOLT_TRANSFORMER_HE_H

#include "BOLTTransformer/bolt_evaluator.h"
#include "BOLTTransformer/bolt_metadata.h"
#include "utils/emp-tool.h"

#include <memory>

namespace sci {
namespace bolt {

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
};

struct BoltHESetupStats {
  double context_and_key_generation_ms = 0.0;
  double evaluation_public_keys_ms = 0.0;
  std::uint64_t evaluation_public_keys_bytes = 0;
  std::uint64_t evaluation_public_keys_rounds = 0;
};

class BoltHE {
public:
  BoltHE(int party, NetIO *io, const BoltHEParameters &parameters,
         bool use_gpu = false);
  ~BoltHE() = default;

  BoltHE(const BoltHE &) = delete;
  BoltHE &operator=(const BoltHE &) = delete;

  BoltHEView view();
  const BoltHEParameters &parameters() const { return parameters_; }
  const BoltHESetupStats &setup_stats() const { return setup_stats_; }
  std::size_t slot_count() const;

private:
  int party_;
  NetIO *io_;
  BoltHEParameters parameters_;
  std::unique_ptr<seal::SEALContext> context_;
  std::unique_ptr<seal::Encryptor> encryptor_;
  std::unique_ptr<seal::Decryptor> decryptor_;
  std::unique_ptr<BoltEvaluator> evaluator_;
  std::unique_ptr<seal::BatchEncoder> encoder_;
  std::unique_ptr<seal::GaloisKeys> galois_keys_;
  std::unique_ptr<seal::RelinKeys> relinearization_keys_;
  BoltHESetupStats setup_stats_;
};

} // namespace bolt
} // namespace sci

#endif
