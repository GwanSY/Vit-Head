#include "BOLTTransformer/bolt_he.h"

#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sci {
namespace bolt {
namespace {

template <class SealObject>
void send_seal_object(NetIO *io, const SealObject &object) {
  std::stringstream stream;
  object.save(stream);
  const std::string bytes = stream.str();
  const std::uint64_t size = bytes.size();
  io->send_data(&size, sizeof(size));
  io->send_data(bytes.data(), size);
}

template <class SealObject>
void receive_seal_object(NetIO *io, seal::SEALContext &context,
                         SealObject &object) {
  std::uint64_t size = 0;
  io->recv_data(&size, sizeof(size));
  if (size == 0 || size > (1ULL << 34)) {
    throw std::runtime_error("BOLT received an invalid SEAL key size");
  }
  std::string bytes(size, '\0');
  io->recv_data(&bytes[0], size);
  std::stringstream stream(bytes);
  object.load(context, stream);
}

} // namespace

void BoltHEView::require_batching() const {
  if (context == nullptr || evaluator == nullptr || encoder == nullptr ||
      encryptor == nullptr || io == nullptr) {
    throw std::logic_error(
        "BOLT fail-closed: BFV context, BatchEncoder, evaluator, encryptor and NetIO are required");
  }
  const auto context_data = context->first_context_data();
  if (context_data == nullptr ||
      !context_data->qualifiers().using_batching ||
      encoder->slot_count() != poly_modulus_degree) {
    throw std::logic_error(
        "BOLT fail-closed: BFV BatchEncoder batching is unavailable");
  }
}

void BoltHEView::require_public_evaluation(bool need_relinearization) const {
  require_batching();
  if (galois_keys == nullptr || galois_keys->size() == 0) {
    throw std::logic_error("BOLT fail-closed: Galois keys are required");
  }
  if (need_relinearization &&
      (relinearization_keys == nullptr || relinearization_keys->size() == 0)) {
    throw std::logic_error(
        "BOLT fail-closed: relinearization keys are required");
  }
}

void BoltHEView::require_decryption() const {
  require_batching();
  if (decryptor == nullptr) {
    throw std::logic_error(
        "BOLT fail-closed: only the secret-key party may decrypt a masked output");
  }
}

BoltHE::BoltHE(int party, NetIO *io, const BoltHEParameters &parameters,
               bool use_gpu)
    : party_(party), io_(io), parameters_(parameters) {
  if (party != ALICE && party != BOB) {
    throw std::invalid_argument("BOLT party must be ALICE or BOB");
  }
  if (io == nullptr) {
    throw std::invalid_argument("BOLT requires a NetIO channel");
  }

  const auto local_setup_start = std::chrono::steady_clock::now();
  seal::EncryptionParameters parms(seal::scheme_type::bfv);
  parms.set_poly_modulus_degree(parameters.poly_modulus_degree);
  parms.set_coeff_modulus(seal::CoeffModulus::Create(
      parameters.poly_modulus_degree, parameters.coeff_modulus_bits));
  parms.set_plain_modulus(parameters.plain_modulus);
  context_.reset(new seal::SEALContext(parms, true, parameters.security_level));
  if (!context_->parameters_set()) {
    throw std::invalid_argument("BOLT BFV encryption parameters are invalid");
  }
  evaluator_.reset(new BoltEvaluator(*context_));
  encoder_.reset(new seal::BatchEncoder(*context_));

  seal::PublicKey generated_public_key;
  seal::GaloisKeys generated_galois_keys;
  seal::RelinKeys generated_relinearization_keys;
  std::unique_ptr<seal::KeyGenerator> generated_keygen;
  if (party == BOB) {
    generated_keygen.reset(new seal::KeyGenerator(*context_));
    generated_keygen->create_public_key(generated_public_key);
    generated_keygen->create_galois_keys(generated_galois_keys);
    generated_keygen->create_relin_keys(generated_relinearization_keys);
  }
  setup_stats_.context_and_key_generation_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - local_setup_start).count();
  const auto public_keys_start = std::chrono::steady_clock::now();
  const std::uint64_t public_keys_before_bytes = io_->counter;
  const std::uint64_t public_keys_before_rounds = io_->num_rounds;

  if (party == BOB) {

    send_seal_object(io_, generated_public_key);
    send_seal_object(io_, generated_galois_keys);
    send_seal_object(io_, generated_relinearization_keys);

    encryptor_.reset(new seal::Encryptor(*context_, generated_public_key));
    decryptor_.reset(new seal::Decryptor(*context_, generated_keygen->secret_key()));
    galois_keys_.reset(new seal::GaloisKeys(generated_galois_keys));
    relinearization_keys_.reset(new seal::RelinKeys(generated_relinearization_keys));
  } else {
    seal::PublicKey public_key;
    galois_keys_.reset(new seal::GaloisKeys());
    relinearization_keys_.reset(new seal::RelinKeys());
    receive_seal_object(io_, *context_, public_key);
    receive_seal_object(io_, *context_, *galois_keys_);
    receive_seal_object(io_, *context_, *relinearization_keys_);
    encryptor_.reset(new seal::Encryptor(*context_, public_key));
  }
  setup_stats_.evaluation_public_keys_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - public_keys_start).count();
  setup_stats_.evaluation_public_keys_bytes =
      io_->counter - public_keys_before_bytes;
  setup_stats_.evaluation_public_keys_rounds =
      io_->num_rounds - public_keys_before_rounds;

  if (use_gpu && party == ALICE) {
    // BOLT GPU is enabled only for BOLT's evaluator party.  BOB remains the
    // original BOLT key/decryption party and never receives BLB's model code.
    evaluator_->enable_gpu(*galois_keys_, *relinearization_keys_, 0);
  }

  view().require_public_evaluation(true);
  if (party == BOB) {
    view().require_decryption();
  }
}

BoltHEView BoltHE::view() {
  BoltHEView result;
  result.party = party_;
  result.io = io_;
  result.poly_modulus_degree = parameters_.poly_modulus_degree;
  result.plain_modulus = parameters_.plain_modulus;
  result.context = context_.get();
  result.encryptor = encryptor_.get();
  result.decryptor = decryptor_.get();
  result.evaluator = evaluator_.get();
  result.encoder = encoder_.get();
  result.galois_keys = galois_keys_.get();
  result.relinearization_keys = relinearization_keys_.get();
  return result;
}

std::size_t BoltHE::slot_count() const {
  return encoder_ == nullptr ? 0 : encoder_->slot_count();
}

} // namespace bolt
} // namespace sci
