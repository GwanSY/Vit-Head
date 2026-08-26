#include "BOLTTransformer/bolt_he.h"

#include "LinearHE/unified-he.h"

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef BLB_VIT_USE_GPU
#include <cuda_runtime_api.h>
#endif

namespace sci {
namespace bolt {
namespace {

std::uint32_t pow_mod_power_of_two(std::uint32_t base,
                                   std::size_t exponent,
                                   std::uint32_t modulus) {
  std::uint64_t result = 1;
  std::uint64_t factor = base;
  while (exponent != 0) {
    if ((exponent & 1U) != 0) result = (result * factor) % modulus;
    factor = (factor * factor) % modulus;
    exponent >>= 1U;
  }
  return static_cast<std::uint32_t>(result);
}

std::uint32_t logical_galois_element(std::size_t poly_modulus_degree,
                                     std::size_t left_steps) {
  const std::size_t row_size = poly_modulus_degree / 2;
  const std::uint32_t m =
      static_cast<std::uint32_t>(2 * poly_modulus_degree);
  return pow_mod_power_of_two(3, left_steps % row_size, m);
}

std::vector<std::uint32_t>
logical_batching_galois_elements(std::size_t poly_modulus_degree) {
  const std::size_t row_size = poly_modulus_degree / 2;
  const std::uint32_t m =
      static_cast<std::uint32_t>(2 * poly_modulus_degree);
  std::vector<std::uint32_t> elements{m - 1}; // rotate_columns
  for (std::size_t step = 1; step < row_size; step <<= 1) {
    elements.push_back(logical_galois_element(poly_modulus_degree, step));
    elements.push_back(logical_galois_element(poly_modulus_degree,
                                              row_size - step));
  }
  std::sort(elements.begin(), elements.end());
  elements.erase(std::unique(elements.begin(), elements.end()),
                 elements.end());
  return elements;
}

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

BoltEvaluator::BoltEvaluator(
    seal::SEALContext *context, seal::Evaluator *host_evaluator,
    std::size_t poly_modulus_degree, PhantomContext *device_context,
    PhantomEvaluator *device_evaluator,
    PhantomRelinKey *device_relinearization_keys,
    PhantomGaloisKey *device_galois_keys, bool use_gpu)
    : context_(context), host_evaluator_(host_evaluator),
      poly_modulus_degree_(poly_modulus_degree),
      device_context_(device_context), device_evaluator_(device_evaluator),
      device_relinearization_keys_(device_relinearization_keys),
      device_galois_keys_(device_galois_keys), use_gpu_(use_gpu) {
  if (context_ == nullptr || host_evaluator_ == nullptr ||
      poly_modulus_degree_ == 0) {
    throw std::invalid_argument("BOLT evaluator received invalid host state");
  }
#ifdef BLB_VIT_USE_GPU
  if (use_gpu_) require_gpu();
#else
  if (use_gpu_) {
    throw std::logic_error(
        "BOLT GPU evaluator requested from a CPU-only build");
  }
#endif
}

void BoltEvaluator::require_gpu() const {
#ifdef BLB_VIT_USE_GPU
  if (!use_gpu_ || device_context_ == nullptr ||
      device_evaluator_ == nullptr || device_relinearization_keys_ == nullptr ||
      device_galois_keys_ == nullptr) {
    throw std::logic_error(
        "BOLT Phantom evaluator is not fully initialized");
  }
#else
  throw std::logic_error("BOLT Phantom evaluator is unavailable");
#endif
}

#ifdef BLB_VIT_USE_GPU
PhantomCiphertext BoltEvaluator::upload(
    const seal::Ciphertext &ciphertext) const {
  require_gpu();
  UnifiedCiphertext unified(ciphertext);
  unified.to_device(*context_, *device_context_);
  return std::move(unified.dcipher());
}

void BoltEvaluator::download(PhantomCiphertext &&ciphertext,
                             seal::Ciphertext &destination) const {
  require_gpu();
  UnifiedCiphertext unified(std::move(ciphertext));
  unified.to_host(*context_, *device_context_);
  destination = std::move(unified.hcipher());
}

PhantomPlaintext BoltEvaluator::upload_plain(
    const seal::Plaintext &plaintext) const {
  require_gpu();
  const std::size_t degree = poly_modulus_degree_;
  if (plaintext.coeff_count() > degree || plaintext.is_ntt_form()) {
    throw std::invalid_argument(
        "BOLT Phantom BFV plaintext must be a coefficient polynomial within the ring degree");
  }
  std::vector<std::uint64_t> host_coefficients(degree, 0);
  std::copy(plaintext.data(), plaintext.data() + plaintext.coeff_count(),
            host_coefficients.begin());
  PhantomPlaintext result;
  result.resize(1, degree, cudaStreamPerThread);
  result.set_chain_index(0);
  PHANTOM_CHECK_CUDA(cudaMemcpy(
      result.data(), host_coefficients.data(),
      degree * sizeof(std::uint64_t),
      cudaMemcpyHostToDevice));
  return result;
}
#endif

void BoltEvaluator::add_inplace(seal::Ciphertext &encrypted1,
                                const seal::Ciphertext &encrypted2) const {
  if (!use_gpu_) {
    host_evaluator_->add_inplace(encrypted1, encrypted2);
    return;
  }
#ifdef BLB_VIT_USE_GPU
  PhantomCiphertext d1 = upload(encrypted1);
  PhantomCiphertext d2 = upload(encrypted2);
  device_evaluator_->add_inplace(d1, d2);
  download(std::move(d1), encrypted1);
#else
  throw std::logic_error("BOLT Phantom evaluator is unavailable");
#endif
}

void BoltEvaluator::add(const seal::Ciphertext &encrypted1,
                        const seal::Ciphertext &encrypted2,
                        seal::Ciphertext &destination) const {
  if (!use_gpu_) {
    host_evaluator_->add(encrypted1, encrypted2, destination);
    return;
  }
#ifdef BLB_VIT_USE_GPU
  PhantomCiphertext d1 = upload(encrypted1);
  PhantomCiphertext d2 = upload(encrypted2);
  PhantomCiphertext out;
  device_evaluator_->add(d1, d2, out);
  download(std::move(out), destination);
#else
  throw std::logic_error("BOLT Phantom evaluator is unavailable");
#endif
}

void BoltEvaluator::add_plain_inplace(seal::Ciphertext &encrypted,
                                      const seal::Plaintext &plain) const {
  if (!use_gpu_) {
    host_evaluator_->add_plain_inplace(encrypted, plain);
    return;
  }
#ifdef BLB_VIT_USE_GPU
  PhantomCiphertext d = upload(encrypted);
  PhantomPlaintext p = upload_plain(plain);
  device_evaluator_->add_plain_inplace(d, p);
  download(std::move(d), encrypted);
#else
  throw std::logic_error("BOLT Phantom evaluator is unavailable");
#endif
}

void BoltEvaluator::sub_plain_inplace(seal::Ciphertext &encrypted,
                                      const seal::Plaintext &plain) const {
  if (!use_gpu_) {
    host_evaluator_->sub_plain_inplace(encrypted, plain);
    return;
  }
#ifdef BLB_VIT_USE_GPU
  PhantomCiphertext d = upload(encrypted);
  PhantomPlaintext p = upload_plain(plain);
  device_evaluator_->sub_plain_inplace(d, p);
  download(std::move(d), encrypted);
#else
  throw std::logic_error("BOLT Phantom evaluator is unavailable");
#endif
}

void BoltEvaluator::multiply_plain_inplace(
    seal::Ciphertext &encrypted, const seal::Plaintext &plain) const {
  if (!use_gpu_) {
    host_evaluator_->multiply_plain_inplace(encrypted, plain);
    return;
  }
#ifdef BLB_VIT_USE_GPU
  PhantomCiphertext d = upload(encrypted);
  PhantomPlaintext p = upload_plain(plain);
  device_evaluator_->multiply_plain_inplace(d, p);
  download(std::move(d), encrypted);
#else
  throw std::logic_error("BOLT Phantom evaluator is unavailable");
#endif
}

void BoltEvaluator::multiply_plain(const seal::Ciphertext &encrypted,
                                    const seal::Plaintext &plain,
                                    seal::Ciphertext &destination) const {
  if (!use_gpu_) {
    host_evaluator_->multiply_plain(encrypted, plain, destination);
    return;
  }
#ifdef BLB_VIT_USE_GPU
  PhantomCiphertext d = upload(encrypted);
  PhantomPlaintext p = upload_plain(plain);
  PhantomCiphertext out;
  device_evaluator_->multiply_plain(d, p, out);
  download(std::move(out), destination);
#else
  throw std::logic_error("BOLT Phantom evaluator is unavailable");
#endif
}

void BoltEvaluator::multiply(const seal::Ciphertext &encrypted1,
                             const seal::Ciphertext &encrypted2,
                             seal::Ciphertext &destination) const {
  if (!use_gpu_) {
    host_evaluator_->multiply(encrypted1, encrypted2, destination);
    return;
  }
#ifdef BLB_VIT_USE_GPU
  PhantomCiphertext d1 = upload(encrypted1);
  PhantomCiphertext d2 = upload(encrypted2);
  PhantomCiphertext out;
  device_evaluator_->multiply(d1, d2, out);
  download(std::move(out), destination);
#else
  throw std::logic_error("BOLT Phantom evaluator is unavailable");
#endif
}

void BoltEvaluator::relinearize_inplace(
    seal::Ciphertext &encrypted, const seal::RelinKeys &relin_keys) const {
  if (!use_gpu_) {
    host_evaluator_->relinearize_inplace(encrypted, relin_keys);
    return;
  }
#ifdef BLB_VIT_USE_GPU
  (void)relin_keys;
  PhantomCiphertext d = upload(encrypted);
  device_evaluator_->relinearize_inplace(d, *device_relinearization_keys_);
  download(std::move(d), encrypted);
#else
  throw std::logic_error("BOLT Phantom evaluator is unavailable");
#endif
}

void BoltEvaluator::mod_switch_to_next_inplace(
    seal::Ciphertext &encrypted) const {
  if (!use_gpu_) {
    host_evaluator_->mod_switch_to_next_inplace(encrypted);
    return;
  }
#ifdef BLB_VIT_USE_GPU
  PhantomCiphertext d = upload(encrypted);
  device_evaluator_->mod_switch_to_next_inplace(d);
  download(std::move(d), encrypted);
#else
  throw std::logic_error("BOLT Phantom evaluator is unavailable");
#endif
}

void BoltEvaluator::rotate_columns(const seal::Ciphertext &encrypted,
                                   const seal::GaloisKeys &galois_keys,
                                   seal::Ciphertext &destination) const {
  if (!use_gpu_) {
    host_evaluator_->rotate_columns(encrypted, galois_keys, destination);
    return;
  }
#ifdef BLB_VIT_USE_GPU
  (void)galois_keys;
  PhantomCiphertext d = upload(encrypted);
  device_evaluator_->complex_conjugate_inplace(d, *device_galois_keys_);
  download(std::move(d), destination);
#else
  throw std::logic_error("BOLT Phantom evaluator is unavailable");
#endif
}

void BoltEvaluator::rotate_rows_logical(
    seal::Ciphertext &encrypted, int steps,
    const seal::GaloisKeys &galois_keys) const {
  const int row_size = static_cast<int>(poly_modulus_degree_ / 2);
  if (steps <= -row_size || steps >= row_size) {
    throw std::invalid_argument("BOLT logical BFV rotation step is out of range");
  }
  if (steps == 0) return;
  if (use_gpu_) {
#ifdef BLB_VIT_USE_GPU
    (void)galois_keys;
    PhantomCiphertext d = upload(encrypted);
    std::size_t remaining = static_cast<std::size_t>(
        steps < 0 ? -static_cast<std::int64_t>(steps) : steps);
    std::size_t power = 1;
    while (remaining != 0) {
      if ((remaining & 1U) != 0) {
        const std::size_t left_steps =
            steps > 0 ? power : static_cast<std::size_t>(row_size) - power;
        const std::uint32_t element =
            logical_galois_element(poly_modulus_degree_, left_steps);
        device_evaluator_->apply_galois_inplace(
            d, element, *device_galois_keys_);
      }
      remaining >>= 1U;
      power <<= 1U;
    }
    download(std::move(d), encrypted);
    return;
#else
    throw std::logic_error("BOLT Phantom evaluator is unavailable");
#endif
  }

  std::size_t remaining = static_cast<std::size_t>(
      steps < 0 ? -static_cast<std::int64_t>(steps) : steps);
  std::size_t power = 1;
  while (remaining != 0) {
    if ((remaining & 1U) != 0) {
      const std::size_t left_steps =
          steps > 0 ? power : static_cast<std::size_t>(row_size) - power;
      const std::uint32_t element =
          logical_galois_element(poly_modulus_degree_, left_steps);
      if (!galois_keys.has_key(element)) {
        throw std::logic_error(
            "BOLT fail-closed: logical BatchEncoder Galois key is missing");
      }
      host_evaluator_->apply_galois_inplace(encrypted, element, galois_keys);
    }
    remaining >>= 1U;
    power <<= 1U;
  }
}

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

void BoltHEView::rotate_rows_logical(const seal::Ciphertext &input, int steps,
                                     seal::Ciphertext &output) const {
  require_public_evaluation(false);
  output = input;
  evaluator->rotate_rows_logical(output, steps, *galois_keys);
}

BoltHE::BoltHE(int party, NetIO *io, const BoltHEParameters &parameters)
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
  host_evaluator_.reset(new seal::Evaluator(*context_));
  encoder_.reset(new seal::BatchEncoder(*context_));

#ifdef BLB_VIT_USE_GPU
  // Only ALICE evaluates HE ciphertexts in the two-party ViT protocol. BOB
  // still builds the same host key set and performs host encryption/decryption
  // so that the CPU and GPU binaries retain the same wire format.
  gpu_enabled_ = (party_ == ALICE);
  if (gpu_enabled_) {
    phantom::EncryptionParameters device_parms(phantom::scheme_type::bfv);
    device_parms.set_poly_modulus_degree(parameters.poly_modulus_degree);
    device_parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
        parameters.poly_modulus_degree, parameters.coeff_modulus_bits));
    device_parms.set_plain_modulus(
        phantom::arith::Modulus(parameters.plain_modulus));
    // Phantom's default rotate_vector helper uses generator 5.  BLB's
    // BatchEncoder matrix representation uses generator 3, so install the
    // explicit BLB element set and call apply_galois_inplace below.
    device_parms.set_galois_elts(
        logical_batching_galois_elements(parameters.poly_modulus_degree));
    device_context_.reset(new PhantomContext(device_parms));
    device_evaluator_.reset(new PhantomEvaluator(*device_context_));
  }
#endif

  seal::PublicKey generated_public_key;
  seal::GaloisKeys generated_galois_keys;
  seal::RelinKeys generated_relinearization_keys;
  std::unique_ptr<seal::KeyGenerator> generated_keygen;
  if (party == BOB) {
    generated_keygen.reset(new seal::KeyGenerator(*context_));
    generated_keygen->create_public_key(generated_public_key);
    auto galois_elements =
        logical_batching_galois_elements(parameters.poly_modulus_degree);
    generated_keygen->create_galois_keys(galois_elements,
                                         generated_galois_keys);
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

#ifdef BLB_VIT_USE_GPU
  if (gpu_enabled_) {
    device_relinearization_keys_.reset(new PhantomRelinKey());
    kswitchkey_to_device(*context_, relinearization_keys_->key(2),
                         *device_context_, *device_relinearization_keys_);
    device_galois_keys_.reset(new PhantomGaloisKey());
    galoiskeys_to_device(*context_, *galois_keys_, *device_context_,
                         *device_galois_keys_);
  }
#endif

  PhantomContext *device_context = nullptr;
  PhantomEvaluator *device_evaluator = nullptr;
  PhantomRelinKey *device_relinearization_keys = nullptr;
  PhantomGaloisKey *device_galois_keys = nullptr;
#ifdef BLB_VIT_USE_GPU
  device_context = device_context_.get();
  device_evaluator = device_evaluator_.get();
  device_relinearization_keys = device_relinearization_keys_.get();
  device_galois_keys = device_galois_keys_.get();
#endif
  evaluator_.reset(new BoltEvaluator(
      context_.get(), host_evaluator_.get(), parameters.poly_modulus_degree,
      device_context, device_evaluator, device_relinearization_keys,
      device_galois_keys, gpu_enabled_));
  setup_stats_.evaluation_public_keys_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - public_keys_start).count();
  setup_stats_.evaluation_public_keys_bytes =
      io_->counter - public_keys_before_bytes;
  setup_stats_.evaluation_public_keys_rounds =
      io_->num_rounds - public_keys_before_rounds;

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
