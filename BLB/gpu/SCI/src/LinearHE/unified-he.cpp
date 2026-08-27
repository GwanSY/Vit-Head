#include "unified-he.h"

#include <sstream>

namespace {

void load_phantom_public_key(PhantomCiphertext &&cipher,
                             PhantomPublicKey &public_key) {
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  cipher.save(stream);
  stream.seekg(0);
  public_key.load(stream);
}

void load_phantom_relin_key(std::vector<PhantomPublicKey> &public_keys,
                            PhantomRelinKey &relin_key) {
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  const std::size_t key_count = public_keys.size();
  stream.write(reinterpret_cast<const char *>(&key_count), sizeof(key_count));
  for (const auto &public_key : public_keys) {
    public_key.save(stream);
  }
  stream.seekg(0);
  relin_key.load(stream);
}

} // namespace

// ******************** UnifiedPlaintext ********************

UnifiedPlaintext::UnifiedPlaintext(const seal::Plaintext &hplain)
    : loc_(HOST), host_plain_(hplain) {}

UnifiedPlaintext::UnifiedPlaintext(seal::Plaintext &&hplain)
    : loc_(HOST), host_plain_(std::move(hplain)) {}

UnifiedPlaintext::UnifiedPlaintext(const PhantomPlaintext &dplain)
    : loc_(DEVICE), device_plain_(dplain) {}

UnifiedPlaintext::UnifiedPlaintext(PhantomPlaintext &&dplain)
    : loc_(DEVICE), device_plain_(std::move(dplain)) {}

UnifiedPlaintext &UnifiedPlaintext::operator=(seal::Plaintext &&other) {
  loc_ = HOST;
  host_plain_ = std::move(other);
  return *this;
}

void UnifiedPlaintext::to_device(const seal::SEALContext &hcontext,
                                 const seal::Plaintext &hplain,
                                 const PhantomContext &dcontext,
                                 PhantomPlaintext &dplain) {
  const auto &first_parms = hcontext.first_context_data()->parms();
  const auto full_data_modulus_size = first_parms.coeff_modulus().size();
  const auto &curr_parms =
      hcontext.get_context_data(hplain.parms_id())->parms();
  auto curr_data_modulus_size = curr_parms.coeff_modulus().size();

  // * [Important] phantom.chain_index + seal.chain_index = size_Q
  // ckks_ntt can leave a plaintext at SEAL's key-context RNS base, which
  // contains one special prime more than the first data context.  Phantom
  // has no key-context plaintext chain; such a plaintext is the first data
  // level and must not be mapped to chain 0.
  if (curr_data_modulus_size > full_data_modulus_size) {
    if (curr_data_modulus_size != full_data_modulus_size + 1) {
      throw std::runtime_error(
          "UnifiedPlaintext: unsupported plaintext RNS level");
    }
    curr_data_modulus_size = full_data_modulus_size;
  }
  auto phantom_chain_idx = full_data_modulus_size - curr_data_modulus_size + 1;

  const auto &dparms = dcontext.get_context_data(phantom_chain_idx).parms();
  const auto coeff_modulus_size = dparms.coeff_modulus().size();
  const auto poly_modulus_degree = dparms.poly_modulus_degree();
  const auto coeff_count = coeff_modulus_size * poly_modulus_degree;

  dplain.resize(coeff_modulus_size, poly_modulus_degree,
                cudaStreamPerThread);
  dplain.set_chain_index(phantom_chain_idx);
  const_cast<double &>(dplain.scale()) = hplain.scale();
  PHANTOM_CHECK_CUDA(cudaMemcpy(dplain.data(), hplain.data(),
                                coeff_count * sizeof(std::uint64_t),
                                cudaMemcpyHostToDevice));
}

void UnifiedPlaintext::to_device(const seal::SEALContext &hcontext,
                                 const PhantomContext &dcontext) {
  if (loc_ != HOST) {
    throw std::runtime_error("UnifiedPlaintext: NOT in HOST");
  }
  to_device(hcontext, host_plain_, dcontext, device_plain_);
  host_plain_.release();
  loc_ = DEVICE;
}

const double &UnifiedPlaintext::scale() const {
  switch (loc_) {
  case HOST_AND_DEVICE:
  case HOST:
    return host_plain_.scale();
  case DEVICE:
    return device_plain_.scale();
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedPlaintext");
  }
}

double &UnifiedPlaintext::scale() {
  switch (loc_) {
  case HOST_AND_DEVICE:
  case HOST:
    return host_plain_.scale();
  case DEVICE:
    return const_cast<double &>(device_plain_.scale());
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedPlaintext");
  }
}

// ******************** UnifiedCiphertext ********************

UnifiedCiphertext::UnifiedCiphertext(LOCATION loc) : loc_(loc) {}

UnifiedCiphertext::UnifiedCiphertext(const seal::Ciphertext &cipher)
    : loc_(HOST), host_cipher_(cipher) {}

UnifiedCiphertext::UnifiedCiphertext(seal::Ciphertext &&cipher)
    : loc_(HOST), host_cipher_(std::move(cipher)) {}

UnifiedCiphertext::UnifiedCiphertext(const PhantomCiphertext &cipher)
    : loc_(DEVICE), device_cipher_(cipher) {}

UnifiedCiphertext::UnifiedCiphertext(PhantomCiphertext &&cipher)
    : loc_(DEVICE), device_cipher_(std::move(cipher)) {}

void UnifiedCiphertext::to_device(const seal::SEALContext &hcontext,
                                  const seal::Ciphertext &hcipher,
                                  const PhantomContext &dcontext,
                                  PhantomCiphertext &dcipher) {
  const auto &first_parms = hcontext.first_context_data()->parms();
  const auto full_data_modulus_size = first_parms.coeff_modulus().size();
  const auto curr_data_modulus_size = hcipher.coeff_modulus_size();

  // * [Important] phantom.chain_index + seal.chain_index = size_Q
  auto phantom_chain_idx =
      (full_data_modulus_size + 1) - curr_data_modulus_size;

  const auto &dparms = dcontext.get_context_data(phantom_chain_idx).parms();
  const auto coeff_modulus_size = dparms.coeff_modulus().size();
  const auto poly_modulus_degree = dparms.poly_modulus_degree();
  const auto coeff_count =
      hcipher.size() * coeff_modulus_size * poly_modulus_degree;

  dcipher.resize(dcontext, phantom_chain_idx, hcipher.size(),
                 cudaStreamPerThread);
  dcipher.set_scale(hcipher.scale());
  dcipher.set_correction_factor(hcipher.correction_factor());
  dcipher.SetNoiseScaleDeg(1);
  dcipher.set_ntt_form(hcipher.is_ntt_form());
  PHANTOM_CHECK_CUDA(cudaMemcpy(dcipher.data(), hcipher.data(),
                                coeff_count * sizeof(std::uint64_t),
                                cudaMemcpyHostToDevice));
}

void UnifiedCiphertext::to_device(const seal::SEALContext &hcontext,
                                  const PhantomContext &dcontext) {
  if (loc_ != HOST) {
    throw std::runtime_error("UnifiedCiphertext: NOT in HOST");
  }
  to_device(hcontext, host_cipher_, dcontext, device_cipher_);
  host_cipher_.release();
  loc_ = DEVICE;
}

void UnifiedCiphertext::to_host(const PhantomContext &dcontext,
                                const PhantomCiphertext &dcipher,
                                const seal::SEALContext &hcontext,
                                seal::Ciphertext &hcipher) {
  const auto chain_idx = dcipher.chain_index();
  const auto curr_data_modulus_size = dcipher.coeff_modulus_size();

  // PhantomCipher only holds the `chain_index`.
  // But SEAL indexes context_data with `param_id`.
  // * [Important] phantom.chain_index + seal.chain_index = size_Q
  // * Match context through `data_modulus_size`
  auto target_context = hcontext.first_context_data();
  while (target_context->parms().coeff_modulus().size() !=
         curr_data_modulus_size) {
    target_context = target_context->next_context_data();
    if (!target_context) {
      throw std::invalid_argument(
          "Phantom ciphertext modulus level is absent from the SEAL context");
    }
  }
  const auto &parms = dcontext.get_context_data(chain_idx).parms();
  const auto coeff_modulus_size = parms.coeff_modulus().size();
  const auto poly_modulus_degree = parms.poly_modulus_degree();
  const auto coeff_count =
      dcipher.size() * coeff_modulus_size * poly_modulus_degree;

  hcipher.resize(hcontext, target_context->parms_id(), dcipher.size());
  hcipher.scale() = dcipher.scale();
  hcipher.correction_factor() = dcipher.correction_factor();
  hcipher.is_ntt_form() = dcipher.is_ntt_form();
  PHANTOM_CHECK_CUDA(cudaMemcpy(hcipher.data(), dcipher.data(),
                                coeff_count * sizeof(std::uint64_t),
                                cudaMemcpyDeviceToHost));
}

void UnifiedCiphertext::to_host(const seal::SEALContext &hcontext,
                                const PhantomContext &dcontext) {
  if (loc_ != DEVICE) {
    throw std::runtime_error("UnifiedCiphertext: NOT in DEVICE");
  }
  const auto chain_idx = device_cipher_.chain_index();

  to_host(dcontext, device_cipher_, hcontext, host_cipher_);
  // device_cipher_.resize(0, 0, 0, cudaStreamPerThread);
  loc_ = HOST;
}

void UnifiedCiphertext::save(std::ostream &stream) const {
  if (loc_ == UNDEF) {
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
  std::size_t loc = loc_;
  stream.write(reinterpret_cast<const char *>(&loc), sizeof(std::size_t));
  switch (loc_) {
  case HOST:
    host_cipher_.save(stream);
    break;
  case DEVICE:
    device_cipher_.save(stream);
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

void UnifiedCiphertext::load(const seal::SEALContext &hcontext,
                             const PhantomContext &dcontext,
                             std::istream &stream) {
  stream.read(reinterpret_cast<char *>(&loc_), sizeof(std::size_t));
  switch (loc_) {
  case HOST:
    // FIXME: why using unsafe_load
    host_cipher_.unsafe_load(hcontext, stream);
    break;
  case DEVICE:
    device_cipher_.load(stream);
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

std::size_t UnifiedCiphertext::coeff_modulus_size() const {
  switch (loc_) {
  case HOST:
    return host_cipher_.coeff_modulus_size();
  case DEVICE:
    return device_cipher_.coeff_modulus_size();
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

const double &UnifiedCiphertext::scale() const {
  switch (loc_) {
  case HOST:
    return host_cipher_.scale();
  case DEVICE:
    return device_cipher_.scale();
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

double &UnifiedCiphertext::scale() {
  switch (loc_) {
  case HOST:
    return host_cipher_.scale();
  case DEVICE:
    return const_cast<double &>(device_cipher_.scale());
    break;
  default:
    throw std::invalid_argument("Invalid UnifiedCiphertext");
  }
}

void kswitchkey_to_device(const seal::SEALContext &hcontext,
                          const std::vector<seal::PublicKey> &hksk,
                          const PhantomContext &dcontext,
                          std::vector<PhantomPublicKey> &dksk) {
  auto dnum = hksk.size();
  dksk.resize(dnum);
  for (size_t i = 0; i < hksk.size(); i++) {
    PhantomCiphertext tmp;
    UnifiedCiphertext::to_device(hcontext, hksk[i].data(), dcontext, tmp);
    load_phantom_public_key(std::move(tmp), dksk[i]);
  }
}

void kswitchkey_to_device(const seal::SEALContext &hcontext,
                          const std::vector<seal::PublicKey> &hksk,
                          const PhantomContext &dcontext,
                          PhantomRelinKey &dksk) {
  std::vector<PhantomPublicKey> public_keys;
  kswitchkey_to_device(hcontext, hksk, dcontext, public_keys);
  load_phantom_relin_key(public_keys, dksk);
}

void galoiskeys_to_device(const seal::SEALContext &hcontext,
                          const seal::GaloisKeys &hks,
                          const PhantomContext &dcontext,
                          PhantomGaloisKey &dks) {
  const auto &galois_elts = dcontext.key_galois_tool_->galois_elts();
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  const std::size_t key_count = galois_elts.size();
  stream.write(reinterpret_cast<const char *>(&key_count), sizeof(key_count));

  for (const auto galois_elt : galois_elts) {
    if (!hks.has_key(galois_elt)) {
      throw std::invalid_argument(
          "SEAL GaloisKeys is missing an element required by Phantom");
    }

    std::vector<PhantomPublicKey> public_keys;
    kswitchkey_to_device(hcontext, hks.key(galois_elt), dcontext,
                         public_keys);
    const std::size_t decomposition_count = public_keys.size();
    stream.write(reinterpret_cast<const char *>(&decomposition_count),
                 sizeof(decomposition_count));
    for (const auto &public_key : public_keys) {
      public_key.save(stream);
    }
  }

  stream.seekg(0);
  dks.load(stream);
}

// ******************** UnifiedEvaluator ********************

UnifiedEvaluator::UnifiedEvaluator(const seal::SEALContext &hcontext,
                                   const PhantomContext &dcontext) {
  hevaluator_ = new seal::Evaluator(hcontext);
  devaluator_ = new PhantomEvaluator(dcontext);
}

UnifiedEvaluator::~UnifiedEvaluator() {
  delete hevaluator_;
  delete devaluator_;
}
