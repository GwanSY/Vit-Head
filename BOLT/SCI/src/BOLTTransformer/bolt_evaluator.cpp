#include "BOLTTransformer/bolt_evaluator.h"

#include <stdexcept>

#if defined(BOLT_USE_HE_GPU)

#include "BOLTTransformer/bolt_phantom_host_compat.h"

#include <cuda_runtime_api.h>
#include <phantom/evaluate.cuh>
#include <phantom/phantom.h>
#include <seal/util/numth.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sci {
namespace bolt {
namespace {

void check_cuda(cudaError_t error, const char *operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("BOLT Phantom CUDA failure in ") +
                             operation + ": " + cudaGetErrorString(error));
  }
}

void load_phantom_public_key(PhantomCiphertext &&cipher,
                             PhantomPublicKey &public_key) {
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  cipher.save(stream);
  stream.seekg(0);
  public_key.load(stream);
}

void host_cipher_to_device(const seal::SEALContext &host_context,
                           const seal::Ciphertext &host_cipher,
                           const PhantomContext &device_context,
                           PhantomCiphertext &device_cipher) {
  const auto first_data = host_context.first_context_data();
  if (first_data == nullptr) {
    throw std::logic_error("BOLT GPU: SEAL data context is unavailable");
  }
  const std::size_t first_data_modulus_size =
      first_data->parms().coeff_modulus().size();
  const std::size_t host_modulus_size = host_cipher.coeff_modulus_size();
  if (host_modulus_size == 0 || host_modulus_size > first_data_modulus_size + 1) {
    throw std::logic_error("BOLT GPU: unsupported SEAL ciphertext modulus level");
  }

  // Phantom has one key-only special modulus.  SEAL's key context therefore
  // maps to Phantom chain 0 and its first data context maps to chain 1.
  const std::size_t device_chain =
      first_data_modulus_size - host_modulus_size + 1;
  if (device_chain >= device_context.total_parm_size()) {
    throw std::logic_error("BOLT GPU: ciphertext level is outside Phantom chain");
  }
  const auto &device_parms = device_context.get_context_data(device_chain).parms();
  if (device_parms.coeff_modulus().size() != host_modulus_size ||
      device_parms.poly_modulus_degree() != host_cipher.poly_modulus_degree()) {
    throw std::logic_error("BOLT GPU: SEAL/Phantom ciphertext parameters differ");
  }

  const auto &stream = cudaStreamPerThread;
  device_cipher.resize(device_context, device_chain, host_cipher.size(), stream);
  device_cipher.set_scale(host_cipher.scale());
  device_cipher.set_correction_factor(host_cipher.correction_factor());
  device_cipher.SetNoiseScaleDeg(1);
  device_cipher.set_ntt_form(host_cipher.is_ntt_form());
  const std::size_t words = host_cipher.size() * host_modulus_size *
                            host_cipher.poly_modulus_degree();
  check_cuda(cudaMemcpy(device_cipher.data(), host_cipher.data(),
                        words * sizeof(std::uint64_t),
                        cudaMemcpyHostToDevice),
             "ciphertext upload");
}

void device_cipher_to_host(const PhantomContext &device_context,
                           const PhantomCiphertext &device_cipher,
                           const seal::SEALContext &host_context,
                           seal::Ciphertext &host_cipher) {
  check_cuda(cudaStreamSynchronize(cudaStreamPerThread),
             "synchronize ciphertext download");
  auto target = host_context.first_context_data();
  while (target != nullptr &&
         target->parms().coeff_modulus().size() !=
             device_cipher.coeff_modulus_size()) {
    target = target->next_context_data();
  }
  if (target == nullptr) {
    throw std::logic_error("BOLT GPU: Phantom modulus level is absent in SEAL");
  }
  const auto &device_parms =
      device_context.get_context_data(device_cipher.chain_index()).parms();
  if (device_parms.poly_modulus_degree() !=
      host_cipher.poly_modulus_degree() && host_cipher.poly_modulus_degree() != 0) {
    throw std::logic_error("BOLT GPU: Phantom/SEAL polynomial degree differs");
  }
  host_cipher.resize(host_context, target->parms_id(), device_cipher.size());
  host_cipher.scale() = device_cipher.scale();
  host_cipher.correction_factor() = device_cipher.correction_factor();
  host_cipher.is_ntt_form() = device_cipher.is_ntt_form();
  const std::size_t words = device_cipher.size() *
                            device_cipher.coeff_modulus_size() *
                            device_cipher.poly_modulus_degree();
  check_cuda(cudaMemcpy(host_cipher.data(), device_cipher.data(),
                        words * sizeof(std::uint64_t),
                        cudaMemcpyDeviceToHost),
             "ciphertext download");
}

void host_plain_to_device(const seal::Plaintext &host_plain,
                          const PhantomContext &device_context,
                          PhantomPlaintext &device_plain) {
  const std::size_t degree = device_context.poly_degree_;
  if (host_plain.coeff_count() > degree) {
    throw std::logic_error("BOLT GPU: plaintext exceeds polynomial degree");
  }
  const auto &stream = cudaStreamPerThread;
  // BFV plaintexts are a single polynomial modulo t.  Do not use Phantom's
  // BatchEncoder here: its public slot generator is intentionally independent
  // of SEAL's generator.  Copying the already encoded polynomial preserves
  // BOLT's exact SEAL BatchEncoder packing.
  device_plain.resize(1, degree, stream);
  check_cuda(cudaMemset(device_plain.data(), 0,
                        degree * sizeof(std::uint64_t)),
             "plaintext clear");
  check_cuda(cudaMemcpy(device_plain.data(), host_plain.data(),
                        host_plain.coeff_count() * sizeof(std::uint64_t),
                        cudaMemcpyHostToDevice),
             "plaintext upload");
}

void convert_key_vector(const seal::SEALContext &host_context,
                        const std::vector<seal::PublicKey> &host_keys,
                        const PhantomContext &device_context,
                        std::vector<PhantomPublicKey> &device_keys) {
  device_keys.clear();
  device_keys.reserve(host_keys.size());
  for (const auto &host_key : host_keys) {
    PhantomCiphertext device_cipher;
    host_cipher_to_device(host_context, host_key.data(), device_context,
                          device_cipher);
    PhantomPublicKey device_key;
    load_phantom_public_key(std::move(device_cipher), device_key);
    device_keys.emplace_back(std::move(device_key));
  }
}

void load_phantom_relin_key(const std::vector<PhantomPublicKey> &public_keys,
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

void convert_relin_key(const seal::SEALContext &host_context,
                       const seal::RelinKeys &host_key,
                       const PhantomContext &device_context,
                       PhantomRelinKey &device_key) {
  if (!host_key.has_key(2)) {
    throw std::logic_error("BOLT GPU: relinearization key for power 2 is missing");
  }
  std::vector<PhantomPublicKey> public_keys;
  convert_key_vector(host_context, host_key.key(2), device_context, public_keys);
  load_phantom_relin_key(public_keys, device_key);
}

void convert_galois_key(const seal::SEALContext &host_context,
                        const seal::GaloisKeys &host_key,
                        const PhantomContext &device_context,
                        PhantomGaloisKey &device_key) {
  const auto &host_data = host_key.data();
  std::vector<std::uint32_t> elements;
  for (std::size_t index = 0; index < host_data.size(); ++index) {
    if (!host_data[index].empty()) {
      elements.push_back(static_cast<std::uint32_t>(index * 2 + 1));
    }
  }
  if (elements.empty()) {
    throw std::logic_error("BOLT GPU: Galois keys are empty");
  }

  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  const std::size_t key_count = elements.size();
  stream.write(reinterpret_cast<const char *>(&key_count), sizeof(key_count));
  for (const auto element : elements) {
    if (!host_key.has_key(element)) {
      throw std::logic_error("BOLT GPU: sparse Galois key table is invalid");
    }
    std::vector<PhantomPublicKey> public_keys;
    convert_key_vector(host_context, host_key.key(element), device_context,
                       public_keys);
    const std::size_t decomposition_count = public_keys.size();
    stream.write(reinterpret_cast<const char *>(&decomposition_count),
                 sizeof(decomposition_count));
    for (const auto &public_key : public_keys) {
      public_key.save(stream);
    }
  }
  stream.seekg(0);
  device_key.load(stream);
}

std::uint32_t galois_element_for_step(std::size_t poly_degree, int steps) {
  const std::uint64_t modulus = static_cast<std::uint64_t>(poly_degree) * 2;
  if (steps == 0) {
    return static_cast<std::uint32_t>(modulus - 1);
  }
  const int half = static_cast<int>(poly_degree / 2);
  int exponent = steps < 0 ? half - (-steps) : steps;
  if (std::abs(steps) >= half) {
    throw std::invalid_argument("BOLT GPU: rotation step is too large");
  }
  exponent %= half;
  if (exponent < 0) exponent += half;
  std::uint64_t result = 1;
  for (int index = 0; index < exponent; ++index) {
    result = (result * 3) & (modulus - 1);
  }
  return static_cast<std::uint32_t>(result);
}

} // namespace

class BoltEvaluator::GpuState {
public:
  GpuState(const seal::SEALContext &host_context,
           const seal::GaloisKeys &galois_keys,
           const seal::RelinKeys &relinearization_keys, int device)
      : host_context_(host_context) {
    const char *resident_setting = std::getenv("BOLT_GPU_RESIDENT");
    resident_enabled_ =
        resident_setting != nullptr && std::string(resident_setting) != "0";
    check_cuda(cudaSetDevice(device), "select device 0");

    const auto &host_key_parms = host_context.key_context_data()->parms();
    phantom::EncryptionParameters device_parms(phantom::scheme_type::bfv);
    device_parms.set_poly_modulus_degree(host_key_parms.poly_modulus_degree());
    std::vector<phantom::arith::Modulus> coeff_modulus;
    coeff_modulus.reserve(host_key_parms.coeff_modulus().size());
    for (const auto &modulus : host_key_parms.coeff_modulus()) {
      coeff_modulus.emplace_back(modulus.value());
    }
    device_parms.set_coeff_modulus(coeff_modulus);
    device_parms.set_plain_modulus(
        phantom::arith::Modulus(host_key_parms.plain_modulus().value()));
    device_parms.set_special_modulus_size(1);

    // Phantom's default rotation generator is 5, while SEAL's is 3.  Give
    // Phantom the exact sparse SEAL element list and invoke apply_galois with
    // the SEAL element below; this preserves BOLT's host packing semantics.
    std::vector<std::uint32_t> galois_elements;
    for (std::size_t index = 0; index < galois_keys.data().size(); ++index) {
      if (!galois_keys.data()[index].empty()) {
        galois_elements.push_back(
            static_cast<std::uint32_t>(index * 2 + 1));
      }
    }
    device_parms.set_galois_elts(galois_elements);

    device_context_ = std::make_unique<PhantomContext>(device_parms);
    convert_relin_key(host_context, relinearization_keys, *device_context_,
                      device_relin_keys_);
    convert_galois_key(host_context, galois_keys, *device_context_,
                       device_galois_keys_);
  }

  void add(const seal::Ciphertext &lhs, const seal::Ciphertext &rhs,
           seal::Ciphertext &destination) const {
    auto dlhs = clone_or_upload(lhs);
    auto drhs = clone_or_upload(rhs);
    phantom::add_inplace(*device_context_, *dlhs, *drhs);
    store(std::move(dlhs), destination);
  }

  void add_inplace(seal::Ciphertext &lhs,
                   const seal::Ciphertext &rhs) const {
    auto dlhs = take_or_upload(lhs);
    auto drhs = &lhs == &rhs ? std::make_unique<PhantomCiphertext>(*dlhs)
                             : clone_or_upload(rhs);
    phantom::add_inplace(*device_context_, *dlhs, *drhs);
    store(std::move(dlhs), lhs);
  }

  void add_plain_inplace(seal::Ciphertext &ciphertext,
                         const seal::Plaintext &plaintext) const {
    auto dcipher = take_or_upload(ciphertext);
    PhantomPlaintext dplain;
    host_plain_to_device(plaintext, *device_context_, dplain);
    phantom::add_plain_inplace(*device_context_, *dcipher, dplain);
    store(std::move(dcipher), ciphertext);
  }

  void sub_plain_inplace(seal::Ciphertext &ciphertext,
                         const seal::Plaintext &plaintext) const {
    auto dcipher = take_or_upload(ciphertext);
    PhantomPlaintext dplain;
    host_plain_to_device(plaintext, *device_context_, dplain);
    phantom::sub_plain_inplace(*device_context_, *dcipher, dplain);
    store(std::move(dcipher), ciphertext);
  }

  void multiply(const seal::Ciphertext &lhs, const seal::Ciphertext &rhs,
                seal::Ciphertext &destination) const {
    auto dlhs = clone_or_upload(lhs);
    auto drhs = clone_or_upload(rhs);
    try {
      phantom::multiply_inplace(*device_context_, *dlhs, *drhs);
    } catch (const std::invalid_argument &error) {
      std::cerr << "BOLT_GPU_MULTIPLY_MISMATCH lhs_chain=" << dlhs->chain_index()
                << " rhs_chain=" << drhs->chain_index()
                << " lhs_size=" << dlhs->size() << " rhs_size=" << drhs->size()
                << " lhs_coeff=" << lhs.coeff_modulus_size()
                << " rhs_coeff=" << rhs.coeff_modulus_size()
                << " lhs_parms0=" << lhs.parms_id()[0] << " rhs_parms0="
                << rhs.parms_id()[0] << " reason=" << error.what() << std::endl;
      throw;
    }
    store(std::move(dlhs), destination);
  }

  void multiply_plain(const seal::Ciphertext &ciphertext,
                      const seal::Plaintext &plaintext,
                      seal::Ciphertext &destination) const {
    auto dcipher = clone_or_upload(ciphertext);
    PhantomPlaintext dplain;
    host_plain_to_device(plaintext, *device_context_, dplain);
    phantom::multiply_plain_inplace(*device_context_, *dcipher, dplain);
    store(std::move(dcipher), destination);
  }

  void multiply_plain_inplace(seal::Ciphertext &ciphertext,
                              const seal::Plaintext &plaintext) const {
    auto dcipher = take_or_upload(ciphertext);
    PhantomPlaintext dplain;
    host_plain_to_device(plaintext, *device_context_, dplain);
    phantom::multiply_plain_inplace(*device_context_, *dcipher, dplain);
    store(std::move(dcipher), ciphertext);
  }

  void relinearize_inplace(seal::Ciphertext &ciphertext) const {
    auto dcipher = take_or_upload(ciphertext);
    phantom::relinearize_inplace(*device_context_, *dcipher,
                                 device_relin_keys_);
    store(std::move(dcipher), ciphertext);
  }

  void mod_switch_to_next_inplace(seal::Ciphertext &ciphertext) const {
    auto dcipher = take_or_upload(ciphertext);
    auto next = std::make_unique<PhantomCiphertext>(
        phantom::mod_switch_to_next(*device_context_, *dcipher));
    store(std::move(next), ciphertext);
  }

  void rotate_rows(const seal::Ciphertext &ciphertext, int steps,
                   seal::Ciphertext &destination) const {
    if (steps == 0) {
      copy_ciphertext(ciphertext, destination);
      return;
    }
    auto dcipher = clone_or_upload(ciphertext);
    const auto poly_degree =
        host_context_.key_context_data()->parms().poly_modulus_degree();

    // SEAL creates the complete Galois-key set as powers of two and reduces
    // arbitrary rotations to their non-adjacent form (NAF).  A direct
    // arbitrary-step lookup would therefore fail for normal BOLT rotations.
    for (const int naf_step : seal::util::naf(steps)) {
      if (std::abs(naf_step) == static_cast<int>(poly_degree / 2)) {
        continue;
      }
      const auto element = galois_element_for_step(poly_degree, naf_step);
      phantom::apply_galois_inplace(*device_context_, *dcipher, element,
                                    device_galois_keys_);
    }
    store(std::move(dcipher), destination);
  }

  void rotate_columns(const seal::Ciphertext &ciphertext,
                      seal::Ciphertext &destination) const {
    auto dcipher = clone_or_upload(ciphertext);
    const auto element = static_cast<std::size_t>(
        host_context_.key_context_data()->parms().poly_modulus_degree() * 2 -
        1);
    phantom::apply_galois_inplace(*device_context_, *dcipher, element,
                                  device_galois_keys_);
    store(std::move(dcipher), destination);
  }

  void copy_ciphertext(const seal::Ciphertext &source,
                       seal::Ciphertext &destination) const {
    if (&source == &destination) {
      return;
    }
    forget(destination);
    const auto it = resident_.find(source.data());
    if (it == resident_.end()) {
      destination = source;
      return;
    }
    auto copy = std::make_unique<PhantomCiphertext>(*it->second);
    set_host_metadata(*copy, destination);
    resident_[destination.data()] = std::move(copy);
  }

  void synchronize(seal::Ciphertext &ciphertext) const {
    const auto it = resident_.find(ciphertext.data());
    if (it == resident_.end()) {
      return;
    }
    auto device_cipher = std::move(it->second);
    resident_.erase(it);
    device_cipher_to_host(*device_context_, *device_cipher, host_context_,
                          ciphertext);
  }

  void release_residency() const {
    check_cuda(cudaStreamSynchronize(cudaStreamPerThread),
               "synchronize resident ciphertext release");
    resident_.clear();
  }

  bool resident_enabled() const noexcept { return resident_enabled_; }

private:
  using HostDataKey = const std::uint64_t *;
  using ResidentMap =
      std::unordered_map<HostDataKey, std::unique_ptr<PhantomCiphertext>>;

  void forget(const seal::Ciphertext &ciphertext) const {
    if (ciphertext.data() != nullptr) {
      resident_.erase(ciphertext.data());
    }
  }

  std::unique_ptr<PhantomCiphertext> clone_or_upload(
      const seal::Ciphertext &ciphertext) const {
    const auto it = resident_.find(ciphertext.data());
    if (it != resident_.end()) {
      return std::make_unique<PhantomCiphertext>(*it->second);
    }
    auto device_cipher = std::make_unique<PhantomCiphertext>();
    host_cipher_to_device(host_context_, ciphertext, *device_context_,
                          *device_cipher);
    return device_cipher;
  }

  std::unique_ptr<PhantomCiphertext> take_or_upload(
      seal::Ciphertext &ciphertext) const {
    const auto key = ciphertext.data();
    const auto it = resident_.find(key);
    if (it != resident_.end()) {
      auto device_cipher = std::move(it->second);
      resident_.erase(it);
      return device_cipher;
    }
    return clone_or_upload(ciphertext);
  }

  void set_host_metadata(const PhantomCiphertext &device_cipher,
                         seal::Ciphertext &host_cipher) const {
    auto target = host_context_.first_context_data();
    while (target != nullptr &&
           target->parms().coeff_modulus().size() !=
               device_cipher.coeff_modulus_size()) {
      target = target->next_context_data();
    }
    if (target == nullptr) {
      throw std::logic_error("BOLT GPU: Phantom modulus level is absent in SEAL");
    }
    const auto &device_parms =
        device_context_->get_context_data(device_cipher.chain_index()).parms();
    const bool same_shape =
        host_cipher.size() == device_cipher.size() &&
        host_cipher.coeff_modulus_size() == device_cipher.coeff_modulus_size() &&
        host_cipher.poly_modulus_degree() == device_cipher.poly_modulus_degree() &&
        host_cipher.parms_id() == target->parms_id();
    if (!same_shape) {
      host_cipher.resize(host_context_, target->parms_id(), device_cipher.size());
    }
    if (device_parms.poly_modulus_degree() !=
        host_cipher.poly_modulus_degree()) {
      throw std::logic_error("BOLT GPU: Phantom/SEAL polynomial degree differs");
    }
    host_cipher.scale() = device_cipher.scale();
    host_cipher.correction_factor() = device_cipher.correction_factor();
    host_cipher.is_ntt_form() = device_cipher.is_ntt_form();
  }

  void store(std::unique_ptr<PhantomCiphertext> device_cipher,
             seal::Ciphertext &host_cipher) const {
    forget(host_cipher);
    set_host_metadata(*device_cipher, host_cipher);
    const auto key = host_cipher.data();
    if (key == nullptr) {
      throw std::logic_error("BOLT GPU: ciphertext has no host storage");
    }
    if (!resident_enabled_) {
      // Strict compatibility mode: materialize every operator result in the
      // SEAL host buffer, matching the original host/device boundary model.
      device_cipher_to_host(*device_context_, *device_cipher, host_context_,
                            host_cipher);
      return;
    }
    resident_[key] = std::move(device_cipher);
  }

  const seal::SEALContext &host_context_;
  bool resident_enabled_ = true;
  std::unique_ptr<PhantomContext> device_context_;
  PhantomRelinKey device_relin_keys_;
  PhantomGaloisKey device_galois_keys_;
  mutable ResidentMap resident_;
};

} // namespace bolt
} // namespace sci

#endif // BOLT_USE_HE_GPU

namespace sci {
namespace bolt {

#if !defined(BOLT_USE_HE_GPU)
class BoltEvaluator::GpuState {};
#endif

BoltEvaluator::BoltEvaluator(const seal::SEALContext &context)
    : context_(&context), host_evaluator_(std::make_unique<seal::Evaluator>(context)) {}

BoltEvaluator::~BoltEvaluator() = default;

void BoltEvaluator::enable_gpu(const seal::GaloisKeys &galois_keys,
                               const seal::RelinKeys &relinearization_keys,
                               int device) {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    throw std::logic_error("BOLT GPU evaluator is already enabled");
  }
  if (device != 0) {
    throw std::invalid_argument("BOLT GPU entry points require device 0");
  }
  gpu_ = std::make_unique<GpuState>(*context_, galois_keys,
                                     relinearization_keys, device);
  std::cerr << "BOLT_HE_BACKEND backend=phantom_bfv device=" << device
            << " protocol=BOLT residency="
            << (gpu_->resident_enabled() ? "on" : "off") << std::endl;
#else
  (void)galois_keys;
  (void)relinearization_keys;
  (void)device;
  throw std::logic_error(
      "BOLT GPU evaluator was not compiled; configure with -DBOLT_USE_HE_GPU=ON");
#endif
}

bool BoltEvaluator::gpu_enabled() const noexcept { return gpu_ != nullptr; }

void BoltEvaluator::add(const seal::Ciphertext &lhs,
                        const seal::Ciphertext &rhs,
                        seal::Ciphertext &destination) const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->add(lhs, rhs, destination);
    return;
  }
#endif
  host_evaluator_->add(lhs, rhs, destination);
}

void BoltEvaluator::add_inplace(seal::Ciphertext &lhs,
                                const seal::Ciphertext &rhs) const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->add_inplace(lhs, rhs);
    return;
  }
#endif
  host_evaluator_->add_inplace(lhs, rhs);
}

void BoltEvaluator::add_plain_inplace(seal::Ciphertext &ciphertext,
                                      const seal::Plaintext &plaintext) const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->add_plain_inplace(ciphertext, plaintext);
    return;
  }
#endif
  host_evaluator_->add_plain_inplace(ciphertext, plaintext);
}

void BoltEvaluator::sub_plain_inplace(seal::Ciphertext &ciphertext,
                                      const seal::Plaintext &plaintext) const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->sub_plain_inplace(ciphertext, plaintext);
    return;
  }
#endif
  host_evaluator_->sub_plain_inplace(ciphertext, plaintext);
}

void BoltEvaluator::multiply(const seal::Ciphertext &lhs,
                             const seal::Ciphertext &rhs,
                             seal::Ciphertext &destination) const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->multiply(lhs, rhs, destination);
    return;
  }
#endif
  host_evaluator_->multiply(lhs, rhs, destination);
}

void BoltEvaluator::multiply_plain(const seal::Ciphertext &ciphertext,
                                   const seal::Plaintext &plaintext,
                                   seal::Ciphertext &destination) const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->multiply_plain(ciphertext, plaintext, destination);
    return;
  }
#endif
  host_evaluator_->multiply_plain(ciphertext, plaintext, destination);
}

void BoltEvaluator::multiply_plain_inplace(
    seal::Ciphertext &ciphertext, const seal::Plaintext &plaintext) const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->multiply_plain_inplace(ciphertext, plaintext);
    return;
  }
#endif
  host_evaluator_->multiply_plain_inplace(ciphertext, plaintext);
}

void BoltEvaluator::relinearize_inplace(
    seal::Ciphertext &ciphertext,
    const seal::RelinKeys &relinearization_keys) const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->relinearize_inplace(ciphertext);
    return;
  }
#endif
  host_evaluator_->relinearize_inplace(ciphertext, relinearization_keys);
}

void BoltEvaluator::mod_switch_to_next_inplace(
    seal::Ciphertext &ciphertext) const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->mod_switch_to_next_inplace(ciphertext);
    return;
  }
#endif
  host_evaluator_->mod_switch_to_next_inplace(ciphertext);
}

void BoltEvaluator::rotate_rows(const seal::Ciphertext &ciphertext, int steps,
                                const seal::GaloisKeys &galois_keys,
                                seal::Ciphertext &destination) const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->rotate_rows(ciphertext, steps, destination);
    return;
  }
#endif
  host_evaluator_->rotate_rows(ciphertext, steps, galois_keys, destination);
}

void BoltEvaluator::rotate_columns(
    const seal::Ciphertext &ciphertext, const seal::GaloisKeys &galois_keys,
    seal::Ciphertext &destination) const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->rotate_columns(ciphertext, destination);
    return;
  }
#endif
  host_evaluator_->rotate_columns(ciphertext, galois_keys, destination);
}

void BoltEvaluator::copy_ciphertext(const seal::Ciphertext &source,
                                    seal::Ciphertext &destination) const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->copy_ciphertext(source, destination);
    return;
  }
#endif
  destination = source;
}

void BoltEvaluator::synchronize(const seal::Ciphertext &ciphertext) const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->synchronize(const_cast<seal::Ciphertext &>(ciphertext));
  }
#else
  (void)ciphertext;
#endif
}

void BoltEvaluator::release_gpu_residency() const {
#if defined(BOLT_USE_HE_GPU)
  if (gpu_) {
    gpu_->release_residency();
  }
#endif
}

} // namespace bolt
} // namespace sci
