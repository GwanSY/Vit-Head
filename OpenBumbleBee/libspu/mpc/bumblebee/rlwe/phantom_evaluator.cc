// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "libspu/mpc/bumblebee/rlwe/phantom_evaluator.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "spdlog/spdlog.h"
#include "seal/util/ntt.h"
#include "seal/util/polyarithsmallmod.h"

#include "libspu/core/parallel_utils.h"
#include "libspu/mpc/bumblebee/rlwe/bumblebee_metrics.h"

#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
#include "libspu/mpc/bumblebee/rlwe/phantom_host_compat.h"
#include "libspu/mpc/bumblebee/rlwe/phantom_bridge_kernels.h"
#include "phantom.h"
#include "cuda_runtime_api.h"
#endif

namespace spu::mpc::bumblebee {
namespace {

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

[[noreturn]] void ThrowBackendError(const std::string& message) {
  SPDLOG_ERROR("OPENBUMBLEBEE_HE_BACKEND {}", message);
  throw std::runtime_error(message);
}

bool VerifyGpuResults() {
  static const bool enabled = [] {
    const char* value = std::getenv("SPU_BUMBLEBEE_VERIFY_GPU");
    return value != nullptr &&
           (std::strcmp(value, "1") == 0 || Lower(value) == "true" ||
            Lower(value) == "on");
  }();
  return enabled;
}

[[maybe_unused]] void ExactSealModSwitchToInplace(
    const seal::SEALContext& context, seal::Ciphertext& cipher,
    seal::parms_id_type target) {
  const auto pool = seal::MemoryManager::GetPool();
  while (cipher.parms_id() != target) {
    const auto current = context.get_context_data(cipher.parms_id());
    if (!current || !current->next_context_data()) {
      throw std::runtime_error(
          "OpenBumbleBee verification modulus level is absent in SEAL");
    }
    const auto* rns_tool = current->rns_tool();
    const auto* ntt_tables = current->small_ntt_tables();
    if (cipher.is_ntt_form()) {
      SEAL_ITERATE(seal::util::iter(cipher), cipher.size(), [&](auto item) {
        rns_tool->divide_and_round_q_last_ntt_inplace(item, ntt_tables, pool);
      });
    } else {
      SEAL_ITERATE(seal::util::iter(cipher), cipher.size(), [&](auto item) {
        rns_tool->divide_and_round_q_last_inplace(item, pool);
      });
    }

    const auto next_context = current->next_context_data();
    seal::Ciphertext next(pool);
    next.resize(context, next_context->parms_id(), cipher.size());
    SEAL_ITERATE(seal::util::iter(cipher, next), cipher.size(), [&](auto item) {
      seal::util::set_poly(std::get<0>(item), cipher.poly_modulus_degree(),
                           cipher.coeff_modulus_size() - 1,
                           std::get<1>(item));
    });
    next.is_ntt_form() = cipher.is_ntt_form();
    next.scale() = cipher.scale();
    next.correction_factor() = cipher.correction_factor();
    std::swap(cipher, next);
  }
}

[[maybe_unused]] void ExactSealAddSubPlainInplace(
    const seal::SEALContext& context, seal::Ciphertext& cipher,
    const seal::Plaintext& plain, bool subtract) {
  const auto context_data = context.get_context_data(cipher.parms_id());
  if (!context_data || cipher.parms_id() != plain.parms_id()) {
    throw std::runtime_error(
        "OpenBumbleBee verification ciphertext/plaintext level mismatch");
  }
  seal::util::RNSIter cipher_c0(cipher.data(0),
                                cipher.poly_modulus_degree());
  seal::util::ConstRNSIter plain_rns(plain.data(),
                                     cipher.poly_modulus_degree());
  const auto& moduli = context_data->parms().coeff_modulus();
  if (subtract) {
    seal::util::sub_poly_coeffmod(cipher_c0, plain_rns, moduli.size(),
                                  moduli, cipher_c0);
  } else {
    seal::util::add_poly_coeffmod(cipher_c0, plain_rns, moduli.size(),
                                  moduli, cipher_c0);
  }
}

void ExactSealNegacyclicRightShiftInplace(
    const seal::SEALContext& context, seal::Ciphertext& cipher,
    std::size_t shift) {
  if (shift == 0 || cipher.size() == 0) {
    return;
  }
  const auto context_data = context.get_context_data(cipher.parms_id());
  if (!context_data || cipher.is_ntt_form() ||
      shift >= cipher.poly_modulus_degree()) {
    throw std::runtime_error(
        "OpenBumbleBee verification invalid negacyclic shift input");
  }
  const auto degree = cipher.poly_modulus_degree();
  std::vector<std::uint64_t> wrapped(shift);
  for (std::size_t poly = 0; poly < cipher.size(); ++poly) {
    auto* destination = cipher.data(poly);
    for (const auto& modulus : context_data->parms().coeff_modulus()) {
      std::copy_n(destination + degree - shift, shift, wrapped.data());
      std::move_backward(destination, destination + degree - shift,
                         destination + degree);
      const auto q = modulus.value();
      for (std::size_t coefficient = 0; coefficient < shift; ++coefficient) {
        const auto value = wrapped[coefficient];
        destination[coefficient] = value == 0 ? 0 : q - value;
      }
      destination += degree;
    }
  }
}

[[maybe_unused]] void VerifyCipherEquivalent(const seal::SEALContext& context,
                            const seal::Ciphertext& expected,
                            const seal::Ciphertext& actual,
                            const char* operation) {
  if (!VerifyGpuResults()) {
    return;
  }
  if (expected.parms_id() != actual.parms_id() ||
      expected.size() != actual.size() ||
      expected.coeff_modulus_size() != actual.coeff_modulus_size() ||
      expected.poly_modulus_degree() != actual.poly_modulus_degree() ||
      expected.is_ntt_form() != actual.is_ntt_form()) {
    throw std::runtime_error(std::string("OpenBumbleBee GPU metadata mismatch in ") +
                             operation);
  }

  auto context_data = context.get_context_data(expected.parms_id());
  if (!context_data) {
    throw std::runtime_error("OpenBumbleBee GPU verification context missing");
  }
  const auto& moduli = context_data->parms().coeff_modulus();
  const std::size_t degree = expected.poly_modulus_degree();
  for (std::size_t poly = 0; poly < expected.size(); ++poly) {
    for (std::size_t modulus = 0; modulus < moduli.size(); ++modulus) {
      const auto q = moduli[modulus].value();
      const auto* lhs = expected.data(poly) + modulus * degree;
      const auto* rhs = actual.data(poly) + modulus * degree;
      for (std::size_t coeff = 0; coeff < degree; ++coeff) {
        if (lhs[coeff] % q != rhs[coeff] % q) {
          throw std::runtime_error(
              std::string("OpenBumbleBee GPU coefficient mismatch in ") +
              operation + " at polynomial=" + std::to_string(poly) +
              " modulus=" + std::to_string(modulus) +
              " coefficient=" + std::to_string(coeff) +
              " expected=" + std::to_string(lhs[coeff] % q) +
              " actual=" + std::to_string(rhs[coeff] % q) +
              " q=" + std::to_string(q));
        }
      }
    }
  }
}

[[maybe_unused]] void VerifyPlainEquivalent(const seal::SEALContext& context,
                                            const seal::Plaintext& expected,
                                            const seal::Plaintext& actual,
                                            const char* operation) {
  if (!VerifyGpuResults()) {
    return;
  }
  if (expected.parms_id() != actual.parms_id() ||
      expected.coeff_count() != actual.coeff_count()) {
    throw std::runtime_error(std::string("OpenBumbleBee GPU plaintext metadata mismatch in ") +
                             operation);
  }
  auto context_data = context.get_context_data(expected.parms_id());
  if (!context_data) {
    throw std::runtime_error(
        "OpenBumbleBee GPU plaintext verification context missing");
  }
  const auto degree = context_data->parms().poly_modulus_degree();
  const auto& moduli = context_data->parms().coeff_modulus();
  if (expected.coeff_count() != degree * moduli.size()) {
    throw std::runtime_error(
        "OpenBumbleBee GPU plaintext verification size mismatch");
  }
  for (std::size_t modulus = 0; modulus < moduli.size(); ++modulus) {
    const auto q = moduli[modulus].value();
    for (std::size_t coeff = 0; coeff < degree; ++coeff) {
      const auto index = modulus * degree + coeff;
      if (expected[index] % q != actual[index] % q) {
        throw std::runtime_error(
            std::string("OpenBumbleBee GPU plaintext coefficient mismatch in ") +
            operation + " at modulus=" + std::to_string(modulus) +
            " coefficient=" + std::to_string(coeff));
      }
    }
  }
}

#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    ThrowBackendError(std::string("OpenBumbleBee Phantom CUDA failure in ") +
                      operation + ": " + cudaGetErrorString(status));
  }
}

void SynchronizeCudaStream(const char* operation) {
  ScopedBumblebeeMetric metric(BumblebeeMetricKind::kSyncWait);
  CheckCuda(cudaStreamSynchronize(cudaStreamPerThread), operation);
}

void SynchronizeCudaDevice(const char* operation) {
  ScopedBumblebeeMetric metric(BumblebeeMetricKind::kSyncWait);
  CheckCuda(cudaDeviceSynchronize(), operation);
}

std::shared_ptr<const seal::SEALContext::ContextData>
FindHostContextByModulusSize(const seal::SEALContext& context,
                             std::size_t modulus_size) {
  auto key = context.key_context_data();
  if (key && key->parms().coeff_modulus().size() == modulus_size) {
    return key;
  }
  auto current = context.first_context_data();
  while (current && current->parms().coeff_modulus().size() != modulus_size) {
    current = current->next_context_data();
  }
  if (!current) {
    throw std::runtime_error(
        "OpenBumbleBee GPU: Phantom modulus level is absent in SEAL");
  }
  return current;
}

std::size_t HostDataChainIndex(const seal::SEALContext& context,
                               seal::parms_id_type parms_id,
                               const char* operation) {
  auto current = context.get_context_data(parms_id);
  if (!current) {
    std::ostringstream message;
    message << "OpenBumbleBee GPU: unknown SEAL parms_id in " << operation
            << " [" << std::hex << parms_id[0] << ',' << parms_id[1] << ','
            << parms_id[2] << ',' << parms_id[3] << ']';
    ThrowBackendError(message.str());
  }
  const auto key_size =
      context.key_context_data()->parms().coeff_modulus().size();
  const auto first_size =
      context.first_context_data()->parms().coeff_modulus().size();
  const auto current_size = current->parms().coeff_modulus().size();
  if (current_size == key_size && key_size > first_size) {
    return 0;
  }
  if (current_size > first_size || current_size == 0) {
    throw std::runtime_error("OpenBumbleBee GPU: invalid SEAL modulus level");
  }
  return first_size + 1 - current_size;
}

phantom::EncryptionParameters MakePhantomParameters(
    const seal::SEALContext& context,
    const std::vector<std::uint32_t>& galois_elts) {
  const auto& key_parms = context.key_context_data()->parms();
  const auto key_size = key_parms.coeff_modulus().size();
  const auto data_size =
      context.first_context_data()->parms().coeff_modulus().size();
  if (key_size < data_size) {
    throw std::runtime_error("OpenBumbleBee GPU: malformed SEAL modulus chain");
  }

  phantom::scheme_type scheme;
  switch (key_parms.scheme()) {
    case seal::scheme_type::bfv:
      scheme = phantom::scheme_type::bfv;
      break;
    case seal::scheme_type::ckks:
      scheme = phantom::scheme_type::ckks;
      break;
    default:
      throw std::runtime_error(
          "OpenBumbleBee GPU supports only the original BFV and CKKS-form "
          "coefficient schemes");
  }

  phantom::EncryptionParameters result(scheme);
  result.set_poly_modulus_degree(key_parms.poly_modulus_degree());
  result.set_special_modulus_size(key_size - data_size);
  std::vector<phantom::arith::Modulus> moduli;
  moduli.reserve(key_size);
  for (const auto& modulus : key_parms.coeff_modulus()) {
    moduli.emplace_back(modulus.value());
  }
  result.set_coeff_modulus(moduli);
  if (scheme == phantom::scheme_type::bfv) {
    result.set_plain_modulus(
        phantom::arith::Modulus(key_parms.plain_modulus().value()));
  }
  result.set_galois_elts(galois_elts);
  return result;
}

phantom::EncryptionParameters MakeExactCkksFormParameters(
    const seal::SEALContext& context) {
  const auto& key_parms = context.key_context_data()->parms();
  const auto key_size = key_parms.coeff_modulus().size();
  const auto data_size =
      context.first_context_data()->parms().coeff_modulus().size();
  phantom::EncryptionParameters result(phantom::scheme_type::ckks);
  result.set_poly_modulus_degree(key_parms.poly_modulus_degree());
  result.set_special_modulus_size(key_size - data_size);
  std::vector<phantom::arith::Modulus> moduli;
  moduli.reserve(key_size);
  for (const auto& modulus : key_parms.coeff_modulus()) {
    moduli.emplace_back(modulus.value());
  }
  result.set_coeff_modulus(moduli);
  return result;
}

std::vector<std::uint32_t> GaloisElements(const seal::GaloisKeys* keys) {
  std::vector<std::uint32_t> elements;
  if (keys == nullptr) {
    return elements;
  }
  const auto& data = keys->data();
  for (std::size_t index = 0; index < data.size(); ++index) {
    if (!data[index].empty()) {
      elements.push_back(static_cast<std::uint32_t>(2 * index + 1));
    }
  }
  return elements;
}

std::shared_ptr<PhantomContext> GetOrCreatePhantomContext(
    const phantom::EncryptionParameters& parameters, int device) {
  struct CacheEntry {
    int device;
    phantom::scheme_type scheme;
    phantom::mul_tech_type mul_tech;
    std::size_t degree;
    std::size_t special_modulus_size;
    std::uint64_t plain_modulus;
    std::vector<std::uint64_t> coeff_moduli;
    std::vector<std::uint32_t> galois_elements;
    std::shared_ptr<PhantomContext> context;
  };

  std::vector<std::uint64_t> coeff_moduli;
  coeff_moduli.reserve(parameters.coeff_modulus().size());
  for (const auto& modulus : parameters.coeff_modulus()) {
    coeff_moduli.push_back(modulus.value());
  }
  const auto galois_elements = parameters.galois_elts();

  static std::mutex cache_mutex;
  // Bumblebee constructs short-lived evaluators inside parallel loops.  Keep
  // the immutable Phantom contexts alive for the process lifetime; a weak
  // cache would repeatedly rebuild identical CUDA tables after each temporary
  // evaluator is destroyed.
  static std::vector<CacheEntry> cache;
  std::lock_guard guard(cache_mutex);
  for (const auto& entry : cache) {
    if (entry.device == device && entry.scheme == parameters.scheme() &&
        entry.mul_tech == parameters.mul_tech() &&
        entry.degree == parameters.poly_modulus_degree() &&
        entry.special_modulus_size == parameters.special_modulus_size() &&
        entry.plain_modulus == parameters.plain_modulus().value() &&
        entry.coeff_moduli == coeff_moduli &&
        entry.galois_elements == galois_elements) {
      return entry.context;
    }
  }
  ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
  auto created = std::make_shared<PhantomContext>(parameters);
  SynchronizeCudaStream("Phantom context initialization");
  cache.push_back(CacheEntry{device,
                             parameters.scheme(),
                             parameters.mul_tech(),
                             parameters.poly_modulus_degree(),
                             parameters.special_modulus_size(),
                             parameters.plain_modulus().value(),
                             std::move(coeff_moduli),
                             galois_elements,
                             created});
  return created;
}

void ExactModSwitchToInplace(const PhantomContext& context,
                             PhantomCiphertext& cipher,
                             std::size_t target_chain) {
  if (target_chain < cipher.chain_index()) {
    throw std::runtime_error(
        "OpenBumbleBee GPU cannot switch to a higher modulus level");
  }

  while (cipher.chain_index() < target_chain) {
    const auto current_chain = cipher.chain_index();
    const auto next_chain = context.get_next_index(current_chain);
    if (next_chain <= current_chain || next_chain > target_chain) {
      throw std::runtime_error(
          "OpenBumbleBee GPU target modulus level is absent in Phantom");
    }

    const auto& context_data = context.get_context_data(current_chain);
    auto& rns_tool = context_data.gpu_rns_tool();
    PhantomCiphertext next;
    next.resize(context, next_chain, cipher.size(), cudaStreamPerThread);

    // OpenBumbleBee's original scale=1 CKKS-form path deliberately supports
    // both coefficient- and NTT-form RLWE and applies SEAL's exact
    // divide-and-round operation in either representation.  Call Phantom's
    // matching CUDA RNS primitives directly instead of its public CKKS
    // modulus-switch API, which only accepts NTT-form ciphertexts and merely
    // drops a modulus.
    if (cipher.is_ntt_form()) {
      rns_tool.divide_and_round_q_last_ntt(
          cipher.data(), cipher.size(), context.gpu_rns_tables(), next.data(),
          cudaStreamPerThread);
    } else {
      rns_tool.divide_and_round_q_last(cipher.data(), cipher.size(),
                                      next.data(), cudaStreamPerThread);
    }
    next.set_ntt_form(cipher.is_ntt_form());
    next.set_scale(cipher.scale());
    next.set_correction_factor(cipher.correction_factor());
    next.SetNoiseScaleDeg(1);
    cipher = std::move(next);
  }
}

void HostCipherToDevice(const seal::SEALContext& host_context,
                        const seal::Ciphertext& host,
                        const PhantomContext& device_context,
                        PhantomCiphertext& device) {
  if (host.parms_id() == seal::parms_id_zero) {
    ThrowBackendError(
        "OpenBumbleBee GPU: ciphertext upload has zero parms_id, size=" +
        std::to_string(host.size()) + " moduli=" +
        std::to_string(host.coeff_modulus_size()) + " degree=" +
        std::to_string(host.poly_modulus_degree()));
  }
  if (host.size() == 2 && host.data(1)[0] == UINT64_MAX) {
    ThrowBackendError(
        "OpenBumbleBee GPU cannot evaluate an unexpanded seeded SEAL "
        "ciphertext; serialize/load it before HE evaluation");
  }
  const auto chain = HostDataChainIndex(host_context, host.parms_id(),
                                        "ciphertext upload");
  const auto& parameters = device_context.get_context_data(chain).parms();
  const auto modulus_size = parameters.coeff_modulus().size();
  const auto degree = parameters.poly_modulus_degree();
  if (host.coeff_modulus_size() != modulus_size ||
      host.poly_modulus_degree() != degree) {
    throw std::runtime_error(
        "OpenBumbleBee GPU: SEAL/Phantom ciphertext parameters differ");
  }
  const auto bytes = host.size() * modulus_size * degree * sizeof(std::uint64_t);
  RecordBumblebeeMetric(BumblebeeMetricKind::kCipherUpload, 0.0, bytes);
  device.resize(device_context, chain, host.size(), cudaStreamPerThread);
  device.set_scale(host.scale());
  device.set_correction_factor(host.correction_factor());
  device.SetNoiseScaleDeg(1);
  device.set_ntt_form(host.is_ntt_form());
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kH2D, bytes);
    CheckCuda(cudaMemcpyAsync(device.data(), host.data(), bytes,
                              cudaMemcpyHostToDevice, cudaStreamPerThread),
              "ciphertext H2D");
    CheckCuda(CanonicalizePhantomRns(
                  device.data(), device_context.gpu_rns_tables().modulus(),
                  host.size(), modulus_size, degree, cudaStreamPerThread),
              "ciphertext canonicalization");
  }
}

void HostCipherPolynomialToDevice(const seal::SEALContext& host_context,
                                  const seal::Ciphertext& host,
                                  std::size_t polynomial,
                                  const PhantomContext& device_context,
                                  PhantomCiphertext& device) {
  if (host.size() == 2 && host.data(1)[0] == UINT64_MAX) {
    ThrowBackendError(
        "OpenBumbleBee GPU cannot evaluate an unexpanded seeded SEAL "
        "ciphertext; serialize/load it before HE evaluation");
  }
  if (polynomial >= host.size()) {
    throw std::runtime_error(
        "OpenBumbleBee GPU ciphertext polynomial index is out of range");
  }
  const auto chain = HostDataChainIndex(host_context, host.parms_id(),
                                        "ciphertext polynomial upload");
  const auto& parameters = device_context.get_context_data(chain).parms();
  const auto modulus_size = parameters.coeff_modulus().size();
  const auto degree = parameters.poly_modulus_degree();
  if (host.coeff_modulus_size() != modulus_size ||
      host.poly_modulus_degree() != degree) {
    throw std::runtime_error(
        "OpenBumbleBee GPU polynomial SEAL/Phantom parameters differ: "
        "chain=" +
        std::to_string(chain) + " host_moduli=" +
        std::to_string(host.coeff_modulus_size()) + " device_moduli=" +
        std::to_string(modulus_size) + " host_degree=" +
        std::to_string(host.poly_modulus_degree()) + " device_degree=" +
        std::to_string(degree));
  }
  const auto bytes = modulus_size * degree * sizeof(std::uint64_t);
  RecordBumblebeeMetric(BumblebeeMetricKind::kCipherUpload, 0.0, bytes);
  device.resize(device_context, chain, 1, cudaStreamPerThread);
  device.set_scale(host.scale());
  device.set_correction_factor(host.correction_factor());
  device.SetNoiseScaleDeg(1);
  device.set_ntt_form(host.is_ntt_form());
  ScopedBumblebeeMetric metric(BumblebeeMetricKind::kH2D, bytes);
  CheckCuda(cudaMemcpyAsync(device.data(), host.data(polynomial), bytes,
                            cudaMemcpyHostToDevice, cudaStreamPerThread),
            "ciphertext polynomial H2D");
  CheckCuda(CanonicalizePhantomRns(
                device.data(), device_context.gpu_rns_tables().modulus(), 1,
                modulus_size, degree, cudaStreamPerThread),
            "ciphertext polynomial canonicalization");
  if (VerifyGpuResults()) {
    SynchronizeCudaStream(
        "ciphertext polynomial canonicalization verification sync");
    std::vector<std::uint64_t> canonical(modulus_size * degree);
    CheckCuda(cudaMemcpy(canonical.data(), device.data(), bytes,
                         cudaMemcpyDeviceToHost),
              "ciphertext polynomial canonicalization verification D2H");
    for (std::size_t modulus = 0; modulus < modulus_size; ++modulus) {
      const auto q = parameters.coeff_modulus()[modulus].value();
      for (std::size_t coeff = 0; coeff < degree; ++coeff) {
        const auto index = modulus * degree + coeff;
        if (canonical[index] >= q ||
            canonical[index] != host.data(polynomial)[index] % q) {
          throw std::runtime_error(
              "OpenBumbleBee GPU ciphertext canonicalization mismatch");
        }
      }
    }
  }
}

void DeviceCipherPolynomialToHost(const PhantomCiphertext& device,
                                  seal::Ciphertext& host,
                                  std::size_t polynomial) {
  if (device.size() != 1 || polynomial >= host.size() ||
      device.coeff_modulus_size() != host.coeff_modulus_size() ||
      device.poly_modulus_degree() != host.poly_modulus_degree()) {
    throw std::runtime_error(
        "OpenBumbleBee GPU ciphertext polynomial metadata mismatch");
  }
  const auto bytes =
      device.coeff_modulus_size() * device.poly_modulus_degree() *
      sizeof(std::uint64_t);
  ScopedBumblebeeMetric metric(BumblebeeMetricKind::kD2H, bytes);
  CheckCuda(cudaMemcpy(host.data(polynomial), device.data(), bytes,
                       cudaMemcpyDeviceToHost),
            "ciphertext polynomial D2H");
}

void DeviceCipherToHost(const PhantomContext& device_context,
                        const PhantomCiphertext& device,
                        const seal::SEALContext& host_context,
                        seal::Ciphertext& host) {
  auto target = FindHostContextByModulusSize(host_context,
                                             device.coeff_modulus_size());
  const auto& parameters =
      device_context.get_context_data(device.chain_index()).parms();
  if (parameters.coeff_modulus().size() != device.coeff_modulus_size() ||
      parameters.poly_modulus_degree() != device.poly_modulus_degree()) {
    throw std::runtime_error(
        "OpenBumbleBee GPU: Phantom ciphertext metadata is inconsistent");
  }
  const auto bytes = device.size() * device.coeff_modulus_size() *
                     device.poly_modulus_degree() * sizeof(std::uint64_t);
  host.resize(host_context, target->parms_id(), device.size());
  host.scale() = device.scale();
  host.correction_factor() = device.correction_factor();
  host.is_ntt_form() = device.is_ntt_form();
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kD2H, bytes);
    CheckCuda(cudaMemcpy(host.data(), device.data(), bytes,
                         cudaMemcpyDeviceToHost),
              "ciphertext D2H");
  }
}

void CloneCipherOnCurrentStream(const PhantomContext& context,
                                const PhantomCiphertext& source,
                                PhantomCiphertext& destination) {
  destination.resize(context, source.chain_index(), source.size(),
                     cudaStreamPerThread);
  destination.set_scale(source.scale());
  destination.set_correction_factor(source.correction_factor());
  destination.SetNoiseScaleDeg(source.GetNoiseScaleDeg());
  destination.set_ntt_form(source.is_ntt_form());
  const auto bytes = source.size() * source.coeff_modulus_size() *
                     source.poly_modulus_degree() * sizeof(std::uint64_t);
  CheckCuda(cudaMemcpyAsync(destination.data(), source.data(), bytes,
                            cudaMemcpyDeviceToDevice, cudaStreamPerThread),
            "resident ciphertext D2D");
}

void HostPlainToDevice(const seal::SEALContext& host_context,
                       const seal::Plaintext& host,
                       const PhantomContext& device_context,
                       PhantomPlaintext& device) {
  const auto degree =
      host_context.key_context_data()->parms().poly_modulus_degree();
  const auto scheme = host_context.key_context_data()->parms().scheme();
  if (scheme == seal::scheme_type::bfv &&
      host.parms_id() == seal::parms_id_zero) {
    if (host.coeff_count() > degree) {
      throw std::runtime_error(
          "OpenBumbleBee GPU: BFV plaintext exceeds polynomial degree");
    }
    device.resize(1, degree, cudaStreamPerThread);
    device.set_chain_index(0);
    const_cast<double&>(device.scale()) = host.scale();
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kH2D,
                                 degree * sizeof(std::uint64_t));
    RecordBumblebeeMetric(BumblebeeMetricKind::kPlainUpload, 0.0,
                          degree * sizeof(std::uint64_t));
    CheckCuda(cudaMemsetAsync(device.data(), 0,
                              degree * sizeof(std::uint64_t),
                              cudaStreamPerThread),
              "BFV plaintext clear padding");
    if (host.coeff_count() != 0) {
      CheckCuda(cudaMemcpyAsync(device.data(), host.data(),
                                host.coeff_count() * sizeof(std::uint64_t),
                                cudaMemcpyHostToDevice,
                                cudaStreamPerThread),
                "BFV plaintext H2D");
    }
    return;
  }
  std::size_t chain = 0;
  std::size_t modulus_size = 0;
  if (host.parms_id() != seal::parms_id_zero) {
    chain = HostDataChainIndex(host_context, host.parms_id(),
                               "plaintext upload");
    modulus_size =
        device_context.get_context_data(chain).parms().coeff_modulus().size();
  } else {
    if (host.coeff_count() % degree != 0) {
      throw std::runtime_error(
          "OpenBumbleBee GPU: non-RNS plaintext cannot enter CKKS-form path");
    }
    modulus_size = host.coeff_count() / degree;
    const auto first_size = host_context.first_context_data()
                                ->parms()
                                .coeff_modulus()
                                .size();
    if (modulus_size == 0 || modulus_size > first_size) {
      throw std::runtime_error(
          "OpenBumbleBee GPU: invalid coefficient plaintext size");
    }
    chain = first_size + 1 - modulus_size;
  }
  const auto count = modulus_size * degree;
  if (host.coeff_count() < count) {
    throw std::runtime_error(
        "OpenBumbleBee GPU: plaintext coefficient buffer is too small");
  }
  device.resize(modulus_size, degree, cudaStreamPerThread);
  device.set_chain_index(chain);
  const_cast<double&>(device.scale()) = host.scale();
  RecordBumblebeeMetric(BumblebeeMetricKind::kPlainUpload, 0.0,
                        count * sizeof(std::uint64_t));
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kH2D,
                                 count * sizeof(std::uint64_t));
    CheckCuda(cudaMemcpyAsync(device.data(), host.data(),
                              count * sizeof(std::uint64_t),
                              cudaMemcpyHostToDevice, cudaStreamPerThread),
              "plaintext H2D");
    CheckCuda(CanonicalizePhantomRns(
                  device.data(), device_context.gpu_rns_tables().modulus(), 1,
                  modulus_size, degree, cudaStreamPerThread),
              "plaintext canonicalization");
  }
}

void DevicePlainToExistingHost(const PhantomPlaintext& device,
                               seal::Plaintext& host) {
  const auto count = device.coeff_count();
  if (host.coeff_count() != count) {
    throw std::runtime_error(
        "OpenBumbleBee GPU: plaintext coefficient count changed");
  }
  ScopedBumblebeeMetric metric(BumblebeeMetricKind::kD2H,
                               count * sizeof(std::uint64_t));
  CheckCuda(cudaMemcpy(host.data(), device.data(), count * sizeof(std::uint64_t),
                       cudaMemcpyDeviceToHost),
            "plaintext D2H");
}

void LoadPhantomPublicKey(PhantomCiphertext&& cipher,
                          PhantomPublicKey& public_key) {
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  cipher.save(stream);
  stream.seekg(0);
  public_key.load(stream);
}

std::shared_ptr<PhantomGaloisKey> TranslateGaloisKeys(
    const seal::SEALContext& host_context, const seal::GaloisKeys& host_keys,
    const std::vector<std::uint32_t>& elements,
    const std::shared_ptr<PhantomContext>& device_context) {
  auto result = std::make_shared<PhantomGaloisKey>();
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  const std::size_t key_count = elements.size();
  stream.write(reinterpret_cast<const char*>(&key_count), sizeof(key_count));
  for (const auto element : elements) {
    if (!host_keys.has_key(element)) {
      throw std::runtime_error(
          "OpenBumbleBee GPU: missing SEAL Galois key element");
    }
    std::vector<PhantomPublicKey> public_keys;
    const auto& decompositions = host_keys.key(element);
    public_keys.reserve(decompositions.size());
    for (const auto& decomposition : decompositions) {
      PhantomCiphertext cipher;
      HostCipherToDevice(host_context, decomposition.data(), *device_context,
                         cipher);
      // PhantomPublicKey::load serializes through host memory; complete this
      // one-time async bridge before entering that serializer.
      SynchronizeCudaStream("Galois-key ciphertext upload");
      public_keys.emplace_back();
      LoadPhantomPublicKey(std::move(cipher), public_keys.back());
    }
    const std::size_t decomposition_count = public_keys.size();
    stream.write(reinterpret_cast<const char*>(&decomposition_count),
                 sizeof(decomposition_count));
    for (const auto& public_key : public_keys) {
      public_key.save(stream);
    }
  }
  stream.seekg(0);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    result->load(stream);
    SynchronizeCudaStream("Phantom Galois-key translation");
  }
  return result;
}

#endif  // OPENBUMBLEBEE_ENABLE_PHANTOM

}  // namespace

BumblebeeHeBackend GetBumblebeeHeBackend() {
  static const BumblebeeHeBackend backend = [] {
    const char* raw = std::getenv("SPU_BUMBLEBEE_HE_BACKEND");
    const std::string value = Lower(raw == nullptr ? "gpu" : raw);
    if (value == "gpu" || value == "phantom") {
      return BumblebeeHeBackend::kGpu;
    }
    if (value == "cpu" || value == "seal") {
      return BumblebeeHeBackend::kCpu;
    }
    ThrowBackendError(
        "SPU_BUMBLEBEE_HE_BACKEND must be gpu/phantom or cpu/seal");
  }();
  return backend;
}

bool UsePhantomHeBackend() {
  return GetBumblebeeHeBackend() == BumblebeeHeBackend::kGpu;
}

const char* BumblebeeHeBackendName() {
  return UsePhantomHeBackend() ? "gpu" : "cpu";
}

void ValidateBumblebeeHeBackend() {
  if (!UsePhantomHeBackend()) {
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  static std::once_flag validated;
  std::call_once(validated, [] {
    int count = 0;
    CheckCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count <= 0) {
      ThrowBackendError(
          "OpenBumbleBee GPU backend requested, but no CUDA GPU is visible; "
          "there is no CPU fallback");
    }
    int device = -1;
    CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    CheckCuda(cudaGetDeviceProperties(&properties, device),
              "cudaGetDeviceProperties");
    SPDLOG_INFO("OPENBUMBLEBEE_HE_BACKEND backend=gpu device={} name={}",
                device, properties.name);
  });
#else
  ThrowBackendError(
      "OpenBumbleBee GPU backend requested, but PhantomFHE was not compiled; "
      "build with --config=gpu. There is no CPU fallback");
#endif
}

struct PhantomEvaluator::GpuState {
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  explicit GpuState(const seal::SEALContext& host_context,
                    const seal::GaloisKeys* host_galois_keys) {
    ValidateBumblebeeHeBackend();
    CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
    elements = GaloisElements(host_galois_keys);
    parameters = MakePhantomParameters(host_context, elements);
    context = GetOrCreatePhantomContext(parameters, device);
    if (host_context.key_context_data()->parms().scheme() ==
        seal::scheme_type::bfv) {
      // Phantom's BFV multiply_plain does not match SEAL coefficient-for-
      // coefficient for arbitrary batched plaintexts. Use Phantom's exact
      // scale=1 CKKS-form NTT multiply over the identical RNS moduli instead.
      // This is the same polynomial product, not approximate CKKS.
      bfv_multiply_parameters = MakeExactCkksFormParameters(host_context);
      bfv_multiply_context =
          GetOrCreatePhantomContext(bfv_multiply_parameters, device);
    }
    if (host_galois_keys != nullptr && !elements.empty()) {
      ScopedBumblebeeMetric cpu_metric(BumblebeeMetricKind::kGpuCpu);
      RecordBumblebeeMetric(BumblebeeMetricKind::kGaloisUpload, 0.0);
      galois_keys = TranslateGaloisKeys(host_context, *host_galois_keys,
                                        elements, context);
    }
  }

  int device = -1;
  phantom::EncryptionParameters parameters{phantom::scheme_type::none};
  phantom::EncryptionParameters bfv_multiply_parameters{
      phantom::scheme_type::none};
  std::vector<std::uint32_t> elements;
  std::shared_ptr<PhantomContext> context;
  std::shared_ptr<PhantomContext> bfv_multiply_context;
  std::shared_ptr<PhantomGaloisKey> galois_keys;
  std::unordered_map<const seal::Ciphertext*, std::size_t>
      multiply_cipher_indices;
  std::unordered_map<const seal::Plaintext*, std::size_t>
      multiply_plain_indices;
  std::vector<PhantomCiphertext> multiply_ciphers;
  std::vector<PhantomPlaintext> multiply_plains;
#endif
};

PhantomEvaluator::PhantomEvaluator(const seal::SEALContext& context,
                                   const seal::GaloisKeys* galois_keys)
    : context_(context), cpu_(std::make_unique<seal::Evaluator>(context)) {
  if (UsePhantomHeBackend()) {
    ValidateBumblebeeHeBackend();
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
    gpu_ = std::make_unique<GpuState>(context, galois_keys);
#endif
  }
}

PhantomEvaluator::~PhantomEvaluator() = default;

bool PhantomEvaluator::using_gpu() const { return gpu_ != nullptr; }

void PhantomEvaluator::prepare_multiply_add_inputs(
    const seal::Ciphertext* ciphers, std::size_t cipher_count,
    const seal::Plaintext* plains, std::size_t plain_count) const {
  if (!using_gpu()) {
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  if ((cipher_count != 0 && ciphers == nullptr) ||
      (plain_count != 0 && plains == nullptr)) {
    throw std::runtime_error(
        "OpenBumbleBee GPU multiply/add cache received null input");
  }
  clear_multiply_add_inputs();
  ScopedBumblebeeMetric cpu_metric(BumblebeeMetricKind::kGpuCpu);
  gpu_->multiply_cipher_indices.reserve(cipher_count);
  gpu_->multiply_plain_indices.reserve(plain_count);
  gpu_->multiply_ciphers.resize(cipher_count);
  gpu_->multiply_plains.resize(plain_count);
  for (std::size_t index = 0; index < cipher_count; ++index) {
    HostCipherToDevice(context_, ciphers[index], *gpu_->context,
                       gpu_->multiply_ciphers[index]);
    gpu_->multiply_cipher_indices.emplace(ciphers + index, index);
  }
  for (std::size_t index = 0; index < plain_count; ++index) {
    HostPlainToDevice(context_, plains[index], *gpu_->context,
                      gpu_->multiply_plains[index]);
    gpu_->multiply_plain_indices.emplace(plains + index, index);
  }
  SynchronizeCudaStream("multiply/add resident input upload");
#else
  (void)ciphers;
  (void)cipher_count;
  (void)plains;
  (void)plain_count;
#endif
}

void PhantomEvaluator::clear_multiply_add_inputs() const {
  if (!using_gpu()) {
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  gpu_->multiply_cipher_indices.clear();
  gpu_->multiply_plain_indices.clear();
  gpu_->multiply_ciphers.clear();
  gpu_->multiply_plains.clear();
#endif
}

void PhantomEvaluator::add_inplace(seal::Ciphertext& lhs,
                                   const seal::Ciphertext& rhs) const {
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    cpu_->add_inplace(lhs, rhs);
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    expected = lhs;
    cpu_->add_inplace(expected, rhs);
  }
  if (lhs.parms_id() != rhs.parms_id() || lhs.size() != rhs.size() ||
      lhs.is_ntt_form() != rhs.is_ntt_form() || lhs.scale() != rhs.scale() ||
      lhs.correction_factor() != rhs.correction_factor()) {
    throw std::runtime_error(
        "OpenBumbleBee GPU ciphertext add metadata mismatch");
  }
  PhantomCiphertext dlhs;
  PhantomCiphertext drhs;
  HostCipherToDevice(context_, lhs, *gpu_->context, dlhs);
  HostCipherToDevice(context_, rhs, *gpu_->context, drhs);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    CheckCuda(AddSubCipherPhantomRns(
                  dlhs.data(), drhs.data(),
                  gpu_->context->gpu_rns_tables().modulus(), dlhs.size(),
                  dlhs.coeff_modulus_size(), dlhs.poly_modulus_degree(),
                  false, cudaStreamPerThread),
              "exact ciphertext add");
    SynchronizeCudaStream("ciphertext add");
  }
  DeviceCipherToHost(*gpu_->context, dlhs, context_, lhs);
  VerifyCipherEquivalent(context_, expected, lhs, "add");
#endif
}

void PhantomEvaluator::sub_inplace(seal::Ciphertext& lhs,
                                   const seal::Ciphertext& rhs) const {
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    cpu_->sub_inplace(lhs, rhs);
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    expected = lhs;
    cpu_->sub_inplace(expected, rhs);
  }
  if (lhs.parms_id() != rhs.parms_id() || lhs.size() != rhs.size() ||
      lhs.is_ntt_form() != rhs.is_ntt_form() || lhs.scale() != rhs.scale() ||
      lhs.correction_factor() != rhs.correction_factor()) {
    throw std::runtime_error(
        "OpenBumbleBee GPU ciphertext subtract metadata mismatch");
  }
  PhantomCiphertext dlhs;
  PhantomCiphertext drhs;
  HostCipherToDevice(context_, lhs, *gpu_->context, dlhs);
  HostCipherToDevice(context_, rhs, *gpu_->context, drhs);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    CheckCuda(AddSubCipherPhantomRns(
                  dlhs.data(), drhs.data(),
                  gpu_->context->gpu_rns_tables().modulus(), dlhs.size(),
                  dlhs.coeff_modulus_size(), dlhs.poly_modulus_degree(), true,
                  cudaStreamPerThread),
              "exact ciphertext subtract");
    SynchronizeCudaStream("ciphertext subtract");
  }
  DeviceCipherToHost(*gpu_->context, dlhs, context_, lhs);
  VerifyCipherEquivalent(context_, expected, lhs, "subtract");
#endif
}

void PhantomEvaluator::negate(const seal::Ciphertext& input,
                              seal::Ciphertext& output) const {
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    cpu_->negate(input, output);
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    cpu_->negate(input, expected);
  }
  PhantomCiphertext device;
  HostCipherToDevice(context_, input, *gpu_->context, device);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    CheckCuda(NegateCipherPhantomRns(
                  device.data(), gpu_->context->gpu_rns_tables().modulus(),
                  device.size(), device.coeff_modulus_size(),
                  device.poly_modulus_degree(), cudaStreamPerThread),
              "exact ciphertext negate");
    SynchronizeCudaStream("ciphertext negate");
  }
  DeviceCipherToHost(*gpu_->context, device, context_, output);
  VerifyCipherEquivalent(context_, expected, output, "negate");
#endif
}

void PhantomEvaluator::add_plain_inplace(seal::Ciphertext& cipher,
                                         const seal::Plaintext& plain) const {
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    cpu_->add_plain_inplace(cipher, plain);
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  const bool coefficient_ckks =
      context_.key_context_data()->parms().scheme() ==
          seal::scheme_type::ckks &&
      !cipher.is_ntt_form();
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    expected = cipher;
    if (coefficient_ckks) {
      ExactSealAddSubPlainInplace(context_, expected, plain, false);
    } else {
      cpu_->add_plain_inplace(expected, plain);
    }
  }
  PhantomCiphertext dcipher;
  PhantomPlaintext dplain;
  HostCipherToDevice(context_, cipher, *gpu_->context, dcipher);
  HostPlainToDevice(context_, plain, *gpu_->context, dplain);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    if (coefficient_ckks) {
      CheckCuda(AddSubPlainPhantomRns(
                    dcipher.data(), dplain.data(),
                    gpu_->context->gpu_rns_tables().modulus(),
                    dcipher.coeff_modulus_size(),
                    dcipher.poly_modulus_degree(), false,
                    cudaStreamPerThread),
                "coefficient-form add plaintext");
    } else {
      phantom::add_plain_inplace(*gpu_->context, dcipher, dplain);
    }
    SynchronizeCudaStream("add plaintext");
  }
  DeviceCipherToHost(*gpu_->context, dcipher, context_, cipher);
  VerifyCipherEquivalent(context_, expected, cipher, "add_plain");
#endif
}

void PhantomEvaluator::sub_plain_inplace(seal::Ciphertext& cipher,
                                         const seal::Plaintext& plain) const {
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    cpu_->sub_plain_inplace(cipher, plain);
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  const bool coefficient_ckks =
      context_.key_context_data()->parms().scheme() ==
          seal::scheme_type::ckks &&
      !cipher.is_ntt_form();
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    expected = cipher;
    if (coefficient_ckks) {
      ExactSealAddSubPlainInplace(context_, expected, plain, true);
    } else {
      cpu_->sub_plain_inplace(expected, plain);
    }
  }
  PhantomCiphertext dcipher;
  PhantomPlaintext dplain;
  HostCipherToDevice(context_, cipher, *gpu_->context, dcipher);
  HostPlainToDevice(context_, plain, *gpu_->context, dplain);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    if (coefficient_ckks) {
      CheckCuda(AddSubPlainPhantomRns(
                    dcipher.data(), dplain.data(),
                    gpu_->context->gpu_rns_tables().modulus(),
                    dcipher.coeff_modulus_size(),
                    dcipher.poly_modulus_degree(), true, cudaStreamPerThread),
                "coefficient-form subtract plaintext");
    } else {
      phantom::sub_plain_inplace(*gpu_->context, dcipher, dplain);
    }
    SynchronizeCudaStream("subtract plaintext");
  }
  DeviceCipherToHost(*gpu_->context, dcipher, context_, cipher);
  VerifyCipherEquivalent(context_, expected, cipher, "sub_plain");
#endif
}

void PhantomEvaluator::multiply_plain(const seal::Ciphertext& cipher,
                                      const seal::Plaintext& plain,
                                      seal::Ciphertext& output) const {
  output = cipher;
  multiply_plain_inplace(output, plain);
}

void PhantomEvaluator::multiply_plain_inplace(
    seal::Ciphertext& cipher, const seal::Plaintext& plain) const {
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    cpu_->multiply_plain_inplace(cipher, plain);
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    expected = cipher;
    cpu_->multiply_plain_inplace(expected, plain);
  }
  const bool is_bfv = context_.key_context_data()->parms().scheme() ==
                      seal::scheme_type::bfv;
  if (is_bfv) {
    // SEAL's exact BFV multiply is: lift the t-polynomial into every qi,
    // NTT, perform the dyadic product, then inverse NTT. Keep SEAL's original
    // lift (a representation conversion), but execute both transforms and the
    // polynomial product through Phantom on CUDA. The temporary Phantom
    // context is CKKS-form solely to select its scale=1 RNS multiply kernel;
    // coefficients, moduli, degree, and BFV ciphertext semantics are unchanged.
    seal::Plaintext lifted;
    {
      ScopedBumblebeeMetric conversion(BumblebeeMetricKind::kConversion);
      auto context_data = context_.get_context_data(cipher.parms_id());
      if (!context_data ||
          !context_data->qualifiers().using_fast_plain_lift) {
        throw std::runtime_error(
            "OpenBumbleBee GPU requires the original Bumblebee fast BFV "
            "plaintext lift parameters; there is no CPU HE fallback");
      }
      const auto degree = context_data->parms().poly_modulus_degree();
      const auto modulus_size =
          context_data->parms().coeff_modulus().size();
      if (plain.coeff_count() > degree) {
        throw std::runtime_error(
            "OpenBumbleBee GPU BFV plaintext exceeds polynomial degree");
      }
      lifted.resize(degree * modulus_size);
      lifted.parms_id() = cipher.parms_id();
      lifted.scale() = plain.scale();
      const auto threshold = context_data->plain_upper_half_threshold();
      const auto increments = context_data->plain_upper_half_increment();
      for (std::size_t modulus = 0; modulus < modulus_size; ++modulus) {
        auto* destination = lifted.data() + modulus * degree;
        for (std::size_t coeff = 0; coeff < plain.coeff_count(); ++coeff) {
          const auto value = plain[coeff];
          destination[coeff] =
              value >= threshold ? value + increments[modulus] : value;
        }
      }
    }
    PhantomPlaintext dplain;
    HostPlainToDevice(context_, lifted, *gpu_->bfv_multiply_context, dplain);
    {
      ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
      nwt_2d_radix8_forward_inplace(
          dplain.data(), gpu_->bfv_multiply_context->gpu_rns_tables(),
          lifted.coeff_count() / cipher.poly_modulus_degree(), 0,
          cudaStreamPerThread);
      SynchronizeCudaStream("exact BFV plaintext NTT");
    }
    for (std::size_t poly = 0; poly < cipher.size(); ++poly) {
      PhantomCiphertext component;
      HostCipherPolynomialToDevice(context_, cipher, poly,
                                   *gpu_->bfv_multiply_context, component);
      {
        ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
        nwt_2d_radix8_forward_inplace(
            component.data(),
            gpu_->bfv_multiply_context->gpu_rns_tables(),
            component.coeff_modulus_size(), 0, cudaStreamPerThread);
        component.set_ntt_form(true);
        CheckCuda(DyadicProductPhantomRns(
                      component.data(), dplain.data(),
                      gpu_->bfv_multiply_context->gpu_rns_tables().modulus(),
                      1, component.coeff_modulus_size(),
                      component.poly_modulus_degree(), cudaStreamPerThread),
                  "exact BFV dyadic product");
        nwt_2d_radix8_backward_inplace(
            component.data(),
            gpu_->bfv_multiply_context->gpu_rns_tables(),
            component.coeff_modulus_size(), 0, cudaStreamPerThread);
        component.set_ntt_form(false);
        SynchronizeCudaStream("exact BFV plaintext multiply");
      }
      DeviceCipherPolynomialToHost(component, cipher, poly);
    }
    VerifyCipherEquivalent(context_, expected, cipher, "multiply_plain");
    return;
  }
  PhantomCiphertext dcipher;
  PhantomPlaintext dplain;
  HostCipherToDevice(context_, cipher, *gpu_->context, dcipher);
  HostPlainToDevice(context_, plain, *gpu_->context, dplain);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    phantom::multiply_plain_inplace(*gpu_->context, dcipher, dplain);
    SynchronizeCudaStream("multiply plaintext");
  }
  DeviceCipherToHost(*gpu_->context, dcipher, context_, cipher);
  VerifyCipherEquivalent(context_, expected, cipher, "multiply_plain");
#endif
}

void PhantomEvaluator::multiply_sub_plain_inplace(
    seal::Ciphertext& cipher, const seal::Plaintext& multiplier,
    const seal::Plaintext& subtractor) const {
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    cpu_->multiply_plain_inplace(cipher, multiplier);
    cpu_->sub_plain_inplace(cipher, subtractor);
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    expected = cipher;
    cpu_->multiply_plain_inplace(expected, multiplier);
    cpu_->sub_plain_inplace(expected, subtractor);
  }
  PhantomCiphertext dcipher;
  PhantomPlaintext dmultiplier;
  PhantomPlaintext dsubtractor;
  HostCipherToDevice(context_, cipher, *gpu_->context, dcipher);
  HostPlainToDevice(context_, multiplier, *gpu_->context, dmultiplier);
  HostPlainToDevice(context_, subtractor, *gpu_->context, dsubtractor);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    phantom::multiply_plain_inplace(*gpu_->context, dcipher, dmultiplier);
    phantom::sub_plain_inplace(*gpu_->context, dcipher, dsubtractor);
    SynchronizeCudaStream("multiply/subtract plaintext sequence");
  }
  DeviceCipherToHost(*gpu_->context, dcipher, context_, cipher);
  VerifyCipherEquivalent(context_, expected, cipher, "multiply_sub_plain");
#endif
}

void PhantomEvaluator::apply_galois_inplace(
    seal::Ciphertext& cipher, std::uint32_t galois_elt,
    const seal::GaloisKeys& galois_keys) const {
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    cpu_->apply_galois_inplace(cipher, galois_elt, galois_keys);
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    expected = cipher;
    cpu_->apply_galois_inplace(expected, galois_elt, galois_keys);
  }
  if (!gpu_->galois_keys) {
    throw std::runtime_error(
        "OpenBumbleBee GPU evaluator was constructed without Galois keys");
  }
  PhantomCiphertext dcipher;
  HostCipherToDevice(context_, cipher, *gpu_->context, dcipher);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    phantom::apply_galois_inplace(*gpu_->context, dcipher, galois_elt,
                                  *gpu_->galois_keys);
    SynchronizeCudaStream("apply Galois");
  }
  DeviceCipherToHost(*gpu_->context, dcipher, context_, cipher);
  VerifyCipherEquivalent(context_, expected, cipher, "apply_galois");
#endif
}

void PhantomEvaluator::mod_switch_to_inplace(
    seal::Ciphertext& cipher, seal::parms_id_type target) const {
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    cpu_->mod_switch_to_inplace(cipher, target);
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    expected = cipher;
    ExactSealModSwitchToInplace(context_, expected, target);
  }
  const auto target_chain =
      HostDataChainIndex(context_, target, "modulus-switch target");
  PhantomCiphertext dcipher;
  HostCipherToDevice(context_, cipher, *gpu_->context, dcipher);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    ExactModSwitchToInplace(*gpu_->context, dcipher, target_chain);
    SynchronizeCudaStream("modulus switch");
  }
  DeviceCipherToHost(*gpu_->context, dcipher, context_, cipher);
  VerifyCipherEquivalent(context_, expected, cipher, "mod_switch");
#endif
}

void PhantomEvaluator::ntt_inplace(seal::Ciphertext& cipher) const {
  if (cipher.is_ntt_form()) {
    return;
  }
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    cpu_->transform_to_ntt_inplace(cipher);
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    expected = cipher;
    cpu_->transform_to_ntt_inplace(expected);
  }
  PhantomCiphertext device;
  HostCipherToDevice(context_, cipher, *gpu_->context, device);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    ScopedBumblebeeMetric kernel_metric(
        BumblebeeMetricKind::kKernelSubmit);
    const auto polynomial_stride =
        device.coeff_modulus_size() * device.poly_modulus_degree();
    for (std::size_t poly = 0; poly < device.size(); ++poly) {
      nwt_2d_radix8_forward_inplace(
          device.data() + poly * polynomial_stride,
          gpu_->context->gpu_rns_tables(),
          device.coeff_modulus_size(), 0, cudaStreamPerThread);
    }
    device.set_ntt_form(true);
  }
  SynchronizeCudaStream("ciphertext NTT");
  DeviceCipherToHost(*gpu_->context, device, context_, cipher);
  VerifyCipherEquivalent(context_, expected, cipher, "NTT");
#endif
}

void PhantomEvaluator::inverse_ntt_inplace(seal::Ciphertext& cipher) const {
  if (!cipher.is_ntt_form()) {
    return;
  }
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    cpu_->transform_from_ntt_inplace(cipher);
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    expected = cipher;
    cpu_->transform_from_ntt_inplace(expected);
  }
  PhantomCiphertext device;
  HostCipherToDevice(context_, cipher, *gpu_->context, device);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    ScopedBumblebeeMetric kernel_metric(
        BumblebeeMetricKind::kKernelSubmit);
    const auto polynomial_stride =
        device.coeff_modulus_size() * device.poly_modulus_degree();
    for (std::size_t poly = 0; poly < device.size(); ++poly) {
      nwt_2d_radix8_backward_inplace(
          device.data() + poly * polynomial_stride,
          gpu_->context->gpu_rns_tables(),
          device.coeff_modulus_size(), 0, cudaStreamPerThread);
    }
    device.set_ntt_form(false);
  }
  SynchronizeCudaStream("ciphertext inverse NTT");
  DeviceCipherToHost(*gpu_->context, device, context_, cipher);
  VerifyCipherEquivalent(context_, expected, cipher, "inverse_NTT");
#endif
}

void PhantomEvaluator::ntt_inplace(seal::Plaintext& plain) const {
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    auto context_data = context_.get_context_data(plain.parms_id());
    if (!context_data) {
      throw std::runtime_error("OpenBumbleBee CPU plaintext NTT level missing");
    }
    const auto modulus_size = context_data->parms().coeff_modulus().size();
    const auto degree = context_data->parms().poly_modulus_degree();
    if (plain.coeff_count() != modulus_size * degree) {
      throw std::runtime_error(
          "OpenBumbleBee CPU plaintext NTT buffer has invalid size");
    }
    auto* data = plain.data();
    const auto* tables = context_data->small_ntt_tables();
    for (std::size_t modulus = 0; modulus < modulus_size; ++modulus) {
      seal::util::ntt_negacyclic_harvey(data, tables[modulus]);
      data += degree;
    }
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  seal::Plaintext expected;
  if (VerifyGpuResults()) {
    expected = plain;
    auto context_data = context_.get_context_data(expected.parms_id());
    auto* data = expected.data();
    const auto degree = context_data->parms().poly_modulus_degree();
    const auto& moduli = context_data->parms().coeff_modulus();
    const auto* tables = context_data->small_ntt_tables();
    for (std::size_t modulus = 0; modulus < moduli.size(); ++modulus) {
      seal::util::ntt_negacyclic_harvey(data, tables[modulus]);
      data += degree;
    }
  }
  PhantomPlaintext device;
  HostPlainToDevice(context_, plain, *gpu_->context, device);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    const auto modulus_size = device.coeff_count() /
                              context_.key_context_data()
                                  ->parms()
                                  .poly_modulus_degree();
    nwt_2d_radix8_forward_inplace(
        device.data(), gpu_->context->gpu_rns_tables(), modulus_size, 0,
        cudaStreamPerThread);
    SynchronizeCudaStream("plaintext NTT");
  }
  DevicePlainToExistingHost(device, plain);
  VerifyPlainEquivalent(context_, expected, plain, "plaintext_NTT");
#endif
}

void PhantomEvaluator::inverse_ntt_inplace(seal::Plaintext& plain) const {
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    auto context_data = context_.get_context_data(plain.parms_id());
    if (!context_data) {
      throw std::runtime_error(
          "OpenBumbleBee CPU plaintext inverse-NTT level missing");
    }
    const auto modulus_size = context_data->parms().coeff_modulus().size();
    const auto degree = context_data->parms().poly_modulus_degree();
    if (plain.coeff_count() != modulus_size * degree) {
      throw std::runtime_error(
          "OpenBumbleBee CPU plaintext inverse-NTT buffer has invalid size");
    }
    auto* data = plain.data();
    const auto* tables = context_data->small_ntt_tables();
    for (std::size_t modulus = 0; modulus < modulus_size; ++modulus) {
      seal::util::inverse_ntt_negacyclic_harvey(data, tables[modulus]);
      data += degree;
    }
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  seal::Plaintext expected;
  if (VerifyGpuResults()) {
    expected = plain;
    auto context_data = context_.get_context_data(expected.parms_id());
    auto* data = expected.data();
    const auto degree = context_data->parms().poly_modulus_degree();
    const auto& moduli = context_data->parms().coeff_modulus();
    const auto* tables = context_data->small_ntt_tables();
    for (std::size_t modulus = 0; modulus < moduli.size(); ++modulus) {
      seal::util::inverse_ntt_negacyclic_harvey(data, tables[modulus]);
      data += degree;
    }
  }
  PhantomPlaintext device;
  HostPlainToDevice(context_, plain, *gpu_->context, device);
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    const auto modulus_size = device.coeff_count() /
                              context_.key_context_data()
                                  ->parms()
                                  .poly_modulus_degree();
    nwt_2d_radix8_backward_inplace(
        device.data(), gpu_->context->gpu_rns_tables(), modulus_size, 0,
        cudaStreamPerThread);
    SynchronizeCudaStream("plaintext inverse NTT");
  }
  DevicePlainToExistingHost(device, plain);
  VerifyPlainEquivalent(context_, expected, plain, "plaintext_inverse_NTT");
#endif
}

void PhantomEvaluator::packing_butterfly_inplace(
    seal::Ciphertext* even, const seal::Ciphertext* odd, std::size_t shift,
    std::uint32_t galois_elt, const seal::GaloisKeys& galois_keys) const {
  if (even == nullptr || (even->size() == 0 && (odd == nullptr || odd->size() == 0))) {
    throw std::runtime_error("OpenBumbleBee packing butterfly has no input");
  }
  if (!using_gpu()) {
    seal::Ciphertext shifted_odd;
    if (odd != nullptr && odd->size() != 0) {
      shifted_odd = *odd;
      ExactSealNegacyclicRightShiftInplace(context_, shifted_odd, shift);
    }
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    if (even->size() != 0) {
      seal::Ciphertext sum = *even;
      if (shifted_odd.size() != 0) {
        cpu_->sub_inplace(*even, shifted_odd);
        cpu_->add_inplace(sum, shifted_odd);
      }
      cpu_->apply_galois_inplace(*even, galois_elt, galois_keys);
      cpu_->add_inplace(*even, sum);
    } else {
      cpu_->negate(shifted_odd, *even);
      cpu_->apply_galois_inplace(*even, galois_elt, galois_keys);
      cpu_->add_inplace(*even, shifted_odd);
    }
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    seal::Ciphertext shifted_odd;
    if (odd != nullptr && odd->size() != 0) {
      shifted_odd = *odd;
      ExactSealNegacyclicRightShiftInplace(context_, shifted_odd, shift);
    }
    if (even->size() != 0) {
      expected = *even;
      seal::Ciphertext sum = expected;
      if (shifted_odd.size() != 0) {
        cpu_->sub_inplace(expected, shifted_odd);
        cpu_->add_inplace(sum, shifted_odd);
      }
      cpu_->apply_galois_inplace(expected, galois_elt, galois_keys);
      cpu_->add_inplace(expected, sum);
    } else {
      cpu_->negate(shifted_odd, expected);
      cpu_->apply_galois_inplace(expected, galois_elt, galois_keys);
      cpu_->add_inplace(expected, shifted_odd);
    }
  }

  if (!gpu_->galois_keys) {
    throw std::runtime_error(
        "OpenBumbleBee GPU evaluator was constructed without Galois keys");
  }
  PhantomCiphertext device_even;
  PhantomCiphertext device_odd;
  const bool has_even = even->size() != 0;
  const bool has_odd = odd != nullptr && odd->size() != 0;
  if (has_even) {
    HostCipherToDevice(context_, *even, *gpu_->context, device_even);
  }
  if (has_odd) {
    HostCipherToDevice(context_, *odd, *gpu_->context, device_odd);
  }
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    if (has_odd && shift != 0) {
      PhantomCiphertext shifted = device_odd;
      CheckCuda(NegacyclicRightShiftPhantomRns(
                    device_odd.data(), shifted.data(),
                    gpu_->context->gpu_rns_tables().modulus(),
                    device_odd.size(), device_odd.coeff_modulus_size(),
                    device_odd.poly_modulus_degree(), shift,
                    cudaStreamPerThread),
                "packing negacyclic right shift");
      device_odd = std::move(shifted);
    }
    if (has_even) {
      PhantomCiphertext sum = device_even;
      if (has_odd) {
        phantom::sub_inplace(*gpu_->context, device_even, device_odd);
        phantom::add_inplace(*gpu_->context, sum, device_odd);
      }
      phantom::apply_galois_inplace(*gpu_->context, device_even, galois_elt,
                                    *gpu_->galois_keys);
      phantom::add_inplace(*gpu_->context, device_even, sum);
    } else {
      device_even = device_odd;
      phantom::negate_inplace(*gpu_->context, device_even);
      phantom::apply_galois_inplace(*gpu_->context, device_even, galois_elt,
                                    *gpu_->galois_keys);
      phantom::add_inplace(*gpu_->context, device_even, device_odd);
    }
    SynchronizeCudaStream("packing butterfly");
  }
  DeviceCipherToHost(*gpu_->context, device_even, context_, *even);
  VerifyCipherEquivalent(context_, expected, *even, "packing_butterfly");
#endif
}

void PhantomEvaluator::packing_butterfly_tree_inplace(
    seal::Ciphertext* ciphers, std::size_t count, std::size_t tree_width,
    const std::vector<std::size_t>& shifts,
    const std::vector<std::uint32_t>& galois_elts,
    const seal::GaloisKeys& galois_keys) const {
  if (ciphers == nullptr || count == 0 || count > tree_width ||
      shifts.size() != galois_elts.size() ||
      (std::size_t{1} << shifts.size()) != tree_width) {
    throw std::runtime_error("OpenBumbleBee invalid packing tree");
  }

  auto evaluate_host_tree = [&](std::vector<seal::Ciphertext>& host) {
    host.resize(tree_width);
    for (std::size_t stage = 0; stage < shifts.size(); ++stage) {
      const auto half = tree_width >> (stage + 1);
      for (std::size_t index = 0; index < half; ++index) {
        auto& even = host[index];
        auto& odd = host[index + half];
        if (even.size() == 0 && odd.size() == 0) {
          continue;
        }
        ExactSealNegacyclicRightShiftInplace(context_, odd, shifts[stage]);
        if (even.size() != 0) {
          seal::Ciphertext sum = even;
          if (odd.size() != 0) {
            cpu_->sub_inplace(even, odd);
            cpu_->add_inplace(sum, odd);
          }
          cpu_->apply_galois_inplace(even, galois_elts[stage], galois_keys);
          cpu_->add_inplace(even, sum);
        } else {
          cpu_->negate(odd, even);
          cpu_->apply_galois_inplace(even, galois_elts[stage], galois_keys);
          cpu_->add_inplace(even, odd);
        }
      }
    }
  };

  if (!using_gpu()) {
    std::vector<seal::Ciphertext> host(ciphers, ciphers + count);
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    evaluate_host_tree(host);
    ciphers[0] = std::move(host[0]);
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  if (!gpu_->galois_keys) {
    throw std::runtime_error(
        "OpenBumbleBee GPU evaluator was constructed without Galois keys");
  }
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    std::vector<seal::Ciphertext> host(ciphers, ciphers + count);
    evaluate_host_tree(host);
    expected = std::move(host[0]);
  }

  std::vector<PhantomCiphertext> device(tree_width);
  yacl::parallel_for(0, count, [&](std::int64_t begin, std::int64_t end) {
    for (auto index = begin; index < end; ++index) {
      if (ciphers[index].size() == 0) {
        continue;
      }
      HostCipherToDevice(context_, ciphers[index], *gpu_->context,
                         device[index]);
    }
  });
  SynchronizeCudaDevice("packing tree upload");

  for (std::size_t stage = 0; stage < shifts.size(); ++stage) {
    const auto half = tree_width >> (stage + 1);
    {
      ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
      yacl::parallel_for(0, half,
                         [&](std::int64_t begin, std::int64_t end) {
        for (auto index = begin; index < end; ++index) {
          auto& even = device[index];
          auto& odd = device[index + half];
          const bool has_even = even.size() != 0;
          const bool has_odd = odd.size() != 0;
          if (!has_even && !has_odd) {
            continue;
          }
          if (has_odd && shifts[stage] != 0) {
            PhantomCiphertext shifted = odd;
            CheckCuda(NegacyclicRightShiftPhantomRns(
                          odd.data(), shifted.data(),
                          gpu_->context->gpu_rns_tables().modulus(), odd.size(),
                          odd.coeff_modulus_size(), odd.poly_modulus_degree(),
                          shifts[stage], cudaStreamPerThread),
                      "packing-tree negacyclic right shift");
            odd = std::move(shifted);
          }
          if (has_even) {
            PhantomCiphertext sum = even;
            if (has_odd) {
              phantom::sub_inplace(*gpu_->context, even, odd);
              phantom::add_inplace(*gpu_->context, sum, odd);
            }
            phantom::apply_galois_inplace(*gpu_->context, even,
                                          galois_elts[stage],
                                          *gpu_->galois_keys);
            phantom::add_inplace(*gpu_->context, even, sum);
          } else {
            even = odd;
            phantom::negate_inplace(*gpu_->context, even);
            phantom::apply_galois_inplace(*gpu_->context, even,
                                          galois_elts[stage],
                                          *gpu_->galois_keys);
            phantom::add_inplace(*gpu_->context, even, odd);
          }
        }
      });
      SynchronizeCudaDevice("packing tree level");
    }
  }
  DeviceCipherToHost(*gpu_->context, device[0], context_, ciphers[0]);
  VerifyCipherEquivalent(context_, expected, ciphers[0],
                         "packing_butterfly_tree");
#endif
}

void PhantomEvaluator::multiply_add_sequence(
    const seal::Ciphertext* ciphers, const seal::Plaintext* plains,
    std::size_t count, seal::Ciphertext& output, bool inverse_ntt_output,
    seal::parms_id_type output_target) const {
  if (count == 0) {
    output.release();
    return;
  }
  if (!using_gpu()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    cpu_->multiply_plain(ciphers[0], plains[0], output);
    for (std::size_t i = 1; i < count; ++i) {
      seal::Ciphertext term;
      cpu_->multiply_plain(ciphers[i], plains[i], term);
      cpu_->add_inplace(output, term);
    }
    if (inverse_ntt_output && output.is_ntt_form()) {
      cpu_->transform_from_ntt_inplace(output);
    }
    if (output_target != seal::parms_id_zero &&
        output.parms_id() != output_target) {
      ExactSealModSwitchToInplace(context_, output, output_target);
    }
    return;
  }
#ifdef OPENBUMBLEBEE_ENABLE_PHANTOM
  seal::Ciphertext expected;
  if (VerifyGpuResults()) {
    cpu_->multiply_plain(ciphers[0], plains[0], expected);
    for (std::size_t i = 1; i < count; ++i) {
      seal::Ciphertext term;
      cpu_->multiply_plain(ciphers[i], plains[i], term);
      cpu_->add_inplace(expected, term);
    }
    if (inverse_ntt_output && expected.is_ntt_form()) {
      cpu_->transform_from_ntt_inplace(expected);
    }
    if (output_target != seal::parms_id_zero &&
        expected.parms_id() != output_target) {
      ExactSealModSwitchToInplace(context_, expected, output_target);
    }
  }
  PhantomCiphertext accumulator;
  std::size_t cipher_cache_hits = 0;
  std::size_t plain_cache_hits = 0;
  for (std::size_t i = 0; i < count; ++i) {
    PhantomCiphertext cipher;
    PhantomPlaintext plain;
    const PhantomPlaintext* device_plain = nullptr;
    const auto cipher_cached = gpu_->multiply_cipher_indices.find(ciphers + i);
    if (cipher_cached != gpu_->multiply_cipher_indices.end()) {
      CloneCipherOnCurrentStream(*gpu_->context,
                                 gpu_->multiply_ciphers[cipher_cached->second],
                                 cipher);
      ++cipher_cache_hits;
    } else {
      HostCipherToDevice(context_, ciphers[i], *gpu_->context, cipher);
    }
    const auto plain_cached = gpu_->multiply_plain_indices.find(plains + i);
    if (plain_cached != gpu_->multiply_plain_indices.end()) {
      device_plain = &gpu_->multiply_plains[plain_cached->second];
      ++plain_cache_hits;
    } else {
      HostPlainToDevice(context_, plains[i], *gpu_->context, plain);
      device_plain = &plain;
    }
    {
      ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
      ScopedBumblebeeMetric kernel_metric(
          BumblebeeMetricKind::kKernelSubmit);
      phantom::multiply_plain_inplace(*gpu_->context, cipher, *device_plain);
      if (i == 0) {
        accumulator = std::move(cipher);
      } else {
        phantom::add_inplace(*gpu_->context, accumulator, cipher);
      }
    }
  }
  if (inverse_ntt_output && accumulator.is_ntt_form()) {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    ScopedBumblebeeMetric kernel_metric(
        BumblebeeMetricKind::kKernelSubmit);
    const auto polynomial_stride = accumulator.coeff_modulus_size() *
                                   accumulator.poly_modulus_degree();
    for (std::size_t poly = 0; poly < accumulator.size(); ++poly) {
      nwt_2d_radix8_backward_inplace(
          accumulator.data() + poly * polynomial_stride,
          gpu_->context->gpu_rns_tables(),
          accumulator.coeff_modulus_size(), 0, cudaStreamPerThread);
    }
    accumulator.set_ntt_form(false);
  }
  if (output_target != seal::parms_id_zero) {
    const auto target_chain = HostDataChainIndex(
        context_, output_target, "multiply/add output modulus target");
    if (accumulator.chain_index() != target_chain) {
      ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
      ScopedBumblebeeMetric kernel_metric(
          BumblebeeMetricKind::kKernelSubmit);
      ExactModSwitchToInplace(*gpu_->context, accumulator, target_chain);
    }
  }
  {
    ScopedBumblebeeMetric metric(BumblebeeMetricKind::kHe);
    SynchronizeCudaStream("multiply/add sequence");
  }
  if (cipher_cache_hits != 0) {
    RecordBumblebeeMetric(BumblebeeMetricKind::kCipherCacheHit, 0.0, 0,
                          cipher_cache_hits);
  }
  if (plain_cache_hits != 0) {
    RecordBumblebeeMetric(BumblebeeMetricKind::kPlainCacheHit, 0.0, 0,
                          plain_cache_hits);
  }
  DeviceCipherToHost(*gpu_->context, accumulator, context_, output);
  VerifyCipherEquivalent(context_, expected, output, "multiply_add_sequence");
#endif
}

}  // namespace spu::mpc::bumblebee
