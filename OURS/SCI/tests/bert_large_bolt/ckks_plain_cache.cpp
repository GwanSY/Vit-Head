#include "ckks_plain_cache.h"

#include "linear_ckks.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace {

using seal::EncryptionParameters;
using seal::Plaintext;
using seal::SEALContext;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("CKKS host/device plaintext cache: " + message);
}

void validate_parameter_level(
    const EncryptionParameters &host_parms,
    const phantom::EncryptionParameters &device_parms,
    const std::string &label) {
  if (host_parms.poly_modulus_degree() !=
      device_parms.poly_modulus_degree()) {
    std::ostringstream os;
    os << label << " polynomial degree mismatch (SEAL "
       << host_parms.poly_modulus_degree() << ", Phantom "
       << device_parms.poly_modulus_degree() << ')';
    fail(os.str());
  }

  const auto &host_moduli = host_parms.coeff_modulus();
  const auto &device_moduli = device_parms.coeff_modulus();
  if (host_moduli.size() != device_moduli.size()) {
    std::ostringstream os;
    os << label << " coefficient-modulus count mismatch (SEAL "
       << host_moduli.size() << ", Phantom " << device_moduli.size() << ')';
    fail(os.str());
  }

  for (std::size_t i = 0; i < host_moduli.size(); ++i) {
    if (host_moduli[i].value() != device_moduli[i].value()) {
      std::ostringstream os;
      os << label << " coefficient modulus[" << i
         << "] mismatch (SEAL " << host_moduli[i].value() << ", Phantom "
         << device_moduli[i].value() << ')';
      fail(os.str());
    }
  }
}

template <typename HostContextData, typename DeviceContextData>
void validate_ntt_level(const HostContextData &host_data,
                        const DeviceContextData &device_data,
                        const std::string &label) {
  const auto *host_tables = host_data.small_ntt_tables();
  const auto &device_tables = device_data.small_ntt_tables();
  if (host_tables == nullptr || !device_tables) {
    fail(label + " NTT tables are missing");
  }

  const auto modulus_count = host_data.parms().coeff_modulus().size();
  if (device_tables->size() != modulus_count) {
    std::ostringstream os;
    os << label << " NTT-table count mismatch (SEAL " << modulus_count
       << ", Phantom " << device_tables->size() << ')';
    fail(os.str());
  }

  // A raw NTT-plaintext copy needs more than equal primes: both backends must
  // use the same evaluation ordering.  Comparing the complete forward root
  // table proves that a SEAL NTT coefficient has the same slot in Phantom.
  for (std::size_t modulus = 0; modulus < modulus_count; ++modulus) {
    const auto &host_ntt = host_tables[modulus];
    const auto &device_ntt = device_tables->get_ntt_at(modulus);
    if (host_ntt.get_root() != device_ntt.get_root() ||
        host_ntt.coeff_count() != device_ntt.coeff_count() ||
        host_ntt.coeff_count_power() != device_ntt.coeff_count_power()) {
      std::ostringstream os;
      os << label << " NTT metadata mismatch at modulus " << modulus;
      fail(os.str());
    }
    const auto &device_roots = device_ntt.get_from_root_powers();
    if (device_roots.size() != host_ntt.coeff_count()) {
      fail(label + " Phantom forward NTT table has an invalid size");
    }
    for (std::size_t index = 0; index < host_ntt.coeff_count(); ++index) {
      if (host_ntt.get_from_root_powers(index).operand !=
          device_roots[index]) {
        std::ostringstream os;
        os << label << " forward NTT table mismatch at modulus " << modulus
           << ", index " << index;
        fail(os.str());
      }
    }
  }
}

const CkksHostDeviceLevel &find_host_level(
    const std::vector<CkksHostDeviceLevel> &levels,
    const seal::parms_id_type &parms_id) {
  const auto it = std::find_if(
      levels.begin(), levels.end(), [&](const CkksHostDeviceLevel &level) {
        return level.seal_parms_id == parms_id;
      });
  if (it == levels.end()) {
    fail("SEAL parms_id is not a CKKS data level in the validated context");
  }
  return *it;
}

const CkksHostDeviceLevel &find_device_level(
    const std::vector<CkksHostDeviceLevel> &levels,
    std::size_t chain_index) {
  const auto it = std::find_if(
      levels.begin(), levels.end(), [&](const CkksHostDeviceLevel &level) {
        return level.phantom_chain_index == chain_index;
      });
  if (it == levels.end()) {
    std::ostringstream os;
    os << "Phantom chain index " << chain_index
       << " is not a CKKS data level in the validated context";
    fail(os.str());
  }
  return *it;
}

bool scales_close(double actual, double expected, double relative_tolerance) {
  const double magnitude =
      std::max({1.0, std::abs(actual), std::abs(expected)});
  return std::abs(actual - expected) <= relative_tolerance * magnitude;
}

PhantomPlaintext upload_plaintext(
    HE_CKKS &he, const Plaintext &source, const CkksPlainUploadSpec &spec,
    const std::vector<CkksHostDeviceLevel> &levels, const std::string &label,
    std::size_t &plaintext_count, std::size_t &device_bytes) {
  if (source.parms_id() == seal::parms_id_zero) {
    fail(label + " has parms_id_zero; raw non-CKKS plaintext upload refused");
  }
  if (spec.scale_relative_tolerance < 0.0 ||
      !std::isfinite(spec.scale_relative_tolerance)) {
    fail(label + " has an invalid scale tolerance");
  }

  const Plaintext *prepared = &source;
  Plaintext level_adjusted;
  if (spec.target_parms_id && source.parms_id() != *spec.target_parms_id) {
    // SEAL validates that the requested target is downstream in the same
    // modulus chain. This is a real CKKS modulus switch, not scale relabeling.
    find_host_level(levels, *spec.target_parms_id);
    level_adjusted = source;
    try {
      he.evaluator->mod_switch_to_inplace(level_adjusted,
                                          *spec.target_parms_id);
    } catch (const std::exception &error) {
      fail(label + " cannot be modulus-switched to its requested level: " +
           error.what());
    }
    prepared = &level_adjusted;
  }

  const auto &host_level = find_host_level(levels, prepared->parms_id());
  if (spec.require_ntt_form && !prepared->is_ntt_form()) {
    fail(label + " is not in NTT form");
  }
  if (!std::isfinite(prepared->scale()) || prepared->scale() <= 0.0) {
    fail(label + " has a non-finite or non-positive CKKS scale");
  }
  if (spec.expected_scale &&
      (!std::isfinite(*spec.expected_scale) || *spec.expected_scale <= 0.0 ||
       !scales_close(prepared->scale(), *spec.expected_scale,
                     spec.scale_relative_tolerance))) {
    std::ostringstream os;
    os << label << " scale mismatch (encoded " << prepared->scale()
       << ", expected "
       << (spec.expected_scale ? *spec.expected_scale : 0.0) << ')';
    fail(os.str());
  }

  const auto expected_coeff_count =
      host_level.coeff_modulus_size * he.poly_modulus_degree;
  if (prepared->coeff_count() != expected_coeff_count) {
    std::ostringstream os;
    os << label << " coefficient count mismatch (plaintext "
       << prepared->coeff_count() << ", context " << expected_coeff_count
       << ')';
    fail(os.str());
  }

  PhantomPlaintext device_plain;
  UnifiedPlaintext::to_device(*he.context, *prepared, *he.dcontext,
                              device_plain);

  if (device_plain.chain_index() != host_level.phantom_chain_index) {
    std::ostringstream os;
    os << label << " mapped to Phantom chain "
       << device_plain.chain_index() << " instead of validated chain "
       << host_level.phantom_chain_index;
    fail(os.str());
  }
  if (device_plain.coeff_count() != expected_coeff_count) {
    fail(label + " device coefficient count changed during upload");
  }
  if (!scales_close(device_plain.scale(), prepared->scale(), 0.0)) {
    fail(label + " device scale changed during upload");
  }

  ++plaintext_count;
  device_bytes += device_plain.coeff_count() * sizeof(std::uint64_t);
  return device_plain;
}

std::vector<PhantomPlaintext> upload_vector(
    HE_CKKS &he, const std::vector<Plaintext> &source,
    const CkksPlainUploadSpec &spec,
    const std::vector<CkksHostDeviceLevel> &levels,
    const std::string &label, std::size_t &plaintext_count,
    std::size_t &device_bytes) {
  std::vector<PhantomPlaintext> result;
  result.reserve(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    result.emplace_back(upload_plaintext(
        he, source[i], spec, levels,
        label + '[' + std::to_string(i) + ']', plaintext_count,
        device_bytes));
  }
  return result;
}

DevicePlainMatrix upload_matrix(
    HE_CKKS &he, const std::vector<std::vector<Plaintext>> &source,
    const CkksPlainUploadSpec &spec,
    const std::vector<CkksHostDeviceLevel> &levels,
    const std::string &label, std::size_t &plaintext_count,
    std::size_t &device_bytes) {
  DevicePlainMatrix result;
  result.reserve(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    result.emplace_back(upload_vector(
        he, source[i], spec, levels,
        label + '[' + std::to_string(i) + ']', plaintext_count,
        device_bytes));
  }
  return result;
}

DevicePackedWeight upload_packed_weight(
    HE_CKKS &he,
    const std::pair<std::vector<std::vector<Plaintext>>,
                    std::vector<std::vector<Plaintext>>> &source,
    const CkksPlainUploadSpec &spec,
    const std::vector<CkksHostDeviceLevel> &levels,
    const std::string &label, std::size_t &plaintext_count,
    std::size_t &device_bytes) {
  DevicePackedWeight result;
  result.first = upload_matrix(he, source.first, spec, levels, label + ".left",
                               plaintext_count, device_bytes);
  result.second =
      upload_matrix(he, source.second, spec, levels, label + ".right",
                    plaintext_count, device_bytes);
  return result;
}

std::vector<DevicePackedWeight> upload_packed_weights(
    HE_CKKS &he,
    const std::vector<std::pair<std::vector<std::vector<Plaintext>>,
                                std::vector<std::vector<Plaintext>>>> &source,
    const CkksPlainUploadSpec &spec,
    const std::vector<CkksHostDeviceLevel> &levels,
    const std::string &label, std::size_t &plaintext_count,
    std::size_t &device_bytes) {
  std::vector<DevicePackedWeight> result;
  result.reserve(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    result.emplace_back(upload_packed_weight(
        he, source[i], spec, levels,
        label + '[' + std::to_string(i) + ']', plaintext_count,
        device_bytes));
  }
  return result;
}

void require_device_he(const HE_CKKS &he) {
  if (!he.is_gpu_enable() || he.context == nullptr || he.dcontext == nullptr ||
      he.evaluator == nullptr) {
    fail("device cache requested for an HE_CKKS instance without a complete "
         "SEAL/Phantom backend");
  }
}

} // namespace

std::vector<CkksHostDeviceLevel>
validate_ckks_host_device_contexts(const SEALContext &host_context,
                                   const PhantomContext &device_context) {
  if (!host_context.parameters_set()) {
    fail(std::string("invalid SEAL context: ") +
         host_context.parameter_error_message());
  }
  const auto host_key = host_context.key_context_data();
  if (!host_key) {
    fail("SEAL key context is missing");
  }
  if (host_key->parms().scheme() != seal::scheme_type::ckks ||
      device_context.key_context_data().parms().scheme() !=
          phantom::scheme_type::ckks) {
    fail("only CKKS context pairs can use the plaintext cache");
  }
  if (device_context.total_parm_size() < 2 ||
      device_context.first_context_data().chain_index() != 1) {
    fail("Phantom context does not reserve chain 0 for keys and chain 1 for "
         "the first data level");
  }

  validate_parameter_level(host_key->parms(),
                           device_context.key_context_data().parms(),
                           "key level");
  validate_ntt_level(*host_key, device_context.key_context_data(),
                     "key level");

  std::vector<CkksHostDeviceLevel> levels;
  auto host_level = host_context.first_context_data();
  std::size_t phantom_chain =
      device_context.first_context_data().chain_index();
  while (host_level) {
    if (phantom_chain >= device_context.total_parm_size()) {
      fail("Phantom data chain is shorter than the SEAL data chain");
    }
    const auto &device_level =
        device_context.get_context_data(phantom_chain);
    if (device_level.chain_index() != phantom_chain) {
      fail("Phantom ContextData reports an inconsistent chain index");
    }

    validate_parameter_level(
        host_level->parms(), device_level.parms(),
        "data level " + std::to_string(levels.size()));
    validate_ntt_level(
        *host_level, device_level,
        "data level " + std::to_string(levels.size()));
    levels.push_back({host_level->parms_id(), host_level->chain_index(),
                      phantom_chain,
                      host_level->parms().coeff_modulus().size()});

    host_level = host_level->next_context_data();
    ++phantom_chain;
  }

  if (phantom_chain != device_context.total_parm_size()) {
    fail("Phantom data chain is longer than the SEAL data chain");
  }
  if (levels.empty()) {
    fail("CKKS context contains no data levels");
  }

  const std::size_t invariant_chain_sum =
      levels.front().seal_chain_index + levels.front().phantom_chain_index;
  for (const auto &level : levels) {
    if (level.seal_chain_index + level.phantom_chain_index !=
        invariant_chain_sum) {
      fail("SEAL/Phantom chain indices do not describe the same level order");
    }
  }
  return levels;
}

std::size_t phantom_chain_index_for(
    const SEALContext &host_context, const PhantomContext &device_context,
    const seal::parms_id_type &host_parms_id) {
  const auto levels =
      validate_ckks_host_device_contexts(host_context, device_context);
  return find_host_level(levels, host_parms_id).phantom_chain_index;
}

seal::parms_id_type seal_parms_id_for(
    const SEALContext &host_context, const PhantomContext &device_context,
    std::size_t phantom_chain_index) {
  const auto levels =
      validate_ckks_host_device_contexts(host_context, device_context);
  return find_device_level(levels, phantom_chain_index).seal_parms_id;
}

void align_ckks_host_plaintext_inplace(
    HE_CKKS &he, Plaintext &plain, const CkksPlainUploadSpec &spec,
    const std::string &label) {
  if (he.context == nullptr || he.evaluator == nullptr) {
    fail(label + " cannot be aligned without a SEAL context/evaluator");
  }
  if (plain.parms_id() == seal::parms_id_zero ||
      !he.context->get_context_data(plain.parms_id())) {
    fail(label + " does not belong to the HE_CKKS SEAL data context");
  }
  if (spec.scale_relative_tolerance < 0.0 ||
      !std::isfinite(spec.scale_relative_tolerance)) {
    fail(label + " has an invalid scale tolerance");
  }
  if (spec.target_parms_id) {
    if (!he.context->get_context_data(*spec.target_parms_id)) {
      fail(label + " requested a target parms_id outside the SEAL context");
    }
    if (plain.parms_id() != *spec.target_parms_id) {
      try {
        he.evaluator->mod_switch_to_inplace(plain, *spec.target_parms_id);
      } catch (const std::exception &error) {
        fail(label + " cannot be modulus-switched to its requested level: " +
             error.what());
      }
    }
  }
  if (spec.require_ntt_form && !plain.is_ntt_form()) {
    fail(label + " is not in NTT form after materialization");
  }
  if (!std::isfinite(plain.scale()) || plain.scale() <= 0.0) {
    fail(label + " has a non-finite or non-positive CKKS scale");
  }
  if (spec.expected_scale &&
      (!std::isfinite(*spec.expected_scale) || *spec.expected_scale <= 0.0 ||
       !scales_close(plain.scale(), *spec.expected_scale,
                     spec.scale_relative_tolerance))) {
    std::ostringstream os;
    os << label << " scale mismatch (encoded " << plain.scale()
       << ", expected " << *spec.expected_scale << ')';
    fail(os.str());
  }
}

void assert_ckks_device_plain_compatible(
    const UnifiedCiphertext &cipher, const PhantomPlaintext &plain,
    CkksCtPtOperation operation, const std::string &label,
    double scale_relative_tolerance) {
  if (cipher.location() != LOCATION::DEVICE) {
    fail(label + " compatibility check requires a device ciphertext");
  }
  if (scale_relative_tolerance < 0.0 ||
      !std::isfinite(scale_relative_tolerance)) {
    fail(label + " has an invalid scale tolerance");
  }
  if (!cipher.dcipher().is_ntt_form()) {
    fail(label + " CKKS ciphertext is not in NTT form");
  }
  if (cipher.dcipher().chain_index() != plain.chain_index()) {
    std::ostringstream os;
    os << label << " level mismatch (cipher chain "
       << cipher.dcipher().chain_index() << ", plaintext chain "
       << plain.chain_index() << ')';
    fail(os.str());
  }
  if (!std::isfinite(cipher.scale()) || cipher.scale() <= 0.0 ||
      !std::isfinite(plain.scale()) || plain.scale() <= 0.0) {
    fail(label + " has a non-finite or non-positive CKKS scale");
  }
  if (operation == CkksCtPtOperation::Add &&
      !scales_close(cipher.scale(), plain.scale(),
                    scale_relative_tolerance)) {
    std::ostringstream os;
    os << label << " add scale mismatch (cipher " << cipher.scale()
       << ", plaintext " << plain.scale() << ')';
    fail(os.str());
  }
}

CkksQkvUploadPlan make_qkv_upload_plan(
    const HE_CKKS &he, std::size_t input_device_chain_index,
    std::optional<double> expected_weight_scale,
    std::optional<double> expected_bias_scale) {
  require_device_he(he);
  if (input_device_chain_index + 1 >= he.dcontext->total_parm_size()) {
    fail("QKV input has no next data level for its post-multiply rescale");
  }

  const auto weight_parms = seal_parms_id_for(
      *he.context, *he.dcontext, input_device_chain_index);
  const auto bias_parms = seal_parms_id_for(
      *he.context, *he.dcontext, input_device_chain_index + 1);

  CkksQkvUploadPlan plan;
  plan.weight.target_parms_id = weight_parms;
  plan.weight.expected_scale = expected_weight_scale;
  plan.bias.target_parms_id = bias_parms;
  plan.bias.expected_scale = expected_bias_scale;
  return plan;
}

CkksLinearUploadPlan make_linear_upload_plan(
    const HE_CKKS &he, std::size_t input_device_chain_index,
    std::optional<double> expected_weight_scale,
    std::optional<double> expected_bias_scale) {
  require_device_he(he);
  const auto input_parms = seal_parms_id_for(
      *he.context, *he.dcontext, input_device_chain_index);

  CkksLinearUploadPlan plan;
  plan.weight.target_parms_id = input_parms;
  plan.weight.expected_scale = expected_weight_scale;
  plan.bias.target_parms_id = input_parms;
  plan.bias.expected_scale = expected_bias_scale;
  return plan;
}

void CkksDevicePlainCache::upload_qkv(
    HE_CKKS &he, const PreprocessParams_1 &host_params,
    const CkksQkvUploadPlan &plan) {
  require_device_he(he);
  clear();
  const auto levels =
      validate_ckks_host_device_contexts(*he.context, *he.dcontext);

  auto uploaded = std::make_unique<DevicePreprocessParams_1>();
  std::size_t count = 0;
  std::size_t bytes = 0;
  uploaded->wq_pack = upload_packed_weights(
      he, host_params.wq_pack, plan.weight, levels, "wq", count, bytes);
  uploaded->wk_pack = upload_packed_weights(
      he, host_params.wk_pack, plan.weight, levels, "wk", count, bytes);
  uploaded->wv_pack = upload_packed_weights(
      he, host_params.wv_pack, plan.weight, levels, "wv", count, bytes);
  uploaded->bq_pack = upload_matrix(he, host_params.bq_pack, plan.bias, levels,
                                    "bq", count, bytes);
  uploaded->bk_pack = upload_matrix(he, host_params.bk_pack, plan.bias, levels,
                                    "bk", count, bytes);
  uploaded->bv_pack = upload_matrix(he, host_params.bv_pack, plan.bias, levels,
                                    "bv", count, bytes);
  // PreprocessParams_1::cross_masks belongs to an older packed-QK helper. The
  // current bert_cipher_plain_bsgs QKV kernel never reads it (QK creates its
  // unit masks at the exact live level), so mirroring it would retain 256
  // highest-level plaintexts -- about 384 MiB for he1 -- for no computation.

  qkv_ = std::move(uploaded);
  linear_.reset();
  stage_ = Stage::Qkv;
  plaintext_count_ = count;
  device_bytes_ = bytes;
}

void CkksDevicePlainCache::upload_linear(
    HE_CKKS &he, const PreprocessParams_2 &host_params,
    const CkksLinearUploadPlan &plan) {
  require_device_he(he);
  clear();
  const auto levels =
      validate_ckks_host_device_contexts(*he.context, *he.dcontext);

  auto uploaded = std::make_unique<DevicePreprocessParams_2>();
  std::size_t count = 0;
  std::size_t bytes = 0;
  uploaded->cross_mat_single = upload_packed_weight(
      he, host_params.cross_mat_single, plan.weight, levels, "weight", count,
      bytes);
  uploaded->cross_bias_single = upload_vector(
      he, host_params.cross_bias_single, plan.bias, levels, "bias", count,
      bytes);

  linear_ = std::move(uploaded);
  qkv_.reset();
  stage_ = Stage::Linear;
  plaintext_count_ = count;
  device_bytes_ = bytes;
}

const DevicePreprocessParams_1 &CkksDevicePlainCache::qkv() const {
  if (stage_ != Stage::Qkv || !qkv_) {
    fail("QKV plaintexts requested while the QKV stage is not resident");
  }
  return *qkv_;
}

const DevicePreprocessParams_2 &CkksDevicePlainCache::linear() const {
  if (stage_ != Stage::Linear || !linear_) {
    fail("linear plaintexts requested while the linear stage is not resident");
  }
  return *linear_;
}

void CkksDevicePlainCache::clear() {
  if (stage_ != Stage::Empty) {
    PHANTOM_CHECK_CUDA(cudaStreamSynchronize(cudaStreamPerThread));
  }
  qkv_.reset();
  linear_.reset();
  stage_ = Stage::Empty;
  plaintext_count_ = 0;
  device_bytes_ = 0;
}
