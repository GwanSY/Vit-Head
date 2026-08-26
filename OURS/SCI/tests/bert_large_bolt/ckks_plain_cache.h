#ifndef BERT_LARGE_BOLT_CKKS_PLAIN_CACHE_H_
#define BERT_LARGE_BOLT_CKKS_PLAIN_CACHE_H_

#include "LinearHE/unified-he.h"

#include <seal/context.h>
#include <seal/plaintext.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class HE_CKKS;
struct PreprocessParams_1;
struct PreprocessParams_2;

// A SEAL data level and the Phantom data level that contains exactly the same
// coefficient moduli. Phantom reserves chain index 0 for key parameters, while
// its first data level starts at chain index 1.
struct CkksHostDeviceLevel {
  seal::parms_id_type seal_parms_id{};
  std::size_t seal_chain_index = 0;
  std::size_t phantom_chain_index = 0;
  std::size_t coeff_modulus_size = 0;
};

// Validates the raw-RNS-copy precondition used by UnifiedPlaintext::to_device.
// This fails closed unless every modulus prime (including the key level), its
// order, and the polynomial degree are identical in SEAL and Phantom.
std::vector<CkksHostDeviceLevel>
validate_ckks_host_device_contexts(const seal::SEALContext &host_context,
                                   const PhantomContext &device_context);

std::size_t phantom_chain_index_for(
    const seal::SEALContext &host_context,
    const PhantomContext &device_context,
    const seal::parms_id_type &host_parms_id);

seal::parms_id_type seal_parms_id_for(
    const seal::SEALContext &host_context,
    const PhantomContext &device_context,
    std::size_t phantom_chain_index);

// A target level is reached with SEAL's normal plaintext modulus switching;
// scale is never changed by metadata assignment. An expected scale therefore
// acts as an assertion, not as a request to relabel CKKS coefficients.
struct CkksPlainUploadSpec {
  std::optional<seal::parms_id_type> target_parms_id;
  std::optional<double> expected_scale;
  // SEAL and Phantom use machine-epsilon scale equality for ct+pt.  Formal
  // model plaintexts are encoded from the exact live scale expression, so the
  // cache must not accept a looser value that the evaluator will later reject.
  double scale_relative_tolerance = 0.0;
  bool require_ntt_form = true;
};

struct CkksQkvUploadPlan {
  CkksPlainUploadSpec weight;
  CkksPlainUploadSpec bias;
};

struct CkksLinearUploadPlan {
  CkksPlainUploadSpec weight;
  CkksPlainUploadSpec bias;
};

// The QKV kernel rescales the ct-pt products before adding bias. This helper
// selects the input level for weights and the next data level for biases. Scale
// assertions are caller supplied because fused formal graphs may use a profile
// whose encoded plaintext scales differ from HE_CKKS::he_scale.
CkksQkvUploadPlan make_qkv_upload_plan(
    const HE_CKKS &he, std::size_t input_device_chain_index,
    std::optional<double> expected_weight_scale = std::nullopt,
    std::optional<double> expected_bias_scale = std::nullopt);

// BSGS2 adds its bias before the first rescale. Weight and bias therefore use
// the input ciphertext level. Their scale assertions are explicitly supplied.
CkksLinearUploadPlan make_linear_upload_plan(
    const HE_CKKS &he, std::size_t input_device_chain_index,
    std::optional<double> expected_weight_scale = std::nullopt,
    std::optional<double> expected_bias_scale = std::nullopt);

// Applies only a real SEAL plaintext modulus switch when a lower target level
// is requested, then checks the encoded scale. This is useful for keeping the
// CPU pp structures and their device mirrors on the exact same materialized
// level. It never assigns Plaintext::scale().
void align_ckks_host_plaintext_inplace(
    HE_CKKS &he, seal::Plaintext &plain, const CkksPlainUploadSpec &spec,
    const std::string &label = "plaintext");

enum class CkksCtPtOperation { Multiply, Add };

// Phantom's current ct-pt kernels do not consistently reject a plaintext from
// the wrong chain. Keep this assertion immediately before every GPU ct-pt
// multiply/add. Add additionally requires matching CKKS scales.
void assert_ckks_device_plain_compatible(
    const UnifiedCiphertext &cipher, const PhantomPlaintext &plain,
    CkksCtPtOperation operation, const std::string &label,
    double scale_relative_tolerance = 0.0);

using DevicePlainMatrix = std::vector<std::vector<PhantomPlaintext>>;
using DevicePackedWeight = std::pair<DevicePlainMatrix, DevicePlainMatrix>;

struct DevicePreprocessParams_1 {
  std::vector<DevicePackedWeight> wq_pack;
  std::vector<DevicePackedWeight> wk_pack;
  std::vector<DevicePackedWeight> wv_pack;

  DevicePlainMatrix bq_pack;
  DevicePlainMatrix bk_pack;
  DevicePlainMatrix bv_pack;
};

struct DevicePreprocessParams_2 {
  DevicePackedWeight cross_mat_single;
  std::vector<PhantomPlaintext> cross_bias_single;
};

// Owns only one model stage at a time. This intentionally keeps the original
// SEAL PreprocessParams structures alive and unchanged for the CPU path, while
// avoiding simultaneous residency of pp_1, pp_2, pp_3 and pp_4 on the GPU.
class CkksDevicePlainCache {
public:
  enum class Stage { Empty, Qkv, Linear };

  CkksDevicePlainCache() = default;
  ~CkksDevicePlainCache() = default;

  CkksDevicePlainCache(const CkksDevicePlainCache &) = delete;
  CkksDevicePlainCache &operator=(const CkksDevicePlainCache &) = delete;
  CkksDevicePlainCache(CkksDevicePlainCache &&) = default;
  CkksDevicePlainCache &operator=(CkksDevicePlainCache &&) = default;

  void upload_qkv(HE_CKKS &he, const PreprocessParams_1 &host_params,
                  const CkksQkvUploadPlan &plan);

  void upload_linear(HE_CKKS &he, const PreprocessParams_2 &host_params,
                     const CkksLinearUploadPlan &plan);

  const DevicePreprocessParams_1 &qkv() const;
  const DevicePreprocessParams_2 &linear() const;

  // Synchronizes the upload/compute stream before releasing device storage.
  // Call this at a model-stage boundary after the last kernel that consumes the
  // cache. Destruction is stream ordered, but explicit clear gives predictable
  // peak memory and stage timing.
  void clear();

  Stage stage() const noexcept { return stage_; }
  std::size_t plaintext_count() const noexcept { return plaintext_count_; }
  std::size_t device_bytes() const noexcept { return device_bytes_; }

private:
  Stage stage_ = Stage::Empty;
  std::unique_ptr<DevicePreprocessParams_1> qkv_;
  std::unique_ptr<DevicePreprocessParams_2> linear_;
  std::size_t plaintext_count_ = 0;
  std::size_t device_bytes_ = 0;
};

#endif // BERT_LARGE_BOLT_CKKS_PLAIN_CACHE_H_
