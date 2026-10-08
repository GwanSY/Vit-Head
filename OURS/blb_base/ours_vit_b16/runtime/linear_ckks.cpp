#include "linear_ckks.h"
#include "packed_block_rotation.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

bool use_direct_device_plaintext_packing(const HE_CKKS &he) {
    if (!he.is_gpu_enable()) return false;
    const char *value = std::getenv("BLB_VIT_GPU_DIRECT_PLAINTEXT");
    if (value == nullptr || std::string(value) == "1") return true;
    if (std::string(value) == "0") return false;
    throw std::runtime_error(
        "BLB_VIT_GPU_DIRECT_PLAINTEXT must be 0 or 1");
}

int direct_device_plaintext_streams() {
    const char *value = std::getenv("BLB_VIT_GPU_LINEAR_STREAMS");
    if (value == nullptr) return 8;
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 1 || parsed > 16) {
        throw std::runtime_error(
            "BLB_VIT_GPU_LINEAR_STREAMS must be an integer in [1, 16]");
    }
    return static_cast<int>(parsed);
}

// Phantom accepts only device plaintexts.  They are created for one evaluator
// call and deliberately do not survive the call or retain a cross-layer
// cache.  GPU runs use the official Phantom encoder on the exact packed slot
// values; CPU runs retain the original SEAL plaintext representation.
enum class CkksCtPtOperation { Multiply, Add };

using DevicePlainMatrix = std::vector<std::vector<PhantomPlaintext>>;
using DevicePackedWeight =
    std::pair<DevicePlainMatrix, DevicePlainMatrix>;

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

#ifdef BLB_HAS_HE_GPU
std::size_t phantom_chain_for_seal_chain(HE_CKKS &he,
                                         std::size_t seal_chain) {
    const auto first = he.context->first_context_data();
    if (!first)
        throw std::runtime_error("Phantom packing: empty SEAL context");
    const auto full_modulus_count = first->parms().coeff_modulus().size();
    for (auto data = first; data; data = data->next_context_data()) {
        if (data->chain_index() == seal_chain) {
            return full_modulus_count -
                   data->parms().coeff_modulus().size() + 1;
        }
    }
    throw std::runtime_error("Phantom packing: SEAL chain is absent");
}

PhantomPlaintext copy_plaintext_to_device(HE_CKKS &he,
                                           const Plaintext &source) {
    PhantomPlaintext result;
    UnifiedPlaintext::to_device(*he.context, source, *he.dcontext, result);
    return result;
}

PhantomPlaintext encode_semantic_plaintext_to_device(
    HE_CKKS &he, PhantomCKKSEncoder &encoder,
    const SemanticPlaintext &source, double scale, std::size_t seal_chain) {
    if (source.size() != he.num_slots || !std::isfinite(scale) || scale <= 0.0)
        throw std::runtime_error(
            "Phantom packing: invalid semantic plaintext metadata");
    PhantomPlaintext result;
    encoder.encode_trusted(*he.dcontext, source, scale, result,
                           phantom_chain_for_seal_chain(he, seal_chain));
    return result;
}


PhantomPlaintext encode_semantic_plaintext_to_device(
    HE_CKKS &he, const SemanticPlaintext &source, double scale,
    std::size_t seal_chain) {
    return encode_semantic_plaintext_to_device(
        he, *he.dencoder, source, scale, seal_chain);
}

DevicePlainMatrix encode_semantic_matrix_to_device(
    HE_CKKS &he, const SemanticPlainMatrix &source, double scale,
    std::size_t seal_chain) {
    DevicePlainMatrix result(source.size());
    for (size_t row = 0; row < source.size(); ++row) {
        result[row].reserve(source[row].size());
        for (const auto &plain : source[row]) {
            result[row].push_back(encode_semantic_plaintext_to_device(
                he, plain, scale, seal_chain));
        }
    }
    return result;
}

DevicePackedWeight encode_semantic_packed_weight_to_device(
    HE_CKKS &he, const SemanticPackedWeight &source, double scale,
    std::size_t seal_chain) {
    return {encode_semantic_matrix_to_device(he, source.first, scale,
                                             seal_chain),
            encode_semantic_matrix_to_device(he, source.second, scale,
                                             seal_chain)};
}

std::vector<DevicePackedWeight> encode_semantic_packed_weights_to_device(
    HE_CKKS &he, const std::vector<SemanticPackedWeight> &source,
    double scale, std::size_t seal_chain) {
    std::vector<DevicePackedWeight> result;
    result.reserve(source.size());
    for (const auto &packed : source) {
        result.push_back(encode_semantic_packed_weight_to_device(
            he, packed, scale, seal_chain));
    }
    return result;
}

DevicePlainMatrix copy_plain_matrix_to_device(
    HE_CKKS &he, const std::vector<std::vector<Plaintext>> &source) {
    DevicePlainMatrix result(source.size());
    for (size_t row = 0; row < source.size(); ++row) {
        result[row].reserve(source[row].size());
        for (const auto &plain : source[row])
            result[row].push_back(copy_plaintext_to_device(he, plain));
    }
    return result;
}

DevicePackedWeight copy_packed_weight_to_device(
    HE_CKKS &he,
    const std::pair<std::vector<std::vector<Plaintext>>,
                    std::vector<std::vector<Plaintext>>> &source) {
    return {copy_plain_matrix_to_device(he, source.first),
            copy_plain_matrix_to_device(he, source.second)};
}

std::vector<DevicePackedWeight> copy_packed_weights_to_device(
    HE_CKKS &he,
    const std::vector<std::pair<std::vector<std::vector<Plaintext>>,
                                std::vector<std::vector<Plaintext>>>> &source) {
    std::vector<DevicePackedWeight> result;
    result.reserve(source.size());
    for (const auto &packed : source)
        result.push_back(copy_packed_weight_to_device(he, packed));
    return result;
}

DevicePreprocessParams_1 copy_qkv_params_to_device(
    HE_CKKS &he, const PreprocessParams_1 &source) {
    DevicePreprocessParams_1 result;
    if (source.device_semantic_packed) {
        result.wq_pack = encode_semantic_packed_weights_to_device(
            he, source.device_wq_pack, source.weight_scale,
            source.weight_chain);
        result.wk_pack = encode_semantic_packed_weights_to_device(
            he, source.device_wk_pack, source.weight_scale,
            source.weight_chain);
        result.wv_pack = encode_semantic_packed_weights_to_device(
            he, source.device_wv_pack, source.weight_scale,
            source.weight_chain);
        result.bq_pack = encode_semantic_matrix_to_device(
            he, source.device_bq_pack, source.bias_scale, source.bias_chain);
        result.bk_pack = encode_semantic_matrix_to_device(
            he, source.device_bk_pack, source.bias_scale, source.bias_chain);
        result.bv_pack = encode_semantic_matrix_to_device(
            he, source.device_bv_pack, source.bias_scale, source.bias_chain);
    } else {
        result.wq_pack = copy_packed_weights_to_device(he, source.wq_pack);
        result.wk_pack = copy_packed_weights_to_device(he, source.wk_pack);
        result.wv_pack = copy_packed_weights_to_device(he, source.wv_pack);
        result.bq_pack = copy_plain_matrix_to_device(he, source.bq_pack);
        result.bk_pack = copy_plain_matrix_to_device(he, source.bk_pack);
        result.bv_pack = copy_plain_matrix_to_device(he, source.bv_pack);
    }
    return result;
}

DevicePreprocessParams_2 copy_linear_params_to_device(
    HE_CKKS &he, const PreprocessParams_2 &source) {
    DevicePreprocessParams_2 result;
    if (source.device_semantic_packed) {
        result.cross_mat_single = encode_semantic_packed_weight_to_device(
            he, source.device_cross_mat_single, source.weight_scale,
            source.target_chain);
        result.cross_bias_single.reserve(
            source.device_cross_bias_single.size());
        for (const auto &plain : source.device_cross_bias_single) {
            result.cross_bias_single.push_back(
                encode_semantic_plaintext_to_device(
                    he, plain, source.bias_scale, source.target_chain));
        }
    } else {
        result.cross_mat_single = copy_packed_weight_to_device(
            he, source.cross_mat_single);
        result.cross_bias_single.reserve(source.cross_bias_single.size());
        for (const auto &plain : source.cross_bias_single)
            result.cross_bias_single.push_back(
                copy_plaintext_to_device(he, plain));
    }
    return result;
}

template <typename DeviceParams>
double release_device_plaintexts(DeviceParams &params, const char *label) {
    const auto start = std::chrono::steady_clock::now();
    // Phantom keeps cudaMallocAsync pages in its default pool. Destroy only
    // the current operator's copied plaintexts and return their free pages to
    // the device before the next original BLB fused MatMul is materialized.
    // This changes neither packing nor evaluator/protocol work.
    params = DeviceParams{};
    auto error = cudaStreamSynchronize(cudaStreamPerThread);
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(label) +
                                 ": Phantom plaintext release sync failed: " +
                                 cudaGetErrorString(error));
    }
    int device = 0;
    cudaMemPool_t pool{};
    error = cudaGetDevice(&device);
    if (error == cudaSuccess)
        error = cudaDeviceGetDefaultMemPool(&pool, device);
    if (error == cudaSuccess)
        error = cudaMemPoolTrimTo(pool, 0);
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(label) +
                                 ": Phantom plaintext pool trim failed: " +
                                 cudaGetErrorString(error));
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "CKKS_DEVICE_PLAINTEXT_RELEASE op=" << label
              << " elapsed_ms=" << seconds * 1000.0 << std::endl;
    return seconds;
}
#endif

bool ckks_scales_close(double lhs, double rhs,
                       double relative_tolerance = 0.0) {
    const double magnitude = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= relative_tolerance * magnitude;
}

void require_device_plain_compatible(
    const UnifiedCiphertext &cipher, const PhantomPlaintext &plain,
    CkksCtPtOperation operation, const std::string &label) {
    if (cipher.location() != LOCATION::DEVICE)
        throw std::runtime_error(label + ": ciphertext is not on DEVICE");
#ifndef BLB_HAS_HE_GPU
    (void)plain;
    (void)operation;
    throw std::runtime_error(
        label + ": DEVICE backend was not compiled with USE_HE_GPU=ON");
#else
    if (cipher.dcipher().chain_index() != plain.chain_index()) {
        throw std::runtime_error(
            label + ": CKKS level mismatch cipher_chain=" +
            std::to_string(cipher.dcipher().chain_index()) +
            " plain_chain=" + std::to_string(plain.chain_index()) +
            " cipher_limbs=" +
            std::to_string(cipher.dcipher().coeff_modulus_size()) +
            " plain_coeff_count=" + std::to_string(plain.coeff_count()));
    }
    if (!std::isfinite(cipher.scale()) || !std::isfinite(plain.scale()) ||
        cipher.scale() <= 0.0 || plain.scale() <= 0.0)
        throw std::runtime_error(label + ": invalid CKKS scale");
    if (operation == CkksCtPtOperation::Add &&
        std::abs(cipher.scale() - plain.scale()) >
            std::max({1.0, std::abs(cipher.scale()), std::abs(plain.scale())}) *
                1e-12)
        throw std::runtime_error(label + ": CKKS add scale mismatch");
#endif
}

std::shared_ptr<const seal::SEALContext::ContextData>
host_context_data_for(const HE_CKKS &he, const UnifiedCiphertext &cipher,
                      const char *label) {
    if (cipher.location() == LOCATION::HOST) {
        auto data = he.context->get_context_data(cipher.hcipher().parms_id());
        if (!data) {
            throw std::runtime_error(std::string(label) +
                                     ": ciphertext parms_id is not in the SEAL context");
        }
        return data;
    }
#ifdef BLB_HAS_HE_GPU
    if (cipher.location() == LOCATION::DEVICE) {
        const auto first = he.context->first_context_data();
        if (!first) {
            throw std::runtime_error(std::string(label) +
                                     ": SEAL context has no first data level");
        }
        const size_t full_modulus_count =
            first->parms().coeff_modulus().size();
        const size_t chain_index = cipher.dcipher().chain_index();
        if (chain_index == 0 || chain_index > full_modulus_count + 1) {
            throw std::runtime_error(std::string(label) +
                                     ": invalid Phantom chain index");
        }
        const size_t expected_modulus_count =
            full_modulus_count - chain_index + 1;
        for (auto data = first; data; data = data->next_context_data()) {
            if (data->parms().coeff_modulus().size() ==
                expected_modulus_count)
                return data;
        }
        throw std::runtime_error(std::string(label) +
                                 ": device chain has no SEAL data-level mirror");
    }
#endif
    throw std::runtime_error(std::string(label) +
                             ": ciphertext is not in a supported backend");
}

void require_uniform_cipher_metadata(
    const std::vector<UnifiedCiphertext> &ciphertexts, const char *label) {
    if (ciphertexts.empty()) {
        throw std::runtime_error(std::string(label) + ": empty ciphertext input");
    }
    const auto location = ciphertexts.front().location();
    const auto modulus_size = ciphertexts.front().coeff_modulus_size();
    const auto scale = ciphertexts.front().scale();
    if (!std::isfinite(scale) || scale <= 0.0) {
        throw std::runtime_error(std::string(label) +
                                 ": invalid ciphertext scale");
    }
    for (std::size_t i = 1; i < ciphertexts.size(); ++i) {
        if (ciphertexts[i].location() != location ||
            ciphertexts[i].coeff_modulus_size() != modulus_size ||
            !ckks_scales_close(ciphertexts[i].scale(), scale)) {
            std::ostringstream os;
            os << label << ": ciphertext metadata differs at index " << i;
            throw std::runtime_error(os.str());
        }
    }
}

void require_fractional_bits(int fractional_bits, const std::string &label) {
    // double can represent every checkpoint integer used by BLB exactly at
    // these small bit widths.  Keeping this bounded also avoids accepting an
    // accidental metadata value as a model denominator.
    if (fractional_bits < 0 || fractional_bits > 52) {
        throw std::runtime_error(label + ": invalid fixed-point fractional bits " +
                                 std::to_string(fractional_bits));
    }
}

double checkpoint_semantic_multiplier(int fractional_bits,
                                      const std::string &label) {
    require_fractional_bits(fractional_bits, label);
    return std::ldexp(1.0, -fractional_bits);
}

double checkpoint_fixed_to_real(uint64_t raw, double semantic_multiplier) {
    // The ViT adapter stores checkpoint values in the protocol's 37-bit
    // two's-complement ring, not in a sign-extended 64-bit container.  A
    // negative value such as -1 is therefore 2^37-1 here; interpreting the
    // uint64_t word as int64_t turns every negative weight into a roughly
    // 2^37-sized positive value and makes the BSGS result explode.  Recover
    // the centered 37-bit representative before applying the public
    // fractional-bit scale.
    constexpr int kCheckpointRingBits = 37;
    constexpr uint64_t kCheckpointRing =
        uint64_t{1} << kCheckpointRingBits;
    constexpr uint64_t kCheckpointMask = kCheckpointRing - 1;
    constexpr uint64_t kCheckpointSign =
        uint64_t{1} << (kCheckpointRingBits - 1);
    const uint64_t centered_word = raw & kCheckpointMask;
    const int64_t signed_value =
        (centered_word & kCheckpointSign)
            ? static_cast<int64_t>(centered_word - kCheckpointRing)
            : static_cast<int64_t>(centered_word);
    return static_cast<double>(signed_value) * semantic_multiplier;
}

void reencode_semantic_plain(HE_CKKS &he, seal::Plaintext &plain,
                             const seal::parms_id_type &target,
                             double target_scale, int fractional_bits,
                             bool source_is_checkpoint_integer,
                             const std::string &label) {
    require_fractional_bits(fractional_bits, label);
    if (!std::isfinite(target_scale) || target_scale <= 0.0) {
        throw std::runtime_error(label + ": invalid target CKKS scale");
    }
    if (plain.parms_id() == seal::parms_id_zero ||
        !he.context->get_context_data(plain.parms_id())) {
        throw std::runtime_error(label + ": plaintext is not in the CKKS data chain");
    }
    if (!plain.is_ntt_form()) {
        throw std::runtime_error(label + ": CKKS plaintext is not in NTT form");
    }
    if (!he.context->get_context_data(target)) {
        throw std::runtime_error(label + ": target level is not in the CKKS data chain");
    }
    if (!source_is_checkpoint_integer &&
        ckks_scales_close(plain.scale(), target_scale)) {
        if (plain.parms_id() == target) {
            return;
        }
        const auto source_context =
            he.context->get_context_data(plain.parms_id());
        const auto target_context = he.context->get_context_data(target);
        if (target_context->chain_index() <= source_context->chain_index()) {
            // A true CKKS plaintext modulus switch preserves both the encoded
            // semantic slots and Plaintext::scale(). It avoids a needless
            // approximate decode/encode round trip when only the level differs.
            try {
                he.evaluator->mod_switch_to_inplace(plain, target);
                return;
            } catch (const std::exception &error) {
                // Some SEAL builds reject an otherwise valid same-scale
                // plaintext switch with a generic "scale mismatch" when the
                // source and target levels have different prime sizes.  The
                // model value is already materialized (source_is_checkpoint_integer
                // is false), so decode/re-encode it at the requested level
                // rather than relabeling the scale or weakening GPU checks.
                std::cerr << "CKKS_PLAIN_LEVEL_REENCODE label=" << label
                          << " reason=" << error.what() << std::endl;
            }
        }
    }

    std::vector<double> slots;
    he.encoder->decode(plain, slots);
    const double denominator = std::ldexp(1.0, fractional_bits);
    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (!std::isfinite(slots[i])) {
            throw std::runtime_error(label + ": decoded a non-finite model value");
        }
        if (source_is_checkpoint_integer) {
            const double rounded = std::nearbyint(slots[i]);
            const double tolerance =
                std::max(1e-5, std::abs(rounded) * 1e-10);
            if (std::abs(slots[i] - rounded) > tolerance) {
                std::ostringstream os;
                os << label
                   << ": decoded checkpoint value is not an integer at slot "
                   << i << " (value=" << slots[i] << ')';
                throw std::runtime_error(os.str());
            }
            // This is the actual model-value conversion.  Changing only
            // Plaintext::scale() would leave the polynomial coefficients (and
            // hence the model) wrong.
            slots[i] = rounded / denominator;
        }
    }

    seal::Plaintext encoded;
    he.encoder->encode(slots, target, target_scale, encoded);
    plain = std::move(encoded);
}

template <typename PackedWeights>
void reencode_packed_weights(HE_CKKS &he, PackedWeights &weights,
                             const seal::parms_id_type &target,
                             double target_scale, int fractional_bits,
                             bool source_is_checkpoint_integer,
                             const std::string &label) {
    for (std::size_t packing = 0; packing < weights.size(); ++packing) {
        auto switch_side = [&](auto &side, const char *side_name) {
            for (std::size_t diag = 0; diag < side.size(); ++diag) {
                for (std::size_t ct = 0; ct < side[diag].size(); ++ct) {
                    reencode_semantic_plain(
                        he, side[diag][ct], target, target_scale,
                        fractional_bits, source_is_checkpoint_integer,
                        label + '[' + std::to_string(packing) + "]." +
                            side_name + '[' + std::to_string(diag) + "][" +
                            std::to_string(ct) + ']');
                }
            }
        };
        switch_side(weights[packing].first, "left");
        switch_side(weights[packing].second, "right");
    }
}

void reencode_linear_weights(
    HE_CKKS &he,
    std::pair<std::vector<std::vector<seal::Plaintext>>,
              std::vector<std::vector<seal::Plaintext>>> &weights,
    const seal::parms_id_type &target, double target_scale,
    int fractional_bits, bool source_is_checkpoint_integer,
    const std::string &label) {
    auto switch_side = [&](auto &side, const char *side_name) {
        for (std::size_t diag = 0; diag < side.size(); ++diag) {
            for (std::size_t ct = 0; ct < side[diag].size(); ++ct) {
                reencode_semantic_plain(
                    he, side[diag][ct], target, target_scale, fractional_bits,
                    source_is_checkpoint_integer,
                    label + '.' + side_name + '[' + std::to_string(diag) +
                        "][" + std::to_string(ct) + ']');
            }
        }
    };
    switch_side(weights.first, "left");
    switch_side(weights.second, "right");
}

void reencode_plain_matrix(HE_CKKS &he,
                           std::vector<std::vector<seal::Plaintext>> &values,
                           const seal::parms_id_type &target,
                           double target_scale, int fractional_bits,
                           bool source_is_checkpoint_integer,
                           const std::string &label) {
    for (std::size_t packing = 0; packing < values.size(); ++packing) {
        for (std::size_t i = 0; i < values[packing].size(); ++i) {
            reencode_semantic_plain(
                he, values[packing][i], target, target_scale,
                fractional_bits, source_is_checkpoint_integer,
                label + '[' + std::to_string(packing) + "][" +
                    std::to_string(i) + ']');
        }
    }
}

void reencode_plain_vector(HE_CKKS &he,
                           std::vector<seal::Plaintext> &values,
                           const seal::parms_id_type &target,
                           double target_scale, int fractional_bits,
                           bool source_is_checkpoint_integer,
                           const std::string &label) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        reencode_semantic_plain(
            he, values[i], target, target_scale, fractional_bits,
            source_is_checkpoint_integer,
            label + '[' + std::to_string(i) + ']');
    }
}

struct MaterializedQkvMetadata {
    seal::parms_id_type weight_level{};
    seal::parms_id_type bias_level{};
    double weight_scale = 0.0;
    double bias_scale = 0.0;
};

MaterializedQkvMetadata materialize_qkv_plaintexts(
    HE_CKKS &he, const std::vector<UnifiedCiphertext> &inputs,
    PreprocessParams_1 &params) {
    const auto materialize_start = std::chrono::steady_clock::now();
    require_uniform_cipher_metadata(inputs, "QKV");
    auto input_context = host_context_data_for(he, inputs.front(), "QKV");
    auto output_context = input_context->next_context_data();
    if (!output_context) {
        throw std::runtime_error("QKV: input level has no rescale target");
    }
    const auto &dropped_moduli = input_context->parms().coeff_modulus();
    if (dropped_moduli.empty()) {
        throw std::runtime_error("QKV: input level has no coefficient modulus");
    }
    if (params.device_semantic_packed) {
        const double expected_bias_scale =
            inputs.front().scale() * params.weight_scale /
            static_cast<double>(dropped_moduli.back().value());
        if (input_context->chain_index() != params.weight_chain ||
            output_context->chain_index() != params.bias_chain ||
            !ckks_scales_close(inputs.front().scale(), params.input_scale,
                               1e-12) ||
            !ckks_scales_close(expected_bias_scale, params.bias_scale,
                               1e-12)) {
            throw std::runtime_error(
                "QKV: Phantom semantic packing metadata does not match input");
        }
        std::cout << "CKKS_PLAIN_MATERIALIZE op=QKV input_chain="
                  << input_context->chain_index() << " output_chain="
                  << output_context->chain_index() << " input_scale_bits="
                  << std::log2(inputs.front().scale())
                  << " weight_scale_bits=" << std::log2(params.weight_scale)
                  << " bias_scale_bits=" << std::log2(params.bias_scale)
                  << " weight_fractional_bits="
                  << params.weight_fractional_bits
                  << " bias_fractional_bits=" << params.bias_fractional_bits
                  << " semantic_conversion=phantom_direct elapsed_ms=0"
                  << std::endl;
        return {input_context->parms_id(), output_context->parms_id(),
                params.weight_scale, params.bias_scale};
    }
    if (params.wq_pack.empty() || params.wq_pack.front().first.empty() ||
        params.wq_pack.front().first.front().empty()) {
        throw std::runtime_error("QKV: weight preprocessing is empty");
    }
    const double weight_scale =
        params.wq_pack.front().first.front().front().scale();
    const double bias_scale = inputs.front().scale() * weight_scale /
                              static_cast<double>(dropped_moduli.back().value());
    require_fractional_bits(params.weight_fractional_bits, "QKV.weight");
    require_fractional_bits(params.bias_fractional_bits, "QKV.bias");
    const bool source_is_checkpoint_integer =
        !params.semantic_values_materialized;

    const bool materialized_at_target =
        params.wq_pack.front().first.front().front().parms_id() ==
            input_context->parms_id() &&
        params.bq_pack.front().front().parms_id() ==
            output_context->parms_id() &&
        ckks_scales_close(params.wq_pack.front().first.front().front().scale(),
                          weight_scale) &&
        ckks_scales_close(params.bq_pack.front().front().scale(),
                          bias_scale);
    if (!materialized_at_target) {
        reencode_packed_weights(
            he, params.wq_pack, input_context->parms_id(), weight_scale,
            params.weight_fractional_bits, source_is_checkpoint_integer, "wq");
        reencode_packed_weights(
            he, params.wk_pack, input_context->parms_id(), weight_scale,
            params.weight_fractional_bits, source_is_checkpoint_integer, "wk");
        reencode_packed_weights(
            he, params.wv_pack, input_context->parms_id(), weight_scale,
            params.weight_fractional_bits, source_is_checkpoint_integer, "wv");
        reencode_plain_matrix(
            he, params.bq_pack, output_context->parms_id(), bias_scale,
            params.bias_fractional_bits, source_is_checkpoint_integer, "bq");
        reencode_plain_matrix(
            he, params.bk_pack, output_context->parms_id(), bias_scale,
            params.bias_fractional_bits, source_is_checkpoint_integer, "bk");
        reencode_plain_matrix(
            he, params.bv_pack, output_context->parms_id(), bias_scale,
            params.bias_fractional_bits, source_is_checkpoint_integer, "bv");
    }
    params.semantic_values_materialized = true;
    const auto materialize_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                  materialize_start).count();

    std::cout << "CKKS_PLAIN_MATERIALIZE op=QKV input_chain="
              << input_context->chain_index() << " output_chain="
              << output_context->chain_index() << " input_scale_bits="
              << std::log2(inputs.front().scale()) << " weight_scale_bits="
              << std::log2(weight_scale) << " bias_scale_bits="
              << std::log2(bias_scale) << " weight_fractional_bits="
              << params.weight_fractional_bits << " bias_fractional_bits="
              << params.bias_fractional_bits << " semantic_conversion="
              << (materialized_at_target ? "offline_cache" :
                  (source_is_checkpoint_integer ? "applied" : "reused"))
              << " elapsed_ms=" << materialize_ms
              << std::endl;
    return {input_context->parms_id(), output_context->parms_id(),
            weight_scale, bias_scale};
}

struct MaterializedLinearMetadata {
    seal::parms_id_type level{};
    double weight_scale = 0.0;
    double bias_scale = 0.0;
};

MaterializedLinearMetadata materialize_linear_plaintexts(
    HE_CKKS &he, const std::vector<UnifiedCiphertext> &inputs,
    PreprocessParams_2 &params) {
    const auto materialize_start = std::chrono::steady_clock::now();
    require_uniform_cipher_metadata(inputs, "BSGS2");
    auto input_context = host_context_data_for(he, inputs.front(), "BSGS2");
    if (params.device_semantic_packed) {
        const std::size_t cached_target_chain = params.target_chain;
        const double cached_input_scale = params.input_scale;
        const double cached_bias_scale = params.bias_scale;
        const double expected_bias_scale =
            inputs.front().scale() * params.weight_scale;
        if (input_context->chain_index() != params.target_chain ||
            !ckks_scales_close(inputs.front().scale(), params.input_scale,
                               1e-12) ||
            !ckks_scales_close(expected_bias_scale, params.bias_scale,
                               1e-12)) {
            // The semantic device vectors do not carry a SEAL context or a
            // CKKS level.  The original fast path cached the level/scale from
            // the baseline diagonal-reuse schedule, while the validation-only
            // full-diagonal schedule can legally consume a different public
            // modulus level before entering BSGS2.  Adopt the actual input
            // metadata before encoding the semantic vectors.  This is a
            // metadata alignment at the existing ct-pt boundary; it does not
            // add a conversion, decrypt/re-encrypt, or evaluator operation.
            params.target_chain = input_context->chain_index();
            params.input_scale = inputs.front().scale();
            params.bias_scale = expected_bias_scale;
            std::cout << "CKKS_DIRECT_METADATA_ALIGN op=BSGS2 status=ADJUST"
                      << " cached_chain=" << cached_target_chain
                      << " actual_chain=" << params.target_chain
                      << " cached_input_scale_bits="
                      << std::log2(cached_input_scale)
                      << " actual_input_scale_bits="
                      << std::log2(params.input_scale)
                      << " cached_bias_scale_bits="
                      << std::log2(cached_bias_scale)
                      << " actual_bias_scale_bits="
                      << std::log2(params.bias_scale)
                      << " context=existing_he_context"
                      << " semantic_values=unchanged"
                      << " client_decrypt_reencrypt=0"
                      << std::endl;
        }
        std::cout << "CKKS_PLAIN_MATERIALIZE op=BSGS2 input_chain="
                  << input_context->chain_index() << " input_scale_bits="
                  << std::log2(inputs.front().scale())
                  << " weight_scale_bits=" << std::log2(params.weight_scale)
                  << " bias_scale_bits=" << std::log2(params.bias_scale)
                  << " weight_fractional_bits="
                  << params.weight_fractional_bits
                  << " bias_fractional_bits=" << params.bias_fractional_bits
                  << " semantic_conversion=phantom_direct elapsed_ms=0"
                  << std::endl;
        return {input_context->parms_id(), params.weight_scale,
                params.bias_scale};
    }
    if (params.cross_mat_single.first.empty() ||
        params.cross_mat_single.first.front().empty()) {
        throw std::runtime_error("BSGS2: weight preprocessing is empty");
    }
    const double weight_scale =
        params.cross_mat_single.first.front().front().scale();
    const double bias_scale = inputs.front().scale() * weight_scale;
    require_fractional_bits(params.weight_fractional_bits, "BSGS2.weight");
    require_fractional_bits(params.bias_fractional_bits, "BSGS2.bias");
    const bool source_is_checkpoint_integer =
        !params.semantic_values_materialized;
    const bool materialized_at_target =
        params.cross_mat_single.first.front().front().parms_id() ==
            input_context->parms_id() &&
        params.cross_bias_single.front().parms_id() ==
            input_context->parms_id() &&
        ckks_scales_close(params.cross_mat_single.first.front().front().scale(),
                          weight_scale) &&
        ckks_scales_close(params.cross_bias_single.front().scale(),
                          bias_scale);
    if (!materialized_at_target) {
        reencode_linear_weights(
            he, params.cross_mat_single, input_context->parms_id(), weight_scale,
            params.weight_fractional_bits, source_is_checkpoint_integer,
            "linear.weight");
        reencode_plain_vector(
            he, params.cross_bias_single, input_context->parms_id(), bias_scale,
            params.bias_fractional_bits, source_is_checkpoint_integer,
            "linear.bias");
    }
    params.semantic_values_materialized = true;
    const auto materialize_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                  materialize_start).count();
    std::cout << "CKKS_PLAIN_MATERIALIZE op=BSGS2 input_chain="
              << input_context->chain_index() << " input_scale_bits="
              << std::log2(inputs.front().scale()) << " weight_scale_bits="
              << std::log2(weight_scale) << " bias_scale_bits="
              << std::log2(bias_scale) << " weight_fractional_bits="
              << params.weight_fractional_bits << " bias_fractional_bits="
              << params.bias_fractional_bits << " semantic_conversion="
              << (materialized_at_target ? "offline_cache" :
                  (source_is_checkpoint_integer ? "applied" : "reused"))
              << " elapsed_ms=" << materialize_ms
              << std::endl;
    return {input_context->parms_id(), weight_scale, bias_scale};
}

// Figure 10's linear stages consume fixed CKKS levels/scales.  The official
// BLB preprocessing therefore materializes the packed plaintexts at those
// levels before the online boundary.  This metadata ciphertext contains no
// model/input data; it only lets the existing materialization routine select
// the published SEAL context level without entering the online protocol.
std::shared_ptr<const seal::SEALContext::ContextData> context_at_chain(
    HE_CKKS &he, std::size_t seal_chain, const char *label) {
    const auto first = he.context->first_context_data();
    if (!first) {
        throw std::runtime_error(std::string(label) +
                                 ": CKKS context has no first data level");
    }
    // Materialize at the exact SEAL data level mirrored by the ciphertext
    // that enters the published Phantom evaluator stage.  The former ViT
    // migration ignored this argument and always selected first_context_data,
    // so QKV/output/FFN1 re-encoded every real packed weight online.  Selecting
    // the already-prescribed level here restores BLB's offline preprocessing;
    // it does not add a cache, packing, rescale, or evaluator operation.
    std::shared_ptr<const seal::SEALContext::ContextData> target_context;
    for (auto data = first; data; data = data->next_context_data()) {
        if (data->chain_index() == seal_chain) {
            target_context = data;
            break;
        }
    }
    if (!target_context) {
        throw std::runtime_error(std::string(label) +
                                 ": invalid offline SEAL chain");
    }
    return target_context;
}

double scale_after_rescale(HE_CKKS &he, std::size_t source_chain,
                           double source_scale, const char *label) {
    const auto source_context = context_at_chain(he, source_chain, label);
    const auto &moduli = source_context->parms().coeff_modulus();
    if (moduli.empty() || !std::isfinite(source_scale) ||
        source_scale <= 0.0) {
        throw std::runtime_error(std::string(label) +
                                 ": invalid rescale metadata");
    }
    return source_scale / static_cast<double>(moduli.back().value());
}

UnifiedCiphertext make_offline_metadata_ciphertext(
    HE_CKKS &he, std::size_t seal_chain, double scale, const char *label) {
    if (!he.is_gpu_enable()) {
        throw std::runtime_error(std::string(label) +
                                 ": offline Phantom metadata requires DEVICE backend");
    }
    const auto target_context = context_at_chain(he, seal_chain, label);
    std::vector<double> zeros(he.num_slots, 0.0);
    Plaintext encoded;
    he.encoder->encode(zeros, target_context->parms_id(), scale, encoded);
    Ciphertext encrypted;
    he.encryptor->encrypt(encoded, encrypted);
    return UnifiedCiphertext(std::move(encrypted));
}

} // namespace


vector<Plaintext> Linear_ckks::generate_cross_packing_masks(HE_CKKS* he, const FCMetadata &data) {
    vector<Plaintext> result(data.image_size * 2);
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < data.image_size; i++) {
        vector<double> mask1(data.slot_count, 0ULL);
        vector<double> mask2(data.slot_count, 0ULL);
        for (int k = 0; k < data.image_size - i; k++) {
            mask1[k + (i * data.image_size) % (data.slot_count / 2)] = 1;
            mask1[k + data.slot_count / 2 + (i * data.image_size) % (data.slot_count / 2)] = 1;
        }
        for (int k = data.image_size - i; k < data.image_size; k++) {
            mask2[k + (i * data.image_size) % (data.slot_count / 2)] = 1;
            mask2[k + data.slot_count / 2 + (i * data.image_size) % (data.slot_count / 2)] = 1;
        }
        Plaintext pt1, pt2;
        he->encoder->encode(mask1, pow(2,he->he_scale), pt1);
        he->encoder->encode(mask2, pow(2,he->he_scale), pt2);
        result[i] = pt1;
        result[i + data.image_size] = pt2;
    }
    return result;
}

PreprocessParams_1 Linear_ckks::params_preprocessing_ct_ct(
    HE_CKKS* he,
    const vector<vector<vector<uint64_t>>> &w_q,
    const vector<vector<vector<uint64_t>>> &w_k,
    const vector<vector<vector<uint64_t>>> &w_v,
    const vector<vector<uint64_t>> &b_q,
    const vector<vector<uint64_t>> &b_k,
    const vector<vector<uint64_t>> &b_v,
    const FCMetadata &data,
    std::size_t input_chain,
    double input_scale
){
    const auto packing_start = std::chrono::steady_clock::now();
    PreprocessParams_1 pp;
    require_fractional_bits(pp.weight_fractional_bits,
                            "QKV preprocessing weight");
    require_fractional_bits(pp.bias_fractional_bits,
                            "QKV preprocessing bias");
    const auto input_context = context_at_chain(
        *he, input_chain, "QKV direct packing");
    const auto output_context = input_context->next_context_data();
    if (!output_context) {
        throw runtime_error("QKV direct packing has no output level");
    }
    const auto &moduli = input_context->parms().coeff_modulus();
    if (moduli.empty()) {
        throw runtime_error("QKV direct packing has no coefficient modulus");
    }
    const double weight_scale = std::ldexp(1.0, he->he_scale);
    const double bias_scale =
        input_scale * weight_scale / static_cast<double>(moduli.back().value());
    pp.weight_chain = input_context->chain_index();
    pp.bias_chain = output_context->chain_index();
    pp.weight_scale = weight_scale;
    pp.bias_scale = bias_scale;
    pp.input_scale = input_scale;
    // The upstream GPU benchmark encodes its (placeholder) plaintext directly
    // with Phantom.  Keep an opt-in real-checkpoint equivalent so the ViT
    // adapter can avoid copying a full SEAL/RNS plaintext over PCIe for every
    // ct-pt multiply.  CPU continues to use the original SEAL representation.
    pp.device_semantic_packed = use_direct_device_plaintext_packing(*he);

    cout << "filter.h, filter.w:" << data.filter_h << "," << data.filter_w << endl;
    
    pp.wq_pack = bert_cross_packing_single_matrix(
        he, w_q, data, pp.weight_fractional_bits,
        input_context->parms_id(), weight_scale,
        pp.device_semantic_packed ? &pp.device_wq_pack : nullptr);
    pp.wk_pack = bert_cross_packing_single_matrix(
        he, w_k, data, pp.weight_fractional_bits,
        input_context->parms_id(), weight_scale,
        pp.device_semantic_packed ? &pp.device_wk_pack : nullptr);
    pp.wv_pack = bert_cross_packing_single_matrix(
        he, w_v, data, pp.weight_fractional_bits,
        input_context->parms_id(), weight_scale,
        pp.device_semantic_packed ? &pp.device_wv_pack : nullptr);
    // cout << "OK here" << endl;
    // exit(0);
    pp.bq_pack = bert_cross_packing_bias(
        he, b_q, data, pp.bias_fractional_bits,
        output_context->parms_id(), bias_scale,
        pp.device_semantic_packed ? &pp.device_bq_pack : nullptr);
    pp.bk_pack = bert_cross_packing_bias(
        he, b_k, data, pp.bias_fractional_bits,
        output_context->parms_id(), bias_scale,
        pp.device_semantic_packed ? &pp.device_bk_pack : nullptr);
    pp.bv_pack = bert_cross_packing_bias(
        he, b_v, data, pp.bias_fractional_bits,
        output_context->parms_id(), bias_scale,
        pp.device_semantic_packed ? &pp.device_bv_pack : nullptr);

	pp.cross_masks = generate_cross_packing_masks(he, data);
	// The packing helpers already convert checkpoint integers to their semantic
	// real values using the published fractional-bit metadata.
	pp.semantic_values_materialized = true;
	const auto packing_ms = std::chrono::duration_cast<
		std::chrono::milliseconds>(std::chrono::steady_clock::now() -
		                          packing_start).count();
	cout << "CKKS_SEMANTIC_PACK op=QKV weight_fractional_bits="
	     << pp.weight_fractional_bits << " bias_fractional_bits="
	     << pp.bias_fractional_bits << " elapsed_ms=" << packing_ms << endl;

    return pp;
}

PreprocessParams_2 Linear_ckks::params_preprocessing_ct_pt(
    HE_CKKS* he,
    int32_t input_dim, 
    int32_t common_dim, 
    int32_t output_dim,
    const vector<vector<uint64_t>> &w,
    const vector<uint64_t> &b,
    const FCMetadata &data,
    int weight_fractional_bits,
    int bias_fractional_bits,
    std::size_t input_chain,
    double input_scale
){
    const auto packing_start = std::chrono::steady_clock::now();
    PreprocessParams_2 pp;
    require_fractional_bits(weight_fractional_bits,
                            "BSGS2 preprocessing weight");
    require_fractional_bits(bias_fractional_bits,
                            "BSGS2 preprocessing bias");
    pp.weight_fractional_bits = weight_fractional_bits;
    pp.bias_fractional_bits = bias_fractional_bits;
    const auto input_context = context_at_chain(
        *he, input_chain, "BSGS2 direct packing");
    const double weight_scale = std::ldexp(1.0, he->he_scale);
    const double bias_scale = input_scale * weight_scale;
    pp.target_chain = input_context->chain_index();
    pp.weight_scale = weight_scale;
    pp.bias_scale = bias_scale;
    pp.input_scale = input_scale;
    pp.device_semantic_packed = use_direct_device_plaintext_packing(*he);

    vector<vector<uint64_t>> matrix_mod_p1_storage(
        common_dim, vector<uint64_t>(output_dim / 2));
    vector<vector<uint64_t>> matrix_mod_p2_storage(
        common_dim, vector<uint64_t>(output_dim / 2));
    vector<uint64_t *> matrix_mod_p1(common_dim);
    vector<uint64_t *> matrix_mod_p2(common_dim);
    cout << "common_dim, output_dim/2:" << common_dim << "," << output_dim/2 << endl;
    #pragma omp parallel for
    for (int i = 0; i < common_dim; i++) {
        matrix_mod_p1[i] = matrix_mod_p1_storage[i].data();
        matrix_mod_p2[i] = matrix_mod_p2_storage[i].data();
        for (int j = 0; j < output_dim / 2; j++) {
            matrix_mod_p1[i][j] = w[i][j];
            matrix_mod_p2[i][j] = w[i][j + output_dim / 2];
        }
    }
    
    pp.cross_mat_single = bert_cross_packing_single_matrix_2(
        he, matrix_mod_p1.data(), matrix_mod_p2.data(), data,
        weight_fractional_bits, input_context->parms_id(), weight_scale,
        pp.device_semantic_packed ? &pp.device_cross_mat_single : nullptr);
    cout << "here OK" << endl;
    pp.cross_bias_single = bert_cross_packing_bias_2(
        he, b.data(), data, bias_fractional_bits,
        input_context->parms_id(), bias_scale,
        pp.device_semantic_packed ? &pp.device_cross_bias_single : nullptr);
    // The packing helper already applied the published fractional-bit scaling.
    pp.semantic_values_materialized = true;
    const auto packing_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                  packing_start).count();
    cout << "CKKS_SEMANTIC_PACK op=BSGS2 filter=" << data.filter_h << 'x'
         << data.filter_w << " weight_fractional_bits="
         << weight_fractional_bits << " bias_fractional_bits="
         << bias_fractional_bits << " elapsed_ms=" << packing_ms << endl;
    return pp;
}

vector<vector<uint64_t>>
Linear_ckks::attention_output_projection_weights_for_bsgs(
    const vector<vector<uint64_t>> &weights) const {
    constexpr int kHalfHeadFeatures = OUTPUT_DIM / 2;
    constexpr int kPairFeatures = 2 * OUTPUT_DIM;
    static_assert(OUTPUT_DIM % 2 == 0,
                  "attention head width must split into two CKKS halves");
    static_assert(PACKING_NUM * OUTPUT_DIM == COMMON_DIM,
                  "attention heads must cover the model width");
    if (weights.size() != COMMON_DIM) {
        throw invalid_argument(
            "W_O packing permutation requires COMMON_DIM input rows");
    }
    vector<vector<uint64_t>> packed(weights.size());
    // interpreted block -> checkpoint/logical block within one head pair.
    constexpr array<int, 4> kLogicalBlock{0, 2, 1, 3};
    for (int interpreted = 0; interpreted < COMMON_DIM; ++interpreted) {
        const int pair_base =
            (interpreted / kPairFeatures) * kPairFeatures;
        const int within_pair = interpreted % kPairFeatures;
        const int interpreted_block = within_pair / kHalfHeadFeatures;
        const int within_block = within_pair % kHalfHeadFeatures;
        const int logical = pair_base +
            kLogicalBlock[interpreted_block] * kHalfHeadFeatures +
            within_block;
        packed[interpreted] = weights[logical];
    }
    cout << "FORMAL_WO_INPUT_LAYOUT status=PASS"
         << " source=av_paired_head_half"
         << " target=bsgs2_contiguous_features"
         << " block_features=" << kHalfHeadFeatures
         << " permutation=0,2,1,3"
         << " online_conversions=0 communication_bytes=0" << endl;
    return packed;
}

void Linear_ckks::load_model_tensors(const BertModel &bm){
    w_ln_1 = bm.w_ln_1;
    b_ln_1 = bm.b_ln_1;
    w_ln_2 = bm.w_ln_2;
    b_ln_2 = bm.b_ln_2;
    w_c = bm.w_c;
    b_c = bm.b_c;
    w_p = bm.w_p;
    b_p = bm.b_p;
}

void Linear_ckks::configure_heterogeneous_layer(
    int layer_id, const vector<int> &protocols,
    const vector<double> &alphas) {
    if (layer_id < 0 || layer_id >= ATTENTION_LAYERS ||
        protocols.size() != PACKING_NUM || alphas.size() != PACKING_NUM) {
        throw invalid_argument(
            "heterogeneous output projection configuration mismatch");
    }
    for (int head = 0; head < PACKING_NUM; ++head) {
        if (protocols[head] < 0 || protocols[head] > 2 ||
            !isfinite(alphas[head])) {
            throw invalid_argument(
                "heterogeneous output projection contains invalid metadata");
        }
    }
    heterogeneous_protocols[layer_id] = protocols;
    heterogeneous_alphas[layer_id] = alphas;
}

void Linear_ckks::weights_preprocess_layer(const BertModel &bm, int layer_id){
    const auto has_layer = [layer_id](const auto &values) {
        return layer_id >= 0 &&
               static_cast<size_t>(layer_id) < values.size();
    };
    if (!has_layer(bm.w_q) || !has_layer(bm.w_k) || !has_layer(bm.w_v) ||
        !has_layer(bm.w_o) || !has_layer(bm.w_i_1) ||
        !has_layer(bm.w_i_2) || !has_layer(bm.b_q) ||
        !has_layer(bm.b_k) || !has_layer(bm.b_v) || !has_layer(bm.b_o) ||
        !has_layer(bm.b_i_1) || !has_layer(bm.b_i_2) ||
        !has_layer(bm.w_ln_1) || !has_layer(bm.b_ln_1) ||
        !has_layer(bm.w_ln_2) || !has_layer(bm.b_ln_2)) {
        throw runtime_error("formal preprocessing refused: incomplete BERT layer weights");
    }
    if (layer_id >= static_cast<int>(pp_1.size()) ||
        layer_id >= static_cast<int>(pp_2.size()) ||
        layer_id >= static_cast<int>(pp_3.size()) ||
        layer_id >= static_cast<int>(pp_4.size())) {
        throw runtime_error("formal preprocessing refused: layer cache is not configured");
    }

    // These are the exact deterministic scales emitted by the existing BLB
    // evaluator immediately before each packed MatMul.  Use the real dropped
    // RNS prime, rather than rounding log2(scale) to an integer, so the bias
    // plaintext is encoded once and does not need online re-materialization.
    // Native Linear needs exactly one level beyond BLB: the packed QK/V
    // product enters W_O at chain 1 and W_O's real rescale lands directly at
    // the one-prime Algorithm-1 conversion boundary.  Its QK scale is solved
    // below from that final dropped prime, so a second padding level is not
    // necessary.
    const size_t native_chain_offset = native_linear_enabled ? 1 : 0;
    const double ln_attention_output_scale = scale_after_rescale(
        *he1, 6 + native_chain_offset,
        std::ldexp(1.0, 3 * he1->he_scale), "QKV input scale");
    auto qk_output_scale = [&](double qkv_input) {
        const double qkv_output = scale_after_rescale(
            *he1, 5 + native_chain_offset,
            qkv_input * std::ldexp(1.0, he1->he_scale),
            "QKV output scale");
        const double q_after_mask = scale_after_rescale(
            *he1, 4 + native_chain_offset, qkv_output * 8.0 *
                         std::ldexp(1.0, he1->he_scale),
            "Q mask-1 scale");
        const double k_after_mask = scale_after_rescale(
            *he1, 4 + native_chain_offset,
            qkv_output * std::ldexp(1.0, he1->he_scale),
            "K mask-1 scale");
        const double qk_product = scale_after_rescale(
            *he1, 3 + native_chain_offset, q_after_mask * k_after_mask,
            "QK product scale");
        return scale_after_rescale(
            *he1, 2 + native_chain_offset,
            qk_product * std::ldexp(1.0, he1->he_scale),
            "QK mask-2 scale");
    };
    // ViT's pre-norm affine naturally exits at roughly 2^72.  Reusing that
    // value in the BERT QK depth schedule leaves the final QK ciphertext at
    // only roughly 2^7; multiplying by an encoded public one later cannot
    // recover the precision already lost.  Solve the public scale schedule
    // backwards from either SELECTIVE's 2^49 conversion target or NATIVE's final-W_O
    // target, and lift the affine output before QKV. QK is quadratic in the
    // QKV input scale, hence the square-root correction.
    double qkv_input_scale = ln_attention_output_scale;
    const double qk_scale_before_lift = qk_output_scale(qkv_input_scale);
    double native_qk_target_scale = std::ldexp(1.0, 49);
    if (native_linear_enabled) {
        const auto wo_input_context = context_at_chain(
            *he1, 1, "NATIVE native W_O input");
        const auto &wo_input_moduli =
            wo_input_context->parms().coeff_modulus();
        if (wo_input_moduli.size() < 2)
            throw runtime_error("NATIVE native W_O lacks its final rescale level");
        const double wo_dropped_modulus =
            static_cast<double>(wo_input_moduli.back().value());
        const double wo_weight_scale = std::ldexp(1.0, he1->he_scale);
        const double conversion_scale = std::ldexp(1.0, 49);
        // AV has scale qk_scale^2.  BSGS2 multiplies by W_O at
        // wo_weight_scale, and the one remaining real rescale must produce
        // conversion_scale exactly.
        native_qk_target_scale = std::sqrt(
            conversion_scale * wo_dropped_modulus / wo_weight_scale);
    }
    if (sequence_length == 256) {
        qkv_input_scale *=
            std::sqrt(native_qk_target_scale / qk_scale_before_lift);
    }
    const double planned_qk_output_scale = qk_output_scale(qkv_input_scale);
    if (!std::isfinite(qkv_input_scale) || qkv_input_scale <= 0.0 ||
        !std::isfinite(planned_qk_output_scale) ||
        planned_qk_output_scale <= 0.0) {
        throw runtime_error("formal QK scale planner produced invalid metadata");
    }
    cout << "FORMAL_QK_SCALE_PLAN status=PASS sequence_length="
         << sequence_length
         << " ln_output_scale_bits=" << log2(ln_attention_output_scale)
         << " qkv_input_scale_bits=" << log2(qkv_input_scale)
         << " qk_without_lift_bits=" << log2(qk_scale_before_lift)
         << " planned_qk_scale_bits=" << log2(planned_qk_output_scale)
         << " native_qk_target_scale_bits="
         << log2(native_qk_target_scale)
         << " target_conversion_scale_bits=49" << endl;
    // Softmax and V meet at chain 4 after the diagonal selector rescale.
    // AV then multiplies both ciphertexts and the public output mask before
    // rescaling with chain 4's last coefficient prime into W_O at chain 3.
    // Using chain 5's adjacent (but non-identical) prime here makes the
    // offline Phantom scale metadata fail closed at the real W_O boundary.
    const double output_projection_input_scale = scale_after_rescale(
        *he2, 4, std::ldexp(1.0, 3 * he2->he_scale),
        "output projection input scale");
    const double ffn1_input_scale = scale_after_rescale(
        *he3, 6, std::ldexp(1.0, 3 * he3->he_scale), "FFN1 input scale");
    const double ffn2_input_scale = std::ldexp(1.0, 38);

    // BERT layer 0 has no preceding LN affine. Match the existing raw-input
    // four-rescale scale planner, rather than caching ViT's LN output level.
    size_t qkv_input_chain = 5 + native_chain_offset;
    if (bert_post_norm && layer_id == 0) {
        auto level = he1->context->first_context_data();
        qkv_input_chain = level->chain_index();
        int drops[4];
        for (int i=0; i<4; ++i) {
            if (!level || level->parms().coeff_modulus().size()<2)
                throw runtime_error("BERT raw QKV requires four rescale levels");
            drops[i] = level->parms().coeff_modulus().back().bit_count();
            level = level->next_context_data();
        }
        const int bits = (49 - 5*he1->he_scale + 2*drops[0] +
                          2*drops[1] + drops[2] + drops[3] + 1) / 2;
        qkv_input_scale = std::ldexp(1.0, bits);
    }
    cout << "Preprocessing layer " << layer_id << endl;
    FCMetadata data = layer_id == 0 ? data_lin1_0 : data_lin1_1;
    pp_1[layer_id] = params_preprocessing_ct_ct(
        he1,
        bm.w_q[layer_id],
        bm.w_k[layer_id],
        bm.w_v[layer_id],
        bm.b_q[layer_id],
        bm.b_k[layer_id],
        bm.b_v[layer_id],
        data,
        qkv_input_chain,
        qkv_input_scale
    );
    if (he1->is_gpu_enable()) {
        // ViT's pre-norm affine re-enters every QKV stage at the same
        // published level/scale; use that exact metadata offline.
        const std::size_t qkv_chain = qkv_input_chain;
        const double qkv_scale = qkv_input_scale;
        auto metadata = make_offline_metadata_ciphertext(
            *he1, qkv_chain, qkv_scale, "QKV");
        materialize_qkv_plaintexts(he1[0], {metadata}, pp_1[layer_id]);
        cout << "CKKS_OFFLINE_MATERIALIZE op=QKV layer=" << layer_id
             << " seal_chain=" << qkv_chain
             << " input_scale_bits=" << setprecision(17) << log2(qkv_scale)
             << defaultfloat << endl;
    }
    cout << "pp_1 done" << endl;
    const auto output_projection_weights =
        attention_output_projection_weights_for_bsgs(bm.w_o[layer_id]);
    if (!native_linear_enabled) {
      pp_2[layer_id] = params_preprocessing_ct_pt(
          he2,
          sequence_length,
          COMMON_DIM,
          COMMON_DIM,
          output_projection_weights,
          bm.b_o[layer_id],
          data_lin2,
          6,
          12,
          3,
          output_projection_input_scale
      );
    }
    if (!native_linear_enabled && he2->is_gpu_enable()) {
        auto metadata = make_offline_metadata_ciphertext(
            *he2, 3, output_projection_input_scale, "OutputProjection");
        materialize_linear_plaintexts(he2[0], {metadata}, pp_2[layer_id]);
        cout << "CKKS_OFFLINE_MATERIALIZE op=BSGS2 layer=" << layer_id
             << " stage=OutputProjection seal_chain=3 input_scale_bits="
             << setprecision(17) << log2(output_projection_input_scale)
             << defaultfloat << endl;
    }
    cout << "pp_2 done mode="
         << (native_linear_enabled ? "assignment_partitioned"
                                      : "original_full") << endl;
    if (native_linear_enabled) {
        if (heterogeneous_protocols[layer_id].size() != PACKING_NUM ||
            heterogeneous_alphas[layer_id].size() != PACKING_NUM) {
            throw runtime_error(
                "NATIVE weights require a public layer assignment before preprocessing");
        }
        vector<vector<uint64_t>> mpc_weights = bm.w_o[layer_id];
        vector<vector<uint64_t>> native_weights = bm.w_o[layer_id];
        vector<uint64_t> zero_bias(COMMON_DIM, 0);
        bool has_mpc = false;
        bool has_linear = false;
        // Alpha is normally close to 1/logical_tokens.  Multiplying the s6
        // checkpoint integer directly would quantize most W_O entries to zero.
        // Retain twenty extra public fractional bits in the offline fused
        // weight and tell the existing semantic packer its true denominator.
        constexpr int kNATIVEAlphaFractionalBits = 20;
        constexpr long double kNATIVEAlphaScale =
            static_cast<long double>(uint64_t{1} <<
                                     kNATIVEAlphaFractionalBits);
        for (int head = 0; head < PACKING_NUM; ++head) {
            const bool linear_head = heterogeneous_protocols[layer_id][head] == 2;
            has_linear = has_linear || linear_head;
            has_mpc = has_mpc || !linear_head;
            for (int feature = 0; feature < OUTPUT_DIM; ++feature) {
                const int row = head * OUTPUT_DIM + feature;
                if (linear_head) {
                    fill(mpc_weights[row].begin(), mpc_weights[row].end(), 0);
                    const long double alpha = heterogeneous_alphas[layer_id][head];
                    for (uint64_t &raw : native_weights[row]) {
                        constexpr uint64_t kCheckpointRing = uint64_t{1} << 37;
                        constexpr uint64_t kCheckpointMask = kCheckpointRing - 1;
                        constexpr uint64_t kCheckpointSign = uint64_t{1} << 36;
                        const uint64_t centered_word = raw & kCheckpointMask;
                        const int64_t centered =
                            (centered_word & kCheckpointSign)
                                ? static_cast<int64_t>(centered_word -
                                                       kCheckpointRing)
                                : static_cast<int64_t>(centered_word);
                        const long double scaled =
                            static_cast<long double>(centered) * alpha *
                            kNATIVEAlphaScale;
                        const long double ring_min =
                            -static_cast<long double>(uint64_t{1} << 36);
                        const long double ring_max =
                            static_cast<long double>((uint64_t{1} << 36) - 1);
                        if (scaled < ring_min || scaled > ring_max) {
                            throw runtime_error("NATIVE alpha-fused W_O overflow");
                        }
                        raw = static_cast<uint64_t>(llround(scaled)) &
                              kCheckpointMask;
                    }
                } else {
                    fill(native_weights[row].begin(), native_weights[row].end(), 0);
                }
            }
        }
        const auto mpc_projection_weights =
            attention_output_projection_weights_for_bsgs(mpc_weights);
        // Keep the MPC-normalized contribution on BLB's original he2 chain.
        // Native Linear heads use the one extra he1 level for QK->AV->WO,
        // but making Softmax/ReLU heads pay for that extended chain is slower
        // than one additional real output conversion in a mixed layer.
        const double native_av_input_scale =
            planned_qk_output_scale * planned_qk_output_scale;
        if (has_mpc) {
            pp_2_mpc[layer_id] = params_preprocessing_ct_pt(
                he2, sequence_length, COMMON_DIM, COMMON_DIM,
                mpc_projection_weights, bm.b_o[layer_id], data_lin2,
                6, 12, 3, output_projection_input_scale);
        }
        if (has_linear) {
            const auto native_projection_weights =
                attention_output_projection_weights_for_bsgs(native_weights);
            // Native QK exits at chain 2 and the diagonal selector consumes
            // one level. V is value-preservingly aligned to the same planned
            // scale; native AV therefore enters W_O at chain 1 and QK^2.
            pp_2_native[layer_id] = params_preprocessing_ct_pt(
                he1, sequence_length, COMMON_DIM, COMMON_DIM,
                native_projection_weights, has_mpc ? zero_bias : bm.b_o[layer_id],
                data_lin2, 6 + kNATIVEAlphaFractionalBits, 12, 1,
                native_av_input_scale);
            if (he1->is_gpu_enable()) {
                auto metadata = make_offline_metadata_ciphertext(
                    *he1, 1, native_av_input_scale, "NATIVEOutputProjection");
                materialize_linear_plaintexts(
                    he1[0], {metadata}, pp_2_native[layer_id]);
            }
        }
        if (has_mpc && he2->is_gpu_enable()) {
            auto metadata = make_offline_metadata_ciphertext(
                *he2, 3, output_projection_input_scale,
                "SELECTIVEGroupedOutputProjection");
            materialize_linear_plaintexts(
                he2[0], {metadata}, pp_2_mpc[layer_id]);
        }
        cout << "OURS_NATIVE_WO_PREPROCESS status=PASS layer=" << layer_id
             << " mpc_heads="
             << count_if(heterogeneous_protocols[layer_id].begin(),
                         heterogeneous_protocols[layer_id].end(),
                         [](int p) { return p != 2; })
             << " linear_heads="
             << count(heterogeneous_protocols[layer_id].begin(),
                      heterogeneous_protocols[layer_id].end(), 2)
             << " alpha_fusion=WO_rows linear_bias="
             << (has_mpc ? "zero" : "original")
             << " mpc_bias=" << (has_mpc ? "original" : "none")
             << " alpha_fractional_bits=" << kNATIVEAlphaFractionalBits
             << " mpc_context=he2 linear_context=he1"
             << " native_cache_materialized=" << (has_linear ? 1 : 0);
        if (has_linear)
            cout << " native_input_chain=1 native_input_scale_bits="
                 << log2(native_av_input_scale);
        cout << endl;
    }
    pp_3[layer_id] = params_preprocessing_ct_pt(
        he3,
        sequence_length,
        COMMON_DIM,
        INTER_DIM,
        bm.w_i_1[layer_id],
        bm.b_i_1[layer_id],
        data_lin3,
        6,
        11,
        5,
        ffn1_input_scale
    );
    if (he3->is_gpu_enable()) {
        auto metadata = make_offline_metadata_ciphertext(
            *he3, 5, ffn1_input_scale, "FFN1");
        materialize_linear_plaintexts(he3[0], {metadata}, pp_3[layer_id]);
        cout << "CKKS_OFFLINE_MATERIALIZE op=BSGS2 layer=" << layer_id
             << " stage=FFN1 seal_chain=5 input_scale_bits="
             << setprecision(17) << log2(ffn1_input_scale)
             << defaultfloat << endl;
    }
    cout << "pp_3 done" << endl;
    pp_4[layer_id] = params_preprocessing_ct_pt(
        he4,
        sequence_length,
        INTER_DIM,
        COMMON_DIM,
        bm.w_i_2[layer_id],
        bm.b_i_2[layer_id],
        data_lin4,
        5,
        9,
        4,
        ffn2_input_scale
    );
    if (he4->is_gpu_enable()) {
        auto metadata = make_offline_metadata_ciphertext(
            *he4, 4, ffn2_input_scale, "FFN2");
        materialize_linear_plaintexts(he4[0], {metadata}, pp_4[layer_id]);
        cout << "CKKS_OFFLINE_MATERIALIZE op=BSGS2 layer=" << layer_id
             << " stage=FFN2 seal_chain=4 input_scale_bits=38" << endl;
    }
    cout << "pp_4 done" << endl;

}

void Linear_ckks::weights_preprocess(const BertModel &bm){
	weights_preprocess(bm, ATTENTION_LAYERS);
}

void Linear_ckks::weights_preprocess(const BertModel &bm,
                                     int layers_to_preprocess){
	if (layers_to_preprocess <= 0 || layers_to_preprocess > ATTENTION_LAYERS) {
		throw invalid_argument(
			"weights_preprocess: layer scope must be in [1, ATTENTION_LAYERS]");
	}
    load_model_tensors(bm);
	for (int layer_id = 0; layer_id < layers_to_preprocess; ++layer_id)
        weights_preprocess_layer(bm, layer_id);
}

void Linear_ckks::release_preprocessed_layer(int layer_id) {
    if (layer_id < 0 || layer_id >= static_cast<int>(pp_1.size()) ||
        layer_id >= static_cast<int>(pp_2.size()) ||
        layer_id >= static_cast<int>(pp_3.size()) ||
        layer_id >= static_cast<int>(pp_4.size())) {
        throw invalid_argument("release_preprocessed_layer: invalid layer");
    }
    // A twelve-layer ViT physical layout cannot retain every materialized
    // plaintext cache at once.  Releasing a completed layer changes only the
    // lifetime of offline data; the BLB packing and online evaluator graph are
    // unchanged.
    pp_1[layer_id] = PreprocessParams_1{};
    pp_2[layer_id] = PreprocessParams_2{};
    pp_2_mpc[layer_id] = PreprocessParams_2{};
    pp_2_native[layer_id] = PreprocessParams_2{};
    pp_3[layer_id] = PreprocessParams_2{};
    pp_4[layer_id] = PreprocessParams_2{};
    cout << "BLB_OFFLINE_CACHE_RELEASE layer=" << layer_id
         << " protocol_change=0 packing_change=0" << endl;
}

void Linear_ckks::linear_1(
		HE_CKKS* he,
		const vector<UnifiedCiphertext> &input_cts,
		PreprocessParams_1 &pp,
		const FCMetadata &data,
        vector<UnifiedCiphertext> &output,
        vector<UnifiedCiphertext> *b_qkv_observer) {

    cout << "input_cts.size: " << input_cts.size() << endl;
    const auto loc = input_cts[0].location();
	vector<UnifiedCiphertext> Cipher_plain_results(
        data.image_size * data.filter_w * 3 * PACKING_NUM / data.slot_count,
        UnifiedCiphertext(loc)
    );
	// Q,K,V
	if (profile_switch_callback != nullptr)
		profile_switch_callback(profile_context, 0);
	const auto qkv_profile_start = high_resolution_clock::now();
    bert_cipher_plain_bsgs(
        he, 
        input_cts, 
        pp,
        data, 
        Cipher_plain_results
    );
	if (profile_he_callback != nullptr)
		profile_he_callback(profile_context, 0,
			chrono::duration<double, milli>(high_resolution_clock::now() -
											 qkv_profile_start).count());
    // cout << "attention part1 time:" << (high_resolution_clock::now() - start).count()/1e+9 << endl;
    // in bert_cipher_plain_bsgs, there is a rescale already
    if (loc == LOCATION::HOST){
        #pragma omp parallel for
        for(int i=0;i<Cipher_plain_results.size();i++){
            // cout << "Cipher_plain_results: " << Cipher_plain_results[i].hcipher().size() << endl;
            he->evaluator->relinearize_inplace(Cipher_plain_results[i], *he->relin_keys);
        } 
    } else {
        /* FIXME: Why need relinearization? */
        // for(int i=0;i<Cipher_plain_results.size();i++){
        //     he->devaluator->relinearize_inplace(Cipher_plain_results[i], *he->drelin_keys);
        // } 
    }
    cout << "scale after WX: " << log2(Cipher_plain_results[0].scale()) << endl;
	cout << "bert_cipher_plain_bsgs done" << endl;
	const size_t q_ciphertexts = static_cast<size_t>(
		data.image_size * data.filter_w * PACKING_NUM / data.slot_count);
	if (Cipher_plain_results.size() != 3 * q_ciphertexts) {
		throw runtime_error("ViT Q/K/V ciphertext partition is incomplete");
	}
	for (size_t index = 0; index < q_ciphertexts; ++index) {
		Cipher_plain_results[index].scale() *= 8.0;
	}
	cout << "VIT_ATTENTION_PUBLIC_SCALE status=PASS target=Q divisor=8"
		 << " mechanism=CKKS_scale_metadata q_cts=" << q_ciphertexts
		 << " extra_multiply=0 extra_rescale=0" << endl;
	// B correctness tracing needs the actual Q/K/V values before the fused
	// QK^T kernel consumes them.  Copying ciphertexts is deliberately gated by
	// the caller and is never part of the benchmark graph.
	if (b_qkv_observer != nullptr) {
		*b_qkv_observer = Cipher_plain_results;
	}
	output.resize(
        data.image_size * data.image_size * PACKING_NUM / data.slot_count 
        + data.image_size * data.filter_w * PACKING_NUM / data.slot_count,
        UnifiedCiphertext(loc)
    );
	//QK^T
	if (profile_switch_callback != nullptr)
		profile_switch_callback(profile_context, 1);
	const auto qkt_profile_start = high_resolution_clock::now();
    bert_cipher_cipher_cross_packing(
        he, data, Cipher_plain_results, output);
	if (profile_he_callback != nullptr)
		profile_he_callback(profile_context, 1,
			chrono::duration<double, milli>(high_resolution_clock::now() -
											 qkt_profile_start).count());
	if (profile_switch_callback != nullptr)
		profile_switch_callback(profile_context, 0);

    // #pragma omp parallel for
    for (int i = 0; i < data.image_size * data.filter_w * PACKING_NUM / data.slot_count; i++) {
        output[data.image_size * data.image_size * PACKING_NUM / data.slot_count + i] = 
            std::move(Cipher_plain_results[data.image_size * data.filter_w * PACKING_NUM * 2 / data.slot_count + i]);
    }
}

void Linear_ckks::linear_2(
    HE_CKKS* he,
    const vector<UnifiedCiphertext> &input_cts, 
    PreprocessParams_2 &pp,
    const FCMetadata &data,
    vector<UnifiedCiphertext> &Cipher_plain_results
    ){
    Cipher_plain_results.resize(
        data.image_size * data.filter_w / data.slot_count,
        UnifiedCiphertext(input_cts[0].location()));
    bert_cipher_plain_bsgs_2(he, input_cts, pp, data, Cipher_plain_results);
}

pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>> 
Linear_ckks::bert_cross_packing_matrix(
	HE_CKKS* he,
	const uint64_t *const *matrix1, 
	const uint64_t *const *matrix2, 
	const FCMetadata &data) {

    vector<vector<Plaintext>> weightMatrix1; // 64 x 48
    vector<vector<Plaintext>> weightMatrix2; // 64 x 48
    vector<double> temp2;
    int num_diag = data.slot_count / data.image_size / 2; // should be 8
    int num_matrix_per_row = data.filter_h / num_diag; // should be 48
    int num_matrix_per_col = data.filter_w / num_diag; // should be 8

    int n1;
    int n2;
    if (data.slot_count == 4096) {
        n1 = 4;
        n2 = 4;
    }
    else {
        n1 = 8;
        n2 = 4;
    }

    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        for (int l = 0; l < num_diag; l++) {
            vector<Plaintext> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    for (int k = 0; k < data.image_size; k++) {
                        if (matrix_flag == 0)
                            temp2.push_back((int64_t)matrix1[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        else
                            temp2.push_back((int64_t)matrix2[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                    }
                    if (temp2.size() % (data.slot_count / 2) == 0) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        if (temp2.size() == data.slot_count) {
                            Plaintext pt;
                            he->encoder->encode(temp2,pow(2, he->he_scale), pt);
                            temp_matrix_diag[matrix_diag_index] = pt;
                            matrix_diag_index++;
                            temp2.clear();
                        }
                    }
                }
            }
            weightMatrix1.push_back(temp_matrix_diag);
        }
    }

    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        for (int l = 0; l < num_diag; l++) {
            vector<Plaintext> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    for (int k = 0; k < data.image_size; k++) {
                        if (matrix_flag == 0)
                            temp2.push_back((int64_t)matrix2[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        else
                            temp2.push_back((int64_t)matrix1[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                    }
                    if (temp2.size() % (data.slot_count / 2) == 0) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        if (temp2.size() == data.slot_count) {
                            std::rotate(temp2.begin(), temp2.begin() + temp2.size() / 2, temp2.end());
                            Plaintext pt;
                            he->encoder->encode(temp2, pow(2, he->he_scale), pt);
                            temp_matrix_diag[matrix_diag_index] = pt;
                            matrix_diag_index++;
                            temp2.clear();
                        }
                    }
                }
            }
            weightMatrix2.push_back(temp_matrix_diag);
        }
    }
    return std::make_pair(weightMatrix1, weightMatrix2);
}

vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>>
Linear_ckks::bert_cross_packing_single_matrix(
    HE_CKKS* he, const vector<vector<vector<uint64_t>>> &weights,
    const FCMetadata &data, int fractional_bits,
    const parms_id_type &target_level, double target_scale,
    vector<SemanticPackedWeight> *device_values) {
    vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> result(
        PACKING_NUM / 2);
    if (device_values != nullptr)
        device_values->resize(PACKING_NUM / 2);
    const double semantic_multiplier = checkpoint_semantic_multiplier(
        fractional_bits, "QKV packed weight");
    const int num_diag = data.slot_count / data.image_size / 2;
    int n1 = data.image_size == 64 ? 16 : 8;
    if (num_diag <= 0 || num_diag % n1 != 0)
        throw runtime_error("QKV BSGS blocking is incompatible with slot layout");
    const int n2 = num_diag / n1;
    const int feature_slots = data.slot_count / data.image_size;
    const int input_ciphertexts = data.filter_h / feature_slots;
    const int num_matrix_per_col = data.filter_w / num_diag;
    if (feature_slots != 2 * num_diag ||
        data.filter_h % feature_slots != 0 ||
        data.filter_w % num_diag != 0 ||
        static_cast<int>(weights.size()) != PACKING_NUM) {
        throw runtime_error("QKV packing shape is not slot aligned");
    }
    for (const auto &head : weights) {
        if (static_cast<int>(head.size()) != data.filter_h)
            throw runtime_error("QKV packing input dimension mismatch");
        for (const auto &row : head)
            if (static_cast<int>(row.size()) != data.filter_w)
                throw runtime_error("QKV packing output dimension mismatch");
    }

    // Derive each diagonal from the rotations performed by
    // bert_cipher_plain_bsgs.  The old implementation selected a head by
    // alternating every half plaintext.  That is not a head selector: the
    // half rotation is part of the lane mapping, so that rule silently
    // replaced rows of head 0 with rows of head 1 (and vice versa).  For a
    // final output slot, first determine which output feature half it belongs
    // to, then read that head's matrix at the exact input lane reached by the
    // baby/giant rotations.  This is the same inverse-rotation construction
    // used by the corrected BSGS2 packer below.
    #pragma omp parallel for num_threads(PACKING_NUM / 2)
    for (int packing_ind = 0; packing_ind < PACKING_NUM / 2;
         ++packing_ind) {
        vector<vector<Plaintext>> weightMatrix1;
        vector<vector<Plaintext>> weightMatrix2;
        SemanticPlainMatrix deviceWeightMatrix1;
        SemanticPlainMatrix deviceWeightMatrix2;
        for (int output_ciphertext = 0;
             output_ciphertext < num_matrix_per_col; ++output_ciphertext) {
            for (int diagonal = 0; diagonal < num_diag; ++diagonal) {
                const int baby = diagonal % n1;
                const int giant = diagonal / n1;
                vector<vector<double>> left_slots(
                    input_ciphertexts, vector<double>(data.slot_count));
                vector<vector<double>> right_slots(
                    input_ciphertexts, vector<double>(data.slot_count));
                for (int input_ciphertext = 0;
                     input_ciphertext < input_ciphertexts;
                     ++input_ciphertext) {
                    for (int source_slot = 0; source_slot < feature_slots;
                         ++source_slot) {
                        const int output_slot =
                            (source_slot + n1 * giant) % feature_slots;
                        // In QKV, the upper half of the CKKS vector is the
                        // paired attention head, not columns
                        // [filter_w/2, filter_w) of the same matrix.  The
                        // output ciphertext selects the 32-column block;
                        // the lane half selects head 2*p or 2*p+1.
                        const bool upper_head = output_slot >= num_diag;
                        const int output_feature =
                            output_ciphertext * num_diag +
                            (output_slot % num_diag);
                        const int matrix_column = output_feature;
                        const int head = packing_ind * 2 +
                            (upper_head ? 1 : 0);
                        const int left_local_feature =
                            baby == 0
                                ? source_slot
                                : (source_slot + num_diag - baby) %
                                      feature_slots;
                        const int right_local_feature =
                            (left_local_feature + num_diag) % feature_slots;
                        const int left_input_feature =
                            input_ciphertext * feature_slots +
                            left_local_feature;
                        const int right_input_feature =
                            input_ciphertext * feature_slots +
                            right_local_feature;
                        const double left_value = checkpoint_fixed_to_real(
                            weights[head][left_input_feature][matrix_column],
                            semantic_multiplier);
                        const double right_value = checkpoint_fixed_to_real(
                            weights[head][right_input_feature][matrix_column],
                            semantic_multiplier);
                        std::fill(
                            left_slots[input_ciphertext].begin() +
                                source_slot * data.image_size,
                            left_slots[input_ciphertext].begin() +
                                (source_slot + 1) * data.image_size,
                            left_value);
                        std::fill(
                            right_slots[input_ciphertext].begin() +
                                source_slot * data.image_size,
                            right_slots[input_ciphertext].begin() +
                                (source_slot + 1) * data.image_size,
                            right_value);
                    }
                }
                if (device_values != nullptr) {
                    deviceWeightMatrix1.push_back(std::move(left_slots));
                    deviceWeightMatrix2.push_back(std::move(right_slots));
                } else {
                    vector<Plaintext> left_plain(input_ciphertexts);
                    vector<Plaintext> right_plain(input_ciphertexts);
                    #pragma omp parallel for num_threads(input_ciphertexts)
                    for (int input_ciphertext = 0;
                         input_ciphertext < input_ciphertexts;
                         ++input_ciphertext) {
                        he->encoder->encode(
                            left_slots[input_ciphertext], target_level,
                            target_scale, left_plain[input_ciphertext]);
                        he->encoder->encode(
                            right_slots[input_ciphertext], target_level,
                            target_scale, right_plain[input_ciphertext]);
                    }
                    weightMatrix1.push_back(std::move(left_plain));
                    weightMatrix2.push_back(std::move(right_plain));
                }
            }
        }
        if (device_values != nullptr) {
            (*device_values)[packing_ind] = std::make_pair(
                std::move(deviceWeightMatrix1),
                std::move(deviceWeightMatrix2));
        } else {
            result[packing_ind] = std::make_pair(
                std::move(weightMatrix1), std::move(weightMatrix2));
        }
    }
    return result;
}

pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>
Linear_ckks::bert_cross_packing_single_matrix_2(
    HE_CKKS* he,
    const uint64_t *const *matrix1,
    const uint64_t *const *matrix2,
    const FCMetadata &data,
    int fractional_bits,
    const parms_id_type &target_level,
    double target_scale,
    SemanticPackedWeight *device_values){
    
    vector<vector<Plaintext>> weightMatrix1; // 64 x 48
    vector<vector<Plaintext>> weightMatrix2; // 64 x 48
    SemanticPlainMatrix deviceWeightMatrix1;
    SemanticPlainMatrix deviceWeightMatrix2;
    const double semantic_multiplier = checkpoint_semantic_multiplier(
        fractional_bits, "BSGS2 packed weight");
    
    int num_diag = data.slot_count / data.image_size / 2; // should be 8
    int num_matrix_per_row = data.filter_h / num_diag; // should be 48
    int num_matrix_per_col = data.filter_w / num_diag / 2; // should be 8

    int n1;
    int n2;
    if (data.filter_h == 3072 && data.filter_w == 768) {
        n1 = 2;
        n2 = 16;
        if (data.image_size == 64) {
            n1 = 4;
        }
    }
    else if (data.filter_h == 768 && data.filter_w == 3072) {
        n1 = 8;
        n2 = 4;
        if (data.image_size == 64) {
            n1 = 16;
        }
    }
    else if (data.filter_h == 768 && data.filter_w == 768) {
        n1 = 4;
        n2 = 8;
        if (data.image_size == 64) {
            n1 = 8;
        }
    }
    else {
        n1=4;
        n2=4;
        // assert (0);
    }
    if (num_diag <= 0 || num_diag % n1 != 0) {
        throw runtime_error("BSGS2 packing is incompatible with slot layout");
    }
    n2 = num_diag / n1;
    cout << "num_matrix_per_col: " << num_matrix_per_col << endl;
    const int feature_slots = data.slot_count / data.image_size;
    const int input_ciphertexts = data.filter_h / feature_slots;
    if (feature_slots != 2 * num_diag ||
        data.filter_h % feature_slots != 0) {
        throw runtime_error("BSGS2 packing feature layout is not slot aligned");
    }

    // Derive the plaintext diagonals from the online kernel itself.  For giant
    // step g and baby step i, the final right rotation moves source slot s to
    // output slot p=s+n1*g.  The two multiplicands read source feature
    // s+num_diag-i (i>0; i=0 is unrotated) and its half-swapped partner.
    // The former public packing used (j+l)%num_diag independently in each
    // half; that does not match these full-vector rotations and therefore
    // produced a deterministic permutation/mixing instead of XW.
    for (int output_ciphertext = 0;
         output_ciphertext < num_matrix_per_col; ++output_ciphertext) {
        cout << "col_ind: " << output_ciphertext << endl;
        for (int diagonal = 0; diagonal < num_diag; ++diagonal) {
            const int baby = diagonal % n1;
            const int giant = diagonal / n1;
            vector<vector<double>> left_slots(
                input_ciphertexts, vector<double>(data.slot_count));
            vector<vector<double>> right_slots(
                input_ciphertexts, vector<double>(data.slot_count));
            vector<Plaintext> left_plain(input_ciphertexts);
            vector<Plaintext> right_plain(input_ciphertexts);

            for (int input_ciphertext = 0;
                 input_ciphertext < input_ciphertexts; ++input_ciphertext) {
                for (int source_slot = 0; source_slot < feature_slots;
                     ++source_slot) {
                    const int output_slot =
                        (source_slot + n1 * giant) % feature_slots;
                    const int output_feature = output_slot < num_diag
                        ? output_ciphertext * num_diag + output_slot
                        : data.filter_w / 2 +
                              output_ciphertext * num_diag +
                              (output_slot - num_diag);
                    const int left_local_feature = baby == 0
                        ? source_slot
                        : (source_slot + num_diag - baby) % feature_slots;
                    const int right_local_feature =
                        (left_local_feature + num_diag) % feature_slots;
                    const int left_input_feature =
                        input_ciphertext * feature_slots + left_local_feature;
                    const int right_input_feature =
                        input_ciphertext * feature_slots + right_local_feature;
                    const bool output_in_upper_half =
                        output_feature >= data.filter_w / 2;
                    const int matrix_column = output_in_upper_half
                        ? output_feature - data.filter_w / 2
                        : output_feature;
                    const uint64_t left_raw = output_in_upper_half
                        ? matrix2[left_input_feature][matrix_column]
                        : matrix1[left_input_feature][matrix_column];
                    const uint64_t right_raw = output_in_upper_half
                        ? matrix2[right_input_feature][matrix_column]
                        : matrix1[right_input_feature][matrix_column];
                    const double left_value = checkpoint_fixed_to_real(
                        left_raw, semantic_multiplier);
                    const double right_value = checkpoint_fixed_to_real(
                        right_raw, semantic_multiplier);
                    std::fill(
                        left_slots[input_ciphertext].begin() +
                            source_slot * data.image_size,
                        left_slots[input_ciphertext].begin() +
                            (source_slot + 1) * data.image_size,
                        left_value);
                    std::fill(
                        right_slots[input_ciphertext].begin() +
                            source_slot * data.image_size,
                        right_slots[input_ciphertext].begin() +
                            (source_slot + 1) * data.image_size,
                        right_value);
                }
            }

            if (device_values != nullptr) {
                deviceWeightMatrix1.push_back(std::move(left_slots));
                deviceWeightMatrix2.push_back(std::move(right_slots));
            } else {
                #pragma omp parallel for num_threads(input_ciphertexts)
                for (int i = 0; i < input_ciphertexts; ++i) {
                    he->encoder->encode(left_slots[i], target_level,
                                        target_scale, left_plain[i]);
                    he->encoder->encode(right_slots[i], target_level,
                                        target_scale, right_plain[i]);
                }
                weightMatrix1.push_back(std::move(left_plain));
                weightMatrix2.push_back(std::move(right_plain));
            }
        }
    }
    
    if (device_values != nullptr) {
        *device_values = std::make_pair(std::move(deviceWeightMatrix1),
                                        std::move(deviceWeightMatrix2));
    }
    return std::make_pair(std::move(weightMatrix1),
                          std::move(weightMatrix2));
}


vector<vector<Plaintext>> Linear_ckks::bert_cross_packing_bias(
    HE_CKKS* he,
    const vector<vector<uint64_t>> &bias, 
    const FCMetadata &data,
    int fractional_bits,
    const parms_id_type &target_level,
    double target_scale,
    SemanticPlainMatrix *device_values) {
    vector<vector<Plaintext>> cross_bias_packing(PACKING_NUM/2);
    if (device_values != nullptr)
        device_values->resize(PACKING_NUM / 2);
    const double semantic_multiplier = checkpoint_semantic_multiplier(
        fractional_bits, "QKV packed bias");
    int current_packing = 0, matrix1_pointer = 0, matrix2_pointer = 0;
    while(current_packing < PACKING_NUM/2) {
        vector<double> temp(data.slot_count, 0);
        int next_flag = 0;
        int row = 0;
        if (matrix1_pointer == data.filter_w && matrix2_pointer == data.filter_w) {
            matrix1_pointer = 0;
            matrix2_pointer = 0;
            current_packing += 1;
            if (current_packing >= PACKING_NUM/2)
                break;
        }
        while (row < data.slot_count) {
            if (row >= data.slot_count / 2) {
                next_flag = 1;
            }
            if (next_flag == 0) {
                const double value = checkpoint_fixed_to_real(
                    bias[current_packing * 2][matrix1_pointer],
                    semantic_multiplier);
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = value;
                }
                matrix1_pointer++;
            }
            else {
                const double value = checkpoint_fixed_to_real(
                    bias[current_packing * 2 + 1][matrix2_pointer],
                    semantic_multiplier);
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = value;
                }
                matrix2_pointer++;
            }
            row += data.image_size;
        }
        if (device_values != nullptr) {
            (*device_values)[current_packing].push_back(std::move(temp));
        } else {
            Plaintext pt;
            he->encoder->encode(temp, target_level, target_scale, pt);
            cross_bias_packing[current_packing].push_back(std::move(pt));
        }
    }
    return cross_bias_packing;
}

vector<Plaintext> Linear_ckks::bert_cross_packing_bias_2(
    HE_CKKS* he,
    const uint64_t *matrix, 
    const FCMetadata &data,
    int fractional_bits,
    const parms_id_type &target_level,
    double target_scale,
    vector<SemanticPlaintext> *device_values){

    vector<Plaintext> cross_bias_packing(data.image_size * data.filter_w / data.slot_count);
    if (device_values != nullptr)
        device_values->resize(cross_bias_packing.size());
    const double semantic_multiplier = checkpoint_semantic_multiplier(
        fractional_bits, "BSGS2 packed bias");
    cout << "num for loop: " << data.image_size * data.filter_w / data.slot_count << endl;
    #pragma omp parallel for num_threads(data.image_size * data.filter_w / data.slot_count)
    for (int packing_num = 0; packing_num < data.image_size * data.filter_w / data.slot_count; packing_num++) {
        vector<double> temp(data.slot_count, 0);
        // Each plaintext owns a disjoint contiguous slice of both output
        // halves.  Derive the pointers from packing_num: the former shared
        // incrementing pointers raced under this OpenMP loop and could attach
        // the wrong real checkpoint bias to an otherwise valid ciphertext.
        const int values_per_half =
            data.slot_count / (2 * data.image_size);
        int matrix_pointer1 = packing_num * values_per_half;
        int matrix_pointer2 = data.filter_w / 2 +
                              packing_num * values_per_half;
        int row = 0;
        while (row < data.slot_count) {
            if (row < data.slot_count / 2) {
                const double value = checkpoint_fixed_to_real(
                    matrix[matrix_pointer1], semantic_multiplier);
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = value;
                }
                matrix_pointer1++;
            }
            else {
                const double value = checkpoint_fixed_to_real(
                    matrix[matrix_pointer2], semantic_multiplier);
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = value;
                }
                matrix_pointer2++;
            }
            row += data.image_size;
        }
        // The existing BSGS2 kernel adds bias before its first rescale, so the
        // bias plaintext must match ct_scale * weight_scale.
        if (device_values != nullptr) {
            (*device_values)[packing_num] = std::move(temp);
        } else {
            Plaintext pt;
            he->encoder->encode(temp, target_level, target_scale, pt);
            cross_bias_packing[packing_num] = std::move(pt);
        }
    }
    return cross_bias_packing;
}

// ckks only support rotate_vector, hence we change all rotate_row and rotate_column to rotate_vector
void Linear_ckks::bert_cipher_plain_bsgs(
    HE_CKKS* he,
    const vector<UnifiedCiphertext> &cts, 
    PreprocessParams_1 &pp,
    const FCMetadata &data, 
    vector<UnifiedCiphertext> &result) {

    int n1 = 8;
    int n2 = 4;
    if (data.image_size == 64) {
        n1 = 16;
    }
    const int num_diag = data.slot_count / data.image_size / 2;
    if (num_diag <= 0 || num_diag % n1 != 0) {
        throw runtime_error("QKV BSGS blocking is incompatible with slot layout");
    }
    n2 = num_diag / n1;
    const auto loc = cts[0].location();
    if (loc != LOCATION::HOST && loc != LOCATION::DEVICE) {
        throw runtime_error("formal QKV requires a HOST or DEVICE ciphertext");
    }
    if (loc == LOCATION::DEVICE && pp.device_semantic_packed) {
        if (pp.device_wq_pack.size() != PACKING_NUM / 2 ||
            pp.device_wk_pack.size() != PACKING_NUM / 2 ||
            pp.device_wv_pack.size() != PACKING_NUM / 2 ||
            pp.device_bq_pack.size() != PACKING_NUM / 2 ||
            pp.device_bk_pack.size() != PACKING_NUM / 2 ||
            pp.device_bv_pack.size() != PACKING_NUM / 2) {
            throw runtime_error(
                "formal QKV refused: Phantom semantic packing is incomplete");
        }
    } else if (pp.wq_pack.size() != PACKING_NUM / 2 ||
               pp.wk_pack.size() != PACKING_NUM / 2 ||
               pp.wv_pack.size() != PACKING_NUM / 2 ||
               pp.bq_pack.size() != PACKING_NUM / 2 ||
               pp.bk_pack.size() != PACKING_NUM / 2 ||
               pp.bv_pack.size() != PACKING_NUM / 2) {
        throw runtime_error("formal QKV refused: pp_1 is incomplete");
    }

    materialize_qkv_plaintexts(*he, cts, pp);
    const DevicePreprocessParams_1 *device_pp = nullptr;
#ifdef BLB_HAS_HE_GPU
    DevicePreprocessParams_1 device_params;
    if (loc == LOCATION::DEVICE && !pp.device_semantic_packed) {
        const auto copy_start = std::chrono::steady_clock::now();
        device_params = copy_qkv_params_to_device(*he, pp);
        const double device_prepare_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - copy_start).count();
        if (pp.device_semantic_packed)
            plaintext_device_encode_time += device_prepare_seconds;
        else
            plaintext_h2d_time += device_prepare_seconds;
        device_pp = &device_params;
    }
#else
    if (loc == LOCATION::DEVICE) {
        throw runtime_error(
            "formal QKV DEVICE backend was not compiled with USE_HE_GPU=ON");
    }
#endif

#ifdef BLB_HAS_HE_GPU
    double qkv_encode_seconds = 0.0;
    auto encode_qkv_plain = [&](PhantomCKKSEncoder &encoder,
                                const SemanticPlaintext &slots, double scale,
                                std::size_t chain, double &elapsed_seconds) {
        const auto encode_start = std::chrono::steady_clock::now();
        auto plain = encode_semantic_plaintext_to_device(
            *he, encoder, slots, scale, chain);
        elapsed_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - encode_start).count();
        return plain;
    };
#endif

    // begin computation
    vector<vector<UnifiedCiphertext>> rotatedIR(
        cts.size(), 
        vector<UnifiedCiphertext>(2 * n1, UnifiedCiphertext(loc)));

    int num_matrix_per_col = data.filter_w / num_diag;
    if (loc == LOCATION::HOST) {
        // ******************** CPU ********************
        #pragma omp parallel for num_threads(cts.size() * n1)
        for (int k = 0; k < cts.size() * n1; k++) {
            int i = k % cts.size();
            int j = k / cts.size();
            Ciphertext temp_rot;
            if (j == 0)
                rotatedIR[i][j] = cts[i];
            else {
                he->evaluator->rotate_vector(cts[i], (num_diag - j) * data.image_size, *(he->gal_keys), temp_rot);
                rotatedIR[i][j] = temp_rot;
            }
            he->evaluator->rotate_vector(rotatedIR[i][j],he->poly_modulus_degree/4, *(he->gal_keys), temp_rot);
            rotatedIR[i][j + n1] = temp_rot;
        }
    } else {
        // ******************** GPU ********************
        for (int k = 0; k < cts.size() * n1; k++) {
            int i = k % cts.size();
            int j = k / cts.size();
            if (j == 0)
                rotatedIR[i][j] = cts[i];
            else {
                he->devaluator->rotate_vector(cts[i], (num_diag - j) * data.image_size, *(he->dgal_keys), rotatedIR[i][j]);
            }
            he->devaluator->rotate_vector(rotatedIR[i][j],he->poly_modulus_degree/4, *(he->dgal_keys), rotatedIR[i][j + n1]);
        }
    }
    rotation_count += cts.size() * (2 * n1 - 1);
    // cout << "OK1" << endl;
    vector<vector<Ciphertext>> temp_results(data.image_size * data.filter_w * 3 * PACKING_NUM / data.slot_count, vector<Ciphertext>(n2));

    int temp_result_size = data.image_size * data.filter_w * 2 / data.slot_count;
    int omp_thread1 = 2, omp_thread2 = 16;
    if (data.image_size == 64) {
        omp_thread1 = 3;
        omp_thread2 = 8;
    }
    ct_pt_multiply_count += static_cast<uint64_t>(PACKING_NUM / 2) *
        cts.size() * n2 * temp_result_size * n1 * 6;
    rotation_count += static_cast<uint64_t>(PACKING_NUM / 2) *
        temp_result_size * (n2 - 1) * 3;
    if (loc == LOCATION::HOST) {
    // ******************** CPU ********************
    omp_set_nested(1);
    #pragma omp parallel for num_threads(PACKING_NUM/2)
    for (int packing_index = 0; packing_index < PACKING_NUM/2; packing_index++) {
        //compute matrix multiplication
        vector<vector<Ciphertext>> temp_results(temp_result_size * 3, vector<Ciphertext>(n2));
        vector<vector<Ciphertext>> temp_results_q(temp_result_size, vector<Ciphertext>(n2 * cts.size()));
        vector<vector<Ciphertext>> temp_results_k(temp_result_size, vector<Ciphertext>(n2 * cts.size()));
        vector<vector<Ciphertext>> temp_results_v(temp_result_size, vector<Ciphertext>(n2 * cts.size()));

        #pragma omp parallel for num_threads(omp_thread2)
        for (int k = 0; k < cts.size() * n2; k++) {
            int j = k / cts.size();
            int ct_i = k % cts.size();
            for (int l = 0; l < data.image_size * data.filter_w * 2 / data.slot_count; l++) {
                for (int i = 0; i < n1; i++) {
                    Ciphertext ct_l_q, ct_r_q, ct_l_k, ct_r_k, ct_l_v, ct_r_v;

                    const size_t diag = l * n1 * n2 + j * n1 + i;
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], pp.wq_pack[packing_index].first[diag][ct_i], ct_l_q);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], pp.wq_pack[packing_index].second[diag][ct_i], ct_r_q);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], pp.wk_pack[packing_index].first[diag][ct_i], ct_l_k);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], pp.wk_pack[packing_index].second[diag][ct_i], ct_r_k);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], pp.wv_pack[packing_index].first[diag][ct_i], ct_l_v);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], pp.wv_pack[packing_index].second[diag][ct_i], ct_r_v);
                    // cout << "mul done" << endl;
                    if (i == 0) {
                        he->evaluator->add(ct_l_q, ct_r_q, temp_results_q[l][k]);
                        he->evaluator->add(ct_l_k, ct_r_k, temp_results_k[l][k]);
                        he->evaluator->add(ct_l_v, ct_r_v, temp_results_v[l][k]);
                    }
                    else {
                        Ciphertext temp_add_ct;
                        he->evaluator->add(ct_l_q, ct_r_q, temp_add_ct);
                        he->evaluator->add_inplace(temp_results_q[l][k], temp_add_ct);
                        he->evaluator->add(ct_l_k, ct_r_k, temp_add_ct);
                        he->evaluator->add_inplace(temp_results_k[l][k], temp_add_ct);
                        he->evaluator->add(ct_l_v, ct_r_v, temp_add_ct);
                        he->evaluator->add_inplace(temp_results_v[l][k], temp_add_ct);
                    }
                    // cout << "add done" << endl;
                }

                ++rescale_count;
                he->evaluator->rescale_to_next_inplace(temp_results_q[l][k]);
                ++rescale_count;
                he->evaluator->rescale_to_next_inplace(temp_results_k[l][k]);
                ++rescale_count;
                he->evaluator->rescale_to_next_inplace(temp_results_v[l][k]);
            }
        }
        // cout << "part 1 ok" << endl;
        #pragma omp parallel for num_threads(n2)
        for (int j = 0; j < n2; j++) {
            for (int ct_i = 0; ct_i < cts.size(); ct_i++) {
                for (int l = 0; l < temp_result_size; l++) {
                    if (ct_i == 0) {
                        temp_results[l][j] = temp_results_q[l][j * cts.size() + ct_i];
                        temp_results[l + temp_result_size][j] = temp_results_k[l][j * cts.size() + ct_i];
                        temp_results[l + temp_result_size * 2][j] = temp_results_v[l][j * cts.size() + ct_i];
                    }
                    else {
                        he->evaluator->add_inplace(temp_results[l][j], temp_results_q[l][j * cts.size() + ct_i]);
                        he->evaluator->add_inplace(temp_results[l + temp_result_size][j], temp_results_k[l][j * cts.size() + ct_i]);
                        he->evaluator->add_inplace(temp_results[l + temp_result_size * 2][j], temp_results_v[l][j * cts.size() + ct_i]);
                    }
                }
            }
        }
        // cout << "OK2" << endl;
        #pragma omp parallel for num_threads(temp_result_size)
        for (int l = 0; l < temp_result_size; l++) {
            Ciphertext ct_q, ct_k, ct_v;
            for (int k = 0; k < n2; k++) {
                if (k == 0) {
                    ct_q = temp_results[l][0];
                    ct_k = temp_results[l + temp_result_size][0];
                    ct_v = temp_results[l + temp_result_size * 2][0];
                }
                else {
                    Ciphertext temp_rot_ct;
                    he->evaluator->rotate_vector(temp_results[l][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct_q, temp_rot_ct);
                    he->evaluator->rotate_vector(temp_results[l + temp_result_size][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct_k, temp_rot_ct);
                    he->evaluator->rotate_vector(temp_results[l + temp_result_size * 2][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct_v, temp_rot_ct);
                }
            }
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count] = ct_q;
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * PACKING_NUM / data.slot_count] = ct_k;
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * PACKING_NUM*2 / data.slot_count] = ct_v;

            he->evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count], pp.bq_pack[packing_index][l]);
            he->evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * PACKING_NUM / data.slot_count], pp.bk_pack[packing_index][l]);
            he->evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * PACKING_NUM*2 / data.slot_count], pp.bv_pack[packing_index][l]);
        }
    }
    } else {
    // ******************** GPU ********************
    const int stream_count = pp.device_semantic_packed
        ? direct_device_plaintext_streams() : 1;
    for (int packing_index = 0; packing_index < PACKING_NUM/2; packing_index++) {
        //compute matrix multiplication
        vector<vector<UnifiedCiphertext>> temp_results(temp_result_size * 3, vector<UnifiedCiphertext>(n2, UnifiedCiphertext(loc)));
        vector<vector<UnifiedCiphertext>> temp_results_q(temp_result_size, vector<UnifiedCiphertext>(n2 * cts.size(), UnifiedCiphertext(loc)));
        vector<vector<UnifiedCiphertext>> temp_results_k(temp_result_size, vector<UnifiedCiphertext>(n2 * cts.size(), UnifiedCiphertext(loc)));
        vector<vector<UnifiedCiphertext>> temp_results_v(temp_result_size, vector<UnifiedCiphertext>(n2 * cts.size(), UnifiedCiphertext(loc)));

#ifdef BLB_HAS_HE_GPU
        if (stream_count > 1) {
            const auto error = cudaStreamSynchronize(cudaStreamPerThread);
            if (error != cudaSuccess) {
                throw runtime_error(
                    string("QKV pre-stream synchronization failed: ") +
                    cudaGetErrorString(error));
            }
        }
        vector<double> stream_encode_seconds(stream_count, 0.0);
        vector<int> stream_errors(stream_count,
                                  static_cast<int>(cudaSuccess));
#pragma omp parallel num_threads(stream_count) if(stream_count > 1)
        {
            const int stream_id = omp_get_thread_num();
            std::unique_ptr<PhantomCKKSEncoder> stream_encoder;
            PhantomCKKSEncoder *encoder = he->dencoder;
            if (stream_count > 1) {
                stream_encoder.reset(new PhantomCKKSEncoder(*he->dcontext));
                encoder = stream_encoder.get();
            }
#pragma omp for schedule(static)
        for (int k = 0; k < static_cast<int>(cts.size()) * n2; k++) {
            int j = k / cts.size();
            int ct_i = k % cts.size();
            for (int l = 0; l < data.image_size * data.filter_w * 2 / data.slot_count; l++) {
                for (int i = 0; i < n1; i++) {
                    UnifiedCiphertext ct_l_q(loc);
                    UnifiedCiphertext ct_r_q(loc);
                    UnifiedCiphertext ct_l_k(loc);
                    UnifiedCiphertext ct_r_k(loc);
                    UnifiedCiphertext ct_l_v(loc);
                    UnifiedCiphertext ct_r_v(loc);

                    const size_t diag = l * n1 * n2 + j * n1 + i;
                    PhantomPlaintext owned_wq_l, owned_wq_r;
                    PhantomPlaintext owned_wk_l, owned_wk_r;
                    PhantomPlaintext owned_wv_l, owned_wv_r;
                    const PhantomPlaintext *wq_l_ptr;
                    const PhantomPlaintext *wq_r_ptr;
                    const PhantomPlaintext *wk_l_ptr;
                    const PhantomPlaintext *wk_r_ptr;
                    const PhantomPlaintext *wv_l_ptr;
                    const PhantomPlaintext *wv_r_ptr;
                    if (pp.device_semantic_packed) {
                        owned_wq_l = encode_qkv_plain(
                            *encoder,
                            pp.device_wq_pack[packing_index].first[diag][ct_i],
                            pp.weight_scale, pp.weight_chain,
                            stream_encode_seconds[stream_id]);
                        owned_wq_r = encode_qkv_plain(
                            *encoder,
                            pp.device_wq_pack[packing_index].second[diag][ct_i],
                            pp.weight_scale, pp.weight_chain,
                            stream_encode_seconds[stream_id]);
                        owned_wk_l = encode_qkv_plain(
                            *encoder,
                            pp.device_wk_pack[packing_index].first[diag][ct_i],
                            pp.weight_scale, pp.weight_chain,
                            stream_encode_seconds[stream_id]);
                        owned_wk_r = encode_qkv_plain(
                            *encoder,
                            pp.device_wk_pack[packing_index].second[diag][ct_i],
                            pp.weight_scale, pp.weight_chain,
                            stream_encode_seconds[stream_id]);
                        owned_wv_l = encode_qkv_plain(
                            *encoder,
                            pp.device_wv_pack[packing_index].first[diag][ct_i],
                            pp.weight_scale, pp.weight_chain,
                            stream_encode_seconds[stream_id]);
                        owned_wv_r = encode_qkv_plain(
                            *encoder,
                            pp.device_wv_pack[packing_index].second[diag][ct_i],
                            pp.weight_scale, pp.weight_chain,
                            stream_encode_seconds[stream_id]);
                        wq_l_ptr = &owned_wq_l;
                        wq_r_ptr = &owned_wq_r;
                        wk_l_ptr = &owned_wk_l;
                        wk_r_ptr = &owned_wk_r;
                        wv_l_ptr = &owned_wv_l;
                        wv_r_ptr = &owned_wv_r;
                    } else {
                        wq_l_ptr = &device_pp->wq_pack[packing_index]
                                          .first[diag][ct_i];
                        wq_r_ptr = &device_pp->wq_pack[packing_index]
                                          .second[diag][ct_i];
                        wk_l_ptr = &device_pp->wk_pack[packing_index]
                                          .first[diag][ct_i];
                        wk_r_ptr = &device_pp->wk_pack[packing_index]
                                          .second[diag][ct_i];
                        wv_l_ptr = &device_pp->wv_pack[packing_index]
                                          .first[diag][ct_i];
                        wv_r_ptr = &device_pp->wv_pack[packing_index]
                                          .second[diag][ct_i];
                    }
                    const auto &wq_l = *wq_l_ptr;
                    const auto &wq_r = *wq_r_ptr;
                    const auto &wk_l = *wk_l_ptr;
                    const auto &wk_r = *wk_r_ptr;
                    const auto &wv_l = *wv_l_ptr;
                    const auto &wv_r = *wv_r_ptr;
                    require_device_plain_compatible(
                        rotatedIR[ct_i][i], wq_l,
                        CkksCtPtOperation::Multiply, "QKV.wq.left");
                    require_device_plain_compatible(
                        rotatedIR[ct_i][i + n1], wq_r,
                        CkksCtPtOperation::Multiply, "QKV.wq.right");
                    require_device_plain_compatible(
                        rotatedIR[ct_i][i], wk_l,
                        CkksCtPtOperation::Multiply, "QKV.wk.left");
                    require_device_plain_compatible(
                        rotatedIR[ct_i][i + n1], wk_r,
                        CkksCtPtOperation::Multiply, "QKV.wk.right");
                    require_device_plain_compatible(
                        rotatedIR[ct_i][i], wv_l,
                        CkksCtPtOperation::Multiply, "QKV.wv.left");
                    require_device_plain_compatible(
                        rotatedIR[ct_i][i + n1], wv_r,
                        CkksCtPtOperation::Multiply, "QKV.wv.right");
                    he->devaluator->multiply_plain(rotatedIR[ct_i][i], wq_l,
                                                   ct_l_q);
                    he->devaluator->multiply_plain(rotatedIR[ct_i][i + n1],
                                                   wq_r, ct_r_q);
                    he->devaluator->multiply_plain(rotatedIR[ct_i][i], wk_l,
                                                   ct_l_k);
                    he->devaluator->multiply_plain(rotatedIR[ct_i][i + n1],
                                                   wk_r, ct_r_k);
                    he->devaluator->multiply_plain(rotatedIR[ct_i][i], wv_l,
                                                   ct_l_v);
                    he->devaluator->multiply_plain(rotatedIR[ct_i][i + n1],
                                                   wv_r, ct_r_v);
                    // cout << "mul done" << endl;
                    if (i == 0) {
                        he->devaluator->add(ct_l_q, ct_r_q, temp_results_q[l][k]);
                        he->devaluator->add(ct_l_k, ct_r_k, temp_results_k[l][k]);
                        he->devaluator->add(ct_l_v, ct_r_v, temp_results_v[l][k]);
                    }
                    else {
                        UnifiedCiphertext temp_add_ct(loc);
                        he->devaluator->add(ct_l_q, ct_r_q, temp_add_ct);
                        he->devaluator->add_inplace(temp_results_q[l][k], temp_add_ct);
                        he->devaluator->add(ct_l_k, ct_r_k, temp_add_ct);
                        he->devaluator->add_inplace(temp_results_k[l][k], temp_add_ct);
                        he->devaluator->add(ct_l_v, ct_r_v, temp_add_ct);
                        he->devaluator->add_inplace(temp_results_v[l][k], temp_add_ct);
                    }
                    // cout << "add done" << endl;
                }

                he->devaluator->rescale_to_next_inplace(temp_results_q[l][k]);
                he->devaluator->rescale_to_next_inplace(temp_results_k[l][k]);
                he->devaluator->rescale_to_next_inplace(temp_results_v[l][k]);
            }
        }
            if (stream_count > 1) {
                stream_errors[stream_id] = static_cast<int>(
                    cudaStreamSynchronize(cudaStreamPerThread));
            }
        }
        for (int error : stream_errors) {
            if (error != static_cast<int>(cudaSuccess)) {
                throw runtime_error(
                    string("QKV worker-stream synchronization failed: ") +
                    cudaGetErrorString(static_cast<cudaError_t>(error)));
            }
        }
        qkv_encode_seconds += *std::max_element(
            stream_encode_seconds.begin(), stream_encode_seconds.end());
        rescale_count += static_cast<uint64_t>(cts.size()) * n2 *
            temp_result_size * 3;
#else
        throw runtime_error(
            "formal QKV DEVICE backend was not compiled with USE_HE_GPU=ON");
#endif
        // cout << "part 1 ok" << endl;
        for (int j = 0; j < n2; j++) {
            for (int ct_i = 0; ct_i < cts.size(); ct_i++) {
                for (int l = 0; l < temp_result_size; l++) {
                    if (ct_i == 0) {
                        temp_results[l][j] = std::move(temp_results_q[l][j * cts.size() + ct_i]);
                        temp_results[l + temp_result_size][j] = std::move(temp_results_k[l][j * cts.size() + ct_i]);
                        temp_results[l + temp_result_size * 2][j] = std::move(temp_results_v[l][j * cts.size() + ct_i]);
                    }
                    else {
                        he->devaluator->add_inplace(temp_results[l][j], temp_results_q[l][j * cts.size() + ct_i]);
                        he->devaluator->add_inplace(temp_results[l + temp_result_size][j], temp_results_k[l][j * cts.size() + ct_i]);
                        he->devaluator->add_inplace(temp_results[l + temp_result_size * 2][j], temp_results_v[l][j * cts.size() + ct_i]);
                    }
                }
            }
        }
        // cout << "OK2" << endl;
        for (int l = 0; l < temp_result_size; l++) {
            UnifiedCiphertext ct_q(loc);
            UnifiedCiphertext ct_k(loc);
            UnifiedCiphertext ct_v(loc);
            for (int k = 0; k < n2; k++) {
                if (k == 0) {
                    ct_q = std::move(temp_results[l][0]);
                    ct_k = std::move(temp_results[l + temp_result_size][0]);
                    ct_v = std::move(temp_results[l + temp_result_size * 2][0]);
                }
                else {
                    // FIXME: Assume that temp_results[l + xxx][k] won't be reuesd
                    he->devaluator->rotate_vector_inplace(temp_results[l][k], -n1 * k * data.image_size, *(he->dgal_keys));
                    he->devaluator->add_inplace(ct_q, temp_results[l][k]);
                    he->devaluator->rotate_vector_inplace(temp_results[l + temp_result_size][k], -n1 * k * data.image_size, *(he->dgal_keys));
                    he->devaluator->add_inplace(ct_k, temp_results[l + temp_result_size][k]);
                    he->devaluator->rotate_vector_inplace(temp_results[l + temp_result_size * 2][k], -n1 * k * data.image_size, *(he->dgal_keys));
                    he->devaluator->add_inplace(ct_v, temp_results[l + temp_result_size * 2][k]);
                }
            }
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count] = std::move(ct_q);
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * PACKING_NUM / data.slot_count] = std::move(ct_k);
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * PACKING_NUM*2 / data.slot_count] = std::move(ct_v);

            auto &q_result = result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count];
            auto &k_result = result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * PACKING_NUM / data.slot_count];
            auto &v_result = result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * PACKING_NUM * 2 / data.slot_count];
            PhantomPlaintext owned_bq, owned_bk, owned_bv;
            const PhantomPlaintext *bq_ptr;
            const PhantomPlaintext *bk_ptr;
            const PhantomPlaintext *bv_ptr;
            if (pp.device_semantic_packed) {
                owned_bq = encode_qkv_plain(
                    *he->dencoder,
                    pp.device_bq_pack[packing_index][l], pp.bias_scale,
                    pp.bias_chain, qkv_encode_seconds);
                owned_bk = encode_qkv_plain(
                    *he->dencoder,
                    pp.device_bk_pack[packing_index][l], pp.bias_scale,
                    pp.bias_chain, qkv_encode_seconds);
                owned_bv = encode_qkv_plain(
                    *he->dencoder,
                    pp.device_bv_pack[packing_index][l], pp.bias_scale,
                    pp.bias_chain, qkv_encode_seconds);
                bq_ptr = &owned_bq;
                bk_ptr = &owned_bk;
                bv_ptr = &owned_bv;
            } else {
                bq_ptr = &device_pp->bq_pack[packing_index][l];
                bk_ptr = &device_pp->bk_pack[packing_index][l];
                bv_ptr = &device_pp->bv_pack[packing_index][l];
            }
            const auto &bq = *bq_ptr;
            const auto &bk = *bk_ptr;
            const auto &bv = *bv_ptr;
            require_device_plain_compatible(
                q_result, bq, CkksCtPtOperation::Add, "QKV.bq");
            require_device_plain_compatible(
                k_result, bk, CkksCtPtOperation::Add, "QKV.bk");
            require_device_plain_compatible(
                v_result, bv, CkksCtPtOperation::Add, "QKV.bv");
            he->devaluator->add_plain_inplace(q_result, bq);
            he->devaluator->add_plain_inplace(k_result, bk);
            he->devaluator->add_plain_inplace(v_result, bv);
        }
    }
    }
#ifdef BLB_HAS_HE_GPU
    if (loc == LOCATION::DEVICE && pp.device_semantic_packed) {
        plaintext_device_encode_time += qkv_encode_seconds;
        cout << "CKKS_DEVICE_PLAINTEXTS op=QKV weight_encodes="
             << static_cast<uint64_t>(PACKING_NUM / 2) * cts.size() *
                    n2 * temp_result_size * n1 * 6
             << " bias_encodes=" << static_cast<uint64_t>(PACKING_NUM / 2) *
                    temp_result_size * 3
             << " cache_policy=once_per_operator_use_bounded_lifetime"
             << " weight_source=real_checkpoint" << endl;
        cout << "CKKS_DEVICE_STREAMS op=QKV streams="
             << direct_device_plaintext_streams()
             << " critical_encode_ms=" << qkv_encode_seconds * 1000.0
             << endl;
    }
    if (loc == LOCATION::DEVICE && device_pp != nullptr) {
        device_pp = nullptr;
        plaintext_device_release_time +=
            release_device_plaintexts(device_params, "QKV");
    }
#endif
    cout << "bert_cipher_plain_bsgs done" << endl;
}

/*
    ckks only support rotate_vector, hence we change all rotate_row and rotate_column to rotate_vector
*/
void Linear_ckks::bert_cipher_plain_bsgs_2(
    HE_CKKS* he,
    const vector<UnifiedCiphertext> &cts, 
    PreprocessParams_2 &pp,
    const FCMetadata &data, 
    vector<UnifiedCiphertext> &result){
    const bool diagnostic_skip_flood =
        std::getenv("OURS_VIT_B_SKIP_BSGS2_FLOOD") != nullptr;
    int n1;
    int n2;
    if (data.filter_h == 3072 && data.filter_w == 768) {
        n1 = 2;
        n2 = 16;
        if (data.image_size == 64) {
            n1 = 4;
            n2 = 16;
        }
    }
    else if (data.filter_h == 768 && data.filter_w == 3072) {
        n1 = 8;
        n2 = 4;
        if (data.image_size == 64) {
            n1 = 16;
            n2 = 4;
        }
    }
    else if (data.filter_h == 768 && data.filter_w == 768) {
        n1 = 4;
        n2 = 8;
        if (data.image_size == 64) {
            n1 = 8;
            n2 = 8;
        }
    }
    else {
        n1=4;
        n2=4;
        // assert (0);
    }
    // The BSGS blocking must cover exactly the diagonal groups present in the
    // active Table-6 slot layout.  This also handles the physical-token
    // expansion used by the existing ViT caller without a separate heuristic.
    const int num_diag = data.slot_count / data.image_size / 2;
    if (num_diag <= 0 || num_diag % n1 != 0) {
        throw runtime_error("BSGS2 online blocking is incompatible with slot layout");
    }
    n2 = num_diag / n1;
    const auto loc = cts[0].location();
    if (loc != LOCATION::HOST && loc != LOCATION::DEVICE) {
        throw runtime_error("formal output/FFN requires a HOST or DEVICE ciphertext");
    }
    const bool use_device_semantics =
        loc == LOCATION::DEVICE && pp.device_semantic_packed;
    if (use_device_semantics) {
        if (pp.device_cross_mat_single.first.empty() ||
            pp.device_cross_mat_single.second.empty() ||
            pp.device_cross_bias_single.empty()) {
            throw runtime_error(
                "formal linear refused: Phantom semantic packing is incomplete");
        }
    } else if (pp.cross_mat_single.first.empty() ||
               pp.cross_mat_single.second.empty() ||
               pp.cross_bias_single.empty()) {
        throw runtime_error("formal linear refused: pp_2/pp_3/pp_4 is incomplete");
    }
    const auto diagonal_groups =
        data.image_size * data.filter_w / data.slot_count;
    const auto required_diagonals =
        diagonal_groups * n1 * n2;
    const auto packed_left_size = use_device_semantics
        ? pp.device_cross_mat_single.first.size()
        : pp.cross_mat_single.first.size();
    const auto packed_right_size = use_device_semantics
        ? pp.device_cross_mat_single.second.size()
        : pp.cross_mat_single.second.size();
    if (diagonal_groups <= 0 ||
        required_diagonals > static_cast<int>(packed_left_size) ||
        required_diagonals > static_cast<int>(packed_right_size)) {
        throw runtime_error(
            "BSGS2 packing shape mismatch: online diagonal count exceeds packed weights");
    }
    const auto required_ciphertexts =
        static_cast<size_t>(cts.size());
    for (int diagonal = 0; diagonal < required_diagonals; ++diagonal) {
        const auto left_width = use_device_semantics
            ? pp.device_cross_mat_single.first[diagonal].size()
            : pp.cross_mat_single.first[diagonal].size();
        const auto right_width = use_device_semantics
            ? pp.device_cross_mat_single.second[diagonal].size()
            : pp.cross_mat_single.second[diagonal].size();
        if (left_width < required_ciphertexts ||
            right_width < required_ciphertexts) {
            throw runtime_error(
                "BSGS2 packing shape mismatch: packed ciphertext width is too small");
        }
    }

    materialize_linear_plaintexts(*he, cts, pp);
    const DevicePreprocessParams_2 *device_pp = nullptr;
#ifdef BLB_HAS_HE_GPU
    DevicePreprocessParams_2 device_params;
    if (loc == LOCATION::DEVICE && !use_device_semantics) {
        const auto copy_start = std::chrono::steady_clock::now();
        device_params = copy_linear_params_to_device(*he, pp);
        const double device_prepare_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - copy_start).count();
        if (pp.device_semantic_packed)
            plaintext_device_encode_time += device_prepare_seconds;
        else
            plaintext_h2d_time += device_prepare_seconds;
        device_pp = &device_params;
    }
#else
    if (loc == LOCATION::DEVICE) {
        throw runtime_error(
            "formal output/FFN DEVICE backend was not compiled with USE_HE_GPU=ON");
    }
#endif

#ifdef BLB_HAS_HE_GPU
    double linear_encode_seconds = 0.0;
    auto encode_linear_plain = [&](PhantomCKKSEncoder &encoder,
                                   const SemanticPlaintext &slots,
                                   double scale, std::size_t chain,
                                   double &elapsed_seconds) {
        const auto encode_start = std::chrono::steady_clock::now();
        auto plain = encode_semantic_plaintext_to_device(
            *he, encoder, slots, scale, chain);
        elapsed_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - encode_start).count();
        return plain;
    };
#endif

    // start computation
    vector<vector<UnifiedCiphertext>> rotatedIR(
        cts.size(),
        vector<UnifiedCiphertext>(n1 * 2, UnifiedCiphertext(loc)));
    if (loc == LOCATION::HOST) {
    // ******************** CPU ********************
    #pragma omp parallel for num_threads(cts.size() * n1)
    for (int k = 0; k < cts.size() * n1; k++) {
        int i = k % cts.size();
        int j = k / cts.size();
        if (j == 0) {
            rotatedIR[i][j] = cts[i];
        }
        else {
            he->evaluator->rotate_vector(cts[i], (num_diag - j) * data.image_size, *(he->gal_keys), rotatedIR[i][j]);
        }
        he->evaluator->rotate_vector(rotatedIR[i][j], he->poly_modulus_degree/4, *(he->gal_keys), rotatedIR[i][j + n1]);
    }
    } else {
    // ******************** GPU ********************
    for (int k = 0; k < cts.size() * n1; k++) {
        int i = k % cts.size();
        int j = k / cts.size();
        if (j == 0) {
            rotatedIR[i][j] = cts[i];
        }
        else {
            he->devaluator->rotate_vector(cts[i], (num_diag - j) * data.image_size, *(he->dgal_keys), rotatedIR[i][j]);
        }
        he->devaluator->rotate_vector(rotatedIR[i][j], he->poly_modulus_degree/4, *(he->dgal_keys), rotatedIR[i][j + n1]);
    }
    rotation_count += cts.size() * (2 * n1 - 1);
    }
    // cout << "OK1" << endl;
    //compute matrix multiplication
    vector<vector<UnifiedCiphertext>> temp_results(
        data.image_size * data.filter_w / data.slot_count,
        vector<UnifiedCiphertext>(n2, UnifiedCiphertext(loc)));
    vector<vector<UnifiedCiphertext>> temp_results1(
        data.image_size * data.filter_w / data.slot_count,
        vector<UnifiedCiphertext>(n2 * cts.size(), UnifiedCiphertext(loc)));
    
    // rotatedIR 48 x 16, enc_mat 64 x 48
    ct_pt_multiply_count += static_cast<uint64_t>(cts.size()) * n2 *
        (data.image_size * data.filter_w / data.slot_count) * n1 * 2;
    rotation_count += static_cast<uint64_t>(data.image_size * data.filter_w /
        data.slot_count) * (n2 - 1);
    if (loc == LOCATION::HOST) {
    // ******************** CPU ********************
    #pragma omp parallel for num_threads(cts.size() * n2)
    for (int k = 0; k < cts.size() * n2; k++) {
        int j = k / cts.size();
        int ct_i = k % cts.size();
        // Keep the original BLB BSGS2 schedule: the outer loop owns the
        // parallelism.  A nested default-sized OpenMP team here multiplies
        // cts.size()*n2 by OMP_NUM_THREADS on the CPU (for ViT this can
        // create thousands of threads and exhaust memory).
        for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
            for (int i = 0; i < n1; i++) {
                Ciphertext ct1_l;
                Ciphertext ct1_r;
                const size_t diag = l * n1 * n2 + j * n1 + i;
                he->evaluator->multiply_plain(rotatedIR[ct_i][i], pp.cross_mat_single.first[diag][ct_i], ct1_l);
                he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], pp.cross_mat_single.second[diag][ct_i], ct1_r);
                // cout << "after scale: " << log2(ct1_l.scale()) << endl;
                if (i == 0)
                    he->evaluator->add(ct1_l, ct1_r, temp_results1[l][k]);
                else {
                    Ciphertext temp_add_ct;
                    he->evaluator->add(ct1_l, ct1_r, temp_add_ct);
                    he->evaluator->add_inplace(temp_results1[l][k], temp_add_ct);
                }
            }
        }
    }
    } else {
    // ******************** GPU ********************
    const int stream_count =
        use_device_semantics ? direct_device_plaintext_streams() : 1;
#ifdef BLB_HAS_HE_GPU
    if (stream_count > 1) {
        const auto error = cudaStreamSynchronize(cudaStreamPerThread);
        if (error != cudaSuccess) {
            throw runtime_error(
                string("BSGS2 pre-stream synchronization failed: ") +
                cudaGetErrorString(error));
        }
    }
    vector<double> stream_encode_seconds(stream_count, 0.0);
    vector<int> stream_errors(stream_count, static_cast<int>(cudaSuccess));
#pragma omp parallel num_threads(stream_count) if(stream_count > 1)
    {
        const int stream_id = omp_get_thread_num();
        std::unique_ptr<PhantomCKKSEncoder> stream_encoder;
        PhantomCKKSEncoder *encoder = he->dencoder;
        if (stream_count > 1) {
            stream_encoder.reset(new PhantomCKKSEncoder(*he->dcontext));
            encoder = stream_encoder.get();
        }
#pragma omp for schedule(static)
    for (int k = 0; k < static_cast<int>(cts.size()) * n2; k++) {
        int j = k / cts.size();
        int ct_i = k % cts.size();
        for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
            for (int i = 0; i < n1; i++) {
                UnifiedCiphertext ct1_l(loc);
                UnifiedCiphertext ct1_r(loc);
                const size_t diag = l * n1 * n2 + j * n1 + i;
                PhantomPlaintext owned_weight_l, owned_weight_r;
                const PhantomPlaintext *weight_l_ptr;
                const PhantomPlaintext *weight_r_ptr;
                if (use_device_semantics) {
                    owned_weight_l = encode_linear_plain(
                        *encoder,
                        pp.device_cross_mat_single.first[diag][ct_i],
                        pp.weight_scale, pp.target_chain,
                        stream_encode_seconds[stream_id]);
                    owned_weight_r = encode_linear_plain(
                        *encoder,
                        pp.device_cross_mat_single.second[diag][ct_i],
                        pp.weight_scale, pp.target_chain,
                        stream_encode_seconds[stream_id]);
                    weight_l_ptr = &owned_weight_l;
                    weight_r_ptr = &owned_weight_r;
                } else {
                    weight_l_ptr =
                        &device_pp->cross_mat_single.first[diag][ct_i];
                    weight_r_ptr =
                        &device_pp->cross_mat_single.second[diag][ct_i];
                }
                const auto &weight_l = *weight_l_ptr;
                const auto &weight_r = *weight_r_ptr;
                require_device_plain_compatible(
                    rotatedIR[ct_i][i], weight_l,
                    CkksCtPtOperation::Multiply, "BSGS2.weight.left");
                require_device_plain_compatible(
                    rotatedIR[ct_i][i + n1], weight_r,
                    CkksCtPtOperation::Multiply, "BSGS2.weight.right");
                he->devaluator->multiply_plain(rotatedIR[ct_i][i], weight_l,
                                               ct1_l);
                he->devaluator->multiply_plain(rotatedIR[ct_i][i + n1],
                                               weight_r, ct1_r);
                // cout << "after scale: " << log2(ct1_l.scale()) << endl;
                if (i == 0) {
                    he->devaluator->add(ct1_l, ct1_r, temp_results1[l][k]);
                }
                else {
                    he->devaluator->add_inplace(temp_results1[l][k], ct1_l);
                    he->devaluator->add_inplace(temp_results1[l][k], ct1_r);
                }
            }
        }
    }
        if (stream_count > 1) {
            stream_errors[stream_id] = static_cast<int>(
                cudaStreamSynchronize(cudaStreamPerThread));
        }
    }
    for (int error : stream_errors) {
        if (error != static_cast<int>(cudaSuccess)) {
            throw runtime_error(
                string("BSGS2 worker-stream synchronization failed: ") +
                cudaGetErrorString(static_cast<cudaError_t>(error)));
        }
    }
    const double critical_encode_seconds = *std::max_element(
        stream_encode_seconds.begin(), stream_encode_seconds.end());
    linear_encode_seconds += critical_encode_seconds;
    if (use_device_semantics) {
        cout << "CKKS_DEVICE_STREAMS op=BSGS2 streams=" << stream_count
             << " critical_encode_ms=" << critical_encode_seconds * 1000.0
             << endl;
    }
#else
    throw runtime_error(
        "formal output/FFN DEVICE backend was not compiled with USE_HE_GPU=ON");
#endif
    }

    // cout << "OK2" << endl;
    // FIXME: optimize this
    if (loc == LOCATION::HOST) {
    // ******************** CPU ********************
    #pragma omp parallel for num_threads(n2)
    for (int j = 0; j < n2; j++) {
        for (int ct_i = 0; ct_i < cts.size(); ct_i++) {
            for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
                if (ct_i == 0)
                    temp_results[l][j] = temp_results1[l][j * cts.size() + ct_i];
                else
                    he->evaluator->add_inplace(temp_results[l][j], temp_results1[l][j * cts.size() + ct_i]);
            }
        }
    }
    } else {
    // ******************** GPU ********************
    for (int j = 0; j < n2; j++) {
        for (int ct_i = 0; ct_i < cts.size(); ct_i++) {
            for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
                if (ct_i == 0) {
                    temp_results[l][j] = std::move(temp_results1[l][j * cts.size() + ct_i]);
                }
                else {
                    he->devaluator->add_inplace(temp_results[l][j], temp_results1[l][j * cts.size() + ct_i]);
                }
            }
        }
    }
    }
    cout << "OK3" << endl;
    if (loc == LOCATION::HOST) {
    // ******************** CPU ********************
    #pragma omp parallel for num_threads(data.image_size * data.filter_w / data.slot_count)
    for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
        Ciphertext ct;
        for (int k = 0; k < n2; k++) {
            if (k == 0)
                ct = temp_results[l][0];
            else {
                Ciphertext temp_rot_ct;
                he->evaluator->rotate_vector(temp_results[l][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                he->evaluator->add_inplace(ct, temp_rot_ct);
            }
        }
        result[l] = ct;
        he->evaluator->add_plain_inplace(result[l], pp.cross_bias_single[l]);
    }
    } else {
    // ******************** GPU ********************
    for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
        for (int k = 0; k < n2; k++) {
            if (k == 0) {
                result[l] = std::move(temp_results[l][0]);
            }
            else {
                UnifiedCiphertext temp_rot_ct(loc);
                he->devaluator->rotate_vector(temp_results[l][k], -n1 * k * data.image_size, *(he->dgal_keys), temp_rot_ct);
                he->devaluator->add_inplace(result[l], temp_rot_ct);
            }
        }
        PhantomPlaintext owned_bias;
        const PhantomPlaintext *bias_ptr;
        if (use_device_semantics) {
            owned_bias = encode_linear_plain(
                *he->dencoder,
                pp.device_cross_bias_single[l], pp.bias_scale,
                pp.target_chain, linear_encode_seconds);
            bias_ptr = &owned_bias;
        } else {
            bias_ptr = &device_pp->cross_bias_single[l];
        }
        const auto &bias = *bias_ptr;
        require_device_plain_compatible(
            result[l], bias, CkksCtPtOperation::Add, "BSGS2.bias");
        he->devaluator->add_plain_inplace(result[l], bias);
    }
    }
#ifdef BLB_HAS_HE_GPU
    if (loc == LOCATION::DEVICE && use_device_semantics) {
        plaintext_device_encode_time += linear_encode_seconds;
        cout << "CKKS_DEVICE_PLAINTEXTS op=BSGS2 weight_encodes="
             << static_cast<uint64_t>(cts.size()) * n2 *
                    (data.image_size * data.filter_w / data.slot_count) * n1 * 2
             << " bias_encodes=" << data.image_size * data.filter_w / data.slot_count
             << " cache_policy=once_per_operator_use_bounded_lifetime"
             << " weight_source=real_checkpoint" << endl;
    }
    if (loc == LOCATION::DEVICE && device_pp != nullptr) {
        device_pp = nullptr;
        plaintext_device_release_time +=
            release_device_plaintexts(device_params, "BSGS2");
    }
#endif
    // Every BSGS2 ciphertext that leaves Alice is subsequently protected by
    // mask_ckks_to_mpc_inplace: its c0 polynomial receives an independent
    // uniform element of the complete current RNS ring.  The client's
    // decrypted polynomial is therefore information-theoretically uniform.
    // An additional coefficient flood is redundant, destroys CKKS accuracy
    // at the final 60-bit conversion level, and forced a GPU->CPU->GPU copy.
    cout << "CKKS_RELEASE_GUARD op=BSGS2"
         << " protection=current_rns_uniform_polynomial_mask"
         << " applied_at=ckks_to_mpc_export"
         << " pre_export_flood=0"
         << " device_residency_preserved=" << (loc == LOCATION::DEVICE ? 1 : 0)
         << " diagnostic_skip_flag=" << (diagnostic_skip_flood ? 1 : 0)
         << endl;
    cout << "OK4" << endl;
    cout << "scale: " << log2(result[0].scale()) << endl;
}

// Compute QK^T
// 1. rotate rhs for 128 x 1-step rotations
// 2. mult with lhs (producing 128 cts)
// 3. for each of the 128 cts, rotate for log(32) times, sum together + 1 time batch rotation
// 4. mult masks (1, 0 (x31), 1, 0 (x31), ... ) and sum together (do the first 32 (1st batch) and then the second batch).

void Linear_ckks::bert_cipher_cipher_cross_packing(
	HE_CKKS* he,
	const FCMetadata &data,
	vector<UnifiedCiphertext> Cipher_plain_result, /* FIXME: D2D copy */
	vector<UnifiedCiphertext> &results) {
    // Official BLB BERT schedule (commit 25e8d86): this is the number of
    // rotations per attention head used by the paper's packed QK^T path.
    // Keep the schedule tied to the BERT dimensions; a generic sequence
    // fallback would silently change the paper's rotation count.
    int rot_ours; // #rotations of each head
    if (data.image_size == 128) {
        if (COMMON_DIM == 768) {
            // Original BLB BERT-base fused QK schedule (840 rotations).
            rot_ours = 840 / 2;
        } else if (COMMON_DIM == 1024) {
            // Original BLB BERT-large fused QK schedule (640 rotations).
            rot_ours = 640 / 2;
        } else {
            throw std::invalid_argument("unsupported BLB BERT hidden dimension");
        }
    } else if (data.image_size == 256) {
        const int packed_feature_ciphertexts =
            data.image_size * data.filter_w * 2 / data.slot_count;
        rot_ours = packed_feature_ciphertexts * data.image_size * 2;
    } else {
        throw std::invalid_argument(
            "unsupported BLB physical token count for packed QK");
    }
    const int feature_blocks = data.slot_count / 2 / data.image_size;
    if (feature_blocks <= 0 || (feature_blocks & (feature_blocks - 1)) != 0)
        throw runtime_error("packed QK requires power-of-two feature blocks");
    const auto loc = Cipher_plain_result[0].location();
    uint32_t num_rot=0;
    int packing_gap = data.image_size * data.filter_w / data.slot_count * 3;
    int temp_result_size = data.image_size * data.filter_w * 2 / data.slot_count;

    // These two public all-one masks are part of the original BLB fused QK
    // schedule.  The first mask moves Q and K to the product level; the
    // second mask performs the paper's final packed-column depth alignment.
    // They are not semantic changes to QK and must remain in the GPU path.
    const vector<double> pt_1(data.slot_count, 1.0);
    UnifiedPlaintext mask_1(loc);
    if (loc == LOCATION::HOST) {
        he->encoder->encode(pt_1, Cipher_plain_result[0].hcipher().parms_id(),
                            pow(2, he->he_scale), mask_1);
    } else {
        he->dencoder->encode(*he->dcontext, pt_1, pow(2, he->he_scale),
                             mask_1, Cipher_plain_result[0].dcipher().chain_index());
    }
    vector<UnifiedPlaintext> cross_masks;
    if (loc == LOCATION::HOST) {
        cross_masks.assign(data.image_size * 2, UnifiedPlaintext(loc));
        auto util_context_data =
            he->context->get_context_data(Cipher_plain_result[0].hcipher().parms_id());
        if (!util_context_data || !util_context_data->next_context_data() ||
            !util_context_data->next_context_data()->next_context_data()) {
            throw runtime_error("QK mask-2 level is absent");
        }
        util_context_data = util_context_data->next_context_data()->next_context_data();
        #pragma omp parallel for
        for (int diagonal = 0; diagonal < data.image_size; ++diagonal) {
            vector<double> forward(data.slot_count, 0.0);
            vector<double> wrapped(data.slot_count, 0.0);
            const int block =
                (diagonal * data.image_size) % (data.slot_count / 2);
            for (int token = 0; token < data.image_size; ++token) {
                auto &mask = token < data.image_size - diagonal
                                 ? forward : wrapped;
                mask[block + token] = 1.0;
                mask[data.slot_count / 2 + block + token] = 1.0;
            }
            he->encoder->encode(
                forward, util_context_data->parms_id(),
                pow(2, he->he_scale), cross_masks[diagonal]);
            he->encoder->encode(
                wrapped, util_context_data->parms_id(),
                pow(2, he->he_scale),
                cross_masks[diagonal + data.image_size]);
        }
    } else {
        cross_masks.assign(data.image_size * 2, UnifiedPlaintext(loc));
        const auto mask_chain = Cipher_plain_result[0].dcipher().chain_index() + 2;
        for (int diagonal = 0; diagonal < data.image_size; ++diagonal) {
            vector<double> forward(data.slot_count, 0.0);
            vector<double> wrapped(data.slot_count, 0.0);
            const int block =
                (diagonal * data.image_size) % (data.slot_count / 2);
            for (int token = 0; token < data.image_size; ++token) {
                auto &mask = token < data.image_size - diagonal
                                 ? forward : wrapped;
                mask[block + token] = 1.0;
                mask[data.slot_count / 2 + block + token] = 1.0;
            }
            he->dencoder->encode(
                *he->dcontext, forward, pow(2, he->he_scale),
                cross_masks[diagonal], mask_chain);
            he->dencoder->encode(
                *he->dcontext, wrapped, pow(2, he->he_scale),
                cross_masks[diagonal + data.image_size], mask_chain);
        }
    }

    if (loc == LOCATION::HOST) {
        #pragma omp parallel for num_threads(Cipher_plain_result.size())
        for (int i = 0; i < static_cast<int>(Cipher_plain_result.size()); ++i) {
            he->evaluator->multiply_plain_inplace(Cipher_plain_result[i], mask_1);
            he->evaluator->relinearize_inplace(Cipher_plain_result[i], *(he->relin_keys));
            he->evaluator->rescale_to_next_inplace(Cipher_plain_result[i]);
        }
    } else {
        for (auto &cipher : Cipher_plain_result) {
            he->devaluator->multiply_plain_inplace(cipher, mask_1);
            he->devaluator->rescale_to_next_inplace(cipher);
        }
    }
    cout << "scale: " << log2(Cipher_plain_result[0].scale()) << endl;
    if (loc == LOCATION::HOST) {
    // ******************** CPU ********************
    #pragma omp parallel for num_threads(PACKING_NUM/2)
    for (int packing_index = 0; packing_index < PACKING_NUM/2; packing_index++) {
        int rot_count=0;
        vector<Ciphertext> rotation_results(data.image_size * 2);
        for (int l = 0; l < temp_result_size; l++) {
            Ciphertext Qi = Cipher_plain_result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count];
            Ciphertext Ki = Cipher_plain_result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * PACKING_NUM / data.slot_count];
            #pragma omp parallel for num_threads(16)
            for (int i = 0; i < data.image_size; i++) {
                vector<Ciphertext> temp_mult;
                if(rot_count<rot_ours){
                    // two rotations inside
                    temp_mult = rotation_by_one_depth3(he, data, Ki, i);
                    rot_count += 2;
                }
                else{
                    temp_mult.push_back(Ki);
                    temp_mult.push_back(Ki);
                }
                if (l == 0) {
                    he->evaluator->multiply(Qi, temp_mult[0], rotation_results[i]);
                    he->evaluator->multiply(Qi, temp_mult[1], rotation_results[i + data.image_size]);
                }
                else {
                    Ciphertext temp_qk;
                    he->evaluator->multiply(Qi, temp_mult[0], temp_qk);
                    he->evaluator->add_inplace(rotation_results[i], temp_qk);
                    he->evaluator->multiply(Qi, temp_mult[1], temp_qk);
                    he->evaluator->add_inplace(rotation_results[i + data.image_size], temp_qk);
                }

            }
        }
        // cout << "part 1 ok" << endl;
        // cout << "scale: " << log2(rotation_results[0].scale()) << endl;
        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < data.image_size * 2; i++) {
            he->evaluator->relinearize_inplace(rotation_results[i], *(he->relin_keys));
            he->evaluator->rescale_to_next_inplace(rotation_results[i]);
        }
        // cout << "part 2 ok" << endl;

        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < data.image_size * 2; i++) {
            {
                const int blocks = data.slot_count / data.image_size / 2;
                const int target_block = (i % data.image_size) % blocks;
                for (int stride = 1; stride < blocks; stride <<= 1) {
                    Ciphertext rotated;
                    he->evaluator->rotate_vector(
                        rotation_results[i],
                        ours_vit::packed_block_rotation(
                            target_block, stride, data.image_size, false),
                        *(he->gal_keys), rotated);
                    he->evaluator->add_inplace(rotation_results[i], rotated);
                }
                he->evaluator->multiply_plain_inplace(
                    rotation_results[i], cross_masks[i]);
            }
            he->evaluator->relinearize_inplace(rotation_results[i], *(he->relin_keys));
            he->evaluator->rescale_to_next_inplace(rotation_results[i]);
        }
        // cout << "part 3 ok" << endl;
        int num_cts_per_res = data.image_size * data.image_size * 2 / data.slot_count; // 1 or 4
        int num_col_per_ct = data.slot_count / 2 / data.image_size; // 64 or 32

        #pragma omp parallel for num_threads(num_cts_per_res)
        for (int i = 0; i < num_cts_per_res; i++) {
            he->evaluator->add(rotation_results[num_col_per_ct * i], rotation_results[num_col_per_ct * i + data.image_size], results[packing_index * num_cts_per_res + i]);
            for (int j = 1; j < num_col_per_ct; j++) {
                he->evaluator->add_inplace(results[packing_index * num_cts_per_res + i], rotation_results[num_col_per_ct * i + j]);
                he->evaluator->add_inplace(results[packing_index * num_cts_per_res + i], rotation_results[num_col_per_ct * i + j + data.image_size]);
            }
        }
        // cout << "part 4 ok" << endl;
    }
    } else {
    // ******************** GPU ********************
    const int qk_streams = min(PACKING_NUM / 2, direct_device_plaintext_streams());
#ifdef BLB_HAS_HE_GPU
    const auto qk_ready = cudaStreamSynchronize(cudaStreamPerThread);
    if (qk_ready != cudaSuccess)
        throw runtime_error("QK input stream synchronization failed");
    vector<int> qk_errors(PACKING_NUM / 2, static_cast<int>(cudaSuccess));
#pragma omp parallel for num_threads(qk_streams) if(qk_streams > 1)
#endif
    for (int packing_index = 0; packing_index < PACKING_NUM/2; packing_index++) {
        int rot_count = 0;
        vector<UnifiedCiphertext> rotation_results(data.image_size * 2, UnifiedCiphertext(loc));
        for (int l = 0; l < temp_result_size; l++) {
            // FIXME: Assume that Cipher_plain_result won't be reuesd
            UnifiedCiphertext Qi = std::move(Cipher_plain_result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count]);
            UnifiedCiphertext Ki = std::move(Cipher_plain_result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * PACKING_NUM / data.slot_count]);
            for (int i = 0; i < data.image_size; i++) {
                if (rot_count < rot_ours) {
                    UnifiedCiphertext ct1(loc);
                    UnifiedCiphertext ct2(loc);
                    he->devaluator->rotate_vector(Ki, i, *(he->dgal_keys), ct1);
                    he->devaluator->rotate_vector(Ki, -(data.image_size - i), *(he->dgal_keys), ct2);
                    if (l == 0) {
                        he->devaluator->multiply(Qi, ct1, rotation_results[i]);
                        he->devaluator->multiply(Qi, ct2, rotation_results[i + data.image_size]);
                    } else {
                        UnifiedCiphertext temp_qk(loc);
                        he->devaluator->multiply(Qi, ct1, temp_qk);
                        he->devaluator->add_inplace(rotation_results[i], temp_qk);
                        he->devaluator->multiply(Qi, ct2, temp_qk);
                        he->devaluator->add_inplace(rotation_results[i + data.image_size], temp_qk);
                    }
                    rot_count += 2;
                } else {
                    if (l == 0) {
                        he->devaluator->multiply(Qi, Ki, rotation_results[i]);
                        he->devaluator->multiply(Qi, Ki, rotation_results[i + data.image_size]);
                    } else {
                        UnifiedCiphertext temp_qk(loc);
                        he->devaluator->multiply(Qi, Ki, temp_qk);
                        he->devaluator->add_inplace(rotation_results[i], temp_qk);
                        he->devaluator->multiply(Qi, Ki, temp_qk);
                        he->devaluator->add_inplace(rotation_results[i + data.image_size], temp_qk);
                    }
                }
            }
        }
        // cout << "part 1 ok" << endl;
        // cout << "scale: " << log2(rotation_results[0].scale()) << endl;
        for (int i = 0; i < data.image_size * 2; i++) {
            he->devaluator->relinearize_inplace(rotation_results[i], *(he->drelin_keys));
            he->devaluator->rescale_to_next_inplace(rotation_results[i]);
        }
        // cout << "part 2 ok" << endl;

        for (int i = 0; i < data.image_size * 2; i++) {
            {
                const int blocks = data.slot_count / data.image_size / 2;
                const int target_block = (i % data.image_size) % blocks;
                for (int stride = 1; stride < blocks; stride <<= 1) {
                    UnifiedCiphertext rotated(loc);
                    he->devaluator->rotate_vector(
                        rotation_results[i],
                        ours_vit::packed_block_rotation(
                            target_block, stride, data.image_size, false),
                        *(he->dgal_keys), rotated);
                    he->devaluator->add_inplace(rotation_results[i], rotated);
                }
                he->devaluator->multiply_plain_inplace(
                    rotation_results[i], cross_masks[i]);
            }
            he->devaluator->rescale_to_next_inplace(rotation_results[i]);
        }
        // cout << "part 3 ok" << endl;
        int num_cts_per_res = data.image_size * data.image_size * 2 / data.slot_count; // 1 or 4
        int num_col_per_ct = data.slot_count / 2 / data.image_size; // 64 or 32

        for (int i = 0; i < num_cts_per_res; i++) {
            he->devaluator->add(rotation_results[num_col_per_ct * i], rotation_results[num_col_per_ct * i + data.image_size], results[packing_index * num_cts_per_res + i]);
            for (int j = 1; j < num_col_per_ct; j++) {
                he->devaluator->add_inplace(results[packing_index * num_cts_per_res + i], rotation_results[num_col_per_ct * i + j]);
                he->devaluator->add_inplace(results[packing_index * num_cts_per_res + i], rotation_results[num_col_per_ct * i + j + data.image_size]);
            }
        }
        // Wait before per-head temporary ciphertexts leave this worker.
#ifdef BLB_HAS_HE_GPU
        qk_errors[packing_index] = static_cast<int>(
            cudaStreamSynchronize(cudaStreamPerThread));
#endif
    }
#ifdef BLB_HAS_HE_GPU
    for (int error : qk_errors)
        if (error != static_cast<int>(cudaSuccess))
            throw runtime_error("QK worker stream synchronization failed");
#endif
    cout << "CKKS_DEVICE_STREAMS op=QK streams=" << qk_streams << endl;
    }
    ct_ct_multiply_count += static_cast<uint64_t>(PACKING_NUM / 2) *
        temp_result_size * data.image_size * 2;
    ct_pt_multiply_count += static_cast<uint64_t>(PACKING_NUM / 2) *
        data.image_size * 2;
    rotation_count += static_cast<uint64_t>(PACKING_NUM / 2) *
        std::min(rot_ours, temp_result_size * data.image_size * 2);
    {
        rotation_count += static_cast<uint64_t>(PACKING_NUM / 2) *
            data.image_size * 2 *
            static_cast<uint64_t>(std::ceil(std::log2(
                data.slot_count / data.image_size / 2)));

    }
    if (COMMON_DIM == 1024) {
        qk_rotation_count += static_cast<uint64_t>(PACKING_NUM) *
            std::min(rot_ours, temp_result_size * data.image_size * 2);
    }
    cout << "num rotations: " << num_rot << endl;
    cout << "OURS_BLB_QK_PACKING status=EXECUTED mode=complete_diagonal"
         << " alignment_rotations=0 block_rotation=direct_target_v1"
         << " rotation_cap=" << rot_ours
         << " mpc_surrogate=0 client_decrypt_reencrypt=0"
         << " b_numerical_validation=PENDING"
         << endl;
    cout << "bert_cipher_cipher_cross_packing done" << endl;
}

vector<Ciphertext> Linear_ckks::rotation_by_one_depth3(
	HE_CKKS* he,
	const FCMetadata &data, 
	const Ciphertext &ct, 
	int k) {

    int m = -(data.image_size - k);
    Ciphertext ct1;
    Ciphertext ct2;
    he->evaluator->rotate_vector(ct, k, *(he->gal_keys), ct1);
    he->evaluator->rotate_vector(ct, m, *(he->gal_keys), ct2);

    return {ct1, ct2};
}

// column-wise packing
vector<Ciphertext> Linear_ckks::bert_efficient_preprocess_vec(
	HE_CKKS* he,
	vector<int64_t> &input, 
	const FCMetadata &data) {

    vector<Ciphertext> cts((data.image_size * data.filter_h) / data.slot_count);

    #pragma omp parallel for num_threads((data.image_size * data.filter_h) / data.slot_count)
    for (int i = 0; i < (data.image_size * data.filter_h) / data.slot_count; i++)
    {
        vector<double> pod_matrix(data.slot_count, 0);
        pod_matrix = vector<double>(input.begin() + i * data.slot_count, input.begin() + (i+1) * data.slot_count);
        Ciphertext ct;
        Plaintext pt;
        he->encoder->encode(pod_matrix,pow(2, he->he_scale), pt);
        he->encryptor->encrypt(pt, ct);
        cts[i] = ct;
    }
    return cts;
}

uint64_t* Linear_ckks::bert_cross_packing_postprocess(
    HE_CKKS* he,
    vector<Ciphertext> &cts, 
    const FCMetadata &data) {
    uint64_t *result = new uint64_t[data.image_size*data.image_size*PACKING_NUM];
    int num_cts_per_mat = data.image_size * data.image_size / data.slot_count;
    for (int packing_num = 0; packing_num < PACKING_NUM; packing_num++) {
        for (int i = 0; i < num_cts_per_mat; i++) {
            vector<double> plain(data.slot_count, 0);
            Plaintext tmp;
            he->decryptor->decrypt(cts[i + packing_num * num_cts_per_mat], tmp);
            he->encoder->decode(tmp, plain);

            #pragma omp parallel for num_threads(128)
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (j < 32) { // k, (k + j) % 128
                    result[k + ((k + j + i * 32) % data.image_size) * data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                }
                else if (j == 32 && i == 0) { // (64 + k) % 128, k
                    result[((k + 64) % data.image_size) + k * data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                }
                else { // (k - 32 + j) % 128, k
                    result[k * data.image_size + (k + j - 32 + i * 32) % 128 + packing_num * data.image_size * data.image_size] = plain[row];
                }
            }
        }
    }
    return result;
}


void Linear_ckks::plain_cross_packing_postprocess(
    uint64_t* input, 
    uint64_t * output,
    bool col_packing,
    const FCMetadata &data) {

    int cts_size = PACKING_NUM*data.image_size*data.image_size / data.slot_count;

    int num_cts_per_res = data.image_size * data.image_size * 2 / data.slot_count; // 1 or 4
    int num_col_per_ct = data.slot_count / 2 / data.image_size; // 64 or 32

    // omp_set_nested(1);
    #pragma omp parallel for num_threads(cts_size)
    for (int ct_ind = 0; ct_ind < cts_size; ct_ind++) {
        // cout << "ct_ind: " << ct_ind << endl;
        vector<uint64_t> plain(&input[ct_ind* data.slot_count], &input[(ct_ind + 1) * data.slot_count]);

        int current_col = ct_ind % num_cts_per_res;
        int current_packing = ct_ind / num_cts_per_res;
        
        if (col_packing) {
            #pragma omp parallel for num_threads(64)
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size + current_col * num_col_per_ct;
                int k = row % data.image_size;
                int next_flag = 0;
                if (row >= data.slot_count / 2) {
                    next_flag = data.image_size * data.image_size;
                    j -= data.slot_count / 2 / data.image_size;
                }
                output[k + (k + j) % data.image_size * data.image_size + current_packing * data.image_size * data.image_size * 2 + next_flag] = plain[row];
            }
        }
        else {
            #pragma omp parallel for num_threads(64)
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size + current_col * num_col_per_ct;
                int k = row % data.image_size;
                int next_flag = 0;
                if (row >= data.slot_count / 2) {
                    next_flag = data.image_size * data.image_size;
                    j -= data.slot_count / 2 / data.image_size;
                }
                output[k * data.image_size + (k + j) % data.image_size + current_packing * data.image_size * data.image_size * 2 + next_flag] = plain[row];
            }
        }
    }
}

void Linear_ckks::plain_cross_packing_postprocess_v(
    uint64_t* input, 
    uint64_t * output,
    bool col_packing,
    const FCMetadata &data) {
    int cts_size = PACKING_NUM*data.image_size*OUTPUT_DIM / data.slot_count; 
    int num_V_per_cts = data.slot_count / (data.image_size * data.filter_w);

    omp_set_nested(1);
    #pragma omp parallel for num_threads(cts_size)
    for (int ct_ind = 0; ct_ind < cts_size; ct_ind++) {
        vector<uint64_t> plain(&input[ct_ind* data.slot_count], &input[(ct_ind + 1) * data.slot_count]);

        if (col_packing) {
            #pragma omp parallel for num_threads(128)
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    j += data.filter_w;
                }
                if (num_V_per_cts == 1) {
                    output[k + j * data.image_size + (ct_ind / 2) * data.image_size * data.filter_w * 2 + (ct_ind % 2) * data.image_size * data.filter_w / 2] = plain[row];
                }
                else if (num_V_per_cts == 2) {
                    output[k + j * data.image_size + ct_ind * data.image_size * data.filter_w * 2] = plain[row];
                }
            }
        }
        else {
            #pragma omp parallel for num_threads(128)
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                int next_flag = 0;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    next_flag = data.filter_w * data.image_size;
                }
                if (num_V_per_cts == 1) {
                    output[k * data.filter_w + j + next_flag + (ct_ind / 2) * data.image_size * data.filter_w * 2 + (ct_ind % 2) * data.filter_w / 2] = plain[row];
                }
                else if (num_V_per_cts == 2) {
                    output[k * data.filter_w + j + next_flag + ct_ind * data.image_size * data.filter_w * 2] = plain[row];
                }
            }
        }
    }
}

void Linear_ckks::plain_col_packing_preprocess(
    uint64_t* input, 
    uint64_t * output,
    uint64_t plain_mod,
    int input_dim,
    int common_dim){
    #pragma omp parallel for num_threads(128)
    for (int j = 0; j < common_dim; j++)
            for (int i = 0; i < input_dim; i++)
                output[j*input_dim + i] = input[i*common_dim +j];
}

void Linear_ckks::plain_col_packing_preprocess_vec(
    vector<vector<uint64_t>> input, 
    uint64_t * output,
    uint64_t plain_mod,
    int input_dim,
    int common_dim){
    for (int j = 0; j < common_dim; j++)
            for (int i = 0; i < input_dim; i++)
                output[j*input_dim + i] = input[i][j];
}

void Linear_ckks::plain_col_packing_postprocess(
    uint64_t* input, 
    uint64_t * output,
    bool col_packing,
    const FCMetadata &data){
    for (int i = 0; i < data.image_size * data.filter_w / data.slot_count; i++) {
        vector<uint64_t> plain(&input[i* data.slot_count], &input[(i+1)* data.slot_count]);
        if (col_packing) {
            #pragma omp parallel for num_threads(128)
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    j += data.filter_w / 2;
                }
                output[k + j * data.image_size + i * data.slot_count / 2] = plain[row];
            }
        }
        else {
            #pragma omp parallel for num_threads(128)
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    j += data.filter_w / 2;
                }
                j += i * data.slot_count / data.image_size / 2;
                output[k * data.filter_w + j] = plain[row];
            }
        }
    }
}

vector<vector<uint64_t>> Linear_ckks::concat_vec(
    uint64_t* att,
    int n,
    int dim1,
    int dim2){
    
    vector<vector<uint64_t>> res;
    for(int j = 0; j < dim1; j++){
        vector<uint64_t> row;
        for(int i = 0; i < n; i++){
            row.insert(row.end(), &att[i*dim1*dim2 + j*dim2], &att[i*dim1*dim2 + j*dim2 + dim2]);
        }
        res.push_back(row);
    }
    return res;
}

void Linear_ckks::concat( 
    uint64_t* input,
    uint64_t* output,
    int n,
    int dim1,
    int dim2){

    for(int j = 0; j < dim1; j++){
        for(int i = 0; i < n; i++){
            memcpy(&output[j*n*dim2 + i*dim2], &input[i*dim1*dim2 + j*dim2], dim2*sizeof(uint64_t));
        }
    }
}

// matrix is row-packed with 12 * 128 rows and 128 cols
vector<Ciphertext> Linear_ckks::preprocess_softmax_s1(HE_CKKS* he, uint64_t* matrix, const FCMetadata &data) {
    int num_cts_per_res = data.image_size * data.image_size * 2 / data.slot_count; // 1 or 4
    int num_col_per_ct = data.slot_count / 2 / data.image_size; // 64 or 32

    int total_cts = PACKING_NUM * data.image_size * data.image_size / data.slot_count;
    vector<Ciphertext> enc_softmax(total_cts);

    #pragma omp parallel for num_threads(total_cts)
    for (int ct_ind = 0; ct_ind < total_cts; ct_ind++) {
        int current_col = ct_ind % num_cts_per_res;
        int current_packing = ct_ind / num_cts_per_res;
        vector<double> pod_matrix(data.slot_count);

        for (int row = 0; row < data.slot_count; row++) {
            int j = row / data.image_size + current_col * num_col_per_ct;
            int k = row % data.image_size;
            int next_flag = 0;
            if (row >= data.slot_count / 2) {
                next_flag = data.image_size * data.image_size;
                j -= data.slot_count / 2 / data.image_size;
            }
            pod_matrix[row] = (int64_t)matrix[k * data.image_size + j + current_packing * data.image_size * data.image_size * 2 + next_flag];
        }
        Ciphertext ct;
        Plaintext pt;
        he->encoder->encode(pod_matrix,pow(2, he->he_scale), pt);
        he->encryptor->encrypt(pt, ct);
        enc_softmax[ct_ind] = ct;
    }
    return enc_softmax;
}

void Linear_ckks::client_S1_V_R(HE_CKKS* he, const uint64_t *softmax_s1, uint64_t* V, uint64_t* result, const FCMetadata &data){
    int total_packing = PACKING_NUM;
    for (int packing_num = 0; packing_num < total_packing; packing_num++) {
        #pragma omp parallel for num_threads(64)
        for(int i = 0; i < data.image_size; i++) {
            for(int j = 0; j < data.filter_w; j++) {
                result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] = 0;
                for(int k = 0; k < data.image_size; k++) {
                    result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] += softmax_s1[packing_num * data.image_size * data.image_size + i * data.image_size + k] * V[k + j * data.image_size + data.image_size * data.filter_w * packing_num];
                    result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] = result[packing_num * data.image_size * data.filter_w + i + j * data.image_size];
                }
            }
        }
    }
}

void Linear_ckks::bert_postprocess_V(HE_CKKS* he, uint64_t* input, uint64_t* result, const FCMetadata &data, const bool &col_packing){
    // int total_packing = cts.size() * data.slot_count / (data.image_size * data.filter_w);
    int num_cts_per_mat_V = data.image_size * data.filter_w / data.slot_count;
    for (int packing_num = 0; packing_num < PACKING_NUM; packing_num++) {
        for (int i = 0; i < num_cts_per_mat_V; i++) {
            // vector<uint64_t> plain(data.slot_count, 0ULL);
            // Plaintext pt;
            // he->decryptor->decrypt(cts[i + packing_num * num_cts_per_mat_V], pt);
            // he->encoder->decode(pt, plain);

            int offset = i + packing_num * num_cts_per_mat_V;
            vector<uint64_t> plain(&input[offset* data.slot_count], &input[offset* data.slot_count + data.slot_count]);

            if (col_packing) {
                #pragma omp parallel for num_threads(128)
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    result[k + j * data.image_size + i * data.slot_count / 2 + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
            else {
                #pragma omp parallel for num_threads(128)
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    j += i * data.slot_count / data.image_size / 2;
                    result[k * data.filter_w + j + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
        }
    }
}

void Linear_ckks::bert_postprocess_V_enc(HE_CKKS* he, vector<Ciphertext> cts, uint64_t* result, const FCMetadata &data, const bool &col_packing){
    // int total_packing = cts.size() * data.slot_count / (data.image_size * data.filter_w);
    int num_cts_per_mat_V = data.image_size * data.filter_w / data.slot_count;
    for (int packing_num = 0; packing_num < PACKING_NUM; packing_num++) {
        for (int i = 0; i < num_cts_per_mat_V; i++) {
            vector<double> plain(data.slot_count, 0);
            Plaintext pt;
            he->decryptor->decrypt(cts[i + packing_num * num_cts_per_mat_V], pt);
            he->encoder->decode(pt, plain);

            if (col_packing) {
                #pragma omp parallel for num_threads(128)
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    result[k + j * data.image_size + i * data.slot_count / 2 + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
            else {
                #pragma omp parallel for num_threads(128)
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    j += i * data.slot_count / data.image_size / 2;
                    result[k * data.filter_w + j + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
        }
    }
}

vector<vector<vector<uint64_t>>> softmax_mask_ckks(const FCMetadata &data) {
    vector<vector<vector<uint64_t>>> mask(2, vector<vector<uint64_t>>(data.image_size, vector<uint64_t>(data.image_size)));
    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < data.image_size; i++) {
        vector<uint64_t> mask1(data.image_size, 0ULL);
        vector<uint64_t> mask2(data.image_size, 0ULL);
        for (int j = 0; j < data.image_size - i; j++) {
            mask1[j] = 1;
        }
        for (int j = data.image_size - i; j < data.image_size; j++) {
            mask2[j] = 1;
        }
        mask[0][i] = mask1;
        mask[1][i] = mask2;
    }
    return mask;
}

vector<vector<vector<Plaintext>>> 
Linear_ckks::preprocess_softmax_s2(HE_CKKS* he, const uint64_t *matrix, const FCMetadata &data){
    
    auto mask = softmax_mask_ckks(data);

    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;
    vector<vector<vector<Plaintext>>> s2_pack(PACKING_NUM/2);

    #pragma omp parallel for num_threads(PACKING_NUM/2)
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; packing_ind++) {
        vector<vector<Plaintext>> weightMatrix1(2, vector<Plaintext>(num_diag));
        #pragma omp parallel for
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {
            vector<int64_t> temp2, temp3;
            vector<double> r1(data.image_size), r2(data.image_size), r3(data.image_size), r4(data.image_size);
            for (int j = 0; j < num_diag; j++) {
                temp2.push_back(matrix[((j + diag_ind) % num_diag) + j * data.image_size + packing_ind * 2 * data.image_size * data.image_size]);
                temp3.push_back(matrix[((j + diag_ind) % num_diag) + j * data.image_size + (packing_ind * 2 + 1) * data.image_size * data.image_size]);
            }
            // std::rotate(temp2.begin(), temp2.begin() + temp2.size() - diag_ind, temp2.end());
            std::transform(temp2.begin(), temp2.end(), mask[0][diag_ind].begin(), r1.begin(), std::multiplies<uint64_t>());
            std::transform(temp2.begin(), temp2.end(), mask[1][diag_ind].begin(), r2.begin(), std::multiplies<uint64_t>());
            std::transform(temp3.begin(), temp3.end(), mask[0][diag_ind].begin(), r3.begin(), std::multiplies<uint64_t>());
            std::transform(temp3.begin(), temp3.end(), mask[1][diag_ind].begin(), r4.begin(), std::multiplies<uint64_t>());
            for (int j = 0; j < std::log2(num_diag_per_ct); j++) {
                r1.reserve(r1.size() + distance(r1.begin(), r1.end()));
                r1.insert(r1.end(), r1.begin(), r1.end());
                r2.reserve(r2.size() + distance(r2.begin(), r2.end()));
                r2.insert(r2.end(), r2.begin(), r2.end());
                r3.reserve(r3.size() + distance(r3.begin(), r3.end()));
                r3.insert(r3.end(), r3.begin(), r3.end());
                r4.reserve(r4.size() + distance(r4.begin(), r4.end()));
                r4.insert(r4.end(), r4.begin(), r4.end());
            }
            r1.insert(r1.end(), r3.begin(), r3.end());
            r2.insert(r2.end(), r4.begin(), r4.end());

            Plaintext pt;
            he->encoder->encode(r1, pow(2, he->he_scale), pt);
            weightMatrix1[0][diag_ind] = pt;
            he->encoder->encode(r2, pow(2, he->he_scale), pt);
            weightMatrix1[1][diag_ind] = pt;
        }
        s2_pack[packing_ind] = weightMatrix1;
    }
    return s2_pack;
}

vector<vector<vector<Plaintext>>> Linear_ckks::bert_softmax_v_packing_single_matrix(
    HE_CKKS* he,
    const vector<vector<vector<uint64_t>>> &weights, 
    const FCMetadata &data) 
    {
    vector<vector<vector<Plaintext>>> result(PACKING_NUM/2);
    int num_diag = data.slot_count / data.image_size / 2;

    int n1 = 4;
    int n2 = 8;
    if (data.image_size == 64) {
        n1 = 8;
        n2 = 8;
    }

    int weight_height = data.image_size;
    int num_matrix_per_row = weight_height / num_diag; // 1 or 4
    int num_matrix_per_col = data.filter_w / num_diag; // 1 or 2

    omp_set_nested(1);
    #pragma omp parallel for num_threads(PACKING_NUM/2)
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; packing_ind++) {
        vector<vector<Plaintext>> weightMatrix1(num_matrix_per_col * num_diag);
        for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
            #pragma omp parallel for num_threads(num_diag/2)
            for (int l = 0; l < num_diag; l++) {
                vector<double> temp2, temp3;
                vector<Plaintext> temp_matrix_diag(num_matrix_per_row);
                int matrix_diag_index = 0;
                for (int i = 0; i < num_matrix_per_row; i++) {
                    for (int j = 0; j < num_diag; j++) {
                        for (int k = 0; k < data.image_size; k++) {
                                temp2.push_back((int64_t)weights[packing_ind * 2][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                                temp3.push_back((int64_t)weights[packing_ind * 2 + 1][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        }
                    }
                    std::rotate(temp2.begin(), temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                    std::rotate(temp3.begin(), temp3.begin() + temp3.size() - (l % n1) * data.image_size, temp3.end());
                    temp2.insert(temp2.end(), temp3.begin(), temp3.end());
                    Plaintext pt;
                    he->encoder->encode(temp2, pow(2, he->he_scale), pt);
                    temp_matrix_diag[matrix_diag_index] = pt;
                    matrix_diag_index++;
                    temp2.clear();
                    temp3.clear();
                }
                weightMatrix1[col_ind * num_diag + l] = temp_matrix_diag;
            }
        }
        result[packing_ind] = weightMatrix1;
    }
    return result;
}

vector<vector<vector<Plaintext>>>
Linear_ckks::preprocess_softmax_v_r(HE_CKKS* he, const uint64_t *matrix, const FCMetadata &data) {
    vector<vector<vector<uint64_t>>> weights_r(PACKING_NUM, vector<vector<uint64_t>>(data.image_size, vector<uint64_t>(data.filter_w)));

    for (int packing_ind = 0; packing_ind < PACKING_NUM; packing_ind++) {
        #pragma omp parallel for num_threads(64)
        for (int i = 0; i < data.image_size; i++) {
            for (int j = 0; j < data.filter_w; j++) {
                weights_r[packing_ind][i][j] = matrix[i + j * data.image_size + packing_ind * data.image_size * data.filter_w];
            }
        }
    }
    vector<vector<vector<Plaintext>>> R_pack = bert_softmax_v_packing_single_matrix(he, weights_r, data);
    return R_pack;
}

void Linear_ckks::bert_softmax_V(HE_CKKS* he, vector<Ciphertext> &softmax_s1, vector<Ciphertext> &V, const FCMetadata &data, vector<Ciphertext> &result) {
    // FIXME: pack R according to ours ctxpt
    // FIXME: compute softmax_s1 x R

    // #pragma omp parallel for
    // for (int i = 0; i < V.size(); i++) {
    //     he->evaluator->mod_switch_to_next_inplace(V[i]);
    // }
    int n1 = 4;
    int n2 = 8;
    if (data.image_size == 64) {
        n1 = 8;
        n2 = 8;
    }
    vector<double> vec_w(he->num_slots,0.5);
    Plaintext w;
    he->encoder->encode(vec_w, pow(2, he->he_scale), w);
    #pragma omp parallel for num_threads(PACKING_NUM/2)
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; packing_ind++) {
        int num_diag = data.slot_count / data.image_size / 2;
        int num_matrix_per_row = data.image_size / num_diag; // 1 or 4
        int num_matrix_per_col = data.filter_w / num_diag; // 1 or 2

        vector<vector<Ciphertext>> rotatedIR(num_matrix_per_row);

        #pragma omp parallel for
        for (int i = 0; i < num_matrix_per_row; i++) {
            vector<Ciphertext> tmp;
            tmp.push_back(softmax_s1[packing_ind * num_matrix_per_row + i]);

            for (int j = 1; j < n1; j++) {
                Ciphertext temp_rot;
                he->evaluator->rotate_rows(softmax_s1[packing_ind * num_matrix_per_row + i], (num_diag - j) * data.image_size, *(he->gal_keys), temp_rot);
                tmp.push_back(temp_rot);
            }

            rotatedIR[i] = tmp;
            tmp.clear();
        }

        //compute matrix multiplication
        vector<vector<Ciphertext>> temp_results(num_matrix_per_col, vector<Ciphertext>(n2));
        vector<vector<Ciphertext>> temp_results1(num_matrix_per_col, vector<Ciphertext>(n2 * num_matrix_per_row));

        #pragma omp parallel for
        for (int k = 0; k < num_matrix_per_row * n2; k++) {
            int j = k / num_matrix_per_row;
            int ct_i = k % num_matrix_per_row;
            for (int l = 0; l < num_matrix_per_col; l++) {
                for (int i = 0; i < n1; i++) {
                    Ciphertext ct1_l;
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], w, ct1_l);
                    if (i == 0)
                        temp_results1[l][k] = ct1_l;
                    else {
                        he->evaluator->add_inplace(temp_results1[l][k], ct1_l);
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int j = 0; j < n2; j++) {
            for (int ct_i = 0; ct_i < num_matrix_per_row; ct_i++) {
                for (int l = 0; l < num_matrix_per_col; l++) {
                    if (ct_i == 0)
                        temp_results[l][j] = temp_results1[l][j * num_matrix_per_row + ct_i];
                    else
                        he->evaluator->add_inplace(temp_results[l][j], temp_results1[l][j * num_matrix_per_row + ct_i]);
                }
            }
        }

        #pragma omp parallel for
        for (int l = 0; l < num_matrix_per_col; l++) {
            Ciphertext ct;
            for (int k = 0; k < n2; k++) {
                if (k == 0)
                    ct = temp_results[l][0];
                else {
                    Ciphertext temp_rot_ct;
                    he->evaluator->rotate_rows(temp_results[l][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct, temp_rot_ct);
                }
            }
            result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + l] = ct;
        }

        // FIXME: pack softmax_s2 according to gazelle
        // FIXME: compute softmax_s2 x V

        num_diag = data.image_size;
        for (int ct_ind = 0; ct_ind < num_matrix_per_col; ct_ind++) {
            vector<Ciphertext> rotation_results(num_diag);

            #pragma omp parallel for
            for (int i = 0; i < num_diag; i++) {
                Ciphertext temp1;
                Ciphertext temp2;
                vector<Ciphertext> temp_mult = rotation_by_one_depth3(he, data, V[packing_ind * num_matrix_per_col + ct_ind], i);
                he->evaluator->multiply_plain(temp_mult[0], w, temp1);
                he->evaluator->multiply_plain(temp_mult[1], w, temp2);
                he->evaluator->add(temp1, temp2, rotation_results[i]);
            }
            for (int i = 0; i < num_diag; i++) {
                he->evaluator->add_inplace(result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + ct_ind], rotation_results[i]);
            }
            rotation_results.clear();
        }
    }

    parms_id_type parms_id = result[0].parms_id();
    shared_ptr<const SEALContext::ContextData> context_data = he->context->get_context_data(parms_id);

    #pragma omp parallel for
    for (int i = 0; i < result.size(); i++) {
        flood_ciphertext(result[i], context_data, SMUDGING_BITLEN_bert1);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
    }
}

vector<Ciphertext> Linear_ckks::w_ln(HE_CKKS* he, vector<Ciphertext> ln){
    int cts_size = ln.size();
    vector<Ciphertext> result(cts_size);
    vector<double> vec_w(he->num_slots,0.5);
    Plaintext w1;
    he->encoder->encode(vec_w, pow(2, he->he_scale), w1);
    #pragma omp parallel for
    for (int i = 0; i < cts_size; i++) {
        he->evaluator->multiply_plain(ln[i], w1, result[i]);
    }

    parms_id_type parms_id = result[0].parms_id();
    shared_ptr<const SEALContext::ContextData> context_data = he->context->get_context_data(parms_id);

    #pragma omp parallel for
    for (int i = 0; i < cts_size; i++) {
        flood_ciphertext(result[i], context_data, SMUDGING_BITLEN_bert3);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
    }

    return result;
}


void Linear_ckks::softmax_v(
    HE_CKKS* he,
    const vector<vector<vector<UnifiedCiphertext>>> &softmax_s2, 
    const vector<UnifiedCiphertext> &V, 
    const FCMetadata &data, 
    vector<UnifiedCiphertext> &result,
    double output_mask_value,
    const vector<bool> *active_heads,
    bool post_av_rescale){
    int rot_ours;
    if(data.image_size == INPUT_DIM){
        if (COMMON_DIM == 1024) {
            // Table 5/7 reports 1056 rotations for all BERT-large heads.
            // One rotation_by_one_depth3 call emits two rotations and the
            // loop covers PACKING_NUM/2 packed groups.
            rot_ours = 1056 / PACKING_NUM;
        } else {
            rot_ours = 3*24;
        }
    }
    else if (data.image_size == 64){
        rot_ours = 3*16;
    }
    else{
        // Preserve the measured BLB ViT schedule exactly: the rotation budget
        // is shared by all V feature ciphertexts in one packed head pair.
        // heterogeneous may skip inactive pairs, but must not expand the baseline HE
        // graph by resetting this budget for every feature ciphertext.
        rot_ours = data.image_size;
    }
    if (active_heads != nullptr &&
        active_heads->size() != static_cast<size_t>(PACKING_NUM)) {
        throw runtime_error("Softmax*V active-head mask has invalid size");
    }
    auto pair_is_active = [&](int pair) {
        return active_heads == nullptr || (*active_heads)[2 * pair] ||
               (*active_heads)[2 * pair + 1];
    };
    int active_pairs = 0;
    int first_active_pair = -1;
    for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
        if (pair_is_active(pair)) {
            if (first_active_pair < 0) first_active_pair = pair;
            ++active_pairs;
        }
    }
    if (active_pairs == 0)
        throw runtime_error("Softmax*V has no active packed head pair");
    const int result_per_pair =
        data.image_size * data.filter_w * 2 / data.slot_count;
    const int reference_v_index = first_active_pair * result_per_pair;
    const auto loc = V[reference_v_index].location();
    // encode weight once
    UnifiedPlaintext mask(loc);
    if (!std::isfinite(output_mask_value) || output_mask_value <= 0.0) {
        throw std::runtime_error("Softmax*V output mask value must be positive");
    }
    vector<double> vec_mask(data.slot_count, output_mask_value);
    // start computation
    if (loc == LOCATION::HOST) {
    // ******************** CPU ********************
    he->encoder->encode(vec_mask,
                        V[reference_v_index].hcipher().parms_id(),
                        pow(2,he->he_scale), mask);
    int num_diag = data.slot_count / data.image_size / 2;
    int num_matrix_per_col = data.filter_w / num_diag; // 1 or 2
    // cout << "num_diag, num_matrix_per_col: " << num_diag << " " << num_matrix_per_col << endl;
    // auto start = chrono::high_resolution_clock::now();
    
    #pragma omp parallel for num_threads(PACKING_NUM/2)
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; packing_ind++) {
        if (!pair_is_active(packing_ind)) continue;

        // cout << "packing_ind: " << packing_ind << endl;
        // FIXME: pack softmax_s2 according to gazelle
        // FIXME: compute softmax_s2 x V

        num_diag = data.image_size;
        for (int ct_ind = 0; ct_ind < num_matrix_per_col; ct_ind++) {
            vector<Ciphertext> rotation_results(num_diag);
            #pragma omp parallel for num_threads(32)
            for (int i = 0; i < num_diag; i++) {
                // one rotation_by_one_depth3 contains 2 rotations
                vector<Ciphertext> temp_mult(2);
                // The rotation budget applies independently to every V
                // feature ciphertext.  Reusing the counter across ct_ind
                // silently skipped all rotations for the second feature
                // block (for ViT, ct_ind=1), while the corresponding
                // diagonal weights still expected rotated V slots.
                if(i < rot_ours){
                    temp_mult = rotation_by_one_depth3(he, data, V[packing_ind * num_matrix_per_col + ct_ind], i);
                }
                else{
                    temp_mult[0]=(V[packing_ind * num_matrix_per_col + ct_ind]);
                    temp_mult[1]=(V[packing_ind * num_matrix_per_col + ct_ind]);
                }
                he->evaluator->multiply_inplace(temp_mult[0], softmax_s2[packing_ind][0][i]);
                he->evaluator->multiply_inplace(temp_mult[1], softmax_s2[packing_ind][1][i]);
                he->evaluator->add(temp_mult[0], temp_mult[1], rotation_results[i]);
                // he->evaluator->relinearize_inplace(rotation_results[i], *(he->relin_keys));
            }
            result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + ct_ind] = rotation_results[0];
            #pragma omp parallel for
            for (int i = 1; i < num_diag; i++) {
                he->evaluator->add_inplace(result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + ct_ind], rotation_results[i]);
            }
            rotation_results.clear();
        }
    }
    // cout << "softmax_v part1 time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    #pragma omp parallel for
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; ++packing_ind) {
      if (!pair_is_active(packing_ind)) continue;
      const int num_matrix_per_col =
          data.filter_w / (data.slot_count / data.image_size / 2);
      for (int ct_ind = 0; ct_ind < num_matrix_per_col; ++ct_ind) {
        const int i = packing_ind * num_matrix_per_col + ct_ind;
        he->evaluator->relinearize_inplace(result[i], *(he->relin_keys));
        if (post_av_rescale) {
          // Preserve the repository's post-MatMul scale-alignment depth.
          he->evaluator->multiply_plain_inplace(result[i], mask);
          ++rescale_count;
          he->evaluator->rescale_to_next_inplace(result[i]);
        }
      }
        // he->evaluator->mod_switch_to_next_inplace(result[i]);
        // he->evaluator->mod_switch_to_next_inplace(result[i]);
    }
    } else {
    // ******************** GPU ********************
    he->dencoder->encode(*he->dcontext, vec_mask, pow(2,he->he_scale),
                         mask,
                         V[reference_v_index].dcipher().chain_index());
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; packing_ind++) {
        if (!pair_is_active(packing_ind)) continue;
        int num_diag = data.slot_count / data.image_size / 2;
        int num_matrix_per_col = data.filter_w / num_diag; // 1 or 2

        // FIXME: pack softmax_s2 according to gazelle
        // FIXME: compute softmax_s2 x V

        num_diag = data.image_size;
        for (int ct_ind = 0; ct_ind < num_matrix_per_col; ct_ind++) {
            vector<UnifiedCiphertext> rotation_results(
                num_diag,
                UnifiedCiphertext(loc));
            for (int i = 0; i < num_diag; i++) {
                UnifiedCiphertext temp1(loc);
                UnifiedCiphertext temp2(loc);
                // See the host path above: each feature ciphertext needs its
                // own complete rotation schedule; a shared counter corrupts
                // the later feature blocks deterministically.
                if(i < rot_ours){
                    UnifiedCiphertext ct1(loc);
                    UnifiedCiphertext ct2(loc);
                    he->devaluator->rotate_vector(V[packing_ind * num_matrix_per_col + ct_ind], i, *(he->dgal_keys), ct1);
                    he->devaluator->rotate_vector(V[packing_ind * num_matrix_per_col + ct_ind], -(data.image_size - i), *(he->dgal_keys), ct2);
                    he->devaluator->multiply(ct1, softmax_s2[packing_ind][0][i], temp1);
                    he->devaluator->multiply(ct2, softmax_s2[packing_ind][1][i], temp2);
                } else {
                    he->devaluator->multiply(V[packing_ind * num_matrix_per_col + ct_ind], softmax_s2[packing_ind][0][i], temp1);
                    he->devaluator->multiply(V[packing_ind * num_matrix_per_col + ct_ind], softmax_s2[packing_ind][1][i], temp2);
                }
                he->devaluator->add(temp1, temp2, rotation_results[i]);
            }
            result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + ct_ind] = std::move(rotation_results[0]);
            for (int i = 1; i < num_diag; i++) {
                he->devaluator->add_inplace(result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + ct_ind], rotation_results[i]);
            }
        }
    }

    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; ++packing_ind) {
      if (!pair_is_active(packing_ind)) continue;
      const int num_matrix_per_col =
          data.filter_w / (data.slot_count / data.image_size / 2);
      for (int ct_ind = 0; ct_ind < num_matrix_per_col; ++ct_ind) {
        const int i = packing_ind * num_matrix_per_col + ct_ind;
        he->devaluator->relinearize_inplace(result[i], *(he->drelin_keys));
        if (post_av_rescale) {
          he->devaluator->multiply_plain_inplace(result[i], mask);
          ++rescale_count;
          he->devaluator->rescale_to_next_inplace(result[i]);
        }
      }
    }  
    }
    // BSGS2 consumes a fixed twelve-ciphertext AV layout.  Materialize a
    // public encrypted zero only for packed pairs skipped above; no inactive
    // head participates in a ct-ct multiplication.
    int reference_index = -1;
    for (int pair = 0; pair < PACKING_NUM / 2 && reference_index < 0; ++pair)
        if (pair_is_active(pair)) reference_index = pair * result_per_pair;
    if (reference_index < 0)
        throw runtime_error("Softmax*V active result reference is missing");
    UnifiedPlaintext zero_plain(loc);
    vector<double> public_zero(data.slot_count, 0.0);
    if (loc == LOCATION::HOST) {
        he->encoder->encode(public_zero,
                            result[reference_index].hcipher().parms_id(),
                            1.0, zero_plain);
    } else {
        he->dencoder->encode(
            *he->dcontext, public_zero, 1.0, zero_plain,
            result[reference_index].dcipher().chain_index());
    }
    for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
        if (pair_is_active(pair)) continue;
        for (int local = 0; local < result_per_pair; ++local) {
            const int index = pair * result_per_pair + local;
            result[index] = result[reference_index];
            if (loc == LOCATION::HOST)
                he->evaluator->multiply_plain_inplace(result[index],
                                                       zero_plain);
            else
                he->devaluator->multiply_plain_inplace(result[index],
                                                        zero_plain);
            ++ct_pt_multiply_count;
        }
    }
    if (COMMON_DIM == 1024) {
        const int num_matrix_per_col =
            data.filter_w / (data.slot_count / data.image_size / 2);
        softmax_v_rotation_count += static_cast<uint64_t>(2 * active_pairs) *
            num_matrix_per_col * std::min(rot_ours, data.image_size);
    }
    const int num_matrix_per_col =
        data.filter_w / (data.slot_count / data.image_size / 2);
    const uint64_t av_diagonals =
        static_cast<uint64_t>(active_pairs) * num_matrix_per_col *
        data.image_size;
    ct_ct_multiply_count += 2 * av_diagonals;
    if (post_av_rescale)
        ct_pt_multiply_count += active_pairs * result_per_pair;
    const uint64_t av_rotations =
        2 * static_cast<uint64_t>(active_pairs) * num_matrix_per_col *
        std::min(rot_ours, data.image_size);
    rotation_count += av_rotations;
    if (COMMON_DIM != 1024)
        softmax_v_rotation_count += av_rotations;
    cout << "OURS_SELECTIVE_AV status=PASS active_heads="
         << (active_heads == nullptr
                 ? PACKING_NUM
                 : count(active_heads->begin(), active_heads->end(), true))
         << " active_pairs=" << active_pairs
         << " total_pairs=" << PACKING_NUM / 2
         << " post_av_rescale=" << (post_av_rescale ? 1 : 0)
         << " inactive_pair_ctct=0" << endl;
}

void Linear_ckks::preprocess_softmax(const uint64_t *input, uint64_t *output, const FCMetadata &data){
    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;
    // vector<vector<uint64_t>> result_uint(data.image_size * data.image_size * 12 / data.slot_count, vector<uint64_t>(data.slot_count));
    #pragma omp parallel for num_threads(PACKING_NUM/2)
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; packing_ind++) {
        vector<uint64_t> temp2, temp3;
        // #pragma omp parallel for num_threads(32)
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {
            for (int j = 0; j < num_diag; j++) {
                temp2.push_back(input[((j + diag_ind) % num_diag) + j * data.image_size + packing_ind * 2 * data.image_size * data.image_size]);
                temp3.push_back(input[((j + diag_ind) % num_diag) + j * data.image_size + (packing_ind * 2 + 1) * data.image_size * data.image_size]);
            }
            if (temp2.size() == data.slot_count / 2) {
                temp2.insert(temp2.end(), temp3.begin(), temp3.end());
                // result_uint[packing_ind * data.image_size * data.image_size * 2 / data.slot_count + diag_ind / num_diag_per_ct] = temp2;

                int offset = packing_ind * data.image_size * data.image_size * 2 / data.slot_count + diag_ind / num_diag_per_ct;
                memcpy(&output[offset*data.slot_count], temp2.data(), data.slot_count*sizeof(uint64_t));
                temp2.clear();
                temp3.clear();
            }
        }
    }
}

vector<vector<Plaintext>> Linear_ckks::softmax_mask_ct_ct(HE_CKKS* he, const FCMetadata &data){
    vector<vector<Plaintext>> mask(2, vector<Plaintext>(data.image_size));
    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;

    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < data.image_size; i++) {
        vector<int64_t> mask1(data.image_size, 0);
        vector<int64_t> mask2(data.image_size, 0);
        for (int j = 0; j < data.image_size - i; j++) {
            mask1[j] = 1;
        }
        for (int j = data.image_size - i; j < data.image_size; j++) {
            mask2[j] = 1;
        }
        vector<double> m1(data.slot_count, 0), m2(data.slot_count, 0);
        int start_ind = (i % num_diag_per_ct) * num_diag;
        for (int j = start_ind; j < num_diag + start_ind; j++) {
            m1[j] = mask1[j - start_ind];
            m1[j + data.slot_count / 2] = mask1[j - start_ind];
            m2[j] = mask2[j - start_ind];
            m2[j + data.slot_count / 2] = mask2[j - start_ind];
        }
        Plaintext pt;
        he->encoder->encode(m1, pow(2,he->he_scale), pt);
        mask[0][i] = pt;
        he->encoder->encode(m2, pow(2,he->he_scale), pt);
        mask[1][i] = pt;
    }
    return mask;
}

void Linear_ckks::preprocess_softmax_s1_ct_ct(
    HE_CKKS* he, 
    const vector<UnifiedCiphertext> &matrix, 
    const FCMetadata &data,
    vector<vector<vector<UnifiedCiphertext>>> &s2_pack,
    const vector<bool> *active_heads) {
    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;
    if (matrix.empty() || num_diag <= 0 || num_diag_per_ct <= 0 ||
        data.slot_count % (2 * data.image_size) != 0 ||
        (num_diag_per_ct & (num_diag_per_ct - 1)) != 0) {
        throw std::runtime_error(
            "Softmax*V diagonal preprocessing received invalid metadata");
    }
    const int ciphertexts_per_pair =
        data.image_size * data.image_size * 2 / data.slot_count;
    if (ciphertexts_per_pair <= 0 ||
        matrix.size() != static_cast<size_t>(PACKING_NUM / 2 *
                                             ciphertexts_per_pair)) {
        throw std::runtime_error(
            "Softmax*V diagonal preprocessing ciphertext count mismatch");
    }
    if (active_heads != nullptr &&
        active_heads->size() != static_cast<size_t>(PACKING_NUM)) {
        throw runtime_error("Softmax*V active-head mask has invalid size");
    }
    auto pair_is_active = [&](int pair) {
        return active_heads == nullptr || (*active_heads)[2 * pair] ||
               (*active_heads)[2 * pair + 1];
    };
    int reference_pair = -1;
    for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
        if (pair_is_active(pair)) {
            reference_pair = pair;
            break;
        }
    }
    if (reference_pair < 0)
        throw runtime_error("Softmax*V has no active packed head pair");
    const auto &matrix_reference =
        matrix[reference_pair * ciphertexts_per_pair];
    const auto loc = matrix_reference.location();
    s2_pack.resize(
        PACKING_NUM/2,
        vector<vector<UnifiedCiphertext>>(
            2, vector<UnifiedCiphertext>(
                num_diag, UnifiedCiphertext(loc))
    ));
    // Each imported Softmax ciphertext holds num_diag_per_ct diagonals in
    // token-sized blocks.  Select one block, split the ordinary and wrapping
    // token rotations, move that block to the first block of each packed
    // head, then replicate it over all feature blocks.  The previous source
    // copied the whole ciphertext for every diagonal: its mask/multiply and
    // rotation loops were commented out, which mixed all diagonals in AV.
    vector<UnifiedPlaintext> forward_masks;
    vector<UnifiedPlaintext> wrap_masks;
    forward_masks.reserve(num_diag);
    wrap_masks.reserve(num_diag);
    // Multiplication by this public 0/1 selector is followed immediately by
    // rescale.  Encode at the exact prime that rescale drops so the selected
    // Softmax value keeps its incoming scale.  Encoding at 2^he_scale (the
    // old placeholder) lost roughly 16 precision bits with ViT's 60-bit
    // level and made the long AV rotation/sum numerically unstable.
    const auto mask_context =
        host_context_data_for(*he, matrix_reference, "Softmax*V mask");
    const auto &mask_moduli = mask_context->parms().coeff_modulus();
    if (mask_moduli.size() < 2)
        throw std::runtime_error(
            "Softmax*V mask preprocessing has no rescale level");
    const double mask_scale =
        static_cast<double>(mask_moduli.back().value());
    for (int diag_ind = 0; diag_ind < num_diag; ++diag_ind) {
        vector<double> forward(data.slot_count, 0.0);
        vector<double> wrap(data.slot_count, 0.0);
        const int block = diag_ind % num_diag_per_ct;
        const int block_start = block * num_diag;
        for (int token = 0; token < num_diag; ++token) {
            const double forward_value = token < num_diag - diag_ind ? 1.0 : 0.0;
            const double wrap_value = token >= num_diag - diag_ind ? 1.0 : 0.0;
            forward[block_start + token] = forward_value;
            wrap[block_start + token] = wrap_value;
            forward[data.slot_count / 2 + block_start + token] =
                forward_value;
            wrap[data.slot_count / 2 + block_start + token] = wrap_value;
        }
        forward_masks.emplace_back(loc);
        wrap_masks.emplace_back(loc);
        if (loc == LOCATION::HOST) {
            he->encoder->encode(
                forward, matrix_reference.hcipher().parms_id(), mask_scale,
                forward_masks.back().hplain());
            he->encoder->encode(
                wrap, matrix_reference.hcipher().parms_id(), mask_scale,
                wrap_masks.back().hplain());
        } else {
            he->dencoder->encode(
                *he->dcontext, forward, mask_scale,
                forward_masks.back().dplain(),
                matrix_reference.dcipher().chain_index());
            he->dencoder->encode(
                *he->dcontext, wrap, mask_scale,
                wrap_masks.back().dplain(),
                matrix_reference.dcipher().chain_index());
        }
    }
    if (loc == LOCATION::HOST) {
    #pragma omp parallel for num_threads(PACKING_NUM/2)
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; packing_ind++) {
        if (!pair_is_active(packing_ind)) continue;
        auto &weightMatrix1 = s2_pack[packing_ind];
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {
            const int cts_ind = packing_ind * ciphertexts_per_pair +
                                diag_ind / num_diag_per_ct;
            Ciphertext cur_ct_l, cur_ct_r;
            he->evaluator->multiply_plain(
                matrix[cts_ind], forward_masks[diag_ind].hplain(), cur_ct_l);
            he->evaluator->multiply_plain(
                matrix[cts_ind], wrap_masks[diag_ind].hplain(), cur_ct_r);
            he->evaluator->rescale_to_next_inplace(cur_ct_l);
            he->evaluator->rescale_to_next_inplace(cur_ct_r);
            const int block = diag_ind % num_diag_per_ct;

            for (int stride = 1; stride < num_diag_per_ct; stride <<= 1) {
                Ciphertext replicated;
                he->evaluator->rotate_vector(
                    cur_ct_l, ours_vit::packed_block_rotation(
                        block, stride, num_diag, true), *(he->gal_keys), replicated);
                he->evaluator->add_inplace(cur_ct_l, replicated);
                he->evaluator->rotate_vector(
                    cur_ct_r, ours_vit::packed_block_rotation(
                        block, stride, num_diag, true), *(he->gal_keys), replicated);
                he->evaluator->add_inplace(cur_ct_r, replicated);
            }
            weightMatrix1[0][diag_ind] = std::move(cur_ct_l);
            weightMatrix1[1][diag_ind] = std::move(cur_ct_r);
        }
    }
    } else {
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; packing_ind++) {
        if (!pair_is_active(packing_ind)) continue;
        auto &weightMatrix1 = s2_pack[packing_ind];
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {
            const int cts_ind = packing_ind * ciphertexts_per_pair +
                                diag_ind / num_diag_per_ct;
            UnifiedCiphertext cur_ct_l(loc), cur_ct_r(loc);
            he->devaluator->multiply_plain(
                matrix[cts_ind], forward_masks[diag_ind].dplain(), cur_ct_l);
            he->devaluator->multiply_plain(
                matrix[cts_ind], wrap_masks[diag_ind].dplain(), cur_ct_r);
            he->devaluator->rescale_to_next_inplace(cur_ct_l);
            he->devaluator->rescale_to_next_inplace(cur_ct_r);
            const int block = diag_ind % num_diag_per_ct;

            for (int stride = 1; stride < num_diag_per_ct; stride <<= 1) {
                UnifiedCiphertext replicated(loc);
                he->devaluator->rotate_vector(
                    cur_ct_l, ours_vit::packed_block_rotation(
                        block, stride, num_diag, true), *(he->dgal_keys), replicated);
                he->devaluator->add_inplace(cur_ct_l, replicated);
                he->devaluator->rotate_vector(
                    cur_ct_r, ours_vit::packed_block_rotation(
                        block, stride, num_diag, true), *(he->dgal_keys), replicated);
                he->devaluator->add_inplace(cur_ct_r, replicated);
            }
            weightMatrix1[0][diag_ind] = std::move(cur_ct_l);
            weightMatrix1[1][diag_ind] = std::move(cur_ct_r);
        }
    }
    }
    uint64_t active_pairs = 0;
    for (int pair = 0; pair < PACKING_NUM / 2; ++pair)
        active_pairs += pair_is_active(pair) ? 1 : 0;
    const uint64_t selected_diagonals = active_pairs * num_diag;
    const uint64_t replication_depth = static_cast<uint64_t>(
        std::ceil(std::log2(num_diag_per_ct)));
    ct_pt_multiply_count += 2 * selected_diagonals;
    rescale_count += 2 * selected_diagonals;
    rotation_count += 2 * selected_diagonals * replication_depth;
    cout << "FORMAL_AV_DIAGONAL_PACK status=PASS"
         << " alignment_rotations=0 block_rotation=direct_target_v1"
         << " softmax_cts=" << matrix.size()
         << " active_pairs=" << active_pairs
         << " packed_diagonals=" << selected_diagonals
         << " feature_blocks=" << num_diag_per_ct
         << " mask_scale_bits=" << std::log2(mask_scale)
         << " output_scale_bits="
         << std::log2(s2_pack[reference_pair].front().front().scale())
         << " mask_multiplies=" << 2 * selected_diagonals
         << " rescales=" << 2 * selected_diagonals
         << endl;
}

Linear_ckks::Linear_ckks(){

}

Linear_ckks::Linear_ckks(int party, sci::NetIO *io, LOCATION backend,
                         bool enable_native_linear) {
	this->party = party;
	this->io = io;
	this->native_linear_enabled = enable_native_linear;

    if (backend != LOCATION::HOST && backend != LOCATION::DEVICE) {
        throw std::invalid_argument("Bert_ckks: backend must be HOST or DEVICE");
    }

    set_HE(backend, enable_native_linear);
    pp_1.resize(ATTENTION_LAYERS);
    pp_2.resize(ATTENTION_LAYERS);
    pp_2_mpc.resize(ATTENTION_LAYERS);
    pp_2_native.resize(ATTENTION_LAYERS);
    pp_3.resize(ATTENTION_LAYERS);
    pp_4.resize(ATTENTION_LAYERS);
    heterogeneous_protocols.resize(ATTENTION_LAYERS);
    heterogeneous_alphas.resize(ATTENTION_LAYERS);

    data_lin1_0.filter_h = COMMON_DIM;
    data_lin1_0.filter_w = OUTPUT_DIM;
    data_lin1_0.image_size = INPUT_DIM;
    data_lin1_0.slot_count = he1->num_slots;

    int input_dim = INPUT_DIM;

    data_lin1_1.filter_h = COMMON_DIM;
    data_lin1_1.filter_w = OUTPUT_DIM;
    data_lin1_1.image_size = input_dim;
    data_lin1_1.slot_count = he1->num_slots;

    data_lin2.filter_h = COMMON_DIM;
    data_lin2.filter_w = COMMON_DIM;
    data_lin2.image_size = input_dim;
    data_lin2.slot_count = he2->num_slots;

    data_lin3.filter_h = COMMON_DIM;
    data_lin3.filter_w = INTER_DIM;
    data_lin3.image_size = input_dim;
    data_lin3.slot_count = he3->num_slots;

    data_lin4.filter_h = INTER_DIM;
    data_lin4.filter_w = COMMON_DIM;
    data_lin4.image_size = input_dim;
    data_lin4.slot_count = he4->num_slots;
    
    // exit(0);
}

void Linear_ckks::configure_sequence_length(int tokens) {
    if (tokens <= 0) {
        throw invalid_argument("BLB sequence length must be positive");
    }
    if (he1 == nullptr || he2 == nullptr || he3 == nullptr || he4 == nullptr) {
        throw logic_error("BLB sequence length configured before HE setup");
    }
    auto require_slot_alignment = [tokens](const HE_CKKS *he, int features,
                                             const char *stage) {
        const long long elements = static_cast<long long>(tokens) * features;
        if (elements % he->num_slots != 0) {
            throw invalid_argument(string("BLB original packing cannot represent ") +
                                   stage + " at sequence length " +
                                   to_string(tokens) + "; elements=" +
                                   to_string(elements) + " slots=" +
                                   to_string(he->num_slots));
        }
    };
    // The original BSGS layout is a full-slot layout.  ViT-B/16 therefore
    // uses 256 physical rows (197 logical rows plus public zero padding).
    require_slot_alignment(he1, COMMON_DIM, "QKV input");
    require_slot_alignment(he2, COMMON_DIM, "attention output");
    require_slot_alignment(he3, COMMON_DIM, "FFN1 input");
    require_slot_alignment(he4, INTER_DIM, "FFN2 input");

    sequence_length = tokens;
    data_lin1_0.image_size = tokens;
    data_lin1_1.image_size = tokens;
    data_lin2.image_size = tokens;
    data_lin3.image_size = tokens;
    data_lin4.image_size = tokens;
}

void Linear_ckks::set_HE(LOCATION backend, bool enable_native_linear){
    cout << "he0 ok" << endl;
	// Table 6, Block 2: the 35-bit middle RNS prime is part of the
	// published depth-7 chain.  The old public source used 34 bits and one
	// fewer level, which is a different CKKS circuit.
	const vector<int> he1_modulus = enable_native_linear
	    ? vector<int>{60, 35, 60, 60, 60, 60, 60, 60, 60}
	    : vector<int>{60, 35, 60, 60, 60, 60, 60, 60};
	this->he1 = new HE_CKKS(
        party, /* Bob or Alice (Client or Server) */
        io, /* Network */
        32768, /* N */
        he1_modulus,
        40, /* fft_scale */
        44, /* he_scale */
        backend /* Evaluator backend */
    );
    cout << "OURS_NATIVE_HE1_CHAIN status=PASS enabled="
         << (enable_native_linear ? 1 : 0)
         << " data_level_extension="
         << (enable_native_linear ? 1 : 0)
         << " modulus_entries=" << he1_modulus.size()
         << " baseline_entries=8 reason=native_QK_AV_WO" << endl;
    cout << "he1 ok" << endl;
    // One additional data level is required by restored WO -> LN statistics.
    // N=32768, total modulus=480 bits; SEAL tc128 remains mandatory.
    this->he2 = new HE_CKKS(
        party, /* Bob or Alice (Client or Server) */
        io, /* Network */
        32768, /* N */
        std::vector<int>{60, 60, 60, 60, 60, 60, 60, 60}, /* modulus chain */
        40, /* fft_scale */
        43, /* he_scale */
        backend /* Evaluator backend */
    );
    cout << "he2 ok" << endl;
    this->he3 = new HE_CKKS(
        party, /* Bob or Alice (Client or Server) */
        io, /* Network */
        32768, /* N */
        std::vector<int>{60, 32, 60, 60, 60, 60, 60, 60}, /* modulus chain */
        40, /* fft_scale */
        46, /* he_scale */
        backend /* Evaluator backend */
    );
    cout << "he3 ok" << endl;
	// Table 6, Block 1 publishes five computational 60-bit RNS primes.
	// Phantom/SEAL reserves one additional 60-bit special prime for
	// relinearization/key-switching; it is not part of the paper's
	// ciphertext-RNS count.  Passing only the five published entries here
	// leaves the GPU variance square without the two published rescale levels.
	this->he4 = new HE_CKKS(
        party, /* Bob or Alice (Client or Server) */
        io, /* Network */
        16384, /* N */
        std::vector<int>{60, 60, 60, 60, 60, 60}, /* 1 special + 5 Table 6 data primes */
        40, /* fft_scale */
        38, /* he_scale */
        backend /* Evaluator backend */
    );
    this->he_softmax = new HE_CKKS(
        party, /* Bob or Alice (Client or Server) */
        io, /* Network */
        32768, /* N */
        std::vector<int>{60, 40, 40, 40, 40, 40, 40, 40, 60}, /* modulus chain */
        40, /* fft_scale */
        40, /* he_scale */
        backend /* Evaluator backend */
    );
    this->he_onetime = new HE_CKKS(
        party, /* Bob or Alice (Client or Server) */
        io, /* Network */
        8192, /* N */
        std::vector<int>{60, 40, 60}, /* modulus chain */
        40, /* fft_scale */
        40, /* he_scale */
        backend /* Evaluator backend */
    );
    cout << "he initial ok" << endl;
}
