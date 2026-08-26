#include "linear_ckks.h"
#include "ckks_plain_cache.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace {

bool ckks_scales_close(double lhs, double rhs,
                       double relative_tolerance = 0.0) {
    const double magnitude = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= relative_tolerance * magnitude;
}

void require_device_plain_compatible(
    const UnifiedCiphertext &cipher, const PhantomPlaintext &plain,
    CkksCtPtOperation operation, const std::string &label) {
#ifdef BLB_HAS_HE_GPU
    assert_ckks_device_plain_compatible(cipher, plain, operation, label);
#else
    (void)cipher;
    (void)plain;
    (void)operation;
    throw std::runtime_error(label +
                             ": DEVICE backend was not compiled with USE_HE_GPU=ON");
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
        const auto parms_id = seal_parms_id_for(
            *he.context, *he.dcontext, cipher.dcipher().chain_index());
        auto data = he.context->get_context_data(parms_id);
        if (!data) {
            throw std::runtime_error(std::string(label) +
                                     ": device chain has no SEAL data-level mirror");
        }
        return data;
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
    // The checkpoint loader stores signed fixed-point integers in uint64_t
    // containers. Preserve the bit pattern instead of relying on an
    // implementation-defined unsigned-to-signed conversion.
    int64_t signed_value = 0;
    static_assert(sizeof(signed_value) == sizeof(raw),
                  "BLB checkpoint word width must remain 64 bits");
    std::memcpy(&signed_value, &raw, sizeof(raw));
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
            he.evaluator->mod_switch_to_inplace(plain, target);
            return;
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
    if (params.wq_pack.empty() || params.wq_pack.front().first.empty() ||
        params.wq_pack.front().first.front().empty()) {
        throw std::runtime_error("QKV: weight preprocessing is empty");
    }
    auto input_context = host_context_data_for(he, inputs.front(), "QKV");
    auto output_context = input_context->next_context_data();
    if (!output_context) {
        throw std::runtime_error("QKV: input level has no rescale target");
    }
    const auto &dropped_moduli = input_context->parms().coeff_modulus();
    if (dropped_moduli.empty()) {
        throw std::runtime_error("QKV: input level has no coefficient modulus");
    }
    const double weight_scale =
        params.wq_pack.front().first.front().front().scale();
    const double bias_scale = inputs.front().scale() * weight_scale /
                              static_cast<double>(dropped_moduli.back().value());
    require_fractional_bits(params.weight_fractional_bits, "QKV.weight");
    require_fractional_bits(params.bias_fractional_bits, "QKV.bias");
    const bool source_is_checkpoint_integer =
        !params.semantic_values_materialized;

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
              << (source_is_checkpoint_integer ? "applied" : "reused")
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
    if (params.cross_mat_single.first.empty() ||
        params.cross_mat_single.first.front().empty()) {
        throw std::runtime_error("BSGS2: weight preprocessing is empty");
    }
    auto input_context = host_context_data_for(he, inputs.front(), "BSGS2");
    double weight_scale =
        params.cross_mat_single.first.front().front().scale();
#ifdef BLB_VIT65_HETERO
    if (he.poly_modulus_degree == 32768 && inputs.size() == 2 &&
        inputs.front().coeff_modulus_size() == 2 &&
        abs(log2(inputs.front().scale()) - 60.0) < 0.25) {
        const auto &moduli = input_context->parms().coeff_modulus();
        if (moduli.size() != 2) {
            throw runtime_error(
                "heterogeneous WO dynamic weight scale requires two limbs");
        }
        weight_scale = ldexp(1.0, 40) *
                       static_cast<double>(moduli.back().value()) /
                       inputs.front().scale();
        if (!isfinite(weight_scale) || weight_scale <= 0.0 ||
            abs(log2(weight_scale) - 14.0) > 0.25) {
            throw runtime_error(
                "heterogeneous WO dynamic weight scale is unsafe");
        }
        cout << setprecision(17)
             << "HETERO_WO_DYNAMIC_SCALE status=PASS weight_scale="
             << weight_scale << " weight_scale_bits="
             << log2(weight_scale) << " target_after_rescale_bits=40"
             << endl;
    }
#endif
    const double bias_scale = inputs.front().scale() * weight_scale;
    require_fractional_bits(params.weight_fractional_bits, "BSGS2.weight");
    require_fractional_bits(params.bias_fractional_bits, "BSGS2.bias");
    const bool source_is_checkpoint_integer =
        !params.semantic_values_materialized;
    reencode_linear_weights(
        he, params.cross_mat_single, input_context->parms_id(), weight_scale,
        params.weight_fractional_bits, source_is_checkpoint_integer,
        "linear.weight");
    reencode_plain_vector(
        he, params.cross_bias_single, input_context->parms_id(), bias_scale,
        params.bias_fractional_bits, source_is_checkpoint_integer,
        "linear.bias");
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
              << (source_is_checkpoint_integer ? "applied" : "reused")
              << " elapsed_ms=" << materialize_ms
              << std::endl;
    return {input_context->parms_id(), weight_scale, bias_scale};
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
    int weight_fractional_bits,
    int bias_fractional_bits
){
    const auto packing_start = std::chrono::steady_clock::now();
    PreprocessParams_1 pp;
    pp.weight_fractional_bits = weight_fractional_bits;
    pp.bias_fractional_bits = bias_fractional_bits;
    require_fractional_bits(pp.weight_fractional_bits,
                            "QKV preprocessing weight");
    require_fractional_bits(pp.bias_fractional_bits,
                            "QKV preprocessing bias");

    cout << "filter.h, filter.w:" << data.filter_h << "," << data.filter_w << endl;
    
    pp.wq_pack = bert_cross_packing_single_matrix(
        he, w_q, data, pp.weight_fractional_bits);
    pp.wk_pack = bert_cross_packing_single_matrix(
        he, w_k, data, pp.weight_fractional_bits);
    pp.wv_pack = bert_cross_packing_single_matrix(
        he, w_v, data, pp.weight_fractional_bits);
    // cout << "OK here" << endl;
    // exit(0);
    pp.bq_pack = bert_cross_packing_bias(
        he, b_q, data, pp.bias_fractional_bits);
    pp.bk_pack = bert_cross_packing_bias(
        he, b_k, data, pp.bias_fractional_bits);
    pp.bv_pack = bert_cross_packing_bias(
        he, b_v, data, pp.bias_fractional_bits);

	pp.cross_masks = generate_cross_packing_masks(he, data);
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
    int bias_fractional_bits
){
    const auto packing_start = std::chrono::steady_clock::now();
    PreprocessParams_2 pp;
    require_fractional_bits(weight_fractional_bits,
                            "BSGS2 preprocessing weight");
    require_fractional_bits(bias_fractional_bits,
                            "BSGS2 preprocessing bias");
    pp.weight_fractional_bits = weight_fractional_bits;
    pp.bias_fractional_bits = bias_fractional_bits;

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
        weight_fractional_bits);
    cout << "here OK" << endl;
    pp.cross_bias_single = bert_cross_packing_bias_2(
        he, b.data(), data, bias_fractional_bits);
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

void Linear_ckks::weights_preprocess(BertModel &bm){
    if (bm.w_q.size() < ATTENTION_LAYERS || bm.w_k.size() < ATTENTION_LAYERS ||
        bm.w_v.size() < ATTENTION_LAYERS || bm.w_o.size() < ATTENTION_LAYERS ||
        bm.w_i_1.size() < ATTENTION_LAYERS || bm.w_i_2.size() < ATTENTION_LAYERS) {
        throw runtime_error("formal preprocessing refused: incomplete BERT layer weights");
    }
    int preprocess_layers = ATTENTION_LAYERS;
#ifdef BLB_VIT65_FULL
    if (const char *requested = std::getenv("BLB_PREPROCESS_LAYERS")) {
        preprocess_layers = std::stoi(requested);
        if (preprocess_layers < 1 || preprocess_layers > ATTENTION_LAYERS) {
            throw runtime_error("BLB_PREPROCESS_LAYERS must be in [1,7]");
        }
    }
    cout << "VIT65_PREPROCESS_SCOPE layers=" << preprocess_layers
         << " production_layers=" << ATTENTION_LAYERS << endl;
#endif
    for(int i = 0; i < preprocess_layers; i++){
        cout << "Preprocessing layer " << i << endl;
        FCMetadata data = data_lin1_0;
        if(i > 0){
            data = data_lin1_1;
        }
        pp_1[i] = params_preprocessing_ct_ct(
            he1,
            bm.w_q[i],
            bm.w_k[i],
            bm.w_v[i],
            bm.b_q[i],
            bm.b_k[i],
            bm.b_v[i],
            data
        );
        cout << "pp_1 done" << endl;
        pp_2[i] = params_preprocessing_ct_pt(
            he2,
            INPUT_DIM,
            COMMON_DIM,
            COMMON_DIM,
            bm.w_o[i],
            bm.b_o[i],
            data_lin2,
            6,
            12
        );
#ifdef BLB_VIT65_HETERO
        const int saved_he_scale = he1->he_scale;
        he1->he_scale = 14;
        pp_2_he1[i] = params_preprocessing_ct_pt(
            he1, INPUT_DIM, COMMON_DIM, COMMON_DIM, bm.w_o[i], bm.b_o[i],
            data_lin2, 6, 12);
        he1->he_scale = saved_he_scale;
#endif
        cout << "pp_2 done" << endl;
        pp_3[i] = params_preprocessing_ct_pt(
            he3,
            INPUT_DIM,
            COMMON_DIM,
            INTER_DIM,
            bm.w_i_1[i],
            bm.b_i_1[i],
            data_lin3,
            6,
            11
        );
        cout << "pp_3 done" << endl;
        pp_4[i] = params_preprocessing_ct_pt(
            he4,
            INPUT_DIM,
            INTER_DIM,
            COMMON_DIM,
            bm.w_i_2[i],
            bm.b_i_2[i],
            data_lin4,
            5,
            9
        );
        cout << "pp_4 done" << endl;
        int slot_count = he1->poly_modulus_degree/2;
        int w_length = data_lin2.image_size*COMMON_DIM;
        int64_t* w_1 = new int64_t[w_length];
        int64_t* w_2 = new int64_t[w_length];

        vector<Plaintext> pt_w_1(w_length / slot_count);
        vector<Plaintext> pt_w_2(w_length / slot_count);
        for(int j = 0; j < data_lin2.image_size; j++){
            memcpy(&w_1[j*COMMON_DIM], bm.w_ln_1[i].data(), COMMON_DIM*sizeof(uint64_t));
            memcpy(&w_2[j*COMMON_DIM], bm.w_ln_2[i].data(), COMMON_DIM*sizeof(uint64_t));
        }
        cout << "half done" << endl;
        for(int i = 0; i < pt_w_1.size(); i++){
            vector<double> tmp_1(&w_1[i*slot_count], &w_1[(i+1)*slot_count]);
            Plaintext pt_1;
            he1->encoder->encode(tmp_1, pow(2,he1->he_scale), pt_1);
            pt_w_1[i] = pt_1;

            vector<double> tmp_2(&w_2[i*slot_count], &w_2[(i+1)*slot_count]);
            Plaintext pt_2;
            he2->encoder->encode(tmp_2, pow(2,he2->he_scale), pt_2);
            pt_w_2[i] = pt_2;
        }

        w_ln_1_pt[i] = pt_w_1;
        w_ln_2_pt[i] = pt_w_2;
    }

    w_ln_1 = bm.w_ln_1;
    b_ln_1 = bm.b_ln_1;

    w_ln_2 = bm.w_ln_2;
    b_ln_2 = bm.b_ln_2;

    w_c = bm.w_c;
    b_c = bm.b_c;
    w_p = bm.w_p;
    b_p = bm.b_p;
}

void Linear_ckks::attention_weights_preprocess(BertModel &bm) {
    if (bm.w_q.size() < ATTENTION_LAYERS ||
        bm.w_k.size() < ATTENTION_LAYERS ||
        bm.w_v.size() < ATTENTION_LAYERS ||
        bm.w_o.size() < ATTENTION_LAYERS ||
        bm.b_q.size() < ATTENTION_LAYERS ||
        bm.b_k.size() < ATTENTION_LAYERS ||
        bm.b_v.size() < ATTENTION_LAYERS ||
        bm.b_o.size() < ATTENTION_LAYERS) {
        throw runtime_error(
            "BLB attention-only preprocessing refused incomplete weights");
    }
    for (int layer = 0; layer < ATTENTION_LAYERS; ++layer) {
        const FCMetadata &qkv_data =
            layer == 0 ? data_lin1_0 : data_lin1_1;
        pp_1[layer] = params_preprocessing_ct_ct(
            he1, bm.w_q[layer], bm.w_k[layer], bm.w_v[layer],
            bm.b_q[layer], bm.b_k[layer], bm.b_v[layer], qkv_data,
            BLB_QKV_WEIGHT_FRACTIONAL_BITS,
            BLB_QKV_BIAS_FRACTIONAL_BITS);
        pp_2[layer] = params_preprocessing_ct_pt(
            he2, INPUT_DIM, COMMON_DIM, COMMON_DIM, bm.w_o[layer],
            bm.b_o[layer], data_lin2, 6, 12);
#ifdef BLB_VIT65_HETERO
        // Heterogeneous AV remains in he1. V is multiplied by an encoded
        // public one to reach 2^60; a 2^20 diagonal score mask
        // matches that scale. AV lands at 2^60 after its 60-bit rescale. A
        // 2^14 WO plaintext then lands at 2^40 after the remaining 34-bit
        // rescale immediately before Algorithm 1. The six
        // checkpoint fractional bits are unchanged.
        const int saved_he_scale = he1->he_scale;
        he1->he_scale = 14;
        pp_2_he1[layer] = params_preprocessing_ct_pt(
            he1, INPUT_DIM, COMMON_DIM, COMMON_DIM, bm.w_o[layer],
            bm.b_o[layer], data_lin2, 6, 12);
        he1->he_scale = saved_he_scale;
#endif
    }
    cout << "BLB_ATTENTION_WEIGHTS status=PASS layers=" << ATTENTION_LAYERS
         << " physical_tokens=" << INPUT_DIM << " hidden=" << COMMON_DIM
         << " heads=" << PACKING_NUM << " head_dim=" << OUTPUT_DIM
         << endl;
}

void Linear_ckks::linear_1(
		HE_CKKS* he,
		const vector<UnifiedCiphertext> &input_cts,
		PreprocessParams_1 &pp,
		const FCMetadata &data,
        vector<UnifiedCiphertext> &output,
        vector<UnifiedCiphertext> *qkv_snapshot) {

    cout << "input_cts.size: " << input_cts.size() << endl;
    const auto loc = input_cts[0].location();
	vector<UnifiedCiphertext> Cipher_plain_results(
        data.image_size * data.filter_w * 3 * PACKING_NUM / data.slot_count,
        UnifiedCiphertext(loc)
    );
	// Q,K,V
    bert_cipher_plain_bsgs(
        he, 
        input_cts, 
        pp,
        data, 
        Cipher_plain_results
    );
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
	if (qkv_snapshot != nullptr) {
		*qkv_snapshot = Cipher_plain_results;
	}
	output.resize(
        data.image_size * data.image_size * PACKING_NUM / data.slot_count 
        + data.image_size * data.filter_w * PACKING_NUM / data.slot_count,
        UnifiedCiphertext(loc)
    );
	//QK^T
    bert_cipher_cipher_cross_packing(he, data, Cipher_plain_results, output);

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
    const FCMetadata &data, int fractional_bits) {
    vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>>
        result(PACKING_NUM / 2);
    const double semantic_multiplier = checkpoint_semantic_multiplier(
        fractional_bits, "QKV packed weight");
    int num_diag = data.slot_count / data.image_size / 2;

#ifdef BLB_VIT65_A0
    // The published CKKS donor leaves the QKV ct-pt multiply wired to unit
    // plaintexts and its dormant packed-weight table does not implement all
    // 64 diagonals.  A0 keeps the same two-head MHP layout and an 8x8 BSGS,
    // but materializes the mathematically complete diagonals.  Slots in each
    // half are [output_channel, token].  Two global rotations implement each
    // cyclic baby step; two split accumulators implement each cyclic giant
    // step.  The wrap selection is fused into the real weight plaintexts, so
    // it neither consumes another CKKS level nor relies on an inaccurate
    // scale-one slot mask.
    constexpr int baby_steps = 8;
    if (data.image_size != 128 || data.filter_h != 256 ||
        data.filter_w != 64 || data.slot_count != 16384 ||
        PACKING_NUM != 4 || num_diag != 64) {
        throw runtime_error(
            "A0 QKV packed-weight geometry differs from 128x256x4x64");
    }
    for (int packing = 0; packing < PACKING_NUM / 2; ++packing) {
        vector<vector<Plaintext>> direct(
            num_diag * 2, vector<Plaintext>(data.filter_h * data.image_size /
                                           data.slot_count));
        vector<vector<Plaintext>> wrapped = direct;
        for (int diagonal = 0; diagonal < num_diag; ++diagonal) {
            const int giant = diagonal / baby_steps;
            const int baby = diagonal % baby_steps;
            const int giant_shift = giant * baby_steps;
            for (int input_ct = 0;
                 input_ct < data.filter_h * data.image_size / data.slot_count;
                 ++input_ct) {
                for (int split = 0; split < 2; ++split) {
                    vector<double> direct_slots(data.slot_count, 0.0);
                    vector<double> wrapped_slots(data.slot_count, 0.0);
                    for (int half = 0; half < 2; ++half) {
                        const int head = packing * 2 + half;
                        for (int position = 0; position < num_diag;
                             ++position) {
                            const bool keep = split == 0
                                ? position >= giant_shift
                                : position < giant_shift;
                            if (!keep) continue;
                            const int output_channel =
                                (position - giant_shift + num_diag) % num_diag;
                            const bool baby_prefix =
                                position < num_diag - baby;
                            const int local_input = baby_prefix
                                ? position + baby
                                : position + baby - num_diag;
                            const int direct_half =
                                baby_prefix ? half : 1 - half;
                            const int wrapped_half =
                                baby_prefix ? 1 - half : half;
                            const int direct_input =
                                input_ct * 2 * num_diag +
                                direct_half * num_diag + local_input;
                            const int wrapped_input =
                                input_ct * 2 * num_diag +
                                wrapped_half * num_diag + local_input;
                            const double direct_value =
                                checkpoint_fixed_to_real(
                                    weights[head][direct_input]
                                           [output_channel],
                                    semantic_multiplier);
                            const double wrapped_value =
                                checkpoint_fixed_to_real(
                                    weights[head][wrapped_input]
                                           [output_channel],
                                    semantic_multiplier);
                            const int base = half * data.slot_count / 2 +
                                             position * data.image_size;
                            fill_n(direct_slots.begin() + base,
                                   data.image_size, direct_value);
                            fill_n(wrapped_slots.begin() + base,
                                   data.image_size, wrapped_value);
                        }
                    }
                    const int packed_index = diagonal * 2 + split;
                    he->encoder->encode(direct_slots, pow(2, he->he_scale),
                                        direct[packed_index][input_ct]);
                    he->encoder->encode(wrapped_slots, pow(2, he->he_scale),
                                        wrapped[packed_index][input_ct]);
                }
            }
        }
        result[packing] = {std::move(direct), std::move(wrapped)};
    }
    cout << "A0_QKV_WEIGHT_PACK status=PASS layout=two_head_mhp"
         << " bsgs=8x8 diagonals=" << num_diag
         << " wrap_split_plaintexts=" << num_diag * 2
         << " input_cts=2" << endl;
    return result;
#endif

    int n1 = 8;
    int n2 = 4;
    if (data.image_size == 64) {
        n1 = 16;
        n2 = 4;
    }

    int weight_height = data.filter_h;

    int num_matrix_per_row = weight_height / num_diag;
    int num_matrix_per_col = data.filter_w / num_diag;
    // cout << "num_matrix_per_col,num_matrix_per_row,num_diag,data.image_size: " << num_matrix_per_col << "," << num_matrix_per_row << "," << num_diag << "," << data.image_size << endl;
    
    #pragma omp parallel for num_threads(PACKING_NUM/2)
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; packing_ind++) {
        vector<double> temp2;
        vector<vector<Plaintext>> weightMatrix1;
        vector<vector<Plaintext>> weightMatrix2;
        int cot = 0;
        for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
            int matrix_flag = 0;
            // #pragma omp parallel for num_threads(4)
            for (int l = 0; l < num_diag; l++) {
                vector<Plaintext> temp_matrix_diag2(weight_height * data.image_size / data.slot_count);
                vector<vector<double>> temp_matrix_diag(weight_height * data.image_size / data.slot_count);
                int matrix_diag_index = 0;
                for (int i = 0; i < num_matrix_per_row; i++) {
                    for (int j = 0; j < num_diag; j++) {
                        const int head = packing_ind * 2 + matrix_flag;
                        const double value = checkpoint_fixed_to_real(
                            weights[head][i * num_diag + j]
                                   [(j + l) % num_diag + col_ind * num_diag],
                            semantic_multiplier);
                        temp2.insert(temp2.end(), data.image_size, value);
                        if (temp2.size() % (data.slot_count / 2) == 0) {
                            matrix_flag = (matrix_flag + 1) % 2;
                            std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                            if (temp2.size() == data.slot_count) {
                                temp_matrix_diag[matrix_diag_index] = temp2;
                                matrix_diag_index++;
                                temp2.clear();
                            }
                        }
                    }
                }
                
                #pragma omp parallel for num_threads(temp_matrix_diag.size())
                for(int i=0;i<temp_matrix_diag.size();i++){
                    Plaintext pt;
                    he->encoder->encode(temp_matrix_diag[i], pow(2, he->he_scale), pt);
                    temp_matrix_diag2[i] = pt;
                }
                weightMatrix1.push_back(temp_matrix_diag2);
            }
        }
        // cout << "OK HERE hhh" << endl;
        for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
            int matrix_flag = 0;
            // #pragma omp parallel for num_threads(4)
            for (int l = 0; l < num_diag; l++) {
                vector<Plaintext> temp_matrix_diag2(weight_height * data.image_size / data.slot_count);
                vector<vector<double>> temp_matrix_diag(weight_height * data.image_size / data.slot_count);
                int matrix_diag_index = 0;
                for (int i = 0; i < num_matrix_per_row; i++) {
                    for (int j = 0; j < num_diag; j++) {
                        const int head = packing_ind * 2 + 1 - matrix_flag;
                        const double value = checkpoint_fixed_to_real(
                            weights[head][i * num_diag + j]
                                   [(j + l) % num_diag + col_ind * num_diag],
                            semantic_multiplier);
                        temp2.insert(temp2.end(), data.image_size, value);
                        if (temp2.size() % (data.slot_count / 2) == 0) {
                            matrix_flag = (matrix_flag + 1) % 2;
                            std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                            if (temp2.size() == data.slot_count) {
                                std::rotate(temp2.begin(), temp2.begin() + temp2.size() / 2, temp2.end());
                                temp_matrix_diag[matrix_diag_index] = temp2;
                                matrix_diag_index++;
                                temp2.clear();
                            }
                        }
                    }
                }
                #pragma omp parallel for num_threads(temp_matrix_diag.size())
                for(int i=0;i<temp_matrix_diag.size();i++){
                    Plaintext pt;
                    he->encoder->encode(temp_matrix_diag[i], pow(2, he->he_scale), pt);
                    temp_matrix_diag2[i] = pt;
                }
                weightMatrix2.push_back(temp_matrix_diag2);
            }
        }
        // cout << "weightMatrix1.size: " << weightMatrix1.size() << "," << weightMatrix1[0].size() << endl;
        // cout << "weightMatrix2.size: " << weightMatrix2.size() << "," << weightMatrix2[0].size() << endl;
        result[packing_ind] = std::make_pair(weightMatrix1, weightMatrix2);
    }
    return result;
}

pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>
Linear_ckks::bert_cross_packing_single_matrix_2(
    HE_CKKS* he,
    const uint64_t *const *matrix1,
    const uint64_t *const *matrix2,
    const FCMetadata &data,
    int fractional_bits){
    
    vector<vector<Plaintext>> weightMatrix1; // 64 x 48
    vector<vector<Plaintext>> weightMatrix2; // 64 x 48
    const double semantic_multiplier = checkpoint_semantic_multiplier(
        fractional_bits, "BSGS2 packed weight");

#ifdef BLB_VIT65_A0
    // General ViT65 BSGS2 table.  Each ciphertext holds
    // slot_count/image_size feature positions, split equally across the two
    // output halves used by the donor layout.  This covers WO (256x256),
    // FFN1 (256x512) and FFN2 (512x256) without changing their mathematics.
    // Store P[g,b](r)=W[input_ct*128+(r+b), output(ct,r-gB)] so the
    // evaluator's final +gB rotation produces the ordinary diagonal product.
    constexpr int baby_steps = 8;
    const int feature_blocks = data.slot_count / data.image_size;
    const int features_per_half = feature_blocks / 2;
    const int giant_steps = feature_blocks / baby_steps;
    const int input_ciphertexts = data.filter_h / feature_blocks;
    const int output_ciphertexts = data.filter_w / feature_blocks;
    if (data.image_size != 128 || feature_blocks <= 0 ||
        feature_blocks % (2 * baby_steps) != 0 ||
        data.filter_h % feature_blocks != 0 ||
        data.filter_w % feature_blocks != 0 ||
        data.filter_w % 2 != 0 || input_ciphertexts <= 0 ||
        output_ciphertexts <= 0) {
        throw runtime_error(
            "ViT65 BSGS2 preprocessing geometry mismatch");
    }
    vector<vector<Plaintext>> packed_weights(
        output_ciphertexts * giant_steps * baby_steps,
        vector<Plaintext>(input_ciphertexts));
    for (int output_ct = 0; output_ct < output_ciphertexts; ++output_ct) {
        for (int giant = 0; giant < giant_steps; ++giant) {
            const int giant_shift = giant * baby_steps;
            for (int baby = 0; baby < baby_steps; ++baby) {
                const int packed_index =
                    (output_ct * giant_steps + giant) * baby_steps + baby;
                for (int input_ct = 0; input_ct < input_ciphertexts;
                     ++input_ct) {
                    vector<double> slots(data.slot_count, 0.0);
                    for (int source_position = 0;
                         source_position < feature_blocks;
                         ++source_position) {
                        const int output_position =
                            (source_position - giant_shift + feature_blocks) %
                            feature_blocks;
                        const int input_feature = input_ct * feature_blocks +
                            (source_position + baby) % feature_blocks;
                        const int output_feature =
                            output_position < features_per_half
                            ? output_ct * features_per_half + output_position
                            : data.filter_w / 2 +
                                  output_ct * features_per_half +
                                  (output_position - features_per_half);
                        const uint64_t raw = output_feature < data.filter_w / 2
                            ? matrix1[input_feature][output_feature]
                            : matrix2[input_feature]
                                     [output_feature - data.filter_w / 2];
                        const double value = checkpoint_fixed_to_real(
                            raw, semantic_multiplier);
                        fill_n(slots.begin() +
                                   source_position * data.image_size,
                               data.image_size, value);
                    }
                    he->encoder->encode(slots, pow(2, he->he_scale),
                                        packed_weights[packed_index][input_ct]);
                }
            }
        }
    }
    cout << "VIT65_BSGS2_WEIGHT_PACK status=PASS"
         << " filter=" << data.filter_h << 'x' << data.filter_w
         << " layout=split_output_halves baby_steps=" << baby_steps
         << " giant_steps=" << giant_steps
         << " diagonals=" << feature_blocks
         << " input_cts=" << input_ciphertexts
         << " output_cts=" << output_ciphertexts << endl;
    return {std::move(packed_weights), {}};
#endif
    
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
        // BSGS2 operates on one output half at a time.  Each physical-token
        // block therefore needs num_diag/2 diagonals.  Preserve the donor's
        // baby-step width and derive the giant-step count rather than using
        // the former BERT-only 4x4 fallback.
        n1 = 4;
        const int half_diagonals = num_diag / 2;
        if (half_diagonals <= 0 || half_diagonals % n1 != 0) {
            throw runtime_error(
                "unsupported BSGS2 token/slot geometry in preprocessing");
        }
        n2 = half_diagonals / n1;
    }
    cout << "num_matrix_per_col: " << num_matrix_per_col << endl;
    int cot = 0;
    vector<double> temp2(data.slot_count);
    // #pragma omp parallel for
    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        cout << "col_ind: " << col_ind << endl;
        for (int l = 0; l < num_diag; l++) {
            vector<vector<double>> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            vector<Plaintext> temp_matrix_diag2(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    const uint64_t raw = matrix_flag == 0
                        ? matrix1[i * num_diag + j]
                                 [(j + l) % num_diag + col_ind * num_diag]
                        : matrix2[i * num_diag + j]
                                 [(j + l) % num_diag + col_ind * num_diag];
                    const double value = checkpoint_fixed_to_real(
                        raw, semantic_multiplier);
                    std::fill(temp2.begin() + cot*data.image_size,
                              temp2.begin()+(cot+1)*data.image_size, value);
                    cot++;
                    if (cot==(data.slot_count / 2/ data.image_size)) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin(), temp2.begin() + data.slot_count / 2 - (l % n1) * data.image_size, temp2.begin()+data.slot_count / 2);
                    }
                    else if (cot==(data.slot_count / data.image_size)) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        temp_matrix_diag[matrix_diag_index] = temp2;
                        matrix_diag_index++;
                        cot = 0;
                    }
                }
            }
            // cout << "temp_matrix_diag[].size: " << temp_matrix_diag.size() << endl;
            #pragma omp parallel for num_threads(temp_matrix_diag.size())
            for(int i=0;i<temp_matrix_diag.size();i++){
                Plaintext pt;
                he->encoder->encode(temp_matrix_diag[i], pow(2, he->he_scale), pt);
                temp_matrix_diag2[i] = pt;
            }
            weightMatrix1.push_back(temp_matrix_diag2);
        }
    }
    
    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        for (int l = 0; l < num_diag; l++) {
            vector<vector<double>> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            vector<Plaintext> temp_matrix_diag2(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    const uint64_t raw = matrix_flag == 0
                        ? matrix2[i * num_diag + j]
                                 [(j + l) % num_diag + col_ind * num_diag]
                        : matrix1[i * num_diag + j]
                                 [(j + l) % num_diag + col_ind * num_diag];
                    const double value = checkpoint_fixed_to_real(
                        raw, semantic_multiplier);
                    std::fill(temp2.begin() + cot*data.image_size,
                              temp2.begin()+(cot+1)*data.image_size, value);
                    cot++;
                    if (cot==(data.slot_count / 2/ data.image_size)) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin(), temp2.begin() + data.slot_count / 2 - (l % n1) * data.image_size, temp2.begin()+data.slot_count / 2);
                    }
                    else if (cot==(data.slot_count / data.image_size)) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        std::rotate(temp2.begin(), temp2.begin() + temp2.size() / 2, temp2.end());
                        temp_matrix_diag[matrix_diag_index] = temp2;
                        matrix_diag_index++;
                        cot = 0;
                    }
                }
            }
            
            #pragma omp parallel for num_threads(temp_matrix_diag.size())
            for(int i=0;i<temp_matrix_diag.size();i++){
                Plaintext pt;
                he->encoder->encode(temp_matrix_diag[i], pow(2, he->he_scale), pt);
                temp_matrix_diag2[i] = pt;
            }
            weightMatrix2.push_back(temp_matrix_diag2);
        }
    }
    
    return std::make_pair(weightMatrix1, weightMatrix2);
}


vector<vector<Plaintext>> Linear_ckks::bert_cross_packing_bias(
    HE_CKKS* he,
    const vector<vector<uint64_t>> &bias, 
    const FCMetadata &data,
    int fractional_bits) {
    vector<vector<Plaintext>> cross_bias_packing(PACKING_NUM/2);
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
        Plaintext pt;
        he->encoder->encode(temp,pow(2, he->he_scale), pt);
        cross_bias_packing[current_packing].push_back(pt);
    }
    return cross_bias_packing;
}

vector<Plaintext> Linear_ckks::bert_cross_packing_bias_2(
    HE_CKKS* he,
    const uint64_t *matrix, 
    const FCMetadata &data,
    int fractional_bits){

    vector<Plaintext> cross_bias_packing(data.image_size * data.filter_w / data.slot_count);
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
        Plaintext pt;
        // The existing BSGS2 kernel adds bias before its first rescale, so the
        // bias plaintext must match ct_scale * weight_scale.
        he->encoder->encode(temp,pow(4, he->he_scale), pt);
        cross_bias_packing[packing_num] = pt;
        temp.clear();
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
        n2 = 4;
    }
    const auto loc = cts[0].location();
    if (loc != LOCATION::HOST && loc != LOCATION::DEVICE) {
        throw runtime_error("formal QKV requires a HOST or DEVICE ciphertext");
    }
    if (pp.wq_pack.size() != PACKING_NUM / 2 ||
        pp.wk_pack.size() != PACKING_NUM / 2 ||
        pp.wv_pack.size() != PACKING_NUM / 2 ||
        pp.bq_pack.size() != PACKING_NUM / 2 ||
        pp.bk_pack.size() != PACKING_NUM / 2 ||
        pp.bv_pack.size() != PACKING_NUM / 2) {
        throw runtime_error("formal QKV refused: pp_1 is incomplete");
    }

    const auto materialized = materialize_qkv_plaintexts(*he, cts, pp);
    const DevicePreprocessParams_1 *device_pp = nullptr;
#ifdef BLB_HAS_HE_GPU
    std::unique_ptr<CkksDevicePlainCache> device_cache;
    if (loc == LOCATION::DEVICE) {
        auto upload_plan = make_qkv_upload_plan(
            *he, cts.front().dcipher().chain_index(),
            materialized.weight_scale, materialized.bias_scale);
        device_cache = std::make_unique<CkksDevicePlainCache>();
        device_cache->upload_qkv(*he, pp, upload_plan);
        device_pp = &device_cache->qkv();
        cout << "CKKS_DEVICE_PLAIN_CACHE stage=QKV plaintexts="
             << device_cache->plaintext_count() << " bytes="
             << device_cache->device_bytes() << endl;
    }
#else
    if (loc == LOCATION::DEVICE) {
        throw runtime_error(
            "formal QKV DEVICE backend was not compiled with USE_HE_GPU=ON");
    }
#endif

#ifdef BLB_VIT65_A0
    if (loc != LOCATION::DEVICE || device_pp == nullptr) {
        throw runtime_error(
            "A0 QKV requires the real Phantom DEVICE plaintext cache");
    }
    constexpr int baby_steps = 8;
    const int num_diag_a0 = data.slot_count / data.image_size / 2;
    const int giant_steps = num_diag_a0 / baby_steps;
    const int input_ct_count = data.filter_h * data.image_size /
                               data.slot_count;
    if (num_diag_a0 != 64 || giant_steps != 8 || input_ct_count != 2 ||
        cts.size() != static_cast<size_t>(input_ct_count) ||
        result.size() != static_cast<size_t>(PACKING_NUM * 3 / 2)) {
        throw runtime_error("A0 QKV 8x8 BSGS runtime geometry mismatch");
    }

    // For a baby shift i, rotate_vector(i*T) and
    // rotate_vector((i-64)*T) each contain one correct and one wrapped region
    // in every MHP half.  The preprocessing above fuses the region selection
    // and its true model weight into the two plaintexts.  This avoids the
    // numerically invalid scale-one CKKS masks used by the previous draft.
    vector<vector<UnifiedCiphertext>> direct_rotations(
        input_ct_count, vector<UnifiedCiphertext>(
                            baby_steps, UnifiedCiphertext(LOCATION::DEVICE)));
    vector<vector<UnifiedCiphertext>> wrapped_rotations(
        input_ct_count, vector<UnifiedCiphertext>(
                            baby_steps, UnifiedCiphertext(LOCATION::DEVICE)));
    for (int input_ct = 0; input_ct < input_ct_count; ++input_ct) {
        for (int baby = 0; baby < baby_steps; ++baby) {
            if (baby == 0) {
                direct_rotations[input_ct][baby] = cts[input_ct];
            } else {
                he->devaluator->rotate_vector(
                    cts[input_ct], baby * data.image_size, *he->dgal_keys,
                    direct_rotations[input_ct][baby]);
            }
            he->devaluator->rotate_vector(
                cts[input_ct], (baby - num_diag_a0) * data.image_size,
                *he->dgal_keys, wrapped_rotations[input_ct][baby]);
        }
    }

    auto evaluate_one = [&](const vector<DevicePackedWeight> &weights,
                            const DevicePlainMatrix &biases,
                            int output_offset, const char *label) {
        for (int packing = 0; packing < PACKING_NUM / 2; ++packing) {
            vector<UnifiedCiphertext> direct_groups(
                giant_steps, UnifiedCiphertext(LOCATION::DEVICE));
            vector<UnifiedCiphertext> wrapped_groups(
                giant_steps, UnifiedCiphertext(LOCATION::DEVICE));
            for (int giant = 0; giant < giant_steps; ++giant) {
                bool direct_initialized = false;
                bool wrapped_initialized = false;
                for (int input_ct = 0; input_ct < input_ct_count; ++input_ct) {
                    for (int baby = 0; baby < baby_steps; ++baby) {
                        const int diagonal = giant * baby_steps + baby;
                        const int split_count = giant == 0 ? 1 : 2;
                        for (int split = 0; split < split_count; ++split) {
                            const int weight_index = diagonal * 2 + split;
                            const auto &direct_weight =
                                weights[packing].first[weight_index][input_ct];
                            const auto &wrapped_weight =
                                weights[packing].second[weight_index][input_ct];
                            require_device_plain_compatible(
                                direct_rotations[input_ct][baby], direct_weight,
                                CkksCtPtOperation::Multiply,
                                string("A0.QKV.") + label + ".direct");
                            require_device_plain_compatible(
                                wrapped_rotations[input_ct][baby],
                                wrapped_weight, CkksCtPtOperation::Multiply,
                                string("A0.QKV.") + label + ".wrapped");
                            UnifiedCiphertext direct(LOCATION::DEVICE);
                            UnifiedCiphertext wrapped(LOCATION::DEVICE);
                            UnifiedCiphertext contribution(LOCATION::DEVICE);
                            he->devaluator->multiply_plain(
                                direct_rotations[input_ct][baby], direct_weight,
                                direct);
                            he->devaluator->multiply_plain(
                                wrapped_rotations[input_ct][baby],
                                wrapped_weight, wrapped);
                            he->devaluator->add(direct, wrapped, contribution);
                            UnifiedCiphertext &group = split == 0
                                ? direct_groups[giant]
                                : wrapped_groups[giant];
                            bool &initialized = split == 0
                                ? direct_initialized
                                : wrapped_initialized;
                            if (!initialized) {
                                group = std::move(contribution);
                                initialized = true;
                            } else {
                                he->devaluator->add_inplace(group,
                                                             contribution);
                            }
                        }
                    }
                }
                if (!direct_initialized ||
                    (giant != 0 && !wrapped_initialized)) {
                    throw runtime_error("A0 QKV BSGS produced an empty group");
                }
                he->devaluator->rescale_to_next_inplace(
                    direct_groups[giant]);
                if (giant != 0) {
                    he->devaluator->rescale_to_next_inplace(
                        wrapped_groups[giant]);
                }
            }

            UnifiedCiphertext combined = std::move(direct_groups[0]);
            for (int giant = 1; giant < giant_steps; ++giant) {
                const int giant_shift = giant * baby_steps;
                UnifiedCiphertext rotated_direct(LOCATION::DEVICE);
                UnifiedCiphertext rotated_wrapped(LOCATION::DEVICE);
                he->devaluator->rotate_vector(
                    direct_groups[giant], giant_shift * data.image_size,
                    *he->dgal_keys, rotated_direct);
                he->devaluator->rotate_vector(
                    wrapped_groups[giant],
                    (giant_shift - num_diag_a0) * data.image_size,
                    *he->dgal_keys, rotated_wrapped);
                he->devaluator->add_inplace(combined, rotated_direct);
                he->devaluator->add_inplace(combined, rotated_wrapped);
            }
            const auto &bias = biases[packing][0];
            require_device_plain_compatible(
                combined, bias, CkksCtPtOperation::Add,
                string("A0.QKV.") + label + ".bias");
            he->devaluator->add_plain_inplace(combined, bias);
            result[output_offset + packing] = std::move(combined);
        }
    };

    evaluate_one(device_pp->wq_pack, device_pp->bq_pack, 0, "q");
    evaluate_one(device_pp->wk_pack, device_pp->bk_pack, PACKING_NUM / 2,
                 "k");
    evaluate_one(device_pp->wv_pack, device_pp->bv_pack, PACKING_NUM, "v");
    if (device_cache) {
        device_cache->clear();
    }
    cout << "A0_QKV_BSGS status=PASS backend=phantom_gpu bsgs=8x8"
         << " heads_per_ciphertext=2 input_cts=" << input_ct_count
         << " output_cts=" << result.size()
         << " scale=" << log2(result.front().scale()) << endl;
    return;
#endif

    // begin computation
    vector<vector<UnifiedCiphertext>> rotatedIR(
        cts.size(), 
        vector<UnifiedCiphertext>(2 * n1, UnifiedCiphertext(loc)));

    int num_diag = data.slot_count / data.image_size / 2;
#ifdef BLB_VIT65_A0
    // The donor hard-codes four giant steps and consequently consumes only
    // 8*4 of the 64 diagonals materialized for a 128-token/64-wide head.  The
    // missing half is observable as a 1/sqrt(2) output-norm loss.  ViT A0
    // derives the giant-step count from the physical layout; the original
    // BERT build keeps its historical schedule unchanged.
    if (num_diag % n1 != 0) {
        throw runtime_error("A0 QKV BSGS diagonal geometry is not divisible");
    }
    n2 = num_diag / n1;
#endif
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

                he->evaluator->rescale_to_next_inplace(temp_results_q[l][k]);
                he->evaluator->rescale_to_next_inplace(temp_results_k[l][k]);
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
    for (int packing_index = 0; packing_index < PACKING_NUM/2; packing_index++) {
        //compute matrix multiplication
        vector<vector<UnifiedCiphertext>> temp_results(temp_result_size * 3, vector<UnifiedCiphertext>(n2, UnifiedCiphertext(loc)));
        vector<vector<UnifiedCiphertext>> temp_results_q(temp_result_size, vector<UnifiedCiphertext>(n2 * cts.size(), UnifiedCiphertext(loc)));
        vector<vector<UnifiedCiphertext>> temp_results_k(temp_result_size, vector<UnifiedCiphertext>(n2 * cts.size(), UnifiedCiphertext(loc)));
        vector<vector<UnifiedCiphertext>> temp_results_v(temp_result_size, vector<UnifiedCiphertext>(n2 * cts.size(), UnifiedCiphertext(loc)));

        for (int k = 0; k < cts.size() * n2; k++) {
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
                    const auto &wq_l = device_pp->wq_pack[packing_index].first[diag][ct_i];
                    const auto &wq_r = device_pp->wq_pack[packing_index].second[diag][ct_i];
                    const auto &wk_l = device_pp->wk_pack[packing_index].first[diag][ct_i];
                    const auto &wk_r = device_pp->wk_pack[packing_index].second[diag][ct_i];
                    const auto &wv_l = device_pp->wv_pack[packing_index].first[diag][ct_i];
                    const auto &wv_r = device_pp->wv_pack[packing_index].second[diag][ct_i];
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
            const auto &bq = device_pp->bq_pack[packing_index][l];
            const auto &bk = device_pp->bk_pack[packing_index][l];
            const auto &bv = device_pp->bv_pack[packing_index][l];
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
    if (device_cache) {
        device_cache->clear();
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
        n1 = 4;
        const int num_diag = data.slot_count / data.image_size / 2;
        const int half_diagonals = num_diag / 2;
        if (half_diagonals <= 0 || half_diagonals % n1 != 0) {
            throw runtime_error(
                "unsupported BSGS2 token/slot geometry in evaluator");
        }
        n2 = half_diagonals / n1;
    }
    const auto loc = cts[0].location();
    if (loc != LOCATION::HOST && loc != LOCATION::DEVICE) {
        throw runtime_error("formal output/FFN requires a HOST or DEVICE ciphertext");
    }
    if (pp.cross_mat_single.first.empty() ||
#ifndef BLB_VIT65_A0
        pp.cross_mat_single.second.empty() ||
#endif
        pp.cross_bias_single.empty()) {
        throw runtime_error("formal linear refused: pp_2/pp_3/pp_4 is incomplete");
    }

    const auto materialized = materialize_linear_plaintexts(*he, cts, pp);
    const DevicePreprocessParams_2 *device_pp = nullptr;
#ifdef BLB_HAS_HE_GPU
    std::unique_ptr<CkksDevicePlainCache> device_cache;
    if (loc == LOCATION::DEVICE) {
        auto upload_plan = make_linear_upload_plan(
            *he, cts.front().dcipher().chain_index(),
            materialized.weight_scale, materialized.bias_scale);
        device_cache = std::make_unique<CkksDevicePlainCache>();
        device_cache->upload_linear(*he, pp, upload_plan);
        device_pp = &device_cache->linear();
        cout << "CKKS_DEVICE_PLAIN_CACHE stage=LINEAR plaintexts="
             << device_cache->plaintext_count() << " bytes="
             << device_cache->device_bytes() << endl;
    }
#else
    if (loc == LOCATION::DEVICE) {
        throw runtime_error(
            "formal output/FFN DEVICE backend was not compiled with USE_HE_GPU=ON");
    }
#endif

#ifdef BLB_VIT65_A0
    if (loc != LOCATION::DEVICE || device_pp == nullptr) {
        throw runtime_error("A0 WO requires Phantom DEVICE plaintexts");
    }
    constexpr int baby_steps = 8;
    const int feature_blocks = data.slot_count / data.image_size;
    const int giant_steps = feature_blocks / baby_steps;
    const int input_ciphertexts = data.filter_h / feature_blocks;
    const int output_ciphertexts = data.filter_w / feature_blocks;
    const auto &weights = device_pp->cross_mat_single.first;
    if (data.image_size != 128 || feature_blocks <= 0 ||
        feature_blocks % (2 * baby_steps) != 0 ||
        data.filter_h % feature_blocks != 0 ||
        data.filter_w % feature_blocks != 0 ||
        cts.size() != input_ciphertexts || result.size() != output_ciphertexts ||
        weights.size() != output_ciphertexts * giant_steps * baby_steps ||
        device_pp->cross_bias_single.size() != output_ciphertexts) {
        throw runtime_error("ViT65 BSGS2 evaluator geometry mismatch");
    }

    vector<vector<UnifiedCiphertext>> baby_rotations(
        input_ciphertexts,
        vector<UnifiedCiphertext>(baby_steps,
                                  UnifiedCiphertext(LOCATION::DEVICE)));
    for (int input_ct = 0; input_ct < input_ciphertexts; ++input_ct) {
        baby_rotations[input_ct][0] = cts[input_ct];
        for (int baby = 1; baby < baby_steps; ++baby) {
            he->devaluator->rotate_vector(
                cts[input_ct], baby * data.image_size, *he->dgal_keys,
                baby_rotations[input_ct][baby]);
        }
    }

    for (int output_ct = 0; output_ct < output_ciphertexts; ++output_ct) {
        bool output_initialized = false;
        for (int giant = 0; giant < giant_steps; ++giant) {
            bool group_initialized = false;
            UnifiedCiphertext group(LOCATION::DEVICE);
            for (int input_ct = 0; input_ct < input_ciphertexts; ++input_ct) {
                for (int baby = 0; baby < baby_steps; ++baby) {
                    const int packed_index =
                        (output_ct * giant_steps + giant) * baby_steps + baby;
                    if (weights[packed_index].size() != input_ciphertexts) {
                        throw runtime_error(
                            "ViT65 BSGS2 packed input coverage mismatch");
                    }
                    const auto &weight = weights[packed_index][input_ct];
                    require_device_plain_compatible(
                        baby_rotations[input_ct][baby], weight,
                        CkksCtPtOperation::Multiply,
                        "ViT65.BSGS2.weight");
                    UnifiedCiphertext contribution(LOCATION::DEVICE);
                    he->devaluator->multiply_plain(
                        baby_rotations[input_ct][baby], weight,
                        contribution);
                    if (!group_initialized) {
                        group = std::move(contribution);
                        group_initialized = true;
                    } else {
                        he->devaluator->add_inplace(group, contribution);
                    }
                }
            }
            if (!group_initialized) {
                throw runtime_error(
                    "ViT65 BSGS2 produced an empty giant group");
            }
            if (giant != 0) {
                he->devaluator->rotate_vector_inplace(
                    group, giant * baby_steps * data.image_size,
                    *he->dgal_keys);
            }
            if (!output_initialized) {
                result[output_ct] = std::move(group);
                output_initialized = true;
            } else {
                he->devaluator->add_inplace(result[output_ct], group);
            }
        }
        const auto &bias = device_pp->cross_bias_single[output_ct];
        require_device_plain_compatible(
            result[output_ct], bias, CkksCtPtOperation::Add,
            "ViT65.BSGS2.bias");
        he->devaluator->add_plain_inplace(result[output_ct], bias);
    }
    if (device_cache) {
        device_cache->clear();
    }
    // Algorithm 1 immediately follows this kernel and masks every active RNS
    // coefficient uniformly before transfer.  The donor's extra CPU
    // flood_ciphertext corrupts the 112-bit live scale and is unnecessary at
    // this already-masked conversion boundary, so A0 does not apply it.
    cout << "VIT65_BSGS2 status=PASS backend=phantom_gpu"
         << " filter=" << data.filter_h << 'x' << data.filter_w
         << " baby_steps=" << baby_steps
         << " giant_steps=" << giant_steps
         << " input_cts=" << input_ciphertexts
         << " output_cts=" << output_ciphertexts
         << " diagonals=" << feature_blocks
         << " conversion_mask=uniform_rns flood=not_required" << endl;
    return;
#endif

    // start computation
    int num_diag = data.slot_count / data.image_size / 2;

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
        #pragma omp parallel for
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
    for (int k = 0; k < cts.size() * n2; k++) {
        int j = k / cts.size();
        int ct_i = k % cts.size();
        for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
            for (int i = 0; i < n1; i++) {
                UnifiedCiphertext ct1_l(loc);
                UnifiedCiphertext ct1_r(loc);
                const size_t diag = l * n1 * n2 + j * n1 + i;
                const auto &weight_l =
                    device_pp->cross_mat_single.first[diag][ct_i];
                const auto &weight_r =
                    device_pp->cross_mat_single.second[diag][ct_i];
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
    parms_id_type parms_id = result[0].hcipher().parms_id();
    shared_ptr<const SEALContext::ContextData> context_data = he->context->get_context_data(parms_id);
    #pragma omp parallel for
    for (int i = 0; i < result.size(); i++) {
        flood_ciphertext(result[i], context_data, SMUDGING_BITLEN_bert2);
        // he->evaluator->rescale_to_next_inplace(result[i]);
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
        const auto &bias = device_pp->cross_bias_single[l];
        require_device_plain_compatible(
            result[l], bias, CkksCtPtOperation::Add, "BSGS2.bias");
        he->devaluator->add_plain_inplace(result[l], bias);
    }
    // Preserve the original BLB smudging step exactly. Phantom currently has
    // no audited equivalent of flood_ciphertext, so cross the backend boundary
    // as ciphertext only; no intermediate plaintext is recovered.
    for (auto &cipher : result) {
        cipher.to_host(*he->context, *he->dcontext);
    }
    parms_id_type parms_id = result[0].hcipher().parms_id();
    auto context_data = he->context->get_context_data(parms_id);
    if (!context_data) {
        throw runtime_error("BSGS2 flooding target level is absent");
    }
    #pragma omp parallel for
    for (int i = 0; i < result.size(); ++i) {
        flood_ciphertext(result[i], context_data, SMUDGING_BITLEN_bert2);
    }
    for (auto &cipher : result) {
        cipher.to_device(*he->context, *he->dcontext);
    }
    cout << "CKKS_BACKEND_BOUNDARY op=BSGS2_FLOOD direction=D2H2D cts="
         << result.size() << endl;
    }
#ifdef BLB_HAS_HE_GPU
    if (device_cache) {
        device_cache->clear();
    }
#endif
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
    int rot_ours;
    if(INPUT_DIM==128){
        rot_ours=366;
    }
    else{
        rot_ours=183;
    }
    const auto loc = Cipher_plain_result[0].location();

#ifdef BLB_VIT65_HETERO
    // Evaluate both heads of each MHP pair together. Pairwise rotations by d
    // and d-T preserve the two 8192-slot head halves. The 2^52 public
    // prefix/tail masks into the ct-ct products removes the A0 half-extraction
    // rescale and leaves three limbs at scale 2^40 for HE-only AV.
    if (loc != LOCATION::DEVICE || data.image_size != 128 ||
        data.filter_w != 64 || data.slot_count != 16384 ||
        PACKING_NUM != 4 || Cipher_plain_result.size() != 6 ||
        results.size() < 4) {
        throw runtime_error("heterogeneous QK Phantom MHP geometry mismatch");
    }
    constexpr int tokens = 128;
    constexpr int head_dim = 64;
    constexpr int mask_scale_bits = 52;
    const size_t qkv_chain =
        Cipher_plain_result.front().dcipher().chain_index();
    vector<UnifiedPlaintext> prefix_masks(
        tokens, UnifiedPlaintext(LOCATION::DEVICE));
    vector<UnifiedPlaintext> tail_masks(
        tokens, UnifiedPlaintext(LOCATION::DEVICE));
    for (int diagonal = 0; diagonal < tokens; ++diagonal) {
        vector<double> prefix(data.slot_count, 0.0);
        vector<double> tail(data.slot_count, 0.0);
        for (int half = 0; half < 2; ++half) {
            const int base = half * data.slot_count / 2;
            fill(prefix.begin() + base,
                 prefix.begin() + base + tokens - diagonal, 1.0);
            fill(tail.begin() + base + tokens - diagonal,
                 tail.begin() + base + tokens, 1.0);
        }
        he->dencoder->encode(*he->dcontext, prefix,
                             ldexp(1.0, mask_scale_bits),
                             prefix_masks[diagonal], qkv_chain);
        he->dencoder->encode(*he->dcontext, tail,
                             ldexp(1.0, mask_scale_bits),
                             tail_masks[diagonal], qkv_chain);
    }

    vector<bool> output_initialized(PACKING_NUM, false);
    for (int packing = 0; packing < PACKING_NUM / 2; ++packing) {
        const auto &q_pair = Cipher_plain_result[packing];
        const auto &k_pair =
            Cipher_plain_result[packing + PACKING_NUM / 2];
        for (int diagonal = 0; diagonal < tokens; ++diagonal) {
            UnifiedCiphertext k_prefix(LOCATION::DEVICE);
            if (diagonal == 0) {
                k_prefix = k_pair;
            } else {
                he->devaluator->rotate_vector(
                    k_pair, diagonal, *he->dgal_keys, k_prefix);
            }
            UnifiedCiphertext product_prefix(LOCATION::DEVICE);
            he->devaluator->multiply(q_pair, k_prefix, product_prefix);
            he->devaluator->relinearize_inplace(product_prefix,
                                                 *he->drelin_keys);
            UnifiedCiphertext product_tail(LOCATION::DEVICE);
            if (diagonal != 0) {
                UnifiedCiphertext k_tail(LOCATION::DEVICE);
                he->devaluator->rotate_vector(
                    k_pair, diagonal - tokens, *he->dgal_keys, k_tail);
                he->devaluator->multiply(q_pair, k_tail, product_tail);
                he->devaluator->relinearize_inplace(product_tail,
                                                     *he->drelin_keys);
            }
            for (int feature_shift = head_dim / 2; feature_shift >= 1;
                 feature_shift /= 2) {
                UnifiedCiphertext rotated(LOCATION::DEVICE);
                he->devaluator->rotate_vector(
                    product_prefix, feature_shift * tokens,
                    *he->dgal_keys, rotated);
                he->devaluator->add_inplace(product_prefix, rotated);
                if (diagonal != 0) {
                    he->devaluator->rotate_vector(
                        product_tail, feature_shift * tokens,
                        *he->dgal_keys, rotated);
                    he->devaluator->add_inplace(product_tail, rotated);
                }
            }
            require_device_plain_compatible(
                product_prefix, prefix_masks[diagonal],
                CkksCtPtOperation::Multiply,
                "HETERO.QK.diagonal_prefix");
            he->devaluator->multiply_plain_inplace(
                product_prefix, prefix_masks[diagonal]);
            he->devaluator->rescale_to_next_inplace(product_prefix);
            if (diagonal != 0) {
                require_device_plain_compatible(
                    product_tail, tail_masks[diagonal],
                    CkksCtPtOperation::Multiply,
                    "HETERO.QK.diagonal_tail");
                he->devaluator->multiply_plain_inplace(
                    product_tail, tail_masks[diagonal]);
                he->devaluator->rescale_to_next_inplace(product_tail);
                he->devaluator->add_inplace(product_prefix, product_tail);
            }
            const int output_block = diagonal % head_dim;
            if (output_block != 0) {
                he->devaluator->rotate_vector_inplace(
                    product_prefix, -output_block * tokens,
                    *he->dgal_keys);
            }
            const int output_ct = packing * 2 + diagonal / head_dim;
            if (!output_initialized[output_ct]) {
                results[output_ct] = std::move(product_prefix);
                output_initialized[output_ct] = true;
            } else {
                he->devaluator->add_inplace(results[output_ct],
                                             product_prefix);
            }
        }
    }
    if (any_of(output_initialized.begin(), output_initialized.end(),
               [](bool value) { return !value; })) {
        throw runtime_error("heterogeneous QK output is incomplete");
    }
    for (int ciphertext = 0; ciphertext < PACKING_NUM; ++ciphertext) {
        he->devaluator->rescale_to_next_inplace(results[ciphertext]);
    }
    cout << "HETERO_QK_MATMUL status=PASS backend=phantom_gpu"
         << " layout=two_head_pairwise feature_reduction=64"
         << " output_cts=4 scale=" << log2(results.front().scale())
         << " limbs=" << results.front().coeff_modulus_size() << endl;
    return;
#elif defined(BLB_VIT65_A0)
    // The donor GPU path below leaves its feature-reduction rotations
    // commented out, so it computes Q .* rot(K), not QK^T.  A0 evaluates the
    // complete 64-term dot products on Phantom.  Q/K remain in the donor's
    // two-head MHP layout; each output ciphertext contains the same two heads
    // and 64 attention diagonals, exactly the layout consumed by the existing
    // public postprocess and Softmax boundary.
    if (loc != LOCATION::DEVICE || data.image_size != 128 ||
        data.filter_w != 64 || data.slot_count != 16384 ||
        PACKING_NUM != 4 || Cipher_plain_result.size() != 6 ||
        results.size() < 4) {
        throw runtime_error("A0 QK Phantom MHP geometry mismatch");
    }
    constexpr int tokens = 128;
    constexpr int head_dim = 64;
    constexpr int head_mask_scale_bits = 40;
    constexpr int diagonal_mask_scale_bits = 52;
    const size_t qkv_chain =
        Cipher_plain_result.front().dcipher().chain_index();

    vector<UnifiedPlaintext> head_masks(
        2, UnifiedPlaintext(LOCATION::DEVICE));
    for (int half = 0; half < 2; ++half) {
        vector<double> slots(data.slot_count, 0.0);
        fill(slots.begin() + half * data.slot_count / 2,
             slots.begin() + (half + 1) * data.slot_count / 2, 1.0);
        he->dencoder->encode(*he->dcontext, slots,
                             ldexp(1.0, head_mask_scale_bits),
                             head_masks[half], qkv_chain);
    }

    // After head extraction and the ct-ct product each ciphertext has dropped
    // two levels.  These real-scale masks select both the one feature block
    // holding the completed dot product and the non-wrapping token region.
    vector<vector<UnifiedPlaintext>> diagonal_prefix(
        2, vector<UnifiedPlaintext>(
               tokens, UnifiedPlaintext(LOCATION::DEVICE)));
    vector<vector<UnifiedPlaintext>> diagonal_tail(
        2, vector<UnifiedPlaintext>(
               tokens, UnifiedPlaintext(LOCATION::DEVICE)));
    for (int half = 0; half < 2; ++half) {
        const int source_block = half * head_dim;
        for (int diagonal = 0; diagonal < tokens; ++diagonal) {
            vector<double> prefix(data.slot_count, 0.0);
            vector<double> tail(data.slot_count, 0.0);
            const int base = source_block * tokens;
            fill(prefix.begin() + base,
                 prefix.begin() + base + (tokens - diagonal), 1.0);
            fill(tail.begin() + base + (tokens - diagonal),
                 tail.begin() + base + tokens, 1.0);
            he->dencoder->encode(*he->dcontext, prefix,
                                 ldexp(1.0, diagonal_mask_scale_bits),
                                 diagonal_prefix[half][diagonal],
                                 qkv_chain + 2);
            he->dencoder->encode(*he->dcontext, tail,
                                 ldexp(1.0, diagonal_mask_scale_bits),
                                 diagonal_tail[half][diagonal],
                                 qkv_chain + 2);
        }
    }

    vector<bool> output_initialized(PACKING_NUM, false);
    for (int packing = 0; packing < PACKING_NUM / 2; ++packing) {
        const auto &q_pair = Cipher_plain_result[packing];
        const auto &k_pair =
            Cipher_plain_result[packing + PACKING_NUM / 2];
        for (int half = 0; half < 2; ++half) {
            UnifiedCiphertext q_head(LOCATION::DEVICE);
            UnifiedCiphertext k_head(LOCATION::DEVICE);
            require_device_plain_compatible(
                q_pair, head_masks[half], CkksCtPtOperation::Multiply,
                "A0.QK.q_head_mask");
            require_device_plain_compatible(
                k_pair, head_masks[half], CkksCtPtOperation::Multiply,
                "A0.QK.k_head_mask");
            he->devaluator->multiply_plain(q_pair, head_masks[half], q_head);
            he->devaluator->multiply_plain(k_pair, head_masks[half], k_head);
            he->devaluator->rescale_to_next_inplace(q_head);
            he->devaluator->rescale_to_next_inplace(k_head);

            for (int diagonal = 0; diagonal < tokens; ++diagonal) {
                UnifiedCiphertext k_prefix(LOCATION::DEVICE);
                if (diagonal == 0) {
                    k_prefix = k_head;
                } else {
                    he->devaluator->rotate_vector(
                        k_head, diagonal, *he->dgal_keys, k_prefix);
                }
                UnifiedCiphertext product_prefix(LOCATION::DEVICE);
                he->devaluator->multiply(q_head, k_prefix, product_prefix);
                he->devaluator->relinearize_inplace(product_prefix,
                                                     *he->drelin_keys);
                he->devaluator->rescale_to_next_inplace(product_prefix);

                UnifiedCiphertext product_tail(LOCATION::DEVICE);
                if (diagonal != 0) {
                    UnifiedCiphertext k_tail(LOCATION::DEVICE);
                    he->devaluator->rotate_vector(
                        k_head, diagonal - tokens, *he->dgal_keys, k_tail);
                    he->devaluator->multiply(q_head, k_tail, product_tail);
                    he->devaluator->relinearize_inplace(product_tail,
                                                         *he->drelin_keys);
                    he->devaluator->rescale_to_next_inplace(product_tail);
                }

                // Only block 0 (lower head) or block 64 (upper head) is later
                // retained.  Positive power-of-two rotations make that block
                // contain the complete sum over all 64 feature blocks.
                for (int feature_shift = head_dim / 2; feature_shift >= 1;
                     feature_shift /= 2) {
                    UnifiedCiphertext rotated(LOCATION::DEVICE);
                    he->devaluator->rotate_vector(
                        product_prefix, feature_shift * tokens,
                        *he->dgal_keys, rotated);
                    he->devaluator->add_inplace(product_prefix, rotated);
                    if (diagonal != 0) {
                        he->devaluator->rotate_vector(
                            product_tail, feature_shift * tokens,
                            *he->dgal_keys, rotated);
                        he->devaluator->add_inplace(product_tail, rotated);
                    }
                }

                const auto &prefix_mask =
                    diagonal_prefix[half][diagonal];
                require_device_plain_compatible(
                    product_prefix, prefix_mask,
                    CkksCtPtOperation::Multiply,
                    "A0.QK.diagonal_prefix");
                he->devaluator->multiply_plain_inplace(product_prefix,
                                                        prefix_mask);
                he->devaluator->rescale_to_next_inplace(product_prefix);
                if (diagonal != 0) {
                    const auto &tail_mask = diagonal_tail[half][diagonal];
                    require_device_plain_compatible(
                        product_tail, tail_mask,
                        CkksCtPtOperation::Multiply,
                        "A0.QK.diagonal_tail");
                    he->devaluator->multiply_plain_inplace(product_tail,
                                                            tail_mask);
                    he->devaluator->rescale_to_next_inplace(product_tail);
                    he->devaluator->add_inplace(product_prefix, product_tail);
                }

                const int output_block = diagonal % head_dim;
                if (output_block != 0) {
                    he->devaluator->rotate_vector_inplace(
                        product_prefix, -output_block * tokens,
                        *he->dgal_keys);
                }
                const int output_ct = packing * 2 + diagonal / head_dim;
                if (!output_initialized[output_ct]) {
                    results[output_ct] = std::move(product_prefix);
                    output_initialized[output_ct] = true;
                } else {
                    he->devaluator->add_inplace(results[output_ct],
                                                 product_prefix);
                }
            }
        }
    }
    if (any_of(output_initialized.begin(), output_initialized.end(),
               [](bool value) { return !value; })) {
        throw runtime_error("A0 QK did not materialize all packed outputs");
    }
    cout << "A0_QK_MATMUL status=PASS backend=phantom_gpu"
         << " layout=two_head_64_diagonals feature_reduction=64"
         << " output_cts=4 scale=" << log2(results.front().scale())
         << " limbs=" << results.front().coeff_modulus_size() << endl;
    return;
#endif

    uint32_t num_rot=0;
    int packing_gap = data.image_size * data.filter_w / data.slot_count * 3;
    int temp_result_size = data.image_size * data.filter_w * 2 / data.slot_count;
    // Unit plaintexts keep the established BLB scale/level schedule around the
    // ct-ct packed MatMul; they are not model weights or security masks.
    const vector<double> pt_1(data.slot_count, 1.0);
    UnifiedPlaintext mask_1(loc);
    if (loc == LOCATION::HOST) {
        he->encoder->encode(pt_1, Cipher_plain_result[0].hcipher().parms_id(), pow(2, he->he_scale), mask_1);
    } else {
        he->dencoder->encode(*he->dcontext, pt_1, pow(2, he->he_scale), mask_1, Cipher_plain_result[0].dcipher().chain_index());
    }
    const vector<double> vec_mask(data.slot_count, 1.0);
    UnifiedPlaintext mask_2(loc);
    if (loc == LOCATION::HOST) {
        // ******************** CPU ********************
        auto util_context_data = he->context->get_context_data(Cipher_plain_result[0].hcipher().parms_id());
        util_context_data = util_context_data->next_context_data()->next_context_data();
        he->encoder->encode(vec_mask, util_context_data->parms_id(), pow(2,he->he_scale), mask_2);
    } else {
        // ******************** GPU ********************
        auto util_chain_index = Cipher_plain_result[0].dcipher().chain_index();
        util_chain_index += 2;
        he->dencoder->encode(*he->dcontext, vec_mask, pow(2,he->he_scale), mask_2, util_chain_index);
    }
    
    // begin computation
    // Preserve the repository's initial packed scale-alignment step.
    if (loc == LOCATION::HOST) {
    // ******************** CPU ********************
    #pragma omp parallel for num_threads(Cipher_plain_result.size())
    for(int i=0;i<Cipher_plain_result.size();i++){
        he->evaluator->multiply_plain_inplace(Cipher_plain_result[i], mask_1);
        he->evaluator->relinearize_inplace(Cipher_plain_result[i], *(he->relin_keys));
        he->evaluator->rescale_to_next_inplace(Cipher_plain_result[i]);
    }
    ct_pt_multiply_count += Cipher_plain_result.size();
    } else {
    // ******************** GPU ********************
    for(int i=0;i<Cipher_plain_result.size();i++){
        he->devaluator->multiply_plain_inplace(Cipher_plain_result[i], mask_1);
        he->devaluator->rescale_to_next_inplace(Cipher_plain_result[i]);
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
        int local_rotation = std::ceil(std::log2(data.slot_count / data.image_size / 2));

        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < data.image_size * 2; i++) {
            // no need to rotate, all rotations are done before
            // for (int k = 0; k < local_rotation; k++) {
            //     Ciphertext temp2;
            //     he->evaluator->rotate_vector(rotation_results[i], (int32_t) pow(2, k) * data.image_size, *(he->gal_keys), temp2);
            //     he->evaluator->add_inplace(rotation_results[i], temp2);
            //     num_rot++;
            // }
            he->evaluator->multiply_plain_inplace(rotation_results[i], mask_2);
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
    for (int packing_index = 0; packing_index < PACKING_NUM/2; packing_index++) {
        int rot_count = 0; /* FIXME: 0 */
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
        int local_rotation = std::ceil(std::log2(data.slot_count / data.image_size / 2));

        for (int i = 0; i < data.image_size * 2; i++) {
            // no need to rotate, all rotations are done before
            // for (int k = 0; k < local_rotation; k++) {
            //     Ciphertext temp2;
            //     he->evaluator->rotate_vector(rotation_results[i], (int32_t) pow(2, k) * data.image_size, *(he->gal_keys), temp2);
            //     he->evaluator->add_inplace(rotation_results[i], temp2);
            //     num_rot++;
            // }
            he->devaluator->multiply_plain_inplace(rotation_results[i], mask_2);
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
        // cout << "part 4 ok" << endl;
    }
    }
    ct_ct_multiply_count += static_cast<uint64_t>(PACKING_NUM / 2) *
        temp_result_size * data.image_size * 2;
    rotation_count += static_cast<uint64_t>(PACKING_NUM / 2) *
        std::min(rot_ours, temp_result_size * data.image_size * 2);
    ct_pt_multiply_count += static_cast<uint64_t>(PACKING_NUM / 2) *
        data.image_size * 2;
    cout << "num rotations: " << num_rot << endl;
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
    vector<UnifiedCiphertext> &result){
    int rot_ours;
#ifdef BLB_VIT65_A0
    // Every one of the 128 score diagonals must rotate V.  The donor's
    // rot_ours=72 shortcut substitutes unrotated V for the remaining 56
    // diagonals and is not a matrix multiplication.
    rot_ours = data.image_size;
#else
    if(INPUT_DIM==128){
        rot_ours = 3*24;
    }
    else if (INPUT_DIM==64){
        rot_ours = 3*16;
    }
    else{
        assert(0);
    }
#endif
    const auto loc = V[0].location();
#ifdef BLB_VIT65_A0
    if (loc != LOCATION::DEVICE || data.image_size != 128 ||
        data.filter_w != 64 || data.slot_count != 16384 || V.size() != 2 ||
        result.size() != 2 || softmax_s2.size() != 2) {
        throw runtime_error("A0 Softmax*V Phantom geometry mismatch");
    }
    vector<UnifiedCiphertext> aligned_v = V;
    for (auto &cipher : aligned_v) {
        he->devaluator->mod_switch_to_next_inplace(cipher);
    }
    for (int packing = 0; packing < PACKING_NUM / 2; ++packing) {
        if (softmax_s2[packing].size() != 2 ||
            softmax_s2[packing][0].size() !=
                static_cast<size_t>(data.image_size) ||
            softmax_s2[packing][1].size() !=
                static_cast<size_t>(data.image_size)) {
            throw runtime_error("A0 Softmax*V diagonal coverage mismatch");
        }
        bool initialized = false;
        for (int diagonal = 0; diagonal < data.image_size; ++diagonal) {
            UnifiedCiphertext v_prefix(LOCATION::DEVICE);
            if (diagonal == 0) {
                v_prefix = aligned_v[packing];
            } else {
                he->devaluator->rotate_vector(
                    aligned_v[packing], diagonal, *he->dgal_keys, v_prefix);
            }
            UnifiedCiphertext contribution(LOCATION::DEVICE);
            he->devaluator->multiply(
                v_prefix, softmax_s2[packing][0][diagonal], contribution);
            if (diagonal != 0) {
                UnifiedCiphertext v_tail(LOCATION::DEVICE);
                UnifiedCiphertext tail_contribution(LOCATION::DEVICE);
                he->devaluator->rotate_vector(
                    aligned_v[packing], diagonal - data.image_size,
                    *he->dgal_keys, v_tail);
                he->devaluator->multiply(
                    v_tail, softmax_s2[packing][1][diagonal],
                    tail_contribution);
                he->devaluator->add_inplace(contribution,
                                             tail_contribution);
            }
            if (!initialized) {
                result[packing] = std::move(contribution);
                initialized = true;
            } else {
                he->devaluator->add_inplace(result[packing], contribution);
            }
        }
        he->devaluator->relinearize_inplace(result[packing],
                                             *he->drelin_keys);
        UnifiedPlaintext alignment(LOCATION::DEVICE);
        he->dencoder->encode(
            *he->dcontext, vector<double>(data.slot_count, 1.0),
            pow(2, he->he_scale), alignment,
            result[packing].dcipher().chain_index());
        require_device_plain_compatible(
            result[packing], alignment.dplain(),
            CkksCtPtOperation::Multiply, "A0.AV.scale_alignment");
        he->devaluator->multiply_plain_inplace(result[packing], alignment);
        he->devaluator->rescale_to_next_inplace(result[packing]);
    }
    cout << "A0_SOFTMAX_V status=PASS backend=phantom_gpu"
         << " pairs=2 diagonals=128 prefix_tail_rotations=real"
         << " output_cts=2 scale=" << log2(result.front().scale())
         << " limbs=" << result.front().coeff_modulus_size() << endl;
    return;
#endif
    // encode weight once
    UnifiedPlaintext mask(loc);
    vector<double> vec_mask(data.slot_count,1.0);
    // start computation
    if (loc == LOCATION::HOST) {
    // ******************** CPU ********************
    he->encoder->encode(vec_mask, V[0].hcipher().parms_id(), pow(2,he->he_scale), mask);
    int num_diag = data.slot_count / data.image_size / 2;
    int num_matrix_per_col = data.filter_w / num_diag; // 1 or 2
    // cout << "num_diag, num_matrix_per_col: " << num_diag << " " << num_matrix_per_col << endl;
    // auto start = chrono::high_resolution_clock::now();
    
    #pragma omp parallel for num_threads(PACKING_NUM/2)
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; packing_ind++) {
        int rot_count = 0;

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
                if(rot_count<rot_ours){
                    temp_mult = rotation_by_one_depth3(he, data, V[packing_ind * num_matrix_per_col + ct_ind], i);
                    rot_count++;
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
    for (int i = 0; i < result.size(); i++) {
        he->evaluator->relinearize_inplace(result[i], *(he->relin_keys));
        // Preserve the repository's post-MatMul scale-alignment depth.
        he->evaluator->multiply_plain_inplace(result[i], mask);
        // he->evaluator->relinearize_inplace(result[i], *(he->relin_keys)); // FIXME: @relin
        he->evaluator->rescale_to_next_inplace(result[i]);
        // he->evaluator->mod_switch_to_next_inplace(result[i]);
        // he->evaluator->mod_switch_to_next_inplace(result[i]);
    }
    } else {
    // ******************** GPU ********************
    he->dencoder->encode(*he->dcontext, vec_mask, pow(2,he->he_scale), mask, V[0].dcipher().chain_index());
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; packing_ind++) {
        int rot_count = 0;
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
                if(rot_count < rot_ours){
                    UnifiedCiphertext ct1(loc);
                    UnifiedCiphertext ct2(loc);
                    he->devaluator->rotate_vector(V[packing_ind * num_matrix_per_col + ct_ind], i, *(he->dgal_keys), ct1);
                    he->devaluator->rotate_vector(V[packing_ind * num_matrix_per_col + ct_ind], -(data.image_size - i), *(he->dgal_keys), ct2);
                    he->devaluator->multiply(ct1, softmax_s2[packing_ind][0][i], temp1);
                    he->devaluator->multiply(ct2, softmax_s2[packing_ind][1][i], temp2);
                    rot_count++;
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

    for (int i = 0; i < result.size(); i++) {
        he->devaluator->relinearize_inplace(result[i], *(he->drelin_keys));
        // Preserve the repository's post-MatMul scale-alignment depth.
        he->devaluator->multiply_plain_inplace(result[i], mask);
        he->devaluator->rescale_to_next_inplace(result[i]);
    }  
    }
}

void Linear_ckks::hetero_scores_v(
    HE_CKKS *he,
    const vector<UnifiedCiphertext> &normalized_scores,
    const vector<UnifiedCiphertext> &V,
    const FCMetadata &data,
    size_t boundary_score_ciphertexts,
    vector<UnifiedCiphertext> &result) {
#ifndef BLB_VIT65_HETERO
    (void)he;
    (void)normalized_scores;
    (void)V;
    (void)data;
    (void)boundary_score_ciphertexts;
    (void)result;
    throw runtime_error("heterogeneous score*V is not enabled");
#else
    constexpr int logical_tokens = 65;
    if (he == nullptr || he != he1 || he->backend() != LOCATION::DEVICE ||
        data.image_size != 128 || data.filter_w != 64 ||
        data.slot_count != 16384 || normalized_scores.size() != 4 ||
        V.size() != 2 || result.size() != 2 ||
        boundary_score_ciphertexts > normalized_scores.size() ||
        boundary_score_ciphertexts % 2 != 0) {
        throw runtime_error("heterogeneous score*V Phantom geometry mismatch");
    }
    for (const auto &score : normalized_scores) {
        if (score.location() != LOCATION::DEVICE ||
            score.coeff_modulus_size() != 3 ||
            abs(log2(score.scale()) - 40.0) > 0.25) {
            throw runtime_error("heterogeneous score ciphertext metadata mismatch");
        }
    }
    const size_t score_limbs =
        normalized_scores.front().coeff_modulus_size();
    vector<UnifiedCiphertext> aligned_v = V;
    for (auto &cipher : aligned_v) {
        if (cipher.coeff_modulus_size() != 5 ||
            abs(log2(cipher.scale()) - 54.0) > 0.25) {
            throw runtime_error("heterogeneous V input metadata mismatch");
        }
        const double lift_scale = ldexp(1.0, 60) / cipher.scale();
        if (!isfinite(lift_scale) || lift_scale <= 0.0 ||
            abs(log2(lift_scale) - 6.0) > 0.25) {
            throw runtime_error("heterogeneous V public scale lift is unsafe");
        }
        UnifiedPlaintext public_one(LOCATION::DEVICE);
        he->dencoder->encode(
            *he->dcontext, vector<double>(data.slot_count, 1.0),
            lift_scale, public_one, cipher.dcipher().chain_index());
        require_device_plain_compatible(
            cipher, public_one.dplain(), CkksCtPtOperation::Multiply,
            "HETERO.AV.v_public_one_lift");
        he->devaluator->multiply_plain_inplace(cipher, public_one);
        while (cipher.coeff_modulus_size() > score_limbs) {
            he->devaluator->mod_switch_to_next_inplace(cipher);
        }
        if (cipher.coeff_modulus_size() != score_limbs ||
            abs(log2(cipher.scale()) - 60.0) > 0.25) {
            throw runtime_error("heterogeneous V modulus/scale alignment failed");
        }
    }
    cout << setprecision(17)
         << "HETERO_AV_SCALE_ALIGNMENT status=PASS imported_score="
         << normalized_scores[0].scale()
         << " boundary_score_cts=" << boundary_score_ciphertexts
         << " v_pair0=" << aligned_v[0].scale()
         << " v_pair1=" << aligned_v[1].scale() << endl;

    const int tokens = data.image_size;
    const int diagonals_per_ct = data.slot_count / tokens / 2;
    vector<vector<vector<UnifiedCiphertext>>> packed_scores(
        PACKING_NUM / 2,
        vector<vector<UnifiedCiphertext>>(
            2, vector<UnifiedCiphertext>(
                   tokens, UnifiedCiphertext(LOCATION::DEVICE))));

    for (int packing = 0; packing < PACKING_NUM / 2; ++packing) {
        for (int diagonal = 0; diagonal < tokens; ++diagonal) {
            const int source_ct = packing * 2 +
                                  diagonal / diagonals_per_ct;
            const int local_block = diagonal % diagonals_per_ct;
            const size_t chain =
                normalized_scores[source_ct].dcipher().chain_index();
            for (int split = 0; split < 2; ++split) {
                vector<double> slots(data.slot_count, 0.0);
                for (int half = 0; half < 2; ++half) {
                    const int block_start = half * data.slot_count / 2 +
                                            local_block * tokens;
                    for (int row = 0; row < tokens; ++row) {
                        const bool prefix = row < tokens - diagonal;
                        if ((split == 0) != prefix) continue;
                        double value = 1.0;
                        if (static_cast<size_t>(source_ct) >=
                            boundary_score_ciphertexts) {
                            const int key = prefix
                                ? row + diagonal
                                : row + diagonal - tokens;
                            value = row < logical_tokens &&
                                            key < logical_tokens
                                        ? 1.0 / logical_tokens
                                        : 0.0;
                        }
                        slots[block_start + row] = value;
                    }
                }
                UnifiedPlaintext mask(LOCATION::DEVICE);
                const double mask_scale =
                    aligned_v[packing].scale() /
                    normalized_scores[source_ct].scale();
                if (!isfinite(mask_scale) || mask_scale <= 0.0 ||
                    abs(log2(mask_scale) - 20.0) > 0.25) {
                    throw runtime_error(
                        "heterogeneous AV dynamic mask scale is unsafe");
                }
                he->dencoder->encode(*he->dcontext, slots,
                                     mask_scale, mask,
                                     chain);
                require_device_plain_compatible(
                    normalized_scores[source_ct], mask.dplain(),
                    CkksCtPtOperation::Multiply,
                    "HETERO.AV.score_diagonal");
                UnifiedCiphertext replicated(LOCATION::DEVICE);
                he->devaluator->multiply_plain(
                    normalized_scores[source_ct], mask, replicated);
                if (local_block != 0) {
                    he->devaluator->rotate_vector_inplace(
                        replicated, local_block * tokens,
                        *he->dgal_keys);
                }
                for (int width = 1; width < diagonals_per_ct; width *= 2) {
                    UnifiedCiphertext shifted(LOCATION::DEVICE);
                    he->devaluator->rotate_vector(
                        replicated, -width * tokens,
                        *he->dgal_keys, shifted);
                    he->devaluator->add_inplace(replicated, shifted);
                }
                packed_scores[packing][split][diagonal] =
                    std::move(replicated);
            }
        }
    }

    for (int packing = 0; packing < PACKING_NUM / 2; ++packing) {
        bool initialized = false;
        for (int diagonal = 0; diagonal < tokens; ++diagonal) {
            UnifiedCiphertext v_prefix(LOCATION::DEVICE);
            if (diagonal == 0) {
                v_prefix = aligned_v[packing];
            } else {
                he->devaluator->rotate_vector(
                    aligned_v[packing], diagonal, *he->dgal_keys,
                    v_prefix);
            }
            UnifiedCiphertext contribution(LOCATION::DEVICE);
            he->devaluator->multiply(
                v_prefix, packed_scores[packing][0][diagonal],
                contribution);
            if (diagonal != 0) {
                UnifiedCiphertext v_tail(LOCATION::DEVICE);
                UnifiedCiphertext tail_contribution(LOCATION::DEVICE);
                he->devaluator->rotate_vector(
                    aligned_v[packing], diagonal - tokens,
                    *he->dgal_keys, v_tail);
                he->devaluator->multiply(
                    v_tail, packed_scores[packing][1][diagonal],
                    tail_contribution);
                he->devaluator->add_inplace(contribution,
                                             tail_contribution);
            }
            if (!initialized) {
                result[packing] = std::move(contribution);
                initialized = true;
            } else {
                he->devaluator->add_inplace(result[packing], contribution);
            }
        }
        he->devaluator->relinearize_inplace(result[packing],
                                             *he->drelin_keys);
        he->devaluator->rescale_to_next_inplace(result[packing]);
    }
    cout << "HETERO_SCORE_V status=PASS backend=phantom_gpu"
         << " assignment=relu,linear_spill,linear,linear"
         << " direct_pair_factor=1/65 valid_keys=65"
         << " output_cts=2 scale=" << log2(result.front().scale())
         << " limbs=" << result.front().coeff_modulus_size() << endl;
#endif
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

    double mask_scale = pow(2, he->he_scale);
#ifdef BLB_VIT65_A0
    // The mask is followed immediately by a rescale.  Encoding this public
    // binary mask at the exact dropped modulus preserves the imported score's
    // 2^43 scale, which Phantom requires to equal V's scale for ct-ct multiply.
    mask_scale = static_cast<double>(
        he->context->first_context_data()->parms().coeff_modulus().back().value());
#endif
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
        he->encoder->encode(m1, mask_scale, pt);
        mask[0][i] = pt;
        he->encoder->encode(m2, mask_scale, pt);
        mask[1][i] = pt;
    }
    return mask;
}

void Linear_ckks::preprocess_softmax_s1_ct_ct(
    HE_CKKS* he, 
    const vector<UnifiedCiphertext> &matrix, 
    const FCMetadata &data,
    vector<vector<vector<UnifiedCiphertext>>> &s2_pack,
    const vector<vector<Plaintext>> *diagonal_masks) {
    const auto loc = matrix[0].location();
    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;
    s2_pack.resize(
        PACKING_NUM/2,
        vector<vector<UnifiedCiphertext>>(
            2, vector<UnifiedCiphertext>(
                num_diag, UnifiedCiphertext(loc))
    ));

#ifdef BLB_VIT65_A0
    if (diagonal_masks != nullptr) {
        if (loc != LOCATION::DEVICE || data.image_size != 128 ||
            data.slot_count != 16384 || matrix.size() != 4 ||
            diagonal_masks->size() != 2 ||
            (*diagonal_masks)[0].size() != static_cast<size_t>(num_diag) ||
            (*diagonal_masks)[1].size() != static_cast<size_t>(num_diag)) {
            throw runtime_error("A0 Softmax diagonal extraction geometry mismatch");
        }
        for (int packing = 0; packing < PACKING_NUM / 2; ++packing) {
            for (int diagonal = 0; diagonal < num_diag; ++diagonal) {
                const int source_ct = packing * 2 +
                                      diagonal / num_diag_per_ct;
                const int local_block = diagonal % num_diag_per_ct;
                for (int split = 0; split < 2; ++split) {
                    UnifiedPlaintext mask((*diagonal_masks)[split][diagonal]);
                    mask.to_device(*he->context, *he->dcontext);
                    require_device_plain_compatible(
                        matrix[source_ct], mask.dplain(),
                        CkksCtPtOperation::Multiply,
                        split == 0 ? "A0.AV.score_prefix"
                                   : "A0.AV.score_tail");
                    UnifiedCiphertext replicated(LOCATION::DEVICE);
                    he->devaluator->multiply_plain(
                        matrix[source_ct], mask, replicated);
                    he->devaluator->rescale_to_next_inplace(replicated);

                    // Move the selected diagonal block to block 0 in each
                    // 64-block MHP half, then duplicate it within each half.
                    // The occupied ranges double as [0,1), [0,2), ...,
                    // [0,64), so none of these rotations mixes the two heads.
                    if (local_block != 0) {
                        he->devaluator->rotate_vector_inplace(
                            replicated, local_block * data.image_size,
                            *he->dgal_keys);
                    }
                    for (int width = 1; width < num_diag_per_ct;
                         width *= 2) {
                        UnifiedCiphertext shifted(LOCATION::DEVICE);
                        he->devaluator->rotate_vector(
                            replicated, -width * data.image_size,
                            *he->dgal_keys, shifted);
                        he->devaluator->add_inplace(replicated, shifted);
                    }
                    s2_pack[packing][split][diagonal] =
                        std::move(replicated);
                }
            }
        }
        cout << "A0_SOFTMAX_DIAGONAL_PACK status=PASS backend=phantom_gpu"
             << " pairs=2 diagonals=128 features_per_head=64"
             << " prefix_tail=2" << endl;
        return;
    }
#endif
    vector<double> vec_w(he->num_slots,0.5);
    UnifiedPlaintext w1(loc);
    if (loc == LOCATION::HOST) {
    he->encoder->encode(vec_w, pow(2, he->he_scale), w1);
    #pragma omp parallel for num_threads(PACKING_NUM/2)
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; packing_ind++) {
        // cout << "packing_ind: " << packing_ind << endl;
        auto &weightMatrix1 = s2_pack[packing_ind];
        // cout << "OK0" << endl;
        #pragma omp parallel for num_threads(num_diag/2)
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {
            // int cur_diag = (packing_ind * num_diag_per_ct + diag_ind) % num_diag;
            int cts_ind = packing_ind * data.image_size * data.image_size * 2 / data.slot_count + diag_ind / num_diag_per_ct;
            Ciphertext cur_ct = matrix[cts_ind];
            Plaintext mask1 = w1;
            Plaintext mask2 = w1;
            Ciphertext cur_ct_l, cur_ct_r;
            cur_ct_l = Ciphertext(cur_ct);
            cur_ct_r = Ciphertext(cur_ct);
            // he->evaluator->multiply_plain(cur_ct, mask1, cur_ct_l);
            // he->evaluator->multiply_plain(cur_ct, mask2, cur_ct_r);
            // cout << "OK1" << endl;
            // we do not need rotation, since we pack in plaintext
            // for (int j = 0; j < std::log2(num_diag_per_ct); j++) {
            //     Ciphertext temp_ct;
            //     he->evaluator->rotate_vector(cur_ct_l, (int64_t)num_diag * std::pow(2, j), *(he->gal_keys), temp_ct);
            //     he->evaluator->add_inplace(cur_ct_l, temp_ct);
            //     he->evaluator->rotate_vector(cur_ct_r, (int64_t)num_diag * std::pow(2, j), *(he->gal_keys), temp_ct);
            //     he->evaluator->add_inplace(cur_ct_r, temp_ct);
            // }
            // cout << "OK2" << endl;
            weightMatrix1[0][diag_ind] = cur_ct_l;
            weightMatrix1[1][diag_ind] = cur_ct_r;
        }
    }
    } else {
    for (int packing_ind = 0; packing_ind < PACKING_NUM/2; packing_ind++) {
        // cout << "packing_ind: " << packing_ind << endl;
        auto &weightMatrix1 = s2_pack[packing_ind];
        // cout << "OK0" << endl;
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {
            // int cur_diag = (packing_ind * num_diag_per_ct + diag_ind) % num_diag;
            int cts_ind = packing_ind * data.image_size * data.image_size * 2 / data.slot_count + diag_ind / num_diag_per_ct;
            UnifiedPlaintext mask1(w1);
            UnifiedPlaintext mask2(w1);
            UnifiedCiphertext cur_ct_l(matrix[cts_ind]);
            UnifiedCiphertext cur_ct_r(matrix[cts_ind]);
            // he->evaluator->multiply_plain(cur_ct, mask1, cur_ct_l);
            // he->evaluator->multiply_plain(cur_ct, mask2, cur_ct_r);
            // cout << "OK1" << endl;
            // we do not need rotation, since we pack in plaintext
            // for (int j = 0; j < std::log2(num_diag_per_ct); j++) {
            //     Ciphertext temp_ct;
            //     he->evaluator->rotate_vector(cur_ct_l, (int64_t)num_diag * std::pow(2, j), *(he->gal_keys), temp_ct);
            //     he->evaluator->add_inplace(cur_ct_l, temp_ct);
            //     he->evaluator->rotate_vector(cur_ct_r, (int64_t)num_diag * std::pow(2, j), *(he->gal_keys), temp_ct);
            //     he->evaluator->add_inplace(cur_ct_r, temp_ct);
            // }
            // cout << "OK2" << endl;
            weightMatrix1[0][diag_ind] = std::move(cur_ct_l);
            weightMatrix1[1][diag_ind] = std::move(cur_ct_r);
        }
    }
    }
}

Linear_ckks::Linear_ckks(){

}

Linear_ckks::Linear_ckks(int party, sci::NetIO *io, LOCATION backend) {
	this->party = party;
	this->io = io;

    if (backend != LOCATION::HOST && backend != LOCATION::DEVICE) {
        throw std::invalid_argument("Bert_ckks: backend must be HOST or DEVICE");
    }

    set_HE(backend);
    pp_1.resize(ATTENTION_LAYERS);
    pp_2.resize(ATTENTION_LAYERS);
    pp_2_he1.resize(ATTENTION_LAYERS);
    pp_3.resize(ATTENTION_LAYERS);
    pp_4.resize(ATTENTION_LAYERS);

    w_ln_1_pt.resize(ATTENTION_LAYERS);
    w_ln_2_pt.resize(ATTENTION_LAYERS);

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

void Linear_ckks::set_HE(LOCATION backend){
    cout << "he0 ok" << endl;
	this->he1 = new HE_CKKS(
        party, /* Bob or Alice (Client or Server) */
        io, /* Network */
        32768, /* N */
        std::vector<int>{60, 34, 60, 60, 60, 60, 60}, /* modulus chain */
        20, /* fft_scale */
        44, /* he_scale */
        backend /* Evaluator backend */
    );
    cout << "he1 ok" << endl;
    this->he2 = new HE_CKKS(
        party, /* Bob or Alice (Client or Server) */
        io, /* Network */
        32768, /* N */
        std::vector<int>{60, 60, 60, 60, 60, 60, 60}, /* modulus chain */
        20, /* fft_scale */
        43, /* he_scale */
        backend /* Evaluator backend */
    );
    cout << "he2 ok" << endl;
    this->he3 = new HE_CKKS(
        party, /* Bob or Alice (Client or Server) */
        io, /* Network */
        32768, /* N */
        std::vector<int>{60, 32, 60, 60, 60, 60, 60, 60}, /* modulus chain */
        20, /* fft_scale */
        46, /* he_scale */
        backend /* Evaluator backend */
    );
    cout << "he3 ok" << endl;
    this->he4 = new HE_CKKS(
        party, /* Bob or Alice (Client or Server) */
        io, /* Network */
        16384, /* N */
        std::vector<int>{60, 34, 60, 60, 60, 60}, /* modulus chain */
        20, /* fft_scale */
        38, /* he_scale */
        backend /* Evaluator backend */
    );
    this->he_softmax = new HE_CKKS(
        party, /* Bob or Alice (Client or Server) */
        io, /* Network */
        32768, /* N */
        std::vector<int>{60, 40, 40, 40, 40, 40, 40, 40, 60}, /* modulus chain */
        20, /* fft_scale */
        40, /* he_scale */
        backend /* Evaluator backend */
    );
    this->he_onetime = new HE_CKKS(
        party, /* Bob or Alice (Client or Server) */
        io, /* Network */
        8192, /* N */
        std::vector<int>{60, 40, 60}, /* modulus chain */
        20, /* fft_scale */
        40, /* he_scale */
        backend /* Evaluator backend */
    );
    cout << "he initial ok" << endl;
}

// plain_mod_bitwidth is the bitwidth of plain_mod
void Linear_ckks::set_HE_bfv(size_t poly_modulus_degree,vector<int> coeff_bit_sizes, uint64_t plain_mod_bitwidth){
    this->he_bfv = new HE(party, io, poly_modulus_degree, coeff_bit_sizes, plain_mod_bitwidth, true);
}
