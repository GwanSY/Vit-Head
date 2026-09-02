// clang-format off
/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// #include "FloatingPoint/fp-math.h"
// #include "FloatingPoint/fixed-point.h"
// #include "BuildingBlocks/aux-protocols.h"
// #include "NonLinear/argmax.h"
#include "ckks_bert.h"
#include "ckks_bert_basic.cpp"
#include "nvtx3/nvToolsExt.h"
#include <cmath>
#include <iostream>

using namespace sci;
using namespace std;
using namespace seal;

#define BOB sci::BOB
// #define SCI_HE

namespace {

// Keep the final CKKS share at high precision through Algorithm 1.  The
// CKKS-to-MPC adapter performs the secure coefficient-domain truncation from
// this scale to each tensor's semantic fixed-point scale (normally s12, V
// uses s11); lowering the ciphertext itself to s12 before decryption would
// amplify CKKS noise in the subsequent fixed FFT.
// Key-switched ViT ciphertexts can have centered coefficient representatives
// close to the 60-bit conversion prime even when their decoded slots are
// small.  At s40, the 20-bit local FFT roots leave too little precision after
// the coefficient-domain mask is decoded to s12.  Use the highest practical
// conversion scale that still leaves 11 signed integer bits at the final
// 60-bit prime; all audited ViT CKKS/MPC boundaries are well inside that
// public range.  Together with the 26-bit local roots below this suppresses
// the former conversion error by roughly 2^15 without changing any semantic
// tensor scale.
constexpr int kFormalConversionScaleBits = 49;

// Work backwards from the scale required by Algorithm 1 instead of reusing
// the scale of the (absent) fused LayerNorm.  The layer-0 QK path consumes
// four rescale levels:
//
//   X --(QKV weight h, drop d0)--> X+h-d0
//     --(public 1 at h, drop d1)--> X+2h-d0-d1
//     --(Q*K, drop d2)-----------> 2X+4h-2d0-2d1-d2
//     --(public 1 at h, drop d3)-> 2X+5h-2d0-2d1-d2-d3.
//
// X is numerical CKKS precision only; the MPC tensor remains an s5 value.
// Choosing X from the live modulus chain keeps QK at (at least) s40 before
// masking/decryption and avoids the former s4 ciphertext followed by a
// precision-free metadata lift.  With he1's current chain this derives X=90.
int layer0_qkv_import_scale_bits(HE_CKKS *he, int qk_target_scale_bits) {
    if (he == nullptr || qk_target_scale_bits <= 0) {
        throw runtime_error("formal layer-0 QKV scale planner input is invalid");
    }
    auto level = he->context->first_context_data();
    vector<int> dropped_bits;
    dropped_bits.reserve(4);
    for (int rescale = 0; rescale < 4; ++rescale) {
        if (!level || level->parms().coeff_modulus().size() < 2) {
            throw runtime_error(
                "formal layer-0 QK path has fewer than four rescale levels");
        }
        dropped_bits.push_back(
            level->parms().coeff_modulus().back().bit_count());
        level = level->next_context_data();
    }

    const int h = he->he_scale;
    const int numerator =
        qk_target_scale_bits - 5 * h + 2 * dropped_bits[0] +
        2 * dropped_bits[1] + dropped_bits[2] + dropped_bits[3];
    if (numerator <= 0) {
        throw runtime_error("formal layer-0 QKV scale schedule is non-positive");
    }
    // Moduli of a requested bit count are slightly below 2^bits. Rounding X
    // upward therefore guarantees that the native QK scale is not below the
    // requested conversion precision; prepare_ckks_to_mpc_inplace performs
    // the exact public-one alignment if the real prime values leave a small
    // excess.
    const int input_scale_bits = (numerator + 1) / 2;
    const int qkv_scale_bits =
        input_scale_bits + h - dropped_bits[0];
    const int q_after_unit_bits =
        qkv_scale_bits + h - dropped_bits[1];
    const int qk_after_ctct_bits =
        2 * q_after_unit_bits - dropped_bits[2];
    const int qk_final_bits =
        qk_after_ctct_bits + h - dropped_bits[3];
    const auto first = he->context->first_context_data();
    if (!first || input_scale_bits <= 0 || qkv_scale_bits <= 0 ||
        q_after_unit_bits <= 0 || qk_after_ctct_bits <= 0 ||
        qk_final_bits < qk_target_scale_bits ||
        input_scale_bits + h >=
            first->total_coeff_modulus_bit_count()) {
        throw runtime_error("formal layer-0 QKV scale schedule is unsafe");
    }
    cout << "FORMAL_QK_SCALE_PLAN status=PASS input_scale_bits="
         << input_scale_bits << " qkv_scale_bits=" << qkv_scale_bits
         << " q_after_unit_bits=" << q_after_unit_bits
         << " qk_after_ctct_bits=" << qk_after_ctct_bits
         << " qk_final_bits=" << qk_final_bits
         << " target_scale_bits=" << qk_target_scale_bits
         << " dropped_modulus_bits=" << dropped_bits[0] << ','
         << dropped_bits[1] << ',' << dropped_bits[2] << ','
         << dropped_bits[3] << endl;
    return input_scale_bits;
}

// The exported LayerNorm tensors are signed fixed-point integers (s12 in the
// BERT artifact). FixArray::s is their semantic scale. The CKKS plaintext
// scale selected at encode time is only numerical precision, so encoding the
// raw integer would multiply the model parameter by 2^s.
double layernorm_fixed_to_real(uint64_t raw, const FixArray &parameter,
                               const char *name) {
    if (!parameter.signed_ || parameter.ell <= 0 || parameter.ell > 63 ||
        parameter.s < 0 || parameter.s > 62) {
        throw runtime_error(string("invalid fixed-point metadata for ") + name);
    }
    return ldexp(static_cast<double>(signed_val(raw, parameter.ell)),
                 -parameter.s);
}

void require_layernorm_parameter_scale(const FixArray &x,
                                       const FixArray &sigma,
                                       const FixArray &w,
                                       const FixArray &b) {
    if (sigma.s != x.s || w.s != x.s || b.s != x.s ||
        sigma.ell != x.ell || w.ell != x.ell || b.ell != x.ell) {
        throw runtime_error(
            "formal LayerNorm fixed-point parameter metadata mismatch");
    }
}

void import_target_scale(Bert_ckks &owner,
                         const vector<vector<HE_CKKS::scalar_t>> &shares,
                         int slot, int ell, int input_scale,
                         int target_scale, HE_CKKS *he,
                         vector<UnifiedPlaintext> &destination,
                         const char *stage) {
    const auto previous_conversion_bucket = owner.profile_conversion_bucket();
    if (string(stage) == "softmax") {
        owner.profile_set_conversion_bucket(
            BlbViTCommAuditBucket::SOFTMAX_TO_CKKS_CONVERSION);
    }
    vector<Plaintext> host_plaintexts = owner.mpc_to_ckks(
        shares, slot, he->get_conversion_prime(), ell, input_scale,
        target_scale, he, false);
    owner.profile_set_conversion_bucket(previous_conversion_bucket);
    const auto first_data = he->context->first_context_data();
    if (host_plaintexts.empty() || !first_data) {
        throw runtime_error(string("formal MPC-to-CKKS import is unavailable: ") +
                            stage);
    }
    const size_t expected_limbs =
        first_data->parms().coeff_modulus().size();
    const double expected_scale = ldexp(1.0, target_scale);
    destination.resize(host_plaintexts.size());
    for (size_t i = 0; i < host_plaintexts.size(); ++i) {
        const auto context_data =
            he->context->get_context_data(host_plaintexts[i].parms_id());
        const double relative_error =
            abs(host_plaintexts[i].scale() - expected_scale) /
            expected_scale;
        if (!context_data ||
            host_plaintexts[i].parms_id() != first_data->parms_id() ||
            context_data->parms().coeff_modulus().size() != expected_limbs ||
            !isfinite(relative_error) || relative_error > 1e-12) {
            throw runtime_error(string("formal MPC-to-CKKS import metadata mismatch: ") +
                                stage);
        }
        destination[i] = std::move(host_plaintexts[i]);
    }
    cout << "FORMAL_MPC_TO_CKKS_IMPORT stage=" << stage
         << " status=PASS ell=" << ell
         << " fixed_scale=" << input_scale
         << " target_scale=" << target_scale
         << " actual_scale_bits=" << log2(expected_scale)
         << " limbs=" << expected_limbs << " cts=" << destination.size()
         << endl;
}

void require_uniform_unified_metadata(
    const vector<UnifiedCiphertext> &ciphertexts, const char *stage) {
    if (ciphertexts.empty()) {
        throw runtime_error(string("formal ciphertext vector is empty: ") +
                            stage);
    }
    const LOCATION location = ciphertexts.front().location();
    const size_t limbs = ciphertexts.front().coeff_modulus_size();
    const double scale = ciphertexts.front().scale();
    if (!isfinite(scale) || scale <= 0.0) {
        throw runtime_error(string("formal ciphertext scale is invalid: ") +
                            stage);
    }
    for (const auto &ciphertext : ciphertexts) {
        const double relative_error = abs(ciphertext.scale() - scale) / scale;
        if (ciphertext.location() != location ||
            ciphertext.coeff_modulus_size() != limbs ||
            !isfinite(relative_error) || relative_error > 1e-12) {
            throw runtime_error(string("formal ciphertext metadata mismatch: ") +
                                stage);
        }
        if (location == LOCATION::HOST) {
            if (ciphertext.hcipher().parms_id() !=
                ciphertexts.front().hcipher().parms_id()) {
                throw runtime_error(string("formal ciphertext parms mismatch: ") +
                                    stage);
            }
        } else if (location == LOCATION::DEVICE) {
            if (ciphertext.dcipher().chain_index() !=
                ciphertexts.front().dcipher().chain_index()) {
                throw runtime_error(string("formal ciphertext chain mismatch: ") +
                                    stage);
            }
        } else {
            throw runtime_error(string("formal ciphertext location is invalid: ") +
                                stage);
        }
    }
}

void lift_unified_with_actual_one(Linear_ckks &linear, HE_CKKS *he,
                                  vector<UnifiedCiphertext> &ciphertexts,
                                  double target_scale, const char *stage) {
    require_uniform_unified_metadata(ciphertexts, stage);
    const LOCATION location = ciphertexts.front().location();
    const double source_scale = ciphertexts.front().scale();
    const size_t source_limbs = ciphertexts.front().coeff_modulus_size();
    const double unit_scale = target_scale / source_scale;
    if (!isfinite(target_scale) || target_scale < source_scale ||
        !isfinite(unit_scale) || unit_scale < 1.0) {
        throw runtime_error(string("formal actual-one lift is invalid: ") +
                            stage);
    }
    const vector<double> public_one(he->num_slots, 1.0);
    UnifiedPlaintext unit_plain(location);
    if (location == LOCATION::HOST) {
        he->encoder->encode(public_one,
                            ciphertexts.front().hcipher().parms_id(),
                            unit_scale, unit_plain);
        for (auto &ciphertext : ciphertexts) {
            he->evaluator->multiply_plain_inplace(ciphertext, unit_plain);
        }
    } else if (location == LOCATION::DEVICE) {
        const auto chain_index =
            ciphertexts.front().dcipher().chain_index();
        he->dencoder->encode(*he->dcontext, public_one, unit_scale,
                             unit_plain, chain_index);
        for (auto &ciphertext : ciphertexts) {
            he->devaluator->multiply_plain_inplace(ciphertext, unit_plain);
        }
    } else {
        throw runtime_error("formal actual-one lift backend is invalid");
    }
    linear.ct_pt_multiply_count += ciphertexts.size();
    require_uniform_unified_metadata(ciphertexts, stage);
    const double relative_error =
        abs(ciphertexts.front().scale() - target_scale) / target_scale;
    if (ciphertexts.front().coeff_modulus_size() != source_limbs ||
        !isfinite(relative_error) || relative_error > 1e-12) {
        throw runtime_error(string("formal actual-one lift failed: ") + stage);
    }
    cout << "FORMAL_ACTUAL_ONE_LIFT stage=" << stage
         << " status=PASS source_scale_bits=" << log2(source_scale)
         << " unit_scale_bits=" << log2(unit_scale)
         << " target_scale_bits=" << log2(target_scale)
         << " limbs=" << source_limbs << " cts=" << ciphertexts.size()
         << " plaintext_value=1" << endl;
}

void align_unified_to_cipher_metadata(
    Linear_ckks &linear, HE_CKKS *he,
    vector<UnifiedCiphertext> &ciphertexts,
    const vector<UnifiedCiphertext> &target, const char *stage) {
    require_uniform_unified_metadata(target, stage);
    lift_unified_with_actual_one(linear, he, ciphertexts,
                                 target.front().scale(), stage);
    const size_t target_limbs = target.front().coeff_modulus_size();
    const LOCATION location = ciphertexts.front().location();
    for (auto &ciphertext : ciphertexts) {
        while (ciphertext.coeff_modulus_size() > target_limbs) {
            if (location == LOCATION::HOST) {
                he->evaluator->mod_switch_to_next_inplace(ciphertext);
            } else {
                he->devaluator->mod_switch_to_next_inplace(ciphertext);
            }
        }
        if (ciphertext.coeff_modulus_size() != target_limbs) {
            throw runtime_error(string("formal level alignment crossed target: ") +
                                stage);
        }
    }
    require_uniform_unified_metadata(ciphertexts, stage);
    bool same_level = false;
    if (location == LOCATION::HOST) {
        same_level = ciphertexts.front().hcipher().parms_id() ==
                     target.front().hcipher().parms_id();
    } else {
        same_level = ciphertexts.front().dcipher().chain_index() ==
                     target.front().dcipher().chain_index();
    }
    const double relative_error =
        abs(ciphertexts.front().scale() - target.front().scale()) /
        target.front().scale();
    if (!same_level || !isfinite(relative_error) || relative_error > 1e-12) {
        throw runtime_error(string("formal add metadata alignment failed: ") +
                            stage);
    }
    cout << "FORMAL_ADD_ALIGNMENT stage=" << stage
         << " status=PASS actual_scale_bits="
         << log2(ciphertexts.front().scale()) << " limbs=" << target_limbs
         << " cts=" << ciphertexts.size() << endl;
}

void prepare_host_conversion_from_actual_scale(
    Bert_ckks &owner, vector<Ciphertext> &ciphertexts, int target_scale_bits,
    HE_CKKS *he, const char *stage) {
    if (ciphertexts.empty()) {
        throw runtime_error(string("formal conversion output is empty: ") +
                            stage);
    }
    const double target_scale = ldexp(1.0, target_scale_bits);
    const size_t initial_limbs = ciphertexts.front().coeff_modulus_size();
    const double initial_scale = ciphertexts.front().scale();
    size_t preliminary_rescales = 0;
    bool upward_actual_one_lift = false;
    // Phantom can return ciphertexts that have the same modulus count and
    // scale but different SEAL parms_id values.  They are still valid inputs
    // to the original per-ciphertext CKKS->MPC preparation.  Rescale each
    // ciphertext against its own current context instead of requiring a
    // cross-ciphertext parms_id match at this migration boundary.
    while (true) {
        bool rescaled_any = false;
        for (auto &ciphertext : ciphertexts) {
            const auto context_data =
                he->context->get_context_data(ciphertext.parms_id());
            if (!context_data) {
                throw runtime_error(
                    string("formal conversion level is invalid: ") + stage);
            }
            const auto &moduli = context_data->parms().coeff_modulus();
            if (moduli.size() >= 2 &&
                target_scale * static_cast<double>(moduli.back().value()) <
                    ciphertext.scale()) {
                ++owner.he_rescale_count;
                he->evaluator->rescale_to_next_inplace(ciphertext);
                rescaled_any = true;
            }
        }
        if (!rescaled_any) {
            break;
        }
        ++preliminary_rescales;
    }
    const double current_scale = ciphertexts.front().scale();
    const double current_relative_error =
        abs(current_scale - target_scale) / target_scale;
    if (current_scale < target_scale && current_relative_error > 1e-12) {
        const double unit_scale = target_scale / current_scale;
        if (!isfinite(unit_scale) || unit_scale < 1.0) {
            throw runtime_error(
                string("formal upward conversion lift is invalid: ") + stage);
        }
        const vector<double> public_one(he->num_slots, 1.0);
        for (auto &ciphertext : ciphertexts) {
            Plaintext per_ciphertext_unit;
            he->encoder->encode(public_one, ciphertext.parms_id(),
                                unit_scale, per_ciphertext_unit);
            he->evaluator->multiply_plain_inplace(ciphertext,
                                                  per_ciphertext_unit);
        }
        owner.linear.ct_pt_multiply_count += ciphertexts.size();
        upward_actual_one_lift = true;
    }
    owner.prepare_ckks_to_mpc_inplace(ciphertexts, target_scale_bits, he);
    for (const auto &ciphertext : ciphertexts) {
        const auto context_data =
            he->context->get_context_data(ciphertext.parms_id());
        const double relative_error =
            abs(ciphertext.scale() - target_scale) / target_scale;
        if (!context_data ||
            context_data->parms().coeff_modulus().size() != 1 ||
            ciphertext.parms_id() != he->context->last_parms_id() ||
            !isfinite(relative_error) || relative_error > 1e-9) {
            throw runtime_error(
                string("formal final conversion metadata mismatch: ") + stage);
        }
    }
    cout << "FORMAL_CKKS_TO_MPC_READY stage=" << stage
         << " status=PASS source_scale_bits=" << log2(initial_scale)
         << " source_limbs=" << initial_limbs
         << " preliminary_rescales=" << preliminary_rescales
         << " upward_actual_one_lift=" << upward_actual_one_lift
         << " target_scale_bits=" << target_scale_bits
         << " limbs=1 cts=" << ciphertexts.size() << endl;
}

// BOLT's BSGS2 kernel consumes a simple feature-major layout
// [feature, token], while the MPC nonlinearities and checkpoint oracle use
// token-major [token, feature].  Both are public permutations, so applying
// them independently to the two additive shares requires no communication
// and reveals no value.
FixArray token_major_to_feature_major(const FixArray &input, int tokens,
                                      int features, const char *stage) {
    if (tokens <= 0 || features <= 0 ||
        input.size != tokens * features) {
        throw runtime_error(string("formal token/feature layout mismatch: ") +
                            stage);
    }
    FixArray output(input.party, input.size, input.signed_, input.ell,
                    input.s);
    for (int token = 0; token < tokens; ++token) {
        for (int feature = 0; feature < features; ++feature) {
            output.data[feature * tokens + token] =
                input.data[token * features + feature];
        }
    }
    return output;
}

FixArray feature_major_to_token_major(const FixArray &input, int tokens,
                                      int features, const char *stage) {
    if (tokens <= 0 || features <= 0 ||
        input.size != tokens * features) {
        throw runtime_error(string("formal feature/token layout mismatch: ") +
                            stage);
    }
    FixArray output(input.party, input.size, input.signed_, input.ell,
                    input.s);
    for (int feature = 0; feature < features; ++feature) {
        for (int token = 0; token < tokens; ++token) {
            output.data[token * features + feature] =
                input.data[feature * tokens + token];
        }
    }
    return output;
}

// BSGS2 does not emit consecutive feature-major ciphertexts.  Ciphertext i
// stores one low feature block in its first slot half and the corresponding
// high feature block in its second half.  This is exactly the inverse of
// Linear::plain_col_packing_postprocess(..., false, data) used by Bert::run.
// Residuals must be put in this layout before ciphertext addition, and an
// Algorithm-1 result must be taken out of it before an MPC nonlinearity.
void require_bsgs_output_layout(const FixArray &input,
                                const FCMetadata &data,
                                const char *stage) {
    const int tokens = data.image_size;
    const int features = data.filter_w;
    const int slots = data.slot_count;
    if (tokens <= 0 || features <= 0 || slots <= 0 ||
        input.size != tokens * features || features % 2 != 0 ||
        slots % (2 * tokens) != 0 || input.size % slots != 0) {
        throw runtime_error(string("formal BSGS output layout mismatch: ") +
                            stage);
    }
    const int ciphertexts = input.size / slots;
    const int features_per_half = slots / (2 * tokens);
    if (ciphertexts * features_per_half != features / 2) {
        throw runtime_error(string("formal BSGS output coverage mismatch: ") +
                            stage);
    }
}

FixArray token_major_to_bsgs_output(const FixArray &input,
                                    const FCMetadata &data,
                                    const char *stage) {
    require_bsgs_output_layout(input, data, stage);
    const int tokens = data.image_size;
    const int features = data.filter_w;
    const int slots = data.slot_count;
    const int features_per_half = slots / (2 * tokens);
    const int ciphertexts = input.size / slots;
    FixArray output(input.party, input.size, input.signed_, input.ell,
                    input.s);
    for (int ciphertext = 0; ciphertext < ciphertexts; ++ciphertext) {
        for (int slot_index = 0; slot_index < slots; ++slot_index) {
            const int token = slot_index % tokens;
            const bool upper_half = slot_index >= slots / 2;
            const int local_feature =
                (slot_index % (slots / 2)) / tokens;
            const int feature = ciphertext * features_per_half +
                local_feature + (upper_half ? features / 2 : 0);
            output.data[ciphertext * slots + slot_index] =
                input.data[token * features + feature];
        }
    }
    return output;
}

FixArray bsgs_output_to_token_major(const FixArray &input,
                                    const FCMetadata &data,
                                    const char *stage) {
    require_bsgs_output_layout(input, data, stage);
    const int tokens = data.image_size;
    const int features = data.filter_w;
    const int slots = data.slot_count;
    const int features_per_half = slots / (2 * tokens);
    const int ciphertexts = input.size / slots;
    FixArray output(input.party, input.size, input.signed_, input.ell,
                    input.s);
    for (int ciphertext = 0; ciphertext < ciphertexts; ++ciphertext) {
        for (int slot_index = 0; slot_index < slots; ++slot_index) {
            const int token = slot_index % tokens;
            const bool upper_half = slot_index >= slots / 2;
            const int local_feature =
                (slot_index % (slots / 2)) / tokens;
            const int feature = ciphertext * features_per_half +
                local_feature + (upper_half ? features / 2 : 0);
            output.data[token * features + feature] =
                input.data[ciphertext * slots + slot_index];
        }
    }
    return output;
}

} // namespace

#define NVTX_PUSH(tag, flag)                                                   \
  do {                                                                         \
    if (flag) {                                                                \
      nvtxRangePush(tag);                                                      \
    }                                                                          \
  } while (0)

#define NVTX_POP(flag)                                                         \
  do {                                                                         \
    if (flag) {                                                                \
      nvtxRangePop();                                                          \
    }                                                                          \
  } while (0)

/*
    do not support multiple threads
    dim is hidden dimension, num_ops is sequence length
*/ 
// std::map<char*,FixArray> Bert_ckks::layer_norm_part1(uint64_t *x, int num_ops, int dim, int bitwidth, int s) {
//     cout << "bitwidth, s:" << bitwidth << "," << s << endl;
//     map<char*,FixArray> result;
//     FPMath *fpmath = nonlinear.fpmath[0];
//     // ArgMaxProtocol<uint64_t> *argmax_oracle;
//     int this_party = party;
//     vector<FixArray> input_array;
//     for(int i = 0; i < num_ops; i++){
//         input_array.push_back(fpmath->fix->input(this_party, dim, &x[i*dim], true, bitwidth, s));
//     }
//     FixArray w = fpmath->fix->input(this_party, num_ops*dim, 1, true, bitwidth, s);
//     FixArray b = fpmath->fix->input(this_party, num_ops*dim, 1, true, bitwidth, s);
//     // layer_norm part1 from bolt, change OT to HE
//     uint64_t bw_x = bitwidth+s;
//     uint64_t ell_mask = (bitwidth == 64 ? -1 : ((1ULL << bitwidth) - 1));
//     uint64_t bw_x_mask = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
//     bool signed_ = input_array[0].signed_;
//     vector<FixArray> ret(num_ops);
//     BoolArray all_0 = fpmath->bool_op->input(ALICE, num_ops*dim, uint8_t(0));
//     BoolArray all_1 = fpmath->bool_op->input(ALICE, num_ops*dim, 1);
//     // sum是(ell,s)，将其扩展至(ell+s,s)
//     FixArray sum = fpmath->fix->tree_sum(input_array);
//     cout << "comm:" << this->get_comm() << endl;
//     cout << "round:" << this->get_round() << endl;
//     sum = fpmath->fix->extend(sum,bitwidth+s);
//     cout << "extend comm:" << this->get_comm() << endl;
//     cout << "extend round:" << this->get_round() << endl;
//     uint64_t dn_scalar = uint64_t(((1.0 / dim) * pow(2, s))) & bw_x_mask;
//     //第一个mul是scaler乘法，维度是N, 乘数还是明文1/n，因此可以将sum位宽扩展为ell+s，这样无需任何协议，本地乘法即可。
//     //avg位宽，scale保持为(ell, s)
//     for(int i = 0; i < num_ops; i++){
//         sum.data[i] = (sum.data[i] * dn_scalar) & bw_x_mask;
//     }
//     sum.s = s+s;
//     // sum是(ell+s,2s)，将其tr至(ell,s)
//     FixArray avg = fpmath->fix->truncate_reduce(sum, s);
//     cout << "tr comm:" << this->get_comm() << endl;
//     cout << "tr round:" << this->get_round() << endl;
//     FixArray avg_flat(party, num_ops*dim, sum.signed_, bitwidth, s);
//     for(int i = 0; i < num_ops; i++){
//         for(int j = 0; j < dim; j++){
//             avg_flat.data[i*dim + j] = avg.data[i];
//         }
//     }
//     // 计算(x-x_avg)^2, x_flat_avg是(ell,s)
//     FixArray x_flat = concat(input_array);
//     FixArray x_flat_avg = fpmath->fix->sub(x_flat, avg_flat);
//     // x_flat_avg = fpmath->fix->extend(x_flat_avg, bitwidth+s);
//     cout << "extend2 comm:" << this->get_comm() << endl;
//     cout << "extend2 round:" << this->get_round() << endl;
//     //第二个mul是element-wise乘法，维度是N*n，输入(ell,s)，输出是(ell,s),采用CKKS HE。
//     FixArray x_flat_avg_square = element_wise_square(x_flat_avg.size,x_flat_avg);
//     // Bolt's implementation
//     // BoolArray msb_x_avg = fpmath->fix->MSB(x_flat_avg);
//     // FixArray x_flat_avg_square = fpmath->fix->mul(x_flat_avg, x_flat_avg, bitwidth + s, msb_x_avg.data, msb_x_avg.data);
//     // x_flat_avg_square = fpmath->fix->truncate_reduce(x_flat_avg_square, s);
//     cout << "element-wise mult. comm:" << this->get_comm() << endl;
//     cout << "element-wise mult. round:" << this->get_round() << endl;
//     //point6
//     vector<FixArray> square_group(num_ops);
//     for(int i = 0; i < num_ops; i++){
//         square_group[i] = FixArray(party, dim, signed_, bitwidth, s);
//         memcpy(square_group[i].data, &x_flat_avg_square.data[i*dim], dim*sizeof(uint64_t));
//     }
//     // square_sum是(ell,s)，将其扩展至(ell+s,s)
//     FixArray square_sum = fpmath->fix->tree_sum(square_group);
//     uint8_t* msb_0 = new uint8_t[square_sum.size];
//     memset(msb_0, 0, square_sum.size);
//     square_sum = fpmath->fix->extend(square_sum,bitwidth+s,msb_0);
//     cout << "ext2 comm:" << this->get_comm() << endl;
//     cout << "ext2 round:" << this->get_round() << endl;
//     // square_sum*(1/n)，本地乘法
//     for(int i = 0; i < square_sum.size; i++){
//         square_sum.data[i] = (square_sum.data[i] * dn_scalar) & bw_x_mask;
//     }
//     square_sum.s = 2*s;
//     // square_sum是(ell+s,2s)，将其tr至(ell,s)
//     // cout << "square_sum: (bitwidth,s):" << square_sum.ell << "," << square_sum.s << endl;
//     square_sum = fpmath->fix->truncate_reduce(square_sum, s, msb_0);
//     cout << "tr2 comm:" << this->get_comm() << endl;
//     cout << "tr2 round:" << this->get_round() << endl;
//     // cout << "after TR, square_sum: (bitwidth,s):" << square_sum.ell << "," << square_sum.s << endl;
//     //point7
//     // 计算平方根倒数
//     FixArray sigma = fpmath->sqrt(square_sum, true);
//     cout << "sqrt comm:" << this->get_comm() << endl;
//     cout << "sqrt round:" << this->get_round() << endl;
//     //point8
//     FixArray sigma_flat(party, num_ops*dim, sum.signed_, bitwidth, s);
//     for(int i = 0; i < num_ops; i++){
//         for(int j = 0; j < dim; j++){
//             sigma_flat.data[i*dim + j] = sigma.data[i];
//         }
//     }
//     // 广播乘法
//     cout << "x_flat_avg bitwidth,s:" << x_flat_avg.ell << "," << x_flat_avg.s << endl;
//     result["x_flat_avg"] = x_flat_avg;
//     result["sigma_flat"] = sigma_flat;
//     result["w"] = w;
//     result["b"] = b;
//     // FixArray x_avg_sigma = fix->mul(x_flat_avg, sigma_flat, ell+s, msb_x_avg.data, all_0.data);
//     // x_avg_sigma = fix->truncate_reduce(x_avg_sigma, s);
//     // //point9
//     // // Weight and Bias
//     // x_avg_sigma = fix->mul(x_avg_sigma, w, ell+s);
//     // x_avg_sigma = fix->truncate_reduce(x_avg_sigma, s);
//     // x_avg_sigma = fix->add(x_avg_sigma, b);
//     cout << "layer_norm_part1 done" << endl;
//     return result;
// }

/*
    step1. 计算layernorm结尾的两个element-wise乘法
    step2. 计算attention
    input 的 bitwidth 与 scale 需要是正确的MPC下的值
*/
vector<vector<HE_CKKS::scalar_t>> Bert_ckks::layer_norm_part2_attention(
    std::map<char*,FixArray> input, int layer_id){
    // mult. depth = 4
    auto start = high_resolution_clock::now();
    const bool raw_input = input.size() == 1 && input.begin()->first != nullptr &&
                           string(input.begin()->first) == "raw_input";
    // The BERT Figure-10 path uses x+y*j half packing. ViT's pre-norm path
    // supplies ordinary full-slot shares and must not interpret the unknown
    // imaginary residue of Algorithm-2's local Encode as a second tensor.
    const bool use_bert_half_packing =
        !raw_input && !pre_norm_attention_output_only;
    const bool ordinary_full_slot_output = !use_bert_half_packing;
    // Aggregate-only Q/K/V audit.  It converts each raw tensor to the same
    // fixed-clear reference domain but opens only scalar error summaries;
    // the normal benchmark path never enables it.
    const bool debug_qkv = getenv("BLB_QKV_AGGREGATE") != nullptr &&
                           string(getenv("BLB_QKV_AGGREGATE")) == "1";
    const bool a0_trace = getenv("OURS_VIT_A0_TRACE") != nullptr &&
                          string(getenv("OURS_VIT_A0_TRACE")) == "1";
    a0_qkv_ready = false;
    if (raw_input) {
        attention_v_he_cts.clear();
        attention_v_he_ready = false;
    } else if (!pre_norm_attention_output_only) {
        // Figure 10 Block 2 produces the post-LayerNorm attention residual
        // that Block 4 consumes after the intervening MPC Softmax.  It is
        // materialized below from the same LN-affine ciphertexts used by
        // QKV; do not let a stale layer's residual cross this boundary.
        attention_input_residual_ready = false;
    }
    FixArray x_flat_avg = raw_input ? input.begin()->second : input["x_flat_avg"];
    FixArray sigma_flat = raw_input ? FixArray() : input["sigma_flat"];
    FixArray w = raw_input ? FixArray() : input["w"];
    FixArray b = raw_input ? FixArray() : input["b"];

    if (x_flat_avg.size <= 0 || x_flat_avg.size % COMMON_DIM != 0) {
        throw runtime_error("formal attention input shape mismatch");
    }
    if (!raw_input &&
        (sigma_flat.size != x_flat_avg.size || w.size != x_flat_avg.size ||
         b.size != x_flat_avg.size)) {
        throw runtime_error("formal LayerNorm/attention share shape mismatch");
    }
    if (!raw_input) {
        require_layernorm_parameter_scale(x_flat_avg, sigma_flat, w, b);
    }
    if (layer_id < 0 || layer_id >= ATTENTION_LAYERS) {
        throw runtime_error("formal attention layer id is out of range");
    }

    if (!raw_input && pre_norm_attention_output_only) {
        const int tokens = x_flat_avg.size / COMMON_DIM;
        auto transpose_for_qkv = [tokens](FixArray &share) {
            vector<uint64_t> feature_major(share.size);
            for (int token = 0; token < tokens; ++token) {
                for (int feature = 0; feature < COMMON_DIM; ++feature) {
                    feature_major[static_cast<size_t>(feature) * tokens +
                                  token] =
                        share.data[static_cast<size_t>(token) * COMMON_DIM +
                                   feature];
                }
            }
            copy(feature_major.begin(), feature_major.end(), share.data);
        };
        // BLB's QKV diagonals repeat one weight over all token slots, so the
        // fused affine/QKV input is [feature][token]. ViT keeps its residual
        // state [token][feature]; transpose each additive share locally at
        // this boundary. This is a permutation only: no opening, round, or
        // communication is introduced.
        transpose_for_qkv(x_flat_avg);
        transpose_for_qkv(sigma_flat);
        transpose_for_qkv(w);
        transpose_for_qkv(b);
        cout << "FORMAL_QKV_INPUT_LAYOUT status=PASS source=token_major"
             << " target=feature_major tokens=" << tokens
             << " features=" << COMMON_DIM
             << " communication=0 rounds=0" << endl;
    }

    const int bitwidth = x_flat_avg.ell;
    const int s = x_flat_avg.s;
    uint64_t dim = x_flat_avg.size;
    cout << "should be 0! comm:" << this->get_comm() << ", round:" << this->get_round() << endl;
    HE_CKKS* he = linear.he1;
    uint64_t slot_count = he->num_slots;
    uint64_t degree = he->poly_modulus_degree;
    uint64_t num_cts = dim / slot_count;
    vector<vector<HE_CKKS::scalar_t>> vec_x_flat_avg(num_cts), vec_sigma_flat(num_cts);
    for(int i = 0; i < num_cts; i++){
        for(int j = 0; j < slot_count; j++){
            vec_x_flat_avg[i].push_back(x_flat_avg.data[i*slot_count + j]);
            if (!raw_input) {
                vec_sigma_flat[i].push_back(sigma_flat.data[i*slot_count + j]);
            }
        }
    }

    vector<UnifiedPlaintext> pts_x_flat_avg;
    vector<UnifiedPlaintext> pts_sigma_flat;
    
    FCMetadata data = linear.data_lin1_0;
    const size_t qk_ct_count = static_cast<size_t>(
        data.image_size * data.image_size * PACKING_NUM / data.slot_count);
    const size_t v_ct_count = static_cast<size_t>(
        data.image_size * data.filter_w * PACKING_NUM / data.slot_count);
    const size_t output_ct_count = qk_ct_count + v_ct_count;
    if (ordinary_full_slot_output &&
        (qk_ct_count * static_cast<size_t>(data.slot_count) !=
             static_cast<size_t>(data.image_size) * data.image_size * PACKING_NUM ||
         v_ct_count * static_cast<size_t>(data.slot_count) !=
             static_cast<size_t>(data.image_size) * data.filter_w * PACKING_NUM)) {
        throw runtime_error(
            "formal raw QK/V ciphertext layout is not slot aligned for this sequence");
    }
    vector<UnifiedCiphertext> output_cts(
        output_ct_count,
        UnifiedCiphertext(LOCATION::HOST)
    );
    vector<Ciphertext> output_cts_half(output_cts.size()/2);
    vector<Plaintext> output_pts(output_ct_count);
    vector<Plaintext> output_pts_half(output_pts.size()/2);
    // The raw layer-0 kernel emits 12 ordinary full-slot QK ciphertexts
    // followed by 6 ordinary full-slot V ciphertexts.  They are not nine
    // real/imag paired ciphertexts, so Algorithm 1 and CKKS->MPC conversion
    // must preserve the two independent layouts and scales.
    const bool qkv_only_diagnostic = raw_input && debug_qkv;
    vector<Ciphertext> raw_qk_cts(
        ordinary_full_slot_output
            ? (qkv_only_diagnostic ? output_ct_count : qk_ct_count)
            : 0);
    vector<Ciphertext> raw_v_cts(
        ordinary_full_slot_output && !qkv_only_diagnostic ? v_ct_count : 0);
    vector<Plaintext> raw_qk_pts(raw_qk_cts.size());
    vector<Plaintext> raw_v_pts(raw_v_cts.size());
    const bool selective_conversion =
        ordinary_full_slot_output && !raw_input &&
        heterogeneous_ablation >= ours_vit::Ablation::A3;
    struct SelectiveGroupConversion {
        ours_vit::HeadProtocol protocol;
        int head_begin = 0;
        int head_end = 0;
        vector<size_t> qk_indices;
        vector<size_t> v_indices;
        vector<Ciphertext> qk_cts;
        vector<Ciphertext> v_cts;
        vector<Plaintext> qk_pts;
        vector<Plaintext> v_pts;
    };
    vector<SelectiveGroupConversion> selective_groups;
    if (selective_conversion) {
        if (heterogeneous_protocols.size() != PACKING_NUM ||
            heterogeneous_alphas.size() != PACKING_NUM) {
            throw runtime_error(
                "A3/A4 selective QKV conversion has no public assignment");
        }
        const size_t ciphertexts_per_pair =
            qk_ct_count / (PACKING_NUM / 2);
        const size_t v_ciphertexts_per_pair =
            v_ct_count / (PACKING_NUM / 2);
        if (qk_ct_count % (PACKING_NUM / 2) != 0 ||
            v_ct_count % (PACKING_NUM / 2) != 0 ||
            ciphertexts_per_pair == 0 || v_ciphertexts_per_pair == 0) {
            throw runtime_error(
                "A3/A4 QK/V head-pair ciphertext layout is invalid");
        }
        for (int begin = 0; begin < PACKING_NUM;) {
            int end = begin + 1;
            while (end < PACKING_NUM &&
                   heterogeneous_protocols[end] ==
                       heterogeneous_protocols[begin]) {
                ++end;
            }
            if (heterogeneous_protocols[begin] !=
                ours_vit::HeadProtocol::HeLinear) {
                SelectiveGroupConversion group;
                group.protocol = heterogeneous_protocols[begin];
                group.head_begin = begin;
                group.head_end = end;
                for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
                    const bool first = 2 * pair >= begin && 2 * pair < end;
                    const bool second = 2 * pair + 1 >= begin &&
                                        2 * pair + 1 < end;
                    if (first || second) {
                        for (size_t local = 0;
                             local < ciphertexts_per_pair; ++local) {
                            group.qk_indices.push_back(
                                pair * ciphertexts_per_pair + local);
                        }
                    }
                }
                for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
                    const bool first = 2 * pair >= begin && 2 * pair < end;
                    const bool second = 2 * pair + 1 >= begin &&
                                        2 * pair + 1 < end;
                    if (first || second) {
                        for (size_t local = 0;
                             local < v_ciphertexts_per_pair; ++local) {
                            group.v_indices.push_back(
                                pair * v_ciphertexts_per_pair + local);
                        }
                    }
                }
                group.qk_cts.resize(group.qk_indices.size());
                group.v_cts.resize(group.v_indices.size());
                group.qk_pts.resize(group.qk_indices.size());
                group.v_pts.resize(group.v_indices.size());
                selective_groups.push_back(std::move(group));
            }
            begin = end;
        }
        if (selective_groups.empty() &&
            heterogeneous_ablation != ours_vit::Ablation::A4) {
            throw runtime_error("A3 assignment has no MPC attention group");
        }
    }
    const size_t projected_tensor_ct_count = v_ct_count;
    vector<Ciphertext> a0_q_cts(a0_trace && !raw_input
                                    ? projected_tensor_ct_count : 0);
    vector<Ciphertext> a0_k_cts(a0_trace && !raw_input
                                    ? projected_tensor_ct_count : 0);
    vector<Ciphertext> a0_v_cts(a0_trace && !raw_input
                                    ? projected_tensor_ct_count : 0);
    vector<Plaintext> a0_q_pts(a0_q_cts.size());
    vector<Plaintext> a0_k_pts(a0_k_cts.size());
    vector<Plaintext> a0_v_pts(a0_v_cts.size());
    // The first half of the ordinary CKKS input layout contains the real and
    // imaginary halves needed by Algorithm 1's half decode.  Exporting this
    // copy once gives the next Figure-10 block the real post-LN residual
    // without re-running LayerNorm or inventing a new packing.
    vector<Ciphertext> normalized_input_cts_half(
        raw_input || pre_norm_attention_output_only ? 0 : num_cts / 2);
    vector<Plaintext> normalized_input_pts(
        raw_input || pre_norm_attention_output_only ? 0 : num_cts / 2);
    cout << this->get_comm() << endl;
    cout << this->get_round() << endl;
    this->he_compute_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    
    start = high_resolution_clock::now();
    // For the post-norm layer-0 entry, main has already applied BERT's real
    // x_scale[0] transition (s12 -> s5) with secure truncate_reduce.  The
    // official BLB path imports this tensor at he1's Table-6 scale.  The
    // numerical CKKS scale is independent of the MPC fixed-point scale; the
    // subsequent QK rescale schedule is the one encoded by the original
    // he1 evaluator and must not be replaced by a new scale planner.
    // Table 6's Block-2 scale (44) is the public plaintext/operator scale.
    // The layer-0 ciphertext crosses four consecutive rescale boundaries
    // (QKV, the public packing mask, Q*K, and the final public mask).  The
    // paper's depth-7 schedule therefore imports the real embedding at the
    // planner-derived scale 90, yielding 74 -> 58 -> 56 -> 40.  Reusing
    // he->he_scale here incorrectly produces 28 -> 12 -> -4 -> -52.
    int x_ckks_scale_bits = raw_input
                                ? layer0_qkv_import_scale_bits(
                                      he, kFormalConversionScaleBits)
                                : he->he_scale;
    if (raw_input && s != 5) {
        throw runtime_error(
            "formal layer-0 QKV input must be securely truncated to scale 5");
    }
    if (raw_input) {
        cout << "RAW_INPUT_SCALE_ALIGNMENT status=PASS fixed_scale=" << s
             << " ckks_scale_bits=" << x_ckks_scale_bits
             << " source=table6_he1_scale" << endl;
    }
    if (raw_input) {
        mpc_to_ckks(pts_x_flat_avg, vec_x_flat_avg, slot_count,
                    he->get_conversion_prime(),
                    bitwidth, s, x_ckks_scale_bits, he);
    } else {
        mpc_to_ckks(pts_x_flat_avg, vec_x_flat_avg, slot_count,
                    he->get_conversion_prime(),
                    bitwidth, s, x_ckks_scale_bits, he);
    }
    // MARK: Can be parallel
    // cout << "mpc_to_ckks1 comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks1 round:" << this->get_round() << endl;
    if (!raw_input) {
        mpc_to_ckks(pts_sigma_flat, vec_sigma_flat, slot_count,
                    he->get_conversion_prime(),
                    bitwidth, s, he->he_scale, he);
    }
    // cout << "mpc_to_ckks2 comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks2 round:" << this->get_round() << endl;
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    
    vector<UnifiedCiphertext> cts_x_flat_full(
        use_bert_half_packing ? 0 : pts_x_flat_avg.size(),
        UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> cts_sigma_flat_full(
        (!raw_input && !use_bert_half_packing) ? pts_sigma_flat.size() : 0,
        UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> cts_x_flat_half(
        use_bert_half_packing ? pts_x_flat_avg.size() / 2 : 0,
        UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> cts_sigma_flat_half(
        use_bert_half_packing ? pts_sigma_flat.size() / 2 : 0,
        UnifiedCiphertext(LOCATION::HOST));
    if (party == BOB) {
        if (!use_bert_half_packing) {
            #pragma omp parallel for
            for (size_t i = 0; i < num_cts; ++i) {
                he->encryptor->encrypt(pts_x_flat_avg[i],
                                       cts_x_flat_full[i]);
                if (!raw_input) {
                    he->encryptor->encrypt(pts_sigma_flat[i],
                                           cts_sigma_flat_full[i]);
                }
            }
        } else {
            #pragma omp parallel for
            for(size_t i=0;i<num_cts/2;i++){
                he->encryptor->encrypt(pts_x_flat_avg[i], cts_x_flat_half[i]);
                he->encryptor->encrypt(pts_sigma_flat[i], cts_sigma_flat_half[i]);
            }
        }
        cout << "send_encrypted_vector" << endl;
			// test_util();
        if (!use_bert_half_packing) {
            send_encrypted_vector(io, cts_x_flat_full);
            if (!raw_input)
                send_encrypted_vector(io, cts_sigma_flat_full);
            cout << (raw_input ? "RAW_INPUT_TRANSFER_LAYOUT"
                               : "FORMAL_TRANSFER_LAYOUT stage=ln_attention_input")
                 << " status=PASS layout=ordinary_full_slot cts="
                 << cts_x_flat_full.size() << " imag_repack=0" << endl;
        } else {
			send_encrypted_vector(io, cts_x_flat_half);
            send_encrypted_vector(io, cts_sigma_flat_half);
        }
        cout << "send_encrypted_vector done" << endl;
		/*
           TODO: output is also in x+y*j format, thus, decrypt, decode can be reduced by half
        */
        if (ordinary_full_slot_output) {
            if (qkv_only_diagnostic) {
                recv_encrypted_vector(he->context, io, raw_qk_cts);
            } else if (selective_conversion) {
                for (auto &group : selective_groups) {
                    recv_encrypted_vector(he->context, io, group.qk_cts);
                    recv_encrypted_vector(he->context, io, group.v_cts);
                    #pragma omp parallel for
                    for (int i = 0;
                         i < static_cast<int>(group.qk_cts.size()); ++i) {
                        he->decryptor->decrypt(group.qk_cts[i],
                                               group.qk_pts[i]);
                    }
                    #pragma omp parallel for
                    for (int i = 0;
                         i < static_cast<int>(group.v_cts.size()); ++i) {
                        he->decryptor->decrypt(group.v_cts[i],
                                               group.v_pts[i]);
                    }
                }
            } else {
                recv_encrypted_vector(he->context, io, raw_qk_cts);
                recv_encrypted_vector(he->context, io, raw_v_cts);
            }
            if (!selective_conversion || qkv_only_diagnostic) {
                #pragma omp parallel for
                for (int i = 0;
                     i < static_cast<int>(raw_qk_cts.size()); ++i) {
                    he->decryptor->decrypt(raw_qk_cts[i], raw_qk_pts[i]);
                }
                if (!qkv_only_diagnostic) {
                    #pragma omp parallel for
                    for (int i = 0;
                         i < static_cast<int>(raw_v_cts.size()); ++i) {
                        he->decryptor->decrypt(raw_v_cts[i], raw_v_pts[i]);
                    }
                }
            }
            if (a0_trace && !raw_input) {
                if (getenv("OURS_VIT_A0_NATIVE_OBSERVER_CHECK") != nullptr) {
                    vector<Ciphertext> native_q(a0_q_cts.size());
                    recv_encrypted_vector(io == nullptr ? nullptr : he->context,
                                         io, native_q);
                    const size_t count = min<size_t>(2, native_q.size());
                    for (size_t index = 0; index < count; ++index) {
                        Plaintext native_plaintext;
                        he->decryptor->decrypt(native_q[index],
                                               native_plaintext);
                        vector<complex<double>> native_decoded;
                        he->encoder->decode(native_plaintext, native_decoded);
                        double max_abs = 0.0;
                        for (const auto &value : native_decoded) {
                            max_abs = max(max_abs, abs(value.real()));
                        }
                        cout << "OURS_A0_NATIVE_Q_OBSERVER index=" << index
                             << " scale_bits="
                             << log2(native_plaintext.scale())
                             << " limbs="
                             << he->context->get_context_data(
                                    native_plaintext.parms_id())
                                    ->parms().coeff_modulus().size()
                             << " max_abs=" << max_abs << " slots=";
                        for (size_t slot_index = 0;
                             slot_index < min<size_t>(8, native_decoded.size());
                             ++slot_index) {
                            if (slot_index != 0) cout << ',';
                            cout << setprecision(17)
                                 << native_decoded[slot_index].real();
                        }
                        cout << endl;
                    }
                }
                recv_encrypted_vector(he->context, io, a0_q_cts);
                recv_encrypted_vector(he->context, io, a0_k_cts);
                recv_encrypted_vector(he->context, io, a0_v_cts);
                #pragma omp parallel for
                for (int i = 0; i < static_cast<int>(a0_q_cts.size()); ++i) {
                    he->decryptor->decrypt(a0_q_cts[i], a0_q_pts[i]);
                    he->decryptor->decrypt(a0_k_cts[i], a0_k_pts[i]);
                    he->decryptor->decrypt(a0_v_cts[i], a0_v_pts[i]);
                }
            }
        } else {
            if (!pre_norm_attention_output_only) {
                recv_encrypted_vector(he->context, io,
                                      normalized_input_cts_half);
                #pragma omp parallel for
                for (int i = 0; i < normalized_input_cts_half.size(); ++i) {
                    he->decryptor->decrypt(normalized_input_cts_half[i],
                                           normalized_input_pts[i]);
                }
            }
            recv_encrypted_vector(he->context, io, output_cts_half);
            #pragma omp parallel for
            for(int i=0;i<output_cts_half.size();i++){
                he->decryptor->decrypt(output_cts_half[i], output_pts_half[i]);
            }
            if (a0_trace) {
                recv_encrypted_vector(he->context, io, a0_q_cts);
                recv_encrypted_vector(he->context, io, a0_k_cts);
                recv_encrypted_vector(he->context, io, a0_v_cts);
                #pragma omp parallel for
                for (int i = 0; i < static_cast<int>(a0_q_cts.size()); ++i) {
                    he->decryptor->decrypt(a0_q_cts[i], a0_q_pts[i]);
                    he->decryptor->decrypt(a0_k_cts[i], a0_k_pts[i]);
                    he->decryptor->decrypt(a0_v_cts[i], a0_v_pts[i]);
                }
            }
        }
	}
    else {
        LOCATION loc = he->backend();
        vector<UnifiedCiphertext> cts_x_flat(pts_x_flat_avg.size(), UnifiedCiphertext(loc));
        vector<UnifiedCiphertext> cts_sigma_flat(
            raw_input ? 0 : pts_sigma_flat.size(), UnifiedCiphertext(loc));
        
        // Revice ciphertexts from BOB (or clinet)
        start = high_resolution_clock::now();
        if (!use_bert_half_packing) {
            recv_encrypted_vector(he->context, io, cts_x_flat_full);
            if (!raw_input)
                recv_encrypted_vector(he->context, io,
                                      cts_sigma_flat_full);
        } else {
            recv_encrypted_vector(he->context, io, cts_x_flat_half);
            recv_encrypted_vector(he->context, io, cts_sigma_flat_half);
        }
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        
        NVTX_PUSH("LN2-ATT", true);
        if (he->is_gpu_enable()) {
            // Load reviced ciphertexts to GPU
            start = high_resolution_clock::now();
            if (!use_bert_half_packing) {
                for_each_n(cts_x_flat_full.begin(), cts_x_flat_full.size(),
                        [he](auto &ct) { ct.to_device(*he->context, *he->dcontext); });
                if (!raw_input) {
                    for_each_n(cts_sigma_flat_full.begin(),
                               cts_sigma_flat_full.size(),
                               [he](auto &ct) {
                                   ct.to_device(*he->context, *he->dcontext);
                               });
                }
            } else {
                for_each_n(cts_x_flat_half.begin(), cts_x_flat_half.size(),
                        [he](auto &ct) { ct.to_device(*he->context, *he->dcontext); });
                for_each_n(cts_sigma_flat_half.begin(), cts_sigma_flat_half.size(),
                        [he](auto &ct) { ct.to_device(*he->context, *he->dcontext); });
            }
            for_each_n(pts_x_flat_avg.begin(), pts_x_flat_avg.size(),
                    [he](auto &pt) { pt.to_device(*he->context, *he->dcontext); });
            if (!raw_input) {
                for_each_n(pts_sigma_flat.begin(), pts_sigma_flat.size(),
                        [he](auto &pt) { pt.to_device(*he->context, *he->dcontext); });
            }
            this->ct_transfer_mem_la += ((high_resolution_clock::now() - start)).count() / 1e+9;
        }

        start = high_resolution_clock::now();
        if (!use_bert_half_packing) {
            for (size_t i = 0; i < cts_x_flat_full.size(); ++i) {
                cts_x_flat[i] = std::move(cts_x_flat_full[i]);
                if (!raw_input) {
                    cts_sigma_flat[i] =
                        std::move(cts_sigma_flat_full[i]);
                }
            }
            cout << (raw_input ? "RAW_INPUT_TRANSFER_LAYOUT"
                               : "FORMAL_TRANSFER_LAYOUT stage=ln_attention_input")
                 << " status=PASS layout=ordinary_full_slot cts="
                 << cts_x_flat.size() << " imag_repack=0" << endl;
        } else {
            const auto x_half_offset = cts_x_flat_half.size();
            const auto sigma_half_offset = cts_sigma_flat_half.size();
            #pragma omp parallel for
            for(int i=0;i<cts_x_flat_half.size();i++){
                he->imag_repack(cts_x_flat_half[i], cts_x_flat[i],
                                cts_x_flat[i + x_half_offset]);
                he->imag_repack(cts_sigma_flat_half[i], cts_sigma_flat[i],
                                cts_sigma_flat[i + sigma_half_offset]);
            }
        }
        cout << "num of input cts:" << cts_x_flat.size()+cts_sigma_flat.size() << endl;
        cout << "bitwidth of input ct:" << cts_x_flat[0].coeff_modulus_size() << endl;
        cout << "recv_encrypted_vector" << endl;
        
        // Encode weight in once.
        vector<UnifiedPlaintext> tmp_w(cts_x_flat.size(), UnifiedPlaintext(loc));
        vector<UnifiedPlaintext> tmp_b(cts_x_flat.size(), UnifiedPlaintext(loc));
        vector<vector<double>> w_matrix(cts_x_flat.size(),vector<double>(slot_count, 0));
        vector<vector<double>> b_matrix(cts_x_flat.size(),vector<double>(slot_count, 0));
        if (!raw_input) {
            for(int i=0;i<cts_x_flat.size();i++){
                for(int j=0;j<slot_count;j++){
                    const double projection_compensation =
                        use_bert_half_packing ? 1.0 : 0.25;
                    w_matrix[i][j] = projection_compensation *
                        layernorm_fixed_to_real(
                            w.data[i*slot_count+j], w,
                            "LayerNorm weight");
                    b_matrix[i][j] = layernorm_fixed_to_real(
                        b.data[i*slot_count+j], b, "LayerNorm bias");
                }
            }
        }
        if (!raw_input && !he->is_gpu_enable()) {
            #pragma omp parallel for
            for(int i=0;i<cts_x_flat.size();i++){
                he->encoder->encode(w_matrix[i], cts_x_flat[i].hcipher().parms_id(), pow(2, he->he_scale), tmp_w[i]);
                he->encoder->encode(b_matrix[i], cts_x_flat[i].hcipher().parms_id(), pow(8, he->he_scale), tmp_b[i]);
            }
        } else if (!raw_input) {
            for(int i=0;i<cts_x_flat.size();i++){
                he->dencoder->encode(*he->dcontext, w_matrix[i], pow(2, he->he_scale), tmp_w[i], cts_x_flat[i].dcipher().chain_index());
                he->dencoder->encode(*he->dcontext, b_matrix[i], pow(8, he->he_scale), tmp_b[i], cts_x_flat[i].dcipher().chain_index());
            }
        }

        // start computation
        #pragma omp parallel for
        for(int i=0;i<cts_x_flat.size();i++){
            /*
                get x_avg and sigma, compute (x_avg * w) * sigma + b
                end with a single rescale + relinearize
            */ 
            if (raw_input) {
                // BERT-base layer 0 is post-norm.  The exported embedding
                // share is sent directly to the existing pp_1/BSGS QKV
                // kernel; no synthetic identity LayerNorm is introduced.
                // BOB encrypted each ordinary full-slot local share above;
                // ALICE adds its matching plaintext share exactly once before
                // QKV. No real/imag reconstruction is valid for this layout.
                if (!he->is_gpu_enable()) {
                    he->evaluator->add_plain_inplace(cts_x_flat[i],
                                                     pts_x_flat_avg[i]);
                } else {
                    he->devaluator->add_plain_inplace(cts_x_flat[i],
                                                      pts_x_flat_avg[i]);
                }
                if (abs(log2(cts_x_flat[i].scale()) - x_ckks_scale_bits) >
                    1e-9) {
                    throw runtime_error(
                        "formal layer-0 high-scale MPC-to-CKKS import was not preserved");
                }
                continue;
            }
            if (!he->is_gpu_enable()) {
                he->evaluator->add_plain_inplace(cts_x_flat[i], pts_x_flat_avg[i]);
                he->evaluator->add_plain_inplace(cts_sigma_flat[i], pts_sigma_flat[i]);
                if (!use_bert_half_packing) {
                    UnifiedCiphertext x_conjugate(LOCATION::HOST);
                    UnifiedCiphertext sigma_conjugate(LOCATION::HOST);
                    he->evaluator->complex_conjugate(
                        cts_x_flat[i], *he->gal_keys, x_conjugate);
                    he->evaluator->complex_conjugate(
                        cts_sigma_flat[i], *he->gal_keys, sigma_conjugate);
                    he->evaluator->add_inplace(cts_x_flat[i], x_conjugate);
                    he->evaluator->add_inplace(cts_sigma_flat[i],
                                               sigma_conjugate);
                }

                // x and sigma enter at the same CKKS scale.  Form their
                // ciphertext product first; applying w first raises x's
                // scale and makes the subsequent ciphertext multiplication
                // invalid.  This is the paper's (x*sigma)*w+b order.
                he->evaluator->multiply_inplace(cts_x_flat[i], cts_sigma_flat[i]);
                he->evaluator->relinearize_inplace(cts_x_flat[i],
                                                   *he->relin_keys);
                he->evaluator->multiply_plain_inplace(cts_x_flat[i], tmp_w[i]);
                he->evaluator->add_plain_inplace(cts_x_flat[i], tmp_b[i]);
                ++this->he_rescale_count;
                he->evaluator->rescale_to_next_inplace(cts_x_flat[i]);
            } else {
                he->devaluator->add_plain_inplace(cts_x_flat[i], pts_x_flat_avg[i]);
                he->devaluator->add_plain_inplace(cts_sigma_flat[i], pts_sigma_flat[i]);
                if (!use_bert_half_packing) {
                    UnifiedCiphertext x_conjugate(LOCATION::DEVICE);
                    UnifiedCiphertext sigma_conjugate(LOCATION::DEVICE);
                    he->devaluator->complex_conjugate(
                        cts_x_flat[i], *he->dgal_keys, x_conjugate);
                    he->devaluator->complex_conjugate(
                        cts_sigma_flat[i], *he->dgal_keys,
                        sigma_conjugate);
                    he->devaluator->add_inplace(cts_x_flat[i], x_conjugate);
                    he->devaluator->add_inplace(cts_sigma_flat[i],
                                                sigma_conjugate);
                }

                he->devaluator->multiply_inplace(cts_x_flat[i], cts_sigma_flat[i]);
                he->devaluator->relinearize_inplace(cts_x_flat[i],
                                                    *he->drelin_keys);
                he->devaluator->multiply_plain_inplace(cts_x_flat[i], tmp_w[i]);

                he->devaluator->add_plain_inplace(cts_x_flat[i], tmp_b[i]);
                ++this->he_rescale_count;
                he->devaluator->rescale_to_next_inplace(cts_x_flat[i]);
            }
        }
        if (raw_input) {
            require_uniform_unified_metadata(cts_x_flat,
                                             "layer0_qkv_input");
            const double target_scale = ldexp(1.0, x_ckks_scale_bits);
            const double relative_error =
                abs(cts_x_flat.front().scale() - target_scale) /
                target_scale;
            if (!isfinite(relative_error) || relative_error > 1e-12) {
                throw runtime_error(
                    "formal layer-0 MPC-to-CKKS metadata mismatch after share addition");
            }
            cout << "RAW_INPUT_SCALE status=PASS actual_scale_bits="
                 << log2(cts_x_flat.front().scale())
                 << " limbs=" << cts_x_flat.front().coeff_modulus_size()
                 << " cts=" << cts_x_flat.size() << endl;
        }
        cout << "scale:" << log2(cts_x_flat[0].scale()) << endl;
        if (!raw_input) {
            cout << "FORMAL_REAL_PROJECTION stage=ln_attention_affine"
                 << " status=PASS mode="
                 << (use_bert_half_packing ? "xyj_imag_repack"
                                           : "both_conjugate")
                 << " gamma_compensation="
                 << (use_bert_half_packing ? 1.0 : 0.25) << endl;
        }
        cout << "(x_avg * sigma) * w + b done" << endl;  
        if (!raw_input && pre_norm_attention_output_only) {
            const double qkv_target_scale =
                linear.pp_1[layer_id].input_scale;
            if (cts_x_flat.front().scale() < qkv_target_scale) {
                lift_unified_with_actual_one(
                    linear, he, cts_x_flat, qkv_target_scale,
                    "vit_pre_norm_qkv_input");
            }
            const double relative_error =
                abs(cts_x_flat.front().scale() - qkv_target_scale) /
                qkv_target_scale;
            if (!isfinite(relative_error) || relative_error > 1e-12) {
                throw runtime_error(
                    "formal ViT pre-norm QKV scale plan was not reached");
            }
        }
        vector<UnifiedCiphertext> q_k_v;
        vector<UnifiedCiphertext> a0_qkv_ciphertexts;
        if (debug_qkv) {
            q_k_v.resize(output_ct_count, UnifiedCiphertext(loc));
            linear.bert_cipher_plain_bsgs(
                he, cts_x_flat, linear.pp_1[layer_id], data, q_k_v);
            if (loc == LOCATION::HOST) {
                #pragma omp parallel for
                for (int i = 0; i < static_cast<int>(q_k_v.size()); ++i) {
                    he->evaluator->relinearize_inplace(
                        q_k_v[i], *he->relin_keys);
                }
            }
        } else {
            linear.linear_1(
                he,
                cts_x_flat,
                linear.pp_1[layer_id],
                data,
                q_k_v,
                a0_trace && !raw_input ? &a0_qkv_ciphertexts : nullptr,
                heterogeneous_ablation >= ours_vit::Ablation::A3
            );
        }
        cout << "qkv_scale:" << log2(q_k_v[0].scale()) << endl;
        /*
            before rescale, ct scale=145, Q=(60,55,60)
            after rescale, ct scale=30, Q=(60,)
        */
        cout << "Q:" << he->get_logQ(q_k_v[0]) << endl;
        cout << "scale:" << log2(q_k_v[0].scale()) << endl;
        cout << "num of output cts:" << q_k_v.size() << endl;
        cout << "bitwidth of output ct:" << q_k_v[0].coeff_modulus_size() << endl;
        this->he_compute_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        
        if (q_k_v.size() != output_ct_count) {
            throw runtime_error("formal QK/V ciphertext count mismatch");
        }
        if (!raw_input && !pre_norm_attention_output_only) {
            for (size_t i = 0; i < normalized_input_cts_half.size(); ++i) {
                if (he->is_gpu_enable()) {
                    cts_x_flat[i].to_host(*he->context, *he->dcontext);
                }
                normalized_input_cts_half[i] = cts_x_flat[i].hcipher();
            }
            prepare_host_conversion_from_actual_scale(
                *this, normalized_input_cts_half, kFormalConversionScaleBits,
                he, "layernorm2_attention_residual");
            mask_ckks_to_mpc_inplace(
                normalized_input_cts_half, normalized_input_pts, he);
            send_encrypted_vector(io, normalized_input_cts_half);
            cout << "FORMAL_TRANSFER_LAYOUT stage=ln2_attention_residual"
                 << " status=PASS layout=ordinary_half_slot cts="
                 << normalized_input_cts_half.size()
                 << " source=Block2_LN_affine_ciphertext" << endl;
        }
        // Load result ciphertexts to CPU.  Algorithm 1 samples masks in the
        // host SEAL RNS ring, but no ciphertext is decrypted on the evaluator.
        start = high_resolution_clock::now();
        if (ordinary_full_slot_output) {
            for (size_t i = 0; i < output_ct_count; ++i) {
                if (he->is_gpu_enable()) {
                    q_k_v[i].to_host(*he->context, *he->dcontext);
                }
                if (qkv_only_diagnostic) {
                    raw_qk_cts[i] = std::move(q_k_v[i].hcipher());
                } else if (i < qk_ct_count) {
                    raw_qk_cts[i] = std::move(q_k_v[i].hcipher());
                } else {
                    raw_v_cts[i - qk_ct_count] =
                        std::move(q_k_v[i].hcipher());
                }
            }
        } else if (!he->is_gpu_enable()) {
            for(int i=0;i<output_cts_half.size();i++){
                output_cts_half[i] = std::move(q_k_v[i].hcipher());
            }
        } else {
            for(int i=0;i<output_cts_half.size();i++){
                q_k_v[i].to_host(*he->context, *he->dcontext);
                output_cts_half[i] = std::move(q_k_v[i].hcipher());
            }
        }
        if (a0_trace && !raw_input) {
            if (a0_qkv_ciphertexts.size() != 3 * projected_tensor_ct_count) {
                throw runtime_error("A0 Q/K/V observer ciphertext partition mismatch");
            }
            for (size_t i = 0; i < projected_tensor_ct_count; ++i) {
                auto move_observer_ciphertext = [&](UnifiedCiphertext &source,
                                                    Ciphertext &target) {
                    if (he->is_gpu_enable()) {
                        source.to_host(*he->context, *he->dcontext);
                    }
                    target = std::move(source.hcipher());
                };
                move_observer_ciphertext(a0_qkv_ciphertexts[i], a0_q_cts[i]);
                move_observer_ciphertext(
                    a0_qkv_ciphertexts[projected_tensor_ct_count + i],
                    a0_k_cts[i]);
                move_observer_ciphertext(
                    a0_qkv_ciphertexts[2 * projected_tensor_ct_count + i],
                    a0_v_cts[i]);
            }
        }
        if (selective_conversion) {
            const size_t ciphertexts_per_pair =
                qk_ct_count / (PACKING_NUM / 2);
            const size_t v_ciphertexts_per_pair =
                v_ct_count / (PACKING_NUM / 2);
            auto mask_pair_halves = [&](Ciphertext &ciphertext,
                                      size_t ciphertext_index,
                                      size_t per_pair,
                                      bool qk_layout,
                                      bool mask_logical_padding,
                                      const auto &head_is_active,
                                      const char *stage) {
                const int pair = static_cast<int>(
                    ciphertext_index / per_pair);
                const bool first = head_is_active(2 * pair);
                const bool second = head_is_active(2 * pair + 1);
                if (!first && !second)
                    throw runtime_error(string(stage) +
                                        ": empty packed head pair");
                vector<double> mask(slot_count, 0.0);
                const size_t local_ciphertext = ciphertext_index % per_pair;
                const int columns_per_ciphertext =
                    static_cast<int>(slot_count / 2 / data.image_size);
                const int column_base =
                    static_cast<int>(local_ciphertext) *
                    columns_per_ciphertext;
                for (size_t slot_index = 0; slot_index < slot_count;
                     ++slot_index) {
                    const bool upper = slot_index >= slot_count / 2;
                    const bool active = upper ? second : first;
                    if (!active) continue;
                    const int within_half = static_cast<int>(
                        slot_index % (slot_count / 2));
                    const int token = within_half % data.image_size;
                    const int packed_column =
                        within_half / data.image_size + column_base;
                    // QK is diagonal-packed: this slot becomes
                    // row=query=token, column=(query+diagonal)%tokens in
                    // plain_cross_packing_postprocess. V is feature-packed
                    // and its slot token is already the key/token index.
                    const bool logical = !mask_logical_padding ||
                        (qk_layout
                             ? (token + packed_column) % data.image_size <
                                   ours_vit::kPlanLogicalTokens
                             : token < ours_vit::kPlanLogicalTokens);
                    mask[slot_index] = logical ? 1.0 : 0.0;
                }
                Plaintext mask_pt;
                he->encoder->encode(mask, ciphertext.parms_id(), 1.0,
                                    mask_pt);
                he->evaluator->multiply_plain_inplace(ciphertext, mask_pt);
                ++linear.ct_pt_multiply_count;
            };

            if (heterogeneous_ablation == ours_vit::Ablation::A4) {
                a4_linear_qk_he1_cts.assign(
                    qk_ct_count, UnifiedCiphertext(LOCATION::HOST));
                a4_linear_v_he1_cts.assign(
                    v_ct_count, UnifiedCiphertext(LOCATION::HOST));
                for (size_t index = 0; index < qk_ct_count; ++index) {
                    const int pair = static_cast<int>(
                        index / ciphertexts_per_pair);
                    const bool first = heterogeneous_protocols[2 * pair] ==
                        ours_vit::HeadProtocol::HeLinear;
                    const bool second =
                        heterogeneous_protocols[2 * pair + 1] ==
                        ours_vit::HeadProtocol::HeLinear;
                    if (!first && !second) continue;
                    a4_linear_qk_he1_cts[index].hcipher() =
                        raw_qk_cts[index];
                    mask_pair_halves(
                        a4_linear_qk_he1_cts[index].hcipher(), index,
                        ciphertexts_per_pair, true, true,
                        [&](int head) {
                            return heterogeneous_protocols[head] ==
                                   ours_vit::HeadProtocol::HeLinear;
                        }, "A4 QK half mask");
                }
                for (size_t index = 0; index < v_ct_count; ++index) {
                    const int pair = static_cast<int>(
                        index / v_ciphertexts_per_pair);
                    const bool first = heterogeneous_protocols[2 * pair] ==
                        ours_vit::HeadProtocol::HeLinear;
                    const bool second = heterogeneous_protocols[2 * pair + 1] ==
                        ours_vit::HeadProtocol::HeLinear;
                    if (!first && !second) continue;
                    a4_linear_v_he1_cts[index].hcipher() = raw_v_cts[index];
                    mask_pair_halves(
                        a4_linear_v_he1_cts[index].hcipher(), index,
                        v_ciphertexts_per_pair, false, true,
                        [&](int head) {
                            return heterogeneous_protocols[head] ==
                                   ours_vit::HeadProtocol::HeLinear;
                        }, "A4 V half mask");
                }
                a4_linear_operands_ready = true;
            } else {
                a4_linear_qk_he1_cts.clear();
                a4_linear_v_he1_cts.clear();
                a4_linear_operands_ready = false;
            }

            for (auto &group : selective_groups) {
                for (size_t local = 0; local < group.qk_indices.size();
                     ++local) {
                    const size_t index = group.qk_indices[local];
                    group.qk_cts[local] = raw_qk_cts[index];
                    mask_pair_halves(
                        group.qk_cts[local], index,
                        ciphertexts_per_pair, true, false,
                        [&](int head) {
                            return head >= group.head_begin &&
                                   head < group.head_end;
                        }, "A3 grouped QK half mask");
                }
                for (size_t local = 0; local < group.v_indices.size();
                     ++local) {
                    const size_t index = group.v_indices[local];
                    group.v_cts[local] = raw_v_cts[index];
                    mask_pair_halves(
                        group.v_cts[local], index,
                        v_ciphertexts_per_pair, false, false,
                        [&](int head) {
                            return head >= group.head_begin &&
                                   head < group.head_end;
                        }, "A3 grouped V half mask");
                }
            }
            cout << "OURS_A3_QKV_PARTITION status=PASS layer=" << layer_id
                 << " mpc_groups=" << selective_groups.size()
                 << " linear_heads="
                 << count(heterogeneous_protocols.begin(),
                          heterogeneous_protocols.end(),
                          ours_vit::HeadProtocol::HeLinear)
                 << " qk_pair_half_mask=1 v_pair_half_mask=1"
                 << " mpc_logical_padding_mask=normalization_output"
                 << " linear_logical_padding_mask=1"
                 << " retained_he1="
                 << (heterogeneous_ablation == ours_vit::Ablation::A4 ? 1 : 0)
                 << endl;
        }
        if (he->is_gpu_enable()) {
            this->ct_transfer_mem_la +=
                ((high_resolution_clock::now() - start)).count()/1e+9;
        }
        NVTX_POP(true);

        // BLB Algorithm 1: mask in the current RNS ring before sending. The
        // evaluator keeps -r as its field-to-ring input share.
        start = high_resolution_clock::now();
        if (ordinary_full_slot_output) {
            if (qkv_only_diagnostic) {
                prepare_host_conversion_from_actual_scale(
                    *this, raw_qk_cts, kFormalConversionScaleBits, he,
                    "debug_qkv");
                mask_ckks_to_mpc_inplace(raw_qk_cts, raw_qk_pts, he);
                send_encrypted_vector(io, raw_qk_cts);
            } else if (selective_conversion) {
                size_t qk_sent = 0;
                size_t v_sent = 0;
                for (auto &group : selective_groups) {
                    prepare_host_conversion_from_actual_scale(
                        *this, group.qk_cts,
                        kFormalConversionScaleBits, he,
                        "a3_group_qk");
                    mask_ckks_to_mpc_inplace(
                        group.qk_cts, group.qk_pts, he);
                    send_encrypted_vector(io, group.qk_cts);
                    prepare_host_conversion_from_actual_scale(
                        *this, group.v_cts,
                        kFormalConversionScaleBits, he,
                        "a3_group_v");
                    mask_ckks_to_mpc_inplace(
                        group.v_cts, group.v_pts, he);
                    send_encrypted_vector(io, group.v_cts);
                    qk_sent += group.qk_cts.size();
                    v_sent += group.v_cts.size();
                    cout << "OURS_A3_SELECTIVE_CKKS_TO_MPC status=PASS"
                         << " layer=" << layer_id
                         << " protocol="
                         << ours_vit::to_string(group.protocol)
                         << " physical_head_begin=" << group.head_begin
                         << " head_count="
                         << group.head_end - group.head_begin
                         << " qk_ciphertexts=" << group.qk_cts.size()
                         << " v_ciphertexts=" << group.v_cts.size()
                         << " packing_granularity=head_pair_half_mask"
                         << endl;
                }
                cout << "FORMAL_QKV_CONVERSION_LAYOUT status=PASS"
                     << " qk_cts=" << qk_sent
                     << " v_cts=" << v_sent
                     << " full_qk_cts=" << qk_ct_count
                     << " full_v_cts=" << v_ct_count
                     << " linear_qk_v_conversion=0"
                     << " packing=ordinary_full_slot_selective"
                     << endl;
            } else {
                prepare_host_conversion_from_actual_scale(
                    *this, raw_qk_cts, kFormalConversionScaleBits, he,
                    "layer0_qk");
                mask_ckks_to_mpc_inplace(raw_qk_cts, raw_qk_pts, he);
                send_encrypted_vector(io, raw_qk_cts);
                // The official BLB Block-2 boundary exports both QK and V from
                // he1 before Block 4 imports V into the distinct he2 context.
                // Keeping he1's V ciphertext in he2 is not a valid FGF edge:
                // Table 6 gives the two blocks different modulus chains.
                prepare_host_conversion_from_actual_scale(
                    *this, raw_v_cts, kFormalConversionScaleBits, he,
                    "layer0_v");
                mask_ckks_to_mpc_inplace(raw_v_cts, raw_v_pts, he);
                send_encrypted_vector(io, raw_v_cts);
            }
            attention_v_he_cts.clear();
            attention_v_he_ready = false;
            if (!selective_conversion)
            cout << "FORMAL_QKV_CONVERSION_LAYOUT status=PASS qk_cts="
                 << raw_qk_cts.size() << " qk_scale=" << NL_SCALE
                 << " v_cts=" << raw_v_cts.size()
                 << " v_scale=" << ATTENTION_V_SCALE
                 << " v_transport=CKKS_to_MPC packing=ordinary_full_slot"
                 << " raw_input=" << (raw_input ? 1 : 0)
                 << " diagnostic_qkv=" << (qkv_only_diagnostic ? 1 : 0)
                 << endl;
            if (a0_trace && !raw_input) {
                if (getenv("OURS_VIT_A0_NATIVE_OBSERVER_CHECK") != nullptr) {
                    // Diagnostic only: expose one unmasked native Q ciphertext
                    // before Algorithm 1 so a bad QKV kernel cannot be
                    // confused with a CKKS->MPC conversion failure. The
                    // normal A0 protocol never sends this extra copy.
                    vector<Ciphertext> native_q = a0_q_cts;
                    send_encrypted_vector(io, native_q);
                }
                prepare_host_conversion_from_actual_scale(
                    *this, a0_q_cts, kFormalConversionScaleBits, he,
                    "a0_q_observer");
                prepare_host_conversion_from_actual_scale(
                    *this, a0_k_cts, kFormalConversionScaleBits, he,
                    "a0_k_observer");
                prepare_host_conversion_from_actual_scale(
                    *this, a0_v_cts, kFormalConversionScaleBits, he,
                    "a0_v_observer");
                mask_ckks_to_mpc_inplace(a0_q_cts, a0_q_pts, he);
                mask_ckks_to_mpc_inplace(a0_k_cts, a0_k_pts, he);
                mask_ckks_to_mpc_inplace(a0_v_cts, a0_v_pts, he);
                send_encrypted_vector(io, a0_q_cts);
                send_encrypted_vector(io, a0_k_cts);
                send_encrypted_vector(io, a0_v_cts);
            }
        } else {
            // The non-raw QKV output crosses the same CKKS->MPC Algorithm-1
            // boundary as layer-0 QK/V.  Bring every ciphertext to the
            // target scale and final single-prime level before sampling the
            // mask; masking alone preserves the current multi-limb ring.
            prepare_host_conversion_from_actual_scale(
                *this, output_cts_half, kFormalConversionScaleBits, he,
                "layernorm2_attention_qkv");
            mask_ckks_to_mpc_inplace(output_cts_half, output_pts_half, he);
            send_encrypted_vector(io, output_cts_half);
            if (a0_trace) {
                prepare_host_conversion_from_actual_scale(
                    *this, a0_q_cts, kFormalConversionScaleBits, he,
                    "a0_q_observer");
                prepare_host_conversion_from_actual_scale(
                    *this, a0_k_cts, kFormalConversionScaleBits, he,
                    "a0_k_observer");
                prepare_host_conversion_from_actual_scale(
                    *this, a0_v_cts, kFormalConversionScaleBits, he,
                    "a0_v_observer");
                mask_ckks_to_mpc_inplace(a0_q_cts, a0_q_pts, he);
                mask_ckks_to_mpc_inplace(a0_k_cts, a0_k_pts, he);
                mask_ckks_to_mpc_inplace(a0_v_cts, a0_v_pts, he);
                send_encrypted_vector(io, a0_q_cts);
                send_encrypted_vector(io, a0_k_cts);
                send_encrypted_vector(io, a0_v_cts);
            }
        }
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    }
    if (ordinary_full_slot_output) {
        attention_v_he_ready = false;
    }
    start = high_resolution_clock::now();
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    // cout << "send and receive comm:" << this->get_comm() << endl;
    // cout << "send and receive round:" << this->get_round() << endl;
    // he->print_parameters(he->context->get_context_data(output_pts[0].parms_id())->parms());
    vector<vector<HE_CKKS::scalar_t>> output_vector;
    if (!raw_input && !pre_norm_attention_output_only) {
        auto normalized_input_vector = ckks_to_mpc(
            normalized_input_pts, degree, he->get_Q(normalized_input_pts[0]),
            NL_ELL, NL_SCALE, he, true);
        attention_input_residual_share = FixArray(
            party, static_cast<int>(x_flat_avg.size), true, NL_ELL, NL_SCALE);
        size_t normalized_offset = 0;
        for (const auto &part : normalized_input_vector) {
            if (part.size() != static_cast<size_t>(degree) ||
                normalized_offset + part.size() >
                    static_cast<size_t>(attention_input_residual_share.size)) {
                throw runtime_error(
                    "formal Block-2 LayerNorm residual decode mismatch");
            }
            copy(part.begin(), part.end(),
                 attention_input_residual_share.data + normalized_offset);
            normalized_offset += part.size();
        }
        if (normalized_offset !=
            static_cast<size_t>(attention_input_residual_share.size)) {
            throw runtime_error(
                "formal Block-2 LayerNorm residual is incomplete");
        }
        attention_input_residual_ready = true;
        cout << "FORMAL_LAYOUT_ADAPTER stage=ln2_attention_residual"
             << " status=PASS source=Block2_LN_affine"
             << " target=token_feature_row_major elements="
             << attention_input_residual_share.size
             << " communication=Algorithm1_once" << endl;
    }
    if (ordinary_full_slot_output) {
        if (qkv_only_diagnostic) {
            output_vector = ckks_to_mpc(
                raw_qk_pts, degree, he->get_Q(raw_qk_pts[0]), NL_ELL,
                ATTENTION_V_SCALE, he, false);
        } else if (selective_conversion) {
            output_vector.assign(
                qk_ct_count + v_ct_count,
                vector<HE_CKKS::scalar_t>(slot_count, 0));
            const uint64_t ring_mask = (uint64_t(1) << NL_ELL) - 1;
            const auto previous_conversion_bucket =
                profile_conversion_bucket();
            for (auto &group : selective_groups) {
                profile_set_conversion_bucket(
                    BlbViTCommAuditBucket::QK_TO_SOFTMAX_CONVERSION);
                auto qk_group = ckks_to_mpc(
                    group.qk_pts, degree,
                    he->get_Q(group.qk_pts.front()), NL_ELL, NL_SCALE,
                    he, false);
                profile_set_conversion_bucket(
                    BlbViTCommAuditBucket::OTHER_CONVERSION);
                auto v_group = ckks_to_mpc(
                    group.v_pts, degree,
                    he->get_Q(group.v_pts.front()), NL_ELL,
                    ATTENTION_V_SCALE, he, false);
                if (qk_group.size() != group.qk_indices.size() ||
                    v_group.size() != group.v_indices.size()) {
                    throw runtime_error(
                        "A3 grouped CKKS-to-MPC decode count mismatch");
                }
                for (size_t local = 0; local < qk_group.size(); ++local) {
                    auto &target = output_vector[group.qk_indices[local]];
                    for (size_t slot_index = 0;
                         slot_index < target.size(); ++slot_index) {
                        target[slot_index] =
                            (target[slot_index] +
                             qk_group[local][slot_index]) & ring_mask;
                    }
                }
                for (size_t local = 0; local < v_group.size(); ++local) {
                    auto &target = output_vector[
                        qk_ct_count + group.v_indices[local]];
                    for (size_t slot_index = 0;
                         slot_index < target.size(); ++slot_index) {
                        target[slot_index] =
                            (target[slot_index] +
                             v_group[local][slot_index]) & ring_mask;
                    }
                }
            }
            profile_set_conversion_bucket(previous_conversion_bucket);
            cout << "OURS_A3_GROUPED_CKKS_TO_MPC status=PASS layer="
                 << layer_id << " group_calls="
                 << selective_groups.size()
                 << " output_qk_slots=" << qk_ct_count * slot_count
                 << " output_v_slots=" << v_ct_count * slot_count
                 << " linear_slots=public_zero_shares" << endl;
        } else {
            const auto previous_conversion_bucket = profile_conversion_bucket();
            profile_set_conversion_bucket(
                BlbViTCommAuditBucket::QK_TO_SOFTMAX_CONVERSION);
            output_vector = ckks_to_mpc(
                raw_qk_pts, degree, he->get_Q(raw_qk_pts[0]), NL_ELL, NL_SCALE,
                he, false);
            profile_set_conversion_bucket(
                BlbViTCommAuditBucket::OTHER_CONVERSION);
            auto v_vector = ckks_to_mpc(
                raw_v_pts, degree, he->get_Q(raw_v_pts[0]), NL_ELL,
                ATTENTION_V_SCALE, he, false);
            output_vector.insert(output_vector.end(),
                                 make_move_iterator(v_vector.begin()),
                                 make_move_iterator(v_vector.end()));
            profile_set_conversion_bucket(previous_conversion_bucket);
        }
    } else {
        output_vector = ckks_to_mpc(output_pts_half, degree,
                                    he->get_Q(output_pts_half[0]), bitwidth, s,
                                    he, true);
    }
    if (a0_trace && !raw_input) {
        auto observer_to_fix = [&](const vector<Plaintext> &points,
                                   int fixed_scale,
                                   const char *stage) -> FixArray {
            auto pieces = ckks_to_mpc(
                points, degree, he->get_Q(points[0]), NL_ELL, fixed_scale,
                he, false);
            const size_t expected =
                projected_tensor_ct_count * static_cast<size_t>(slot_count);
            FixArray result(party, static_cast<int>(expected), true, NL_ELL,
                            fixed_scale);
            size_t offset = 0;
            for (const auto &piece : pieces) {
                if (piece.size() != static_cast<size_t>(slot_count) ||
                    offset + piece.size() > expected) {
                    throw runtime_error(string("A0 observer decode mismatch: ") +
                                        stage);
                }
                copy(piece.begin(), piece.end(), result.data + offset);
                offset += piece.size();
            }
            if (offset != expected) {
                throw runtime_error(string("A0 observer is incomplete: ") +
                                    stage);
            }
            return result;
        };
        a0_q_share = observer_to_fix(a0_q_pts, NL_SCALE, "q");
        a0_k_share = observer_to_fix(a0_k_pts, NL_SCALE, "k");
        a0_v_share = observer_to_fix(a0_v_pts, ATTENTION_V_SCALE, "v");
        a0_qkv_ready = true;
        cout << "OURS_A0_QKV_OBSERVER status=PASS q_elements="
             << a0_q_share.size << " k_elements=" << a0_k_share.size
             << " v_elements=" << a0_v_share.size
             << " production_path=0 intermediate_reconstruction=1" << endl;
    }
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    // cout << "ckks_to_mpc comm:" << this->get_comm() << endl;
    // cout << "ckks_to_mpc round:" << this->get_round() << endl;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    return output_vector;
}

vector<vector<HE_CKKS::scalar_t>> Bert_ckks::attention_from_real_input(
    const FixArray &input, int layer_id) {
    if (input.size <= 0 || input.size % COMMON_DIM != 0) {
        throw runtime_error("formal real BERT input shape mismatch");
    }
    map<char*, FixArray> raw;
    raw["raw_input"] = input;
    return layer_norm_part2_attention(raw, layer_id);
}

map<char*, FixArray> Bert_ckks::heterogeneous_attention_output_a3_a4(
    FixArray normalized_qk, int layer_id) {
    if (heterogeneous_ablation != ours_vit::Ablation::A3 &&
        heterogeneous_ablation != ours_vit::Ablation::A4) {
        throw runtime_error("selective attention output requires A3 or A4");
    }
    if (!pre_norm_attention_output_only ||
        heterogeneous_protocols.size() != PACKING_NUM ||
        heterogeneous_alphas.size() != PACKING_NUM ||
        normalized_qk.ell != NL_ELL || normalized_qk.s != NL_SCALE ||
        normalized_qk.size != PACKING_NUM * linear.data_lin1_0.image_size *
                                  linear.data_lin1_0.image_size ||
        !attention_v_ready ||
        attention_v_share.size !=
            linear.data_lin1_0.image_size * COMMON_DIM ||
        !attention_input_residual_ready ||
        attention_input_residual_share.size !=
            linear.data_lin1_0.image_size * COMMON_DIM) {
        throw runtime_error("A3/A4 attention output state mismatch");
    }

    const FCMetadata av_data = linear.data_lin1_0;
    const int tokens = av_data.image_size;
    const int slot = av_data.slot_count;
    const int qk_ct_count =
        PACKING_NUM * tokens * tokens / slot;
    const int v_ct_count = PACKING_NUM * tokens * OUTPUT_DIM / slot;
    const int qk_cts_per_pair =
        qk_ct_count / (PACKING_NUM / 2);
    const int v_cts_per_pair =
        v_ct_count / (PACKING_NUM / 2);
    if (qk_ct_count % (PACKING_NUM / 2) != 0 ||
        v_ct_count % (PACKING_NUM / 2) != 0 ||
        qk_cts_per_pair <= 0 || v_cts_per_pair <= 0)
        throw runtime_error("A3/A4 packed conversion geometry is invalid");

    vector<bool> mpc_heads(PACKING_NUM, false);
    vector<bool> linear_heads(PACKING_NUM, false);
    for (int head = 0; head < PACKING_NUM; ++head) {
        linear_heads[head] = heterogeneous_protocols[head] ==
                             ours_vit::HeadProtocol::HeLinear;
        mpc_heads[head] = !linear_heads[head];
    }
    const int mpc_head_count = count(mpc_heads.begin(), mpc_heads.end(), true);
    const int linear_head_count =
        count(linear_heads.begin(), linear_heads.end(), true);
    int linear_pair_count = 0;
    for (int pair = 0; pair < PACKING_NUM / 2; ++pair)
        linear_pair_count +=
            (linear_heads[2 * pair] || linear_heads[2 * pair + 1]) ? 1 : 0;
    if (heterogeneous_ablation == ours_vit::Ablation::A3 &&
        linear_head_count != 0)
        throw runtime_error("A3 cannot execute a native Linear head");
    auto add_plain_share = [](HE_CKKS *he,
                              vector<UnifiedCiphertext> &ciphertexts,
                              const vector<UnifiedPlaintext> &shares) {
        if (ciphertexts.size() != shares.size())
            throw runtime_error("grouped MPC-to-CKKS share count mismatch");
        for (size_t i = 0; i < ciphertexts.size(); ++i)
            he->evaluator->add_plain_inplace(ciphertexts[i], shares[i]);
    };
    auto add_or_move_ciphertext = [](HE_CKKS *he,
                                     UnifiedCiphertext &target,
                                     uint8_t &ready,
                                     UnifiedCiphertext source) {
        if (!ready) {
            target = std::move(source);
            ready = true;
        } else {
            he->evaluator->add_inplace(target, source);
        }
    };
    auto zero_host_ciphertext = [&](HE_CKKS *he,
                                    const UnifiedCiphertext &reference) {
        UnifiedCiphertext result = reference;
        vector<double> zeros(he->num_slots, 0.0);
        UnifiedPlaintext zero(LOCATION::HOST);
        he->encoder->encode(zeros, reference.hcipher().parms_id(), 1.0,
                            zero);
        he->evaluator->multiply_plain_inplace(result, zero);
        ++linear.ct_pt_multiply_count;
        return result;
    };

    vector<UnifiedCiphertext> mpc_qk_cts(
        qk_ct_count, UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> mpc_v_cts(
        v_ct_count, UnifiedCiphertext(LOCATION::HOST));
    vector<uint8_t> mpc_qk_ready(qk_ct_count, 0);
    vector<uint8_t> mpc_v_ready(v_ct_count, 0);
    if (mpc_head_count > 0) {
        vector<uint64_t> packed_qk(normalized_qk.size);
        linear.preprocess_softmax(normalized_qk.data, packed_qk.data(),
                                  av_data);
        // MPC-normalized groups stay on BLB's original he2 schedule.  The
        // extended he1 chain is reserved exclusively for native A4 Linear
        // heads, so Softmax/ReLU work never pays its extra modulus limbs.
        HE_CKKS *he = linear.he2;
        const int import_scale_bits = he->he_scale;
        for (int begin = 0; begin < PACKING_NUM;) {
            int end = begin + 1;
            while (end < PACKING_NUM &&
                   heterogeneous_protocols[end] ==
                       heterogeneous_protocols[begin])
                ++end;
            if (heterogeneous_protocols[begin] ==
                ours_vit::HeadProtocol::HeLinear) {
                begin = end;
                continue;
            }

            vector<int> qk_indices;
            vector<vector<HE_CKKS::scalar_t>> qk_shares;
            for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
                const bool first = 2 * pair >= begin && 2 * pair < end;
                const bool second = 2 * pair + 1 >= begin &&
                                    2 * pair + 1 < end;
                if (!first && !second) continue;
                for (int local = 0; local < qk_cts_per_pair; ++local) {
                    const int index = pair * qk_cts_per_pair + local;
                    qk_indices.push_back(index);
                    vector<HE_CKKS::scalar_t> share(
                        packed_qk.begin() + static_cast<size_t>(index) * slot,
                        packed_qk.begin() +
                            static_cast<size_t>(index + 1) * slot);
                    if (!first)
                        fill(share.begin(), share.begin() + slot / 2, 0);
                    if (!second)
                        fill(share.begin() + slot / 2, share.end(), 0);
                    qk_shares.push_back(std::move(share));
                }
            }
            vector<int> v_indices;
            vector<vector<HE_CKKS::scalar_t>> v_shares;
            for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
                const bool first = 2 * pair >= begin && 2 * pair < end;
                const bool second = 2 * pair + 1 >= begin &&
                                    2 * pair + 1 < end;
                if (!first && !second) continue;
                for (int local = 0; local < v_cts_per_pair; ++local) {
                    const int index = pair * v_cts_per_pair + local;
                    v_indices.push_back(index);
                    vector<HE_CKKS::scalar_t> share(
                        attention_v_share.data +
                            static_cast<size_t>(index) * slot,
                        attention_v_share.data +
                            static_cast<size_t>(index + 1) * slot);
                    if (!first)
                        fill(share.begin(), share.begin() + slot / 2, 0);
                    if (!second)
                        fill(share.begin() + slot / 2, share.end(), 0);
                    v_shares.push_back(std::move(share));
                }
            }
            vector<UnifiedPlaintext> qk_pts;
            vector<UnifiedPlaintext> v_pts;
            const auto conversion_start = high_resolution_clock::now();
            const uint64_t conversion_comm_start = current_comm();
            const uint64_t conversion_round_start = current_round();
            import_target_scale(*this, qk_shares, slot, NL_ELL, NL_SCALE,
                                import_scale_bits, he, qk_pts,
                                "a3_grouped_qk");
            import_target_scale(*this, v_shares, slot, NL_ELL,
                                ATTENTION_V_SCALE, import_scale_bits, he, v_pts,
                                "a3_grouped_v");
            vector<UnifiedCiphertext> group_qk(
                qk_pts.size(), UnifiedCiphertext(LOCATION::HOST));
            vector<UnifiedCiphertext> group_v(
                v_pts.size(), UnifiedCiphertext(LOCATION::HOST));
            if (party == BOB) {
                for (size_t i = 0; i < group_qk.size(); ++i)
                    he->encryptor->encrypt(qk_pts[i], group_qk[i]);
                for (size_t i = 0; i < group_v.size(); ++i)
                    he->encryptor->encrypt(v_pts[i], group_v[i]);
                send_encrypted_vector(io, group_qk);
                send_encrypted_vector(io, group_v);
            } else {
                recv_encrypted_vector(he->context, io, group_qk);
                recv_encrypted_vector(he->context, io, group_v);
                add_plain_share(he, group_qk, qk_pts);
                add_plain_share(he, group_v, v_pts);
                for (size_t local = 0; local < group_qk.size(); ++local)
                    add_or_move_ciphertext(
                        he, mpc_qk_cts[qk_indices[local]],
                        mpc_qk_ready[qk_indices[local]],
                        std::move(group_qk[local]));
                for (size_t local = 0; local < group_v.size(); ++local)
                    add_or_move_ciphertext(
                        he, mpc_v_cts[v_indices[local]],
                        mpc_v_ready[v_indices[local]],
                        std::move(group_v[local]));
            }
            cout << "OURS_A3_SELECTIVE_MPC_TO_CKKS status=PASS layer="
                 << layer_id << " protocol="
                 << ours_vit::to_string(heterogeneous_protocols[begin])
                 << " physical_head_begin=" << begin
                 << " head_count=" << end - begin
                 << " qk_ciphertexts=" << qk_indices.size()
                 << " v_ciphertexts=" << v_indices.size()
                 << " latency_ms="
                 << chrono::duration<double, milli>(
                        high_resolution_clock::now() - conversion_start)
                        .count()
                 << " communication_bytes="
                 << current_comm() - conversion_comm_start
                 << " rounds="
                 << current_round() - conversion_round_start
                 << " packing=head_pair_half_mask"
                 << " grouped_call=1 dynamic_selection=0" << endl;
            begin = end;
        }
    }

    // Preserve BLB's add-before-final-rescale residual boundary.  A mixed A4
    // layer anchors the residual to the he2 MPC-head contribution; an
    // all-Linear layer anchors it to the native he1 contribution.
    HE_CKKS *output_he = mpc_head_count > 0 ? linear.he2 : linear.he1;
    const int output_ct_count =
        linear.data_lin2.image_size * linear.data_lin2.filter_w / slot;
    vector<UnifiedCiphertext> residual_cts;
    {
        FixArray residual_packed = token_major_to_bsgs_output(
            attention_input_residual_share, linear.data_lin2,
            "a3_a4_attention_residual_to_h3");
        vector<vector<HE_CKKS::scalar_t>> residual_shares(output_ct_count);
        for (int i = 0; i < output_ct_count; ++i) {
            residual_shares[i].assign(
                residual_packed.data + static_cast<size_t>(i) * slot,
                residual_packed.data + static_cast<size_t>(i + 1) * slot);
        }
        vector<UnifiedPlaintext> residual_pts;
        residual_cts.assign(
            output_ct_count, UnifiedCiphertext(LOCATION::HOST));
        const auto residual_conversion_start = high_resolution_clock::now();
        const uint64_t residual_conversion_comm_start = current_comm();
        const uint64_t residual_conversion_round_start = current_round();
        import_target_scale(
            *this, residual_shares, slot, attention_input_residual_share.ell,
            attention_input_residual_share.s, output_he->he_scale, output_he,
            residual_pts, "a3_a4_attention_residual");
        if (party == BOB) {
            for (int i = 0; i < output_ct_count; ++i)
                output_he->encryptor->encrypt(residual_pts[i], residual_cts[i]);
            send_encrypted_vector(io, residual_cts);
        } else {
            recv_encrypted_vector(output_he->context, io, residual_cts);
            add_plain_share(output_he, residual_cts, residual_pts);
        }
        cout << "OURS_A3_A4_RESIDUAL_MPC_TO_CKKS status=PASS layer="
             << layer_id << " context="
             << (output_he == linear.he1 ? "he1" : "he2")
             << " ciphertexts=" << output_ct_count
             << " latency_ms="
             << chrono::duration<double, milli>(
                    high_resolution_clock::now() - residual_conversion_start)
                    .count()
             << " communication_bytes="
             << current_comm() - residual_conversion_comm_start
             << " rounds="
             << current_round() - residual_conversion_round_start
             << " layout=bsgs2_split_feature_halves" << endl;
    }

    vector<UnifiedCiphertext> non_linear_h3;
    if (mpc_head_count > 0 && party == sci::ALICE) {
        const auto he_path_start = high_resolution_clock::now();
        HE_CKKS *he = linear.he2;
        // In a boundary pair, the inactive V operand is an encrypted public
        // zero at identical metadata. This preserves BLB half/complex packing
        // without executing or importing the inactive head.
        int reference_v = -1;
        for (int i = 0; i < v_ct_count; ++i)
            if (mpc_v_ready[i]) { reference_v = i; break; }
        if (reference_v < 0)
            throw runtime_error("A3 grouped V import is empty");
        for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
            if (!mpc_heads[2 * pair] && !mpc_heads[2 * pair + 1]) continue;
            for (int local = 0; local < v_cts_per_pair; ++local) {
                const int index = pair * v_cts_per_pair + local;
                if (!mpc_v_ready[index]) {
                    mpc_v_cts[index] =
                        zero_host_ciphertext(he, mpc_v_cts[reference_v]);
                    mpc_v_ready[index] = true;
                }
            }
        }
        // Both imported operands are ordinary full-slot additive shares.
        // Project to the real subspace exactly as the original BLB ViT path;
        // its factor four is absorbed by the unchanged AV mask 0.25.
        for (int i = 0; i < qk_ct_count; ++i) {
            if (!mpc_qk_ready[i]) continue;
            UnifiedCiphertext conjugate(LOCATION::HOST);
            he->evaluator->complex_conjugate(
                mpc_qk_cts[i], *he->gal_keys, conjugate);
            he->evaluator->add_inplace(mpc_qk_cts[i], conjugate);
        }
        for (int i = 0; i < v_ct_count; ++i) {
            if (!mpc_v_ready[i]) continue;
            UnifiedCiphertext conjugate(LOCATION::HOST);
            he->evaluator->complex_conjugate(
                mpc_v_cts[i], *he->gal_keys, conjugate);
            he->evaluator->add_inplace(mpc_v_cts[i], conjugate);
        }
        linear.rotation_count +=
            count(mpc_qk_ready.begin(), mpc_qk_ready.end(), true) +
            count(mpc_v_ready.begin(), mpc_v_ready.end(), true);
        if (he->is_gpu_enable()) {
            for (int i = 0; i < qk_ct_count; ++i)
                if (mpc_qk_ready[i])
                    mpc_qk_cts[i].to_device(*he->context, *he->dcontext);
            for (int i = 0; i < v_ct_count; ++i)
                if (mpc_v_ready[i])
                    mpc_v_cts[i].to_device(*he->context, *he->dcontext);
        }
        // The measured BLB AV kernel reuses packed diagonals without the
        // selector multiply/rescale.  Drop the same public modulus level once
        // on the compact source vectors, before diagonal aliases are created,
        // so AV still enters the unchanged W_O cache at chain 3 / scale 2^69.
        uint64_t blb_fast_modswitches = 0;
        for (int i = 0; i < qk_ct_count; ++i) {
            if (!mpc_qk_ready[i]) continue;
            if (mpc_qk_cts[i].location() == LOCATION::HOST)
                he->evaluator->mod_switch_to_next_inplace(mpc_qk_cts[i]);
            else
                he->devaluator->mod_switch_to_next_inplace(mpc_qk_cts[i]);
            ++blb_fast_modswitches;
        }
        for (int i = 0; i < v_ct_count; ++i) {
            if (!mpc_v_ready[i]) continue;
            if (mpc_v_cts[i].location() == LOCATION::HOST)
                he->evaluator->mod_switch_to_next_inplace(mpc_v_cts[i]);
            else
                he->devaluator->mod_switch_to_next_inplace(mpc_v_cts[i]);
            ++blb_fast_modswitches;
        }
        cout << "OURS_BLB_FAST_AV_LEVEL status=PASS layer=" << layer_id
             << " context=he2 mod_switch_only=" << blb_fast_modswitches
             << " extra_rescales=0 extra_multiplies=0" << endl;
        if (profile_enabled_)
            profile_switch(BlbViTProfileStage::SOFTMAX_V);
        const auto softmax_v_profile_start = high_resolution_clock::now();
        vector<vector<vector<UnifiedCiphertext>>> packed_qk;
        linear.preprocess_softmax_s1_ct_ct(
            he, mpc_qk_cts, av_data, packed_qk, &mpc_heads, true);
        const auto &level_reference = [&]() -> const UnifiedCiphertext & {
            for (int pair = 0; pair < PACKING_NUM / 2; ++pair)
                if (mpc_heads[2 * pair] || mpc_heads[2 * pair + 1])
                    return packed_qk[pair][0][0];
            throw runtime_error("A3 packed QK reference is unavailable");
        }();
        const size_t target_limbs = level_reference.coeff_modulus_size();
        for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
            if (!mpc_heads[2 * pair] && !mpc_heads[2 * pair + 1]) continue;
            for (int local = 0; local < v_cts_per_pair; ++local) {
                auto &ciphertext =
                    mpc_v_cts[pair * v_cts_per_pair + local];
                while (ciphertext.coeff_modulus_size() > target_limbs)
                    he->devaluator->mod_switch_to_next_inplace(ciphertext);
                if (ciphertext.coeff_modulus_size() != target_limbs)
                    throw runtime_error("A3 V crossed below packed QK level");
            }
        }
        vector<UnifiedCiphertext> av_result(
            v_ct_count, UnifiedCiphertext(he->backend()));
        linear.softmax_v(he, packed_qk, mpc_v_cts, av_data, av_result,
                         0.25, &mpc_heads, true);
        if (profile_enabled_)
            profile_record_he(
                BlbViTProfileStage::SOFTMAX_V,
                chrono::duration<double, milli>(
                    high_resolution_clock::now() - softmax_v_profile_start)
                    .count());
        if (profile_enabled_)
            profile_switch(BlbViTProfileStage::OUTPUT_PROJECTION);
        const auto output_projection_profile_start =
            high_resolution_clock::now();
        linear.linear_2(
            he, av_result,
            heterogeneous_ablation == ours_vit::Ablation::A4
                ? linear.pp_2_mpc[layer_id]
                : linear.pp_2[layer_id],
            linear.data_lin2, non_linear_h3);
        if (profile_enabled_)
            profile_record_he(
                BlbViTProfileStage::OUTPUT_PROJECTION,
                chrono::duration<double, milli>(
                    high_resolution_clock::now() -
                    output_projection_profile_start)
                    .count());
        const double he_path_seconds = chrono::duration<double>(
            high_resolution_clock::now() - he_path_start).count();
        he_compute_la += he_path_seconds;
        cout << "OURS_A3_HE_PATH_PROFILE status=PASS layer=" << layer_id
             << " latency_ms=" << he_path_seconds * 1000.0
             << " active_heads=" << mpc_head_count
             << " execution_domain=he2_grouped"
             << endl;
    }

    vector<UnifiedCiphertext> linear_h3;
    if (linear_head_count > 0 && party == sci::ALICE) {
        const auto he_path_start = high_resolution_clock::now();
        const uint64_t he_path_comm_start = current_comm();
        const uint64_t he_path_round_start = current_round();
        const double he_path_conversion_start = conversion_la;
        const uint64_t he_path_conversion_ct_start = conversion_ciphertexts;
        const uint64_t he_path_ct_pt_start = linear.ct_pt_multiply_count;
        const uint64_t he_path_ct_ct_start = linear.ct_ct_multiply_count;
        const uint64_t he_path_rotation_start = linear.rotation_count;
        const uint64_t he_path_rescale_start = linear.rescale_count;
        if (!a4_linear_operands_ready ||
            a4_linear_qk_he1_cts.size() != static_cast<size_t>(qk_ct_count) ||
            a4_linear_v_he1_cts.size() != static_cast<size_t>(v_ct_count)) {
            throw runtime_error("A4 native he1 operands are unavailable");
        }
        HE_CKKS *he = linear.he1;
        int reference_qk = -1;
        int reference_v = -1;
        for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
            if (!linear_heads[2 * pair] && !linear_heads[2 * pair + 1])
                continue;
            if (reference_qk < 0) reference_qk = pair * qk_cts_per_pair;
            if (reference_v < 0) reference_v = pair * v_cts_per_pair;
        }
        if (reference_qk < 0 || reference_v < 0)
            throw runtime_error("A4 active packed pair is empty");
        for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
            if (!linear_heads[2 * pair] && !linear_heads[2 * pair + 1])
                continue;
            for (int local = 0; local < v_cts_per_pair; ++local) {
                const int index = pair * v_cts_per_pair + local;
                if (a4_linear_v_he1_cts[index].hcipher().size() == 0)
                    throw runtime_error("A4 active V pair was not retained");
            }
        }
        if (profile_enabled_)
            profile_switch(BlbViTProfileStage::SOFTMAX_V);
        const auto softmax_v_profile_start = high_resolution_clock::now();
        // Lower native V from the QKV scale to native QK's exact scale using
        // one public value-one multiply and the single added A4 level. This is
        // real CKKS arithmetic; no metadata relabel or client bridge occurs.
        const double source_v_scale =
            a4_linear_v_he1_cts[reference_v].scale();
        const double target_qk_scale =
            a4_linear_qk_he1_cts[reference_qk].scale();
        auto context = he->context->get_context_data(
            a4_linear_v_he1_cts[reference_v].hcipher().parms_id());
        if (!context || !context->next_context_data() ||
            context->parms().coeff_modulus().empty())
            throw runtime_error("A4 V scale alignment lacks one level");
        const double dropped_modulus =
            context->parms().coeff_modulus().back().value();
        const double unit_scale =
            target_qk_scale * dropped_modulus / source_v_scale;
        vector<double> ones(slot, 1.0);
        for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
            if (!linear_heads[2 * pair] && !linear_heads[2 * pair + 1])
                continue;
            for (int local = 0; local < v_cts_per_pair; ++local) {
                auto &ciphertext = a4_linear_v_he1_cts[
                    pair * v_cts_per_pair + local];
                Plaintext one;
                he->encoder->encode(ones, ciphertext.hcipher().parms_id(),
                                    unit_scale, one);
                he->evaluator->multiply_plain_inplace(ciphertext, one);
                he->evaluator->rescale_to_next_inplace(ciphertext);
                ++linear.ct_pt_multiply_count;
                ++linear.rescale_count;
            }
        }
        const size_t qk_raw_limbs =
            a4_linear_qk_he1_cts[reference_qk].coeff_modulus_size();
        if (qk_raw_limbs < 1)
            throw runtime_error("A4 QK has no native CKKS data level");
        // A3/A4 deliberately use the measured BLB diagonal-reuse kernel,
        // which consumes no QK level.  V's one native alignment rescale lands
        // directly on the raw QK level.
        const size_t av_target_limbs = qk_raw_limbs;
        for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
            if (!linear_heads[2 * pair] && !linear_heads[2 * pair + 1])
                continue;
            for (int local = 0; local < v_cts_per_pair; ++local) {
                auto &ciphertext = a4_linear_v_he1_cts[
                    pair * v_cts_per_pair + local];
                while (ciphertext.coeff_modulus_size() > av_target_limbs)
                    he->evaluator->mod_switch_to_next_inplace(ciphertext);
                const double error =
                    abs(ciphertext.scale() - target_qk_scale) /
                    target_qk_scale;
                if (ciphertext.coeff_modulus_size() != av_target_limbs ||
                    !isfinite(error) || error > 1e-9)
                    throw runtime_error("A4 V scale/level alignment failed");
            }
        }
        if (he->is_gpu_enable()) {
            for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
                if (!linear_heads[2 * pair] && !linear_heads[2 * pair + 1])
                    continue;
                for (int local = 0; local < qk_cts_per_pair; ++local)
                    a4_linear_qk_he1_cts[pair * qk_cts_per_pair + local]
                        .to_device(*he->context, *he->dcontext);
                for (int local = 0; local < v_cts_per_pair; ++local)
                    a4_linear_v_he1_cts[pair * v_cts_per_pair + local]
                        .to_device(*he->context, *he->dcontext);
            }
        }
        // As above, consume the selector's public level once on the compact
        // native operands.  QK and V remain encrypted in he1; this is a CKKS
        // modulus switch, not a conversion or a surrogate computation.
        uint64_t blb_fast_modswitches = 0;
        for (int pair = 0; pair < PACKING_NUM / 2; ++pair) {
            if (!linear_heads[2 * pair] && !linear_heads[2 * pair + 1])
                continue;
            for (int local = 0; local < qk_cts_per_pair; ++local) {
                auto &ciphertext = a4_linear_qk_he1_cts[
                    pair * qk_cts_per_pair + local];
                if (ciphertext.location() == LOCATION::HOST)
                    he->evaluator->mod_switch_to_next_inplace(ciphertext);
                else
                    he->devaluator->mod_switch_to_next_inplace(ciphertext);
                ++blb_fast_modswitches;
            }
            for (int local = 0; local < v_cts_per_pair; ++local) {
                auto &ciphertext = a4_linear_v_he1_cts[
                    pair * v_cts_per_pair + local];
                if (ciphertext.location() == LOCATION::HOST)
                    he->evaluator->mod_switch_to_next_inplace(ciphertext);
                else
                    he->devaluator->mod_switch_to_next_inplace(ciphertext);
                ++blb_fast_modswitches;
            }
        }
        cout << "OURS_BLB_FAST_AV_LEVEL status=PASS layer=" << layer_id
             << " context=he1 mod_switch_only=" << blb_fast_modswitches
             << " extra_rescales=0 extra_multiplies=0"
             << " client_decrypt_reencrypt=0 mpc_reference=0" << endl;
        vector<vector<vector<UnifiedCiphertext>>> packed_qk;
        linear.preprocess_softmax_s1_ct_ct(
            he, a4_linear_qk_he1_cts, av_data, packed_qk, &linear_heads,
            true);
        vector<UnifiedCiphertext> av_result(
            v_ct_count, UnifiedCiphertext(he->backend()));
        linear.softmax_v(he, packed_qk, a4_linear_v_he1_cts, av_data,
                         av_result, 1.0, &linear_heads, false);
        if (profile_enabled_)
            profile_record_he(
                BlbViTProfileStage::SOFTMAX_V,
                chrono::duration<double, milli>(
                    high_resolution_clock::now() - softmax_v_profile_start)
                    .count());
        if (profile_enabled_)
            profile_switch(BlbViTProfileStage::OUTPUT_PROJECTION);
        const auto output_projection_profile_start =
            high_resolution_clock::now();
        linear.linear_2(he, av_result, linear.pp_2_a4[layer_id],
                        linear.data_lin2, linear_h3);
        if (profile_enabled_)
            profile_record_he(
                BlbViTProfileStage::OUTPUT_PROJECTION,
                chrono::duration<double, milli>(
                    high_resolution_clock::now() -
                    output_projection_profile_start)
                    .count());
        cout << "OURS_A4_NATIVE_CKKS status=PASS layer=" << layer_id
             << " linear_heads=" << linear_head_count
             << " active_pairs=" << linear_pair_count
             << " qk_scale_bits=" << log2(target_qk_scale)
             << " v_scale_bits="
             << log2(a4_linear_v_he1_cts[reference_v].scale())
             << " alpha_fused_into=WO"
             << " v_alignment_rescales=1"
             << " client_decrypt_reencrypt=0 mpc_reference=0" << endl;
        const double he_path_seconds = chrono::duration<double>(
            high_resolution_clock::now() - he_path_start).count();
        he_compute_la += he_path_seconds;
        const uint64_t he_path_conversion_ct =
            conversion_ciphertexts - he_path_conversion_ct_start;
        const uint64_t he_path_he_muls =
            linear.ct_pt_multiply_count - he_path_ct_pt_start +
            linear.ct_ct_multiply_count - he_path_ct_ct_start;
        cout << "OURS_A4_HE_PATH_PROFILE status=PASS layer=" << layer_id
             << " latency_ms=" << he_path_seconds * 1000.0
             << " he_ms=" << he_path_seconds * 1000.0
             << " mpc_ms=0 conversion_ms="
             << (conversion_la - he_path_conversion_start) * 1000.0
             << " conversion_ct=" << he_path_conversion_ct
             << " he_muls=" << he_path_he_muls
             << " he_rotations="
             << linear.rotation_count - he_path_rotation_start
             << " he_rescales="
             << linear.rescale_count - he_path_rescale_start
             << " communication_bytes="
             << current_comm() - he_path_comm_start
             << " rounds=" << current_round() - he_path_round_start
             << " active_heads=" << linear_head_count
             << " active_pairs=" << linear_pair_count
             << " execution_domain=he1_native" << endl;
    }

    if (party == sci::ALICE) {
        vector<UnifiedCiphertext> *residual_anchor =
            mpc_head_count > 0 ? &non_linear_h3 : &linear_h3;
        if (residual_anchor->size() != residual_cts.size() ||
            residual_anchor->empty())
            throw runtime_error("A3/A4 residual merge size mismatch");
        const auto residual_he_start = high_resolution_clock::now();
        if (output_he->is_gpu_enable()) {
            for (auto &ciphertext : residual_cts)
                ciphertext.to_device(*output_he->context,
                                     *output_he->dcontext);
        }
        align_unified_to_cipher_metadata(
            linear, output_he, residual_cts, *residual_anchor,
            "a3_a4_attention_residual");
        for (size_t i = 0; i < residual_anchor->size(); ++i) {
            if (output_he->is_gpu_enable()) {
                output_he->devaluator->add_inplace((*residual_anchor)[i],
                                                    residual_cts[i]);
                output_he->devaluator->rescale_to_next_inplace(
                    (*residual_anchor)[i]);
            } else {
                output_he->evaluator->add_inplace((*residual_anchor)[i],
                                                   residual_cts[i]);
                output_he->evaluator->rescale_to_next_inplace(
                    (*residual_anchor)[i]);
            }
            ++linear.rescale_count;
        }
        size_t final_rescales = residual_anchor->size();
        // Mixed A4 contributions live in distinct CKKS parameter contexts.
        // The he2 contribution receives the residual above.  Finish the he1
        // Linear contribution independently, then combine their two genuine
        // Algorithm-1 exports as additive MPC shares below.
        if (heterogeneous_ablation == ours_vit::Ablation::A4 &&
            mpc_head_count > 0 && linear_head_count > 0) {
            if (linear_h3.size() != static_cast<size_t>(output_ct_count))
                throw runtime_error("A4 Linear WO contribution is incomplete");
            for (auto &ciphertext : linear_h3) {
                if (ciphertext.location() == LOCATION::DEVICE)
                    linear.he1->devaluator->rescale_to_next_inplace(ciphertext);
                else
                    linear.he1->evaluator->rescale_to_next_inplace(ciphertext);
                ++linear.rescale_count;
            }
            final_rescales += linear_h3.size();
        }
        const double residual_he_seconds = chrono::duration<double>(
            high_resolution_clock::now() - residual_he_start).count();
        he_compute_la += residual_he_seconds;
        cout << "OURS_A3_A4_CKKS_RESIDUAL_MERGE status=PASS layer="
             << layer_id << " ciphertexts=" << residual_anchor->size()
             << " add_before_rescale=1 final_rescales="
             << final_rescales
             << " residual_context="
             << (output_he == linear.he1 ? "he1" : "he2")
             << " latency_ms=" << residual_he_seconds * 1000.0 << endl;
    }

    auto export_contribution = [&](HE_CKKS *he,
                                   vector<UnifiedCiphertext> &ciphertexts,
                                   bool enabled,
                                   const char *stage) -> FixArray {
        const auto conversion_start = high_resolution_clock::now();
        const uint64_t conversion_comm_start = current_comm();
        const uint64_t conversion_round_start = current_round();
        const int output_cts =
            linear.data_lin2.image_size * linear.data_lin2.filter_w /
            he->num_slots;
        vector<Ciphertext> transport(output_cts);
        vector<Plaintext> points(output_cts);
        if (party == BOB) {
            if (enabled) {
                recv_encrypted_vector(he->context, io, transport);
                for (int i = 0; i < output_cts; ++i)
                    he->decryptor->decrypt(transport[i], points[i]);
            }
        } else if (enabled) {
            if (ciphertexts.size() != static_cast<size_t>(output_cts))
                throw runtime_error(string(stage) +
                                    ": output ciphertext count mismatch");
            for (int i = 0; i < output_cts; ++i) {
                if (he->is_gpu_enable())
                    ciphertexts[i].to_host(*he->context, *he->dcontext);
                transport[i] = std::move(ciphertexts[i].hcipher());
            }
            prepare_host_conversion_from_actual_scale(
                *this, transport, kFormalConversionScaleBits, he, stage);
            mask_ckks_to_mpc_inplace(transport, points, he);
            send_encrypted_vector(io, transport);
        }
        if (!enabled)
            return FixArray(party, tokens * COMMON_DIM, true, NL_ELL,
                            NL_SCALE);
        auto pieces = ckks_to_mpc(
            points, he->poly_modulus_degree, he->get_Q(points.front()),
            NL_ELL, NL_SCALE, he, false);
        FixArray packed(party, output_cts * he->num_slots, true, NL_ELL,
                        NL_SCALE);
        size_t offset = 0;
        for (const auto &piece : pieces) {
            if (piece.size() != static_cast<size_t>(he->num_slots) ||
                offset + piece.size() > static_cast<size_t>(packed.size))
                throw runtime_error(string(stage) + ": decode mismatch");
            copy(piece.begin(), piece.end(), packed.data + offset);
            offset += piece.size();
        }
        if (offset != static_cast<size_t>(packed.size))
            throw runtime_error(string(stage) + ": incomplete decode");
        cout << "OURS_SELECTIVE_OUTPUT_CONVERSION status=PASS layer="
             << layer_id << " stage=" << stage
             << " ciphertexts=" << output_cts
             << " latency_ms="
             << chrono::duration<double, milli>(
                    high_resolution_clock::now() - conversion_start).count()
             << " communication_bytes="
             << current_comm() - conversion_comm_start
             << " rounds=" << current_round() - conversion_round_start
             << " algorithm=BLB_Algorithm_1 client_reencrypt=0" << endl;
        return bsgs_output_to_token_major(
            packed, linear.data_lin2, stage);
    };

    FixArray result(party, tokens * COMMON_DIM, true, NL_ELL, NL_SCALE);
    if (heterogeneous_ablation == ours_vit::Ablation::A3) {
        result = export_contribution(
            linear.he2, non_linear_h3, true, "a3_mpc_heads_wo");
    } else {
        FixArray mpc_contribution(
            party, tokens * COMMON_DIM, true, NL_ELL, NL_SCALE);
        FixArray linear_contribution(
            party, tokens * COMMON_DIM, true, NL_ELL, NL_SCALE);
        if (mpc_head_count > 0)
            mpc_contribution = export_contribution(
                linear.he2, non_linear_h3, true, "a4_mpc_heads_wo");
        if (linear_head_count > 0)
            linear_contribution = export_contribution(
                linear.he1, linear_h3, true, "a4_linear_heads_wo");
        if (mpc_head_count > 0 && linear_head_count > 0) {
            result = nonlinear.fpmath[0]->fix->add(
                mpc_contribution, linear_contribution);
        } else if (mpc_head_count > 0) {
            result = std::move(mpc_contribution);
        } else {
            result = std::move(linear_contribution);
        }
    }
    attention_v_ready = false;
    attention_input_residual_ready = false;
    a4_linear_operands_ready = false;
    a4_linear_qk_he1_cts.clear();
    a4_linear_v_he1_cts.clear();
    cout << "OURS_A3_A4_ATTENTION_OUTPUT status=PASS layer=" << layer_id
         << " ablation=" << ours_vit::to_string(heterogeneous_ablation)
         << " mpc_heads=" << mpc_head_count
         << " linear_heads=" << linear_head_count
         << " residual_merge=CKKS_before_rescale"
         << " a4_cross_context_merge="
         << (heterogeneous_ablation == ours_vit::Ablation::A4 &&
                     mpc_head_count > 0 && linear_head_count > 0
                 ? "MPC_additive_after_two_real_conversions"
                 : "single_context_one_real_conversion")
         << " dynamic_selection=0 surrogate=0" << endl;
    map<char*, FixArray> output;
    output["residual"] = std::move(result);
    return output;
}

FixArray Bert_ckks::linear2_from_mpc(const FixArray &input,
                                     PreprocessParams_2 &params,
                                     const FCMetadata &data,
                                     int layer_id) {
    const bool final_he_logits_diagnostic =
        layer_id == ATTENTION_LAYERS &&
        getenv("OURS_VIT_A0_FINAL_HE_LOGITS_PATH") != nullptr;
    if (input.size <= 0 || input.size != data.image_size * data.filter_h ||
        input.size % data.slot_count != 0 ||
        (data.image_size * data.filter_w) % data.slot_count != 0 ||
        data.filter_h != COMMON_DIM) {
        throw runtime_error("formal generic BSGS2 input shape is not slot aligned");
    }
    if (input.ell != NL_ELL || input.s != NL_SCALE) {
        throw runtime_error("formal generic BSGS2 input requires the paper nonlinear ell/s");
    }
    HE_CKKS *he = linear.he2;
    const LOCATION loc = he->backend();
    const int slot = he->num_slots;
    const size_t num_cts = static_cast<size_t>(input.size / slot);
    const size_t output_num_cts = static_cast<size_t>(
        data.image_size * data.filter_w / slot);
    FixArray feature_major = token_major_to_feature_major(
        input, data.image_size, data.filter_h, "generic_bsgs_input");
    vector<vector<HE_CKKS::scalar_t>> shares(num_cts);
    for (size_t i = 0; i < num_cts; ++i) {
        shares[i].assign(feature_major.data + i * slot,
                         feature_major.data + (i + 1) * slot);
    }
    vector<UnifiedPlaintext> pts;
    import_target_scale(*this, shares, slot, input.ell, input.s,
                            he->he_scale, he, pts, "generic_bsgs_input");

    // The generic ViT matrices use the same full-real input layout as the
    // existing ViT attention/FFN paths.  The BERT compatibility branch sends
    // half as many complex ciphertexts, but mpc_to_ckks produces independent
    // real plaintexts here; applying imag_repack would therefore discard the
    // second half of Bob's additive share.
    vector<UnifiedCiphertext> input_cts(
        num_cts, UnifiedCiphertext(LOCATION::HOST));
    vector<Ciphertext> output_pts_host;
    vector<Plaintext> output_pts(output_num_cts);
    if (party == BOB) {
        #pragma omp parallel for
        for (size_t i = 0; i < input_cts.size(); ++i) {
            he->encryptor->encrypt(pts[i], input_cts[i]);
        }
        send_encrypted_vector(io, input_cts);
        if (final_he_logits_diagnostic) {
            vector<Ciphertext> diagnostic_cts(output_num_cts);
            recv_encrypted_vector(he->context, io, diagnostic_cts);
            vector<double> packed_values(
                static_cast<size_t>(data.image_size) * data.filter_w);
            for (size_t ciphertext = 0; ciphertext < diagnostic_cts.size();
                 ++ciphertext) {
                Plaintext diagnostic_plaintext;
                he->decryptor->decrypt(diagnostic_cts[ciphertext],
                                       diagnostic_plaintext);
                vector<double> slots;
                he->encoder->decode(diagnostic_plaintext, slots);
                if (slots.size() != static_cast<size_t>(slot)) {
                    throw runtime_error(
                        "final HE logits diagnostic slot mismatch");
                }
                copy(slots.begin(), slots.end(),
                     packed_values.begin() + ciphertext * slot);
            }
            vector<double> token_major(packed_values.size());
            const int features_per_half = slot / (2 * data.image_size);
            for (int ciphertext = 0;
                 ciphertext < static_cast<int>(diagnostic_cts.size());
                 ++ciphertext) {
                for (int slot_index = 0; slot_index < slot; ++slot_index) {
                    const int token = slot_index % data.image_size;
                    const bool upper_half = slot_index >= slot / 2;
                    const int local_feature =
                        (slot_index % (slot / 2)) / data.image_size;
                    const int feature =
                        ciphertext * features_per_half + local_feature +
                        (upper_half ? data.filter_w / 2 : 0);
                    token_major[static_cast<size_t>(token) * data.filter_w +
                                feature] =
                        packed_values[static_cast<size_t>(ciphertext) * slot +
                                      slot_index];
                }
            }
            ofstream diagnostic_out(
                getenv("OURS_VIT_A0_FINAL_HE_LOGITS_PATH"));
            if (!diagnostic_out) {
                throw runtime_error(
                    "cannot write final HE logits diagnostic");
            }
            diagnostic_out << "OURS_VIT_FINAL_HE_LOGITS_V1\n"
                           << "elements " << min(1000, data.filter_w)
                           << "\nvalues\n" << fixed << setprecision(10);
            for (int feature = 0; feature < min(1000, data.filter_w);
                 ++feature) {
                diagnostic_out << token_major[feature] << '\n';
            }
            diagnostic_out << "end\n";
            cout << "OURS_A0_FINAL_HE_LOGITS status=CAPTURED"
                 << " final_output_only=1 intermediate_reconstruction=0"
                 << " elements=" << min(1000, data.filter_w) << endl;
        }
        output_pts_host.resize(output_num_cts);
        recv_encrypted_vector(he->context, io, output_pts_host);
        #pragma omp parallel for
        for (size_t i = 0; i < output_pts_host.size(); ++i) {
            he->decryptor->decrypt(output_pts_host[i], output_pts[i]);
        }
    } else {
        recv_encrypted_vector(he->context, io, input_cts);
        if (he->is_gpu_enable()) {
            for (auto &ct : input_cts) {
                ct.to_device(*he->context, *he->dcontext);
            }
            for (auto &pt : pts) {
                pt.to_device(*he->context, *he->dcontext);
            }
        }
        cout << "CKKS_TRANSFER_LAYOUT stage=generic_bsgs_input"
             << " layout=ordinary_full_slot cts=" << input_cts.size()
             << " imag_repack=0 source=ViT_original_path" << endl;
        for (size_t i = 0; i < input_cts.size(); ++i) {
            if (loc == LOCATION::HOST) {
                he->evaluator->add_plain_inplace(input_cts[i], pts[i]);
            } else {
                he->devaluator->add_plain_inplace(input_cts[i], pts[i]);
            }
        }
        vector<UnifiedCiphertext> output_cts(
            output_num_cts, UnifiedCiphertext(loc));
        linear.linear_2(he, input_cts, params, data, output_cts);
        vector<Ciphertext> output_host(output_num_cts);
        for (size_t i = 0; i < output_cts.size(); ++i) {
            if (he->is_gpu_enable()) {
                output_cts[i].to_host(*he->context, *he->dcontext);
            }
            output_host[i] = std::move(output_cts[i].hcipher());
        }
        prepare_host_conversion_from_actual_scale(
            *this, output_host, kFormalConversionScaleBits, he,
            "generic_bsgs_output");
        if (final_he_logits_diagnostic) {
            vector<Ciphertext> diagnostic_cts = output_host;
            send_encrypted_vector(io, diagnostic_cts);
        }
        mask_ckks_to_mpc_inplace(output_host, output_pts, he);
        send_encrypted_vector(io, output_host);
    }
    ct_transfer_comm += get_comm();
    ct_transfer_round += get_round();

    auto decoded = ckks_to_mpc(
        output_pts, he->poly_modulus_degree, he->get_Q(output_pts[0]),
        input.ell, input.s, he, false);
    const int output_size = data.image_size * data.filter_w;
    FixArray packed(party, output_size, true, input.ell, input.s);
    size_t offset = 0;
    for (const auto &part : decoded) {
        if (part.size() != static_cast<size_t>(slot) ||
            offset + part.size() > static_cast<size_t>(packed.size)) {
            throw runtime_error("formal generic BSGS2 output decode mismatch");
        }
        copy(part.begin(), part.end(), packed.data + offset);
        offset += part.size();
    }
    if (offset != static_cast<size_t>(packed.size)) {
        throw runtime_error("formal generic BSGS2 output is incomplete");
    }
    return bsgs_output_to_token_major(packed, data, "generic_bsgs_output_to_token");
}

FixArray Bert_ckks::layer_norm_affine_ckks(
    const map<char*,FixArray> &input) {
    auto get_share = [&input](const char *name) {
        for (const auto &entry : input) {
            if (entry.first != nullptr && string(entry.first) == name) {
                return entry.second;
            }
        }
        throw runtime_error(string("formal final LayerNorm share is missing: ") +
                            name);
    };
    FixArray x_flat_avg = get_share("x_flat_avg");
    FixArray sigma_flat = get_share("sigma_flat");
    FixArray w = get_share("w");
    FixArray b = get_share("b");
    if (x_flat_avg.size <= 0 || x_flat_avg.size % COMMON_DIM != 0 ||
        sigma_flat.size != x_flat_avg.size || w.size != x_flat_avg.size ||
        b.size != x_flat_avg.size) {
        throw runtime_error(
            "formal final LayerNorm affine requires real shape-matched shares");
    }
    require_layernorm_parameter_scale(x_flat_avg, sigma_flat, w, b);
    HE_CKKS *he = linear.he4;
    const LOCATION loc = he->backend();
    const int bitwidth = x_flat_avg.ell;
    const int scale = x_flat_avg.s;
    if (bitwidth != NL_ELL || scale != NL_SCALE) {
        throw runtime_error(
            "formal final LayerNorm affine requires the paper nonlinear ell/s shares");
    }
    const int slot = he->num_slots;
    const int degree = he->poly_modulus_degree;
    if (x_flat_avg.size % slot != 0) {
        throw runtime_error("formal final LayerNorm affine is not slot aligned");
    }
    const size_t num_cts = static_cast<size_t>(x_flat_avg.size / slot);
    if (num_cts == 0) {
        throw runtime_error(
            "formal final LayerNorm affine requires at least one ciphertext");
    }

    auto pack = [slot, num_cts](const FixArray &share) {
        vector<vector<HE_CKKS::scalar_t>> packed(num_cts);
        for (size_t i = 0; i < num_cts; ++i) {
            packed[i].assign(share.data + i * slot,
                             share.data + (i + 1) * slot);
        }
        return packed;
    };
    vector<UnifiedPlaintext> x_pts;
    vector<UnifiedPlaintext> sigma_pts;
    auto start = high_resolution_clock::now();
    import_target_scale(*this, pack(x_flat_avg), slot, bitwidth, scale,
                            he->he_scale, he, x_pts,
                            "final_ln_centered_input");
    import_target_scale(*this, pack(sigma_flat), slot, bitwidth, scale,
                            he->he_scale, he, sigma_pts,
                            "final_ln_inv_sigma_input");
    this->conversion_la +=
        chrono::duration<double>(high_resolution_clock::now() - start).count();
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();

    vector<UnifiedCiphertext> x_cts(num_cts,
                                   UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> sigma_cts(num_cts,
                                       UnifiedCiphertext(LOCATION::HOST));
    vector<Plaintext> output_pts(num_cts);
    auto transfer_start = high_resolution_clock::now();
    if (party == BOB) {
        #pragma omp parallel for
        for (size_t i = 0; i < num_cts; ++i) {
            he->encryptor->encrypt(x_pts[i], x_cts[i]);
            he->encryptor->encrypt(sigma_pts[i], sigma_cts[i]);
        }
        send_encrypted_vector(io, x_cts);
        send_encrypted_vector(io, sigma_cts);
        vector<Ciphertext> output_cts(num_cts);
        recv_encrypted_vector(he->context, io, output_cts);
        #pragma omp parallel for
        for (size_t i = 0; i < num_cts; ++i) {
            he->decryptor->decrypt(output_cts[i], output_pts[i]);
        }
    } else {
        recv_encrypted_vector(he->context, io, x_cts);
        recv_encrypted_vector(he->context, io, sigma_cts);
        if (he->is_gpu_enable()) {
            auto copy_start = high_resolution_clock::now();
            for_each_n(x_cts.begin(), x_cts.size(), [he](auto &ct) {
                ct.to_device(*he->context, *he->dcontext);
            });
            for_each_n(sigma_cts.begin(), sigma_cts.size(), [he](auto &ct) {
                ct.to_device(*he->context, *he->dcontext);
            });
            for_each_n(x_pts.begin(), x_pts.size(), [he](auto &pt) {
                pt.to_device(*he->context, *he->dcontext);
            });
            for_each_n(sigma_pts.begin(), sigma_pts.size(), [he](auto &pt) {
                pt.to_device(*he->context, *he->dcontext);
            });
            this->ct_transfer_mem_la +=
                ((high_resolution_clock::now() - copy_start)).count() / 1e9;
        }
        cout << "FORMAL_TRANSFER_LAYOUT stage=final_layernorm_input status=PASS"
             << " layout=ordinary_full_slot cts=" << num_cts
             << " imag_repack=0" << endl;

        vector<UnifiedPlaintext> tmp_w(num_cts, UnifiedPlaintext(loc));
        vector<UnifiedPlaintext> tmp_b(num_cts, UnifiedPlaintext(loc));
        vector<vector<double>> w_matrix(num_cts,
                                        vector<double>(slot, 0.0));
        vector<vector<double>> b_matrix(num_cts,
                                        vector<double>(slot, 0.0));
        const double ln_affine_weight_scale = ldexp(1.0, he->he_scale);
        const double ln_affine_bias_scale =
            ln_affine_weight_scale * ln_affine_weight_scale *
            ln_affine_weight_scale;
        const double ln_affine_input_scale =
            sqrt(ln_affine_bias_scale / ln_affine_weight_scale);
        for (size_t i = 0; i < num_cts; ++i) {
            for (int j = 0; j < slot; ++j) {
                // Algorithm-2's share-local fixed-point Encode preserves the
                // real slots but may carry a large imaginary sharing residue.
                // Both encrypted operands are projected with ct+conj(ct)
                // below, yielding 2*Re(x) and 2*Re(sigma). Absorb the public
                // factor four into gamma without another level/rescale.
                w_matrix[i][j] = 0.25 * layernorm_fixed_to_real(
                    w.data[i * slot + j], w, "LayerNorm weight");
                b_matrix[i][j] = layernorm_fixed_to_real(
                    b.data[i * slot + j], b, "LayerNorm bias");
            }
            if (!he->is_gpu_enable()) {
                he->encoder->encode(w_matrix[i], x_cts[i].hcipher().parms_id(),
                                    ln_affine_weight_scale, tmp_w[i]);
                he->encoder->encode(b_matrix[i], x_cts[i].hcipher().parms_id(),
                                    ln_affine_bias_scale, tmp_b[i]);
                he->evaluator->add_plain_inplace(x_cts[i], x_pts[i]);
                he->evaluator->add_plain_inplace(sigma_cts[i], sigma_pts[i]);
                UnifiedCiphertext x_conjugate(LOCATION::HOST);
                UnifiedCiphertext sigma_conjugate(LOCATION::HOST);
                he->evaluator->complex_conjugate(x_cts[i], *he->gal_keys,
                                                  x_conjugate);
                he->evaluator->complex_conjugate(sigma_cts[i], *he->gal_keys,
                                                  sigma_conjugate);
                he->evaluator->add_inplace(x_cts[i], x_conjugate);
                he->evaluator->add_inplace(sigma_cts[i], sigma_conjugate);
                if (abs(x_cts[i].scale() - ln_affine_input_scale) /
                            ln_affine_input_scale >
                        1e-12 ||
                    abs(sigma_cts[i].scale() - ln_affine_input_scale) /
                            ln_affine_input_scale >
                        1e-12) {
                    throw runtime_error(
                        "formal final LN Algorithm-2 target scale mismatch");
                }
                // Keep the ct-ct product scale-aligned: x and sigma are both
                // at the input scale, so multiply them before applying the
                // public weight plaintext.  The previous order produced
                // scale^2 for x and then attempted a ct-ct multiply against
                // sigma at scale, which Phantom correctly rejects.
                he->evaluator->multiply_inplace(x_cts[i], sigma_cts[i]);
                he->evaluator->relinearize_inplace(x_cts[i],
                                                   *he->relin_keys);
                he->evaluator->multiply_plain_inplace(x_cts[i], tmp_w[i]);
                he->evaluator->add_plain_inplace(x_cts[i], tmp_b[i]);
                ++this->he_rescale_count;
                he->evaluator->rescale_to_next_inplace(x_cts[i]);
            } else {
                const auto chain_index = x_cts[i].dcipher().chain_index();
                he->dencoder->encode(*he->dcontext, w_matrix[i],
                                     ln_affine_weight_scale, tmp_w[i],
                                     chain_index);
                he->dencoder->encode(*he->dcontext, b_matrix[i],
                                     ln_affine_bias_scale, tmp_b[i],
                                     chain_index);
                he->devaluator->add_plain_inplace(x_cts[i], x_pts[i]);
                he->devaluator->add_plain_inplace(sigma_cts[i], sigma_pts[i]);
                UnifiedCiphertext x_conjugate(LOCATION::DEVICE);
                UnifiedCiphertext sigma_conjugate(LOCATION::DEVICE);
                he->devaluator->complex_conjugate(x_cts[i], *he->dgal_keys,
                                                   x_conjugate);
                he->devaluator->complex_conjugate(sigma_cts[i],
                                                   *he->dgal_keys,
                                                   sigma_conjugate);
                he->devaluator->add_inplace(x_cts[i], x_conjugate);
                he->devaluator->add_inplace(sigma_cts[i], sigma_conjugate);
                if (abs(x_cts[i].scale() - ln_affine_input_scale) /
                            ln_affine_input_scale >
                        1e-12 ||
                    abs(sigma_cts[i].scale() - ln_affine_input_scale) /
                            ln_affine_input_scale >
                        1e-12) {
                    throw runtime_error(
                        "formal final LN Algorithm-2 target scale mismatch");
                }
                he->devaluator->multiply_inplace(x_cts[i], sigma_cts[i]);
                he->devaluator->relinearize_inplace(x_cts[i],
                                                    *he->drelin_keys);
                he->devaluator->multiply_plain_inplace(x_cts[i], tmp_w[i]);
                he->devaluator->add_plain_inplace(x_cts[i], tmp_b[i]);
                ++this->he_rescale_count;
                he->devaluator->rescale_to_next_inplace(x_cts[i]);
            }
        }
        cout << "FORMAL_LAYERNORM_SCALE_PLAN stage=final_affine status=PASS"
             << " input_scale_bits=" << log2(ln_affine_input_scale)
             << " weight_scale_bits=" << log2(ln_affine_weight_scale)
             << " bias_scale_bits=" << log2(ln_affine_bias_scale)
             << " real_projection=both gamma_compensation=0.25"
             << endl;
        linear.ct_pt_multiply_count += num_cts;
        linear.ct_ct_multiply_count += num_cts;
        vector<Ciphertext> output_cts(num_cts);
        if (he->is_gpu_enable()) {
            auto copy_start = high_resolution_clock::now();
            for (size_t i = 0; i < num_cts; ++i) {
                UnifiedCiphertext::to_host(*he->dcontext,
                                           x_cts[i].dcipher(), *he->context,
                                           output_cts[i]);
            }
            this->ct_transfer_mem_la +=
                ((high_resolution_clock::now() - copy_start)).count() / 1e9;
        } else {
            for (size_t i = 0; i < num_cts; ++i) {
                output_cts[i] = x_cts[i].hcipher();
            }
        }
        prepare_host_conversion_from_actual_scale(
            *this, output_cts, kFormalConversionScaleBits, he,
            "final_layernorm_affine");
        mask_ckks_to_mpc_inplace(output_cts, output_pts, he);
        send_encrypted_vector(io, output_cts);
        cout << "FORMAL_TRANSFER_LAYOUT stage=final_layernorm_output status=PASS"
             << " layout=ordinary_full_slot cts=" << num_cts
             << " half_decode=0" << endl;
    }
    this->ct_transfer_la +=
        chrono::duration<double>(high_resolution_clock::now() - transfer_start).count();
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();

    start = high_resolution_clock::now();
    auto output_vector = ckks_to_mpc(
        output_pts, degree, he->get_Q(output_pts[0]), bitwidth,
        scale, he, false);
    this->conversion_la +=
        chrono::duration<double>(high_resolution_clock::now() - start).count();
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    FixArray output(party, x_flat_avg.size, x_flat_avg.signed_, bitwidth,
                    scale);
    size_t offset = 0;
    for (const auto &packed : output_vector) {
        if (packed.size() != static_cast<size_t>(slot) ||
            offset + packed.size() > static_cast<size_t>(output.size)) {
            throw runtime_error("formal final LayerNorm affine decode mismatch");
        }
        copy(packed.begin(), packed.end(), output.data + offset);
        offset += packed.size();
    }
    if (offset != static_cast<size_t>(output.size)) {
        throw runtime_error("formal final LayerNorm affine output is incomplete");
    }
    return output;
}

/*
    scale已经搞定了，差一个matmul_cc到matmul_cp的packing
*/
std::map<char*,FixArray> Bert_ckks::Softmax_v_linear_layer_norm_part1(FixArray input, int layer_id){
    FCMetadata data = linear.data_lin1_0;
    int bitwidth,s;
    bitwidth = input.ell;
    s = input.s;
    if (bitwidth != NL_ELL || s != NL_SCALE) {
        throw runtime_error("formal FFN2 requires the paper nonlinear ell/s GeLU shares");
    }
    if (bitwidth != NL_ELL || s != NL_SCALE) {
        throw runtime_error(
            "formal Softmax*V requires a paper ell/s Softmax share");
    }
    int softmax_dim = data.image_size;
    const bool use_bert_half_packing = !pre_norm_attention_output_only;
    int softmax_output_size = PACKING_NUM*softmax_dim*softmax_dim;
    uint64_t* softmax_output_pack = new uint64_t[softmax_output_size];
    linear.preprocess_softmax(
        input.data,
        softmax_output_pack,
        data
    );
    cout << "OK2" << endl;
    HE_CKKS* he = linear.he2;
    int poly_modulus_degree = he->poly_modulus_degree;
    int slot = poly_modulus_degree / 2;
    int num_cts_softmax = softmax_output_size / slot;
    int num_cts_ln = data.image_size * COMMON_DIM / slot;
    const bool use_he_v = attention_v_he_ready;
    if (use_he_v) {
        if (party == sci::ALICE &&
            attention_v_he_cts.size() != static_cast<size_t>(num_cts_ln)) {
            throw runtime_error(
                "formal Softmax*V HE V ciphertext count is unavailable");
        }
    } else {
        if (!attention_v_ready ||
            attention_v_share.size != data.image_size * COMMON_DIM) {
            throw runtime_error(
                "formal Softmax*V refused: real V share is unavailable");
        }
        if (attention_v_share.ell != NL_ELL ||
            attention_v_share.s != ATTENTION_V_SCALE) {
            throw runtime_error(
                "formal Softmax*V requires the real V share at paper ell/s11");
        }
    }
    if (!attention_input_residual_ready ||
        attention_input_residual_share.size != data.image_size * COMMON_DIM) {
        throw runtime_error("formal attention output refused: real block input residual is unavailable");
    }
    if (attention_input_residual_share.ell != NL_ELL ||
        attention_input_residual_share.s != NL_SCALE) {
        throw runtime_error(
            "formal attention residual must remain at the paper nonlinear ell/s");
    }
    // linear_2 emits BSGS split-half slots.  The checkpoint input retained
    // for the residual is token-major, so place each local share in exactly
    // the output layout before encrypting and adding it to h3.
    FixArray attention_residual_packed = token_major_to_bsgs_output(
        attention_input_residual_share, linear.data_lin2,
        "attention_residual_to_h3");
    vector<vector<HE_CKKS::scalar_t>> softmax_output_pack_vector(num_cts_softmax);
    vector<vector<HE_CKKS::scalar_t>> v_share_vector(
        use_he_v ? 0 : num_cts_ln);
    vector<vector<HE_CKKS::scalar_t>> attention_residual_vector(num_cts_ln);
    vector<UnifiedPlaintext> softmax_output_pack_pts;
    for(int i=0;i<num_cts_softmax;i++){
        for(int j=0;j<slot;j++){
            softmax_output_pack_vector[i].push_back(softmax_output_pack[i*slot+j]);
        }
    }
    if (!use_he_v) {
        for (int i = 0; i < num_cts_ln; ++i) {
            v_share_vector[i].assign(attention_v_share.data + i * slot,
                                     attention_v_share.data + (i + 1) * slot);
        }
    }
    for (int i = 0; i < num_cts_ln; ++i) {
        attention_residual_vector[i].assign(
            attention_residual_packed.data + i * slot,
            attention_residual_packed.data + (i + 1) * slot);
    }
    cout << "FORMAL_LAYOUT_ADAPTER stage=attention_residual_to_h3"
         << " status=PASS source=token_feature_row_major"
         << " target=bsgs2_split_feature_halves elements="
         << attention_residual_packed.size
         << " communication=0 public_share_permutation=1" << endl;
    auto start = high_resolution_clock::now();
    import_target_scale(*this, softmax_output_pack_vector, slot,
                            bitwidth, s, he->he_scale, he,
                            softmax_output_pack_pts, "softmax");
    vector<UnifiedPlaintext> v_share_pts;
    vector<UnifiedPlaintext> attention_residual_pts;
    if (!use_he_v) {
        import_target_scale(*this, v_share_vector, slot,
                                attention_v_share.ell, attention_v_share.s,
                                he->he_scale, he, v_share_pts, "attention_v");
    }
    import_target_scale(*this, attention_residual_vector, slot,
                            attention_input_residual_share.ell,
                            attention_input_residual_share.s, he->he_scale,
                            he, attention_residual_pts,
                            "attention_residual");
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    vector<Plaintext> x_avg_pts(num_cts_ln);
    vector<Plaintext> x_avg_pts_half(num_cts_ln % 2 == 0 ? num_cts_ln / 2 : 0);
    vector<Plaintext> sigma_pts(1);
    vector<Ciphertext> enc_x_avg(num_cts_ln);
    vector<Ciphertext> enc_x_avg_half(num_cts_ln % 2 == 0 ? num_cts_ln / 2 : 0);
    vector<Ciphertext> enc_sigma(1);
    
    vector<UnifiedCiphertext> enc_softmax(
        num_cts_softmax, UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> enc_softmax_half(
        num_cts_softmax % 2 == 0 ? num_cts_softmax / 2 : 0,
        UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> enc_v(
        use_he_v ? 0 : num_cts_ln, UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> enc_attention_residual(
        num_cts_ln, UnifiedCiphertext(LOCATION::HOST));
    if (use_he_v && party == sci::ALICE) {
        enc_v = std::move(attention_v_he_cts);
    }
    if (party==BOB){
        #pragma omp parallel for
        for(int i=0;i<(use_bert_half_packing ? enc_softmax_half.size()
                                             : enc_softmax.size());i++){
            he->encryptor->encrypt(
                softmax_output_pack_pts[i],
                use_bert_half_packing ? enc_softmax_half[i] : enc_softmax[i]);
        }
        if (!use_he_v) {
            #pragma omp parallel for
            for(int i=0;i<num_cts_ln;i++){
                he->encryptor->encrypt(v_share_pts[i], enc_v[i]);
            }
        }
        #pragma omp parallel for
        for(int i=0;i<num_cts_ln;i++){
            he->encryptor->encrypt(attention_residual_pts[i],
                                   enc_attention_residual[i]);
        }
        if (use_bert_half_packing && enc_softmax_half.empty()) {
            throw runtime_error("formal Softmax half-packing requires an even ciphertext count");
        }
        if (use_bert_half_packing) {
            send_encrypted_vector(io, enc_softmax_half);
        } else {
            send_encrypted_vector(io, enc_softmax);
        }
        if (!use_he_v) {
            send_encrypted_vector(io, enc_v);
        }
        send_encrypted_vector(io, enc_attention_residual);
        cout << "FORMAL_TRANSFER_LAYOUT stage=softmax_v_input status=PASS"
             << " layout=ordinary_full_slot softmax_cts=" << num_cts_softmax
             << " v_cts=" << enc_v.size() << " residual_cts=" << num_cts_ln
             << " v_transport=" << (use_he_v ? "HE_at_Alice" : "MPC_to_CKKS")
             << " imag_repack=0" << endl;
        cout << "send_encrypted_vector done" << endl;
        if (profile_enabled_)
            profile_switch(BlbViTProfileStage::OUTPUT_PROJECTION);
        if (use_bert_half_packing && !enc_x_avg_half.empty()) {
            recv_encrypted_vector(he->context, io, enc_x_avg_half);
        } else {
            recv_encrypted_vector(he->context, io, enc_x_avg);
        }
        if (!pre_norm_attention_output_only) {
            recv_encrypted_vector(he->context, io, enc_sigma);
        }
        if (use_bert_half_packing && !enc_x_avg_half.empty()) {
            #pragma omp parallel for
            for(int i=0;i<enc_x_avg_half.size();i++){
                he->decryptor->decrypt(enc_x_avg_half[i], x_avg_pts_half[i]);
            }
        } else {
            #pragma omp parallel for
            for(int i=0;i<enc_x_avg.size();i++){
                he->decryptor->decrypt(enc_x_avg[i], x_avg_pts[i]);
            }
        }
        cout << "recv_encrypted_vector done" << endl;
        if (!pre_norm_attention_output_only) {
            he->decryptor->decrypt(enc_sigma[0], sigma_pts[0]);
        }
    }
    else{
        LOCATION loc = he->backend();

        auto require_uniform_metadata = [&](const vector<UnifiedCiphertext> &cts,
                                            const char *name) {
            if (cts.empty()) {
                throw runtime_error(string("formal ciphertext vector is empty: ") +
                                    name);
            }
            const LOCATION expected_location = cts.front().location();
            const size_t expected_limbs = cts.front().coeff_modulus_size();
            const double expected_scale = cts.front().scale();
            for (const auto &ct : cts) {
                if (ct.location() != expected_location ||
                    ct.coeff_modulus_size() != expected_limbs) {
                    throw runtime_error(string("formal ciphertext level mismatch: ") +
                                        name);
                }
                const double relative_error =
                    abs(ct.scale() - expected_scale) / expected_scale;
                if (!isfinite(relative_error) || relative_error > 1e-12) {
                    throw runtime_error(string("formal ciphertext scale mismatch: ") +
                                        name);
                }
                if (expected_location == LOCATION::HOST) {
                    if (ct.hcipher().parms_id() !=
                        cts.front().hcipher().parms_id()) {
                        throw runtime_error(
                            string("formal ciphertext parms mismatch: ") + name);
                    }
                } else if (expected_location == LOCATION::DEVICE) {
                    if (ct.dcipher().chain_index() !=
                        cts.front().dcipher().chain_index()) {
                        throw runtime_error(
                            string("formal ciphertext chain mismatch: ") + name);
                    }
                } else {
                    throw runtime_error(
                        string("formal ciphertext location is invalid: ") + name);
                }
            }
        };
        auto lift_with_actual_public_one = [&] (
            vector<UnifiedCiphertext> &cts, double target_scale,
            const char *name) {
            require_uniform_metadata(cts, name);
            const double source_scale = cts.front().scale();
            const size_t source_limbs = cts.front().coeff_modulus_size();
            if (!isfinite(source_scale) || source_scale <= 0.0 ||
                !isfinite(target_scale) || target_scale < source_scale) {
                throw runtime_error(string("formal public-one lift is invalid: ") +
                                    name);
            }
            const double unit_scale = target_scale / source_scale;
            if (!isfinite(unit_scale) || unit_scale < 1.0) {
                throw runtime_error(
                    string("formal public-one plaintext is not encodable: ") +
                    name);
            }
            const vector<double> public_one(slot, 1.0);
            UnifiedPlaintext unit_plain(loc);
            if (loc == LOCATION::HOST) {
                he->encoder->encode(public_one,
                                    cts.front().hcipher().parms_id(),
                                    unit_scale, unit_plain);
                for (auto &ct : cts) {
                    he->evaluator->multiply_plain_inplace(ct, unit_plain);
                }
            } else if (loc == LOCATION::DEVICE) {
                const auto chain_index = cts.front().dcipher().chain_index();
                he->dencoder->encode(*he->dcontext, public_one, unit_scale,
                                     unit_plain, chain_index);
                for (auto &ct : cts) {
                    he->devaluator->multiply_plain_inplace(ct, unit_plain);
                }
            } else {
                throw runtime_error("formal public-one lift backend is invalid");
            }
            linear.ct_pt_multiply_count += cts.size();
            require_uniform_metadata(cts, name);
            const double relative_error =
                abs(cts.front().scale() - target_scale) / target_scale;
            if (!isfinite(relative_error) || relative_error > 1e-12 ||
                cts.front().coeff_modulus_size() != source_limbs) {
                throw runtime_error(string("formal public-one lift failed: ") +
                                    name);
            }
            cout << "FORMAL_ACTUAL_ONE_LIFT stage=" << name
                 << " status=PASS source_scale_bits=" << log2(source_scale)
                 << " unit_scale_bits=" << log2(unit_scale)
                 << " target_scale_bits=" << log2(target_scale)
                 << " limbs=" << source_limbs << " cts=" << cts.size()
                 << " plaintext_value=1" << endl;
        };

        start = high_resolution_clock::now();
        if (use_bert_half_packing && enc_softmax_half.empty()) {
            throw runtime_error("formal Softmax half-packing requires an even ciphertext count");
        }
        if (use_bert_half_packing) {
            recv_encrypted_vector(he->context, io, enc_softmax_half);
            #pragma omp parallel for
            for (size_t i = 0; i < enc_softmax_half.size(); ++i) {
                he->imag_repack(enc_softmax_half[i], enc_softmax[i],
                                enc_softmax[i + enc_softmax_half.size()]);
            }
        } else {
            recv_encrypted_vector(he->context, io, enc_softmax);
        }
        if (!use_he_v) {
            recv_encrypted_vector(he->context, io, enc_v);
        }
        recv_encrypted_vector(he->context, io, enc_attention_residual);
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;

        // In the HE-V path, perform the existing share additions and the
        // required level alignment while the received ciphertexts are still
        // host SEAL objects.  Uploading an already-aligned ciphertext avoids
        // the Phantom GPU sequence that previously faulted after a device
        // plaintext add; no arithmetic or packing operation is changed.
        bool he_v_host_prepared = false;
        if (use_he_v && he->is_gpu_enable()) {
            for (size_t i = 0; i < enc_softmax.size(); ++i) {
                he->evaluator->add_plain_inplace(
                    enc_softmax[i], softmax_output_pack_pts[i]);
            }
            for (size_t i = 0; i < enc_attention_residual.size(); ++i) {
                he->evaluator->add_plain_inplace(
                    enc_attention_residual[i], attention_residual_pts[i]);
            }
            require_uniform_metadata(enc_v, "attention_v_he_host");
            const size_t target_limbs = enc_v.front().coeff_modulus_size();
            for (auto &ciphertext : enc_softmax) {
                while (ciphertext.coeff_modulus_size() > target_limbs) {
                    he->evaluator->mod_switch_to_next_inplace(ciphertext);
                }
                if (ciphertext.coeff_modulus_size() != target_limbs) {
                    throw runtime_error(
                        "formal Softmax HE level is below the native V level");
                }
            }
            // Phantom requires equal CKKS operand scales for the ciphertext
            // products in the unchanged Softmax*V kernel.  The original BLB
            // path obtains this naturally from its HE boundary; the formal
            // MPC import has scale he_scale.  Lift only the public scale of
            // the imported Softmax ciphertext to native V's scale, preserving
            // its plaintext value and the original packing/evaluator path.
            if (enc_softmax.front().scale() < enc_v.front().scale()) {
                lift_unified_with_actual_one(
                    linear, he, enc_softmax, enc_v.front().scale(),
                    "softmax_v_operand_scale");
            }
            if (abs(enc_softmax.front().scale() - enc_v.front().scale()) /
                    enc_v.front().scale() > 1e-12) {
                throw runtime_error(
                    "formal Softmax and native V CKKS scales are incompatible");
            }
            cout << "FORMAL_HE_LEVEL_ALIGNMENT stage=softmax_v"
                 << " status=PASS v_limbs=" << target_limbs
                 << " softmax_limbs=" << enc_softmax.front().coeff_modulus_size()
                 << " v_scale_bits=" << log2(enc_v.front().scale())
                 << " softmax_scale_bits=" << log2(enc_softmax.front().scale())
                 << " v_transport=HE_at_Alice host_prepared=1" << endl;
            he_v_host_prepared = true;
        }
        
        NVTX_PUSH("SM-LN1", true);
        if (he->is_gpu_enable()) {
            // Load reviced ciphertexts to GPU
            start = high_resolution_clock::now();
            for_each_n(enc_softmax.begin(), enc_softmax.size(),
                    [he](auto &ct) { ct.to_device(*he->context, *he->dcontext); });
            for_each_n(enc_v.begin(), enc_v.size(),
                    [he](auto &ct) { ct.to_device(*he->context, *he->dcontext); });
            for_each_n(enc_attention_residual.begin(),
                    enc_attention_residual.size(),
                    [he](auto &ct) { ct.to_device(*he->context, *he->dcontext); });
            for_each_n(softmax_output_pack_pts.begin(), softmax_output_pack_pts.size(),
                    [he](auto &pt) { pt.to_device(*he->context, *he->dcontext); });
            for_each_n(v_share_pts.begin(), v_share_pts.size(),
                    [he](auto &pt) { pt.to_device(*he->context, *he->dcontext); });
            for_each_n(attention_residual_pts.begin(),
                    attention_residual_pts.size(),
                    [he](auto &pt) { pt.to_device(*he->context, *he->dcontext); });
            this->ct_transfer_mem_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        }

        // The original BLB HE boundary keeps V at the level emitted by the
        // QKV kernel.  The MPC-imported Softmax ciphertext starts at the HE
        // key level, so align that ciphertext down to V's existing level
        // before the unchanged Softmax*V evaluator is called.  GPU runs do
        // this on host SEAL above; the CPU path keeps the same operation here.
        if (use_he_v && !he_v_host_prepared) {
            require_uniform_metadata(enc_v, "attention_v_he");
            const size_t target_limbs = enc_v.front().coeff_modulus_size();
            for (auto &ciphertext : enc_softmax) {
                while (ciphertext.coeff_modulus_size() > target_limbs) {
                    if (loc == LOCATION::HOST) {
                        he->evaluator->mod_switch_to_next_inplace(ciphertext);
                    } else {
                        he->devaluator->mod_switch_to_next_inplace(ciphertext);
                    }
                }
                if (ciphertext.coeff_modulus_size() != target_limbs) {
                    throw runtime_error(
                        "formal Softmax HE level is below the native V level");
                }
            }
            if (enc_softmax.front().scale() < enc_v.front().scale()) {
                lift_unified_with_actual_one(
                    linear, he, enc_softmax, enc_v.front().scale(),
                    "softmax_v_operand_scale");
            }
            if (abs(enc_softmax.front().scale() - enc_v.front().scale()) /
                    enc_v.front().scale() > 1e-12) {
                throw runtime_error(
                    "formal Softmax and native V CKKS scales are incompatible");
            }
            cout << "FORMAL_HE_LEVEL_ALIGNMENT stage=softmax_v"
                 << " status=PASS v_limbs=" << target_limbs
                 << " softmax_limbs=" << enc_softmax.front().coeff_modulus_size()
                 << " v_scale_bits=" << log2(enc_v.front().scale())
                 << " softmax_scale_bits=" << log2(enc_softmax.front().scale())
                 << " v_transport=HE_at_Alice" << endl;
        }
        
        start = high_resolution_clock::now();
        cout << "FORMAL_TRANSFER_LAYOUT stage=softmax_v_input status=PASS"
             << " layout=ordinary_full_slot softmax_cts=" << enc_softmax.size()
             << " v_cts=" << enc_v.size()
             << " residual_cts=" << enc_attention_residual.size()
             << " imag_repack=0" << endl;
        cout << "num of input cts:" << enc_softmax.size() << endl;
        cout << "recv_encrypted_vector" << endl;

        if (!he->is_gpu_enable()) {
            #pragma omp parallel for num_threads(4)
            for(int i=0;i<enc_softmax.size();i++){
                he->evaluator->add_plain_inplace(enc_softmax[i], softmax_output_pack_pts[i]);
            }
            for (int i = 0; i < enc_v.size(); ++i) {
                he->evaluator->add_plain_inplace(enc_v[i], v_share_pts[i]);
            }
            for (int i = 0; i < enc_attention_residual.size(); ++i) {
                he->evaluator->add_plain_inplace(
                    enc_attention_residual[i], attention_residual_pts[i]);
            }
        } else if (!he_v_host_prepared) {
            for(int i=0;i<enc_softmax.size();i++){
                he->devaluator->add_plain_inplace(enc_softmax[i], softmax_output_pack_pts[i]);
            }
            for (int i = 0; i < enc_v.size(); ++i) {
                he->devaluator->add_plain_inplace(enc_v[i], v_share_pts[i]);
            }
            for (int i = 0; i < enc_attention_residual.size(); ++i) {
                he->devaluator->add_plain_inplace(
                    enc_attention_residual[i], attention_residual_pts[i]);
            }
        }
        require_uniform_metadata(enc_softmax, "softmax");
        if (!use_he_v) {
            require_uniform_metadata(enc_v, "attention_v");
        }
        require_uniform_metadata(enc_attention_residual,
                                 "attention_residual");
        const double imported_scale = ldexp(1.0, he->he_scale);
        if (use_he_v) {
            const double residual_relative_error =
                abs(enc_attention_residual.front().scale() - imported_scale) /
                imported_scale;
            const double softmax_v_relative_error =
                abs(enc_softmax.front().scale() - enc_v.front().scale()) /
                enc_v.front().scale();
            if (!isfinite(residual_relative_error) ||
                residual_relative_error > 1e-12 ||
                !isfinite(softmax_v_relative_error) ||
                softmax_v_relative_error > 1e-12) {
                throw runtime_error(
                    "formal HE-V boundary scale alignment was not preserved");
            }
        } else {
            vector<const vector<UnifiedCiphertext> *> imported_ciphertext_vectors = {
                &enc_softmax, &enc_attention_residual, &enc_v};
            for (const auto *entry : imported_ciphertext_vectors) {
                const double relative_error =
                    abs(entry->front().scale() - imported_scale) /
                    imported_scale;
                if (!isfinite(relative_error) || relative_error > 1e-12) {
                    throw runtime_error(
                        "formal high-scale MPC-to-CKKS import was not preserved");
                }
            }
        }
        cout << "OK3" << endl;

        // Algorithm-2's share-local MPC-to-CKKS Encode preserves the real
        // slots, but an ordinary full-slot import can contain a large,
        // sharing-dependent imaginary component.  Softmax*V is a true
        // ciphertext-ciphertext product, so those two imaginary components
        // would multiply into (and corrupt) the real result.  Project both
        // imported operands with ct+conj(ct).  Each projection contributes a
        // public factor two; the existing post-AV mask multiply absorbs the
        // combined factor 1/4 without adding a multiply or modulus level.
        if (!use_bert_half_packing) {
            const double softmax_scale_before = enc_softmax.front().scale();
            const double v_scale_before = enc_v.front().scale();
            if (loc == LOCATION::HOST) {
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < static_cast<int>(enc_softmax.size()); ++i) {
                    UnifiedCiphertext conjugate(LOCATION::HOST);
                    he->evaluator->complex_conjugate(
                        enc_softmax[i], *he->gal_keys, conjugate);
                    he->evaluator->add_inplace(enc_softmax[i], conjugate);
                }
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < static_cast<int>(enc_v.size()); ++i) {
                    UnifiedCiphertext conjugate(LOCATION::HOST);
                    he->evaluator->complex_conjugate(
                        enc_v[i], *he->gal_keys, conjugate);
                    he->evaluator->add_inplace(enc_v[i], conjugate);
                }
            } else {
                for (size_t i = 0; i < enc_softmax.size(); ++i) {
                    UnifiedCiphertext conjugate(LOCATION::DEVICE);
                    he->devaluator->complex_conjugate(
                        enc_softmax[i], *he->dgal_keys, conjugate);
                    he->devaluator->add_inplace(enc_softmax[i], conjugate);
                }
                for (size_t i = 0; i < enc_v.size(); ++i) {
                    UnifiedCiphertext conjugate(LOCATION::DEVICE);
                    he->devaluator->complex_conjugate(
                        enc_v[i], *he->dgal_keys, conjugate);
                    he->devaluator->add_inplace(enc_v[i], conjugate);
                }
            }
            require_uniform_metadata(enc_softmax, "softmax_real_projected");
            require_uniform_metadata(enc_v, "attention_v_real_projected");
            if (enc_softmax.front().scale() != softmax_scale_before ||
                enc_v.front().scale() != v_scale_before) {
                throw runtime_error(
                    "formal Softmax*V real projection changed CKKS scale");
            }
            linear.rotation_count += enc_softmax.size() + enc_v.size();
            cout << "FORMAL_REAL_PROJECTION stage=softmax_v status=PASS"
                 << " mode=both_conjugate factor_compensation=av_mask_0.25"
                 << " softmax_cts=" << enc_softmax.size()
                 << " v_cts=" << enc_v.size()
                 << " extra_multiply=0 extra_rescale=0" << endl;
        }
        cout << "OK4" << endl;
        vector<vector<vector<UnifiedCiphertext>>> pack_softmax_ct;
        linear.preprocess_softmax_s1_ct_ct(he, enc_softmax, data, pack_softmax_ct);
        if (pack_softmax_ct.empty() || pack_softmax_ct.front().empty() ||
            pack_softmax_ct.front().front().empty()) {
            throw runtime_error("formal Softmax*V diagonal packing is empty");
        }
        const auto &av_level_reference =
            pack_softmax_ct.front().front().front();
        const size_t av_target_limbs =
            av_level_reference.coeff_modulus_size();
        for (auto &ciphertext : enc_v) {
            while (ciphertext.coeff_modulus_size() > av_target_limbs) {
                if (loc == LOCATION::HOST) {
                    he->evaluator->mod_switch_to_next_inplace(ciphertext);
                } else {
                    he->devaluator->mod_switch_to_next_inplace(ciphertext);
                }
            }
            if (ciphertext.coeff_modulus_size() != av_target_limbs) {
                throw runtime_error(
                    "formal V level is below the packed Softmax level");
            }
        }
        require_uniform_metadata(enc_v, "attention_v_av_aligned");
        bool av_same_level = false;
        if (loc == LOCATION::HOST) {
            av_same_level = enc_v.front().hcipher().parms_id() ==
                            av_level_reference.hcipher().parms_id();
        } else {
            av_same_level = enc_v.front().dcipher().chain_index() ==
                            av_level_reference.dcipher().chain_index();
        }
        if (!av_same_level) {
            throw runtime_error(
                "formal V and packed Softmax levels do not match");
        }
        cout << "FORMAL_AV_LEVEL_ALIGNMENT status=PASS"
             << " limbs=" << av_target_limbs
             << " softmax_scale_bits=" << log2(av_level_reference.scale())
             << " v_scale_bits=" << log2(enc_v.front().scale())
             << " v_cts=" << enc_v.size() << endl;
        cout << "OK5" << endl;
        vector<UnifiedCiphertext> softmax_V_result(
            PACKING_NUM * data.image_size * data.filter_w / data.slot_count,
            UnifiedCiphertext(loc));
        if (profile_enabled_)
            profile_switch(BlbViTProfileStage::SOFTMAX_V);
        const auto softmax_v_profile_start = high_resolution_clock::now();
        linear.softmax_v(he, pack_softmax_ct, enc_v, data, softmax_V_result,
                         use_bert_half_packing ? 1.0 : 0.25);
        if (profile_enabled_)
            profile_record_he(
                BlbViTProfileStage::SOFTMAX_V,
                chrono::duration<double, milli>(high_resolution_clock::now() -
                                                softmax_v_profile_start)
                    .count());
        // softmax_V_result's pack may be changed
        data = linear.data_lin2;
        cout << "OK6" << endl;
        vector<UnifiedCiphertext> h3;
        if (profile_enabled_)
            profile_switch(
                BlbViTProfileStage::OUTPUT_PROJECTION);
        const auto output_projection_profile_start =
            high_resolution_clock::now();
        linear.linear_2(
            he,
            softmax_V_result, 
            linear.pp_2[layer_id],
            data,
            h3
        );
        if (profile_enabled_)
            profile_record_he(
                BlbViTProfileStage::OUTPUT_PROJECTION,
                chrono::duration<double, milli>(
                    high_resolution_clock::now() - output_projection_profile_start)
                    .count());
        // encode weight once
        int L = data.image_size;
        vector<double> inv_D(slot, 1.0/768);
        vector<double> all_1(slot, 1);
        vector<double> mask(slot,0);
        UnifiedPlaintext inv_D_p(loc);
        UnifiedPlaintext inv_D_p2(loc);
        UnifiedPlaintext mask_p(loc);
        UnifiedPlaintext scale_up(loc);
        for(int i=0;i<L;i++){
            mask[i] = 1;
        }
        if (!he->is_gpu_enable()) {
            auto util_context_data = he->context->get_context_data(h3[0].hcipher().parms_id());
            util_context_data = util_context_data->next_context_data();
            he->encoder->encode(inv_D, util_context_data->parms_id(), pow(2,he->he_scale), inv_D_p);
            he->encoder->encode(all_1, util_context_data->parms_id(), pow(2,he->he_scale), scale_up);
        } else {
            auto util_chain_index = h3[0].dcipher().chain_index() + 1;
            he->dencoder->encode(*he->dcontext, inv_D, pow(2,he->he_scale), inv_D_p, util_chain_index);
            he->dencoder->encode(*he->dcontext, all_1, pow(2,he->he_scale), scale_up, util_chain_index);
        }

        // begin computation
        cout << "OK7" << endl;
        // Add the actual encrypted block-input residual before LayerNorm.
        if (h3.size() != enc_attention_residual.size()) {
            throw runtime_error(
                "formal attention residual ciphertext count mismatch");
        }
        require_uniform_metadata(h3, "attention_output_projection");
        require_uniform_metadata(enc_attention_residual,
                                 "attention_residual_before_alignment");
        const double residual_target_scale = h3.front().scale();
        const size_t residual_target_limbs =
            h3.front().coeff_modulus_size();
        lift_with_actual_public_one(enc_attention_residual,
                                    residual_target_scale,
                                    "attention_residual");
        for (auto &residual : enc_attention_residual) {
            while (residual.coeff_modulus_size() > residual_target_limbs) {
                if (loc == LOCATION::HOST) {
                    he->evaluator->mod_switch_to_next_inplace(residual);
                } else {
                    he->devaluator->mod_switch_to_next_inplace(residual);
                }
            }
            if (residual.coeff_modulus_size() != residual_target_limbs) {
                throw runtime_error(
                    "formal attention residual crossed below the h3 level");
            }
        }
        require_uniform_metadata(enc_attention_residual,
                                 "attention_residual_after_alignment");
        const double residual_scale_error =
            abs(enc_attention_residual.front().scale() -
                residual_target_scale) /
            residual_target_scale;
        bool residual_level_matches = true;
        if (loc == LOCATION::HOST) {
            residual_level_matches =
                enc_attention_residual.front().hcipher().parms_id() ==
                h3.front().hcipher().parms_id();
        } else {
            residual_level_matches =
                enc_attention_residual.front().dcipher().chain_index() ==
                h3.front().dcipher().chain_index();
        }
        if (!residual_level_matches || residual_scale_error > 1e-12) {
            throw runtime_error(
                "formal attention residual did not match h3 metadata");
        }
        cout << "FORMAL_RESIDUAL_ALIGNMENT stage=attention_output"
             << " status=PASS target_scale_bits="
             << log2(residual_target_scale)
             << " target_limbs=" << residual_target_limbs
             << " cts=" << enc_attention_residual.size()
             << " value_lift=real_1" << endl;
        if (!he->is_gpu_enable()) {
            for(int i=0;i<h3.size();i++){
                he->evaluator->add_inplace(h3[i], enc_attention_residual[i]);
                ++this->he_rescale_count;
                he->evaluator->rescale_to_next_inplace(h3[i]);
            }
        } else {
            for(int i=0;i<h3.size();i++){
                he->devaluator->add_inplace(h3[i], enc_attention_residual[i]);
                ++this->he_rescale_count;
                he->devaluator->rescale_to_next_inplace(h3[i]);
            }
        }
        attention_input_residual_ready = false;
        attention_v_he_ready = false;
        attention_v_he_cts.clear();
        cout << "OK1" << endl;
        if (pre_norm_attention_output_only) {
            // ViT is pre-norm: expose h3 immediately after the original
            // Softmax*V -> output projection -> residual CKKS path.  The
            // conversion is exactly Algorithm 1; no LayerNorm protocol is
            // skipped elsewhere because this branch is the model's residual
            // boundary, not an optimization.
            vector<Ciphertext> h3_full(h3.size());
            for (size_t i = 0; i < h3.size(); ++i) {
                if (he->is_gpu_enable()) {
                    h3[i].to_host(*he->context, *he->dcontext);
                }
                h3_full[i] = std::move(h3[i].hcipher());
            }
            prepare_host_conversion_from_actual_scale(
                *this, h3_full, kFormalConversionScaleBits, he,
                "vit_attention_residual_output");
            mask_ckks_to_mpc_inplace(h3_full, x_avg_pts, he);
            send_encrypted_vector(io, h3_full);
            cout << "FORMAL_TRANSFER_LAYOUT stage=vit_attention_residual"
                 << " status=PASS layout=bsgs2_to_full_slot cts="
                 << h3_full.size() << " sigma_cts=0" << endl;
        } else {
        if (profile_enabled_)
            profile_switch(BlbViTProfileStage::LAYERNORM1);
        // 开始计算layernorm
        // 1. X_sum
        vector<UnifiedCiphertext> X_avg(h3.size(), UnifiedCiphertext(loc));
        vector<UnifiedCiphertext> sigma(h3.size(), UnifiedCiphertext(loc));
        for(int i=0;i<h3.size();i++){
            X_avg[i] = h3[i];
        }
        for(int i = 1;i <= std::log2(slot/L);i ++){
            if (!he->is_gpu_enable()) {
                #pragma omp parallel for num_threads(4)
                for(int j=0;j<h3.size();j++){
                    Ciphertext tmp;
                    he->evaluator->rotate_vector(X_avg[j],pow(2,i-1)*L,*he->gal_keys,tmp);
                    he->evaluator->add_inplace(X_avg[j],tmp);
                }
            } else {
                for(int j=0;j<h3.size();j++){
                    UnifiedCiphertext tmp(loc);
                    he->devaluator->rotate_vector(X_avg[j],pow(2,i-1)*L,*he->dgal_keys,tmp);
                    he->devaluator->add_inplace(X_avg[j],tmp);
                }
            }
        }
        cout << "OK1" << endl;
        UnifiedCiphertext X_sum = X_avg[0];
        for(int i=1;i<X_avg.size();i++){
            if (!he->is_gpu_enable()) {
                he->evaluator->add_inplace(X_sum,X_avg[i]);
            } else {
                he->devaluator->add_inplace(X_sum, X_avg[i]);
            }
        }
        // #pragma omp parallel for num_threads(4)
        for(int i=0;i<X_avg.size();i++){
            X_avg[i] = X_sum;
        }
        cout << "OK2" << endl;
        // 2. X_avg, scale=4s-60
        for(int i=0;i<X_avg.size();i++){
            if (!he->is_gpu_enable()) {
                he->evaluator->multiply_plain_inplace(X_avg[i], inv_D_p);
            } else {
                he->devaluator->multiply_plain_inplace(X_avg[i], inv_D_p);
            }
        }
        cout << "OK3" << endl;
        /*
           3. scale up h3 (that is, x) from 3s-60 to 4s-60, thus h3 can sub X_avg
           4. sigma = h3 = h3-X_avg
           do 1*rs; then scale = 4s-60*2=52, Q = [60, 40, 60, 60]
        */ 
        cout << "h3.scale:" << log2(h3[0].scale()) << endl;
        cout << "X_avg.scale:" << log2(X_avg[0].scale()) << endl;
        for(int i=0;i<h3.size();i++){
           if (!he->is_gpu_enable()) {
                he->evaluator->multiply_plain_inplace(h3[i],scale_up);
                he->evaluator->sub_inplace(h3[i],X_avg[i]);
                ++this->he_rescale_count;
                he->evaluator->rescale_to_next_inplace(h3[i]);
            } else {
                he->devaluator->multiply_plain_inplace(h3[i],scale_up);
                he->devaluator->sub_inplace(h3[i],X_avg[i]);
                ++this->he_rescale_count;
                he->devaluator->rescale_to_next_inplace(h3[i]);
            }
        }
        cout << "OK4" << endl;
        /*
            compute X_sum2=sigma^2, 注意sigma^2维度需要mask成(L,1),为了降低后面sqrt维度
            同时把h3转化为 scale: 12, Q:60 by 2*ms, 1*rs
            sigma scale = 52^2-60, Q = [60, 40, 60]
        */ 
        cout << "h3.scale:" << log2(h3[0].scale()) << endl;
        for(int i=0;i<h3.size();i++){
            sigma[i] = h3[i];
            if (!he->is_gpu_enable()) {
                he->evaluator->square_inplace(sigma[i]);
                he->evaluator->relinearize_inplace(sigma[i], *he->relin_keys);
                ++this->he_rescale_count;
                he->evaluator->rescale_to_next_inplace(sigma[i]);
            } else {
                he->devaluator->square_inplace(sigma[i]);
                he->devaluator->relinearize_inplace(sigma[i], *he->drelin_keys);
                ++this->he_rescale_count;
                he->devaluator->rescale_to_next_inplace(sigma[i]);
            }
        }
        cout << "OK5" << endl;
        for(int i=1;i<=std::log2(slot/L);i++){
            if (!he->is_gpu_enable()) {
                #pragma omp parallel for num_threads(4)
                for(int j=0;j<sigma.size();j++){
                    Ciphertext tmp;
                    he->evaluator->rotate_vector(sigma[j],pow(2,i-1)*L,*he->gal_keys,tmp);
                    he->evaluator->add_inplace(sigma[j],tmp);
                }
            } else {
                for(int j=0;j<sigma.size();j++){
                    UnifiedCiphertext tmp(loc);
                    he->devaluator->rotate_vector(sigma[j],pow(2,i-1)*L,*he->dgal_keys,tmp);
                    he->devaluator->add_inplace(sigma[j],tmp);
                }
            }
        }
        cout << "OK5" << endl;
        UnifiedCiphertext X_sum2 = std::move(sigma[0]);
        for(int i=1;i<sigma.size();i++){
            if (!he->is_gpu_enable()) {
            he->evaluator->add_inplace(X_sum2,sigma[i]);
            } else {
            he->devaluator->add_inplace(X_sum2,sigma[i]);
            }
        }
        cout << "OK5" << endl;
        const double variance_source_scale = X_sum2.scale();
        const size_t variance_initial_limbs = X_sum2.coeff_modulus_size();
        // The original BLB variance schedule enters the public scale
        // alignment at Q=[60,40].  Phantom can retain an otherwise unused
        // higher level after the parallel square/rotation reductions; drop
        // only those extra levels before applying the same public masks.
        while (X_sum2.coeff_modulus_size() > 2) {
            if (loc == LOCATION::HOST) {
                he->evaluator->mod_switch_to_next_inplace(X_sum2);
            } else {
                he->devaluator->mod_switch_to_next_inplace(X_sum2);
            }
        }
        if (variance_initial_limbs != X_sum2.coeff_modulus_size()) {
            cout << "FORMAL_VARIANCE_LEVEL_ALIGNMENT stage=ln1"
                 << " status=PASS initial_limbs=" << variance_initial_limbs
                 << " final_limbs=" << X_sum2.coeff_modulus_size()
                 << " target_limbs=2" << endl;
        }
        const double variance_target_scale =
            ldexp(1.0, kFormalConversionScaleBits);
        uint64_t variance_dropped_modulus = 0;
        if (loc == LOCATION::HOST) {
            const auto variance_context = he->context->get_context_data(
                X_sum2.hcipher().parms_id());
            if (!variance_context ||
                variance_context->parms().coeff_modulus().size() != 2) {
                throw runtime_error(
                    "formal LN1 variance must enter alignment with two limbs");
            }
            variance_dropped_modulus =
                variance_context->parms().coeff_modulus().back().value();
        } else {
            const auto variance_chain = X_sum2.dcipher().chain_index();
            // Phantom's device chain index is intentionally the reverse of
            // SEAL's modulus-chain order.  Some Phantom operations preserve
            // the ciphertext limb count while carrying a chain index from a
            // copied operand, so use the actual ciphertext limb count for
            // this metadata lookup (the arithmetic schedule remains the
            // original two-limb variance schedule).
            const size_t variance_limbs = X_sum2.coeff_modulus_size();
            size_t variance_context_index = 0;
            bool found_variance_context = false;
            for (size_t i = 0; i < he->dcontext->context_data_.size(); ++i) {
                if (he->dcontext->context_data_[i]
                        .parms().coeff_modulus().size() == variance_limbs) {
                    variance_context_index = i;
                    found_variance_context = true;
                    break;
                }
            }
            if (!found_variance_context || variance_limbs != 2) {
                cout << "FORMAL_VARIANCE_CONTEXT_LOOKUP stage=ln1"
                     << " status=BLOCKED ciphertext_chain=" << variance_chain
                     << " ciphertext_limbs=" << variance_limbs
                     << " context_sizes=";
                for (size_t i = 0; i < he->dcontext->context_data_.size(); ++i) {
                    if (i != 0) {
                        cout << ',';
                    }
                    cout << he->dcontext->context_data_[i]
                                .parms()
                                .coeff_modulus()
                                .size();
                }
                cout << endl;
                throw runtime_error(
                    "formal LN1 variance DEVICE level is not two limbs "
                    "(inspect FORMAL_VARIANCE_CONTEXT_LOOKUP)");
            }
            const auto &variance_context =
                he->dcontext->get_context_data(variance_context_index);
            cout << "FORMAL_VARIANCE_CONTEXT_LOOKUP stage=ln1"
                 << " status=PASS ciphertext_chain=" << variance_chain
                 << " context_index=" << variance_context_index
                 << " limbs=" << variance_limbs << endl;
            variance_dropped_modulus =
                variance_context.parms().coeff_modulus().back().value();
        }
        const double required_plain_scale_product =
            variance_target_scale *
            static_cast<double>(variance_dropped_modulus) /
            variance_source_scale;
        const double variance_mask_scale =
            sqrt(required_plain_scale_product);
        const double variance_inv_d_scale =
            required_plain_scale_product / variance_mask_scale;
        if (!isfinite(variance_mask_scale) ||
            !isfinite(variance_inv_d_scale) ||
            variance_mask_scale < 1.0 || variance_inv_d_scale < 1.0) {
            throw runtime_error(
                "formal LN1 variance public scale is not encodable");
        }
        if (loc == LOCATION::HOST) {
            const auto parms_id = X_sum2.hcipher().parms_id();
            he->encoder->encode(mask, parms_id, variance_mask_scale, mask_p);
            he->encoder->encode(inv_D, parms_id, variance_inv_d_scale,
                                inv_D_p2);
        } else {
            const auto chain_index = X_sum2.dcipher().chain_index();
            he->dencoder->encode(*he->dcontext, mask, variance_mask_scale,
                                 mask_p, chain_index);
            he->dencoder->encode(*he->dcontext, inv_D,
                                 variance_inv_d_scale, inv_D_p2,
                                 chain_index);
        }
        cout << "FORMAL_VARIANCE_SCALE_PLAN stage=ln1 status=PASS"
             << " source_scale_bits=" << log2(variance_source_scale)
             << " dropped_modulus_bits="
             << log2(static_cast<double>(variance_dropped_modulus))
             << " mask_scale_bits=" << log2(variance_mask_scale)
             << " inv_d_scale_bits=" << log2(variance_inv_d_scale)
             << " target_scale_bits=" << kFormalConversionScaleBits
             << " source_limbs=2" << endl;
        if (!he->is_gpu_enable()) {
        he->evaluator->multiply_plain_inplace(X_sum2,mask_p);
        } else {
        he->devaluator->multiply_plain_inplace(X_sum2,mask_p);
        }
        cout << "OK6" << endl;
        if (!he->is_gpu_enable()) {
        he->evaluator->multiply_plain_inplace(X_sum2,inv_D_p2);
        //before scale=130, Q=[60, 40, 60]; after 2*rs scale = 30, Q = 60
        ++this->he_rescale_count;
        he->evaluator->rescale_to_next_inplace(X_sum2);
        // he->evaluator->rescale_to_next_inplace(X_sum2);
        } else {
        he->devaluator->multiply_plain_inplace(X_sum2,inv_D_p2);
        ++this->he_rescale_count;
        he->devaluator->rescale_to_next_inplace(X_sum2);
        }
        const double variance_scale_error =
            abs(X_sum2.scale() - variance_target_scale) /
            variance_target_scale;
        if (X_sum2.coeff_modulus_size() != 1 ||
            !isfinite(variance_scale_error) || variance_scale_error > 1e-9) {
            throw runtime_error(
                "formal LN1 variance did not reach the high-scale final prime");
        }
        cout << "FORMAL_VARIANCE_SCALE_READY stage=ln1 status=PASS"
             << " actual_scale_bits=" << log2(X_sum2.scale())
             << " limbs=" << X_sum2.coeff_modulus_size() << endl;
        this->he_compute_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        
        if (he->is_gpu_enable()) {
        // Load result ciphertexts to CPU
        start = high_resolution_clock::now();
        X_sum2.to_host(*he->context, *he->dcontext);
        this->ct_transfer_mem_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        }
        NVTX_POP(true);
        
        start = high_resolution_clock::now();
        cout << "num of output cts:" << sigma.size()+1 << endl;
        cout << "bitwidth of h3 ct:" << h3[0].coeff_modulus_size() << endl;
        cout << "bitwidth of X_sum2 ct:" << X_sum2.coeff_modulus_size() << endl;
        vector<Ciphertext> enc_sigma(1);
        enc_sigma[0] = std::move(X_sum2);
        cout << "h3.scale:" << log2(h3[0].scale()) << endl;
        cout << "enc_sigma.scale:" << log2(enc_sigma[0].scale()) << endl;
        vector<Ciphertext> h3_full(num_cts_ln);
        if (h3.size() != h3_full.size()) {
            throw runtime_error("formal LayerNorm full-slot ciphertext count mismatch");
        }
        for(int i=0;i<h3_full.size();i++){
            if (he->is_gpu_enable()) {
                h3[i].to_host(*he->context, *he->dcontext);
            }
            h3_full[i] = std::move(h3[i].hcipher());
        }
        // Algorithm 1 masks only final-prime ciphertexts at the common high
        // conversion scale.  ckks_to_mpc then securely truncates to s12.
        prepare_host_conversion_from_actual_scale(
            *this, h3_full, kFormalConversionScaleBits, he,
            "ln1_centered");
        prepare_host_conversion_from_actual_scale(
            *this, enc_sigma, kFormalConversionScaleBits, he,
            "ln1_variance");
        if (num_cts_ln % 2 != 0) {
            throw runtime_error("formal LayerNorm output half-packing requires an even ciphertext count");
        }
        vector<Ciphertext> h3_half(num_cts_ln / 2);
        for (size_t i = 0; i < h3_half.size(); ++i) {
            h3_half[i] = std::move(h3_full[i]);
        }
        vector<Plaintext> x_avg_pts_half_send;
        mask_ckks_to_mpc_inplace(h3_half, x_avg_pts_half_send, he);
        send_encrypted_vector(io, h3_half);
        x_avg_pts_half = std::move(x_avg_pts_half_send);
        mask_ckks_to_mpc_inplace(enc_sigma, sigma_pts, he);
        send_encrypted_vector(io, enc_sigma);
        cout << "FORMAL_TRANSFER_LAYOUT stage=softmax_v_ln_output status=PASS"
             << " layout=xyj_half_slot h3_cts=" << h3_half.size()
             << " sigma_cts=1 half_decode=1 source=Figure10_Block4" << endl;
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        }
    }
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    // cout << "send and receive cts, comm:" << this->get_comm() << endl;
    // cout << "send and receive cts, round:" << this->get_round() << endl;
    cout << "send and receive done in linear4_add_layernorm_part1" << endl;
    if (pre_norm_attention_output_only) {
        vector<vector<HE_CKKS::scalar_t>> residual_vector = ckks_to_mpc(
            x_avg_pts, poly_modulus_degree,
            he->get_Q(x_avg_pts[0]), bitwidth, s, he, false);
        FixArray packed(party, num_cts_ln * slot, true, bitwidth, s);
        size_t offset = 0;
        for (const auto &part : residual_vector) {
            if (part.size() != static_cast<size_t>(slot) ||
                offset + part.size() > static_cast<size_t>(packed.size)) {
                throw runtime_error("formal ViT attention residual decode mismatch");
            }
            copy(part.begin(), part.end(), packed.data + offset);
            offset += part.size();
        }
        if (offset != static_cast<size_t>(packed.size)) {
            throw runtime_error("formal ViT attention residual is incomplete");
        }
        map<char*,FixArray> result;
        result["residual"] = bsgs_output_to_token_major(
            packed, linear.data_lin2, "vit_attention_residual_to_token");
        this->conversion_comm += this->get_comm();
        this->conversion_round += this->get_round();
        return result;
    }
    vector<vector<HE_CKKS::scalar_t>> vec_x_avg(x_avg_pts.size());
    vector<vector<HE_CKKS::scalar_t>> vec_sigma(sigma_pts.size());
    start = high_resolution_clock::now();
    if (!pre_norm_attention_output_only && !x_avg_pts_half.empty()) {
        vec_x_avg = ckks_to_mpc(x_avg_pts_half, poly_modulus_degree,
                                he->get_Q(x_avg_pts_half[0]), bitwidth, s,
                                he, true);
    } else {
        vec_x_avg = ckks_to_mpc(x_avg_pts, poly_modulus_degree,
                                he->get_Q(x_avg_pts[0]), bitwidth, s, he,
                                false);
    }
    vec_sigma = ckks_to_mpc(sigma_pts, poly_modulus_degree, he->get_Q(sigma_pts[0]), bitwidth, s, he, false);
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    cout << "vec_sigma.size():" << vec_sigma.size() << ", should be 1" << endl;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    // cout << "mpc_to_ckks comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks round:" << this->get_round() << endl;
    cout << "bitwidth,s:" << bitwidth << "," << s << endl;
    FixArray x_flat_avg, sigma_flat;
    FixArray x_flat_avg_packed(
        party, num_cts_ln * slot, true, bitwidth, s);
    sigma_flat = FixArray(
        party, data.image_size, true, bitwidth, s);
    for(int i=0;i<vec_x_avg.size();i++){
        for(int j=0;j<vec_x_avg[i].size();j++){
            x_flat_avg_packed.data[i*vec_x_avg[i].size()+j] =
                vec_x_avg[i][j];
        }
    }
    x_flat_avg = bsgs_output_to_token_major(
        x_flat_avg_packed, linear.data_lin2, "h3_to_ln1_mpc");
    cout << "FORMAL_LAYOUT_ADAPTER stage=h3_to_ln1_mpc"
         << " status=PASS source=bsgs2_split_feature_halves"
         << " target=token_feature_row_major elements=" << x_flat_avg.size
         << " communication=0 public_share_permutation=1" << endl;
    for(int j=0;j<data.image_size;j++){
        sigma_flat.data[j] = vec_sigma[0][j];
    }
    map<char*,FixArray> result;
    result["x_flat_avg"] = x_flat_avg;
    result["sigma_flat"] = sigma_flat;
    return result;
}

/*
    do not support multiple threads
    num_ops is sequence*dim
*/ 
// std::map<char*,FixArray> Bert_ckks::gelu_part1(uint64_t *x, int num_ops, int bitwidth, int s){
//     FPMath *fpmath = nonlinear.fpmath[0];
//     map<char*,FixArray> result;
//     int this_party = party;
//     int N = num_ops;
//     FixArray x_array;
//     x_array = fpmath->fix->input(this_party, num_ops, x, true, bitwidth, s);
//     vector<Plaintext> gelu_poly_pts = eval_gelu_poly_old(x_array);
//     vector<vector<HE_CKKS::scalar_t>> gelu_poly_mpc = ckks_to_mpc(gelu_poly_pts, poly_modulus_degree, linear.he->get_Q(gelu_poly_pts[0]), bitwidth, s);
//     FixArray gelu_poly_flat(this_party, num_ops, true, bitwidth, s);
//     for(int i=0;i<gelu_poly_mpc.size();i++){
//         for(int j=0;j<gelu_poly_mpc[i].size();j++){
//             gelu_poly_flat.data[i*gelu_poly_mpc[i].size()+j] = gelu_poly_mpc[i][j];
//         }
//     }
//     result["gelu_poly_flat"] = gelu_poly_flat;
//     result["x_array"] = x_array;
//     return result;
// }

/*
    input 的 bitwidth 与 scale 需要是正确的MPC下的值
*/
std::map<char*,FixArray> Bert_ckks::ln_part2_linear3_gelu_part1(
    std::map<char*,FixArray> input, int layer_id){
    // multi depth = 6
    
    FixArray x_flat_avg = input["x_flat_avg"];
    FixArray sigma_flat = input["sigma_flat"];
    FixArray w = input["w"];
    FixArray b = input["b"];
    if (x_flat_avg.size <= 0 || sigma_flat.size != x_flat_avg.size ||
        w.size != x_flat_avg.size || b.size != x_flat_avg.size) {
        throw runtime_error(
            "formal LayerNorm/FFN affine requires shape-matched shares");
    }
    require_layernorm_parameter_scale(x_flat_avg, sigma_flat, w, b);
    map<char*,FixArray> result;
    int bitwidth = x_flat_avg.ell;
    int s = x_flat_avg.s;
    if (bitwidth != NL_ELL || s != NL_SCALE) {
        throw runtime_error(
            "formal LN1 affine/FFN1 requires the paper nonlinear ell/s shares");
    }
    if (layer_id < 0 || layer_id >= ATTENTION_LAYERS) {
        throw runtime_error("formal FFN1 layer id is out of range");
    }
    const int ln1_tokens = linear.data_lin3.image_size;
    const int ln1_features = linear.data_lin3.filter_h;
    FixArray x_flat_avg_column = token_major_to_feature_major(
        x_flat_avg, ln1_tokens, ln1_features, "ln1_centered_to_ffn1");
    FixArray sigma_flat_column = token_major_to_feature_major(
        sigma_flat, ln1_tokens, ln1_features, "ln1_sigma_to_ffn1");
    FixArray w_column = token_major_to_feature_major(
        w, ln1_tokens, ln1_features, "ln1_weight_to_ffn1");
    FixArray b_column = token_major_to_feature_major(
        b, ln1_tokens, ln1_features, "ln1_bias_to_ffn1");
    cout << "FORMAL_LAYOUT_ADAPTER stage=ln1_affine_to_ffn1"
         << " status=PASS source=token_feature_row_major"
         << " target=feature_token_column_major elements="
         << x_flat_avg_column.size
         << " tensors=x,sigma,weight,bias communication=0"
         << " public_share_permutation=1" << endl;
    cout << "in ln_part2_linear3_gelu_part1, bitwidth,s:" << bitwidth << "," << s << endl;
    cout << "should be 0! comm:" << this->get_comm() << ", round:" << this->get_round() << endl;
    HE_CKKS* he = linear.he3;
    uint64_t dim = x_flat_avg.size;
    uint64_t poly_modulus_degree = he->poly_modulus_degree;
    uint64_t slot_count = poly_modulus_degree / 2;
    uint64_t num_cts = dim / slot_count;
    FixArray gelu_poly_flat, x_array;

    vector<vector<HE_CKKS::scalar_t>> vec_x_flat_avg(num_cts), vec_sigma_flat(num_cts);
    for(int i = 0; i < num_cts; i++){
        for(int j = 0; j < slot_count; j++){
            vec_x_flat_avg[i].push_back(
                x_flat_avg_column.data[i*slot_count + j]);
            vec_sigma_flat[i].push_back(
                sigma_flat_column.data[i*slot_count + j]);
        }
    }
    vector<UnifiedPlaintext> pts_x_flat_avg;
    vector<UnifiedPlaintext> pts_sigma_flat;
    
    FCMetadata data = linear.data_lin3;
    // Figure 10's x+y*j boundary is the BERT post-norm path.  The shared
    // routine is also used by the pre-norm ViT path; keep that path on its
    // original full-slot transfers.
    const bool use_bert_half_packing = !pre_norm_ffn_residual_from_external;
    int gelu_input_size = data.image_size*INTER_DIM;
    int gelu_cts_size = gelu_input_size / slot_count;
    vector<Ciphertext> output_cts(
        gelu_cts_size
    );
    vector<Ciphertext> output_cts_half(
        gelu_cts_size % 2 == 0 ? gelu_cts_size / 2 : 0);
    vector<Plaintext> output_pts(
        gelu_cts_size
    );
    vector<Plaintext> output_pts_half(
        gelu_cts_size % 2 == 0 ? gelu_cts_size / 2 : 0);
    // The MPC tail selection in gelu_nonlinear needs the true FFN1 output x,
    // not a synthetic constant.  Export a second, independently masked
    // Algorithm-1 branch from linear3_output while retaining the original
    // ciphertext vector for eval_gelu_poly.
    vector<Ciphertext> pre_gelu_cts(gelu_cts_size);
    vector<Ciphertext> pre_gelu_cts_half(
        gelu_cts_size % 2 == 0 ? gelu_cts_size / 2 : 0);
    vector<Plaintext> pre_gelu_pts(gelu_cts_size);
    vector<Plaintext> pre_gelu_pts_half(
        gelu_cts_size % 2 == 0 ? gelu_cts_size / 2 : 0);
    // The FFN residual is the *post-attention LayerNorm affine output* (H4)
    // in post-norm BERT.  It is exported as an additive share from the same
    // CKKS ciphertext before FFN1; the old pre-LayerNorm h3 transfer is not a
    // valid FFN residual.
    vector<Ciphertext> normalized_residual_cts(
        pre_norm_ffn_residual_from_external ? 0 : num_cts);
    vector<Plaintext> normalized_residual_pts(
        pre_norm_ffn_residual_from_external ? 0 : num_cts);
    cout << "gelu_cts_size:" << gelu_cts_size << endl;
    auto start = high_resolution_clock::now();
    import_target_scale(*this, vec_x_flat_avg, slot_count, bitwidth, s,
                            he->he_scale, he, pts_x_flat_avg,
                            "ln1_centered_input");
    import_target_scale(*this, vec_sigma_flat, slot_count, bitwidth, s,
                            he->he_scale, he, pts_sigma_flat,
                            "ln1_inv_sigma_input");
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    // cout << "mpc_to_ckks comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks round:" << this->get_round() << endl;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    
    LOCATION loc = he->backend();
    vector<UnifiedCiphertext> cts_x_flat(
        pts_x_flat_avg.size(),
        UnifiedCiphertext(LOCATION::HOST)
    );
    vector<UnifiedCiphertext> cts_sigma_flat(
        pts_sigma_flat.size(),
        UnifiedCiphertext(LOCATION::HOST)
    );
    vector<UnifiedCiphertext> cts_x_flat_half(
        pts_x_flat_avg.size() % 2 == 0 ? pts_x_flat_avg.size() / 2 : 0,
        UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> cts_sigma_flat_half(
        pts_sigma_flat.size() % 2 == 0 ? pts_sigma_flat.size() / 2 : 0,
        UnifiedCiphertext(LOCATION::HOST));
    cout << "cts_x_flat.size():" << cts_x_flat.size() << endl;
    cout << "cts_sigma_flat.size():" << cts_sigma_flat.size() << endl;
    if (party == BOB) {
        cout << "send_encrypted_vector" << endl;
        if (use_bert_half_packing) {
            if (cts_x_flat_half.empty() || cts_sigma_flat_half.empty()) {
                throw runtime_error("formal FFN1 input half-packing requires an even ciphertext count");
            }
            #pragma omp parallel for
            for(size_t i=0;i<cts_x_flat_half.size();i++){
                he->encryptor->encrypt(pts_x_flat_avg[i], cts_x_flat_half[i]);
                he->encryptor->encrypt(pts_sigma_flat[i], cts_sigma_flat_half[i]);
            }
            send_encrypted_vector(io, cts_x_flat_half);
            send_encrypted_vector(io, cts_sigma_flat_half);
            cout << "FORMAL_TRANSFER_LAYOUT stage=ln_ffn_input status=PASS"
                 << " layout=xyj_half_slot x_cts=" << cts_x_flat_half.size()
                 << " sigma_cts=" << cts_sigma_flat_half.size()
                 << " imag_repack=1 source=Figure10_Block5" << endl;
        } else {
            #pragma omp parallel for
            for(size_t i=0;i<cts_x_flat.size();i++){
                he->encryptor->encrypt(pts_x_flat_avg[i], cts_x_flat[i]);
                he->encryptor->encrypt(pts_sigma_flat[i], cts_sigma_flat[i]);
            }
            send_encrypted_vector(io, cts_x_flat);
            send_encrypted_vector(io, cts_sigma_flat);
            cout << "FORMAL_TRANSFER_LAYOUT stage=ln_ffn_input status=PASS"
                 << " layout=ordinary_full_slot x_cts=" << cts_x_flat.size()
                 << " sigma_cts=" << cts_sigma_flat.size()
                 << " imag_repack=0 source=ViT_original_path" << endl;
        }
        cout << "send_encrypted_vector done" << endl;
		// send_encrypted_vector(io, cts);
		// vector<Ciphertext> share_client(dim);
        if (!pre_norm_ffn_residual_from_external) {
            recv_encrypted_vector(he->context, io, normalized_residual_cts);
            #pragma omp parallel for
            for (int i = 0; i < normalized_residual_cts.size(); ++i) {
                he->decryptor->decrypt(normalized_residual_cts[i],
                                       normalized_residual_pts[i]);
            }
        }
        if (use_bert_half_packing) {
            if (pre_gelu_cts_half.empty() || output_cts_half.empty()) {
                throw runtime_error("formal GeLU boundary half-packing requires an even ciphertext count");
            }
            recv_encrypted_vector(he->context, io, pre_gelu_cts_half);
            #pragma omp parallel for
            for (int i = 0; i < pre_gelu_cts_half.size(); ++i) {
                he->decryptor->decrypt(pre_gelu_cts_half[i], pre_gelu_pts_half[i]);
            }
            recv_encrypted_vector(he->context, io, output_cts_half);
            #pragma omp parallel for
            for(int i=0;i<output_cts_half.size();i++){
                he->decryptor->decrypt(output_cts_half[i], output_pts_half[i]);
            }
        } else {
            recv_encrypted_vector(he->context, io, pre_gelu_cts);
            #pragma omp parallel for
            for (int i = 0; i < pre_gelu_cts.size(); ++i) {
                he->decryptor->decrypt(pre_gelu_cts[i], pre_gelu_pts[i]);
            }
            recv_encrypted_vector(he->context, io, output_cts);
            #pragma omp parallel for
            for(int i=0;i<output_cts.size();i++){
                he->decryptor->decrypt(output_cts[i], output_pts[i]);
            }
        }
	}
    else{
        start = high_resolution_clock::now();
        if (use_bert_half_packing) {
            if (cts_x_flat_half.empty() || cts_sigma_flat_half.empty()) {
                throw runtime_error("formal FFN1 input half-packing requires an even ciphertext count");
            }
            recv_encrypted_vector(he->context, io, cts_x_flat_half);
            recv_encrypted_vector(he->context, io, cts_sigma_flat_half);
            #pragma omp parallel for
            for (size_t i = 0; i < cts_x_flat_half.size(); ++i) {
                he->imag_repack(cts_x_flat_half[i], cts_x_flat[i],
                                cts_x_flat[i + cts_x_flat_half.size()]);
                he->imag_repack(cts_sigma_flat_half[i], cts_sigma_flat[i],
                                cts_sigma_flat[i + cts_sigma_flat_half.size()]);
            }
        } else {
            recv_encrypted_vector(he->context, io, cts_x_flat);
            recv_encrypted_vector(he->context, io, cts_sigma_flat);
        }
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        
        NVTX_PUSH("LN2-L3-GL1", true);
        if (he->is_gpu_enable()) {
        // Load revice ciphertexts to GPU
        start = high_resolution_clock::now();
        for_each_n(cts_x_flat.begin(), cts_x_flat.size(),
            [he](auto &ct) { ct.to_device(*he->context, *he->dcontext); });
        for_each_n(cts_sigma_flat.begin(), cts_sigma_flat.size(),
                [he](auto &ct) { ct.to_device(*he->context, *he->dcontext); });
        for_each_n(pts_x_flat_avg.begin(), pts_x_flat_avg.size(),
                [he](auto &pt) { pt.to_device(*he->context, *he->dcontext); });
        for_each_n(pts_sigma_flat.begin(), pts_sigma_flat.size(),
                [he](auto &pt) { pt.to_device(*he->context, *he->dcontext); });
        this->ct_transfer_mem_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        }

        start = high_resolution_clock::now();
        cout << "FORMAL_TRANSFER_LAYOUT stage=ln_ffn_input status=PASS"
             << " layout=" << (use_bert_half_packing ? "xyj_half_slot" : "ordinary_full_slot")
             << " x_cts=" << (use_bert_half_packing ? cts_x_flat_half.size() : cts_x_flat.size())
             << " sigma_cts=" << (use_bert_half_packing ? cts_sigma_flat_half.size() : cts_sigma_flat.size())
             << " imag_repack=" << (use_bert_half_packing ? 1 : 0)
             << " source=" << (use_bert_half_packing ? "Figure10_Block5" : "ViT_original_path") << endl;
        cout << "num of input cts:" << cts_x_flat.size()+cts_sigma_flat.size() << endl;
        cout << "bitwidth of input ct:" << cts_x_flat[0].coeff_modulus_size() << endl;
        cout << "recv_encrypted_vector" << endl;
        
        // Encode weight in once.
        vector<UnifiedPlaintext> tmp_w(cts_x_flat.size(), UnifiedPlaintext(loc));
        vector<UnifiedPlaintext> tmp_b(cts_x_flat.size(), UnifiedPlaintext(loc));
        vector<vector<double>> w_matrix(cts_x_flat.size(),vector<double>(slot_count, 0));
        vector<vector<double>> b_matrix(cts_x_flat.size(),vector<double>(slot_count, 0));
        for(int i=0;i<cts_x_flat.size();i++){
            for(int j=0;j<slot_count;j++){
                const double projection_compensation =
                    use_bert_half_packing ? 1.0 : 0.25;
                w_matrix[i][j] = projection_compensation *
                    layernorm_fixed_to_real(
                        w_column.data[i*slot_count+j], w_column,
                        "LayerNorm weight");
                b_matrix[i][j] = layernorm_fixed_to_real(
                    b_column.data[i*slot_count+j], b_column,
                    "LayerNorm bias");
            }
        }
        const double ln_affine_weight_scale = ldexp(1.0, he->he_scale);
        const double ln_affine_bias_scale =
            ln_affine_weight_scale * ln_affine_weight_scale *
            ln_affine_weight_scale;
        const double ln_affine_input_scale =
            sqrt(ln_affine_bias_scale / ln_affine_weight_scale);
        if (!isfinite(ln_affine_input_scale) ||
            ln_affine_input_scale < pts_x_flat_avg.front().scale()) {
            throw runtime_error("formal LN1 affine input scale is invalid");
        }
        if (!he->is_gpu_enable()) {
        // Parallelize host codes through OpenMP
        #pragma omp parallel for
        for(int i=0;i<cts_x_flat.size();i++){
            he->encoder->encode(w_matrix[i],
                                cts_x_flat[i].hcipher().parms_id(),
                                ln_affine_weight_scale, tmp_w[i]);
            he->encoder->encode(b_matrix[i],
                                cts_x_flat[i].hcipher().parms_id(),
                                ln_affine_bias_scale, tmp_b[i]);
        }
        } else {
        // Parallelize device codes through CUDA Stream
        for(int i=0;i<cts_x_flat.size();i++){
            const auto chain_index = cts_x_flat[i].dcipher().chain_index(); 
            he->dencoder->encode(*he->dcontext, w_matrix[i],
                                 ln_affine_weight_scale, tmp_w[i],
                                 chain_index);
            he->dencoder->encode(*he->dcontext, b_matrix[i],
                                 ln_affine_bias_scale, tmp_b[i],
                                 chain_index);
        }
        }

        // Recombine the encrypted additive shares at their honest semantic
        // scale before applying the public scale lift.
        for(int i=0;i<cts_x_flat.size();i++){
            if (!he->is_gpu_enable()) {
                he->evaluator->add_plain_inplace(cts_x_flat[i],
                                                 pts_x_flat_avg[i]);
                he->evaluator->add_plain_inplace(cts_sigma_flat[i],
                                                 pts_sigma_flat[i]);
                if (!use_bert_half_packing) {
                    UnifiedCiphertext x_conjugate(LOCATION::HOST);
                    UnifiedCiphertext sigma_conjugate(LOCATION::HOST);
                    he->evaluator->complex_conjugate(
                        cts_x_flat[i], *he->gal_keys, x_conjugate);
                    he->evaluator->complex_conjugate(
                        cts_sigma_flat[i], *he->gal_keys, sigma_conjugate);
                    he->evaluator->add_inplace(cts_x_flat[i], x_conjugate);
                    he->evaluator->add_inplace(cts_sigma_flat[i],
                                               sigma_conjugate);
                }
            } else {
                he->devaluator->add_plain_inplace(cts_x_flat[i],
                                                  pts_x_flat_avg[i]);
                he->devaluator->add_plain_inplace(cts_sigma_flat[i],
                                                  pts_sigma_flat[i]);
                if (!use_bert_half_packing) {
                    UnifiedCiphertext x_conjugate(LOCATION::DEVICE);
                    UnifiedCiphertext sigma_conjugate(LOCATION::DEVICE);
                    he->devaluator->complex_conjugate(
                        cts_x_flat[i], *he->dgal_keys, x_conjugate);
                    he->devaluator->complex_conjugate(
                        cts_sigma_flat[i], *he->dgal_keys,
                        sigma_conjugate);
                    he->devaluator->add_inplace(cts_x_flat[i], x_conjugate);
                    he->devaluator->add_inplace(cts_sigma_flat[i],
                                                sigma_conjugate);
                }
            }
        }
        require_uniform_unified_metadata(cts_x_flat, "ln1_affine_x");
        require_uniform_unified_metadata(cts_sigma_flat, "ln1_affine_sigma");
        const double ln1_x_scale_error =
            abs(cts_x_flat.front().scale() - ln_affine_input_scale) /
            ln_affine_input_scale;
        const double ln1_sigma_scale_error =
            abs(cts_sigma_flat.front().scale() - ln_affine_input_scale) /
            ln_affine_input_scale;
        if (ln1_x_scale_error > 1e-12 || ln1_sigma_scale_error > 1e-12) {
            throw runtime_error(
                "formal LN1 Algorithm-2 target scale mismatch");
        }
        cout << "FORMAL_LAYERNORM_SCALE_PLAN stage=ln1_affine status=PASS"
             << " input_scale_bits=" << log2(ln_affine_input_scale)
             << " weight_scale_bits=" << log2(ln_affine_weight_scale)
             << " bias_scale_bits=" << log2(ln_affine_bias_scale)
             << " limbs=" << cts_x_flat.front().coeff_modulus_size()
             << " real_projection="
             << (use_bert_half_packing ? "xyj_imag_repack"
                                       : "both_conjugate")
             << " gamma_compensation="
             << (use_bert_half_packing ? 1.0 : 0.25)
             << endl;

        // get x_avg and sigma, compute (x_avg * sigma) * w + b
        #pragma omp parallel for
        for(int i=0;i<cts_x_flat.size();i++){
            if (!he->is_gpu_enable()) {
                // x and sigma share the input scale.  Form that ct-ct
                // product first, then multiply by the public weight so the
                // resulting scale matches the bias plaintext.
                he->evaluator->multiply_inplace(cts_x_flat[i],
                                                cts_sigma_flat[i]);
                he->evaluator->relinearize_inplace(cts_x_flat[i],
                                                   *he->relin_keys);
                he->evaluator->multiply_plain_inplace(cts_x_flat[i], tmp_w[i]);
                he->evaluator->add_plain_inplace(cts_x_flat[i], tmp_b[i]);
                ++this->he_rescale_count;
                he->evaluator->rescale_to_next_inplace(cts_x_flat[i]);
            } else {
                he->devaluator->multiply_inplace(cts_x_flat[i],
                                                 cts_sigma_flat[i]);
                he->devaluator->relinearize_inplace(cts_x_flat[i],
                                                    *he->drelin_keys);
                he->devaluator->multiply_plain_inplace(cts_x_flat[i], tmp_w[i]);
                he->devaluator->add_plain_inplace(cts_x_flat[i], tmp_b[i]);
                ++this->he_rescale_count;
                he->devaluator->rescale_to_next_inplace(cts_x_flat[i]);
            }
        }
        
        cout << "(x_avg * sigma) * w + b done" << endl;  
        cout << "Q, scale:" << he->get_logQ(cts_x_flat[0]) << "," << log2(cts_x_flat[0].scale()) << endl;
        if (he->is_gpu_enable()) {
            // Export a host copy only for the existing Algorithm-1 masking
            // boundary.  Keep cts_x_flat on DEVICE for the following FFN1;
            // no unmasked ciphertext and no reconstructed intermediate is
            // exposed to either party.
            auto copy_start = high_resolution_clock::now();
            for (size_t i = 0; i < normalized_residual_cts.size(); ++i) {
                UnifiedCiphertext::to_host(
                    *he->dcontext, cts_x_flat[i].dcipher(), *he->context,
                    normalized_residual_cts[i]);
            }
            this->ct_transfer_mem_la +=
                ((high_resolution_clock::now() - copy_start)).count() / 1e9;
        } else {
            for (size_t i = 0; i < normalized_residual_cts.size(); ++i) {
                normalized_residual_cts[i] = cts_x_flat[i].hcipher();
            }
        }
        if (!pre_norm_ffn_residual_from_external) {
            prepare_host_conversion_from_actual_scale(
                *this, normalized_residual_cts,
                kFormalConversionScaleBits, he,
                "ln1_affine_residual");
            mask_ckks_to_mpc_inplace(normalized_residual_cts,
                                     normalized_residual_pts, he);
            send_encrypted_vector(io, normalized_residual_cts);
        }
        /*
            计算linear3
        */
        FCMetadata data = linear.data_lin3;
        vector<UnifiedCiphertext> linear3_output;
        const auto linear3_profile_start = high_resolution_clock::now();
        linear.linear_2(
            he,
            cts_x_flat, 
            linear.pp_3[layer_id],
            data,
            linear3_output
        );
        if (profile_enabled_)
            profile_record_he(
                BlbViTProfileStage::FFN1,
                chrono::duration<double, milli>(high_resolution_clock::now() -
                                                linear3_profile_start)
                    .count());
        cout << "linear2 done" << endl;
        cout << "scale:" << log2(linear3_output[0].scale()) << endl;

        #pragma omp parallel for
        for(int i=0;i<linear3_output.size();i++){
            if (!he->is_gpu_enable()) {
            ++this->he_rescale_count;
            he->evaluator->rescale_to_next_inplace(linear3_output[i]);
            he->evaluator->relinearize_inplace(linear3_output[i], *he->relin_keys); // FIXME: @relin
            } else {
            ++this->he_rescale_count;
            he->devaluator->rescale_to_next_inplace(linear3_output[i]);
            }
        }
        cout << "scale:" << log2(linear3_output[0].scale()) << endl;
        require_uniform_unified_metadata(linear3_output,
                                         "ffn1_pre_gelu_source");
        // Snapshot the true post-linear3, pre-polynomial ciphertext.  The
        // snapshot is ciphertext-only and remains masked before it leaves
        // Alice.  linear3_output itself stays available to eval_gelu_poly.
        this->he_compute_la +=
            ((high_resolution_clock::now() - start)).count()/1e+9;
        if (he->is_gpu_enable()) {
            auto copy_start = high_resolution_clock::now();
            for (size_t i = 0; i < linear3_output.size(); ++i) {
                UnifiedCiphertext::to_host(
                    *he->dcontext, linear3_output[i].dcipher(), *he->context,
                    pre_gelu_cts[i]);
            }
            this->ct_transfer_mem_la +=
                ((high_resolution_clock::now() - copy_start)).count()/1e+9;
        } else {
            for (size_t i = 0; i < linear3_output.size(); ++i) {
                pre_gelu_cts[i] = linear3_output[i].hcipher();
            }
        }
        cout << "FORMAL_PRE_GELU_SOURCE status=PASS"
             << " source=real_linear3_output cts=" << pre_gelu_cts.size()
             << " source_scale_bits=" << log2(linear3_output[0].scale())
             << " source_limbs=" << linear3_output[0].coeff_modulus_size()
             << " plaintext_recovery=0 constant_placeholder=0" << endl;
        start = high_resolution_clock::now();
        /*
            before: scale 64, Q= (60, 32, 60, 60, 60)
        */
        vector<UnifiedCiphertext> gelu_output;
        eval_gelu_poly(gelu_output, linear3_output, he,
                       kFormalConversionScaleBits);
        this->he_compute_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        
        if (he->is_gpu_enable()) {
        // Load result ciphertexts to CPU
        start = high_resolution_clock::now();
        for(size_t i = 0;i < gelu_output.size();i ++){
            gelu_output[i].to_host(*he->context, *he->dcontext);
        }
        this->ct_transfer_mem_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        }
        NVTX_POP(true);

        start = high_resolution_clock::now();
        cout << "num of output cts:" << gelu_output.size() << endl;
        cout << "bitwidth of output ct:" << gelu_output[0].coeff_modulus_size() << endl;
        if (gelu_output.size() != output_cts.size()) {
            throw runtime_error("formal GeLU full-slot ciphertext count mismatch");
        }
        if (pre_gelu_cts.size() != output_cts.size()) {
            throw runtime_error(
                "formal pre-GeLU full-slot ciphertext count mismatch");
        }
        for(int i=0;i<output_cts.size();i++){
            output_cts[i] = std::move(gelu_output[i].hcipher());
        }
        if (use_bert_half_packing) {
            if (pre_gelu_cts_half.empty() || output_cts_half.empty()) {
                throw runtime_error("formal GeLU boundary half-packing requires an even ciphertext count");
            }
            for (size_t i = 0; i < pre_gelu_cts_half.size(); ++i) {
                pre_gelu_cts_half[i] = std::move(pre_gelu_cts[i]);
                output_cts_half[i] = std::move(output_cts[i]);
            }
            prepare_host_conversion_from_actual_scale(
                *this, pre_gelu_cts_half, kFormalConversionScaleBits, he,
                "ffn1_pre_gelu");
            mask_ckks_to_mpc_inplace(pre_gelu_cts_half, pre_gelu_pts_half, he);
            send_encrypted_vector(io, pre_gelu_cts_half);
            prepare_host_conversion_from_actual_scale(
                *this, output_cts_half, kFormalConversionScaleBits, he,
                "ffn1_gelu");
            mask_ckks_to_mpc_inplace(output_cts_half, output_pts_half, he);
            send_encrypted_vector(io, output_cts_half);
            cout << "FORMAL_TRANSFER_LAYOUT stage=ln_ffn_output status=PASS"
                 << " layout=xyj_half_slot residual_cts="
                 << normalized_residual_cts.size()
                 << " pre_gelu_cts=" << pre_gelu_cts_half.size()
                 << " gelu_cts=" << output_cts_half.size()
                 << " masked_algorithm1=1 half_decode=1 intermediate_tensor_opened=0"
                 << " source=Figure10_Block5" << endl;
        } else {
            prepare_host_conversion_from_actual_scale(
                *this, pre_gelu_cts, kFormalConversionScaleBits, he,
                "ffn1_pre_gelu");
            mask_ckks_to_mpc_inplace(pre_gelu_cts, pre_gelu_pts, he);
            send_encrypted_vector(io, pre_gelu_cts);
            prepare_host_conversion_from_actual_scale(
                *this, output_cts, kFormalConversionScaleBits, he,
                "ffn1_gelu");
            mask_ckks_to_mpc_inplace(output_cts, output_pts, he);
            send_encrypted_vector(io, output_cts);
            cout << "FORMAL_TRANSFER_LAYOUT stage=ln_ffn_output status=PASS"
                 << " layout=ordinary_full_slot residual_cts="
                 << normalized_residual_cts.size()
                 << " pre_gelu_cts=" << pre_gelu_cts.size()
                 << " gelu_cts=" << output_cts.size()
                 << " masked_algorithm1=1 half_decode=0 intermediate_tensor_opened=0"
                 << " source=ViT_original_path" << endl;
        }
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        cout << "linear3_output.size():" << linear3_output.size() << endl;
        cout << "output_pts.size():" << output_pts.size() << endl;
    }

    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    cout << "send and receive done in ln_part2_linear3_gelu_part1" << endl;
    vector<vector<HE_CKKS::scalar_t>> normalized_residual_vector;
    if (!normalized_residual_pts.empty()) {
        normalized_residual_vector = ckks_to_mpc(
            normalized_residual_pts, poly_modulus_degree,
            he->get_Q(normalized_residual_pts[0]), bitwidth, s, he, false);
        FixArray ffn_residual_column(
            party, linear.data_lin1_0.image_size * COMMON_DIM, true,
            bitwidth, s);
        size_t residual_offset = 0;
        for (const auto &packed : normalized_residual_vector) {
            for (auto value : packed) {
                if (residual_offset <
                    static_cast<size_t>(ffn_residual_column.size)) {
                    ffn_residual_column.data[residual_offset++] = value;
                }
            }
        }
        if (residual_offset !=
            static_cast<size_t>(ffn_residual_column.size)) {
            throw runtime_error("formal post-attention LayerNorm residual decode mismatch");
        }
        ffn_residual_share = feature_major_to_token_major(
            ffn_residual_column, ln1_tokens, ln1_features,
            "ln1_affine_residual_to_mpc");
        cout << "FORMAL_LAYOUT_ADAPTER stage=ln1_affine_residual_to_mpc"
             << " status=PASS source=feature_token_column_major"
             << " target=token_feature_row_major elements="
             << ffn_residual_share.size
             << " communication=0 public_share_permutation=1" << endl;
        ffn_residual_ready = true;
    }
    vector<vector<HE_CKKS::scalar_t>> vec_pre_gelu(
        use_bert_half_packing ? pre_gelu_pts_half.size() : pre_gelu_pts.size());
    vector<vector<HE_CKKS::scalar_t>> vec_output(
        use_bert_half_packing ? output_pts_half.size() : output_pts.size());
    start = high_resolution_clock::now();
    if (use_bert_half_packing) {
        vec_pre_gelu = ckks_to_mpc(
            pre_gelu_pts_half, poly_modulus_degree,
            he->get_Q(pre_gelu_pts_half[0]), bitwidth, s, he, true);
        vec_output = ckks_to_mpc(
            output_pts_half, poly_modulus_degree,
            he->get_Q(output_pts_half[0]), bitwidth, s, he, true);
    } else {
        vec_pre_gelu = ckks_to_mpc(
            pre_gelu_pts, poly_modulus_degree,
            he->get_Q(pre_gelu_pts[0]), bitwidth, s, he, false);
        vec_output = ckks_to_mpc(
            output_pts, poly_modulus_degree,
            he->get_Q(output_pts[0]), bitwidth, s, he, false);
    }
    
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    // cout << "mpc_to_ckks comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks round:" << this->get_round() << endl;
    cout << "bitwidth,s:" << bitwidth << "," << s << endl;
    FixArray pre_gelu_packed(
        party, gelu_input_size, true, bitwidth, s);
    FixArray gelu_poly_packed(
        party, gelu_input_size, true, bitwidth, s);
    cout << "gelu_input_size:" << gelu_input_size << endl;
    cout << "vec_pre_gelu.size()*vec_pre_gelu[0].size():"
         << vec_pre_gelu.size()*vec_pre_gelu[0].size() << endl;
    cout << "vec_output.size()*vec_output[0].size():" << vec_output.size()*vec_output[0].size() << endl;
    for(int i=0;i<vec_pre_gelu.size();i++){
        for(int j=0;j<vec_pre_gelu[i].size();j++){
            pre_gelu_packed.data[i*vec_pre_gelu[i].size()+j] =
                vec_pre_gelu[i][j];
        }
    }
    for(int i=0;i<vec_output.size();i++){
        for(int j=0;j<vec_output[i].size();j++){
            gelu_poly_packed.data[i*vec_output[i].size()+j] =
                vec_output[i][j];
        }
    }
    x_array = bsgs_output_to_token_major(
        pre_gelu_packed, linear.data_lin3, "ffn1_pre_gelu_to_mpc");
    gelu_poly_flat = bsgs_output_to_token_major(
        gelu_poly_packed, linear.data_lin3, "ffn1_gelu_to_mpc");
    cout << "FORMAL_LAYOUT_ADAPTER stage=ffn1_pre_gelu_to_mpc"
         << " status=PASS source=bsgs2_split_feature_halves"
         << " target=token_feature_row_major elements=" << x_array.size
         << " communication=0 public_share_permutation=1" << endl;
    cout << "FORMAL_LAYOUT_ADAPTER stage=ffn1_gelu_to_mpc"
         << " status=PASS source=bsgs2_split_feature_halves"
         << " target=token_feature_row_major elements="
         << gelu_poly_flat.size
         << " communication=0 public_share_permutation=1" << endl;
    if (x_array.ell != bitwidth || x_array.s != s ||
        x_array.size != gelu_input_size) {
        throw runtime_error("formal pre-GeLU MPC metadata mismatch");
    }
    cout << "FORMAL_PRE_GELU_EXPORT status=PASS"
         << " source=real_linear3_output"
         << " masked_algorithm1=1 output_ell=" << x_array.ell
         << " output_scale=" << x_array.s
         << " elements=" << x_array.size
         << " plaintext_recovery=0 constant_placeholder=0" << endl;
    gelu_poly_flat.ell = bitwidth;
    gelu_poly_flat.s = s;
    cout << "gelu_poly_flat, bitwidth,s:" << gelu_poly_flat.ell << "," << gelu_poly_flat.s << endl;
    result["gelu_poly_flat"] = gelu_poly_flat;
    result["x_array"] = x_array;
    cout << "ln_part2_linear3_gelu_part1" << endl;
    return result;
}


/*
    input 维度是 (L,4096)，linear4后降维至 (L,1024)
*/
std::map<char*,FixArray> Bert_ckks::linear4_add_layernorm_part1(FixArray input, int layer_id){
    cout << "in linear4_add_layernorm_part1" << endl;
    FCMetadata data = linear.data_lin4;
    // The shared routine has a pre-norm ViT mode.  Figure 10's half-slot
    // boundary belongs only to post-norm BERT; preserve ViT's full-slot
    // transfer path here.
    const bool use_bert_half_packing = !pre_norm_ffn_output_only;
    HE_CKKS* he = linear.he4;
    int bitwidth,s;
    bitwidth = input.ell;
    s = input.s;
    int gelu_input_size = data.image_size*COMMON_DIM*4;
    int poly_modulus_degree = he->poly_modulus_degree;
    int slot = poly_modulus_degree / 2;
    int num_cts_gelu = gelu_input_size / slot;
    int num_cts_ln = data.image_size*COMMON_DIM / slot;
    if (!ffn_residual_ready || ffn_residual_share.size != data.image_size * COMMON_DIM) {
        throw runtime_error("formal FFN2 refused: real post-attention residual is unavailable");
    }
    if (ffn_residual_share.ell != bitwidth ||
        ffn_residual_share.s != s) {
        throw runtime_error(
            "formal FFN2 residual metadata must match the paper nonlinear ell/s input");
    }
    // The MPC GeLU output is token-major.  linear_2 consumes the same simple
    // feature-major layout produced by Bert::run's
    // plain_col_packing_preprocess.  Its output is the distinct BSGS
    // split-half layout, so prepare the row-major residual for that output
    // layout rather than adding unlike slot coordinates.
    FixArray h6_column = token_major_to_feature_major(
        input, data.image_size, data.filter_h, "gelu_to_ffn2");
    FixArray residual_packed = token_major_to_bsgs_output(
        ffn_residual_share, data, "ffn2_residual_to_h7");
    cout << "FORMAL_LAYOUT_ADAPTER stage=gelu_to_ffn2"
         << " status=PASS source=token_feature_row_major"
         << " target=feature_token_column_major elements=" << h6_column.size
         << " communication=0 public_share_permutation=1" << endl;
    cout << "FORMAL_LAYOUT_ADAPTER stage=ffn2_residual_to_h7"
         << " status=PASS source=token_feature_row_major"
         << " target=bsgs2_split_feature_halves elements="
         << residual_packed.size
         << " communication=0 public_share_permutation=1" << endl;
    vector<vector<HE_CKKS::scalar_t>> h6(num_cts_gelu);
    vector<vector<HE_CKKS::scalar_t>> residual_share(num_cts_ln);
    vector<UnifiedPlaintext> h6_pts;
    cout << num_cts_gelu*slot << endl;
    for(int i=0;i<num_cts_gelu;i++){
        for(int j=0;j<slot;j++){
            h6[i].push_back(h6_column.data[i*slot+j]);
        }
    }
    for (int i = 0; i < num_cts_ln; ++i) {
        residual_share[i].assign(
            residual_packed.data + i * slot,
            residual_packed.data + (i + 1) * slot);
    }
    auto start = high_resolution_clock::now();
    import_target_scale(*this, h6, slot, bitwidth, s, he->he_scale,
                            he, h6_pts, "ffn2_gelu_input");
    vector<UnifiedPlaintext> residual_pts;
    import_target_scale(*this, residual_share, slot, bitwidth, s,
                            he->he_scale, he, residual_pts,
                            "ffn2_residual_input");
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    
    vector<Plaintext> x_avg_pts(num_cts_ln);
    vector<Plaintext> x_avg_pts_half(num_cts_ln % 2 == 0 ? num_cts_ln / 2 : 0);
    vector<Plaintext> sigma_pts(1);
    vector<Ciphertext> enc_x_avg(num_cts_ln);
    vector<Ciphertext> enc_x_avg_half(num_cts_ln % 2 == 0 ? num_cts_ln / 2 : 0);
    vector<Ciphertext> enc_sigma(1);
    vector<UnifiedCiphertext> enc_h6(num_cts_gelu,
                                     UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> enc_h6_half(
        num_cts_gelu % 2 == 0 ? num_cts_gelu / 2 : 0,
        UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> enc_residual(num_cts_ln,
                                           UnifiedCiphertext(LOCATION::HOST));
    if (party==BOB){
        if (use_bert_half_packing) {
            if (enc_h6_half.empty()) {
                throw runtime_error("formal FFN2 input half-packing requires an even ciphertext count");
            }
            #pragma omp parallel for
            for(int i=0;i<enc_h6_half.size();i++){
                he->encryptor->encrypt(h6_pts[i], enc_h6_half[i]);
            }
            send_encrypted_vector(io, enc_h6_half);
            cout << "FORMAL_TRANSFER_LAYOUT stage=ffn2_input status=PASS"
                 << " layout=xyj_half_slot h6_cts=" << enc_h6_half.size()
                 << " residual_cts=" << num_cts_ln
                 << " imag_repack=1 source=Figure10_Block1" << endl;
        } else {
            #pragma omp parallel for
            for(int i=0;i<enc_h6.size();i++){
                he->encryptor->encrypt(h6_pts[i], enc_h6[i]);
            }
            send_encrypted_vector(io, enc_h6);
            cout << "FORMAL_TRANSFER_LAYOUT stage=ffn2_input status=PASS"
                 << " layout=ordinary_full_slot h6_cts=" << enc_h6.size()
                 << " residual_cts=" << num_cts_ln
                 << " imag_repack=0 source=ViT_original_path" << endl;
        }
        #pragma omp parallel for
        for (int i = 0; i < num_cts_ln; ++i) {
            he->encryptor->encrypt(residual_pts[i], enc_residual[i]);
        }
        send_encrypted_vector(io, enc_residual);
        cout << "send_encrypted_vector done" << endl;
        if (!pre_norm_ffn_output_only && !enc_x_avg_half.empty()) {
            recv_encrypted_vector(he->context, io, enc_x_avg_half);
        } else {
            recv_encrypted_vector(he->context, io, enc_x_avg);
        }
        if (!pre_norm_ffn_output_only) {
            recv_encrypted_vector(he->context, io, enc_sigma);
        }
        if (!pre_norm_ffn_output_only && !enc_x_avg_half.empty()) {
            for(int i=0;i<enc_x_avg_half.size();i++){
                he->decryptor->decrypt(enc_x_avg_half[i], x_avg_pts_half[i]);
            }
        } else {
            for(int i=0;i<enc_x_avg.size();i++){
                he->decryptor->decrypt(enc_x_avg[i], x_avg_pts[i]);
            }
        }
        if (!pre_norm_ffn_output_only) {
            he->decryptor->decrypt(enc_sigma[0], sigma_pts[0]);
        }
    }
    else{
        LOCATION loc = he->backend();

        start = high_resolution_clock::now();
        if (use_bert_half_packing) {
            if (enc_h6_half.empty()) {
                throw runtime_error("formal FFN2 input half-packing requires an even ciphertext count");
            }
            recv_encrypted_vector(he->context, io, enc_h6_half);
            #pragma omp parallel for
            for (size_t i = 0; i < enc_h6_half.size(); ++i) {
                he->imag_repack(enc_h6_half[i], enc_h6[i],
                                enc_h6[i + enc_h6_half.size()]);
            }
        } else {
            recv_encrypted_vector(he->context, io, enc_h6);
        }
        recv_encrypted_vector(he->context, io, enc_residual);
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        
        NVTX_PUSH("L4-LN1", true);
        if (he->is_gpu_enable()) {
        // Load revice ciphertexts to GPU
        start = high_resolution_clock::now();
        for_each_n(enc_h6.begin(), enc_h6.size(),
                [he](auto &ct) { ct.to_device(*he->context, *he->dcontext); });
        for_each_n(enc_residual.begin(), enc_residual.size(),
                [he](auto &ct) { ct.to_device(*he->context, *he->dcontext); });
        for_each_n(h6_pts.begin(), h6_pts.size(),
                [he](auto &pt) { pt.to_device(*he->context, *he->dcontext); });
        for_each_n(residual_pts.begin(), residual_pts.size(),
                [he](auto &pt) { pt.to_device(*he->context, *he->dcontext); });
        this->ct_transfer_mem_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        }

        start = high_resolution_clock::now();
        cout << "FORMAL_TRANSFER_LAYOUT stage=ffn2_input status=PASS"
             << " layout=" << (use_bert_half_packing ? "xyj_half_slot" : "ordinary_full_slot")
             << " h6_cts=" << (use_bert_half_packing ? enc_h6_half.size() : enc_h6.size())
             << " residual_cts=" << enc_residual.size()
             << " imag_repack=" << (use_bert_half_packing ? 1 : 0)
             << " source=" << (use_bert_half_packing ? "Figure10_Block1" : "ViT_original_path") << endl;
        cout << "num of input cts:" << enc_h6.size() << endl;
        cout << "recv_encrypted_vector" << endl;

        if (!he->is_gpu_enable()) {
            #pragma omp parallel for num_threads(4)
            for(int i=0;i<enc_h6.size();i++){
                he->evaluator->add_plain_inplace(enc_h6[i], h6_pts[i]);
            }
            for (int i = 0; i < enc_residual.size(); ++i) {
                he->evaluator->add_plain_inplace(enc_residual[i], residual_pts[i]);
            }
        } else {
            for(int i=0;i < enc_h6.size();i++){
                he->devaluator->add_plain_inplace(enc_h6[i], h6_pts[i]);
            }
            for (int i = 0; i < enc_residual.size(); ++i) {
                he->devaluator->add_plain_inplace(enc_residual[i],
                                                  residual_pts[i]);
            }
        }
        vector<UnifiedCiphertext> h7;
        const auto linear4_profile_start = high_resolution_clock::now();
        linear.linear_2(
            he,
            enc_h6, 
            linear.pp_4[layer_id],
            data,
            h7
        );
        if (profile_enabled_)
            profile_record_he(
                BlbViTProfileStage::FFN2,
                chrono::duration<double, milli>(high_resolution_clock::now() -
                                                linear4_profile_start)
                    .count());
        cout << "h7.size():" << h7.size() << ", should be:" << num_cts_ln << endl;
        cout << "scale:" << log2(h7[0].scale()) << endl;

        if (h7.size() != enc_residual.size()) {
            throw runtime_error("formal FFN2 residual ciphertext count mismatch");
        }
        require_uniform_unified_metadata(enc_h6, "ffn2_gelu_input");
        require_uniform_unified_metadata(enc_residual,
                                         "ffn2_residual_input");
        const double imported_scale = ldexp(1.0, he->he_scale);
        if (abs(enc_h6.front().scale() - imported_scale) / imported_scale >
                1e-12 ||
            abs(enc_residual.front().scale() - imported_scale) /
                    imported_scale >
                1e-12) {
            throw runtime_error(
                "formal FFN2 high-scale Algorithm-2 import was not preserved");
        }
        align_unified_to_cipher_metadata(linear, he, enc_residual, h7,
                                         "ffn2_residual");

        // Encode weight once
        int L = data.image_size;
        vector<double> inv_D(slot, 1.0/768);
        vector<double> all_1(slot, 1);
        vector<double> mask(slot,0);
        UnifiedPlaintext inv_D_p(loc);
        UnifiedPlaintext inv_D_p2(loc);
        UnifiedPlaintext mask_p(loc);
        UnifiedPlaintext scale_up(loc);
        for(int i=0;i<L;i++){
            mask[i] = 1;
        }
        uint64_t centered_drop_modulus = 0;
        if (!he->is_gpu_enable()) {
            auto centered_context = he->context->get_context_data(
                h7[0].hcipher().parms_id());
            if (!centered_context ||
                centered_context->parms().coeff_modulus().empty()) {
                throw runtime_error(
                    "formal FFN2 centered scale level is unavailable");
            }
            centered_drop_modulus =
                centered_context->parms().coeff_modulus().back().value();
        } else {
            const auto centered_chain = h7[0].dcipher().chain_index();
            const auto &centered_context =
                he->dcontext->get_context_data(centered_chain);
            centered_drop_modulus =
                centered_context.parms().coeff_modulus().back().value();
        }
        const double centered_target_scale =
            ldexp(1.0, kFormalConversionScaleBits);
        const double centered_plain_scale =
            centered_target_scale *
            static_cast<double>(centered_drop_modulus) /
            h7.front().scale();
        if (!isfinite(centered_plain_scale) || centered_plain_scale < 1.0) {
            throw runtime_error(
                "formal FFN2 centered public scale is not encodable");
        }
        if (!he->is_gpu_enable()) {
            he->encoder->encode(inv_D, h7[0].hcipher().parms_id(),
                                centered_plain_scale, inv_D_p);
            he->encoder->encode(all_1, h7[0].hcipher().parms_id(),
                                centered_plain_scale, scale_up);
        } else {
            const auto centered_chain = h7[0].dcipher().chain_index();
            he->dencoder->encode(*he->dcontext, inv_D,
                                 centered_plain_scale, inv_D_p,
                                 centered_chain);
            he->dencoder->encode(*he->dcontext, all_1,
                                 centered_plain_scale, scale_up,
                                 centered_chain);
        }
        cout << "FORMAL_CENTERED_SCALE_PLAN stage=ln2 status=PASS"
             << " source_scale_bits=" << log2(h7.front().scale())
             << " mean_plain_scale_bits=" << log2(centered_plain_scale)
             << " centered_drop_bits="
             << log2(static_cast<double>(centered_drop_modulus))
             << " target_scale_bits=" << kFormalConversionScaleBits
             << endl;

        // begin computation
        for(int i=0;i<h7.size();i++){
            if (!he->is_gpu_enable()) {
            he->evaluator->add_inplace(h7[i], enc_residual[i]);
            } else {
            he->devaluator->add_inplace(h7[i], enc_residual[i]);
            }
        }
        ffn_residual_ready = false;
        cout << "OK1" << endl;
        if (pre_norm_ffn_output_only) {
            // ViT's second residual is h7 = FFN2(GELU(FFN1(LN2(h3)))) + h3.
            // Export h7 at the same original Algorithm-1 boundary and leave
            // the final LayerNorm to the ViT driver.
            vector<Ciphertext> h7_full(h7.size());
            for (size_t i = 0; i < h7.size(); ++i) {
                if (he->is_gpu_enable()) {
                    h7[i].to_host(*he->context, *he->dcontext);
                }
                h7_full[i] = std::move(h7[i].hcipher());
            }
            prepare_host_conversion_from_actual_scale(
                *this, h7_full, kFormalConversionScaleBits, he,
                "vit_ffn_residual_output");
            mask_ckks_to_mpc_inplace(h7_full, x_avg_pts, he);
            send_encrypted_vector(io, h7_full);
            cout << "FORMAL_TRANSFER_LAYOUT stage=vit_ffn_residual"
                 << " status=PASS layout=bsgs2_to_full_slot cts="
                 << h7_full.size() << " sigma_cts=0" << endl;
        } else {
        if (profile_enabled_)
            profile_switch(BlbViTProfileStage::LAYERNORM2);
        // 开始计算layernorm
        // 1. X_sum
        vector<UnifiedCiphertext> X_avg(h7.size(), UnifiedCiphertext(loc));
        vector<UnifiedCiphertext> sigma(num_cts_ln, UnifiedCiphertext(loc));
         #pragma omp parallel for
        for(int i=0;i<h7.size();i++){
            X_avg[i] = h7[i];
        }
        for(int i=1;i<=std::log2(slot/L);i++){
            if (!he->is_gpu_enable()) {
            #pragma omp parallel for num_threads(4)
                for(int j=0;j<h7.size();j++){
                    Ciphertext tmp;
                    he->evaluator->rotate_vector(X_avg[j],pow(2,i-1)*L,*he->gal_keys,tmp);
                    he->evaluator->add_inplace(X_avg[j],tmp);
                }
            } else {
                for(int j=0;j<h7.size();j++){
                    UnifiedCiphertext tmp(loc);
                    he->devaluator->rotate_vector(X_avg[j], pow(2,i-1)*L, *he->dgal_keys, tmp);
                    he->devaluator->add_inplace(X_avg[j],tmp);
                }
            }
        }
        cout << "OK1" << endl;
        UnifiedCiphertext X_sum = X_avg[0];
        for(int i=1;i<num_cts_ln;i++){
            if (!he->is_gpu_enable()) {
            he->evaluator->add_inplace(X_sum,X_avg[i]);
            } else {
            he->devaluator->add_inplace(X_sum, X_avg[i]);
            }
        }
        // #pragma omp parallel for num_threads(4)
        for(int i=0;i<num_cts_ln;i++){
            X_avg[i] = X_sum;
        }
        cout << "OK2" << endl;
        // 2. X_avg, scale=s*3
        #pragma omp parallel for
        for(int i=0;i<num_cts_ln;i++){
            if (!he->is_gpu_enable()) {
                he->evaluator->multiply_plain_inplace(X_avg[i], inv_D_p);
            } else {
                he->devaluator->multiply_plain_inplace(X_avg[i], inv_D_p);
            }
        }
        cout << "OK3" << endl;
        /*
           3. scale up h7 from s*2 to s*3
           4. h7 = h7-X_avg
           scale = 3*s-60, Q=[60, 34, 60, 60]
        */ 
        cout << "h7.scale:" << log2(h7[0].scale()) << endl;
        cout << "X_avg.scale:" << log2(X_avg[0].scale()) << endl;
        #pragma omp parallel for
        for(int i=0;i<num_cts_ln;i++){
            if (!he->is_gpu_enable()) {
            he->evaluator->multiply_plain_inplace(h7[i],scale_up);
            he->evaluator->sub_inplace(h7[i],X_avg[i]);
            ++this->he_rescale_count;
            he->evaluator->rescale_to_next_inplace(h7[i]);
            } else {
            he->devaluator->multiply_plain_inplace(h7[i],scale_up);
            he->devaluator->sub_inplace(h7[i],X_avg[i]);
            ++this->he_rescale_count;
            he->devaluator->rescale_to_next_inplace(h7[i]);
            }
        }
        cout << "OK4" << endl;
        /*
            compute X_sum2=sigma^2, 注意sigma^2维度需要mask成(L,1)，为了降低后面sqrt维度
            同时将h7 转化为 scale: 20, Q: 60 by 2*ms, 1*rs
        */ 
        cout << "h7.scale:" << log2(h7[0].scale()) << endl;
        // #pragma omp parallel for
        for(int i=0;i<num_cts_ln;i++){
            sigma[i] = h7[i];
        }
        // The centered subtraction above has already consumed the one
        // published LayerNorm rescale and is planned directly at the common
        // conversion scale.  The final Algorithm-1 boundary performs only
        // local modulus switching to the last prime; no extra rescale is
        // inserted here.
        const size_t centered_input_limbs = h7.front().coeff_modulus_size();
        if (centered_input_limbs < 3) {
            throw runtime_error(
                "formal LN2 centered path has insufficient Table-6 levels");
        }
        const int centered_mod_switches = 0;
        cout << "FORMAL_LN2_LEVEL_PLAN status=PASS input_limbs="
             << centered_input_limbs
             << " explicit_mod_switches=" << centered_mod_switches
             << " final_rescale=0" << endl;
        #pragma omp parallel for
        for(int i=0;i<num_cts_ln;i++){
            if (!he->is_gpu_enable()) {
            he->evaluator->square_inplace(sigma[i]);
            he->evaluator->relinearize_inplace(sigma[i], *he->relin_keys);
            for (int step = 0; step < centered_mod_switches; ++step) {
                he->evaluator->mod_switch_to_next_inplace(h7[i]);
            }
            } else {
            he->devaluator->square_inplace(sigma[i]);
            he->devaluator->relinearize_inplace(sigma[i], *he->drelin_keys);
            for (int step = 0; step < centered_mod_switches; ++step) {
                he->devaluator->mod_switch_to_next_inplace(h7[i]);
            }
            }
        }
        require_uniform_unified_metadata(h7, "ln2_centered_output");
        const double centered_scale_error =
            abs(h7.front().scale() - centered_target_scale) /
            centered_target_scale;
        if (h7.front().coeff_modulus_size() < 1 ||
            !isfinite(centered_scale_error) || centered_scale_error > 1e-9) {
            throw runtime_error(
                "formal LN2 centered output did not reach high-scale final prime");
        }
        cout << "FORMAL_CENTERED_SCALE_READY stage=ln2 status=PASS"
             << " actual_scale_bits=" << log2(h7.front().scale())
             << " limbs=" << h7.front().coeff_modulus_size() << endl;
        cout << "OK5" << endl;
        for(int i=1;i<=std::log2(slot/L);i++){
            if (!he->is_gpu_enable()) {
            #pragma omp parallel for num_threads(4)
            for(int j=0;j<num_cts_ln;j++){
                Ciphertext tmp;
                he->evaluator->rotate_vector(sigma[j],pow(2,i-1)*L,*he->gal_keys,tmp);
                he->evaluator->add_inplace(sigma[j],tmp);
            }
            } else {
            for(int j=0;j<num_cts_ln;j++){
                UnifiedCiphertext tmp(loc);
                he->devaluator->rotate_vector(sigma[j], pow(2,i-1)*L, *he->dgal_keys, tmp);
                he->devaluator->add_inplace(sigma[j], tmp);
            }
            }
        }
        cout << "OK5" << endl;
        UnifiedCiphertext X_sum2 = std::move(sigma[0]);
        for(int i=1;i<num_cts_ln;i++){
            if (!he->is_gpu_enable()) {
            he->evaluator->add_inplace(X_sum2,sigma[i]);
            } else {
            he->devaluator->add_inplace(X_sum2, sigma[i]);
            }
        }
        cout << "OK5" << endl;
        
        const double variance_source_scale = X_sum2.scale();
        const size_t variance_source_limbs = X_sum2.coeff_modulus_size();
        uint64_t variance_drop_modulus_1 = 0;
        uint64_t variance_drop_modulus_2 = 0;
        if (!he->is_gpu_enable()) {
            auto variance_context = he->context->get_context_data(
                X_sum2.hcipher().parms_id());
            if (!variance_context ||
                variance_context->parms().coeff_modulus().size() < 3) {
                throw runtime_error(
                    "formal LN2 variance requires two rescale levels");
            }
            variance_drop_modulus_1 =
                variance_context->parms().coeff_modulus().back().value();
            variance_context = variance_context->next_context_data();
            if (!variance_context) {
                throw runtime_error(
                    "formal LN2 second variance rescale level is unavailable");
            }
            variance_drop_modulus_2 =
                variance_context->parms().coeff_modulus().back().value();
        } else {
            const auto variance_chain = X_sum2.dcipher().chain_index();
            const size_t variance_limbs = X_sum2.coeff_modulus_size();
            size_t variance_context_index = 0;
            bool found_variance_context = false;
            for (size_t i = 0; i < he->dcontext->context_data_.size(); ++i) {
                if (he->dcontext->context_data_[i]
                        .parms().coeff_modulus().size() == variance_limbs) {
                    variance_context_index = i;
                    found_variance_context = true;
                    break;
                }
            }
            if (!found_variance_context || variance_limbs < 3 ||
                variance_context_index + 1 >=
                    he->dcontext->context_data_.size()) {
                cout << "FORMAL_VARIANCE_CONTEXT_LOOKUP stage=ln2"
                     << " status=BLOCKED ciphertext_chain=" << variance_chain
                     << " ciphertext_limbs=" << variance_limbs
                     << " context_index=" << variance_context_index << endl;
                throw runtime_error(
                    "formal LN2 DEVICE variance requires two rescale levels");
            }
            const auto &variance_context_1 =
                he->dcontext->get_context_data(variance_context_index);
            const auto &variance_context_2 =
                he->dcontext->get_context_data(variance_context_index + 1);
            cout << "FORMAL_VARIANCE_CONTEXT_LOOKUP stage=ln2"
                 << " status=PASS"
                 << " ciphertext_chain=" << variance_chain
                 << " ciphertext_limbs=" << variance_limbs
                 << " context_index=" << variance_context_index
                 << " next_context_limbs="
                 << variance_context_2.parms().coeff_modulus().size() << endl;
            if (variance_context_1.parms().coeff_modulus().size() < 3 ||
                variance_context_2.parms().coeff_modulus().size() + 1 !=
                    variance_limbs) {
                throw runtime_error(
                    "formal LN2 DEVICE variance context chain is not contiguous");
            }
            variance_drop_modulus_1 = variance_context_1.parms()
                                                  .coeff_modulus()
                                                  .back()
                                                  .value();
            variance_drop_modulus_2 = variance_context_2.parms()
                                                  .coeff_modulus()
                                                  .back()
                                                  .value();
        }
        const double variance_target_scale =
            ldexp(1.0, kFormalConversionScaleBits);
        const double variance_plain_scale_product =
            variance_target_scale *
            static_cast<double>(variance_drop_modulus_1) *
            static_cast<double>(variance_drop_modulus_2) /
            variance_source_scale;
        const double variance_mask_scale =
            sqrt(variance_plain_scale_product);
        const double variance_inv_d_scale =
            variance_plain_scale_product / variance_mask_scale;
        if (!isfinite(variance_mask_scale) ||
            !isfinite(variance_inv_d_scale) || variance_mask_scale < 1.0 ||
            variance_inv_d_scale < 1.0) {
            throw runtime_error(
                "formal LN2 variance public scale is not encodable");
        }
        if (!he->is_gpu_enable()) {
            const auto parms_id = X_sum2.hcipher().parms_id();
            he->encoder->encode(mask, parms_id, variance_mask_scale, mask_p);
            he->encoder->encode(inv_D, parms_id, variance_inv_d_scale,
                                inv_D_p2);
            he->evaluator->multiply_plain_inplace(X_sum2,mask_p);
            he->evaluator->multiply_plain_inplace(X_sum2,inv_D_p2);
            ++this->he_rescale_count;
            he->evaluator->rescale_to_next_inplace(X_sum2);
            ++this->he_rescale_count;
            he->evaluator->rescale_to_next_inplace(X_sum2);
        } else {
            const auto chain_index = X_sum2.dcipher().chain_index();
            he->dencoder->encode(*he->dcontext, mask, variance_mask_scale,
                                 mask_p, chain_index);
            he->dencoder->encode(*he->dcontext, inv_D,
                                 variance_inv_d_scale, inv_D_p2,
                                 chain_index);
            he->devaluator->multiply_plain_inplace(X_sum2,mask_p);
            he->devaluator->multiply_plain_inplace(X_sum2,inv_D_p2);
            ++this->he_rescale_count;
            he->devaluator->rescale_to_next_inplace(X_sum2);
            ++this->he_rescale_count;
            he->devaluator->rescale_to_next_inplace(X_sum2);
        }
        const double variance_scale_error =
            abs(X_sum2.scale() - variance_target_scale) /
            variance_target_scale;
        if (X_sum2.coeff_modulus_size() != variance_source_limbs - 2 ||
            !isfinite(variance_scale_error) || variance_scale_error > 1e-9) {
            throw runtime_error(
                "formal LN2 variance did not reach its high conversion scale");
        }
        cout << "FORMAL_VARIANCE_SCALE_PLAN stage=ln2 status=PASS"
             << " source_scale_bits=" << log2(variance_source_scale)
             << " drop1_bits="
             << log2(static_cast<double>(variance_drop_modulus_1))
             << " drop2_bits="
             << log2(static_cast<double>(variance_drop_modulus_2))
             << " mask_scale_bits=" << log2(variance_mask_scale)
             << " inv_d_scale_bits=" << log2(variance_inv_d_scale)
             << " target_scale_bits=" << kFormalConversionScaleBits
             << " output_limbs=" << X_sum2.coeff_modulus_size() << endl;
        // 6. send cts to client, and ckks to mpc
        this->he_compute_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        
        if (he->is_gpu_enable()) {
        // Load result ciphertexts to CPU
        start = high_resolution_clock::now();
        X_sum2.to_host(*he->context, *he->dcontext);
        for(size_t i = 0;i < h7.size();i ++){
            h7[i].to_host(*he->context, *he->dcontext);
        }
        this->ct_transfer_mem_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        }
        NVTX_POP(true);

        start = high_resolution_clock::now();
        cout << "num of output cts:" << sigma.size()+1 << endl;
        cout << "bitwidth of h7 ct:" << h7[0].coeff_modulus_size() << endl;
        cout << "bitwidth of X_sum2 ct:" << X_sum2.coeff_modulus_size() << endl;
        vector<Ciphertext> enc_sigma(1);
        enc_sigma[0] = std::move(X_sum2);
        cout << "h7.scale:" << log2(h7[0].scale()) << endl;
        cout << "enc_sigma.scale:" << log2(enc_sigma[0].scale()) << endl;
        vector<Ciphertext> h7_full(h7.size());
        if (h7_full.size() != static_cast<size_t>(num_cts_ln)) {
            throw runtime_error("formal FFN2 LayerNorm ciphertext count mismatch");
        }
        for(int i=0;i<h7_full.size();i++){
            h7_full[i] = std::move(h7[i].hcipher());
        }
        prepare_host_conversion_from_actual_scale(
            *this, h7_full, kFormalConversionScaleBits, he,
            "ln2_centered");
        prepare_host_conversion_from_actual_scale(
            *this, enc_sigma, kFormalConversionScaleBits, he,
            "ln2_variance");
        if (num_cts_ln % 2 != 0) {
            throw runtime_error("formal FFN2 LayerNorm output half-packing requires an even ciphertext count");
        }
        vector<Ciphertext> h7_half(num_cts_ln / 2);
        for (size_t i = 0; i < h7_half.size(); ++i) {
            h7_half[i] = std::move(h7_full[i]);
        }
        vector<Plaintext> x_avg_pts_half_send;
        mask_ckks_to_mpc_inplace(h7_half, x_avg_pts_half_send, he);
        mask_ckks_to_mpc_inplace(enc_sigma, sigma_pts, he);
        send_encrypted_vector(io, h7_half);
        x_avg_pts_half = std::move(x_avg_pts_half_send);
        send_encrypted_vector(io, enc_sigma);
        cout << "FORMAL_TRANSFER_LAYOUT stage=ffn2_ln_output status=PASS"
             << " layout=xyj_half_slot h7_cts=" << h7_half.size()
             << " sigma_cts=1 half_decode=1 source=Figure10_Block1" << endl;
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        }
    }
    // cout << "send and receive cts, comm:" << this->get_comm() << endl;
    // cout << "send and receive cts, round:" << this->get_round() << endl;
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    cout << "send and receive done in linear4_add_layernorm_part1" << endl;
    if (pre_norm_ffn_output_only) {
        vector<vector<HE_CKKS::scalar_t>> residual_vector = ckks_to_mpc(
            x_avg_pts, poly_modulus_degree,
            he->get_Q(x_avg_pts[0]), bitwidth, s, he, false);
        FixArray packed(party, num_cts_ln * slot, true, bitwidth, s);
        size_t offset = 0;
        for (const auto &part : residual_vector) {
            if (part.size() != static_cast<size_t>(slot) ||
                offset + part.size() > static_cast<size_t>(packed.size)) {
                throw runtime_error("formal ViT FFN residual decode mismatch");
            }
            copy(part.begin(), part.end(), packed.data + offset);
            offset += part.size();
        }
        if (offset != static_cast<size_t>(packed.size)) {
            throw runtime_error("formal ViT FFN residual is incomplete");
        }
        map<char*,FixArray> result;
        result["residual"] = bsgs_output_to_token_major(
            packed, data, "vit_ffn_residual_to_token");
        this->conversion_comm += this->get_comm();
        this->conversion_round += this->get_round();
        return result;
    }
    vector<vector<HE_CKKS::scalar_t>> vec_x_avg(x_avg_pts.size());
    vector<vector<HE_CKKS::scalar_t>> vec_sigma(sigma_pts.size());
    start = high_resolution_clock::now();
    if (!x_avg_pts_half.empty()) {
        vec_x_avg = ckks_to_mpc(x_avg_pts_half, poly_modulus_degree,
                                he->get_Q(x_avg_pts_half[0]), bitwidth, s,
                                he, true);
    } else {
        vec_x_avg = ckks_to_mpc(x_avg_pts, poly_modulus_degree,
                                he->get_Q(x_avg_pts[0]), bitwidth, s, he,
                                false);
    }
    vec_sigma = ckks_to_mpc(sigma_pts, poly_modulus_degree,
                            he->get_Q(sigma_pts[0]), bitwidth, s, he, false);
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    cout << "vec_sigma.size():" << vec_sigma.size() << ", should be 1" << endl;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    // cout << "mpc_to_ckks comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks round:" << this->get_round() << endl;
    cout << "bitwidth,s:" << bitwidth << "," << s << endl;
    FixArray x_flat_avg, sigma_flat;
    FixArray x_flat_avg_packed(
        party, num_cts_ln * slot, true, bitwidth, s);
    sigma_flat = FixArray(
        party, data.image_size, true, bitwidth, s);
    for(int i=0;i<vec_x_avg.size();i++){
        for(int j=0;j<vec_x_avg[i].size();j++){
            x_flat_avg_packed.data[i*vec_x_avg[i].size()+j] =
                vec_x_avg[i][j];
        }
    }
    x_flat_avg = bsgs_output_to_token_major(
        x_flat_avg_packed, data, "h7_to_ln2_mpc");
    cout << "FORMAL_LAYOUT_ADAPTER stage=h7_to_ln2_mpc"
         << " status=PASS source=bsgs2_split_feature_halves"
         << " target=token_feature_row_major elements=" << x_flat_avg.size
         << " communication=0 public_share_permutation=1" << endl;
    for(int j=0;j<data.image_size;j++){
        sigma_flat.data[j] = vec_sigma[0][j];
    }
    x_flat_avg.ell = bitwidth;
    x_flat_avg.s = s;
    sigma_flat.ell = bitwidth;
    sigma_flat.s = s;
    map<char*,FixArray> result;
    result["x_flat_avg"] = x_flat_avg;
    result["sigma_flat"] = sigma_flat;
    return result;
}
