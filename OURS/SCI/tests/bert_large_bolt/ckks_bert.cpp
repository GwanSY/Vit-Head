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

#define MAX_THREADS 1
#define BOB sci::BOB
// #define SCI_HE

namespace {

// Keep the final CKKS share at high precision through Algorithm 1.  The
// CKKS-to-MPC adapter performs the secure coefficient-domain truncation from
// this scale to each tensor's semantic fixed-point scale (normally s12, V
// uses s11); lowering the ciphertext itself to s12 before decryption would
// amplify CKKS noise in the subsequent fixed FFT.
constexpr int kFormalConversionScaleBits = 40;

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
#ifdef BLB_VIT65_HETERO
    // The heterogeneous pairwise QK path omits A0's per-head extraction
    // rescale, then uses a high-precision 2^52 diagonal mask and two 60-bit
    // rescales. X=70 gives Q/K/V scale 54 and QK scale 40 with three limbs:
    //   QKV: 70 + 44 - 60 = 54
    //   QK: 2*54 + 52 - 60 - 60 = 40.
    if (qk_target_scale_bits == 40) {
        cout << "HETERO_QK_SCALE_PLAN status=PASS input_scale_bits=70"
             << " qkv_scale_bits=54 qk_mask_scale_bits=52"
             << " qk_final_bits=40 target_scale_bits=40"
             << " qk_final_limbs=3" << endl;
        return 70;
    }
#endif
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

void import_rns_target_scale(Bert_ckks &owner,
                             const vector<vector<HE_CKKS::scalar_t>> &shares,
                             int slot, int ell, int input_scale,
                             int target_scale, HE_CKKS *he,
                             vector<UnifiedPlaintext> &destination,
                             const char *stage) {
    vector<Plaintext> host_plaintexts = owner.mpc_to_ckks_rns(
        shares, slot, ell, input_scale, target_scale, he, false);
    const auto first_data = he->context->first_context_data();
    if (host_plaintexts.empty() || !first_data) {
        throw runtime_error(string("formal RNS import is unavailable: ") +
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
            throw runtime_error(string("formal RNS import metadata mismatch: ") +
                                stage);
        }
        destination[i] = std::move(host_plaintexts[i]);
    }
    cout << "FORMAL_RNS_IMPORT stage=" << stage
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
    while (true) {
        const auto context_data =
            he->context->get_context_data(ciphertexts.front().parms_id());
        if (!context_data) {
            throw runtime_error(string("formal conversion level is invalid: ") +
                                stage);
        }
        const auto &moduli = context_data->parms().coeff_modulus();
        if (moduli.size() < 2 ||
            target_scale * static_cast<double>(moduli.back().value()) >=
                ciphertexts.front().scale()) {
            break;
        }
        // Capture the common level before mutating any ciphertext.  Comparing
        // against ciphertexts.front() inside this loop is invalid: rescaling
        // the first ciphertext changes its parms_id before the remaining
        // ciphertexts have been visited.
        const auto common_parms_id = ciphertexts.front().parms_id();
        for (auto &ciphertext : ciphertexts) {
            if (ciphertext.parms_id() != common_parms_id) {
                throw runtime_error(
                    string("formal pre-rescale level mismatch: ") + stage);
            }
            he->evaluator->rescale_to_next_inplace(ciphertext);
        }
        owner.host_rescale_count += ciphertexts.size();
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
        Plaintext unit_plain;
        he->encoder->encode(public_one, ciphertexts.front().parms_id(),
                            unit_scale, unit_plain);
        const auto common_parms_id = ciphertexts.front().parms_id();
        for (auto &ciphertext : ciphertexts) {
            if (ciphertext.parms_id() != common_parms_id) {
                throw runtime_error(
                    string("formal upward conversion level mismatch: ") +
                    stage);
            }
            he->evaluator->multiply_plain_inplace(ciphertext, unit_plain);
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
    std::map<string, FixArray> input, int layer_id){
    // mult. depth = 4
    auto start = high_resolution_clock::now();
    const bool raw_input = input.size() == 1 &&
                           input.begin()->first == "raw_input";
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
    
    if (layer_id < 0 || layer_id >= ATTENTION_LAYERS) {
        throw invalid_argument("attention layer index is out of range");
    }
    FCMetadata data = layer_id == 0 ? linear.data_lin1_0
                                    : linear.data_lin1_1;
    const size_t qk_ct_count = static_cast<size_t>(
        data.image_size * data.image_size * PACKING_NUM / data.slot_count);
    const size_t v_ct_count = static_cast<size_t>(
        data.image_size * data.filter_w * PACKING_NUM / data.slot_count);
    const size_t output_ct_count = qk_ct_count + v_ct_count;
    if (raw_input &&
        (qk_ct_count * data.slot_count !=
             static_cast<size_t>(data.image_size) * data.image_size *
                 PACKING_NUM ||
         v_ct_count * data.slot_count !=
             static_cast<size_t>(data.image_size) * data.filter_w *
                 PACKING_NUM)) {
        throw runtime_error(
            "formal QK/V ciphertext layout is not full-slot exact");
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
#ifdef BLB_VIT65_HETERO
    // Protocol-aware head reorder makes all MPC heads a prefix.  The donor's
    // two-head packing rounds that prefix to an even number of heads; an
    // adjacent linear head may therefore spill into MPC, but no private-input
    // branch is introduced.
    const size_t boundary_qk_ct_count =
        raw_input ? hetero_boundary_qk_ct_count : 0;
    const size_t boundary_v_ct_count = 0;
#else
    const size_t boundary_qk_ct_count = raw_input ? qk_ct_count : 0;
    const size_t boundary_v_ct_count = raw_input ? v_ct_count : 0;
#endif
    vector<Ciphertext> raw_qk_cts(boundary_qk_ct_count);
    vector<Ciphertext> raw_v_cts(boundary_v_ct_count);
    vector<Plaintext> raw_qk_pts(boundary_qk_ct_count);
    vector<Plaintext> raw_v_pts(boundary_v_ct_count);
    cout << this->get_comm() << endl;
    cout << this->get_round() << endl;
    this->he_compute_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    
    start = high_resolution_clock::now();
    // For the post-norm layer-0 entry, main has already applied BERT's real
    // x_scale[0] transition (s12 -> s5) with secure truncate_reduce.  Choose
    // the numerical CKKS input precision by working backwards through the
    // actual QKV/QK rescale graph so QK reaches Algorithm 1 at s40.  The MPC
    // tensor remains an s5 value; this is not a substitute for the secure
    // fixed-point truncation.
    int x_ckks_scale_bits = he->he_scale;
    if (raw_input && s != 5) {
        throw runtime_error(
            "formal layer-0 QKV input must be securely truncated to scale 5");
    }
    if (raw_input) {
        x_ckks_scale_bits = layer0_qkv_import_scale_bits(
            he, kFormalConversionScaleBits);
        cout << "RAW_INPUT_SCALE_ALIGNMENT status=PASS fixed_scale=" << s
             << " ckks_scale_bits=" << x_ckks_scale_bits
             << " qk_conversion_scale_bits="
             << kFormalConversionScaleBits << endl;
    }
    if (raw_input) {
        vector<Plaintext> rns_plaintexts = mpc_to_ckks_rns(
            vec_x_flat_avg, slot_count, bitwidth, s,
            x_ckks_scale_bits, he, false);
        pts_x_flat_avg.resize(rns_plaintexts.size());
        for (size_t i = 0; i < rns_plaintexts.size(); ++i) {
            pts_x_flat_avg[i] = std::move(rns_plaintexts[i]);
        }
    } else {
        mpc_to_ckks(pts_x_flat_avg, vec_x_flat_avg, slot_count, he->get_Q(),
                    bitwidth, s, x_ckks_scale_bits, he);
    }
    // MARK: Can be parallel
    // cout << "mpc_to_ckks1 comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks1 round:" << this->get_round() << endl;
    if (!raw_input) {
        mpc_to_ckks(pts_sigma_flat, vec_sigma_flat, slot_count, he->get_Q(),
                    bitwidth, s, he->he_scale, he);
    }
    // cout << "mpc_to_ckks2 comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks2 round:" << this->get_round() << endl;
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    
    vector<UnifiedCiphertext> cts_x_flat_full(
        raw_input ? pts_x_flat_avg.size() : 0,
        UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> cts_x_flat_half(
        raw_input ? 0 : pts_x_flat_avg.size() / 2,
        UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> cts_sigma_flat_half(
        raw_input ? 0 : pts_sigma_flat.size() / 2,
        UnifiedCiphertext(LOCATION::HOST));
    if (party == BOB) {
        if (raw_input) {
            #pragma omp parallel for
            for (size_t i = 0; i < num_cts; ++i) {
                he->encryptor->encrypt(pts_x_flat_avg[i],
                                       cts_x_flat_full[i]);
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
        if (raw_input) {
            send_encrypted_vector(io, cts_x_flat_full);
            cout << "RAW_INPUT_TRANSFER_LAYOUT status=PASS layout=ordinary_full_slot cts="
                 << cts_x_flat_full.size() << " imag_repack=0" << endl;
        } else {
			send_encrypted_vector(io, cts_x_flat_half);
            send_encrypted_vector(io, cts_sigma_flat_half);
        }
        cout << "send_encrypted_vector done" << endl;
		/*
           TODO: output is also in x+y*j format, thus, decrypt, decode can be reduced by half
        */
		if (raw_input) {
            recv_encrypted_vector(he->context, io, raw_qk_cts);
            if (!raw_v_cts.empty()) {
                recv_encrypted_vector(he->context, io, raw_v_cts);
            }
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(raw_qk_cts.size()); ++i) {
                he->decryptor->decrypt(raw_qk_cts[i], raw_qk_pts[i]);
            }
            if (!raw_v_cts.empty()) {
                #pragma omp parallel for
                for (int i = 0; i < static_cast<int>(raw_v_cts.size()); ++i) {
                    he->decryptor->decrypt(raw_v_cts[i], raw_v_pts[i]);
                }
            }
        } else {
            recv_encrypted_vector(he->context, io, output_cts_half);
            #pragma omp parallel for
            for(int i=0;i<output_cts_half.size();i++){
                he->decryptor->decrypt(output_cts_half[i], output_pts_half[i]);
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
        if (raw_input) {
            recv_encrypted_vector(he->context, io, cts_x_flat_full);
        } else {
            recv_encrypted_vector(he->context, io, cts_x_flat_half);
            recv_encrypted_vector(he->context, io, cts_sigma_flat_half);
        }
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        
        NVTX_PUSH("LN2-ATT", true);
        if (he->is_gpu_enable()) {
            // Load reviced ciphertexts to GPU
            start = high_resolution_clock::now();
            if (raw_input) {
                for_each_n(cts_x_flat_full.begin(), cts_x_flat_full.size(),
                        [he](auto &ct) { ct.to_device(*he->context, *he->dcontext); });
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
        if (raw_input) {
            for (size_t i = 0; i < cts_x_flat_full.size(); ++i) {
                cts_x_flat[i] = std::move(cts_x_flat_full[i]);
            }
            cout << "RAW_INPUT_TRANSFER_LAYOUT status=PASS layout=ordinary_full_slot cts="
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
                    w_matrix[i][j] = layernorm_fixed_to_real(
                        w.data[i*slot_count+j], w, "LayerNorm weight");
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
                        "formal layer-0 high-scale RNS import was not preserved");
                }
                continue;
            }
            if (!he->is_gpu_enable()) {
                he->evaluator->add_plain_inplace(cts_x_flat[i], pts_x_flat_avg[i]);
                he->evaluator->add_plain_inplace(cts_sigma_flat[i], pts_sigma_flat[i]);

                he->evaluator->multiply_plain_inplace(cts_x_flat[i], tmp_w[i]);
                he->evaluator->multiply_inplace(cts_x_flat[i], cts_sigma_flat[i]);
                he->evaluator->add_plain_inplace(cts_x_flat[i], tmp_b[i]);
                he->evaluator->rescale_to_next_inplace(cts_x_flat[i]);
                he->evaluator->relinearize_inplace(cts_x_flat[i], *he->relin_keys);
            } else {
                he->devaluator->add_plain_inplace(cts_x_flat[i], pts_x_flat_avg[i]);
                he->devaluator->add_plain_inplace(cts_sigma_flat[i], pts_sigma_flat[i]);

                he->devaluator->multiply_plain_inplace(cts_x_flat[i], tmp_w[i]);
                he->devaluator->multiply_inplace(cts_x_flat[i], cts_sigma_flat[i]);

                he->devaluator->add_plain_inplace(cts_x_flat[i], tmp_b[i]);
                he->devaluator->rescale_to_next_inplace(cts_x_flat[i]);
                he->devaluator->relinearize_inplace(cts_x_flat[i], *he->drelin_keys);
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
                    "formal layer-0 RNS import metadata mismatch after share addition");
            }
            cout << "RAW_INPUT_RNS_SCALE status=PASS actual_scale_bits="
                 << log2(cts_x_flat.front().scale())
                 << " limbs=" << cts_x_flat.front().coeff_modulus_size()
                 << " cts=" << cts_x_flat.size() << endl;
        }
        cout << "scale:" << log2(cts_x_flat[0].scale()) << endl;
        cout << "(x_avg * sigma) * w + b done" << endl;  
        vector<UnifiedCiphertext> q_k_v;
        linear.linear_1(
            he,
            cts_x_flat,
            linear.pp_1[layer_id],
            data,
            q_k_v,
            a0_trace_enabled ? &a0_qkv_ciphertexts : nullptr
        );
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
        if (a0_trace_enabled) {
            a0_qk_ciphertexts.assign(q_k_v.begin(),
                                     q_k_v.begin() + qk_ct_count);
        }
#ifdef BLB_VIT65_HETERO
        if (!raw_input || qk_ct_count != 4 || v_ct_count != 2 ||
            boundary_qk_ct_count > qk_ct_count ||
            boundary_qk_ct_count % 2 != 0 || boundary_v_ct_count != 0) {
            throw runtime_error("heterogeneous selective-boundary geometry mismatch");
        }
        hetero_qk_ciphertexts.assign(
            qk_ct_count, UnifiedCiphertext(LOCATION::DEVICE));
        for (size_t index = boundary_qk_ct_count; index < qk_ct_count;
             ++index) {
            hetero_qk_ciphertexts[index] = std::move(q_k_v[index]);
        }
        hetero_v_ciphertexts.clear();
        hetero_v_ciphertexts.reserve(v_ct_count);
        for (size_t index = 0; index < v_ct_count; ++index) {
            hetero_v_ciphertexts.emplace_back(
                std::move(q_k_v[qk_ct_count + index]));
        }
#endif
        // Load result ciphertexts to CPU.  Algorithm 1 samples masks in the
        // host SEAL RNS ring, but no ciphertext is decrypted on the evaluator.
        start = high_resolution_clock::now();
        if (raw_input) {
            for (size_t i = 0; i < boundary_qk_ct_count; ++i) {
                if (he->is_gpu_enable()) {
                    q_k_v[i].to_host(*he->context, *he->dcontext);
                }
                raw_qk_cts[i] = std::move(q_k_v[i].hcipher());
            }
            for (size_t i = 0; i < boundary_v_ct_count; ++i) {
                const size_t source = qk_ct_count + i;
                if (he->is_gpu_enable()) {
                    q_k_v[source].to_host(*he->context, *he->dcontext);
                }
                raw_v_cts[i] = std::move(q_k_v[source].hcipher());
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
        if (he->is_gpu_enable()) {
            this->ct_transfer_mem_la +=
                ((high_resolution_clock::now() - start)).count()/1e+9;
        }
        NVTX_POP(true);

        // BLB Algorithm 1: mask in the current RNS ring before sending. The
        // evaluator keeps -r as its field-to-ring input share.
        start = high_resolution_clock::now();
        if (raw_input) {
            prepare_host_conversion_from_actual_scale(
                *this, raw_qk_cts, kFormalConversionScaleBits, he,
                "layer0_qk");
            mask_ckks_to_mpc_inplace(raw_qk_cts, raw_qk_pts, he);
            send_encrypted_vector(io, raw_qk_cts);
            if (!raw_v_cts.empty()) {
                prepare_host_conversion_from_actual_scale(
                    *this, raw_v_cts, kFormalConversionScaleBits, he,
                    "layer0_v");
                mask_ckks_to_mpc_inplace(raw_v_cts, raw_v_pts, he);
                send_encrypted_vector(io, raw_v_cts);
            }
            cout << "RAW_QKV_CONVERSION_LAYOUT status=PASS qk_cts="
                 << raw_qk_cts.size() << " qk_scale=" << NL_SCALE
                 << " v_cts=" << raw_v_cts.size()
                 << " v_scale=" << ATTENTION_V_SCALE
                 << " packing=ordinary_full_slot" << endl;
        } else {
            mask_ckks_to_mpc_inplace(output_cts_half, output_pts_half, he);
            send_encrypted_vector(io, output_cts_half);
        }
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    }
    start = high_resolution_clock::now();
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    // cout << "send and receive comm:" << this->get_comm() << endl;
    // cout << "send and receive round:" << this->get_round() << endl;
    // he->print_parameters(he->context->get_context_data(output_pts[0].parms_id())->parms());
    vector<vector<HE_CKKS::scalar_t>> output_vector;
    if (raw_input) {
        vector<vector<HE_CKKS::scalar_t>> qk_output = ckks_to_mpc(
            raw_qk_pts, degree, he->get_Q(raw_qk_pts[0]), NL_ELL, NL_SCALE,
            he, false);
        vector<vector<HE_CKKS::scalar_t>> v_output;
        if (!raw_v_pts.empty()) {
            v_output = ckks_to_mpc(
                raw_v_pts, degree, he->get_Q(raw_v_pts[0]), NL_ELL,
                ATTENTION_V_SCALE, he, false);
        }
        output_vector.reserve(qk_output.size() + v_output.size());
        for (auto &packed : qk_output) {
            output_vector.emplace_back(std::move(packed));
        }
        for (auto &packed : v_output) {
            output_vector.emplace_back(std::move(packed));
        }
    } else {
        output_vector = ckks_to_mpc(output_pts_half, degree,
                                    he->get_Q(output_pts_half[0]), bitwidth, s,
                                    he, true);
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
    map<string, FixArray> raw;
    raw["raw_input"] = input;
    return layer_norm_part2_attention(raw, layer_id);
}

#ifdef BLB_VIT65_HETERO
void Bert_ckks::set_heterogeneous_head_protocols(
    const vector<string> &protocols) {
    if (protocols.size() != PACKING_NUM) {
        throw invalid_argument("heterogeneous protocol count mismatch");
    }
    bool saw_linear = false;
    size_t relu_heads = 0;
    for (const string &protocol : protocols) {
        if (protocol != "relu" && protocol != "linear") {
            throw invalid_argument("unknown heterogeneous head protocol");
        }
        if (protocol == "linear") {
            saw_linear = true;
        } else {
            if (saw_linear) {
                throw invalid_argument(
                    "heterogeneous heads must be protocol-reordered");
            }
            ++relu_heads;
        }
    }
    hetero_head_protocols = protocols;
    hetero_boundary_qk_ct_count =
        std::min<size_t>(PACKING_NUM, (relu_heads + 1) & ~size_t(1));
    cout << "HETERO_PUBLIC_ASSIGNMENT status=PASS relu_heads="
         << relu_heads << " boundary_qk_cts="
         << hetero_boundary_qk_ct_count
         << " private_branching=0" << endl;
}
#endif

FixArray Bert_ckks::attention_output_from_softmax(FixArray input,
                                                  int layer_id) {
    if (layer_id < 0 || layer_id >= ATTENTION_LAYERS) {
        throw invalid_argument("BLB attention layer index is out of range");
    }
    const FCMetadata attention_data = linear.data_lin1_0;
    const size_t softmax_values = static_cast<size_t>(PACKING_NUM) *
        attention_data.image_size * attention_data.image_size;
    if (input.size != static_cast<int>(softmax_values) ||
        input.ell != NL_ELL || input.s != NL_SCALE) {
        throw runtime_error(
            "BLB attention-only Softmax input must be ell37/s12 and full-sized");
    }
    if (!attention_v_ready ||
        attention_v_share.size != attention_data.image_size * COMMON_DIM ||
        attention_v_share.ell != NL_ELL ||
        attention_v_share.s != ATTENTION_V_SCALE) {
        throw runtime_error(
            "BLB attention-only path requires the real packed V share");
    }

    HE_CKKS *he = linear.he2;
    const int slot = he->num_slots;
    const size_t softmax_ct_count = softmax_values / slot;
    const size_t value_count = static_cast<size_t>(attention_data.image_size) *
        COMMON_DIM;
    const size_t value_ct_count = value_count / slot;
    if (softmax_ct_count * slot != softmax_values ||
        value_ct_count * slot != value_count) {
        throw runtime_error(
            "BLB attention-only tensors do not fill CKKS ciphertexts exactly");
    }

    vector<uint64_t> softmax_packed(softmax_values);
    linear.preprocess_softmax(input.data, softmax_packed.data(),
                              attention_data);
    vector<vector<HE_CKKS::scalar_t>> softmax_share_vectors(
        softmax_ct_count, vector<HE_CKKS::scalar_t>(slot));
    vector<vector<HE_CKKS::scalar_t>> value_share_vectors(
        value_ct_count, vector<HE_CKKS::scalar_t>(slot));
    for (size_t ct = 0; ct < softmax_ct_count; ++ct) {
        copy_n(softmax_packed.data() + ct * slot, slot,
               softmax_share_vectors[ct].begin());
    }
    for (size_t ct = 0; ct < value_ct_count; ++ct) {
        copy_n(attention_v_share.data + ct * slot, slot,
               value_share_vectors[ct].begin());
    }

    auto conversion_start = high_resolution_clock::now();
    vector<UnifiedPlaintext> softmax_share_plain;
    vector<UnifiedPlaintext> value_share_plain;
    import_rns_target_scale(*this, softmax_share_vectors, slot, input.ell,
                            input.s, he->he_scale, he,
                            softmax_share_plain, "a0_softmax");
    import_rns_target_scale(*this, value_share_vectors, slot,
                            attention_v_share.ell,
                            attention_v_share.s, he->he_scale, he,
                            value_share_plain, "a0_value");
    conversion_la +=
        ((high_resolution_clock::now() - conversion_start)).count() / 1e9;
    conversion_comm += get_comm();
    conversion_round += get_round();

    vector<UnifiedCiphertext> encrypted_softmax(
        softmax_ct_count, UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> encrypted_value(
        value_ct_count, UnifiedCiphertext(LOCATION::HOST));
    if (party == BOB) {
        for (size_t ct = 0; ct < softmax_ct_count; ++ct) {
            he->encryptor->encrypt(softmax_share_plain[ct],
                                   encrypted_softmax[ct]);
        }
        for (size_t ct = 0; ct < value_ct_count; ++ct) {
            he->encryptor->encrypt(value_share_plain[ct],
                                   encrypted_value[ct]);
        }
        send_encrypted_vector(io, encrypted_softmax);
        send_encrypted_vector(io, encrypted_value);
    } else {
        auto transfer_start = high_resolution_clock::now();
        recv_encrypted_vector(he->context, io, encrypted_softmax);
        recv_encrypted_vector(he->context, io, encrypted_value);
        ct_transfer_la +=
            ((high_resolution_clock::now() - transfer_start)).count() / 1e9;
        if (he->is_gpu_enable()) {
            auto device_start = high_resolution_clock::now();
            for (auto &cipher : encrypted_softmax) {
                cipher.to_device(*he->context, *he->dcontext);
            }
            for (auto &cipher : encrypted_value) {
                cipher.to_device(*he->context, *he->dcontext);
            }
            for (auto &plain : softmax_share_plain) {
                plain.to_device(*he->context, *he->dcontext);
            }
            for (auto &plain : value_share_plain) {
                plain.to_device(*he->context, *he->dcontext);
            }
            ct_transfer_mem_la +=
                ((high_resolution_clock::now() - device_start)).count() /
                1e9;
        }
        for (size_t ct = 0; ct < softmax_ct_count; ++ct) {
            if (he->is_gpu_enable()) {
                he->devaluator->add_plain_inplace(encrypted_softmax[ct],
                                                   softmax_share_plain[ct]);
            } else {
                he->evaluator->add_plain_inplace(encrypted_softmax[ct],
                                                  softmax_share_plain[ct]);
            }
        }
        for (size_t ct = 0; ct < value_ct_count; ++ct) {
            if (he->is_gpu_enable()) {
                he->devaluator->add_plain_inplace(encrypted_value[ct],
                                                   value_share_plain[ct]);
            } else {
                he->evaluator->add_plain_inplace(encrypted_value[ct],
                                                  value_share_plain[ct]);
            }
        }
    }
    ct_transfer_comm += get_comm();
    ct_transfer_round += get_round();

    vector<UnifiedCiphertext> wo_ciphertexts;
    if (party == ALICE) {
        const LOCATION location = he->backend();
        auto he_start = high_resolution_clock::now();
        auto softmax_masks = linear.softmax_mask_ct_ct(he, attention_data);
        vector<vector<vector<UnifiedCiphertext>>> packed_softmax;
        linear.preprocess_softmax_s1_ct_ct(
            he, encrypted_softmax, attention_data, packed_softmax,
            &softmax_masks);
        vector<UnifiedCiphertext> av_ciphertexts(
            value_ct_count, UnifiedCiphertext(location));
        linear.softmax_v(he, packed_softmax, encrypted_value,
                         attention_data, av_ciphertexts);
        if (a0_trace_enabled) {
            a0_av_ciphertexts = av_ciphertexts;
        }
        linear.linear_2(he, av_ciphertexts, linear.pp_2[layer_id],
                        linear.data_lin2, wo_ciphertexts);
        he_compute_la +=
            ((high_resolution_clock::now() - he_start)).count() / 1e9;
    } else {
        wo_ciphertexts.resize(value_ct_count,
                              UnifiedCiphertext(LOCATION::HOST));
    }

    vector<Ciphertext> host_output(value_ct_count);
    vector<Plaintext> output_plain(value_ct_count);
    auto transfer_start = high_resolution_clock::now();
    if (party == ALICE) {
        if (wo_ciphertexts.size() != value_ct_count) {
            throw runtime_error(
                "BLB attention-only WO ciphertext count mismatch");
        }
        for (size_t ct = 0; ct < value_ct_count; ++ct) {
            if (he->is_gpu_enable()) {
                wo_ciphertexts[ct].to_host(*he->context, *he->dcontext);
            }
            host_output[ct] = std::move(wo_ciphertexts[ct].hcipher());
        }
        prepare_host_conversion_from_actual_scale(
            *this, host_output, kFormalConversionScaleBits, he, "a0_wo");
        mask_ckks_to_mpc_inplace(host_output, output_plain, he);
        send_encrypted_vector(io, host_output);
    } else {
        recv_encrypted_vector(he->context, io, host_output);
        for (size_t ct = 0; ct < value_ct_count; ++ct) {
            he->decryptor->decrypt(host_output[ct], output_plain[ct]);
        }
    }
    ct_transfer_la +=
        ((high_resolution_clock::now() - transfer_start)).count() / 1e9;
    ct_transfer_comm += get_comm();
    ct_transfer_round += get_round();

    conversion_start = high_resolution_clock::now();
    vector<vector<HE_CKKS::scalar_t>> packed_output = ckks_to_mpc(
        output_plain, he->poly_modulus_degree,
        he->get_Q(output_plain.front()), NL_ELL, NL_SCALE, he, false);
    conversion_la +=
        ((high_resolution_clock::now() - conversion_start)).count() / 1e9;
    conversion_comm += get_comm();
    conversion_round += get_round();
    if (packed_output.size() != value_ct_count) {
        throw runtime_error("BLB attention-only WO conversion count mismatch");
    }
    FixArray packed_fix(party, static_cast<int>(value_count), true, NL_ELL,
                        NL_SCALE);
    size_t offset = 0;
    for (const auto &cipher_slots : packed_output) {
        if (cipher_slots.size() != static_cast<size_t>(slot)) {
            throw runtime_error(
                "BLB attention-only WO conversion slot mismatch");
        }
        copy(cipher_slots.begin(), cipher_slots.end(),
             packed_fix.data + offset);
        offset += cipher_slots.size();
    }
    attention_v_ready = false;
    FixArray token_major = bsgs_output_to_token_major(
        packed_fix, linear.data_lin2, "a0_wo_to_token_major");
    cout << "BLB_VIT65_A0_OUTPUT status=PASS layout=token_hidden_row_major"
         << " elements=" << token_major.size << " ell=" << token_major.ell
         << " scale=" << token_major.s << endl;
    return token_major;
}

#ifdef BLB_VIT65_HETERO
FixArray Bert_ckks::attention_output_from_heterogeneous_scores(
    FixArray normalized_pair, int layer_id) {
    if (layer_id < 0 || layer_id >= ATTENTION_LAYERS) {
        throw invalid_argument("heterogeneous attention layer is out of range");
    }
    const FCMetadata attention_data = linear.data_lin1_0;
    const size_t all_score_values = static_cast<size_t>(PACKING_NUM) *
        attention_data.image_size * attention_data.image_size;
    if (normalized_pair.size != static_cast<int>(all_score_values) ||
        normalized_pair.ell != NL_ELL || normalized_pair.s != NL_SCALE) {
        throw runtime_error(
            "heterogeneous normalized pair must be full ell37/s12 scores");
    }

    HE_CKKS *he = linear.he1;
    const int slot = he->num_slots;
    const size_t qk_ct_count = all_score_values / slot;
    const size_t pair_boundary_ct_count = hetero_boundary_qk_ct_count;
    const size_t value_count = static_cast<size_t>(attention_data.image_size) *
                               COMMON_DIM;
    const size_t value_ct_count = value_count / slot;
    if (qk_ct_count != 4 || pair_boundary_ct_count > qk_ct_count ||
        pair_boundary_ct_count % 2 != 0 ||
        value_ct_count != 2 ||
        all_score_values % slot != 0 || value_count % slot != 0) {
        throw runtime_error("heterogeneous continuation layout mismatch");
    }

    vector<uint64_t> packed_scores(all_score_values);
    linear.preprocess_softmax(normalized_pair.data, packed_scores.data(),
                              attention_data);
    vector<vector<HE_CKKS::scalar_t>> pair_share_vectors(
        pair_boundary_ct_count, vector<HE_CKKS::scalar_t>(slot));
    for (size_t ct = 0; ct < pair_boundary_ct_count; ++ct) {
        copy_n(packed_scores.data() + ct * slot, slot,
               pair_share_vectors[ct].begin());
    }

    auto conversion_start = high_resolution_clock::now();
    vector<UnifiedPlaintext> pair_share_plain;
    if (!pair_share_vectors.empty()) {
        import_rns_target_scale(*this, pair_share_vectors, slot,
                                normalized_pair.ell, normalized_pair.s,
                                kFormalConversionScaleBits, he,
                                pair_share_plain, "hetero_normalized_pair");
    }
    conversion_la +=
        ((high_resolution_clock::now() - conversion_start)).count() / 1e9;
    conversion_comm += get_comm();
    conversion_round += get_round();

    vector<UnifiedCiphertext> encrypted_pair(
        pair_boundary_ct_count, UnifiedCiphertext(LOCATION::HOST));
    if (party == BOB && pair_boundary_ct_count != 0) {
        for (size_t ct = 0; ct < pair_boundary_ct_count; ++ct) {
            he->encryptor->encrypt(pair_share_plain[ct], encrypted_pair[ct]);
        }
        send_encrypted_vector(io, encrypted_pair);
    } else if (party == ALICE && pair_boundary_ct_count != 0) {
        auto transfer_start = high_resolution_clock::now();
        recv_encrypted_vector(he->context, io, encrypted_pair);
        ct_transfer_la +=
            ((high_resolution_clock::now() - transfer_start)).count() / 1e9;
        if (hetero_qk_ciphertexts.size() != qk_ct_count ||
            hetero_v_ciphertexts.size() != value_ct_count) {
            throw runtime_error(
                "heterogeneous timed-path device snapshots are incomplete");
        }
        const size_t direct_limbs = pair_boundary_ct_count < qk_ct_count
            ? hetero_qk_ciphertexts[pair_boundary_ct_count]
                  .coeff_modulus_size()
            : 3;
        for (size_t ct = 0; ct < pair_boundary_ct_count; ++ct) {
            encrypted_pair[ct].to_device(*he->context, *he->dcontext);
            pair_share_plain[ct].to_device(*he->context, *he->dcontext);
            he->devaluator->add_plain_inplace(encrypted_pair[ct],
                                               pair_share_plain[ct]);
            while (encrypted_pair[ct].coeff_modulus_size() > direct_limbs) {
                he->devaluator->mod_switch_to_next_inplace(
                    encrypted_pair[ct]);
            }
            if (encrypted_pair[ct].coeff_modulus_size() != direct_limbs ||
                abs(log2(encrypted_pair[ct].scale()) -
                    kFormalConversionScaleBits) > 0.25) {
                throw runtime_error(
                    "heterogeneous imported score metadata alignment failed");
            }
        }
    }
    ct_transfer_comm += get_comm();
    ct_transfer_round += get_round();

    vector<UnifiedCiphertext> wo_ciphertexts;
    if (party == ALICE) {
        vector<UnifiedCiphertext> score_ciphertexts(
            qk_ct_count, UnifiedCiphertext(LOCATION::DEVICE));
        for (size_t ct = 0; ct < pair_boundary_ct_count; ++ct) {
            score_ciphertexts[ct] = std::move(encrypted_pair[ct]);
        }
        for (size_t ct = pair_boundary_ct_count; ct < qk_ct_count; ++ct) {
            score_ciphertexts[ct] =
                std::move(hetero_qk_ciphertexts[ct]);
        }

        auto he_start = high_resolution_clock::now();
        vector<UnifiedCiphertext> av_ciphertexts(
            value_ct_count, UnifiedCiphertext(LOCATION::DEVICE));
        linear.hetero_scores_v(he, score_ciphertexts,
                               hetero_v_ciphertexts, attention_data,
                               pair_boundary_ct_count, av_ciphertexts);
        if (a0_trace_enabled) {
            a0_av_ciphertexts = av_ciphertexts;
        }
        linear.linear_2(he, av_ciphertexts, linear.pp_2_he1[layer_id],
                        linear.data_lin2, wo_ciphertexts);
        he_compute_la +=
            ((high_resolution_clock::now() - he_start)).count() / 1e9;
    } else {
        wo_ciphertexts.resize(value_ct_count,
                              UnifiedCiphertext(LOCATION::HOST));
    }

    vector<Ciphertext> host_output(value_ct_count);
    vector<Plaintext> output_plain(value_ct_count);
    auto transfer_start = high_resolution_clock::now();
    if (party == ALICE) {
        if (wo_ciphertexts.size() != value_ct_count) {
            throw runtime_error("heterogeneous WO ciphertext count mismatch");
        }
        for (size_t ct = 0; ct < value_ct_count; ++ct) {
            wo_ciphertexts[ct].to_host(*he->context, *he->dcontext);
            host_output[ct] = std::move(wo_ciphertexts[ct].hcipher());
        }
        const auto wo_context =
            he->context->get_context_data(host_output.front().parms_id());
        if (!wo_context ||
            wo_context->parms().coeff_modulus().size() != 2) {
            throw runtime_error(
                "heterogeneous WO metadata snap requires two limbs");
        }
        const double desired_pre_rescale =
            ldexp(1.0, kFormalConversionScaleBits) *
            static_cast<double>(
                wo_context->parms().coeff_modulus().back().value());
        const double snap_relative_error =
            abs(host_output.front().scale() - desired_pre_rescale) /
            desired_pre_rescale;
        if (!isfinite(snap_relative_error) || snap_relative_error > 1e-9) {
            throw runtime_error(
                "heterogeneous WO scale metadata snap is not negligible");
        }
        for (auto &ciphertext : host_output) {
            const double local_relative_error =
                abs(ciphertext.scale() - desired_pre_rescale) /
                desired_pre_rescale;
            if (!isfinite(local_relative_error) ||
                local_relative_error > 1e-9) {
                throw runtime_error(
                    "heterogeneous WO ciphertext scales are inconsistent");
            }
            ciphertext.scale() = desired_pre_rescale;
        }
        cout << "HETERO_WO_SCALE_SNAP status=PASS relative_error="
             << snap_relative_error << " target_pre_rescale_bits="
             << log2(desired_pre_rescale)
             << " coefficient_change=0" << endl;
        prepare_host_conversion_from_actual_scale(
            *this, host_output, kFormalConversionScaleBits, he,
            "hetero_wo");
        mask_ckks_to_mpc_inplace(host_output, output_plain, he);
        send_encrypted_vector(io, host_output);
    } else {
        recv_encrypted_vector(he->context, io, host_output);
        for (size_t ct = 0; ct < value_ct_count; ++ct) {
            he->decryptor->decrypt(host_output[ct], output_plain[ct]);
        }
    }
    ct_transfer_la +=
        ((high_resolution_clock::now() - transfer_start)).count() / 1e9;
    ct_transfer_comm += get_comm();
    ct_transfer_round += get_round();

    conversion_start = high_resolution_clock::now();
    vector<vector<HE_CKKS::scalar_t>> packed_output = ckks_to_mpc(
        output_plain, he->poly_modulus_degree,
        he->get_Q(output_plain.front()), NL_ELL, NL_SCALE, he, false);
    conversion_la +=
        ((high_resolution_clock::now() - conversion_start)).count() / 1e9;
    conversion_comm += get_comm();
    conversion_round += get_round();
    if (packed_output.size() != value_ct_count) {
        throw runtime_error("heterogeneous WO conversion count mismatch");
    }
    FixArray packed_fix(party, static_cast<int>(value_count), true, NL_ELL,
                        NL_SCALE);
    size_t offset = 0;
    for (const auto &cipher_slots : packed_output) {
        if (cipher_slots.size() != static_cast<size_t>(slot)) {
            throw runtime_error("heterogeneous WO conversion slot mismatch");
        }
        copy(cipher_slots.begin(), cipher_slots.end(),
             packed_fix.data + offset);
        offset += cipher_slots.size();
    }
    hetero_qk_ciphertexts.clear();
    hetero_v_ciphertexts.clear();
    FixArray token_major = bsgs_output_to_token_major(
        packed_fix, linear.data_lin2, "hetero_wo_to_token_major");
    cout << "BLB_VIT65_HETERO_OUTPUT status=PASS"
         << " layout=token_hidden_row_major elements=" << token_major.size
         << " ell=" << token_major.ell << " scale=" << token_major.s
         << endl;
    return token_major;
}
#endif

map<string, FixArray> Bert_ckks::materialize_a0_attention_trace() {
    if (!a0_trace_enabled) {
        throw runtime_error("A0 HE trace materialization was not enabled");
    }
    HE_CKKS *he = linear.he1;
    const size_t value_ct_count = static_cast<size_t>(INPUT_DIM) *
        COMMON_DIM / he->num_slots;
    const size_t qkv_ct_count = value_ct_count * 3;
    const size_t qk_ct_count = static_cast<size_t>(PACKING_NUM) * INPUT_DIM *
                               INPUT_DIM / he->num_slots;
    if (party == ALICE && a0_qkv_ciphertexts.size() != qkv_ct_count) {
        throw runtime_error("A0 QKV trace ciphertext snapshot is incomplete");
    }
    if (party == ALICE && a0_av_ciphertexts.size() != value_ct_count) {
        throw runtime_error("A0 AV trace ciphertext snapshot is incomplete");
    }
    if (party == ALICE && a0_qk_ciphertexts.size() != qk_ct_count) {
        throw runtime_error("A0 QK trace ciphertext snapshot is incomplete");
    }

    auto convert_snapshot = [&](HE_CKKS *snapshot_he,
                                vector<UnifiedCiphertext> source,
                                size_t expected_cts, int output_scale,
                                const char *stage) {
        vector<Ciphertext> host_ciphertexts(expected_cts);
        vector<Plaintext> local_plaintexts(expected_cts);
        if (party == ALICE) {
            if (source.size() != expected_cts) {
                throw runtime_error(string("A0 trace ciphertext count mismatch: ") +
                                    stage);
            }
            for (size_t index = 0; index < expected_cts; ++index) {
                if (snapshot_he->is_gpu_enable()) {
                    source[index].to_host(*snapshot_he->context,
                                          *snapshot_he->dcontext);
                }
                host_ciphertexts[index] =
                    std::move(source[index].hcipher());
            }
            prepare_host_conversion_from_actual_scale(
                *this, host_ciphertexts, kFormalConversionScaleBits,
                snapshot_he, stage);
            mask_ckks_to_mpc_inplace(host_ciphertexts, local_plaintexts,
                                     snapshot_he);
            send_encrypted_vector(io, host_ciphertexts);
        } else {
            recv_encrypted_vector(snapshot_he->context, io,
                                  host_ciphertexts);
            for (size_t index = 0; index < expected_cts; ++index) {
                snapshot_he->decryptor->decrypt(host_ciphertexts[index],
                                                local_plaintexts[index]);
            }
        }
        vector<vector<HE_CKKS::scalar_t>> packed = ckks_to_mpc(
            local_plaintexts, snapshot_he->poly_modulus_degree,
            snapshot_he->get_Q(local_plaintexts.front()), NL_ELL,
            output_scale, snapshot_he, false);
        FixArray result(party,
                        static_cast<int>(expected_cts * snapshot_he->num_slots),
                        true, NL_ELL, output_scale);
        size_t offset = 0;
        for (const auto &cipher_slots : packed) {
            if (cipher_slots.size() !=
                static_cast<size_t>(snapshot_he->num_slots)) {
                throw runtime_error(string("A0 trace slot mismatch: ") +
                                    stage);
            }
            copy(cipher_slots.begin(), cipher_slots.end(),
                 result.data + offset);
            offset += cipher_slots.size();
        }
        cout << "A0_TRACE_CONVERSION stage=" << stage
             << " status=PASS cts=" << expected_cts
             << " scale=" << output_scale
             << " validation_only=1" << endl;
        return result;
    };

    vector<UnifiedCiphertext> q_ciphertexts;
    vector<UnifiedCiphertext> k_ciphertexts;
    vector<UnifiedCiphertext> v_ciphertexts;
    vector<UnifiedCiphertext> qk_ciphertexts;
    if (party == ALICE) {
        q_ciphertexts.assign(a0_qkv_ciphertexts.begin(),
                             a0_qkv_ciphertexts.begin() + value_ct_count);
        k_ciphertexts.assign(a0_qkv_ciphertexts.begin() + value_ct_count,
                             a0_qkv_ciphertexts.begin() + 2 * value_ct_count);
        v_ciphertexts.assign(a0_qkv_ciphertexts.begin() + 2 * value_ct_count,
                             a0_qkv_ciphertexts.end());
        qk_ciphertexts = a0_qk_ciphertexts;
    }
    FixArray q_packed = convert_snapshot(
        linear.he1, std::move(q_ciphertexts), value_ct_count,
        ATTENTION_V_SCALE,
        "a0_trace_q");
    FixArray k_packed = convert_snapshot(
        linear.he1, std::move(k_ciphertexts), value_ct_count,
        ATTENTION_V_SCALE,
        "a0_trace_k");
    FixArray v_packed = convert_snapshot(
        linear.he1, std::move(v_ciphertexts), value_ct_count,
        ATTENTION_V_SCALE, "a0_trace_v");
    FixArray qk_packed = convert_snapshot(
        linear.he1, std::move(qk_ciphertexts), qk_ct_count,
        NL_SCALE, "a0_trace_qk");
    FixArray av_packed = convert_snapshot(
#ifdef BLB_VIT65_HETERO
        linear.he1,
#else
        linear.he2,
#endif
        party == ALICE ? a0_av_ciphertexts : vector<UnifiedCiphertext>(),
        value_ct_count, NL_SCALE + ATTENTION_V_SCALE, "a0_trace_av");

    map<string, FixArray> result;
    for (const auto &item : vector<pair<string, FixArray *>>{
             {"q", &q_packed}, {"k", &k_packed}, {"v", &v_packed},
             {"av", &av_packed}}) {
        FixArray unpacked(party, item.second->size, true, item.second->ell,
                          item.second->s);
        linear.plain_cross_packing_postprocess_v(
            item.second->data, unpacked.data, false, linear.data_lin1_0);
        result[item.first] = unpacked;
    }
    FixArray qk_unpacked(party, qk_packed.size, true, qk_packed.ell,
                         qk_packed.s);
    linear.plain_cross_packing_postprocess(
        qk_packed.data, qk_unpacked.data, false, linear.data_lin1_0);
    result["qk"] = qk_unpacked;
#ifdef BLB_VIT65_HETERO
    // Heterogeneous normalization is traced directly by the executable; it
    // has no Softmax exp/sum/div intermediates.
#else
    if (a0_softmax_exp_share.size <= 0 ||
        a0_softmax_sum_share.size <= 0 ||
        a0_softmax_div_share.size <= 0) {
        throw runtime_error("A0 Softmax validation snapshots are incomplete");
    }
    result["softmax_exp"] = a0_softmax_exp_share;
    result["softmax_sum"] = a0_softmax_sum_share;
    result["softmax_div"] = a0_softmax_div_share;
#endif
    a0_qkv_ciphertexts.clear();
    a0_qk_ciphertexts.clear();
    a0_av_ciphertexts.clear();
#ifndef BLB_VIT65_HETERO
    a0_softmax_exp_share = FixArray();
    a0_softmax_sum_share = FixArray();
    a0_softmax_div_share = FixArray();
#endif
    return result;
}

FixArray Bert_ckks::layer_norm_affine_ckks(
    const map<string, FixArray> &input) {
    auto get_share = [&input](const string &name) {
        const auto entry = input.find(name);
        if (entry != input.end()) return entry->second;
        throw runtime_error("formal final LayerNorm share is missing: " + name);
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
            "formal final LayerNorm affine requires ell37/s12 shares");
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
    import_rns_target_scale(*this, pack(x_flat_avg), slot, bitwidth, scale,
                            he->he_scale, he, x_pts,
                            "final_ln_centered_input");
    import_rns_target_scale(*this, pack(sigma_flat), slot, bitwidth, scale,
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
                w_matrix[i][j] = layernorm_fixed_to_real(
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
                if (abs(x_cts[i].scale() - ln_affine_input_scale) /
                            ln_affine_input_scale >
                        1e-12 ||
                    abs(sigma_cts[i].scale() - ln_affine_input_scale) /
                            ln_affine_input_scale >
                        1e-12) {
                    throw runtime_error(
                        "formal final LN Algorithm-2 target scale mismatch");
                }
                he->evaluator->multiply_inplace(x_cts[i], sigma_cts[i]);
                he->evaluator->relinearize_inplace(x_cts[i], *he->relin_keys);
                he->evaluator->multiply_plain_inplace(x_cts[i], tmp_w[i]);
                he->evaluator->add_plain_inplace(x_cts[i], tmp_b[i]);
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
                he->devaluator->rescale_to_next_inplace(x_cts[i]);
            }
        }
        cout << "FORMAL_LAYERNORM_SCALE_PLAN stage=final_affine status=PASS"
             << " input_scale_bits=" << log2(ln_affine_input_scale)
             << " weight_scale_bits=" << log2(ln_affine_weight_scale)
             << " bias_scale_bits=" << log2(ln_affine_bias_scale)
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
std::map<string, FixArray> Bert_ckks::Softmax_v_linear_layer_norm_part1(
    FixArray input, int layer_id){
    FCMetadata data = linear.data_lin1_0;
    int bitwidth,s;
    bitwidth = input.ell;
    s = input.s;
    if (bitwidth != NL_ELL || s != NL_SCALE) {
        throw runtime_error("formal FFN2 requires ell37/s12 GeLU shares");
    }
    if (bitwidth != NL_ELL || s != NL_SCALE) {
        throw runtime_error(
            "formal Softmax*V requires an ell37/s12 Softmax share");
    }
    int softmax_dim = data.image_size;
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
    if (!attention_v_ready || attention_v_share.size != data.image_size * COMMON_DIM) {
        throw runtime_error("formal Softmax*V refused: real V share is unavailable");
    }
    if (attention_v_share.ell != NL_ELL ||
        attention_v_share.s != ATTENTION_V_SCALE) {
        throw runtime_error(
            "formal Softmax*V requires the real V share at ell37/s11");
    }
    if (!attention_input_residual_ready ||
        attention_input_residual_share.size != data.image_size * COMMON_DIM) {
        throw runtime_error("formal attention output refused: real block input residual is unavailable");
    }
    if (attention_input_residual_share.ell != NL_ELL ||
        attention_input_residual_share.s != NL_SCALE) {
        throw runtime_error(
            "formal attention residual must remain at ell37/s12");
    }
    // linear_2 emits BSGS split-half slots.  The checkpoint input retained
    // for the residual is token-major, so place each local share in exactly
    // the output layout before encrypting and adding it to h3.
    FixArray attention_residual_packed = token_major_to_bsgs_output(
        attention_input_residual_share, linear.data_lin2,
        "attention_residual_to_h3");
    vector<vector<HE_CKKS::scalar_t>> softmax_output_pack_vector(num_cts_softmax);
    vector<vector<HE_CKKS::scalar_t>> v_share_vector(num_cts_ln);
    vector<vector<HE_CKKS::scalar_t>> attention_residual_vector(num_cts_ln);
    vector<UnifiedPlaintext> softmax_output_pack_pts;
    for(int i=0;i<num_cts_softmax;i++){
        for(int j=0;j<slot;j++){
            softmax_output_pack_vector[i].push_back(softmax_output_pack[i*slot+j]);
        }
    }
    for (int i = 0; i < num_cts_ln; ++i) {
        v_share_vector[i].assign(attention_v_share.data + i * slot,
                                 attention_v_share.data + (i + 1) * slot);
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
    import_rns_target_scale(*this, softmax_output_pack_vector, slot,
                            bitwidth, s, he->he_scale, he,
                            softmax_output_pack_pts, "softmax");
    vector<UnifiedPlaintext> v_share_pts;
    vector<UnifiedPlaintext> attention_residual_pts;
    import_rns_target_scale(*this, v_share_vector, slot,
                            attention_v_share.ell, attention_v_share.s,
                            he->he_scale, he, v_share_pts, "attention_v");
    import_rns_target_scale(*this, attention_residual_vector, slot,
                            attention_input_residual_share.ell,
                            attention_input_residual_share.s, he->he_scale,
                            he, attention_residual_pts,
                            "attention_residual");
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    vector<Plaintext> x_avg_pts(num_cts_ln);
    vector<Plaintext> sigma_pts(1);
    vector<Ciphertext> enc_x_avg(num_cts_ln);
    vector<Ciphertext> enc_sigma(1);
    
    vector<UnifiedCiphertext> enc_softmax(
        num_cts_softmax, UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> enc_v(
        num_cts_ln, UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> enc_attention_residual(
        num_cts_ln, UnifiedCiphertext(LOCATION::HOST));
    if (party==BOB){
        #pragma omp parallel for
        for(int i=0;i<num_cts_softmax;i++){
            he->encryptor->encrypt(softmax_output_pack_pts[i], enc_softmax[i]);
        }
        #pragma omp parallel for
        for(int i=0;i<num_cts_ln;i++){
            he->encryptor->encrypt(v_share_pts[i], enc_v[i]);
            he->encryptor->encrypt(attention_residual_pts[i],
                                   enc_attention_residual[i]);
        }
        send_encrypted_vector(io, enc_softmax);
        send_encrypted_vector(io, enc_v);
        send_encrypted_vector(io, enc_attention_residual);
        cout << "FORMAL_TRANSFER_LAYOUT stage=softmax_v_input status=PASS"
             << " layout=ordinary_full_slot softmax_cts=" << num_cts_softmax
             << " v_cts=" << num_cts_ln << " residual_cts=" << num_cts_ln
             << " imag_repack=0" << endl;
        cout << "send_encrypted_vector done" << endl;
        recv_encrypted_vector(he->context, io, enc_x_avg);
        recv_encrypted_vector(he->context, io, enc_sigma);
        #pragma omp parallel for
        for(int i=0;i<enc_x_avg.size();i++){
            he->decryptor->decrypt(enc_x_avg[i], x_avg_pts[i]);
        }
        cout << "recv_encrypted_vector done" << endl;
        he->decryptor->decrypt(enc_sigma[0], sigma_pts[0]);
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
        recv_encrypted_vector(he->context, io, enc_softmax);
        recv_encrypted_vector(he->context, io, enc_v);
        recv_encrypted_vector(he->context, io, enc_attention_residual);
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        
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
                he->evaluator->add_plain_inplace(enc_attention_residual[i],
                                                 attention_residual_pts[i]);
            }
        } else {
            for(int i=0;i<enc_softmax.size();i++){
                he->devaluator->add_plain_inplace(enc_softmax[i], softmax_output_pack_pts[i]);
            }
            for (int i = 0; i < enc_v.size(); ++i) {
                he->devaluator->add_plain_inplace(enc_v[i], v_share_pts[i]);
                he->devaluator->add_plain_inplace(enc_attention_residual[i],
                                                  attention_residual_pts[i]);
            }
        }
        require_uniform_metadata(enc_softmax, "softmax");
        require_uniform_metadata(enc_v, "attention_v");
        require_uniform_metadata(enc_attention_residual,
                                 "attention_residual");
        const double imported_scale = ldexp(1.0, he->he_scale);
        for (const auto *entry : {&enc_softmax, &enc_v,
                                  &enc_attention_residual}) {
            const double relative_error =
                abs(entry->front().scale() - imported_scale) /
                imported_scale;
            if (!isfinite(relative_error) || relative_error > 1e-12) {
                throw runtime_error(
                    "formal high-scale RNS import was not preserved");
            }
        }
        cout << "OK3" << endl;
        auto soft_mask_ct = linear.softmax_mask_ct_ct(he, data);
        cout << "OK4" << endl;
        vector<vector<vector<UnifiedCiphertext>>> pack_softmax_ct;
        linear.preprocess_softmax_s1_ct_ct(he, enc_softmax, data, pack_softmax_ct);
        cout << "OK5" << endl;
        vector<UnifiedCiphertext> softmax_V_result(
            PACKING_NUM * data.image_size * data.filter_w / data.slot_count,
            UnifiedCiphertext(loc));
        linear.softmax_v(he, pack_softmax_ct, enc_v, data, softmax_V_result);
        // softmax_V_result's pack may be changed
        data = linear.data_lin2;
        cout << "OK6" << endl;
        vector<UnifiedCiphertext> h3;
        linear.linear_2(
            he,
            softmax_V_result, 
            linear.pp_2[layer_id],
            data,
            h3
        );
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
                he->evaluator->rescale_to_next_inplace(h3[i]);
            }
        } else {
            for(int i=0;i<h3.size();i++){
                he->devaluator->add_inplace(h3[i], enc_attention_residual[i]);
                he->devaluator->rescale_to_next_inplace(h3[i]);
            }
        }
        attention_input_residual_ready = false;
        cout << "OK1" << endl;
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
                he->evaluator->rescale_to_next_inplace(h3[i]);
            } else {
                he->devaluator->multiply_plain_inplace(h3[i],scale_up);
                he->devaluator->sub_inplace(h3[i],X_avg[i]);
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
                he->evaluator->rescale_to_next_inplace(sigma[i]);
            } else {
                he->devaluator->square_inplace(sigma[i]);
                he->devaluator->relinearize_inplace(sigma[i], *he->drelin_keys);
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
            const auto &variance_context =
                he->dcontext->get_context_data(variance_chain);
            if (variance_context.parms().coeff_modulus().size() != 2) {
                throw runtime_error(
                    "formal LN1 variance DEVICE level is not two limbs");
            }
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
        he->evaluator->rescale_to_next_inplace(X_sum2);
        // he->evaluator->rescale_to_next_inplace(X_sum2);
        } else {
        he->devaluator->multiply_plain_inplace(X_sum2,inv_D_p2);
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
        for_each_n(h3.begin(), h3.size(),
            [he](auto &ct) { ct.to_host(*he->context, *he->dcontext); });
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
        mask_ckks_to_mpc_inplace(h3_full, x_avg_pts, he);
        send_encrypted_vector(io, h3_full);
        mask_ckks_to_mpc_inplace(enc_sigma, sigma_pts, he);
        send_encrypted_vector(io, enc_sigma);
        cout << "FORMAL_TRANSFER_LAYOUT stage=softmax_v_ln_output status=PASS"
             << " layout=ordinary_full_slot h3_cts=" << h3_full.size()
             << " sigma_cts=1 half_decode=0" << endl;
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    }
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    // cout << "send and receive cts, comm:" << this->get_comm() << endl;
    // cout << "send and receive cts, round:" << this->get_round() << endl;
    cout << "send and receive done in linear4_add_layernorm_part1" << endl;
    vector<vector<HE_CKKS::scalar_t>> vec_x_avg(x_avg_pts.size());
    vector<vector<HE_CKKS::scalar_t>> vec_sigma(sigma_pts.size());
    start = high_resolution_clock::now();
    vec_x_avg = ckks_to_mpc(x_avg_pts, poly_modulus_degree,
                            he->get_Q(x_avg_pts[0]), bitwidth, s, he, false);
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
    map<string, FixArray> result;
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
std::map<string, FixArray> Bert_ckks::ln_part2_linear3_gelu_part1(
    std::map<string, FixArray> input, int layer_id){
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
    map<string, FixArray> result;
    int bitwidth = x_flat_avg.ell;
    int s = x_flat_avg.s;
    if (bitwidth != NL_ELL || s != NL_SCALE) {
        throw runtime_error(
            "formal LN1 affine/FFN1 requires ell37/s12 shares");
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
    int gelu_input_size = data.image_size*INTER_DIM;
    int gelu_cts_size = gelu_input_size / slot_count;
    vector<Ciphertext> output_cts(
        gelu_cts_size
    );
    vector<Plaintext> output_pts(
        gelu_cts_size
    );
    // The MPC tail selection in gelu_nonlinear needs the true FFN1 output x,
    // not a synthetic constant.  Export a second, independently masked
    // Algorithm-1 branch from linear3_output while retaining the original
    // ciphertext vector for eval_gelu_poly.
    vector<Ciphertext> pre_gelu_cts(gelu_cts_size);
    vector<Plaintext> pre_gelu_pts(gelu_cts_size);
    // The FFN residual is the *post-attention LayerNorm affine output* (H4)
    // in post-norm BERT.  It is exported as an additive share from the same
    // CKKS ciphertext before FFN1; the old pre-LayerNorm h3 transfer is not a
    // valid FFN residual.
    vector<Ciphertext> normalized_residual_cts(num_cts);
    vector<Plaintext> normalized_residual_pts(num_cts);
    cout << "gelu_cts_size:" << gelu_cts_size << endl;
    auto start = high_resolution_clock::now();
    import_rns_target_scale(*this, vec_x_flat_avg, slot_count, bitwidth, s,
                            he->he_scale, he, pts_x_flat_avg,
                            "ln1_centered_input");
    import_rns_target_scale(*this, vec_sigma_flat, slot_count, bitwidth, s,
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
    cout << "cts_x_flat.size():" << cts_x_flat.size() << endl;
    cout << "cts_sigma_flat.size():" << cts_sigma_flat.size() << endl;
    if (party == BOB) {
        cout << "send_encrypted_vector" << endl;
        #pragma omp parallel for
        for(size_t i=0;i<num_cts;i++){
            he->encryptor->encrypt(pts_x_flat_avg[i], cts_x_flat[i]);
            he->encryptor->encrypt(pts_sigma_flat[i], cts_sigma_flat[i]);
        }
        send_encrypted_vector(io, cts_x_flat);
        send_encrypted_vector(io, cts_sigma_flat);
        cout << "FORMAL_TRANSFER_LAYOUT stage=ln_ffn_input status=PASS"
             << " layout=ordinary_full_slot x_cts=" << num_cts
             << " sigma_cts=" << num_cts << " imag_repack=0" << endl;
        cout << "send_encrypted_vector done" << endl;
		// send_encrypted_vector(io, cts);
		// vector<Ciphertext> share_client(dim);
        recv_encrypted_vector(he->context, io, normalized_residual_cts);
        #pragma omp parallel for
        for (int i = 0; i < normalized_residual_cts.size(); ++i) {
            he->decryptor->decrypt(normalized_residual_cts[i],
                                   normalized_residual_pts[i]);
        }
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
    else{
        start = high_resolution_clock::now();
        recv_encrypted_vector(he->context, io, cts_x_flat);
        recv_encrypted_vector(he->context, io, cts_sigma_flat);
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
             << " layout=ordinary_full_slot x_cts=" << cts_x_flat.size()
             << " sigma_cts=" << cts_sigma_flat.size()
             << " imag_repack=0" << endl;
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
                w_matrix[i][j] = layernorm_fixed_to_real(
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
            } else {
                he->devaluator->add_plain_inplace(cts_x_flat[i],
                                                  pts_x_flat_avg[i]);
                he->devaluator->add_plain_inplace(cts_sigma_flat[i],
                                                  pts_sigma_flat[i]);
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
             << endl;

        // get x_avg and sigma, compute (x_avg * sigma) * w + b
        #pragma omp parallel for
        for(int i=0;i<cts_x_flat.size();i++){
            if (!he->is_gpu_enable()) {
                he->evaluator->multiply_inplace(cts_x_flat[i],
                                                cts_sigma_flat[i]);
                he->evaluator->relinearize_inplace(cts_x_flat[i],
                                                   *he->relin_keys);
                he->evaluator->multiply_plain_inplace(cts_x_flat[i], tmp_w[i]);
                he->evaluator->add_plain_inplace(cts_x_flat[i], tmp_b[i]);
                he->evaluator->rescale_to_next_inplace(cts_x_flat[i]);
            } else {
                he->devaluator->multiply_inplace(cts_x_flat[i],
                                                 cts_sigma_flat[i]);
                he->devaluator->relinearize_inplace(cts_x_flat[i],
                                                    *he->drelin_keys);
                he->devaluator->multiply_plain_inplace(cts_x_flat[i], tmp_w[i]);
                he->devaluator->add_plain_inplace(cts_x_flat[i], tmp_b[i]);
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
        prepare_host_conversion_from_actual_scale(
            *this, normalized_residual_cts,
            kFormalConversionScaleBits, he,
            "ln1_affine_residual");
        mask_ckks_to_mpc_inplace(normalized_residual_cts,
                                 normalized_residual_pts, he);
        send_encrypted_vector(io, normalized_residual_cts);
        /*
            计算linear3
        */
        FCMetadata data = linear.data_lin3;
        vector<UnifiedCiphertext> linear3_output;
        linear.linear_2(
            he,
            cts_x_flat, 
            linear.pp_3[layer_id],
            data,
            linear3_output
        );
        cout << "linear2 done" << endl;
        cout << "scale:" << log2(linear3_output[0].scale()) << endl;

        #pragma omp parallel for
        for(int i=0;i<linear3_output.size();i++){
            if (!he->is_gpu_enable()) {
            he->evaluator->rescale_to_next_inplace(linear3_output[i]);
            he->evaluator->relinearize_inplace(linear3_output[i], *he->relin_keys); // FIXME: @relin
            } else {
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
             << " masked_algorithm1=1 half_decode=0"
             << " intermediate_tensor_opened=0" << endl;
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
    vector<vector<HE_CKKS::scalar_t>> vec_pre_gelu(pre_gelu_pts.size());
    vector<vector<HE_CKKS::scalar_t>> vec_output(output_pts.size());
    start = high_resolution_clock::now();
    vec_pre_gelu = ckks_to_mpc(
        pre_gelu_pts, poly_modulus_degree, he->get_Q(pre_gelu_pts[0]),
        bitwidth, s, he, false);
    vec_output = ckks_to_mpc(output_pts, poly_modulus_degree,
                             he->get_Q(output_pts[0]), bitwidth, s, he, false);
    
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
std::map<string, FixArray> Bert_ckks::linear4_add_layernorm_part1(
    FixArray input, int layer_id){
    cout << "in linear4_add_layernorm_part1" << endl;
    FCMetadata data = linear.data_lin4;
    HE_CKKS* he = linear.he4;
    int bitwidth,s;
    bitwidth = input.ell;
    s = input.s;
    int gelu_input_size = data.image_size * data.filter_h;
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
            "formal FFN2 residual metadata must match ell37/s12 input");
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
    import_rns_target_scale(*this, h6, slot, bitwidth, s, he->he_scale,
                            he, h6_pts, "ffn2_gelu_input");
    vector<UnifiedPlaintext> residual_pts;
    import_rns_target_scale(*this, residual_share, slot, bitwidth, s,
                            he->he_scale, he, residual_pts,
                            "ffn2_residual_input");
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    
    vector<Plaintext> x_avg_pts(num_cts_ln);
    vector<Plaintext> sigma_pts(1);
    vector<Plaintext> block_output_pts(num_cts_ln);
    vector<Ciphertext> enc_x_avg(num_cts_ln);
    vector<Ciphertext> enc_sigma(1);
    vector<Ciphertext> enc_block_output(num_cts_ln);
    vector<UnifiedCiphertext> enc_h6(num_cts_gelu,
                                     UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> enc_residual(num_cts_ln,
                                           UnifiedCiphertext(LOCATION::HOST));
    if (party==BOB){
        #pragma omp parallel for
        for(int i=0;i<num_cts_gelu;i++){
            he->encryptor->encrypt(h6_pts[i], enc_h6[i]);
        }
        send_encrypted_vector(io, enc_h6);
        #pragma omp parallel for
        for (int i = 0; i < num_cts_ln; ++i) {
            he->encryptor->encrypt(residual_pts[i], enc_residual[i]);
        }
        send_encrypted_vector(io, enc_residual);
        cout << "FORMAL_TRANSFER_LAYOUT stage=ffn2_input status=PASS"
             << " layout=ordinary_full_slot h6_cts=" << num_cts_gelu
             << " residual_cts=" << num_cts_ln << " imag_repack=0" << endl;
        cout << "send_encrypted_vector done" << endl;
        recv_encrypted_vector(he->context, io, enc_block_output);
        recv_encrypted_vector(he->context, io, enc_x_avg);
        recv_encrypted_vector(he->context, io, enc_sigma);
        #pragma omp parallel for
        for(int i=0;i<enc_x_avg.size();i++){
            he->decryptor->decrypt(enc_block_output[i], block_output_pts[i]);
            he->decryptor->decrypt(enc_x_avg[i], x_avg_pts[i]);
        }
        he->decryptor->decrypt(enc_sigma[0], sigma_pts[0]);
    }
    else{
        LOCATION loc = he->backend();

        start = high_resolution_clock::now();
        recv_encrypted_vector(he->context, io, enc_h6);
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
             << " layout=ordinary_full_slot h6_cts=" << enc_h6.size()
             << " residual_cts=" << enc_residual.size()
             << " imag_repack=0" << endl;
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
        linear.linear_2(
            he,
            enc_h6, 
            linear.pp_4[layer_id],
            data,
            h7
        );
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
        vector<double> inv_D(slot, 1.0 / COMMON_DIM);
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
        uint64_t final_centered_drop_modulus = 0;
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
            for (int step = 0; step < 3; ++step) {
                centered_context = centered_context->next_context_data();
                if (!centered_context) {
                    throw runtime_error(
                        "formal FFN2 final centered rescale level is unavailable");
                }
            }
            final_centered_drop_modulus =
                centered_context->parms().coeff_modulus().back().value();
        } else {
            const auto centered_chain = h7[0].dcipher().chain_index();
            const auto &centered_context =
                he->dcontext->get_context_data(centered_chain);
            const auto &final_centered_context =
                he->dcontext->get_context_data(centered_chain + 3);
            centered_drop_modulus =
                centered_context.parms().coeff_modulus().back().value();
            final_centered_drop_modulus = final_centered_context.parms()
                                                  .coeff_modulus()
                                                  .back()
                                                  .value();
        }
        const double centered_target_scale =
            ldexp(1.0, kFormalConversionScaleBits);
        const double centered_plain_scale =
            centered_target_scale *
            static_cast<double>(centered_drop_modulus) *
            static_cast<double>(final_centered_drop_modulus) /
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
             << " final_drop_bits="
             << log2(static_cast<double>(final_centered_drop_modulus))
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
        // Preserve the actual FFN residual output before centering.  The next
        // pre-norm ViT block consumes its normalized form for Q/K/V but must
        // add this unnormalized tensor on the residual branch.
        for (int i = 0; i < num_cts_ln; ++i) {
            if (he->is_gpu_enable()) {
                UnifiedCiphertext::to_host(*he->dcontext, h7[i].dcipher(),
                                           *he->context,
                                           enc_block_output[i]);
            } else {
                enc_block_output[i] = h7[i].hcipher();
            }
        }
        prepare_host_conversion_from_actual_scale(
            *this, enc_block_output, kFormalConversionScaleBits, he,
            "vit_block_output");
        mask_ckks_to_mpc_inplace(enc_block_output, block_output_pts, he);
        send_encrypted_vector(io, enc_block_output);
        cout << "VIT_BLOCK_RESIDUAL_EXPORT status=PASS layer=" << layer_id
             << " cts=" << enc_block_output.size()
             << " plaintext_recovery=0" << endl;
        cout << "OK1" << endl;
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
            he->evaluator->rescale_to_next_inplace(h7[i]);
            } else {
            he->devaluator->multiply_plain_inplace(h7[i],scale_up);
            he->devaluator->sub_inplace(h7[i],X_avg[i]);
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
        #pragma omp parallel for
        for(int i=0;i<num_cts_ln;i++){
            if (!he->is_gpu_enable()) {
            he->evaluator->square_inplace(sigma[i]);
            he->evaluator->relinearize_inplace(sigma[i], *he->relin_keys);
            he->evaluator->mod_switch_to_next_inplace(h7[i]);
            he->evaluator->mod_switch_to_next_inplace(h7[i]);
            he->evaluator->rescale_to_next_inplace(h7[i]);
            } else {
            he->devaluator->square_inplace(sigma[i]);
            he->devaluator->relinearize_inplace(sigma[i], *he->drelin_keys);
            he->devaluator->mod_switch_to_next_inplace(h7[i]);
            he->devaluator->mod_switch_to_next_inplace(h7[i]);
            he->devaluator->rescale_to_next_inplace(h7[i]);
            }
        }
        require_uniform_unified_metadata(h7, "ln2_centered_output");
        const double centered_scale_error =
            abs(h7.front().scale() - centered_target_scale) /
            centered_target_scale;
        if (h7.front().coeff_modulus_size() != 1 ||
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
            const auto &variance_context_1 =
                he->dcontext->get_context_data(variance_chain);
            const auto &variance_context_2 =
                he->dcontext->get_context_data(variance_chain + 1);
            if (variance_context_1.parms().coeff_modulus().size() < 3) {
                throw runtime_error(
                    "formal LN2 DEVICE variance requires two rescale levels");
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
            he->evaluator->rescale_to_next_inplace(X_sum2);
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
            he->devaluator->rescale_to_next_inplace(X_sum2);
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
        mask_ckks_to_mpc_inplace(h7_full, x_avg_pts, he);
        mask_ckks_to_mpc_inplace(enc_sigma, sigma_pts, he);
        send_encrypted_vector(io, h7_full);
        send_encrypted_vector(io, enc_sigma);
        cout << "FORMAL_TRANSFER_LAYOUT stage=ffn2_ln_output status=PASS"
             << " layout=ordinary_full_slot h7_cts=" << h7_full.size()
             << " sigma_cts=1 half_decode=0" << endl;
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    }
    // cout << "send and receive cts, comm:" << this->get_comm() << endl;
    // cout << "send and receive cts, round:" << this->get_round() << endl;
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    cout << "send and receive done in linear4_add_layernorm_part1" << endl;
    vector<vector<HE_CKKS::scalar_t>> vec_block_output(block_output_pts.size());
    vector<vector<HE_CKKS::scalar_t>> vec_x_avg(x_avg_pts.size());
    vector<vector<HE_CKKS::scalar_t>> vec_sigma(sigma_pts.size());
    start = high_resolution_clock::now();
    vec_block_output = ckks_to_mpc(
        block_output_pts, poly_modulus_degree,
        he->get_Q(block_output_pts[0]), bitwidth, s, he, false);
    vec_x_avg = ckks_to_mpc(x_avg_pts, poly_modulus_degree,
                            he->get_Q(x_avg_pts[0]), bitwidth, s, he, false);
    vec_sigma = ckks_to_mpc(sigma_pts, poly_modulus_degree,
                            he->get_Q(sigma_pts[0]), bitwidth, s, he, false);
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    cout << "vec_sigma.size():" << vec_sigma.size() << ", should be 1" << endl;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    // cout << "mpc_to_ckks comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks round:" << this->get_round() << endl;
    cout << "bitwidth,s:" << bitwidth << "," << s << endl;
    FixArray block_output, x_flat_avg, sigma_flat;
    FixArray block_output_packed(
        party, num_cts_ln * slot, true, bitwidth, s);
    FixArray x_flat_avg_packed(
        party, num_cts_ln * slot, true, bitwidth, s);
    sigma_flat = FixArray(
        party, data.image_size, true, bitwidth, s);
    for (int i = 0; i < static_cast<int>(vec_block_output.size()); ++i) {
        for (int j = 0; j < static_cast<int>(vec_block_output[i].size()); ++j) {
            block_output_packed.data[i * vec_block_output[i].size() + j] =
                vec_block_output[i][j];
        }
    }
    block_output = bsgs_output_to_token_major(
        block_output_packed, data, "vit_block_output_to_mpc");
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
    map<string, FixArray> result;
    result["block_output"] = block_output;
    result["x_flat_avg"] = x_flat_avg;
    result["sigma_flat"] = sigma_flat;
    return result;
}
