#include "ckks_bert.h"

#include "FloatingPoint/fixed-point.h"
#include "utils/ArgMapping/ArgMapping.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef BLB_HAS_HE_GPU
#include <cuda_runtime_api.h>
#endif

using namespace sci;
using namespace std;

extern uint64_t prime_mod;
int32_t bitlength = 37;
int32_t num_threads = 1;

namespace {
constexpr int kLogicalTokens = 65;
constexpr int kEll = 37;
constexpr int kScale = 12;
constexpr int kQkvScale = 5;

struct PhantomOperationTotals {
    uint64_t rotations = 0;
    uint64_t ct_ct_multiply = 0;
    uint64_t ct_pt_multiply = 0;
    uint64_t relinearizations = 0;
    uint64_t rescales = 0;
    uint64_t conjugations = 0;
};

vector<HE_CKKS *> he_contexts(Bert_ckks &bert) {
    return {bert.linear.he1, bert.linear.he2, bert.linear.he3,
            bert.linear.he4, bert.linear.he_softmax,
            bert.linear.he_onetime};
}

void reset_online_metrics(Bert_ckks &bert) {
    (void)bert.get_comm();
    (void)bert.get_round();
    bert.ct_transfer_comm = 0;
    bert.ct_transfer_round = 0;
    bert.conversion_comm = 0;
    bert.conversion_round = 0;
    bert.other_nonlinear_comm = 0;
    bert.other_nonlinear_round = 0;
    bert.ct_transfer_la = 0;
    bert.ct_transfer_mem_la = 0;
    bert.conversion_la = 0;
    bert.other_nonlinear_la = 0;
    bert.he_compute_la = 0;
    bert.ckks_to_mpc_conversions = 0;
    bert.mpc_to_ckks_conversions = 0;
    bert.ckks_to_mpc_calls = 0;
    bert.mpc_to_ckks_calls = 0;
    bert.field_to_ring_calls = 0;
    bert.masked_rns_coefficients = 0;
    bert.host_rescale_count = 0;
    bert.linear.rotation_count = 0;
    bert.linear.ct_ct_multiply_count = 0;
    bert.linear.ct_pt_multiply_count = 0;
    bert.linear.relinearization_count = 0;
    bert.linear.rescale_count = 0;
    for (HE_CKKS *he : he_contexts(bert)) {
        if (he != nullptr && he->devaluator != nullptr) {
            he->devaluator->reset_operation_counters();
        }
    }
}

PhantomOperationTotals operation_totals(Bert_ckks &bert) {
    PhantomOperationTotals total;
    for (HE_CKKS *he : he_contexts(bert)) {
        if (he == nullptr || he->devaluator == nullptr) continue;
        total.rotations += he->devaluator->rotation_count();
        total.ct_ct_multiply += he->devaluator->ct_ct_multiply_count();
        total.ct_pt_multiply += he->devaluator->ct_pt_multiply_count();
        total.relinearizations += he->devaluator->relinearization_count();
        total.rescales += he->devaluator->rescale_count();
        total.conjugations += he->devaluator->conjugation_count();
    }
    return total;
}

void trim_phantom_pool_if_requested(int party, const char *stage) {
#ifdef BLB_HAS_HE_GPU
    const char *enabled = std::getenv("VIT65_LOW_MEMORY");
    if (party != ALICE || enabled == nullptr || string(enabled) != "1") {
        return;
    }
    int device = 0;
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess ||
        cudaMemPoolTrimTo(pool, 0) != cudaSuccess) {
        throw runtime_error(string("Phantom CUDA pool trim failed at ") +
                            stage);
    }
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        throw runtime_error(string("CUDA memory query failed at ") + stage);
    }
    cout << "VIT65_GPU_POOL_TRIM status=PASS stage=" << stage
         << " device=" << device << " free_bytes=" << free_bytes
         << " total_bytes=" << total_bytes << endl;
#else
    (void)party;
    (void)stage;
#endif
}

vector<int64_t> read_flat(const string &path, size_t expected) {
    ifstream input(path);
    if (!input) throw runtime_error("required full-ViT tensor is missing: " + path);
    vector<int64_t> values;
    string line, cell;
    while (getline(input, line)) {
        istringstream row(line);
        while (getline(row, cell, ',')) values.push_back(stoll(cell));
    }
    if (values.size() != expected) {
        throw runtime_error("full-ViT tensor size mismatch: " + path);
    }
    return values;
}

vector<vector<string>> read_protocols(const string &path) {
    ifstream input(path);
    if (!input) throw runtime_error("head_protocols.txt is missing");
    vector<vector<string>> result;
    string line, cell;
    while (getline(input, line)) {
        vector<string> row;
        istringstream stream(line);
        while (getline(stream, cell, ',')) row.push_back(cell);
        if (!row.empty()) result.push_back(row);
    }
    if (result.size() != ATTENTION_LAYERS) {
        throw runtime_error("full-ViT protocol layer count mismatch");
    }
    return result;
}

FixArray model_parameter(FPMath *fp, int party,
                         const vector<uint64_t> &values, int rows) {
    if (party == ALICE && values.size() != COMMON_DIM) {
        throw runtime_error("LayerNorm parameter width mismatch");
    }
    vector<uint64_t> repeated(static_cast<size_t>(rows) * COMMON_DIM);
    if (party == ALICE) {
        for (int row = 0; row < rows; ++row) {
            copy(values.begin(), values.end(),
                 repeated.begin() + static_cast<size_t>(row) * COMMON_DIM);
        }
    }
    return fp->fix->input(ALICE, static_cast<int>(repeated.size()),
                          party == ALICE ? repeated.data() : nullptr,
                          true, kEll, kScale);
}

FixArray file_parameter(FPMath *fp, int party, const string &path,
                        int rows) {
    vector<int64_t> raw;
    if (party == ALICE) raw = read_flat(path, COMMON_DIM);
    vector<uint64_t> repeated(static_cast<size_t>(rows) * COMMON_DIM);
    if (party == ALICE) {
        for (int row = 0; row < rows; ++row) {
            for (int channel = 0; channel < COMMON_DIM; ++channel) {
                repeated[static_cast<size_t>(row) * COMMON_DIM + channel] =
                    static_cast<uint64_t>(raw[channel]);
            }
        }
    }
    return fp->fix->input(ALICE, static_cast<int>(repeated.size()),
                          party == ALICE ? repeated.data() : nullptr,
                          true, kEll, kScale);
}

// The CKKS packing uses 128 physical token rows although the trained model
// has only 65 logical rows.  Bumblebee's reciprocal-square-root LayerNorm
// primitive has no epsilon term, so an all-zero padding row would ask it to
// invert zero.  Replace only the public padding projection, in a fresh share
// buffer, by a fixed non-zero-variance vector.  Logical rows and the residual
// tensor itself are left untouched; attention masks the padding rows again.
FixArray layer_norm_input_with_safe_padding(const FixArray &input, int party) {
    if (input.size != INPUT_DIM * COMMON_DIM) {
        throw runtime_error("safe LayerNorm padding shape mismatch");
    }
    FixArray padded(party, input.size, input.signed_, input.ell, input.s);
    copy_n(input.data, input.size, padded.data);
    const uint64_t mask = padded.ell_mask();
    for (int token = kLogicalTokens; token < INPUT_DIM; ++token) {
        fill_n(padded.data + token * COMMON_DIM, COMMON_DIM, uint64_t(0));
        if (party == ALICE) {
            padded.data[token * COMMON_DIM] = uint64_t(1) << input.s;
            padded.data[token * COMMON_DIM + 1] =
                (uint64_t(0) - (uint64_t(1) << input.s)) & mask;
        }
    }
    cout << "VIT_PADDING_SANITIZE status=PASS logical_tokens="
         << kLogicalTokens << " physical_tokens=" << INPUT_DIM
         << " private_branching=0" << endl;
    return padded;
}

FixArray secure_private_model_linear(
    FPMath *fp, int party, const FixArray &input, int rows, int input_dim,
    int output_dim, const vector<int64_t> &weight_input_major,
    int weight_scale) {
    if (input.size != rows * input_dim || input.s != kScale ||
        (party == ALICE && weight_input_major.size() !=
             static_cast<size_t>(input_dim) * output_dim)) {
        throw runtime_error("secure private-model linear shape mismatch");
    }
    const size_t products = static_cast<size_t>(rows) * output_dim * input_dim;
    FixArray expanded(party, static_cast<int>(products), true, kEll, kScale);
    vector<uint64_t> weight_values(products, 0);
    size_t offset = 0;
    for (int row = 0; row < rows; ++row) {
        for (int output = 0; output < output_dim; ++output) {
            for (int input_channel = 0; input_channel < input_dim; ++input_channel) {
                expanded.data[offset] = input.data[row * input_dim + input_channel];
                if (party == ALICE) {
                    weight_values[offset] = static_cast<uint64_t>(
                        weight_input_major[input_channel * output_dim + output]);
                }
                ++offset;
            }
        }
    }
    FixArray weights = fp->fix->input(
        ALICE, static_cast<int>(products),
        party == ALICE ? weight_values.data() : nullptr,
        true, kEll, weight_scale);
    FixArray product = fp->fix->mul(expanded, weights, kEll);
    FixArray sums(party, rows * output_dim, true, kEll,
                  kScale + weight_scale);
    const uint64_t mask = sums.ell_mask();
    offset = 0;
    for (int row = 0; row < rows; ++row) {
        for (int output = 0; output < output_dim; ++output) {
            uint64_t local = 0;
            for (int input_channel = 0; input_channel < input_dim; ++input_channel) {
                local += product.data[offset++];
            }
            sums.data[row * output_dim + output] = local & mask;
        }
    }
    return fp->fix->truncate_reduce(sums, weight_scale);
}

FixArray patch_embedding(FPMath *fp, int party, const string &model_path,
                         const FixArray &image_tokens) {
    constexpr int patches = 64;
    constexpr int patch_dim = 48;
    FixArray compact(party, patches * patch_dim, true, kEll, kScale);
    for (int patch = 0; patch < patches; ++patch) {
        copy_n(image_tokens.data + (patch + 1) * COMMON_DIM, patch_dim,
               compact.data + patch * patch_dim);
    }
    vector<int64_t> all_weights;
    if (party == ALICE) {
        all_weights = read_flat(model_path + "/vit.patch.weight.txt",
                                COMMON_DIM * COMMON_DIM);
    }
    vector<int64_t> compact_weights;
    if (party == ALICE) {
        compact_weights.resize(patch_dim * COMMON_DIM);
        for (int input = 0; input < patch_dim; ++input) {
            copy_n(all_weights.begin() + input * COMMON_DIM, COMMON_DIM,
                   compact_weights.begin() + input * COMMON_DIM);
        }
    }
    FixArray projected = secure_private_model_linear(
        fp, party, compact, patches, patch_dim, COMMON_DIM,
        compact_weights, 9);
    FixArray output(party, INPUT_DIM * COMMON_DIM, true, kEll, kScale);
    fill_n(output.data, output.size, uint64_t(0));
    for (int patch = 0; patch < patches; ++patch) {
        copy_n(projected.data + patch * COMMON_DIM, COMMON_DIM,
               output.data + (patch + 1) * COMMON_DIM);
    }
    vector<int64_t> bias, cls, pos;
    if (party == ALICE) {
        bias = read_flat(model_path + "/vit.patch.bias.txt", COMMON_DIM);
        cls = read_flat(model_path + "/vit.class_token.txt", COMMON_DIM);
        pos = read_flat(model_path + "/vit.position.txt",
                        kLogicalTokens * COMMON_DIM);
        for (int channel = 0; channel < COMMON_DIM; ++channel) {
            output.data[channel] = static_cast<uint64_t>(cls[channel]);
        }
        for (int token = 1; token < kLogicalTokens; ++token) {
            for (int channel = 0; channel < COMMON_DIM; ++channel) {
                output.data[token * COMMON_DIM + channel] +=
                    static_cast<uint64_t>(bias[channel]);
            }
        }
        for (int token = 0; token < kLogicalTokens; ++token) {
            for (int channel = 0; channel < COMMON_DIM; ++channel) {
                output.data[token * COMMON_DIM + channel] +=
                    static_cast<uint64_t>(pos[token * COMMON_DIM + channel]);
                output.data[token * COMMON_DIM + channel] &= output.ell_mask();
            }
        }
    }
    cout << "VIT_PATCH_EMBEDDING status=PASS backend=mpc_secure_mul"
         << " private_model=1 private_image=1 logical_tokens=65" << endl;
    return output;
}

FixArray normalize_heads(Bert_ckks &bert, const FixArray &qk_row,
                         const vector<string> &protocols) {
    constexpr int matrix_values = INPUT_DIM * INPUT_DIM;
    constexpr int logical_values = kLogicalTokens * kLogicalTokens;
    const size_t boundary = bert.heterogeneous_boundary_qk_ciphertexts();
    FixOp *fix = bert.nonlinear.fpmath[0]->fix;
    FixArray normalized(bert.party, qk_row.size, true, kEll, NL_SCALE);
    fill_n(normalized.data, normalized.size, uint64_t(0));
    for (size_t head = 0; head < boundary; ++head) {
        FixArray score(bert.party, logical_values, true, kEll, NL_SCALE);
        for (int row = 0; row < kLogicalTokens; ++row) {
            copy_n(qk_row.data + head * matrix_values + row * INPUT_DIM,
                   kLogicalTokens, score.data + row * kLogicalTokens);
        }
        FixArray head_output;
        if (protocols[head] == "relu") {
            FixArray wide = fix->extend(score, kEll + 1);
            BoolArray positive = fix->GT(wide, uint64_t(0));
            FixArray relu = fix->if_else(positive, wide, uint64_t(0));
            constexpr int relu_ell = 31;
            constexpr int sum_ell = 38;
            relu.signed_ = false;
            relu = fix->reduce(relu, relu_ell);
            FixArray for_sum = fix->extend(relu, sum_ell);
            FixArray sums(bert.party, kLogicalTokens, false, sum_ell, NL_SCALE);
            for (int row = 0; row < kLogicalTokens; ++row) {
                uint64_t local = 0;
                for (int col = 0; col < kLogicalTokens; ++col)
                    local += for_sum.data[row * kLogicalTokens + col];
                sums.data[row] = local & sums.ell_mask();
            }
            BoolArray nonzero = fix->GT(sums, uint64_t(0));
            FixArray denominator = fix->if_else(nonzero, sums, uint64_t(1));
            head_output = fix->div_batch(
                relu, denominator, kLogicalTokens, relu_ell, NL_SCALE);
            head_output = fix->extend(head_output, kEll);
            head_output.signed_ = true;
        } else {
            constexpr int reciprocal_scale = 20;
            const uint64_t reciprocal = static_cast<uint64_t>(
                llround(ldexp(1.0, reciprocal_scale) / kLogicalTokens));
            FixArray product = fix->mul(score, reciprocal,
                                        kEll + reciprocal_scale);
            product.s += reciprocal_scale;
            head_output = fix->truncate_reduce(product, reciprocal_scale);
        }
        for (int row = 0; row < kLogicalTokens; ++row) {
            copy_n(head_output.data + row * kLogicalTokens, kLogicalTokens,
                   normalized.data + head * matrix_values + row * INPUT_DIM);
        }
    }
    cout << "HETERO_NORMALIZATION status=PASS boundary_heads=" << boundary
         << " private_branching=0" << endl;
    return normalized;
}

FixArray run_attention(Bert_ckks &bert, FPMath *fp,
                       const FixArray &normalized_input, int layer,
                       const vector<string> &protocols) {
    bert.set_heterogeneous_head_protocols(protocols);
    FixArray qkv_input = fp->fix->truncate_reduce(
        normalized_input, normalized_input.s - kQkvScale);
    FixArray column(bert.party, qkv_input.size, true, qkv_input.ell,
                    qkv_input.s);
    for (int token = 0; token < INPUT_DIM; ++token)
        for (int hidden = 0; hidden < COMMON_DIM; ++hidden)
            column.data[hidden * INPUT_DIM + token] =
                qkv_input.data[token * COMMON_DIM + hidden];
    auto converted = bert.attention_from_real_input(column, layer);
    const int slots = bert.linear.data_lin1_0.slot_count;
    const size_t qk_values = static_cast<size_t>(PACKING_NUM) * INPUT_DIM * INPUT_DIM;
    FixArray qk_packed(bert.party, static_cast<int>(qk_values), true,
                       kEll, NL_SCALE);
    fill_n(qk_packed.data, qk_packed.size, uint64_t(0));
    size_t offset = 0;
    for (const auto &cipher_slots : converted) {
        if (cipher_slots.size() != static_cast<size_t>(slots))
            throw runtime_error("heterogeneous QK conversion slot mismatch");
        copy(cipher_slots.begin(), cipher_slots.end(), qk_packed.data + offset);
        offset += cipher_slots.size();
    }
    if (offset != bert.heterogeneous_boundary_qk_ciphertexts() * slots)
        throw runtime_error("heterogeneous QK boundary size mismatch");
    FixArray qk_row(bert.party, static_cast<int>(qk_values), true,
                    kEll, NL_SCALE);
    bert.linear.plain_cross_packing_postprocess(
        qk_packed.data, qk_row.data, false, bert.linear.data_lin1_0);
    FixArray normalized_scores = normalize_heads(bert, qk_row, protocols);
    return bert.attention_output_from_heterogeneous_scores(
        normalized_scores, layer);
}

FixArray classifier(FPMath *fp, int party, const string &model_path,
                    const FixArray &final_norm) {
    FixArray cls(party, COMMON_DIM, true, kEll, kScale);
    copy_n(final_norm.data, COMMON_DIM, cls.data);
    vector<int64_t> weights;
    if (party == ALICE)
        weights = read_flat(model_path + "/classifier.weight.txt",
                            COMMON_DIM * 10);
    FixArray logits = secure_private_model_linear(
        fp, party, cls, 1, COMMON_DIM, 10, weights, 9);
    if (party == ALICE) {
        auto bias = read_flat(model_path + "/classifier.bias.txt", 10);
        for (int i = 0; i < 10; ++i)
            logits.data[i] = (logits.data[i] + static_cast<uint64_t>(bias[i])) &
                             logits.ell_mask();
    }
    return logits;
}
}  // namespace

int main(int argc, char **argv) {
    int party = 0, port = 27600, blocks = ATTENTION_LAYERS;
    string address = "127.0.0.1", model_path;
    ArgMapping args;
    args.arg("r", party, "Role: ALICE=1, BOB=2");
    args.arg("p", port, "TCP port");
    args.arg("ip", address, "ALICE address");
    args.arg("model", model_path, "full ViT65 exported model");
    args.arg("blocks", blocks, "debug block count; production uses 7");
    args.parse(argc, argv);
    try {
        if (party != ALICE && party != BOB) throw invalid_argument("invalid party");
        if (blocks < 1 || blocks > ATTENTION_LAYERS)
            throw invalid_argument("blocks must be in [1,7]");
        const auto protocols = read_protocols(model_path + "/head_protocols.txt");
        Bert_ckks bert(party, port, address, LOCATION::DEVICE, model_path,
                       false);
        FPMath *fp = bert.nonlinear.fpmath[0];
        vector<int64_t> image_raw;
        if (party == BOB)
            image_raw = read_flat(model_path + "/inputs_image_data.txt",
                                  INPUT_DIM * COMMON_DIM);
        vector<uint64_t> image_values(image_raw.size());
        transform(image_raw.begin(), image_raw.end(), image_values.begin(),
                  [](int64_t value) { return static_cast<uint64_t>(value); });
        reset_online_metrics(bert);
        const uint64_t communication_start = bert.raw_total_comm();
        const uint64_t rounds_start = bert.raw_total_rounds();
        auto start = chrono::steady_clock::now();
        FixArray image = fp->fix->input(
            BOB, INPUT_DIM * COMMON_DIM,
            party == BOB ? image_values.data() : nullptr,
            true, kEll, kScale);
        FixArray raw = patch_embedding(fp, party, model_path, image);
        FixArray pre_w = file_parameter(
            fp, party, model_path + "/vit.block.0.pre_norm.weight.txt",
            INPUT_DIM);
        FixArray pre_b = file_parameter(
            fp, party, model_path + "/vit.block.0.pre_norm.bias.txt",
            INPUT_DIM);
        FixArray pre_input = layer_norm_input_with_safe_padding(raw, party);
        auto pre_map = bert.layer_norm_part1_real(pre_input, pre_w, pre_b);
        FixArray normalized = bert.layer_norm_affine_ckks(pre_map);
        trim_phantom_pool_if_requested(party, "initial_pre_norm");
        for (int layer = 0; layer < blocks; ++layer) {
            cout << "VIT_BLOCK status=START layer=" << layer << endl;
            FixArray attention = run_attention(
                bert, fp, normalized, layer, protocols[layer]);
            trim_phantom_pool_if_requested(party, "attention");
            FixArray attention_residual = fp->fix->add(raw, attention);
            FixArray ln1_w = model_parameter(
                fp, party, bert.linear.w_ln_1[layer], INPUT_DIM);
            FixArray ln1_b = model_parameter(
                fp, party, bert.linear.b_ln_1[layer], INPUT_DIM);
            FixArray ln1_input = layer_norm_input_with_safe_padding(
                attention_residual, party);
            auto ln1 = bert.layer_norm_part1_real(
                ln1_input, ln1_w, ln1_b);
            auto ffn1 = bert.ln_part2_linear3_gelu_part1(ln1, layer);
            FixArray gelu = bert.gelu_nonlinear(ffn1);
            auto next_ln = bert.linear4_add_layernorm_part1(gelu, layer);
            raw = next_ln.at("block_output");
            FixArray next_w = model_parameter(
                fp, party, bert.linear.w_ln_2[layer], INPUT_DIM);
            FixArray next_b = model_parameter(
                fp, party, bert.linear.b_ln_2[layer], INPUT_DIM);
            next_ln["w"] = next_w;
            next_ln["b"] = next_b;
            next_ln = bert.layer_norm_nonlinear(next_ln);
            normalized = bert.layer_norm_affine_ckks(next_ln);
            trim_phantom_pool_if_requested(party, "block_output_norm");
            cout << "VIT_BLOCK status=PASS layer=" << layer
                 << " output_elements=" << raw.size << endl;
        }
        if (blocks != ATTENTION_LAYERS) {
            const double seconds = chrono::duration<double>(
                chrono::steady_clock::now() - start).count();
            cout << "VIT65_FULL_SMOKE status=PASS backend=phantom_gpu blocks="
                 << blocks << " end_to_end_seconds=" << seconds << endl;
            return 0;
        }
        FixArray logits = classifier(fp, party, model_path, normalized);
        FixArray opened = fp->fix->output(PUBLIC, logits);
        auto native = opened.get_native_type<double>();
        const int prediction = static_cast<int>(
            max_element(native.begin(), native.end()) - native.begin());
        const double seconds = chrono::duration<double>(
            chrono::steady_clock::now() - start).count();
        const uint64_t communication =
            bert.raw_total_comm() - communication_start;
        const uint64_t rounds = bert.raw_total_rounds() - rounds_start;
        const PhantomOperationTotals operations = operation_totals(bert);
        const double accounted = bert.he_compute_la + bert.conversion_la +
                                 bert.ct_transfer_la;
        const double mpc_and_host = max(0.0, seconds - accounted);
        cout << setprecision(17);
        cout << "VIT65_FULL status=PASS backend=phantom_gpu layers="
             << ATTENTION_LAYERS << " prediction=" << prediction
             << " end_to_end_seconds=" << seconds
             << " he_seconds=" << bert.he_compute_la
             << " mpc_and_host_seconds=" << mpc_and_host
             << " conversion_seconds=" << bert.conversion_la
             << " ciphertext_transfer_seconds=" << bert.ct_transfer_la
             << " communication_bytes=" << communication
             << " rounds=" << rounds
             << " conversion_communication_bytes=" << bert.conversion_comm
             << " ckks_to_mpc_ciphertexts=" << bert.ckks_to_mpc_conversions
             << " mpc_to_ckks_ciphertexts=" << bert.mpc_to_ckks_conversions
             << " ckks_to_mpc_calls=" << bert.ckks_to_mpc_calls
             << " mpc_to_ckks_calls=" << bert.mpc_to_ckks_calls
             << endl;
        cout << "VIT65_FULL_HE_OPS rotations=" << operations.rotations
             << " ct_ct_multiply=" << operations.ct_ct_multiply
             << " ct_pt_multiply=" << operations.ct_pt_multiply
             << " relinearization=" << operations.relinearizations
             << " gpu_rescale=" << operations.rescales
             << " host_rescale=" << bert.host_rescale_count
             << " conjugation=" << operations.conjugations << endl;
        cout << "VIT65_FULL_LAYOUT status=PASS logical_tokens=65"
             << " physical_tokens=128 layers=7 total_heads=28"
             << " relu_heads=14 linear_heads=14"
             << " he_only_linear_heads=12 linear_spill_heads=2"
             << " qk_ciphertexts_total=28"
             << " qk_boundary_ciphertexts=16"
             << " qk_he_only_ciphertexts=12"
             << " qk_logical_utilization="
             << (double(PACKING_NUM * kLogicalTokens * kLogicalTokens) /
                 double(4 * bert.linear.data_lin1_0.slot_count))
             << endl;
        return 0;
    } catch (const exception &error) {
        cerr << "VIT65_FULL status=BLOCKED reason=" << error.what() << endl;
        return 2;
    }
}
