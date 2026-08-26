#include <BuildingBlocks/truncation.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ckks_bert.h"

using namespace sci;
using namespace std;

// SCI's legacy LinearHE utility imports this executable-owned ring width.
// Keep it fixed to the donor's accepted nonlinear domain.
int bitlength = NL_ELL;

namespace {

constexpr int kLogicalTokens = 65;
constexpr int kEll = NL_ELL;
constexpr int kInputScale = NL_SCALE;
constexpr int kQkvInputScale = 5;

struct PhantomOperationTotals {
    uint64_t rotations = 0;
    uint64_t ct_ct_multiply = 0;
    uint64_t ct_pt_multiply = 0;
    uint64_t relinearizations = 0;
    uint64_t rescales = 0;
    uint64_t conjugations = 0;
};

vector<HE_CKKS *> a0_he_contexts(Bert_ckks &bert) {
    return {bert.linear.he1, bert.linear.he2, bert.linear.he3,
            bert.linear.he4, bert.linear.he_softmax,
            bert.linear.he_onetime};
}

void reset_phantom_operation_counters(Bert_ckks &bert) {
    for (HE_CKKS *he : a0_he_contexts(bert)) {
        if (he != nullptr && he->devaluator != nullptr) {
            he->devaluator->reset_operation_counters();
        }
    }
}

PhantomOperationTotals phantom_operation_totals(Bert_ckks &bert) {
    PhantomOperationTotals total;
    for (HE_CKKS *he : a0_he_contexts(bert)) {
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

vector<uint64_t> read_signed_csv(const string &path, size_t expected) {
    ifstream input(path);
    if (!input.good()) {
        throw runtime_error("missing A0 input: " + path);
    }
    vector<uint64_t> values;
    string line;
    while (getline(input, line)) {
        istringstream row(line);
        string cell;
        while (getline(row, cell, ',')) {
            if (!cell.empty()) {
                values.push_back(static_cast<uint64_t>(stoll(cell)));
            }
        }
    }
    if (values.size() != expected) {
        throw runtime_error("A0 input shape mismatch: " + path);
    }
    return values;
}

vector<int64_t> read_public_mask(const string &path) {
    ifstream input(path);
    if (!input.good()) {
        throw runtime_error("missing A0 public mask: " + path);
    }
    vector<int64_t> mask;
    int64_t value = 0;
    while (input >> value) {
        if (value > 0 || value < -(int64_t(1) << (kEll - kInputScale - 1))) {
            throw runtime_error("A0 public mask is outside the signed ring");
        }
        mask.push_back(value);
    }
    if (mask.size() != INPUT_DIM) {
        throw runtime_error("A0 public mask must have 128 physical tokens");
    }
    for (int token = 0; token < INPUT_DIM; ++token) {
        if ((token < kLogicalTokens && mask[token] != 0) ||
            (token >= kLogicalTokens && mask[token] >= 0)) {
            throw runtime_error("A0 public mask does not encode 65+63 padding");
        }
    }
    return mask;
}

#ifdef BLB_VIT65_HETERO
vector<string> read_public_head_protocols(const string &path) {
    ifstream input(path);
    if (!input.good()) {
        throw runtime_error("missing exported public head protocols: " + path);
    }
    vector<string> protocols;
    string value;
    while (input >> value) protocols.push_back(value);
    const vector<string> supported{"relu", "linear", "linear", "linear"};
    if (protocols != supported) {
        throw runtime_error(
            "this heterogeneous binary requires exported alpha [1,0,0,0]");
    }
    cout << "HETERO_ASSIGNMENT_SOURCE status=PASS"
         << " source=head_protocols.txt protocols=relu,linear,linear,linear"
         << " private_branching=0" << endl;
    return protocols;
}
#endif

void write_opened(const string &directory, const string &stage,
                  Bert_ckks &bert, const FixArray &share) {
    if (directory.empty()) return;
    FixArray opened = bert.nonlinear.fpmath[0]->fix->output(PUBLIC, share);
    ofstream out(directory + "/secure_" + stage + ".csv");
    if (!out.good()) {
        throw runtime_error("cannot write A0 trace stage " + stage);
    }
    const uint64_t modulus = uint64_t(1) << opened.ell;
    const uint64_t sign = uint64_t(1) << (opened.ell - 1);
    for (int i = 0; i < opened.size; ++i) {
        const uint64_t raw = opened.data[i] & (modulus - 1);
        const int64_t value = raw >= sign
            ? static_cast<int64_t>(raw - modulus)
            : static_cast<int64_t>(raw);
        out << value << '\n';
    }
    cout << "A0_TRACE stage=" << stage << " elements=" << opened.size
         << " ell=" << opened.ell << " scale=" << opened.s << endl;
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
    reset_phantom_operation_counters(bert);
    truncation_call_count.store(0);
}

#ifdef BLB_VIT65_HETERO
FixArray normalize_heterogeneous_pair(Bert_ckks &bert,
                                      const FixArray &qk_row) {
    constexpr int matrix_values = INPUT_DIM * INPUT_DIM;
    constexpr int logical_matrix_values =
        kLogicalTokens * kLogicalTokens;
    if (qk_row.size != PACKING_NUM * matrix_values ||
        qk_row.ell != kEll || qk_row.s != NL_SCALE) {
        throw runtime_error("heterogeneous QK row-major input mismatch");
    }
    FixOp *fix = bert.nonlinear.fpmath[0]->fix;
    FixArray head0(bert.party, logical_matrix_values, true, kEll, NL_SCALE);
    FixArray head1(bert.party, logical_matrix_values, true, kEll, NL_SCALE);
    for (int row = 0; row < kLogicalTokens; ++row) {
        for (int col = 0; col < kLogicalTokens; ++col) {
            const int compact = row * kLogicalTokens + col;
            head0.data[compact] = qk_row.data[row * INPUT_DIM + col];
            head1.data[compact] =
                qk_row.data[matrix_values + row * INPUT_DIM + col];
        }
    }

    FixArray head0_wide = fix->extend(head0, kEll + 1);
    BoolArray positive = fix->GT(head0_wide, uint64_t(0));
    FixArray relu = fix->if_else(positive, head0_wide, uint64_t(0));
    // div_batch forms an l_out+adjust.s intermediate in a uint64 ring. Keep
    // the non-negative score in a public, conservative 31-bit range and add
    // ceil(log2(65)) accumulator bits; using the original ell38 numerator
    // would silently overflow that intermediate and collapse the quotient.
    constexpr int relu_ell = 31;
    constexpr int sum_ell = relu_ell + 7;
    relu.signed_ = false;
    relu = fix->reduce(relu, relu_ell);
    FixArray relu_for_sum = fix->extend(relu, sum_ell);
    FixArray row_sum(bert.party, kLogicalTokens, false, sum_ell,
                     NL_SCALE);
    for (int row = 0; row < kLogicalTokens; ++row) {
        uint64_t local_sum = 0;
        for (int col = 0; col < kLogicalTokens; ++col) {
            local_sum += relu_for_sum.data[row * kLogicalTokens + col];
        }
        row_sum.data[row] = local_sum & row_sum.ell_mask();
    }
    BoolArray nonzero = fix->GT(row_sum, uint64_t(0));
    FixArray denominator = fix->if_else(nonzero, row_sum, uint64_t(1));
    FixArray relu_norm = fix->div_batch(
        relu, denominator, kLogicalTokens, relu_ell, NL_SCALE);
    relu_norm = fix->extend(relu_norm, kEll);
    relu_norm.signed_ = true;

    constexpr int reciprocal_scale = 20;
    const uint64_t reciprocal = static_cast<uint64_t>(
        llround(ldexp(1.0, reciprocal_scale) / kLogicalTokens));
    FixArray linear_product = fix->mul(
        head1, reciprocal, kEll + reciprocal_scale);
    linear_product.s += reciprocal_scale;
    FixArray linear_norm =
        fix->truncate_reduce(linear_product, reciprocal_scale);
    if (relu_norm.ell != kEll || relu_norm.s != NL_SCALE ||
        linear_norm.ell != kEll || linear_norm.s != NL_SCALE) {
        throw runtime_error("heterogeneous normalization metadata mismatch");
    }

    FixArray normalized(bert.party, qk_row.size, true, kEll, NL_SCALE);
    fill_n(normalized.data, normalized.size, uint64_t(0));
    for (int row = 0; row < kLogicalTokens; ++row) {
        for (int col = 0; col < kLogicalTokens; ++col) {
            const int compact = row * kLogicalTokens + col;
            normalized.data[row * INPUT_DIM + col] =
                relu_norm.data[compact];
            normalized.data[matrix_values + row * INPUT_DIM + col] =
                linear_norm.data[compact];
        }
    }
    cout << "HETERO_NORMALIZATION status=PASS"
         << " head0=relu_norm_mpc head1=linear_spill_mpc"
         << " head2=he_linear head3=he_linear"
         << " reciprocal_raw=" << reciprocal
         << " reciprocal_scale=" << reciprocal_scale << endl;
    return normalized;
}
#endif

}  // namespace

int main(int argc, char **argv) {
    int party = 0;
    int port = 8000;
    string address = "127.0.0.1";
    string model_path;
    string backend_mode = "gpu";
    string trace_dir;

    ArgMapping args;
    args.arg("r", party, "Role: ALICE=1, BOB=2");
    args.arg("p", port, "TCP port");
    args.arg("ip", address, "ALICE address");
    args.arg("model", model_path, "A0 exported weights/input directory");
    args.arg("backend", backend_mode, "gpu/device only for accepted A0");
    args.arg("trace_dir", trace_dir,
             "optional correctness trace directory (excluded from timing)");
    args.parse(argc, argv);

    try {
        if (party != ALICE && party != BOB) {
            throw invalid_argument("A0 party must be 1 or 2");
        }
        if (model_path.empty()) {
            throw invalid_argument("A0 --model is required");
        }
        if (backend_mode != "gpu" && backend_mode != "device") {
            throw runtime_error(
                "A0 refuses CPU CKKS: Phantom GPU backend is mandatory");
        }

#ifdef BLB_VIT65_HETERO
        cout << "A0_CONFIGURATION protocol=HEAD_HETERO_FIXED"
             << " assignment=relu,linear_spill,linear,linear"
             << " donor=bert_large_bolt"
#else
        cout << "A0_CONFIGURATION protocol=BLB_SOFTMAX donor=bert_large_bolt"
#endif
             << " backend=phantom_gpu logical_tokens=" << kLogicalTokens
             << " physical_tokens=" << INPUT_DIM
             << " hidden=" << COMMON_DIM << " heads=" << PACKING_NUM
             << " head_dim=" << OUTPUT_DIM
             << " qkv_weight_scale_bits="
             << BLB_QKV_WEIGHT_FRACTIONAL_BITS << endl;

        Bert_ckks bert(party, port, address, LOCATION::DEVICE, model_path,
                       true);
        bert.enable_a0_trace(!trace_dir.empty());
        FPMath *fpmath = bert.nonlinear.fpmath[0];
        const size_t input_values_count =
            static_cast<size_t>(INPUT_DIM) * COMMON_DIM;
        vector<uint64_t> input_values;
        if (party == BOB) {
            input_values = read_signed_csv(model_path + "/inputs_0_data.txt",
                                           input_values_count);
        }
        const vector<int64_t> public_mask =
            read_public_mask(model_path + "/inputs_0_mask.txt");
        bert.set_public_attention_mask(public_mask);
#ifdef BLB_VIT65_HETERO
        const vector<string> public_head_protocols =
            read_public_head_protocols(model_path + "/head_protocols.txt");
        (void)public_head_protocols;
#endif

        reset_online_metrics(bert);
        const uint64_t online_comm_start = bert.raw_total_comm();
        const uint64_t online_round_start = bert.raw_total_rounds();
        const auto online_start = chrono::high_resolution_clock::now();
        FixArray input = fpmath->fix->input(
            BOB, static_cast<int>(input_values_count),
            party == BOB ? input_values.data() : nullptr,
            true, kEll, kInputScale);
        FixArray qkv_input = fpmath->fix->truncate_reduce(
            input, kInputScale - kQkvInputScale);
        FixArray qkv_column(party, qkv_input.size, true, qkv_input.ell,
                            qkv_input.s);
        for (int token = 0; token < INPUT_DIM; ++token) {
            for (int hidden = 0; hidden < COMMON_DIM; ++hidden) {
                qkv_column.data[hidden * INPUT_DIM + token] =
                    qkv_input.data[token * COMMON_DIM + hidden];
            }
        }

        vector<vector<HE_CKKS::scalar_t>> qk_v =
            bert.attention_from_real_input(qkv_column);
        const int slots = bert.linear.data_lin1_0.slot_count;
        const size_t qk_values = static_cast<size_t>(PACKING_NUM) *
            INPUT_DIM * INPUT_DIM;
        const size_t v_values = static_cast<size_t>(INPUT_DIM) * COMMON_DIM;
        const size_t qk_cts = qk_values / slots;
        const size_t v_cts = v_values / slots;
        if (qk_values % slots || v_values % slots ||
#ifdef BLB_VIT65_HETERO
            qk_v.size() != qk_cts / 2) {
#else
            qk_v.size() != qk_cts + v_cts) {
#endif
            throw runtime_error("A0 QK/V full-slot ciphertext layout mismatch");
        }
        FixArray qk_packed(party, static_cast<int>(qk_values), true,
                           kEll, NL_SCALE);
        fill_n(qk_packed.data, qk_packed.size, uint64_t(0));
#ifndef BLB_VIT65_HETERO
        bert.attention_v_share = FixArray(
            party, static_cast<int>(v_values), true, kEll,
            ATTENTION_V_SCALE);
#endif
        size_t packed_offset = 0;
        for (const auto &cipher_slots : qk_v) {
            if (cipher_slots.size() != static_cast<size_t>(slots)) {
                throw runtime_error("A0 conversion returned a partial slot vector");
            }
            for (uint64_t value : cipher_slots) {
                if (packed_offset < qk_values) {
                    qk_packed.data[packed_offset] = value;
#ifndef BLB_VIT65_HETERO
                } else {
                    bert.attention_v_share.data[packed_offset - qk_values] =
                        value;
#endif
                }
                ++packed_offset;
            }
        }
#ifndef BLB_VIT65_HETERO
        bert.attention_v_ready = true;
#endif

        FixArray qk_row(party, static_cast<int>(qk_values), true,
                        kEll, NL_SCALE);
        bert.linear.plain_cross_packing_postprocess(
            qk_packed.data, qk_row.data, false,
            bert.linear.data_lin1_0);
#ifndef BLB_VIT65_HETERO
        FixArray v_head(party, static_cast<int>(v_values), true,
                        kEll, ATTENTION_V_SCALE);
        bert.linear.plain_cross_packing_postprocess_v(
            bert.attention_v_share.data, v_head.data, false,
            bert.linear.data_lin1_0);
#endif

        const double he_before_softmax = bert.he_compute_la;
        const double conv_before_softmax = bert.conversion_la;
        const double transfer_before_softmax = bert.ct_transfer_la;
        const auto softmax_start = chrono::high_resolution_clock::now();
#ifdef BLB_VIT65_HETERO
        FixArray softmax = normalize_heterogeneous_pair(bert, qk_row);
#else
        FixArray softmax = bert.softmax_he(qk_packed);
#endif
        const double softmax_wall = chrono::duration<double>(
            chrono::high_resolution_clock::now() - softmax_start).count();
        const double softmax_he = bert.he_compute_la - he_before_softmax;
        const double softmax_conversion =
            bert.conversion_la - conv_before_softmax;
        const double softmax_transfer =
            bert.ct_transfer_la - transfer_before_softmax;

#ifdef BLB_VIT65_HETERO
        FixArray output =
            bert.attention_output_from_heterogeneous_scores(softmax, 0);
#else
        FixArray output = bert.attention_output_from_softmax(softmax, 0);
#endif
        const double online_seconds = chrono::duration<double>(
            chrono::high_resolution_clock::now() - online_start).count();
        const uint64_t online_comm =
            bert.raw_total_comm() - online_comm_start;
        const uint64_t online_rounds =
            bert.raw_total_rounds() - online_round_start;
        const PhantomOperationTotals phantom_ops =
            phantom_operation_totals(bert);
        const uint64_t online_host_rescales = bert.host_rescale_count;
        const uint64_t online_ckks_to_mpc_ciphertexts =
            bert.ckks_to_mpc_conversions;
        const uint64_t online_mpc_to_ckks_ciphertexts =
            bert.mpc_to_ckks_conversions;
        const uint64_t online_ckks_to_mpc_calls = bert.ckks_to_mpc_calls;
        const uint64_t online_mpc_to_ckks_calls = bert.mpc_to_ckks_calls;
        const uint64_t online_field_to_ring_calls = bert.field_to_ring_calls;
        const uint64_t online_masked_rns_coefficients =
            bert.masked_rns_coefficients;
        const uint64_t online_secure_truncations =
            truncation_call_count.load();

        // Correctness traces deliberately run after the online timer.  They
        // reveal reconstructed intermediates and therefore must never be
        // charged to, or mistaken for, secure A0 performance.
        if (!trace_dir.empty()) {
            map<string, FixArray> he_trace =
                bert.materialize_a0_attention_trace();
            write_opened(trace_dir, "q", bert, he_trace.at("q"));
            write_opened(trace_dir, "k", bert, he_trace.at("k"));
#ifdef BLB_VIT65_HETERO
            write_opened(trace_dir, "qk", bert, he_trace.at("qk"));
            write_opened(trace_dir, "v", bert, he_trace.at("v"));
            write_opened(trace_dir, "normalization", bert, softmax);
#else
            write_opened(trace_dir, "qk", bert, qk_row);
            write_opened(trace_dir, "v", bert, v_head);
            write_opened(trace_dir, "softmax", bert, softmax);
#endif
            write_opened(trace_dir, "av", bert, he_trace.at("av"));
            write_opened(trace_dir, "wo", bert, output);
#ifndef BLB_VIT65_HETERO
            write_opened(trace_dir, "softmax_exp", bert,
                         he_trace.at("softmax_exp"));
            write_opened(trace_dir, "softmax_sum", bert,
                         he_trace.at("softmax_sum"));
            write_opened(trace_dir, "softmax_div", bert,
                         he_trace.at("softmax_div"));
#endif
        }

        const double accounted = bert.he_compute_la + bert.conversion_la +
            bert.ct_transfer_la;
        const double mpc_and_overhead = max(0.0, online_seconds - accounted);
        const double softmax_mpc = max(
            0.0, softmax_wall - softmax_he - softmax_conversion -
                     softmax_transfer);
        const double input_slot_util =
            double(kLogicalTokens * COMMON_DIM) /
            double((input_values_count / slots) * slots);
        const double qk_slot_util =
            double(PACKING_NUM * kLogicalTokens * kLogicalTokens) /
            double(qk_cts * slots);
#ifdef BLB_VIT65_HETERO
        const char *metric_prefix = "HETERO";
#else
        const char *metric_prefix = "A0";
#endif

        cout << setprecision(17);
        cout << metric_prefix << "_LAYOUT status=PASS qkv_input_cts="
             << input_values_count / slots << " qk_cts=" << qk_cts
             << " v_cts=" << v_cts
#ifdef BLB_VIT65_HETERO
             << " mpc_normalized_cts=" << qk_cts / 2
             << " he_linear_score_cts=" << qk_cts / 2
#else
             << " softmax_cts=" << qk_cts
#endif
             << " av_cts=" << v_cts << " wo_cts=" << v_cts
             << " physical_slot_fill=1 input_logical_util=" << input_slot_util
             << " qk_logical_util=" << qk_slot_util << endl;
        cout << metric_prefix << "_TIMING end_to_end_seconds=" << online_seconds
             << " he_seconds=" << bert.he_compute_la
             << " mpc_and_host_seconds=" << mpc_and_overhead
             << " conversion_seconds=" << bert.conversion_la
             << " ciphertext_transfer_seconds=" << bert.ct_transfer_la
             << " h2d_d2h_seconds=" << bert.ct_transfer_mem_la
             << " normalization_wall_seconds=" << softmax_wall
             << " normalization_mpc_seconds=" << softmax_mpc << endl;
        cout << metric_prefix << "_COMM total_local_bytes=" << online_comm
             << " rounds=" << online_rounds
             << " conversion_local_bytes=" << bert.conversion_comm
             << " ciphertext_transfer_local_bytes=" << bert.ct_transfer_comm
             << endl;
        cout << metric_prefix << "_CONVERSIONS ckks_to_mpc_ciphertexts="
             << online_ckks_to_mpc_ciphertexts
             << " mpc_to_ckks_ciphertexts="
             << online_mpc_to_ckks_ciphertexts
             << " ckks_to_mpc_calls=" << online_ckks_to_mpc_calls
             << " mpc_to_ckks_calls=" << online_mpc_to_ckks_calls
             << " field_to_ring_calls=" << online_field_to_ring_calls
             << " masked_rns_coefficients="
             << online_masked_rns_coefficients
             << endl;
        cout << metric_prefix << "_HE_OPS rotations=" << phantom_ops.rotations
             << " ct_ct_multiply=" << phantom_ops.ct_ct_multiply
             << " ct_pt_multiply=" << phantom_ops.ct_pt_multiply
             << " relinearization=" << phantom_ops.relinearizations
             << " gpu_rescale=" << phantom_ops.rescales
             << " host_rescale=" << online_host_rescales
             << " conjugation=" << phantom_ops.conjugations
             << " secure_truncation_calls=" << online_secure_truncations
             << endl;
        cout << metric_prefix
             << " status=PASS backend=phantom_gpu protocol="
#ifdef BLB_VIT65_HETERO
             << "head_heterogeneous assignment=relu,linear_spill,linear,linear"
#else
             << "softmax"
#endif
             << " logical_tokens=65 physical_tokens=128" << endl;
        return 0;
    } catch (const exception &error) {
#ifdef BLB_VIT65_HETERO
        cerr << "HETERO status=BLOCKED reason=" << error.what() << endl;
#else
        cerr << "A0 status=BLOCKED reason=" << error.what() << endl;
#endif
        return 2;
    }
}
