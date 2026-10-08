#ifndef BERT_H__
#define BERT_H__

// #include "linear.h"
#include "hetero_plan.h"
#include "linear_ckks.h"
#include "nonlinear.h"
#include <seal/util/rlwe.h>
#include <array>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <vector>

#define NL_NTHREADS 64
// Keep the same fixed-point precision as the reproduced BLB BERT targets.
#define NL_ELL 37

#define GELU_ELL 20
#define GELU_SCALE 11
#define NL_SCALE 12
#define ATTENTION_V_SCALE 11
#define NUM_CLASS 2

// #define BERT_DEBUG
#define BERT_PERF
#define XTS
// #define BERT_SAVE_RESULTS


using namespace std;

enum class BlbViTProfileStage : size_t {
    QKV = 0,
    QK_TRANSPOSE,
    CKKS_TO_MPC,
    SOFTMAX,
    MPC_TO_CKKS,
    SOFTMAX_V,
    OUTPUT_PROJECTION,
    LAYERNORM1,
    FFN1,
    GELU,
    FFN2,
    LAYERNORM2,
};

constexpr size_t kBlbViTProfileStageCount = 12;

enum class BlbViTCommAuditBucket : size_t {
    QK_TO_SOFTMAX_CONVERSION = 0,
    SOFTMAX_TO_CKKS_CONVERSION,
    SOFTMAX_REC_CONVERSION,
    OTHER_CONVERSION,
    SOFTMAX_REC,
    SOFTMAX_CMP,
    SOFTMAX_MUX,
};

constexpr size_t kBlbViTCommAuditBucketCount = 7;

struct BlbViTCommAuditMetric {
    uint64_t send_bytes = 0;
    uint64_t calls = 0;
};

struct BlbViTProfileMetric {
    double wall_ms = 0.0;
    uint64_t send_bytes = 0;
    uint64_t rounds = 0;
    // Exclusive stage deltas; order matches profile_operator_snapshot().
    array<uint64_t, 5> operators{};
    double plaintext_device_encode_ms = 0.0;
};

class Bert_ckks
{
public:
    // Leave 22 coefficient magnitude bits for the unselected GELU tails.
    // MPC still receives the original ell=37, scale=12 representation.
    static constexpr int kGeluConversionScaleBits = 37;
    int party;
    string address;
    int port;

    sci::NetIO *io;
    double n_rounds = 0;
    double n_comm = 0;
    double ct_transfer_comm = 0;
    double ct_transfer_round = 0;
    double conversion_comm = 0;
    double conversion_round = 0;
    double other_nonlinear_comm = 0;
    double other_nonlinear_round = 0;
    double total_la = 0;
    double ct_transfer_la = 0;
    double ct_transfer_mem_la = 0;
    double h2d_time = 0;
    double d2h_time = 0;
    double conversion_la = 0;
    double other_nonlinear_la = 0;
    double he_compute_la = 0;
    uint64_t block_online_comm_bytes = 0;
    double block_online_profile_seconds = 0;
    uint64_t ckks_to_mpc_conversions = 0;
    uint64_t mpc_to_ckks_conversions = 0;
    uint64_t conversion_ciphertexts = 0;
    uint64_t field_to_ring_calls = 0;
    uint64_t ring_to_field_calls = 0;
    uint64_t masked_rns_coefficients = 0;
    uint64_t he_rescale_count = 0;
    Linear_ckks linear;
    NonLinear nonlinear;
    uint64_t poly_modulus_degree = 8192;
    bool prune;
    FixArray ffn_residual_share;
    bool ffn_residual_ready = false;
    FixArray attention_v_share;
    // Validation-only B shares. They are populated only when
    // OURS_VIT_B_TRACE=1 and are opened by the driver after the normal
    // Algorithm-1 conversion. The production inference path never allocates
    // or reconstructs these tensors.
    FixArray b_q_share;
    FixArray b_k_share;
    FixArray b_v_share;
    bool b_qkv_ready = false;
    // B-only observers for the two encrypted stages after normalization.
    // AV is retained in the same packed per-Head layout as V; W_O is stored
    // in token-major order before the block-input residual is added.  These
    // shares are never read unless OURS_VIT_B_TRACE=1.
    FixArray b_av_share;
    FixArray b_wo_share;
    bool b_av_wo_ready = false;
    // Retained for the ViT adapter only. The BERT path must not carry a V
    // ciphertext from Block 2's he1 context into Block 4's distinct he2
    // context; the official boundary exports/imports V through MPC.
    vector<UnifiedCiphertext> attention_v_he_cts;
    bool attention_v_he_ready = false;
    // heterogeneous public plan and the native he1 operands retained only by Alice.
    // The packed QK vector has eight ciphertexts per two-head pair; public
    // half masks zero every non-Linear half before it reaches the NATIVE kernel.
    vector<ours_vit::HeadProtocol> heterogeneous_protocols;
    vector<double> heterogeneous_alphas;
    ours_vit::Version heterogeneous_version = ours_vit::Version::B;
    // Paper version switch.  The default is the production packed path.
    // Setting it false only splits otherwise identical public-plan groups and
    // disables diagonal reuse; it never changes the Head assignment.
    bool heterogeneous_group_packing_enabled = true;
    // The original BLB diagonal-reuse schedule is independent of the later
    // heterogeneous Head grouping switch.  Keep this explicit so the
    // allocator can remain per-Head while using the established fast GPU
    // end-to-end schedule.
    // Validation-only observers used by the paired Packed/Unpacked trace.
    FixArray heterogeneous_av_share;
    FixArray heterogeneous_wo_share;
    bool heterogeneous_av_wo_ready = false;
    vector<UnifiedCiphertext> native_linear_qk_he1_cts;
    vector<UnifiedCiphertext> native_linear_v_he1_cts;
    bool native_linear_operands_ready = false;
    FixArray attention_input_residual_share;
    bool attention_v_ready = false;
    bool attention_input_residual_ready = false;
    // These switches expose the same original CKKS/MPC kernels at a ViT
    // pre-norm residual boundary.  They do not select another protocol.
    bool pre_norm_attention_output_only = false;
    bool pre_norm_ffn_residual_from_external = false;
    bool pre_norm_ffn_output_only = false;
    // Explicit model graph: only the real BERT driver enables post-norm.
    // ViT keeps its existing execution and conversion boundaries.
    bool bert_post_norm = false;
    // Public BERT key mask in unscaled score units (0 for a valid key and a
    // negative value for padding).  It is added to exactly one additive MPC
    // share immediately before Softmax; no model or client-private value is
    // exposed by this public operation.
    vector<int64_t> public_attention_mask;
    bool public_attention_mask_ready = false;

    LOCATION location_ = LOCATION::HOST;

    Bert_ckks(int party, int port, string address, LOCATION backend = LOCATION::HOST,
              const string &model_path = "checkpoints/official-blb-mrpc/weights_txt",
              int layers_to_preprocess = ATTENTION_LAYERS);
    Bert_ckks(int party, int port, string address, LOCATION backend,
              const BertModel *external_model, int sequence_length,
              bool preprocess_weights = true,
              int layers_to_preprocess = ATTENTION_LAYERS,
              bool enable_native_linear = false);
    // Lightweight protocol-regression entry: reuse a caller-owned transport
    // and initialize only the MPC conversion channels. It deliberately skips
    // all model loading and the six production HE contexts; callers supply the
    // HE_CKKS instance explicitly to the conversion methods under test.
    Bert_ckks(int party, sci::NetIO *io, string address,
              int nonlinear_port);
    ~Bert_ckks();
    Bert_ckks();

    // These methods only account for already-executing regions. They are
    // deliberately inactive unless the ViT driver enables profiling.
    inline void profile_start(BlbViTProfileStage stage) {
        profile_enabled_ = true;
        profile_stack_.clear();
        for (auto &metric : profile_metrics_) metric = BlbViTProfileMetric{};
        profile_he_ms_.fill(0.0);
        profile_comm_audit_.fill(BlbViTCommAuditMetric{});
        profile_conversion_bucket_ = BlbViTCommAuditBucket::OTHER_CONVERSION;
        profile_stage_ = -1;
        profile_begin(stage);
    }

    inline void profile_push(BlbViTProfileStage stage) {
        if (!profile_enabled_) return;
        profile_close();
        profile_stack_.push_back(profile_stage_);
        profile_begin(stage);
    }

    inline void profile_pop() {
        if (!profile_enabled_) return;
        profile_close();
        if (profile_stack_.empty()) {
            profile_stage_ = -1;
            return;
        }
        const int resume = profile_stack_.back();
        profile_stack_.pop_back();
        if (resume >= 0) profile_begin(resume);
        else profile_stage_ = -1;
    }

    inline void profile_switch(BlbViTProfileStage stage) {
        if (!profile_enabled_) return;
        profile_close();
        profile_begin(stage);
    }

    inline void profile_stop() {
        if (!profile_enabled_) return;
        profile_close();
        profile_stage_ = -1;
        profile_stack_.clear();
        profile_enabled_ = false;
    }

    inline const BlbViTProfileMetric &profile_metric(
        BlbViTProfileStage stage) const {
        return profile_metrics_[static_cast<size_t>(stage)];
    }

    inline void profile_record_he(BlbViTProfileStage stage, double ms) {
        if (profile_enabled_) {
            profile_he_ms_[static_cast<size_t>(stage)] += ms;
        }
    }

    inline double profile_he_ms(BlbViTProfileStage stage) const {
        return profile_he_ms_[static_cast<size_t>(stage)];
    }

    inline bool profile_is_enabled() const { return profile_enabled_; }

    inline void profile_record_comm(BlbViTCommAuditBucket bucket,
                                    uint64_t send_bytes) {
        if (!profile_enabled_) return;
        lock_guard<mutex> lock(profile_comm_audit_mutex_);
        auto &metric = profile_comm_audit_[static_cast<size_t>(bucket)];
        metric.send_bytes += send_bytes;
        metric.calls++;
    }

    inline const BlbViTCommAuditMetric &profile_comm_audit(
        BlbViTCommAuditBucket bucket) const {
        return profile_comm_audit_[static_cast<size_t>(bucket)];
    }

    inline BlbViTCommAuditBucket profile_conversion_bucket() const {
        return profile_conversion_bucket_;
    }

    inline void profile_set_conversion_bucket(BlbViTCommAuditBucket bucket) {
        profile_conversion_bucket_ = bucket;
    }

    inline uint64_t current_comm() const {
        uint64_t total = io == nullptr ? 0 : io->counter;
        for (int i = 0; i < MAX_THREADS; ++i) {
            if (nonlinear.iopackArr[i] != nullptr) {
                total += nonlinear.iopackArr[i]->get_comm();
            }
        }
        return total;
    }

    inline uint64_t current_round() const {
        uint64_t total = io == nullptr ? 0 : io->num_rounds;
        uint64_t nonlinear_max = 0;
        for (int i = 0; i < MAX_THREADS; ++i) {
            if (nonlinear.iopackArr[i] != nullptr) {
                nonlinear_max = max(
                    nonlinear_max,
                    nonlinear.iopackArr[i]->get_rounds());
            }
        }
        return total + nonlinear_max;
    }

    inline array<uint64_t, 3> current_stream_comm() const {
        array<uint64_t, 3> totals = {
            io == nullptr ? 0 : io->counter, 0, 0};
        for (int i = 0; i < MAX_THREADS; ++i) {
            const auto *pack = nonlinear.iopackArr[i];
            if (pack == nullptr) continue;
            totals[0] += pack->io->counter;
            totals[1] += pack->io_rev->counter;
            totals[2] += pack->io_GC->counter;
        }
        return totals;
    }

private:
    inline array<uint64_t, 5> profile_operator_snapshot() const {
        return {linear.rotation_count, linear.ct_pt_multiply_count,
                linear.ct_ct_multiply_count, linear.rescale_count + he_rescale_count,
                conversion_ciphertexts};
    }

    inline void profile_begin(BlbViTProfileStage stage) {
        profile_begin(static_cast<int>(stage));
    }

    inline void profile_begin(int stage) {
        profile_stage_ = stage;
        profile_start_time_ = chrono::steady_clock::now();
        profile_start_comm_ = current_comm();
        profile_start_round_ = current_round();
        profile_start_operators_ = profile_operator_snapshot();
        profile_start_encode_ = linear.plaintext_device_encode_time;
    }

    inline void profile_close() {
        if (profile_stage_ < 0) return;
        const auto now = chrono::steady_clock::now();
        const size_t stage = static_cast<size_t>(profile_stage_);
        profile_metrics_[stage].wall_ms +=
            chrono::duration<double, milli>(now - profile_start_time_).count();
        const uint64_t comm = current_comm();
        const uint64_t rounds = current_round();
        if (comm < profile_start_comm_ || rounds < profile_start_round_)
            throw runtime_error("profiling communication counter moved backwards");
        profile_metrics_[stage].send_bytes += comm - profile_start_comm_;
        profile_metrics_[stage].rounds += rounds - profile_start_round_;
        const auto operators = profile_operator_snapshot();
        for (size_t i = 0; i < operators.size(); ++i) {
            if (operators[i] < profile_start_operators_[i])
                throw runtime_error("profiling operator counter moved backwards");
            profile_metrics_[stage].operators[i] += operators[i] - profile_start_operators_[i];
        }
        profile_metrics_[stage].plaintext_device_encode_ms +=
            (linear.plaintext_device_encode_time - profile_start_encode_) * 1000.0;
    }

    bool profile_enabled_ = false;
    int profile_stage_ = -1;
    chrono::steady_clock::time_point profile_start_time_{};
    uint64_t profile_start_comm_ = 0;
    uint64_t profile_start_round_ = 0;
    array<uint64_t, 5> profile_start_operators_{};
    double profile_start_encode_ = 0.0;
    array<BlbViTProfileMetric, kBlbViTProfileStageCount> profile_metrics_{};
    array<double, kBlbViTProfileStageCount> profile_he_ms_{};
    array<BlbViTCommAuditMetric, kBlbViTCommAuditBucketCount>
        profile_comm_audit_{};
    mutex profile_comm_audit_mutex_;
    BlbViTCommAuditBucket profile_conversion_bucket_ =
        BlbViTCommAuditBucket::OTHER_CONVERSION;
    vector<int> profile_stack_;

public:

    inline uint64_t get_comm(){
        uint64_t total_comm = io->counter;
        for(int i = 0; i < MAX_THREADS; i++){
            if (nonlinear.iopackArr[i] != nullptr) {
                total_comm += nonlinear.iopackArr[i]->get_comm();
            }
        }
        uint64_t ret_comm = total_comm - n_comm;
        n_comm = total_comm;
        return ret_comm;
    }

    inline uint64_t get_round(){
        uint64_t total_round = io->num_rounds;
        uint64_t nonlinear_max = 0;
        for(int i = 0; i < MAX_THREADS; i++){
            if (nonlinear.iopackArr[i] != nullptr) {
                nonlinear_max = max(
                    nonlinear_max,
                    nonlinear.iopackArr[i]->get_rounds());
            }
        }
        total_round += nonlinear_max;
        uint64_t ret_round = total_round - n_rounds;
        n_rounds = total_round;
        return ret_round;
    }

    vector<vector<HE_CKKS::scalar_t>> ckks_to_mpc(const vector<Plaintext> &pts, int degree, uint64_t Q, int bit_out, int s_out, HE_CKKS* he, bool half=false);
    void pack_real_ckks_pairs(vector<Ciphertext>& ciphertexts, HE_CKKS* he);
    vector<vector<HE_CKKS::scalar_t>> ckks_pairs_to_mpc(
        const vector<Plaintext>& pts, size_t real_ciphertexts,
        int bit_out, int s_out, HE_CKKS* he);

    // BLB Algorithm 1: sample a fresh uniform element directly in the current
    // RNS polynomial ring, add it to c0, and return the additive -r share.
    // No slot-vector encoding is involved.
    void mask_ckks_to_mpc_inplace(vector<Ciphertext> &ciphertexts,
                                  vector<Plaintext> &server_shares,
                                  HE_CKKS *he);

    // size是slot_num
    vector<Plaintext> mpc_to_ckks(const vector<vector<HE_CKKS::scalar_t>> &input, int slot, uint64_t Q, int bit_out, int s_in, int s_out, HE_CKKS* he, bool half=false, bool known_nonnegative=false);

    // Before Algorithm 1, align to the CKKS conversion scale and
    // modulus-switch to the final single-prime level. ckks_to_mpc performs
    // the paper's Appendix-C.4 extend-then-local-arithmetic-shift operation
    // inside conversion only, then evaluates the public FFT locally. The
    // process-wide MPC truncation mode remains unchanged for Softmax,
    // LayerNorm, and GELU.
    void prepare_ckks_to_mpc_inplace(vector<Ciphertext> &ciphertexts,
                                     int target_scale_bits, HE_CKKS *he);
    inline void mpc_to_ckks(vector<UnifiedPlaintext> &out,
                            const vector<vector<HE_CKKS::scalar_t>> &in, int slot,
                            uint64_t Q, int bit_out, int s_in, int s_out,
                            HE_CKKS *he, bool half = false,
                            bool known_nonnegative = false) {
        vector<Plaintext> tmp = std::move(mpc_to_ckks(
            in, slot, Q, bit_out, s_in, s_out, he, half,
            known_nonnegative));
        out.resize(tmp.size());
        for (size_t i = 0;i < out.size();i ++) {
            out[i] = std::move(tmp[i]);
        }
    }
    
    FixArray element_wise_square(int32_t dim, FixArray input);

    // Build the first LayerNorm fusion inputs from the real secret-shared
    // layer-0 tensor.  This is the pre-rsqrt half of the existing BLB path;
    // it never reconstructs the tensor between protocol domains.
    std::map<char*,FixArray> layer_norm_part1_real(const FixArray &input,
                                                   const FixArray &w,
                                                   const FixArray &b);

    // Generic BSGS2 entry used by the ViT-specific patch projection.  The
    // conversion, encryption, masking, and evaluator operations are the
    // original BLB CKKS path; the method only removes the BERT-specific
    // LayerNorm/attention wrapper around it.
    FixArray linear2_from_mpc(const FixArray &input,
                              PreprocessParams_2 &params,
                              const FCMetadata &data,
                              int layer_id = 0);

    FixArray div_batch_he(const FixArray& nm, const FixArray& dn, int batch_dn_size, int l_out, int s_out, bool normalized_dn = false);

    FixArray div_batch_he_V2(const FixArray& nm, const FixArray& dn, int batch_dn_size, int l_out, int s_out, bool normalized_dn = false, bool softmax_denominator = false, bool compact_softmax_denominator = false);

    FixArray element_wise_mul(FixArray x, FixArray y);

    FixArray element_wise_square_bfv(int32_t dim, FixArray input);

    map<char*,FixArray> layer_norm_nonlinear(map<char*,FixArray> input);

    // Complete the existing Figure-10 LayerNorm path after MPC rsqrt:
    // CKKS evaluates centered * inv_sigma * real weight + real bias.
    FixArray layer_norm_affine_ckks(const map<char*,FixArray> &input);

    vector<vector<HE_CKKS::scalar_t>> layer_norm_part2_attention(
        std::map<char*,FixArray> input, int layer_id = 0);

    // BERT-base is post-norm: layer 0 starts with the exported embedding
    // tensor, so Q/K/V must consume the real input directly.  This entry
    // reuses the existing MPC↔CKKS and pp_1/BSGS path without manufacturing
    // an identity LayerNorm share.
    vector<vector<HE_CKKS::scalar_t>> attention_from_real_input(
        const FixArray &input, int layer_id = 0);

    std::map<char*,FixArray> gelu_part1(uint64_t *x, int num_ops, int bitwidth, int s);

    FixArray gelu_nonlinear(std::map<char*,FixArray> input);
    // my softmax
    FixArray softmax_ot(FixArray input);

    FixArray softmax_he(FixArray input);

    vector<FixArray> softmax_fix_he(int party, FixArray x, FPMath* fpmath[],
                                    int active_heads = PACKING_NUM);

    // HETERO deliberately retains BLB's full CKKS->MPC->CKKS boundary for every
    // head.  Only the public, pre-inference normalization protocol differs;
    // HE-linear candidates use the calibrated alpha in MPC at this version
    // so HETERO cannot claim selective-conversion savings.
    FixArray heterogeneous_attention_norm(
        FixArray input,
        const vector<ours_vit::HeadProtocol> &protocols,
        const vector<double> &alphas, int logical_tokens,
        int layer_id, bool group_contiguous_protocols);

    // Bind one public offline assignment before preprocessing/execution of a
    // layer.  This is never called from an input-dependent branch.
    void configure_heterogeneous_attention_layer(
        int layer_id, const vector<ours_vit::HeadProtocol> &protocols,
        const vector<double> &alphas, ours_vit::Version version);

    // ViT pre-norm heterogeneous output path: grouped MPC->CKKS for Softmax/ReLU,
    // native he1 Linear QK->AV->WO, then Algorithm-1 export and MPC residual
    // merge.  No decrypt/re-encrypt bridge exists in this method.
    std::map<char*,FixArray> heterogeneous_attention_output(
        FixArray normalized_qk, int layer_id);

    FixArray exp_he(const FixArray &x, bool apply_attention_mask = true);

    vector<Plaintext> eval_gelu_poly_old(FixArray x);

    void eval_gelu_poly(vector<UnifiedCiphertext> &output_cts,
                        vector<UnifiedCiphertext> x, HE_CKKS* he,
                        int output_scale_bits);

    std::map<char*,FixArray> Softmax_v_linear_layer_norm_part1(FixArray input, int layer_id);
    std::map<char*,FixArray> ln_part2_linear3_gelu_part1(
        std::map<char*,FixArray> input, int layer_id = 0);
    std::map<char*,FixArray> linear4_add_layernorm_part1(FixArray input, int layer_id);
	void mp2ml_test();

    void set_public_attention_mask(const vector<int64_t> &mask) {
        public_attention_mask = mask;
        public_attention_mask_ready = true;
    }

    // The six HE contexts, their keys and their encoded model caches are
    // bound to the constructor backend.  Silently changing only this field
    // would create a mixed HOST/DEVICE graph, so fail closed instead.
    void set_he_backend(LOCATION location) {
        if (location != location_) {
            throw logic_error("Bert_ckks backend is fixed at construction");
        }
    }
};

#endif
