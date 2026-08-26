#ifndef BERT_H__
#define BERT_H__

// #include "linear.h"
#include "linear_ckks.h"
#include "nonlinear.h"
#include <seal/util/rlwe.h>

#define NL_NTHREADS 32
#define NL_ELL 37

#define GELU_ELL 20
#define GELU_SCALE 11
#define NL_SCALE 12
#define ATTENTION_V_SCALE 11
#ifndef NUM_CLASS
#define NUM_CLASS 2
#endif

// #define BERT_DEBUG
#define BERT_PERF
#define XTS
// #define BERT_SAVE_RESULTS


using namespace std;

class Bert_ckks
{
public:
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
    double conversion_la = 0;
    double other_nonlinear_la = 0;
    double he_compute_la = 0;
    uint64_t ckks_to_mpc_conversions = 0;
    uint64_t mpc_to_ckks_conversions = 0;
    uint64_t ckks_to_mpc_calls = 0;
    uint64_t mpc_to_ckks_calls = 0;
    uint64_t field_to_ring_calls = 0;
    uint64_t masked_rns_coefficients = 0;
    uint64_t host_rescale_count = 0;
    Linear_ckks linear;
    NonLinear nonlinear;
    uint64_t poly_modulus_degree = 8192;
    bool prune;
    FixArray ffn_residual_share;
    bool ffn_residual_ready = false;
    FixArray attention_v_share;
    FixArray attention_input_residual_share;
    bool attention_v_ready = false;
    bool attention_input_residual_ready = false;
    // Public BERT key mask in unscaled score units (0 for a valid key and a
    // negative value for padding).  It is added to exactly one additive MPC
    // share immediately before Softmax; no model or client-private value is
    // exposed by this public operation.
    vector<int64_t> public_attention_mask;
    bool public_attention_mask_ready = false;
    bool a0_trace_enabled = false;
    vector<UnifiedCiphertext> a0_qkv_ciphertexts;
    vector<UnifiedCiphertext> a0_qk_ciphertexts;
    vector<UnifiedCiphertext> a0_av_ciphertexts;
#ifdef BLB_VIT65_HETERO
    // Timed-path device snapshots.  These are never opened: pair 0 QK is
    // selectively converted, while pair 1 QK and all V ciphertexts remain in
    // he1 until AV and WO complete.
    vector<UnifiedCiphertext> hetero_qk_ciphertexts;
    vector<UnifiedCiphertext> hetero_v_ciphertexts;
    vector<string> hetero_head_protocols;
    size_t hetero_boundary_qk_ct_count = 0;
#endif
    FixArray a0_softmax_exp_share;
    FixArray a0_softmax_sum_share;
    FixArray a0_softmax_div_share;

    LOCATION location_ = LOCATION::HOST;

    Bert_ckks(int party, int port, string address, LOCATION backend = LOCATION::HOST,
              const string &model_path = "checkpoints/official-blb-mrpc/weights_txt",
              bool attention_only = false);
    // Lightweight protocol-regression entry: reuse a caller-owned transport
    // and initialize only the MPC conversion channels. It deliberately skips
    // all model loading and the six production HE contexts; callers supply the
    // HE_CKKS instance explicitly to the conversion methods under test.
    Bert_ckks(int party, sci::NetIO *io, string address,
              int nonlinear_port);
    ~Bert_ckks();
    Bert_ckks();

    inline uint64_t get_comm(){
        uint64_t total_comm = io->counter;
        for(int i = 0; i < MAX_THREADS; i++){
            total_comm += nonlinear.iopackArr[i]->get_comm();
        }
        uint64_t ret_comm = total_comm - n_comm;
        n_comm = total_comm;
        return ret_comm;
    }

    inline uint64_t get_round(){
        uint64_t total_round = io->num_rounds;
        for(int i = 0; i < MAX_THREADS; i++){
            total_round += nonlinear.iopackArr[i]->get_rounds();
        }
        uint64_t ret_round = total_round - n_rounds;
        n_rounds = total_round;
        return ret_round;
    }

    inline uint64_t raw_total_comm() const {
        uint64_t total_comm = io->counter;
        for (int i = 0; i < MAX_THREADS; ++i) {
            total_comm += nonlinear.iopackArr[i]->get_comm();
        }
        return total_comm;
    }

    inline uint64_t raw_total_rounds() const {
        uint64_t total_round = io->num_rounds;
        for (int i = 0; i < MAX_THREADS; ++i) {
            total_round += nonlinear.iopackArr[i]->get_rounds();
        }
        return total_round;
    }

    vector<vector<HE_CKKS::scalar_t>> ckks_to_mpc(const vector<Plaintext> &pts, int degree, uint64_t Q, int bit_out, int s_out, HE_CKKS* he, bool half=false);

    // BLB Algorithm 1: sample a fresh uniform element directly in the current
    // RNS polynomial ring, add it to c0, and return the additive -r share.
    // No slot-vector encoding is involved.
    void mask_ckks_to_mpc_inplace(vector<Ciphertext> &ciphertexts,
                                  vector<Plaintext> &server_shares,
                                  HE_CKKS *he);

    // size是slot_num
    vector<Plaintext> mpc_to_ckks(const vector<vector<HE_CKKS::scalar_t>> &input, int slot, uint64_t Q, int bit_out, int s_in, int s_out, HE_CKKS* he, bool half=false);

    // Algorithm 2 at an RNS CKKS entry level. Before the secure FFT, grow the
    // ring width and scale by the same public amount allowed by the 63-bit FFT
    // product bound. This injective additive map preserves the full original
    // fixed-point domain while reducing butterfly rounding. Each prime-field
    // share is then raised publicly from that working scale to target_scale.
    vector<Plaintext> mpc_to_ckks_rns(
        const vector<vector<HE_CKKS::scalar_t>> &input, int slot,
        int bit_out, int input_scale, int target_scale, HE_CKKS *he,
        bool half = false);

    // Compatibility entry point for callers that intentionally keep the
    // fixed-point and CKKS scales identical.
    vector<Plaintext> mpc_to_ckks_rns_same_scale(
        const vector<vector<HE_CKKS::scalar_t>> &input, int slot,
        int bit_out, int scale, HE_CKKS *he, bool half = false);

    // Before Algorithm 1, align to an honest high CKKS conversion scale and
    // modulus-switch to the final single-prime level. ckks_to_mpc keeps that
    // scale through field->ring, then uses the existing secure
    // truncate_reduce to reach an elevated secure-FFT working scale. After
    // decoding, a second secure truncation returns to the requested MPC
    // metadata. Keeping both HE noise and FFT rounding in discarded low bits
    // is essential.
    void prepare_ckks_to_mpc_inplace(vector<Ciphertext> &ciphertexts,
                                     int target_scale_bits, HE_CKKS *he);
    inline void mpc_to_ckks(vector<UnifiedPlaintext> &out,
                            const vector<vector<HE_CKKS::scalar_t>> &in, int slot,
                            uint64_t Q, int bit_out, int s_in, int s_out,
                            HE_CKKS *he, bool half = false) {
        vector<Plaintext> tmp = std::move(mpc_to_ckks(in, slot, Q, bit_out, s_in, s_out, he, half));
        out.resize(tmp.size());
        for (size_t i = 0;i < out.size();i ++) {
            out[i] = std::move(tmp[i]);
        }
    }
    
    FixArray element_wise_square(int32_t dim, FixArray input);

    // Build the first LayerNorm fusion inputs from the real secret-shared
    // layer-0 tensor.  This is the pre-rsqrt half of the existing BLB path;
    // it never reconstructs the tensor between protocol domains.
    std::map<string, FixArray> layer_norm_part1_real(const FixArray &input,
                                                     const FixArray &w,
                                                     const FixArray &b);

    FixArray div_batch_he(const FixArray& nm, const FixArray& dn, int batch_dn_size, int l_out, int s_out, bool normalized_dn = false);

    FixArray div_batch_he_V2(const FixArray& nm, const FixArray& dn, int batch_dn_size, int l_out, int s_out, bool normalized_dn = false);

    FixArray element_wise_mul(FixArray x, FixArray y);

    FixArray element_wise_square_bfv(int32_t dim, FixArray input);

    map<string, FixArray> layer_norm_nonlinear(
        map<string, FixArray> input);

    // Complete the existing Figure-10 LayerNorm path after MPC rsqrt:
    // CKKS evaluates centered * inv_sigma * real weight + real bias.
    FixArray layer_norm_affine_ckks(const map<string, FixArray> &input);

    vector<vector<HE_CKKS::scalar_t>> layer_norm_part2_attention(
        std::map<string, FixArray> input, int layer_id = 0);

    // BERT-base is post-norm: layer 0 starts with the exported embedding
    // tensor, so Q/K/V must consume the real input directly.  This entry
    // reuses the existing MPC↔CKKS and pp_1/BSGS path without manufacturing
    // an identity LayerNorm share.
    vector<vector<HE_CKKS::scalar_t>> attention_from_real_input(
        const FixArray &input, int layer_id = 0);

    // Validation-only snapshots.  Ciphertexts are copied during the normal
    // HE execution and converted/opened only after the timed A0 region.
    void enable_a0_trace(bool enabled) { a0_trace_enabled = enabled; }
    map<string, FixArray> materialize_a0_attention_trace();

    std::map<string, FixArray> gelu_part1(uint64_t *x, int num_ops,
                                         int bitwidth, int s);

    FixArray gelu_nonlinear(std::map<string, FixArray> input);
    // my softmax
    FixArray softmax_ot(FixArray input);

    FixArray softmax_he(FixArray input);

    // Attention-only A0 continuation: import the MPC Softmax shares through
    // Algorithm 2, evaluate the donor's CKKS Softmax*V and BSGS output
    // projection, then export WO through Algorithm 1.  No residual,
    // LayerNorm or MPC reference matrix multiplication is included.
    FixArray attention_output_from_softmax(FixArray input, int layer_id = 0);

#ifdef BLB_VIT65_HETERO
    void set_heterogeneous_head_protocols(const vector<string> &protocols);
    size_t heterogeneous_boundary_qk_ciphertexts() const {
        return hetero_boundary_qk_ct_count;
    }
    // Continue the fixed, public [ReLU, Linear, Linear, Linear] assignment.
    // normalized_pair contains full physical score matrices, but only heads
    // 0 and 1 are populated; heads 2 and 3 stay encrypted in he1.
    FixArray attention_output_from_heterogeneous_scores(
        FixArray normalized_pair, int layer_id = 0);
#endif

    vector<FixArray> softmax_fix_he(int party, FixArray x, FPMath* fpmath[]);

    FixArray exp_he(const FixArray &x);

    vector<Plaintext> eval_gelu_poly_old(FixArray x);

    void eval_gelu_poly(vector<UnifiedCiphertext> &output_cts,
                        vector<UnifiedCiphertext> x, HE_CKKS* he,
                        int output_scale_bits);

    std::map<string, FixArray> Softmax_v_linear_layer_norm_part1(
        FixArray input, int layer_id);
    std::map<string, FixArray> ln_part2_linear3_gelu_part1(
        std::map<string, FixArray> input, int layer_id = 0);
    std::map<string, FixArray> linear4_add_layernorm_part1(
        FixArray input, int layer_id);
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
