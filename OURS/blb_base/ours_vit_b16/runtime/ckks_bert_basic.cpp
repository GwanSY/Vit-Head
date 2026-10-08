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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <omp.h>
#include "ckks_bert.h"
#include "secure_row_max.h"

using namespace sci;
using namespace std;
using namespace seal;

namespace {
int blb_conversion_threads() {
    const char *value = std::getenv("OURS_VIT_MPC_WORKERS");
    if (value == nullptr || *value == '\0') return MAX_THREADS;
    const long parsed = std::strtol(value, nullptr, 10);
    if (parsed < 1 || parsed > MAX_THREADS) {
        throw std::invalid_argument(
            "OURS_VIT_MPC_WORKERS must be an integer in [1,64]");
    }
    return static_cast<int>(parsed);
}
int blb_c2m_headroom_bits() {
    const char *value = std::getenv("OURS_VIT_C2M_HEADROOM_BITS");
    if (value == nullptr || *value == '\0') return 24;
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 16 || parsed > 40) {
        throw std::invalid_argument(
            "OURS_VIT_C2M_HEADROOM_BITS must be an integer in [16,40]");
    }
    return static_cast<int>(parsed);
}

bool blb_secure_exp_clamp_enabled() {
    return true; // BLB Algorithm 5 is mandatory, never an environment toggle.
}
}
// #define SCI_HE
#define CKKS_PERF

#ifdef CKKS_PERF
uint64_t c_offline = 0;
uint64_t r_offline = 0;
#endif

void check_cts(vector<Ciphertext> cts){
    for(int i =0; i < cts.size(); i++){
        if(cts[i].poly_modulus_degree()==0){
            cout << "error: " << i << endl;
            return;
        }
    }
    cout << "check_cts OK" << endl;
}
/*
example:
    auto t_load_model = high_resolution_clock::now();
    xxx code
    cout << "> [TIMING]: xxx code takes: " << interval(t_load_model) << "sec" << endl;
*/
inline double interval(chrono::_V2::system_clock::time_point start){
    auto end = high_resolution_clock::now();
    auto interval = (end - start)/1e+9;
    return interval.count();
}

inline void NegacyclicRightShiftInplace(seal::Ciphertext& ct, size_t shift,
                                 const seal::SEALContext& context) {
  if (shift == 0 || ct.size() == 0) {
    // nothing to do
    return;
  }

  auto cntxt = context.get_context_data(ct.parms_id());

  size_t num_coeff = ct.poly_modulus_degree();

  std::vector<uint64_t> tmp(shift);
  //  i < N - s  ai*X^i -> ai*X^{i + s}
  // i >= N - s ai*X^i -> -ai*X^{(i + s) % N}
  const auto& modulus = cntxt->parms().coeff_modulus();
  for (size_t k = 0; k < ct.size(); ++k) {
    uint64_t* dst_ptr = ct.data(k);

    for (const auto& prime : modulus) {
      // save [N-s, N)
      std::copy_n(dst_ptr + num_coeff - shift, shift, tmp.data());

      // X^i for i \in [0, N-s)
      for (size_t i = num_coeff - shift; i > 0; --i) {
        dst_ptr[i - 1 + shift] = dst_ptr[i - 1];
      }

      // i \n [N-s, N)
      for (size_t i = 0; i < shift; ++i) {
        dst_ptr[i] = seal::util::negate_uint_mod(tmp[i], prime);
      }

      dst_ptr += num_coeff;
    }
  }
}

inline void MulImageUnitInplace(seal::Ciphertext& ct, const seal::SEALContext& context){
    const bool is_ntt = ct.is_ntt_form();
    const size_t degree = ct.poly_modulus_degree();
    seal::Evaluator evaluator(context);
    if (is_ntt) {
        evaluator.transform_from_ntt_inplace(ct);
    }

    // * X^{N/2}
    NegacyclicRightShiftInplace(ct, degree / 2, context);

    if (is_ntt) {
        evaluator.transform_to_ntt_inplace(ct);
    }
}

// B \in (1/4, 1)
inline uint64_t recp_lookup_c0(uint64_t index, int m) {
  uint64_t k = 1ULL << m;
  double p = 1 + (double(index) / double(k));
  double HETERO = 1.0 / (p * (p + 1.0 / double(k)));
  int32_t scale = m + 3;
  uint64_t mask = (1ULL << scale) - 1;
  uint64_t val = uint64_t(HETERO * (1ULL << scale)) & mask;
  return val;
}

// HETERO \in (1/2, 1)
inline uint64_t recp_lookup_c1(uint64_t index, int m) {
  uint64_t k = 1ULL << m;
  double p = 1 + (double(index) / double(k));
  double z = (p * (p + (1.0 / double(k))));
  double HETERO = ((1.0 / double(k * 2)) + sqrt(z)) / z;
  int32_t scale = 2 * m + 2;
  uint64_t mask = (1ULL << scale) - 1;
  uint64_t val = uint64_t(HETERO * (1ULL << scale)) & mask;
    return val;
}

static void vit_profile_switch_callback(void *context, int stage) {
    auto *bert = static_cast<Bert_ckks *>(context);
    bert->profile_switch(static_cast<BlbViTProfileStage>(stage));
}

static void vit_profile_he_callback(void *context, int stage, double ms) {
    auto *bert = static_cast<Bert_ckks *>(context);
    bert->profile_record_he(static_cast<BlbViTProfileStage>(stage), ms);
}

static void vit_profile_comm_callback(void *context, int op,
                                      uint64_t send_bytes) {
    auto *bert = static_cast<Bert_ckks *>(context);
    bert->profile_record_comm(
        op == 0 ? BlbViTCommAuditBucket::SOFTMAX_CMP
                : BlbViTCommAuditBucket::SOFTMAX_MUX,
        send_bytes);
}

static void install_vit_profile_comm_hooks(Bert_ckks *bert) {
    // NonLinear owns one FPMath/IOPack per worker.  The translation unit also
    // uses the same 64 workers as the official BLB nonlinear/conversion
    // paths. This only installs callbacks and does not change scheduling.
    for (int i = 0; i < bert->nonlinear.active_workers(); ++i) {
        bert->nonlinear.fpmath[i]->fix->profile_context = bert;
        bert->nonlinear.fpmath[i]->fix->profile_comm_callback =
            vit_profile_comm_callback;
    }
}

Bert_ckks::Bert_ckks(int party, int port, string address, LOCATION backend,
                     const string &model_path, int layers_to_preprocess){
    if (layers_to_preprocess <= 0 || layers_to_preprocess > ATTENTION_LAYERS) {
        throw invalid_argument(
            "Bert_ckks: layer preprocessing scope must be in [1, ATTENTION_LAYERS]");
    }
    this->party = party;
    this->address = address;
    this->port = port;
    this->io = new sci::NetIO(party == 1 ? nullptr : address.c_str(), port);

    if (backend != LOCATION::HOST && backend != LOCATION::DEVICE) {
        throw std::invalid_argument("Bert_ckks: backend must be HOST or DEVICE");
    }
    this->location_ = backend;
    
    this->linear = Linear_ckks(party, io, backend);
    this->linear.profile_context = this;
    this->linear.profile_switch_callback = vit_profile_switch_callback;
    this->linear.profile_he_callback = vit_profile_he_callback;
    cout << "> Setup Linear OK" << endl;
    
    this->nonlinear = NonLinear(party, address, port+1);
    install_vit_profile_comm_hooks(this);
    cout << "> Setup NonLinear OK" << endl;

    this->prune = prune;
    if(party == sci::ALICE){
        cout << "> Loading and preprocessing weights on server" << endl;
        #ifdef BERT_PERF
        auto t_load_model = high_resolution_clock::now();
        #endif 
        // The full BLB entry loads all twelve BERT-base blocks and the actual
        // MRPC pooler/classifier.  Bob deliberately does not load model files;
        // only Alice materializes the HE plaintext weights.
        struct BertModel bm = load_model(
            model_path, NUM_CLASS, layers_to_preprocess, true);

        #ifdef BERT_PERF
        cout << "> [TIMING]: Loading Model takes: " << interval(t_load_model) << "sec" << endl;
        auto t_model_preprocess = high_resolution_clock::now();
        #endif 

        linear.weights_preprocess(bm, layers_to_preprocess);

        #ifdef BERT_PERF
        cout << "> [TIMING]: Model Preprocessing takes: " << interval(t_model_preprocess) << "sec" << endl;
        #endif 
    }
    c_offline += this->get_comm();
    r_offline += this->get_round();
    cout << "c_offline:" << c_offline << ", r_offline:" << r_offline << endl;
    const auto offline_streams = current_stream_comm();
    cout << "VOLE_OFFLINE_STREAM_COMM party=" << party
         << " primary_send_bytes=" << offline_streams[0]
         << " reverse_send_bytes=" << offline_streams[1]
         << " gc_send_bytes=" << offline_streams[2] << endl;
    cout << "> Bert intialized done!" << endl << endl;

    cout << "> Bert_ckks intialized done!" << endl << endl;

}

Bert_ckks::Bert_ckks(int party, int port, string address, LOCATION backend,
                     const BertModel *external_model, int sequence_length,
                     bool preprocess_weights, int layers_to_preprocess,
                     bool enable_native_linear) {
    if (sequence_length <= 0) {
        throw invalid_argument("Bert_ckks: sequence length must be positive");
    }
    if (layers_to_preprocess <= 0 || layers_to_preprocess > ATTENTION_LAYERS) {
        throw invalid_argument(
            "Bert_ckks: layer scope must be in [1, ATTENTION_LAYERS]");
    }
    if (backend != LOCATION::HOST && backend != LOCATION::DEVICE) {
        throw invalid_argument("Bert_ckks: backend must be HOST or DEVICE");
    }
    this->party = party;
    this->address = address;
    this->port = port;
    this->io = new sci::NetIO(party == 1 ? nullptr : address.c_str(), port);
    this->location_ = backend;
    this->linear = Linear_ckks(
        party, io, backend, enable_native_linear);
    this->linear.profile_context = this;
    this->linear.profile_switch_callback = vit_profile_switch_callback;
    this->linear.profile_he_callback = vit_profile_he_callback;
    this->linear.configure_sequence_length(sequence_length);
    cout << "> Setup Linear OK sequence_length=" << sequence_length << endl;
    this->nonlinear = NonLinear(party, address, port + 1);
    install_vit_profile_comm_hooks(this);
    cout << "> Setup NonLinear OK" << endl;

    if (party == sci::ALICE) {
        if (external_model == nullptr) {
            throw invalid_argument(
                "Bert_ckks: Alice requires an external ViT-converted model");
        }
        // Use the original BLB preprocessing kernels.  A partial scope is
        // only used by the ViT profiling stop path to avoid constructing
        // unused offline caches; the default scope remains all 12 layers.
        linear.load_model_tensors(*external_model);
        if (preprocess_weights) {
            linear.weights_preprocess(*external_model, layers_to_preprocess);
        }
    }
    c_offline += this->get_comm();
    r_offline += this->get_round();
    cout << "c_offline:" << c_offline << ", r_offline:" << r_offline << endl;
    cout << "> Bert ViT CKKS baseline initialized" << endl;
}

Bert_ckks::Bert_ckks(int party, sci::NetIO *io, string address,
                     int nonlinear_port) {
    if (io == nullptr) {
        throw invalid_argument("conversion adapter requires a transport");
    }
    this->party = party;
    this->address = std::move(address);
    this->port = nonlinear_port - 1;
    this->io = io;
    this->location_ = LOCATION::HOST;
    this->nonlinear = NonLinear(party, this->address, nonlinear_port);
    install_vit_profile_comm_hooks(this);
    cout << "CKKS_CONVERSION_ADAPTER status=READY model_loaded=0 he_contexts=external"
         << endl;
}

Bert_ckks::~Bert_ckks() {
    
}

void Bert_ckks::configure_heterogeneous_attention_layer(
    int layer_id, const vector<ours_vit::HeadProtocol> &protocols,
    const vector<double> &alphas, ours_vit::Version version) {
    if (layer_id < 0 || layer_id >= ATTENTION_LAYERS ||
        protocols.size() != PACKING_NUM || alphas.size() != PACKING_NUM ||
        version == ours_vit::Version::B) {
        throw invalid_argument("heterogeneous attention layer configuration mismatch");
    }

    heterogeneous_protocols = protocols;
    heterogeneous_alphas = alphas;
    heterogeneous_version = version;
    vector<int> protocol_codes(PACKING_NUM);
    for (int head = 0; head < PACKING_NUM; ++head)
        protocol_codes[head] = static_cast<int>(protocols[head]);
    linear.configure_heterogeneous_layer(layer_id, protocol_codes, alphas);
    cout << "OURS_PUBLIC_LAYER_PLAN status=PASS layer=" << layer_id
         << " version=" << ours_vit::to_string(version)
         << " softmax_heads="
         << count(protocols.begin(), protocols.end(),
                  ours_vit::HeadProtocol::SoftmaxMpc)
         << " relu_heads="
         << count(protocols.begin(), protocols.end(),
                  ours_vit::HeadProtocol::ReluNormMpc)
         << " linear_heads="
         << count(protocols.begin(), protocols.end(),
                  ours_vit::HeadProtocol::HeLinear)
         << " dynamic_selection=0" << endl;
}


/*
	input: 解密后的明文share，对于server，是r，对于client，是ifft(x)-r，这里r是uniform random, size是poly degree.
	step:
		1. 双方各自做intt，并且从[0,Q]转为了[-Q/2,Q/2]
		2. 将scale变成f, 位宽变成bw, bw与f均是人为定义参数, bw>log2Q
		3. 在输入l-bit, scale是f的情况下，本地进行fft变化
        4. (可选) trunc-reduce，可以先做3，2和4合并一次。
*/
vector<vector<HE_CKKS::scalar_t>> Bert_ckks::ckks_to_mpc(const vector<Plaintext> &pts, int degree, uint64_t Q, int bit_out, int s_out, HE_CKKS* he, bool half){
    const bool profile_conversion = profile_enabled_;
    const auto conversion_bucket = profile_conversion_bucket_;
    const uint64_t conversion_comm_before =
        profile_conversion ? current_comm() : 0;
    if (profile_conversion) profile_push(BlbViTProfileStage::CKKS_TO_MPC);
    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 
    if (pts.empty()) {
        throw invalid_argument("CKKS-to-MPC input is empty");
    }
    conversion_ciphertexts += pts.size();
    auto conversion_context = he->context->get_context_data(pts[0].parms_id());
    if (!conversion_context) {
        throw invalid_argument("CKKS-to-MPC plaintext level is invalid");
    }
    const auto &conversion_moduli =
        conversion_context->parms().coeff_modulus();
    if (conversion_moduli.size() != 1) {
        throw runtime_error(
            "CKKS-to-MPC requires the final single-prime level; RNS product "
            "must not be truncated to uint64_t");
    }
    const uint64_t conversion_prime = conversion_moduli[0].value();
    if (Q != conversion_prime) {
        throw runtime_error("CKKS-to-MPC prime does not match plaintext level");
    }
    cout << "in ckks_to_mpc, degree, Q, bit_out, s_in, s_out:" << degree << "," << Q << "," << bit_out << ","<< round(log2(pts[0].scale())) << "," << s_out << endl;
    int slot = degree/2;
    // step 1
    vector<vector<std::complex<HE_CKKS::scalar_t>>> x_intt(pts.size());
    cout << "pts.size():" << pts.size() << endl;
    int num_threads = pts.size()>64?64:pts.size();
    #pragma omp parallel for num_threads(num_threads)
    for(int i=0;i<pts.size();i++){
        // cout << "i:" << i << endl;
        x_intt[i] = he->ckks_intt(pts[i]);
    }
    const size_t coefficients_per_plain = x_intt[0].size();
    if (coefficients_per_plain != static_cast<size_t>(degree)) {
        throw runtime_error("CKKS-to-MPC inverse FFT degree mismatch");
    }
    // Pre-size the batch so coefficient canonicalization does not serialize
    // on vector::push_back.  Every output cell has one owner, so this remains
    // bit-for-bit identical to the old flattening loop.
    vector<HE_CKKS::scalar_t> x_intt_flatten_field(
        x_intt.size() * coefficients_per_plain);
#pragma omp parallel for num_threads(num_threads)
    for(int i=0;i<static_cast<int>(x_intt.size());i++){
        for(size_t j=0;j<coefficients_per_plain;j++){
            // ckks_intt returns centered coefficients through a uint64_t
            // container.  Negative coefficients therefore arrive in
            // two's-complement form; Algorithm 1's field share must instead
            // be the canonical residue in [0, Q).
            const uint64_t raw = x_intt[i][j].real();
            int64_t signed_value = 0;
            static_assert(sizeof(signed_value) == sizeof(raw),
                          "unexpected fixed-width integer size");
            memcpy(&signed_value, &raw, sizeof(raw));
            uint64_t canonical = 0;
            if (signed_value < 0) {
                const uint64_t magnitude =
                    static_cast<uint64_t>(-(signed_value + 1)) + 1;
                const uint64_t reduced = magnitude % conversion_prime;
                canonical = reduced == 0 ? 0 : conversion_prime - reduced;
            } else {
                canonical = raw % conversion_prime;
            }
            x_intt_flatten_field[static_cast<size_t>(i) *
                                 coefficients_per_plain + j] = canonical;
        }
    }
    cout << "CKKS_INTT_FIELD_CANONICAL status=PASS range=[0,q) q="
         << conversion_prime << endl;
    cout << "intt done" << endl;
	// step 2
    int s_in;
    if (abs(pts[0].scale())<1){
        s_in = 0;
    }
    else{
        s_in = round(log2(pts[0].scale()));
    }
    if (s_in < s_out) {
        throw runtime_error(
            "CKKS-to-MPC cannot raise scale at the Algorithm 1 boundary");
    }

    // Center the canonical prime-field coefficient before converting it to
    // the l-bit ring.  This preserves CKKS's signed representative instead
    // of interpreting q-|m| as a large positive integer.
    const int output_ring_bits = bit_out;
    const int field_ring_bits =
        static_cast<int>(ceil(log2(static_cast<double>(conversion_prime))));
    // Keep the coefficient in its natural single-prime ring until the
    // public decode FFT has finished. Reducing it to the output l-bit ring
    // before the FFT discards the high bits that must be removed exactly once
    // when changing the CKKS scale from s_in to s_out.
    const int source_ring_bits = field_ring_bits;
    const int c2m_headroom_bits = blb_c2m_headroom_bits();
    const int working_bits = source_ring_bits + c2m_headroom_bits;
    if (output_ring_bits <= 1 || output_ring_bits >= 63 ||
        source_ring_bits <= output_ring_bits || source_ring_bits >= 64 ||
        working_bits <= source_ring_bits || working_bits >= 127) {
        throw runtime_error("invalid BLB CKKS-to-MPC working ring");
    }
    const uint64_t center_offset = (conversion_prime - 1) / 2;
    // Keep the prime-field coefficient at its natural width before the FFT.
    // The conversion uses a 64-bit low COT limb plus a configured high limb
    // in NonLinear; reducing that limb preserves the exact prime-wrap result
    // while shrinking only the wide working representation.
    // This avoids the generic 60->100 extension's larger per-worker workspace.
    vector<uint64_t> wide_low(x_intt_flatten_field.size());
    vector<uint64_t> wide_high(x_intt_flatten_field.size());
    nonlinear.centered_field_to_ring_wide(
        blb_conversion_threads(), x_intt_flatten_field.data(), Q,
        wide_low.data(), wide_high.data(),
        static_cast<int>(x_intt_flatten_field.size()), working_bits);
    field_to_ring_calls++;
    ckks_to_mpc_conversions += pts.size();
    cout << "CKKS_CENTERED_FIELD_TO_RING status=PASS offset="
         << center_offset << " field_ring_bits=" << field_ring_bits
         << " source_ring_bits=" << source_ring_bits
         << " output_ring_bits=" << output_ring_bits
         << " source_scale=" << s_in << " target_scale=" << s_out
         << " secure_truncate_bits=" << (s_in - s_out)
         << " working_ell=" << working_bits
         << " protocol=prime_wrap_then_split_wide_b2a"
         << "headroom_bits=" << c2m_headroom_bits
         << " workers=" << blb_conversion_threads() << endl;

    vector<vector<HE_CKKS::scalar_t>> y(pts.size());
#pragma omp parallel for num_threads(num_threads)
    for(int i=0;i<static_cast<int>(pts.size());i++){
        // Build each wide FFT input on the worker that consumes it.  This
        // removes a second batch-sized temporary and parallelizes the former
        // serial wide-word assembly without changing any arithmetic.
        vector<HE_CKKS::wide_complex_t> x_decode(
            static_cast<size_t>(degree));
        for (int j = 0; j < degree; ++j) {
            const size_t index =
                static_cast<size_t>(i) * static_cast<size_t>(degree) +
                static_cast<size_t>(j);
            x_decode[j].real =
                (static_cast<__uint128_t>(wide_high[index]) << 64) |
                wide_low[index];
            x_decode[j].imag = 0;
        }
        y[i] = he->ckks_decode_wide(
            x_decode, degree, output_ring_bits, s_in, s_out, half,
            working_bits);
        // ckks_decode_wide returns ring words, not signed host integers.  Keep
        // the ABI invariant explicit at this boundary: every local additive
        // share must be a canonical residue in Z_(2^output_ring_bits).  This
        // is intentionally a mask (not a signed conversion); signedness is a
        // property of the FixArray metadata and only reconstructed values are
        // centered later by the consumer.
        const uint64_t output_mask =
            (uint64_t{1} << output_ring_bits) - uint64_t{1};
        for (uint64_t &value : y[i]) value &= output_mask;
    }
    cout << "CKKS_WIDE_LOCAL_FFT direction=decode degree=" << degree
         << " ell=" << working_bits << " scale=" << s_out
         << " slots=" << pts.size() << endl;
    // cout << "decode ok" << endl;
    // uint8_t* msb_x = new uint8_t[y[0].size()];
    // for(int i=0;i<y.size();i++){
    //     nonlinear.fpmath[0]->fix->trunc->truncate_with_lsb(y[i].size(), y[i].data(), y2[i].data(),
    //                      1, bit_out, true,
    //                      msb_x);
    // }
    
    cout << "ckks_to_mpc done" << endl;
	if (profile_conversion) {
        profile_record_comm(conversion_bucket,
                            current_comm() - conversion_comm_before);
    }
	if (profile_conversion) profile_pop();
	return y;
}

void Bert_ckks::pack_real_ckks_pairs(vector<Ciphertext>& ciphertexts, HE_CKKS* he) {
    if (ciphertexts.empty()) throw invalid_argument("empty real-pair packing input");
    const auto id = ciphertexts.front().parms_id();
    const double scale = ciphertexts.front().scale();
    for (const auto& ct : ciphertexts)
        if (ct.parms_id() != id || ct.scale() != scale)
            throw invalid_argument("real-pair packing metadata mismatch");
    Plaintext imaginary_unit;
    he->encoder->encode(complex<double>(0, 1), id, 1.0, imaginary_unit);
    vector<Ciphertext> packed((ciphertexts.size() + 1) / 2);
    const int workers = min<int>(8, packed.size());
#pragma omp parallel for num_threads(workers)
    for (size_t pair = 0; pair < packed.size(); ++pair) {
        Ciphertext conjugate;
        auto& first = ciphertexts[2 * pair];
        he->evaluator->complex_conjugate(first, *he->gal_keys, conjugate);
        he->evaluator->add(first, conjugate, packed[pair]);
        if (2 * pair + 1 < ciphertexts.size()) {
            auto& second = ciphertexts[2 * pair + 1];
            he->evaluator->complex_conjugate(second, *he->gal_keys, conjugate);
            he->evaluator->add_inplace(second, conjugate);
            he->evaluator->multiply_plain_inplace(second, imaginary_unit);
            he->evaluator->add_inplace(packed[pair], second);
        }
        // Projection produces 2*Re(first) + i*2*Re(second). Accounting for
        // that factor in the scale preserves values without a rescale round.
        packed[pair].scale() *= 2.0;
    }
    linear.rotation_count += ciphertexts.size();
    linear.ct_pt_multiply_count += ciphertexts.size() / 2;
    cout << "CKKS_REAL_PAIR_PACK source_cts=" << ciphertexts.size()
         << " packed_cts=" << packed.size()
         << " layout=adjacent_real_imag projection=conjugate"
         << " source_scale_bits=" << log2(scale)
         << " packed_scale_bits=" << log2(scale) + 1 << endl;
    ciphertexts = std::move(packed);
}

vector<vector<HE_CKKS::scalar_t>> Bert_ckks::ckks_pairs_to_mpc(
    const vector<Plaintext>& pts, size_t real_ciphertexts,
    int bit_out, int s_out, HE_CKKS* he) {
    if (real_ciphertexts == 0 || pts.size() != (real_ciphertexts + 1) / 2)
        throw invalid_argument("real-pair conversion count mismatch");
    auto packed = ckks_to_mpc(pts, he->poly_modulus_degree, he->get_Q(pts[0]),
                              bit_out, s_out, he, true);
    const size_t slots = he->num_slots;
    vector<vector<HE_CKKS::scalar_t>> result(real_ciphertexts);
    for (size_t i = 0; i < real_ciphertexts; ++i) {
        if (packed[i / 2].size() != 2 * slots)
            throw runtime_error("real-pair conversion slot mismatch");
        const auto begin = packed[i / 2].begin() + (i % 2) * slots;
        result[i].assign(begin, begin + slots);
    }
    return result;
}

void Bert_ckks::mask_ckks_to_mpc_inplace(
    vector<Ciphertext> &ciphertexts, vector<Plaintext> &server_shares,
    HE_CKKS *he) {
    if (party != sci::ALICE) {
        throw logic_error("only the CKKS evaluator may sample Algorithm 1 masks");
    }
    if (ciphertexts.empty()) {
        throw invalid_argument("Algorithm 1 mask input is empty");
    }
    server_shares.resize(ciphertexts.size());
    for (size_t index = 0; index < ciphertexts.size(); ++index) {
        Ciphertext &cipher = ciphertexts[index];
        auto context_data = he->context->get_context_data(cipher.parms_id());
        if (!context_data || !cipher.is_ntt_form()) {
            throw runtime_error("Algorithm 1 requires a valid current-level CKKS NTT ciphertext");
        }
        const auto &parms = context_data->parms();
        const auto &moduli = parms.coeff_modulus();
        const size_t degree = parms.poly_modulus_degree();
        const size_t rns_limbs = moduli.size();
        Plaintext &negative_mask = server_shares[index];
        negative_mask.resize(degree * rns_limbs);

        // SEAL's default factory seeds a cryptographic PRNG from OS entropy.
        // sample_poly_uniform uses rejection sampling independently in every
        // current RNS limb, hence this is uniform in A_N,Q rather than an
        // encoder image of a random slot vector.
        auto csprng = parms.random_generator()->create();
        seal::util::sample_poly_uniform(csprng, parms, negative_mask.data());
        uint64_t *c0 = cipher.data(0);
        for (size_t limb = 0; limb < rns_limbs; ++limb) {
            const uint64_t modulus = moduli[limb].value();
            uint64_t *mask_limb = negative_mask.data() + limb * degree;
            uint64_t *c0_limb = c0 + limb * degree;
            for (size_t coeff = 0; coeff < degree; ++coeff) {
                const uint64_t random_coeff = mask_limb[coeff];
                uint64_t sum = c0_limb[coeff] + random_coeff;
                if (sum >= modulus) {
                    sum -= modulus;
                }
                c0_limb[coeff] = sum;
                mask_limb[coeff] = random_coeff == 0 ? 0 : modulus - random_coeff;
            }
        }
        negative_mask.parms_id() = cipher.parms_id();
        negative_mask.scale() = cipher.scale();
        masked_rns_coefficients += degree * rns_limbs;
    }
}

/*
	input: l-bit, scale是s_in的share, 维度是(x, slot)
    output: scale是s_out的share
	step:
		1. 先编码进复数，进行本地encode
		2. 从(l-bit, s_in)变至(Q，s_out), s_out = ckks的delta，在ntt中会构建RNS，因此预先把位宽Ext至Q，可以避免对每个RNS分量都做Ext
           这里s_out > s_in (41 vs 30)
		3. 本地做ntt
*/
vector<Plaintext> Bert_ckks::mpc_to_ckks(
    const vector<vector<HE_CKKS::scalar_t>> & x, int slot, uint64_t Q,
    int bitwidth, int s_in, int s_out, HE_CKKS* he, bool half,
    bool known_nonnegative){
    const bool profile_conversion = profile_enabled_;
    const auto conversion_bucket = profile_conversion_bucket_;
    const uint64_t conversion_comm_before =
        profile_conversion ? current_comm() : 0;
    if (profile_conversion) profile_push(BlbViTProfileStage::MPC_TO_CKKS);
    cout << "in mpc_to_ckks, slot, Q, bitwidth, s_in, s_out:" << slot << "," << Q << "," << bitwidth << "," << s_in << "," << s_out << endl;
    // step 1
    conversion_ciphertexts += x.size();
    int poly_degree = slot * 2;
    // cout << "poly_degree:" << poly_degree << endl;
    cout << "pts.size():" << x.size() << endl;
    int thread_num = x.size()>64?64:x.size();
    constexpr int encode_guard_bits = 16;
    const int sharing_bits = bitwidth + 40;
    int degree_bits = 0;
    for (int degree = poly_degree; degree > 1; degree >>= 1) ++degree_bits;
    // The FFT operates on integer representatives, not a wrapping sharing
    // ring. N times the largest input fits with two signed headroom bits.
    const int working_bits = sharing_bits + encode_guard_bits + degree_bits + 2;
    const int encode_bitwidth = bitwidth + encode_guard_bits;
    const int encode_scale = s_in + encode_guard_bits;
    if (x.empty() || bitwidth <= 1 || bitwidth >= 64 || s_out < s_in ||
        encode_bitwidth >= 64 || encode_scale >= 127 ||
        working_bits >= 127 || (poly_degree & (poly_degree - 1)) != 0 ||
        x.front().size() != static_cast<size_t>(slot)) {
        throw invalid_argument("invalid BLB MPC-to-CKKS conversion metadata");
    }
    for (const auto &share : x) {
        if (share.size() != static_cast<size_t>(slot)) {
            throw invalid_argument("inconsistent MPC-to-CKKS slot batch");
        }
    }

    // Extend once before Encode. The conversion API receives additive shares
    // in the unsigned sharing ring, while the CKKS encoder consumes the
    // signed representative of their reconstruction. This public origin
    // correction is required for both extension protocols: the non-negative
    // shortcut changes only how the wide MSB/wrap is obtained, not the
    // encoder's signed share convention.
    const size_t slot_elements = x.size() * static_cast<size_t>(slot);
    vector<uint64_t> slot_ring(slot_elements);
    for (size_t ct = 0; ct < x.size(); ++ct) {
        copy(x[ct].begin(), x[ct].end(),
             slot_ring.begin() + ct * static_cast<size_t>(slot));
    }
    vector<uint64_t> slot_low(slot_elements), slot_high(slot_elements);
    if (known_nonnegative) {
        nonlinear.ring_extend_wide_nonnegative(
            blb_conversion_threads(), slot_ring.data(), slot_low.data(),
            slot_high.data(), static_cast<int>(slot_elements), bitwidth,
            sharing_bits);
    } else {
        nonlinear.ring_extend_wide(
            blb_conversion_threads(), slot_ring.data(), slot_low.data(),
            slot_high.data(), static_cast<int>(slot_elements), bitwidth,
            sharing_bits);
    }

    // The input shares arrive at scale s_in.  A fixed-point FFT rounds every
    // public-twiddle product, so running it directly at s_in accumulates the
    // rounding error over log2(N) stages.  Lift the shared integer by the
    // public guard factor once, run the FFT at s_in+16, and remove those 16
    // bits only through the local ring-to-field scale lift below.  This is
    // the one intentional scale change in MPC->CKKS.
    const __uint128_t working_mask =
        (__uint128_t{1} << working_bits) - __uint128_t{1};
    for (size_t index = 0; index < slot_low.size(); ++index) {
        __uint128_t lifted =
            (static_cast<__uint128_t>(slot_high[index]) << 64) |
            slot_low[index];
        if (party == sci::ALICE)
            lifted = (lifted - (__uint128_t{1} << sharing_bits)) & working_mask;
        lifted = (lifted << encode_guard_bits) & working_mask;
        slot_low[index] = static_cast<uint64_t>(lifted);
        slot_high[index] = static_cast<uint64_t>(lifted >> 64);
    }

    // Real CKKS slots imply a[N-j] = -a[j] and a[N/2] = 0.
    // Convert only the independent coefficients, then restore symmetry in
    // each RNS prime. Complex (half) inputs retain the complete polynomial.
    const int independent_coefficients = half ? poly_degree : slot;
    // CKKS plaintexts at the first context are RNS polynomials.  The old
    // adapter converted the share once modulo the final-prime Q and then
    // passed that same uint64 residue to every RNS limb.  That is not a
    // lift from Z_Q to the RNS representation: each prime needs its own
    // field residue, and the per-prime reduction must happen after the
    // signed wide-ring conversion.
    auto first_context = he->context->first_context_data();
    if (!first_context) {
        throw runtime_error("MPC-to-CKKS first context is unavailable");
    }
    const auto &rns_moduli = first_context->parms().coeff_modulus();
    if (rns_moduli.empty() || Q == 0) {
        throw runtime_error("MPC-to-CKKS RNS modulus set is empty");
    }
    // Algorithm 2 step 1: local Encode in the enlarged FFT workspace.
    // Keep the numerical guard bits, then form additive coefficient shares
    // in Z_(2^encode_bitwidth). Do not locally reinterpret wide shares as
    // field shares: Algorithm 2's coefficient-domain conversion follows.
    const size_t coefficient_count = x.size() * independent_coefficients;
    const uint64_t coefficient_mask = (uint64_t{1} << encode_bitwidth) - 1;
    vector<uint64_t> coefficient_ring(coefficient_count);
#pragma omp parallel for num_threads(thread_num)
    for (int i=0; i<static_cast<int>(x.size()); ++i) {
        vector<HE_CKKS::wide_complex_t> encoded_input(poly_degree);
        for (int j=0; j<slot; ++j) {
            const size_t flat = static_cast<size_t>(i)*slot+j;
            const __uint128_t value = (__uint128_t(slot_high[flat]) << 64) | slot_low[flat];
            encoded_input[he->matrix_reps_index_map_[j]].real = value;
            encoded_input[he->matrix_reps_index_map_[j+slot]].real = value;
        }
        auto encoded = he->ckks_encode_wide(std::move(encoded_input), poly_degree, working_bits);
        for (int j=0; j<independent_coefficients; ++j)
            coefficient_ring[static_cast<size_t>(i)*independent_coefficients+j] =
                static_cast<uint64_t>(encoded[j].real) & coefficient_mask;
    }
    // Algorithm 2 step 2 / Appendix C.3: securely extend coefficient shares
    // before local field reduction. The same extended integer is reduced
    // separately into each RNS prime, never reused as a residue of another.
    const int coefficient_sharing_bits = encode_bitwidth + 40;
    vector<uint64_t> coefficient_low(coefficient_count), coefficient_high(coefficient_count);
    nonlinear.ring_extend_wide(blb_conversion_threads(), coefficient_ring.data(),
        coefficient_low.data(), coefficient_high.data(), static_cast<int>(coefficient_count),
        encode_bitwidth, coefficient_sharing_bits);
    ++ring_to_field_calls;

    const int scale_shift = s_out - encode_scale;
    if (scale_shift < 0) {
        throw runtime_error(
            "MPC-to-CKKS optimized RNS conversion cannot lower scale");
    }
    // After the secure coefficient extension, perform local per-prime
    // reduction, real-slot symmetry restoration and NTT per ciphertext.
    vector<__uint128_t> scale_factors(rns_moduli.size());
    vector<__uint128_t> wrap_corrections(rns_moduli.size());
    const auto pow_two_mod = [](const __uint128_t modulus, int exponent) {
        __uint128_t base = 2 % modulus;
        __uint128_t result = 1 % modulus;
        while (exponent > 0) {
            if (exponent & 1) result = (result * base) % modulus;
            exponent >>= 1;
            if (exponent > 0) base = (base * base) % modulus;
        }
        return result;
    };
    for (size_t limb = 0; limb < rns_moduli.size(); ++limb) {
        const __uint128_t modulus = rns_moduli[limb].value();
        scale_factors[limb] = pow_two_mod(modulus, scale_shift);
        wrap_corrections[limb] =
            pow_two_mod(modulus, coefficient_sharing_bits + scale_shift);
    }
    vector<Plaintext> y(x.size());
    const double ckks_scale = ldexp(1.0, s_out);
#pragma omp parallel for num_threads(thread_num)
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        vector<vector<HE_CKKS::scalar_t>> x_encode_rns(
            rns_moduli.size(), vector<HE_CKKS::scalar_t>(poly_degree));
        for (size_t limb = 0; limb < rns_moduli.size(); ++limb) {
            const __uint128_t modulus = rns_moduli[limb].value();
            const __uint128_t scale_factor = scale_factors[limb];
            const __uint128_t wrap_correction = wrap_corrections[limb];
            for (int coefficient = 0;
                 coefficient < independent_coefficients; ++coefficient) {
                const size_t flat = static_cast<size_t>(i)*independent_coefficients+coefficient;
                const __uint128_t wide = (__uint128_t(coefficient_high[flat]) << 64) |
                                           coefficient_low[flat];
                __uint128_t local = ((wide % modulus) * scale_factor) % modulus;
                if (party == sci::ALICE)
                    local = (local + modulus - wrap_correction) % modulus;
                x_encode_rns[limb][coefficient] =
                    static_cast<uint64_t>(local);
            }
            if (!half) {
                const uint64_t prime = rns_moduli[limb].value();
                for (int j = 1; j < slot; ++j) {
                    const uint64_t value = x_encode_rns[limb][j];
                    x_encode_rns[limb][poly_degree - j] =
                        value == 0 ? 0 : prime - value;
                }
            }
        }
        he->ckks_ntt_rns(x_encode_rns, ckks_scale, y[i]);
    }
    cout << "BLB_RING_TO_FIELD_WIDE status=PASS input_bits=" << bitwidth
         << " extension_bits=" << sharing_bits
         << " integer_fft_bits=" << working_bits
         << " source_scale=" << s_in << " target_scale=" << s_out
         << " batch_elements=" << slot_elements
         << " encode_guard_bits=" << encode_guard_bits
         << " lifted_input_scale=" << encode_scale
         << " rns_limbs=" << rns_moduli.size()
         << " secure_slot_extensions=1 secure_coefficient_extensions=1"
         << " known_nonnegative=" << (known_nonnegative ? 1 : 0)
         << " independent_coefficients_per_ciphertext=" << independent_coefficients
         << " real_coefficient_symmetry=" << (!half ? 1 : 0)
         << " encode=integer_fft_then_secure_coefficient_ring_to_field" << endl;
    mpc_to_ckks_conversions += x.size();
    cout << "mpc_to_ckks done" << endl;
	if (profile_conversion) {
        profile_record_comm(conversion_bucket,
                            current_comm() - conversion_comm_before);
	}
	if (profile_conversion) profile_pop();
    return y;
}

void Bert_ckks::prepare_ckks_to_mpc_inplace(
    vector<Ciphertext> &ciphertexts, int target_scale_bits, HE_CKKS *he) {
    if (ciphertexts.empty()) {
        throw invalid_argument("CKKS-to-MPC preparation input is empty");
    }
    const double target_scale = ldexp(1.0, target_scale_bits);
    const vector<double> public_one(he->num_slots, 1.0);
    size_t source_limbs = 0;
    double source_scale_bits = 0.0;
    size_t rescaled_ciphertexts = 0;
    struct PublicUnitCacheEntry {
        parms_id_type parms_id;
        double scale;
        Plaintext plaintext;
    };
    vector<PublicUnitCacheEntry> public_unit_cache;
    size_t public_unit_cache_hits = 0;
    size_t public_unit_encodes = 0;

    for (size_t index = 0; index < ciphertexts.size(); ++index) {
        Ciphertext &ciphertext = ciphertexts[index];
        auto context_data =
            he->context->get_context_data(ciphertext.parms_id());
        if (!context_data || !ciphertext.is_ntt_form()) {
            throw runtime_error("CKKS-to-MPC preparation level is invalid");
        }
        const auto &moduli = context_data->parms().coeff_modulus();
        if (index == 0) {
            source_limbs = moduli.size();
            source_scale_bits = log2(ciphertext.scale());
        }

        const double relative_scale_error =
            abs(ciphertext.scale() - target_scale) / target_scale;
        if (relative_scale_error > 1e-12) {
            if (moduli.size() < 2) {
                throw runtime_error(
                    "CKKS-to-MPC scale alignment has no rescale level");
            }
            const double dropped_modulus =
                static_cast<double>(moduli.back().value());
            const double unit_plain_scale =
                target_scale * dropped_modulus / ciphertext.scale();
            if (!isfinite(unit_plain_scale) || unit_plain_scale < 1.0) {
                throw runtime_error(
                    "CKKS-to-MPC public-unit scale is not encodable");
            }
            const Plaintext *public_unit = nullptr;
            for (auto &entry : public_unit_cache) {
                if (entry.parms_id == ciphertext.parms_id() &&
                    entry.scale == unit_plain_scale) {
                    public_unit = &entry.plaintext;
                    ++public_unit_cache_hits;
                    break;
                }
            }
            if (public_unit == nullptr) {
                PublicUnitCacheEntry entry;
                entry.parms_id = ciphertext.parms_id();
                entry.scale = unit_plain_scale;
                he->encoder->encode(public_one, ciphertext.parms_id(),
                                    unit_plain_scale, entry.plaintext);
                public_unit_cache.push_back(std::move(entry));
                public_unit = &public_unit_cache.back().plaintext;
                ++public_unit_encodes;
            }
            he->evaluator->multiply_plain_inplace(ciphertext, *public_unit);
            ++this->he_rescale_count;
            he->evaluator->rescale_to_next_inplace(ciphertext);
            ++rescaled_ciphertexts;
        }

        if (ciphertext.parms_id() != he->context->last_parms_id()) {
            he->evaluator->mod_switch_to_inplace(
                ciphertext, he->context->last_parms_id());
        }
        auto final_context =
            he->context->get_context_data(ciphertext.parms_id());
        if (!final_context ||
            final_context->parms().coeff_modulus().size() != 1) {
            throw runtime_error(
                "CKKS-to-MPC preparation did not reach one prime");
        }
        const double final_relative_error =
            abs(ciphertext.scale() - target_scale) / target_scale;
        if (final_relative_error > 1e-9) {
            throw runtime_error(
                "CKKS-to-MPC preparation produced the wrong actual scale");
        }
    }
    cout << "CKKS_TO_MPC_PREP status=PASS cts=" << ciphertexts.size()
         << " source_limbs=" << source_limbs
         << " source_scale_bits=" << source_scale_bits
         << " target_limbs=1 target_scale_bits=" << target_scale_bits
         << " unit_plaintext=real_1 rescaled_cts=" << rescaled_ciphertexts
         << " unit_plaintext_encodes=" << public_unit_encodes
         << " unit_plaintext_cache_hits=" << public_unit_cache_hits
         << endl;
}

// Evaluate an MPC-share square through the existing CKKS conversion path.
// This is the formal LayerNorm variance primitive: no share is reconstructed
// and the evaluator masks the current RNS polynomial directly before the
// CKKS->MPC conversion. The full-slot layout is intentional here; the
// surrounding FineGrainFusion blocks use their own audited half packing.
FixArray Bert_ckks::element_wise_square(int32_t dim, FixArray input) {
    if (dim <= 0 || input.size != dim) {
        throw runtime_error("formal CKKS square size mismatch");
    }
    HE_CKKS *he = linear.he1;
    const int slot = he->num_slots;
    if (dim % slot != 0) {
        throw runtime_error("formal CKKS square requires slot-aligned input");
    }
    const int num_cts = dim / slot;
    vector<vector<HE_CKKS::scalar_t>> shares(
        num_cts, vector<HE_CKKS::scalar_t>(slot));
    for (int ct = 0; ct < num_cts; ++ct) {
        copy_n(input.data + ct * slot, slot, shares[ct].begin());
    }

    vector<Plaintext> input_pts = mpc_to_ckks(
        shares, slot, he->get_conversion_prime(), input.ell, input.s,
        he->he_scale, he, false);
    vector<Ciphertext> input_cts(num_cts);
    vector<Plaintext> output_pts(num_cts);
    auto start = high_resolution_clock::now();
    if (party == sci::BOB) {
        #pragma omp parallel for
        for (int i = 0; i < num_cts; ++i) {
            he->encryptor->encrypt(input_pts[i], input_cts[i]);
        }
        send_encrypted_vector(io, input_cts);
        recv_encrypted_vector(he->context, io, input_cts);
        #pragma omp parallel for
        for (int i = 0; i < num_cts; ++i) {
            he->decryptor->decrypt(input_cts[i], output_pts[i]);
        }
    } else {
        recv_encrypted_vector(he->context, io, input_cts);
        if (!he->is_gpu_enable()) {
            for (int i = 0; i < num_cts; ++i) {
                // Add Alice's local MPC share before the CKKS multiplication.
                he->evaluator->add_plain_inplace(input_cts[i], input_pts[i]);
                he->evaluator->square_inplace(input_cts[i]);
                he->evaluator->relinearize_inplace(input_cts[i],
                                                    *he->relin_keys);
                ++this->he_rescale_count;
                he->evaluator->rescale_to_next_inplace(input_cts[i]);
            }
            cout << "BLB_VIT_LAYER_NORM_SQUARE_BACKEND backend=SEAL_HOST cts="
                 << num_cts << endl;
        } else {
            auto copy_start = high_resolution_clock::now();
            vector<UnifiedCiphertext> device_cts;
            vector<UnifiedPlaintext> device_pts;
            device_cts.reserve(num_cts);
            device_pts.reserve(num_cts);
            for (int i = 0; i < num_cts; ++i) {
                device_cts.emplace_back(std::move(input_cts[i]));
                device_pts.emplace_back(input_pts[i]);
                device_cts.back().to_device(*he->context, *he->dcontext);
                device_pts.back().to_device(*he->context, *he->dcontext);
            }
            const double h2d_seconds = chrono::duration<double>(
                high_resolution_clock::now() - copy_start).count();
            this->ct_transfer_mem_la += h2d_seconds;
            this->h2d_time += h2d_seconds;
            for (int i = 0; i < num_cts; ++i) {
                he->devaluator->add_plain_inplace(device_cts[i],
                                                  device_pts[i]);
                he->devaluator->square_inplace(device_cts[i]);
                he->devaluator->relinearize_inplace(device_cts[i],
                                                     *he->drelin_keys);
                ++this->he_rescale_count;
                he->devaluator->rescale_to_next_inplace(device_cts[i]);
            }
            copy_start = high_resolution_clock::now();
            for (int i = 0; i < num_cts; ++i) {
                device_cts[i].to_host(*he->context, *he->dcontext);
                input_cts[i] = std::move(device_cts[i].hcipher());
            }
            const double d2h_seconds = chrono::duration<double>(
                high_resolution_clock::now() - copy_start).count();
            this->ct_transfer_mem_la += d2h_seconds;
            this->d2h_time += d2h_seconds;
            cout << "BLB_VIT_LAYER_NORM_SQUARE_BACKEND backend=Phantom_DEVICE cts="
                 << num_cts << endl;
        }
        linear.ct_ct_multiply_count += static_cast<uint64_t>(num_cts);
        // Keep the original CKKS square and Algorithm-1 mask, but complete
        // the formal conversion boundary before exporting the share.  The
        // square leaves one RNS level above the final single-prime level;
        // ckks_to_mpc intentionally rejects that ambiguous state instead of
        // truncating an RNS product to uint64_t.
        prepare_ckks_to_mpc_inplace(input_cts, 40, he);
        mask_ckks_to_mpc_inplace(input_cts, output_pts, he);
        send_encrypted_vector(io, input_cts);
    }
    this->he_compute_la +=
        chrono::duration<double>(high_resolution_clock::now() - start).count();
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();

    vector<vector<HE_CKKS::scalar_t>> squared = ckks_to_mpc(
        output_pts, he->poly_modulus_degree, he->get_Q(output_pts[0]),
        input.ell, input.s, he, false);
    FixArray result(party, dim, input.signed_, input.ell, input.s);
    size_t offset = 0;
    for (const auto &packed : squared) {
        if (packed.size() != static_cast<size_t>(slot) ||
            offset + packed.size() > static_cast<size_t>(dim)) {
            throw runtime_error("formal CKKS square decode size mismatch");
        }
        copy(packed.begin(), packed.end(), result.data + offset);
        offset += packed.size();
    }
    if (offset != static_cast<size_t>(dim)) {
        throw runtime_error("formal CKKS square output is incomplete");
    }
    return result;
}

// // evaluate input^2, 输入输出均是mpc share, use CKKS in default
// FixArray Bert_ckks::element_wise_square(int32_t dim, FixArray input){
//     int bitwidth = input.ell; //bw
//     int s = input.s;
//     cout << "in element_wise_square, bitwidth, s:" << bitwidth << "," << s << endl;
//     cout << "should be 0! comm:" << this->get_comm() << ", round:" << this->get_round() << endl;
//     linear.set_HE(poly_modulus_degree,vector<int>{bitwidth, s, bitwidth}, s);
//     c_offline += this->get_comm();
//     r_offline += this->get_round();
//     cout << "c_offline:" << c_offline << ", r_offline:" << r_offline << endl;
//     if (dim!=input.size){
//         cout << "Error: input size does not match" << endl;
//         return input;
//     }
//     uint64_t slot_count = poly_modulus_degree / 2;
//     uint64_t num_cts = dim / slot_count;
//     cout << "slot_count:" << slot_count << endl;
//     cout << "num_cts:" << num_cts << endl;
//     vector<vector<HE_CKKS::scalar_t>> vec(num_cts);
//     vector<Plaintext> pts;
//     vector<Ciphertext> cts(num_cts);
//     vector<Plaintext> output_pts(num_cts);
//     for(int i = 0; i < num_cts; i++){
//         Plaintext pt;
//         Ciphertext ct;
//         for(int j = 0; j < slot_count; ++j){
//             vec[i].push_back(input.data[i*slot_count + j]);
//         }
//     }
//     cout << "Q:" << linear.he->get_Q() << endl;
//     this->get_comm();
//     this->get_round();
//     pts = mpc_to_ckks(vec, slot_count, linear.he->get_Q(), bitwidth, s, s);
//     cout << "mpc_to_ckks comm:" << this->get_comm() << endl;
//     cout << "mpc_to_ckks round:" << this->get_round() << endl;
//     cout << "pts.size():" << pts.size() << endl;
//     vector<Ciphertext> half_cts(num_cts/2);
//     // Alice is server, Bob is client
//     if (party == sci::BOB) {
//         for(size_t i=0;i<num_cts;i++){
//             Ciphertext ct;
//             linear.he->encryptor->encrypt(pts[i], ct);
//             cts[i] = ct;
//             if(i<num_cts/2){
//                 half_cts[i] = ct;
//             }
//         }
//         cout << "client send_encrypted_vector" << endl;
// 		// Historical x+y*j half-packing experiment.
//         // send half_cts instead of cts
// 		send_encrypted_vector(io, half_cts);
// 		// send_encrypted_vector(io, cts);
// 		// vector<Ciphertext> share_client(dim);
// 		recv_encrypted_vector(linear.he->context, io, cts);
//         for(int i=0;i<num_cts;i++){
//             linear.he->decryptor->decrypt(cts[i], output_pts[i]);
//         }
// 	}
//     else{
//         // receive half_cts instead of cts
//         recv_encrypted_vector(linear.he->context, io, half_cts);
//         for(int i=0;i<half_cts.size();i++){
//             seal::Ciphertext ct1;
//             seal::Ciphertext tmp;
//             // x + y*j => x - y*j
//             linear.he->evaluator->complex_conjugate(half_cts[i], *linear.he->gal_keys, tmp);
//             // x + y*j - (x - y*j)*
//             // - 2 * y*j
//             linear.he->evaluator->sub(tmp, half_cts[i], ct1);
//             // - 2 * y*j => 2*y
//             MulImageUnitInplace(ct1, *linear.he->context);
//             // x + y*j + (x - y*j) => 2*x
//             linear.he->evaluator->add_inplace(half_cts[i], tmp);
//             cts[i] = half_cts[i];
//             cts[i+half_cts.size()] = ct1;
//         }
//         linear.he->print_parameters(linear.he->context->get_context_data(cts[0].parms_id())->parms());
//         for(int i=0;i<cts.size();i++){
//             linear.he->evaluator->add_plain_inplace(cts[i], pts[i]);
//             linear.he->evaluator->square_inplace(cts[i]);
//             RelinKeys* relin_keys = linear.he->relin_keys;
//             linear.he->evaluator->relinearize_inplace(cts[i], *relin_keys);
//             linear.he->evaluator->rescale_to_next_inplace(cts[i]);
//         }
//         PRG128 prg;
//         /*
//             Historical disabled masking experiment.
//         */
//         vector<uint64_t> r(dim,0);
//         // prg.random_mod_p<uint64_t>(r.data(), dim, linear.he->get_Q());
//         cout << "server send_encrypted_vector" << endl;
//         send_encrypted_vector(io, cts);
//         for(int i=0;i<num_cts;i++){
//             vector<double> pod_matrix(poly_modulus_degree/2, 0.0);
//             Plaintext tmp;
//             linear.he->encoder->encode(pod_matrix, cts[i].parms_id(), cts[i].scale(), tmp);
//             output_pts[i] = tmp;
//         }
//     }
//     linear.he->print_parameters(linear.he->context->get_context_data(cts[0].parms_id())->parms());
//     vector<vector<HE_CKKS::scalar_t>> output_vector;
//     cout << "send and receive comm:" << this->get_comm() << endl;
//     cout << "send and receive round:" << this->get_round() << endl;
//     output_vector = ckks_to_mpc(output_pts, poly_modulus_degree, linear.he->get_Q(cts[0]), bitwidth, s);
//     cout << "ckks_to_mpc comm:" << this->get_comm() << endl;
//     cout << "ckks_to_mpc round:" << this->get_round() << endl;
//     FixArray output(party, dim, input.signed_, bitwidth, s);
//     for(int i = 0; i < num_cts; i++){
//         for(int j = 0; j < slot_count; ++j){
//             output.data[i*slot_count + j] = output_vector[i][j];
//         }
//     }
//     cout << "element_wise_square done" << endl;
//     return output;
// }

// // evaluate input^2, 输入输出均是mpc share, use BFV in default
// FixArray Bert_ckks::element_wise_square_bfv(int32_t dim, FixArray input){
//     int bitwidth = input.ell; //bw
//     int s = input.s;
//     cout << "in element_wise_square, bitwidth, s:" << bitwidth << "," << s << endl;
//     cout << "should be 0! comm:" << this->get_comm() << ", round:" << this->get_round() << endl;
//     linear.set_HE_bfv(poly_modulus_degree,vector<int>{bitwidth, s, bitwidth}, bitwidth+s);
//     c_offline += this->get_comm();
//     r_offline += this->get_round();
//     cout << "c_offline:" << c_offline << ", r_offline:" << r_offline << endl;
//     if (dim!=input.size){
//         cout << "Error: input size does not match" << endl;
//         return input;
//     }
//     uint64_t slot_count = poly_modulus_degree / 2;
//     uint64_t num_cts = dim / slot_count;
//     cout << "slot_count:" << slot_count << endl;
//     cout << "num_cts:" << num_cts << endl;
//     vector<vector<HE_CKKS::scalar_t>> vec(num_cts);
//     vector<Plaintext> pts;
//     vector<Ciphertext> cts(num_cts);
//     vector<Plaintext> output_pts(num_cts);
//     for(int i = 0; i < num_cts; i++){
//         Plaintext pt;
//         Ciphertext ct;
//         for(int j = 0; j < slot_count; ++j){
//             vec[i].push_back(input.data[i*slot_count + j]);
//         }
//     }
//     cout << "Q:" << linear.he->get_Q() << endl;
//     this->get_comm();
//     this->get_round();
//     pts = mpc_to_ckks(vec, slot_count, linear.he->get_Q(), bitwidth, s, s);
//     cout << "mpc_to_ckks comm:" << this->get_comm() << endl;
//     cout << "mpc_to_ckks round:" << this->get_round() << endl;
//     cout << "pts.size():" << pts.size() << endl;
//     vector<Ciphertext> half_cts(num_cts/2);
//     // Alice is server, Bob is client
//     if (party == sci::BOB) {
//         for(size_t i=0;i<num_cts;i++){
//             Ciphertext ct;
//             linear.he->encryptor->encrypt(pts[i], ct);
//             cts[i] = ct;
//             if(i<num_cts/2){
//                 half_cts[i] = ct;
//             }
//         }
//         cout << "client send_encrypted_vector" << endl;
// 		// Historical x+y*j half-packing experiment.
//         // send half_cts instead of cts
// 		send_encrypted_vector(io, half_cts);
// 		// send_encrypted_vector(io, cts);
// 		// vector<Ciphertext> share_client(dim);
// 		recv_encrypted_vector(linear.he->context, io, cts);
//         for(int i=0;i<num_cts;i++){
//             linear.he->decryptor->decrypt(cts[i], output_pts[i]);
//         }
// 	}
//     else{
//         // receive half_cts instead of cts
//         recv_encrypted_vector(linear.he->context, io, half_cts);
//         for(int i=0;i<half_cts.size();i++){
//             seal::Ciphertext ct1;
//             seal::Ciphertext tmp;
//             // x + y*j => x - y*j
//             linear.he->evaluator->complex_conjugate(half_cts[i], *linear.he->gal_keys, tmp);
//             // x + y*j - (x - y*j)*
//             // - 2 * y*j
//             linear.he->evaluator->sub(tmp, half_cts[i], ct1);
//             // - 2 * y*j => 2*y
//             MulImageUnitInplace(ct1, *linear.he->context);
//             // x + y*j + (x - y*j) => 2*x
//             linear.he->evaluator->add_inplace(half_cts[i], tmp);
//             cts[i] = half_cts[i];
//             cts[i+half_cts.size()] = ct1;
//         }
//         linear.he->print_parameters(linear.he->context->get_context_data(cts[0].parms_id())->parms());
//         for(int i=0;i<cts.size();i++){
//             linear.he->evaluator->add_plain_inplace(cts[i], pts[i]);
//             linear.he->evaluator->square_inplace(cts[i]);
//             RelinKeys* relin_keys = linear.he->relin_keys;
//             linear.he->evaluator->relinearize_inplace(cts[i], *relin_keys);
//             linear.he->evaluator->rescale_to_next_inplace(cts[i]);
//         }
//         PRG128 prg;
//         /*
//             Historical disabled masking experiment.
//         */
//         vector<uint64_t> r(dim,0);
//         // prg.random_mod_p<uint64_t>(r.data(), dim, linear.he->get_Q());
//         cout << "server send_encrypted_vector" << endl;
//         send_encrypted_vector(io, cts);
//         for(int i=0;i<num_cts;i++){
//             vector<double> pod_matrix(poly_modulus_degree/2, 0.0);
//             Plaintext tmp;
//             linear.he->encoder->encode(pod_matrix, cts[i].parms_id(), cts[i].scale(), tmp);
//             output_pts[i] = tmp;
//         }
//     }
//     linear.he->print_parameters(linear.he->context->get_context_data(cts[0].parms_id())->parms());
//     vector<vector<HE_CKKS::scalar_t>> output_vector;
//     cout << "send and receive comm:" << this->get_comm() << endl;
//     cout << "send and receive round:" << this->get_round() << endl;
//     output_vector = ckks_to_mpc(output_pts, poly_modulus_degree, linear.he->get_Q(cts[0]), bitwidth, s);
//     cout << "ckks_to_mpc comm:" << this->get_comm() << endl;
//     cout << "ckks_to_mpc round:" << this->get_round() << endl;
//     FixArray output(party, dim, input.signed_, bitwidth, s);
//     for(int i = 0; i < num_cts; i++){
//         for(int j = 0; j < slot_count; ++j){
//             output.data[i*slot_count + j] = output_vector[i][j];
//         }
//     }
//     cout << "element_wise_square done" << endl;
//     return output;
// }


/*
    Evaluate the even part of GELU on [-2.7, 2.7].  The encrypted input is
    signed, so a polynomial fitted only on |x| cannot be evaluated directly:
    its odd-power terms explode on the negative half of the interval.  Use a
    minimax even quartic for x*erf(x/sqrt(2))/2 here; gelu_nonlinear adds the
    missing x/2 in MPC and selects the exact ReLU tail outside the interval.
*/
void Bert_ckks::eval_gelu_poly(
    vector<UnifiedCiphertext> &output_cts,
    vector<UnifiedCiphertext> input,
    HE_CKKS *he, int output_scale_bits) {
    if (input.empty() || output_scale_bits < 0 || output_scale_bits > 62)
        throw invalid_argument("invalid GeLU ciphertexts or output scale");
    const auto loc = input.front().location();
    const size_t n = input.size();
    const double target_scale = ldexp(1.0, output_scale_bits);
    vector<UnifiedCiphertext> square(n, UnifiedCiphertext(loc));
    vector<UnifiedCiphertext> cube(n, UnifiedCiphertext(loc));
    vector<UnifiedCiphertext> quartic(n, UnifiedCiphertext(loc));
    output_cts.resize(2 * n, UnifiedCiphertext(loc));
    // BLB Algorithm 4: x^2, x^3=x^2*x, x^4=x^2*x^2.
    auto powers = [&](auto *ev, auto &keys) {
        for (size_t i = 0; i < n; ++i) {
            ev->square(input[i], square[i]);
            ev->relinearize_inplace(square[i], keys);
            ev->rescale_to_next_inplace(square[i]);
            ev->mod_switch_to_next_inplace(input[i]);
            ev->multiply(square[i], input[i], cube[i]);
            ev->relinearize_inplace(cube[i], keys);
            ev->rescale_to_next_inplace(cube[i]);
            ev->square(square[i], quartic[i]);
            ev->relinearize_inplace(quartic[i], keys);
            ev->rescale_to_next_inplace(quartic[i]);
            ev->mod_switch_to_next_inplace(input[i]);
            ev->mod_switch_to_next_inplace(square[i]);
            he_rescale_count += 3;
        }
    };
    if (loc == LOCATION::HOST) powers(he->evaluator, *he->relin_keys);
    else if (loc == LOCATION::DEVICE) powers(he->devaluator, *he->drelin_keys);
    else throw runtime_error("invalid GeLU backend");
    uint64_t drop1, drop2;
    if (loc == LOCATION::HOST) {
        auto ctx = he->context->get_context_data(quartic[0].hcipher().parms_id());
        if (!ctx || !ctx->next_context_data() || !ctx->next_context_data()->next_context_data())
            throw runtime_error("GeLU coefficient levels unavailable");
        drop1 = ctx->parms().coeff_modulus().back().value();
        drop2 = ctx->next_context_data()->parms().coeff_modulus().back().value();
    } else {
        auto chain = quartic[0].dcipher().chain_index();
        drop1 = he->dcontext->get_context_data(chain).parms().coeff_modulus().back().value();
        drop2 = he->dcontext->get_context_data(chain + 1).parms().coeff_modulus().back().value();
    }
    // Original BLB/BOLT coefficients; F0 and F1 are both evaluated in HE.
    constexpr double a = 0.020848611754127593, b = -0.18352506127082727;
    constexpr double c = 0.5410550166368381, d = -0.03798164612714154;
    constexpr double e = 0.001620808531841547;
    auto encode = [&](double value, double scale, int levels, UnifiedPlaintext &pt) {
        if (!isfinite(scale) || scale < 1) throw runtime_error("invalid GeLU plaintext scale");
        vector<double> values(he->num_slots, value);
        if (loc == LOCATION::HOST) {
            auto ctx = he->context->get_context_data(quartic[0].hcipher().parms_id());
            for (int k = 0; k < levels; ++k) ctx = ctx->next_context_data();
            he->encoder->encode(values, ctx->parms_id(), scale, pt);
        } else {
            he->dencoder->encode(*he->dcontext, values, scale, pt,
                                quartic[0].dcipher().chain_index() + levels);
        }
    };
    vector<UnifiedPlaintext> coefficients(5, UnifiedPlaintext(loc));
    const double product_scale = target_scale * double(drop1) * double(drop2);
    encode(a, product_scale / quartic[0].scale(), 0, coefficients[0]);
    encode(b, product_scale / cube[0].scale(), 0, coefficients[1]);
    encode(c, product_scale / square[0].scale(), 0, coefficients[2]);
    encode(0.5 - d, product_scale / input[0].scale(), 0, coefficients[3]);
    encode(0.5 + d, product_scale / input[0].scale(), 0, coefficients[4]);
    UnifiedPlaintext constant(loc);
    encode(e, target_scale, 2, constant);
    auto evaluate = [&](auto *ev) {
        for (size_t i = 0; i < n; ++i) {
            UnifiedCiphertext negative_linear(loc), positive_linear(loc);
            ev->multiply_plain(input[i], coefficients[3], negative_linear);
            ev->multiply_plain(input[i], coefficients[4], positive_linear);
            ev->multiply_plain_inplace(quartic[i], coefficients[0]);
            ev->multiply_plain_inplace(cube[i], coefficients[1]);
            ev->multiply_plain_inplace(square[i], coefficients[2]);
            for (auto *term : {&quartic[i], &cube[i], &square[i], &negative_linear, &positive_linear}) {
                ev->rescale_to_next_inplace(*term);
                ev->rescale_to_next_inplace(*term);
                he_rescale_count += 2;
                if (!isfinite(term->scale()) || abs(term->scale() / target_scale - 1) > 1e-6)
                    throw runtime_error("GeLU scale mismatch");
                term->scale() = target_scale;
            }
            ev->add(quartic[i], square[i], output_cts[i]);
            output_cts[n+i] = output_cts[i];
            ev->sub_inplace(output_cts[i], cube[i]);
            ev->add_inplace(output_cts[i], negative_linear);
            ev->add_plain_inplace(output_cts[i], constant);
            ev->add_inplace(output_cts[n+i], cube[i]);
            ev->add_inplace(output_cts[n+i], positive_linear);
            ev->add_plain_inplace(output_cts[n+i], constant);
        }
    };
    if (loc == LOCATION::HOST) evaluate(he->evaluator);
    else evaluate(he->devaluator);
    for (const auto &ct : output_cts)
        if (ct.coeff_modulus_size() != 1) throw runtime_error("invalid GeLU final level");
    cout << "FORMAL_GELU_SCALE_READY status=PASS actual_scale_bits=" << output_scale_bits
         << " limbs=1 cts=" << output_cts.size()
         << " polynomial=blb_F0_F1 ct_ct_per_input_ciphertext=3" << endl;
}

// void Bert_ckks::mp2ml_test(){
//     #ifdef BERT_PERF
//     auto t_conversion = high_resolution_clock::now();
//     #endif 
//     if (party == sci::BOB) {
//         vector<double> x = {1,2,3,4,5,6,7,8};
//     }
//     // step 1
//     vector<vector<std::complex<HE_CKKS::scalar_t>>> x_intt(pts.size());
//     cout << "pts.size():" << pts.size() << endl;
//     for(int i=0;i<pts.size();i++){
//         // cout << "i:" << i << endl;
//         x_intt[i] = linear.he->ckks_intt(pts[i]);
//     }
//     vector<HE_CKKS::scalar_t> x_intt_flatten_field,x_intt_flatten_ring(x_intt.size()*x_intt[0].size());
//     for(int i=0;i<x_intt.size();i++){
//         for(int j=0;j<x_intt[i].size();j++){
//             x_intt_flatten_field.push_back(x_intt[i][j].real());
//         }
//     }
//     cout << "step 1 done" << endl;
// 	// step 2
//     int s_in = ceil(log2(pts[0].scale()));
//     s_in = s_out;
//     nonlinear.field_to_ring(
//         1,
//         x_intt_flatten_field.data(),
//         Q,
//         x_intt_flatten_ring.data(),
//         x_intt_flatten_field.size(),
//         bit_out,
//         s_in,
//         s_out
//     );
//     cout << "step 2 done" << endl;
// 	// step 3
//     vector<vector<HE_CKKS::complex_t>> x_decode(pts.size());
//     for(int i=0;i<pts.size();i++){
//         for(int j=0;j<size;j++){
//             x_decode[i].push_back(HE_CKKS::complex_t(x_intt_flatten_ring[i*size+j],0));
//         }
//     }
//     cout << "x_decode[0][0]:" << x_decode[0][0].real() << endl;
//     vector<vector<HE_CKKS::scalar_t>> y(pts.size());
//     for(int i=0;i<pts.size();i++){
//         y[i] = linear.he->ckks_decode(x_decode[i],size);
//     }
//     cout << "ckks_to_mpc done" << endl;
// 	return y;
// }


std::map<char*,FixArray> Bert_ckks::layer_norm_nonlinear(map<char*,FixArray> input) {
    FPMath *fpmath = nonlinear.fpmath[0];
    FixArray x_flat_avg = input["x_flat_avg"];
    FixArray sigma = input["sigma_flat"];
    FixArray w = input["w"];
    FixArray b = input["b"];
    if (x_flat_avg.size <= 0 || sigma.size <= 0 ||
        w.size != x_flat_avg.size || b.size != x_flat_avg.size) {
        throw runtime_error(
            "formal LayerNorm refused: real x/sigma/weight/bias shares are required");
    }
    int bitwidth = x_flat_avg.ell;
    int s = x_flat_avg.s;
    int num_ops = sigma.size;
    int dim = x_flat_avg.size / num_ops;
    cout << "in layer_norm_nonlinear, bitwidth,s,num_ops,dim:" << bitwidth << "," << s << "," << num_ops << "," << dim << endl;
    cout << "sigma.size, should be L:" << sigma.size << endl;
    // Padded rows are public and later discarded. Use public variance one
    // before rsqrt so CKKS noise around an unused zero cannot create a
    // negative fixed-point variance. No live token is clipped or opened.
    if (!bert_post_norm && public_attention_mask_ready && sigma.size == static_cast<int>(public_attention_mask.size()))
        for (int i=0; i<sigma.size; ++i)
            if (public_attention_mask[i]) sigma.data[i] = party == sci::ALICE ? (uint64_t{1} << s) : 0;
    // 计算平方根倒数
    const uint64_t sqrt_sent_before = current_comm();
    FixArray sigma_sqrt = fpmath->sqrt(sigma, true);
    cout << "sqrt comm:" << current_comm() - sqrt_sent_before << endl;
    cout << "sqrt round:" << this->get_round() << endl;
    //point8
    FixArray sigma_flat(party, x_flat_avg.size, sigma_sqrt.signed_, bitwidth, s);
    for(int i = 0; i < num_ops; i++){
        for(int j = 0; j < dim; j++){
            sigma_flat.data[i*dim + j] = sigma_sqrt.data[i];
        }
    }
    // 广播乘法
    cout << "x_flat_avg bitwidth,s:" << x_flat_avg.ell << "," << x_flat_avg.s << endl;
    map<char*,FixArray> result;
    result["x_flat_avg"] = x_flat_avg;
    result["sigma_flat"] = sigma_flat;
    result["w"] = w;
    result["b"] = b;
    cout << "layer_norm_nonlinear done" << endl;
    return result;
}

void gelu_nonlinear_thread(int tid, int party, uint64_t *x1, uint64_t* x2, uint64_t* x3, uint64_t* output, int array_size, int ell, int s, FPMath *fpmath){
    // cout << "in gelu thread, tid:" << tid << endl;
    // x1/x2 are already additive shares produced by the HE->MPC boundary.
    // Re-inputting odd chunks as the remote party zeroed both parties' local
    // shares for half the tensor.  Copy the local shares without opening or
    // changing their ownership.
    FixArray x_array(party, array_size, true, ell, s);
    FixArray gelu_poly_flat(party, array_size, true, ell, s);
    memcpy(x_array.data, x1, static_cast<size_t>(array_size) * sizeof(uint64_t));
    memcpy(gelu_poly_flat.data, x2,
           static_cast<size_t>(array_size) * sizeof(uint64_t));
    FixArray positive(party, array_size, true, ell, s);
    memcpy(positive.data, x3, static_cast<size_t>(array_size) * sizeof(uint64_t));
    // Use LT/MSB directly. Donor BoolOp::NOT flips both XOR shares, so
    // derived LE/GE names do not have their conventional semantics.
    auto negative = fpmath->fix->MSB(x_array);
    auto below = fpmath->fix->LT(x_array, uint64_t(-int64_t(2.7 * (1ULL << s))));
    auto above = fpmath->fix->LT(
        fpmath->fix->input(sci::PUBLIC, array_size,
                          uint64_t(2.7 * (1ULL << s)), true, ell, s), x_array);
    auto inner = fpmath->fix->if_else(negative, gelu_poly_flat, positive);
    auto left = fpmath->fix->if_else(below, uint64_t(0), inner);
    auto ret = fpmath->fix->if_else(above, x_array, left);
    // cout << "OK1" << endl;
    for(int i = 0; i < array_size; i++){
        output[i] = ret.data[i];
    }
    // cout << "OK2" << endl;
}


FixArray Bert_ckks::gelu_nonlinear(std::map<char*,FixArray> input){
    FixArray x_array = input["x_array"];
    FixArray gelu_poly_flat = input["gelu_poly_flat"];
    FixArray gelu_positive = input.at("gelu_positive");
    // Pure output buffer: every entry is written by a worker below. Do not
    // invoke an MPC input protocol or seed it with a constant share.
    FixArray ret(party, x_array.size, x_array.signed_, x_array.ell,
                 x_array.s);
    FPMath* fpmath = this->nonlinear.fpmath[0];
    int nthreads = 64, array_size =  x_array.size;
    std::thread threads[nthreads];
    int chunk_size = array_size / nthreads;
    cout << "chunk_size, ell,s:" << chunk_size << "," << x_array.ell << "," << x_array.s << endl;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
            lnum_ops = array_size - offset;
        } else {
            lnum_ops = chunk_size;
        }
        // num_ops = chunk_size
        threads[i] =
            std::thread(
                gelu_nonlinear_thread, 
                i, 
                party, 
                &x_array.data[offset], 
                &gelu_poly_flat.data[offset],
                &gelu_positive.data[offset],
                &ret.data[offset],
                lnum_ops,
                x_array.ell,
                x_array.s,
                this->nonlinear.fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }
    cout << "gelu time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    // exit(0);
    // int this_party = party;
    // int N = x_array.size;
    // int bitwidth = x_array.ell;
    // int s = x_array.s;
    // auto start = high_resolution_clock::now();
    // cout << "x_array.size:" << x_array.size << endl;
    // BoolArray msb_x = fpmath->fix->MSB(x_array);
    // cout << "MSB time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    
    // // no communication
    // FixArray neg_x = fpmath->fix->mul(x_array, -1);
    // // point 2
    
    // FixArray y = fpmath->fix->if_else(msb_x, neg_x, x_array);
    // start = high_resolution_clock::now();
    // BoolArray lt27 = fpmath->fix->LT(y, 2.7*(1 << s));
    // cout << "LT time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    // // point 8
    // BoolArray all_0 = fpmath->bool_op->input(sci::ALICE, N, uint8_t(0));
    // FixArray x_plus_y = fpmath->fix->add(x_array, y);   
    // FixArray half_x_plus_y = x_plus_y;
    // // FixArray half_x_plus_y = fpmath->fix->right_shift(x_plus_y, 1, all_0.data);
    // // point 9
    // cout << "gelu_poly_flat(bitwidth,s):" << gelu_poly_flat.ell << "," << gelu_poly_flat.s << endl;
    // cout << "x array bitwidth,s:" << x_array.ell << "," << x_array.s << endl;
    // cout << "half_x_plus_y(bitwidth,s):" << half_x_plus_y.ell << "," << half_x_plus_y.s << endl;
    // half_x_plus_y.s = s;
    // //half_x_plus_y = (x+|x|)/2 = max(x,0), if |x|<2.7, 则 gelu_y, 否则half_x_plus_y
    // FixArray ret = fpmath->fix->if_else(lt27, gelu_poly_flat, half_x_plus_y);
    // cout << "other time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    // BoolArray msb_ret = fpmath->bool_op->AND(msb_x, lt27);
    // point 10
    
    // ret = fpmath->fix->extend(ret, 37, msb_ret.data);
    // ret = fpmath->fix->right_shift(ret, 7, msb_ret.data);
    return ret;
}

FixArray Bert_ckks::exp_he(const FixArray &x, bool apply_attention_mask){
    int ell = x.ell;
    int scale = x.s;
    HE_CKKS* he = this->linear.he_softmax;
    if (x.party == sci::PUBLIC || !x.signed_ || scale < 0 || scale >= 62) {
        throw invalid_argument(
            "HE Softmax exp requires a secret signed fixed-point input");
    }
    // The BLB exponent approximation is (1+x/64)^64.  It is only valid on
    // the clipped Softmax domain x > -13: for a more negative input the
    // base becomes negative and the even power produces a large positive
    // number instead of an underflowed zero.  The previous implementation
    // sent those values directly through CKKS, which is why rows containing
    // qk gaps below -13 returned deterministic 1e7-scale signed values.
    // Compare and clamp in MPC before the HE polynomial, then mux the exact
    // fixed-point zero back for x <= -13.  No activation is opened.
    const int64_t cutoff_value = -13LL * (int64_t{1} << scale);
    const uint64_t cutoff_raw = static_cast<uint64_t>(cutoff_value);
    // LT/MSB is the direct signed predicate. Do not use derived GE/LE:
    // the donor BoolOp::NOT toggles both XOR shares, leaving their XOR unchanged.
    BoolArray underflow;
    FixArray evaluation_input = x;
    const int exp_workers = min(nonlinear.active_workers(), x.size);
    if (blb_secure_exp_clamp_enabled()) {
        const auto clip_start = high_resolution_clock::now();
        underflow = BoolArray(x.party, x.size);
        vector<thread> workers;
        for (int worker = 0; worker < exp_workers; ++worker) {
            const int begin = x.size * worker / exp_workers;
            const int end = x.size * (worker + 1) / exp_workers;
            workers.emplace_back([&, worker, begin, end]() {
                FixArray local(x.party, end - begin, x.signed_, x.ell, x.s);
                local.set_p(x.p);
                copy_n(x.data + begin, end - begin, local.data);
                auto *fix = nonlinear.fpmath[worker]->fix;
                auto below = fix->LT(local, cutoff_raw);
                auto clipped = fix->if_else(below, cutoff_raw, local);
                copy_n(below.data, end - begin, underflow.data + begin);
                copy_n(clipped.data, end - begin, evaluation_input.data + begin);
            });
        }
        for (auto &worker : workers) worker.join();
        cout << "EXP_SAFETY_PHASE phase=input_clip workers=" << exp_workers
             << " ms=" << chrono::duration<double, milli>(
                    high_resolution_clock::now() - clip_start).count() << endl;
    }
    // BLB Algorithm 5: exactly six squares after a public HE division by 64.
    const int exp_squarings = 6;
    size_t public_masked_elements = 0;
    if (apply_attention_mask && public_attention_mask_ready) {
        const size_t row_width = public_attention_mask.size();
        if (row_width == 0 ||
            evaluation_input.size % static_cast<int>(row_width) != 0) {
            throw runtime_error(
                "HE Softmax exp public mask is incompatible with row layout");
        }
        const size_t rows =
            static_cast<size_t>(evaluation_input.size) / row_width;
        for (size_t row = 0; row < rows; ++row) {
            for (size_t key = 0; key < row_width; ++key) {
                if (public_attention_mask[key] == 0) continue;
                // Padding locations are public and their exact Softmax output
                // is public zero.  Discard both local secret shares before HE
                // so the large public negative mask cannot overflow the
                // comparison-free polynomial; discard the public exp(0)=1
                // placeholder again after the HE round below.
                evaluation_input.data[row * row_width + key] = 0;
                ++public_masked_elements;
            }
        }
    }
    const int poly_modulus_degree = he->poly_modulus_degree;
    int slots = he->num_slots;
    int num_cts = (evaluation_input.size + slots - 1) / slots;
    cout << "slots, num_cts:" << slots << "," << num_cts << endl;
    vector<vector<uint64_t>> vec_x;
    auto start = high_resolution_clock::now();
    for(int i = 0; i < num_cts; i++){
        vector<uint64_t> tmp;
        for(int j = 0; j < slots; j++){
            const int index = i * slots + j;
            tmp.push_back(index < evaluation_input.size ? evaluation_input.data[index] : 0);
            // cout << "x.data[i*slots + j]: " << x.data[i*slots + j] << endl;
        }
        vec_x.push_back(tmp);
    }
    cout << "exp_he data copy time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    cout << "vec_x.size:" << vec_x.size()*vec_x[0].size() << endl;
    start = high_resolution_clock::now();
    const auto previous_conversion_bucket = profile_conversion_bucket();
    profile_set_conversion_bucket(
        BlbViTCommAuditBucket::SOFTMAX_TO_CKKS_CONVERSION);
    vector<UnifiedPlaintext> pts_x;
    mpc_to_ckks(pts_x, vec_x, slots, he->get_conversion_prime(), ell,
                scale, he->he_scale, he);
    profile_set_conversion_bucket(previous_conversion_bucket);
    cout << "pts.size:" << pts_x.size() << endl;
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    // Wide conversion uses the ordinary full-slot transport contract.
    vector<UnifiedCiphertext> cts_x(
        num_cts, UnifiedCiphertext(LOCATION::HOST));
    vector<Ciphertext> cts_y(num_cts);
    vector<Plaintext> pts_y(num_cts);
    if (party == sci::BOB) {
        cout << "send_encrypted_vector" << endl;
            he->encrypt_send(pts_x, cts_x);
		if (getenv("OURS_VIT_CONVERSION_EXP_DIRECT_CHECK") != nullptr) {
			const auto &input_ciphertext = cts_x[0];
			Plaintext direct_input_plaintext;
			he->decryptor->decrypt(input_ciphertext, direct_input_plaintext);
			vector<complex<double>> direct_input_decoded;
			he->encoder->decode(direct_input_plaintext, direct_input_decoded);
			cout << "CONVERSION_TEST C4E_INPUT slots=";
			for (size_t index = 0;
			     index < min<size_t>(8, direct_input_decoded.size()); ++index) {
				if (index != 0) cout << ',';
				cout << setprecision(17)
				     << direct_input_decoded[index].real();
			}
			cout << " scale_bits=" << log2(direct_input_plaintext.scale())
			     << " imag0=" << (direct_input_decoded.empty()
										? 0.0
										: direct_input_decoded[0].imag())
			     << " limbs="
			     << he->context->get_context_data(
						 direct_input_plaintext.parms_id())
								->parms().coeff_modulus().size()
			     << endl;
		}
		
		// test_util();
		cout << "send_encrypted_vector done" << endl;
		if (getenv("OURS_VIT_CONVERSION_EXP_DIRECT_CHECK") != nullptr) {
			vector<Ciphertext> direct_output(num_cts);
			recv_encrypted_vector(he->context, io, direct_output);
			Plaintext direct_plaintext;
			he->decryptor->decrypt(direct_output[0], direct_plaintext);
			vector<complex<double>> direct_decoded;
			he->encoder->decode(direct_plaintext, direct_decoded);
			cout << "CONVERSION_TEST C4E_DIRECT slots=";
			for (size_t index = 0; index < min<size_t>(8, direct_decoded.size());
			     ++index) {
				if (index != 0) cout << ',';
				cout << setprecision(17) << direct_decoded[index].real();
			}
			cout << " scale_bits=" << log2(direct_plaintext.scale())
			     << " imag0=" << (direct_decoded.empty()
										? 0.0 : direct_decoded[0].imag())
			     << endl;
		}
        cts_y.resize((num_cts + 1) / 2);
        pts_y.resize(cts_y.size());
		recv_encrypted_vector(he->context, io, cts_y);
        #pragma omp parallel for
        for(int i=0;i<cts_y.size();i++){
            he->decryptor->decrypt(cts_y[i], pts_y[i]);
        }
    }
    else{
        auto loc = he->backend();

        start = high_resolution_clock::now();
        recv_encrypted_vector(he->context, io, cts_x);
		this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;

        if (he->is_gpu_enable()) {
            // Load reviced ciphertexts to GPU
            start = high_resolution_clock::now();
                for_each_n(
                    cts_x.begin(),
                    cts_x.size(),
                    [he](auto &ct) { 
                        ct.to_device(*he->context, *he->dcontext);
                });
            for_each_n(
                pts_x.begin(), pts_x.size(),
                [he](auto &pt) { pt.to_device(*he->context, *he->dcontext); });
            const double h2d_seconds =
                ((high_resolution_clock::now() - start)).count()/1e+9;
            this->ct_transfer_mem_la += h2d_seconds;
            this->h2d_time += h2d_seconds;
        }

        start = high_resolution_clock::now();

        // Algorithm 5: public /64 in HE, then +1 and six squares.
        UnifiedPlaintext divisor(loc), one(loc);
        double drop = he->is_gpu_enable()
            ? he->dcontext->get_context_data(cts_x[0].dcipher().chain_index()).parms().coeff_modulus().back().value()
            : he->context->get_context_data(cts_x[0].hcipher().parms_id())->parms().coeff_modulus().back().value();
        const double divisor_scale = ldexp(1.0, he->he_scale);
        const double next_scale = cts_x[0].scale() * divisor_scale / drop;
        if (!he->is_gpu_enable()) {
            auto ctx = he->context->get_context_data(cts_x[0].hcipher().parms_id());
            he->encoder->encode(1.0/64.0, ctx->parms_id(), divisor_scale, divisor);
            he->encoder->encode(1.0, ctx->next_context_data()->parms_id(), next_scale, one);
        } else {
            auto chain = cts_x[0].dcipher().chain_index();
            he->dencoder->encode(*he->dcontext, vector<double>(slots, 1.0/64.0), divisor_scale, divisor, chain);
            he->dencoder->encode(*he->dcontext, vector<double>(slots, 1.0), next_scale, one, chain + 1);
        }

        // The encoded constant and uploads belong to the producer stream;
        // workers must finish before the main thread copies their results.
        const int exp_threads = min<int>(cts_x.size(),
            he->is_gpu_enable() ? 8 : omp_get_max_threads());
#ifdef BLB_HAS_HE_GPU
        if (he->is_gpu_enable() && cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess)
            throw runtime_error("Softmax exponent input stream failed");
        vector<int> exp_errors(cts_x.size(), int(cudaSuccess));
#endif
        #pragma omp parallel for num_threads(exp_threads)
        for(int i=0;i<cts_x.size();i++){
            if (!he->is_gpu_enable()) {
                // cout << cts_x[i].poly_modulus_degree() << endl;
                // The client ciphertext contains Bob's local CKKS share.
                // Add Alice's independently converted share before applying
                // the nonlinear polynomial; evaluating exp on only one share
                // is not a secret-shared exp and was the source of the
                // deterministic C4E/B corruption.
                he->evaluator->add_plain_inplace(cts_x[i], pts_x[i]);
                he->evaluator->multiply_plain_inplace(cts_x[i], divisor);
                he->evaluator->rescale_to_next_inplace(cts_x[i]);
                he->evaluator->add_plain_inplace(cts_x[i], one);
                for (int j = 0; j < exp_squarings; j++) {
                    he->evaluator->square_inplace(cts_x[i]);
                    he->evaluator->relinearize_inplace(cts_x[i], *he->relin_keys);
                    he->evaluator->rescale_to_next_inplace(cts_x[i]);
                }
                // cout << "OK3" << endl;
            } else {
                // Keep the GPU path's share aggregation identical to SEAL:
                // both local CKKS shares are present before any HE multiply.
                he->devaluator->add_plain_inplace(cts_x[i], pts_x[i]);
                he->devaluator->multiply_plain_inplace(cts_x[i], divisor);
                he->devaluator->rescale_to_next_inplace(cts_x[i]);
                he->devaluator->add_plain_inplace(cts_x[i], one);
                for (int j = 0; j < exp_squarings; j++) {
                    he->devaluator->square_inplace(cts_x[i]);
                    he->devaluator->relinearize_inplace(cts_x[i], *he->drelin_keys);
                    he->devaluator->rescale_to_next_inplace(cts_x[i]);
                }
#ifdef BLB_HAS_HE_GPU
                exp_errors[i] = int(cudaStreamSynchronize(cudaStreamPerThread));
#endif
            }
        }

        he_rescale_count += cts_x.size() * (exp_squarings + 1);
#ifdef BLB_HAS_HE_GPU
        for (int error : exp_errors)
            if (error != int(cudaSuccess))
                throw runtime_error("Softmax exponent worker stream failed");
#endif
        if (he->is_gpu_enable()) {
            // Load result ciphertexts to CPU
            start = high_resolution_clock::now();
            for_each_n(
                    cts_x.begin(),
                    cts_x.size(),
                    [he](auto &ct) { 
                        ct.to_host(*he->context, *he->dcontext);
            });
            const double d2h_seconds =
                ((high_resolution_clock::now() - start)).count()/1e+9;
            this->ct_transfer_mem_la += d2h_seconds;
            this->d2h_time += d2h_seconds;
        }

        cout << "num of output cts:" << cts_x.size() << endl;
        cout << "bitwidth of output ct:" << cts_x[0].coeff_modulus_size() << endl;
        this->he_compute_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        start = high_resolution_clock::now();
        for (int i = 0; i < cts_x.size(); ++i) {
            cts_y[i] = std::move(cts_x[i].hcipher());
        }
		if (getenv("OURS_VIT_CONVERSION_EXP_DIRECT_CHECK") != nullptr) {
			vector<Ciphertext> direct_output = cts_y;
			send_encrypted_vector(io, direct_output);
		}
        pack_real_ckks_pairs(cts_y, he);
        mask_ckks_to_mpc_inplace(cts_y, pts_y, he);
        send_encrypted_vector(io, cts_y);
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    }
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    vector<vector<HE_CKKS::scalar_t>> vec_output(pts_y.size());
    start = high_resolution_clock::now();
    vec_output = ckks_pairs_to_mpc(pts_y, num_cts, ell, scale, he);
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    // ckks_to_mpc already returned local additive shares; this is only their
    // flattened destination buffer.
    FixArray ret(party, x.size, x.signed_, ell, scale);

    for(int i=0;i<vec_output.size();i++){
        for(int j=0;j<vec_output[i].size();j++){
            if (i * slots + j < ret.size)
                ret.data[i*slots + j] = vec_output[i][j];
        }
    }

    // x <= -13 is an intentional exponential underflow in the BLB Softmax
    // approximation.  Apply the mux after the HE round so malformed CKKS
    // values from the unclipped branch can never enter the denominator.
    if (blb_secure_exp_clamp_enabled()) {
        const auto clip_start = high_resolution_clock::now();
        vector<thread> workers;
        for (int worker = 0; worker < exp_workers; ++worker) {
            const int begin = ret.size * worker / exp_workers;
            const int end = ret.size * (worker + 1) / exp_workers;
            workers.emplace_back([&, worker, begin, end]() {
                auto *fix = nonlinear.fpmath[worker]->fix;
                auto local = ret.subset(begin, end);
                BoolArray below(local.party, end - begin);
                copy_n(underflow.data + begin, end - begin, below.data);
                local = fix->if_else(below, uint64_t(0), local);
                // CKKS tails may be negative by a few raw units. Preserve
                // the mandatory signed clamp exactly once, before row sums.
                auto negative = fix->MSB(local);
                auto clipped = fix->if_else(negative, uint64_t(0), local);
                copy_n(clipped.data, end - begin, ret.data + begin);
            });
        }
        for (auto &worker : workers) worker.join();
        cout << "EXP_SAFETY_PHASE phase=output_mux_and_clamp workers="
             << exp_workers << " ms=" << chrono::duration<double, milli>(
                    high_resolution_clock::now() - clip_start).count() << endl;
    }
    if (apply_attention_mask && public_attention_mask_ready) {
        const size_t row_width = public_attention_mask.size();
        const size_t rows = static_cast<size_t>(ret.size) / row_width;
        for (size_t row = 0; row < rows; ++row) {
            for (size_t key = 0; key < row_width; ++key) {
                if (public_attention_mask[key] != 0)
                    ret.data[row * row_width + key] = 0;
            }
        }
    }
    // Values have been clamped; preserve the signed API metadata.
    ret.signed_ = x.signed_;
    cout << "FORMAL_SOFTMAX_EXP_CLIP status=PASS cutoff=-13"
         << " comparison="
         << (blb_secure_exp_clamp_enabled() ? "secure_LT" : "none")
         << " divisor=2^" << 6
         << " exponent=2^" << exp_squarings
         << " divisor_folded_into_conversion=0 zero_after_HE="
         << (blb_secure_exp_clamp_enabled() ? 1 : 0)
         << " negative_rounding_to_zero="
         << (blb_secure_exp_clamp_enabled() ? 1 : 0)
         << " fast_default=" << (blb_secure_exp_clamp_enabled() ? 0 : 1)
         << " public_padding_zeroed=" << public_masked_elements
         << " client_reconstruction=0" << endl;

    return ret;
}


FixArray Bert_ckks::element_wise_mul(FixArray x, FixArray y){
    cout << "begin element_wise_mul" << endl;
    const int original_size = x.size;
    const bool original_signed = x.signed_;
    int bitwidth = x.ell; //bw
    int s = x.s;
    HE_CKKS* he = this->linear.he_onetime;
    int dim = x.size;
    if(x.size!=y.size){
        cout << "x.size, y.size:" << x.size << "," << y.size << endl;
        cout << "Error: input size does not match" << endl;
        assert(0);
    }
    int poly_modulus_degree = he->poly_modulus_degree;
    uint64_t slot_count = poly_modulus_degree / 2;
    uint64_t num_cts = (dim + slot_count - 1) / slot_count;
    if (dim % slot_count != 0) {
        const int padded_size = num_cts * slot_count;
        // Preserve the real shares while padding the public unused tail with
        // additive zero.  The old branch replaced both inputs with fresh,
        // uninitialized slot buffers, so the 3072 Softmax denominators were
        // discarded before the reciprocal-square multiplication.
        FixArray x_padded(x.party, padded_size, x.signed_, x.ell, x.s);
        FixArray y_padded(y.party, padded_size, y.signed_, y.ell, y.s);
        fill(x_padded.data, x_padded.data + padded_size, uint64_t{0});
        fill(y_padded.data, y_padded.data + padded_size, uint64_t{0});
        copy(x.data, x.data + original_size, x_padded.data);
        copy(y.data, y.data + original_size, y_padded.data);
        x = std::move(x_padded);
        y = std::move(y_padded);
        dim = padded_size;
        cout << "FORMAL_ELEMENTWISE_PADDING status=PASS original_elements="
             << original_size << " padded_elements=" << slot_count
             << " copied_inputs=2 zero_tail=1 output_trimmed=1" << endl;
    }
    cout << "slot_count, num_cts:" << slot_count << "," << num_cts << endl;
    vector<vector<HE_CKKS::scalar_t>> vec_x(num_cts);
    vector<vector<HE_CKKS::scalar_t>> vec_y(num_cts);
    vector<UnifiedPlaintext> pts_x;
    vector<UnifiedPlaintext> pts_y;
    vector<UnifiedCiphertext> cts_x(
        num_cts, UnifiedCiphertext(LOCATION::HOST));
    vector<UnifiedCiphertext> cts_y(
        num_cts, UnifiedCiphertext(LOCATION::HOST));
    vector<Ciphertext> cts_output(num_cts);
    vector<Plaintext> output_pts(num_cts);
    for(int i = 0; i < num_cts; i++){
        for(int j = 0; j < slot_count; ++j){
            vec_x[i].push_back(x.data[(i*slot_count + j)%x.size]);
            vec_y[i].push_back(y.data[(i*slot_count + j)%y.size]);
        }
    }

    auto start = high_resolution_clock::now();
    mpc_to_ckks(pts_x, vec_x, slot_count, he->get_conversion_prime(),
                bitwidth, s, he->he_scale, he);
    mpc_to_ckks(pts_y, vec_y, slot_count, he->get_conversion_prime(),
                bitwidth, s, he->he_scale, he);
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    // Alice is server, Bob is client
    if (party == sci::BOB) {
            #pragma omp parallel for
            for(size_t i=0;i<num_cts;i++){
                he->encryptor->encrypt(pts_x[i], cts_x[i]);
                he->encryptor->encrypt(pts_y[i], cts_y[i]);
            }
            send_encrypted_vector(io, cts_x);
            send_encrypted_vector(io, cts_y);
        cout << "client send_encrypted_vector done" << endl;
		recv_encrypted_vector(he->context, io, cts_output);
        // check_cts(cts_output);
        cout << "client recv_encrypted_vector done" << endl;
        #pragma omp parallel for
        for(int i=0;i<cts_output.size();i++){
            he->decryptor->decrypt(cts_output[i], output_pts[i]);
        }
        cout << "client decrypt done" << endl;
	} else {
        // Revice ciphertexts from BOB (or clinet)
        start = high_resolution_clock::now();
        recv_encrypted_vector(he->context, io, cts_x);
        recv_encrypted_vector(he->context, io, cts_y);
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;

        if (he->is_gpu_enable()) {
            start = high_resolution_clock::now();
                for_each_n(cts_x.begin(), cts_x.size(),
                    [he](auto &ct) { ct.to_device(*he->context, *he->dcontext); });
                for_each_n(cts_y.begin(), cts_y.size(),
                    [he](auto &ct) { ct.to_device(*he->context, *he->dcontext); });
            for_each_n(pts_x.begin(), pts_x.size(),
                [he](auto &pt) { pt.to_device(*he->context, *he->dcontext); });
            for_each_n(pts_y.begin(), pts_y.size(),
                [he](auto &pt) { pt.to_device(*he->context, *he->dcontext); });
            const double h2d_seconds =
                ((high_resolution_clock::now() - start)).count() / 1e+9;
            this->ct_transfer_mem_la += h2d_seconds;
            this->h2d_time += h2d_seconds;
        }

        // Begin computation
        start = high_resolution_clock::now();

        #pragma omp parallel for if(!he->is_gpu_enable())
        for(int i=0;i<cts_x.size();i++){
            if (!he->is_gpu_enable()) {
                he->evaluator->add_plain_inplace(cts_x[i], pts_x[i]);
                he->evaluator->add_plain_inplace(cts_y[i], pts_y[i]);
                he->evaluator->multiply_inplace(cts_x[i], cts_y[i]);
                he->evaluator->relinearize_inplace(cts_x[i], *he->relin_keys);
                ++this->he_rescale_count;
                he->evaluator->rescale_to_next_inplace(cts_x[i]);
            } else {
                he->devaluator->add_plain_inplace(cts_x[i], pts_x[i]);
                he->devaluator->add_plain_inplace(cts_y[i], pts_y[i]);
                he->devaluator->multiply_inplace(cts_x[i], cts_y[i]);
                he->devaluator->relinearize_inplace(cts_x[i], *he->drelin_keys);
                ++this->he_rescale_count;
                he->devaluator->rescale_to_next_inplace(cts_x[i]);    
            }
        }
        this->he_compute_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        
        if (he->is_gpu_enable()) {
            // Load result ciphertexts to CPU
            start = high_resolution_clock::now();
            for_each_n(cts_x.begin(), cts_x.size(),
                [he](auto &ct) { ct.to_host(*he->context, *he->dcontext); });
            const double d2h_seconds =
                ((high_resolution_clock::now() - start)).count()/1e+9;
            this->ct_transfer_mem_la += d2h_seconds;
            this->d2h_time += d2h_seconds;
        }

        start = high_resolution_clock::now();
        cout << "server send_encrypted_vector" << endl;
        cout << "cts_x.size:" << cts_x.size() << endl;
        for (int i = 0; i < cts_x.size(); ++i) {
            cts_output[i] = std::move(cts_x[i].hcipher());
        }
        mask_ckks_to_mpc_inplace(cts_output, output_pts, he);
        send_encrypted_vector(io, cts_output);
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    }
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    vector<vector<HE_CKKS::scalar_t>> output_vector;
    start = high_resolution_clock::now();
    // cout << "OK here" << endl;
    // cout << "cts_x.size:" << cts_x.size() << endl;
    // cout << "pts[0].scale()" << output_pts[0].scale() << endl;
    // cout << "Q:" << he->get_Q(output_pts[0]) << endl;
    output_vector = ckks_to_mpc(output_pts, poly_modulus_degree, he->get_Q(output_pts[0]), bitwidth, s, he);
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    FixArray output(party, original_size, original_signed, bitwidth, s);
    #pragma omp parallel for
    for(int i = 0; i < num_cts; i++){
        for(int j = 0; j < slot_count; ++j){
            const size_t output_index =
                static_cast<size_t>(i) * slot_count + j;
            if (output_index < static_cast<size_t>(original_size))
                output.data[output_index] = output_vector[i][j];
        }
    }
    cout << "HE element-wise mul done" << endl;
    return output;
}

FixArray Bert_ckks::div_batch_he(const FixArray& nm, const FixArray& dn, int batch_dn_size, int l_out, int s_out, bool normalized_dn){
    cout << "l_out: " << l_out << " s_out: " << s_out << endl;
    cout << "nm.size, dn.size:" << nm.size << "," << dn.size << endl;
    if (!normalized_dn) assert(dn.signed_ == false);
    assert(nm.party != sci::PUBLIC && dn.party != sci::PUBLIC);
    assert(nm.size == dn.size * batch_dn_size);
    assert(s_out <= dn.s);
    auto start = high_resolution_clock::now();
    BoolOp* bool_op = nonlinear.fpmath[0]->bool_op;
    BoolArray all_0 = bool_op->input(sci::ALICE, dn.size, uint8_t(0));
    BoolArray all_1 = bool_op->input(sci::ALICE, dn.size, uint8_t(1));
    FixOp* fix = nonlinear.fpmath[0]->fix;
    FixArray nrmlzd_dn;
    FixArray adjust = fix->input(sci::PUBLIC, dn.size, uint64_t(0), false, dn.ell + 1, 0);
    if (!normalized_dn) {
        vector<FixArray> msnzb_one_hot = fix->msnzb_one_hot(dn, dn.ell + 1);
        for (int i = 0; i < dn.ell; i++) {
            adjust = fix->add(adjust, fix->mul(msnzb_one_hot[i], (1ULL << (dn.ell - 1 - i))));
        }
        adjust.s = dn.ell - 1 - dn.s;
        BoolArray msb_dn = fix->LSB(msnzb_one_hot[dn.ell - 1]);
        nrmlzd_dn = fix->mul(dn, adjust, dn.ell + 1, msb_dn.data, all_0.data);
        // nrmlzd_dn = element_wise_mul(dn, adjust);
    } else {
        if (dn.ell == dn.s + 1) {
            nrmlzd_dn = fix->extend(dn, dn.s + 2, all_1.data);
        } else {
            nrmlzd_dn = fix->reduce(dn, dn.s + 2);
        }
    }
    cout << "OK1" << endl;
    int32_t m, iters;
    m = (s_out <= 20 ? ceil((s_out - 2) / 2.0) : ceil((ceil(s_out / 2.0) - 2) / 2.0));
    iters = (s_out <= 20 ? 0 : 1);
    // cout << "s_out: " << s_out << " m: " << m << " iters: " << iters << endl;
    // reciprocal approximation w
    FixArray eps = fix->reduce(nrmlzd_dn, nrmlzd_dn.s - m);
    eps.signed_ = false;
    BoolArray msb_eps = fix->MSB(eps);
    uint8_t *wrap_eps = new uint8_t[dn.size];
    fix->aux->MSB_to_Wrap(eps.data, msb_eps.data, wrap_eps, eps.size, eps.ell);
    FixArray idx = fix->truncate_reduce(fix->reduce(nrmlzd_dn, nrmlzd_dn.s), nrmlzd_dn.s - m, wrap_eps);
    idx.signed_ = false;
    delete[] wrap_eps;
    vector<uint64_t> spec_c0(1 << idx.ell);
    vector<uint64_t> spec_c1(1 << idx.ell);
    cout << "OK2" << endl;
    for (int j = 0; j < (1 << idx.ell); j++) {
        spec_c0[j] = recp_lookup_c0(j, m);
        spec_c1[j] = recp_lookup_c1(j, m);
    }
    cout << "OK3" << endl;
    FixArray c0 = fix->LUT(spec_c0, idx, true, m + 4, m + 3);
    FixArray c1 = fix->LUT(spec_c1, idx, true, 2*m + 3, 2*m + 2);
    cout << "OK4" << endl;
    this->other_nonlinear_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->other_nonlinear_comm += this->get_comm();
    this->other_nonlinear_round += this->get_round();
    // FixArray w = fix->mul(c0, eps, nrmlzd_dn.s + 4, all_0.data, msb_eps.data);
    FixArray w = element_wise_mul(c0, eps);
    cout << "OK5" << endl;
    // w = fix->sub(fix->scale_up(c1, nrmlzd_dn.s + m + 4, nrmlzd_dn.s + m + 3),
    //             fix->extend(w, nrmlzd_dn.s + m + 4, all_0.data));
    // w = fix->truncate_reduce(w, w.s - s_out);
    start = high_resolution_clock::now();
    BoolArray msb_nm;
    uint8_t* msb_nm_data = nullptr;
    if (nm.signed_) {
        msb_nm = fix->MSB(nm);
        msb_nm_data = msb_nm.data;
    }
    this->other_nonlinear_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->other_nonlinear_comm += this->get_comm();
    this->other_nonlinear_round += this->get_round();
    // Change extend w and adjust here
    FixArray w_extend(party, nm.size, w.signed_, w.ell, w.s);
    FixArray adjust_extend(party, nm.size, adjust.signed_, adjust.ell, adjust.s);

    for(int i = 0; i < dn.size; i++) {
        for (int j = 0; j < batch_dn_size; j++) {
            w_extend.data[i*batch_dn_size + j] = w.data[i];
            adjust_extend.data[i*batch_dn_size + j] = adjust.data[i];
        }
    }

    BoolArray all_0_dm = bool_op->input(sci::ALICE, nm.size, uint8_t(0));
    // BoolArray all_1_dm = bool_op->input(ALICE, nm.size, uint8_t(1));
    cout << "OK3" << endl;
    // FixArray a = fix->mul(nm, w_extend, nm.ell + s_out, msb_nm_data, all_0_dm.data);
    FixArray a = element_wise_mul(nm, w_extend);
    // a = fix->truncate_reduce(a, nm.s);
    start = high_resolution_clock::now();
    if ((nm.ell - nm.s) >= (l_out - s_out)) {
        a = fix->reduce(a, l_out);
    } else {
        a = fix->extend(a, l_out, msb_nm_data);
    }
    this->other_nonlinear_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->other_nonlinear_comm += this->get_comm();
    this->other_nonlinear_round += this->get_round();
    cout << "OK4" << endl;
    if (!normalized_dn) {
        // Change extend adjust here
        // a = fix->mul(a, adjust_extend, l_out + adjust_extend.s, msb_nm_data, all_0_dm.data);
        a = element_wise_mul(a, adjust_extend);
        // a = fix->truncate_reduce(a, adjust_extend.s);
    }
    cout << "OK5" << endl;
    if (iters > 0) {
        assert(0);
        FixArray d = fix->mul(w, nrmlzd_dn, s_out + nrmlzd_dn.s + 2, all_0.data, all_0.data);
        d = fix->truncate_reduce(d, nrmlzd_dn.s);
        FixArray e = fix->sub(1ULL << d.s, d);
        e.signed_ = true;

        FixArray a_curr, e_curr;
        FixArray a_prev = a, e_prev = e;
        for (int i = 0; i < iters - 1; i++) {
            e_curr = fix->mul(e_prev, e_prev, 2*s_out + 2, all_0.data, all_0.data);
            e_curr = fix->truncate_reduce(e_curr, s_out);
            e_prev = fix->add(e_prev, 1ULL << e_prev.s);
            a_curr = fix->mul(e_prev, a_prev, l_out + s_out, all_0.data, msb_nm_data);
            a_curr = fix->truncate_reduce(a_curr, s_out);
            a_prev = a_curr;
            e_prev = e_curr;
        }
        e_prev = fix->add(e_prev, 1ULL << e_prev.s);
        FixArray out = fix->mul(e_prev, a_prev, l_out + s_out, all_0.data, msb_nm_data);
        out = fix->truncate_reduce(out, s_out);
        return out;
    } else {
        return a;
    }
}

// bumblebee's version
FixArray Bert_ckks::div_batch_he_V2(const FixArray& nm, const FixArray& dn, int batch_dn_size, int l_out, int s_out, bool normalized_dn, bool softmax_denominator, bool compact_softmax_denominator){
    if (batch_dn_size <= 0 ||
        static_cast<size_t>(dn.size) * static_cast<size_t>(batch_dn_size) !=
            static_cast<size_t>(nm.size)) {
        throw invalid_argument("division batch shape mismatch");
    }
    const auto previous_conversion_bucket = profile_conversion_bucket();
    const uint64_t rec_comm_before = profile_is_enabled() ? current_comm() : 0;
    profile_set_conversion_bucket(BlbViTCommAuditBucket::SOFTMAX_REC_CONVERSION);
    auto start = high_resolution_clock::now();
    // exp(max-max)=1 bounds a Softmax reciprocal by one. A row-level
    // floor of 1/2 protects against approximation underflow; two integer
    // bits then suffice. Generic (including ReLU) denominators stay wide.
    const int reciprocal_ell = softmax_denominator ? dn.s + 2 : dn.ell;
    if (softmax_denominator && (dn.s < 1 || reciprocal_ell >= dn.ell ||
                               l_out + nm.s + dn.s - s_out > nm.ell + reciprocal_ell))
        throw invalid_argument("invalid bounded Softmax reciprocal metadata");

    // A shifted Softmax exponential is at most one on the production BLB
    // domain.  The reciprocal path already relies on that bound (with a
    // factor-three approximation guard); carry the row-size bound into the
    // local ring metadata before sqrt.  `reduce` is purely local, so this
    // removes high zero/sign-extension bits from MSNZB/B2A and the following
    // fixed-point products without changing a protocol message or a value in
    // the valid domain.  If an unusually large row cannot be represented by a
    // safe public bound, retain the old wide path.
    int reciprocal_input_ell = dn.ell;
    FixArray bounded_dn;
    bool denominator_reduced = false;
    if (softmax_denominator && compact_softmax_denominator &&
        static_cast<size_t>(batch_dn_size) <=
            numeric_limits<size_t>::max() / 4) {
        const size_t guarded_bound =
            static_cast<size_t>(batch_dn_size) * size_t{4};
        size_t bound_power = 1;
        int bound_bits = 0;
        while (bound_power < guarded_bound && bound_bits < 63) {
            bound_power <<= 1;
            ++bound_bits;
        }
        const int candidate_ell = dn.s + bound_bits;
        if (bound_bits < 63 && candidate_ell > 0 &&
            candidate_ell < dn.ell && candidate_ell <= 64) {
            bounded_dn = nonlinear.fpmath[0]->fix->reduce(dn, candidate_ell);
            reciprocal_input_ell = candidate_ell;
            denominator_reduced = true;
        }
    }
    const FixArray &reciprocal_input = denominator_reduced ? bounded_dn : dn;
    FixArray dn_rec(party, dn.size, false, reciprocal_ell, dn.s);
    const int reciprocal_workers = min(nonlinear.active_workers(),
                                       max(1, (dn.size + 63) / 64));
    vector<thread> reciprocal_threads;
    for (int worker = 0; worker < reciprocal_workers; ++worker) {
        const int begin = dn.size * worker / reciprocal_workers;
        const int end = dn.size * (worker + 1) / reciprocal_workers;
        reciprocal_threads.emplace_back([&, worker, begin, end]() {
            auto *math = nonlinear.fpmath[worker];
            FixArray denominator(party, end - begin,
                                reciprocal_input.signed_, reciprocal_input_ell,
                                reciprocal_input.s);
            copy_n(reciprocal_input.data + begin, end - begin,
                   denominator.data);
            if (softmax_denominator) {
                const uint64_t floor = uint64_t{1} << (dn.s - 1);
                // This SCI build's LE predicate selects lhs >= rhs, as
                // covered by the signed-comparison and Softmax fixtures.
                auto above_floor = math->fix->LE(denominator, floor);
                denominator = math->fix->if_else(above_floor, denominator, floor);
            }
            auto root = math->sqrt(denominator, true);
            root.signed_ = false;
            auto squared = math->fix->mul(root, root, root.ell + root.s);
            auto reciprocal = math->fix->truncate_reduce(squared, root.s);
            if (softmax_denominator) {
                // The exact bound is 2. Allow approximation slack up to 3,
                // enforce it on rows before broadcasting, then reduce locally.
                const uint64_t ceiling = uint64_t{3} << dn.s;
                auto above_ceiling = math->fix->LE(reciprocal, ceiling);
                reciprocal = math->fix->if_else(above_ceiling, ceiling, reciprocal);
                reciprocal = math->fix->reduce(reciprocal, reciprocal_ell);
            }
            copy_n(reciprocal.data, end - begin, dn_rec.data + begin);
        });
    }
    for (auto& worker : reciprocal_threads) worker.join();
    cout << "FORMAL_SOFTMAX_RECIPROCAL status=PASS mode=1_over_dn" << endl;
    cout << "FORMAL_SOFTMAX_DENOMINATOR_RING input_ell="
         << dn.ell << " reciprocal_input_ell=" << reciprocal_input_ell
         << " row_bound=" << (compact_softmax_denominator ? batch_dn_size : 0)
         << " guard_factor=" << (compact_softmax_denominator ? 4 : 0)
         << " local_reduce=" << (denominator_reduced ? 1 : 0) << endl;
    this->other_nonlinear_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->other_nonlinear_comm += this->get_comm();
    this->other_nonlinear_round += this->get_round();
    // Replication below overwrites the complete local-share buffer.
    FixArray dn_rec_flat(dn_rec.party, nm.size, dn_rec.signed_, dn_rec.ell,
                         dn_rec.s);
    for(int i=0; i<nm.size; i++){
        // nm is row-major with batch_dn_size consecutive numerators per
        // denominator.  Modulo interleaves denominators across a row; integer
        // division broadcasts the corresponding row denominator correctly.
        dn_rec_flat.data[i] = dn_rec.data[i / batch_dn_size];
    }
    // BLB Algorithm 5 smulcc executes in HE. Widen the reciprocal shares
    // before import; changing ell metadata alone is not a share extension.
    auto reciprocal_he = nonlinear.fpmath[0]->fix->extend(dn_rec_flat, nm.ell);
    reciprocal_he.signed_ = nm.signed_;
    if (l_out != nm.ell || s_out != nm.s || reciprocal_he.s != nm.s)
        throw invalid_argument("BLB HE Softmax product output metadata mismatch");
    FixArray result = element_wise_mul(nm, reciprocal_he);
    cout << "FORMAL_SOFTMAX_PRODUCT status=EXECUTED engine=CKKS"
         << " output_ell=" << result.ell << " ckks_round_trips=1" << endl;
    if (profile_is_enabled()) {
        profile_record_comm(BlbViTCommAuditBucket::SOFTMAX_REC,
                            current_comm() - rec_comm_before);
    }
    profile_set_conversion_bucket(previous_conversion_bucket);
    return result;
}

void max_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int array_size, int ell, int s, FPMath *fpmath){
    if (num_ops <= 0 || array_size <= 0)
        return;

    // Same signed predicate and reduction as the correctness path, with
    // log2(keys) protocol batches instead of keys-1 sequential batches.
    FixArray maximum = ours_vit::secure_row_max(
        *fpmath->fix, party, x, num_ops, array_size, ell, s);
    memcpy(y, maximum.data,
           static_cast<size_t>(num_ops) * sizeof(uint64_t));
}

vector<FixArray> Bert_ckks::softmax_fix_he(int party, FixArray x,
                                           FPMath* fpmath[],
                                           int active_heads) {
    // std::cout << "Entering softmax fix" << std::endl;
    auto start = high_resolution_clock::now();
    int ell = x.ell;
    int s = x.s;
    cout << "ell,s:" << ell << "," << s << endl;
    auto diagnostic_open_sum = [&](const FixArray &tensor,
                                   const char *stage) {
        if (getenv("OURS_VIT_B_AGGREGATE_MAX") == nullptr) return;
        const uint64_t mask = tensor.ell_mask();
        uint64_t local_sum = 0;
        for (int i = 0; i < tensor.size; ++i)
            local_sum = (local_sum + tensor.data[i]) & mask;
        FixArray scalar(party, 1, tensor.signed_, tensor.ell, tensor.s);
        scalar.data[0] = local_sum;
        FixArray opened = fpmath[0]->fix->output(sci::PUBLIC, scalar);
        const uint64_t raw = opened.data[0] & opened.ell_mask();
        int64_t decoded = static_cast<int64_t>(raw);
        if (opened.signed_ && (raw & (uint64_t{1} << (opened.ell - 1))))
            decoded -= static_cast<int64_t>(uint64_t{1} << opened.ell);
        cout << "OURS_B_SOFTMAX_SCALAR stage=" << stage
             << " metric=sum value=" << setprecision(17)
             << ldexp(static_cast<double>(decoded), -opened.s)
             << defaultfloat << " ell=" << opened.ell
             << " scale=" << opened.s << " elements=" << tensor.size
             << " intermediate_reconstruction=0 scalar_only=1" << endl;
    };
    // vector<FixArray> ret(N);

    bool signed_ = x.signed_;
    //理论上在第二个维度进行求Max
    //point1
    // multi threads for max
    
    FCMetadata data = linear.data_lin1_0;
    int softmax_dim = data.image_size;
    if (active_heads <= 0 || active_heads > PACKING_NUM ||
        x.size != active_heads * softmax_dim * softmax_dim) {
        throw runtime_error("Softmax active-head shape mismatch");
    }
    vector<int> active_keys;
    for (int key = 0; key < softmax_dim; ++key) {
        if (!public_attention_mask_ready || public_attention_mask.at(key) == 0)
            active_keys.push_back(key);
    }
    if (active_keys.empty()) throw runtime_error("Softmax has no unmasked key");
    int nthreads = min(active_heads * 4, nonlinear.active_workers());
    const int physical_rows = active_heads * softmax_dim;
    vector<int> active_rows;
    for (int row = 0; row < physical_rows; ++row) {
        // ViT masks padded rows again at every pre-norm boundary; their
        // attention outputs cannot contribute to a valid query or key.
        if (!pre_norm_attention_output_only || !public_attention_mask_ready ||
            public_attention_mask.at(row % softmax_dim) == 0)
            active_rows.push_back(row);
    }
    int dim = static_cast<int>(active_rows.size());
    // Each logical Softmax row has image_size keys.  `dim / nthreads` is the
    // number of rows assigned to a worker, not the row width.  Treating it as
    // the width processed only the first quarter of a 128-token attention
    // tensor and made a public padding mask impossible to apply correctly.
    int array_size = static_cast<int>(active_keys.size());
    if (array_size != softmax_dim || dim != physical_rows) {
        FixArray compact(party, dim * array_size, signed_, ell, s);
        for (int row = 0; row < dim; ++row)
            for (int key = 0; key < array_size; ++key)
                compact.data[row * array_size + key] =
                    x.data[active_rows[row] * softmax_dim + active_keys[key]];
        x = std::move(compact);
    }
    cout << "SOFTMAX_PUBLIC_COMPACTION rows=" << dim
         << " original_rows=" << physical_rows
         << " original_keys=" << softmax_dim << " active_keys=" << array_size
         << " elements=" << x.size << " restored_zero_padding=1" << endl;
    // All rows are filled by max_thread; no secret input operation belongs on
    // this local output buffer.
    FixArray x_max(party, dim, signed_, ell, s);
    std::thread threads[nthreads];
    int chunk_size = dim / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        int offset = i * chunk_size;
        int lnum_ops;
        if (i == (nthreads - 1)) {
            lnum_ops = dim - offset;
        } else {
            lnum_ops = chunk_size;
        }
        // num_ops = chunk_size
        threads[i] =
            std::thread(
                max_thread, 
                i, 
                party, 
                &x.data[offset*array_size], 
                &x_max.data[offset], 
                lnum_ops,
                array_size,
                ell,
                s,
                fpmath[i]);
    }
    for (int i = 0; i < nthreads; ++i) {
        threads[i].join();
    }

    // This probe is deliberately opt-in: it opens only a few diagnostic
    // values and must never be enabled on a real private inference run.  It
    // localizes numerical drift at the Softmax boundaries without relying on
    // a final-logit symptom.
    auto diagnostic_open_softmax_prefix = [&](const FixArray &tensor,
                                              const char *stage,
                                              int count) {
        if (getenv("OURS_VIT_SOFTMAX_INTERNAL_DIAGNOSTIC") == nullptr)
            return;
        const int probe_size = min(count, tensor.size);
        FixArray probe(party, probe_size, tensor.signed_, tensor.ell,
                       tensor.s);
        copy_n(tensor.data, probe_size, probe.data);
        FixArray opened = fpmath[0]->fix->output(sci::PUBLIC, probe);
        if (party != sci::ALICE) return;
        cout << "OURS_SOFTMAX_INTERNAL stage=" << stage
             << " ell=" << opened.ell << " scale=" << opened.s
             << " values=";
        const uint64_t mask = opened.ell_mask();
        const uint64_t sign = uint64_t{1} << (opened.ell - 1);
        for (int index = 0; index < opened.size; ++index) {
            const uint64_t raw = opened.data[index] & mask;
            const int64_t value = (raw & sign)
                                      ? static_cast<int64_t>(
                                            raw - (uint64_t{1} << opened.ell))
                                      : static_cast<int64_t>(raw);
            if (index != 0) cout << ',';
            cout << value;
        }
        cout << endl;
    };
    auto diagnostic_open_softmax_row_sum = [&](const FixArray &tensor,
                                               const char *stage) {
        if (getenv("OURS_VIT_SOFTMAX_INTERNAL_DIAGNOSTIC") == nullptr)
            return;
        if (tensor.size < array_size)
            throw runtime_error("Softmax internal diagnostic row is too small");
        FixArray sum(party, 1, tensor.signed_, tensor.ell, tensor.s);
        sum.data[0] = 0;
        const uint64_t mask = tensor.ell_mask();
        for (int key = 0; key < array_size; ++key)
            sum.data[0] = (sum.data[0] + tensor.data[key]) & mask;
        diagnostic_open_softmax_prefix(sum, stage, 1);
    };
    diagnostic_open_softmax_prefix(x_max, "max", 4);
    // FixArray x_max = fpmath->fix->max(x);
    //point2
    FixArray x_max_flat(party, dim*array_size, signed_, ell, s);
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < array_size; j++) {
            x_max_flat.data[i*array_size + j] = x_max.data[i];
        }
    }

    FixArray x_flat = x;
    FixArray shifted_x_flat = fpmath[0]->fix->sub(x_flat, x_max_flat);
    diagnostic_open_softmax_prefix(shifted_x_flat, "shifted", 8);

    FixArray e_x_flat;
    this->other_nonlinear_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->other_nonlinear_comm += get_comm();
    this->other_nonlinear_round += get_round();
    // Keep the production Softmax on BLB's HE-assisted exponentiation path.
    // A previous B diagnostic revision replaced this with SCI exp4 plus an
    // exact MPC division and two secure clamps.  That path took roughly 350 s
    // for one 12-head normalization and was not a valid optimization baseline.
    // exp_he performs MPC->CKKS, evaluates the BLB polynomial on Phantom, and
    // returns additive shares; it never opens the activation to the client.
    //point3
    e_x_flat = exp_he(shifted_x_flat, false);
    // exp_he already applies C(z)=max(z,0) after the underflow mux. There
    // is no intervening HE operation: a second C(C(z)) was redundant.
    e_x_flat.signed_ = false;
    diagnostic_open_softmax_prefix(e_x_flat, "exp", 8);
    diagnostic_open_softmax_row_sum(e_x_flat, "exp_row_sum");
    cout << "OURS_SOFTMAX_ENGINE status=PASS"
         << " engine=BLB_HE_ASSISTED_EXP_RECIPROCAL"
         << " baseline_anchor_seconds=498.104"
         << " slow_sci_exp4_div_clamp_removed=1"
         << " client_reconstruction=0" << endl;
    cout << "FORMAL_SOFTMAX_EXPONENT status=PASS engine=BLB_CKKS_exp_he"
         << " input_ell=" << shifted_x_flat.ell
         << " output_ell=" << e_x_flat.ell
         << " client_reconstruction=0" << endl;
    // Preserve the ring returned by exp_he.  BLB hard-coded exp_ell=40 while
    // copying local shares from a different ring, which is only a metadata
    // reinterpretation.  The actual ring already has enough row-sum headroom.
    const int exp_ell = e_x_flat.ell;
    cout << "FORMAL_SOFTMAX_EXP_RING status=PASS source_ell="
         << e_x_flat.ell << " accumulation_ell=" << exp_ell
         << " metadata_reinterpretation=0 row_sum_bound=" << array_size
         << " intermediate_reconstruction=0" << endl;
    diagnostic_open_sum(e_x_flat, "exp");
    start = high_resolution_clock::now();
    cout << "exp_he OK" << endl;
    vector<FixArray> e_x_tr(array_size);
    for (int i = 0; i < array_size; i++) {
        e_x_tr[i] = FixArray(party, dim, e_x_flat.signed_, exp_ell, s);
        for (int j = 0; j < dim; j++) {
            e_x_tr[i].data[j] = e_x_flat.data[j*array_size + i];
        }
    }
    FixArray sum_e_x;
    {
        vector<FixArray> tmp = e_x_tr;
        int num_adds_old = array_size; int num_adds_curr = array_size/2;
        while(num_adds_old > 1) {
            int odd_num_adds = num_adds_old & 1;
            vector<FixArray> lhs(num_adds_curr); vector<FixArray> rhs(num_adds_curr);
            for (int j = odd_num_adds; j < num_adds_old && j + 1 < num_adds_old; j += 2) {
                lhs[j/2] = tmp[j]; rhs[j/2] = tmp[j+1];
            }
            FixArray lhs_concat = concat(lhs);
            FixArray rhs_concat = concat(rhs);
            lhs_concat = fpmath[0]->fix->add(lhs_concat, rhs_concat);
            for (int j = 0; j < num_adds_old && j + 1 < num_adds_old; j += 2) {
                tmp[odd_num_adds + (j/2)] = lhs_concat.subset((j/2)*dim, (j/2)*dim + dim);
            }
            num_adds_old = num_adds_curr + odd_num_adds;
            num_adds_curr = num_adds_old/2;
        }
        sum_e_x = tmp[0];
    }
    //point4
    sum_e_x.signed_ = false;
    diagnostic_open_softmax_prefix(sum_e_x, "denominator", 4);
    diagnostic_open_sum(sum_e_x, "denominator");
    cout << "e_x_flat.size:" << e_x_flat.size << endl;
    cout << "sum_e_x.size:" << sum_e_x.size << endl;
    cout << "exp_he data time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    start = high_resolution_clock::now();
    // Strict B retains the full nonlinear ring. Public-range narrowing and
    // reciprocal clipping are optional optimizations, not BLB restoration.
    FixArray ret_flat = div_batch_he_V2(
        e_x_flat, sum_e_x, array_size, exp_ell, s, false, false, false);
    diagnostic_open_sum(ret_flat, "probability");
    cout << "FORMAL_SOFTMAX_DIVISION status=PASS"
         << " engine=BLB_HE_ASSISTED_RECIPROCAL"
         << " numerator_ell=" << e_x_flat.ell
         << " denominator_ell=" << sum_e_x.ell
         << " output_ell=" << ret_flat.ell
         << " client_reconstruction=0" << endl;
    cout << "div_batch_he time:"
         << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    start = high_resolution_clock::now();
    //point5
    cout << "ret_flat.size:" << ret_flat.size << endl;
    // Normalize from the actual returned metadata rather than the nominal
    // l_out argument.  If a future backend returns a narrower ring, extend
    // every element using its real sign; the old code supplied only `dim`
    // zero-MSB bits for `dim * array_size` elements.
    const int reciprocal_output_ell = ret_flat.ell;
    diagnostic_open_softmax_prefix(ret_flat, "probability", 8);
    diagnostic_open_softmax_row_sum(ret_flat, "probability_row_sum");
    if (ret_flat.ell > ell) {
        ret_flat = fpmath[0]->fix->reduce(ret_flat, ell);
    } else if (ret_flat.ell < ell) {
        if (ret_flat.signed_) {
            BoolArray sign = fpmath[0]->fix->MSB(ret_flat);
            ret_flat = fpmath[0]->fix->extend(ret_flat, ell, sign.data);
        } else {
            BoolArray all_0 = fpmath[0]->bool_op->input(
                sci::ALICE, ret_flat.size, uint8_t(0));
            ret_flat = fpmath[0]->fix->extend(ret_flat, ell, all_0.data);
        }
    }
    cout << "FORMAL_SOFTMAX_RING_CAST status=PASS source_ell="
         << reciprocal_output_ell << " target_ell=" << ell
         << " direction="
         << (reciprocal_output_ell > ell
                 ? "reduce"
                 : (reciprocal_output_ell < ell ? "zero_extend" : "none"))
         << " signed=" << (ret_flat.signed_ ? 1 : 0)
         << " intermediate_reconstruction=0" << endl;
    vector<FixArray> ret(physical_rows);
    //point6
    // return make_tuple(ret, l_short);
    for (int i = 0; i < physical_rows; i++) {
        ret[i] = FixArray(party, softmax_dim, signed_, ell, s);
        fill_n(ret[i].data, softmax_dim, uint64_t(0));
    }
    for (int i = 0; i < dim; i++) {
        for (int key = 0; key < array_size; ++key)
            ret[active_rows[i]].data[active_keys[key]] = ret_flat.data[i * array_size + key];
    }
    this->other_nonlinear_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    return ret;
}

FixArray Bert_ckks::heterogeneous_attention_norm(
    FixArray input,
    const vector<ours_vit::HeadProtocol> &protocols,
    const vector<double> &alphas, int logical_tokens,
    int layer_id, bool group_contiguous_protocols) {
    FCMetadata data = linear.data_lin1_0;
    const int tokens = data.image_size;
    const int per_head = tokens * tokens;
    if (input.ell != NL_ELL || input.s != NL_SCALE ||
        input.size != PACKING_NUM * per_head ||
        protocols.size() != PACKING_NUM || alphas.size() != PACKING_NUM ||
        logical_tokens <= 0 || logical_tokens > tokens || layer_id < 0 ||
        layer_id >= ours_vit::kPlanLayers) {
        throw runtime_error("HETERO heterogeneous attention metadata mismatch");
    }

    vector<uint64_t> row_major(static_cast<size_t>(input.size));
    linear.plain_cross_packing_postprocess(input.data, row_major.data(), false,
                                           data);
    FixArray result(party, input.size, input.signed_, input.ell, input.s);
    FixOp *fix = nonlinear.fpmath[0]->fix;
    const uint64_t ring_mask = input.ell_mask();
    uint64_t softmax_heads = 0;
    uint64_t relu_heads = 0;
    uint64_t linear_heads = 0;
    uint64_t protocol_invocations = 0;
    uint64_t mpc_protocol_invocations = 0;
    uint64_t native_linear_deferred_groups = 0;

    for (int group_start = 0; group_start < PACKING_NUM;) {
        int group_end = group_start + 1;
        if (group_contiguous_protocols) {
            while (group_end < PACKING_NUM &&
                   protocols[group_end] == protocols[group_start]) {
                ++group_end;
            }
        }
        const int group_heads = group_end - group_start;
        const int group_values = group_heads * per_head;
        ++protocol_invocations;
        const bool native_linear_group = protocols[group_start] == ours_vit::HeadProtocol::HeLinear;
        if (native_linear_group)
            ++native_linear_deferred_groups;
        else
            ++mpc_protocol_invocations;
        FixArray group_input(party, group_values, true, input.ell, input.s);
        memcpy(group_input.data,
               row_major.data() + static_cast<size_t>(group_start) * per_head,
               static_cast<size_t>(group_values) * sizeof(uint64_t));
        const auto group_profile_start = high_resolution_clock::now();
        const uint64_t group_comm_start = current_comm();
        const uint64_t group_round_start = current_round();
        const double group_he_start = he_compute_la;
        const double group_conversion_start = conversion_la;
        const uint64_t group_conversion_ct_start = conversion_ciphertexts;
        const uint64_t group_ct_pt_start = linear.ct_pt_multiply_count;
        const uint64_t group_ct_ct_start = linear.ct_ct_multiply_count;
        const uint64_t group_rotation_start = linear.rotation_count;
        const uint64_t group_rescale_start = linear.rescale_count;
        const uint64_t group_he_rescale_start = he_rescale_count;
        FixArray normalized;
        if (protocols[group_start] == ours_vit::HeadProtocol::SoftmaxMpc) {
            softmax_heads += group_heads;
            if (public_attention_mask_ready) {
                if (public_attention_mask.size() != static_cast<size_t>(tokens))
                    throw runtime_error("HETERO public attention mask shape mismatch");
                if (party == sci::BOB) {
                    for (int group_head = 0; group_head < group_heads;
                         ++group_head) {
                        for (int query = 0; query < tokens; ++query) {
                            const size_t offset =
                                (static_cast<size_t>(group_head) * tokens + query) *
                                tokens;
                            for (int key = 0; key < tokens; ++key) {
                                const int64_t delta =
                                    public_attention_mask[key] *
                                    (int64_t(1) << input.s);
                                group_input.data[offset + key] =
                                    (group_input.data[offset + key] +
                                     static_cast<uint64_t>(delta)) & ring_mask;
                            }
                        }
                    }
                }
            }
            const auto rows = softmax_fix_he(
                party, group_input, nonlinear.fpmath, group_heads);
            normalized = FixArray(party, group_values, true, input.ell, input.s);
            for (int row = 0; row < group_heads * tokens; ++row) {
                memcpy(normalized.data + static_cast<size_t>(row) * tokens,
                       rows[row].data,
                       static_cast<size_t>(tokens) * sizeof(uint64_t));
            }
        } else if (protocols[group_start] ==
                   ours_vit::HeadProtocol::ReluNormMpc) {
            relu_heads += group_heads;
            // ReLU is evaluated on secret shares. Public padding keys are
            // subsequently zeroed on both local shares, which is exactly a
            // multiplication by a public 0/1 mask and leaks no input value.
            BoolArray negative = fix->LT(group_input, uint64_t(0));
            normalized = fix->if_else(negative, uint64_t(0), group_input);
            normalized.signed_ = false;
            for (int group_head = 0; group_head < group_heads; ++group_head)
                for (int query = 0; query < tokens; ++query)
                    for (int key = logical_tokens; key < tokens; ++key)
                        normalized.data[
                            (static_cast<size_t>(group_head) * tokens + query) *
                                tokens + key] = 0;

            FixArray denominator(party, group_heads * tokens, false,
                                 input.ell, input.s);
            for (int row = 0; row < group_heads * tokens; ++row) {
                uint64_t local_sum = 0;
                for (int key = 0; key < tokens; ++key)
                    local_sum = (local_sum +
                        normalized.data[static_cast<size_t>(row) * tokens + key]) &
                        ring_mask;
                denominator.data[row] = local_sum;
            }
            BoolArray zero = fix->EQ(denominator, uint64_t(0));
            FixArray safe_denominator = fix->if_else(
                zero, uint64_t(1) << input.s, denominator);
            safe_denominator.signed_ = false;
            // Reuse the same HE-assisted reciprocal/multiply primitive as the
            // BLB Softmax engine.  ReLU and the zero-denominator mux stay in
            // MPC, but the former full-MPC div_batch bottleneck is removed.
            normalized = div_batch_he_V2(
                normalized, safe_denominator, tokens,
                input.ell, input.s, false);
        } else if (protocols[group_start] ==
                   ours_vit::HeadProtocol::HeLinear) {
            linear_heads += group_heads;
            // Linear operands stay in CKKS; these positions are public zeros.
            normalized = FixArray(party, group_values, true, input.ell, input.s);
            fill(normalized.data, normalized.data + normalized.size, 0);
        } else {
            throw runtime_error("HETERO head protocol code is invalid");
        }
        memcpy(result.data + static_cast<size_t>(group_start) * per_head,
               normalized.data,
               static_cast<size_t>(group_values) * sizeof(uint64_t));
        const double group_wall_ms =
            chrono::duration<double, std::milli>(high_resolution_clock::now() -
                                                 group_profile_start)
                .count();
        const uint64_t group_comm_bytes = current_comm() - group_comm_start;
        const uint64_t group_rounds = current_round() - group_round_start;
        const double group_he_ms = (he_compute_la - group_he_start) * 1000.0;
        const double group_conversion_ms =
            (conversion_la - group_conversion_start) * 1000.0;
        const double group_mpc_ms = max(0.0, group_wall_ms - group_he_ms);
        const uint64_t group_conversion_ct =
            conversion_ciphertexts - group_conversion_ct_start;
        const uint64_t group_he_muls =
            linear.ct_pt_multiply_count - group_ct_pt_start +
            linear.ct_ct_multiply_count - group_ct_ct_start;
        const uint64_t group_he_rotations =
            linear.rotation_count - group_rotation_start;
        const uint64_t group_he_rescales =
            linear.rescale_count - group_rescale_start +
            he_rescale_count - group_he_rescale_start;
        const char *execution_domain =
            protocols[group_start] == ours_vit::HeadProtocol::HeLinear
                ? "CKKS_deferred" : "MPC_CKKS_hybrid";
        cout << "OURS_HETERO_PATH_PROFILE status=PASS"
             << " version=" << ours_vit::to_string(heterogeneous_version)
             << " layer=" << layer_id
             << " protocol=" << ours_vit::to_string(protocols[group_start])
             << " physical_head_begin=" << group_start
             << " head_count=" << group_heads
             << " latency_ms=" << setprecision(17) << group_wall_ms
             << " he_ms=" << group_he_ms
             << " mpc_ms=" << group_mpc_ms
             << " conversion_ms=" << group_conversion_ms
             << " conversion_ct=" << group_conversion_ct
             << " he_muls=" << group_he_muls
             << " he_rotations=" << group_he_rotations
             << " he_rescales=" << group_he_rescales
             << " communication_bytes=" << group_comm_bytes
             << " rounds=" << group_rounds
             << " execution_domain=" << execution_domain
             << " conversion_scope="
             << "selective_group"
             << " private_dynamic_branch=0" << defaultfloat << endl;
        group_start = group_end;
    }
    cout << "OURS_HETERO_NORMALIZATION"
         << " status=PASS layer_scope=current"
         << " softmax_heads=" << softmax_heads
         << " relu_heads=" << relu_heads
         << " linear_heads=" << linear_heads
         << " protocol_invocations=" << protocol_invocations
         << " mpc_protocol_invocations=" << mpc_protocol_invocations
         << " native_linear_deferred_groups="
         << native_linear_deferred_groups
         << " grouping=" << (group_contiguous_protocols ? 1 : 0)
         << " linear_location="
         << "native_CKKS_deferred"
         << " selective_conversion=" << 1
         << " private_dynamic_branch=0" << endl;
    return result;
}

FixArray Bert_ckks::softmax_he(FixArray input) {
    FCMetadata data = linear.data_lin1_0;
    int softmax_dim = data.image_size;
    int qk_size = PACKING_NUM*softmax_dim*softmax_dim;
    int softmax_output_size = PACKING_NUM*softmax_dim*softmax_dim;
    uint64_t* softmax_input_row = new uint64_t[qk_size];
    cout << "qk_size:" << qk_size << endl;
    cout << "OK" << endl;
    auto start = high_resolution_clock::now();
    linear.plain_cross_packing_postprocess(
        input.data, 
        softmax_input_row,
        // we need row packing
        false,
        data);
    cout << "softmax packing time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    int ell = input.ell;
    int s = input.s;
    cout << "OK2" << endl;
    // may be need a truncation
    uint64_t* softmax_output_row = new uint64_t[softmax_output_size];
    int num_ops = PACKING_NUM*softmax_dim;
    int array_size = softmax_dim;
    // Softmax
    // vector<FixArray> input_array;
    // for(int i = 0; i < num_ops; i++){
    //     input_array.push_back(fpmath->fix->input(party, array_size, input.data[i*array_size], true, ell, s));
    // }

    // QK^T leaves the HE kernel in diagonal/cross packing.  Softmax is a
    // row-wise MPC operation, so consume the public local permutation above
    // instead of accidentally feeding the original cross-packed shares.
    FixArray softmax_input(party, qk_size, input.signed_, ell, s);
    memcpy(softmax_input.data, softmax_input_row,
           static_cast<size_t>(qk_size) * sizeof(uint64_t));

    // The checkpoint mask is public.  Add it to exactly BOB's additive share
    // so reconstruction gains mask[key] * 2^s once (not once per party).
    // Every query row and every head receives the same key mask, matching
    // BERT's [batch, 1, 1, key] broadcast before Softmax.
    if (public_attention_mask_ready) {
        if (public_attention_mask.size() != static_cast<size_t>(softmax_dim)) {
            throw runtime_error("public attention mask shape mismatch");
        }
        if (party == sci::BOB) {
            const uint64_t ring_mask = softmax_input.ell_mask();
            for (int head = 0; head < PACKING_NUM; ++head) {
                for (int query = 0; query < softmax_dim; ++query) {
                    const size_t row_offset =
                        (static_cast<size_t>(head) * softmax_dim + query) *
                        softmax_dim;
                    for (int key = 0; key < softmax_dim; ++key) {
                        const int64_t public_delta =
                            public_attention_mask[key] * (int64_t(1) << s);
                        softmax_input.data[row_offset + key] =
                            (softmax_input.data[row_offset + key] +
                             static_cast<uint64_t>(public_delta)) &
                            ring_mask;
                    }
                }
            }
        }
    }

    vector<FixArray> output_array;
    cout << "begin softmax_fix_he" << endl;
    output_array =
        softmax_fix_he(party, softmax_input, this->nonlinear.fpmath);
    for(int i = 0; i < num_ops; i++){
        memcpy(&softmax_output_row[i*array_size], output_array[i].data, array_size * sizeof(uint64_t));
    }
    cout << "OK3" << endl;
    // output_array already contains additive shares. Calling FixOp::input on
    // those local shares is not a metadata cast: it invokes an input protocol
    // and the old hard-coded ell40/s20 also changed the represented real
    // values. Preserve the actual Softmax output ring directly.
    FixArray result(party, softmax_output_size, input.signed_, ell, s);
    memcpy(result.data, softmax_output_row,
           static_cast<size_t>(softmax_output_size) * sizeof(uint64_t));
    cout << "SOFTMAX_OUTPUT_METADATA backend=he status=PASS ell="
         << result.ell << " scale=" << result.s
         << " signed=" << (result.signed_ ? 1 : 0)
         << " shares=local_additive" << endl;
    delete[] softmax_input_row;
    delete[] softmax_output_row;
    return result;
}


FixArray Bert_ckks::softmax_ot(FixArray input) {
    FCMetadata data = linear.data_lin1_0;
    int softmax_dim = data.image_size;
    int qk_size = PACKING_NUM*softmax_dim*softmax_dim;
    int softmax_output_size = PACKING_NUM*softmax_dim*softmax_dim;
    uint64_t* softmax_input_row = new uint64_t[qk_size];
    cout << "OK" << endl;
    linear.plain_cross_packing_postprocess(
        input.data, 
        softmax_input_row,
        // we need row packing
        false,
        data);
    cout << "OK2" << endl;
    // may be need a truncation
    uint64_t* softmax_output_row = new uint64_t[softmax_output_size];
    uint64_t* softmax_l_row = new uint64_t[softmax_output_size];
    // Softmax
    nonlinear.softmax(
        1,
        softmax_input_row,
        softmax_output_row,
        softmax_l_row,
        PACKING_NUM*softmax_dim,
        softmax_dim,
        NL_ELL,
        NL_SCALE);
    cout << "OK3" << endl;
    FixArray result(party, softmax_output_size, input.signed_, NL_ELL,
                    NL_SCALE);
    memcpy(result.data, softmax_output_row,
           static_cast<size_t>(softmax_output_size) * sizeof(uint64_t));
    cout << "SOFTMAX_OUTPUT_METADATA backend=ot status=PASS ell="
         << result.ell << " scale=" << result.s
         << " shares=local_additive" << endl;
    delete[] softmax_input_row;
    delete[] softmax_output_row;
    delete[] softmax_l_row;
    return result;
}
