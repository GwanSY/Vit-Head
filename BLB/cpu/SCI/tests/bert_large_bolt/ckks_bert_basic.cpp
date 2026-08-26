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
#include <iostream>
#include <thread>
#include <cmath>
#include <vector>
#include "ckks_bert.h"

using namespace sci;
using namespace std;
using namespace seal;

#define MAX_THREADS 1
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

// A0 \in (1/4, 1)
inline uint64_t recp_lookup_c0(uint64_t index, int m) {
  uint64_t k = 1ULL << m;
  double p = 1 + (double(index) / double(k));
  double A1 = 1.0 / (p * (p + 1.0 / double(k)));
  int32_t scale = m + 3;
  uint64_t mask = (1ULL << scale) - 1;
  uint64_t val = uint64_t(A1 * (1ULL << scale)) & mask;
  return val;
}

// A1 \in (1/2, 1)
inline uint64_t recp_lookup_c1(uint64_t index, int m) {
  uint64_t k = 1ULL << m;
  double p = 1 + (double(index) / double(k));
  double z = (p * (p + (1.0 / double(k))));
  double A1 = ((1.0 / double(k * 2)) + sqrt(z)) / z;
  int32_t scale = 2 * m + 2;
  uint64_t mask = (1ULL << scale) - 1;
  uint64_t val = uint64_t(A1 * (1ULL << scale)) & mask;
  return val;
}

Bert_ckks::Bert_ckks(int party, int port, string address){
    this->party = party;
    this->address = address;
    this->port = port;
    this->io = new sci::NetIO(party == 1 ? nullptr : address.c_str(), port);

    
    this->linear = Linear_ckks(party, io);
    cout << "> Setup Linear OK" << endl;
    
    this->nonlinear = NonLinear(party, address, port+1);
    cout << "> Setup NonLinear OK" << endl;

    this->prune = prune;
    if(party == sci::ALICE){
        cout << "> Loading and preprocessing weights on server" << endl;
        #ifdef BERT_PERF
        auto t_load_model = high_resolution_clock::now();
        #endif 
        struct BertModel bm = 
            load_model("../../dataset/quantize/mrpc/weights_txt/", NUM_CLASS);

        #ifdef BERT_PERF
        cout << "> [TIMING]: Loading Model takes: " << interval(t_load_model) << "sec" << endl;
        auto t_model_preprocess = high_resolution_clock::now();
        #endif 

        linear.weights_preprocess(bm);

        #ifdef BERT_PERF
        cout << "> [TIMING]: Model Preprocessing takes: " << interval(t_model_preprocess) << "sec" << endl;
        #endif 
    }
    c_offline += this->get_comm();
    r_offline += this->get_round();
    cout << "c_offline:" << c_offline << ", r_offline:" << r_offline << endl;
    cout << "> Bert intialized done!" << endl << endl;

    cout << "> Bert_ckks intialized done!" << endl << endl;

}

Bert_ckks::~Bert_ckks() {
    
}


/*
	input: 解密后的明文share，对于server，是r，对于client，是ifft(x)-r，这里r是uniform random, size是poly degree.
	step:
		1. 双方各自做intt，并且从[0,Q]转为了[-Q/2,Q/2]
		2. 将scale变成f, 位宽变成bw, bw与f均是人为定义参数, bw>log2Q
		3. 在输入l-bit, scale是f的情况下，本地进行fft变化
        4. (可选) trunc-reduce，可以先做3，2和4合并一次。
*/
vector<vector<HE_CKKS::scalar_t>> Bert_ckks::ckks_to_mpc(vector<Plaintext> pts, int degree, uint64_t Q, int bit_out, int s_out, HE_CKKS* he, bool half){
    // cout << "begin ckks_to_mpc" << endl;
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
    vector<HE_CKKS::scalar_t> x_intt_flatten_field,x_intt_flatten_ring(x_intt.size()*x_intt[0].size());
    // #pragma omp parallel for
    for(int i=0;i<x_intt.size();i++){
        for(int j=0;j<x_intt[i].size();j++){
            x_intt_flatten_field.push_back(x_intt[i][j].real());
        }
    }
    cout << "intt done" << endl;
	// step 2
    int s_in;
    if (abs(pts[0].scale())<1){
        s_in = 0;
    }
    else{
        s_in = round(log2(pts[0].scale()));
    }
    // from (Q,s_in) to (2^bit_out, s_out), s_in is expected to be 0
    nonlinear.field_to_ring(
        64,
        x_intt_flatten_field.data(),
        Q,
        x_intt_flatten_ring.data(),
        x_intt_flatten_field.size(),
        bit_out,
        s_in,
        s_out
    );
    cout << "field_to_ring done" << endl;
	// step 3
    vector<vector<HE_CKKS::complex_t>> x_decode(pts.size());
    // #pragma omp parallel for
    for(int i=0;i<pts.size();i++){
        for(int j=0;j<degree;j++){
            x_decode[i].push_back(HE_CKKS::complex_t(x_intt_flatten_ring[i*degree+j],0));
        }
    }
    cout << "pts.size():" << pts.size() << endl;
    vector<vector<HE_CKKS::scalar_t>> y(pts.size());
    vector<vector<HE_CKKS::scalar_t>> y2(pts.size(),vector<HE_CKKS::scalar_t>(slot,0));
    // #pragma omp parallel for num_threads(num_threads)
    for(int i=0;i<pts.size();i++){
        // cout << "i:" << i << endl;
        y[i] = he->ckks_decode(x_decode[i],degree,half);
    }
    // cout << "decode ok" << endl;
    // uint8_t* msb_x = new uint8_t[y[0].size()];
    // for(int i=0;i<y.size();i++){
    //     nonlinear.fpmath[0]->fix->trunc->truncate_with_lsb(y[i].size(), y[i].data(), y2[i].data(),
    //                      1, bit_out, true,
    //                      msb_x);
    // }
    
    cout << "ckks_to_mpc done" << endl;
	return y;
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
vector<Plaintext> Bert_ckks::mpc_to_ckks(vector<vector<HE_CKKS::scalar_t>> x, int slot, uint64_t Q, int bitwidth, int s_in, int s_out, HE_CKKS* he, bool half){
    cout << "in mpc_to_ckks, slot, Q, bitwidth, s_in, s_out:" << slot << "," << Q << "," << bitwidth << "," << s_in << "," << s_out << endl;
    // step 1
    int poly_degree = slot * 2;
    // cout << "poly_degree:" << poly_degree << endl;
    vector<vector<HE_CKKS::scalar_t>> x_ifft(x.size());
    vector<vector<HE_CKKS::scalar_t>> x_ifft2(x.size(),vector<HE_CKKS::scalar_t>(poly_degree,0));
    cout << "pts.size():" << x.size() << endl;
    int thread_num = x.size()>64?64:x.size();
    #pragma omp parallel for num_threads(thread_num)
    for(int i=0;i<x.size();i++){
        x_ifft[i] = he->ckks_encode(x[i],slot,half);
    }
    // cout << "encode ok" << endl;
    // uint8_t* msb_x = new uint8_t[x_ifft[0].size()];
    // for(int i=0;i<x_ifft.size();i++){
    //     // TODO, now is fake trunc
    //     nonlinear.fpmath[0]->fix->trunc->truncate_with_lsb(x_ifft[i].size(), x_ifft[i].data(), x_ifft2[i].data(),
    //                      1, bitwidth, true,
    //                      msb_x);
    // }
    // cout << "trunc comm:" << get_comm() << endl;
    // cout << "trunc round:" << get_round() << endl;
    // cout<<"step 1 done" << endl;
    // step 2
    vector<HE_CKKS::scalar_t> x_intt_flatten_field(x_ifft.size()*x_ifft[0].size()),x_intt_flatten_ring;
    // #pragma omp parallel for
    for(int i=0;i<x_ifft.size();i++){
        for(int j=0;j<x_ifft[i].size();j++){
            x_intt_flatten_ring.push_back(x_ifft[i][j]);
        }
    }
    // cout << "num ops:"  << x_intt_flatten_ring.size() << endl;
    nonlinear.ring_to_field(64, x_intt_flatten_ring.data(), Q, x_intt_flatten_field.data(), x_intt_flatten_ring.size(), bitwidth, s_in, s_out);
    // cout << "step 2 done" << endl;
    // step 3
    vector<vector<HE_CKKS::scalar_t>> x_encode(poly_degree);
    // #pragma omp parallel for
    for(int i=0;i<x.size();i++){
        for(int j=0;j<poly_degree;j++){
            x_encode[i].push_back(x_intt_flatten_field[i*poly_degree+j]);
        }
    }
    vector<Plaintext> y(x.size());
    #pragma omp parallel for num_threads(thread_num)
    for(int i=0;i<x.size();i++){
        he->ckks_ntt(x_encode[i],poly_degree,pow(2,s_out),y[i]);
    }
    
    cout << "mpc_to_ckks done" << endl;
    return y;
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
// 		// simulate x+y*j optimization, reduce input comm by half
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
//             生成r并计算x-r，这一过程极为复杂，暂时令r=0
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
// 		// simulate x+y*j optimization, reduce input comm by half
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
//             生成r并计算x-r，这一过程极为复杂，暂时令r=0
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
    evaluate gelu poly, 输入输出均是mpc share
    ax^4+bx^3+cx^2+dx+e+0.5x
    a = 0.020848611754127593
    b = -0.18352506127082727
    c = 0.5410550166368381
    d = -0.03798164612714154
    e = 0.001620808531841547
*/ 
vector<Ciphertext> Bert_ckks::eval_gelu_poly(vector<Ciphertext> poly1, HE_CKKS* he){
    double cons_list[] = {0.020848611754127593,-0.18352506127082727,0.5410550166368381,-0.03798164612714154,0.001620808531841547};
    int num_cts = poly1.size();
    int poly_modulus_degree = he->poly_modulus_degree;
    int slot_num = poly_modulus_degree / 2;
    vector<Ciphertext> poly4(num_cts),poly3(num_cts),poly2(num_cts), output_cts(num_cts);
    // 先计算x,x^2,x^3,x^4，再对齐它们的level和scale
    // he->print_parameters(he->context->get_context_data(poly1[0].parms_id())->parms());

    // encode weight once
    vector<double> scale_up(slot_num, 1);
    Plaintext scale_up1, scale_up2, scale_up3;
    Ciphertext util_ct,util_ct2;
    {
        util_ct = poly1[0];
        util_ct.scale() = pow(util_ct.scale(),2);
        he->evaluator->rescale_to_next_inplace(util_ct);
        util_ct.scale() = pow(util_ct.scale(),2);
        he->evaluator->rescale_to_next_inplace(util_ct);

        util_ct2 = poly1[0];
        he->evaluator->mod_switch_to_next_inplace(util_ct2);
        he->evaluator->mod_switch_to_next_inplace(util_ct2);
        util_ct2.scale() = util_ct2.scale()*pow(2,12+he->he_scale);
        he->evaluator->rescale_to_next_inplace(util_ct2);
        he->evaluator->rescale_to_next_inplace(util_ct2);
    }
    he->encoder->encode(scale_up, util_ct.parms_id(), pow(2,12), scale_up1);
    he->encoder->encode(scale_up, util_ct.parms_id(), pow(2,8), scale_up2);
    he->encoder->encode(scale_up, util_ct.parms_id(), pow(2,4), scale_up3);

    // encode cons_list
    vector<Plaintext> cons_pts(5);
    vector<double> cons_matrix4(slot_num, cons_list[0]);
    vector<double> cons_matrix3(slot_num, cons_list[1]);
    vector<double> cons_matrix2(slot_num, cons_list[2]);
    vector<double> cons_matrix1(slot_num, cons_list[3]);
    vector<double> cons_matrix0(slot_num, cons_list[4]);
    vector<double> cons_matrix(slot_num, 0.5);
    he->encoder->encode(cons_matrix4, util_ct.parms_id(), pow(2,he->he_scale), cons_pts[0]);
    he->encoder->encode(cons_matrix3, util_ct.parms_id(), pow(2,he->he_scale), cons_pts[1]);
    he->encoder->encode(cons_matrix2, util_ct.parms_id(), pow(2,he->he_scale), cons_pts[2]);
    he->encoder->encode(cons_matrix1, util_ct.parms_id(), pow(2,he->he_scale), cons_pts[3]);
    he->encoder->encode(cons_matrix0, util_ct2.parms_id(), util_ct2.scale(), cons_pts[4]);
    auto start = high_resolution_clock::now();
    // start computation
    cout << "num_cts:" << num_cts << endl;
    #pragma omp parallel for
    for(int i=0;i<num_cts;i++){
        he->evaluator->square(poly1[i], poly2[i]);
        he->evaluator->relinearize_inplace(poly2[i], *he->relin_keys);
        he->evaluator->rescale_to_next_inplace(poly2[i]);
        he->evaluator->square(poly2[i], poly4[i]);
        he->evaluator->relinearize_inplace(poly4[i], *he->relin_keys);
        he->evaluator->rescale_to_next_inplace(poly4[i]);
        // cout << "part1 ok" << endl;
        he->evaluator->mod_switch_to_next_inplace(poly1[i]);
        he->evaluator->multiply(poly2[i], poly1[i], poly3[i]);
        he->evaluator->relinearize_inplace(poly3[i], *he->relin_keys);
        he->evaluator->rescale_to_next_inplace(poly3[i]);
        
        he->evaluator->mod_switch_to_next_inplace(poly1[i]);
        he->evaluator->mod_switch_to_next_inplace(poly2[i]);
        he->evaluator->multiply_plain_inplace(poly1[i], scale_up1);
        he->evaluator->multiply_plain_inplace(poly2[i], scale_up2);
        he->evaluator->multiply_plain_inplace(poly3[i], scale_up3);
        // poly1[i].scale() = poly3[i].scale();
        // poly2[i].scale() = poly3[i].scale();
        // poly4[i].scale() = poly3[i].scale();
    }
    // he->print_parameters(he->context->get_context_data(poly2[0].parms_id())->parms());
    // he->print_parameters(he->context->get_context_data(poly3[0].parms_id())->parms());
    // he->print_parameters(he->context->get_context_data(poly4[0].parms_id())->parms());
    cout << "poly1~4 scale:" << endl;
    cout << log2(poly1[0].scale()) << "," << poly1[0].scale() << endl;
    cout << log2(poly2[0].scale()) << "," << poly2[0].scale() << endl;
    cout << log2(poly3[0].scale()) << "," << poly3[0].scale() << endl;
    cout << log2(poly4[0].scale()) << "," << poly4[0].scale() << endl;
    cout << "part2 ok" << endl;
    
    // 注意规则：相同level才可乘，相同level且相同scale才可加，我们忽略了0.5x，显然这一项可以和|x|合并
    #pragma omp parallel for
    for(int i=0;i<num_cts;i++){
        // cout << "poly4 (Q,scale):" <<log2(he->get_Q(poly4[i]))<<","<< log2(poly4[i].scale()) << endl;
        // cout << "cons_pts[0] (Q,scale):" <<log2(he->get_Q(cons_pts[0]))<<","<< log2(cons_pts[0].scale()) << endl;
        he->evaluator->multiply_plain_inplace(poly4[i], cons_pts[0]);
        he->evaluator->multiply_plain_inplace(poly3[i], cons_pts[1]);
        he->evaluator->multiply_plain_inplace(poly2[i], cons_pts[2]);
        he->evaluator->multiply_plain_inplace(poly1[i], cons_pts[3]);
        // Ciphertext ct_half_x;
        // Plaintext pt_half_x;
        // he->encoder->encode(cons_matrix, poly1[i].parms_id(), poly1[i].scale(), pt_half_x);
        // he->evaluator->multiply_plain(poly1[i], pt_half_x, ct_half_x);
        // cout << "part3 ok" << endl;
        
        // cout << poly1[0].scale() << endl;
        // cout << poly2[0].scale() << endl;
        // cout << poly3[0].scale() << endl;
        // cout << poly4[0].scale() << endl;
        he->evaluator->rescale_to_next_inplace(poly4[i]);
        he->evaluator->rescale_to_next_inplace(poly3[i]);
        he->evaluator->rescale_to_next_inplace(poly2[i]);
        he->evaluator->rescale_to_next_inplace(poly1[i]);
        he->evaluator->rescale_to_next_inplace(poly4[i]);
        he->evaluator->rescale_to_next_inplace(poly3[i]);
        he->evaluator->rescale_to_next_inplace(poly2[i]);
        he->evaluator->rescale_to_next_inplace(poly1[i]);
        // now, scale=30, Q=60
        // cout << "scale" << endl;
        // cout << log2(poly1[0].scale()) << endl;
        // cout << log2(poly2[0].scale()) << endl;
        // cout << log2(poly3[0].scale()) << endl;
        // cout << log2(poly4[0].scale()) << endl;
        poly2[i].scale() = poly1[i].scale();
        poly3[i].scale() = poly1[i].scale();
        poly4[i].scale() = poly1[i].scale();
        he->evaluator->add(poly1[i], poly2[i], output_cts[i]);
        // he->evaluator->mod_switch_to_next_inplace(output_cts[i]);
        // cout << "part4 ok" << endl;
        // cout << output_cts[i].scale() << endl;
        // cout << poly3[i].scale() << endl;
        // cout << poly4[i].scale() << endl;
        // output_cts[i].scale() = poly3[i].scale();
        he->evaluator->add_inplace(output_cts[i], poly3[i]);
        // output_cts[i].scale() = poly4[i].scale();
        he->evaluator->add_inplace(output_cts[i], poly4[i]);
        
        // cout << "part5 ok" << endl;
        // cout << output_cts[i].scale() << endl;
        // cout << cons_pts[i].scale() << endl;
        he->evaluator->add_plain_inplace(output_cts[i], cons_pts[4]);
        // cout << "part6 ok" << endl;
    }
    cout << "gelu poly time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    // exit(0);
    cout << "eval_gelu_poly done" << endl;
    return output_cts;
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
    int bitwidth = x_flat_avg.ell;
    int s = x_flat_avg.s;
    int num_ops = sigma.size;
    int dim = x_flat_avg.size / num_ops;
    cout << "in layer_norm_nonlinear, bitwidth,s,num_ops,dim:" << bitwidth << "," << s << "," << num_ops << "," << dim << endl;
    cout << "sigma.size, should be L:" << sigma.size << endl;
    // 计算平方根倒数
    FixArray sigma_sqrt = fpmath->sqrt(sigma, true);
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
    FixArray w = fpmath->fix->input(party, x_flat_avg.size, 1, true, bitwidth, s);
    FixArray b = fpmath->fix->input(party, x_flat_avg.size, 1, true, bitwidth, s);
    result["w"] = w;
    result["b"] = b;
    cout << "layer_norm_nonlinear done" << endl;
    return result;
}

void gelu_nonlinear_thread(int tid, int party, uint64_t *x1, uint64_t* x2, uint64_t* output, int array_size, int ell, int s, FPMath *fpmath){
    // cout << "in gelu thread, tid:" << tid << endl;
    int this_party;
    if (tid & 1) {
        this_party = 3 - party;
    } else {
        this_party = party;
    }
    FixArray x_array, gelu_poly_flat;
    x_array = fpmath->fix->input(this_party, array_size, x1, true, ell, s);
    gelu_poly_flat = fpmath->fix->input(this_party, array_size, x2, true, ell, s);
    auto start = high_resolution_clock::now();
    // cout << "x_array.size:" << x_array.size << endl;
    BoolArray msb_x = fpmath->fix->MSB(x_array);
    // cout << "MSB time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    // no communication
    FixArray neg_x = fpmath->fix->mul(x_array, -1);
    // point 2
    
    FixArray y = fpmath->fix->if_else(msb_x, neg_x, x_array);

    BoolArray lt27 = fpmath->fix->LT(y, 2.7*(1 << s));
    // point 8
    BoolArray all_0 = fpmath->bool_op->input(sci::ALICE, array_size, uint8_t(0));
    FixArray x_plus_y = fpmath->fix->add(x_array, y);   
    FixArray half_x_plus_y = x_plus_y;
    // FixArray half_x_plus_y = fpmath->fix->right_shift(x_plus_y, 1, all_0.data);
    // point 9
    // cout << "gelu_poly_flat(bitwidth,s):" << gelu_poly_flat.ell << "," << gelu_poly_flat.s << endl;
    // cout << "x array bitwidth,s:" << x_array.ell << "," << x_array.s << endl;
    // cout << "half_x_plus_y(bitwidth,s):" << half_x_plus_y.ell << "," << half_x_plus_y.s << endl;
    half_x_plus_y.s = s;
    //half_x_plus_y = (x+|x|)/2 = max(x,0), if |x|<2.7, 则 gelu_y, 否则half_x_plus_y
    FixArray ret = fpmath->fix->if_else(lt27, gelu_poly_flat, half_x_plus_y);
    // cout << "OK1" << endl;
    for(int i = 0; i < array_size; i++){
        output[i] = ret.data[i];
    }
    // cout << "OK2" << endl;
}


FixArray Bert_ckks::gelu_nonlinear(std::map<char*,FixArray> input){
    FixArray x_array = input["x_array"];
    FixArray gelu_poly_flat = input["gelu_poly_flat"];
    FixArray ret = this->nonlinear.fpmath[0]->fix->input(party, x_array.size, 1, true, x_array.ell, x_array.s);
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
                &ret.data[offset],
                chunk_size,
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

FixArray Bert_ckks::exp_he(const FixArray &x){
    int ell = x.ell;
    int scale = x.s;
    HE_CKKS* he = this->linear.he_softmax; 
    int slots = he->num_slots;
    int num_cts = x.size/slots;
    cout << "slots, num_cts:" << slots << "," << num_cts << endl;
    vector<vector<uint64_t>> vec_x;
    auto start = high_resolution_clock::now();
    for(int i = 0; i < num_cts; i++){
        vector<uint64_t> tmp;
        for(int j = 0; j < slots; j++){
            tmp.push_back(x.data[i*slots + j]);
            // cout << "x.data[i*slots + j]: " << x.data[i*slots + j] << endl;
        }
        vec_x.push_back(tmp);
    }
    cout << "exp_he data copy time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    cout << "vec_x.size:" << vec_x.size()*vec_x[0].size() << endl;
    start = high_resolution_clock::now();
    vector<Plaintext> pts_x;
    pts_x = mpc_to_ckks(vec_x, slots, he->get_Q(), ell, scale, he->he_scale, he);
    cout << "pts.size:" << pts_x.size() << endl;
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    vector<Ciphertext> cts_x(num_cts); 
    vector<Ciphertext> cts_y(num_cts);
    bool half = num_cts %2 == 0;
    vector<Ciphertext> cts_x_half; 
    if(half){
        cts_x_half.resize(num_cts/2);
    }
    vector<Plaintext> pts_y(num_cts);
    if (party == sci::BOB) {
        cout << "send_encrypted_vector" << endl;
        if (half){
            #pragma omp parallel for
            for(size_t i=0;i<num_cts/2;i++){
                he->encryptor->encrypt(pts_x[i], cts_x_half[i]);
            }
            send_encrypted_vector(io, cts_x_half);
        }
        else{
            #pragma omp parallel for
            for(size_t i=0;i<num_cts;i++){
                he->encryptor->encrypt(pts_x[i], cts_x[i]);
            }
            send_encrypted_vector(io, cts_x);
        }
        
		// test_util();
        cout << "send_encrypted_vector done" << endl;
		recv_encrypted_vector(he->context, io, cts_y);
        #pragma omp parallel for
        for(int i=0;i<cts_y.size();i++){
            he->decryptor->decrypt(cts_y[i], pts_y[i]);
        }
    }
    else{
        start = high_resolution_clock::now();
        if(half){
            recv_encrypted_vector(he->context, io, cts_x_half);
            this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
            start = high_resolution_clock::now();
            #pragma omp parallel for
            for(int i=0;i<cts_x_half.size();i++){
                Ciphertext ct1,tmp;
                // x + y*j => x - y*j
                he->evaluator->complex_conjugate(cts_x_half[i], *he->gal_keys, tmp);
                // x + y*j - (x - y*j)*
                // - 2 * y*j
                he->evaluator->sub(tmp, cts_x_half[i], ct1);
                // - 2 * y*j => 2*y
                MulImageUnitInplace(ct1, *he->context);
                // x + y*j + (x - y*j) => 2*x
                he->evaluator->add_inplace(cts_x_half[i], tmp);
                cts_x[i] = cts_x_half[i];
                cts_x[i+cts_x_half.size()] = ct1;
            }
        }
        else{
            recv_encrypted_vector(he->context, io, cts_x);
            this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
            start = high_resolution_clock::now();
        }
        // encode weight once
        Plaintext one, inverse_128;
        he->encoder->encode(0.015625, pow(2,he->he_scale), inverse_128);
        Ciphertext util_ct;
        {
            he->evaluator->multiply_plain(cts_x[0], inverse_128, util_ct);
            he->evaluator->rescale_to_next_inplace(util_ct);
        }
        he->encoder->encode(1.0, util_ct.parms_id(), util_ct.scale(), one);
        
        // begin computation
        #pragma omp parallel for
        for(int i=0;i<cts_x.size();i++){
            // cout << cts_x[i].poly_modulus_degree() << endl;
            he->evaluator->multiply_plain_inplace(cts_x[i], inverse_128);
            he->evaluator->rescale_to_next_inplace(cts_x[i]);
            he->evaluator->add_plain_inplace(cts_x[i], one);
            // x^128
            // cout << "OK2" << endl;
            for (int j = 0; j < log2(64); j++) {
                he->evaluator->square_inplace(cts_x[i]);
                he->evaluator->relinearize_inplace(cts_x[i], *he->relin_keys);
                he->evaluator->rescale_to_next_inplace(cts_x[i]);
            }
            // cout << "OK3" << endl;
        }
        PRG128 prg;
        /*
            生成r并计算x-r，这一过程极为复杂，暂时令r=0
        */
        // prg.random_mod_p<uint64_t>(r.data(), dim, linear.he->get_Q());
        cout << "num of output cts:" << cts_x.size() << endl;
        cout << "bitwidth of output ct:" << cts_x[0].coeff_modulus_size() << endl;
        this->he_compute_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        start = high_resolution_clock::now();
        send_encrypted_vector(io, cts_x);
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        start = high_resolution_clock::now();
        #pragma omp parallel for
        for(int i=0;i<cts_x.size();i++){
            vector<double> pod_matrix(poly_modulus_degree/2, 0.0);
            Plaintext tmp;
            he->encoder->encode(pod_matrix, cts_x[i].parms_id(), cts_x[i].scale(), tmp);
            pts_y[i] = tmp;
        }
        this->he_compute_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    }
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    vector<vector<HE_CKKS::scalar_t>> vec_output(pts_y.size());
    start = high_resolution_clock::now();
    vec_output = ckks_to_mpc(pts_y, poly_modulus_degree, he->get_Q(pts_y[0]), ell, scale, he);
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    FixArray ret = this->nonlinear.fpmath[0]->fix->input(party, x.size, 1, true, ell, scale);

    for(int i=0;i<vec_output.size();i++){
        for(int j=0;j<vec_output[i].size();j++){
            ret.data[i*slots + j] = vec_output[i][j];
        }
    }

    return ret;
}


FixArray Bert_ckks::element_wise_mul(FixArray x, FixArray y){
    cout << "begin element_wise_mul" << endl;
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
    uint64_t num_cts = dim / slot_count;
    if(x.size<slot_count){
        x = FixArray(x.party, slot_count, x.signed_, x.ell, x.s);
        y = FixArray(y.party, slot_count, y.signed_, y.ell, y.s);
        dim = slot_count;
        num_cts = 1;
    }
    cout << "slot_count, num_cts:" << slot_count << "," << num_cts << endl;
    vector<vector<HE_CKKS::scalar_t>> vec_x(num_cts);
    vector<vector<HE_CKKS::scalar_t>> vec_y(num_cts);
    vector<Plaintext> pts_x,pts_y;
    vector<Ciphertext> cts_x(num_cts),cts_y(num_cts);
    bool half = num_cts %2 == 0;
    vector<Ciphertext> cts_x_half,cts_y_half;
    if(half){
        cts_x_half.resize(num_cts/2);
        cts_y_half.resize(num_cts/2);
    }
    vector<Ciphertext> cts_output(num_cts);
    vector<Plaintext> output_pts(num_cts);
    for(int i = 0; i < num_cts; i++){
        for(int j = 0; j < slot_count; ++j){
            vec_x[i].push_back(x.data[(i*slot_count + j)%x.size]);
            vec_y[i].push_back(y.data[(i*slot_count + j)%y.size]);
        }
    }
    auto start = high_resolution_clock::now();
    pts_x = mpc_to_ckks(vec_x, slot_count, he->get_Q(), bitwidth, s, he->he_scale, he);
    pts_y = mpc_to_ckks(vec_y, slot_count, he->get_Q(), bitwidth, s, he->he_scale, he);
    this->conversion_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    vector<Ciphertext> half_cts(num_cts/2);
    // Alice is server, Bob is client
    if (party == sci::BOB) {
        if(half){
            #pragma omp parallel for
            for(size_t i=0;i<num_cts/2;i++){
                he->encryptor->encrypt(pts_x[i], cts_x_half[i]);
                he->encryptor->encrypt(pts_y[i], cts_y_half[i]);
            }
            send_encrypted_vector(io, cts_x_half);
            send_encrypted_vector(io, cts_y_half);
        }
        else{
            #pragma omp parallel for
            for(size_t i=0;i<num_cts;i++){
                he->encryptor->encrypt(pts_x[i], cts_x[i]);
                he->encryptor->encrypt(pts_y[i], cts_y[i]);
            }
            send_encrypted_vector(io, cts_x);
            send_encrypted_vector(io, cts_y);
        }
        cout << "client send_encrypted_vector done" << endl;
		recv_encrypted_vector(he->context, io, cts_output);
        // check_cts(cts_output);
        cout << "client recv_encrypted_vector done" << endl;
        #pragma omp parallel for
        for(int i=0;i<cts_output.size();i++){
            he->decryptor->decrypt(cts_output[i], output_pts[i]);
        }
        cout << "client decrypt done" << endl;
	}
    else{
        // receive half_cts instead of cts
        start = high_resolution_clock::now();
        if(half){
            recv_encrypted_vector(he->context, io, cts_x_half);
            recv_encrypted_vector(he->context, io, cts_y_half);
            this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
            start = high_resolution_clock::now();
            #pragma omp parallel for
            for(int i=0;i<cts_x_half.size();i++){
                Ciphertext ct1,ct2,tmp2,tmp;
                // x + y*j => x - y*j
                he->evaluator->complex_conjugate(cts_x_half[i], *he->gal_keys, tmp);
                // x + y*j - (x - y*j)*
                // - 2 * y*j
                he->evaluator->sub(tmp, cts_x_half[i], ct1);
                // - 2 * y*j => 2*y
                MulImageUnitInplace(ct1, *he->context);
                // x + y*j + (x - y*j) => 2*x
                he->evaluator->add_inplace(cts_x_half[i], tmp);
                cts_x[i] = cts_x_half[i];
                cts_x[i+cts_x_half.size()] = ct1;


                he->evaluator->complex_conjugate(cts_y_half[i], *he->gal_keys, tmp2);
                he->evaluator->sub(tmp2, cts_y_half[i], ct2);
                MulImageUnitInplace(ct2, *he->context);
                he->evaluator->add_inplace(cts_y_half[i], tmp2);
                cts_y[i] = cts_y_half[i];
                cts_y[i+cts_y_half.size()] = ct2;
            }
        }
        else{
            recv_encrypted_vector(he->context, io, cts_x);
            recv_encrypted_vector(he->context, io, cts_y);
            this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
            start = high_resolution_clock::now();
        }
        #pragma omp parallel for
        for(int i=0;i<cts_x.size();i++){
            he->evaluator->add_plain_inplace(cts_x[i], pts_x[i]);
            he->evaluator->add_plain_inplace(cts_y[i], pts_y[i]);
            he->evaluator->multiply_inplace(cts_x[i], cts_y[i]);
            he->evaluator->relinearize_inplace(cts_x[i], *he->relin_keys);
            he->evaluator->rescale_to_next_inplace(cts_x[i]);
        }
        this->he_compute_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        PRG128 prg;
        /*
            生成r并计算x-r，这一过程极为复杂，暂时令r=0
        */
        vector<uint64_t> r(dim,0);
        start = high_resolution_clock::now();
        cout << "server send_encrypted_vector" << endl;
        cout << "cts_x.size:" << cts_x.size() << endl;
        send_encrypted_vector(io, cts_x);
        this->ct_transfer_la += ((high_resolution_clock::now() - start)).count()/1e+9;
        start = high_resolution_clock::now();
        #pragma omp parallel for
        for(int i=0;i<num_cts;i++){
            vector<double> pod_matrix(poly_modulus_degree/2, 0.0);
            Plaintext tmp;
            he->encoder->encode(pod_matrix, cts_x[i].parms_id(), cts_x[i].scale(), tmp);
            output_pts[i] = tmp;
        }
        this->he_compute_la += ((high_resolution_clock::now() - start)).count()/1e+9;
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
    FixArray output(party, x.size, x.signed_, bitwidth, s);
    #pragma omp parallel for
    for(int i = 0; i < num_cts; i++){
        for(int j = 0; j < slot_count; ++j){
            output.data[(i*slot_count + j)%x.size] = output_vector[i][j];
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
FixArray Bert_ckks::div_batch_he_V2(const FixArray& nm, const FixArray& dn, int batch_dn_size, int l_out, int s_out, bool normalized_dn){
    assert(nm.size == dn.size * batch_dn_size);
    auto fpmath = this->nonlinear.fpmath[0];
    auto start = high_resolution_clock::now();
    FixArray dn_rec = fpmath->sqrt(dn, true);
    this->other_nonlinear_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->other_nonlinear_comm += this->get_comm();
    this->other_nonlinear_round += this->get_round();
    FixArray dn_rec_flat = fpmath->fix->input(dn_rec.party, nm.size,1, dn_rec.signed_, dn_rec.ell, dn_rec.s);
    for(int i=0; i<nm.size; i++){
        dn_rec_flat.data[i] = dn_rec.data[i%dn_rec.size];
    }
    return element_wise_mul(nm, dn_rec_flat);
}

void max_thread(int tid, int party, uint64_t *x, uint64_t *y, int num_ops, int array_size, int ell, int s, FPMath *fpmath){
    int this_party;
    if (tid & 1) {
        this_party = 3 - party;
    } else {
        this_party = party;
    }
    vector<FixArray> input_array;
    for(int i = 0; i < num_ops; i++){
        input_array.push_back(fpmath->fix->input(this_party, array_size, &x[i*array_size], true, ell, s));
    }
    FixArray x_max = fpmath->fix->max(input_array);
    for(int i = 0; i < num_ops; i++){
        y[i] = x_max.data[i];
    }
}

vector<FixArray> Bert_ckks::softmax_fix_he(int party, FixArray x, FPMath* fpmath[]) {
    // std::cout << "Entering softmax fix" << std::endl;
    auto start = high_resolution_clock::now();
    int ell = x.ell;
    int s = x.s;
    cout << "ell,s:" << ell << "," << s << endl;
    // vector<FixArray> ret(N);

    bool signed_ = x.signed_;
    //理论上在第二个维度进行求Max
    //point1
    // multi threads for max
    
    FCMetadata data = linear.data_lin1_0;
    int softmax_dim = data.image_size;
    int nthreads = PACKING_NUM*4, dim = PACKING_NUM*softmax_dim, array_size=dim/nthreads;
    FixArray x_max = fpmath[0]->fix->input(party, dim, 1, signed_, ell, s);
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

    FixArray e_x_flat;
    this->other_nonlinear_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    this->other_nonlinear_comm += get_comm();
    this->other_nonlinear_round += get_round();
    //point3
    int exp_ell = 40;
    auto start2 = high_resolution_clock::now();
    e_x_flat= exp_he(shifted_x_flat);
    cout << "block 3 time:" << ((high_resolution_clock::now() - start2)).count()/1e+9 << endl;
    // exit(0);
    start = high_resolution_clock::now();
    cout << "exp_he OK" << endl;
    vector<FixArray> e_x_tr(array_size);
    for (int i = 0; i < array_size; i++) {
        e_x_tr[i] = FixArray(party, dim, signed_, exp_ell, s);
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
    cout << "e_x_flat.size:" << e_x_flat.size << endl;
    cout << "sum_e_x.size:" << sum_e_x.size << endl;
    // FixArray ret_flat = fpmath->fix->div_batch(e_x_flat, sum_e_x, n ,exp_ell, s);
    cout << "exp_he data time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    start = high_resolution_clock::now();
    FixArray ret_flat = div_batch_he_V2(e_x_flat, sum_e_x, array_size ,exp_ell, s);
    cout << "div_batch_he time:" << ((high_resolution_clock::now() - start)).count()/1e+9 << endl;
    start = high_resolution_clock::now();
    //point5
    BoolArray all_0 = fpmath[0]->bool_op->input(sci::ALICE, dim, uint8_t(0));
    // uint8_t* msb_ret = new uint8_t[ret_flat.size];
    // memset(msb_ret, 0, sizeof(msb_ret));
    cout << "ret_flat.size:" << ret_flat.size << endl;
    ret_flat = fpmath[0]->fix->extend(ret_flat, ell, all_0.data);
    vector<FixArray> ret(dim);
    //point6
    // return make_tuple(ret, l_short);
    for (int i = 0; i < dim; i++) {
        ret[i] = FixArray(party, array_size, signed_, ell, s);
        memcpy(ret[i].data, ret_flat.data + i*array_size, array_size*sizeof(uint64_t));
    }
    this->other_nonlinear_la += ((high_resolution_clock::now() - start)).count()/1e+9;
    return ret;
}

FixArray Bert_ckks::softmax_he(FixArray input) {
    FPMath *fpmath = nonlinear.fpmath[0];
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

    vector<FixArray> output_array;
    cout << "begin softmax_fix_he" << endl;
    output_array = softmax_fix_he(party, input, this->nonlinear.fpmath);
    for(int i = 0; i < num_ops; i++){
        memcpy(&softmax_output_row[i*array_size], output_array[i].data, array_size * sizeof(uint64_t));
    }
    cout << "OK3" << endl;
    FixArray result = fpmath->fix->input(party, softmax_output_size, softmax_output_row, true, 40, 20);
    return result;
}


FixArray Bert_ckks::softmax_ot(FixArray input) {
    FPMath *fpmath = nonlinear.fpmath[0];
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
    FixArray result = fpmath->fix->input(party, softmax_output_size, softmax_output_row, true, 40, 20);
    return result;
}

Ciphertext inner_rot(Ciphertext x, HE_CKKS* he,int i){
    Ciphertext tmp1,tmp2,result;
    vector<double> mask_1(he->num_slots,1),mask_2(he->num_slots,1);
    Plaintext mask_1_pt,mask_2_pt;
    he->encoder->encode(mask_1, x.parms_id(), pow(2,he->he_scale), mask_1_pt);
    he->encoder->encode(mask_2, x.parms_id(), pow(2,he->he_scale), mask_2_pt);
    he->evaluator->multiply_plain(x, mask_1_pt, tmp1);
    he->evaluator->multiply_plain(x, mask_2_pt, tmp2);
    he->evaluator->rotate_vector_inplace(tmp1, i,*(he->gal_keys));
    he->evaluator->rotate_vector_inplace(tmp2, -i,*(he->gal_keys));
    he->evaluator->add(tmp1,tmp2,result);
    return result;
}

//只测bert-large+GPT2
vector<Ciphertext> Bert_ckks::ct_ct_mul_blb(vector<Ciphertext> x,vector<Ciphertext> y, HE_CKKS* he){
    auto data = this->linear.data_lin1_0;
    //bert-large
    // int num_x=8,num_y=8,num_res=16;
    // int rot1 = 6*8,rot2=22*8,rot3=128;
    // int mul1 = 16*8,mul2=128*8,mul3=128;
    //GPT2
    int num_x=6,num_y=6,num_res=12;
    int rot1 = 44,rot2=84,rot3=128;
    int mul1 = 16*4+32*2,mul2=64*6,mul3=128;
    vector<Ciphertext> HE_result(
        num_res
    );
    vector<double> mask_1(he->num_slots,1);
    Plaintext mask_1_pt,mask_2_pt;
    he->encoder->encode(mask_1, x[0].parms_id(), pow(2,he->he_scale), mask_1_pt);
    // step 1
    #pragma omp parallel for num_threads(rot1/2)
    for(int i=0;i<rot1;i++){
        auto tmp = inner_rot(x[0],he,i%6);
    }
    #pragma omp parallel for
    for(int i=0;i<mul1-rot1;i++){
        Ciphertext tmp;
        he->evaluator->multiply_plain(x[0], mask_1_pt, tmp);
        he->evaluator->multiply_plain(x[0], mask_1_pt, tmp);
    }
    for(int i=0;i<y.size();i++){
        y[i].scale() = pow(2,he->he_scale+he->he_scale);
    }
    cout << "step1 ok" << endl;
    // step 2
    #pragma omp parallel for num_threads(rot2/4)
    for(int i=0;i<rot2;i++){
        auto tmp = inner_rot(y[0],he,i%22);
    }
    #pragma omp parallel for num_threads(64)
    for(int i=0;i<mul2-y.size();i++){
        Ciphertext tmp;
        he->evaluator->multiply(y[0], x[0], tmp);
    }

    #pragma omp parallel for num_threads(2)
    for(int i=0;i<y.size();i++){
        y[i].scale() = pow(2,he->he_scale+he->he_scale+he->he_scale);
        x[i].scale() = pow(2,he->he_scale+he->he_scale);
        he->evaluator->rescale_to_next_inplace(y[i]);
        he->evaluator->rescale_to_next_inplace(x[i]);
        he->evaluator->multiply(x[i], y[i], HE_result[i]);
        HE_result[i+y.size()] = HE_result[i];
        he->evaluator->relinearize_inplace(HE_result[i], *he->relin_keys);
        he->evaluator->relinearize_inplace(HE_result[i+y.size()], *he->relin_keys);
        he->evaluator->rescale_to_next_inplace(HE_result[i]);
        he->evaluator->rescale_to_next_inplace(HE_result[i+y.size()]);
    }
    cout << "step2 ok" << endl;
    // step3
    #pragma omp parallel for num_threads(64)
    for(int i=0;i<rot3;i++){
        auto tmp = inner_rot(HE_result[0],he,i%120);
    }
    #pragma omp parallel for num_threads(2)
    for(int i =0;i<HE_result.size();i++){
        HE_result[i].scale() = HE_result[i].scale() * pow(2,he->he_scale);
        he->evaluator->rescale_to_next_inplace(HE_result[i]);
    }
    cout << "step3 ok" << endl;
    return HE_result;
}

vector<Ciphertext> Bert_ckks::ct_ct_mul_powerformer(vector<Ciphertext> x,vector<Ciphertext> y, HE_CKKS* he){
    auto data = this->linear.data_lin1_0;
    //bert-large
    int num_x=8,num_y=8,num_res=16;
    int rot1 = 856/2,rot2=4872/2,rot3=128/2;
    int mul1 = 16*8,mul2=128*8,mul3=128;
    //GPT2
    // int num_x=6,num_y=6,num_res=12;
    // int rot1 = 152,rot2=872,rot3=128;
    // int mul1 = 16*4+32*2,mul2=64*6,mul3=128;
    vector<Ciphertext> HE_result(
        num_res
    );
    vector<double> mask_1(he->num_slots,1);
    Plaintext mask_1_pt,mask_2_pt;
    he->encoder->encode(mask_1, x[0].parms_id(), pow(2,he->he_scale), mask_1_pt);
    // step 1
    #pragma omp parallel for num_threads(rot1/2)
    for(int i=0;i<rot1;i++){
        auto tmp = inner_rot(x[0],he,i%6);
    }
    #pragma omp parallel for
    for(int i=0;i<mul1-rot1;i++){
        Ciphertext tmp;
        he->evaluator->multiply_plain(x[0], mask_1_pt, tmp);
        he->evaluator->multiply_plain(x[0], mask_1_pt, tmp);
    }
    for(int i=0;i<y.size();i++){
        y[i].scale() = pow(2,he->he_scale+he->he_scale);
    }
    cout << "step1 ok" << endl;
    // step 2
    #pragma omp parallel for num_threads(64)
    for(int i=0;i<rot2;i++){
        auto tmp = inner_rot(y[0],he,i%22);
    }
    // #pragma omp parallel for num_threads(64)
    // for(int i=0;i<mul2-y.size();i++){
    //     Ciphertext tmp;
    //     he->evaluator->multiply(y[0], x[0], tmp);
    // }

    #pragma omp parallel for num_threads(2)
    for(int i=0;i<y.size();i++){
        y[i].scale() = pow(2,he->he_scale+he->he_scale+he->he_scale);
        x[i].scale() = pow(2,he->he_scale+he->he_scale);
        he->evaluator->rescale_to_next_inplace(y[i]);
        he->evaluator->rescale_to_next_inplace(x[i]);
        he->evaluator->multiply(x[i], y[i], HE_result[i]);
        HE_result[i+y.size()] = HE_result[i];
        he->evaluator->relinearize_inplace(HE_result[i], *he->relin_keys);
        he->evaluator->relinearize_inplace(HE_result[i+y.size()], *he->relin_keys);
        he->evaluator->rescale_to_next_inplace(HE_result[i]);
        he->evaluator->rescale_to_next_inplace(HE_result[i+y.size()]);
    }
    cout << "step2 ok" << endl;
    // step3
    #pragma omp parallel for num_threads(64)
    for(int i=0;i<rot3;i++){
        auto tmp = inner_rot(HE_result[0],he,i%120);
    }
    #pragma omp parallel for num_threads(2)
    for(int i =0;i<HE_result.size();i++){
        HE_result[i].scale() = HE_result[i].scale() * pow(2,he->he_scale);
        he->evaluator->rescale_to_next_inplace(HE_result[i]);
    }
    cout << "step3 ok" << endl;
    return HE_result;
}
